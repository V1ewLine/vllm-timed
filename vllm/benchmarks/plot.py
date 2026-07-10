# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate plots for benchmark results."""

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    import plotly.express as px
    import plotly.io as pio
except ImportError:
    px = None
    pio = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def generate_timeline_plot(
    results: list[dict[str, Any]],
    output_path: Path,
    colors: list[str] | None = None,
    itl_thresholds: list[float] | None = None,
    labels: list[str] | None = None,
) -> None:
    """
    Generate an HTML timeline plot from benchmark results.

    Args:
        results: List of per-request result dictionaries containing:
            - start_time: Request start time (seconds)
            - ttft: Time to first token (seconds)
            - itl: List of inter-token latencies (seconds)
            - latency: Total request latency (seconds)
            - prompt_len: Number of prompt tokens
            - output_tokens: Number of output tokens
        output_path: Path where the HTML file will be saved
        colors: List of colors for ITL categories (default: green, orange, red, black)
        itl_thresholds: ITL thresholds in seconds (default: [1.0, 4.0, 6.0])
        labels: Labels for ITL categories (default based on thresholds)
    """
    if px is None or pio is None:
        raise ImportError("Timeline plotting requires Plotly and pandas.")

    # Set defaults
    if colors is None:
        colors = ["#109618", "#FF7F0E", "#D62728"]
    if itl_thresholds is None:
        itl_thresholds = [0.025, 0.050]
    if labels is None:
        labels = [
            f"ITL < {itl_thresholds[0] * 1000:.0f}ms",
            f"{itl_thresholds[0] * 1000:.0f}ms ≤ ITL < {itl_thresholds[1] * 1000:.0f}ms",  # noqa
            f"ITL ≥ {itl_thresholds[1] * 1000:.0f}ms",
        ]

    labels_colors = {"TTFT": "#636EFA", **dict(zip(labels, colors))}
    labels_order = ["TTFT"] + labels

    timeline_data = construct_timeline_data(results, itl_thresholds, labels)

    if not timeline_data:
        print("No timeline data to plot")
        return

    # Create the plot
    fig = px.timeline(
        timeline_data,
        x_start="start",
        x_end="end",
        y="request_id",
        color="type",
        color_discrete_map=labels_colors,
        category_orders={"type": labels_order},
        hover_data=[
            "prompt_tokens",
            "output_tokens",
            "req_start_time",
            "req_finish_time",
            "segment_start",
            "segment_end",
            "duration",
        ],
    )

    # Customize hover template to show only time without date
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>"
        "Type: %{fullData.name}<br>"
        "Start: %{customdata[4]}<br>"
        "End: %{customdata[5]}<br>"
        "Duration: %{customdata[6]}<br>"
        "Prompt Tokens: %{customdata[0]}<br>"
        "Output Tokens: %{customdata[1]}<br>"
        "Request Start Time: %{customdata[2]}<br>"
        "Request End Time: %{customdata[3]}<br>"
        "<extra></extra>"
    )

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Request ID",
        showlegend=True,
    )

    # Save to HTML
    pio.write_html(fig, str(output_path))
    print(f"Timeline plot saved to: {output_path}")


def construct_timeline_data(
    requests_data: list[dict[str, Any]],
    itl_thresholds: list[float],
    labels: list[str],
) -> list[dict[str, Any]]:
    """
    Construct timeline data from request results.

    Args:
        requests_data: List of per-request result dictionaries
        itl_thresholds: ITL thresholds in seconds
        labels: Labels for ITL categories

    Returns:
        List of timeline segments for plotting
    """

    def tostr(sec_time: float) -> str:
        """Convert seconds to HH:MM:SS.mmm format."""
        h = int(sec_time // 3600)
        assert h < 100, "time seems to last more than 100 hours"
        m = int((sec_time % 3600) // 60)
        s = sec_time % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    def itl_type(itl: float) -> str:
        """Categorize ITL based on thresholds."""
        if itl < itl_thresholds[0]:
            return labels[0]
        elif itl < itl_thresholds[1]:
            return labels[1]
        else:
            return labels[2]

    # Find the earliest start time to use as t0
    t0 = None
    for request in requests_data:
        start_time = request.get("start_time")
        if start_time is not None and (t0 is None or start_time < t0):
            t0 = start_time

    if t0 is None:
        return []

    timeline_data = []

    for i, request in enumerate(requests_data):
        start_time = request.get("start_time")
        ttft = request.get("ttft")
        itl = request.get("itl", [])
        latency = request.get("latency")
        prompt_len = request.get("prompt_len", 0)
        output_tokens = request.get("output_tokens", 0)

        # Skip requests without required data
        if start_time is None or ttft is None or latency is None:
            continue

        # Normalize start time
        start_time = start_time - t0
        start_time_str = tostr(start_time)

        # TTFT segment
        ttft_end = start_time + ttft
        ttft_end_str = tostr(ttft_end)

        timeline_data.append(
            {
                "request_id": f"Req {i}",
                "start": start_time_str,
                "end": ttft_end_str,
                "type": "TTFT",
                "prompt_tokens": prompt_len,
                "output_tokens": output_tokens,
                "req_start_time": tostr(start_time),
                "req_finish_time": tostr(start_time + latency),
                "segment_start": start_time_str,
                "segment_end": ttft_end_str,
                "duration": f"{ttft:.3f}s",
            }
        )

        # ITL segments
        prev_time = ttft_end
        prev_time_str = ttft_end_str

        for itl_value in itl:
            itl_end = prev_time + itl_value
            itl_end_str = tostr(itl_end)

            timeline_data.append(
                {
                    "request_id": f"Req {i}",
                    "start": prev_time_str,
                    "end": itl_end_str,
                    "type": itl_type(itl_value),
                    "prompt_tokens": prompt_len,
                    "output_tokens": output_tokens,
                    "req_start_time": tostr(start_time),
                    "req_finish_time": tostr(start_time + latency),
                    "segment_start": prev_time_str,
                    "segment_end": itl_end_str,
                    "duration": f"{itl_value:.3f}s",
                }
            )

            prev_time = itl_end
            prev_time_str = itl_end_str

    return timeline_data


def construct_trace_plot_data(
    arrival_times: list[float],
    start_times: list[float],
    latencies: list[float],
    successes: list[bool],
    bin_width: float | None = None,
) -> dict[str, Any]:
    """Build scheduled and observed request-rate and cumulative series."""
    lengths = {
        len(arrival_times),
        len(start_times),
        len(latencies),
        len(successes),
    }
    if len(lengths) != 1:
        raise ValueError("Trace plot inputs must have the same length.")
    if not arrival_times:
        return {}
    if any(not math.isfinite(value) for value in arrival_times + start_times):
        raise ValueError("Trace plot timestamps must be finite.")
    if any(not math.isfinite(value) or value < 0 for value in latencies):
        raise ValueError("Trace plot latencies must be finite and non-negative.")

    first_start = min(start_times)
    observed_starts = [start_time - first_start for start_time in start_times]
    completion_times = sorted(
        observed_start + latency
        for observed_start, latency, success in zip(
            observed_starts, latencies, successes
        )
        if success
    )
    scheduled_arrivals = sorted(arrival_times)
    observed_starts.sort()

    max_time = max(
        scheduled_arrivals[-1],
        observed_starts[-1],
        completion_times[-1] if completion_times else 0.0,
    )
    if bin_width is None:
        target_width = max_time / 300 if max_time > 0 else 1.0
        nice_widths = [
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
            30.0,
            60.0,
            120.0,
            300.0,
            600.0,
        ]
        bin_width = next(
            (width for width in nice_widths if width >= target_width),
            nice_widths[-1],
        )
    if not math.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("Trace plot bin width must be finite and positive.")

    bucket_count = max(1, int(max_time // bin_width) + 1)
    bucket_centers = [(index + 0.5) * bin_width for index in range(bucket_count)]

    def request_rates(timestamps: list[float]) -> list[float]:
        counts = [0] * bucket_count
        for timestamp in timestamps:
            bucket = min(int(timestamp // bin_width), bucket_count - 1)
            counts[bucket] += 1
        return [count / bin_width for count in counts]

    return {
        "bin_width": bin_width,
        "bucket_centers": bucket_centers,
        "scheduled_rates": request_rates(scheduled_arrivals),
        "observed_rates": request_rates(observed_starts),
        "completion_rates": request_rates(completion_times),
        "scheduled_arrivals": scheduled_arrivals,
        "observed_starts": observed_starts,
        "completion_times": completion_times,
    }


def generate_trace_plot(
    arrival_times: list[float],
    start_times: list[float],
    latencies: list[float],
    successes: list[bool],
    output_path: Path,
) -> None:
    """Generate a PNG comparing scheduled and observed request traffic."""
    if plt is None:
        raise ImportError("Static benchmark plotting requires Matplotlib.")
    data = construct_trace_plot_data(
        arrival_times=arrival_times,
        start_times=start_times,
        latencies=latencies,
        successes=successes,
    )
    if not data:
        print("No request trace data to plot")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (rate_ax, cumulative_ax) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1]},
    )

    series = [
        ("Scheduled arrivals", "scheduled", "--"),
        ("Observed starts", "observed", "-"),
        ("Successful completions", "completion", ":"),
    ]
    colors = {
        "scheduled": "#6B7280",
        "observed": "#2563EB",
        "completion": "#D97706",
    }
    for label, key, line_style in series:
        rate_ax.plot(
            data["bucket_centers"],
            data[f"{key}_rates"],
            label=label,
            color=colors[key],
            linestyle=line_style,
            linewidth=1.8,
        )
        timestamps = data[
            {
                "scheduled": "scheduled_arrivals",
                "observed": "observed_starts",
                "completion": "completion_times",
            }[key]
        ]
        cumulative_ax.step(
            timestamps,
            range(1, len(timestamps) + 1),
            where="post",
            color=colors[key],
            linestyle=line_style,
            linewidth=1.8,
        )

    rate_ax.set_title(
        f"Request Rate ({data['bin_width']:g}s buckets)", loc="left", fontsize=12
    )
    rate_ax.set_ylabel("Requests / second")
    rate_ax.legend(loc="upper right", ncols=3, frameon=False)
    rate_ax.grid(True, color="#D1D5DB", alpha=0.55, linewidth=0.8)
    rate_ax.set_ylim(bottom=0)

    cumulative_ax.set_title("Cumulative Requests", loc="left", fontsize=12)
    cumulative_ax.set_xlabel("Seconds since first observed request")
    cumulative_ax.set_ylabel("Request count")
    cumulative_ax.grid(True, color="#D1D5DB", alpha=0.55, linewidth=0.8)
    cumulative_ax.set_ylim(bottom=0)

    fig.suptitle(
        "Scheduled vs Observed Benchmark Trace",
        x=0.07,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Request trace plot saved to: {output_path}")


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def _distribution_bins(
    values: Sequence[int], count: int = 40
) -> tuple[list[float], bool]:
    minimum = min(values)
    maximum = max(values)
    use_log_scale = minimum > 0 and maximum / minimum >= 20
    if not use_log_scale or minimum == maximum:
        width = max(1.0, (maximum - minimum) / count)
        return [minimum + index * width for index in range(count + 1)], False

    ratio = (maximum / minimum) ** (1 / count)
    return [minimum * ratio**index for index in range(count + 1)], True


def generate_length_distribution_plot(
    values: list[int],
    output_path: Path,
    *,
    title: str,
    xlabel: str,
    color: str,
) -> None:
    """Generate a request-length histogram with key workload quantiles."""
    if plt is None:
        raise ImportError("Static benchmark plotting requires Matplotlib.")
    if not values:
        raise ValueError(f"No values available for {title}.")
    if any(value < 0 for value in values):
        raise ValueError(f"{title} values must be non-negative.")

    bins, use_log_scale = _distribution_bins(values)
    quantiles = [
        ("P50", _percentile(values, 0.50), "-"),
        ("P90", _percentile(values, 0.90), "--"),
        ("P99", _percentile(values, 0.99), ":"),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.hist(values, bins=bins, color=color, edgecolor="#1F2937", alpha=0.82)
    for label, value, line_style in quantiles:
        ax.axvline(
            value,
            color="#111827",
            linestyle=line_style,
            linewidth=1.4,
            label=f"{label}: {value:,.0f}",
        )
    if use_log_scale:
        ax.set_xscale("log")
        scale_note = "Log-scaled token axis"
    else:
        scale_note = "Linear token axis"

    fig.suptitle(
        title,
        x=0.08,
        y=0.98,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.93,
        f"{len(values):,} requests | {scale_note}",
        color="#4B5563",
        fontsize=10,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Request count")
    ax.grid(True, axis="y", color="#D1D5DB", alpha=0.55, linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Length distribution plot saved to: {output_path}")


def generate_input_output_distribution_plot(
    input_lens: list[int],
    output_lens: list[int],
    output_path: Path,
) -> None:
    """Generate the joint input/output token distribution for all requests."""
    if plt is None:
        raise ImportError("Static benchmark plotting requires Matplotlib.")
    if not input_lens or len(input_lens) != len(output_lens):
        raise ValueError("Input and output lengths must be non-empty and aligned.")
    if any(value <= 0 for value in input_lens):
        raise ValueError("Input lengths must be positive.")
    if any(value < 0 for value in output_lens):
        raise ValueError("Output lengths must be non-negative.")

    output_scale = "log" if min(output_lens) > 0 else "linear"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    density = ax.hexbin(
        input_lens,
        output_lens,
        gridsize=42,
        mincnt=1,
        bins="log",
        xscale="log",
        yscale=output_scale,
        cmap="Blues",
        linewidths=0.3,
    )
    colorbar = fig.colorbar(density, ax=ax, pad=0.02)
    colorbar.set_label("Samples per bin (log scale)")
    fig.suptitle(
        "Input vs Output Length Distribution",
        x=0.08,
        y=0.98,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.93,
        f"All {len(input_lens):,} requests from this benchmark",
        color="#4B5563",
        fontsize=10,
    )
    ax.set_xlabel("Input tokens (log scale)")
    ax.set_ylabel(f"Output tokens ({output_scale} scale)")
    ax.grid(True, which="major", color="#D1D5DB", alpha=0.45, linewidth=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Input/output distribution plot saved to: {output_path}")


def generate_benchmark_plots(
    result: Mapping[str, Any],
    output_dir: Path,
    prefix: str = "benchmark",
    plots: Sequence[str] = ("trace", "input", "output", "input-output"),
) -> list[Path]:
    """Generate selected plots from one detailed benchmark result."""
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = set(plots)
    unknown = requested - {"trace", "input", "output", "input-output"}
    if unknown:
        raise ValueError(f"Unknown benchmark plots: {sorted(unknown)}")

    created: list[Path] = []
    if "trace" in requested:
        trace_path = output_dir / f"{prefix}.trace.png"
        generate_trace_plot(
            arrival_times=list(result.get("arrival_times", [])),
            start_times=list(result.get("start_times", [])),
            latencies=list(result.get("latencies", [])),
            successes=list(result.get("successes", [])),
            output_path=trace_path,
        )
        created.append(trace_path)

    input_lens = [int(value) for value in result.get("input_lens", [])]
    output_lens = [int(value) for value in result.get("output_lens", [])]
    if "input" in requested:
        input_path = output_dir / f"{prefix}.input_distribution.png"
        generate_length_distribution_plot(
            input_lens,
            input_path,
            title="Input Length Distribution",
            xlabel="Input tokens",
            color="#2563EB",
        )
        created.append(input_path)
    if "output" in requested:
        output_path = output_dir / f"{prefix}.output_distribution.png"
        generate_length_distribution_plot(
            output_lens,
            output_path,
            title="Output Length Distribution",
            xlabel="Actually generated output tokens",
            color="#D97706",
        )
        created.append(output_path)
    if "input-output" in requested:
        joint_path = output_dir / f"{prefix}.input_output_distribution.png"
        generate_input_output_distribution_plot(
            input_lens=input_lens,
            output_lens=output_lens,
            output_path=joint_path,
        )
        created.append(joint_path)

    return created


def generate_dataset_stats_plot(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Generate a matplotlib figure with dataset statistics.

    Creates a figure with 4 subplots:
    - Top-left: Prompt tokens distribution (histogram)
    - Top-right: Output tokens distribution (histogram)
    - Bottom-left: Prompt+output tokens distribution (histogram)
    - Bottom-right: Stacked bar chart (request_id vs tokens)

    Args:
        results: List of per-request result dictionaries containing:
            - prompt_len: Number of prompt tokens
            - output_tokens: Number of output tokens
        output_path: Path where the figure will be saved
    """
    if plt is None:
        raise ImportError("Dataset statistics plotting requires Matplotlib.")

    # Extract data
    prompt_tokens = []
    output_tokens = []
    total_tokens = []

    for request in results:
        prompt_len = request.get("prompt_len", 0)
        output_len = request.get("output_tokens", 0)

        prompt_tokens.append(prompt_len)
        output_tokens.append(output_len)
        total_tokens.append(prompt_len + output_len)

    if not prompt_tokens:
        print("No data available for dataset statistics plot")
        return

    # Create figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Prompt tokens distribution
    ax1.hist(prompt_tokens, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Prompt Tokens")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Prompt Tokens Distribution")
    ax1.grid(True, alpha=0.3)

    # Top-right: Output tokens distribution
    ax2.hist(output_tokens, bins=30, color="coral", edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Output Tokens")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Output Tokens Distribution")
    ax2.grid(True, alpha=0.3)

    # Bottom-left: Prompt+output tokens distribution
    ax3.hist(
        total_tokens, bins=30, color="mediumseagreen", edgecolor="black", alpha=0.7
    )
    ax3.set_xlabel("Total Tokens (Prompt + Output)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Total Tokens Distribution")
    ax3.grid(True, alpha=0.3)

    # Bottom-right: Stacked bar chart
    request_ids = list(range(len(prompt_tokens)))
    ax4.bar(
        request_ids, prompt_tokens, label="Prompt Tokens", color="steelblue", alpha=0.7
    )
    ax4.bar(
        request_ids,
        output_tokens,
        bottom=prompt_tokens,
        label="Output Tokens",
        color="coral",
        alpha=0.7,
    )
    ax4.set_xlabel("Request ID")
    ax4.set_ylabel("Tokens")
    ax4.set_title("Tokens per Request (Stacked)")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save figure
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Dataset statistics plot saved to: {output_path}")


def generate_benchmark_plots_from_file(
    result_json: Path,
    output_dir: Path | None = None,
    prefix: str | None = None,
    plots: Sequence[str] = ("trace", "input", "output", "input-output"),
) -> list[Path]:
    """Generate plots independently from a saved detailed benchmark result."""
    with result_json.open(encoding="utf-8") as file:
        result = json.load(file)
    if not isinstance(result, dict):
        raise ValueError("Benchmark result JSON must contain one object.")

    return generate_benchmark_plots(
        result=result,
        output_dir=output_dir or result_json.parent,
        prefix=prefix or result_json.stem,
        plots=plots,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate static plots from one detailed benchmark result."
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        required=True,
        help="Detailed benchmark result JSON generated by vllm bench serve.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to the result JSON directory.",
    )
    parser.add_argument(
        "--prefix",
        help="Output filename prefix. Defaults to the result JSON filename stem.",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=("trace", "input", "output", "input-output"),
        default=("trace", "input", "output", "input-output"),
        help="Plots to generate. Defaults to all four plots.",
    )
    args = parser.parse_args()
    created = generate_benchmark_plots_from_file(
        result_json=args.result_json,
        output_dir=args.output_dir,
        prefix=args.prefix,
        plots=args.plots,
    )
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
