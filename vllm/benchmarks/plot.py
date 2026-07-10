# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate plots for benchmark results."""

import math
from pathlib import Path
from typing import Any

from vllm.utils.import_utils import PlaceholderModule

try:
    import plotly.express as px
except ImportError:
    _plotly = PlaceholderModule("plotly")
    px = _plotly.placeholder_attr("express")

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
except ImportError:
    _plotly = PlaceholderModule("plotly")
    go = _plotly.placeholder_attr("graph_objects")
    pio = _plotly.placeholder_attr("io")
    make_subplots = _plotly.placeholder_attr("subplots.make_subplots")

try:
    import matplotlib.pyplot as plt
except ImportError:
    _matplotlib = PlaceholderModule("matplotlib")
    plt = _matplotlib.placeholder_attr("pyplot")


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
    """Generate an HTML plot comparing scheduled and observed request traffic."""
    data = construct_trace_plot_data(
        arrival_times=arrival_times,
        start_times=start_times,
        latencies=latencies,
        successes=successes,
    )
    if not data:
        print("No request trace data to plot")
        return

    colors = {
        "scheduled": "#6B7280",
        "observed": "#2563EB",
        "completed": "#D97706",
    }
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            f"Request rate ({data['bin_width']:g}s buckets)",
            "Cumulative requests",
        ),
    )

    rate_series = [
        ("Scheduled arrivals", "scheduled_rates", "scheduled", "dash"),
        ("Observed starts", "observed_rates", "observed", "solid"),
        ("Successful completions", "completion_rates", "completed", "dot"),
    ]
    for name, field, color, dash in rate_series:
        fig.add_trace(
            go.Scatter(
                x=data["bucket_centers"],
                y=data[field],
                mode="lines",
                name=name,
                legendgroup=name,
                line={"color": colors[color], "width": 2, "dash": dash},
                hovertemplate="Time: %{x:.3f}s<br>Rate: %{y:.3f} req/s<extra></extra>",
            ),
            row=1,
            col=1,
        )

    cumulative_series = [
        ("Scheduled arrivals", "scheduled_arrivals", "scheduled", "dash"),
        ("Observed starts", "observed_starts", "observed", "solid"),
        ("Successful completions", "completion_times", "completed", "dot"),
    ]
    for name, field, color, dash in cumulative_series:
        timestamps = data[field]
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=list(range(1, len(timestamps) + 1)),
                mode="lines",
                name=name,
                legendgroup=name,
                showlegend=False,
                line={"color": colors[color], "width": 2, "dash": dash},
                line_shape="hv",
                hovertemplate="Time: %{x:.3f}s<br>Requests: %{y}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_xaxes(title_text="Seconds since first observed request", row=2, col=1)
    fig.update_yaxes(
        title_text="Requests / second", rangemode="tozero", row=1, col=1
    )
    fig.update_yaxes(title_text="Request count", rangemode="tozero", row=2, col=1)
    fig.update_layout(
        title={
            "text": "Scheduled vs observed request trace",
            "x": 0.02,
            "xanchor": "left",
        },
        template="plotly_white",
        hovermode="x unified",
        height=800,
        legend={"orientation": "h", "y": 1.08, "x": 0},
        margin={"l": 80, "r": 30, "t": 110, "b": 70},
    )
    pio.write_html(fig, str(output_path))
    print(f"Request trace plot saved to: {output_path}")


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
