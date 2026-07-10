# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.benchmarks.plot import (
    construct_trace_plot_data,
    generate_benchmark_plots,
)


@pytest.mark.benchmark
def test_construct_trace_plot_data_uses_observed_request_times():
    data = construct_trace_plot_data(
        arrival_times=[0.0, 1.0, 2.0],
        start_times=[100.0, 101.5, 103.0],
        latencies=[1.0, 0.5, 2.0],
        successes=[True, False, True],
        bin_width=1.0,
    )

    assert data["scheduled_arrivals"] == [0.0, 1.0, 2.0]
    assert data["observed_starts"] == [0.0, 1.5, 3.0]
    assert data["completion_times"] == [1.0, 5.0]
    assert data["scheduled_rates"] == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert data["observed_rates"] == [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    assert data["completion_rates"] == [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


@pytest.mark.benchmark
def test_construct_trace_plot_data_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        construct_trace_plot_data(
            arrival_times=[0.0],
            start_times=[],
            latencies=[1.0],
            successes=[True],
        )


@pytest.mark.benchmark
def test_generate_benchmark_plots_writes_four_pngs(tmp_path):
    result = {
        "arrival_times": [0.0, 0.5, 1.0, 1.5],
        "start_times": [100.0, 100.6, 101.2, 101.8],
        "latencies": [0.8, 0.7, 0.9, 0.6],
        "successes": [True, True, False, True],
        "input_lens": [128, 512, 2048, 8192],
        "output_lens": [32, 128, 256, 1024],
    }

    paths = generate_benchmark_plots(result, tmp_path, prefix="run")

    assert {path.name for path in paths} == {
        "run.trace.png",
        "run.input_distribution.png",
        "run.output_distribution.png",
        "run.input_output_distribution.png",
    }
    assert all(path.stat().st_size > 0 for path in paths)
