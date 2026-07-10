# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.benchmarks.plot import construct_trace_plot_data


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
