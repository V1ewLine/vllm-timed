# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.benchmarks.datasets import TimedTraceDataset


class FakeTokenizer:
    vocab_size = 128
    all_special_ids = [0, 1, 2]
    added_tokens_decoder = {}

    def decode(self, token_ids, **kwargs):
        return " ".join(str(token_id) for token_id in token_ids)

    def encode(self, text, **kwargs):
        if not text:
            return []
        return [int(token_id) for token_id in text.split()]

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=self.encode(text))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.mark.benchmark
def test_timed_trace_dataset_replays_hash_chunks_and_timestamps(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    _write_jsonl(
        trace_path,
        [
            {
                "hash_ids": [7, 8, 9],
                "input_length": 5,
                "output_length": 2,
                "timestamp": 10.0,
            },
            {
                "hash_ids": [7, 8, 10],
                "input_length": 5,
                "output_length": 3,
                "timestamp": 12.5,
            },
        ],
    )

    dataset = TimedTraceDataset(dataset_path=str(trace_path))
    samples = dataset.sample(
        tokenizer=FakeTokenizer(),
        num_requests=2,
        chunk_hash_size=2,
        sec_multiplier=0.5,
        request_id_prefix="trace-",
    )

    assert len(samples) == 2
    assert samples[0].prompt_len == 5
    assert samples[0].expected_output_len == 2
    assert samples[0].request_id == "trace-0"
    assert samples[0].arrival_time == 0.0
    assert samples[1].arrival_time == 1.25

    assert isinstance(samples[0].prompt, list)
    assert isinstance(samples[1].prompt, list)
    assert len(samples[0].prompt) == 5
    assert len(samples[1].prompt) == 5
    assert samples[0].prompt[:2] == samples[1].prompt[:2]
    assert samples[0].prompt[2:4] == samples[1].prompt[2:4]


@pytest.mark.benchmark
def test_timed_trace_dataset_compresses_idle_gaps(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    _write_jsonl(
        trace_path,
        [
            {
                "hash_ids": [1],
                "input_length": 1,
                "output_length": 1,
                "timestamp": 10.0,
            },
            {
                "hash_ids": [2],
                "input_length": 1,
                "output_length": 1,
                "timestamp": 30.0,
            },
            {
                "hash_ids": [3],
                "input_length": 1,
                "output_length": 1,
                "timestamp": 150.0,
            },
        ],
    )

    dataset = TimedTraceDataset(dataset_path=str(trace_path))
    samples = dataset.sample(
        tokenizer=FakeTokenizer(),
        num_requests=3,
        chunk_hash_size=1,
        sec_multiplier=0.5,
        idle_gap_threshold=60.0,
        idle_sec_multiplier=0.1,
    )

    assert [sample.arrival_time for sample in samples] == pytest.approx(
        [0.0, 10.0, 46.0]
    )


@pytest.mark.benchmark
def test_timed_trace_idle_threshold_uses_regular_multiplier_by_default(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    _write_jsonl(
        trace_path,
        [
            {
                "hash_ids": [1],
                "input_length": 1,
                "output_length": 1,
                "timestamp": 0.0,
            },
            {
                "hash_ids": [2],
                "input_length": 1,
                "output_length": 1,
                "timestamp": 120.0,
            },
        ],
    )

    dataset = TimedTraceDataset(dataset_path=str(trace_path))
    samples = dataset.sample(
        tokenizer=FakeTokenizer(),
        num_requests=2,
        chunk_hash_size=1,
        sec_multiplier=0.5,
        idle_gap_threshold=60.0,
    )

    assert samples[1].arrival_time == 60.0


@pytest.mark.benchmark
def test_timed_trace_dataset_rejects_unsorted_timestamps(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    _write_jsonl(
        trace_path,
        [
            {
                "hash_ids": [1],
                "input_length": 1,
                "output_length": 1,
                "timestamp": 2.0,
            },
            {
                "hash_ids": [2],
                "input_length": 1,
                "output_length": 1,
                "timestamp": 1.0,
            },
        ],
    )

    dataset = TimedTraceDataset(dataset_path=str(trace_path))
    with pytest.raises(ValueError, match="timestamp is not non-decreasing"):
        dataset.sample(
            tokenizer=FakeTokenizer(),
            num_requests=2,
            chunk_hash_size=1,
        )


@pytest.mark.benchmark
def test_timed_trace_dataset_can_return_text_prompt(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    _write_jsonl(
        trace_path,
        [
            {
                "hash_ids": [11, 12],
                "input_length": 4,
                "output_length": 2,
                "timestamp": 0.0,
            },
        ],
    )

    dataset = TimedTraceDataset(dataset_path=str(trace_path))
    samples = dataset.sample(
        tokenizer=FakeTokenizer(),
        num_requests=1,
        chunk_hash_size=2,
        return_token_ids=False,
    )

    assert isinstance(samples[0].prompt, str)
    assert samples[0].prompt_len == 4
    assert len(FakeTokenizer().encode(samples[0].prompt)) == 4
