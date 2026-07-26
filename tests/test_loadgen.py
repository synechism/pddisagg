from __future__ import annotations

import random

import pytest

from pd_disagg.loadgen import (
    RunConfig,
    make_specs,
    parse_lengths,
    percentile,
    poisson_offsets,
    sse_payload,
    summarize,
)


def config(**overrides: object) -> RunConfig:
    values = {
        "endpoint": "http://localhost:8000",
        "model": "model",
        "input_lengths": (8, 16),
        "output_lengths": (4, 8),
        "request_count": 5,
        "arrival_rate": 2.0,
        "max_concurrency": 4,
        "warmup_requests": 1,
        "seed": 7,
        "timeout_s": 60.0,
    }
    values.update(overrides)
    return RunConfig(**values)


def test_specs_are_exact_and_deterministic() -> None:
    warmups_a, measured_a = make_specs(config(), [10, 11, 12])
    warmups_b, measured_b = make_specs(config(), [10, 11, 12])
    assert (warmups_a, measured_a) == (warmups_b, measured_b)
    assert len(warmups_a) == 1
    assert len(measured_a) == 5
    assert all(len(spec.input_token_ids) in (8, 16) for spec in measured_a)
    assert all(spec.output_length in (4, 8) for spec in measured_a)


def test_poisson_offsets_are_monotonic() -> None:
    offsets = poisson_offsets(20, 5.0, random.Random(1))
    assert offsets[0] == 0
    assert offsets == sorted(offsets)
    assert offsets[-1] > 0


def test_sse_parser() -> None:
    assert sse_payload("event: message") is None
    assert sse_payload("data: [DONE]") is None
    assert sse_payload('data: {"choices": []}') == {"choices": []}


def test_percentile_interpolates() -> None:
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([1.0, 2.0], 0.9) == pytest.approx(1.9)


def test_summary_excludes_warmup_and_invalid_itl() -> None:
    records = [
        {
            "included_in_summary": False,
            "status": "ok",
            "ttft_s": 99.0,
            "total_s": 99.0,
            "itl_valid": True,
            "itl_s": [99.0],
        },
        {
            "included_in_summary": True,
            "status": "ok",
            "ttft_s": 1.0,
            "total_s": 2.0,
            "itl_valid": True,
            "itl_s": [0.1, 0.2],
        },
        {
            "included_in_summary": True,
            "status": "ok",
            "ttft_s": 3.0,
            "total_s": 4.0,
            "itl_valid": False,
            "itl_s": [5.0],
        },
    ]
    result = summarize(records)
    assert result["request_count"] == 2
    assert result["ttft_s"]["median"] == 2.0
    assert result["itl_s"]["count"] == 2


def test_validation_rejects_invalid_rate() -> None:
    with pytest.raises(ValueError):
        config(arrival_rate=0).validate()


def test_parse_lengths() -> None:
    assert parse_lengths("1, 4,8") == (1, 4, 8)

