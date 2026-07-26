from __future__ import annotations

import pytest

from pd_disagg.router import (
    create_app,
    make_decode_payload,
    make_prefill_payload,
    validate_request,
)


def test_pd_payloads_preserve_original_request() -> None:
    original = {
        "model": "model",
        "prompt": [1, 2, 3],
        "max_tokens": 17,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    prefill = make_prefill_payload(original)
    decode = make_decode_payload(original, {"uuid": 9})
    local_decode = make_decode_payload(original, None)

    assert prefill["max_tokens"] == 1
    assert prefill["stream"] is False
    assert "stream_options" not in prefill
    assert decode["max_tokens"] == 17
    assert decode["stream"] is True
    assert decode["kv_transfer_params"] == {"uuid": 9}
    assert "kv_transfer_params" not in local_decode
    assert "kv_transfer_params" not in original


def test_local_decode_fallback_removes_stale_transfer_metadata() -> None:
    original = {
        "model": "model",
        "prompt": [1, 2, 3],
        "max_tokens": 4,
        "kv_transfer_params": {"uuid": 9},
    }

    decode = make_decode_payload(original, {})

    assert "kv_transfer_params" not in decode
    assert original["kv_transfer_params"] == {"uuid": 9}


def test_pd_router_rejects_semantically_unsupported_batching() -> None:
    with pytest.raises(ValueError):
        validate_request({"prompt": [[1, 2], [3, 4]], "max_tokens": 4})
    with pytest.raises(ValueError):
        validate_request({"prompt": [1, 2], "max_tokens": 4, "n": 2})


def test_pd_router_requires_paired_worker_counts() -> None:
    with pytest.raises(ValueError):
        create_app(
            prefill_endpoint="http://p0,http://p1",
            decode_endpoint="http://d0",
        )
