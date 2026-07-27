from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

import pd_disagg.router as router_module
from pd_disagg.router import (
    create_app,
    decode_worker_headers,
    make_decode_payload,
    make_prefill_payload,
    should_retry_decode_locally,
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


def test_transfer_failure_retries_with_fresh_worker_request() -> None:
    response = httpx.Response(500)

    assert should_retry_decode_locally(response, {"uuid": 9})
    assert not should_retry_decode_locally(response, None)
    assert not should_retry_decode_locally(httpx.Response(400), {"uuid": 9})
    assert decode_worker_headers("request-1") == {
        "X-PD-Request-ID": "request-1",
        "X-Request-Id": "request-1",
    }
    assert decode_worker_headers("request-1", local_retry=True) == {
        "X-PD-Request-ID": "request-1-local-retry",
        "X-Request-Id": "request-1-local-retry",
    }


def test_non_streaming_transfer_failure_retries_locally(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class FakeAsyncClient:
        def __init__(self, responses):
            self.responses = iter(responses)
            self.calls = []

        async def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return next(self.responses)

        async def aclose(self):
            return None

    request = httpx.Request("POST", "http://worker/v1/completions")
    prefill_client = FakeAsyncClient([
        httpx.Response(
            200,
            request=request,
            json={"kv_transfer_params": {"uuid": 9}},
        )
    ])
    decode_client = FakeAsyncClient([
        httpx.Response(500, request=request, json={"error": "transfer failed"}),
        httpx.Response(
            200,
            request=request,
            json={"choices": [{"token_ids": [1, 2]}]},
        ),
    ])
    clients = iter((prefill_client, decode_client))
    monkeypatch.setattr(
        router_module.httpx,
        "AsyncClient",
        lambda **_kwargs: next(clients),
    )
    app = create_app(
        prefill_endpoint="http://prefill",
        decode_endpoint="http://decode",
        record_path=tmp_path / "router.jsonl",
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            headers={"X-PD-Request-ID": "outer-request"},
            json={
                "model": "model",
                "prompt": [1, 2, 3],
                "max_tokens": 2,
                "stream": False,
            },
        )

    assert response.status_code == 200
    assert response.headers["x-pd-kv-transfer"] == "local_retry"
    assert len(decode_client.calls) == 2
    first_payload = decode_client.calls[0][1]["json"]
    retry_payload = decode_client.calls[1][1]["json"]
    assert first_payload["kv_transfer_params"] == {"uuid": 9}
    assert "kv_transfer_params" not in retry_payload
    assert decode_client.calls[1][1]["headers"]["X-Request-Id"] == (
        "outer-request-local-retry"
    )


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
