"""Minimal measured prefill/decode router for OpenAI-compatible vLLM servers."""

from __future__ import annotations

import argparse
import asyncio
import copy
import itertools
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def validate_request(payload: dict[str, Any]) -> None:
    prompt = payload.get("prompt")
    if not isinstance(prompt, (str, list)):
        raise ValueError("PD router requires one string or token-ID-list prompt")
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], list):
        raise ValueError("PD router does not support batched prompts")
    if int(payload.get("n", 1)) != 1:
        raise ValueError("PD router requires n=1")
    if int(payload.get("best_of", 1)) != 1:
        raise ValueError("PD router requires best_of=1")


def make_prefill_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    result["max_tokens"] = 1
    result["ignore_eos"] = True
    result["stream"] = False
    result.pop("stream_options", None)
    result.pop("kv_transfer_params", None)
    return result


def make_decode_payload(
    payload: dict[str, Any],
    kv_transfer_params: dict[str, Any] | None,
) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    if kv_transfer_params:
        result["kv_transfer_params"] = kv_transfer_params
    else:
        # vLLM intentionally omits transfer metadata when the prompt has no
        # complete KV block. Let the decode worker prefill locally instead of
        # failing an otherwise valid request.
        result.pop("kv_transfer_params", None)
    return result


class JsonlRecorder:
    def __init__(self, path: Path):
        self.path = path
        self.lock = asyncio.Lock()

    async def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":")) + "\n"
        async with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as output:
                output.write(line)


def create_app(
    *,
    prefill_endpoint: str | None = None,
    decode_endpoint: str | None = None,
    record_path: Path | None = None,
) -> FastAPI:
    prefill_value = (
        prefill_endpoint
        or os.environ.get("PREFILL_ENDPOINT", "http://10.128.0.2:8000")
    )
    decode_value = (
        decode_endpoint
        or os.environ.get("DECODE_ENDPOINT", "http://10.128.0.4:8000")
    )
    prefill_urls = [
        endpoint.strip().rstrip("/")
        for endpoint in prefill_value.split(",")
        if endpoint.strip()
    ]
    decode_urls = [
        endpoint.strip().rstrip("/")
        for endpoint in decode_value.split(",")
        if endpoint.strip()
    ]
    if not prefill_urls or len(prefill_urls) != len(decode_urls):
        raise ValueError(
            "PREFILL_ENDPOINT and DECODE_ENDPOINT must contain the same "
            "non-zero number of comma-separated endpoints"
        )
    recorder = JsonlRecorder(
        record_path
        or Path(
            os.environ.get(
                "PD_ROUTER_RECORDS",
                str(Path.home() / "pd-results" / "router.jsonl"),
            )
        )
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        timeout = httpx.Timeout(600.0, connect=30.0)
        limits = httpx.Limits(max_connections=256, max_keepalive_connections=128)
        app.state.prefill_client = httpx.AsyncClient(timeout=timeout, limits=limits)
        app.state.decode_client = httpx.AsyncClient(timeout=timeout, limits=limits)
        app.state.pair_iterator = itertools.cycle(range(len(prefill_urls)))
        app.state.pair_lock = asyncio.Lock()
        yield
        await app.state.prefill_client.aclose()
        await app.state.decode_client.aclose()

    app = FastAPI(title="TPU PD measurement router", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        async def check(client: httpx.AsyncClient, endpoint: str) -> bool:
            try:
                response = await client.get(f"{endpoint}/health")
                return response.is_success
            except httpx.HTTPError:
                return False

        statuses = await asyncio.gather(
            *[
                check(app.state.prefill_client, endpoint)
                for endpoint in prefill_urls
            ],
            *[
                check(app.state.decode_client, endpoint)
                for endpoint in decode_urls
            ],
        )
        if not all(statuses):
            raise HTTPException(
                status_code=503,
                detail={
                    "prefill": statuses[: len(prefill_urls)],
                    "decode": statuses[len(prefill_urls) :],
                },
            )
        return {
            "prefill": len(prefill_urls),
            "decode": len(decode_urls),
        }

    @app.post("/tokenize")
    async def tokenize(request: Request) -> Response:
        response = await app.state.prefill_client.post(
            f"{prefill_urls[0]}/tokenize",
            content=await request.body(),
            headers={"content-type": "application/json"},
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
        )

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        try:
            payload = await request.json()
            validate_request(payload)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        request_id = request.headers.get("x-pd-request-id", uuid.uuid4().hex)
        async with app.state.pair_lock:
            pair_index = next(app.state.pair_iterator)
        prefill_url = prefill_urls[pair_index]
        decode_url = decode_urls[pair_index]
        started = time.perf_counter()
        record: dict[str, Any] = {
            "schema_version": 1,
            "request_id": request_id,
            "pair_index": pair_index,
            "created_at": utc_now(),
            "prefill_endpoint": prefill_url,
            "decode_endpoint": decode_url,
            "stream": bool(payload.get("stream", False)),
            "input_length": (
                len(payload["prompt"])
                if isinstance(payload.get("prompt"), list)
                else None
            ),
            "output_length": payload.get("max_tokens"),
            "status": "started",
            "prefill_s": None,
            "decoder_headers_s": None,
            "first_decoder_byte_s": None,
            "total_s": None,
            "kv_transfer": None,
            "error": None,
        }

        prefill_started = time.perf_counter()
        try:
            prefill_response = await app.state.prefill_client.post(
                f"{prefill_url}/v1/completions",
                json=make_prefill_payload(payload),
                headers={
                    "X-PD-Request-ID": request_id,
                    "X-Request-Id": request_id,
                },
            )
            record["prefill_s"] = time.perf_counter() - prefill_started
            prefill_response.raise_for_status()
            prefill_result = prefill_response.json()
            kv_params = prefill_result.get("kv_transfer_params")
            if kv_params is not None and not isinstance(kv_params, dict):
                raise RuntimeError(
                    "prefill response contained invalid kv_transfer_params"
                )

            decode_payload = make_decode_payload(payload, kv_params)
            record["kv_transfer"] = "remote" if kv_params else "local_fallback"
            if not payload.get("stream", False):
                decode_started = time.perf_counter()
                decode_response = await app.state.decode_client.post(
                    f"{decode_url}/v1/completions",
                    json=decode_payload,
                    headers={
                        "X-PD-Request-ID": request_id,
                        "X-Request-Id": request_id,
                    },
                )
                record["decoder_headers_s"] = time.perf_counter() - decode_started
                decode_response.raise_for_status()
                record["status"] = "ok"
                record["total_s"] = time.perf_counter() - started
                await recorder.write(record)
                return JSONResponse(
                    content=decode_response.json(),
                    status_code=decode_response.status_code,
                    headers={
                        "X-PD-Prefill-Ms": f"{record['prefill_s'] * 1000:.6f}",
                        "X-PD-KV-Transfer": record["kv_transfer"],
                    },
                )

            decode_started = time.perf_counter()
            decode_request = app.state.decode_client.build_request(
                "POST",
                f"{decode_url}/v1/completions",
                json=decode_payload,
                headers={
                    "X-PD-Request-ID": request_id,
                    "X-Request-Id": request_id,
                },
            )
            decode_response = await app.state.decode_client.send(
                decode_request,
                stream=True,
            )
            record["decoder_headers_s"] = time.perf_counter() - decode_started
            decode_response.raise_for_status()
        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["total_s"] = time.perf_counter() - started
            await recorder.write(record)
            raise HTTPException(status_code=502, detail=record["error"]) from exc

        async def relay() -> AsyncIterator[bytes]:
            try:
                async for chunk in decode_response.aiter_raw():
                    if record["first_decoder_byte_s"] is None and chunk:
                        record["first_decoder_byte_s"] = time.perf_counter() - started
                    yield chunk
                record["status"] = "ok"
            except Exception as exc:
                record["status"] = "error"
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                await decode_response.aclose()
                record["total_s"] = time.perf_counter() - started
                await recorder.write(record)

        return StreamingResponse(
            relay(),
            status_code=decode_response.status_code,
            media_type="text/event-stream",
            headers={
                "X-PD-Prefill-Ms": f"{record['prefill_s'] * 1000:.6f}",
                "X-PD-KV-Transfer": record["kv_transfer"],
                "Cache-Control": "no-cache",
            },
        )

    return app


app = create_app()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--host", default="0.0.0.0")
    result.add_argument("--port", type=int, default=9000)
    return result


def main() -> None:
    args = parser().parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
