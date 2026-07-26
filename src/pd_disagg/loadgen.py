"""Streaming OpenAI-compatible load generator with raw per-request timings."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import math
import random
import statistics
import time
import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

SCHEMA_VERSION = 1
TOKEN_SEED_TEXT = " ".join(
    f"measurement token{i % 997} context{i % 389}" for i in range(4096)
)


@dataclasses.dataclass(frozen=True)
class RequestSpec:
    request_index: int
    scheduled_offset_s: float
    input_token_ids: tuple[int, ...]
    output_length: int
    warmup: bool = False


@dataclasses.dataclass(frozen=True)
class RunConfig:
    endpoint: str
    model: str
    input_lengths: tuple[int, ...]
    output_lengths: tuple[int, ...]
    request_count: int
    arrival_rate: float
    max_concurrency: int
    warmup_requests: int
    seed: int
    timeout_s: float

    def validate(self) -> None:
        if not self.input_lengths or min(self.input_lengths) < 1:
            raise ValueError("input lengths must all be positive")
        if not self.output_lengths or min(self.output_lengths) < 1:
            raise ValueError("output lengths must all be positive")
        if self.request_count < 1:
            raise ValueError("request count must be positive")
        if not math.isfinite(self.arrival_rate) or self.arrival_rate <= 0:
            raise ValueError("arrival rate must be finite and positive")
        if self.max_concurrency < 1:
            raise ValueError("max concurrency must be positive")
        if self.warmup_requests < 0:
            raise ValueError("warmup request count cannot be negative")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_lengths(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def poisson_offsets(
    request_count: int,
    arrival_rate: float,
    rng: random.Random,
) -> list[float]:
    offsets: list[float] = []
    offset = 0.0
    for request_index in range(request_count):
        if request_index:
            offset += rng.expovariate(arrival_rate)
        offsets.append(offset)
    return offsets


def make_specs(
    config: RunConfig,
    token_pool: Sequence[int],
) -> tuple[list[RequestSpec], list[RequestSpec]]:
    if not token_pool:
        raise ValueError("token pool cannot be empty")
    rng = random.Random(config.seed)

    def make_one(index: int, offset: float, warmup: bool) -> RequestSpec:
        input_length = rng.choice(config.input_lengths)
        output_length = rng.choice(config.output_lengths)
        tokens = tuple(rng.choice(token_pool) for _ in range(input_length))
        return RequestSpec(
            request_index=index,
            scheduled_offset_s=offset,
            input_token_ids=tokens,
            output_length=output_length,
            warmup=warmup,
        )

    warmups = [
        make_one(index, 0.0, True)
        for index in range(-config.warmup_requests, 0)
    ]
    offsets = poisson_offsets(config.request_count, config.arrival_rate, rng)
    measured = [
        make_one(index, offset, False) for index, offset in enumerate(offsets)
    ]
    return warmups, measured


async def fetch_token_pool(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
) -> list[int]:
    response = await client.post(
        f"{endpoint.rstrip('/')}/tokenize",
        json={"model": model, "prompt": TOKEN_SEED_TEXT},
    )
    response.raise_for_status()
    payload = response.json()
    tokens = payload.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise RuntimeError(f"tokenize response did not contain tokens: {payload}")
    return [int(token) for token in tokens]


def sse_payload(line: str) -> dict[str, Any] | None:
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None
    return json.loads(data)


async def execute_request(
    *,
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    run_id: str,
    spec: RequestSpec,
    run_start: float,
    semaphore: asyncio.Semaphore,
    seed: int,
) -> dict[str, Any]:
    scheduled_at = run_start + spec.scheduled_offset_s
    await asyncio.sleep(max(0.0, scheduled_at - time.perf_counter()))
    semaphore_wait_start = time.perf_counter()

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "request_id": f"{run_id}-{spec.request_index}",
        "request_index": spec.request_index,
        "warmup": spec.warmup,
        "included_in_summary": not spec.warmup,
        "scheduled_offset_s": spec.scheduled_offset_s,
        "input_length_requested": len(spec.input_token_ids),
        "output_length_requested": spec.output_length,
        "prompt_token_sha256": hashlib.sha256(
            ",".join(map(str, spec.input_token_ids)).encode()
        ).hexdigest(),
        "client_start_time": utc_now(),
        "status": "started",
        "error": None,
        "http_status": None,
        "client_queue_s": None,
        "pd_prefill_s": None,
        "ttft_s": None,
        "total_s": None,
        "token_event_offsets_s": [],
        "itl_s": [],
        "stream_text_chunk_count": 0,
        "prompt_tokens_server": None,
        "completion_tokens_server": None,
        "total_tokens_server": None,
        "itl_valid": False,
    }

    async with semaphore:
        request_start = time.perf_counter()
        record["client_queue_s"] = request_start - semaphore_wait_start
        token_event_offsets: list[float] = []
        usage: dict[str, Any] | None = None
        payload = {
            "model": model,
            "prompt": list(spec.input_token_ids),
            "max_tokens": spec.output_length,
            "temperature": 0,
            "seed": seed + (
                1_000_000 + abs(spec.request_index)
                if spec.warmup
                else spec.request_index
            ),
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        try:
            async with client.stream(
                "POST",
                f"{endpoint.rstrip('/')}/v1/completions",
                json=payload,
                headers={"X-PD-Request-ID": record["request_id"]},
            ) as response:
                record["http_status"] = response.status_code
                prefill_ms = response.headers.get("x-pd-prefill-ms")
                record["pd_prefill_s"] = (
                    float(prefill_ms) / 1000 if prefill_ms is not None else None
                )
                response.raise_for_status()
                async for line in response.aiter_lines():
                    event = sse_payload(line)
                    if event is None:
                        continue
                    event_usage = event.get("usage")
                    if isinstance(event_usage, dict):
                        usage = event_usage
                    choices = event.get("choices")
                    if not choices:
                        continue
                    text = choices[0].get("text", "")
                    if text:
                        token_event_offsets.append(
                            time.perf_counter() - request_start
                        )

            request_end = time.perf_counter()
            record["status"] = "ok"
            record["total_s"] = request_end - request_start
            record["token_event_offsets_s"] = token_event_offsets
            record["stream_text_chunk_count"] = len(token_event_offsets)
            if token_event_offsets:
                record["ttft_s"] = token_event_offsets[0]
                record["itl_s"] = [
                    current - previous
                    for previous, current in zip(
                        token_event_offsets,
                        token_event_offsets[1:],
                        strict=False,
                    )
                ]

            if usage:
                record["prompt_tokens_server"] = usage.get("prompt_tokens")
                record["completion_tokens_server"] = usage.get("completion_tokens")
                record["total_tokens_server"] = usage.get("total_tokens")
                record["itl_valid"] = (
                    usage.get("completion_tokens") == len(token_event_offsets)
                )
        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["total_s"] = time.perf_counter() - request_start

    record["client_end_time"] = utc_now()
    return record


def percentile(samples: Sequence[float], quantile: float) -> float:
    if not samples:
        raise ValueError("cannot compute percentile of empty sequence")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def metric_summary(samples: Iterable[float]) -> dict[str, float | int] | None:
    values = list(samples)
    if not values:
        return None
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    included = [record for record in records if record["included_in_summary"]]
    successful = [record for record in included if record["status"] == "ok"]
    return {
        "request_count": len(included),
        "success_count": len(successful),
        "error_count": len(included) - len(successful),
        "ttft_s": metric_summary(
            record["ttft_s"]
            for record in successful
            if record["ttft_s"] is not None
        ),
        "total_s": metric_summary(
            record["total_s"]
            for record in successful
            if record["total_s"] is not None
        ),
        "itl_s": metric_summary(
            latency
            for record in successful
            if record["itl_valid"]
            for latency in record["itl_s"]
        ),
        "itl_valid_request_count": sum(
            bool(record["itl_valid"]) for record in successful
        ),
    }


async def run(config: RunConfig, output: Path) -> dict[str, Any]:
    config.validate()
    output.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    timeout = httpx.Timeout(config.timeout_s, connect=30.0)
    limits = httpx.Limits(
        max_connections=config.max_concurrency,
        max_keepalive_connections=config.max_concurrency,
    )

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        token_pool = await fetch_token_pool(
            client,
            config.endpoint,
            config.model,
        )
        warmups, measured = make_specs(config, token_pool)
        semaphore = asyncio.Semaphore(config.max_concurrency)

        warmup_records: list[dict[str, Any]] = []
        for spec in warmups:
            warmup_records.append(
                await execute_request(
                    client=client,
                    endpoint=config.endpoint,
                    model=config.model,
                    run_id=run_id,
                    spec=spec,
                    run_start=time.perf_counter(),
                    semaphore=semaphore,
                    seed=config.seed,
                )
            )

        run_start = time.perf_counter()
        measured_records = await asyncio.gather(
            *[
                execute_request(
                    client=client,
                    endpoint=config.endpoint,
                    model=config.model,
                    run_id=run_id,
                    spec=spec,
                    run_start=run_start,
                    semaphore=semaphore,
                    seed=config.seed,
                )
                for spec in measured
            ]
        )

    records = sorted(
        [*warmup_records, *measured_records],
        key=lambda record: record["request_index"],
    )
    output.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": utc_now(),
        "config": dataclasses.asdict(config),
        "token_pool_size": len(token_pool),
        "raw_records": str(output),
        "summary": summarize(records),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--endpoint", default="http://127.0.0.1:8000")
    result.add_argument("--model", default="Qwen/Qwen3-4B")
    result.add_argument("--input-lengths", type=parse_lengths, default=(128,))
    result.add_argument("--output-lengths", type=parse_lengths, default=(32,))
    result.add_argument("--requests", type=int, default=10)
    result.add_argument("--arrival-rate", type=float, default=1.0)
    result.add_argument("--max-concurrency", type=int, default=64)
    result.add_argument("--warmup-requests", type=int, default=1)
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--timeout-s", type=float, default=600.0)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    config = RunConfig(
        endpoint=args.endpoint,
        model=args.model,
        input_lengths=args.input_lengths,
        output_lengths=args.output_lengths,
        request_count=args.requests,
        arrival_rate=args.arrival_rate,
        max_concurrency=args.max_concurrency,
        warmup_requests=args.warmup_requests,
        seed=args.seed,
        timeout_s=args.timeout_s,
    )
    summary = asyncio.run(run(config, args.output))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
