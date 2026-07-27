"""Stress selective TPU PD pulls with deterministic concurrent requests."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import httpx


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def token_hash(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest()


def make_cases(
    source: dict[str, Any],
    count: int,
    input_length: int,
    prefix_lengths: list[int],
    start_index: int = 0,
) -> list[dict[str, Any]]:
    source_cases = source["cases"]
    cases = []
    for relative_index in range(count):
        index = start_index + relative_index
        base_source = source_cases[index % len(source_cases)]["prompt_token_ids"]
        tail_source = source_cases[(index + 1) % len(source_cases)][
            "prompt_token_ids"
        ]
        base_repeats = (input_length + len(base_source) - 1) // len(base_source)
        tail_repeats = (input_length + len(tail_source) - 1) // len(tail_source)
        base = (base_source * base_repeats)[:input_length]
        tail = (tail_source * tail_repeats)[:input_length]

        # Give every request family a distinct first block while keeping all
        # tokens inside Qwen's vocabulary.
        base[0] = 1000 + index
        prefix_length = prefix_lengths[index % len(prefix_lengths)]
        if not 0 < prefix_length < input_length:
            raise ValueError("prefix lengths must be inside the prompt")
        variant = base[:prefix_length] + tail[prefix_length:]
        variant[prefix_length] = 2000 + index
        cases.append({
            "case_id": index,
            "prefix_tokens": prefix_length,
            "base": base,
            "variant": variant,
            "seed": 7000 + index,
        })
    return cases


async def complete(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    prompt: list[int],
    output_length: int,
    seed: int,
    request_id: str,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_length,
        "ignore_eos": True,
        "temperature": 0,
        "seed": seed,
        "return_token_ids": True,
    }
    started = time.perf_counter()
    try:
        response = await client.post(
            f"{endpoint.rstrip('/')}/v1/completions",
            json=payload,
            headers={"X-PD-Request-ID": request_id},
        )
        elapsed = time.perf_counter() - started
        response.raise_for_status()
        tokens = [
            int(token) for token in response.json()["choices"][0]["token_ids"]
        ]
        return {
            "status": "ok",
            "elapsed_s": elapsed,
            "prefill_ms": float(response.headers.get("x-pd-prefill-ms", "nan")),
            "output_token_ids": tokens,
            "output_sha256": token_hash(tokens),
        }
    except Exception as exc:
        return {
            "status": "error",
            "elapsed_s": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.prompts.read_text(encoding="utf-8"))
    prefix_lengths = [
        int(value) for value in args.prefix_tokens.split(",") if value
    ]
    cases = make_cases(
        source,
        args.cases,
        args.input_length,
        prefix_lengths,
        args.case_offset,
    )
    expected = (
        json.loads(args.expected.read_text(encoding="utf-8"))
        if args.expected
        else None
    )
    expected_by_id = (
        {
            int(item["case_id"]): item["output_token_ids"]
            for item in expected["results"]
        }
        if expected
        else {}
    )

    timeout = httpx.Timeout(600.0, connect=30.0)
    limits = httpx.Limits(max_connections=256, max_keepalive_connections=128)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        if args.warmup:
            warmup = make_cases(
                source,
                1,
                args.input_length,
                prefix_lengths,
                start_index=10_000,
            )[0]
            for prompt, seed, request_id in (
                (warmup["base"], warmup["seed"] - 1, "stress-warmup-base"),
                (warmup["variant"], warmup["seed"], "stress-warmup-variant"),
            ):
                result = await complete(
                    client,
                    args.endpoint,
                    source["model"],
                    prompt,
                    args.output_length,
                    seed,
                    request_id,
                )
                if result["status"] != "ok":
                    raise RuntimeError(f"failed warmup request: {result}")

        if args.prime:
            for case in cases:
                prime = await complete(
                    client,
                    args.endpoint,
                    source["model"],
                    case["base"],
                    args.output_length,
                    case["seed"] - 1,
                    f"stress-prime-{case['case_id']}",
                )
                if prime["status"] != "ok":
                    raise RuntimeError(
                        f"failed to prime case {case['case_id']}: {prime}"
                    )

        semaphore = asyncio.Semaphore(args.concurrency)
        start = asyncio.Event()

        async def issue(case: dict[str, Any]) -> dict[str, Any]:
            await start.wait()
            async with semaphore:
                result = await complete(
                    client,
                    args.endpoint,
                    source["model"],
                    case["variant"],
                    args.output_length,
                    case["seed"],
                    f"stress-variant-{case['case_id']}",
                )
            result["case_id"] = case["case_id"]
            result["prefix_tokens"] = case["prefix_tokens"]
            expected_tokens = expected_by_id.get(case["case_id"])
            result["matches_expected"] = (
                None
                if expected_tokens is None
                else result.get("output_token_ids") == expected_tokens
            )
            return result

        tasks = [asyncio.create_task(issue(case)) for case in cases]
        await asyncio.sleep(0)
        burst_started = time.perf_counter()
        start.set()
        results = await asyncio.gather(*tasks)
        burst_s = time.perf_counter() - burst_started

    latencies_ms = [
        result["elapsed_s"] * 1000
        for result in results
        if result["status"] == "ok"
    ]
    return {
        "model": source["model"],
        "endpoint": args.endpoint,
        "cases": args.cases,
        "case_offset": args.case_offset,
        "concurrency": args.concurrency,
        "input_tokens": args.input_length,
        "prefix_tokens": prefix_lengths,
        "output_tokens": args.output_length,
        "primed": args.prime,
        "warmed_up": args.warmup,
        "expected_provided": expected is not None,
        "burst_s": burst_s,
        "successful": sum(result["status"] == "ok" for result in results),
        "exact_matches": (
            sum(
                result["status"] == "ok"
                and result["matches_expected"] is True
                for result in results
            )
            if expected is not None
            else None
        ),
        "latency_ms": {
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
            "p99": percentile(latencies_ms, 0.99),
            "max": max(latencies_ms) if latencies_ms else None,
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--input-length", type=int, default=512)
    parser.add_argument("--prefix-tokens", default="384")
    parser.add_argument("--output-length", type=int, default=16)
    parser.add_argument("--prime", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()

    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    if result["successful"] != args.cases:
        raise SystemExit(1)
    if args.expected and result["exact_matches"] != args.cases:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
