"""Measure lossless prefix-aware PD transfers with paired prompts."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx


def token_hash(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest()


def output_tokens(payload: dict[str, Any]) -> list[int]:
    return [int(token) for token in payload["choices"][0]["token_ids"]]


async def complete(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    prompt: list[int],
    output_length: int,
    seed: int,
) -> tuple[list[int], float]:
    started = time.perf_counter()
    response = await client.post(
        f"{endpoint.rstrip('/')}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": output_length,
            "ignore_eos": True,
            "temperature": 0,
            "seed": seed,
            "return_token_ids": True,
        },
    )
    response.raise_for_status()
    return output_tokens(response.json()), time.perf_counter() - started


async def run(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.prompts.read_text(encoding="utf-8"))
    prompt_cases = source["cases"]
    expected = (
        json.loads(args.expected.read_text(encoding="utf-8"))
        if args.expected
        else None
    )
    results = []
    timeout = httpx.Timeout(600.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for pair_index in range(min(args.pairs, len(prompt_cases) - 1)):
            base = prompt_cases[pair_index]["prompt_token_ids"] * args.repeat
            tail_source = (
                prompt_cases[pair_index + 1]["prompt_token_ids"] * args.repeat
            )
            variant = base[:args.prefix_tokens] + tail_source[args.prefix_tokens:]
            await complete(
                client,
                args.endpoint,
                source["model"],
                base,
                args.output_length,
                args.seed + pair_index * 2,
            )
            generated, elapsed = await complete(
                client,
                args.endpoint,
                source["model"],
                variant,
                args.output_length,
                args.seed + pair_index * 2 + 1,
            )
            expected_tokens = (
                expected["pairs"][pair_index]["output_token_ids"]
                if expected
                else generated
            )
            results.append({
                "pair_index": pair_index,
                "input_tokens": len(variant),
                "prefix_tokens": args.prefix_tokens,
                "output_token_ids": generated,
                "output_sha256": token_hash(generated),
                "matches_expected": generated == expected_tokens,
                "total_s": elapsed,
            })
    return {
        "model": source["model"],
        "endpoint": args.endpoint,
        "pairs": results,
        "all_match": all(result["matches_expected"] for result in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--prefix-tokens", type=int, default=384)
    parser.add_argument("--output-length", type=int, default=16)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=5150)
    args = parser.parse_args()

    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
