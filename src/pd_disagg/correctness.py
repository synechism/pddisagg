"""Capture and verify deterministic token-level serving references."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import httpx

from pd_disagg.loadgen import TOKEN_SEED_TEXT, parse_lengths, utc_now


async def token_pool(client: httpx.AsyncClient, endpoint: str, model: str) -> list[int]:
    response = await client.post(
        f"{endpoint.rstrip('/')}/tokenize",
        json={"model": model, "prompt": TOKEN_SEED_TEXT},
    )
    response.raise_for_status()
    tokens = response.json().get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise RuntimeError("tokenizer returned no token IDs")
    return [int(token) for token in tokens]


def output_token_ids(response: dict[str, Any]) -> list[int]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError(f"expected exactly one completion choice: {response}")
    tokens = choices[0].get("token_ids")
    if not isinstance(tokens, list):
        raise RuntimeError(f"completion did not return token IDs: {response}")
    return [int(token) for token in tokens]


async def complete(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    prompt: list[int],
    output_length: int,
    seed: int,
) -> dict[str, Any]:
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
    result = response.json()
    usage = result.get("usage", {})
    if usage.get("prompt_tokens") != len(prompt):
        raise RuntimeError(
            f"prompt token mismatch: expected {len(prompt)}, got {usage}"
        )
    if usage.get("completion_tokens") != output_length:
        raise RuntimeError(
            f"completion token mismatch: expected {output_length}, got {usage}"
        )
    if len(output_token_ids(result)) != output_length:
        raise RuntimeError("returned token ID count does not match requested output")
    return result


async def capture(
    *,
    endpoint: str,
    model: str,
    input_lengths: tuple[int, ...],
    output_length: int,
    seed: int,
) -> dict[str, Any]:
    timeout = httpx.Timeout(600.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        pool = await token_pool(client, endpoint, model)
        rng = random.Random(seed)
        cases = []
        for case_index, input_length in enumerate(input_lengths):
            prompt = [rng.choice(pool) for _ in range(input_length)]
            result = await complete(
                client,
                endpoint,
                model,
                prompt,
                output_length,
                seed + case_index,
            )
            generated = output_token_ids(result)
            cases.append(
                {
                    "case_index": case_index,
                    "input_length": input_length,
                    "output_length": output_length,
                    "prompt_token_ids": prompt,
                    "prompt_sha256": hashlib.sha256(
                        ",".join(map(str, prompt)).encode()
                    ).hexdigest(),
                    "output_token_ids": generated,
                    "output_sha256": hashlib.sha256(
                        ",".join(map(str, generated)).encode()
                    ).hexdigest(),
                }
            )
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "endpoint": endpoint,
        "model": model,
        "seed": seed,
        "cases": cases,
    }


async def verify(endpoint: str, reference: dict[str, Any]) -> dict[str, Any]:
    timeout = httpx.Timeout(600.0, connect=30.0)
    mismatches: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for case in reference["cases"]:
            result = await complete(
                client,
                endpoint,
                reference["model"],
                case["prompt_token_ids"],
                case["output_length"],
                reference["seed"] + case["case_index"],
            )
            actual = output_token_ids(result)
            expected = case["output_token_ids"]
            if actual != expected:
                mismatches.append(
                    {
                        "case_index": case["case_index"],
                        "input_length": case["input_length"],
                        "expected_sha256": case["output_sha256"],
                        "actual_sha256": hashlib.sha256(
                            ",".join(map(str, actual)).encode()
                        ).hexdigest(),
                        "first_mismatch": next(
                            (
                                index
                                for index, (left, right) in enumerate(
                                    zip(expected, actual, strict=False)
                                )
                                if left != right
                            ),
                            None,
                        ),
                    }
                )
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "endpoint": endpoint,
        "case_count": len(reference["cases"]),
        "match_count": len(reference["cases"]) - len(mismatches),
        "all_match": not mismatches,
        "mismatches": mismatches,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--endpoint", required=True)
    capture_parser.add_argument("--model", default="Qwen/Qwen3-4B")
    capture_parser.add_argument(
        "--input-lengths",
        type=parse_lengths,
        default=(16, 17, 128, 511, 512, 2048),
    )
    capture_parser.add_argument("--output-length", type=int, default=32)
    capture_parser.add_argument("--seed", type=int, default=314159)
    capture_parser.add_argument("--output", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--endpoint", required=True)
    verify_parser.add_argument("--reference", type=Path, required=True)
    verify_parser.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "capture":
        result = asyncio.run(
            capture(
                endpoint=args.endpoint,
                model=args.model,
                input_lengths=args.input_lengths,
                output_length=args.output_length,
                seed=args.seed,
            )
        )
    else:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        result = asyncio.run(verify(args.endpoint, reference))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if args.command == "verify" and not result["all_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
