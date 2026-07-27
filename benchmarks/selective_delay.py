"""Delay decode after prefill to exercise TPU PD transfer expiration."""

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


def make_prompt(source: dict[str, Any], case_id: int, length: int) -> list[int]:
    source_tokens = source["cases"][case_id % len(source["cases"])][
        "prompt_token_ids"
    ]
    repeats = (length + len(source_tokens) - 1) // len(source_tokens)
    prompt = (source_tokens * repeats)[:length]
    prompt[0] = 30_000 + case_id
    return prompt


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.prompts.read_text(encoding="utf-8"))
    prompt = make_prompt(source, args.case_id, args.input_length)
    payload = {
        "model": source["model"],
        "prompt": prompt,
        "max_tokens": args.output_length,
        "ignore_eos": True,
        "temperature": 0,
        "seed": 90_000 + args.case_id,
        "return_token_ids": True,
    }
    prefill_payload = dict(payload)
    prefill_payload["max_tokens"] = 1
    request_id = f"stress-delay-{args.case_id}"
    headers = {
        "X-PD-Request-ID": request_id,
        "X-Request-Id": request_id,
    }
    timeout = httpx.Timeout(600.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        prefill_started = time.perf_counter()
        prefill_response = await client.post(
            f"{args.prefill_endpoint.rstrip('/')}/v1/completions",
            json=prefill_payload,
            headers=headers,
        )
        prefill_elapsed = time.perf_counter() - prefill_started
        prefill_response.raise_for_status()
        kv_transfer_params = prefill_response.json().get("kv_transfer_params")
        if not isinstance(kv_transfer_params, dict):
            raise RuntimeError("prefill did not return KV transfer metadata")

        await asyncio.sleep(args.delay)

        decode_payload = dict(payload)
        decode_payload["kv_transfer_params"] = kv_transfer_params
        decode_started = time.perf_counter()
        try:
            decode_response = await client.post(
                f"{args.decode_endpoint.rstrip('/')}/v1/completions",
                json=decode_payload,
                headers=headers,
            )
            decode_elapsed = time.perf_counter() - decode_started
            try:
                body: Any = decode_response.json()
            except ValueError:
                body = decode_response.text
            token_ids = (
                [int(token) for token in body["choices"][0]["token_ids"]]
                if decode_response.is_success and isinstance(body, dict)
                else None
            )
            result = {
                "decode_status": decode_response.status_code,
                "decode_elapsed_s": decode_elapsed,
                "decode_output_token_ids": token_ids,
                "decode_output_sha256": (
                    token_hash(token_ids) if token_ids is not None else None
                ),
            }
            if not decode_response.is_success:
                result["decode_error_body"] = body
        except Exception as exc:
            result = {
                "decode_status": None,
                "decode_elapsed_s": time.perf_counter() - decode_started,
                "decode_error": f"{type(exc).__name__}: {exc}",
            }

        if args.verify_local:
            local_started = time.perf_counter()
            local_response = await client.post(
                f"{args.decode_endpoint.rstrip('/')}/v1/completions",
                json=payload,
                headers={
                    "X-PD-Request-ID": f"{request_id}-local",
                    "X-Request-Id": f"{request_id}-local",
                },
            )
            local_elapsed = time.perf_counter() - local_started
            local_response.raise_for_status()
            local_tokens = [
                int(token)
                for token in local_response.json()["choices"][0]["token_ids"]
            ]
            result.update({
                "local_status": local_response.status_code,
                "local_elapsed_s": local_elapsed,
                "local_output_token_ids": local_tokens,
                "local_output_sha256": token_hash(local_tokens),
                "matches_local": (
                    result.get("decode_output_token_ids") == local_tokens
                ),
            })

    return {
        "case_id": args.case_id,
        "delay_s": args.delay,
        "prefill_status": prefill_response.status_code,
        "prefill_elapsed_s": prefill_elapsed,
        "uuid": kv_transfer_params["uuid"],
        **result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-endpoint", required=True)
    parser.add_argument("--decode-endpoint", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", type=int, required=True)
    parser.add_argument("--delay", type=float, required=True)
    parser.add_argument("--input-length", type=int, default=512)
    parser.add_argument("--output-length", type=int, default=16)
    parser.add_argument("--verify-local", action="store_true")
    args = parser.parse_args()

    result = asyncio.run(main_async(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
