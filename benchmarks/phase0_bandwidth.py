#!/usr/bin/env python3
"""Measure basic TPU HBM and ICI payload bandwidth.

This is a hardware bring-up diagnostic, not the final analytical-model
benchmark. The first invocation of each compiled function is excluded.
Every timed sample is retained in the JSON output.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from functools import partial
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(samples: list[float]) -> dict[str, object]:
    return {
        "samples": samples,
        "median": statistics.median(samples),
        "p90": percentile(samples, 0.90),
        "minimum": min(samples),
        "maximum": max(samples),
    }


def time_bandwidth(
    operation: Callable[[jax.Array], jax.Array],
    value: jax.Array,
    logical_bytes: int,
    repetitions: int,
) -> dict[str, object]:
    # Compilation and first execution are deliberately excluded.
    operation(value).block_until_ready()

    durations: list[float] = []
    bandwidths: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        operation(value).block_until_ready()
        duration = time.perf_counter() - start
        durations.append(duration)
        bandwidths.append(logical_bytes / duration / 1e9)

    return {
        "logical_bytes_per_iteration": logical_bytes,
        "duration_seconds": summarize(durations),
        "payload_gbps": summarize(bandwidths),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mib-per-device", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    devices = jax.local_devices()
    bytes_per_device = args.size_mib_per_device * 1024 * 1024
    elements = bytes_per_device // np.dtype(np.float32).itemsize
    shards = [np.ones(elements, dtype=np.float32) for _ in devices]
    mesh = Mesh(np.array(devices), ("device",))
    sharding = NamedSharding(mesh, P("device"))
    value = jax.device_put(np.stack(shards), sharding)

    @jax.pmap
    def hbm_triad(x: jax.Array) -> jax.Array:
        # One add reads x twice and writes the result once.
        return x + x

    aggregate_hbm_bytes = 3 * bytes_per_device * len(devices)
    ici_result = None
    if len(devices) >= 2:
        ring_permutation = [
            (index, (index + 1) % len(devices)) for index in range(len(devices))
        ]

        @partial(jax.pmap, axis_name="device")
        def ici_ring(x: jax.Array) -> jax.Array:
            return jax.lax.ppermute(x, "device", ring_permutation)

        aggregate_ici_payload_bytes = bytes_per_device * len(devices)
        ici_result = time_bandwidth(
            ici_ring,
            value,
            aggregate_ici_payload_bytes,
            args.repetitions,
        )

    result = {
        "schema_version": 1,
        "timestamp_unix_seconds": time.time(),
        "host": platform.node(),
        "jax_version": jax.__version__,
        "devices": [str(device) for device in devices],
        "device_count": len(devices),
        "size_mib_per_device": args.size_mib_per_device,
        "repetitions": args.repetitions,
        "measurement_notes": {
            "hbm": (
                "Aggregate logical bytes count two operand reads and one result "
                "write per device."
            ),
            "ici": (
                "Aggregate payload bytes count one outgoing buffer per device; "
                "this is not doubled for simultaneous receive traffic."
            ),
        },
        "hbm_triad": time_bandwidth(
            hbm_triad,
            value,
            aggregate_hbm_bytes,
            args.repetitions,
        ),
        "ici_ring": ici_result,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
