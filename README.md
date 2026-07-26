# Generic Prefix-Aware PD Disaggregation on TPU

This repository contains the exact vLLM TPU prefill/decode implementation that
was run on a TPU v6e-4. The main optimization is lossless and model-agnostic:
when the decode worker already owns a cached prompt prefix, the prefill worker
transfers only the uncached KV-block suffix.

The implementation does not contain the separate Qwen-specific cache-layout
optimization investigated earlier.

## Result

Matched Qwen3-4B requests used 2,048 input tokens, a 1,536-token cached prefix
(75% token reuse), and 16 deterministic output tokens.

| Metric | Stock PD | Selective pull | Change |
| --- | ---: | ---: | ---: |
| KV payload | 288 MiB | 72 MiB | 4.00x less |
| Median KV pull | 17.84 ms | 5.49 ms | 3.25x faster |
| Warm median end-to-end | 190.89 ms | 180.91 ms | 5.23% lower |
| Output match | 4/4 | 4/4 | Exact token IDs |

The end-to-end gain is smaller than the connector gain because this experiment
used two chips in one TPU VM, where the link is fast and model execution remains
most of the request. Cross-host PD should make avoided bytes more valuable, but
that has not yet been measured.

## Actual optimized code

- [`implementation/tpu-inference/tpu_inference/distributed/tpu_connector.py`](implementation/tpu-inference/tpu_inference/distributed/tpu_connector.py)
  is the exact connector loaded by the benchmarked vLLM workers.
- [`implementation/tpu-inference/tpu_inference/worker/tpu_worker.py`](implementation/tpu-inference/tpu_inference/worker/tpu_worker.py)
  contains the generic EngineCore TPU-affinity fix used for co-located workers.
- [`implementation/tpu-inference/tests/distributed/test_tpu_connector.py`](implementation/tpu-inference/tests/distributed/test_tpu_connector.py)
  is the exact 42-test connector suite that passed on the TPU VM.
- [`patches/tpu-inference-v0.25`](patches/tpu-inference-v0.25) contains the
  three apply-ready commits relative to upstream tag `v0.25.0`.
- [`scripts/remote/serve_pd_worker.sh`](scripts/remote/serve_pd_worker.sh) is
  the worker launcher used for both prefill and decode.
- [`benchmarks/prefix_selective.py`](benchmarks/prefix_selective.py) is the
  paired-prefix correctness and latency benchmark.
- [`benchmarks/results/selective-pull`](benchmarks/results/selective-pull)
  contains the matched control and optimized response records.

The source snapshots match commit `b77ea5fb` byte-for-byte. The branch contains
only these commits after upstream `v0.25.0`:

```text
afac0f54 Optimize and stabilize TPU PD transfers
69ba34f4 Transfer only uncached KV blocks for prefix hits
b77ea5fb Release producer state on full prefix hits
```

## How selective pull works

1. Prefill computes the prompt normally, but defers KV gather and P2P
   registration.
2. After decode allocates blocks, it calculates which prompt blocks are
   already cached.
3. Decode sends the uncached remote-block suffix over the connector's side
   channel.
4. Prefill validates that request, gathers only those blocks, and registers the
   smaller transfer.
5. Decode pulls and inserts only the missing suffix. A full hit sends an
   immediate release notification and performs no transfer.

No KV values are quantized or approximated. The optimized and stock runs
returned identical output token IDs.

## Apply to upstream

```bash
git clone https://github.com/vllm-project/tpu-inference.git
cd tpu-inference
git checkout v0.25.0
git am /path/to/PD\ Disagg/patches/tpu-inference-v0.25/*.patch
```

Enable the feature in the vLLM connector configuration:

```json
{
  "kv_connector": "TPUConnector",
  "kv_connector_module_path": "tpu_inference.distributed.tpu_connector",
  "kv_role": "kv_consumer",
  "kv_connector_extra_config": {
    "tpu_kv_selective_pull": true
  }
}
```

Both producer and consumer must enable it. The current implementation
automatically disables selective pull for the unsupported Ray multihost and D2H
paths.

See [`notes/selective_pull_results_2026-07-26.md`](notes/selective_pull_results_2026-07-26.md)
for experimental details and cache-hit assumptions.
