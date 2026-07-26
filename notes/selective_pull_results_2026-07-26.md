# Lossless selective KV pull — TPU results, 2026-07-26

## What changed

Upstream TPU PD always transferred every prompt KV block, even when decode's
prefix cache already contained most of them. Its source included a TODO saying
a side channel was needed before prefill could prepare only the non-hit KV.

The new protocol implements that side channel:

- prefill defers gather and transfer registration;
- decode reports the uncached block suffix after allocation;
- prefill validates and gathers only that suffix;
- decode pulls and inserts only those blocks;
- full hits release producer state without a transfer;
- transfer shapes are power-of-two bucketed to bound JAX recompilation;
- spawned EngineCore processes restore the `--device-ids` TPU affinity before
  their first JAX device query.

This protocol contains no model-name checks, no Qwen cache-layout changes, and
no lossy KV representation.

## Test configuration

- TPU: v6e-4, prefill on chip 0 and decode on chip 1 of the same VM
- Model: `Qwen/Qwen3-4B`
- Model revision: `1cfa9a7208912126459214e8b04321603b3df60c`
- Runtime: `vllm-tpu==0.25.0`, `tpu-inference==0.25.0`
- KV cache: BF16, unchanged between control and experiment
- Block size: 128 tokens
- Input: 2,048 tokens
- Cached prefix: 1,536 tokens / 12 blocks / 75%
- Output: 16 tokens, temperature 0, fixed seeds
- Method: each base prompt populated the cache, then a variant reused its
  first 1,536 tokens and changed the final 512
- Warm E2E statistic: median of pairs 2–4; pair 1 was excluded because it
  included new JAX compilation
- Connector statistic: median of all four variant pulls

## Matched result

| Metric | Stock full pull | Selective pull | Improvement |
| --- | ---: | ---: | ---: |
| Blocks requested | 16 | 4 | 4.00x fewer |
| KV payload | 288 MiB | 72 MiB | 4.00x less |
| Median KV pull | 17.84 ms | 5.49 ms | 3.25x faster |
| Warm median end-to-end | 190.888 ms | 180.908 ms | 5.23% lower |
| Deterministic outputs | 4/4 baseline matches | 4/4 baseline matches | Exact |

The exact response records are in
`benchmarks/results/selective-pull/control-2048.json` and
`benchmarks/results/selective-pull/optimized-2048.json`.

The 512-token check showed the same direction:

| Metric | Stock full pull | Selective pull |
| --- | ---: | ---: |
| KV payload | 72 MiB | 18 MiB |
| Median KV pull | ~6.65 ms | ~2.24 ms |
| Warm median end-to-end | 169.844 ms | 161.505 ms |

## Is a 75% prefix hit realistic?

It is realistic for a high-reuse PD workload, not for arbitrary one-shot
traffic. Expected token-prefix reuse depends heavily on workload and routing:

- unrelated one-shot requests: roughly 0–10%;
- general multi-turn chat with sticky routing: roughly 40–70%;
- long chats, coding agents, or repeated tool loops: roughly 60–90%;
- a mixed production fleet with cache-aware routing: use roughly 30–60% as a
  conservative first model.

Seventy-five percent is therefore a strong but credible scenario for agentic
or long-conversation traffic. Without sticky or cache-aware routing, the same
logical reuse will not become a physical decode-cache hit.

The benefit scales approximately with the cached fraction at the connector:
a 50% hit avoids about half the unpadded KV bytes, while a 75% hit avoids about
three quarters. Power-of-two transfer bucketing can make the exact wire ratio
stepwise.

## Correctness and provenance

- `42` connector tests passed on the TPU VM's vLLM/JAX environment.
- All optimized benchmark outputs matched stock token-for-token.
- A separate transfer-only INT4 experiment was rejected and removed because it
  changed outputs and was slower on the same-host path.
- The final branch starts at upstream tag `v0.25.0` and contains only:
  `afac0f54`, `69ba34f4`, and `b77ea5fb`.
- A fresh fetch and search across upstream remote refs found no implementation
  of this TPU selective-pull protocol.

The live stack was left running with `tpu_kv_selective_pull=true` and prefix
caching enabled.
