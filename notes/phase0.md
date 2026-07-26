# Phase 0 environment record

## Access inventory

- Project ID: `disagg-503619`
- Active grant observed: 16 on-demand and 16 preemptible TPU v5e pod-slice
  chips in each of `us-central1` and `us-east5`
- Initial zone: `us-central1-a`
- Initial correctness slice: `v5litepod-1` (one TPU v5e chip)
- First ICI slice: `v5litepod-4` (requested after correctness bring-up)
- Runtime image: `v2-alpha-tpuv5-lite`
- Initial correctness model: `Qwen/Qwen3-4B`

## Decisions still requiring measurement or owner approval

- Primary crossover model
- Final input/output sweep grid and SLOs
- Whether the reported DCN tier uses the default VPC MTU or an optimized
  jumbo-frame network

## Phase 0 evidence

- TPU node creation timestamp: `2026-07-26T19:50:40.244404973Z`
- TPU state at verification: `READY`, `HEALTHY`
- TPU runtime: `v2-alpha-tpuv5-lite`
- Package versions: `vllm-tpu==0.25.0`, `tpu-inference==0.25.0`,
  `jax==0.10.2`
- Model revision: `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`
- HBM after model load: 7.49/15.75 GiB
- Allocated KV cache: 50,688 tokens
- Correctness prompt: `The capital of France is`
- Correctness response begins: ` Paris.`
- Correctness response SHA-256:
  `854fd6043e523f199ff34d684561c7da18c8ae736cfe3df535e81eb2a2e993ca`
- Cold-start weight download: 45.26 seconds
- Cold-start weight load: 29.26 seconds
- HBM logical triad payload: median 532.14 GB/s, p90 534.62 GB/s,
  minimum 484.95 GB/s, maximum 535.43 GB/s over 10 retained samples
- Measured ICI bandwidth: pending four-chip slice
- Measured DCN bandwidth: pending second slice

Raw Phase 0 artifacts are stored locally under `artifacts/phase0/` and are
excluded from version control.
