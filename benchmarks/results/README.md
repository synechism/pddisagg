# Benchmark results

This directory contains the evidence collected while bringing up and testing
TPU prefill/decode disaggregation. It intentionally includes both successful
results and intermediate failure cases so the final implementation decisions
can be traced back to observed behavior.

## Result sets

| Path | Contents |
| --- | --- |
| `selective-pull/` | Small, review-friendly control/optimized comparison for the primary 2,048-token, 75%-prefix-hit result. |
| `archive/2026-07-26/phase0/` | TPU runtime inventory, smoke response, package snapshot, worker log, and single-chip HBM bandwidth measurements. |
| `archive/2026-07-26/phase1/` | Five baseline load-generator runs, with per-request JSONL, per-run summaries, and aggregate statistics. |
| `archive/2026-07-26/phase1-network/` | Separately retrieved copy of the Phase 1 baseline result set. |
| `archive/2026-07-26/v6e/` | TPU v6e copy of the Phase 1 baseline result set. |
| `archive/2026-07-26/selective-pull/` | Raw 512- and 2,048-token selective-pull responses plus producer and consumer logs for the matched comparison. |
| `archive/2026-07-26/selective-stress/` | Concurrency, delayed-registration, expiration, recovery, router-retry, and final production-mode stress results and logs. |
| `archive/2026-07-26/correctness/` | Direct colocated reference outputs used for token-level correctness checks. |

`archive/2026-07-26/MANIFEST.sha256` records the SHA-256 digest of every raw
artifact. See the archive's `README.md` for interpretation notes.

## Primary matched comparison

The headline experiment used Qwen3-4B on a colocated v6e-4, with prefill on
chip 0 and decode on chip 1. The input was 2,048 tokens, the decoder already
held a 1,536-token prefix, block size was 128, and generation was
deterministic.

| Metric | Stock full pull | Selective pull | Difference |
| --- | ---: | ---: | ---: |
| Producer blocks transferred | 16 | 4 | 4.00x fewer |
| KV payload | 288 MiB | 72 MiB | 4.00x less |
| Median KV pull | 17.84 ms | 5.49 ms | 3.25x faster |
| Warm median end-to-end | 190.888 ms | 180.908 ms | 5.23% lower |
| Exact output matches | 4/4 | 4/4 | No token changes |

The compact records are `selective-pull/control-2048.json` and
`selective-pull/optimized-2048.json`. The corresponding raw worker logs live
under `archive/2026-07-26/selective-pull/`.

## Reading the archive

- `*.json` files are complete response records or aggregate benchmark output.
- `*.jsonl` files contain one load-generator request record per line.
- `*.summary.json` files summarize the corresponding JSONL run.
- `*-producer.log` and `*-consumer.log` contain the worker-side transfer
  traces used to confirm requested block counts, transfer sizes, timings, and
  lifecycle behavior.
- Stress files contain top-level fields such as `successful`,
  `exact_matches`, and `latency_ms`. A `null` exactness field means the run did
  not receive a reference file; it does not mean the outputs mismatched.

## Important interpretation notes

- The archive is chronological laboratory evidence, not a collection in which
  every run represents the final configuration.
- Delay, expiration, recomputation, and case-replay files deliberately capture
  failure paths encountered while hardening the protocol.
- The final warm concurrency-8 production run is
  `stress-final-production-warm-c8.json`: 8/8 requests completed with
  277.2 ms p50 and 279.1 ms p95. Exactness is `null` because that particular
  stress invocation did not supply a reference file.
- `stress-final-router-retry-valid.json` and
  `stress-final-router-retry-local-reference.json` together document that the
  request-scoped transfer failure fallback returned the same token sequence as
  a direct local reference.
- Raw logs contain ephemeral process IDs, request UUIDs, localhost endpoints,
  and experiment paths. Those values identify a run but are not stable
  configuration defaults.

