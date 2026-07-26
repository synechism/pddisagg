# Phase 1 measurement harness

## Implemented

- Exact prompt lengths using token-ID inputs
- Exact requested output lengths with EOS ignored
- Poisson arrival process with a deterministic seed
- Bounded client concurrency with client-side queue time
- Raw per-request JSONL records
- TTFT, total latency, streaming token-event offsets, and ITLs
- Warmup records retained but marked as excluded
- Independent-run aggregation with a hard minimum of five runs
- Exact input/output token-count validation against server usage

## TPU-host diagnostic

This run validates the harness but is not publishable because the client shared
the TPU VM host with vLLM.

- Model: `Qwen/Qwen3-4B`
- Input length: 512 tokens
- Output length: 64 tokens
- Arrival rate: 2 requests/second
- Runs: 5
- Measured requests: 100
- Successes: 100
- Across-run median TTFT: 65.39 ms
- P90 of run-median TTFT: 67.15 ms
- Across-run median ITL: 11.73 ms
- P90 of run-median ITL: 11.89 ms
- All input lengths exact: yes
- All output lengths exact: yes
- All streaming token counts matched server completion counts: yes

Raw records are under `artifacts/phase1/phase1-baseline-i512-o64-r2/`.

## Independent load-generator baseline

This is a valid network-separated harness measurement: the client runs on a
separate `n2-standard-8` VM and reaches vLLM through the private VPC. It is not
the final colocated comparison baseline because chunk size has not yet been
swept and selected for this workload.

- Model: `Qwen/Qwen3-4B`
- Input length: 512 tokens
- Output length: 64 tokens
- Arrival rate: 2 requests/second
- Runs: 5
- Measured requests: 100
- Successes: 100
- Across-run median TTFT: 65.15 ms
- P90 of run-median TTFT: 67.44 ms
- Median of run-p90 TTFT: 70.77 ms
- P90 of run-p90 TTFT: 106.96 ms
- Across-run median total latency: 837.19 ms
- P90 of run-median total latency: 865.93 ms
- Across-run median ITL: 11.727 ms
- P90 of run-median ITL: 11.864 ms
- All input lengths exact: yes
- All output lengths exact: yes
- All streaming token counts matched server completion counts: yes

The median values are indistinguishable from the same-host diagnostic at this
load. Two runs had a larger TTFT tail, retained in the run-p90 error bar.
Raw records are under
`artifacts/phase1-network/phase1-baseline-i512-o64-r2/`.

## Infrastructure blocker

A four-chip `v5litepod-4` ICI request cannot currently be placed:

- `us-central1-a` has a separate four-chip serving quota; the active one-chip
  node makes the requested total five.
- `us-east5-a` rejects the TRC principal as unauthorized for the v5e-4 queue,
  despite visible regional quota.

No ICI performance claim will be made until a real four-chip slice is allocated.

After persisting the one-chip correctness and baseline data, that node was
released so a four-chip request fits the per-zone serving limit exactly. The
current `v5litepod-4` Spot request will be partitioned 2P:2D for the first ICI
implementation. A separate cross-slice/DCN allocation remains future work.

Availability checks for a DCN fallback were stopped after two attempts:

- `us-east5-b` returned explicit insufficient capacity for `v5litepod-1`.
- `us-east5-c` rejected this TRC principal for the `v5litepod-1` queue.

The empty persistent disks created before those failed submissions were
deleted. No result will be relabeled as DCN or ICI based on a workaround.
