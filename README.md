# Selective KV pull for TPU PD disaggregation

Here's the problem this fixes: when you split prefill and decode onto
different TPU workers, the decode worker has to pull the whole KV cache over
the network before it can do anything. Every time. Even if it already has
most of that cache sitting in memory from the last request because you're
just continuing the same conversation. The upstream code straight up admits
this in a comment — there's a TODO saying someone should build a side channel
so prefill only sends the part decode doesn't already have. Nobody had
gotten around to it.

So we built it.

## What's actually going on

Decode checks its local cache, figures out which blocks it's missing, and
just asks for those instead of everything. Prefill checks that the request
actually makes sense (can't have decode asking for random blocks it
shouldn't get), gathers only what's needed, and sends it over. If decode
already has the entire thing cached, no transfer happens at all — prefill
just gets told "you're good, free that memory."

Nothing lossy about it. Same KV values, same math, we just stopped shipping
bytes nobody asked for.

## Does it work

Yeah. Tested on a TPU v6e-4 running Qwen3-4B, two chips on the same VM, 2048
token prompts with a 1536 token cached prefix (75% of the prompt already
sitting in decode's cache):

| Metric | Stock | Selective | Change |
|---|---:|---:|---:|
| KV payload | 288 MiB | 72 MiB | 4x less |
| Median pull time | 17.84 ms | 5.49 ms | 3.25x faster |
| Warm end-to-end | 190.9 ms | 180.9 ms | 5.2% lower |
| Output match | 4/4 | 4/4 | identical, token for token |

Ran it again at a smaller 512-token prefix and got the same direction: 72 MiB
down to 18 MiB, pull time cut by more than half, end-to-end down a bit too.

Real talk on the end-to-end number though — it's smaller than the connector
number because this was two chips on one box, so the actual network hop is
fast and most of your time is still spent running the model, not moving
bytes. Cross-host is where this should actually start mattering a lot more,
since real network hops aren't free. Haven't measured that yet. Not gonna
pretend we did.

Also, is 75% cache hit even realistic? Depends what you're serving. Random
one-off requests, no, you'll see like 0-10%. Normal back-and-forth chat, more
like 40-70%. Long conversations, coding agents, anything doing repeated tool
calls where the same huge system prompt gets reused over and over — that's
genuinely 60-90% territory. So 75% isn't us picking a flattering number, it's
just what agentic-style traffic actually looks like.

## Trying to break it

Getting it working was honestly the easy part. The real work was trying to
make it fall over on purpose, because a benchmark that only tests the happy
path doesn't tell you anything about whether you can actually ship this.

- **`selective_stress.py`** — throws concurrent requests at it. Found a real
  bug this way: at 8 concurrent fresh requests, there's a race where decode
  can ask for something before prefill has finished registering it, and that
  used to take down the *entire decode engine*, not just fail the one
  request. Reproduced it clean, twice, from a fresh process.
- **`selective_delay.py`** — delays decode on purpose to poke at the timeout
  logic. Found the same bug wearing a different hat: a request that shows up
  late can hit an already-expired entry and crash the engine the same way.
- **`prefix_selective.py`** — the actual correctness/latency benchmark, the
  one that produced the numbers above. Paired prompts, deterministic
  generation, checks every output token matches the stock run exactly.
- **`phase0_bandwidth.py`** — just a sanity check on raw TPU bandwidth so we
  weren't fooling ourselves with a weird environment.

Both crashes came down to the same root cause: a missing or rejected transfer
ID was treated as a reason to kill the whole engine instead of just failing
that one request. Fixed now — failures are scoped to a single request, and
if a remote pull does fail, the router just retries it once as a normal local
decode. Tested that path specifically: forced a request to fail, watched it
retry, and the retry matched a direct local run token for token while nothing
else on either engine went down.

While we were in there we also fixed the control thread blocking on cold JAX
compiles (was costing up to ~440ms and stalling every other message behind
it) and added a lock so a background KV gather can't race the model's actual
forward pass and stomp on the same memory.

Ran the stress test again after all the fixes: 8 out of 8 concurrent
requests, clean, p50 277ms / p95 279ms. No crashes.

## The code

- [`implementation/tpu-inference/tpu_inference/distributed/tpu_connector.py`](implementation/tpu-inference/tpu_inference/distributed/tpu_connector.py) — the whole thing lives here
- [`implementation/tpu-inference/tpu_inference/runner/tpu_runner.py`](implementation/tpu-inference/tpu_inference/runner/tpu_runner.py) — the KV cache lock
- [`implementation/tpu-inference/tpu_inference/worker/tpu_worker.py`](implementation/tpu-inference/tpu_inference/worker/tpu_worker.py) — TPU chip affinity fix for when prefill and decode share a box
- [`implementation/tpu-inference/tests/distributed/test_tpu_connector.py`](implementation/tpu-inference/tests/distributed/test_tpu_connector.py) — 46 tests, all green on real TPU hardware
- [`patches/tpu-inference-v0.25/`](patches/tpu-inference-v0.25) — four clean commits on top of upstream `v0.25.0`
- [`benchmarks/`](benchmarks) — everything above, results are in [`benchmarks/results/selective-pull/`](benchmarks/results/selective-pull)

The whole branch is just upstream `v0.25.0` plus these four:

```
afac0f54  Optimize and stabilize TPU PD transfers
69ba34f4  Transfer only uncached KV blocks for prefix hits
b77ea5fb  Release producer state on full prefix hits
cae7bdb4  Make selective TPU transfers race-safe
```

Checked upstream's remote branches too — nobody else has this for TPU yet,
as far as we could find.

## How to actually use it

```bash
git clone https://github.com/vllm-project/tpu-inference.git
cd tpu-inference
git checkout v0.25.0
git am /path/to/pddisagg/patches/tpu-inference-v0.25/*.patch
```

Then flip it on in your connector config, on both the prefill and decode
side:

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

Off by default. Also auto-disables itself on Ray multihost and the D2H
(host-memory) path since we haven't tested it there.

## What we haven't done yet

- Real cross-host testing. Everything here is same-VM, two chips. The whole
  point of this feature matters more once you're crossing an actual network,
  and we just haven't gotten hardware to test that yet.
- Tried compressing the transfer with INT4 at one point. Killed it — changed
  the outputs and was actually slower on this same-host setup. Not worth
  keeping around.

Full writeup with the exact test config and hit-rate reasoning is in
[`notes/selective_pull_results_2026-07-26.md`](notes/selective_pull_results_2026-07-26.md)
if you want to go deeper.