# Selective KV pull for TPU PD disaggregation

While serving LLMs, standard practice nowadays is to have separate pools of GPU workers for prefill and decode (PD Disaggregation). Open source TPU inference still has a lot of jagged edges, and while investigating how vLLM handles PD Disagg, we found a pretty gigantic inefficiency: when you split prefill and decode onto different TPU workers, the decode worker has to pull the whole KV cache over
the network before it can do anything. This is extremely redundant as the decode workers already have
most of that cache sitting in memory from the last request because you're
just continuing the same conversation. 


## Results

Tested on a TPU v6e-4 running Qwen3-4B, two chips on the same VM, 2048
token prompts with a 1536 token cached prefix (75% of the prompt already
sitting in decode's cache):

| Metric | Stock | Selective | Change |
|---|---:|---:|---:|
| KV payload | 288 MiB | 72 MiB | 4x less |
| Median pull time | 17.84 ms | 5.49 ms | 3.25x faster |
| Warm end-to-end | 190.9 ms | 180.9 ms | 5.2% lower |
| Output match | 4/4 | 4/4 | identical, token for token |


The cool thing is that as your conversation gets longer (i.e. a coding using your agent of choice) this technique actually works better, as more of your conversation will become a prefix on each turn.
