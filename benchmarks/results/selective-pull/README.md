# Selective-pull matched results

- `control-2048.json`: prefix caching enabled, selective pull disabled.
- `optimized-2048.json`: prefix caching enabled, selective pull enabled.

Both runs used the same four paired prompts and deterministic generation
settings. Every optimized output token sequence exactly matches its control.

The first pair includes JAX compilation. Warm end-to-end medians use pairs
2–4. Connector medians use all four variant pulls from the archived worker
logs.
