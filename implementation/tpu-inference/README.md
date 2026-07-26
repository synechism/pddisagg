# Benchmarked tpu-inference source snapshot

These are the exact modified upstream files loaded by the TPU workers for the
selective-pull benchmark:

| File | SHA-256 |
| --- | --- |
| `tpu_inference/distributed/tpu_connector.py` | `1aa7ba4ab9e9d0e5f18b1d2036619953184a03259bb68388cf080da79f234e2f` |
| `tpu_inference/worker/tpu_worker.py` | `558b4a938f976d40accbbb8fda0358e6766ba2057e89eeaa13b2357c03c5ce76` |
| `tests/distributed/test_tpu_connector.py` | `9c54d6fd26b50903a1fe98e786d1f1ec3b1c1f754fe1288cae20f14cab9deaf2` |

They are based on `vllm-project/tpu-inference` tag `v0.25.0` and include the
three commits in `../../patches/tpu-inference-v0.25`.

The snapshot is provided for direct inspection. Use the patch series to apply
the changes to a complete upstream checkout.
