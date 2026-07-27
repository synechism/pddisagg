# Benchmarked tpu-inference source snapshot

These are the exact modified upstream files loaded by the TPU workers for the
selective-pull benchmark:

| File | SHA-256 |
| --- | --- |
| `tpu_inference/distributed/tpu_connector.py` | `5a195eafd4d5d68857f0ca4d68f4c1a413fad91f72c0430ba1e6fde41869426b` |
| `tpu_inference/worker/tpu_worker.py` | `558b4a938f976d40accbbb8fda0358e6766ba2057e89eeaa13b2357c03c5ce76` |
| `tpu_inference/runner/tpu_runner.py` | `ecdb5ccdd9cec463ffd4549f5b3af9762c4c4cbbefef1ccc6784c893d8386325` |
| `tests/distributed/test_tpu_connector.py` | `3d4619d14bf82e5785c09aede2b6afd3d236f577c94b19455a0c38d17709b1db` |

They are based on `vllm-project/tpu-inference` tag `v0.25.0` and include the
four commits in `../../patches/tpu-inference-v0.25`.

The snapshot is provided for direct inspection. Use the patch series to apply
the changes to a complete upstream checkout.
