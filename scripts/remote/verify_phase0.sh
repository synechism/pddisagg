#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=phase0_env.sh
source "${script_dir}/phase0_env.sh"
mount_point="${PD_DISAGG_MOUNT}"

artifact="${mount_point}/artifacts/phase0-runtime.txt"
{
  date --iso-8601=seconds
  uname -a
  echo
  lscpu
  echo
  df -h "${mount_point}"
  echo
  python - <<'PY'
import importlib.metadata
import jax
import vllm
from vllm.platforms import current_platform

print(f"vllm={vllm.__version__}")
print(f"tpu_inference={importlib.metadata.version('tpu_inference')}")
print(f"jax={jax.__version__}")
print(f"platform={current_platform.get_device_name()}")
print(f"devices={jax.devices()}")
PY
} | tee "${artifact}"
