#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=phase0_env.sh
source "${script_dir}/phase0_env.sh"

output="${PD_DISAGG_MOUNT}/artifacts/phase0-bandwidth.json"
python "${HOME}/pd-disagg-benchmarks/phase0_bandwidth.py" \
  --size-mib-per-device 128 \
  --repetitions 10 \
  --output "${output}"

