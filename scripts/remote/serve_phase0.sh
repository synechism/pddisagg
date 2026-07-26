#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=phase0_env.sh
source "${script_dir}/phase0_env.sh"
mount_point="${PD_DISAGG_MOUNT}"

model_id="${MODEL_ID:-Qwen/Qwen3-4B}"
log_file="${mount_point}/logs/phase0-vllm.log"
pid_file="${mount_point}/phase0-vllm.pid"

if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
  echo "vLLM is already running with PID $(cat "${pid_file}")"
  exit 0
fi

nohup vllm serve "${model_id}" \
  --revision "${MODEL_REVISION}" \
  --host 0.0.0.0 \
  --port 8000 \
  --download-dir "${HF_HOME}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len 8192 \
  >"${log_file}" 2>&1 &

echo "$!" >"${pid_file}"
echo "Started vLLM PID $(cat "${pid_file}"); log: ${log_file}"
