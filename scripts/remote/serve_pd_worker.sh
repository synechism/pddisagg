#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=phase0_env.sh
source "${script_dir}/phase0_env.sh"

role="${PD_ROLE:?set PD_ROLE to producer or consumer}"
case "${role}" in
  producer)
    kv_role="kv_producer"
    ;;
  consumer)
    kv_role="kv_consumer"
    ;;
  *)
    echo "PD_ROLE must be producer or consumer, got: ${role}" >&2
    exit 1
    ;;
esac

host_ip="${VLLM_HOST_IP:?set VLLM_HOST_IP to this TPU VM private IP}"
port="${VLLM_PORT:-8000}"
chunk_size="${MAX_NUM_BATCHED_TOKENS:-2048}"
block_size="${VLLM_BLOCK_SIZE:-128}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.3}"
device_ids="${PD_DEVICE_IDS:-}"
instance_id="${PD_INSTANCE_ID:-${role}}"
tp_size="${PD_TENSOR_PARALLEL_SIZE:-1}"
selective_pull="${TPU_KV_SELECTIVE_PULL:-false}"
enable_prefix_caching="${ENABLE_PREFIX_CACHING:-false}"
log_file="${PD_DISAGG_MOUNT}/logs/vllm-${instance_id}.log"
pid_file="${PD_DISAGG_MOUNT}/pd-vllm-${instance_id}.pid"
kv_config="{\"kv_connector\":\"TPUConnector\",\"kv_connector_module_path\":\"tpu_inference.distributed.tpu_connector\",\"kv_role\":\"${kv_role}\",\"kv_connector_extra_config\":{\"tpu_kv_selective_pull\":${selective_pull}}}"

if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
  echo "PD ${role} is already running with PID $(cat "${pid_file}")"
  exit 0
fi

export VLLM_HOST_IP="${host_ip}"
export PD_TPU_VISIBLE_CHIPS="${PD_TPU_VISIBLE_CHIPS:-${TPU_VISIBLE_CHIPS:-}}"
export TPU_KV_TRANSFER_PORT="${TPU_KV_TRANSFER_PORT:-9100}"
export TPU_SIDE_CHANNEL_PORT="${TPU_SIDE_CHANNEL_PORT:-9600}"
export VLLM_XLA_CHECK_RECOMPILATION="${VLLM_XLA_CHECK_RECOMPILATION:-1}"
export VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-/tmp/jax-cache-${instance_id}}"
export SKIP_JAX_PRECOMPILE="${SKIP_JAX_PRECOMPILE:-1}"
export JAX_COORDINATOR_ADDRESS="${JAX_COORDINATOR_ADDRESS:-127.0.0.1:1200}"
export JAX_PROCESS_ID="${JAX_PROCESS_ID:-0}"
export JAX_NUM_PROCESSES="${JAX_NUM_PROCESSES:-1}"

device_args=()
if [[ -n "${device_ids}" ]]; then
  device_args=(--device-ids "${device_ids}")
fi

prefix_cache_args=(--no-enable-prefix-caching)
if [[ "${enable_prefix_caching}" == "true" ]]; then
  prefix_cache_args=(--enable-prefix-caching)
fi

nohup setsid vllm serve "${MODEL_ID}" \
  --revision "${MODEL_REVISION}" \
  --host 0.0.0.0 \
  --port "${port}" \
  --download-dir "${HF_HOME}" \
  --tensor-parallel-size "${tp_size}" \
  "${device_args[@]}" \
  --max-model-len 8192 \
  --gpu-memory-utilization "${gpu_memory_utilization}" \
  --enable-chunked-prefill \
  "${prefix_cache_args[@]}" \
  --max-num-batched-tokens "${chunk_size}" \
  --block-size "${block_size}" \
  --kv-transfer-config "${kv_config}" \
  >"${log_file}" 2>&1 &

echo "$!" >"${pid_file}"
echo "Started PD ${role} ${instance_id} PID $(cat "${pid_file}"); log: ${log_file}"
