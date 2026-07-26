#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
host_ip="${VLLM_HOST_IP:?set VLLM_HOST_IP to this TPU VM private IP}"
pool_size="${PD_POOL_SIZE:-2}"

if [[ "${pool_size}" -ne 2 ]]; then
  echo "The initial v5e-4 ICI layout requires PD_POOL_SIZE=2" >&2
  exit 1
fi

bash "${script_dir}/stop_vllm.sh"

for worker_index in 0 1; do
  TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
  TPU_PROCESS_BOUNDS=1,1,1 \
  TPU_VISIBLE_CHIPS="${worker_index}" \
  PD_ROLE=producer \
  PD_INSTANCE_ID="producer-${worker_index}" \
  VLLM_HOST_IP="${host_ip}" \
  VLLM_PORT="$((8400 + worker_index))" \
  TPU_KV_TRANSFER_PORT="$((7100 + worker_index))" \
  TPU_SIDE_CHANNEL_PORT="$((6100 + worker_index))" \
  bash "${script_dir}/serve_pd_worker.sh"
done

for worker_index in 0 1; do
  TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
  TPU_PROCESS_BOUNDS=1,1,1 \
  TPU_VISIBLE_CHIPS="$((worker_index + 2))" \
  PD_ROLE=consumer \
  PD_INSTANCE_ID="consumer-${worker_index}" \
  VLLM_HOST_IP="${host_ip}" \
  VLLM_PORT="$((9400 + worker_index))" \
  TPU_KV_TRANSFER_PORT="$((7200 + worker_index))" \
  TPU_SIDE_CHANNEL_PORT="$((6100 + worker_index))" \
  bash "${script_dir}/serve_pd_worker.sh"
done

echo "Started paired 2P:2D ICI workers on ${host_ip}"
