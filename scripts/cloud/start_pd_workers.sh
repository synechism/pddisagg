#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=phase0_env.sh
source "${script_dir}/phase0_env.sh"

prefill_name="${PREFILL_TPU_NAME:-pd-phase0-v5e-1}"
decode_name="${DECODE_TPU_NAME:-pd-decode-v5e-1}"

prefill_ip="$(gcloud compute tpus tpu-vm describe "${prefill_name}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --format='value(networkEndpoints[0].ipAddress)')"
decode_ip="$(gcloud compute tpus tpu-vm describe "${decode_name}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --format='value(networkEndpoints[0].ipAddress)')"

gcloud compute tpus tpu-vm ssh "${prefill_name}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --command="bash ~/pd-disagg-remote/stop_vllm.sh && PD_ROLE=producer VLLM_HOST_IP=${prefill_ip} bash ~/pd-disagg-remote/serve_pd_worker.sh"

gcloud compute tpus tpu-vm ssh "${decode_name}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --command="bash ~/pd-disagg-remote/stop_vllm.sh && PD_ROLE=consumer VLLM_HOST_IP=${decode_ip} bash ~/pd-disagg-remote/serve_pd_worker.sh"

echo "PD workers started: producer=${prefill_ip}:8000 consumer=${decode_ip}:8000"
