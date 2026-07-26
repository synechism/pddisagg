#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=phase0_env.sh
source "${script_dir}/phase0_env.sh"

prefill_name="${PREFILL_TPU_NAME:-pd-phase0-v5e-1}"
decode_name="${DECODE_TPU_NAME:-pd-decode-v5e-1}"
loadgen_name="${LOADGEN_NAME:-pd-loadgen}"

prefill_ip="$(gcloud compute tpus tpu-vm describe "${prefill_name}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --format='value(networkEndpoints[0].ipAddress)')"
decode_ip="$(gcloud compute tpus tpu-vm describe "${decode_name}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --format='value(networkEndpoints[0].ipAddress)')"

gcloud compute ssh "${loadgen_name}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --command="PREFILL_ENDPOINT=http://${prefill_ip}:8000 DECODE_ENDPOINT=http://${decode_ip}:8000 bash ~/pd-disagg-driver/serve_router.sh"

echo "PD router started on ${loadgen_name}:9000"
