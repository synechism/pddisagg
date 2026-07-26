#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=phase0_env.sh
source "${script_dir}/phase0_env.sh"

loadgen_name="${LOADGEN_NAME:-pd-loadgen}"
tpu_ip="$(gcloud compute tpus tpu-vm describe "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --format='value(networkEndpoints[0].ipAddress)')"

gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --command="VLLM_HOST_IP=${tpu_ip} bash ~/pd-disagg-remote/serve_pd_ici.sh"

gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --command='for port in 8400 8401 9400 9401; do ready=0; for _ in $(seq 1 240); do if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null; then ready=1; break; fi; sleep 5; done; if [[ "${ready}" -ne 1 ]]; then echo "Worker on port ${port} did not become healthy" >&2; exit 1; fi; done'

gcloud compute ssh "${loadgen_name}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --command="PREFILL_ENDPOINT=http://${tpu_ip}:8400,http://${tpu_ip}:8401 DECODE_ENDPOINT=http://${tpu_ip}:9400,http://${tpu_ip}:9401 bash ~/pd-disagg-driver/serve_router.sh"

echo "ICI PD endpoint ready on ${loadgen_name}:9000"
