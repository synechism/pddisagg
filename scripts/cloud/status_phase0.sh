#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=phase0_env.sh
source "${script_dir}/phase0_env.sh"

echo "Project: ${PROJECT_ID}"
echo "Zone: ${ZONE}"
echo "Requested accelerator: ${ACCELERATOR_TYPE}"

echo
echo "Queued resource:"
gcloud compute tpus queued-resources describe "${QUEUED_RESOURCE_ID}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --format='yaml(name,state,guaranteed,nodeSpec)' 2>/dev/null || echo "not created"

echo
echo "TPU node:"
gcloud compute tpus tpu-vm describe "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --format='yaml(name,acceleratorType,state,health,runtimeVersion,networkEndpoints)' \
  2>/dev/null || echo "not allocated"

echo
echo "Data disk:"
gcloud compute disks describe "${DATA_DISK_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --format='yaml(name,sizeGb,type.basename(),status,users)' 2>/dev/null || echo "not created"

