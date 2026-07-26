#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=phase0_env.sh
source "${script_dir}/phase0_env.sh"

active_account="$(gcloud auth list --filter='status:ACTIVE' --format='value(account)')"
if [[ -z "${active_account}" ]]; then
  echo "No active gcloud account. Run: gcloud auth login" >&2
  exit 1
fi

gcloud config set project "${PROJECT_ID}" >/dev/null

for service in compute.googleapis.com tpu.googleapis.com; do
  if ! gcloud services list \
    --enabled \
    --project="${PROJECT_ID}" \
    --format='value(config.name)' | grep -qx "${service}"; then
    echo "Required API is not enabled: ${service}" >&2
    exit 1
  fi
done

if ! gcloud compute disks describe "${DATA_DISK_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" >/dev/null 2>&1; then
  gcloud compute disks create "${DATA_DISK_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --size="${DATA_DISK_SIZE}" \
    --type="${DATA_DISK_TYPE}" \
    --labels=project=pd-disagg,phase=phase0
fi

if gcloud compute tpus tpu-vm describe "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" >/dev/null 2>&1; then
  echo "TPU node already exists: ${TPU_NAME}"
  exit 0
fi

if gcloud compute tpus queued-resources describe "${QUEUED_RESOURCE_ID}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" >/dev/null 2>&1; then
  echo "Queued resource already exists: ${QUEUED_RESOURCE_ID}"
  exit 0
fi

provisioning_flag=""
case "${TPU_PROVISIONING_MODEL}" in
  on-demand)
    ;;
  guaranteed)
    provisioning_flag="--guaranteed"
    ;;
  spot)
    provisioning_flag="--spot"
    ;;
  *)
    echo "Unsupported TPU_PROVISIONING_MODEL: ${TPU_PROVISIONING_MODEL}" >&2
    exit 1
    ;;
esac

create_args=(
  compute tpus queued-resources create "${QUEUED_RESOURCE_ID}"
  "--project=${PROJECT_ID}"
  "--zone=${ZONE}"
  "--node-id=${TPU_NAME}"
  "--accelerator-type=${ACCELERATOR_TYPE}"
  "--runtime-version=${RUNTIME_VERSION}"
  "--data-disk=source=projects/${PROJECT_ID}/zones/${ZONE}/disks/${DATA_DISK_NAME},mode=read-write"
  "--network=${NETWORK}"
)
if [[ -n "${provisioning_flag}" ]]; then
  create_args+=("${provisioning_flag}")
fi
gcloud "${create_args[@]}"

echo "Submitted ${ACCELERATOR_TYPE} (${TPU_PROVISIONING_MODEL}) as queued resource ${QUEUED_RESOURCE_ID}."
