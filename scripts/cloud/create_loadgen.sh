#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-disagg-503619}"
ZONE="${ZONE:-us-central1-a}"
LOADGEN_NAME="${LOADGEN_NAME:-pd-loadgen}"
LOADGEN_MACHINE_TYPE="${LOADGEN_MACHINE_TYPE:-n2-standard-8}"

if gcloud compute instances describe "${LOADGEN_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" >/dev/null 2>&1; then
  echo "Load-generator VM already exists: ${LOADGEN_NAME}"
  exit 0
fi

gcloud compute instances create "${LOADGEN_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --machine-type="${LOADGEN_MACHINE_TYPE}" \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=30GB \
  --boot-disk-type=pd-balanced \
  --network=default \
  --maintenance-policy=MIGRATE \
  --labels=project=pd-disagg,role=loadgen
