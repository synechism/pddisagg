#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-disagg-503619}"
ZONE="${ZONE:-us-central1-a}"
LOADGEN_NAME="${LOADGEN_NAME:-pd-loadgen}"

gcloud compute instances describe "${LOADGEN_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --format='yaml(name,status,machineType.basename(),networkInterfaces,creationTimestamp)'

