#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=phase0_env.sh
source "${script_dir}/phase0_env.sh"
mount_point="${PD_DISAGG_MOUNT}"

response_file="${mount_point}/artifacts/phase0-smoke-response.json"
metadata_file="${mount_point}/artifacts/phase0-smoke-metadata.txt"

curl --fail --silent --show-error \
  --max-time 300 \
  http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${MODEL_ID}\",
    \"prompt\": \"The capital of France is\",
    \"max_tokens\": 8,
    \"temperature\": 0,
    \"seed\": 1
  }" \
  | tee "${response_file}"

{
  date --iso-8601=seconds
  sha256sum "${response_file}"
} | tee "${metadata_file}"
