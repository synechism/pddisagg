#!/usr/bin/env bash
set -euo pipefail

TPU_ENDPOINT="${TPU_ENDPOINT:?set TPU_ENDPOINT to the TPU server URL}"
source "${HOME}/pd-loadgen-venv/bin/activate"
export PYTHONPATH="${HOME}/pd-disagg-src"

result_dir="${HOME}/pd-results/phase1-baseline-i512-o64-r2"
mkdir -p "${result_dir}"

curl --fail --silent --show-error "${TPU_ENDPOINT}/health" >/dev/null

for run_index in 1 2 3 4 5; do
  python -m pd_disagg.loadgen \
    --endpoint "${TPU_ENDPOINT}" \
    --model Qwen/Qwen3-4B \
    --input-lengths 512 \
    --output-lengths 64 \
    --requests 20 \
    --arrival-rate 2 \
    --max-concurrency 64 \
    --warmup-requests 1 \
    --seed "$((2000 + run_index))" \
    --output "${result_dir}/run-${run_index}.jsonl"
done

