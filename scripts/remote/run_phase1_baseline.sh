#!/usr/bin/env bash
set -euo pipefail

source /mnt/pd-disagg/venvs/vllm-tpu/bin/activate
export PYTHONPATH="${HOME}/pd-disagg-src"

result_dir="/mnt/pd-disagg/results/phase1-baseline-i512-o64-r2"
mkdir -p "${result_dir}"

for run_index in 1 2 3 4 5; do
  python -m pd_disagg.loadgen \
    --endpoint http://127.0.0.1:8000 \
    --model Qwen/Qwen3-4B \
    --input-lengths 512 \
    --output-lengths 64 \
    --requests 20 \
    --arrival-rate 2 \
    --max-concurrency 64 \
    --warmup-requests 1 \
    --seed "$((1000 + run_index))" \
    --output "${result_dir}/run-${run_index}.jsonl"
done

