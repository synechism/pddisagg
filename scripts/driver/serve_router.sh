#!/usr/bin/env bash
set -euo pipefail

PREFILL_ENDPOINT="${PREFILL_ENDPOINT:?set comma-separated PREFILL_ENDPOINT values}"
DECODE_ENDPOINT="${DECODE_ENDPOINT:?set comma-separated DECODE_ENDPOINT values}"
router_port="${PD_ROUTER_PORT:-9000}"
result_dir="${HOME}/pd-results"
pid_file="${result_dir}/pd-router.pid"
log_file="${result_dir}/pd-router.log"

mkdir -p "${result_dir}"

if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
  kill "$(cat "${pid_file}")"
fi

source "${HOME}/pd-loadgen-venv/bin/activate"
export PYTHONPATH="${HOME}/pd-disagg-src"
export PREFILL_ENDPOINT DECODE_ENDPOINT
export PD_ROUTER_RECORDS="${result_dir}/router.jsonl"

nohup python -m pd_disagg.router \
  --host 0.0.0.0 \
  --port "${router_port}" \
  >"${log_file}" 2>&1 &

echo "$!" >"${pid_file}"
echo "Started PD router PID $(cat "${pid_file}"); log: ${log_file}"
