#!/usr/bin/env bash
set -euo pipefail

mount_point="${PD_DISAGG_MOUNT:-/mnt/pd-disagg}"

pid_files=()
pids=()
process_groups=()
for pid_file in \
  "${mount_point}/phase0-vllm.pid" \
  "${mount_point}/pd-vllm.pid" \
  "${mount_point}"/pd-vllm-*.pid; do
  if [[ ! -f "${pid_file}" ]]; then
    continue
  fi
  pid="$(cat "${pid_file}")"
  pid_files+=("${pid_file}")
  pids+=("${pid}")
  if kill -0 "${pid}" 2>/dev/null; then
    pgid="$(ps -o pgid= -p "${pid}" | tr -d ' ')"
    if [[ "${pgid}" == "${pid}" ]]; then
      process_groups+=("${pgid}")
      kill -TERM -- "-${pgid}"
    else
      kill "${pid}"
    fi
  fi
done

for _ in $(seq 1 20); do
  running=0
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      running=1
      break
    fi
  done
  if [[ "${running}" -eq 0 ]]; then
    break
  fi
  sleep 1
done

for pid in "${pids[@]}"; do
  if kill -0 "${pid}" 2>/dev/null; then
    kill -9 "${pid}"
  fi
done

for pgid in "${process_groups[@]}"; do
  if kill -0 -- "-${pgid}" 2>/dev/null; then
    kill -KILL -- "-${pgid}"
  fi
done

mapfile -t orphan_engine_pids < <(
  pgrep -u "$(id -u)" -f '[V]LLM::EngineCore' || true
)
if [[ "${#orphan_engine_pids[@]}" -gt 0 ]]; then
  kill -KILL "${orphan_engine_pids[@]}"
fi

for pid_file in "${pid_files[@]}"; do
  rm -f "${pid_file}"
done
rm -f /tmp/libtpu_lockfile

echo "vLLM stopped"
