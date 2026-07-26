#!/usr/bin/env bash
set -euo pipefail

venv_dir="${HOME}/pd-loadgen-venv"
uv_bin="${HOME}/bin/uv"
uv_version="${UV_VERSION:-0.11.32}"
python_version="${PYTHON_VERSION:-3.12.13}"

if [[ ! -x "${uv_bin}" ]]; then
  uv_archive="$(mktemp)"
  curl -LsSf https://astral.sh/uv/install.sh -o "${uv_archive}"
  UV_VERSION="${uv_version}" \
    UV_UNMANAGED_INSTALL="${HOME}/bin" \
    sh "${uv_archive}"
  rm -f "${uv_archive}"
fi

"${uv_bin}" venv "${venv_dir}" \
  --python "${python_version}" \
  --clear
"${uv_bin}" pip install \
  --python "${venv_dir}/bin/python" \
  "fastapi==0.116.1" \
  "httpx==0.28.1" \
  "uvicorn==0.35.0"

echo "Load-generator environment ready: ${venv_dir}"
