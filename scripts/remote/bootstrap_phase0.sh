#!/usr/bin/env bash
set -euo pipefail

data_disk_id="${DATA_DISK_ID:-google-persistent-disk-1}"
data_device="/dev/disk/by-id/${data_disk_id}"
mount_point="${PD_DISAGG_MOUNT:-/mnt/pd-disagg}"
use_boot_disk="${USE_BOOT_DISK:-0}"
venv_dir="${mount_point}/venvs/vllm-tpu"
uv_bin="${mount_point}/bin/uv"
uv_version="${UV_VERSION:-0.11.32}"
python_version="${PYTHON_VERSION:-3.12.13}"
vllm_tpu_version="${VLLM_TPU_VERSION:-0.25.0}"

if [[ "${use_boot_disk}" == "1" ]]; then
  sudo mkdir -p "${mount_point}"
  sudo chown "$(id -u):$(id -g)" "${mount_point}"
else
  if [[ ! -e "${data_device}" ]]; then
    echo "Attached data disk not found: ${data_device}" >&2
    ls -la /dev/disk/by-id >&2 || true
    exit 1
  fi

  if ! sudo blkid "${data_device}" >/dev/null 2>&1; then
    sudo mkfs.ext4 -F -m 0 -L pd-disagg "${data_device}"
  fi

  sudo mkdir -p "${mount_point}"
  disk_uuid="$(sudo blkid -s UUID -o value "${data_device}")"
  if ! grep -q "UUID=${disk_uuid}" /etc/fstab; then
    echo "UUID=${disk_uuid} ${mount_point} ext4 defaults,nofail 0 2" \
      | sudo tee -a /etc/fstab >/dev/null
  fi
  sudo mount "${mount_point}" 2>/dev/null || sudo mount -a
  sudo chown "$(id -u):$(id -g)" "${mount_point}"
fi

mkdir -p \
  "${mount_point}/bin" \
  "${mount_point}/cache/huggingface" \
  "${mount_point}/artifacts" \
  "${mount_point}/logs" \
  "${mount_point}/venvs"

if [[ ! -x "${uv_bin}" ]]; then
  uv_archive="$(mktemp)"
  curl -LsSf https://astral.sh/uv/install.sh -o "${uv_archive}"
  UV_VERSION="${uv_version}" \
    UV_UNMANAGED_INSTALL="${mount_point}/bin" \
    sh "${uv_archive}"
  rm -f "${uv_archive}"
fi

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  "${uv_bin}" venv "${venv_dir}" --python "${python_version}"
fi

"${uv_bin}" pip install \
  --python "${venv_dir}/bin/python" \
  --upgrade \
  "vllm-tpu==${vllm_tpu_version}"

"${uv_bin}" pip freeze --python "${venv_dir}/bin/python" \
  | sort >"${mount_point}/artifacts/phase0-pip-freeze.txt"

echo "Phase 0 environment installed at ${mount_point}"
