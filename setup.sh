#!/usr/bin/env bash
set -euo pipefail

# One-command bootstrap for legged_wbc_mjlab.  The Python helper handles
# Python/PyPI mirror selection and the machine-specific PyTorch backend.
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Keep uv's downloads across projects and shell sessions.  Override these
# values by exporting them before running setup.sh if a shared disk/cache is
# preferred.
uv_cache_default="${XDG_CACHE_HOME:-${HOME}/.cache}/uv"
uv_python_default="${HOME}/.local/share/uv/python"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${uv_cache_default}}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${uv_python_default}}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-60}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-8}"
if [[ -n "${UV_PROXY:-}" ]]; then
  export HTTP_PROXY="${HTTP_PROXY:-${UV_PROXY}}"
  export HTTPS_PROXY="${HTTPS_PROXY:-${UV_PROXY}}"
  export http_proxy="${http_proxy:-${UV_PROXY}}"
  export https_proxy="${https_proxy:-${UV_PROXY}}"
fi

# A Python standalone archive is much larger than a package-index page.  Give
# one channel enough time to finish, while still allowing the helper to move
# on to a backup channel when a route is genuinely stuck.
python_install_timeout="${UV_PYTHON_INSTALL_TIMEOUT:-600}"
python_download_workers="${UV_PYTHON_DOWNLOAD_WORKERS:-4}"
python_version="${UV_PYTHON_VERSION:-3.11}"
if [[ -z "${UV_PYTHON_VERSION+x}" && -f "${project_root}/.python-version" ]]; then
  IFS= read -r python_version < "${project_root}/.python-version"
  python_version="${python_version%%#*}"
  python_version="${python_version//[[:space:]]/}"
  [[ -n "${python_version}" ]] || python_version="3.11"
fi

if command -v uv >/dev/null 2>&1; then
  uv_bin="$(command -v uv)"
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
  uv_bin="${HOME}/.local/bin/uv"
else
  echo "未找到 uv，正在安装 uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv_bin="${HOME}/.local/bin/uv"
  if [[ ! -x "${uv_bin}" ]]; then
    echo "uv 安装后仍未找到可执行文件，请检查 ~/.local/bin 是否在 PATH 中。" >&2
    exit 127
  fi
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "未找到 python3，无法运行安装引导脚本。请先安装系统 Python 3。" >&2
  exit 127
fi

setup_args=()
if [[ -n "${UV_WHEELHOUSE:-}" ]]; then
  setup_args+=(--wheelhouse "${UV_WHEELHOUSE}")
fi
if [[ "${UV_OFFLINE:-0}" == "1" || "${UV_OFFLINE:-}" == "true" ]]; then
  setup_args+=(--offline)
fi

exec python3 "${project_root}/tools/uv_auto_sync.py" \
  --uv "${uv_bin}" \
  --python-version "${python_version}" \
  --python-install-timeout "${python_install_timeout}" \
  --python-download-workers "${python_download_workers}" \
  "${setup_args[@]}" \
  "$@"
