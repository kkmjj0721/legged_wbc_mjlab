#!/usr/bin/env python3
"""Choose a fast Python package index and synchronize the uv environment.

The probe downloads a package-index page and a small prefix of one wheel.  It
does not install anything, and it uses the same PEP 503 ``/simple/`` endpoint
that uv uses.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from subprocess import run


DEFAULT_INDEXES = (
    "https://pypi.org/simple",
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.aliyun.com/pypi/simple",
    "https://mirrors.ustc.edu.cn/pypi/web/simple",
    "https://mirrors.cloud.tencent.com/pypi/simple",
    "https://repo.huaweicloud.com/repository/pypi/simple",
    "https://mirror.sjtu.edu.cn/pypi/web/simple",
    "https://mirrors.bfsu.edu.cn/pypi/web/simple",
    "https://mirrors.nju.edu.cn/pypi/web/simple",
    "https://mirror.nankai.edu.cn/pypi/web/simple",
    "https://mirrors.zju.edu.cn/pypi/web/simple",
    "https://pypi.doubanio.com/simple",
    "https://mirrors.sustech.edu.cn/pypi/simple",
    "https://pypi.mirrors.ustc.edu.cn/simple",
    "https://mirrors.hit.edu.cn/pypi/web/simple",
    "https://mirrors.sdu.edu.cn/pypi/web/simple",
)
PROBE_PACKAGE = "numpy"
DEFAULT_TIMEOUT = 6.0
DEFAULT_PROBE_BYTES = 1024 * 1024
DEFAULT_SYNC_TIMEOUT = 300.0
DEFAULT_PYTHON_INSTALL_TIMEOUT = 600.0
DEFAULT_TORCH_TIMEOUT = 1800.0
MAX_INDEX_BYTES = 4 * 1024 * 1024
DEFAULT_PYTHON_TIMEOUT = 8.0
DEFAULT_PYTHON_PROBE_BYTES = 64 * 1024
DEFAULT_PYTHON_REQUEST = "3.11"
DEFAULT_PYTHON_DOWNLOAD_WORKERS = 4
DEFAULT_PYTHON_REQUEST_TIMEOUT = 20.0
TORCH_MINIMUM = "2.6.0"
# A wheel built for a lower CUDA runtime can run with a newer compatible
# NVIDIA driver.  Select the newest backend supported by the detected runtime.
CUDA_BACKENDS = (
    ((13, 2), "cu132"),
    ((13, 0), "cu130"),
    ((12, 9), "cu129"),
    ((12, 8), "cu128"),
    ((12, 6), "cu126"),
    ((12, 5), "cu125"),
    ((12, 4), "cu124"),
    ((12, 3), "cu123"),
    ((12, 2), "cu122"),
    ((12, 1), "cu121"),
    ((12, 0), "cu120"),
    ((11, 8), "cu118"),
    ((11, 7), "cu117"),
    ((11, 6), "cu116"),
    ((11, 3), "cu113"),
    ((11, 1), "cu111"),
    ((11, 0), "cu110"),
    ((10, 2), "cu102"),
    ((10, 1), "cu101"),
    ((10, 0), "cu100"),
)
TORCH_BACKENDS = {backend for _, backend in CUDA_BACKENDS} | {"cpu"}
PYTHON_BUILD_STANDALONE_BASE = (
    "https://github.com/astral-sh/python-build-standalone/releases/download"
)
PYTHON_MIRRORS = (
    "https://mirrors.ustc.edu.cn/github-release/astral-sh/python-build-standalone",
    "https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone",
    "https://releases.astral.sh/github/python-build-standalone/releases/download",
    PYTHON_BUILD_STANDALONE_BASE,
)


@dataclass(frozen=True)
class ProbeResult:
    index: str
    elapsed: float | None = None
    bytes_read: int = 0
    error: str | None = None

    @property
    def speed(self) -> float:
        if self.elapsed is None or self.elapsed <= 0:
            return 0.0
        return self.bytes_read / self.elapsed

    @property
    def available(self) -> bool:
        return self.error is None and self.bytes_read > 0


@dataclass(frozen=True)
class CudaInfo:
    backend: str
    version: str | None
    source: str


@dataclass(frozen=True)
class PythonMirrorResult:
    mirror: str
    elapsed: float | None = None
    bytes_read: int = 0
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.error is None and self.bytes_read > 0

    @property
    def speed(self) -> float:
        if self.elapsed is None or self.elapsed <= 0:
            return 0.0
        return self.bytes_read / self.elapsed


def _parse_cuda_version(output: str) -> tuple[int, int] | None:
    """Extract a CUDA major/minor version from nvidia-smi or nvcc output."""

    patterns = (
        r"CUDA Version:\s*(\d+)\.(\d+)",
        r"release\s+(\d+)\.(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def detect_cuda() -> CudaInfo:
    """Detect a usable NVIDIA driver and select a compatible PyTorch backend.

    ``nvidia-smi`` reports the maximum CUDA runtime supported by the driver,
    which is the relevant value for PyTorch wheels.  ``nvcc`` is only used as
    a fallback when nvidia-smi is not installed (common in build containers).
    """

    smi_path = shutil.which("nvidia-smi")
    if smi_path:
        try:
            completed = subprocess.run(
                [smi_path], capture_output=True, text=True, timeout=4, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            completed = None
        if completed is None or completed.returncode != 0:
            # A present but unusable driver should not cause a CUDA wheel to be
            # installed merely because nvcc happens to be on PATH.
            return CudaInfo("cpu", None, "nvidia-smi 未检测到可用 CUDA 驱动")
        version = _parse_cuda_version(completed.stdout + completed.stderr)
        if version:
            for minimum, backend in CUDA_BACKENDS:
                if version >= minimum:
                    return CudaInfo(backend, f"{version[0]}.{version[1]}", "nvidia-smi")
        return CudaInfo("cpu", None, "nvidia-smi 未返回 CUDA 版本")

    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        try:
            completed = subprocess.run(
                [nvcc_path, "--version"], capture_output=True, text=True, timeout=4, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            completed = None
        if completed is not None and completed.returncode == 0:
            version = _parse_cuda_version(completed.stdout + completed.stderr)
            if version:
                for minimum, backend in CUDA_BACKENDS:
                    if version >= minimum:
                        return CudaInfo(backend, f"{version[0]}.{version[1]}", "nvcc")

    return CudaInfo("cpu", None, "未检测到 NVIDIA CUDA，使用 CPU 版 PyTorch")


def _python_asset_url(uv: str, python_request: str, timeout: float) -> str | None:
    """Ask uv for the download URL of the requested CPython build."""

    command = [
        uv,
        "python",
        "list",
        python_request,
        "--only-downloads",
        "--show-urls",
        "--output-format",
        "json",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    try:
        downloads = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    for download in downloads:
        if (
            download.get("implementation") == "cpython"
            and download.get("variant", "default") == "default"
            and not download.get("prerelease")
            and download.get("url")
        ):
            return str(download["url"])
    return None


def _version_matches_request(version: str, request: str) -> bool:
    """Match ``3.11`` against any stable 3.11.x, or an exact version."""

    try:
        wanted = tuple(int(part) for part in request.split("."))
        actual = tuple(int(part) for part in version.split("."))
    except ValueError:
        return False
    return actual[: len(wanted)] == wanted


def _uv_managed_python(uv: str, python_request: str, timeout: float) -> str | None:
    """Find only a uv-managed stable CPython, never a system or project venv."""

    try:
        listed = subprocess.run(
            [
                uv,
                "python",
                "list",
                python_request,
                "--only-installed",
                "--output-format",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        python_dir = subprocess.run(
            [uv, "python", "dir"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if listed.returncode != 0 or python_dir.returncode != 0:
        return None
    try:
        downloads = json.loads(listed.stdout)
    except json.JSONDecodeError:
        return None
    managed_root = Path(python_dir.stdout.strip()).expanduser().resolve()
    for download in downloads:
        path = download.get("path")
        if (
            download.get("implementation") == "cpython"
            and download.get("variant", "default") == "default"
            and not download.get("prerelease")
            and path
            and _version_matches_request(str(download.get("version", "")), python_request)
        ):
            candidate = Path(path).expanduser()
            try:
                if candidate.resolve().is_relative_to(managed_root):
                    return str(candidate)
            except (OSError, RuntimeError):
                continue
    return None


def _python_mirror_url(mirror: str, asset_url: str) -> str:
    """Apply uv's ``--mirror`` replacement to a concrete asset URL."""

    # uv replaces the canonical GitHub release prefix, preserving the release
    # tag and filename.  This also works for uv builds whose default source is
    # the Astral CDN (the URL returned above is only used for the suffix).
    suffix = asset_url
    for prefix in (
        PYTHON_BUILD_STANDALONE_BASE,
        "https://releases.astral.sh/github/python-build-standalone/releases/download",
    ):
        if asset_url.startswith(prefix):
            suffix = asset_url[len(prefix) :].lstrip("/")
            break
    return f"{mirror.rstrip('/')}/{suffix}"


def probe_python_mirror(
    mirror: str, asset_url: str, timeout: float, probe_bytes: int
) -> PythonMirrorResult:
    """Read a small prefix of the CPython archive to measure a mirror."""

    request = urllib.request.Request(
        _python_mirror_url(mirror, asset_url),
        headers={
            "Range": f"bytes=0-{probe_bytes - 1}",
            "Accept-Encoding": "identity",
            "User-Agent": "legged-wbc-mjlab/uv-auto-sync",
        },
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read(probe_bytes)
        elapsed = time.perf_counter() - started
        return PythonMirrorResult(mirror, elapsed, len(payload))
    except (OSError, urllib.error.URLError, ValueError) as exc:
        return PythonMirrorResult(mirror, error=f"{type(exc).__name__}: {exc}")


def probe_python_mirrors(
    mirrors: list[str], asset_url: str, timeout: float, probe_bytes: int
) -> list[PythonMirrorResult]:
    results: dict[str, PythonMirrorResult] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(mirrors))) as executor:
        futures = {
            executor.submit(probe_python_mirror, mirror, asset_url, timeout, probe_bytes): mirror
            for mirror in mirrors
        }
        for future in as_completed(futures):
            result = future.result()
            results[result.mirror] = result
    positions = {mirror: position for position, mirror in enumerate(mirrors)}
    return sorted(
        results.values(),
        key=lambda result: (
            not result.available,
            -result.speed,
            positions[result.mirror],
        ),
    )


def _python_archive_paths(cache_dir: Path, asset_url: str) -> tuple[Path, Path]:
    """Return the cached archive and local ``--mirror`` root for one asset."""

    parsed = urllib.parse.urlparse(asset_url)
    parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"无法解析 Python 下载地址：{asset_url}")
    tag, filename = parts[-2], parts[-1]
    mirror_root = cache_dir / "python-standalone-mirror"
    return mirror_root / tag / filename, mirror_root


def _python_archive_size(url: str, timeout: float | None) -> int | None:
    """Read the remote archive size with a one-byte ranged request."""

    request = urllib.request.Request(
        url,
        headers={
            "Range": "bytes=0-0",
            "Accept-Encoding": "identity",
            "User-Agent": "legged-wbc-mjlab/uv-auto-sync",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_range = response.headers.get("Content-Range", "")
            match = re.search(r"/([0-9]+)$", content_range)
            if match:
                return int(match.group(1))
            content_length = response.headers.get("Content-Length")
            if content_length and response.status == 206:
                return int(content_length)
    except (OSError, urllib.error.URLError, ValueError):
        return None
    return None


def _download_python_sequential(
    source_url: str,
    target: Path,
    timeout: float | None,
    total_size: int | None = None,
) -> bool:
    """Download one archive in a single stream, showing byte progress."""

    request = urllib.request.Request(
        source_url,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "legged-wbc-mjlab/uv-auto-sync",
        },
    )
    temporary = target.with_name(target.name + ".part")
    downloaded = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    print(
                        f"\rPython 下载进度：{downloaded / total_size:.1%} "
                        f"({downloaded / 1024 / 1024:.1f}/{total_size / 1024 / 1024:.1f} MiB)",
                        end="",
                        flush=True,
                    )
        print()
        if total_size is not None and downloaded != total_size:
            raise RuntimeError("顺序下载文件大小校验失败")
        os.replace(temporary, target)
        target.with_name(target.name + ".complete").touch()
        return True
    except (OSError, urllib.error.URLError, RuntimeError):
        temporary.unlink(missing_ok=True)
        return False


def _download_python_archive(
    source_url: str,
    target: Path,
    timeout: float | None,
    workers: int,
) -> bool:
    """Download an archive using ranged chunks, with a sequential fallback."""

    target.parent.mkdir(parents=True, exist_ok=True)
    complete_marker = target.with_name(target.name + ".complete")
    if complete_marker.exists() and target.exists() and target.stat().st_size > 0:
        print(f"复用 Python 下载缓存：{target}")
        return True

    # ``install_timeout`` is the timeout for a whole channel attempt.  Do not
    # use it as the socket timeout for every ranged request: a dead connection
    # would otherwise look frozen for ten minutes.  A shorter inactivity limit
    # keeps failover responsive while allowing slow but continuously streaming
    # links to finish.
    request_timeout = (
        min(timeout, DEFAULT_PYTHON_REQUEST_TIMEOUT)
        if timeout is not None
        else DEFAULT_PYTHON_REQUEST_TIMEOUT
    )
    print("正在检查 Python 压缩包是否支持分段下载...", flush=True)
    total_size = _python_archive_size(source_url, request_timeout)
    if total_size is None:
        # Some proxies strip Range headers.  A normal streaming download still
        # works, and it is safer than issuing several full-file requests.
        return _download_python_sequential(source_url, target, request_timeout)

    workers = max(1, min(workers, 16, math.ceil(total_size / (4 * 1024 * 1024))))
    chunk_size = math.ceil(total_size / workers)
    part_paths = [target.with_name(f"{target.name}.part{position}") for position in range(workers)]

    progress_lock = threading.Lock()
    progress_bytes = 0
    next_progress = 1024 * 1024

    def record_progress(count: int) -> None:
        nonlocal progress_bytes, next_progress
        with progress_lock:
            progress_bytes += count
            if progress_bytes >= next_progress or progress_bytes == total_size:
                print(
                    f"\rPython 下载进度：{progress_bytes / total_size:.1%} "
                    f"({progress_bytes / 1024 / 1024:.1f}/{total_size / 1024 / 1024:.1f} MiB)",
                    end="",
                    flush=True,
                )
                next_progress = ((progress_bytes // (1024 * 1024)) + 1) * 1024 * 1024

    def fetch_chunk(position: int) -> None:
        start = position * chunk_size
        end = min(total_size - 1, start + chunk_size - 1)
        request = urllib.request.Request(
            source_url,
            headers={
                "Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity",
                "User-Agent": "legged-wbc-mjlab/uv-auto-sync",
            },
        )
        with urllib.request.urlopen(request, timeout=request_timeout) as response:
            content_range = response.headers.get("Content-Range", "")
            if response.status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
                raise RuntimeError("下载服务器未返回分段响应")
            expected = end - start + 1
            with part_paths[position].open("wb") as output:
                remaining = expected
                while remaining:
                    payload = response.read(min(1024 * 1024, remaining))
                    if not payload:
                        raise RuntimeError("分段下载提前结束")
                    output.write(payload)
                    record_progress(len(payload))
                    remaining -= len(payload)

    print(f"启用 {workers} 路并行下载 Python（{total_size / 1024 / 1024:.1f} MiB）")
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(fetch_chunk, position) for position in range(workers)]
            for position, future in enumerate(as_completed(futures), start=1):
                future.result()
                print(f"\nPython 分段下载完成：{position}/{workers}")
        with target.open("wb") as output:
            for part_path in part_paths:
                with part_path.open("rb") as part:
                    while payload := part.read(1024 * 1024):
                        output.write(payload)
        if target.stat().st_size != total_size:
            raise RuntimeError("下载文件大小校验失败")
        complete_marker.touch()
        return True
    except (OSError, urllib.error.URLError, RuntimeError):
        print("\n分段下载不可用，回退到单流下载...", file=sys.stderr)
        return _download_python_sequential(source_url, target, request_timeout, total_size)
    finally:
        for part_path in part_paths:
            part_path.unlink(missing_ok=True)


def download_python_to_local_mirror(
    mirror: str,
    asset_url: str,
    cache_dir: Path,
    timeout: float | None,
    workers: int,
) -> tuple[Path, Path] | None:
    """Download via ``mirror`` and return ``(archive, mirror_root)``."""

    target, mirror_root = _python_archive_paths(cache_dir, asset_url)
    source_url = _python_mirror_url(mirror, asset_url)
    print(f"Python 缓存文件：{target}")
    try:
        if _download_python_archive(source_url, target, timeout, workers):
            return target, mirror_root
    except (OSError, ValueError) as exc:
        print(f"Python 下载缓存准备失败：{exc}", file=sys.stderr)
    return None


def _run_uv(
    uv: str,
    args: list[str],
    cwd: Path,
    timeout: float | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [uv, *args],
        cwd=cwd,
        timeout=timeout,
        check=False,
        capture_output=capture_output,
        text=True,
    )


def ensure_python(
    uv: str,
    project_root: Path,
    python_request: str,
    python_mirrors: list[str],
    timeout: float,
    probe_bytes: int,
    install_timeout: float | None,
    cache_dir: Path,
    download_workers: int,
    offline: bool = False,
    allow_install: bool = True,
) -> str:
    """Return a usable interpreter, downloading CPython only when necessary."""

    interpreter = _uv_managed_python(uv, python_request, timeout)
    if interpreter is not None:
        print(f"复用已有 Python {python_request}：{interpreter}")
        return interpreter

    if offline:
        raise RuntimeError(
            f"离线模式下没有已缓存的 uv Python {python_request}；"
            "请先在有网机器运行一次 setup.sh。"
        )

    asset_url = _python_asset_url(uv, python_request, timeout)
    if asset_url is None:
        raise RuntimeError(f"uv 无法获取 Python {python_request} 的下载信息。")
    print(f"本机未找到 Python {python_request}，正在测速 Python 发行包下载通道...")
    if not python_mirrors:
        python_mirrors = list(PYTHON_MIRRORS)
    mirror_results = probe_python_mirrors(
        python_mirrors, asset_url, timeout=timeout, probe_bytes=probe_bytes
    )
    for position, result in enumerate(mirror_results, start=1):
        if result.available:
            print(
                f"{position}. {result.mirror} "
                f"{result.speed / 1024:.1f} KiB/s, {result.elapsed:.2f}s"
            )
        else:
            print(f"{position}. {result.mirror} 不可用 ({result.error})")

    if not allow_install:
        return f"未安装（dry-run；推荐镜像：{mirror_results[0].mirror}）"

    available = [result for result in mirror_results if result.available]
    # If the probe is blocked by a proxy, still let uv attempt its configured
    # default source rather than failing before the useful error is displayed.
    attempts: list[str | None] = [result.mirror for result in available] or [None]
    for position, mirror in enumerate(attempts, start=1):
        local_mirror: Path | None = None
        if mirror is not None:
            downloaded = download_python_to_local_mirror(
                mirror,
                asset_url,
                cache_dir,
                timeout=install_timeout,
                workers=download_workers,
            )
            if downloaded is None:
                if position < len(attempts):
                    print("该 Python 下载通道失败，正在切换备用通道...", file=sys.stderr)
                continue
            _, local_mirror = downloaded
        install_args = ["python", "install", python_request]
        if local_mirror is not None:
            install_args.extend(("--mirror", local_mirror.as_uri()))
        print(
            f"安装 Python {python_request}（{position}/{len(attempts)}）"
            + (f"：{mirror}" if mirror else "：uv 默认源")
        )
        try:
            installed = _run_uv(uv, install_args, project_root, timeout=install_timeout)
        except subprocess.TimeoutExpired:
            print(
                f"该 Python 下载通道超过 {install_timeout:g} 秒仍未完成，正在切换...",
                file=sys.stderr,
            )
            continue
        except FileNotFoundError as exc:
            raise RuntimeError(f"找不到 uv：{uv}") from exc
        if installed.returncode == 0:
            break
        if position < len(attempts):
            print("该 Python 下载通道失败，正在切换备用通道...", file=sys.stderr)
    else:
        raise RuntimeError(f"Python {python_request} 下载失败。")

    interpreter = _uv_managed_python(uv, python_request, timeout)
    if interpreter is None:
        raise RuntimeError(f"Python {python_request} 已安装但无法定位解释器。")
    return interpreter


def project_venv_python(project_root: Path) -> Path:
    """Return the platform-specific Python path inside the project venv."""

    candidates = (
        project_root / ".venv" / "bin" / "python",
        project_root / ".venv" / "Scripts" / "python.exe",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # On Linux this is the path uv creates; returning it keeps the eventual uv
    # error actionable if sync failed before creating the environment.
    return candidates[0]


def normalize_index(index: str) -> str:
    """Return a canonical PEP 503 index URL without duplicate slashes."""

    normalized = index.strip().rstrip("/")
    if not normalized:
        raise ValueError("索引地址不能为空")
    return normalized


def display_index(index: str) -> str:
    """Hide credentials if an index URL contains embedded authentication."""

    parsed = urllib.parse.urlsplit(index)
    if parsed.username is None:
        return index
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urllib.parse.urlunsplit((parsed.scheme, f"{parsed.username}:***@{host}", parsed.path, parsed.query, parsed.fragment))


def probe_index(index: str, timeout: float, probe_bytes: int) -> ProbeResult:
    """Measure package-file throughput, not merely index-page latency."""

    probe_url = f"{index}/{PROBE_PACKAGE}/"
    index_request = urllib.request.Request(
        probe_url,
        headers={"User-Agent": "legged-wbc-mjlab/uv-auto-sync"},
    )
    try:
        with urllib.request.urlopen(index_request, timeout=timeout) as response:
            if not 200 <= response.status < 300:
                raise urllib.error.HTTPError(
                    probe_url,
                    response.status,
                    f"HTTP {response.status}",
                    response.headers,
                    None,
                )
            index_payload = response.read(MAX_INDEX_BYTES)

        page = index_payload.decode("utf-8", errors="ignore")
        links = re.findall(r'''href=["']([^"']+\.whl(?:#[^"']*)?)["']''', page, re.IGNORECASE)
        if not links:
            raise ValueError("索引页面没有可测速的 wheel 文件")
        preferred = [
            link
            for link in links
            if "cp311" in link and "manylinux" in link and "x86_64" in link
        ]
        artifact = html.unescape((preferred or links)[-1]).split("#", maxsplit=1)[0]
        artifact_url = urllib.parse.urljoin(probe_url, artifact)
        artifact_request = urllib.request.Request(
            artifact_url,
            headers={
                "Range": f"bytes=0-{probe_bytes - 1}",
                "User-Agent": "legged-wbc-mjlab/uv-auto-sync",
            },
        )
        started = time.perf_counter()
        with urllib.request.urlopen(artifact_request, timeout=timeout) as response:
            payload = response.read(probe_bytes)
        elapsed = time.perf_counter() - started
        return ProbeResult(index=index, elapsed=elapsed, bytes_read=len(payload))
    except (OSError, urllib.error.URLError, ValueError) as exc:
        return ProbeResult(index=index, error=f"{type(exc).__name__}: {exc}")


def probe_indexes(indexes: list[str], timeout: float, probe_bytes: int) -> list[ProbeResult]:
    """Probe indexes concurrently while retaining a deterministic tie order."""

    results: dict[str, ProbeResult] = {}
    workers = min(8, len(indexes))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(probe_index, index, timeout, probe_bytes): position
            for position, index in enumerate(indexes)
        }
        positions = {index: position for position, index in enumerate(indexes)}
        for future in as_completed(futures):
            result = future.result()
            results[result.index] = result

    # Speed is the primary metric; the original order is a stable tie breaker.
    return sorted(
        results.values(),
        key=lambda result: (
            not result.available,
            -result.speed,
            positions[result.index],
        ),
    )


def format_result(result: ProbeResult) -> str:
    if not result.available:
        return f"不可用 ({result.error})"
    return f"{result.speed / 1024:.1f} KiB/s, {result.elapsed:.2f}s"


def normalize_torch_requirement(value: str) -> str:
    """Turn a version constraint into a valid uv package requirement.

    ``uv pip install`` expects a package name.  Passing ``>=2.6.0`` by itself
    is invalid; it must be combined with the package name as
    ``torch>=2.6.0``.
    """

    requirement = value.strip()
    if re.match(r"^torch(?:\[[^]]+\])?(?:[<>=!~].*)?$", requirement, re.IGNORECASE):
        return requirement
    if re.match(r"^[<>=!~]", requirement):
        return f"torch{requirement}"
    if re.fullmatch(r"\d+(?:\.\d+){1,3}", requirement):
        return f"torch=={requirement}"
    raise ValueError(
        "PyTorch 版本必须是约束（如 >=2.6.0）、版本号（如 2.6.0），"
        "或完整包要求（如 torch==2.6.0）"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="测速 PyPI 镜像并使用最快的镜像执行 uv sync。",
        epilog=(
            "示例: python3 tools/uv_auto_sync.py --extra sim2sim\n"
            "      python3 tools/uv_auto_sync.py --index https://pypi.org/simple"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--index",
        action="append",
        dest="indexes",
        help="自定义镜像地址；指定后只测速这些地址（可重复传入）。",
    )
    parser.add_argument(
        "--index-file",
        type=Path,
        help="包含镜像地址的文本文件（每行一个，空行和 # 注释会忽略）。指定后只测速文件中的地址。",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"单个镜像的测速超时时间（秒，默认 {DEFAULT_TIMEOUT:g}）。",
    )
    parser.add_argument(
        "--probe-bytes",
        type=int,
        default=DEFAULT_PROBE_BYTES,
        help="测速时从真实 wheel 文件读取的字节数（默认 1 MiB）。",
    )
    parser.add_argument(
        "--sync-timeout",
        type=float,
        default=DEFAULT_SYNC_TIMEOUT,
        help=(
            f"单个镜像的实际安装尝试最长等待时间（秒，默认 {DEFAULT_SYNC_TIMEOUT:g}）；"
            "设为 0 表示不限制。超时会自动切换到下一个镜像。"
        ),
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="传给 uv sync 的可选依赖组，例如 --extra sim2sim（可重复传入）。",
    )
    parser.add_argument(
        "--uv",
        default="uv",
        help="uv 可执行文件路径（默认从 PATH 查找）。",
    )
    parser.add_argument(
        "--python-version",
        default=DEFAULT_PYTHON_REQUEST,
        help=f"项目 Python 版本（默认 {DEFAULT_PYTHON_REQUEST}）。",
    )
    parser.add_argument(
        "--python-mirror",
        action="append",
        dest="python_mirrors",
        help="Python standalone 下载镜像；指定后只测速这些地址（可重复传入）。",
    )
    parser.add_argument(
        "--python-mirror-file",
        type=Path,
        help="包含 Python standalone 镜像地址的文本文件（每行一个）。",
    )
    parser.add_argument(
        "--python-timeout",
        type=float,
        default=DEFAULT_PYTHON_TIMEOUT,
        help=f"Python 下载通道测速超时时间（秒，默认 {DEFAULT_PYTHON_TIMEOUT:g}）。",
    )
    parser.add_argument(
        "--python-install-timeout",
        type=float,
        default=DEFAULT_PYTHON_INSTALL_TIMEOUT,
        help=(
            f"单个 Python 下载通道最长等待时间（秒，默认 {DEFAULT_PYTHON_INSTALL_TIMEOUT:g}）；"
            "设为 0 表示不限制。"
        ),
    )
    parser.add_argument(
        "--python-download-workers",
        type=int,
        default=DEFAULT_PYTHON_DOWNLOAD_WORKERS,
        help=(
            f"Python standalone 分段并行下载路数（默认 {DEFAULT_PYTHON_DOWNLOAD_WORKERS}，"
            "范围 1-16）。"
        ),
    )
    parser.add_argument(
        "--torch-backend",
        default="auto",
        help=(
            "PyTorch 后端：auto 根据 nvidia-smi/nvcc 选择，或手动指定 cpu、cu118、cu124 等；"
            "默认 auto。"
        ),
    )
    parser.add_argument(
        "--torch-version",
        default=f">={TORCH_MINIMUM}",
        help=f"PyTorch 版本约束（默认 >= {TORCH_MINIMUM}）。",
    )
    parser.add_argument(
        "--torch-timeout",
        type=float,
        default=DEFAULT_TORCH_TIMEOUT,
        help=(
            f"PyTorch 安装最长等待时间（秒，默认 {DEFAULT_TORCH_TIMEOUT:g}）；"
            "设为 0 表示不限制。"
        ),
    )
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="跳过 PyTorch 的单独安装步骤（不推荐，适合已有环境的场景）。",
    )
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        help="本地 wheel 目录；可与网络源一起使用，也可配合 --offline 完全离线安装。",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="完全离线安装，只使用 uv 缓存和 --wheelhouse，不测速或访问网络。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只测速并显示将要使用的镜像，不执行 uv sync。",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="同步失败时不切换到测速排名靠后的镜像。",
    )
    parser.add_argument(
        "uv_args",
        nargs=argparse.REMAINDER,
        help="在 '--' 后追加传给 uv sync 的参数。",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout 必须大于 0")
    if args.probe_bytes <= 0:
        parser.error("--probe-bytes 必须大于 0")
    if args.sync_timeout < 0:
        parser.error("--sync-timeout 不能小于 0")
    if args.python_timeout <= 0:
        parser.error("--python-timeout 必须大于 0")
    if args.python_install_timeout < 0:
        parser.error("--python-install-timeout 不能小于 0")
    if not 1 <= args.python_download_workers <= 16:
        parser.error("--python-download-workers 必须在 1 到 16 之间")
    if args.torch_timeout < 0:
        parser.error("--torch-timeout 不能小于 0")
    if not re.fullmatch(r"\d+(?:\.\d+){0,2}", args.python_version):
        parser.error("--python-version 应为类似 3.11 或 3.11.16 的版本号")

    # UV_DEFAULT_INDEX is the current uv setting.  UV_INDEX_URL remains as a
    # compatibility fallback for older uv setups, but is deprecated upstream.
    configured_index = os.environ.get("UV_DEFAULT_INDEX") or os.environ.get("UV_INDEX_URL")
    file_indexes: list[str] = []
    if args.index_file is not None:
        try:
            file_indexes = [
                line.split("#", maxsplit=1)[0].strip()
                for line in args.index_file.expanduser().read_text(encoding="utf-8").splitlines()
            ]
            file_indexes = [line for line in file_indexes if line]
        except OSError as exc:
            parser.error(f"无法读取镜像列表文件 {args.index_file}：{exc}")
    if args.indexes or file_indexes:
        raw_indexes = list(args.indexes or []) + file_indexes
    else:
        raw_indexes = ([configured_index] if configured_index else []) + list(DEFAULT_INDEXES)
    try:
        indexes = [normalize_index(index) for index in raw_indexes]
    except ValueError as exc:
        parser.error(str(exc))

    # Keep the first occurrence if a user accidentally supplies a duplicate.
    indexes = list(dict.fromkeys(indexes))
    torch_backend = args.torch_backend.lower()
    if torch_backend == "auto":
        cuda_info = detect_cuda()
        torch_backend = cuda_info.backend
        detected_message = (
            f"CUDA 检测：{cuda_info.version or '无'}（{cuda_info.source}），"
            f"PyTorch 后端：{torch_backend}"
        )
    else:
        if torch_backend not in TORCH_BACKENDS:
            parser.error(
                f"不支持的 --torch-backend={args.torch_backend!r}；可用值包括 auto、cpu、"
                + "、".join(sorted(TORCH_BACKENDS - {"cpu"}))
            )
        detected_message = f"PyTorch 后端：{torch_backend}（手动指定）"

    try:
        torch_requirement = normalize_torch_requirement(args.torch_version)
    except ValueError as exc:
        parser.error(str(exc))

    project_root = Path(__file__).resolve().parents[1]
    cache_dir = Path(
        os.environ.get(
            "UV_CACHE_DIR",
            str(Path.home() / ".cache" / "uv"),
        )
    ).expanduser().resolve()
    configured_python_mirror = os.environ.get("UV_PYTHON_INSTALL_MIRROR")
    python_file_mirrors: list[str] = []
    if args.python_mirror_file is not None:
        try:
            python_file_mirrors = [
                line.split("#", maxsplit=1)[0].strip()
                for line in args.python_mirror_file.expanduser()
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            python_file_mirrors = [line for line in python_file_mirrors if line]
        except OSError as exc:
            parser.error(f"无法读取 Python 镜像列表文件 {args.python_mirror_file}：{exc}")
    python_mirrors = list(
        dict.fromkeys(
            (args.python_mirrors or [])
            + python_file_mirrors
            + ([configured_python_mirror] if configured_python_mirror else [])
            + list(PYTHON_MIRRORS)
        )
    )
    wheelhouse: Path | None = None
    if args.wheelhouse is not None:
        wheelhouse = args.wheelhouse.expanduser().resolve()
        if not wheelhouse.is_dir():
            parser.error(f"wheelhouse 目录不存在：{wheelhouse}")
    try:
        python_interpreter = ensure_python(
            args.uv,
            project_root,
            args.python_version,
            python_mirrors,
            timeout=args.python_timeout,
            probe_bytes=DEFAULT_PYTHON_PROBE_BYTES,
            install_timeout=(
                None if args.python_install_timeout == 0 else args.python_install_timeout
            ),
            cache_dir=cache_dir,
            download_workers=args.python_download_workers,
            offline=args.offline,
            allow_install=not args.dry_run,
        )
    except RuntimeError as exc:
        print(f"Python 环境准备失败：{exc}", file=sys.stderr)
        return 3

    if args.offline:
        print("离线模式：跳过镜像测速，只使用 uv 缓存和本地 wheelhouse。")
        available: list[ProbeResult] = [
            ProbeResult("离线缓存", elapsed=0.0, bytes_read=1),
        ]
    else:
        print(f"正在测速 {len(indexes)} 个 PyPI 镜像（探测 {PROBE_PACKAGE}）...")
        results = probe_indexes(indexes, args.timeout, args.probe_bytes)
        for position, result in enumerate(results, start=1):
            print(f"{position}. {display_index(result.index):<58} {format_result(result)}")

        available = [result for result in results if result.available]
        if not available:
            print("没有可用的 PyPI 镜像，请检查网络，或用 --index 指定内部镜像。", file=sys.stderr)
            return 2

    if args.uv_args and args.uv_args[0] == "--":
        args.uv_args = args.uv_args[1:]
    if any(option in {"--locked", "--frozen"} for option in args.uv_args):
        print(
            "提示：使用 --locked/--frozen 时 uv 不会更新 uv.lock，已锁定的下载地址可能仍来自旧镜像。",
            file=sys.stderr,
        )

    if args.dry_run:
        print(f"将使用最快镜像：{available[0].index}")
        print(f"将使用 Python：{python_interpreter}")
        if not args.skip_torch:
            print(detected_message)
        return 0

    sync_timeout = None if args.sync_timeout == 0 else args.sync_timeout
    sync_attempts = available if not args.no_fallback else available[:1]
    selected_result: ProbeResult | None = None
    last_returncode = 1
    for attempt, result in enumerate(sync_attempts, start=1):
        # PyTorch is intentionally installed separately.  uv's project lock
        # file contains one resolved torch artifact, while the right wheel is
        # a machine-local choice (CPU, CUDA, ROCm, ...).
        command = [
            args.uv,
            "sync",
            "--python",
            python_interpreter,
            "--inexact",
        ]
        if not args.offline:
            command.extend(("--default-index", result.index))
        else:
            command.extend(("--offline", "--no-index"))
        if wheelhouse is not None:
            command.extend(("--find-links", str(wheelhouse)))
        if not args.skip_torch:
            command.extend(("--no-install-package", "torch"))
        for extra in args.extra:
            command.extend(("--extra", extra))
        command.extend(args.uv_args)
        print(f"\n使用镜像（{attempt}/{len(sync_attempts)}）：{result.index}")
        try:
            completed = run(command, cwd=project_root, check=False, timeout=sync_timeout)
        except subprocess.TimeoutExpired:
            completed = None
            print(
                f"该镜像安装超过 {args.sync_timeout:g} 秒仍未完成，正在切换到下一个镜像...",
                file=sys.stderr,
            )
        except FileNotFoundError:
            print(f"找不到 uv：{args.uv}，请先安装 uv 或通过 --uv 指定路径。", file=sys.stderr)
            return 127
        if completed is not None:
            last_returncode = completed.returncode
        if completed is not None and completed.returncode == 0:
            selected_result = result
            break
        if attempt < len(sync_attempts):
            print("该镜像同步失败，正在切换到下一个可用镜像...", file=sys.stderr)

    if selected_result is None:
        print("所有候选镜像的项目依赖同步都失败了。", file=sys.stderr)
        return last_returncode

    if args.skip_torch:
        print("uv 环境同步完成（已跳过 PyTorch）。")
        return 0

    # PyTorch uses its own CUDA/CPU backend index.  A failure here is not
    # fixed by repeating the already-successful project sync against every
    # ordinary PyPI mirror, so report it directly instead of masking the real
    # error with a misleading mirror retry sequence.
    torch_command = [
        args.uv,
        "pip",
        "install",
        "--reinstall-package",
        "torch",
        "--python",
        str(project_venv_python(project_root)),
    ]
    if args.offline:
        # In offline mode the wheelhouse itself determines CPU/CUDA; asking uv
        # to consult the remote PyTorch backend index would defeat the point.
        torch_command.extend(("--offline", "--no-index"))
    else:
        torch_command.extend(("--torch-backend", torch_backend, "--default-index", selected_result.index))
    if wheelhouse is not None:
        torch_command.extend(("--find-links", str(wheelhouse)))
    torch_command.append(torch_requirement)
    torch_timeout = None if args.torch_timeout == 0 else args.torch_timeout
    print(f"{detected_message}，正在安装兼容的 PyTorch：{torch_requirement}")
    try:
        torch_completed = run(
            torch_command,
            cwd=project_root,
            check=False,
            timeout=torch_timeout,
        )
    except subprocess.TimeoutExpired:
        print(
            f"PyTorch 安装超过 {args.torch_timeout:g} 秒仍未完成。"
            "项目依赖已完成，请检查 PyTorch 下载通道或手动重试。",
            file=sys.stderr,
        )
        return 124
    except FileNotFoundError:
        print(f"找不到 uv：{args.uv}，请先安装 uv 或通过 --uv 指定路径。", file=sys.stderr)
        return 127
    if torch_completed.returncode != 0:
        print(
            "PyTorch 安装失败；项目普通依赖已经完成，不再重复尝试其他 PyPI 镜像。",
            file=sys.stderr,
        )
        return torch_completed.returncode
    print("uv 环境和 PyTorch 同步完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
