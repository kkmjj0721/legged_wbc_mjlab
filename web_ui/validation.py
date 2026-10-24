"""Serialized subprocess jobs and transactional generation for the local UI."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class JobManager:
    def __init__(self, repo_root: Path, workspace: Path):
        self.repo_root = repo_root
        self.workspace = workspace
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robot-migration")
        self._lock = threading.RLock()
        self._jobs = {}

    def submit(self, mode: str, task_ids: list[str], operation=None, spec=None) -> dict:
        job_id = uuid.uuid4().hex
        directory = self.workspace / "jobs" / job_id
        directory.mkdir(parents=True)
        with self._lock:
            job = dict(id=job_id, mode=mode, task_ids=task_ids, status="queued", result=None,
                       error=None, created_at=time.time(), directory=directory, process=None,
                       cancelled=threading.Event(), finished=threading.Event())
            self._jobs[job_id] = job
            self._pool.submit(self._run, job, operation, spec)
        return {"job_id": job_id, "id": job_id, "status": "queued"}

    def _run(self, job, operation, spec):
        try:
            if job["cancelled"].is_set():
                job["status"] = "stopped"
                return
            job["status"] = "running"
            result = operation(job) if operation else self.run_worker(job, job["mode"], job["task_ids"], spec=spec)
            job["result"] = result
            job["status"] = result.get("status", "passed")
        except Exception as exc:
            job["error"] = str(exc)
            job["status"] = "stopped" if job["cancelled"].is_set() else "failed"
            self.log(job, f"失败: {exc}")
        finally:
            job["process"] = None
            try:
                from .lifecycle import cleanup_temporary

                cleanup = cleanup_temporary(job["directory"])
                if isinstance(job["result"], dict):
                    job["result"].setdefault("cleanup", cleanup)
                else:
                    job["cleanup"] = cleanup
            except Exception as exc:
                self.log(job, f"临时文件清理提示：{exc}")
            job["finished_at"] = time.time()
            self._write_state(job)
            job["finished"].set()

    def _write_state(self, job):
        public = self.get(job["id"])
        (job["directory"] / "job.json").write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")

    def log(self, job, message):
        with (job["directory"] / "output.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    def run_worker(self, job, mode, task_ids, *, spec=None, stage_root=None):
        if job["cancelled"].is_set():
            raise RuntimeError("任务已停止")
        directory = job["directory"]
        tag = f"{mode}-{uuid.uuid4().hex[:8]}"
        result_path = directory / f"{tag}.json"
        if mode == "pd_response":
            command = [sys.executable, "-B", "-m", "web_ui.pd_response", "--result", str(result_path),
                       "--video-dir", str(directory / "videos")]
        else:
            command = [sys.executable, "-B", "-m", "web_ui.worker", "--repo-root", str(self.repo_root),
                       "--mode", mode, "--result", str(result_path)]
        for task_id in task_ids:
            command += ["--task-id", task_id]
        if spec is not None:
            spec_path = directory / ("request.json" if mode == "pd_response" else "spec.json")
            spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            command += ["--request" if mode == "pd_response" else "--spec", str(spec_path)]
        if stage_root:
            command += ["--stage-root", str(stage_root)]
        temporary = directory / "tmp"
        temporary.mkdir(exist_ok=True)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(temporary), PYTHONUNBUFFERED="1")
        env.setdefault("MUJOCO_GL", "egl")
        for key, folder in (("WARP_CACHE_PATH", "warp"), ("CUDA_CACHE_PATH", "cuda"), ("MPLCONFIGDIR", "mpl")):
            env[key] = str(directory / "cache" / folder)
        self.log(job, f"开始 {mode}: {', '.join(task_ids)}")
        with (directory / "output.log").open("a", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=self.repo_root, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            with self._lock:
                job["process"] = process
                if job["cancelled"].is_set():
                    self._terminate(process)
            deadline = time.monotonic() + (1800 if mode == "simulation" else 300)
            while process.poll() is None:
                if time.monotonic() > deadline:
                    self._terminate(process)
                    raise RuntimeError(f"{mode} 验证超时，子进程已停止")
                job["cancelled"].wait(0.1)
                if job["cancelled"].is_set():
                    self._terminate(process)
            job["process"] = None
        if job["cancelled"].is_set():
            raise RuntimeError("任务已停止")
        if not result_path.is_file():
            raise RuntimeError(f"验证进程退出 {process.returncode}，详情见日志")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if process.returncode or result.get("status") == "failed":
            raise RuntimeError(result.get("error", f"{mode} 检查失败"))
        return result

    @staticmethod
    def _terminate(process):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        except ProcessLookupError:
            pass

    def stop(self, job_id):
        with self._lock:
            job = self._jobs[job_id]
            if job["status"] not in ("queued", "running"):
                return self.get(job_id)
            job["cancelled"].set()
            if job["status"] == "queued":
                job["status"] = "stopped"
            if job["process"]:
                self._terminate(job["process"])
        return self.get(job_id)

    def get(self, job_id):
        with self._lock:
            job = self._jobs[job_id]
            result = {key: value for key, value in job.items() if key not in ("directory", "process", "cancelled", "finished")}
            if result.get("result") is not None:
                result["result"] = deepcopy(result["result"])
                for video in result["result"].get("videos", []):
                    filename = video.get("filename") or Path(video.get("path", "")).name
                    if filename and Path(filename).name == filename:
                        video["url"] = f"/api/jobs/{job_id}/videos/{filename}"
            log_path = job["directory"] / "output.log"
            result["log"] = log_path.read_text(encoding="utf-8", errors="replace")[-200_000:] if log_path.exists() else ""
            result["artifact_dir"] = str(job["directory"].relative_to(self.repo_root))
            return result

    def check_spec(self, spec):
        submitted = self.submit("precheck", [], spec=spec)
        job = self._jobs[submitted["id"]]
        if not job["finished"].wait(310):
            self.stop(job["id"])
            raise ValueError("配置预检查超时，请检查任务日志")
        if job["status"] != "passed":
            raise ValueError(job["error"] or "配置预检查失败")
        return job["result"]

    def close(self):
        with self._lock:
            ids = list(self._jobs)
        for job_id in ids:
            self.stop(job_id)
        self._pool.shutdown(wait=True, cancel_futures=True)


def output_roots(files: dict) -> list[str]:
    roots = set()
    for name in files:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or len(path.parts) < 4:
            raise ValueError(f"生成器返回了不允许的路径: {name}")
        if path.parts[:3] in (("src", "assets", "robots"), ("src", "tasks", "locomotion")):
            roots.add(str(Path(*path.parts[:4])))
        elif path.parts[:2] == ("src", "config"):
            roots.add(str(Path(*path.parts[:3])))
        else:
            raise ValueError(f"生成器只能写入机器人、配置和任务目录: {name}")
    return sorted(roots)


def check_conflicts(repo_root: Path, files: dict) -> list[str]:
    roots = output_roots(files)
    for name in roots:
        target = repo_root / name
        for parent in (target, *target.parents):
            if parent == repo_root:
                break
            if parent.is_symlink():
                raise ValueError(f"输出路径包含符号链接: {parent}")
        if target.exists():
            raise ValueError(f"目标已存在，拒绝覆盖: {name}")
    return roots


def _rename_new(source: Path, destination: Path):
    """Atomically refuse an existing destination, including empty directories."""
    import ctypes
    import errno

    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise RuntimeError("当前系统缺少 renameat2，无法保证生成时拒绝覆盖已有目录")
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        if code in (errno.EEXIST, errno.ENOTEMPTY):
            raise ValueError(f"写入时检测到名称冲突: {destination}")
        raise OSError(code, os.strerror(code), str(destination))


def generate_transaction(manager: JobManager, job: dict, spec: dict, rendered: dict):
    import shutil

    root = manager.repo_root
    files = rendered["files"]
    roots = check_conflicts(root, files)
    stage = manager.workspace / "staging" / job["id"]
    stage.mkdir(parents=True, exist_ok=False)
    written = []
    try:
        for name, content in files.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
        manager.log(job, "暂存完成，开始独立进程 CPU 检查")
        preflight = manager.run_worker(job, "cpu", rendered["task_ids"], spec=spec, stage_root=stage)
        check_conflicts(root, files)
        if job["cancelled"].is_set():
            raise RuntimeError("任务已停止")
        for name in roots:
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                raise ValueError(f"写入时检测到名称冲突: {name}")
            _rename_new(stage / name, destination)
            written.append(destination)
        manager.log(job, "已写入新目录，复查任务注册与最终配置")
        verification = manager.run_worker(job, "cpu", rendered["task_ids"], spec=spec)
        if job["cancelled"].is_set():
            raise RuntimeError("任务已停止")
        manifest = dict(rendered.get("manifest", {}), task_ids=rendered["task_ids"],
                        output_roots=roots, written_files=sorted(files), validation=verification)
        from .lifecycle import cleanup_generation, cleanup_temporary

        cleanup = cleanup_temporary(job["directory"], remove_payloads=True)
        if stage.exists():
            cleanup["removed_files"] += sum(path.is_file() for path in stage.rglob("*"))
            cleanup["removed_bytes"] += sum(path.stat().st_size for path in stage.rglob("*") if path.is_file())
            shutil.rmtree(stage)
            cleanup["removed_dirs"].append(str(stage.relative_to(manager.workspace)))
        with manager._lock:
            active = [item["id"] for item in manager._jobs.values()
                      if item["id"] != job["id"] and item["status"] in ("queued", "running")]
        related = cleanup_generation(manager.workspace, spec, active_job_ids=active)
        cleanup["removed_files"] += related["removed_files"]
        cleanup["removed_bytes"] += related["removed_bytes"]
        cleanup["removed_dirs"].extend(related["removed_dirs"])
        cleanup["related_job_ids"] = related["job_ids"]
        with manager._lock:
            for job_id in related["job_ids"]:
                previous = manager._jobs.get(job_id)
                if previous and isinstance(previous.get("result"), dict):
                    previous["result"]["videos"] = []
                    previous["result"]["videos_removed_after_generation"] = True
                    manager._write_state(previous)
        manifest["cleanup"] = cleanup
        manifest_path = job["directory"] / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        manager.log(job, "生成完成，CPU 检查通过")
        return {"status": "passed", "task_ids": rendered["task_ids"], "output_roots": roots, "cleanup": cleanup,
                "manifest": manifest, "preflight": preflight,
                "commands": {task: {"train": f"uv run --no-sync python -m scripts.train {task}",
                                    "play": f"uv run --no-sync python -m scripts.play {task} --agent zero --num-envs 1 --viewer viser"}
                             for task in rendered["task_ids"]}}
    except BaseException:
        for destination in reversed(written):
            shutil.rmtree(destination)
        manager.log(job, "本次新建目录已回滚；已有目录保持原样")
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
