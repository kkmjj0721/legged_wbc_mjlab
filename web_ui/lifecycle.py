"""Clean owned temporary work while retaining models, drafts and evidence."""

from pathlib import Path
import json
import os
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]


def _owned_directory(directory):
    directory = Path(directory).absolute()
    if (not directory.is_relative_to(ROOT / ".artifacts") or directory.is_symlink()
            or not directory.resolve().is_relative_to(ROOT / ".artifacts")):
        raise ValueError("仅能清理本项目 .artifacts 中的任务临时目录")
    for parent in directory.parents:
        if parent == ROOT:
            break
        if parent.is_symlink():
            raise ValueError("临时目录包含符号链接，未执行清理")
    return directory


def cleanup_temporary(directory, *, remove_payloads=False, remove_videos=False):
    directory = _owned_directory(directory)
    removed_files, removed_bytes, removed_dirs = 0, 0, []
    for name in ("tmp", "cache", *(("videos",) if remove_videos else ())):
        target = directory / name
        if not target.is_dir() or target.is_symlink():
            continue
        for current, children, files in os.walk(target, followlinks=False):
            children[:] = [child for child in children if not (Path(current) / child).is_symlink()]
            for filename in files:
                path = Path(current) / filename
                if not path.is_symlink():
                    removed_files += 1
                    removed_bytes += path.stat().st_size
        shutil.rmtree(target)
        removed_dirs.append(name)
    if remove_payloads:
        for path in [directory / "spec.json", directory / "request.json", *directory.glob("cpu-*.json"),
                     *directory.glob("precheck-*.json"), *directory.glob("pd_response-*.json")]:
            if path.is_file() and not path.is_symlink():
                removed_files += 1
                removed_bytes += path.stat().st_size
                path.unlink()
    return {"removed_files": removed_files, "removed_bytes": removed_bytes,
            "removed_dirs": removed_dirs,
            "preserved": ["生成的机器人、配置和任务", "原始模型与当前模型工作副本", "迁移草稿",
                          "电机曲线图片及确认数据", "测试报告、日志和生成清单"],
            "note": "生成成功后删除关联模型的测试视频、缓存和重复检查输入；保留源模型、草稿和报告。"}


def cleanup_generation(workspace, spec, *, active_job_ids=()):
    """Remove only finished job intermediates belonging to this imported model."""
    workspace = _owned_directory(workspace)
    jobs_directory = _owned_directory(workspace / "jobs")
    model = spec.get("model", {})
    identity = (model.get("id"), model.get("root"))
    if not all(identity):
        raise ValueError("生成清理缺少导入模型身份")
    referenced = set()
    for scope in spec.get("pd_analysis", {}).get("scopes", {}).values():
        for video in scope.get("response_report", {}).get("videos", []):
            path = Path(video.get("path", "")).absolute()
            if path.parent.name == "videos" and path.parent.parent.parent == workspace / "jobs":
                referenced.add(path.parent.parent.name)
    result = {"removed_files": 0, "removed_bytes": 0, "removed_dirs": [], "job_ids": []}
    for directory in sorted(jobs_directory.iterdir()):
        if (directory.is_symlink() or not directory.is_dir() or not re.fullmatch(r"[a-f0-9]{32}", directory.name)
                or directory.name in active_job_ids):
            continue
        matches = directory.name in referenced
        for filename in ("request.json", "spec.json"):
            path = directory / filename
            if path.is_file() and not path.is_symlink():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    source = payload.get("spec", payload).get("model", {})
                    matches |= (source.get("id"), source.get("root")) == identity
                except (ValueError, AttributeError):
                    continue
        if not matches:
            continue
        removed = cleanup_temporary(directory, remove_payloads=True, remove_videos=True)
        result["removed_files"] += removed["removed_files"]
        result["removed_bytes"] += removed["removed_bytes"]
        result["removed_dirs"].extend(f"jobs/{directory.name}/{name}" for name in removed["removed_dirs"])
        result["job_ids"].append(directory.name)
        state = directory / "job.json"
        if state.is_file() and not state.is_symlink():
            try:
                saved = json.loads(state.read_text(encoding="utf-8"))
                if isinstance(saved.get("result"), dict):
                    saved["result"]["videos"] = []
                    saved["result"]["videos_removed_after_generation"] = True
                saved["generation_cleanup"] = removed
                state.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
            except (ValueError, AttributeError):
                pass
    return result
