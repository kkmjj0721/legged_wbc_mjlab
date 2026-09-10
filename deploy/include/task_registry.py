from __future__ import annotations

from pathlib import Path

from .config import REPO_ROOT


def task_config(task: str) -> Path:
    path = REPO_ROOT / "deploy" / "task" / task / "config" / "sim2sim.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown task '{task}', expected config at {path}")
    return path
