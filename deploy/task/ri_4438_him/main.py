#!/usr/bin/env python3
"""Compatibility launcher; use ``python deploy/main.py --task ri_4438_him``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deploy.include.entrypoint import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["--task", "ri_4438_him", *sys.argv[1:]]))
