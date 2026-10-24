"""Start the local robot migration workbench."""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="本地四足机器人迁移工作台")
    parser.add_argument("--port", type=int, default=5003, help="本机 HTTP 端口 (默认 5003)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("端口必须为 1–65535")
    root = Path(__file__).resolve().parents[1]
    temporary = root / ".artifacts" / "robot-migration" / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temporary)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    try:
        from web_ui.app import create_app
        app = create_app(root)
    except ModuleNotFoundError as exc:
        if exc.name == "flask":
            parser.error("缺少 Web 可选依赖，请执行 bash setup.sh --extra web")
        raise
    print(f"本地迁移工作台: http://127.0.0.1:{args.port}", flush=True)

    def stop_server(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_server)
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False, threaded=True)
    finally:
        app.extensions["migration"]["close"]()


if __name__ == "__main__":
    main()
