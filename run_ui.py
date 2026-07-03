#!/usr/bin/env python3
"""Launch the pipeline dashboard.

Usage:
    python run_ui.py                  # http://127.0.0.1:8000
    python run_ui.py --port 8080
    python run_ui.py --host 0.0.0.0   # expose beyond localhost (not recommended:
                                       # the dashboard has no auth and can execute
                                       # the pipeline subprocess)
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def main() -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "Missing UI dependencies. Install them with:\n"
            "    pip install -e \".[ui]\"\n",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(description="Run the pipeline dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open a browser tab")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print(f"Starting pipeline dashboard at {url}")
    if not args.no_browser and args.host in ("127.0.0.1", "localhost"):
        try:
            webbrowser.open(url)
        except Exception:
            pass

    import os
    # Suppress TensorFlow oneDNN verbose noise that leaks into the UI logs.
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    # Exclude generated/artifact directories so that files written during a
    # pipeline run (e.g. the onnx2tf wrapper script, TFLite checkpoints, ...) 
    # do NOT trigger a server reload mid-run.
    reload_excludes = [
        "artifacts_*",
        "dataset_*",
        "generated_configs",
        "runs",
        "build",
        "__pycache__",
        "*.tmp",
    ]

    uvicorn.run(
        "webui.app:app",
        host=args.host,
        port=args.port,
        reload=True,
        reload_excludes=reload_excludes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
