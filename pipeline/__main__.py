"""Enable ``python -m pipeline`` execution."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``src.*`` imports work.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pipeline.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
