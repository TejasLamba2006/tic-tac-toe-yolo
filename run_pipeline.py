#!/usr/bin/env python3
"""Run the automated ML pipeline.

Usage:
    python run_pipeline.py                          # Full pipeline
    python run_pipeline.py --stage training         # Single stage
    python run_pipeline.py --from-stage export      # Resume from stage
    python run_pipeline.py --dry-run                # Validate only
    python run_pipeline.py --list-stages            # Show available stages
    python run_pipeline.py --config my_config.yaml  # Custom config
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so all imports resolve.
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pipeline.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
