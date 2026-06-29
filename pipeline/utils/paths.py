"""Path resolution and validation helpers."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_project_root() -> Path:
    """Return the project root directory.

    Walks upward from this file until it finds a directory containing
    ``data.yaml`` (the Ultralytics dataset descriptor) or falls back to
    the current working directory.
    """
    candidate = Path(__file__).resolve().parent
    for _ in range(10):
        if (candidate / "data.yaml").exists():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent

    fallback = Path.cwd()
    logger.debug("Project root not found via data.yaml; using cwd: %s", fallback)
    return fallback


def ensure_directory(path: Path) -> Path:
    """Create *path* (and parents) if it does not exist, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(raw: str | Path, base: Path | None = None) -> Path:
    """Resolve *raw* relative to *base* (defaults to project root).

    Absolute paths are returned as-is.
    """
    p = Path(raw)
    if p.is_absolute():
        return p
    if base is None:
        base = resolve_project_root()
    return (base / p).resolve()
