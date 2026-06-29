"""Centralized logging configuration for the pipeline.

Every module obtains its logger via ``logging.getLogger(__name__)``.
This module configures the root ``pipeline`` logger once so that all
sub-loggers inherit the same handlers and format.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


_CONFIGURED = False

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-36s | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    log_to_console: bool = True,
) -> Path | None:
    """Configure the ``pipeline`` root logger.

    Parameters
    ----------
    level:
        Logging level name (DEBUG, INFO, WARNING, ERROR).
    log_dir:
        Directory for timestamped log files.  ``None`` disables file logging.
    log_to_console:
        Whether to attach a stderr stream handler.

    Returns
    -------
    Path to the log file if file logging is enabled, else ``None``.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return None

    root_logger = logging.getLogger("pipeline")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if log_to_console:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(formatter)
        root_logger.addHandler(console)

    log_file_path: Path | None = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_file_path = log_dir / f"pipeline_{timestamp}.log"
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _CONFIGURED = True
    return log_file_path
