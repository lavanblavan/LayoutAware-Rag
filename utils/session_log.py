"""
Session-scoped logging — one log file per process start.

Log file: logs/session_YYYYMMDD_HHMMSS.log
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOG_DIR = _PROJECT_ROOT / "logs"
_SESSION_ID: str | None = None
_LOG_FILE: Path | None = None
_CONFIGURED = False

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def session_id() -> str:
    global _SESSION_ID
    if _SESSION_ID is None:
        _SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _SESSION_ID


def log_file_path() -> Path:
    global _LOG_FILE
    if _LOG_FILE is None:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        _LOG_FILE = _LOG_DIR / f"session_{session_id()}.log"
    return _LOG_FILE


def configure_logging(level: int = logging.INFO, console: bool = True) -> Path:
    """Configure root logger once per process."""
    global _CONFIGURED
    path = log_file_path()
    if _CONFIGURED:
        return path

    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        root.addHandler(stream_handler)

    root.info("Session log started -> %s", path)
    _CONFIGURED = True
    return path


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
