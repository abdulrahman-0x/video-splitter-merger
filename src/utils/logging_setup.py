"""Logging configuration with rotating file handler for application diagnostics."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "app.log"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_BACKUP_COUNT = 3


def setup_logging(
    log_dir: Path | None = None,
    log_level: int = logging.INFO,
    console_output: bool = True,
) -> logging.Logger:
    """Initialize application root logger with a rotating file handler and optional console handler."""
    target_dir = log_dir or _LOG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "app.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if root_logger.handlers:
        return logging.getLogger("video_splitter_merger")

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler with UTF-8 encoding
    file_handler = RotatingFileHandler(
        filename=target_file,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    logger = logging.getLogger("video_splitter_merger")
    logger.info("تم تهيئة نظام السجلات بنجاح. ملف السجل: %s", target_file.resolve())
    return logger
