"""
Structured logging for the ML Playground backend.
Provides JSON-formatted logs with context for debugging and monitoring.
"""
import logging
import json
import sys
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """Format log records as JSON with consistent fields."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        # Add exception info if present
        if record.exc_info and record.exc_info[0]:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_data, default=str)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = True,
) -> None:
    """
    Configure application-wide logging.
    Call this once at application startup.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    if json_format:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


# Context managers for adding extra fields to log records
class LogContext:
    """Thread-local context for adding extra fields to logs."""

    def __init__(self, **kwargs: Any):
        self.old_extra: Dict[str, Any] = {}
        self.new_extra = kwargs

    def __enter__(self) -> "LogContext":
        logger = logging.getLogger()
        self.old_extra = getattr(logger, "extra", {})
        logger.extra = {**self.old_extra, **self.new_extra}
        return self

    def __exit__(self, *args: Any) -> None:
        logger = logging.getLogger()
        logger.extra = self.old_extra


def log_with_context(logger: logging.Logger, level: int, message: str, **kwargs: Any) -> None:
    """Log a message with additional context fields."""
    extra = getattr(logger, "extra", {})
    extra.update(kwargs)
    logger.log(level, message, extra=extra)