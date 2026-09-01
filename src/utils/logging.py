"""
Structured JSON logging with console and file handlers.

Features:
  - Console handler: human-readable, colorized output for interactive use
  - File handler: newline-delimited JSON with full provenance fields:
      timestamp, level, model, provider, latency, tokens, error_type
  - Per-run log files in logs/experiment_{timestamp}.log
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir

_EXTRA_FIELDS = (
    "model",
    "provider",
    "latency_seconds",
    "input_tokens",
    "output_tokens",
    "error_type",
    "prompt_id",
    "retry_attempt",
)


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects.

    Includes standard fields (timestamp, level, logger, message) plus
    any custom fields passed via ``extra={...}``.
    """

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for field_name in _EXTRA_FIELDS:
            value = getattr(record, field_name, None)
            if value is not None:
                data[field_name] = value

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            data["exception"] = record.exc_text

        return json.dumps(data, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Clean, human-readable console formatter."""

    LEVEL_COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelname, "")
        time_str = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        msg = f"{time_str} [{color}{record.levelname:<7}{self.RESET}] {record.getMessage()}"

        model = getattr(record, "model", None)
        latency = getattr(record, "latency_seconds", None)
        tokens = getattr(record, "output_tokens", None)

        extras = []
        if model:
            extras.append(f"model={model}")
        if latency is not None:
            extras.append(f"{latency:.2f}s")
        if tokens is not None:
            extras.append(f"{tokens}tok")

        if extras:
            msg += f" \033[90m({', '.join(extras)})\033[0m"

        return msg


def setup_logging(
    log_dir: str | Path = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_name_prefix: str = "experiment",
) -> Path:
    """Configure structured logging for an experiment run.

    Args:
        log_dir: Directory to store log files.
        console_level: Logging level for stdout.
        file_level: Logging level for the JSON log file.
        log_name_prefix: Prefix for the log file name.

    Returns:
        Path to the created log file.
    """
    log_path = ensure_dir(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{log_name_prefix}_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    root_logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ConsoleFormatter())
    root_logger.addHandler(console_handler)

    for noisy in ("httpcore", "httpx", "urllib3", "openai", "google", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file
