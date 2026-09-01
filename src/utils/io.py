"""
I/O utilities with crash-safe atomic writes and JSON helpers.

Provides:
  - ensure_dir: create parent directories if missing
  - load_json / save_json: JSON read/write with atomic file replacement
  - response_exists: check if a model response already exists on disk
  - get_response_path: deterministic path resolution for responses
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists, creating parents if necessary."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path) -> Any:
    """Load JSON from a file path."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(
    data: Any,
    path: str | Path,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """Save data to JSON using atomic write to prevent corruption on crash.

    Writes to a temporary file first, then atomically renames to the target.
    """
    target = Path(path)
    ensure_dir(target.parent)

    tmp_path = target.with_suffix(f".tmp_{os.getpid()}_{target.suffix}")

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            f.flush()
            os.fsync(f.fileno())

        tmp_path.replace(target)

    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _sanitize_filename(name: str) -> str:
    """Sanitize a model name or string for use in a directory/file name."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def get_response_path(
    output_dir: str | Path,
    model_name: str,
    prompt_id: int | str,
) -> Path:
    """Return the canonical file path for a model response."""
    safe_model = _sanitize_filename(model_name)
    return Path(output_dir) / safe_model / f"prompt_{prompt_id}.json"


def response_exists(
    output_dir: str | Path,
    model_name: str,
    prompt_id: int | str,
) -> bool:
    """Check whether a valid response file exists on disk."""
    path = get_response_path(output_dir, model_name, prompt_id)
    return path.exists() and path.stat().st_size > 0


def get_judgment_path(
    judgments_dir: str | Path,
    prompt_id: int | str,
    anonymized_model: str,
) -> Path:
    """Return the canonical file path for a judge evaluation."""
    return Path(judgments_dir) / f"prompt_{prompt_id}_{anonymized_model}.json"


def judgment_exists(
    judgments_dir: str | Path,
    prompt_id: int | str,
    anonymized_model: str,
) -> bool:
    """Check whether a judgment file exists on disk and is non-empty."""
    path = get_judgment_path(judgments_dir, prompt_id, anonymized_model)
    return path.exists() and path.stat().st_size > 0
