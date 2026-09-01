"""
Model anonymization for blind judging.

Generates a randomized mapping from real model names to anonymous labels
(Model_A, Model_B, ...) so the judge never sees brand names. This controls
for self-preference and brand bias in LLM-as-a-Judge scoring.

The mapping is stored separately from judge outputs and only de-anonymized
at the metrics/analysis stage.
"""

from __future__ import annotations

import logging
import random
import string
from pathlib import Path
from typing import Any

from src.utils.io import load_json, save_json

logger = logging.getLogger(__name__)


def generate_anonymization_map(
    model_names: list[str],
    seed: int = 7,
) -> dict[str, str]:
    """Create a randomized anonymous-label → real-name mapping.

    Args:
        model_names: List of real model names to anonymize.
        seed: Random seed for reproducible shuffling.

    Returns:
        Dict mapping anonymous labels to real model names.
        Example: {"Model_A": "gpt-5", "Model_B": "gemini-2.5-pro", ...}
    """
    rng = random.Random(seed)

    shuffled = list(model_names)
    rng.shuffle(shuffled)

    labels = _generate_labels(len(shuffled))

    mapping = {label: real_name for label, real_name in zip(labels, shuffled)}

    logger.info(
        f"Generated anonymization map: {len(mapping)} models, seed={seed}"
    )
    return mapping


def _generate_labels(n: int) -> list[str]:
    """Generate n unique anonymous labels.

    Produces: Model_A, Model_B, ..., Model_Z, Model_AA, Model_AB, ...
    """
    labels = []
    chars = string.ascii_uppercase

    for i in range(n):
        if i < 26:
            labels.append(f"Model_{chars[i]}")
        else:
            first = chars[(i - 26) // 26]
            second = chars[i % 26]
            labels.append(f"Model_{first}{second}")

    return labels


def save_anonymization_map(
    mapping: dict[str, str],
    path: str | Path,
) -> None:
    """Save the anonymization map to a JSON file.

    Args:
        mapping: The anonymous-label → real-name mapping.
        path: Output file path.
    """
    save_json(mapping, path)
    logger.info(f"Anonymization map saved to {path}")


def load_anonymization_map(path: str | Path) -> dict[str, str]:
    """Load a previously saved anonymization map.

    Args:
        path: Path to the anonymization map JSON.

    Returns:
        The anonymous-label → real-name mapping.
    """
    return load_json(path)


def invert_anonymization_map(mapping: dict[str, str]) -> dict[str, str]:
    """Create a real-name → anonymous-label mapping (reverse lookup).

    Args:
        mapping: The anonymous-label → real-name mapping.

    Returns:
        Dict mapping real model names to their anonymous labels.
    """
    return {real: anon for anon, real in mapping.items()}


def anonymize_model_name(
    real_name: str,
    mapping: dict[str, str],
) -> str:
    """Look up the anonymous label for a real model name.

    Args:
        real_name: The real model name.
        mapping: The anonymous-label → real-name mapping.

    Returns:
        The anonymous label, or "Model_Unknown" if not found.
    """
    inverted = invert_anonymization_map(mapping)
    return inverted.get(real_name, "Model_Unknown")
