"""
TaskBench dataset loader.

Reads TaskBench data from the microsoft/JARVIS repository and normalizes
each sample into the framework's internal ``RawSample`` schema.

TaskBench structure (per sample):
  - user_request: natural-language instruction
  - task_steps: list of decomposition step strings (our reference)
  - task_nodes: tool graph nodes with arguments and dependencies
  - task_links: edges between nodes (resource/temporal dependencies)

The loader infers difficulty from graph complexity (node × edge count)
and maps tool domains to the professor's task categories.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from src.utils.io import load_json

logger = logging.getLogger(__name__)


@dataclass
class RawSample:
    """Normalized sample from any benchmark dataset.

    This is the common schema that both TaskBench and AgentBench loaders
    produce. The curator operates on lists of these.
    """

    source: str
    original_id: str
    instruction: str
    reference: dict = field(default_factory=dict)
    category: str = "general_reasoning"
    difficulty: str = "medium"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict for JSON serialization."""
        return asdict(self)


_TASKBENCH_CATEGORY_MAP: dict[str, str] = {
    "code": "coding",
    "data": "coding",
    "software": "coding",

    "nlp": "general_reasoning",
    "text": "general_reasoning",
    "translation": "general_reasoning",

    "image": "planning",
    "audio": "planning",
    "video": "planning",
    "multimedia": "planning",
    "multimodal": "planning",

    "complex": "planning",
    "chain": "planning",
    "dag": "planning",

    "knowledge": "logical_reasoning",
    "qa": "logical_reasoning",
    "reasoning": "logical_reasoning",
}

_DEFAULT_CATEGORY = "general_reasoning"


def _infer_category(sample_data: dict, source_file: str = "") -> str:
    """Infer a task category from the sample data and source path.

    Uses the source file path and tool names to guess the best category.
    Falls back to general_reasoning.
    """
    source_parts = [p.lower() for p in Path(source_file).parts]
    clean_parts = [p[5:] if p.startswith("data_") else p for p in source_parts]
    source_clean = "/".join(clean_parts)

    for keyword, category in _TASKBENCH_CATEGORY_MAP.items():
        if keyword == "data":
            if any(part == "data" for part in clean_parts):
                return category
            continue
        if keyword in source_clean:
            return category

    nodes = sample_data.get("task_nodes", [])
    tool_names = " ".join(
        str(node.get("task", "")).lower() for node in nodes if isinstance(node, dict)
    )
    for keyword, category in _TASKBENCH_CATEGORY_MAP.items():
        if keyword in tool_names:
            return category

    return _DEFAULT_CATEGORY


def _infer_difficulty(sample_data: dict) -> str:
    """Infer difficulty from task graph complexity.

    Score = node_count + 0.5 * edge_count, bucketed into terciles.
    This is a rough heuristic for stratified sampling, not ground truth.
    """
    nodes = sample_data.get("task_nodes", [])
    links = sample_data.get("task_links", [])
    node_count = len(nodes) if isinstance(nodes, list) else 0
    link_count = len(links) if isinstance(links, list) else 0

    complexity = node_count + 0.5 * link_count

    if complexity <= 2:
        return "low"
    elif complexity <= 5:
        return "medium"
    else:
        return "high"


def _parse_possibly_stringified(value: Any) -> Any:
    """Return ``value`` as a Python object, parsing it if it is a JSON string.

    In TaskBench's ``data.json`` JSONL files the graph fields (``tool_nodes``,
    ``tool_links``, ``tool_steps``) are stored as JSON-encoded *strings* inside
    each line's JSON object rather than as native arrays.  This function
    transparently handles both cases:

    - If ``value`` is already a list/dict, it is returned unchanged.
    - If ``value`` is a non-empty string, ``json.loads`` is attempted.
      On success the parsed object is returned; on failure ``None`` is returned
      and the caller is expected to skip the field.
    - ``None`` and empty strings are returned as ``None`` so callers can treat
      them as "field absent".
    """
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            logger.debug(
                f"_parse_possibly_stringified: could not parse string as JSON: {value[:80]!r}"
            )
            return None
    return None


class TaskBenchLoader:
    """Loader for the TaskBench dataset (microsoft/JARVIS).

    Reads JSON files from the TaskBench data directory and produces
    normalized ``RawSample`` objects.

    Args:
        data_dir: Path to the TaskBench data directory (e.g. datasets/taskbench/).
    """

    DATASET_LICENSE = "Apache-2.0"
    DATASET_CITATION = (
        "Shen et al., 'TaskBench: Benchmarking Large Language Models "
        "for Task Automation', ICLR 2024"
    )

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._tool_pool_cache: dict[str, list[dict]] = {}

    def load(self) -> list[RawSample]:
        """Load all TaskBench samples from the data directory.

        Searches for JSON files recursively and extracts samples that
        contain a ``user_request`` field.

        Returns:
            List of normalized RawSample objects.
        """
        if not self.data_dir.exists():
            logger.warning(
                f"TaskBench data directory not found: {self.data_dir}. "
                f"Run 'python scripts/download_datasets.py' first."
            )
            return []

        samples: list[RawSample] = []
        json_files = sorted(self.data_dir.rglob("*.json"))

        logger.info(f"Found {len(json_files)} JSON files in {self.data_dir}")

        for json_file in json_files:
            try:
                data = load_json(json_file)
                file_samples = self._extract_samples(data, json_file)
                samples.extend(file_samples)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                continue

        logger.info(f"Loaded {len(samples)} TaskBench samples total")
        return samples

    def _extract_samples(
        self, data: Any, source_file: Path
    ) -> list[RawSample]:
        """Extract RawSamples from a loaded JSON structure.

        TaskBench data can be a single dict or a list of dicts.

        The data.json files are JSONL where each line is a JSON object whose
        graph fields (tool_nodes, tool_links, tool_steps) are themselves stored
        as JSON-encoded *strings* rather than native arrays.  This method
        calls ``_parse_possibly_stringified`` on every graph field so the
        stored reference is always a proper Python list, not a raw string.
        """
        if isinstance(data, dict):
            items = [data]
        elif isinstance(data, list):
            items = data
        else:
            return []

        samples: list[RawSample] = []
        rel_path = str(source_file.relative_to(self.data_dir))

        domain_dir = source_file.parent
        tool_pool = self._get_tool_pool(domain_dir)

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            instruction = item.get("user_request", "")
            if not instruction:
                instruction = item.get("instruction", "")
            if not instruction:
                continue

            original_id = f"taskbench_{rel_path}_{idx}"

            reference: dict[str, Any] = {}

            raw_nodes = item.get("tool_nodes", item.get("task_nodes", None))
            raw_links = item.get("tool_links", item.get("task_links", None))
            raw_steps = item.get("tool_steps", item.get("task_steps", None))

            if raw_nodes is not None:
                parsed = _parse_possibly_stringified(raw_nodes)
                if parsed is not None:
                    reference["task_nodes"] = parsed
            if raw_links is not None:
                parsed = _parse_possibly_stringified(raw_links)
                if parsed is not None:
                    reference["task_links"] = parsed
            if raw_steps is not None:
                parsed = _parse_possibly_stringified(raw_steps)
                if parsed is not None:
                    reference["task_steps"] = parsed

            sample = RawSample(
                source="taskbench",
                original_id=original_id,
                instruction=instruction.strip(),
                reference=reference,
                category=_infer_category(item, rel_path),
                difficulty=_infer_difficulty(item),
                metadata={
                    "source_file": rel_path,
                    "dataset_license": self.DATASET_LICENSE,
                    "dataset_citation": self.DATASET_CITATION,
                    "original_index": idx,
                    "tool_pool": tool_pool,
                },
            )
            samples.append(sample)

        return samples

    def _get_tool_pool(self, domain_dir: Path) -> list[dict]:
        """Load and cache the tool pool for a domain directory.

        Reads ``tool_desc.json`` from ``domain_dir`` and returns the ``nodes``
        list.  Returns an empty list if the file does not exist (e.g. for the
        AgentBench loader which shares this pattern but has no tool_desc).

        Results are cached per directory path so the file is read at most once
        per domain per ``TaskBenchLoader`` instance.
        """
        cache_key = str(domain_dir)
        if cache_key in self._tool_pool_cache:
            return self._tool_pool_cache[cache_key]

        tool_desc_path = domain_dir / "tool_desc.json"
        if not tool_desc_path.exists():
            logger.debug(f"No tool_desc.json in {domain_dir} — tool_pool will be empty")
            self._tool_pool_cache[cache_key] = []
            return []

        try:
            with open(tool_desc_path, encoding="utf-8") as f:
                desc = json.load(f)
            pool = desc.get("nodes", [])
            logger.debug(
                f"Loaded {len(pool)} tools from {tool_desc_path}"
            )
        except Exception as exc:
            logger.warning(f"Failed to load {tool_desc_path}: {exc}")
            pool = []

        self._tool_pool_cache[cache_key] = pool
        return pool
