"""
AgentBench dataset loader — natural-language task description extractor.

Per the project scope decision: we extract ONLY the natural-language task
descriptions from AgentBench and treat them as decomposition prompts.
We do NOT stand up AgentBench's live execution environments.

AgentBench has 8 environments; we extract from 6 (skipping Digital Card
Game and Lateral Thinking Puzzles per user decision):
  - OS: shell commands / system tasks
  - DB: natural-language-to-SQL database queries
  - KG: knowledge graph reasoning questions
  - HH (ALFWorld): household task goals
  - WS (WebShop): shopping instructions
  - WB (Mind2Web): web browsing task descriptions
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.utils.io import load_json
from src.datasets.taskbench_loader import RawSample

logger = logging.getLogger(__name__)


_ENVIRONMENT_CONFIG: dict[str, dict[str, str]] = {
    "os": {
        "category": "coding",
        "description": "Operating system shell tasks",
        "instruction_keys": ["instruction", "query", "task", "description"],
    },
    "db": {
        "category": "coding",
        "description": "Database / SQL tasks",
        "instruction_keys": ["instruction", "question", "query", "description"],
    },
    "kg": {
        "category": "logical_reasoning",
        "description": "Knowledge graph reasoning",
        "instruction_keys": ["question", "instruction", "query"],
    },
    "alfworld": {
        "category": "planning",
        "description": "Household tasks (ALFWorld)",
        "instruction_keys": ["goal", "task", "instruction"],
    },
    "webshop": {
        "category": "planning",
        "description": "Web shopping tasks",
        "instruction_keys": ["instruction", "goal", "task", "query"],
    },
    "mind2web": {
        "category": "planning",
        "description": "Web browsing tasks (Mind2Web)",
        "instruction_keys": ["task", "instruction", "confirmed_task", "annotation_id"],
    },
}

_DIR_ALIASES: dict[str, str] = {
    "os": "os",
    "operating_system": "os",
    "db": "db",
    "database": "db",
    "kg": "kg",
    "knowledge_graph": "kg",
    "alfworld": "alfworld",
    "hh": "alfworld",
    "house": "alfworld",
    "householding": "alfworld",
    "webshop": "webshop",
    "ws": "webshop",
    "web_shopping": "webshop",
    "mind2web": "mind2web",
    "wb": "mind2web",
    "web_browsing": "mind2web",
}


class AgentBenchLoader:
    """Loader for AgentBench task descriptions.

    Searches the AgentBench data directory for environment-specific
    data files and extracts natural-language task descriptions.

    Args:
        data_dir: Path to the AgentBench data directory.
    """

    DATASET_LICENSE = "Apache-2.0"
    DATASET_CITATION = (
        "Liu et al., 'AgentBench: Evaluating LLMs as Agents', "
        "ICLR 2024"
    )

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    def load(self) -> list[RawSample]:
        """Load task descriptions from all supported AgentBench environments.

        Returns:
            List of normalized RawSample objects.
        """
        if not self.data_dir.exists():
            logger.warning(
                f"AgentBench data directory not found: {self.data_dir}. "
                f"Run 'python scripts/download_datasets.py' first."
            )
            return []

        samples: list[RawSample] = []

        for subdir in sorted(self.data_dir.iterdir()):
            if not subdir.is_dir():
                continue

            env_key = _DIR_ALIASES.get(subdir.name.lower())
            if env_key is None:
                logger.debug(f"Skipping unknown AgentBench directory: {subdir.name}")
                continue

            env_config = _ENVIRONMENT_CONFIG[env_key]
            env_samples = self._load_environment(subdir, env_key, env_config)
            samples.extend(env_samples)
            logger.info(
                f"Loaded {len(env_samples)} samples from AgentBench/{subdir.name} "
                f"(→ {env_key})"
            )

        for json_file in sorted(self.data_dir.glob("*.json")):
            try:
                data = load_json(json_file)
                flat_samples = self._extract_from_flat(data, json_file)
                samples.extend(flat_samples)
            except Exception as e:
                logger.debug(f"Skipping {json_file}: {e}")

        logger.info(f"Loaded {len(samples)} AgentBench samples total")
        return samples

    def _load_environment(
        self,
        env_dir: Path,
        env_key: str,
        env_config: dict[str, str],
    ) -> list[RawSample]:
        """Load samples from a single AgentBench environment directory."""
        samples: list[RawSample] = []

        json_files = sorted(env_dir.rglob("*.json"))
        jsonl_files = sorted(env_dir.rglob("*.jsonl"))

        for json_file in json_files:
            try:
                data = load_json(json_file)
                file_samples = self._extract_samples(
                    data, json_file, env_key, env_config
                )
                samples.extend(file_samples)
            except Exception as e:
                logger.debug(f"Failed to parse {json_file}: {e}")

        for jsonl_file in jsonl_files:
            try:
                file_samples = self._load_jsonl(jsonl_file, env_key, env_config)
                samples.extend(file_samples)
            except Exception as e:
                logger.debug(f"Failed to parse {jsonl_file}: {e}")

        return samples

    def _load_jsonl(
        self,
        jsonl_file: Path,
        env_key: str,
        env_config: dict[str, str],
    ) -> list[RawSample]:
        """Load samples from a JSONL file (one JSON object per line)."""
        import json

        samples: list[RawSample] = []
        rel_path = str(jsonl_file.relative_to(self.data_dir))

        with open(jsonl_file, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    instruction = self._extract_instruction(
                        item, env_config["instruction_keys"]
                    )
                    if instruction:
                        sample = self._make_sample(
                            instruction, item, env_key, env_config,
                            rel_path, idx,
                        )
                        samples.append(sample)
                except json.JSONDecodeError:
                    continue

        return samples

    def _extract_samples(
        self,
        data: Any,
        source_file: Path,
        env_key: str,
        env_config: dict[str, str],
    ) -> list[RawSample]:
        """Extract samples from a loaded JSON structure."""
        if isinstance(data, dict):
            items = self._unwrap_container(data)
        elif isinstance(data, list):
            items = data
        else:
            return []

        samples: list[RawSample] = []
        rel_path = str(source_file.relative_to(self.data_dir))

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            instruction = self._extract_instruction(
                item, env_config["instruction_keys"]
            )
            if not instruction:
                continue

            sample = self._make_sample(
                instruction, item, env_key, env_config, rel_path, idx
            )
            samples.append(sample)

        return samples

    def _extract_from_flat(
        self, data: Any, source_file: Path
    ) -> list[RawSample]:
        """Try to extract samples from flat JSON files in the root."""
        if isinstance(data, dict):
            items = self._unwrap_container(data)
        elif isinstance(data, list):
            items = data
        else:
            return []

        samples: list[RawSample] = []
        rel_path = str(source_file.relative_to(self.data_dir))

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            all_keys = []
            for cfg in _ENVIRONMENT_CONFIG.values():
                all_keys.extend(cfg["instruction_keys"])
            all_keys = list(dict.fromkeys(all_keys))

            instruction = self._extract_instruction(item, all_keys)
            if not instruction:
                continue

            sample = RawSample(
                source="agentbench",
                original_id=f"agentbench_{rel_path}_{idx}",
                instruction=instruction,
                category="general_reasoning",
                difficulty="medium",
                metadata={
                    "source_file": rel_path,
                    "environment": "unknown",
                    "dataset_license": self.DATASET_LICENSE,
                    "dataset_citation": self.DATASET_CITATION,
                },
            )
            samples.append(sample)

        return samples

    @staticmethod
    def _extract_instruction(item: dict, keys: list[str] | str) -> str:
        """Try multiple keys to find the instruction text."""
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _unwrap_container(data: dict) -> list:
        """Unwrap common container patterns ({"data": [...]} etc.)."""
        for key in ("data", "examples", "tasks", "samples", "items", "instances"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]

    def _make_sample(
        self,
        instruction: str,
        item: dict,
        env_key: str,
        env_config: dict[str, str],
        rel_path: str,
        idx: int,
    ) -> RawSample:
        """Construct a RawSample from extracted data."""
        length = len(instruction.split())
        if length < 20:
            difficulty = "low"
        elif length < 50:
            difficulty = "medium"
        else:
            difficulty = "high"

        return RawSample(
            source="agentbench",
            original_id=f"agentbench_{env_key}_{rel_path}_{idx}",
            instruction=instruction,
            reference={},
            category=env_config["category"],
            difficulty=difficulty,
            metadata={
                "source_file": rel_path,
                "environment": env_key,
                "environment_description": env_config["description"],
                "dataset_license": self.DATASET_LICENSE,
                "dataset_citation": self.DATASET_CITATION,
                "original_index": idx,
            },
        )
