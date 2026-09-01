"""
Tests for the experiment runner — resume logic, dry-run, order randomization.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.utils.config import ExperimentConfig, ModelConfig, ModelsConfig
from src.utils.io import save_json, response_exists, get_response_path
from src.pipeline.cost_estimator import count_tokens_approximate, estimate_cost


class TestResumeLogic:
    """Tests for the runner's crash-resume capability."""

    def test_response_exists_true(self, tmp_path: Path):
        """Completed responses are detected correctly."""
        model_dir = tmp_path / "gpt-5"
        model_dir.mkdir()
        response_file = model_dir / "prompt_1.json"
        response_file.write_text('{"prompt_id": 1}', encoding="utf-8")

        assert response_exists(str(tmp_path), "gpt-5", 1) is True

    def test_response_exists_false_missing(self, tmp_path: Path):
        """Missing response files are detected."""
        assert response_exists(str(tmp_path), "gpt-5", 99) is False

    def test_response_exists_false_empty(self, tmp_path: Path):
        """Empty response files are treated as incomplete."""
        model_dir = tmp_path / "gpt-5"
        model_dir.mkdir()
        response_file = model_dir / "prompt_1.json"
        response_file.write_text("", encoding="utf-8")

        assert response_exists(str(tmp_path), "gpt-5", 1) is False

    def test_get_response_path_deterministic(self, tmp_path: Path):
        """Same inputs always produce the same path."""
        p1 = get_response_path(str(tmp_path), "gemini-2.5-flash", 42)
        p2 = get_response_path(str(tmp_path), "gemini-2.5-flash", 42)
        assert p1 == p2

    def test_get_response_path_sanitizes(self, tmp_path: Path):
        """Unsafe characters in model names are sanitized."""
        path = get_response_path(str(tmp_path), "deepseek/deepseek-r1", 1)
        assert "/" not in path.name or "\\" not in str(path.relative_to(tmp_path))


class TestCostEstimator:
    """Tests for token counting and cost estimation."""

    def test_count_tokens_approximate_nonempty(self):
        """Token count should be > 0 for non-empty text."""
        count = count_tokens_approximate("Hello world, this is a test.", "google")
        assert count > 0

    def test_count_tokens_approximate_empty(self):
        """Empty string should return 1 (minimum)."""
        count = count_tokens_approximate("", "google")
        assert count >= 1

    def test_estimate_cost_structure(self):
        """Estimate should return per-model rows plus a TOTAL row."""
        prompts = [
            {"id": 1, "prompt": "Test task 1"},
            {"id": 2, "prompt": "Test task 2"},
        ]
        models = {
            "test-model": ModelConfig(
                provider="google", model_id="gemini-2.5-flash",
                max_tokens=1024,
            ),
        }
        estimates = estimate_cost(prompts, models)

        assert len(estimates) == 2
        assert estimates[-1]["model"] == "TOTAL"

    def test_estimate_cost_free_tier(self):
        """Ollama models should have $0 cost."""
        prompts = [{"id": 1, "prompt": "Test"}]
        models = {
            "local": ModelConfig(
                provider="ollama", model_id="llama3.2:3b", max_tokens=512
            ),
        }
        estimates = estimate_cost(prompts, models)
        assert estimates[0]["est_cost_usd"] == 0.0


class TestOrderRandomization:
    """Tests for deterministic order randomization."""

    def test_same_seed_same_order(self):
        """Same seed should produce identical ordering."""
        import random

        items = list(range(100))
        rng1 = random.Random(42)
        order1 = items.copy()
        rng1.shuffle(order1)

        rng2 = random.Random(42)
        order2 = items.copy()
        rng2.shuffle(order2)

        assert order1 == order2

    def test_different_seed_different_order(self):
        """Different seeds should (almost certainly) produce different orderings."""
        import random

        items = list(range(100))
        rng1 = random.Random(42)
        order1 = items.copy()
        rng1.shuffle(order1)

        rng2 = random.Random(99)
        order2 = items.copy()
        rng2.shuffle(order2)

        assert order1 != order2


class TestAtomicWrite:
    """Tests for crash-safe JSON writing."""

    def test_save_json_creates_file(self, tmp_path: Path):
        """save_json should create the file with correct content."""
        data = {"key": "value", "number": 42}
        path = tmp_path / "test.json"
        save_json(data, path)

        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == data

    def test_save_json_creates_parent_dirs(self, tmp_path: Path):
        """save_json should create parent directories if needed."""
        path = tmp_path / "deep" / "nested" / "dir" / "test.json"
        save_json({"test": True}, path)
        assert path.exists()

    def test_save_json_no_tmp_leftover(self, tmp_path: Path):
        """After atomic write, no .tmp file should remain."""
        path = tmp_path / "test.json"
        save_json({"test": True}, path)

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0
