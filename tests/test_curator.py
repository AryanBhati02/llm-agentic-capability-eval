"""
Tests for the dataset curation engine (deduplication, sampling, formatting).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.datasets.curator import Curator
from src.datasets.taskbench_loader import RawSample


class TestDeduplication:
    """Tests for embedding-based deduplication."""

    def test_deduplicate_removes_near_identical(self):
        curator = Curator(similarity_threshold=0.90)

        samples = [
            RawSample(source="tb", original_id="1", instruction="Do task A"),
            RawSample(source="tb", original_id="2", instruction="Do task A please"),
            RawSample(source="tb", original_id="3", instruction="Completely different task Z"),
        ]

        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        deduped, _ = curator._deduplicate(samples, embeddings)

        assert len(deduped) == 2
        assert deduped[0].original_id == "1"
        assert deduped[1].original_id == "3"

    def test_deduplicate_keeps_distinct(self):
        curator = Curator(similarity_threshold=0.90)

        samples = [
            RawSample(source="tb", original_id="1", instruction="Task 1"),
            RawSample(source="tb", original_id="2", instruction="Task 2"),
        ]

        embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        deduped, _ = curator._deduplicate(samples, embeddings)
        assert len(deduped) == 2

    def test_deduplicate_empty_and_single(self):
        curator = Curator()
        assert curator._deduplicate([], np.empty((0, 384)))[0] == []

        single = [RawSample(source="tb", original_id="1", instruction="Solo")]
        single_emb = np.array([[1.0, 0.0]])
        assert len(curator._deduplicate(single, single_emb)[0]) == 1


class TestStratifiedSampling:
    """Tests for balanced category and difficulty sampling."""

    def test_stratified_sample_balance(self):
        curator = Curator(n_per_category=2, random_seed=42)

        samples = [
            RawSample(source="tb", original_id=f"c1_{i}", instruction=f"Code {i}",
                      category="coding", difficulty=d)
            for i, d in enumerate(["low", "low", "medium", "high", "high"])
        ] + [
            RawSample(source="ab", original_id=f"p1_{i}", instruction=f"Plan {i}",
                      category="planning", difficulty=d)
            for i, d in enumerate(["low", "medium", "high"])
        ]

        embeddings = np.random.randn(len(samples), 10)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        selected = curator._stratified_sample(samples, embeddings)

        assert len(selected) == 4

        coding_selected = [s for s, _ in selected if s.category == "coding"]
        planning_selected = [s for s, _ in selected if s.category == "planning"]
        assert len(coding_selected) == 2
        assert len(planning_selected) == 2


class TestOutputFormatting:
    """Tests for the curated JSON output schema."""

    def test_format_output_schema(self, sample_raw_samples: list[RawSample]):
        curator = Curator()
        embeddings = np.random.randn(len(sample_raw_samples), 10)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        pairs = list(zip(sample_raw_samples, embeddings))
        output = curator._format_output(pairs)

        assert len(output) == len(sample_raw_samples)

        for item in output:
            assert "id" in item
            assert "dataset" in item
            assert item["dataset"] in ("TaskBench", "AgentBench")
            assert "category" in item
            assert "difficulty" in item
            assert "prompt" in item
            assert "reference" in item
            assert "metadata" in item
            assert "original_id" in item["metadata"]
            assert "embedding_similarity_to_nearest" in item["metadata"]
