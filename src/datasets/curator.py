"""
Dataset curation engine.

Selects representative samples from TaskBench and AgentBench, with:
  - Stratified sampling across categories and difficulty levels
  - Embedding-based deduplication using sentence-transformers
  - Representativeness checking via pairwise distance analysis
  - Full provenance logging

Usage:
    from src.datasets.curator import Curator
    curator = Curator(n_per_category=12)
    curated = curator.curate(raw_samples)
    curator.save(curated, "datasets/curated/prompts.json")
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.datasets.taskbench_loader import RawSample
from src.utils.io import save_json

logger = logging.getLogger(__name__)

_DEDUP_THRESHOLD = 0.92


class Curator:
    """Curates representative samples from raw benchmark data.

    Pipeline:
      1. Pool all RawSamples from both loaders
      2. Compute sentence embeddings (lazy-loaded, CPU-friendly)
      3. Deduplicate by cosine similarity
      4. Stratified sampling: n_per_category, balanced across difficulties
      5. Representativeness check: flag low-diversity categories
      6. Export to JSON with full provenance

    Args:
        n_per_category: Target number of samples per category.
        similarity_threshold: Cosine similarity above which two samples
            are considered duplicates.
        embedding_model: Sentence-transformers model name. Default is
            all-MiniLM-L6-v2 (22M params, runs on CPU in <1s/sample).
        random_seed: Seed for reproducible sampling.
    """

    def __init__(
        self,
        n_per_category: int = 12,
        similarity_threshold: float = _DEDUP_THRESHOLD,
        embedding_model: str = "all-MiniLM-L6-v2",
        random_seed: int = 42,
    ) -> None:
        self.n_per_category = n_per_category
        self.similarity_threshold = similarity_threshold
        self.embedding_model_name = embedding_model
        self.random_seed = random_seed
        self._embedder = None

    def _get_embedder(self):
        """Lazy-load the sentence-transformers model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def curate(self, samples: list[RawSample]) -> list[dict[str, Any]]:
        """Run the full curation pipeline.

        Args:
            samples: Raw samples from all loaders.

        Returns:
            List of curated prompt dicts in the output schema.
        """
        if not samples:
            logger.warning("No samples to curate.")
            return []

        logger.info(f"Starting curation of {len(samples)} raw samples")

        instructions = [s.instruction for s in samples]
        embeddings = self._compute_embeddings(instructions)

        samples, embeddings = self._deduplicate(samples, embeddings)
        logger.info(f"After deduplication: {len(samples)} samples remain")

        selected = self._stratified_sample(samples, embeddings)
        logger.info(f"After stratified sampling: {len(selected)} samples selected")

        self._check_representativeness(selected)

        curated = self._format_output(selected)

        return curated

    def _compute_embeddings(self, texts: list[str]) -> np.ndarray:
        """Compute sentence embeddings for all texts."""
        embedder = self._get_embedder()
        logger.info(f"Computing embeddings for {len(texts)} texts...")
        embeddings = embedder.encode(
            texts,
            show_progress_bar=True,
            batch_size=64,
            normalize_embeddings=True,
        )
        return np.array(embeddings)

    def _deduplicate(
        self,
        samples: list[RawSample],
        embeddings: np.ndarray,
    ) -> tuple[list[RawSample], np.ndarray]:
        """Remove near-duplicate samples based on embedding similarity.

        Uses an optimized greedy approach: iterate through samples, and
        for each accepted sample, discard all subsequent samples that are too
        similar to it.
        """
        if len(samples) <= 1:
            return samples, embeddings

        n = len(samples)
        discard = np.zeros(n, dtype=bool)

        for i in range(n):
            if discard[i]:
                continue
            if i < n - 1:
                similarities = embeddings[i+1:] @ embeddings[i]
                duplicates = similarities >= self.similarity_threshold
                discard[i+1:] |= duplicates

        kept_indices = np.where(~discard)[0].tolist()

        deduped_samples = [samples[i] for i in kept_indices]
        deduped_embeddings = embeddings[kept_indices]

        removed_count = len(samples) - len(deduped_samples)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} near-duplicates")

        return deduped_samples, deduped_embeddings

    def _stratified_sample(
        self,
        samples: list[RawSample],
        embeddings: np.ndarray,
    ) -> list[tuple[RawSample, np.ndarray]]:
        """Select samples balanced across categories and difficulties.

        For each category, attempts to get equal representation of
        low/medium/high difficulty. If a category has fewer samples
        than n_per_category, takes all available.
        """
        rng = random.Random(self.random_seed)

        by_category: dict[str, list[tuple[int, RawSample]]] = defaultdict(list)
        for idx, sample in enumerate(samples):
            by_category[sample.category].append((idx, sample))

        selected: list[tuple[RawSample, np.ndarray]] = []

        for category, cat_samples in sorted(by_category.items()):
            by_difficulty: dict[str, list[tuple[int, RawSample]]] = defaultdict(list)
            for idx, sample in cat_samples:
                by_difficulty[sample.difficulty].append((idx, sample))

            difficulties = list(by_difficulty.keys())
            n_per_diff = max(1, self.n_per_category // max(len(difficulties), 1))

            cat_selected: list[tuple[int, RawSample]] = []

            for diff in difficulties:
                diff_pool = by_difficulty[diff]
                rng.shuffle(diff_pool)
                cat_selected.extend(diff_pool[:n_per_diff])

            selected_ids = {idx for idx, _ in cat_selected}
            remaining = [(i, s) for i, s in cat_samples if i not in selected_ids]
            rng.shuffle(remaining)

            while len(cat_selected) < self.n_per_category and remaining:
                cat_selected.append(remaining.pop(0))

            for idx, sample in cat_selected:
                selected.append((sample, embeddings[idx]))

            logger.info(
                f"Category '{category}': {len(cat_selected)}/{self.n_per_category} "
                f"samples selected (available: {len(cat_samples)})"
            )

        return selected

    def _check_representativeness(
        self,
        selected: list[tuple[RawSample, np.ndarray]],
    ) -> None:
        """Flag categories with suspiciously low or high internal diversity.

        Computes mean pairwise cosine distance within each category.
        """
        by_category: dict[str, list[np.ndarray]] = defaultdict(list)
        for sample, embedding in selected:
            by_category[sample.category].append(embedding)

        for category, cat_embeddings in sorted(by_category.items()):
            if len(cat_embeddings) < 2:
                logger.warning(
                    f"Category '{category}' has only {len(cat_embeddings)} sample(s) — "
                    f"cannot assess representativeness"
                )
                continue

            embeds = np.stack(cat_embeddings)
            sim_matrix = embeds @ embeds.T
            n = len(embeds)
            upper_mask = np.triu_indices(n, k=1)
            pairwise_sims = sim_matrix[upper_mask]

            mean_sim = float(np.mean(pairwise_sims))
            mean_dist = 1.0 - mean_sim

            if mean_dist < 0.15:
                logger.warning(
                    f"Category '{category}': LOW diversity (mean distance={mean_dist:.3f}). "
                    f"Samples may be too similar."
                )
            elif mean_dist > 0.7:
                logger.warning(
                    f"Category '{category}': HIGH scatter (mean distance={mean_dist:.3f}). "
                    f"Samples may be too heterogeneous for this category."
                )
            else:
                logger.info(
                    f"Category '{category}': diversity OK "
                    f"(mean distance={mean_dist:.3f})"
                )

    def _format_output(
        self,
        selected: list[tuple[RawSample, np.ndarray]],
    ) -> list[dict[str, Any]]:
        """Convert selected samples to the output JSON schema."""
        curated: list[dict[str, Any]] = []

        for idx, (sample, embedding) in enumerate(selected, start=1):
            nn_dist = self._nearest_neighbor_distance(embedding, selected, idx - 1)

            entry = {
                "id": idx,
                "dataset": "TaskBench" if sample.source == "taskbench" else "AgentBench",
                "category": sample.category,
                "difficulty": sample.difficulty,
                "prompt": sample.instruction,
                "reference": sample.reference,
                "metadata": {
                    **sample.metadata,
                    "original_id": sample.original_id,
                    "embedding_similarity_to_nearest": round(1.0 - nn_dist, 3),
                },
            }
            curated.append(entry)

        return curated

    @staticmethod
    def _nearest_neighbor_distance(
        embedding: np.ndarray,
        all_selected: list[tuple[RawSample, np.ndarray]],
        current_idx: int,
    ) -> float:
        """Compute the distance to the nearest other selected sample."""
        min_dist = float("inf")
        for i, (_, other_emb) in enumerate(all_selected):
            if i == current_idx:
                continue
            sim = float(embedding @ other_emb)
            dist = 1.0 - sim
            if dist < min_dist:
                min_dist = dist
        return min_dist if min_dist != float("inf") else 0.0

    def save(
        self,
        curated: list[dict[str, Any]],
        output_path: str | Path,
        log_path: str | Path | None = None,
    ) -> None:
        """Save curated prompts and curation log to JSON files.

        Args:
            curated: The curated prompt list.
            output_path: Path for the main prompts.json file.
            log_path: Path for the curation log. Defaults to
                ``{output_dir}/curation_log.json``.
        """
        output_path = Path(output_path)
        save_json(curated, output_path)
        logger.info(f"Saved {len(curated)} curated prompts to {output_path}")

        if log_path is None:
            log_path = output_path.parent / "curation_log.json"

        log_data = {
            "total_curated": len(curated),
            "n_per_category_target": self.n_per_category,
            "similarity_threshold": self.similarity_threshold,
            "embedding_model": self.embedding_model_name,
            "random_seed": self.random_seed,
            "category_counts": self._count_by_field(curated, "category"),
            "difficulty_counts": self._count_by_field(curated, "difficulty"),
            "dataset_counts": self._count_by_field(curated, "dataset"),
        }
        save_json(log_data, log_path)
        logger.info(f"Saved curation log to {log_path}")

    @staticmethod
    def _count_by_field(
        items: list[dict[str, Any]], field: str
    ) -> dict[str, int]:
        """Count occurrences of each value for a given field."""
        counts: dict[str, int] = defaultdict(int)
        for item in items:
            counts[item.get(field, "unknown")] += 1
        return dict(sorted(counts.items()))
