"""
Curate v2 evaluation prompts from LLM-generated candidates.

Reads from datasets/llm_generated/candidates_v2.json, applies the curation
pipeline (embeddings → deduplication → stratified sampling), and saves
to datasets/curated/prompts_v2.json.

Usage:
    python scripts/curate_prompts_v2.py
    python scripts/curate_prompts_v2.py --n-per-category 12
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from src.datasets.curator import Curator
from src.datasets.taskbench_loader import RawSample
from src.utils.io import load_json
from src.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Curate v2 evaluation prompts from LLM-generated candidates."
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default="datasets/llm_generated/candidates_v2.json",
        help="Path to candidates_v2.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/curated/prompts_v2.json",
        help="Output path for curated v2 prompts",
    )
    parser.add_argument(
        "--n-per-category",
        type=int,
        default=12,
        help="Target number of prompts per category (default: 12 → 48 total)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.92,
        help="Cosine similarity deduplication threshold",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )

    args = parser.parse_args()
    setup_logging(log_name_prefix="curation_v2")

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"ERROR: Candidates file not found: {candidates_path}")
        return 1

    print(f"Loading candidates from {candidates_path}...")
    candidates_raw = load_json(candidates_path)
    print(f"  Loaded {len(candidates_raw)} candidate prompts")

    samples: list[RawSample] = []
    for item in candidates_raw:
        sample = RawSample(
            source="llm_generated_v2",
            original_id=f"v2_{item.get('id', len(samples))}",
            instruction=item.get("prompt", "").strip(),
            reference=item.get("reference", {}),
            category=item.get("category", "general_reasoning"),
            difficulty=item.get("difficulty", "medium"),
            metadata={
                "candidate_id": item.get("id"),
                "generation_model": item.get("generation_model", "gpt-5.4-mini"),
                "notes": item.get("notes", ""),
            },
        )
        if sample.instruction:
            samples.append(sample)

    print(f"  Converted {len(samples)} valid RawSample objects")

    print("\nRunning curation pipeline (embeddings → dedup → stratified sample)...")
    curator = Curator(
        n_per_category=args.n_per_category,
        similarity_threshold=args.threshold,
        random_seed=args.seed,
    )
    curated = curator.curate(samples)

    output_path = Path(args.output)
    curator.save(curated, output_path)
    print(f"\nSaved {len(curated)} curated v2 prompts to {output_path}")

    print("\n" + "=" * 60)
    print("  V2 CURATION SUMMARY")
    print("=" * 60)

    by_cat_diff: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p in curated:
        by_cat_diff[p["category"]][p["difficulty"]] += 1

    print(f"{'Category':<25} {'Low':>6} {'Med':>6} {'High':>6} {'Total':>6}")
    print("-" * 60)
    for cat in sorted(by_cat_diff.keys()):
        diffs = by_cat_diff[cat]
        low = diffs.get("low", 0)
        med = diffs.get("medium", 0)
        high = diffs.get("high", 0)
        total = low + med + high
        print(f"{cat:<25} {low:>6} {med:>6} {high:>6} {total:>6}")
    print("-" * 60)
    print(f"{'TOTAL':<25} {'':>6} {'':>6} {'':>6} {len(curated):>6}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
