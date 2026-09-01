"""
Curate representative evaluation prompts from TaskBench and AgentBench.

Usage:
    python scripts/curate_prompts.py
    python scripts/curate_prompts.py --n-per-category 15 --threshold 0.90
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from src.datasets.agentbench_loader import AgentBenchLoader
from src.datasets.curator import Curator
from src.datasets.taskbench_loader import TaskBenchLoader
from src.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Curate evaluation prompts from benchmark datasets."
    )
    parser.add_argument(
        "--taskbench-dir",
        type=str,
        default="datasets/taskbench",
        help="Path to TaskBench data directory",
    )
    parser.add_argument(
        "--agentbench-dir",
        type=str,
        default="datasets/agentbench",
        help="Path to AgentBench data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/curated/prompts.json",
        help="Output path for curated prompts",
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
    setup_logging(log_name_prefix="curation")

    print(f"Loading TaskBench from {args.taskbench_dir}...")
    tb_loader = TaskBenchLoader(args.taskbench_dir)
    tb_samples = tb_loader.load()
    print(f"  Loaded {len(tb_samples)} TaskBench samples")

    print(f"Loading AgentBench from {args.agentbench_dir}...")
    ab_loader = AgentBenchLoader(args.agentbench_dir)
    ab_samples = ab_loader.load()
    print(f"  Loaded {len(ab_samples)} AgentBench samples")

    all_samples = tb_samples + ab_samples
    print(f"\nTotal pooled samples: {len(all_samples)}")

    if not all_samples:
        print("ERROR: No samples found. Did you run 'python scripts/download_datasets.py'?")
        return 1

    print("\nRunning curation pipeline (embeddings → dedup → stratified sample)...")
    curator = Curator(
        n_per_category=args.n_per_category,
        similarity_threshold=args.threshold,
        random_seed=args.seed,
    )
    curated = curator.curate(all_samples)

    output_path = Path(args.output)
    curator.save(curated, output_path)
    print(f"\nSaved {len(curated)} curated prompts to {output_path}")

    print("\n" + "=" * 60)
    print("  CURATION SUMMARY")
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
