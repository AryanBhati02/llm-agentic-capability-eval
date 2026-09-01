"""
Generate evaluation summary tables and plots from stored judgments.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --output-dir metrics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.evaluation.metrics import MetricsAggregator
from src.utils.config import load_judge_config
from src.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate evaluation summary report from judge outputs."
    )
    parser.add_argument(
        "--judge-config",
        type=str,
        default="config/judge.yaml",
        help="Path to judge configuration YAML",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="datasets/curated/prompts.json",
        help="Path to curated prompts JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="metrics",
        help="Directory to save summary CSV, markdown, and plots",
    )

    args = parser.parse_args()
    setup_logging(log_name_prefix="report")

    judge_config = load_judge_config(args.judge_config)

    judgments_dir = Path(judge_config.judgments_dir)
    map_path = Path(judge_config.anonymization_map_path)
    prompts_path = Path(args.prompts)

    if not judgments_dir.exists():
        print(f"ERROR: Judgments directory not found: {judgments_dir}")
        print("Did you run 'python scripts/run_judge.py'?")
        return 1

    if not map_path.exists():
        print(f"ERROR: Anonymization map not found: {map_path}")
        return 1

    aggregator = MetricsAggregator(
        judgments_dir=judgments_dir,
        anonymization_map_path=map_path,
        prompts_path=prompts_path,
        output_dir=args.output_dir,
    )

    df = aggregator.aggregate()

    if df.empty:
        print("No judgments found to summarize.")
        return 1

    aggregator.generate_summary(df)

    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    overall = (
        df.groupby("model")["score"]
        .agg(["mean", "std", "count"])
        .round(2)
        .sort_values("mean", ascending=False)
    )
    print(overall.to_string())
    print("\n" + "=" * 60)
    print(f"Reports saved to {args.output_dir}/:")
    print(f"  - {args.output_dir}/summary.csv")
    print(f"  - {args.output_dir}/summary.md")
    print(f"  - {args.output_dir}/plots/*.png")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
