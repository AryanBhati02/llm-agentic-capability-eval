"""
Generate function-call accuracy report and summary table from model responses.

Reads:
  - datasets/function_calling/prompts.json (ground truth)
  - outputs/function_calls/{model}/prompt_N.json (model outputs)

Produces:
  - metrics/function_calling_accuracy.csv (summary per response)
  - metrics/function_calling_accuracy.md (markdown report with per-tool breakdowns)

Usage:
    python scripts/generate_function_calling_report.py
    python scripts/generate_function_calling_report.py --responses-dir outputs/function_calls
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.evaluation.function_call_scorer import FunctionCallMetricsAggregator
from src.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate function-call accuracy report from response files."
    )
    parser.add_argument(
        "--responses-dir",
        type=str,
        default="outputs/function_calls",
        help="Path to directory containing per-model response subdirs",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="datasets/function_calling/prompts.json",
        help="Path to function calling prompts JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="metrics",
        help="Directory to save summary CSV and markdown",
    )

    args = parser.parse_args()
    setup_logging(log_name_prefix="function_calling_report")

    responses_dir = Path(args.responses_dir)
    prompts_path = Path(args.prompts)

    if not responses_dir.exists():
        print(f"ERROR: Responses directory not found: {responses_dir}")
        print("Run 'python scripts/run_function_calling.py' first.")
        return 1

    if not prompts_path.exists():
        print(f"ERROR: Prompts file not found: {prompts_path}")
        return 1

    aggregator = FunctionCallMetricsAggregator(
        responses_dir=responses_dir,
        prompts_path=prompts_path,
        output_dir=args.output_dir,
    )

    df = aggregator.aggregate()

    if df.empty:
        print("No scoreable responses found. Check directory paths.")
        return 1

    aggregator.generate_summary(df)

    print("\n" + "=" * 60)
    print("  FUNCTION CALL ACCURACY SUMMARY")
    print("=" * 60)

    parseable = df[df["parse_ok"]]
    if not parseable.empty:
        summary = (
            parseable.groupby("model")
            .agg(
                tool_accuracy=("tool_correct", "mean"),
                full_accuracy=("full_correct", "mean"),
                count=("tool_correct", "count"),
            )
            .round(3)
        )
        print(summary.to_string())

    print("\nReport written to:")
    print(f"  - {args.output_dir}/function_calling_accuracy.csv")
    print(f"  - {args.output_dir}/function_calling_accuracy.md")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
