"""
Generate tool-call accuracy summary tables and plots from model responses.

Reads response files from outputs/tool_calls/{model}/prompt_N.json, scores them
against reference tool graphs in datasets/curated/prompts.json using TaskBench
TaskEval F1 metrics (Node F1, Edge F1, Tool-Param F1, Value F1), and writes:
  - metrics/tool_call_accuracy.csv
  - metrics/tool_call_accuracy.md
  - metrics/plots/tool_call_*.png

Usage:
    python scripts/generate_tool_report.py
    python scripts/generate_tool_report.py --tool-calls-dir outputs/tool_calls --output-dir metrics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.evaluation.tool_call_scorer import ToolCallMetricsAggregator
from src.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate tool-call accuracy report using TaskBench TaskEval metrics."
    )
    parser.add_argument(
        "--tool-calls-dir",
        type=str,
        default="outputs/tool_calls",
        help="Path to tool-call responses root directory",
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
    setup_logging(log_name_prefix="tool_report")

    tool_calls_dir = Path(args.tool_calls_dir)
    prompts_path = Path(args.prompts)

    if not tool_calls_dir.exists():
        print(f"ERROR: Tool-calls directory not found: {tool_calls_dir}")
        print("Run 'python scripts/run_tool_invocation.py' first.")
        return 1

    if not prompts_path.exists():
        print(f"ERROR: Prompts file not found: {prompts_path}")
        return 1

    aggregator = ToolCallMetricsAggregator(
        tool_calls_dir=tool_calls_dir,
        prompts_path=prompts_path,
        output_dir=args.output_dir,
    )

    df = aggregator.aggregate()

    if df.empty:
        print("No tool-call responses found to score.")
        return 1

    aggregator.generate_summary(df)

    print("\n" + "=" * 60)
    print("  TOOL CALL ACCURACY SUMMARY (TaskBench TaskEval)")
    print("=" * 60)
    scoreable = df[df["has_reference"] & df["parse_ok"]]
    if not scoreable.empty:
        summary = (
            scoreable.groupby("model")[["n_f1", "e_f1", "t_f1", "v_f1"]]
            .mean()
            .round(3)
            .sort_values("n_f1", ascending=False)
        )
        print(summary.to_string())

    print("\n" + "=" * 60)
    print(f"Reports saved to {args.output_dir}/:")
    print(f"  - {args.output_dir}/tool_call_accuracy.csv")
    print(f"  - {args.output_dir}/tool_call_accuracy.md")
    print(f"  - {args.output_dir}/plots/tool_call_*.png")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
