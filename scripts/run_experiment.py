"""
Run the task decomposition experiment across all models.

Usage:
    python scripts/run_experiment.py --dry-run
    python scripts/run_experiment.py --pilot
    python scripts/run_experiment.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from src.pipeline.runner import ExperimentRunner
from src.utils.config import load_experiment_config, load_models_config
from src.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run task decomposition evaluation across models."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment.yaml",
        help="Path to experiment configuration YAML",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="config/models.yaml",
        help="Path to models configuration YAML",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="datasets/curated/prompts.json",
        help="Path to curated prompts JSON",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="prompts/decomposition.txt",
        help="Path to decomposition prompt template",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate token count and cost without making API calls",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run pilot mode using local Ollama models only",
    )

    args = parser.parse_args()
    setup_logging(log_name_prefix="experiment")

    exp_config = load_experiment_config(args.config)
    models_config = load_models_config(args.models)

    if args.dry_run:
        exp_config.dry_run = True
    if args.pilot:
        exp_config.pilot_mode = True

    template_path = Path(args.template)
    if not template_path.exists():
        print(f"ERROR: Prompt template not found: {template_path}")
        return 1
    prompt_template = template_path.read_text(encoding="utf-8")

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        print(f"ERROR: Curated prompts not found: {prompts_path}")
        print("Run 'python scripts/curate_prompts.py' first.")
        return 1

    runner = ExperimentRunner(
        experiment_config=exp_config,
        models_config=models_config,
        prompts_path=prompts_path,
        prompt_template=prompt_template,
    )

    summary = asyncio.run(runner.run())

    if summary.get("dry_run"):
        return 0

    print("\n" + "=" * 60)
    print("  EXPERIMENT EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Completed: {summary.get('completed', 0)}")
    print(f"Skipped (already on disk): {summary.get('skipped', 0)}")
    print(f"Failed: {summary.get('failed', 0)}")
    print("=" * 60 + "\n")

    return 0 if summary.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
