"""
Run the LLM-as-a-Judge evaluation pass on stored responses.

Usage:
    python scripts/run_judge.py
    python scripts/run_judge.py --pilot
    python scripts/run_judge.py --judge-model gpt-5.4-mini
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from src.evaluation.judge import JudgePipeline
from src.utils.config import load_judge_config, load_models_config
from src.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run LLM-as-a-Judge evaluation on stored responses."
    )
    parser.add_argument(
        "--judge-config",
        type=str,
        default="config/judge.yaml",
        help="Path to judge configuration YAML",
    )
    parser.add_argument(
        "--models-config",
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
        "--responses-dir",
        type=str,
        default="outputs/responses",
        help="Path to responses directory",
    )
    parser.add_argument(
        "--judgments-dir",
        type=str,
        default=None,
        help="Optional override for judgments output directory (e.g. outputs/judgments_v2)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Judge pilot responses using a local Ollama model as judge",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Override the judge model specified in judge.yaml",
    )

    args = parser.parse_args()
    setup_logging(log_name_prefix="judge")

    judge_config = load_judge_config(args.judge_config)
    models_config = load_models_config(args.models_config)

    if args.judge_model:
        judge_config.judge_model = args.judge_model

    if args.pilot:
        judge_config.judge_provider = "ollama"
        judge_config.judge_model = "llama3.2:3b"
        judge_config.judge_api_key_env = ""
        judge_config.judgments_dir = "outputs/_pilot/judgments"
        judge_config.anonymization_map_path = "outputs/_pilot/anonymization_map.json"
        responses_dir = "outputs/_pilot"
        active_models = list(models_config.get_active_models(pilot_mode=True).keys())
    else:
        responses_dir = args.responses_dir
        active_models = list(models_config.get_active_models(pilot_mode=False).keys())

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        print(f"ERROR: Curated prompts not found: {prompts_path}")
        return 1

    resp_path = Path(responses_dir)
    if not resp_path.exists():
        print(f"ERROR: Responses directory not found: {resp_path}")
        print("Did you run 'python scripts/run_experiment.py'?")
        return 1

    pipeline = JudgePipeline(
        judge_config=judge_config,
        prompts_path=prompts_path,
        responses_dir=resp_path,
        model_names=active_models,
        judgments_dir=args.judgments_dir,
    )

    summary = asyncio.run(pipeline.run())

    print("\n" + "=" * 60)
    print("  JUDGE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Completed: {summary.get('completed', 0)}")
    print(f"Skipped (already on disk): {summary.get('skipped', 0)}")
    print(f"Failed: {summary.get('failed', 0)}")
    print("=" * 60 + "\n")

    return 0 if summary.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
