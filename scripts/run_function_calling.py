"""
Run the function-calling accuracy evaluation across all models.

Injects the 20-tool catalog into prompts/function_calling.txt, runs each of the
50 single-tool requests across all research models, and writes responses to
outputs/function_calls/{model}/prompt_N.json.

Usage:
    python scripts/run_function_calling.py --dry-run
    python scripts/run_function_calling.py --pilot
    python scripts/run_function_calling.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from src.pipeline.runner import ExperimentRunner
from src.utils.config import load_experiment_config, load_models_config
from src.utils.io import load_json
from src.utils.logging import setup_logging


def _format_tools_for_prompt(tools: list[dict]) -> str:
    """Format the tools JSON into a clear schema block for the prompt."""
    lines = []
    for t in tools:
        name = t.get("name", "")
        desc = t.get("description", "")
        params = t.get("parameters", {})
        param_desc = []
        for p_name, p_info in params.items():
            req_str = " (required)" if p_info.get("required") else ""
            p_type = p_info.get("type", "string")
            p_doc = p_info.get("description", "")
            param_desc.append(f"    - {p_name} ({p_type}{req_str}): {p_doc}")
        params_block = "\n".join(param_desc) if param_desc else "    (none)"
        lines.append(f"### `{name}`\n{desc}\nParameters:\n{params_block}")
    return "\n\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run function-calling evaluation across models."
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
        default="datasets/function_calling/prompts.json",
        help="Path to function calling prompts JSON",
    )
    parser.add_argument(
        "--tools",
        type=str,
        default="datasets/function_calling/tools.json",
        help="Path to tools definition JSON",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="prompts/function_calling.txt",
        help="Path to function calling prompt template",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/function_calls",
        help="Output directory for responses",
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
    setup_logging(log_name_prefix="function_calling")

    exp_config = load_experiment_config(args.config)
    models_config = load_models_config(args.models)

    if args.dry_run:
        exp_config.dry_run = True
    if args.pilot:
        exp_config.pilot_mode = True
        exp_config.pilot_output_dir = "outputs/_pilot_function_calls"
    else:
        exp_config.output_dir = args.output_dir

    prompts_path = Path(args.prompts)
    tools_path = Path(args.tools)
    template_path = Path(args.template)

    if not prompts_path.exists():
        print(f"ERROR: Prompts file not found: {prompts_path}")
        return 1
    if not tools_path.exists():
        print(f"ERROR: Tools file not found: {tools_path}")
        return 1
    if not template_path.exists():
        print(f"ERROR: Template file not found: {template_path}")
        return 1

    tools = load_json(tools_path)
    tools_text = _format_tools_for_prompt(tools)
    raw_template = template_path.read_text(encoding="utf-8")
    full_template = raw_template.replace("{tools_description}", tools_text)

    prompts_raw = load_json(prompts_path)
    normalized_prompts = []
    for p in prompts_raw:
        item = dict(p)
        if "query" in item and "prompt" not in item:
            item["prompt"] = item["query"]
        normalized_prompts.append(item)

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(normalized_prompts, f, ensure_ascii=False)
        temp_prompts_path = Path(f.name)

    try:
        runner = ExperimentRunner(
            experiment_config=exp_config,
            models_config=models_config,
            prompts_path=temp_prompts_path,
            prompt_template=full_template,
        )

        runner.prompts_path = prompts_path

        summary = asyncio.run(runner.run())

        if summary.get("dry_run"):
            return 0

        print("\n" + "=" * 60)
        print("  FUNCTION CALLING EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Completed: {summary.get('completed', 0)}")
        print(f"Skipped (already on disk): {summary.get('skipped', 0)}")
        print(f"Failed: {summary.get('failed', 0)}")
        print("=" * 60 + "\n")

        return 0 if summary.get("failed", 0) == 0 else 1

    finally:
        temp_prompts_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
