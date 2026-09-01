"""
Tool Invocation Experiment Runner.

Orchestrates the tool-invocation evaluation pass across models and prompts.
Differs from the standard ``ExperimentRunner`` in three ways:

1. **Prompt Construction**: Dynamically formats ``tool_invocation.txt`` with
   both the ``{task_prompt}`` AND the available ``{tool_pool_text}`` extracted
   from TaskBench domain files.
2. **Output Separation**: Writes to ``outputs/tool_calls/`` (or
   ``outputs/_pilot_tool_calls/``) to keep tool-invocation runs cleanly
   separated from the original natural-language decomposition outputs.
3. **Reference Tool Pool**: Looks up the corresponding ``tool_desc.json`` for
   each TaskBench sample to present the model with the valid candidate tool set.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.llm.base import BaseLLMClient
from src.llm.factory import create_clients_from_config
from src.pipeline.cost_estimator import estimate_cost, print_cost_table
from src.pipeline.runner import ExperimentRunner
from src.utils.config import ExperimentConfig, ModelsConfig
from src.utils.io import (
    ensure_dir,
    get_response_path,
    load_json,
    response_exists,
    save_json,
)

logger = logging.getLogger(__name__)


def _format_tool_pool(tools: list[dict[str, Any]]) -> str:
    """Format a list of tool definitions into a readable text block.

    Each tool in TaskBench's ``tool_desc.json`` has:
        - name: tool identifier
        - description: what the tool does
        - parameters: parameter schema/dict
    """
    if not tools:
        return "(No specific tool pool provided — use standard tool names)"

    lines = []
    for t in tools:
        name = t.get("name", t.get("tool_name", "Unknown"))
        desc = t.get("description", t.get("desc", ""))
        params = t.get("parameters", t.get("args", {}))

        param_str = ""
        if isinstance(params, dict) and params:
            param_parts = [f"{k}: {v}" for k, v in params.items()]
            param_str = f" [Params: {', '.join(param_parts)}]"
        elif isinstance(params, list) and params:
            param_str = f" [Params: {', '.join(str(p) for p in params)}]"

        lines.append(f"- **{name}**: {desc}{param_str}")

    return "\n".join(lines)


class ToolInvocationRunner(ExperimentRunner):
    """Experiment runner specialised for tool invocation evaluation.

    Extends ``ExperimentRunner`` to support per-prompt tool pools and
    independent output directories.

    Args:
        experiment_config: Inference parameters and execution flags.
        models_config: Model catalog with provider configurations.
        prompts_path: Path to the curated prompts JSON file.
        prompt_template: Content of ``prompts/tool_invocation.txt``.
        taskbench_data_dir: Path to raw TaskBench datasets for tool lookup.
        output_dir_override: Optional custom output directory.
    """

    DEFAULT_OUTPUT_DIR = "outputs/tool_calls"
    DEFAULT_PILOT_DIR = "outputs/_pilot_tool_calls"

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        models_config: ModelsConfig,
        prompts_path: str | Path,
        prompt_template: str,
        taskbench_data_dir: str | Path = "datasets/taskbench",
        output_dir_override: str | Path | None = None,
    ) -> None:
        super().__init__(
            experiment_config=experiment_config,
            models_config=models_config,
            prompts_path=prompts_path,
            prompt_template=prompt_template,
        )
        self.taskbench_data_dir = Path(taskbench_data_dir)

        if output_dir_override:
            self.output_dir = Path(output_dir_override)
        elif self.exp_config.pilot_mode:
            self.output_dir = Path(self.DEFAULT_PILOT_DIR)
        else:
            self.output_dir = Path(self.DEFAULT_OUTPUT_DIR)

        self._tool_cache: dict[str, list[dict]] = {}

    def _load_tool_pool_for_prompt(self, prompt_data: dict[str, Any]) -> list[dict]:
        """Retrieve the available tool pool for a given curated prompt.

        Resolution strategy:
        1. Read ``metadata.tool_pool`` if present in the prompt record.
        2. Fall back to loading ``tool_desc.json`` from the TaskBench domain
           directory referenced in ``metadata.source_file``.
        3. Return an empty list if no tool definitions are available.
        """
        metadata = prompt_data.get("metadata", {})

        if "tool_pool" in metadata and isinstance(metadata["tool_pool"], list):
            return metadata["tool_pool"]

        source_file = metadata.get("source_file", "")
        if not source_file:
            return []

        source_path = Path(source_file)
        domain_dir = self.taskbench_data_dir / source_path.parent
        cache_key = str(domain_dir)

        if cache_key in self._tool_cache:
            return self._tool_cache[cache_key]

        tool_desc_path = domain_dir / "tool_desc.json"
        if tool_desc_path.exists():
            try:
                data = load_json(tool_desc_path)
                tools = data if isinstance(data, list) else data.get("nodes", [])
                self._tool_cache[cache_key] = tools
                return tools
            except Exception as e:
                logger.warning(f"Failed to load tool pool from {tool_desc_path}: {e}")

        self._tool_cache[cache_key] = []
        return []

    def build_full_prompt(self, task_text: str, prompt_id: int | str | None = None) -> str:
        """Format the tool invocation prompt template.

        Injects both ``{task_prompt}`` and ``{tool_pool_text}``.
        """
        prompt_data = {}
        if prompt_id is not None:
            for p in self.prompts:
                if str(p.get("id")) == str(prompt_id):
                    prompt_data = p
                    break

        tools = self._load_tool_pool_for_prompt(prompt_data)
        tool_pool_text = _format_tool_pool(tools)

        template = self.prompt_template
        if "{task_prompt}" in template and "{tool_pool_text}" in template:
            return template.format(
                task_prompt=task_text,
                tool_pool_text=tool_pool_text,
            )
        elif "{task_prompt}" in template:
            return template.format(task_prompt=task_text)
        else:
            return (
                f"{template}\n\n"
                f"Available Tools:\n{tool_pool_text}\n\n"
                f"Task: {task_text}"
            )

    async def _generate_one(
        self,
        prompt_id: int,
        model_name: str,
        prompt_data: dict,
        client: BaseLLMClient,
        semaphore: asyncio.Semaphore,
    ) -> bool:
        """Generate and save a single tool-invocation response."""
        task_text = prompt_data.get("prompt", "")
        full_prompt = self.build_full_prompt(task_text, prompt_id=prompt_id)

        async with semaphore:
            try:
                response = await client.generate(full_prompt, self.exp_config)

                output_record = {
                    "prompt_id": prompt_id,
                    "model": model_name,
                    "provider": client.provider,
                    "model_id": client.model_id,
                    "prompt": task_text,
                    "full_prompt": full_prompt,
                    "response": response.text,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "latency_seconds": response.latency_seconds,
                    "model_version_reported": response.model_version_reported,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "raw_response": response.raw_response,
                }

                output_path = get_response_path(
                    self.output_dir, model_name, prompt_id
                )
                save_json(output_record, output_path)

                logger.info(
                    f"Saved [{model_name}] tool-call prompt {prompt_id} "
                    f"({response.latency_seconds:.2f}s, "
                    f"{response.input_tokens}in/{response.output_tokens}out)"
                )
                return True

            except Exception as e:
                logger.error(
                    f"Tool-call generation failed for [{model_name}] prompt {prompt_id}: {e}"
                )
                return False
