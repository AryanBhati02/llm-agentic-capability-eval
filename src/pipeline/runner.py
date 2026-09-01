"""
Experiment runner — orchestrates LLM generation across all models and prompts.

Features:
  - Crash-resume: checks disk before every call, skips already-completed
  - Order randomization: deterministic shuffle of (prompt, model) pairs
  - Concurrency limits: per-model Semaphore controls parallel requests
  - Atomic JSON writes: temporary file + rename prevents corruption
  - Run configuration snapshotting: saves exact config for reproducibility
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.llm.base import BaseLLMClient
from src.llm.factory import create_clients_from_config
from src.pipeline.cost_estimator import estimate_cost, print_cost_table
from src.utils.config import ExperimentConfig, ModelsConfig
from src.utils.io import (
    ensure_dir,
    get_response_path,
    load_json,
    response_exists,
    save_json,
)

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates the full decomposition experiment.

    Args:
        experiment_config: Inference parameters and execution flags.
        models_config: Model catalog with provider configurations.
        prompts_path: Path to the curated prompts JSON file.
        prompt_template: The decomposition prompt template string.
    """

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        models_config: ModelsConfig,
        prompts_path: str | Path,
        prompt_template: str,
    ) -> None:
        self.exp_config = experiment_config
        self.models_config = models_config
        self.prompts_path = Path(prompts_path)
        self.prompt_template = prompt_template

        self.prompts = load_json(self.prompts_path)

        if self.exp_config.pilot_mode:
            self.output_dir = Path(self.exp_config.pilot_output_dir)
        else:
            self.output_dir = Path(self.exp_config.output_dir)

    async def run(self) -> dict[str, Any]:
        """Execute the experiment.

        Returns:
            Summary dict with completed, skipped, and failed counts.
        """
        active_models = self.models_config.get_active_models(
            self.exp_config.pilot_mode
        )

        all_pairs = [
            (p["id"], model_name, p)
            for p in self.prompts
            for model_name in active_models
        ]

        pending = [
            (pid, mname, p)
            for pid, mname, p in all_pairs
            if not response_exists(self.output_dir, mname, pid)
        ]

        skipped = len(all_pairs) - len(pending)
        if skipped > 0:
            logger.info(
                f"Resuming: {skipped}/{len(all_pairs)} responses already "
                f"exist on disk — skipping them"
            )

        if self.exp_config.dry_run:
            logger.info("DRY-RUN mode: estimating cost only (no API calls)")
            estimates = estimate_cost(self.prompts, active_models)
            print_cost_table(estimates)
            return {
                "dry_run": True,
                "total_pairs": len(all_pairs),
                "pending_pairs": len(pending),
                "skipped_pairs": skipped,
                "estimates": estimates,
            }

        if not pending:
            logger.info("All prompt-model pairs are already completed.")
            return {"completed": 0, "skipped": skipped, "failed": 0}

        return await self._execute_pending(pending, active_models)

    async def _execute_pending(
        self,
        pending_pairs: list[tuple[int, str, dict]],
        active_models: dict,
    ) -> dict[str, Any]:
        """Execute the pending (prompt, model) generation calls."""
        if self.exp_config.randomize_order:
            rng = random.Random(self.exp_config.random_seed)
            rng.shuffle(pending_pairs)
            logger.info(
                f"Randomized execution order (seed={self.exp_config.random_seed})"
            )

        clients = create_clients_from_config(
            self.models_config, self.exp_config.pilot_mode
        )

        self._save_run_config(active_models)

        semaphores = {
            mname: asyncio.Semaphore(mcfg.concurrency_limit)
            for mname, mcfg in active_models.items()
        }

        completed = 0
        failed = 0

        logger.info(
            f"Starting generation: {len(pending_pairs)} calls across "
            f"{len(active_models)} models"
        )

        tasks = [
            self._generate_one(
                pid, mname, pdata, clients[mname], semaphores[mname]
            )
            for pid, mname, pdata in pending_pairs
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                failed += 1
                logger.error(f"Task failed with unhandled exception: {result}")
            elif result:
                completed += 1
            else:
                failed += 1

        for client in clients.values():
            if hasattr(client, "close"):
                await client.close()

        summary = {
            "total_pending": len(pending_pairs),
            "completed": completed,
            "failed": failed,
            "skipped": len(self.prompts) * len(active_models) - len(pending_pairs),
        }
        logger.info(f"Experiment run complete: {summary}")
        return summary

    def _build_prompt(self, task_text: str) -> str:
        """Format the prompt template with the task description."""
        if "{task_prompt}" in self.prompt_template:
            return self.prompt_template.format(task_prompt=task_text)
        return f"{self.prompt_template}\n\nTask: {task_text}"

    async def _generate_one(
        self,
        prompt_id: int,
        model_name: str,
        prompt_data: dict,
        client: BaseLLMClient,
        semaphore: asyncio.Semaphore,
    ) -> bool:
        """Generate and save a single response under concurrency control."""
        task_text = prompt_data.get("prompt", "")
        full_prompt = self._build_prompt(task_text)

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
                    f"Saved [{model_name}] prompt {prompt_id} "
                    f"({response.latency_seconds:.2f}s, "
                    f"{response.input_tokens}in/{response.output_tokens}out)"
                )
                return True

            except Exception as e:
                logger.error(
                    f"Generation failed for [{model_name}] prompt {prompt_id}: {e}"
                )
                return False

    def _save_run_config(self, active_models: dict) -> None:
        """Save the exact run configuration to the output directory."""
        ensure_dir(self.output_dir)
        run_config = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment_config": self.exp_config.model_dump(),
            "models": {k: v.model_dump() for k, v in active_models.items()},
            "prompts_file": str(self.prompts_path),
            "prompts_count": len(self.prompts),
        }
        config_path = self.output_dir / "_run_config.json"
        save_json(run_config, config_path)
        logger.info(f"Run configuration saved to {config_path}")
