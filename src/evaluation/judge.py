"""
LLM-as-a-Judge evaluation pipeline.

Scores each model response against a fixed rubric using a judge LLM.
The judge sees only anonymized model labels, never real names.

Rubric criteria (all 1-5 scale):
  - Completeness: covers all necessary sub-tasks
  - Logical Ordering: feasible execution order with correct dependencies
  - Correctness: steps are correct, actionable, and feasible
  - Granularity: appropriate level of detail

Judge output is structured JSON, parsed and validated before storage.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.anonymizer import (
    anonymize_model_name,
    generate_anonymization_map,
    load_anonymization_map,
    save_anonymization_map,
)
from src.llm.factory import LLMClient
from src.utils.config import ExperimentConfig, JudgeConfig, ModelConfig
from src.utils.io import (
    ensure_dir,
    get_judgment_path,
    judgment_exists,
    load_json,
    save_json,
)

logger = logging.getLogger(__name__)


_JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator of task decomposition quality.

Given the following task and a model's decomposition of it, score the decomposition on each criterion below using a {scale_min}-{scale_max} scale. Return your evaluation as a JSON object.

## Task
{task_prompt}

{reference_section}

## Model's Decomposition
{anonymized_model}: {response_text}

## Scoring Criteria
{criteria_section}

## Required Output Format
Respond with ONLY this JSON object, no other text:
{{
{json_fields}
    "justification": "<one paragraph explaining your scores>"
}}"""


def _build_criteria_section(judge_config: JudgeConfig) -> str:
    """Build the criteria description block for the judge prompt."""
    lines = []
    for c in judge_config.criteria:
        lines.append(
            f"- {c.name} ({judge_config.scale_min}-{judge_config.scale_max}): "
            f"{c.description}\n"
            f"  ({judge_config.scale_min} = {c.anchor_low}; "
            f"{judge_config.scale_max} = {c.anchor_high})"
        )
    return "\n".join(lines)


def _build_json_fields(judge_config: JudgeConfig) -> str:
    """Build the JSON field template for the expected output format."""
    lines = []
    for c in judge_config.criteria:
        lines.append(f'    "{c.name}": <int {judge_config.scale_min}-{judge_config.scale_max}>,')
    return "\n".join(lines)


def build_judge_prompt(
    task_prompt: str,
    response_text: str,
    anonymized_model: str,
    judge_config: JudgeConfig,
    reference: dict | None = None,
) -> str:
    """Build the full judge prompt for a single response evaluation.

    Args:
        task_prompt: The original task description.
        response_text: The model's decomposition response.
        anonymized_model: Anonymous model label (e.g. "Model_A").
        judge_config: Judge configuration with rubric.
        reference: Optional reference decomposition.

    Returns:
        Complete judge prompt string.
    """
    if reference and reference.get("task_steps"):
        steps = reference["task_steps"]
        if isinstance(steps, list):
            ref_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
        else:
            ref_text = str(steps)
        reference_section = f"## Reference Decomposition (for comparison)\n{ref_text}"
    else:
        reference_section = ""

    return _JUDGE_PROMPT_TEMPLATE.format(
        scale_min=judge_config.scale_min,
        scale_max=judge_config.scale_max,
        task_prompt=task_prompt,
        reference_section=reference_section,
        anonymized_model=anonymized_model,
        response_text=response_text,
        criteria_section=_build_criteria_section(judge_config),
        json_fields=_build_json_fields(judge_config),
    )


def _extract_json_object(text: str) -> str | None:
    """Extract the first complete JSON object from ``text``.

    Unlike a simple ``\\{[^{}]*\\}`` regex, this function correctly handles:

    - Braces ``{`` / ``}`` that appear **inside JSON string values** (they are
      skipped because the parser tracks whether it is currently inside a string).
    - Escaped characters (``\\"`` inside a string does not end the string).
    - Preamble text before the ``{`` and trailing text after the ``}``.
    - Nested JSON objects (depth counter).

    Args:
        text: Raw text that may contain a JSON object anywhere within it.

    Returns:
        The extracted JSON object substring, or ``None`` if no complete
        ``{...}`` pair is found.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue

        if ch == "\\" and in_string:
            escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def _strip_markdown_fence(text: str) -> str | None:
    """Return the content inside the first ```json ... ``` or ``` ... ``` block.

    Returns ``None`` if no fence is found.
    """
    fence_start_patterns = ['```json', '```']
    for opener in fence_start_patterns:
        start = text.find(opener)
        if start == -1:
            continue
        content_start = start + len(opener)
        end = text.find('```', content_start)
        if end != -1:
            return text[content_start:end].strip()
    return None


def parse_judge_response(
    raw_text: str,
    judge_config: JudgeConfig,
) -> dict[str, Any] | None:
    """Parse and validate the judge's JSON response.

    Strategy (in order):

    1. **Direct parse** — ``json.loads(raw_text.strip())``.
       Works when the model returns only the JSON object with no extra text.

    2. **Markdown fence extraction** — strip the ```json ... ``` wrapper
       first, then direct-parse the extracted content.
       Handles responses wrapped in fenced code blocks.

    3. **Brace-balanced extraction** — scan ``raw_text`` character by character,
       tracking string context so that ``{`` / ``}`` inside JSON string values
       are correctly ignored.  This finds the outermost complete ``{...}``
       object regardless of preamble, trailing commentary, or brace characters
       inside the justification text.

    Unlike the previous ``\\{[^{}]*\\}`` regex approach, the brace-balanced
    extractor cannot be tricked by fragments like ``{open editor, import media}``
    that appear inside the justification string.

    Args:
        raw_text: The judge model's raw text response.
        judge_config: Config with criteria names and scale bounds.

    Returns:
        A ``{"scores": {...}, "justification": "..."}`` dict on success,
        or ``None`` if all parse strategies and validation fail.
    """
    parsed: dict[str, Any] | None = None

    try:
        parsed = json.loads(raw_text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    if parsed is None:
        fenced_content = _strip_markdown_fence(raw_text)
        if fenced_content is not None:
            try:
                parsed = json.loads(fenced_content)
            except (json.JSONDecodeError, ValueError):
                raw_text_for_extraction = fenced_content
            else:
                raw_text_for_extraction = raw_text
        else:
            raw_text_for_extraction = raw_text
    else:
        raw_text_for_extraction = raw_text

    if parsed is None:
        candidate = _extract_json_object(raw_text_for_extraction)
        if candidate is not None:
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.debug(
                    f"Brace-balanced extraction found candidate but JSON is "
                    f"invalid: {exc}  candidate={candidate[:120]!r}"
                )

    if parsed is None:
        logger.warning(
            f"Failed to parse judge response as JSON. "
            f"Full response ({len(raw_text)} chars): {raw_text!r}"
        )
        return None

    return _validate_scores(parsed, judge_config)


def _validate_scores(
    parsed: dict[str, Any],
    judge_config: JudgeConfig,
) -> dict[str, Any] | None:
    """Validate that parsed scores are within bounds and complete."""
    scores: dict[str, int] = {}
    criteria_names = [c.name for c in judge_config.criteria]

    for name in criteria_names:
        value = parsed.get(name)
        if value is None:
            logger.warning(f"Missing criterion '{name}' in judge response")
            return None

        try:
            int_value = int(value)
        except (TypeError, ValueError):
            logger.warning(f"Non-integer score for '{name}': {value}")
            return None

        if not (judge_config.scale_min <= int_value <= judge_config.scale_max):
            logger.warning(
                f"Score for '{name}' out of range: {int_value} "
                f"(expected {judge_config.scale_min}-{judge_config.scale_max})"
            )
            int_value = max(judge_config.scale_min,
                            min(judge_config.scale_max, int_value))

        scores[name] = int_value

    justification = parsed.get("justification", "")

    return {
        "scores": scores,
        "justification": str(justification),
    }


class JudgePipeline:
    """Runs the LLM-as-a-Judge evaluation on stored responses.

    Args:
        judge_config: Judge configuration.
        prompts_path: Path to the curated prompts JSON.
        responses_dir: Directory containing model response JSONs.
        model_names: List of real model names that were evaluated.
        judgments_dir: Optional override for the judgments output directory.
            When provided, takes priority over ``judge_config.judgments_dir``.
            Use this to redirect v2 judgments to a separate path without
            modifying judge.yaml (which would break the original report's
            reproducibility).
    """

    def __init__(
        self,
        judge_config: JudgeConfig,
        prompts_path: str | Path,
        responses_dir: str | Path,
        model_names: list[str],
        judgments_dir: str | Path | None = None,
    ) -> None:
        self.judge_config = judge_config
        self.prompts = {p["id"]: p for p in load_json(prompts_path)}
        self.responses_dir = Path(responses_dir)
        self.model_names = model_names
        self.judgments_dir = str(
            Path(judgments_dir) if judgments_dir is not None
            else judge_config.judgments_dir
        )

        map_path = Path(judge_config.anonymization_map_path)
        if map_path.exists():
            self.anon_map = load_anonymization_map(map_path)
            logger.info(f"Loaded existing anonymization map from {map_path}")
        else:
            self.anon_map = generate_anonymization_map(
                model_names, seed=judge_config.anonymization_seed
            )
            save_anonymization_map(self.anon_map, map_path)

    async def run(self) -> dict[str, Any]:
        """Run the judge on all stored responses.

        Returns:
            Summary dict with completed/skipped/failed counts.
        """
        judge_model_config = ModelConfig(
            provider=self.judge_config.judge_provider,
            model_id=self.judge_config.judge_model,
            max_tokens=self.judge_config.judge_max_tokens,
            timeout_seconds=self.judge_config.judge_timeout_seconds,
            api_key_env=self.judge_config.judge_api_key_env,
            concurrency_limit=self.judge_config.judge_concurrency_limit,
        )

        judge_client = LLMClient(
            provider=self.judge_config.judge_provider,
            model=self.judge_config.judge_model,
            model_name="judge",
            config=judge_model_config,
        )

        judge_experiment_config = ExperimentConfig(
            temperature=0.0,
            top_p=1.0,
            retry_count=3,
            retry_backoff_base=2.0,
        )

        to_judge = self._collect_responses()

        pending = [
            (pid, mname, resp) for pid, mname, resp in to_judge
            if not judgment_exists(
                self.judgments_dir,
                pid,
                anonymize_model_name(mname, self.anon_map),
            )
        ]

        skipped = len(to_judge) - len(pending)
        if skipped > 0:
            logger.info(f"Skipping {skipped} already-judged responses")

        if not pending:
            logger.info("All responses already judged.")
            return {"completed": 0, "skipped": skipped, "failed": 0}

        logger.info(f"Judging {len(pending)} responses...")

        semaphore = asyncio.Semaphore(self.judge_config.judge_concurrency_limit)
        completed = 0
        failed = 0

        tasks = [
            self._judge_one(
                pid, mname, resp, judge_client, judge_experiment_config, semaphore
            )
            for pid, mname, resp in pending
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                failed += 1
                logger.error(f"Judge task failed: {result}")
            elif result:
                completed += 1
            else:
                failed += 1

        if hasattr(judge_client, "close"):
            await judge_client.close()

        summary = {
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
        }
        logger.info(f"Judge pipeline complete: {summary}")
        return summary

    def _collect_responses(self) -> list[tuple[int, str, dict]]:
        """Collect all response files to be judged."""
        responses = []

        for model_name in self.model_names:
            from src.utils.io import _sanitize_filename
            safe_name = _sanitize_filename(model_name)
            model_dir = self.responses_dir / safe_name

            if not model_dir.exists():
                logger.warning(f"No responses found for model: {model_name}")
                continue

            for json_file in sorted(model_dir.glob("prompt_*.json")):
                try:
                    resp = load_json(json_file)
                    prompt_id = resp.get("prompt_id")
                    if prompt_id is not None:
                        responses.append((prompt_id, model_name, resp))
                except Exception as e:
                    logger.warning(f"Failed to load {json_file}: {e}")

        return responses

    async def _judge_one(
        self,
        prompt_id: int,
        model_name: str,
        response_data: dict,
        judge_client,
        judge_experiment_config: ExperimentConfig,
        semaphore: asyncio.Semaphore,
    ) -> bool:
        """Judge a single response."""
        anon_label = anonymize_model_name(model_name, self.anon_map)
        prompt_data = self.prompts.get(prompt_id, {})
        task_prompt = prompt_data.get("prompt", "")
        reference = prompt_data.get("reference")
        response_text = response_data.get("response", "")

        judge_prompt = build_judge_prompt(
            task_prompt=task_prompt,
            response_text=response_text,
            anonymized_model=anon_label,
            judge_config=self.judge_config,
            reference=reference,
        )

        async with semaphore:
            try:
                judge_response = await judge_client.generate(
                    judge_prompt, judge_experiment_config
                )

                parsed = parse_judge_response(
                    judge_response.text, self.judge_config
                )

                if parsed is None:
                    logger.warning(
                        f"Failed to parse judge scores for "
                        f"prompt {prompt_id} / {anon_label}"
                    )
                    return False

                judgment = {
                    "prompt_id": prompt_id,
                    "anonymized_model": anon_label,
                    "judge_model": self.judge_config.judge_model,
                    "judge_provider": self.judge_config.judge_provider,
                    "scores": parsed["scores"],
                    "justification": parsed["justification"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "judge_latency_seconds": judge_response.latency_seconds,
                    "judge_input_tokens": judge_response.input_tokens,
                    "judge_output_tokens": judge_response.output_tokens,
                }

                path = get_judgment_path(
                    self.judgments_dir, prompt_id, anon_label
                )
                save_json(judgment, path)

                logger.info(
                    f"Judged prompt {prompt_id} / {anon_label}: "
                    f"scores={parsed['scores']}"
                )
                return True

            except Exception as e:
                logger.error(
                    f"Judge failed for prompt {prompt_id} / {anon_label}: {e}"
                )
                return False
