"""
Function-call accuracy scorer — single-tool-call accuracy metrics.

Supersedes the TaskBench-graph-based tool_call_scorer.py approach with a simpler
accuracy-based evaluation:  given a natural-language request and a fixed 20-tool
catalog, did the model pick the right tool and extract the right parameters?

Metrics
-------
Tool Selection Accuracy
    Fraction of prompts where ``predicted_tool == correct_tool`` (exact string
    match, case-insensitive).

Full Accuracy
    Fraction of prompts where the tool AND every parameter match.  Parameter
    matching is normalised: case-insensitive, whitespace-trimmed, numeric
    parameters compared as numbers (not strings).

Parse failures (malformed JSON) are tracked separately — never folded into
"wrong answer."

JSON Parsing
-------------
Uses the same three-pass strategy proven in ``judge.py`` and
``tool_call_scorer.py``:

1. Direct ``json.loads``.
2. Strip markdown fence, then direct parse.
3. Brace-balanced character-by-character extractor.

Output schema (per response)
------------------------------
    {
        "prompt_id": int,
        "model": str,
        "correct_tool": str,
        "predicted_tool": str | null,
        "parse_ok": bool,
        "tool_correct": bool,
        "full_correct": bool,
    }
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import ensure_dir, load_json, _sanitize_filename

logger = logging.getLogger(__name__)


def _strip_markdown_fence(text: str) -> str | None:
    """Return content inside the first ```json...``` or ```...``` block."""
    for opener in ["```json", "```"]:
        start = text.find(opener)
        if start == -1:
            continue
        content_start = start + len(opener)
        end = text.find("```", content_start)
        if end != -1:
            return text[content_start:end].strip()
    return None


def _extract_json_object(text: str) -> str | None:
    """Brace-balanced JSON object extractor (handles braces in string values)."""
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
                return text[start: i + 1]

    return None


def parse_function_call_response(raw_text: str) -> dict[str, Any] | None:
    """Parse the model's function-call JSON output.

    Expected format::

        {"tool": "<tool_name>", "parameters": {"<param>": "<value>", ...}}

    Applies the three-pass strategy:
    1. Direct ``json.loads``.
    2. Strip markdown fence, then direct parse.
    3. Brace-balanced extraction, then parse.

    Returns the parsed dict on success, or None on failure.
    """
    parsed: dict[str, Any] | None = None

    try:
        parsed = json.loads(raw_text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    if parsed is None:
        fenced = _strip_markdown_fence(raw_text)
        raw_for_extraction = fenced if fenced is not None else raw_text
        if fenced is not None:
            try:
                parsed = json.loads(fenced)
            except (json.JSONDecodeError, ValueError):
                pass
    else:
        raw_for_extraction = raw_text

    if parsed is None:
        candidate = _extract_json_object(raw_for_extraction)
        if candidate is not None:
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.debug(
                    f"Brace-balanced extraction found candidate but JSON invalid: {exc} "
                    f"candidate={candidate[:120]!r}"
                )

    if parsed is None:
        logger.warning(
            f"Failed to parse function-call response as JSON "
            f"({len(raw_text)} chars): {raw_text[:200]!r}"
        )
        return None

    if not isinstance(parsed, dict):
        logger.warning(f"Parsed JSON is not a dict: {type(parsed)}")
        return None

    return parsed


def _normalise_value(v: Any) -> Any:
    """Normalise a parameter value for comparison.

    - Strings: lowercased + stripped.
    - Numbers: kept as float (so "100" == 100 == 100.0).
    - Other types: converted to lowered/stripped strings.
    """
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    try:
        return float(s)
    except (ValueError, TypeError):
        return s


def _params_match(predicted: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Check whether predicted parameters match expected parameters.

    Rules:
    - Both must have the same set of keys (after lowercasing key names).
    - Each value is normalised (case-insensitive, trimmed, numeric coercion).
    """
    pred_normalised = {
        str(k).strip().lower(): _normalise_value(v)
        for k, v in predicted.items()
    }
    exp_normalised = {
        str(k).strip().lower(): _normalise_value(v)
        for k, v in expected.items()
    }

    if set(pred_normalised.keys()) != set(exp_normalised.keys()):
        return False

    for key in exp_normalised:
        if pred_normalised[key] != exp_normalised[key]:
            return False

    return True


@dataclass
class FunctionCallScore:
    """Scores for a single (prompt, model) function-call response."""

    prompt_id: int | str
    model: str
    query: str
    correct_tool: str
    predicted_tool: str | None
    tool_correct: bool
    correct_parameters: str
    predicted_parameters: str | None
    full_correct: bool
    parse_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_response(
    response_record: dict[str, Any],
    prompt_data: dict[str, Any],
) -> FunctionCallScore:
    """Score a single model response against ground truth.

    Args:
        response_record: JSON dict from ``outputs/function_calls/{model}/prompt_N.json``.
        prompt_data: Corresponding entry from ``datasets/function_calling/prompts.json``.

    Returns:
        FunctionCallScore dataclass.
    """
    prompt_id = response_record.get("prompt_id", prompt_data.get("id"))
    model = response_record.get("model", "unknown")
    query = prompt_data.get("query", "")
    correct_tool = prompt_data.get("correct_tool", "")
    correct_params = prompt_data.get("correct_parameters", {})
    correct_params_json = json.dumps(correct_params, ensure_ascii=False)

    raw_text = response_record.get("response", "")
    parsed = parse_function_call_response(raw_text)

    if parsed is None:
        return FunctionCallScore(
            prompt_id=prompt_id,
            model=model,
            query=query,
            correct_tool=correct_tool,
            predicted_tool=None,
            tool_correct=False,
            correct_parameters=correct_params_json,
            predicted_parameters=None,
            full_correct=False,
            parse_ok=False,
        )

    predicted_tool = str(parsed.get("tool", "")).strip()
    predicted_params = parsed.get("parameters", {})
    if not isinstance(predicted_params, dict):
        predicted_params = {}

    tool_match = predicted_tool.lower() == correct_tool.lower()
    param_match = _params_match(predicted_params, correct_params) if tool_match else False

    return FunctionCallScore(
        prompt_id=prompt_id,
        model=model,
        query=query,
        correct_tool=correct_tool,
        predicted_tool=predicted_tool,
        tool_correct=tool_match,
        correct_parameters=correct_params_json,
        predicted_parameters=json.dumps(predicted_params, ensure_ascii=False),
        full_correct=tool_match and param_match,
        parse_ok=True,
    )


def score_all_responses(
    responses_dir: str | Path,
    prompts_path: str | Path,
) -> list[dict[str, Any]]:
    """Score all response files found under ``responses_dir``.

    Iterates ``{responses_dir}/{model}/prompt_*.json``, matches each to its
    prompt by ``prompt_id``, and returns a flat list of score dicts.

    Args:
        responses_dir: Root output directory (e.g. ``outputs/function_calls``).
        prompts_path: Path to ``datasets/function_calling/prompts.json``.

    Returns:
        List of dicts (one per response), each from ``FunctionCallScore.to_dict()``.
    """
    responses_dir = Path(responses_dir)
    prompts = {p["id"]: p for p in load_json(prompts_path)}
    results: list[dict[str, Any]] = []

    if not responses_dir.exists():
        logger.warning(f"Function-call output directory not found: {responses_dir}")
        return results

    model_dirs = sorted(
        d for d in responses_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )

    for model_dir in model_dirs:
        model_name = model_dir.name
        response_files = sorted(model_dir.glob("prompt_*.json"))

        if not response_files:
            logger.warning(f"No response files in {model_dir}")
            continue

        for resp_file in response_files:
            try:
                record = load_json(resp_file)
            except Exception as exc:
                logger.warning(f"Failed to load {resp_file}: {exc}")
                continue

            prompt_id = record.get("prompt_id")
            prompt_data = prompts.get(prompt_id)
            if prompt_data is None:
                logger.warning(
                    f"Prompt ID {prompt_id} from {resp_file} not found in prompts.json"
                )
                continue

            scores = score_response(record, prompt_data)
            row = scores.to_dict()
            row["model"] = record.get("model", model_name)
            results.append(row)

    logger.info(
        f"Scored {len(results)} responses across {len(model_dirs)} model(s)"
    )
    return results


class FunctionCallMetricsAggregator:
    """Aggregate function-call scores into summary tables.

    Produces:
    - Per-model Tool Selection Accuracy and Full Accuracy.
    - Per-tool breakdown showing which tools each model struggles with.
    - Parse failure rate per model.
    - CSV and markdown outputs.

    Args:
        responses_dir: Directory containing ``{model}/prompt_*.json`` files.
        prompts_path: Path to ``datasets/function_calling/prompts.json``.
        output_dir: Directory for CSV and markdown outputs.
    """

    def __init__(
        self,
        responses_dir: str | Path,
        prompts_path: str | Path,
        output_dir: str | Path = "metrics",
    ) -> None:
        self.responses_dir = Path(responses_dir)
        self.prompts_path = Path(prompts_path)
        self.output_dir = Path(output_dir)

    def aggregate(self) -> pd.DataFrame:
        """Load and score all responses, returning a flat DataFrame."""
        records = score_all_responses(self.responses_dir, self.prompts_path)
        df = pd.DataFrame(records)
        if df.empty:
            logger.warning("No scored records found — is responses_dir populated?")
        else:
            logger.info(f"Loaded {len(df)} scored records")
        return df

    def generate_summary(self, df: pd.DataFrame | None = None) -> None:
        """Generate CSV, detailed responses CSV, and markdown reports.

        Args:
            df: Pre-computed DataFrame.  If None, calls ``aggregate()``.
        """
        if df is None:
            df = self.aggregate()

        if df.empty:
            logger.warning("No data to summarise.")
            return

        ensure_dir(self.output_dir)

        csv_path = self.output_dir / "function_calling_accuracy.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {csv_path}")

        detail_cols = [
            "prompt_id", "model", "query", "correct_tool", "predicted_tool",
            "tool_correct", "correct_parameters", "predicted_parameters",
            "full_correct", "parse_ok",
        ]
        detail_df = df[[c for c in detail_cols if c in df.columns]].copy()
        detail_df = detail_df.sort_values(["model", "prompt_id"]).reset_index(drop=True)
        detail_csv_path = self.output_dir / "function_calling_responses.csv"
        detail_df.to_csv(detail_csv_path, index=False)
        logger.info(f"Saved {detail_csv_path}")

        md_path = self.output_dir / "function_calling_accuracy.md"
        self._write_markdown(df, md_path)
        logger.info(f"Saved {md_path}")

    def _write_markdown(self, df: pd.DataFrame, path: Path) -> None:
        """Write the function_calling_accuracy.md summary."""

        parseable = df[df["parse_ok"]].copy()
        parse_failures = df[~df["parse_ok"]].copy()

        lines = [
            "# Function Calling Accuracy — Single-Tool-Call Evaluation",
            "",
            "> **Tool Selection Accuracy** — % of prompts where the predicted tool "
            "matches the correct tool (exact, case-insensitive).",
            "> **Full Accuracy** — % of prompts where the tool AND every parameter "
            "match (normalised: case-insensitive, trimmed, numeric coercion).",
            "",
            f"**Total responses scored:** {len(parseable)}",
            f"**Parse failures (excluded from accuracy):** {len(parse_failures)}",
            f"**Total prompts in dataset:** 50",
            "",
        ]

        if not parseable.empty:
            lines.append("## Overall Accuracy by Model")
            lines.append("")

            model_summary = (
                parseable.groupby("model")
                .agg(
                    prompts_scored=("tool_correct", "count"),
                    tool_selection_accuracy=("tool_correct", "mean"),
                    full_accuracy=("full_correct", "mean"),
                )
                .round(4)
                .sort_values("tool_selection_accuracy", ascending=False)
            )
            display = model_summary.copy()
            display["tool_selection_accuracy"] = (
                display["tool_selection_accuracy"] * 100
            ).round(1).astype(str) + "%"
            display["full_accuracy"] = (
                display["full_accuracy"] * 100
            ).round(1).astype(str) + "%"
            display.columns = ["Prompts Scored", "Tool Selection Acc", "Full Acc"]
            lines.append(display.to_markdown())
            lines.append("")

            lines.append("## Parse Failure Rate by Model")
            lines.append("")
            lines.append(
                "> Proportion of responses that could not be parsed as valid JSON."
            )
            lines.append("")

            model_parse = (
                df.groupby("model")
                .agg(
                    total=("parse_ok", "count"),
                    parse_failures=("parse_ok", lambda x: (~x).sum()),
                    parse_failure_rate=("parse_ok", lambda x: (~x).mean()),
                )
                .round(3)
                .sort_values("parse_failure_rate", ascending=False)
            )
            model_parse["parse_failure_rate"] = (
                model_parse["parse_failure_rate"] * 100
            ).round(1).astype(str) + "%"
            model_parse.columns = ["Total", "Failures", "Failure Rate"]
            lines.append(model_parse.to_markdown())
            lines.append("")

            lines.append("## Per-Tool Accuracy Breakdown (Tool × Model)")
            lines.append("")
            lines.append(
                "> Each cell shows **correct / total** attempts. "
                "Rows = tools (sorted alphabetically), columns = models."
            )
            lines.append("")

            models_sorted = sorted(parseable["model"].unique())

            lines.append("### Tool Selection Accuracy")
            lines.append("")
            tool_sel_table = self._build_cross_tab(
                parseable, "tool_correct", models_sorted
            )
            lines.append(tool_sel_table)
            lines.append("")

            lines.append("### Full Accuracy (Tool + Parameters)")
            lines.append("")
            tool_full_table = self._build_cross_tab(
                parseable, "full_correct", models_sorted
            )
            lines.append(tool_full_table)
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _build_cross_tab(
        df: pd.DataFrame,
        metric_col: str,
        models: list[str],
    ) -> str:
        """Build a tool × model cross-tab markdown table.

        Each cell is "correct/total" for that (tool, model) pair.
        """
        tools_sorted = sorted(df["correct_tool"].unique())

        header = "| Tool | " + " | ".join(models) + " |"
        separator = "|:---" + ("|:---:" * len(models)) + "|"

        rows = [header, separator]
        for tool in tools_sorted:
            tool_df = df[df["correct_tool"] == tool]
            cells = []
            for model in models:
                model_tool_df = tool_df[tool_df["model"] == model]
                total = len(model_tool_df)
                correct = int(model_tool_df[metric_col].sum()) if total > 0 else 0
                cells.append(f"{correct}/{total}" if total > 0 else "—")
            rows.append(f"| {tool} | " + " | ".join(cells) + " |")

        return "\n".join(rows)
