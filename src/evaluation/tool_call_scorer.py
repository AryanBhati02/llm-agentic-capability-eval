"""
Superseded by function_call_scorer.py (2026-07-27); kept for reference.

Tool-call accuracy scorer — TaskBench TaskEval metrics.

Implements the four F1 metrics from the TaskBench paper (Shen et al., ICLR 2024)
as pure set-comparison math on parsed JSON text output.  No LLM call is made.

Metrics
-------
n-F1 (Node F1)
    Precision/recall over predicted tool names vs. reference tool names.
    Order-independent; a predicted node counts as correct if its name matches
    any ground-truth node name.

e-F1 (Edge F1)
    Precision/recall over predicted (source_tool, target_tool) dependency pairs
    vs. reference (source_tool, target_tool) pairs.  Both tool names must match
    exactly.  Edges are resolved from the model's (from_step_id → to_step_id)
    pairs via the step_id→tool mapping; reference edges already use tool names
    directly (confirmed from data inspection).

t-F1 (Tool-parameter F1)
    Precision/recall over (tool_name, parameter_name) pairs.
    Available only for DailyLife-domain prompts whose reference ``arguments``
    are ``{name, value}`` dicts.  For HuggingFace/Multimedia prompts whose
    reference arguments are positional plain strings, the reference t-pair set
    is empty and t-F1 = 0.0 (documented limitation, not a bug).

v-F1 (Value F1)
    Precision/recall over (tool_name, parameter_name, parameter_value) triples.
    Stricter than t-F1; the value must also match.  Same domain caveat as t-F1.

JSON Parsing
------------
Uses the same three-pass strategy as ``judge.py``:
1. Direct ``json.loads``.
2. Strip markdown fence, then direct parse.
3. Brace-balanced character-by-character extractor.

parse_ok=False rows are tracked separately.  The aggregator excludes them from
F1 means and reports them as a parse-failure rate.

Output schema (per response)
-----------------------------
    {
        "prompt_id": int,
        "model": str,
        "category": str,
        "difficulty": str,
        "has_reference": bool,
        "parse_ok": bool,
        "n_f1": float | null,
        "e_f1": float | null,
        "t_f1": float | null,
        "v_f1": float | null,
    }

``null`` means the metric could not be computed (no reference or parse failure).
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


def parse_tool_response(raw_text: str) -> dict[str, Any] | None:
    """Parse the model's tool-graph JSON output.

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
            f"Failed to parse tool-graph response as JSON "
            f"({len(raw_text)} chars): {raw_text[:200]!r}"
        )
        return None

    if not isinstance(parsed, dict):
        logger.warning(f"Parsed JSON is not a dict: {type(parsed)}")
        return None

    return parsed


def _extract_predicted_sets(
    parsed: dict[str, Any],
) -> tuple[
    frozenset[str],
    frozenset[tuple[str, str]],
    frozenset[tuple[str, str]],
    frozenset[tuple[str, str, str]],
]:
    """Extract the four prediction sets from a parsed tool-graph dict.

    step_id → tool mapping is built first; dependency (from, to) step_id pairs
    are then resolved to (source_tool, target_tool) pairs for edge scoring.
    """
    tools_list: list[dict] = parsed.get("tools", [])
    deps_list: list[dict] = parsed.get("dependencies", [])

    step_to_tool: dict[str, str] = {}
    for entry in tools_list:
        sid = str(entry.get("step_id", ""))
        tool = str(entry.get("tool", "")).strip()
        if sid and tool:
            step_to_tool[sid] = tool

    pred_nodes: frozenset[str] = frozenset(step_to_tool.values())

    pred_edges_raw: set[tuple[str, str]] = set()
    for dep in deps_list:
        from_sid = str(dep.get("from", ""))
        to_sid = str(dep.get("to", ""))
        src_tool = step_to_tool.get(from_sid)
        tgt_tool = step_to_tool.get(to_sid)
        if src_tool and tgt_tool:
            pred_edges_raw.add((src_tool, tgt_tool))
    pred_edges: frozenset[tuple[str, str]] = frozenset(pred_edges_raw)

    pred_t_pairs_raw: set[tuple[str, str]] = set()
    pred_v_triples_raw: set[tuple[str, str, str]] = set()
    for entry in tools_list:
        tool = str(entry.get("tool", "")).strip()
        params = entry.get("parameters", {})
        if not tool or not isinstance(params, dict):
            continue
        for param_name, param_value in params.items():
            pred_t_pairs_raw.add((tool, str(param_name)))
            pred_v_triples_raw.add((tool, str(param_name), str(param_value)))

    return (
        pred_nodes,
        pred_edges,
        frozenset(pred_t_pairs_raw),
        frozenset(pred_v_triples_raw),
    )


def _extract_reference_sets(
    reference: dict[str, Any],
) -> tuple[
    frozenset[str],
    frozenset[tuple[str, str]],
    frozenset[tuple[str, str]],
    frozenset[tuple[str, str, str]],
]:
    """Extract the four reference sets from a curated prompt's reference dict.

    Field layout (after loader fix):
        reference["task_nodes"]: list of {task, arguments, ...}
        reference["task_links"]: list of {source, target}  — already tool names

    Argument schema differs by domain:
        DailyLife: arguments = list of {name, value} dicts
            → (tool, arg["name"]) for t-pairs; (tool, arg["name"], arg["value"]) for v-triples
        HuggingFace/Multimedia: arguments = list of plain strings (positional)
            → t-pairs and v-triples are empty (documented limitation)
    """
    task_nodes: list[dict] = reference.get("task_nodes", [])
    task_links: list[dict] = reference.get("task_links", [])

    ref_nodes: frozenset[str] = frozenset(
        str(node.get("task", "")).strip()
        for node in task_nodes
        if isinstance(node, dict) and node.get("task")
    )

    ref_edges: frozenset[tuple[str, str]] = frozenset(
        (str(link.get("source", "")), str(link.get("target", "")))
        for link in task_links
        if isinstance(link, dict)
        and link.get("source")
        and link.get("target")
    )

    ref_t_pairs_raw: set[tuple[str, str]] = set()
    ref_v_triples_raw: set[tuple[str, str, str]] = set()
    for node in task_nodes:
        if not isinstance(node, dict):
            continue
        tool = str(node.get("task", "")).strip()
        arguments = node.get("arguments", [])
        if not tool or not isinstance(arguments, list):
            continue
        for arg in arguments:
            if isinstance(arg, dict) and "name" in arg and "value" in arg:
                ref_t_pairs_raw.add((tool, str(arg["name"])))
                ref_v_triples_raw.add((tool, str(arg["name"], arg["value"])))

    return (
        ref_nodes,
        ref_edges,
        frozenset(ref_t_pairs_raw),
        frozenset(ref_v_triples_raw),
    )


def _f1(pred: frozenset, ref: frozenset) -> float:
    """Compute F1 from two sets.

    Returns 0.0 when both sets are empty (avoids 0/0).
    Returns 0.0 when one set is empty and the other is not (no overlap possible).
    """
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    tp = len(pred & ref)
    precision = tp / len(pred)
    recall = tp / len(ref)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


@dataclass
class ToolCallScores:
    """Scores for a single (prompt, model) tool-call response."""

    prompt_id: int | str
    model: str
    category: str
    difficulty: str
    has_reference: bool
    parse_ok: bool
    n_f1: float | None
    e_f1: float | None
    t_f1: float | None
    v_f1: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_response(
    response_record: dict[str, Any],
    prompt_data: dict[str, Any],
) -> ToolCallScores:
    """Score a single model response against the reference tool graph.

    Args:
        response_record: JSON dict from ``outputs/tool_calls/{model}/prompt_N.json``.
        prompt_data: Corresponding entry from ``datasets/curated/prompts.json``.

    Returns:
        ToolCallScores dataclass.
    """
    prompt_id = response_record.get("prompt_id", prompt_data.get("id"))
    model = response_record.get("model", "unknown")
    category = prompt_data.get("category", "unknown")
    difficulty = prompt_data.get("difficulty", "unknown")
    reference = prompt_data.get("reference", {})

    has_reference = bool(reference.get("task_nodes"))
    if not has_reference:
        logger.info(
            f"Prompt {prompt_id}: no reference tool graph "
            f"(likely user_requests.json source) — skipping scoring"
        )
        return ToolCallScores(
            prompt_id=prompt_id,
            model=model,
            category=category,
            difficulty=difficulty,
            has_reference=False,
            parse_ok=False,
            n_f1=None,
            e_f1=None,
            t_f1=None,
            v_f1=None,
        )

    raw_text = response_record.get("response", "")
    parsed = parse_tool_response(raw_text)
    if parsed is None:
        return ToolCallScores(
            prompt_id=prompt_id,
            model=model,
            category=category,
            difficulty=difficulty,
            has_reference=True,
            parse_ok=False,
            n_f1=None,
            e_f1=None,
            t_f1=None,
            v_f1=None,
        )

    pred_nodes, pred_edges, pred_t, pred_v = _extract_predicted_sets(parsed)
    ref_nodes, ref_edges, ref_t, ref_v = _extract_reference_sets(reference)

    return ToolCallScores(
        prompt_id=prompt_id,
        model=model,
        category=category,
        difficulty=difficulty,
        has_reference=True,
        parse_ok=True,
        n_f1=_f1(pred_nodes, ref_nodes),
        e_f1=_f1(pred_edges, ref_edges),
        t_f1=_f1(pred_t, ref_t),
        v_f1=_f1(pred_v, ref_v),
    )


def score_all_responses(
    tool_calls_dir: str | Path,
    prompts_path: str | Path,
) -> list[dict[str, Any]]:
    """Score all response files found under ``tool_calls_dir``.

    Iterates ``{tool_calls_dir}/{model}/prompt_*.json``, matches each to its
    curated prompt by ``prompt_id``, and returns a flat list of score dicts.

    Args:
        tool_calls_dir: Root output directory (e.g. ``outputs/tool_calls``).
        prompts_path: Path to ``datasets/curated/prompts.json``.

    Returns:
        List of dicts (one per response), each from ``ToolCallScores.to_dict()``.
    """
    tool_calls_dir = Path(tool_calls_dir)
    prompts = {p["id"]: p for p in load_json(prompts_path)}
    results: list[dict[str, Any]] = []

    if not tool_calls_dir.exists():
        logger.warning(f"Tool-call output directory not found: {tool_calls_dir}")
        return results

    model_dirs = sorted(
        d for d in tool_calls_dir.iterdir() if d.is_dir() and not d.name.startswith("_")
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


class ToolCallMetricsAggregator:
    """Aggregate tool-call scores into summary tables and plots.

    Mirrors MetricsAggregator from metrics.py but operates on the four
    TaskBench F1 metrics instead of LLM-as-a-judge rubric scores.

    Args:
        tool_calls_dir: Directory containing ``{model}/prompt_*.json`` files.
        prompts_path: Path to curated prompts JSON.
        output_dir: Directory for CSV, markdown, and plot outputs.
    """

    METRICS = ["n_f1", "e_f1", "t_f1", "v_f1"]
    METRIC_LABELS = {
        "n_f1": "Node F1",
        "e_f1": "Edge F1",
        "t_f1": "Tool-Param F1",
        "v_f1": "Value F1",
    }

    def __init__(
        self,
        tool_calls_dir: str | Path,
        prompts_path: str | Path,
        output_dir: str | Path = "metrics",
    ) -> None:
        self.tool_calls_dir = Path(tool_calls_dir)
        self.prompts_path = Path(prompts_path)
        self.output_dir = Path(output_dir)

    def aggregate(self) -> pd.DataFrame:
        """Load and score all responses, returning a flat DataFrame.

        Columns: model, prompt_id, category, difficulty, has_reference,
                 parse_ok, n_f1, e_f1, t_f1, v_f1.
        """
        records = score_all_responses(self.tool_calls_dir, self.prompts_path)
        df = pd.DataFrame(records)
        if df.empty:
            logger.warning("No scored records found — is tool_calls_dir populated?")
        else:
            logger.info(f"Loaded {len(df)} scored records")
        return df

    def generate_summary(self, df: pd.DataFrame | None = None) -> None:
        """Generate CSV, markdown, and plots.

        Args:
            df: Pre-computed DataFrame. If None, calls ``aggregate()``.
        """
        if df is None:
            df = self.aggregate()

        if df.empty:
            logger.warning("No data to summarise.")
            return

        ensure_dir(self.output_dir)
        ensure_dir(self.output_dir / "plots")

        scoreable = df[df["has_reference"] & df["parse_ok"]].copy()
        parse_failures = df[df["has_reference"] & ~df["parse_ok"]]
        no_reference = df[~df["has_reference"]]

        logger.info(
            f"Scoreable: {len(scoreable)} | "
            f"Parse failures: {len(parse_failures)} | "
            f"No reference: {len(no_reference)}"
        )

        csv_path = self.output_dir / "tool_call_accuracy.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {csv_path}")

        md_path = self.output_dir / "tool_call_accuracy.md"
        self._write_markdown(scoreable, parse_failures, no_reference, df, md_path)
        logger.info(f"Saved {md_path}")

        if not scoreable.empty:
            self._generate_plots(scoreable)

    def _write_markdown(
        self,
        scoreable: pd.DataFrame,
        parse_failures: pd.DataFrame,
        no_reference: pd.DataFrame,
        full_df: pd.DataFrame,
        path: Path,
    ) -> None:
        """Write the tool_call_accuracy.md summary."""
        lines = [
            "# Tool Call Accuracy — TaskBench TaskEval Metrics",
            "",
            "> Metrics: **n-F1** (Node), **e-F1** (Edge), **t-F1** (Tool-Param), **v-F1** (Value).",
            "> All are pure set-comparison F1 — no LLM judge used.",
            "",
            f"**Total responses scored:** {len(scoreable)}",
            f"**Parse failures (excluded from F1 means):** {len(parse_failures)}",
            f"**Prompts without reference graph (skipped):** {len(no_reference)}",
            "",
            "> [!NOTE]",
            "> t-F1 and v-F1 are 0.0 for HuggingFace/Multimedia domain prompts by",
            "> construction — their reference arguments are positional strings with no",
            "> parameter names. This is a dataset-domain limitation, not a model failure.",
            "",
        ]

        if not scoreable.empty:
            lines.append("## Overall Mean F1 by Model")
            lines.append("")
            overall = (
                scoreable.groupby("model")[self.METRICS]
                .mean()
                .round(3)
                .sort_values("n_f1", ascending=False)
            )
            overall.columns = [self.METRIC_LABELS[m] for m in overall.columns]
            lines.append(overall.to_markdown())
            lines.append("")

            all_models = full_df[full_df["has_reference"]].groupby("model")
            fail_rate = (
                all_models.apply(lambda g: (~g["parse_ok"]).mean())
                .rename("parse_failure_rate")
                .round(3)
                .reset_index()
                .sort_values("parse_failure_rate", ascending=False)
            )
            lines.append("## Parse Failure Rate by Model")
            lines.append("")
            lines.append(
                "> Proportion of responses that could not be parsed as valid JSON."
            )
            lines.append("")
            lines.append(fail_rate.to_markdown(index=False))
            lines.append("")

            lines.append("## Mean F1 by Model × Category")
            lines.append("")
            for category in sorted(scoreable["category"].unique()):
                cat_df = scoreable[scoreable["category"] == category]
                cat_pivot = (
                    cat_df.groupby("model")[self.METRICS]
                    .mean()
                    .round(3)
                    .sort_values("n_f1", ascending=False)
                )
                cat_pivot.columns = [self.METRIC_LABELS[m] for m in cat_pivot.columns]
                lines.append(f"### {category.replace('_', ' ').title()}")
                lines.append("")
                lines.append(cat_pivot.to_markdown())
                lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")

    def _generate_plots(self, scoreable: pd.DataFrame) -> None:
        """Generate bar charts and heatmap for tool-call F1 metrics."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not available — skipping plots")
            return

        plots_dir = self.output_dir / "plots"
        palette = sns.color_palette("viridis", scoreable["model"].nunique())

        fig, ax = plt.subplots(figsize=(14, 7))
        pivot = scoreable.groupby("model")[self.METRICS].mean()
        pivot.columns = [self.METRIC_LABELS[m] for m in pivot.columns]
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylabel("Mean F1")
        ax.set_title("Tool Call Accuracy — Mean F1 by Model and Metric")
        ax.set_ylim(0, 1.05)
        ax.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(plots_dir / "tool_call_by_model_metric.png", dpi=150)
        plt.close(fig)
        logger.info("Saved tool_call_by_model_metric.png")

        fig, ax = plt.subplots(figsize=(10, max(4, scoreable["model"].nunique())))
        hmap_data = scoreable.groupby("model")[self.METRICS].mean().round(3)
        hmap_data.columns = [self.METRIC_LABELS[m] for m in hmap_data.columns]
        hmap_data = hmap_data.sort_values(self.METRIC_LABELS["n_f1"], ascending=False)
        sns.heatmap(
            hmap_data,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            ax=ax,
        )
        ax.set_title("Tool Call F1 Heatmap: Model × Metric")
        plt.tight_layout()
        fig.savefig(plots_dir / "tool_call_heatmap.png", dpi=150)
        plt.close(fig)
        logger.info("Saved tool_call_heatmap.png")

        if scoreable["category"].nunique() > 1:
            fig, ax = plt.subplots(figsize=(12, 6))
            cat_pivot = (
                scoreable.groupby(["model", "category"])["n_f1"]
                .mean()
                .unstack(fill_value=0)
                .round(3)
            )
            cat_pivot.plot(kind="bar", ax=ax)
            ax.set_ylabel("Mean Node F1")
            ax.set_title("Node F1 by Model and Category")
            ax.set_ylim(0, 1.05)
            ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fig.savefig(plots_dir / "tool_call_n_f1_by_category.png", dpi=150)
            plt.close(fig)
            logger.info("Saved tool_call_n_f1_by_category.png")
