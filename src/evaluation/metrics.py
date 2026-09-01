"""
Metrics aggregation and visualization.

Loads judge scores, de-anonymizes them, and produces:
  - summary.csv: flat table (model × category × criterion → mean, std, n)
  - summary.md: formatted markdown table for the paper
  - plots/: bar charts per criterion and a heatmap (model × category)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.anonymizer import load_anonymization_map
from src.utils.io import ensure_dir, load_json, save_json

logger = logging.getLogger(__name__)


class MetricsAggregator:
    """Aggregate judge scores into summary tables and plots.

    Args:
        judgments_dir: Directory containing judgment JSON files.
        anonymization_map_path: Path to the anonymization map JSON.
        prompts_path: Path to curated prompts JSON (for category info).
        output_dir: Directory for CSV, markdown, and plots.
    """

    def __init__(
        self,
        judgments_dir: str | Path,
        anonymization_map_path: str | Path,
        prompts_path: str | Path,
        output_dir: str | Path = "metrics",
    ) -> None:
        self.judgments_dir = Path(judgments_dir)
        self.output_dir = Path(output_dir)
        self.anon_map = load_anonymization_map(anonymization_map_path)
        self.prompts = {p["id"]: p for p in load_json(prompts_path)}

    def aggregate(self) -> pd.DataFrame:
        """Load all judgments and produce a flat DataFrame.

        Returns:
            DataFrame with columns: model, category, criterion, score,
            prompt_id, justification.
        """
        records: list[dict[str, Any]] = []

        for json_file in sorted(self.judgments_dir.glob("*.json")):
            try:
                judgment = load_json(json_file)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                continue

            anon_label = judgment.get("anonymized_model", "")
            real_model = self.anon_map.get(anon_label, anon_label)
            prompt_id = judgment.get("prompt_id")
            prompt_data = self.prompts.get(prompt_id, {})
            category = prompt_data.get("category", "unknown")

            scores = judgment.get("scores", {})
            for criterion, score in scores.items():
                records.append({
                    "model": real_model,
                    "anonymized_model": anon_label,
                    "category": category,
                    "criterion": criterion,
                    "score": score,
                    "prompt_id": prompt_id,
                    "justification": judgment.get("justification", ""),
                })

        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} score records from {len(records)} entries")
        return df

    def generate_summary(self, df: pd.DataFrame | None = None) -> None:
        """Generate all summary outputs: CSV, markdown, and plots.

        Args:
            df: Pre-computed DataFrame. If None, calls ``aggregate()``.
        """
        if df is None:
            df = self.aggregate()

        if df.empty:
            logger.warning("No scores to summarize.")
            return

        ensure_dir(self.output_dir)
        ensure_dir(self.output_dir / "plots")

        summary_df = self._compute_summary_table(df)
        csv_path = self.output_dir / "summary.csv"
        summary_df.to_csv(csv_path, index=False)
        logger.info(f"Summary CSV saved to {csv_path}")

        md_path = self.output_dir / "summary.md"
        self._write_markdown_summary(summary_df, df, md_path)
        logger.info(f"Summary markdown saved to {md_path}")

        self._generate_plots(df)

    def _compute_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute mean/std/count per model × category × criterion."""
        grouped = (
            df.groupby(["model", "category", "criterion"])["score"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        grouped.columns = ["model", "category", "criterion", "mean", "std", "n"]
        grouped["mean"] = grouped["mean"].round(2)
        grouped["std"] = grouped["std"].round(2)
        return grouped

    def _write_markdown_summary(
        self,
        summary_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        path: Path,
    ) -> None:
        """Write a formatted markdown summary."""
        lines = [
            "# LLM-as-a-Judge Evaluation Summary",
            "",
            f"**Judge model:** {self._get_judge_model()}",
            f"**Total evaluations:** {len(raw_df) // len(raw_df['criterion'].unique()) if not raw_df.empty else 0}",
            f"**Models evaluated:** {raw_df['model'].nunique() if not raw_df.empty else 0}",
            f"**Categories:** {', '.join(sorted(raw_df['category'].unique())) if not raw_df.empty else 'N/A'}",
            "",
            "> [!IMPORTANT]",
            "> The judge model may also be one of the models under evaluation.",
            "> Self-preference bias is a known limitation disclosed in the methods section.",
            "",
        ]

        if not raw_df.empty:
            lines.append("## Overall Scores by Model")
            lines.append("")
            overall = (
                raw_df.groupby(["model", "criterion"])["score"]
                .mean()
                .unstack(fill_value=0)
                .round(2)
            )
            overall["mean_overall"] = overall.mean(axis=1).round(2)
            overall = overall.sort_values("mean_overall", ascending=False)
            lines.append(overall.to_markdown())
            lines.append("")

            lines.append("## Scores by Model × Category")
            lines.append("")
            for category in sorted(raw_df["category"].unique()):
                cat_df = raw_df[raw_df["category"] == category]
                cat_pivot = (
                    cat_df.groupby(["model", "criterion"])["score"]
                    .mean()
                    .unstack(fill_value=0)
                    .round(2)
                )
                if not cat_pivot.empty:
                    cat_pivot["mean"] = cat_pivot.mean(axis=1).round(2)
                    cat_pivot = cat_pivot.sort_values("mean", ascending=False)
                    lines.append(f"### {category.replace('_', ' ').title()}")
                    lines.append("")
                    lines.append(cat_pivot.to_markdown())
                    lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")

    def _generate_plots(self, df: pd.DataFrame) -> None:
        """Generate bar charts and heatmap."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not available, skipping plots")
            return

        plots_dir = self.output_dir / "plots"

        fig, ax = plt.subplots(figsize=(12, 6))
        overall = df.groupby("model")["score"].mean().sort_values(ascending=True)
        overall.plot(kind="barh", ax=ax, color=sns.color_palette("viridis", len(overall)))
        ax.set_xlabel("Mean Score (1-5)")
        ax.set_title("Overall Mean Score by Model")
        ax.set_xlim(1, 5)
        plt.tight_layout()
        fig.savefig(plots_dir / "overall_by_model.png", dpi=150)
        plt.close(fig)
        logger.info("Saved overall_by_model.png")

        fig, ax = plt.subplots(figsize=(14, 7))
        pivot = df.groupby(["model", "criterion"])["score"].mean().unstack()
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylabel("Mean Score (1-5)")
        ax.set_title("Mean Score by Model and Criterion")
        ax.set_ylim(1, 5)
        ax.legend(title="Criterion", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(plots_dir / "by_model_criterion.png", dpi=150)
        plt.close(fig)
        logger.info("Saved by_model_criterion.png")

        fig, ax = plt.subplots(figsize=(12, 8))
        heatmap_data = df.groupby(["model", "category"])["score"].mean().unstack()
        if not heatmap_data.empty:
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=".2f",
                cmap="YlOrRd",
                vmin=1,
                vmax=5,
                ax=ax,
            )
            ax.set_title("Mean Score: Model × Category")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fig.savefig(plots_dir / "heatmap_model_category.png", dpi=150)
        plt.close(fig)
        logger.info("Saved heatmap_model_category.png")

    def _get_judge_model(self) -> str:
        """Extract judge model name from the first judgment file."""
        for json_file in self.judgments_dir.glob("*.json"):
            try:
                data = load_json(json_file)
                return data.get("judge_model", "unknown")
            except Exception:
                continue
        return "unknown"
