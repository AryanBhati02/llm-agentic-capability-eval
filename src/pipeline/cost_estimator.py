"""
Token counting and cost estimation for dry-run mode.

Estimates the total token count and approximate USD cost across all
(prompt, model) pairs before making any API calls.

Pricing table is updated periodically. Free-tier models (Ollama, Gemini Flash)
are estimated at $0.00.
"""

from __future__ import annotations

import logging
from typing import Any

from src.utils.config import ModelConfig

logger = logging.getLogger(__name__)

_PRICING_PER_M: dict[str, tuple[float, float]] = {
    "llama3.2:3b": (0.0, 0.0),
    "phi3.5": (0.0, 0.0),
    "qwen2.5:3b": (0.0, 0.0),
    "gemma2:2b": (0.0, 0.0),

    "gpt-5": (2.50, 10.00),
    "gpt-5.4-mini": (0.15, 0.60),

    "gemini-2.5-pro": (1.25, 5.00),
    "gemini-2.5-flash": (0.0, 0.0),

    "claude-opus-4-8": (15.00, 75.00),
    "claude-sonnet-4-8": (3.00, 15.00),

    "deepseek/deepseek-r1": (0.55, 2.19),
    "meta-llama/llama-4-maverick": (0.80, 2.40),
    "qwen/qwen3-235b-a22b": (0.60, 1.80),
    "mistralai/mistral-large": (2.00, 6.00),
}


def _get_pricing(model_id: str) -> tuple[float, float]:
    """Look up pricing for a model ID. Returns (input_per_M, output_per_M)."""
    for key, pricing in _PRICING_PER_M.items():
        if key in model_id.lower() or model_id.lower() in key:
            return pricing
    return (1.00, 3.00)


def count_tokens_approximate(text: str, provider: str = "") -> int:
    """Fast, dependency-light token count approximation.

    Uses tiktoken if available, otherwise falls back to character-based
    heuristic (~4 chars per token for English).
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except (ImportError, Exception):
        return max(1, len(text) // 4)


def estimate_cost(
    prompts: list[dict[str, Any]],
    models: dict[str, ModelConfig],
    estimated_output_tokens: int = 500,
) -> list[dict[str, Any]]:
    """Estimate token usage and cost for an experiment run.

    Args:
        prompts: Curated prompts list.
        models: Active models dict (model_name → ModelConfig).
        estimated_output_tokens: Expected completion length per prompt.

    Returns:
        List of dicts with per-model estimates and a total row.
    """
    results: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost_usd = 0.0

    prompt_texts = [p.get("prompt", "") for p in prompts]
    total_prompt_tokens = sum(
        count_tokens_approximate(t) for t in prompt_texts
    )

    for model_name, model_cfg in sorted(models.items()):
        n_prompts = len(prompts)
        input_tokens = total_prompt_tokens
        output_tokens = n_prompts * estimated_output_tokens

        input_price_per_m, output_price_per_m = _get_pricing(model_cfg.model_id)

        cost = (
            (input_tokens / 1_000_000) * input_price_per_m
            + (output_tokens / 1_000_000) * output_price_per_m
        )

        results.append({
            "model": model_name,
            "provider": model_cfg.provider,
            "model_id": model_cfg.model_id,
            "prompts": n_prompts,
            "est_input_tokens": input_tokens,
            "est_output_tokens": output_tokens,
            "est_total_tokens": input_tokens + output_tokens,
            "est_cost_usd": round(cost, 4),
        })

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_cost_usd += cost

    results.append({
        "model": "TOTAL",
        "provider": "-",
        "model_id": "-",
        "prompts": len(prompts) * len(models),
        "est_input_tokens": total_input_tokens,
        "est_output_tokens": total_output_tokens,
        "est_total_tokens": total_input_tokens + total_output_tokens,
        "est_cost_usd": round(total_cost_usd, 4),
    })

    return results


def print_cost_table(estimates: list[dict[str, Any]]) -> None:
    """Print the cost estimation table to stdout."""
    header = (
        f"{'Model':<22} {'Provider':<12} {'Prompts':>8} "
        f"{'Input Tok':>10} {'Output Tok':>10} {'Est Cost ($)':>12}"
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print("  EXPERIMENT COST ESTIMATE (DRY-RUN)")
    print(separator)
    print(header)
    print(separator)

    for row in estimates[:-1]:
        print(
            f"{row['model']:<22} {row['provider']:<12} {row['prompts']:>8} "
            f"{row['est_input_tokens']:>10,d} {row['est_output_tokens']:>10,d} "
            f"${row['est_cost_usd']:>11.4f}"
        )

    print(separator)
    total = estimates[-1]
    print(
        f"{total['model']:<22} {total['provider']:<12} {total['prompts']:>8} "
        f"{total['est_input_tokens']:>10,d} {total['est_output_tokens']:>10,d} "
        f"${total['est_cost_usd']:>11.4f}"
    )
    print(separator + "\n")
