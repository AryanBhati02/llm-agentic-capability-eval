"""
Configuration schemas and loaders using Pydantic v2.

Defines typed dataclasses for:
  - ExperimentConfig: inference parameters, retry policy, execution flags
  - ModelConfig: provider-specific settings, timeouts, concurrency limits
  - JudgeConfig: judge model settings, rubric criteria, scale bounds
  - ModelsConfig: top-level container mapping model names to ModelConfigs

All loaders accept a YAML file path or dict and validate with Pydantic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class ExperimentConfig(BaseModel):
    """Inference parameters applied identically to every model call.

    Per-model overrides here are forbidden by design to guarantee
    equal-parameter comparison.
    """

    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    retry_count: int = Field(default=3, ge=0, le=10)
    retry_backoff_base: float = Field(default=2.0, ge=1.0, le=10.0)
    dry_run: bool = Field(default=False)
    pilot_mode: bool = Field(default=False)
    randomize_order: bool = Field(default=True)
    random_seed: int = Field(default=42)
    output_dir: str = Field(default="outputs/responses")
    pilot_output_dir: str = Field(default="outputs/_pilot")


class ModelConfig(BaseModel):
    """Configuration for a single model within the LLM client hierarchy."""

    provider: str
    model_id: str
    max_tokens: int = Field(default=4096, ge=1)
    timeout_seconds: int = Field(default=60, ge=1)
    api_key_env: str | None = None
    concurrency_limit: int = Field(default=5, ge=1)
    model_snapshot: str | None = None
    base_url: str | None = None
    endpoint: str | None = None

    def resolve_api_key(self) -> str | None:
        """Resolve the API key from the environment variable named in config."""
        if self.api_key_env is None:
            return None
        key = os.getenv(self.api_key_env)
        if not key:
            raise ValueError(
                f"Environment variable '{self.api_key_env}' is not set. "
                f"Please set it in your .env file or environment."
            )
        return key

    def resolve_foundry_base_url(self) -> str | None:
        """Resolve the Azure AI Foundry base URL for the Responses API.

        Normalises the endpoint to ``https://<host>/openai/v1`` which is
        what ``openai.AsyncOpenAI`` expects.
        """
        raw = self.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not raw:
            return None

        raw = raw.split("?")[0].rstrip("/")

        if "/openai/v1" in raw:
            idx = raw.index("/openai/v1") + len("/openai/v1")
            return raw[:idx]

        return f"{raw}/openai/v1"


class CriterionConfig(BaseModel):
    """A single scoring criterion in the judge rubric."""

    name: str
    description: str
    anchor_low: str
    anchor_high: str


class JudgeConfig(BaseModel):
    """Configuration for the LLM-as-a-Judge evaluation pass."""

    judge_provider: str = Field(default="azure")
    judge_model: str = Field(default="gpt-5.4-mini")
    judge_max_tokens: int = Field(default=1024, ge=1)
    judge_timeout_seconds: int = Field(default=60, ge=1)
    judge_api_key_env: str = Field(default="AZURE_OPENAI_API_KEY")
    judge_concurrency_limit: int = Field(default=5, ge=1)
    criteria: list[CriterionConfig] = Field(default_factory=list)
    scale_min: int = Field(default=1)
    scale_max: int = Field(default=5)
    anonymization_seed: int = Field(default=7)
    judgments_dir: str = Field(default="outputs/judgments")
    anonymization_map_path: str = Field(default="outputs/anonymization_map.json")


class ModelsConfig(BaseModel):
    """Top-level container for models.yaml."""

    models: dict[str, ModelConfig] = Field(default_factory=dict)
    pilot_models: list[str] = Field(default_factory=list)

    def get_active_models(self, pilot_mode: bool = False) -> dict[str, ModelConfig]:
        """Return the dict of models that should be executed.

        If pilot_mode is True, returns only models listed in pilot_models.
        Otherwise, returns all research models (excluding pilot models).
        """
        if pilot_mode:
            return {
                name: cfg
                for name, cfg in self.models.items()
                if name in self.pilot_models
            }
        return {
            name: cfg
            for name, cfg in self.models.items()
            if name not in self.pilot_models
        }


def _load_yaml_file(path: str | Path) -> dict[str, Any]:
    """Read a YAML file and return the parsed dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_experiment_config(path: str | Path = "config/experiment.yaml") -> ExperimentConfig:
    """Load and validate ExperimentConfig from a YAML file."""
    data = _load_yaml_file(path)
    return ExperimentConfig.model_validate(data)


def load_models_config(path: str | Path = "config/models.yaml") -> ModelsConfig:
    """Load and validate ModelsConfig from a YAML file."""
    data = _load_yaml_file(path)
    return ModelsConfig.model_validate(data)


def load_judge_config(path: str | Path = "config/judge.yaml") -> JudgeConfig:
    """Load and validate JudgeConfig from a YAML file."""
    data = _load_yaml_file(path)
    return JudgeConfig.model_validate(data)
