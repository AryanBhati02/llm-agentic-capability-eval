"""
Shared pytest fixtures for the research framework test suite.
"""

from __future__ import annotations

import pytest

from src.datasets.taskbench_loader import RawSample
from src.utils.config import (
    CriterionConfig,
    ExperimentConfig,
    JudgeConfig,
    ModelConfig,
    ModelsConfig,
)


@pytest.fixture
def sample_experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        temperature=0.2,
        top_p=1.0,
        retry_count=2,
        retry_backoff_base=1.5,
        dry_run=False,
        pilot_mode=False,
        randomize_order=True,
        random_seed=42,
    )


@pytest.fixture
def sample_judge_config() -> JudgeConfig:
    return JudgeConfig(
        judge_provider="azure",
        judge_model="gpt-5.4-mini",
        judge_max_tokens=1024,
        judge_timeout_seconds=30,
        judge_api_key_env="AZURE_OPENAI_API_KEY",
        judge_concurrency_limit=3,
        criteria=[
            CriterionConfig(
                name="completeness",
                description="Covers all steps",
                anchor_low="Misses steps",
                anchor_high="All steps",
            ),
            CriterionConfig(
                name="logical_ordering",
                description="Correct ordering",
                anchor_low="Incoherent",
                anchor_high="Optimal",
            ),
            CriterionConfig(
                name="correctness",
                description="Steps are correct",
                anchor_low="Wrong",
                anchor_high="All correct",
            ),
            CriterionConfig(
                name="granularity",
                description="Right detail level",
                anchor_low="Too vague/detailed",
                anchor_high="Well-calibrated",
            ),
        ],
        scale_min=1,
        scale_max=5,
        anonymization_seed=7,
    )


@pytest.fixture
def sample_models_config() -> ModelsConfig:
    return ModelsConfig(
        models={
            "llama3.2-3b": ModelConfig(
                provider="ollama", model_id="llama3.2:3b", max_tokens=512
            ),
            "gpt-5": ModelConfig(
                provider="azure",
                model_id="gpt-5",
                model_snapshot="2026-06-01",
                max_tokens=2048,
                api_key_env="AZURE_OPENAI_API_KEY",
            ),
            "gemini-2.5-pro": ModelConfig(
                provider="google",
                model_id="gemini-2.5-pro",
                max_tokens=2048,
                api_key_env="GOOGLE_AI_STUDIO_API_KEY",
            ),
        },
        pilot_models=["llama3.2-3b"],
    )


@pytest.fixture
def sample_raw_samples() -> list[RawSample]:
    return [
        RawSample(
            source="taskbench",
            original_id="tb_1",
            instruction="Write a Python script to scrape a webpage and save to CSV.",
            category="coding",
            difficulty="medium",
            reference={"task_steps": ["Fetch HTML", "Parse DOM", "Extract data", "Write CSV"]},
        ),
        RawSample(
            source="taskbench",
            original_id="tb_2",
            instruction="Write a Python function to download a web page and parse it into CSV format.",
            category="coding",
            difficulty="medium",
            reference={"task_steps": ["Download page", "Parse content", "Save CSV"]},
        ),
        RawSample(
            source="agentbench",
            original_id="ab_1",
            instruction="Find the cheapest flight from NYC to London departing next Friday.",
            category="planning",
            difficulty="high",
            reference={},
        ),
        RawSample(
            source="agentbench",
            original_id="ab_2",
            instruction="If all squares are rectangles and all rectangles are polygons, are all squares polygons?",
            category="logical_reasoning",
            difficulty="low",
            reference={},
        ),
    ]
