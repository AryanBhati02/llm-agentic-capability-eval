"""
Tests for LLM-as-a-Judge prompt building, response parsing, and anonymization.
"""

from __future__ import annotations

import json
import pytest

from src.evaluation.anonymizer import (
    anonymize_model_name,
    generate_anonymization_map,
    invert_anonymization_map,
)
from src.evaluation.judge import (
    _extract_json_object,
    _strip_markdown_fence,
    _validate_scores,
    build_judge_prompt,
    parse_judge_response,
)
from src.utils.config import CriterionConfig, JudgeConfig


class TestAnonymizer:
    """Tests for blind model anonymization."""

    def test_generate_map_deterministic(self):
        models = ["gpt-5", "gemini-2.5-pro", "claude-opus-4-8"]
        map1 = generate_anonymization_map(models, seed=42)
        map2 = generate_anonymization_map(models, seed=42)
        assert map1 == map2

    def test_generate_map_different_seeds(self):
        models = [f"model_{i}" for i in range(10)]
        map1 = generate_anonymization_map(models, seed=1)
        map2 = generate_anonymization_map(models, seed=99)
        assert map1 != map2

    def test_generate_map_all_models_present(self):
        models = ["a", "b", "c", "d"]
        mapping = generate_anonymization_map(models, seed=7)
        assert set(mapping.values()) == set(models)
        assert len(mapping) == 4

    def test_labels_format(self):
        models = ["m1", "m2", "m3"]
        mapping = generate_anonymization_map(models)
        for label in mapping.keys():
            assert label.startswith("Model_")

    def test_anonymize_and_invert(self):
        models = ["gpt-5", "gemini-2.5-pro"]
        mapping = generate_anonymization_map(models, seed=7)

        anon_gpt = anonymize_model_name("gpt-5", mapping)
        assert anon_gpt in mapping
        assert mapping[anon_gpt] == "gpt-5"

    def test_anonymize_unknown_model(self):
        mapping = {"Model_A": "gpt-5"}
        assert anonymize_model_name("unknown-model", mapping) == "Model_Unknown"


class TestJudgePromptBuilding:
    """Tests for judge prompt construction."""

    def test_prompt_contains_criteria(self, sample_judge_config: JudgeConfig):
        prompt = build_judge_prompt(
            task_prompt="Write a sorting function",
            response_text="Step 1: define function\nStep 2: implement quicksort",
            anonymized_model="Model_A",
            judge_config=sample_judge_config,
        )

        assert "completeness" in prompt
        assert "logical_ordering" in prompt
        assert "correctness" in prompt
        assert "granularity" in prompt
        assert "Model_A" in prompt
        assert "Write a sorting function" in prompt
        assert "1-5" in prompt

    def test_prompt_includes_reference_when_present(self, sample_judge_config: JudgeConfig):
        reference = {"task_steps": ["Step A", "Step B"]}
        prompt = build_judge_prompt(
            task_prompt="Task X",
            response_text="My steps",
            anonymized_model="Model_B",
            judge_config=sample_judge_config,
            reference=reference,
        )
        assert "Reference Decomposition" in prompt
        assert "Step A" in prompt

    def test_prompt_omits_reference_when_absent(self, sample_judge_config: JudgeConfig):
        prompt = build_judge_prompt(
            task_prompt="Task X",
            response_text="My steps",
            anonymized_model="Model_B",
            judge_config=sample_judge_config,
            reference=None,
        )
        assert "Reference Decomposition" not in prompt


class TestJudgeResponseParsing:
    """Tests for JSON response extraction and validation."""

    def test_parse_clean_json(self, sample_judge_config: JudgeConfig):
        raw = json.dumps({
            "completeness": 5,
            "logical_ordering": 4,
            "correctness": 5,
            "granularity": 4,
            "justification": "Excellent decomposition.",
        })
        result = parse_judge_response(raw, sample_judge_config)
        assert result is not None
        assert result["scores"]["completeness"] == 5
        assert result["scores"]["logical_ordering"] == 4
        assert result["justification"] == "Excellent decomposition."

    def test_parse_json_with_markdown_fence(self, sample_judge_config: JudgeConfig):
        raw = """Here is my evaluation:
```json
{
    "completeness": 4,
    "logical_ordering": 4,
    "correctness": 3,
    "granularity": 4,
    "justification": "Good but step 3 is slightly vague."
}
```
Hope this helps!"""
        result = parse_judge_response(raw, sample_judge_config)
        assert result is not None
        assert result["scores"]["correctness"] == 3

    def test_parse_json_with_preamble_and_trailing_text(
        self, sample_judge_config: JudgeConfig
    ):
        raw = """I evaluated the model decomposition:
{"completeness": 3, "logical_ordering": 5, "correctness": 4, "granularity": 2, "justification": "Too detailed."}
Thank you."""
        result = parse_judge_response(raw, sample_judge_config)
        assert result is not None
        assert result["scores"]["granularity"] == 2

    def test_parse_missing_criterion_fails(self, sample_judge_config: JudgeConfig):
        raw = json.dumps({
            "completeness": 4,
            "logical_ordering": 4,
            "justification": "Missing criteria.",
        })
        result = parse_judge_response(raw, sample_judge_config)
        assert result is None

    def test_parse_unparseable_text_fails(self, sample_judge_config: JudgeConfig):
        raw = "I think the model did a great job! Score: 5/5."
        result = parse_judge_response(raw, sample_judge_config)
        assert result is None

    def test_parse_clamps_out_of_range_scores(self, sample_judge_config: JudgeConfig):
        raw = json.dumps({
            "completeness": 6,
            "logical_ordering": 0,
            "correctness": 5,
            "granularity": 3,
            "justification": "Scores out of bounds.",
        })
        result = parse_judge_response(raw, sample_judge_config)
        assert result is not None
        assert result["scores"]["completeness"] == 5
        assert result["scores"]["logical_ordering"] == 1


class TestBraceBalancedExtractor:
    """Tests for _extract_json_object: strings with braces, escapes, nesting."""

    def test_handles_braces_inside_strings(self):
        text = '{"a": 1, "justification": "use {open editor, import media} then save"}'
        result = _extract_json_object(text)
        assert result == text

    def test_handles_escaped_quotes_inside_strings(self):
        text = r'{"a": 1, "justification": "he said \"hello {world}\" here"}'
        result = _extract_json_object(text)
        assert result == text

    def test_handles_preamble_and_trailing_text(self):
        inner = '{"completeness": 5, "justification": "all good"}'
        full = f"Preamble text here.\n\n{inner}\n\nTrailing notes."
        result = _extract_json_object(full)
        assert result == inner

    def test_returns_none_when_no_braces(self):
        assert _extract_json_object("no json here at all") is None

    def test_returns_none_for_unclosed_brace(self):
        assert _extract_json_object('{"unclosed": 1') is None

    def test_strip_markdown_fence_json_tag(self):
        fenced = '```json\n{"a": 1}\n```'
        assert _strip_markdown_fence(fenced) == '{"a": 1}'

    def test_strip_markdown_fence_no_tag(self):
        fenced = '```\n{"a": 1}\n```'
        assert _strip_markdown_fence(fenced) == '{"a": 1}'

    def test_strip_markdown_fence_none_when_no_fence(self):
        assert _strip_markdown_fence('{"a": 1}') is None
