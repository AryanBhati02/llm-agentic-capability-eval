"""
Unit tests for src/evaluation/tool_call_scorer.py

Tests cover:
  - JSON parsing (three-pass strategy): clean JSON, fenced, preamble, failure
  - _extract_predicted_sets: step_id → tool resolution for edges, parameter extraction
  - _extract_reference_sets: tool-name edges, DailyLife {name,value} args, HuggingFace positional args
  - _f1: boundary cases (empty sets, perfect match, partial match, no overlap)
  - score_response: no-reference skip, parse failure, full scoring
  - score_all_responses: file-system traversal (using tmp_path)
  - ToolCallMetricsAggregator.aggregate: calls through correctly
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.tool_call_scorer import (
    _extract_predicted_sets,
    _extract_reference_sets,
    _f1,
    parse_tool_response,
    score_response,
    score_all_responses,
    ToolCallMetricsAggregator,
)


SIMPLE_GRAPH = {
    "tools": [
        {"step_id": "1", "tool": "Translation", "parameters": {"text": "hello"}},
        {"step_id": "2", "tool": "Summarization", "parameters": {}},
    ],
    "dependencies": [
        {"from": "1", "to": "2"},
    ],
}

SIMPLE_GRAPH_JSON = json.dumps(SIMPLE_GRAPH)

REFERENCE_HF = {
    "task_nodes": [
        {"task": "Translation", "arguments": ["hello world"]},
        {"task": "Summarization", "arguments": ["<node-0>"]},
    ],
    "task_links": [
        {"source": "Translation", "target": "Summarization"},
    ],
}

REFERENCE_DL = {
    "task_nodes": [
        {
            "task": "get_weather",
            "arguments": [
                {"name": "location", "value": "Paris"},
                {"name": "date", "value": "2024-01-01"},
            ],
        },
        {
            "task": "send_email",
            "arguments": [
                {"name": "to", "value": "alice@example.com"},
            ],
        },
    ],
    "task_links": [
        {"source": "get_weather", "target": "send_email"},
    ],
}


class TestParseToolResponse:
    def test_clean_json(self):
        result = parse_tool_response(SIMPLE_GRAPH_JSON)
        assert result == SIMPLE_GRAPH

    def test_fenced_json(self):
        fenced = f"```json\n{SIMPLE_GRAPH_JSON}\n```"
        result = parse_tool_response(fenced)
        assert result == SIMPLE_GRAPH

    def test_fenced_no_language_tag(self):
        fenced = f"```\n{SIMPLE_GRAPH_JSON}\n```"
        result = parse_tool_response(fenced)
        assert result == SIMPLE_GRAPH

    def test_preamble_and_trailing_text(self):
        text = f"Sure! Here is the tool graph:\n\n{SIMPLE_GRAPH_JSON}\n\nI hope that helps."
        result = parse_tool_response(text)
        assert result == SIMPLE_GRAPH

    def test_braces_in_justification_string(self):
        graph = {
            "tools": [{"step_id": "1", "tool": "Text Generation", "parameters": {"prompt": "write {title}"}}],
            "dependencies": [],
        }
        text = "Here: " + json.dumps(graph)
        result = parse_tool_response(text)
        assert result == graph

    def test_returns_none_on_garbage(self):
        result = parse_tool_response("This is not JSON at all.")
        assert result is None

    def test_returns_none_on_empty(self):
        result = parse_tool_response("")
        assert result is None

    def test_returns_none_on_non_dict_json(self):
        result = parse_tool_response("[1, 2, 3]")
        assert result is None


class TestExtractPredictedSets:
    def test_nodes(self):
        nodes, edges, t, v = _extract_predicted_sets(SIMPLE_GRAPH)
        assert nodes == frozenset({"Translation", "Summarization"})

    def test_edges_resolved_to_tool_names(self):
        nodes, edges, t, v = _extract_predicted_sets(SIMPLE_GRAPH)
        assert edges == frozenset({("Translation", "Summarization")})

    def test_parameter_t_pairs(self):
        nodes, edges, t, v = _extract_predicted_sets(SIMPLE_GRAPH)
        assert ("Translation", "text") in t

    def test_parameter_v_triples(self):
        nodes, edges, t, v = _extract_predicted_sets(SIMPLE_GRAPH)
        assert ("Translation", "text", "hello") in v

    def test_empty_parameters_excluded_from_t_v(self):
        nodes, edges, t, v = _extract_predicted_sets(SIMPLE_GRAPH)
        assert not any(triple[0] == "Summarization" for triple in v)

    def test_unknown_step_id_in_deps_skipped(self):
        graph = {
            "tools": [{"step_id": "1", "tool": "ToolA", "parameters": {}}],
            "dependencies": [{"from": "1", "to": "99"}],
        }
        nodes, edges, t, v = _extract_predicted_sets(graph)
        assert edges == frozenset()

    def test_no_tools_no_deps(self):
        nodes, edges, t, v = _extract_predicted_sets({"tools": [], "dependencies": []})
        assert not nodes
        assert not edges

    def test_dag_multiple_edges(self):
        graph = {
            "tools": [
                {"step_id": "1", "tool": "A", "parameters": {}},
                {"step_id": "2", "tool": "B", "parameters": {}},
                {"step_id": "3", "tool": "C", "parameters": {}},
            ],
            "dependencies": [
                {"from": "1", "to": "2"},
                {"from": "1", "to": "3"},
            ],
        }
        _, edges, _, _ = _extract_predicted_sets(graph)
        assert edges == frozenset({("A", "B"), ("A", "C")})


class TestExtractReferenceSets:
    def test_hf_nodes(self):
        ref_nodes, ref_edges, ref_t, ref_v = _extract_reference_sets(REFERENCE_HF)
        assert ref_nodes == frozenset({"Translation", "Summarization"})

    def test_hf_edges_are_tool_names(self):
        _, ref_edges, _, _ = _extract_reference_sets(REFERENCE_HF)
        assert ref_edges == frozenset({("Translation", "Summarization")})

    def test_hf_no_t_pairs(self):
        _, _, ref_t, ref_v = _extract_reference_sets(REFERENCE_HF)
        assert len(ref_t) == 0
        assert len(ref_v) == 0

    def test_dailylife_nodes(self):
        ref_nodes, _, _, _ = _extract_reference_sets(REFERENCE_DL)
        assert ref_nodes == frozenset({"get_weather", "send_email"})

    def test_dailylife_edges(self):
        _, ref_edges, _, _ = _extract_reference_sets(REFERENCE_DL)
        assert ref_edges == frozenset({("get_weather", "send_email")})

    def test_dailylife_t_pairs(self):
        _, _, ref_t, _ = _extract_reference_sets(REFERENCE_DL)
        assert ("get_weather", "location") in ref_t
        assert ("get_weather", "date") in ref_t
        assert ("send_email", "to") in ref_t

    def test_dailylife_v_triples(self):
        _, _, _, ref_v = _extract_reference_sets(REFERENCE_DL)
        assert ("get_weather", "location", "Paris") in ref_v
        assert ("send_email", "to", "alice@example.com") in ref_v

    def test_empty_reference(self):
        ref_nodes, ref_edges, ref_t, ref_v = _extract_reference_sets({})
        assert not ref_nodes
        assert not ref_edges

    def test_mixed_args_only_dicts_counted(self):
        ref = {
            "task_nodes": [
                {"task": "SomeTool", "arguments": ["plain_str", {"name": "p", "value": "v"}]},
            ],
            "task_links": [],
        }
        _, _, ref_t, ref_v = _extract_reference_sets(ref)
        assert ("SomeTool", "p") in ref_t
        assert ("SomeTool", "p", "v") in ref_v
        assert len(ref_t) == 1


class TestF1:
    def test_perfect_match(self):
        s = frozenset({"A", "B"})
        assert _f1(s, s) == 1.0

    def test_no_overlap(self):
        assert _f1(frozenset({"A"}), frozenset({"B"})) == 0.0

    def test_partial_match(self):
        pred = frozenset({"A", "B"})
        ref = frozenset({"A", "C"})
        assert abs(_f1(pred, ref) - 0.5) < 1e-9

    def test_both_empty(self):
        assert _f1(frozenset(), frozenset()) == 1.0

    def test_pred_empty_ref_nonempty(self):
        assert _f1(frozenset(), frozenset({"A"})) == 0.0

    def test_pred_nonempty_ref_empty(self):
        assert _f1(frozenset({"A"}), frozenset()) == 0.0

    def test_asymmetric_partial(self):
        pred = frozenset({"A", "B", "C"})
        ref = frozenset({"A", "B"})
        expected = 2 * (2/3) * 1.0 / (2/3 + 1.0)
        assert abs(_f1(pred, ref) - expected) < 1e-9


class TestScoreResponse:
    def _make_response(self, graph_dict: dict, model: str = "test_model") -> dict:
        return {
            "prompt_id": 1,
            "model": model,
            "response": json.dumps(graph_dict),
        }

    def _make_prompt(self, reference: dict) -> dict:
        return {
            "id": 1,
            "category": "planning",
            "difficulty": "medium",
            "reference": reference,
        }

    def test_no_reference_returns_has_reference_false(self):
        result = score_response(
            self._make_response(SIMPLE_GRAPH),
            self._make_prompt({}),
        )
        assert result.has_reference is False
        assert result.parse_ok is False
        assert result.n_f1 is None

    def test_parse_failure_returns_parse_ok_false(self):
        bad_response = {"prompt_id": 1, "model": "m", "response": "not json"}
        result = score_response(bad_response, self._make_prompt(REFERENCE_HF))
        assert result.has_reference is True
        assert result.parse_ok is False
        assert result.n_f1 is None

    def test_perfect_hf_score(self):
        result = score_response(
            self._make_response(SIMPLE_GRAPH),
            self._make_prompt(REFERENCE_HF),
        )
        assert result.has_reference is True
        assert result.parse_ok is True
        assert result.n_f1 == 1.0
        assert result.e_f1 == 1.0
        assert result.t_f1 == 0.0
        assert result.v_f1 == 0.0

    def test_wrong_tools_lowers_n_f1(self):
        wrong_graph = {
            "tools": [
                {"step_id": "1", "tool": "WrongTool", "parameters": {}},
            ],
            "dependencies": [],
        }
        result = score_response(
            self._make_response(wrong_graph),
            self._make_prompt(REFERENCE_HF),
        )
        assert result.n_f1 == 0.0

    def test_dailylife_t_and_v_scored(self):
        dl_graph = {
            "tools": [
                {"step_id": "1", "tool": "get_weather", "parameters": {"location": "Paris", "date": "2024-01-01"}},
                {"step_id": "2", "tool": "send_email", "parameters": {"to": "alice@example.com"}},
            ],
            "dependencies": [{"from": "1", "to": "2"}],
        }
        result = score_response(
            self._make_response(dl_graph),
            self._make_prompt(REFERENCE_DL),
        )
        assert result.n_f1 == 1.0
        assert result.e_f1 == 1.0
        assert result.t_f1 == 1.0
        assert result.v_f1 == 1.0


class TestScoreAllResponses:
    def test_loads_and_scores_files(self, tmp_path):
        prompts = [
            {"id": 1, "category": "planning", "difficulty": "medium", "reference": REFERENCE_HF},
            {"id": 2, "category": "general_reasoning", "difficulty": "low", "reference": {}},
        ]
        prompts_file = tmp_path / "prompts.json"
        prompts_file.write_text(json.dumps(prompts), encoding="utf-8")

        model_dir = tmp_path / "tool_calls" / "gpt_4"
        model_dir.mkdir(parents=True)
        (model_dir / "prompt_1.json").write_text(
            json.dumps({"prompt_id": 1, "model": "gpt-4", "response": json.dumps(SIMPLE_GRAPH)}),
            encoding="utf-8",
        )
        (model_dir / "prompt_2.json").write_text(
            json.dumps({"prompt_id": 2, "model": "gpt-4", "response": json.dumps(SIMPLE_GRAPH)}),
            encoding="utf-8",
        )

        results = score_all_responses(tmp_path / "tool_calls", prompts_file)
        assert len(results) == 2

        by_id = {r["prompt_id"]: r for r in results}
        assert by_id[1]["has_reference"] is True
        assert by_id[1]["parse_ok"] is True
        assert by_id[1]["n_f1"] == 1.0
        assert by_id[2]["has_reference"] is False
        assert by_id[2]["n_f1"] is None

    def test_missing_dir_returns_empty(self, tmp_path):
        prompts_file = tmp_path / "prompts.json"
        prompts_file.write_text("[]", encoding="utf-8")
        results = score_all_responses(tmp_path / "nonexistent", prompts_file)
        assert results == []


class TestToolCallMetricsAggregator:
    def test_aggregate_returns_dataframe(self, tmp_path):
        prompts = [
            {"id": 1, "category": "planning", "difficulty": "medium", "reference": REFERENCE_HF},
        ]
        prompts_file = tmp_path / "prompts.json"
        prompts_file.write_text(json.dumps(prompts), encoding="utf-8")

        model_dir = tmp_path / "tool_calls" / "model_a"
        model_dir.mkdir(parents=True)
        (model_dir / "prompt_1.json").write_text(
            json.dumps({"prompt_id": 1, "model": "model_a", "response": json.dumps(SIMPLE_GRAPH)}),
            encoding="utf-8",
        )

        agg = ToolCallMetricsAggregator(
            tool_calls_dir=tmp_path / "tool_calls",
            prompts_path=prompts_file,
            output_dir=tmp_path / "metrics",
        )
        df = agg.aggregate()
        assert len(df) == 1
        assert "n_f1" in df.columns
        assert df.iloc[0]["n_f1"] == 1.0

    def test_generate_summary_creates_files(self, tmp_path):
        prompts = [
            {"id": 1, "category": "planning", "difficulty": "medium", "reference": REFERENCE_HF},
        ]
        prompts_file = tmp_path / "prompts.json"
        prompts_file.write_text(json.dumps(prompts), encoding="utf-8")

        model_dir = tmp_path / "tool_calls" / "model_a"
        model_dir.mkdir(parents=True)
        (model_dir / "prompt_1.json").write_text(
            json.dumps({"prompt_id": 1, "model": "model_a", "response": json.dumps(SIMPLE_GRAPH)}),
            encoding="utf-8",
        )

        output_dir = tmp_path / "metrics"
        agg = ToolCallMetricsAggregator(
            tool_calls_dir=tmp_path / "tool_calls",
            prompts_path=prompts_file,
            output_dir=output_dir,
        )
        agg.generate_summary()

        assert (output_dir / "tool_call_accuracy.csv").exists()
        assert (output_dir / "tool_call_accuracy.md").exists()
