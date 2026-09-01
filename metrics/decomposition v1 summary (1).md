# LLM-as-a-Judge Evaluation Summary

**Judge model:** gpt-5.4-mini
**Total evaluations:** 96
**Models evaluated:** 4
**Categories:** general_reasoning, planning

> [!IMPORTANT]
> The judge model may also be one of the models under evaluation.
> Self-preference bias is a known limitation disclosed in the methods section.

## Overall Scores by Model

| model       |   completeness |   correctness |   granularity |   logical_ordering |   mean_overall |
|:------------|---------------:|--------------:|--------------:|-------------------:|---------------:|
| gemma2-2b   |           3.62 |          3.21 |          3.79 |               4.33 |           3.74 |
| phi3.5      |           3.38 |          2.83 |          3.21 |               3.75 |           3.29 |
| qwen2.5-3b  |           3.38 |          2.71 |          3.17 |               3.71 |           3.24 |
| llama3.2-3b |           3.04 |          2.67 |          3.21 |               3.46 |           3.1  |

## Scores by Model × Category

### General Reasoning

| model       |   completeness |   correctness |   granularity |   logical_ordering |   mean |
|:------------|---------------:|--------------:|--------------:|-------------------:|-------:|
| gemma2-2b   |           3.5  |          3    |          3.83 |               4.25 |   3.64 |
| phi3.5      |           3.08 |          2.75 |          3.25 |               3.67 |   3.19 |
| llama3.2-3b |           2.75 |          2.58 |          3.17 |               3.5  |   3    |
| qwen2.5-3b  |           3.17 |          2.33 |          2.83 |               3.42 |   2.94 |

### Planning

| model       |   completeness |   correctness |   granularity |   logical_ordering |   mean |
|:------------|---------------:|--------------:|--------------:|-------------------:|-------:|
| gemma2-2b   |           3.75 |          3.42 |          3.75 |               4.42 |   3.84 |
| qwen2.5-3b  |           3.58 |          3.08 |          3.5  |               4    |   3.54 |
| phi3.5      |           3.67 |          2.92 |          3.17 |               3.83 |   3.4  |
| llama3.2-3b |           3.33 |          2.75 |          3.25 |               3.42 |   3.19 |
