# LLM-as-a-Judge Evaluation Summary

**Judge model:** gpt-5.4-mini
**Total evaluations:** 253
**Models evaluated:** 4
**Categories:** coding, general_reasoning, logical_reasoning, mathematics, planning

> [!IMPORTANT]
> The judge model may also be one of the models under evaluation.
> Self-preference bias is a known limitation disclosed in the methods section.

## Overall Scores by Model

| model       |   completeness |   correctness |   granularity |   logical_ordering |   mean_overall |
|:------------|---------------:|--------------:|--------------:|-------------------:|---------------:|
| gemma2-2b   |           3.38 |          3.48 |          3.54 |               4.03 |           3.61 |
| qwen2.5-3b  |           3.32 |          3.37 |          3.58 |               4.05 |           3.58 |
| phi3.5      |           3.55 |          3.14 |          3.58 |               3.92 |           3.55 |
| llama3.2-3b |           3.28 |          3.2  |          3.61 |               3.97 |           3.52 |

## Scores by Model × Category

### Coding

| model       |   completeness |   correctness |   granularity |   logical_ordering |   mean |
|:------------|---------------:|--------------:|--------------:|-------------------:|-------:|
| qwen2.5-3b  |           3.46 |          3.69 |          3.77 |               4.38 |   3.82 |
| gemma2-2b   |           3.25 |          4    |          3.25 |               4.42 |   3.73 |
| phi3.5      |           3.46 |          3.38 |          3.77 |               4.15 |   3.69 |
| llama3.2-3b |           3.23 |          3.31 |          3.62 |               4.08 |   3.56 |

### General Reasoning

| model       |   completeness |   correctness |   granularity |   logical_ordering |   mean |
|:------------|---------------:|--------------:|--------------:|-------------------:|-------:|
| phi3.5      |           4    |          3.38 |          3.62 |               4.23 |   3.81 |
| gemma2-2b   |           3.67 |          3.67 |          3.58 |               4.17 |   3.77 |
| llama3.2-3b |           3.54 |          3.31 |          3.69 |               4.23 |   3.69 |
| qwen2.5-3b  |           3.38 |          3.38 |          3.31 |               3.85 |   3.48 |

### Logical Reasoning

| model       |   completeness |   correctness |   granularity |   logical_ordering |   mean |
|:------------|---------------:|--------------:|--------------:|-------------------:|-------:|
| gemma2-2b   |           2.77 |          3    |          3.46 |               3.77 |   3.25 |
| qwen2.5-3b  |           2.69 |          2.77 |          3.46 |               3.62 |   3.14 |
| phi3.5      |           2.77 |          2.46 |          3.31 |               3.23 |   2.94 |
| llama3.2-3b |           2.36 |          2.64 |          3    |               3.27 |   2.82 |

### Mathematics

| model       |   completeness |   correctness |   granularity |   logical_ordering |   mean |
|:------------|---------------:|--------------:|--------------:|-------------------:|-------:|
| qwen2.5-3b  |           3.92 |          3.85 |          4.15 |               4.62 |   4.14 |
| llama3.2-3b |           3.58 |          3.25 |          3.83 |               4    |   3.66 |
| phi3.5      |           3.5  |          2.92 |          3.58 |               4    |   3.5  |
| gemma2-2b   |           3.38 |          3.15 |          3.62 |               3.69 |   3.46 |

### Planning

| model       |   completeness |   correctness |   granularity |   logical_ordering |   mean |
|:------------|---------------:|--------------:|--------------:|-------------------:|-------:|
| gemma2-2b   |           3.85 |          3.62 |          3.77 |               4.15 |   3.85 |
| phi3.5      |           4    |          3.54 |          3.62 |               4    |   3.79 |
| llama3.2-3b |           3.58 |          3.42 |          3.83 |               4.17 |   3.75 |
| qwen2.5-3b  |           3.15 |          3.15 |          3.23 |               3.77 |   3.32 |
