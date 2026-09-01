# Tool Call Accuracy — TaskBench TaskEval Metrics

> Metrics: **n-F1** (Node), **e-F1** (Edge), **t-F1** (Tool-Param), **v-F1** (Value).
> All are pure set-comparison F1 — no LLM judge used.

**Total responses scored:** 50
**Parse failures (excluded from F1 means):** 2
**Prompts without reference graph (skipped):** 44

> [!NOTE]
> t-F1 and v-F1 are 0.0 for HuggingFace/Multimedia domain prompts by
> construction — their reference arguments are positional strings with no
> parameter names. This is a dataset-domain limitation, not a model failure.

## Overall Mean F1 by Model

| model       |   Node F1 |   Edge F1 |   Tool-Param F1 |   Value F1 |
|:------------|----------:|----------:|----------------:|-----------:|
| qwen2.5-3b  |     0.428 |     0.154 |           0.144 |      0.144 |
| llama3.2-3b |     0.344 |     0.046 |           0.16  |      0.117 |
| phi3.5      |     0.278 |     0.097 |           0.028 |      0     |
| gemma2-2b   |     0.204 |     0.153 |           0     |      0     |

## Parse Failure Rate by Model

> Proportion of responses that could not be parsed as valid JSON.

| model       |   parse_failure_rate |
|:------------|---------------------:|
| gemma2-2b   |                0.077 |
| phi3.5      |                0.077 |
| llama3.2-3b |                0     |
| qwen2.5-3b  |                0     |

## Mean F1 by Model × Category

### General Reasoning

| model       |   Node F1 |   Edge F1 |   Tool-Param F1 |   Value F1 |
|:------------|----------:|----------:|----------------:|-----------:|
| llama3.2-3b |     0.492 |     0.1   |           0.346 |      0.254 |
| qwen2.5-3b  |     0.362 |     0.167 |           0.312 |      0.312 |
| gemma2-2b   |     0.157 |     0.067 |           0     |      0     |
| phi3.5      |     0.151 |     0     |           0.067 |      0     |

### Planning

| model       |   Node F1 |   Edge F1 |   Tool-Param F1 |   Value F1 |
|:------------|----------:|----------:|----------------:|-----------:|
| qwen2.5-3b  |     0.486 |     0.143 |               0 |          0 |
| phi3.5      |     0.369 |     0.167 |               0 |          0 |
| gemma2-2b   |     0.238 |     0.214 |               0 |          0 |
| llama3.2-3b |     0.217 |     0     |               0 |          0 |
