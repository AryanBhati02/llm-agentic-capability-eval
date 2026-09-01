# LLM-as-a-Judge for Task Decomposition

A modular, reproducible research framework for evaluating the **task-decomposition ability** of LLMs using an **LLM-as-a-Judge** methodology.

## What This Does

1. **Curates** representative task-decomposition prompts from [TaskBench](https://github.com/microsoft/JARVIS) and [AgentBench](https://github.com/THUDM/AgentBench)
2. **Sends identical prompts** to 8–10 LLMs via a unified async API client (GPT-5, Gemini, Claude, DeepSeek, Llama, Qwen, Mistral, etc.)
3. **Scores each response** using an LLM-as-a-Judge rubric (pointwise, anonymized, 4 criteria)
4. **Produces per-model, per-category summary tables** for the paper

> **Scope:** This framework does _not_ build or run a multi-agent system. It gives a single LLM a complex task, collects its decomposition/plan, and has a judge model score that output. Multi-agent systems are the _motivating context_ for why decomposition quality matters.

## Quick Start

### 1. Clone & Setup

```bash
git clone <repo-url>
cd research_framework

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
copy .env.example .env
# Edit .env with your actual API keys
```

### 3. Download Datasets

```bash
python scripts/download_datasets.py
```

### 4. Curate Prompts

```bash
python scripts/curate_prompts.py --n-per-category 12
```

### 5. Run Experiment

```bash
# Pilot run (Ollama local models, validates pipeline)
python scripts/run_experiment.py --pilot --dry-run   # check cost estimate first
python scripts/run_experiment.py --pilot              # run for real

# Research run (real API models)
python scripts/run_experiment.py --dry-run            # check cost estimate
python scripts/run_experiment.py                      # run for real
```

### 6. Run Judge

```bash
python scripts/run_judge.py              # judge research responses
python scripts/run_judge.py --pilot      # judge pilot responses
```

### 7. Generate Report

```bash
python scripts/generate_report.py
# Outputs: metrics/summary.csv, metrics/summary.md, metrics/plots/
```

## Adding a New Model

**One config change, zero new code** (in most cases):

1. Open `config/models.yaml`
2. Add an entry:

```yaml
  my-new-model:
    provider: openrouter          # ollama | azure | google | openrouter | anthropic
    model_id: "org/model-name"    # as expected by the provider's API
    max_tokens: 4096
    timeout_seconds: 60
    api_key_env: OPENROUTER_API_KEY
    concurrency_limit: 5
```

3. If needed, add the API key to `.env`
4. Run the experiment — the new model is automatically included

## Reproducing a Prior Run

Every experiment run saves a config snapshot (`_run_config.json`) in its output directory, including:
- Exact inference parameters (temperature, top_p)
- Config hash (SHA-256)
- Random seed for order randomization
- Timestamp

To reproduce:
1. Use the same `config/experiment.yaml` and `config/models.yaml`
2. Use the same `datasets/curated/prompts.json`
3. Use the same random seed (deterministic ordering)

## Project Structure

```
research_framework/
├── config/                     # YAML configurations
│   ├── models.yaml             # Per-model: provider, ID, max_tokens, timeout
│   ├── experiment.yaml         # Experiment-level: temperature, top_p, retry
│   └── judge.yaml              # Judge: model, rubric criteria, scale
├── datasets/
│   ├── taskbench/              # Raw TaskBench data (git-ignored, downloaded)
│   ├── agentbench/             # Raw AgentBench data (git-ignored, downloaded)
│   └── curated/
│       ├── prompts.json        # Canonical curated prompt set
│       └── curation_log.json   # Provenance and sampling stats
├── prompts/
│   └── decomposition.txt       # Shared decomposition prompt template
├── outputs/
│   ├── _pilot/                 # Pilot outputs (NEVER mixed with research data)
│   ├── responses/{model}/      # Real run responses
│   ├── judgments/               # Judge scores per response
│   └── anonymization_map.json  # Model_A → real name mapping
├── metrics/
│   ├── summary.csv             # Flat summary table
│   ├── summary.md              # Formatted markdown
│   └── plots/                  # Bar charts, heatmaps
├── src/
│   ├── llm/                    # LLM client hierarchy (5 providers)
│   ├── datasets/               # Dataset loaders + curator
│   ├── pipeline/               # Experiment runner + cost estimator
│   ├── evaluation/             # Anonymizer, judge, metrics
│   └── utils/                  # Config, logging, I/O helpers
├── tests/                      # pytest suite (all mocked, no API calls)
├── scripts/                    # Entry-point scripts
├── .env.example                # API key template
├── pyproject.toml              # Project metadata + dependencies
└── requirements.txt            # Pinned dependencies
```

## LLM Client Hierarchy

```
BaseLLMClient (abstract)
    ├── OllamaClient          — local models (pilot only)
    ├── AzureAIFoundryClient   — GPT-5 via Azure OpenAI
    ├── GoogleClient           — Gemini via AI Studio free tier
    ├── OpenRouterClient       — open-weight models (DeepSeek, Llama, Qwen, Mistral)
    └── AnthropicClient        — Claude via Anthropic API
```

Factory pattern: `LLMClient(provider="google", model="gemini-2.5-pro")` returns the right client automatically.

## Judge Rubric

Pointwise scoring, 1–5 scale per criterion:

| Criterion | What It Measures |
|---|---|
| **Completeness** | Does the decomposition cover all necessary sub-tasks? |
| **Logical Ordering** | Are steps in a feasible execution order? |
| **Correctness** | Are individual steps correct and actionable? |
| **Granularity** | Are steps at an appropriate level of detail? |

The judge sees anonymized model labels (Model_A, Model_B, ...) — never real names.

## Known Limitations

- **Self-preference bias:** The judge model (Gemini 2.5 Flash) may also be under evaluation. This is disclosed in the methods section.
- **Category coverage:** TaskBench/AgentBench don't natively cover mathematics, essay writing, or business tasks. These are documented as out-of-scope for the first result.
- **Azure for Students:** May not cover Azure OpenAI due to policy restrictions. OpenRouter is the fallback for GPT-class model access.

## Running Tests

```bash
# All tests (no API calls, all mocked)
.\venv\Scripts\python.exe -m pytest tests/ -v

# Specific test file
.\venv\Scripts\python.exe -m pytest tests/test_llm_clients.py -v
```

## License

MIT

## Citations

- **TaskBench:** Shen et al., "TaskBench: Benchmarking Large Language Models for Task Automation", ICLR 2024. Apache-2.0 license.
- **AgentBench:** Liu et al., "AgentBench: Evaluating LLMs as Agents", ICLR 2024. Apache-2.0 license.
