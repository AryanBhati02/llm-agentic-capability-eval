# Function Calling Accuracy — Single-Tool-Call Evaluation

> **Tool Selection Accuracy** — % of prompts where the predicted tool matches the correct tool (exact, case-insensitive).
> **Full Accuracy** — % of prompts where the tool AND every parameter match (normalised: case-insensitive, trimmed, numeric coercion).

**Total responses scored:** 196
**Parse failures (excluded from accuracy):** 4
**Total prompts in dataset:** 50

## Overall Accuracy by Model

| model       |   Prompts Scored | Tool Selection Acc   | Full Acc   |
|:------------|-----------------:|:---------------------|:-----------|
| phi3.5      |               48 | 100.0%               | 77.1%      |
| qwen2.5-3b  |               50 | 98.0%                | 74.0%      |
| llama3.2-3b |               50 | 96.0%                | 78.0%      |
| gemma2-2b   |               48 | 95.8%                | 62.5%      |

## Parse Failure Rate by Model

> Proportion of responses that could not be parsed as valid JSON.

| model       |   Total |   Failures | Failure Rate   |
|:------------|--------:|-----------:|:---------------|
| gemma2-2b   |      50 |          2 | 4.0%           |
| phi3.5      |      50 |          2 | 4.0%           |
| llama3.2-3b |      50 |          0 | 0.0%           |
| qwen2.5-3b  |      50 |          0 | 0.0%           |

## Per-Tool Accuracy Breakdown (Tool × Model)

> Each cell shows **correct / total** attempts. Rows = tools (sorted alphabetically), columns = models.

### Tool Selection Accuracy

| Tool | gemma2-2b | llama3.2-3b | phi3.5 | qwen2.5-3b |
|:---|:---:|:---:|:---:|:---:|
| calculate | 3/3 | 3/3 | 3/3 | 3/3 |
| convert_currency | 3/3 | 3/3 | 3/3 | 3/3 |
| convert_units | 1/2 | 2/2 | 2/2 | 2/2 |
| define_word | 2/2 | 2/2 | 2/2 | 2/2 |
| detect_language | 2/2 | 1/2 | 1/1 | 2/2 |
| generate_qr_code | 2/2 | 2/2 | 2/2 | 2/2 |
| get_directions | 3/3 | 3/3 | 3/3 | 3/3 |
| get_flight_status | 2/2 | 2/2 | 2/2 | 2/2 |
| get_news | 3/3 | 3/3 | 3/3 | 3/3 |
| get_stock_price | 3/3 | 3/3 | 3/3 | 3/3 |
| get_time_in_timezone | 2/2 | 2/2 | 2/2 | 2/2 |
| get_weather | 4/4 | 4/4 | 4/4 | 4/4 |
| search_hotel | 2/2 | 2/2 | 2/2 | 2/2 |
| search_movie | 2/3 | 2/3 | 3/3 | 2/3 |
| search_recipe | 3/3 | 3/3 | 3/3 | 3/3 |
| search_restaurant | 2/2 | 2/2 | 2/2 | 2/2 |
| send_email | 1/1 | 2/2 | 1/1 | 2/2 |
| set_reminder | 2/2 | 2/2 | 2/2 | 2/2 |
| summarize_text | 1/1 | 2/2 | 2/2 | 2/2 |
| translate_text | 3/3 | 3/3 | 3/3 | 3/3 |

### Full Accuracy (Tool + Parameters)

| Tool | gemma2-2b | llama3.2-3b | phi3.5 | qwen2.5-3b |
|:---|:---:|:---:|:---:|:---:|
| calculate | 0/3 | 2/3 | 1/3 | 2/3 |
| convert_currency | 3/3 | 3/3 | 2/3 | 2/3 |
| convert_units | 0/2 | 2/2 | 2/2 | 2/2 |
| define_word | 2/2 | 2/2 | 2/2 | 2/2 |
| detect_language | 2/2 | 1/2 | 1/1 | 2/2 |
| generate_qr_code | 2/2 | 2/2 | 2/2 | 2/2 |
| get_directions | 2/3 | 3/3 | 2/3 | 2/3 |
| get_flight_status | 2/2 | 2/2 | 2/2 | 2/2 |
| get_news | 3/3 | 3/3 | 3/3 | 3/3 |
| get_stock_price | 2/3 | 3/3 | 3/3 | 3/3 |
| get_time_in_timezone | 1/2 | 1/2 | 1/2 | 0/2 |
| get_weather | 2/4 | 3/4 | 3/4 | 3/4 |
| search_hotel | 1/2 | 2/2 | 1/2 | 2/2 |
| search_movie | 2/3 | 2/3 | 3/3 | 2/3 |
| search_recipe | 3/3 | 3/3 | 3/3 | 3/3 |
| search_restaurant | 2/2 | 2/2 | 1/2 | 2/2 |
| send_email | 0/1 | 0/2 | 0/1 | 0/2 |
| set_reminder | 0/2 | 0/2 | 1/2 | 0/2 |
| summarize_text | 1/1 | 2/2 | 2/2 | 2/2 |
| translate_text | 0/3 | 1/3 | 2/3 | 1/3 |
