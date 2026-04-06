# arabic-agent-eval

**The first Arabic function-calling benchmark**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://python.org)

> Because Arabic agents deserve evaluation too.

## Why

Arabic tool-calling accuracy drops 5-10% compared to English across every model. Parameters get transliterated. Dialect instructions get ignored. Nobody measures this.

No Arabic function-calling benchmark exists. No Arabic tool-calling evaluation dataset. No Arabic agent evaluation framework. This is the first.

51 evaluation items. 6 categories. 5 dialect variants. 22 Arabic-context functions. One score that tells you how well a model actually handles Arabic function calling.

## Install

```bash
pip install arabic-agent-eval
```

## Quick Start

```bash
# Set up API keys
aae config

# Quick single-provider benchmark (12 items)
aae quick openai

# Full benchmark across all configured providers
aae run

# Compare two providers
aae compare openai anthropic
```

## Evaluation Categories

| Category | Arabic | What it tests |
|---|---|---|
| Simple Function Calling | استدعاء بسيط | Pick the right function, extract correct parameters |
| Parameter Extraction | استخراج المعاملات | Extract Arabic parameters from natural text |
| Multi-Step Reasoning | تفكير متعدد الخطوات | Chain multiple function calls in sequence |
| Dialect Handling | معالجة اللهجات | Understand Gulf, Egyptian, Levantine, Maghrebi dialects |
| Tool Selection | اختيار الأداة | Pick the right tool from 10 options |
| Error Recovery | معالجة الأخطاء | Handle Arabic error responses correctly |

## Dialect Coverage

Every category includes dialect variants:

| Dialect | Example |
|---|---|
| MSA (فصحى) | أريد حجز فندق في دبي غداً |
| Gulf (خليجي) | ابي أحجز فندق في دبي بكرة |
| Egyptian (مصري) | عايز أحجز فندق في دبي بكره |
| Levantine (شامي) | بدي احجز فندق بدبي بكرا |
| Maghrebi (مغاربي) | بغيت نحجز فندق في دبي غدا |

## Functions

22 Arabic-context functions including:

| Function | Arabic | Context |
|---|---|---|
| search_flights | البحث عن رحلات | Regional airlines |
| get_prayer_times | مواقيت الصلاة | Islamic calendar |
| calculate_zakat | حساب الزكاة | Islamic finance |
| find_quran_verse | البحث في القرآن | Quran search |
| check_visa_status | حالة التأشيرة | GCC visa systems |
| get_stock_price | سعر السهم | Tadawul, ADX, DFM |
| convert_currency | تحويل العملات | SAR, AED, EGP, MAD |
| book_car | حجز سيارة | Regional ride-hailing |
| order_food | طلب طعام | Local restaurants |
| get_traffic | حالة المرور | City traffic |

## Scoring

Each item is scored on 4 dimensions:

| Dimension | What it measures |
|---|---|
| Function Selection | Did the model pick the right function? (0 or 1) |
| Argument Accuracy | Are the extracted arguments correct? (0-1 scale) |
| Arabic Preservation | Are Arabic values preserved, not transliterated? (0 or 1) |
| Dialect Understanding | Did the model understand the dialect? (dialect category only) |

Overall score = weighted average across all 6 categories.

## Supported Providers

| Provider | Default Model |
|---|---|
| OpenAI | gpt-4o |
| Anthropic | claude-sonnet-4-6 |
| Google | gemini-2.0-flash |
| DeepSeek | deepseek-chat |
| Groq | llama-3.3-70b-versatile |
| Mistral | mistral-large-latest |
| Qwen | qwen-plus |
| xAI | grok-2 |
| Cohere | command-r-plus |
| Together | Qwen2.5-72B |
| Fireworks | Qwen2.5-72B |

## As a Library

```python
from arabic_agent_eval import Dataset, Evaluator

dataset = Dataset()

def my_call_fn(instruction, tools, functions):
    # Call your model here
    return {"calls": [...], "raw": "..."}

evaluator = Evaluator(call_fn=my_call_fn, provider="my-model", model="v1")
result = evaluator.evaluate(dataset)
print(f"Score: {result.overall_score:.1%} ({result.overall_grade})")
```

## CI / Automation

```bash
# JSON output for pipelines
aae run --json-output

# Fail if score drops below threshold
aae run --provider openai --min-score 0.7
```

## Dataset Stats

```bash
aae dataset
```

- 51 evaluation items
- 6 categories (weighted)
- 5 Arabic dialects
- 22 function definitions
- 3 difficulty levels

## Security

API keys are stored in `~/.aae/config.json` with `0600` permissions. Environment variables are the recommended way to provide keys in CI.

## Community

Built with input from the [Saudi AI Community](https://x.com/i/communities/2032184341682643429).

## License

MIT -- Musa the Carpenter
