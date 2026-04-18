---
language:
- ar
license: cc-by-4.0
task_categories:
- question-answering
- text-generation
tags:
- function-calling
- tool-use
- arabic
- dialects
- agent-evaluation
- benchmark
size_categories:
- n<1K
pretty_name: Arabic Agent Eval
configs:
- config_name: default
  data_files:
  - split: msa
    path: data/msa.jsonl
  - split: gulf
    path: data/gulf.jsonl
  - split: egyptian
    path: data/egyptian.jsonl
  - split: levantine
    path: data/levantine.jsonl
  - split: maghrebi
    path: data/maghrebi.jsonl
  - split: all
    path: data/all.jsonl
---

# Arabic Agent Eval — Dataset Card

**The first Arabic function-calling benchmark with dialect splits.**

## Dataset summary

51 evaluation items spanning 6 categories and 5 dialects of Arabic, testing whether large language models can (a) select the right tool, (b) extract arguments from natural Arabic instructions, (c) preserve Arabic text in tool arguments instead of transliterating, and (d) understand dialectal framing.

## Supported tasks

- **Function-calling evaluation** — items pair Arabic instructions with expected structured tool calls.
- **Tool-use argument extraction** — items declare `expected_calls[i].arguments` that the model's output is compared against via canonical structured-call grading.
- **Dialect-robustness evaluation** — dialect-labeled items enable cross-dialect variance analysis for any agent.

## Languages

Arabic only. Five dialect registers: Modern Standard Arabic (`msa`), Gulf (`gulf`), Egyptian (`egyptian`), Levantine (`levantine`), Maghrebi (`maghrebi`).

## Dataset structure

Each row is a single evaluation item. See [docs/schema.md](docs/schema.md) for full schema. Example:

```json
{
  "id": "dialect_001",
  "category": "dialect_handling",
  "instruction": "ابي أحجز فندق في دبي بكرة",
  "dialect": "gulf",
  "available_functions": ["book_hotel", "search_flights", "get_weather"],
  "expected_calls": [
    {"function": "book_hotel", "arguments": {"city": "دبي", "check_in": "غداً"}}
  ],
  "difficulty": "easy"
}
```

## Splits

| Split | Items | Purpose |
|---|---:|---|
| `msa` | 32 | Modern Standard Arabic baseline |
| `gulf` | 10 | Gulf dialect (Saudi, UAE, Kuwait, Qatar, Bahrain, Oman) |
| `egyptian` | 3 | Egyptian dialect |
| `levantine` | 4 | Levantine dialect (Jordan, Palestine, Syria, Lebanon) |
| `maghrebi` | 2 | Maghrebi dialect (Morocco, Algeria, Tunisia) |
| `all` | 51 | concatenation of the five dialect splits |

Non-MSA dialects are undersampled in v0.1.0 (see [docs/baselines.md](docs/baselines.md)). Use per-dialect averages as directional signal, not as final rankings.

## Categories

| Category | Arabic | Items | Weight |
|---|---|---:|---:|
| Simple Function Calling | استدعاء بسيط | 10 | 0.20 |
| Parameter Extraction | استخراج المعاملات | 9 | 0.20 |
| Multi-Step Reasoning | تفكير متعدد الخطوات | 7 | 0.20 |
| Dialect Handling | معالجة اللهجات | 12 | 0.15 |
| Tool Selection | اختيار الأداة | 7 | 0.15 |
| Error Recovery | معالجة الأخطاء | 6 | 0.10 |

## Tool registry

22 Arabic-context tool definitions ship alongside the items, covering Islamic services (prayer times, zakat, Quran search), GCC financial services (Tadawul stocks, multi-currency conversion, money transfer), regional consumer actions (ride-hailing, food delivery, hotel booking), and communication (WhatsApp, SMS, email). See `data/functions.json`.

## Source data

Items are hand-written native Arabic instructions drawn from real regional use cases (Saudi, UAE, Egyptian, Levantine, Moroccan). Authored by the repository maintainer (a native Arabic speaker and digital-marketing practitioner). No data scraped or model-generated.

## Personal and sensitive information

None. Items use fictitious names (e.g. `أحمد`, `سارة`, `محمد`) and generic city references.

## Licensing

**Data:** CC-BY-4.0 (`data/`, `dataset_card.md`, `docs/`)
**Code:** Apache-2.0 (`arabic_agent_eval/`, `tests/`, `scripts/`)

See [LICENSES.md](LICENSES.md).

## Citation

```bibtex
@software{arabic_agent_eval_2026,
  title = {Arabic Agent Eval: The first Arabic function-calling benchmark with dialect splits},
  author = {Abumazin, Mousa},
  year = {2026},
  url = {https://github.com/Moshe-ship/arabic-agent-eval}
}
```

## Limitations and biases

- **Dialect coverage is uneven** in v0.1.0 (32 MSA / 10 Gulf / 4 Levantine / 3 Egyptian / 2 Maghrebi). Per-dialect statistics from the rare splits are directional only.
- **GCC bias** in the tool registry — Gulf-region contexts dominate (Saudi addresses, Tadawul, AED/SAR currency defaults).
- **No intra-dialect paraphrase variance** in v0.1.0. Scheduled for expansion.
- **English tool names** with Arabic descriptions — models trained primarily on English function-calling may have a spurious advantage on function selection. See [docs/grading.md](docs/grading.md).
- **No automatic audio, RTL-UI, or code-switching tests.** Text-only evaluation.
- **Grading normalizes Arabic variants conservatively** (alef, ya, ta-marbuta, tatweel) but does not apply deep lemmatization. Edge cases may score lower than a human would award.
