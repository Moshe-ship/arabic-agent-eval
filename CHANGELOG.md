# Changelog

All notable changes to arabic-agent-eval.

## 0.1.0 — initial public release

### Added

- **51-item Arabic function-calling benchmark** covering 6 categories: simple_function_calling, parameter_extraction, multi_step, tool_selection, dialect_handling, error_recovery.
- **5 dialect splits**: MSA, Gulf, Egyptian, Levantine, Maghrebi. Each item carries a declared `dialect` field.
- **Canonical structured-call grader** (`arabic_agent_eval.scoring`) with four axes:
  - `function_selection` — binary tool-name match (wildcard `"*"` accepts any non-null actual).
  - `argument_accuracy` — per-key ladder: exact → Arabic-normalized → case-insensitive.
  - `arabic_preservation` — fraction of Arabic-valued expected arguments returned in Arabic script.
  - `dialect_understanding` / `error_handling` — category-specific axes.
- **Multi-call scoring** — every expected call in a multi-step item is graded; extra actual calls dilute all three axes (denominator = `max(len(expected), len(actual))`), so unrelated or destructive appends cannot score 1.0.
- **Arabic normalization** (`normalize_arabic`): alef variants (آأإ → ا), ya (ى → ي), ta-marbuta (ة → ه), strip tatweel.
- **Providers** (`arabic_agent_eval.providers`): OpenAI, Anthropic, OpenRouter, Hermes. `HERMES_BASE_URL` / `OPENROUTER_BASE_URL` env overrides.
- **Exporter** (`arabic_agent_eval.exporter`) emits `data/{msa,gulf,egyptian,levantine,maghrebi,all}.jsonl` + `functions.json` + `categories.json` for HuggingFace publishing.
- **Report** (`arabic_agent_eval.report`): markdown scorecard with category and dialect breakdowns.
- **Atropos environment** (`atropos/arabic_tool_calling/`): RL environment wrapping the benchmark for Nous Atropos.
- **Licensing**: code Apache-2.0, data CC-BY-4.0 (separate `data/LICENSE`).
- **Docs** (`docs/`): `schema.md`, `grading.md`, `related_work.md`, `baselines.md`, `dataset_card.md`.

### Tests

59 tests covering scoring, dataset loading, evaluator multi-call behavior, provider routing, exporter, and Atropos environment integration.
