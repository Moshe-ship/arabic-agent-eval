# Canonical Structured-Call Grading

The evaluator grades each `(expected_call, actual_call)` pair along four axes. This is **canonical structured-call comparison** — not AST matching. The grader reads tool calls as structured JSON, normalizes Arabic text, and emits four independent scores.

## The four axes

| Axis | Range | What it measures |
|---|:-:|---|
| `function_selection` | 0 or 1 | Did the model pick the expected tool name? Binary. |
| `argument_accuracy` | 0.0 – 1.0 | Per-key structural match across expected arguments, after normalization. |
| `arabic_preservation` | 0.0 – 1.0 | Fraction of Arabic-valued expected arguments that were returned in Arabic script (not transliterated, not translated). |
| `dialect_understanding` | 0 or 1 | For `dialect_handling` category only — did the model correctly route despite the dialectal framing? Equals `function_selection` for those items. |

A fifth axis, `error_handling`, applies only to the `error_recovery` category and follows the same structure.

## Tool-equality rules

- Function names are compared **case-sensitively, exact match**. `search_flights != SearchFlights`.
- The literal wildcard string `"*"` in `expected_call.function` accepts any actual function name (used for multi-step items with runtime-dependent intermediates).

## Argument-comparison rules

For each key in `expected_call.arguments`, the grader checks the corresponding key in `actual_call.arguments` and awards credit per this ladder:

1. **Exact string match** after stripping whitespace → **1.0**
2. **Arabic-normalized match** (see below) → **0.9**
3. **Case-insensitive match** → **0.8**
4. **Any other result** → **0.0**

Numeric values are stringified before comparison. Missing keys contribute 0. Extra keys the model emits are ignored (not penalized).

The per-call `argument_accuracy` is `sum(key_scores) / len(expected_arguments)`.

## Multi-call scoring

Items can declare multiple `expected_calls`. The grader scores each expected call independently and averages across `max(len(expected_calls), len(actual_calls))`. Concretely:

- A model that emits fewer calls than expected gets 0 for every missing slot, diluting all three axes toward 0.
- A model that emits **extra** calls beyond the expected count contributes 0 to the numerator for each extra but increases the denominator by 1. This penalizes models that append unrelated or destructive calls. A perfect 3-call sequence followed by one extra delete scores 0.75 on all three axes, not 1.0.
- Missing argument keys contribute 0; extra keys in a call are ignored (unchanged).

`dialect_understanding` and `error_handling` (category-specific axes) follow `function_selection`.

## Arabic normalization rules

The normalizer (`arabic_agent_eval.scoring.normalize_arabic`) is idempotent and applies to any string containing Arabic script. Non-Arabic strings pass through lowercased.

| Transform | Example |
|---|---|
| Strip tatweel (U+0640) | `مــرحبا` → `مرحبا` |
| Unify alef variants (آأإ → ا) | `أحمد` → `احمد` |
| Unify ya (ى → ي) | `مكتبى` → `مكتبي` |
| Unify ta-marbuta (ة → ه) | `فاطمة` → `فاطمه` |

These are conservative normalizations used widely in Arabic NLP. They are sufficient for evaluation comparison but **NOT** lossless — don't round-trip normalized strings back to users.

## Transliteration-failure rules

An argument **fails Arabic preservation** if the expected value contains Arabic script (`\u0600–\u06FF` or `\u0750–\u077F`) and the actual value contains no Arabic characters. Examples that fail:

| Expected | Actual | Status |
|---|---|---|
| `الرياض` | `Riyadh` | ❌ transliterated / translated |
| `الرياض` | `Al-Riyadh` | ❌ transliterated |
| `الرياض` | `الرياض` | ✅ preserved |

Partial Arabic content passes (e.g. `الرياض 2025` actual matches `الرياض` expected under normalization).

## Dialect-preservation rules

`dialect_handling` items carry a declared `dialect` field in their metadata. The grader does not enforce dialect on actual output (models returning MSA for a Gulf prompt are not punished at the preservation axis). Instead, `dialect_understanding` is binary and equal to `function_selection` — if the model picked the right tool despite the dialectal framing, it understood the dialect well enough to act. Finer-grained dialect scoring is reserved for [MTG](https://github.com/Moshe-ship/mtg).

## Per-category score composition

Simple, parameter-extraction, multi-step, and tool-selection categories:

```
score = 0.40 * function_selection
      + 0.35 * argument_accuracy
      + 0.25 * arabic_preservation
```

Dialect category:

```
score = 0.30 * function_selection
      + 0.20 * argument_accuracy
      + 0.20 * arabic_preservation
      + 0.30 * dialect_understanding
```

Error-recovery category:

```
score = 0.30 * function_selection
      + 0.20 * argument_accuracy
      + 0.20 * arabic_preservation
      + 0.30 * error_handling
```

## Overall score

Weighted average of category averages:

```
overall = Σ (category_avg_total * category_weight) / Σ category_weight
```

## Implementation

All of the above is in `arabic_agent_eval/scoring.py`:

- `score_function_call(expected_fn, actual_fn, expected_args, actual_args) → (func, arg, arabic)`
- `normalize_arabic(text) → str`
- `Score` dataclass with `.total` property per category
- `CategoryScore` + `compute_overall_score` for aggregation
