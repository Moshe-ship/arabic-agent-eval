# Baselines

This page hosts baseline results. Entries are produced by running `aae run --md-output docs/baseline-runs/<model>.md` and copying the summary row here.

## v0.1.0 dataset composition

The reference set has 51 items distributed unevenly across dialects. This is deliberately documented — the dataset is a **reference set**, not a complete coverage sweep:

| Dialect | Items | Coverage |
|---|---:|---|
| MSA (فصحى) | 32 | primary — default across all categories |
| Gulf (خليجي) | 10 | dialect-handling + tool-selection |
| Levantine (شامي) | 4 | dialect-handling focused |
| Egyptian (مصري) | 3 | dialect-handling focused |
| Maghrebi (مغاربي) | 2 | dialect-handling focused |

Growing the Egyptian/Levantine/Maghrebi splits to parity with Gulf is tracked as [expansion issues](https://github.com/Moshe-ship/arabic-agent-eval/issues). Current results should be read as **directional**, not as statistically significant per-dialect rankings.

## Baseline table template

Copy this structure for each model run:

| Provider | Model | Overall | Func Sel | Arg Acc | Ar Pres | Dialect Und | Date |
|---|---|---:|---:|---:|---:|---:|---|
| _example_ | _example-model_ | XX.X% | XX.X% | XX.X% | XX.X% | XX.X% | YYYY-MM-DD |

The `_example_` row is a placeholder — all numbers below should be **real** runs with a date and command log.

## Reproducing a baseline

```bash
# Install
pip install -e .

# Configure the provider key
export OPENROUTER_API_KEY=sk-or-v1-...

# Run full eval
aae run --provider openrouter --model nousresearch/hermes-4-70b \
        --json-output > docs/baseline-runs/hermes-4-70b.json

aae run --provider openrouter --model nousresearch/hermes-4-70b \
        --md-output  docs/baseline-runs/hermes-4-70b.md
```

## Published runs

_None yet — populate this section after running against real models. See `examples/report.json` for the shape of the output._
