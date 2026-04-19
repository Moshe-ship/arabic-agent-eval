# Synthetic example bundle — NOT A RESULT

This bundle is **fabricated**. No model was queried. No benchmark was
run. The five rows under `matrix.json` come from hand-written "actual
calls" that happen to be realistic Arabic content, scored against
Hurmoz's x-mtg-annotated tool schemas.

**Why it exists:** developers need a bundle-shaped artifact to:

- Read the format without generating one
- Point tooling at (linters, HTML viewers, schema validators)
- Test the publish gate end-to-end
- Verify the bundle manifest integrity model

**Why it is NOT under `bundles/`:** the CI publish gate workflow
(`.github/workflows/publish-gate.yml`) scans `bundles/**/MANIFEST.json`
for real results. Keeping this bundle under `examples/` prevents it
from being accidentally treated as a real one.

## Regenerating

This bundle is committed alongside the code that produces it. If the
bundle format changes, the committed files may disagree with the
writer. Regenerate with:

```bash
python scripts/gen_example_bundle.py
```

The generator expects Hurmoz to be checked out as a sibling of this
repo (for the tool schemas).

## Verifying

The bundle passes the publish gate (with `--allow-no-runs` because
there are no source run JSONs — no model was called):

```bash
python scripts/check_publish_ready.py examples/synthetic_bundle --allow-no-runs
```

## What's in the matrix row

| field | value | note |
|---|---|---|
| provider | `example` | synthetic marker |
| model | `synthetic-v1` | synthetic marker |
| diagnostic | `false` | schema-bound, clean |
| schema_bound_rate | 1.0 | every arg had x-mtg |
| heuristic_scan_rate | 0.0 | no heuristic fallback |
| baseline_score | 0.90 | fabricated |
| baseline_ci_95 | populated | bootstrap over 5 items |

The row's numbers are arbitrary. Do not quote them anywhere as a real
measurement. They exist so `scan_with_schemas` produces a non-empty,
non-diagnostic row for format demonstration.
