## Result matrix

`baseline` and `repaired` include bootstrap 95% CIs over the items scored (within-run variance only — NOT model-run variance). `heur. scan %` reports the fraction of args scored via the heuristic fallback; rows with `heur. scan > 10%` are marked ⚠ diagnostic and must not be published as clean results.

| provider | model | baseline (95% CI) | repaired (95% CI) | Δ repair | viol. % | translit % | drift % | bidi % | homoglyph % | heur. scan % | items | calls | cost | p50 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| example | synthetic-v1 | 0.900 (0.700–1.000) | 0.900 (0.700–1.000) | +0.000 | 8.3% | 0.0% | 8.3% | 0.0% | 0.0% | 0.0% | 5 | 12 | — | — |

## 3-layer taxonomy

Each failing argument is classified into exactly one layer. **Surface** = schema-shape failures (script mismatch). **Language** = register, morphology, canonicalization, overflow. **Security** = Unicode-layer attacks (BiDi, homoglyph, invisibles). Row sums are ≤ `viol. %` in the matrix above.

| provider | model | surface | language | security |
|---|---|---:|---:|---:|
| example | synthetic-v1 | 0.0% | 8.3% | 0.0% |

## Why failed — family breakdown

Granular view of the same classification (worst family wins per arg).

| provider | model | script | canonicalization | dialect | overflow | bidi | homoglyph |
|---|---|---:|---:|---:|---:|---:|---:|
| example | synthetic-v1 | 0.0% | 0.0% | 8.3% | 0.0% | 0.0% | 0.0% |

## Schema coverage

`schema-bound %` = fraction of scanned args with an x-mtg block on their tool schema. Higher is stronger evidence. Rows with low coverage end up marked ⚠ diagnostic in the main table above.

| provider | model | schema-bound % | schema-bound args | heuristic args | tools covered |
|---|---|---:|---:|---:|---|
| example | synthetic-v1 | 100.0% | 12 | 0 | `find_quran_verse`, `get_prayer_times`, `lookup_saudi_address`, `send_message_gulf` |

## Run provenance

- `example` / `synthetic-v1` — run_id `1172e2f6` · scanned 2026-04-20T02:17:37+00:00 · 5 items, 0 errors · scanner `mtg-matrix/0.6`