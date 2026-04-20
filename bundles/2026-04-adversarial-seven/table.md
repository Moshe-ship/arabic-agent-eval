## Result matrix

`baseline` and `repaired` include bootstrap 95% CIs over the items scored (within-run variance only — NOT model-run variance). `heur. scan %` reports the fraction of args scored via the heuristic fallback; rows with `heur. scan > 10%` are marked ⚠ diagnostic and must not be published as clean results.

| provider | model | baseline (95% CI) | repaired (95% CI) | Δ repair | viol. % | translit % | drift % | bidi % | homoglyph % | heur. scan % | items | calls | cost | p50 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| openrouter | qwen/qwen3.6-plus | 0.407 (0.234–0.615) | 0.418 (0.234–0.615) | +0.011 | 12.5% | 0.0% | 0.0% | 4.2% | 0.0% | 0.0% | 24 | 24 | — | — |
| openrouter | anthropic/claude-opus-4.7 | 0.342 (0.181–0.540) | 0.353 (0.181–0.540) | +0.010 | 4.5% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 24 | 22 | — | — |
| openrouter | minimax/minimax-m2.7 | 0.324 (0.169–0.494) | 0.324 (0.169–0.494) | +0.000 | 4.5% | 0.0% | 0.0% | 0.0% | 4.5% | 0.0% | 24 | 22 | — | — |
| openrouter | moonshotai/kimi-k2.5 | 0.320 (0.151–0.477) | 0.313 (0.151–0.477) | -0.007 | 8.7% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 24 | 23 | — | — |
| openrouter | z-ai/glm-5.1 | 0.280 (0.136–0.526) | 0.317 (0.136–0.526) | +0.037 | 4.8% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 24 | 21 | — | — |
| openrouter | openai/gpt-5.4 | 0.243 (0.100–0.403) | 0.253 (0.100–0.403) | +0.010 | 5.9% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 24 | 17 | — | — |
| openrouter | google/gemini-2.5-pro | 0.130 (0.017–0.286) | 0.140 (0.017–0.286) | +0.010 | 14.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 24 | 7 | — | — |

## 3-layer taxonomy

Each failing argument is classified into exactly one layer. **Surface** = schema-shape failures (script mismatch). **Language** = register, morphology, canonicalization, overflow. **Security** = Unicode-layer attacks (BiDi, homoglyph, invisibles). Row sums are ≤ `viol. %` in the matrix above.

| provider | model | surface | language | security |
|---|---|---:|---:|---:|
| openrouter | qwen/qwen3.6-plus | 0.0% | 0.0% | 8.3% |
| openrouter | anthropic/claude-opus-4.7 | 0.0% | 0.0% | 0.0% |
| openrouter | minimax/minimax-m2.7 | 0.0% | 0.0% | 4.5% |
| openrouter | moonshotai/kimi-k2.5 | 0.0% | 0.0% | 4.3% |
| openrouter | z-ai/glm-5.1 | 0.0% | 0.0% | 0.0% |
| openrouter | openai/gpt-5.4 | 0.0% | 0.0% | 0.0% |
| openrouter | google/gemini-2.5-pro | 0.0% | 0.0% | 0.0% |

## Why failed — family breakdown

Granular view of the same classification (worst family wins per arg).

| provider | model | script | canonicalization | dialect | overflow | bidi | homoglyph |
|---|---|---:|---:|---:|---:|---:|---:|
| openrouter | qwen/qwen3.6-plus | 0.0% | 0.0% | 0.0% | 0.0% | 8.3% | 0.0% |
| openrouter | anthropic/claude-opus-4.7 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| openrouter | minimax/minimax-m2.7 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 4.5% |
| openrouter | moonshotai/kimi-k2.5 | 0.0% | 0.0% | 0.0% | 0.0% | 4.3% | 0.0% |
| openrouter | z-ai/glm-5.1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| openrouter | openai/gpt-5.4 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| openrouter | google/gemini-2.5-pro | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Schema coverage

`schema-bound %` = fraction of scanned args with an x-mtg block on their tool schema. Higher is stronger evidence. Rows with low coverage end up marked ⚠ diagnostic in the main table above.

| provider | model | schema-bound % | schema-bound args | heuristic args | tools covered |
|---|---|---:|---:|---:|---|
| openrouter | qwen/qwen3.6-plus | 100.0% | 24 | 0 | `convert_currency`, `get_news`, `get_stock_price`, `get_weather`, `search_restaurants`, `send_message`, `translate_text` |
| openrouter | anthropic/claude-opus-4.7 | 100.0% | 22 | 0 | `convert_currency`, `get_news`, `get_stock_price`, `get_weather`, `search_restaurants`, `send_message`, `send_money`, `translate_text` |
| openrouter | minimax/minimax-m2.7 | 100.0% | 22 | 0 | `convert_currency`, `get_news`, `get_stock_price`, `get_weather`, `search_flights`, `search_restaurants`, `send_message`, `send_money` |
| openrouter | moonshotai/kimi-k2.5 | 100.0% | 23 | 0 | `book_car`, `convert_currency`, `get_news`, `get_stock_price`, `get_weather`, `search_restaurants`, `send_message`, `translate_text` |
| openrouter | z-ai/glm-5.1 | 100.0% | 21 | 0 | `convert_currency`, `get_news`, `get_stock_price`, `get_weather`, `search_restaurants`, `translate_text` |
| openrouter | openai/gpt-5.4 | 100.0% | 17 | 0 | `convert_currency`, `get_news`, `get_weather`, `search_restaurants`, `send_message`, `translate_text` |
| openrouter | google/gemini-2.5-pro | 100.0% | 7 | 0 | `get_weather`, `search_restaurants`, `translate_text` |

## Run provenance

- `openrouter` / `qwen/qwen3.6-plus` — run_id `3ddc9bbc` · scanned 2026-04-20T15:49:10+00:00 · 24 items, 1 errors · scanner `mtg-matrix/0.6`
- `openrouter` / `anthropic/claude-opus-4.7` — run_id `63dda5a2` · scanned 2026-04-20T15:49:10+00:00 · 24 items, 0 errors · scanner `mtg-matrix/0.6`
- `openrouter` / `minimax/minimax-m2.7` — run_id `48603223` · scanned 2026-04-20T15:49:10+00:00 · 24 items, 0 errors · scanner `mtg-matrix/0.6`
- `openrouter` / `moonshotai/kimi-k2.5` — run_id `79ed9a40` · scanned 2026-04-20T15:49:10+00:00 · 24 items, 0 errors · scanner `mtg-matrix/0.6`
- `openrouter` / `z-ai/glm-5.1` — run_id `2f054f4c` · scanned 2026-04-20T15:49:10+00:00 · 24 items, 2 errors · scanner `mtg-matrix/0.6`
- `openrouter` / `openai/gpt-5.4` — run_id `85702395` · scanned 2026-04-20T15:49:10+00:00 · 24 items, 0 errors · scanner `mtg-matrix/0.6`
- `openrouter` / `google/gemini-2.5-pro` — run_id `4f1bb103` · scanned 2026-04-20T15:49:11+00:00 · 24 items, 0 errors · scanner `mtg-matrix/0.6`