## Result matrix

`baseline` and `repaired` include bootstrap 95% CIs over the items scored (within-run variance only — NOT model-run variance). `heur. scan %` reports the fraction of args scored via the heuristic fallback; rows with `heur. scan > 10%` are marked ⚠ diagnostic and must not be published as clean results.

| provider | model | baseline (95% CI) | repaired (95% CI) | Δ repair | viol. % | translit % | drift % | bidi % | homoglyph % | heur. scan % | items | calls | cost | p50 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| openrouter | z-ai/glm-5.1 | 0.839 (0.750–0.925) | 0.840 (0.750–0.925) | +0.001 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 51 | 102 | — | — |
| openrouter | minimax/minimax-m2.7 | 0.809 (0.717–0.888) | 0.805 (0.717–0.888) | -0.005 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 51 | 101 | — | — |
| openrouter | anthropic/claude-opus-4.7 | 0.803 (0.700–0.891) | 0.800 (0.700–0.891) | -0.003 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 51 | 96 | — | — |
| openrouter | qwen/qwen3.6-plus | 0.798 (0.711–0.893) | 0.801 (0.711–0.893) | +0.003 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 51 | 100 | — | — |
| openrouter | moonshotai/kimi-k2.5 | 0.634 (0.539–0.767) | 0.647 (0.539–0.767) | +0.013 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 51 | 74 | — | — |
| openrouter | google/gemini-2.5-pro | 0.620 (0.509–0.731) | 0.618 (0.509–0.731) | -0.002 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 51 | 71 | — | — |
| openrouter | openai/gpt-5.4 | 0.543 (0.431–0.678) | 0.555 (0.431–0.678) | +0.012 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 51 | 59 | — | — |

## 3-layer taxonomy

Each failing argument is classified into exactly one layer. **Surface** = schema-shape failures (script mismatch). **Language** = register, morphology, canonicalization, overflow. **Security** = Unicode-layer attacks (BiDi, homoglyph, invisibles). Row sums are ≤ `viol. %` in the matrix above.

| provider | model | surface | language | security |
|---|---|---:|---:|---:|
| openrouter | z-ai/glm-5.1 | 0.0% | 0.0% | 0.0% |
| openrouter | minimax/minimax-m2.7 | 0.0% | 0.0% | 0.0% |
| openrouter | anthropic/claude-opus-4.7 | 0.0% | 0.0% | 0.0% |
| openrouter | qwen/qwen3.6-plus | 0.0% | 0.0% | 0.0% |
| openrouter | moonshotai/kimi-k2.5 | 0.0% | 0.0% | 0.0% |
| openrouter | google/gemini-2.5-pro | 0.0% | 0.0% | 0.0% |
| openrouter | openai/gpt-5.4 | 0.0% | 0.0% | 0.0% |

## Why failed — family breakdown

Granular view of the same classification (worst family wins per arg).

| provider | model | script | canonicalization | dialect | overflow | bidi | homoglyph |
|---|---|---:|---:|---:|---:|---:|---:|
| openrouter | z-ai/glm-5.1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| openrouter | minimax/minimax-m2.7 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| openrouter | anthropic/claude-opus-4.7 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| openrouter | qwen/qwen3.6-plus | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| openrouter | moonshotai/kimi-k2.5 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| openrouter | google/gemini-2.5-pro | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| openrouter | openai/gpt-5.4 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Schema coverage

`schema-bound %` = fraction of scanned args with an x-mtg block on their tool schema. Higher is stronger evidence. Rows with low coverage end up marked ⚠ diagnostic in the main table above.

| provider | model | schema-bound % | schema-bound args | heuristic args | tools covered |
|---|---|---:|---:|---:|---|
| openrouter | z-ai/glm-5.1 | 100.0% | 102 | 0 | `book_car`, `book_hotel`, `book_table`, `calculate_zakat`, `check_visa_status`, `convert_currency`, `find_quran_verse`, `get_news`, `get_prayer_times`, `get_stock_price`, `get_time`, `get_traffic`, `get_weather`, `order_food`, `schedule_meeting`, `search_flights`, `search_jobs`, `search_restaurants`, `send_message`, `send_money`, `set_reminder`, `translate_text` |
| openrouter | minimax/minimax-m2.7 | 100.0% | 101 | 0 | `book_car`, `book_hotel`, `book_table`, `calculate_zakat`, `check_visa_status`, `convert_currency`, `find_quran_verse`, `get_news`, `get_prayer_times`, `get_stock_price`, `get_time`, `get_traffic`, `get_weather`, `order_food`, `schedule_meeting`, `search_flights`, `search_jobs`, `search_restaurants`, `send_message`, `send_money`, `set_reminder`, `translate_text` |
| openrouter | anthropic/claude-opus-4.7 | 100.0% | 96 | 0 | `book_hotel`, `book_table`, `calculate_zakat`, `check_visa_status`, `convert_currency`, `find_quran_verse`, `get_news`, `get_prayer_times`, `get_stock_price`, `get_time`, `get_traffic`, `get_weather`, `order_food`, `schedule_meeting`, `search_flights`, `search_jobs`, `search_restaurants`, `send_message`, `send_money`, `set_reminder`, `translate_text` |
| openrouter | qwen/qwen3.6-plus | 100.0% | 100 | 0 | `book_car`, `book_hotel`, `book_table`, `calculate_zakat`, `check_visa_status`, `convert_currency`, `find_quran_verse`, `get_news`, `get_prayer_times`, `get_stock_price`, `get_time`, `get_traffic`, `get_weather`, `order_food`, `schedule_meeting`, `search_flights`, `search_jobs`, `search_restaurants`, `send_message`, `send_money`, `set_reminder`, `translate_text` |
| openrouter | moonshotai/kimi-k2.5 | 100.0% | 74 | 0 | `book_table`, `calculate_zakat`, `check_visa_status`, `convert_currency`, `find_quran_verse`, `get_news`, `get_prayer_times`, `get_stock_price`, `get_time`, `get_traffic`, `get_weather`, `order_food`, `search_jobs`, `search_restaurants`, `send_message`, `send_money`, `set_reminder`, `translate_text` |
| openrouter | google/gemini-2.5-pro | 100.0% | 71 | 0 | `book_car`, `book_table`, `calculate_zakat`, `check_visa_status`, `convert_currency`, `find_quran_verse`, `get_news`, `get_prayer_times`, `get_stock_price`, `get_time`, `get_traffic`, `get_weather`, `order_food`, `search_jobs`, `search_restaurants`, `send_message`, `send_money`, `set_reminder`, `translate_text` |
| openrouter | openai/gpt-5.4 | 100.0% | 59 | 0 | `calculate_zakat`, `check_visa_status`, `convert_currency`, `find_quran_verse`, `get_news`, `get_prayer_times`, `get_stock_price`, `get_time`, `get_traffic`, `get_weather`, `search_jobs`, `send_message`, `send_money`, `set_reminder`, `translate_text` |

## Run provenance

- `openrouter` / `z-ai/glm-5.1` — run_id `8554a4e9` · scanned 2026-04-20T06:28:31+00:00 · 51 items, 0 errors · scanner `mtg-matrix/0.6`
- `openrouter` / `minimax/minimax-m2.7` — run_id `ac8ce160` · scanned 2026-04-20T06:28:31+00:00 · 51 items, 0 errors · scanner `mtg-matrix/0.6`
- `openrouter` / `anthropic/claude-opus-4.7` — run_id `9fde7f82` · scanned 2026-04-20T06:28:31+00:00 · 51 items, 0 errors · scanner `mtg-matrix/0.6`
- `openrouter` / `qwen/qwen3.6-plus` — run_id `375831d3` · scanned 2026-04-20T06:28:31+00:00 · 51 items, 0 errors · scanner `mtg-matrix/0.6`
- `openrouter` / `moonshotai/kimi-k2.5` — run_id `25d30c28` · scanned 2026-04-20T06:28:31+00:00 · 51 items, 0 errors · scanner `mtg-matrix/0.6`
- `openrouter` / `google/gemini-2.5-pro` — run_id `be7dfa12` · scanned 2026-04-20T06:28:31+00:00 · 51 items, 0 errors · scanner `mtg-matrix/0.6`
- `openrouter` / `openai/gpt-5.4` — run_id `b693aa78` · scanned 2026-04-20T06:28:32+00:00 · 51 items, 0 errors · scanner `mtg-matrix/0.6`