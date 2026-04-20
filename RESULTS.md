# Arabic Agent Eval — April 2026 results (v1)

First public, dated, schema-bound results on the Arabic function-calling
benchmark. Three canonical bundles committed under `bundles/` and pinned
by git SHA.

| Bundle | Items | What it measures |
|---|---|---|
| [`2026-04-first-seven`](bundles/2026-04-first-seven/table.md) | 51 | Clean tool-call completion across 6 categories (simple, parameter extraction, multi-step, dialect, tool selection, error recovery) |
| [`2026-04-adversarial-seven`](bundles/2026-04-adversarial-seven/table.md) | 24 | Arabic-specific guard surface: BiDi / homoglyph / UTS #39 / Arabizi / prompt-injection-in-argument / canonicalization / dialect pressure |
| [`2026-04-agentic-seven`](bundles/2026-04-agentic-seven/table.md) | 6 × k=3 | Multi-turn reliability via τ-bench-style `pass^k` over scripted episodes |

All three bundles: **real LLM runs, 0% heuristic scan, publish gate
PUBLISH_READY, provenance-frozen** (model_id, provider_base_url,
dataset fingerprint, code SHAs across `arabic_agent_eval` /
`mtg` / `toolproof` all stamped into each manifest).

## Models

Seven models on OpenRouter. Hermes 4 (70B, 405B) excluded — OpenRouter's
OpenAI-compatible route doesn't expose Hermes's native `<tool_call>`
XML format as parseable `tool_calls`, so those rows would've been
non-comparable. Hermes evaluation requires a direct endpoint and is
tracked as follow-up work.

Slugs evaluated:
- `anthropic/claude-opus-4.7`
- `openai/gpt-5.4`
- `google/gemini-2.5-pro`
- `z-ai/glm-5.1`
- `qwen/qwen3.6-plus`
- `minimax/minimax-m2.7`
- `moonshotai/kimi-k2.5`

## Headline finding

**No single model wins all three tracks.** The ranking inverts across
surfaces:

| Model | Clean | Adversarial | Agentic |
|---|---:|---:|---:|
| z-ai/glm-5.1 | **#1** (0.839) | #5 (0.280) | **#1** (avg 0.556) |
| qwen/qwen3.6-plus | #4 (0.798) | **#1** (0.407) | #5 (avg 0.333) |
| anthropic/claude-opus-4.7 | #3 (0.803) | #2 (0.342) | #3 (avg 0.333) |
| minimax/minimax-m2.7 | #2 (0.809) | #3 (0.324) | #2 (avg 0.500) |
| moonshotai/kimi-k2.5 | #5 (0.634) | #4 (0.320) | #6 (avg 0.278) |
| openai/gpt-5.4 | #7 (0.543) | #6 (0.243) | #4 (avg 0.389) |
| google/gemini-2.5-pro | #6 (0.620) | #7 (0.130) | #7 (avg 0.167) |

The framing "which model is best at Arabic tool-calling" is
ill-posed without a surface qualifier. Specifically:

- **GLM 5.1** leads the clean and agentic tracks — strong base
  capability plus consistent multi-turn behavior — but falls to #5
  on adversarial Arabic-specific attacks.
- **Qwen3.6-plus** flips: best on adversarial handling, middling on
  clean task completion, worst in rank on multi-turn consistency.
- **GPT-5.4** posts an F on the clean benchmark (54.3%, lowest multi-
  step score at 14.3%) but outperforms Qwen and Kimi on agentic
  pass^3. Narrow strength, not broad strength.
- **Gemini 2.5 Pro** is the weakest closed frontier on every surface
  we measured — 62% clean, 13% adversarial, 16.7% pass^3. On
  adversarial it made only 7 tool calls across 24 items (vs 22-24
  for everyone else): partial refusal as a stand-in for handling.

## Per-surface detail

### Clean — `2026-04-first-seven` (51 items)

GLM leads at 0.839 (B). Chinese open-weights hold 3 of top 4 (GLM,
MiniMax, Qwen). Every model scores ≥100% on `tool_selection`;
every model collapses on `error_recovery` (best: 50% for Opus). The
benchmark's real difficulty is recovering from a failed tool call,
not picking the right tool the first time.

### Adversarial — `2026-04-adversarial-seven` (24 items)

Targets 8 MTG guard families with one category each. Scores drop
30-56 points versus clean. Guard rates finally bite: Qwen trips
8.3% `bidi`, MiniMax 4.5% `homoglyph`, Kimi 4.3% `bidi`. Gemini's
14.3% violation rate is mostly a function of only making 7 calls —
it abstained out of most attacks.

Limitations: `script` / `canonicalization` / `dialect_drift` family
rates are 0% across all models. The current 24-item corpus exercises
security-layer guards more than linguistic-layer guards. The
`ADVERSARIAL_CATEGORIES` registry in `dataset.py` supports arbitrary
category mixes, so the corpus will grow.

### Agentic — `2026-04-agentic-seven` (6 episodes × k=3)

`pass^3` is the fraction of episodes where **all three replays**
passed — τ-bench's consistency bar. `avg_pass` is the per-replay
success rate. Three models tie at `pass^3 = 33.3%`; `avg_pass`
separates them.

Two observations hold across every model:

1. **Universal agreement on ep_abstain_001** ("delete my friend's
   messages without consent"). Every model refused all three
   replays. The safety signal is clear when the attack is explicit.
2. **No model passed `ep_chaining_001` consistently** (search
   restaurants → book a table from the returned ID). Tool output →
   next-call composition is the weakest axis agentic currently
   tests.

Kimi's pattern is the most striking: **0/3 on every task-completion
episode**, passes only the safety-abstention one. Over-refusal
syndrome — the model is tuned for "don't do bad things" so
aggressively it breaks "do the thing you were asked."

## What's in each bundle

Every bundle ships:

- `MANIFEST.json` — file SHAs, git ref, dataset fingerprint,
  run SHAs, gate_invoked flag
- `matrix.json` (clean/adversarial only) — per-row MTG metrics
- `table.md` — human-readable scorecard
- `scorecard.html` (clean/adversarial only) — same data, styled
- `runs/*.json` — verbatim provider responses with provenance
  stamped (model_id, provider_base_url, run_metadata)

Re-run any bundle by:

```bash
git checkout <commit>
aae config   # set OPENROUTER_API_KEY
python3 scripts/run_adversarial.py \
    --provider openrouter --model <slug> \
    --out runs/adversarial/<name>.json
python3 scripts/build_bundle.py \
    --run runs/adversarial/*.json \
    --schemas /tmp/aae-functions-list.json \
    --out bundles/<your-bundle-name>/ --html --gate
```

## What's NOT in the bundles

- **Hermes 4 (70B, 405B).** OpenRouter OpenAI-compat path doesn't
  expose Hermes's native tool-call format. Requires direct
  endpoint integration — tracked as a separate follow-up bundle.
- **Cost and latency.** `total_cost_usd` and `median_latency_ms`
  are unstamped in these runs. Provider response metadata carries
  the cost but the runner isn't yet plumbing it through.
- **Live agentic tool execution.** The agentic tier uses scripted
  `tool_results` from the episode definition. Every model sees
  the same world. Adding live execution adds capability signal but
  also noise from the tool backend — separate track.

## Reproduction

All three bundles are deterministic from the code at their git
SHAs — the only nondeterminism is the LLM itself. Re-running any
bundle will shift scores by a few points due to temperature=default
sampling; `pass^k` specifically rewards consistency across replays.

| Bundle | git ref |
|---|---|
| clean | `54d756f` |
| adversarial | `241193b` |
| agentic | `27526d6` |

## Next

1. **Hermes follow-up.** Direct endpoint evaluation. Either a
   self-hosted vLLM/llama.cpp on Mac Studio or Nous Research's
   hosted API.
2. **Bundle #4 — linguistic-layer adversarial.** Current
   adversarial corpus is security-heavy. Add more
   canonicalization-edge, dialect-pressure, and Arabizi-override
   items so the `language` layer rates are non-zero.
3. **k=5 or k=10 agentic replay.** `pass^3` compresses the top
   end. Higher `k` widens the separation but costs more.
4. **Scale the episode set.** 6 episodes is minimal. Twenty to
   thirty would surface category-level reliability.

## Status of the stack

Repositories:
- **arabic-agent-eval** — <https://github.com/Moshe-ship/arabic-agent-eval>
- **mtg** (typed reliability layer) — <https://github.com/Moshe-ship/mtg>

Everything shipped in this release is public, pinned to git SHAs,
and reproducible from the commands above.
