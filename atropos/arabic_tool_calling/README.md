# arabic_tool_calling — Atropos environment

**An Atropos community environment for training and evaluating agents on Arabic tool-call behavior.**

Native Arabic instructions across five dialect registers (MSA, Gulf, Egyptian, Levantine, Maghrebi), 22 Arabic-context tools, 4-axis canonical structured-call grading. Imports from the [arabic-agent-eval](https://github.com/Moshe-ship/arabic-agent-eval) benchmark — no forked ontology.

## Status

v0.1.0 — community environment. Single-turn by default; multi-turn supported for items with chained expected calls.

## Install

```bash
# From source, alongside atropos
pip install -e ".[dev]"
pip install arabic-agent-eval>=0.1.0
```

Or drop this folder at `environments/community/arabic_tool_calling/` in the atropos tree.

## Usage

```python
from atropos.arabic_tool_calling.env import ArabicToolCallingEnv, EnvConfig

env = ArabicToolCallingEnv(EnvConfig(
    n_tasks_per_rollout=32,
    seed=42,
    dialect_filter=None,  # or "gulf", "egy", etc. to filter dialects
))
env.setup()
task = env.next_task()
prompt = env.build_prompt(task)
# ... policy generates response ...
reward = env.score_response(task, response)
```

## Reward shape

Reward is a scalar in `[0, 1]`, computed as an equal-weighted average of the four axes from arabic-agent-eval:

```
reward = 0.25 * function_selection
       + 0.25 * argument_accuracy
       + 0.25 * arabic_preservation
       + 0.25 * dialect_understanding
```

Weights are configurable via `EnvConfig(reward_weights=RewardWeights(...))`. Equal-weighting is a starting point; teams training on specific objectives should override.

## Multi-turn

Items with `len(expected_calls) > 1` score each subsequent call; total reward is `0.6 * primary + 0.4 * mean(extras)`. Configure `max_turns` on `EnvConfig` to cap rollout length.

## Dialect personas

The env injects a dialect-aware system-prompt persona per task. Example for Gulf:

> أنت مساعد يفهم اللهجة الخليجية (ابي، ابغى، الحين، بكرا). استخدم الأدوات المناسبة.

Personas are defined in `tasks.py` and are intentionally minimal — they nudge register but don't leak expected function names.

## Data source

All task items come from `arabic_agent_eval.Dataset()` (51 items, 5 dialects). When arabic-agent-eval's coverage expands, this env picks up the new items automatically.

Dataset split counts (v0.1.0):
- MSA: 32
- Gulf: 10
- Levantine: 4
- Egyptian: 3
- Maghrebi: 2

Non-MSA dialects are undersampled; per-dialect rewards should be read as directional signal, not as final rankings. Growing the dialects is tracked in [arabic-agent-eval issues](https://github.com/Moshe-ship/arabic-agent-eval/issues).

## Files

- `env.py` — `ArabicToolCallingEnv(_UpstreamBaseEnv)`. Has a late-binding import shim so syntax checks pass without the atropos package installed.
- `tasks.py` — task sampling, dialect personas, tool registry builder.
- `scoring.py` — `RewardWeights`, `reward_from_score`, `score_turn`.
- `__init__.py` — exports.
- `requirements.txt` — pins arabic-agent-eval.

## Integration with MTG (optional)

When [mtg](https://github.com/Moshe-ship/mtg) is installed, callers can wrap agent tool calls with MTG guards before scoring to capture violation rates alongside rewards. The env itself does not hard-depend on mtg.

## License

Apache-2.0 (matches the parent atropos repo convention). Data redistributed from arabic-agent-eval under CC-BY-4.0.

## Citation

```bibtex
@software{arabic_agent_eval_2026,
  title = {Arabic Agent Eval: The first Arabic function-calling benchmark with dialect splits},
  author = {Abumazin, Mousa},
  year = {2026},
  url = {https://github.com/Moshe-ship/arabic-agent-eval}
}
```
