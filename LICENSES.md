# Licenses

This repository ships under a two-license split to match conventions for open benchmarks:

| Scope | License | File |
|---|---|---|
| **Code** — `arabic_agent_eval/`, `tests/`, `scripts/`, top-level Python and tooling | Apache-2.0 | [LICENSE](LICENSE) |
| **Data** — `data/`, `dataset_card.md`, `docs/` | CC-BY-4.0 | [data/LICENSE](data/LICENSE) |

## Why two licenses

- **Apache-2.0** for code — compatible with Hugging Face, NousResearch, and most upstream agent frameworks. Includes a patent grant which is the modern open-source norm for AI tooling.
- **CC-BY-4.0** for data — the HF-recommended license for datasets. Allows redistribution and remixing with attribution, including commercial use, which is the usual expectation for benchmark datasets.

## Attribution

If you use this benchmark or its data, cite as:

```bibtex
@software{arabic_agent_eval_2026,
  title = {Arabic Agent Eval: The first Arabic function-calling benchmark with dialect splits},
  author = {Abumazin, Mousa},
  year = {2026},
  url = {https://github.com/Moshe-ship/arabic-agent-eval},
  license = {Apache-2.0 (code) / CC-BY-4.0 (data)}
}
```

## Derivative work

You are free to fork, modify, extend, and redistribute under the respective license terms. Contributions back to this repository are welcome.
