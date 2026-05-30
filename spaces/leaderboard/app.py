"""Arabic Agent Eval — leaderboard Space.

Renders the dated, provenance-frozen result bundles as a leaderboard. Data
lives in leaderboard.json (built from the committed bundles in the
arabic-agent-eval repo). Adding more models is open work — see the repo.
"""

import json
from pathlib import Path

import gradio as gr
import pandas as pd

LB = json.loads(Path(__file__).with_name("leaderboard.json").read_text())


def _df(rows):
    df = pd.DataFrame(rows)
    df.insert(0, "rank", range(1, len(df) + 1))
    df["score (95% CI)"] = df.apply(
        lambda r: f"{r['score']:.3f} ({r['ci_low']:.3f}–{r['ci_high']:.3f})", axis=1
    )
    df["bidi viol %"] = (df["bidi_violation_rate"] * 100).round(1)
    df["dialect drift %"] = (df["dialect_drift_rate"] * 100).round(1)
    return df[["rank", "model", "provider", "score (95% CI)", "bidi viol %", "dialect drift %"]]


INTRO = f"""
# Arabic Agent Eval — Leaderboard

An open, installable, dialect-split Arabic function-calling benchmark.
Measures whether models select the right tool, extract arguments from natural
Arabic, **keep Arabic text in arguments instead of transliterating**, and
honor dialectal framing.

**Provenance:** {LB['provenance']}
**Bundles:** {', '.join(LB['generated_from'])} · updated {LB['updated']}

- Code & data: https://github.com/Moshe-ship/arabic-agent-eval
- Dataset: https://huggingface.co/datasets/Mosescreates/arabic-agent-eval

> Complementary to the arXiv:2601.05101 line of work ("Arabic Prompts with
> English Tools"); this benchmark uses native Arabic functions and dialect splits.
"""

with gr.Blocks(title="Arabic Agent Eval — Leaderboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)
    with gr.Tab("Clean (51 items)"):
        gr.Markdown("Tool-call completion across 6 categories.")
        gr.Dataframe(_df(LB["clean"]), interactive=False, wrap=True)
    with gr.Tab("Adversarial (24 items)"):
        gr.Markdown(
            "Arabic guard surface: BiDi / homoglyph / UTS #39 / Arabizi / "
            "prompt-injection-in-argument / canonicalization / dialect pressure. "
            "Lower scores are expected — this set is hard on purpose."
        )
        gr.Dataframe(_df(LB["adversarial"]), interactive=False, wrap=True)

if __name__ == "__main__":
    demo.launch()
