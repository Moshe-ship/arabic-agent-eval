"""Generate a static index.html leaderboard from leaderboard.json.

Static = no runtime, no dependencies, no build failures. The leaderboard is
read-only display, so it does not need gradio. Run:

    python build_static.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

HERE = Path(__file__).parent
LB = json.loads((HERE / "leaderboard.json").read_text())


def _rows(rows: list[dict]) -> str:
    out = []
    for i, r in enumerate(rows, 1):
        out.append(
            "<tr>"
            f"<td class='r'>{i}</td>"
            f"<td>{html.escape(r['model'])}</td>"
            f"<td class='muted'>{html.escape(r.get('provider',''))}</td>"
            f"<td class='r'>{r['score']:.3f}</td>"
            f"<td class='r muted'>{r['ci_low']:.3f}–{r['ci_high']:.3f}</td>"
            f"<td class='r'>{r['bidi_violation_rate']*100:.1f}%</td>"
            f"<td class='r'>{r['dialect_drift_rate']*100:.1f}%</td>"
            "</tr>"
        )
    return "\n".join(out)


def _table(title: str, note: str, rows: list[dict]) -> str:
    head = (
        "<tr><th class='r'>#</th><th>model</th><th>provider</th>"
        "<th class='r'>score</th><th class='r'>95% CI</th>"
        "<th class='r'>bidi viol</th><th class='r'>dialect drift</th></tr>"
    )
    return (
        f"<h2>{html.escape(title)}</h2><p class='muted'>{html.escape(note)}</p>"
        f"<table>{head}{_rows(rows)}</table>"
    )


HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(LB['benchmark'])} — Leaderboard</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.6 ui-monospace, "SF Mono", Menlo, monospace;
         max-width: 940px; margin: 3rem auto; padding: 0 1.2rem;
         background: #fff; color: #111; }}
  @media (prefers-color-scheme: dark) {{ body {{ background:#0c0c0c; color:#eaeaea; }} }}
  h1 {{ font-size: 1.5rem; margin-bottom: .2rem; }}
  h2 {{ font-size: 1.05rem; margin-top: 2.2rem; border-bottom: 1px solid #8884; padding-bottom: .3rem; }}
  a {{ color: inherit; }}
  .muted {{ opacity: .6; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: .6rem; }}
  th, td {{ text-align: left; padding: .35rem .6rem; border-bottom: 1px solid #8882; }}
  th {{ font-weight: 600; opacity: .8; }}
  td.r, th.r {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr:first-child td {{ font-weight: 600; }}
  footer {{ margin-top: 2.5rem; font-size: .85rem; }}
</style>
</head>
<body>
  <h1>{html.escape(LB['benchmark'])} — Leaderboard</h1>
  <p>An open, installable, dialect-split Arabic function-calling benchmark.
     Measures tool selection, argument extraction, <b>keeping Arabic in arguments
     instead of transliterating</b>, and dialectal framing.</p>
  <p class="muted">{html.escape(LB['provenance'])}<br>
     Bundles: {html.escape(', '.join(LB['generated_from']))} · updated {html.escape(LB['updated'])}</p>

  {_table("Clean (51 items)", "Tool-call completion across 6 categories.", LB['clean'])}
  {_table("Adversarial (24 items)",
          "Arabic guard surface: BiDi / homoglyph / UTS #39 / Arabizi / injection / "
          "canonicalization / dialect pressure. Lower is expected — hard on purpose.",
          LB['adversarial'])}

  <footer class="muted">
    Code &amp; data: <a href="https://github.com/Moshe-ship/arabic-agent-eval">github.com/Moshe-ship/arabic-agent-eval</a> ·
    Dataset: <a href="https://huggingface.co/datasets/Mosescreates/arabic-agent-eval">huggingface.co/datasets/Mosescreates/arabic-agent-eval</a><br>
    Complementary to arXiv:2601.05101 ("Arabic Prompts with English Tools"); this benchmark uses native Arabic functions and dialect splits.
    Adding more models — including Hermes via a native endpoint — is open work.
  </footer>
</body>
</html>
"""

(HERE / "index.html").write_text(HTML, encoding="utf-8")
print(f"wrote index.html ({len(HTML)} bytes): {len(LB['clean'])} clean + {len(LB['adversarial'])} adversarial rows")
