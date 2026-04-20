#!/usr/bin/env python3
"""Assemble the agentic-tier bundle from k-replay episode runs.

Mirrors the shape of scripts/build_bundle.py but works on the
agentic run format (runs/episodes/*-k3.json). Produces:

  bundles/<name>/
    MANIFEST.json  — file SHAs, git ref, schema fingerprint
    table.md       — per-model pass^k + per-episode breakdown
    runs/*.json    — copied input runs (verbatim)

Deliberately minimal. The agentic tier uses different scoring axes
(per-episode outcome: pass / fail / abstain / error) than the
standard benchmark (per-arg score), so it publishes as its own
bundle rather than trying to fold into the main matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256(path.read_bytes())


def _git_ref() -> tuple[str, str]:
    try:
        ref = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip()
    except Exception:
        return "", ""
    return ref, branch


def _dataset_fingerprint(path: Path) -> str:
    return _sha256(path.read_bytes())


def _render_table(rows: list[dict], episodes: list[str]) -> str:
    """Render the per-model pass^k table + per-episode grid."""
    out: list[str] = []
    out.append("## Agentic tier — pass^k across 6 scripted episodes")
    out.append("")
    out.append(
        "`pass^k` = fraction of episodes where ALL k replays passed "
        "(τ-bench style — rewards consistency). `avg_pass` = fraction "
        "of individual replays that passed. Abstention counts as pass."
    )
    out.append("")
    out.append("| rank | provider | model | k | pass^k | avg_pass | episodes_passed |")
    out.append("|---:|---|---|---:|---:|---:|---|")
    ranked = sorted(rows, key=lambda r: (-r["pass_k_rate"], -r["avg_pass_rate"]))
    for i, r in enumerate(ranked, 1):
        out.append(
            f"| {i} | {r['provider']} | {r['model']} | {r['k']} | "
            f"{round(r['pass_k_rate']*100,1)}% | "
            f"{round(r['avg_pass_rate']*100,1)}% | "
            f"{r['episodes_passed']}/{r['n_episodes']} |"
        )
    out.append("")
    out.append("## Per-episode pass matrix")
    out.append("")
    out.append(
        "Each cell shows replays-passed / k for that episode. "
        "Abstention episodes pass by refusing the harmful request."
    )
    out.append("")
    out.append(
        "| model | " + " | ".join(episodes) + " |"
    )
    out.append(
        "|---|" + "|".join(["---:"] * len(episodes)) + "|"
    )
    for r in ranked:
        cells = []
        per_ep = r["per_episode"]
        for ep_id in episodes:
            info = per_ep.get(ep_id) or {}
            outcomes = info.get("outcomes") or []
            passes = sum(1 for o in outcomes if o in ("pass", "abstain"))
            mark = "✓" if info.get("all_passed") else " "
            cells.append(f"{passes}/{r['k']} {mark}")
        out.append(f"| {r['model']} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="Path to one *-k3.json agentic run. Repeat.")
    ap.add_argument("--dataset", type=Path,
                    default=REPO_ROOT / "data" / "episodes.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(exist_ok=True)

    rows: list[dict] = []
    file_shas: dict[str, str] = {}
    run_paths_used: list[str] = []

    for r_path in args.run:
        p = Path(r_path)
        d = json.loads(p.read_text(encoding="utf-8"))
        row = {
            "provider": d.get("provider"),
            "model": d.get("model"),
            "model_id": d.get("model_id") or d.get("model"),
            "provider_base_url": d.get("provider_base_url"),
            "k": d.get("k"),
            "pass_k_rate": d.get("pass_k_rate", 0.0),
            "avg_pass_rate": d.get("avg_pass_rate", 0.0),
            "n_episodes": d.get("n_episodes", 0),
            "episodes_passed": sum(
                1 for e in (d.get("per_episode") or {}).values()
                if e.get("all_passed")
            ),
            "per_episode": d.get("per_episode") or {},
            "run_metadata": d.get("run_metadata") or {},
        }
        rows.append(row)
        dest = out_dir / "runs" / p.name
        shutil.copy(p, dest)
        file_shas[f"runs/{p.name}"] = _sha256_file(dest)
        run_paths_used.append(p.name)

    # Episode order from dataset
    with args.dataset.open(encoding="utf-8") as f:
        episodes = [json.loads(l)["id"] for l in f if l.strip()]

    table_md = _render_table(rows, episodes)
    (out_dir / "table.md").write_text(table_md, encoding="utf-8")
    file_shas["table.md"] = _sha256_file(out_dir / "table.md")

    git_ref, git_branch = _git_ref()
    manifest = {
        "bundle_version": "agentic-1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": file_shas,
        "has_runs": True,
        "invocation": {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "generator": "scripts/build_agentic_bundle.py",
            "generator_version": "agentic_bundle/0.1",
            "git_branch": git_branch,
            "git_ref": git_ref,
            "run_json_sha256": {p: _sha256_file(out_dir / "runs" / p) for p in run_paths_used},
            "dataset": args.dataset.name,
            "dataset_fingerprint": _dataset_fingerprint(args.dataset),
        },
        "n_episodes": len(episodes),
        "n_models": len(rows),
        "tier": "agentic",
        "pass_k": rows[0]["k"] if rows else None,
        "rows": rows,
    }
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"wrote agentic bundle → {out_dir}")
    for r in rows:
        print(
            f"  {r['provider']}/{r['model']}: "
            f"pass^{r['k']}={round(r['pass_k_rate']*100,1)}%  "
            f"avg_pass={round(r['avg_pass_rate']*100,1)}%  "
            f"({r['episodes_passed']}/{r['n_episodes']} episodes consistent)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
