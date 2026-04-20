#!/usr/bin/env python3
"""One-command bundle builder.

Loads N benchmark-run JSON files, optionally loads a tool-schema map
(from a directory of JSON files or a single file), scans each run
with MTG, writes a canonical bundle, and optionally runs the publish
gate on the result.

Designed as the single obvious command a developer runs after a model
matrix completes. Everything else (Evaluator, providers, Markdown,
HTML) is expected to be invoked via this entrypoint or its library
equivalents.

Usage:

    # Heuristic-only bundle (will mark every row diagnostic)
    python scripts/build_bundle.py --run runs/hermes.json --out bundles/heuristic/

    # Schema-bound bundle using Hurmoz's annotated tool schemas
    python scripts/build_bundle.py \\
        --run runs/hermes-70b.json --run runs/claude.json \\
        --schemas ../hurmoz/tool-schemas \\
        --out bundles/2026-04-hermes-vs-claude/ \\
        --gate

`--schemas` accepts either a directory (loads every `*.json` file
inside) or a single JSON file. Each file is expected to be a
tool-schema dict with a `name` key; they're indexed by name into a
map passed to `scan_with_schemas`.

`--gate` runs `scripts/check_publish_ready.py` on the produced bundle
and exits non-zero if the gate fails. Use this in CI or as a
pre-commit check.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Allow running directly from a source checkout.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arabic_agent_eval.bundle import write_bundle  # noqa: E402
from arabic_agent_eval.matrix import (  # noqa: E402
    build_matrix,
    load_benchmark_result_from_json,
    render_markdown,
)

BUILD_BUNDLE_VERSION = "build_bundle/0.2"


def _git_ref() -> tuple[str, str]:
    """Return (git_ref, git_branch) for the current checkout, or ("", "")
    if not in a git repo or git is unavailable."""
    try:
        ref = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "", ""
    return ref, branch


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_schema_map(path: Path) -> dict[str, dict]:
    """Load a tool-schema map from a directory of JSON files or a single
    JSON file. Each schema dict must have a `name` field; indexed by
    name. Duplicate names raise."""
    schemas: dict[str, dict] = {}
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for entry in data:
                _merge_schema(schemas, entry, path)
        else:
            _merge_schema(schemas, data, path)
    elif path.is_dir():
        for f in sorted(path.glob("*.json")):
            try:
                entry = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{f}: malformed JSON: {exc}") from exc
            _merge_schema(schemas, entry, f)
    else:
        raise FileNotFoundError(f"schemas path not found: {path}")
    return schemas


def _merge_schema(bucket: dict[str, dict], entry: dict, source: Path) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"{source}: expected an object, got {type(entry).__name__}")
    # Support OpenAI `{name, ...}`, Anthropic `{name, input_schema}`,
    # and wrapped `{type: "function", function: {name, ...}}`.
    if "function" in entry and isinstance(entry["function"], dict):
        inner = entry["function"]
        name = inner.get("name")
    else:
        name = entry.get("name")
    if not name:
        raise ValueError(f"{source}: schema missing `name` field")
    if name in bucket:
        raise ValueError(f"duplicate tool schema for `{name}` in {source}")
    bucket[name] = entry


def _try_build_html(matrix) -> str | None:
    """Render an HTML scorecard via mtg.report, if mtg is installed.

    The scorecard aggregates receipt chains, not matrix rows, so this
    is best-effort: we build a synthetic aggregate from the matrix so
    users get a shareable single-page HTML in the bundle.
    """
    try:
        from mtg.report import Scorecard, render_html
    except ImportError:
        return None
    # Build a minimal Scorecard that reflects the matrix numbers.
    # Intentionally lightweight — the detailed view lives in table.md.
    card = Scorecard()
    for row in matrix.rows:
        outcome = "fail" if row.violation_rate > 0.5 else "pass"
        card._record({
            "tool_name": f"{row.provider}/{row.model}",
            "outcome": outcome,
            "mtg_violations": [],
        })
    return render_html(card)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build a canonical matrix bundle from benchmark runs."
    )
    p.add_argument(
        "--run", action="append", required=True, type=Path,
        help="Path to a BenchmarkResult JSON. Pass multiple times.",
    )
    p.add_argument(
        "--schemas", type=Path, default=None,
        help="Path to a tool-schema map: directory of *.json or single file. "
             "When omitted, every arg falls to the heuristic spec (bundle "
             "will be diagnostic).",
    )
    p.add_argument(
        "--out", type=Path, required=True,
        help="Output bundle directory.",
    )
    p.add_argument(
        "--html", action="store_true",
        help="Also render scorecard.html via mtg.report (requires mtg-guards).",
    )
    p.add_argument(
        "--gate", action="store_true",
        help="Run check_publish_ready.py on the bundle after writing. "
             "Non-zero exit if the gate fails.",
    )
    p.add_argument(
        "--gate-allow-diagnostic", action="store_true",
        help="Pass --allow-diagnostic to the gate (use when publishing a "
             "deliberately-diagnostic run).",
    )
    p.add_argument(
        "--allow-dirty", action="store_true",
        help="Allow building from a dirty git checkout. Unsafe — the "
             "code_sha won't match the worktree that produced the "
             "bundle. Stamped into invocation.overrides so the "
             "relaxation is auditable.",
    )
    p.add_argument(
        "--allow-no-schemas", action="store_true",
        help="Allow building without --schemas. Every arg will fall to "
             "heuristic spec resolution and every row will be diagnostic.",
    )
    p.add_argument(
        "--allow-diagnostic", action="store_true",
        help="Allow bundle to contain diagnostic rows. Without this "
             "flag, build_bundle.py refuses to write a bundle where "
             "any row is diagnostic.",
    )
    p.add_argument(
        "--allow-missing-shas", action="store_true",
        help="Allow building when required code_shas are empty (e.g. "
             "running from a pip-installed package, not a git checkout).",
    )
    p.add_argument(
        "--synthetic", action="store_true",
        help="Build a synthetic example bundle. Stamps "
             "`invocation.synthetic = true` and waives the dirty-tree, "
             "missing-SHA, and diagnostic-row checks (synthetic bundles "
             "don't describe real model calls). The publish gate will "
             "later require its own --synthetic flag to match.",
    )
    args = p.parse_args()

    # Strict default: refuse ambiguous provenance. Each override flag
    # flips a check off and gets recorded in invocation.overrides.
    refusals: list[str] = []
    overrides: list[str] = []

    schema_map = None
    if args.schemas is not None:
        schema_map = _load_schema_map(args.schemas)
        print(f"loaded {len(schema_map)} tool schemas from {args.schemas}")
    else:
        if args.allow_no_schemas:
            overrides.append("allow_no_schemas")
        else:
            refusals.append(
                "no --schemas provided — every row would be diagnostic. "
                "Pass an annotated tool-schema map or --allow-no-schemas "
                "to acknowledge the diagnostic-only build."
            )

    # Dirty-tree check: build from aae's worktree (where this script
    # lives). Synthetic bundles waive this — they don't describe a
    # real model call, so pinning a clean SHA isn't load-bearing.
    from arabic_agent_eval.matrix import _git_clean_for_package  # reuse helper
    clean = _git_clean_for_package("arabic_agent_eval")
    if clean is False and not args.synthetic:
        if args.allow_dirty:
            overrides.append("allow_dirty")
        else:
            refusals.append(
                "arabic_agent_eval repo is dirty — code_sha won't match the "
                "worktree that produced the bundle. Commit/stash your changes "
                "or pass --allow-dirty to acknowledge the drift."
            )

    benchmarks = []
    for run in args.run:
        if not run.exists():
            print(f"error: run file not found: {run}", file=sys.stderr)
            return 2
        benchmarks.append(load_benchmark_result_from_json(run))
    print(f"loaded {len(benchmarks)} benchmark runs")

    matrix = build_matrix(benchmarks, tool_schema_map=schema_map)

    # Post-build provenance checks, done on the actual matrix rows.
    # Synthetic bundles waive the diagnostic-rows + missing-SHA checks
    # (synthetic examples frequently use degenerate inputs).
    diagnostic_rows = [r for r in matrix.rows if r.diagnostic]
    if diagnostic_rows and not args.synthetic:
        if args.allow_diagnostic:
            overrides.append("allow_diagnostic")
        else:
            labels = ", ".join(f"{r.provider}/{r.model}" for r in diagnostic_rows)
            refusals.append(
                f"{len(diagnostic_rows)} diagnostic row(s): {labels}. "
                f"Pass --allow-diagnostic to publish them anyway."
            )

    missing_shas: list[str] = []
    for row in matrix.rows:
        code_shas = (row.run_metadata or {}).get("code_shas") or {}
        if not code_shas.get("arabic_agent_eval"):
            missing_shas.append(f"{row.provider}/{row.model}:arabic_agent_eval")
    if missing_shas and not args.synthetic:
        if args.allow_missing_shas:
            overrides.append("allow_missing_shas")
        else:
            refusals.append(
                "missing required code_shas: " + ", ".join(missing_shas) +
                ". Pass --allow-missing-shas to build anyway "
                "(the bundle's code provenance will be weak)."
            )

    if refusals:
        print("build_bundle.py refuses to build:", file=sys.stderr)
        for r in refusals:
            print(f"  - {r}", file=sys.stderr)
        return 3

    html = _try_build_html(matrix) if args.html else None

    # Invocation provenance — freeze the exact build-command context so
    # reviewers can reproduce this bundle from scratch.
    git_ref, git_branch = _git_ref()
    run_shas: dict[str, str] = {
        run.name: _sha256_file(run) for run in args.run
    }
    invocation: dict[str, Any] = {
        "generator": "scripts/build_bundle.py",
        "generator_version": BUILD_BUNDLE_VERSION,
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_ref": git_ref,
        "git_branch": git_branch,
        "schema_map_source": str(args.schemas) if args.schemas else "<none>",
        "run_json_sha256": run_shas,
        "html_requested": bool(args.html),
        "gate_invoked": bool(args.gate),
        "gate_allow_diagnostic": bool(args.gate_allow_diagnostic),
        "overrides": sorted(overrides),
    }
    if args.synthetic:
        invocation["synthetic"] = True

    bundle_path = write_bundle(
        matrix,
        args.out,
        html=html,
        run_json_files=list(args.run),
        invocation=invocation,
    )
    print(f"wrote bundle → {bundle_path}")

    # Also print a short summary so the user knows what just happened
    # without opening table.md.
    for row in matrix.rows:
        flag = "⚠ DIAGNOSTIC" if row.diagnostic else "clean"
        print(
            f"  {row.provider}/{row.model}: {flag} · "
            f"schema_bound={row.schema_bound_rate:.0%} · "
            f"heuristic={row.heuristic_scan_rate:.0%} · "
            f"baseline={row.baseline_score:.3f}"
        )

    if args.gate:
        gate_script = _REPO_ROOT / "scripts" / "check_publish_ready.py"
        gate_args = [sys.executable, str(gate_script), str(bundle_path)]
        if args.gate_allow_diagnostic:
            gate_args.append("--allow-diagnostic")
        if args.synthetic:
            gate_args.append("--synthetic")
        if args.allow_dirty:
            gate_args.append("--allow-dirty")
        print(f"running publish gate: {' '.join(gate_args)}")
        return subprocess.call(gate_args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
