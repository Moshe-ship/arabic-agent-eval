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
import json
import subprocess
import sys
from pathlib import Path

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
    args = p.parse_args()

    schema_map = None
    if args.schemas is not None:
        schema_map = _load_schema_map(args.schemas)
        print(f"loaded {len(schema_map)} tool schemas from {args.schemas}")

    benchmarks = []
    for run in args.run:
        if not run.exists():
            print(f"error: run file not found: {run}", file=sys.stderr)
            return 2
        benchmarks.append(load_benchmark_result_from_json(run))
    print(f"loaded {len(benchmarks)} benchmark runs")

    matrix = build_matrix(benchmarks, tool_schema_map=schema_map)

    html = _try_build_html(matrix) if args.html else None

    bundle_path = write_bundle(
        matrix,
        args.out,
        html=html,
        run_json_files=list(args.run),
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
        print(f"running publish gate: {' '.join(gate_args)}")
        return subprocess.call(gate_args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
