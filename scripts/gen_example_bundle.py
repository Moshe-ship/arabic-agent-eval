#!/usr/bin/env python3
"""Regenerate the committed synthetic example bundle.

Produces `examples/synthetic_bundle/` — a fully-formed bundle that
passes the publish gate, built from fabricated "model output" that
happens to be realistic Arabic content. NOT a result. The bundle
exists so developers have something to point tooling at and inspect
the format without running a real matrix.

Run this script any time the bundle format or schema-coverage logic
changes — the committed bundle will stay in sync.

Usage:

    python scripts/gen_example_bundle.py

Writes to `examples/synthetic_bundle/` (relative to this repo root).
The README.md in that directory is preserved; everything else is
rewritten.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arabic_agent_eval.bundle import write_bundle  # noqa: E402
from arabic_agent_eval.dataset import EvalItem, ExpectedCall  # noqa: E402
from arabic_agent_eval.evaluator import BenchmarkResult, EvalResult  # noqa: E402
from arabic_agent_eval.matrix import build_matrix  # noqa: E402
from arabic_agent_eval.scoring import Score  # noqa: E402


HURMOZ_SCHEMAS_CANDIDATES = [
    _REPO_ROOT.parent / "hurmoz" / "tool-schemas",
]


def _load_hurmoz_schemas() -> dict[str, dict]:
    for p in HURMOZ_SCHEMAS_CANDIDATES:
        if p.is_dir():
            return _load_dir(p)
    raise FileNotFoundError(
        "Hurmoz tool-schemas not found; clone Moshe-ship/hurmoz as a "
        "sibling of this repo or set HURMOZ_SCHEMAS env variable"
    )


def _load_dir(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in sorted(path.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        name = data.get("name")
        if not name:
            continue
        out[name] = data
    return out


def _result(
    item_id: str,
    function: str,
    args: dict,
    category: str = "simple_function_calling",
    dialect: str = "gulf",
    score_total: float = 1.0,
) -> EvalResult:
    item = EvalItem(
        id=item_id, category=category, instruction="synthetic",
        dialect=dialect, available_functions=[function],
        expected_calls=[ExpectedCall(function=function, arguments=args)],
        difficulty="easy",
    )
    return EvalResult(
        item=item,
        score=Score(
            item_id=item_id, category=category,
            function_selection=score_total, argument_accuracy=score_total,
            arabic_preservation=score_total,
        ),
        actual_calls=[{"function": function, "arguments": args}],
    )


def build_synthetic_benchmark() -> BenchmarkResult:
    """Five synthetic items against Hurmoz-annotated tool schemas.

    Realistic Arabic content for each call so schema-bound replay
    produces non-trivial numbers. Mix of clean + one mild dialect
    mismatch so the taxonomy / CIs / layer rates have signal.
    """
    results = [
        # Clean Gulf
        _result(
            "synth_001", "send_message_gulf",
            {"recipient": "أحمد", "platform": "whatsapp",
             "message": "أبي أحجز فندق في دبي"},
            dialect="gulf", score_total=1.0,
        ),
        # Clean MSA Quran search
        _result(
            "synth_002", "find_quran_verse",
            {"query": "الرحمن الرحيم", "surah": 1},
            category="simple_function_calling", dialect="msa", score_total=1.0,
        ),
        # Saudi address — clean mixed-script
        _result(
            "synth_003", "lookup_saudi_address",
            {"postal_code": "12345", "additional_number": "6789",
             "city": "الرياض"},
            dialect="msa", score_total=1.0,
        ),
        # Egyptian dialect drift in a Gulf-bound slot
        _result(
            "synth_004", "send_message_gulf",
            {"recipient": "محمد", "platform": "telegram",
             "message": "عايز أبعت رسالة دلوقتي"},
            dialect="egy", score_total=0.5,
        ),
        # Clean prayer-times
        _result(
            "synth_005", "get_prayer_times",
            {"city": "مكة", "country": "SA"},
            dialect="msa", score_total=1.0,
        ),
    ]
    return BenchmarkResult(
        provider="example",
        model="synthetic-v1",
        results=results,
    )


def _git_ref() -> tuple[str, str]:
    """Return (ref, branch) for the current checkout, or ("", "") if
    git isn't available."""
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


def main() -> int:
    out_dir = _REPO_ROOT / "examples" / "synthetic_bundle"
    schemas = _load_hurmoz_schemas()
    print(f"loaded {len(schemas)} tool schemas from Hurmoz")

    br = build_synthetic_benchmark()
    matrix = build_matrix([br], tool_schema_map=schemas)

    # Preserve README.md during bundle write (write_bundle only
    # overwrites files it knows about).
    readme = out_dir / "README.md"
    readme_content = readme.read_text(encoding="utf-8") if readme.exists() else None

    # Synthetic invocation so the committed example shows the shape.
    # Clearly labeled as synthetic — no run_json_sha256 because there
    # are no real runs.
    git_ref, git_branch = _git_ref()
    invocation = {
        "generator": "scripts/gen_example_bundle.py",
        "generator_version": "gen_example_bundle/0.2",
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_ref": git_ref,
        "git_branch": git_branch,
        "schema_map_source": "../hurmoz/tool-schemas",
        "synthetic": True,
        "note": "Fabricated bundle for format demonstration — NOT a model result.",
    }

    bundle = write_bundle(matrix, out_dir, invocation=invocation)
    if readme_content is not None:
        readme.write_text(readme_content, encoding="utf-8")

    row = matrix.rows[0]
    flag = "⚠ DIAGNOSTIC" if row.diagnostic else "clean"
    print(
        f"wrote synthetic bundle → {bundle}\n"
        f"  {row.provider}/{row.model}: {flag} · "
        f"schema_bound={row.schema_bound_rate:.0%} · "
        f"heuristic={row.heuristic_scan_rate:.0%} · "
        f"baseline={row.baseline_score:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
