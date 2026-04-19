#!/usr/bin/env python3
"""Publish gate for canonical matrix bundles.

Validates that a bundle is clean enough to publish:

- Manifest integrity (every listed file sha256 matches)
- scanner_version pinned and non-empty
- Every row has run_metadata (run_id, scanned_at, provider, model)
- Bootstrap CIs present when n_items >= 2 — within-run variance at
  minimum
- heuristic_scan_rate <= threshold on every row (default 0.10)
  unless --allow-diagnostic is passed

Exits 0 on pass. Exits non-zero with one human-readable reason per
violation on fail. Used by CI to block weak result bundles and by
humans as a pre-commit check before publishing.

Usage:

    python scripts/check_publish_ready.py bundle/
    python scripts/check_publish_ready.py bundle/ --allow-diagnostic
    python scripts/check_publish_ready.py bundle/ --heuristic-max 0.25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script directly from a repo checkout without
# `pip install`. Prepends the repo root so `arabic_agent_eval` imports.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arabic_agent_eval.bundle import BundleError, DEFAULT_THRESHOLDS, load_bundle  # noqa: E402


REQUIRED_METADATA_KEYS = ("run_id", "scanned_at", "provider", "model", "scanner_version")


def check_bundle(
    path: Path,
    *,
    heuristic_max: float,
    allow_diagnostic: bool,
) -> list[str]:
    """Return a list of failure reasons. Empty list == publish-ready."""
    reasons: list[str] = []
    try:
        manifest, matrix = load_bundle(path)
    except BundleError as exc:
        return [f"bundle integrity: {exc}"]

    if not manifest.scanner_version:
        reasons.append(
            "scanner_version is empty in MANIFEST.json — rows either "
            "disagreed on version or lacked the pin entirely; re-scan "
            "with a single scanner_version to produce a publishable bundle"
        )

    rows = matrix.get("rows") or []
    if not rows:
        reasons.append("matrix.json contains no rows")
        return reasons

    for i, row in enumerate(rows):
        label = f"rows[{i}] ({row.get('provider', '?')}/{row.get('model', '?')})"

        md = row.get("run_metadata") or {}
        missing = [k for k in REQUIRED_METADATA_KEYS if not md.get(k)]
        if missing:
            reasons.append(
                f"{label}: run_metadata missing {missing} — every "
                f"published row must trace back to a run_id + timestamp"
            )

        hsr = float(row.get("heuristic_scan_rate", 0.0))
        if hsr > heuristic_max:
            if allow_diagnostic:
                if not row.get("diagnostic"):
                    reasons.append(
                        f"{label}: heuristic_scan_rate={hsr:.2%} exceeds "
                        f"{heuristic_max:.0%} but `diagnostic` flag is false; "
                        f"re-scan so the marker is stamped in the manifest"
                    )
                # Else: allow-diagnostic + flag set → accepted
            else:
                reasons.append(
                    f"{label}: heuristic_scan_rate={hsr:.2%} exceeds "
                    f"threshold {heuristic_max:.0%} — bundle is not "
                    f"schema-grounded enough to publish as a clean "
                    f"result. Re-scan with an x-mtg-annotated tool-schema "
                    f"map, or re-run the gate with --allow-diagnostic"
                )

        n_items = int((row.get("run_metadata") or {}).get("n_items") or row.get("total_items") or 0)
        if n_items >= 2:
            if row.get("baseline_ci_95") is None:
                reasons.append(
                    f"{label}: baseline_ci_95 is null despite n_items={n_items}; "
                    f"bootstrap should have produced a within-run CI"
                )
            if row.get("repaired_ci_95") is None:
                reasons.append(
                    f"{label}: repaired_ci_95 is null despite n_items={n_items}"
                )

        # Manifest row_summary should agree with matrix.json
        summary = next(
            (
                s for s in manifest.row_summaries
                if s.get("provider") == row.get("provider")
                and s.get("model") == row.get("model")
            ),
            None,
        )
        if summary is None:
            reasons.append(
                f"{label}: no matching entry in MANIFEST.json row_summaries"
            )
        else:
            manifest_hsr = summary.get("heuristic_scan_rate")
            if manifest_hsr is not None and abs(float(manifest_hsr) - hsr) > 1e-6:
                reasons.append(
                    f"{label}: manifest heuristic_scan_rate={manifest_hsr} "
                    f"disagrees with matrix.json {hsr}; bundle was written "
                    f"partially or mutated after write"
                )

    return reasons


def main() -> int:
    p = argparse.ArgumentParser(description="Publish gate for MTG result bundles")
    p.add_argument("bundle", type=Path, help="Path to bundle directory")
    p.add_argument(
        "--heuristic-max", type=float,
        default=DEFAULT_THRESHOLDS["heuristic_scan_rate_max"],
        help=f"Max acceptable heuristic_scan_rate (default: "
             f"{DEFAULT_THRESHOLDS['heuristic_scan_rate_max']:.2f})",
    )
    p.add_argument(
        "--allow-diagnostic", action="store_true",
        help="Allow rows with heuristic_scan_rate above threshold if "
             "their diagnostic flag is set. Use when publishing a "
             "deliberately-diagnostic run.",
    )
    args = p.parse_args()

    reasons = check_bundle(
        args.bundle,
        heuristic_max=args.heuristic_max,
        allow_diagnostic=args.allow_diagnostic,
    )
    if not reasons:
        print(f"PUBLISH_READY: {args.bundle}")
        return 0

    print(f"NOT_PUBLISH_READY: {args.bundle}", file=sys.stderr)
    for r in reasons:
        print(f"  - {r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
