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


REQUIRED_METADATA_KEYS = (
    "run_id", "scanned_at", "provider", "model", "scanner_version",
    "dataset_fingerprint", "dataset_version",
)

# Every MTG-stack package that was importable at scan time must carry
# a non-empty git SHA. Packages not importable produce None in
# code_shas and are considered absent (not failures).
REQUIRED_CODE_SHA_PACKAGES = ("arabic_agent_eval",)


def check_bundle(
    path: Path,
    *,
    heuristic_max: float,
    allow_diagnostic: bool,
    allow_no_runs: bool = False,
    allow_dirty: bool = False,
    synthetic: bool = False,
    min_non_diagnostic: int = 1,
) -> list[str]:
    """Return a list of failure reasons. Empty list == publish-ready.

    Two publish modes:

    - **Real** (default): bundle is a model-evaluation result. Requires
      runs/, clean code_shas, non-diagnostic rows, and pinned
      scanner/dataset versions.
    - **Synthetic** (`--synthetic`): bundle is explicitly marked
      `invocation.synthetic = true` and is a format example, not a
      result. Waives runs/ and clean-tree requirements; still requires
      provenance fields and manifest integrity. The `--synthetic` flag
      MUST match `invocation.synthetic` — gate rejects mismatches to
      block "silently synthetic" bundles.
    """
    reasons: list[str] = []
    try:
        manifest, matrix = load_bundle(path)
    except BundleError as exc:
        return [f"bundle integrity: {exc}"]

    # Synthetic-mode consistency check. Both sides must agree: the
    # caller passed --synthetic AND the manifest declares
    # invocation.synthetic=true. Any mismatch means the bundle is
    # either unmarked or mismarked.
    if manifest.is_synthetic and not synthetic:
        reasons.append(
            "manifest declares `invocation.synthetic: true` but the gate "
            "was not invoked with --synthetic. Either pass --synthetic to "
            "publish this bundle as an example, or remove the synthetic "
            "marker from invocation to publish it as a real result."
        )
    if synthetic and not manifest.is_synthetic:
        reasons.append(
            "--synthetic was passed but the manifest does not declare "
            "`invocation.synthetic: true`. The synthetic marker must be "
            "stamped into the bundle at build time — add `synthetic: true` "
            "to the invocation dict passed to write_bundle."
        )

    if not manifest.scanner_version:
        reasons.append(
            "scanner_version is empty in MANIFEST.json — rows either "
            "disagreed on version or lacked the pin entirely; re-scan "
            "with a single scanner_version to produce a publishable bundle"
        )

    # runs/ presence — real bundles must carry source run JSONs so the
    # numbers are reproducible. Synthetic bundles waive this (they
    # describe no real model call). `--allow-no-runs` overrides for
    # real bundles when runs are tracked elsewhere.
    has_runs_file = any(
        name.startswith("runs/") for name in manifest.files.keys()
    )
    if not has_runs_file and not allow_no_runs and not synthetic:
        reasons.append(
            "bundle has no `runs/` files — published real bundles must "
            "carry the source run JSONs so numbers are reproducible. Pass "
            "--allow-no-runs to waive (when source runs are tracked "
            "elsewhere) or --synthetic for example bundles."
        )

    rows = matrix.get("rows") or []
    if not rows:
        reasons.append("matrix.json contains no rows")
        return reasons

    # Diagnostic-row accounting — used after the per-row loop.
    n_non_diagnostic = 0

    for i, row in enumerate(rows):
        label = f"rows[{i}] ({row.get('provider', '?')}/{row.get('model', '?')})"

        md = row.get("run_metadata") or {}
        missing = [k for k in REQUIRED_METADATA_KEYS if not md.get(k)]
        if missing:
            reasons.append(
                f"{label}: run_metadata missing {missing} — every "
                f"published row must trace back to a run_id + timestamp + "
                f"dataset_fingerprint"
            )

        # schema_map_tools is expected in run_metadata — declares which
        # tool schemas the scanner was wired against. Missing means the
        # provenance for schema coverage is unclear.
        if "schema_map_tools" not in md:
            reasons.append(
                f"{label}: run_metadata missing `schema_map_tools` list — "
                f"cannot verify which tool-schema map the scanner used"
            )

        # environment.fingerprint is required on real bundles — it
        # captures the full transitive install tree, so identical
        # fingerprints mean identical environments and divergence
        # points at dep drift. Synthetic bundles waive.
        if not synthetic:
            env = md.get("environment") or {}
            fp = env.get("fingerprint")
            if not fp:
                reasons.append(
                    f"{label}: environment.fingerprint is missing or empty — "
                    f"published real bundles must stamp a full env fingerprint "
                    f"so reproducibility is enforceable, not conventional. "
                    f"Re-scan with mtg-matrix/0.5+ (which writes the field)."
                )

        # provider_provenance — real bundles must carry provider_base_url
        # and model_id so "same model name, different backend settings"
        # is impossible. request_config_fingerprint is recommended but
        # not required (a caller may legitimately not capture config).
        if not synthetic:
            pp = md.get("provider_provenance") or {}
            for required_key in ("provider_base_url", "model_id"):
                if not pp.get(required_key):
                    reasons.append(
                        f"{label}: provider_provenance.{required_key} is "
                        f"missing or empty — published real bundles must "
                        f"pin the backend URL and canonical model ID so "
                        f"same-name/different-settings collisions are "
                        f"detectable. Set it via "
                        f"`setattr(benchmark_result, {required_key!r}, "
                        f"...)` before scanning."
                    )

        # Pair-require request_config_fingerprint ↔ schema_version.
        # Either both are stamped or neither is — a fingerprint without
        # its canonicalization-version pin is ambiguous to compare
        # later. Applies to all bundles (real and synthetic) since the
        # invariant is structural.
        pp_for_pair = md.get("provider_provenance") or {}
        fp = pp_for_pair.get("request_config_fingerprint")
        sv = pp_for_pair.get("request_config_schema_version")
        if (fp and not sv) or (sv and not fp):
            reasons.append(
                f"{label}: request_config_fingerprint and "
                f"request_config_schema_version must be set together or "
                f"not at all (got fingerprint={fp!r}, "
                f"schema_version={sv!r}). A fingerprint without its "
                f"canonicalization-version pin is ambiguous to compare "
                f"across runs."
            )

        # code_shas must carry a non-empty git SHA for every REQUIRED
        # package. Optional packages (mtg, toolproof) are permitted to
        # be None if they weren't importable at scan time, but the
        # REQUIRED set (aae) must always be present.
        code_shas = md.get("code_shas") or {}
        for pkg in REQUIRED_CODE_SHA_PACKAGES:
            sha = code_shas.get(pkg)
            if not sha:
                reasons.append(
                    f"{label}: code_shas is missing a non-empty SHA for "
                    f"`{pkg}` (required) — cannot trace this row back "
                    f"to a specific code revision. Re-run the scan "
                    f"from a git checkout."
                )

        # code_clean — for real bundles, the required packages must
        # have been clean (committed) when the scan ran. A dirty tree
        # means the code_sha is one commit behind what actually ran.
        # Synthetic bundles waive this; real bundles override via
        # --allow-dirty which is surfaced in the reason message.
        if not synthetic and not allow_dirty:
            code_clean = md.get("code_clean") or {}
            for pkg in REQUIRED_CODE_SHA_PACKAGES:
                clean = code_clean.get(pkg)
                if clean is False:
                    reasons.append(
                        f"{label}: code_clean.{pkg} is false — the scan "
                        f"ran from a dirty worktree, so the code_sha "
                        f"doesn't represent the code that actually ran. "
                        f"Commit/stash and re-scan, or pass --allow-dirty "
                        f"to publish anyway."
                    )
                elif clean is None:
                    reasons.append(
                        f"{label}: code_clean.{pkg} is null — cannot "
                        f"verify whether the scan ran from a clean "
                        f"checkout. Pass --allow-dirty if this is "
                        f"deliberate (pip-installed, etc.)."
                    )

        hsr = float(row.get("heuristic_scan_rate", 0.0))
        row_diagnostic = bool(row.get("diagnostic", False))
        if hsr > heuristic_max:
            if allow_diagnostic:
                if not row_diagnostic:
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
        if not row_diagnostic:
            n_non_diagnostic += 1

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

    # Aggregate diagnostic checks. Even with --allow-diagnostic, we
    # never publish a bundle where EVERY row is diagnostic — that is
    # pure heuristic theater and should never hit main.
    if n_non_diagnostic < min_non_diagnostic:
        reasons.append(
            f"only {n_non_diagnostic}/{len(rows)} rows are non-diagnostic; "
            f"publish gate requires at least {min_non_diagnostic}. "
            f"A bundle with zero schema-grounded rows is theater, not a "
            f"result — re-scan with an annotated tool-schema map"
        )

    # Override-asymmetry enforcement for real bundles. If the bundle
    # was BUILT with a weakening override (allow_dirty / allow_diagnostic
    # / allow_no_schemas / allow_missing_shas) but the gate was invoked
    # WITHOUT the matching flag, the publisher is effectively hiding
    # the weakening — the gate's checks pass because they can't see a
    # problem the data already absorbed, not because the bundle is
    # clean. Require the gate to acknowledge every build-time
    # relaxation explicitly, so publishing a real bundle means someone
    # consciously opted into every weakening in play.
    #
    # Synthetic bundles waive: synthetic invocation.overrides is
    # informational; the --synthetic flag itself already acknowledges
    # the waiver.
    if not synthetic:
        build_overrides = set(manifest.invocation.get("overrides") or [])
        gate_active = {
            "allow_diagnostic": allow_diagnostic,
            "allow_no_runs": allow_no_runs,
            "allow_dirty": allow_dirty,
        }
        gate_set = {name for name, on in gate_active.items() if on}
        unreceived = build_overrides - gate_set
        if unreceived:
            reasons.append(
                f"build-time overrides were applied that the gate did "
                f"NOT receive: {sorted(unreceived)}. Re-run the gate with "
                f"the matching flag(s) or strip the override from the "
                f"bundle's invocation. Publishing a real bundle requires "
                f"acknowledging every weakening that was in play."
            )

    return reasons


def _render_override_audit(
    manifest_overrides: list[str],
    gate_flags: dict[str, bool],
) -> list[str]:
    """Produce audit lines describing how build-time overrides compare
    to gate-time flags. The goal is one place to read the full
    relaxation history, so a reviewer knows whether the bundle was
    weakened at build, at gate, or both."""
    lines: list[str] = []
    gate_active = [name for name, active in gate_flags.items() if active]

    if manifest_overrides:
        lines.append(
            f"build-time overrides: {', '.join(sorted(manifest_overrides))} "
            f"(stamped in manifest.invocation.overrides)"
        )
    if gate_active:
        lines.append(
            f"gate-time overrides: {', '.join(sorted(gate_active))} "
            f"(passed on the gate command line)"
        )

    # Flag build-time relaxations the gate isn't seeing. If the bundle
    # was built with allow_dirty but the gate runs without --allow-dirty,
    # the gate's own clean-tree check should still fire (it looks at
    # the manifest's code_clean, which captures the dirty state).
    build_set = set(manifest_overrides)
    gate_set = {name for name, active in gate_flags.items() if active}
    # Normalize names: build-side `allow_dirty` corresponds to gate
    # flag `allow_dirty`. Same tokens.
    asymmetric = build_set - gate_set
    if asymmetric:
        lines.append(
            f"build-time overrides present that the gate did NOT receive: "
            f"{sorted(asymmetric)}. Review whether the bundle's relaxations "
            f"are still acceptable under the current gate run."
        )
    return lines


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
    p.add_argument(
        "--allow-no-runs", action="store_true",
        help="Waive the requirement that bundles carry source run JSONs "
             "under runs/. Use for synthetic / example bundles.",
    )
    p.add_argument(
        "--min-non-diagnostic", type=int, default=1,
        help="Minimum number of non-diagnostic rows required "
             "(default 1). Gate always rejects a bundle with zero "
             "schema-grounded rows.",
    )
    p.add_argument(
        "--allow-dirty", action="store_true",
        help="Allow rows whose code_clean.<required-package> is false. "
             "Use when the scan was intentionally run from a dirty "
             "checkout. Surfaced in reason messages so the override "
             "is auditable.",
    )
    p.add_argument(
        "--synthetic", action="store_true",
        help="Publish as a synthetic example bundle. Waives runs/ and "
             "clean-tree requirements but requires the manifest to "
             "declare `invocation.synthetic = true`. --synthetic and "
             "invocation.synthetic must match — gate rejects mismatches.",
    )
    args = p.parse_args()

    reasons = check_bundle(
        args.bundle,
        heuristic_max=args.heuristic_max,
        allow_diagnostic=args.allow_diagnostic,
        allow_no_runs=args.allow_no_runs,
        allow_dirty=args.allow_dirty,
        synthetic=args.synthetic,
        min_non_diagnostic=args.min_non_diagnostic,
    )

    # Build-vs-gate override audit — surface regardless of pass/fail so
    # reviewers can see which relaxations were applied at each stage.
    # Load manifest directly (check_bundle may have failed at integrity
    # check, but we still want to report whatever the manifest says).
    audit_lines: list[str] = []
    try:
        manifest, _ = load_bundle(args.bundle)
        mf_overrides = list(manifest.invocation.get("overrides") or [])
        audit_lines = _render_override_audit(
            mf_overrides,
            gate_flags={
                "allow_diagnostic": args.allow_diagnostic,
                "allow_no_runs": args.allow_no_runs,
                "allow_dirty": args.allow_dirty,
                "synthetic": args.synthetic,
            },
        )
    except BundleError:
        # Bundle is broken; the main reasons list will carry that.
        pass

    if not reasons:
        print(f"PUBLISH_READY: {args.bundle}")
        for line in audit_lines:
            print(f"  note: {line}")
        return 0

    print(f"NOT_PUBLISH_READY: {args.bundle}", file=sys.stderr)
    for r in reasons:
        print(f"  - {r}", file=sys.stderr)
    for line in audit_lines:
        print(f"  note: {line}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
