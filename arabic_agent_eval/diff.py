"""Bundle diffing — compare two result bundles, emit a structured delta.

The diff answers three questions at a glance:

1. Did scores move? (`baseline`, `repaired`, plus CI overlap so we
   can tell whether a delta is statistically meaningful given the
   within-run variance we captured.)
2. Did schema coverage change? Did a row become schema-grounded, or
   drop back to heuristic?
3. Did failure-family / layer rates shift? Often the headline score
   moves for reasons the aggregate masks — a dialect regression
   hiding behind a script fix, for example.

Match rows across bundles by (provider, model). Unmatched rows are
reported as added/removed. Rows that share provider+model but have
different run_ids are compared cell-by-cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RowDiff:
    """Structural delta between two matched MatrixRows."""

    provider: str
    model: str
    status: str  # "changed" | "added" | "removed" | "unchanged"
    baseline_delta: Optional[float] = None
    repaired_delta: Optional[float] = None
    baseline_ci_overlap: Optional[bool] = None
    repaired_ci_overlap: Optional[bool] = None
    heuristic_scan_rate_delta: Optional[float] = None
    schema_bound_rate_delta: Optional[float] = None
    schema_covered_tools_added: list[str] = field(default_factory=list)
    schema_covered_tools_removed: list[str] = field(default_factory=list)
    family_deltas: dict[str, float] = field(default_factory=dict)
    layer_deltas: dict[str, float] = field(default_factory=dict)
    diagnostic_changed: Optional[tuple[bool, bool]] = None  # (before, after)
    before_run_id: Optional[str] = None
    after_run_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "baseline_delta": self.baseline_delta,
            "repaired_delta": self.repaired_delta,
            "baseline_ci_overlap": self.baseline_ci_overlap,
            "repaired_ci_overlap": self.repaired_ci_overlap,
            "heuristic_scan_rate_delta": self.heuristic_scan_rate_delta,
            "schema_bound_rate_delta": self.schema_bound_rate_delta,
            "schema_covered_tools_added": list(self.schema_covered_tools_added),
            "schema_covered_tools_removed": list(self.schema_covered_tools_removed),
            "family_deltas": dict(self.family_deltas),
            "layer_deltas": dict(self.layer_deltas),
            "diagnostic_changed": list(self.diagnostic_changed)
                if self.diagnostic_changed else None,
            "before_run_id": self.before_run_id,
            "after_run_id": self.after_run_id,
        }


@dataclass
class BundleDiff:
    """Top-level diff between two bundles."""

    before_label: str
    after_label: str
    rows: list[RowDiff] = field(default_factory=list)
    before_scanner_version: str = ""
    after_scanner_version: str = ""
    before_schema_fingerprint: Optional[str] = None
    after_schema_fingerprint: Optional[str] = None
    scanner_version_changed: bool = False
    schema_fingerprint_changed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "before_label": self.before_label,
            "after_label": self.after_label,
            "before_scanner_version": self.before_scanner_version,
            "after_scanner_version": self.after_scanner_version,
            "before_schema_fingerprint": self.before_schema_fingerprint,
            "after_schema_fingerprint": self.after_schema_fingerprint,
            "scanner_version_changed": self.scanner_version_changed,
            "schema_fingerprint_changed": self.schema_fingerprint_changed,
            "rows": [r.to_dict() for r in self.rows],
        }


def _ci_overlaps(a: Optional[list], b: Optional[list]) -> Optional[bool]:
    """True when CIs overlap, False when disjoint, None when either is
    missing. Used to signal whether a score delta is plausibly within
    within-run variance (overlap → deltas are weak evidence)."""
    if not a or not b or len(a) != 2 or len(b) != 2:
        return None
    # CIs overlap if a's upper >= b's lower AND b's upper >= a's lower
    return (a[1] >= b[0]) and (b[1] >= a[0])


def _diff_family(before: dict, after: dict) -> dict[str, float]:
    """Return rate-deltas for every family that appears in either side."""
    keys = set(before.keys()) | set(after.keys())
    return {
        k: round(float(after.get(k, 0.0)) - float(before.get(k, 0.0)), 4)
        for k in sorted(keys)
    }


def _row_key(row: dict) -> tuple[str, str]:
    return (row.get("provider") or "", row.get("model") or "")


def diff_bundles(
    before: dict,
    after: dict,
    *,
    before_label: str = "before",
    after_label: str = "after",
) -> BundleDiff:
    """Compute a structured diff between two loaded matrix.json payloads.

    `before` and `after` are dicts as produced by ResultMatrix.to_dict()
    (the shape stored in `matrix.json` inside a bundle).
    """
    by_key_before = {_row_key(r): r for r in (before.get("rows") or [])}
    by_key_after = {_row_key(r): r for r in (after.get("rows") or [])}
    all_keys = sorted(set(by_key_before) | set(by_key_after))

    # Scanner version + schema fingerprint come from the first row's
    # run_metadata — scanners stamp it identically per-run.
    def _scanner(d: dict) -> str:
        rows = d.get("rows") or []
        if not rows:
            return ""
        return (rows[0].get("run_metadata") or {}).get("scanner_version", "")

    def _fingerprint(d: dict) -> Optional[str]:
        rows = d.get("rows") or []
        if not rows:
            return None
        return (rows[0].get("run_metadata") or {}).get("schema_map_fingerprint")

    before_scan = _scanner(before)
    after_scan = _scanner(after)
    before_fp = _fingerprint(before)
    after_fp = _fingerprint(after)

    row_diffs: list[RowDiff] = []
    for key in all_keys:
        before_row = by_key_before.get(key)
        after_row = by_key_after.get(key)
        provider, model = key
        if before_row is None:
            row_diffs.append(RowDiff(
                provider=provider, model=model, status="added",
                after_run_id=(after_row.get("run_metadata") or {}).get("run_id"),
            ))
            continue
        if after_row is None:
            row_diffs.append(RowDiff(
                provider=provider, model=model, status="removed",
                before_run_id=(before_row.get("run_metadata") or {}).get("run_id"),
            ))
            continue

        before_tools = set(before_row.get("schema_covered_tools") or [])
        after_tools = set(after_row.get("schema_covered_tools") or [])
        diagnostic_before = bool(before_row.get("diagnostic", False))
        diagnostic_after = bool(after_row.get("diagnostic", False))

        baseline_delta = round(
            float(after_row.get("baseline_score", 0.0))
            - float(before_row.get("baseline_score", 0.0)),
            4,
        )
        repaired_delta = round(
            float(after_row.get("repaired_score", 0.0))
            - float(before_row.get("repaired_score", 0.0)),
            4,
        )

        any_field_differs = (
            baseline_delta != 0.0
            or repaired_delta != 0.0
            or before_tools != after_tools
            or diagnostic_before != diagnostic_after
            or before_row.get("heuristic_scan_rate") != after_row.get("heuristic_scan_rate")
            or before_row.get("family_rates") != after_row.get("family_rates")
            or before_row.get("layer_rates") != after_row.get("layer_rates")
        )

        row_diffs.append(RowDiff(
            provider=provider, model=model,
            status="changed" if any_field_differs else "unchanged",
            baseline_delta=baseline_delta,
            repaired_delta=repaired_delta,
            baseline_ci_overlap=_ci_overlaps(
                before_row.get("baseline_ci_95"),
                after_row.get("baseline_ci_95"),
            ),
            repaired_ci_overlap=_ci_overlaps(
                before_row.get("repaired_ci_95"),
                after_row.get("repaired_ci_95"),
            ),
            heuristic_scan_rate_delta=round(
                float(after_row.get("heuristic_scan_rate", 0.0))
                - float(before_row.get("heuristic_scan_rate", 0.0)),
                4,
            ),
            schema_bound_rate_delta=round(
                float(after_row.get("schema_bound_rate", 0.0))
                - float(before_row.get("schema_bound_rate", 0.0)),
                4,
            ),
            schema_covered_tools_added=sorted(after_tools - before_tools),
            schema_covered_tools_removed=sorted(before_tools - after_tools),
            family_deltas=_diff_family(
                before_row.get("family_rates") or {},
                after_row.get("family_rates") or {},
            ),
            layer_deltas=_diff_family(
                before_row.get("layer_rates") or {},
                after_row.get("layer_rates") or {},
            ),
            diagnostic_changed=(
                (diagnostic_before, diagnostic_after)
                if diagnostic_before != diagnostic_after else None
            ),
            before_run_id=(before_row.get("run_metadata") or {}).get("run_id"),
            after_run_id=(after_row.get("run_metadata") or {}).get("run_id"),
        ))

    return BundleDiff(
        before_label=before_label,
        after_label=after_label,
        rows=row_diffs,
        before_scanner_version=before_scan,
        after_scanner_version=after_scan,
        before_schema_fingerprint=before_fp,
        after_schema_fingerprint=after_fp,
        scanner_version_changed=before_scan != after_scan,
        schema_fingerprint_changed=before_fp != after_fp,
    )


def render_markdown(diff: BundleDiff) -> str:
    """Render a BundleDiff as Markdown suitable for a PR comment or a
    review thread."""
    parts: list[str] = [
        f"# Bundle diff: `{diff.before_label}` → `{diff.after_label}`",
        "",
    ]

    # Provenance-change notices — these affect how the reader should
    # interpret the row-level deltas.
    if diff.scanner_version_changed:
        parts.extend([
            f"> ⚠ scanner_version changed: `{diff.before_scanner_version}` "
            f"→ `{diff.after_scanner_version}`. Row deltas may reflect "
            f"scoring-logic changes, not model behavior.",
            "",
        ])
    if diff.schema_fingerprint_changed:
        parts.extend([
            f"> ⚠ schema_map_fingerprint changed. Schema-coverage "
            f"deltas reflect changed x-mtg annotations, not just "
            f"model output.",
            "",
        ])

    if not diff.rows:
        parts.append("_No rows in either bundle._")
        return "\n".join(parts)

    header = (
        "| provider | model | status | baseline Δ | repaired Δ | CI overlap | "
        "heuristic Δ | schema-bound Δ | diagnostic |\n"
        "|---|---|---|---:|---:|:-:|---:|---:|:-:|"
    )
    rows_lines = []

    def _fmt_delta(v: Optional[float]) -> str:
        if v is None:
            return "—"
        return f"{v:+.3f}"

    def _fmt_overlap(b: Optional[bool], r: Optional[bool]) -> str:
        if b is None and r is None:
            return "—"
        bstr = "✓" if b else "✗" if b is False else "—"
        rstr = "✓" if r else "✗" if r is False else "—"
        return f"b{bstr}/r{rstr}"

    def _fmt_diag(dc: Optional[tuple[bool, bool]]) -> str:
        if dc is None:
            return "—"
        before, after = dc
        if before == after:
            return "—"
        if before and not after:
            return "→ clean"
        return "→ ⚠"

    for r in diff.rows:
        if r.status == "added":
            rows_lines.append(
                f"| {r.provider} | {r.model} | **added** | — | — | — | — | — | — |"
            )
            continue
        if r.status == "removed":
            rows_lines.append(
                f"| {r.provider} | {r.model} | **removed** | — | — | — | — | — | — |"
            )
            continue
        rows_lines.append(
            "| {p} | {m} | {st} | {b} | {rep} | {ov} | {h} | {sb} | {dg} |".format(
                p=r.provider, m=r.model, st=r.status,
                b=_fmt_delta(r.baseline_delta),
                rep=_fmt_delta(r.repaired_delta),
                ov=_fmt_overlap(r.baseline_ci_overlap, r.repaired_ci_overlap),
                h=_fmt_delta(r.heuristic_scan_rate_delta),
                sb=_fmt_delta(r.schema_bound_rate_delta),
                dg=_fmt_diag(r.diagnostic_changed),
            )
        )

    parts.extend([
        "## Row-level deltas",
        "",
        "`CI overlap`: `b` = baseline, `r` = repaired. `✓` = overlap "
        "(weak evidence of a real change); `✗` = disjoint (stronger).",
        "",
        header,
        *rows_lines,
    ])

    # Non-zero family deltas per row, so reviewers see where shifts
    # actually landed.
    family_tables: list[str] = []
    for r in diff.rows:
        non_zero = {k: v for k, v in r.family_deltas.items() if abs(v) > 1e-6}
        if not non_zero:
            continue
        line = " · ".join(f"`{k}` {v:+.1%}" for k, v in sorted(non_zero.items()))
        family_tables.append(f"- `{r.provider}/{r.model}`: {line}")
    if family_tables:
        parts.extend([
            "",
            "## Family-rate deltas (non-zero only)",
            "",
            *family_tables,
        ])

    layer_tables: list[str] = []
    for r in diff.rows:
        non_zero = {k: v for k, v in r.layer_deltas.items() if abs(v) > 1e-6}
        if not non_zero:
            continue
        line = " · ".join(f"`{k}` {v:+.1%}" for k, v in sorted(non_zero.items()))
        layer_tables.append(f"- `{r.provider}/{r.model}`: {line}")
    if layer_tables:
        parts.extend([
            "",
            "## 3-layer deltas (non-zero only)",
            "",
            *layer_tables,
        ])

    schema_changes: list[str] = []
    for r in diff.rows:
        if r.schema_covered_tools_added or r.schema_covered_tools_removed:
            added = ", ".join(f"`{t}`" for t in r.schema_covered_tools_added) or "—"
            removed = ", ".join(f"`{t}`" for t in r.schema_covered_tools_removed) or "—"
            schema_changes.append(
                f"- `{r.provider}/{r.model}`: +{added} / -{removed}"
            )
    if schema_changes:
        parts.extend([
            "",
            "## Schema coverage changes",
            "",
            *schema_changes,
        ])

    return "\n".join(parts)


__all__ = ["BundleDiff", "RowDiff", "diff_bundles", "render_markdown"]
