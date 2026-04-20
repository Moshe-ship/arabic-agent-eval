"""Bundle-to-bundle trend reporting.

The diff tool (`arabic_agent_eval.diff`) answers "what changed between
these two runs?". This module answers "what's changed over N runs?" —
a compact time-series view across several bundles so reviewers can
see whether coverage, scores, and layer failures are improving.

Each bundle contributes one point per (provider, model) row. The
trend groups points by that key and presents the ordered sequence.
Bundles are consumed in the order given — callers are responsible for
sorting by build_at / created_at when that's what they want.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


@dataclass
class TrendPoint:
    """One (provider, model) row's numbers at one point in time."""

    bundle_label: str
    created_at: str
    dataset_version: str
    scanner_version: str
    baseline_score: float
    repaired_score: float
    baseline_ci_95: Optional[tuple[float, float]] = None
    heuristic_scan_rate: float = 0.0
    schema_bound_rate: float = 0.0
    layer_rates: dict[str, float] = field(default_factory=dict)
    diagnostic: bool = False
    run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_label": self.bundle_label,
            "created_at": self.created_at,
            "dataset_version": self.dataset_version,
            "scanner_version": self.scanner_version,
            "baseline_score": self.baseline_score,
            "repaired_score": self.repaired_score,
            "baseline_ci_95": list(self.baseline_ci_95) if self.baseline_ci_95 else None,
            "heuristic_scan_rate": self.heuristic_scan_rate,
            "schema_bound_rate": self.schema_bound_rate,
            "layer_rates": dict(self.layer_rates),
            "diagnostic": self.diagnostic,
            "run_id": self.run_id,
        }


@dataclass
class TrendSeries:
    """All points for one (provider, model) pair, in bundle order."""

    provider: str
    model: str
    points: list[TrendPoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "points": [p.to_dict() for p in self.points],
        }

    @property
    def first(self) -> Optional[TrendPoint]:
        return self.points[0] if self.points else None

    @property
    def last(self) -> Optional[TrendPoint]:
        return self.points[-1] if self.points else None


@dataclass
class Trend:
    """Full cross-bundle trend view."""

    bundle_labels: list[str] = field(default_factory=list)
    series: list[TrendSeries] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_labels": list(self.bundle_labels),
            "series": [s.to_dict() for s in self.series],
        }


def _row_point(bundle_label: str, created_at: str, row: dict) -> TrendPoint:
    md = row.get("run_metadata") or {}
    ci = row.get("baseline_ci_95")
    return TrendPoint(
        bundle_label=bundle_label,
        created_at=created_at,
        dataset_version=md.get("dataset_version", ""),
        scanner_version=md.get("scanner_version", ""),
        baseline_score=float(row.get("baseline_score", 0.0)),
        repaired_score=float(row.get("repaired_score", 0.0)),
        baseline_ci_95=(float(ci[0]), float(ci[1])) if ci else None,
        heuristic_scan_rate=float(row.get("heuristic_scan_rate", 0.0)),
        schema_bound_rate=float(row.get("schema_bound_rate", 0.0)),
        layer_rates=dict(row.get("layer_rates") or {}),
        diagnostic=bool(row.get("diagnostic", False)),
        run_id=md.get("run_id", ""),
    )


def build_trend(
    bundles: Iterable[tuple[str, str, dict]],
) -> Trend:
    """Build a Trend from an ordered sequence of
    `(bundle_label, created_at, matrix_dict)` tuples. Callers resolve
    the bundle label however they like (directory name, tag, etc.);
    matrix_dict is the parsed matrix.json from each bundle."""
    labels: list[str] = []
    bucket: dict[tuple[str, str], TrendSeries] = {}
    order: list[tuple[str, str]] = []
    for label, created_at, matrix in bundles:
        labels.append(label)
        for row in matrix.get("rows") or []:
            key = (row.get("provider") or "", row.get("model") or "")
            series = bucket.get(key)
            if series is None:
                series = TrendSeries(provider=key[0], model=key[1])
                bucket[key] = series
                order.append(key)
            series.points.append(_row_point(label, created_at, row))
    return Trend(bundle_labels=labels, series=[bucket[k] for k in order])


def _fmt_delta(first: float, last: float) -> str:
    delta = last - first
    return f"{delta:+.3f}"


def render_markdown(trend: Trend) -> str:
    """Render a Trend as Markdown — a compact per-(provider, model)
    table with headline score/coverage/diagnostic across bundles."""
    if not trend.bundle_labels:
        return "_Empty trend._\n"

    parts: list[str] = [
        "# Bundle trend",
        "",
        f"Bundles in order: "
        + " → ".join(f"`{l}`" for l in trend.bundle_labels),
        "",
    ]

    for series in trend.series:
        if not series.points:
            continue
        first, last = series.first, series.last
        assert first is not None and last is not None
        parts.extend([
            f"## `{series.provider} / {series.model}`",
            "",
            f"**Baseline:** {first.baseline_score:.3f} → {last.baseline_score:.3f} "
            f"({_fmt_delta(first.baseline_score, last.baseline_score)}) · "
            f"**Repaired:** {first.repaired_score:.3f} → {last.repaired_score:.3f} "
            f"({_fmt_delta(first.repaired_score, last.repaired_score)}) · "
            f"**Schema coverage:** {first.schema_bound_rate:.1%} → "
            f"{last.schema_bound_rate:.1%}",
            "",
        ])

        header = (
            "| bundle | created_at | baseline | repaired | schema-bound % | "
            "surface | language | security | diag |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|:-:|"
        )
        rows = []
        for p in series.points:
            diag = "⚠" if p.diagnostic else "—"
            ci = f" [{p.baseline_ci_95[0]:.2f}–{p.baseline_ci_95[1]:.2f}]" \
                if p.baseline_ci_95 else ""
            rows.append(
                "| `{bl}` | {ts} | {base:.3f}{ci} | {rep:.3f} | {sb:.1%} | "
                "{su:.1%} | {la:.1%} | {se:.1%} | {dg} |".format(
                    bl=p.bundle_label, ts=p.created_at or "—",
                    base=p.baseline_score, ci=ci,
                    rep=p.repaired_score,
                    sb=p.schema_bound_rate,
                    su=p.layer_rates.get("surface", 0.0),
                    la=p.layer_rates.get("language", 0.0),
                    se=p.layer_rates.get("security", 0.0),
                    dg=diag,
                )
            )
        parts.extend([header, *rows, ""])

        # Dataset / scanner version changes over the series — flag when
        # they shift, because numbers-to-numbers comparison isn't clean
        # across scorer changes.
        versions_dataset = {p.dataset_version for p in series.points if p.dataset_version}
        versions_scanner = {p.scanner_version for p in series.points if p.scanner_version}
        if len(versions_dataset) > 1 or len(versions_scanner) > 1:
            parts.extend([
                "> ⚠ version changes across this series:",
                f"> - dataset_version: {sorted(versions_dataset)}" if len(versions_dataset) > 1 else "",
                f"> - scanner_version: {sorted(versions_scanner)}" if len(versions_scanner) > 1 else "",
                "",
            ])
            # drop empty placeholder lines
            parts = [p for p in parts if p != ""]
            parts.append("")  # restore spacing

    return "\n".join(parts)


__all__ = ["Trend", "TrendPoint", "TrendSeries", "build_trend", "render_markdown"]
