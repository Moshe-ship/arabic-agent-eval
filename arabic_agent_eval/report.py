"""Markdown report generation for Arabic Agent Eval."""

from __future__ import annotations

from datetime import datetime, timezone

from arabic_agent_eval.dataset import CATEGORIES, DIALECTS
from arabic_agent_eval.evaluator import BenchmarkResult


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def benchmark_result_to_markdown(result: BenchmarkResult) -> str:
    """Render a BenchmarkResult as a human-readable markdown report.

    Includes overall score, per-category scores, per-dialect scores,
    and a short error summary. Used by `aae run --md-output report.md`.
    """
    lines: list[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines.append(f"# Arabic Agent Eval — {result.provider}")
    lines.append("")
    lines.append(f"**Model:** `{result.model}`  ")
    lines.append(f"**Overall score:** {_pct(result.overall_score)} (grade **{result.overall_grade}**)  ")
    lines.append(f"**Total items:** {result.total_items}  ")
    lines.append(f"**Errors:** {len(result.errors)}  ")
    lines.append(f"**Generated:** {now}")
    lines.append("")

    lines.append("## Category scores")
    lines.append("")
    lines.append("| Category | Arabic | Items | Function | Arguments | Arabic preservation | Total | Weight |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for cs in result.category_scores:
        lines.append(
            f"| {cs.category} | {cs.name_ar} | {cs.count} | "
            f"{_pct(cs.avg_function_selection)} | "
            f"{_pct(cs.avg_argument_accuracy)} | "
            f"{_pct(cs.avg_arabic_preservation)} | "
            f"{_pct(cs.avg_total)} | "
            f"{_pct(cs.weight)} |"
        )
    lines.append("")

    lines.append("## Dialect scores")
    lines.append("")
    lines.append("| Dialect | Items | Function | Arguments | Arabic preservation | Total |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    dialect_rows = _aggregate_by_dialect(result)
    for dialect in DIALECTS:
        row = dialect_rows.get(dialect)
        if not row:
            continue
        lines.append(
            f"| {dialect} | {row['count']} | "
            f"{_pct(row['avg_function'])} | "
            f"{_pct(row['avg_argument'])} | "
            f"{_pct(row['avg_arabic'])} | "
            f"{_pct(row['avg_total'])} |"
        )
    lines.append("")

    if result.errors:
        lines.append("## Errors")
        lines.append("")
        for r in result.errors[:20]:
            lines.append(f"- `{r.item.id}` — {r.error}")
        if len(result.errors) > 20:
            lines.append(f"- … {len(result.errors) - 20} more errors truncated")
        lines.append("")

    return "\n".join(lines)


def _aggregate_by_dialect(result: BenchmarkResult) -> dict[str, dict[str, float]]:
    bucket: dict[str, list] = {d: [] for d in DIALECTS}
    for r in result.results:
        if r.item.dialect in bucket:
            bucket[r.item.dialect].append(r.score)

    agg: dict[str, dict[str, float]] = {}
    for dialect, scores in bucket.items():
        if not scores:
            continue
        count = len(scores)
        agg[dialect] = {
            "count": count,
            "avg_function": sum(s.function_selection for s in scores) / count,
            "avg_argument": sum(s.argument_accuracy for s in scores) / count,
            "avg_arabic": sum(s.arabic_preservation for s in scores) / count,
            "avg_total": sum(s.total for s in scores) / count,
        }
    return agg
