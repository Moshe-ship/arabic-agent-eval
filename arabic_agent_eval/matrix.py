"""Result-matrix scaffolding — MTG-assisted re-validation over a benchmark run.

Take a completed `BenchmarkResult` (produced by Evaluator against any
provider), replay every emitted tool call through MTG's validation
pipeline, and aggregate the hard-result-table numbers:

- baseline function-calling score
- MTG violation rate (per item, per call, per code)
- transliteration rate (fraction of Arabic-valued args that the model
  returned in a non-Arabic script)
- dialect-drift rate (for dialect-bound items)
- repaired score after reconciled-mode repair (where MTG can propose a
  concrete replacement)
- cost / latency deltas — captured if the caller populates `cost_usd`
  and `latency_ms` on EvalResult

Deliberately decoupled from model providers — this module does NOT make
API calls. Run your provider matrix the usual way (via Evaluator), then
pass the resulting BenchmarkResult into `build_matrix`. That keeps API
keys, rate limits, and cost out of MTG.

Output is Markdown + CSV. The Markdown is designed to drop straight into
a blog post or paper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from arabic_agent_eval.evaluator import BenchmarkResult, EvalResult


try:
    from mtg.adapters.openai import guard_tool
    from mtg.pipeline import validate_pre
    from mtg.types import GuardSpec
    _MTG_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MTG_AVAILABLE = False


@dataclass
class MatrixRow:
    """One row of the result table — typically a (provider, model) pair."""

    provider: str
    model: str
    baseline_score: float = 0.0
    # MTG re-validation aggregates
    total_calls_scanned: int = 0
    violation_rate: float = 0.0
    transliteration_rate: float = 0.0
    dialect_drift_rate: float = 0.0
    bidi_violation_rate: float = 0.0
    homoglyph_rate: float = 0.0
    # Repair path
    calls_with_concrete_repair: int = 0
    repair_rate: float = 0.0
    repaired_score: float = 0.0  # baseline_score if repaired args were used
    # Cost / latency
    total_cost_usd: Optional[float] = None
    median_latency_ms: Optional[float] = None
    total_items: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "baseline_score": round(self.baseline_score, 4),
            "repaired_score": round(self.repaired_score, 4),
            "violation_rate": round(self.violation_rate, 4),
            "transliteration_rate": round(self.transliteration_rate, 4),
            "dialect_drift_rate": round(self.dialect_drift_rate, 4),
            "bidi_violation_rate": round(self.bidi_violation_rate, 4),
            "homoglyph_rate": round(self.homoglyph_rate, 4),
            "repair_rate": round(self.repair_rate, 4),
            "total_items": self.total_items,
            "total_calls_scanned": self.total_calls_scanned,
            "calls_with_concrete_repair": self.calls_with_concrete_repair,
            "total_cost_usd": (
                round(self.total_cost_usd, 4) if self.total_cost_usd is not None else None
            ),
            "median_latency_ms": self.median_latency_ms,
        }


@dataclass
class ResultMatrix:
    """Table of MatrixRow, one per provider-model pair."""

    rows: list[MatrixRow] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"rows": [r.to_dict() for r in self.rows]}


def _guess_spec_for_value(value: str) -> GuardSpec:
    """Best-effort GuardSpec for a raw argument value when no schema is
    attached. Used when we only have the model's emitted calls and want
    to run them through MTG as a diagnostic layer.

    Heuristic: if value contains any Arabic character, assume it's an
    ar-declared free-text slot in reconciled mode. If value is pure
    Latin, assume latn free-text. Never raises; never mutates input.
    """
    has_arabic = any("\u0600" <= ch <= "\u06FF" for ch in value)
    return GuardSpec.from_dict(
        {
            "slot_type": "free_text",
            "script": "ar" if has_arabic else "latn",
            "transliteration_allowed": not has_arabic,
            "mode": "reconciled",
        },
        validate=True,
    )


def _iter_call_values(eval_result: "EvalResult") -> Iterable[tuple[str, str, Any]]:
    """Yield (item_id, arg_name, arg_value) for each argument in each actual call."""
    for call in eval_result.actual_calls or []:
        args = call.get("arguments") or call.get("args") or {}
        if not isinstance(args, dict):
            continue
        for arg_name, arg_value in args.items():
            yield eval_result.item.id, arg_name, arg_value


def scan_with_mtg(benchmark_result: "BenchmarkResult") -> MatrixRow:
    """Replay every emitted tool-call argument through MTG and produce
    a single MatrixRow for this benchmark run.

    Does NOT call any LLM. Does NOT make network calls. Pure aggregation
    over values the caller already captured.
    """
    if not _MTG_AVAILABLE:
        raise RuntimeError(
            "mtg-guards is not installed; install it to run MTG-assisted scoring"
        )

    row = MatrixRow(
        provider=benchmark_result.provider,
        model=benchmark_result.model,
        baseline_score=benchmark_result.overall_score,
        total_items=benchmark_result.total_items,
    )

    violation_hits = 0
    translit_hits = 0
    dialect_hits = 0
    bidi_hits = 0
    homoglyph_hits = 0
    total_calls = 0
    repair_hits = 0
    # For repaired_score: per-item, if EVERY Arabic arg got a concrete
    # repair, treat the item as "could have been fixed"; contributes
    # baseline-or-1.0 to repaired_score depending on the original score.
    repaired_item_scores: dict[str, float] = {}

    for result in benchmark_result.results:
        if result.error:
            continue
        original_total = result.score.total
        item_could_repair = False
        item_had_violations = False
        for _item_id, _arg_name, arg_value in _iter_call_values(result):
            if not isinstance(arg_value, str) or not arg_value:
                continue
            total_calls += 1
            try:
                spec = _guess_spec_for_value(arg_value)
                guard = validate_pre(arg_value, spec)
            except Exception:
                continue

            if guard.violations:
                violation_hits += 1
                item_had_violations = True
            codes = {v.code for v in guard.violations}
            if "TRANSLITERATION_VIOLATION" in codes:
                translit_hits += 1
            if "DIALECT_DRIFT" in codes:
                dialect_hits += 1
            if "BIDI_CONTROL_SMUGGLING" in codes:
                bidi_hits += 1
            if "SCRIPT_HOMOGLYPH" in codes:
                homoglyph_hits += 1

            if guard.repaired_surface:
                repair_hits += 1
                item_could_repair = True

        # Repaired-score bookkeeping: if the item had violations AND got a
        # concrete repair proposal for at least one arg, we credit the
        # caller with 1.0 on a weighted basis (they can replay). This is
        # deliberately optimistic; use it as an upper bound.
        if item_had_violations and item_could_repair:
            repaired_item_scores[result.item.id] = 1.0
        else:
            repaired_item_scores[result.item.id] = original_total

    row.total_calls_scanned = total_calls
    row.violation_rate = violation_hits / max(1, total_calls)
    row.transliteration_rate = translit_hits / max(1, total_calls)
    row.dialect_drift_rate = dialect_hits / max(1, total_calls)
    row.bidi_violation_rate = bidi_hits / max(1, total_calls)
    row.homoglyph_rate = homoglyph_hits / max(1, total_calls)
    row.calls_with_concrete_repair = repair_hits
    row.repair_rate = repair_hits / max(1, total_calls)
    if repaired_item_scores:
        row.repaired_score = sum(repaired_item_scores.values()) / len(repaired_item_scores)

    # Cost / latency — opt-in, only if caller populated them on EvalResult
    costs = [getattr(r, "cost_usd", None) for r in benchmark_result.results]
    costs = [c for c in costs if c is not None]
    if costs:
        row.total_cost_usd = sum(costs)

    latencies = [getattr(r, "latency_ms", None) for r in benchmark_result.results]
    latencies = sorted(lat for lat in latencies if lat is not None)
    if latencies:
        row.median_latency_ms = latencies[len(latencies) // 2]

    return row


def build_matrix(benchmark_results: Iterable["BenchmarkResult"]) -> ResultMatrix:
    """Fold N benchmark runs (typically one per provider-model pair) into
    a single ResultMatrix."""
    rows = [scan_with_mtg(br) for br in benchmark_results]
    return ResultMatrix(rows=rows)


# ---------- renderers ----------


def render_markdown(matrix: ResultMatrix) -> str:
    """Render the matrix as a single Markdown table suitable for a blog
    post, arXiv appendix, or README section."""
    header = (
        "| provider | model | baseline | repaired | Δ repair | "
        "viol. % | translit % | drift % | bidi % | homoglyph % | "
        "items | calls | cost | p50 ms |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in matrix.rows:
        delta = row.repaired_score - row.baseline_score
        cost = f"${row.total_cost_usd:.3f}" if row.total_cost_usd is not None else "—"
        latency = f"{row.median_latency_ms:.0f}" if row.median_latency_ms is not None else "—"
        lines.append(
            "| {prov} | {model} | {base:.3f} | {rep:.3f} | {delta:+.3f} | "
            "{v:.1%} | {t:.1%} | {d:.1%} | {b:.1%} | {h:.1%} | "
            "{items} | {calls} | {cost} | {lat} |".format(
                prov=row.provider, model=row.model,
                base=row.baseline_score, rep=row.repaired_score, delta=delta,
                v=row.violation_rate, t=row.transliteration_rate,
                d=row.dialect_drift_rate, b=row.bidi_violation_rate,
                h=row.homoglyph_rate,
                items=row.total_items, calls=row.total_calls_scanned,
                cost=cost, lat=latency,
            )
        )
    return "\n".join(lines)


def render_csv(matrix: ResultMatrix) -> str:
    """Render the matrix as CSV for spreadsheets / plotting."""
    cols = [
        "provider", "model",
        "baseline_score", "repaired_score",
        "violation_rate", "transliteration_rate", "dialect_drift_rate",
        "bidi_violation_rate", "homoglyph_rate",
        "repair_rate", "total_items", "total_calls_scanned",
        "total_cost_usd", "median_latency_ms",
    ]
    lines = [",".join(cols)]
    for row in matrix.rows:
        d = row.to_dict()
        lines.append(",".join(
            "" if d.get(c) is None else str(d[c])
            for c in cols
        ))
    return "\n".join(lines)


def load_benchmark_result_from_json(path: Path) -> "BenchmarkResult":
    """Load a previously-saved BenchmarkResult JSON file (the shape
    produced by BenchmarkResult.to_dict()). Used by the CLI runner."""
    from arabic_agent_eval.dataset import EvalItem, ExpectedCall
    from arabic_agent_eval.evaluator import BenchmarkResult, EvalResult
    from arabic_agent_eval.scoring import Score

    data = json.loads(path.read_text(encoding="utf-8"))
    results: list[EvalResult] = []
    for r in data.get("results", []):
        item_data = r.get("item") or {}
        # Best-effort item reconstruction — matrix.py only needs id/category/dialect
        item = EvalItem(
            id=item_data.get("id") or r.get("item_id", ""),
            category=item_data.get("category") or r.get("category", ""),
            instruction=item_data.get("instruction") or r.get("instruction", ""),
            dialect=item_data.get("dialect") or r.get("dialect", "msa"),
            available_functions=item_data.get("available_functions", []),
            expected_calls=[
                ExpectedCall(function=c.get("function", ""), arguments=c.get("arguments", {}))
                for c in item_data.get("expected_calls", [])
            ],
            difficulty=item_data.get("difficulty", "easy"),
        )
        score_data = r.get("score", {})
        score = Score(
            item_id=score_data.get("item_id", item.id),
            function_selection=score_data.get("function_selection", 0.0),
            argument_accuracy=score_data.get("argument_accuracy", 0.0),
            arabic_preservation=score_data.get("arabic_preservation", 0.0),
            dialect_understanding=score_data.get("dialect_understanding", 0.0),
            error_handling=score_data.get("error_handling", 0.0),
            category=score_data.get("category", item.category),
        )
        er = EvalResult(
            item=item,
            score=score,
            actual_calls=r.get("actual_calls", []),
            raw_response=r.get("raw_response", ""),
            error=r.get("error"),
        )
        # Optional cost/latency telemetry
        if "cost_usd" in r:
            setattr(er, "cost_usd", r["cost_usd"])
        if "latency_ms" in r:
            setattr(er, "latency_ms", r["latency_ms"])
        results.append(er)

    return BenchmarkResult(
        provider=data.get("provider", "unknown"),
        model=data.get("model", "unknown"),
        results=results,
    )


__all__ = [
    "MatrixRow",
    "ResultMatrix",
    "build_matrix",
    "load_benchmark_result_from_json",
    "render_csv",
    "render_markdown",
    "scan_with_mtg",
]
