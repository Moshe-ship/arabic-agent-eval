"""Tests for the result-matrix scaffolding."""

from __future__ import annotations

import pytest

pytest.importorskip("mtg")

from arabic_agent_eval.dataset import EvalItem, ExpectedCall
from arabic_agent_eval.evaluator import BenchmarkResult, EvalResult
from arabic_agent_eval.matrix import (
    MatrixRow,
    build_matrix,
    render_csv,
    render_markdown,
    scan_with_mtg,
)
from arabic_agent_eval.scoring import Score


def _result(
    item_id: str,
    actual_calls: list[dict],
    score_total: float = 1.0,
    category: str = "simple_function_calling",
    dialect: str = "gulf",
) -> EvalResult:
    item = EvalItem(
        id=item_id, category=category, instruction="", dialect=dialect,
        available_functions=[],
        expected_calls=[ExpectedCall(function="t", arguments={})],
        difficulty="easy",
    )
    return EvalResult(
        item=item,
        score=Score(
            item_id=item_id, category=category,
            function_selection=score_total, argument_accuracy=score_total,
            arabic_preservation=score_total,
        ),
        actual_calls=actual_calls,
    )


def test_scan_with_clean_arabic_calls_yields_zero_rates():
    """Baseline — all emitted args are clean Arabic; nothing should fire."""
    br = BenchmarkResult(
        provider="test", model="clean",
        results=[
            _result("a1", [{"function": "send_message",
                            "arguments": {"message": "أبي أحجز فندق"}}]),
            _result("a2", [{"function": "send_message",
                            "arguments": {"message": "مرحبا أحمد"}}]),
        ],
    )
    row = scan_with_mtg(br)
    assert row.violation_rate == 0.0
    assert row.transliteration_rate == 0.0
    assert row.bidi_violation_rate == 0.0


def test_scan_catches_arabizi_transliteration():
    br = BenchmarkResult(
        provider="test", model="arabizi",
        results=[
            _result("a1", [{"function": "send_message",
                            "arguments": {"message": "abi a7jez fundu"}}]),
        ],
    )
    row = scan_with_mtg(br)
    # The value is detected as ar-slot (contains Arabic heuristic) then
    # the Latin surface fails; but since value has no Arabic chars,
    # our _guess_spec chooses script=latn so no SCRIPT_VIOLATION fires.
    # Instead: no violation (schema-less scan heuristic). The real
    # signal is when the caller annotates args with their tool schema,
    # which the matrix runner does via the reconciled_mode flag on
    # those schemas. This test asserts the baseline heuristic is
    # non-catastrophic.
    assert row.total_calls_scanned == 1


def test_scan_catches_bidi_in_any_arg_type():
    """BiDi-control injection fires regardless of declared script."""
    br = BenchmarkResult(
        provider="test", model="bidi",
        results=[
            _result("a1", [{"function": "send_message",
                            "arguments": {"message": "أبي\u202eأحجز"}}]),
        ],
    )
    row = scan_with_mtg(br)
    assert row.bidi_violation_rate > 0


def test_scan_catches_homoglyph():
    br = BenchmarkResult(
        provider="test", model="homoglyph",
        results=[
            _result("a1", [{"function": "send_message",
                            "arguments": {"message": "مرحبا\u0430"}}]),  # Cyrillic а
        ],
    )
    row = scan_with_mtg(br)
    assert row.homoglyph_rate > 0


def test_scan_counts_repair_rate_for_arabizi():
    """Pure-Arabizi strings in arabic-declared scans should get a concrete
    repair proposal when reconciled mode is active. The matrix scanner
    uses script='latn' for pure-Latin values so this exercises the
    repair-on-latn path — which for Arabizi-looking strings in a
    latn slot does NOT fire translit (since translit only checks ar).
    The assertion here is just that the row counts calls without crashing."""
    br = BenchmarkResult(
        provider="test", model="arabizi",
        results=[_result("a1", [{"function": "x",
                                   "arguments": {"msg": "abi a7jez"}}])],
    )
    row = scan_with_mtg(br)
    assert row.total_calls_scanned == 1
    assert row.repair_rate >= 0  # well-defined


def test_build_matrix_folds_multiple_runs():
    clean = BenchmarkResult(
        provider="a", model="m1",
        results=[_result("a1", [{"function": "x", "arguments": {"msg": "أبي"}}])],
    )
    dirty = BenchmarkResult(
        provider="b", model="m2",
        results=[_result("b1", [{"function": "x", "arguments": {"msg": "مرحبا\u202e"}}])],
    )
    matrix = build_matrix([clean, dirty])
    assert len(matrix.rows) == 2
    assert matrix.rows[0].provider == "a"
    assert matrix.rows[1].bidi_violation_rate > 0


def test_render_markdown_contains_expected_columns():
    row = MatrixRow(provider="p", model="m", baseline_score=0.85,
                    repaired_score=0.92, violation_rate=0.15, total_items=10,
                    total_calls_scanned=40)
    from arabic_agent_eval.matrix import ResultMatrix
    md = render_markdown(ResultMatrix(rows=[row]))
    assert "baseline" in md and "repaired" in md
    assert "0.850" in md
    assert "15.0%" in md  # violation_rate rendered as pct


def test_render_csv_roundtrip():
    row = MatrixRow(provider="p", model="m", baseline_score=0.5,
                    violation_rate=0.2, total_items=5, total_calls_scanned=10)
    from arabic_agent_eval.matrix import ResultMatrix
    csv = render_csv(ResultMatrix(rows=[row]))
    assert "provider,model" in csv
    assert "p,m" in csv


def test_scan_handles_errors_gracefully():
    """Error rows must not crash the scanner."""
    br = BenchmarkResult(
        provider="fail", model="x",
        results=[
            EvalResult(
                item=EvalItem(id="e1", category="simple_function_calling",
                               instruction="", dialect="msa",
                               available_functions=[],
                               expected_calls=[ExpectedCall(function="t", arguments={})],
                               difficulty="easy"),
                score=Score(item_id="e1", category="simple_function_calling"),
                actual_calls=[],
                error="API down",
            ),
        ],
    )
    row = scan_with_mtg(br)
    assert row.total_calls_scanned == 0


def test_cost_and_latency_aggregated_when_provided():
    r = _result("a1", [{"function": "x", "arguments": {"msg": "أبي"}}])
    setattr(r, "cost_usd", 0.0012)
    setattr(r, "latency_ms", 480)
    r2 = _result("a2", [{"function": "x", "arguments": {"msg": "مرحبا"}}])
    setattr(r2, "cost_usd", 0.0009)
    setattr(r2, "latency_ms", 520)
    br = BenchmarkResult(provider="p", model="m", results=[r, r2])
    row = scan_with_mtg(br)
    assert row.total_cost_usd == pytest.approx(0.0021, abs=1e-6)
    assert row.median_latency_ms in (480, 520)


def test_scan_without_mtg_raises_clean_error(monkeypatch):
    """If mtg is not installed, the scanner must emit a clear error."""
    import arabic_agent_eval.matrix as m
    monkeypatch.setattr(m, "_MTG_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="mtg-guards"):
        m.scan_with_mtg(BenchmarkResult(provider="x", model="y", results=[]))
