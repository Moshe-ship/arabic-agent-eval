"""Tests for arabic_agent_eval.diff — bundle diffing."""

from __future__ import annotations

import copy

import pytest

pytest.importorskip("mtg")

from arabic_agent_eval.diff import diff_bundles, render_markdown


def _row(
    provider: str = "p", model: str = "m",
    baseline: float = 0.8, repaired: float = 0.9,
    baseline_ci=(0.7, 0.9), repaired_ci=(0.8, 1.0),
    heuristic_scan_rate: float = 0.0,
    schema_bound_rate: float = 1.0,
    schema_covered_tools=None,
    family_rates=None, layer_rates=None,
    diagnostic: bool = False,
    run_id: str = "run-1",
    scanner_version: str = "mtg-matrix/0.3",
    schema_map_fingerprint: str = "fingerprint-a",
) -> dict:
    return {
        "provider": provider, "model": model,
        "baseline_score": baseline,
        "repaired_score": repaired,
        "baseline_ci_95": list(baseline_ci) if baseline_ci else None,
        "repaired_ci_95": list(repaired_ci) if repaired_ci else None,
        "heuristic_scan_rate": heuristic_scan_rate,
        "schema_bound_rate": schema_bound_rate,
        "schema_covered_tools": list(schema_covered_tools or []),
        "family_rates": dict(family_rates or {}),
        "layer_rates": dict(layer_rates or {}),
        "diagnostic": diagnostic,
        "run_metadata": {
            "run_id": run_id,
            "scanner_version": scanner_version,
            "schema_map_fingerprint": schema_map_fingerprint,
        },
    }


def test_diff_detects_unchanged_row():
    r = _row()
    before = {"rows": [r]}
    after = {"rows": [copy.deepcopy(r)]}
    d = diff_bundles(before, after)
    assert len(d.rows) == 1
    assert d.rows[0].status == "unchanged"
    assert d.rows[0].baseline_delta == 0.0


def test_diff_detects_score_delta_with_ci_overlap():
    """CIs that overlap → weak evidence (baseline_ci_overlap=True)."""
    before = {"rows": [_row(baseline=0.80, baseline_ci=(0.70, 0.90))]}
    after = {"rows": [_row(baseline=0.85, baseline_ci=(0.75, 0.95),
                            run_id="run-2")]}
    d = diff_bundles(before, after)
    assert d.rows[0].status == "changed"
    assert d.rows[0].baseline_delta == pytest.approx(0.05, abs=1e-4)
    assert d.rows[0].baseline_ci_overlap is True


def test_diff_detects_ci_disjoint_strong_evidence():
    """CIs that don't overlap → stronger evidence of real change."""
    before = {"rows": [_row(baseline=0.20, baseline_ci=(0.10, 0.30))]}
    after = {"rows": [_row(baseline=0.80, baseline_ci=(0.70, 0.90))]}
    d = diff_bundles(before, after)
    assert d.rows[0].baseline_ci_overlap is False


def test_diff_detects_added_row():
    before = {"rows": []}
    after = {"rows": [_row(provider="new", model="v1")]}
    d = diff_bundles(before, after)
    assert len(d.rows) == 1
    assert d.rows[0].status == "added"
    assert d.rows[0].provider == "new"


def test_diff_detects_removed_row():
    before = {"rows": [_row(provider="gone", model="v1")]}
    after = {"rows": []}
    d = diff_bundles(before, after)
    assert len(d.rows) == 1
    assert d.rows[0].status == "removed"


def test_diff_detects_schema_coverage_changes():
    before = {"rows": [_row(schema_covered_tools=["a", "b"])]}
    after = {"rows": [_row(schema_covered_tools=["a", "c"])]}
    d = diff_bundles(before, after)
    assert d.rows[0].schema_covered_tools_added == ["c"]
    assert d.rows[0].schema_covered_tools_removed == ["b"]


def test_diff_detects_diagnostic_transition():
    before = {"rows": [_row(diagnostic=True)]}
    after = {"rows": [_row(diagnostic=False)]}
    d = diff_bundles(before, after)
    assert d.rows[0].diagnostic_changed == (True, False)


def test_diff_flags_scanner_version_change():
    """Different scanner_version → the diff's interpretation banner
    warns that deltas may be scoring-logic changes."""
    before = {"rows": [_row(scanner_version="mtg-matrix/0.2")]}
    after = {"rows": [_row(scanner_version="mtg-matrix/0.3")]}
    d = diff_bundles(before, after)
    assert d.scanner_version_changed is True
    md = render_markdown(d)
    assert "scanner_version changed" in md


def test_diff_flags_schema_fingerprint_change():
    before = {"rows": [_row(schema_map_fingerprint="aaa")]}
    after = {"rows": [_row(schema_map_fingerprint="bbb")]}
    d = diff_bundles(before, after)
    assert d.schema_fingerprint_changed is True
    md = render_markdown(d)
    assert "schema_map_fingerprint changed" in md


def test_diff_family_deltas_only_include_non_zero_in_markdown():
    before = {"rows": [_row(family_rates={"script": 0.0, "dialect": 0.2})]}
    after = {"rows": [_row(family_rates={"script": 0.1, "dialect": 0.2})]}
    d = diff_bundles(before, after)
    md = render_markdown(d)
    assert "Family-rate deltas" in md
    assert "script" in md  # changed family listed
    unchanged_sentinel = "`dialect` +0.0%"
    assert unchanged_sentinel not in md


def test_diff_layer_deltas_surfaced_in_markdown():
    before = {"rows": [_row(layer_rates={"surface": 0.1})]}
    after = {"rows": [_row(layer_rates={"surface": 0.05, "security": 0.02})]}
    d = diff_bundles(before, after)
    md = render_markdown(d)
    assert "3-layer deltas" in md
    assert "surface" in md
    assert "security" in md


def test_diff_json_shape_is_stable():
    before = {"rows": [_row()]}
    after = {"rows": [_row(baseline=0.9, run_id="run-2")]}
    d = diff_bundles(before, after)
    data = d.to_dict()
    assert "rows" in data and isinstance(data["rows"], list)
    row = data["rows"][0]
    for key in (
        "provider", "model", "status",
        "baseline_delta", "repaired_delta",
        "baseline_ci_overlap", "repaired_ci_overlap",
        "heuristic_scan_rate_delta", "schema_bound_rate_delta",
        "family_deltas", "layer_deltas", "before_run_id", "after_run_id",
    ):
        assert key in row, f"missing key: {key}"


def test_diff_empty_bundles_render_cleanly():
    d = diff_bundles({"rows": []}, {"rows": []})
    md = render_markdown(d)
    assert "No rows" in md
