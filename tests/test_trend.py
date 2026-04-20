"""Tests for arabic_agent_eval.trend — bundle-to-bundle trends."""

from __future__ import annotations

import pytest

pytest.importorskip("mtg")

from arabic_agent_eval.trend import Trend, build_trend, render_markdown


def _row(
    provider: str = "p", model: str = "m",
    baseline: float = 0.8, repaired: float = 0.9,
    schema_bound_rate: float = 1.0,
    heuristic_scan_rate: float = 0.0,
    layer_rates=None,
    diagnostic: bool = False,
    dataset_version: str = "2026-04-arabic-v1",
    scanner_version: str = "mtg-matrix/0.5",
    run_id: str = "run-1",
    baseline_ci=(0.7, 0.9),
) -> dict:
    return {
        "provider": provider, "model": model,
        "baseline_score": baseline,
        "repaired_score": repaired,
        "baseline_ci_95": list(baseline_ci) if baseline_ci else None,
        "heuristic_scan_rate": heuristic_scan_rate,
        "schema_bound_rate": schema_bound_rate,
        "layer_rates": dict(layer_rates or {}),
        "diagnostic": diagnostic,
        "run_metadata": {
            "dataset_version": dataset_version,
            "scanner_version": scanner_version,
            "run_id": run_id,
        },
    }


def test_build_trend_orders_points_as_given():
    """Trend preserves the order of bundles the caller passed in."""
    trend = build_trend([
        ("apr17", "2026-04-17", {"rows": [_row(baseline=0.5, run_id="r1")]}),
        ("apr24", "2026-04-24", {"rows": [_row(baseline=0.6, run_id="r2")]}),
        ("may01", "2026-05-01", {"rows": [_row(baseline=0.7, run_id="r3")]}),
    ])
    assert trend.bundle_labels == ["apr17", "apr24", "may01"]
    series = trend.series[0]
    assert [p.bundle_label for p in series.points] == ["apr17", "apr24", "may01"]
    assert [p.baseline_score for p in series.points] == [0.5, 0.6, 0.7]


def test_build_trend_groups_by_provider_and_model():
    """Different (provider, model) pairs become separate series."""
    trend = build_trend([
        ("a", "2026-04-17", {"rows": [
            _row(provider="p1", model="m1"),
            _row(provider="p2", model="m2"),
        ]}),
        ("b", "2026-04-24", {"rows": [
            _row(provider="p1", model="m1"),
            _row(provider="p2", model="m2"),
        ]}),
    ])
    assert len(trend.series) == 2
    labels = {(s.provider, s.model) for s in trend.series}
    assert labels == {("p1", "m1"), ("p2", "m2")}
    for series in trend.series:
        assert len(series.points) == 2


def test_build_trend_handles_added_row_across_bundles():
    """A model that appears only in the second bundle gets a series
    with one point."""
    trend = build_trend([
        ("a", "2026-04-17", {"rows": [_row(provider="p1", model="m1")]}),
        ("b", "2026-04-24", {"rows": [
            _row(provider="p1", model="m1"),
            _row(provider="p2", model="new-model"),
        ]}),
    ])
    m2 = next(s for s in trend.series if s.model == "new-model")
    assert len(m2.points) == 1
    assert m2.points[0].bundle_label == "b"


def test_trend_point_exposes_ci_layer_rates_and_diagnostic():
    trend = build_trend([
        ("a", "2026-04-17", {"rows": [
            _row(baseline=0.6, baseline_ci=(0.5, 0.7),
                  layer_rates={"surface": 0.1, "language": 0.05},
                  diagnostic=True)
        ]}),
    ])
    p = trend.series[0].points[0]
    assert p.baseline_ci_95 == (0.5, 0.7)
    assert p.layer_rates == {"surface": 0.1, "language": 0.05}
    assert p.diagnostic is True


def test_render_markdown_shows_delta_and_series_header():
    trend = build_trend([
        ("a", "2026-04-17", {"rows": [_row(baseline=0.5)]}),
        ("b", "2026-04-24", {"rows": [_row(baseline=0.8)]}),
    ])
    md = render_markdown(trend)
    assert "Bundle trend" in md
    assert "Baseline:" in md
    assert "+0.300" in md  # the delta from 0.5 → 0.8


def test_render_markdown_flags_scanner_version_change():
    trend = build_trend([
        ("a", "2026-04-17", {"rows": [_row(scanner_version="mtg-matrix/0.4")]}),
        ("b", "2026-04-24", {"rows": [_row(scanner_version="mtg-matrix/0.5")]}),
    ])
    md = render_markdown(trend)
    assert "version changes" in md
    assert "scanner_version" in md


def test_render_markdown_flags_dataset_version_change():
    trend = build_trend([
        ("a", "2026-04-17", {"rows": [_row(dataset_version="2026-04-arabic-v1")]}),
        ("b", "2026-05-01", {"rows": [_row(dataset_version="2026-05-arabic-v2")]}),
    ])
    md = render_markdown(trend)
    assert "version changes" in md
    assert "dataset_version" in md


def test_empty_trend_renders_cleanly():
    trend = Trend(bundle_labels=[], series=[])
    md = render_markdown(trend)
    assert "Empty trend" in md


def test_trend_json_round_trip():
    trend = build_trend([
        ("a", "2026-04-17", {"rows": [_row()]}),
    ])
    data = trend.to_dict()
    assert "series" in data and len(data["series"]) == 1
    series = data["series"][0]
    assert series["points"][0]["baseline_score"] == 0.8


def test_first_and_last_accessors():
    trend = build_trend([
        ("a", "2026-04-17", {"rows": [_row(baseline=0.5)]}),
        ("b", "2026-04-24", {"rows": [_row(baseline=0.6)]}),
        ("c", "2026-05-01", {"rows": [_row(baseline=0.7)]}),
    ])
    series = trend.series[0]
    assert series.first.baseline_score == 0.5
    assert series.last.baseline_score == 0.7
