"""Tests for arabic_agent_eval.bundle — canonical matrix bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mtg")

from arabic_agent_eval.bundle import (
    BUNDLE_VERSION,
    BundleError,
    BundleManifest,
    load_bundle,
    validate_bundle,
    write_bundle,
)
from arabic_agent_eval.dataset import EvalItem, ExpectedCall
from arabic_agent_eval.evaluator import BenchmarkResult, EvalResult
from arabic_agent_eval.matrix import (
    MatrixRow,
    ResultMatrix,
    build_matrix,
    scan_with_mtg,
)
from arabic_agent_eval.scoring import Score


def _minimal_matrix() -> ResultMatrix:
    """Build a small ResultMatrix from a synthetic benchmark run."""
    item = EvalItem(
        id="a1", category="simple_function_calling", instruction="",
        dialect="gulf", available_functions=[],
        expected_calls=[ExpectedCall(function="t", arguments={})],
        difficulty="easy",
    )
    br = BenchmarkResult(
        provider="p", model="m",
        results=[
            EvalResult(
                item=item,
                score=Score(
                    item_id="a1", category="simple_function_calling",
                    function_selection=1.0, argument_accuracy=1.0,
                    arabic_preservation=1.0,
                ),
                actual_calls=[{"function": "send_message",
                                "arguments": {"message": "أبي أحجز"}}],
            ),
        ],
    )
    return build_matrix([br])


def test_write_bundle_creates_expected_files(tmp_path: Path):
    matrix = _minimal_matrix()
    out = write_bundle(matrix, tmp_path / "bundle")
    assert (out / "MANIFEST.json").exists()
    assert (out / "matrix.json").exists()
    assert (out / "table.md").exists()
    assert (out / "table.csv").exists()
    # scorecard.html is optional — not written here
    assert not (out / "scorecard.html").exists()


def test_write_bundle_optional_html(tmp_path: Path):
    matrix = _minimal_matrix()
    out = write_bundle(matrix, tmp_path / "bundle", html="<html>scorecard</html>")
    assert (out / "scorecard.html").exists()
    manifest = BundleManifest.from_dict(
        json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    )
    assert "scorecard.html" in manifest.files


def test_write_bundle_copies_run_files(tmp_path: Path):
    matrix = _minimal_matrix()
    run_src = tmp_path / "src.json"
    run_src.write_text(json.dumps({"provider": "p", "model": "m"}), encoding="utf-8")
    out = write_bundle(matrix, tmp_path / "bundle", run_json_files=[run_src])
    copy = out / "runs" / "src.json"
    assert copy.exists()
    manifest = BundleManifest.from_dict(
        json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    )
    assert "runs/src.json" in manifest.files


def test_load_bundle_passes_integrity_check(tmp_path: Path):
    matrix = _minimal_matrix()
    out = write_bundle(matrix, tmp_path / "bundle")
    manifest, matrix_json = load_bundle(out)
    assert manifest.bundle_version == BUNDLE_VERSION
    assert "rows" in matrix_json


def test_load_bundle_rejects_tampered_file(tmp_path: Path):
    matrix = _minimal_matrix()
    out = write_bundle(matrix, tmp_path / "bundle")
    # Tamper with table.md after write
    (out / "table.md").write_text("I am tampered content\n", encoding="utf-8")
    with pytest.raises(BundleError, match="sha256 mismatch"):
        load_bundle(out)


def test_load_bundle_rejects_missing_file(tmp_path: Path):
    matrix = _minimal_matrix()
    out = write_bundle(matrix, tmp_path / "bundle")
    (out / "table.csv").unlink()
    with pytest.raises(BundleError, match="missing file"):
        load_bundle(out)


def test_load_bundle_rejects_bad_version(tmp_path: Path):
    matrix = _minimal_matrix()
    out = write_bundle(matrix, tmp_path / "bundle")
    manifest_path = out / "MANIFEST.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["bundle_version"] = "999"
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with pytest.raises(BundleError, match="bundle_version"):
        load_bundle(out)


def test_load_bundle_rejects_missing_manifest(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(BundleError, match="MANIFEST"):
        load_bundle(empty)


def test_row_summary_surfaces_heuristic_rate_in_manifest(tmp_path: Path):
    """The manifest must carry heuristic_scan_rate per row so the gate
    can inspect it without touching matrix.json (cross-check)."""
    matrix = _minimal_matrix()
    out = write_bundle(matrix, tmp_path / "bundle")
    manifest = BundleManifest.from_dict(
        json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    )
    assert len(manifest.row_summaries) == 1
    summary = manifest.row_summaries[0]
    assert "heuristic_scan_rate" in summary
    assert "run_id" in summary
    assert "scanner_version" in summary


def test_validate_bundle_thin_wrapper(tmp_path: Path):
    matrix = _minimal_matrix()
    out = write_bundle(matrix, tmp_path / "bundle")
    validate_bundle(out)  # must not raise
