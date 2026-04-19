"""Tests for scripts/check_publish_ready.py publish gate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mtg")

from arabic_agent_eval.bundle import load_bundle, write_bundle
from arabic_agent_eval.dataset import EvalItem, ExpectedCall
from arabic_agent_eval.evaluator import BenchmarkResult, EvalResult
from arabic_agent_eval.matrix import build_matrix
from arabic_agent_eval.scoring import Score


ROOT = Path(__file__).resolve().parent.parent
GATE_SCRIPT = ROOT / "scripts" / "check_publish_ready.py"


def _item(item_id: str, value: str = "أبي أحجز") -> EvalResult:
    item = EvalItem(
        id=item_id, category="simple_function_calling", instruction="",
        dialect="gulf", available_functions=[],
        expected_calls=[ExpectedCall(function="t", arguments={})],
        difficulty="easy",
    )
    return EvalResult(
        item=item,
        score=Score(
            item_id=item_id, category="simple_function_calling",
            function_selection=1.0, argument_accuracy=1.0, arabic_preservation=1.0,
        ),
        actual_calls=[{"function": "send_message", "arguments": {"message": value}}],
    )


def _clean_bundle(tmp_path: Path) -> Path:
    """A publish-ready bundle: 2+ items, low heuristic_scan_rate."""
    br = BenchmarkResult(
        provider="p", model="m",
        results=[_item("a"), _item("b")],
    )
    matrix = build_matrix([br])
    return write_bundle(matrix, tmp_path / "bundle")


def _run_gate(bundle: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GATE_SCRIPT), str(bundle), *args],
        capture_output=True, text=True,
    )


def test_gate_passes_on_clean_bundle_via_cli(tmp_path: Path):
    bundle = _clean_bundle(tmp_path)
    # The default FUNCTIONS registry has no x-mtg so heuristic_scan_rate
    # will be 1.0 — expected to fail the default threshold. Loosen
    # threshold to match the shape of a pure-heuristic run but still
    # require provenance + CIs.
    result = _run_gate(bundle, "--heuristic-max", "1.0")
    assert result.returncode == 0, result.stderr
    assert "PUBLISH_READY" in result.stdout


def test_gate_fails_when_threshold_exceeded(tmp_path: Path):
    bundle = _clean_bundle(tmp_path)
    # Default threshold is 0.10 but this run is pure-heuristic.
    result = _run_gate(bundle)
    assert result.returncode != 0
    assert "heuristic_scan_rate" in result.stderr


def test_allow_diagnostic_passes_only_when_flag_is_set(tmp_path: Path):
    bundle = _clean_bundle(tmp_path)
    # Without --allow-diagnostic: fails
    r1 = _run_gate(bundle)
    assert r1.returncode != 0
    # With --allow-diagnostic: gate inspects diagnostic marker (which
    # the scanner set because heuristic_scan_rate > threshold).
    r2 = _run_gate(bundle, "--allow-diagnostic")
    assert r2.returncode == 0, r2.stderr


def test_gate_detects_tampered_manifest(tmp_path: Path):
    bundle = _clean_bundle(tmp_path)
    # Tamper with table.md → manifest sha256 mismatch
    (bundle / "table.md").write_text("mutated\n", encoding="utf-8")
    result = _run_gate(bundle)
    assert result.returncode != 0
    assert "integrity" in result.stderr or "sha256" in result.stderr


def test_gate_detects_empty_scanner_version(tmp_path: Path):
    """If rows disagreed on scanner_version, the bundle writer emits an
    empty scanner_version in the manifest — gate must block that."""
    bundle = _clean_bundle(tmp_path)
    manifest_path = bundle / "MANIFEST.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["scanner_version"] = ""
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    # sha256 for manifest itself isn't checked (it's the index), so this
    # doesn't break integrity; the gate must still flag it.
    result = _run_gate(bundle, "--heuristic-max", "1.0")
    assert result.returncode != 0
    assert "scanner_version" in result.stderr


def test_gate_detects_missing_run_metadata(tmp_path: Path):
    bundle = _clean_bundle(tmp_path)
    # Corrupt matrix.json: strip run_metadata. This breaks integrity
    # (manifest sha256 won't match), but we want to test the metadata
    # check specifically — so re-author the manifest too.
    matrix_path = bundle / "matrix.json"
    matrix_data = json.loads(matrix_path.read_text(encoding="utf-8"))
    for row in matrix_data["rows"]:
        row["run_metadata"] = {}
    matrix_path.write_text(
        json.dumps(matrix_data, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    # Recompute the sha256 for matrix.json so integrity still passes —
    # we're testing the gate's metadata check in isolation.
    import hashlib
    new_hash = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    manifest_path = bundle / "MANIFEST.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_data["files"]["matrix.json"] = new_hash
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True), encoding="utf-8"
    )

    result = _run_gate(bundle, "--heuristic-max", "1.0", "--allow-diagnostic")
    assert result.returncode != 0
    assert "run_metadata missing" in result.stderr


def test_gate_rejects_single_item_without_ci(tmp_path: Path):
    """n_items >= 2 triggers the CI requirement. Single-item runs are
    allowed to skip CI (can't bootstrap one observation)."""
    br = BenchmarkResult(
        provider="p", model="m",
        results=[_item("solo")],
    )
    matrix = build_matrix([br])
    bundle_dir = tmp_path / "solo"
    write_bundle(matrix, bundle_dir)
    # Single item → no CI required → gate passes on the CI check
    # (other checks may still fire; relax threshold for clarity).
    result = _run_gate(bundle_dir, "--heuristic-max", "1.0")
    # scanner_version still pinned, run_metadata complete — should pass.
    assert result.returncode == 0, result.stderr


def test_gate_cross_checks_manifest_heuristic_rate(tmp_path: Path):
    """Manifest row_summaries must agree with matrix.json. If someone
    edits the manifest to hide the heuristic rate, the cross-check
    surfaces the tampering."""
    bundle = _clean_bundle(tmp_path)
    manifest_path = bundle / "MANIFEST.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    for s in data["row_summaries"]:
        s["heuristic_scan_rate"] = 0.0  # lie
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    result = _run_gate(bundle, "--allow-diagnostic")
    assert result.returncode != 0
    assert "manifest heuristic_scan_rate" in result.stderr
