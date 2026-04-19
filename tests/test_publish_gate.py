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


_SCHEMA_MAP = {
    "send_message": {
        "name": "send_message",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "x-mtg": {
                        "slot_type": "free_text",
                        "script": "ar",
                        "mode": "advisory",
                    },
                },
            },
        },
    },
}


def _clean_bundle(tmp_path: Path) -> Path:
    """A publish-ready bundle: 2+ items, schema-bound (heuristic_scan_rate=0),
    includes a `runs/` source copy. Non-diagnostic."""
    br = BenchmarkResult(
        provider="p", model="m",
        results=[_item("a"), _item("b")],
    )
    matrix = build_matrix([br], tool_schema_map=_SCHEMA_MAP)
    # Provide a source run file so `runs/` populates in the bundle
    run_src = tmp_path / "src.json"
    run_src.write_text(
        json.dumps({"provider": "p", "model": "m", "results": []}),
        encoding="utf-8",
    )
    return write_bundle(matrix, tmp_path / "bundle", run_json_files=[run_src])


def _heuristic_bundle(tmp_path: Path) -> Path:
    """Pure-heuristic bundle: every row diagnostic. Useful for testing
    gate-reject paths on schema-coverage grounds."""
    br = BenchmarkResult(
        provider="p", model="m",
        results=[_item("a"), _item("b")],
    )
    matrix = build_matrix([br])  # no schema map → 100% heuristic
    return write_bundle(matrix, tmp_path / "heuristic-bundle")


def _run_gate(bundle: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GATE_SCRIPT), str(bundle), *args],
        capture_output=True, text=True,
    )


def test_gate_passes_on_clean_bundle_via_cli(tmp_path: Path):
    """A schema-bound bundle with runs/ present passes the default gate."""
    bundle = _clean_bundle(tmp_path)
    result = _run_gate(bundle)
    assert result.returncode == 0, result.stderr
    assert "PUBLISH_READY" in result.stdout


def test_gate_fails_when_threshold_exceeded(tmp_path: Path):
    """Pure-heuristic bundle fails the default heuristic threshold."""
    bundle = _heuristic_bundle(tmp_path)
    # Needs --allow-no-runs because the heuristic bundle has no runs/.
    # The heuristic_scan_rate failure is what we want to observe.
    result = _run_gate(bundle, "--allow-no-runs")
    assert result.returncode != 0
    assert "heuristic_scan_rate" in result.stderr


def test_allow_diagnostic_alone_rejects_all_diagnostic_bundle(tmp_path: Path):
    """Even with --allow-diagnostic, a bundle where EVERY row is
    diagnostic is theater and must be rejected."""
    bundle = _heuristic_bundle(tmp_path)
    result = _run_gate(bundle, "--allow-no-runs", "--allow-diagnostic")
    assert result.returncode != 0
    assert "theater" in result.stderr or "non-diagnostic" in result.stderr


def test_allow_diagnostic_plus_min_zero_publishes_heuristic(tmp_path: Path):
    """Deliberate override path: --allow-diagnostic + --min-non-diagnostic 0
    lets a pure-diagnostic bundle through (documented escape hatch)."""
    bundle = _heuristic_bundle(tmp_path)
    result = _run_gate(
        bundle, "--allow-no-runs", "--allow-diagnostic",
        "--min-non-diagnostic", "0",
    )
    assert result.returncode == 0, result.stderr


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
    result = _run_gate(bundle)
    assert result.returncode != 0
    assert "scanner_version" in result.stderr


def test_gate_detects_missing_run_metadata(tmp_path: Path):
    bundle = _clean_bundle(tmp_path)
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

    result = _run_gate(bundle)
    assert result.returncode != 0
    assert "run_metadata missing" in result.stderr


def test_gate_requires_runs_by_default(tmp_path: Path):
    """A bundle without runs/ fails the default gate."""
    bundle = _heuristic_bundle(tmp_path)  # no runs/ by construction
    result = _run_gate(bundle)
    assert result.returncode != 0
    assert "runs/" in result.stderr or "runs" in result.stderr


def test_gate_allows_no_runs_with_flag(tmp_path: Path):
    """--allow-no-runs waives the runs/ requirement for synthetic bundles."""
    bundle = _clean_bundle(tmp_path)
    # Remove the runs/ file + update the manifest so the integrity
    # check still passes — we're testing the gate's runs/ requirement
    # in isolation, not manifest integrity.
    runs_key = next(k for k in (bundle / "MANIFEST.json").read_text().split()
                    if "runs/" in k) if False else None
    import hashlib
    manifest_path = bundle / "MANIFEST.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    for rel in list(manifest_data["files"].keys()):
        if rel.startswith("runs/"):
            (bundle / rel).unlink()
            manifest_data["files"].pop(rel)
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True), encoding="utf-8"
    )
    # Default gate: should still fail for missing runs/
    r1 = _run_gate(bundle)
    assert r1.returncode != 0
    # With --allow-no-runs: should pass
    r2 = _run_gate(bundle, "--allow-no-runs")
    assert r2.returncode == 0, r2.stderr


def test_gate_rejects_all_diagnostic_bundle_even_with_allow_diagnostic(tmp_path: Path):
    """Default --min-non-diagnostic=1 blocks pure-theater bundles even
    when --allow-diagnostic is passed."""
    bundle = _heuristic_bundle(tmp_path)
    result = _run_gate(bundle, "--allow-no-runs", "--allow-diagnostic")
    assert result.returncode != 0
    assert "theater" in result.stderr or "non-diagnostic" in result.stderr


def test_gate_flags_missing_schema_map_tools(tmp_path: Path):
    """run_metadata must carry schema_map_tools so the provenance of
    the scanner's schema coverage is auditable."""
    bundle = _clean_bundle(tmp_path)
    matrix_path = bundle / "matrix.json"
    matrix_data = json.loads(matrix_path.read_text(encoding="utf-8"))
    for row in matrix_data["rows"]:
        row["run_metadata"].pop("schema_map_tools", None)
    matrix_path.write_text(
        json.dumps(matrix_data, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    import hashlib
    new_hash = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    manifest_path = bundle / "MANIFEST.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_data["files"]["matrix.json"] = new_hash
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True), encoding="utf-8"
    )
    result = _run_gate(bundle)
    assert result.returncode != 0
    assert "schema_map_tools" in result.stderr


def test_gate_rejects_single_item_with_missing_runs(tmp_path: Path):
    """Single-item runs skip the CI requirement but still need runs/."""
    br = BenchmarkResult(
        provider="p", model="m",
        results=[_item("solo")],
    )
    matrix = build_matrix([br], tool_schema_map=_SCHEMA_MAP)
    bundle_dir = tmp_path / "solo"
    write_bundle(matrix, bundle_dir)
    # Default gate: fails because no runs/
    r1 = _run_gate(bundle_dir)
    assert r1.returncode != 0
    # With --allow-no-runs: passes
    r2 = _run_gate(bundle_dir, "--allow-no-runs")
    assert r2.returncode == 0, r2.stderr


def test_gate_cross_checks_manifest_heuristic_rate(tmp_path: Path):
    """Manifest row_summaries must agree with matrix.json. If someone
    edits the manifest to hide the heuristic rate, the cross-check
    surfaces the tampering."""
    bundle = _heuristic_bundle(tmp_path)
    manifest_path = bundle / "MANIFEST.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    for s in data["row_summaries"]:
        s["heuristic_scan_rate"] = 0.0  # lie
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    result = _run_gate(
        bundle, "--allow-no-runs", "--allow-diagnostic", "--min-non-diagnostic", "0",
    )
    assert result.returncode != 0
    assert "manifest heuristic_scan_rate" in result.stderr
