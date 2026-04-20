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


# ---------- has_runs semantics ----------


def test_has_runs_false_when_no_runs_copied(tmp_path: Path):
    """A bundle without run_json_files must manifest has_runs=false,
    regardless of what schema_map_tools might suggest."""
    matrix = _minimal_matrix()
    out = write_bundle(matrix, tmp_path / "bundle")
    manifest = BundleManifest.from_dict(
        json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    )
    assert manifest.has_runs is False
    # row_summaries MUST NOT carry has_runs (it's a bundle-level fact)
    for s in manifest.row_summaries:
        assert "has_runs" not in s


def test_has_runs_true_when_runs_copied(tmp_path: Path):
    """When run_json_files are supplied, manifest.has_runs must be true."""
    matrix = _minimal_matrix()
    run_src = tmp_path / "hermes.json"
    run_src.write_text(json.dumps({"provider": "p", "model": "m"}), encoding="utf-8")
    out = write_bundle(matrix, tmp_path / "bundle", run_json_files=[run_src])
    manifest = BundleManifest.from_dict(
        json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    )
    assert manifest.has_runs is True


def test_load_bundle_rejects_manifest_has_runs_lie(tmp_path: Path):
    """A manifest that claims has_runs=true without actually shipping
    runs/ files must fail integrity — otherwise the flag is
    unauditable."""
    matrix = _minimal_matrix()
    out = write_bundle(matrix, tmp_path / "bundle")  # has_runs=false
    manifest_path = out / "MANIFEST.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["has_runs"] = True  # lie
    manifest_path.write_text(
        json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(BundleError, match="has_runs"):
        validate_bundle(out)


# ---------- invocation provenance ----------


def test_invocation_default_empty(tmp_path: Path):
    """When no invocation dict is passed, manifest carries an empty one
    (not a missing key) so downstream schemas are stable."""
    matrix = _minimal_matrix()
    out = write_bundle(matrix, tmp_path / "bundle")
    manifest = BundleManifest.from_dict(
        json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    )
    assert manifest.invocation == {}


def test_invocation_passed_through_to_manifest(tmp_path: Path):
    """write_bundle's `invocation` kwarg lands in MANIFEST.json so
    build_bundle.py can stamp the exact build-command context."""
    matrix = _minimal_matrix()
    invocation = {
        "generator": "scripts/build_bundle.py",
        "generator_version": "build_bundle/0.2",
        "git_ref": "abc123",
        "schema_map_source": "../hurmoz/tool-schemas",
        "run_json_sha256": {"hermes.json": "deadbeef" * 8},
    }
    out = write_bundle(matrix, tmp_path / "bundle", invocation=invocation)
    manifest = BundleManifest.from_dict(
        json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    )
    assert manifest.invocation == invocation


def test_invocation_survives_round_trip_and_integrity(tmp_path: Path):
    """Manifest integrity check must accept bundles with non-empty
    invocation (the field is part of the signed manifest JSON but
    not of the per-file sha256 checks)."""
    matrix = _minimal_matrix()
    out = write_bundle(
        matrix, tmp_path / "bundle",
        invocation={"generator": "test", "generator_version": "0.0.1"},
    )
    manifest, _ = load_bundle(out)  # must not raise
    assert manifest.invocation["generator"] == "test"


# ---------- raw evidence ----------


def test_raw_files_copied_into_bundle(tmp_path: Path):
    """write_bundle(raw_files=...) copies each file under bundle/raw/
    and registers it in the manifest."""
    matrix = _minimal_matrix()
    raw_src = tmp_path / "src"
    raw_src.mkdir()
    (raw_src / "request-1.json").write_text("{\"redacted\": true}", encoding="utf-8")
    (raw_src / "trace.log").write_text("sample trace", encoding="utf-8")

    out = write_bundle(
        matrix, tmp_path / "bundle", raw_files=list(raw_src.iterdir()),
    )
    assert (out / "raw" / "request-1.json").exists()
    assert (out / "raw" / "trace.log").exists()
    manifest = BundleManifest.from_dict(
        json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    )
    raw_entries = sorted(k for k in manifest.files if k.startswith("raw/"))
    assert raw_entries == ["raw/request-1.json", "raw/trace.log"]


def test_raw_files_covered_by_manifest_integrity(tmp_path: Path):
    """Tampering a raw file after publish must break load_bundle."""
    matrix = _minimal_matrix()
    raw_src = tmp_path / "src"
    raw_src.mkdir()
    (raw_src / "note.txt").write_text("original", encoding="utf-8")
    out = write_bundle(
        matrix, tmp_path / "bundle", raw_files=list(raw_src.iterdir()),
    )
    (out / "raw" / "note.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(BundleError, match="sha256 mismatch"):
        load_bundle(out)


def test_raw_files_optional_and_default_absent(tmp_path: Path):
    """No raw_files → no raw/ directory + no raw/* entries in manifest."""
    matrix = _minimal_matrix()
    out = write_bundle(matrix, tmp_path / "bundle")
    assert not (out / "raw").exists()
    manifest = BundleManifest.from_dict(
        json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    )
    assert not any(k.startswith("raw/") for k in manifest.files)


def test_raw_files_reject_missing_source(tmp_path: Path):
    """Non-existent source paths fail loud."""
    matrix = _minimal_matrix()
    missing = tmp_path / "nope.txt"
    with pytest.raises(BundleError, match="raw file not found"):
        write_bundle(
            matrix, tmp_path / "bundle", raw_files=[missing],
        )


# ---------- raw evidence index ----------


def test_raw_index_stamped_in_manifest(tmp_path: Path):
    """raw_index entries land in manifest.raw_index, keyed by the
    `raw/<filename>` path that matches the files dict."""
    matrix = _minimal_matrix()
    raw_src = tmp_path / "src"
    raw_src.mkdir()
    (raw_src / "req-1.json").write_text("{}", encoding="utf-8")
    out = write_bundle(
        matrix, tmp_path / "bundle",
        raw_files=list(raw_src.iterdir()),
        raw_index={
            "req-1.json": {
                "type": "request",
                "source": "openrouter",
                "redacted": True,
                "redaction_note": "API key stripped",
            },
        },
    )
    manifest = BundleManifest.from_dict(
        json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    )
    assert "raw/req-1.json" in manifest.raw_index
    entry = manifest.raw_index["raw/req-1.json"]
    assert entry["type"] == "request"
    assert entry["source"] == "openrouter"
    assert entry["redacted"] is True
    assert entry["redaction_note"] == "API key stripped"


def test_raw_index_rejects_dangling_entry(tmp_path: Path):
    """A raw_index entry without a matching raw file must fail loud —
    stale entries would be misleading."""
    matrix = _minimal_matrix()
    with pytest.raises(BundleError, match="does not correspond"):
        write_bundle(
            matrix, tmp_path / "bundle",
            raw_files=None,
            raw_index={"ghost.json": {"type": "request"}},
        )


def test_raw_index_accepts_prefixed_or_bare_key(tmp_path: Path):
    """Callers can pass keys as `req.json` or `raw/req.json` — the
    writer normalizes to the `raw/` form."""
    matrix = _minimal_matrix()
    raw_src = tmp_path / "src"
    raw_src.mkdir()
    (raw_src / "a.json").write_text("{}", encoding="utf-8")
    (raw_src / "b.json").write_text("{}", encoding="utf-8")
    out = write_bundle(
        matrix, tmp_path / "bundle",
        raw_files=list(raw_src.iterdir()),
        raw_index={
            "a.json": {"type": "request"},
            "raw/b.json": {"type": "trace"},
        },
    )
    manifest = BundleManifest.from_dict(
        json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    )
    assert manifest.raw_index["raw/a.json"]["type"] == "request"
    assert manifest.raw_index["raw/b.json"]["type"] == "trace"


def test_raw_index_empty_by_default(tmp_path: Path):
    """Without raw_index the manifest section stays {} but is present."""
    matrix = _minimal_matrix()
    out = write_bundle(matrix, tmp_path / "bundle")
    manifest = BundleManifest.from_dict(
        json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    )
    assert manifest.raw_index == {}
