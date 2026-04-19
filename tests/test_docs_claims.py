"""Tests for scripts/check_docs_claims.py — the docs-claims gate."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import check_docs_claims as gate  # type: ignore[import-not-found]


def test_is_claim_line_detects_percentage():
    assert gate._is_claim_line("MTG reduces violations by 78.4%")


def test_is_claim_line_detects_accuracy():
    assert gate._is_claim_line("The model achieves 0.85 accuracy on dialect")


def test_is_claim_line_detects_beats():
    assert gate._is_claim_line("Hermes beats Claude by 0.12 on MSA")


def test_is_claim_line_ignores_plain_prose():
    assert not gate._is_claim_line("This README describes the bundle format")


def test_is_claim_line_ignores_synthetic_prefix():
    assert not gate._is_claim_line("synthetic: score is 0.90 in the example bundle")


def test_is_claim_line_ignores_example_prefix():
    assert not gate._is_claim_line("example: accuracy 0.85 — not a real measurement")


def test_is_claim_line_ignores_empty():
    assert not gate._is_claim_line("")
    assert not gate._is_claim_line("   ")


def test_is_claim_line_ignores_pure_markdown_markers():
    assert not gate._is_claim_line("# Heading")
    assert not gate._is_claim_line("- bullet with no numbers")


def test_is_claim_line_catches_benchmark_number():
    assert gate._is_claim_line("Achieves 92% function selection accuracy")


def test_references_bundle_match():
    assert gate._references_bundle(["see bundles/2026-04-hermes-run/"])
    assert gate._references_bundle(
        ["Baseline: 0.85", "Source: bundles/main/table.md"]
    )


def test_references_bundle_no_match():
    assert not gate._references_bundle(
        ["Baseline: 0.85", "Source: see the docs folder"]
    )


def test_claim_disclaimer_case_insensitive():
    """SYNTHETIC / Example / Illustrative all match regardless of case."""
    assert not gate._is_claim_line("Synthetic: score 0.85 — not real")
    assert not gate._is_claim_line("EXAMPLE: 0.92 accuracy")
    assert not gate._is_claim_line("Illustrative: the model reaches 78%")


# ---------- stale-bundle citation detection ----------


def test_stale_bundle_citation_detected(tmp_path, monkeypatch):
    """A doc that cites bundles/missing/ must flag when the directory
    doesn't exist."""
    # Redirect the gate's repo root to a temp dir for this test
    monkeypatch.setattr(gate, "_REPO_ROOT", tmp_path)
    (tmp_path / "bundles").mkdir()
    # Write a markdown file that cites a non-existent bundle
    doc = tmp_path / "README.md"
    doc.write_text(
        "See bundles/missing-one/table.md for the results.\n"
        "And also bundles/missing-two/ for comparison.\n",
        encoding="utf-8",
    )
    stale = gate._stale_bundle_citations(doc)
    names = sorted(s[2] for s in stale)
    assert names == ["missing-one", "missing-two"]


def test_existing_bundle_not_flagged(tmp_path, monkeypatch):
    """A cited bundle that exists (with MANIFEST.json) must NOT flag."""
    monkeypatch.setattr(gate, "_REPO_ROOT", tmp_path)
    # Create a real bundle
    bundle = tmp_path / "bundles" / "real-run"
    bundle.mkdir(parents=True)
    (bundle / "MANIFEST.json").write_text("{}", encoding="utf-8")
    doc = tmp_path / "README.md"
    doc.write_text("See bundles/real-run/ for numbers.\n", encoding="utf-8")
    stale = gate._stale_bundle_citations(doc)
    assert stale == []


def test_fenced_code_block_excluded_from_stale_scan(tmp_path, monkeypatch):
    """Bundle references inside fenced code blocks are examples, not
    live citations — don't flag them as stale."""
    monkeypatch.setattr(gate, "_REPO_ROOT", tmp_path)
    (tmp_path / "bundles").mkdir()
    doc = tmp_path / "README.md"
    doc.write_text(
        "Example usage:\n"
        "```\n"
        "python scripts/diff_bundles.py bundles/before/ bundles/after/\n"
        "```\n",
        encoding="utf-8",
    )
    stale = gate._stale_bundle_citations(doc)
    assert stale == []


def test_wildcard_bundle_references_not_flagged(tmp_path, monkeypatch):
    """`bundles/**` is a glob pattern, not a citation — ignore."""
    monkeypatch.setattr(gate, "_REPO_ROOT", tmp_path)
    (tmp_path / "bundles").mkdir()
    doc = tmp_path / "README.md"
    doc.write_text("CI runs on bundles/**/MANIFEST.json\n", encoding="utf-8")
    stale = gate._stale_bundle_citations(doc)
    assert stale == []


def test_existing_bundle_names_filters_unfinished_directories(tmp_path, monkeypatch):
    """A directory under bundles/ without MANIFEST.json is NOT treated
    as a real bundle — cited names must match a manifest'd dir."""
    monkeypatch.setattr(gate, "_REPO_ROOT", tmp_path)
    (tmp_path / "bundles" / "half-done").mkdir(parents=True)
    (tmp_path / "bundles" / "done").mkdir(parents=True)
    (tmp_path / "bundles" / "done" / "MANIFEST.json").write_text("{}", encoding="utf-8")
    assert gate._existing_bundle_names() == {"done"}
