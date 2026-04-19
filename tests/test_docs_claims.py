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
