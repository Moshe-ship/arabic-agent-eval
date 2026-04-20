"""Discipline test: dataset content changes must bump DATASET_VERSION.

If the dataset content changes (items added, removed, or mutated),
the fingerprint computed over the full dataset will differ from the
value locked in `DATASET_FINGERPRINT_LOCK`. This test fails loudly in
that case, with the new fingerprint in the message, forcing the
developer to either revert the change or bump the version + add a new
lock entry.

Complements `test_dataset_changelog.py` — that one ensures the
version label is documented, this one ensures the label actually
tracks content.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mtg")

from arabic_agent_eval import DATASET_FINGERPRINT_LOCK, DATASET_VERSION
from arabic_agent_eval.dataset import Dataset
from arabic_agent_eval.evaluator import BenchmarkResult, EvalResult
from arabic_agent_eval.matrix import _fingerprint_benchmark_items
from arabic_agent_eval.scoring import Score


def _current_fingerprint() -> str:
    ds = Dataset()
    results = [
        EvalResult(
            item=item,
            score=Score(item_id=item.id, category=item.category),
            actual_calls=[],
        )
        for item in ds
    ]
    br = BenchmarkResult(provider="lock", model="lock", results=results)
    return _fingerprint_benchmark_items(br)


def test_current_version_is_in_lock():
    assert DATASET_VERSION in DATASET_FINGERPRINT_LOCK, (
        f"DATASET_VERSION={DATASET_VERSION!r} has no entry in "
        f"DATASET_FINGERPRINT_LOCK. Add one with the computed fingerprint "
        f"(see test_dataset_content_matches_locked_fingerprint for the value)."
    )


def test_dataset_content_matches_locked_fingerprint():
    """If this fails, someone changed dataset content without bumping
    DATASET_VERSION. Either revert the change or bump the version and
    add a new lock entry."""
    expected = DATASET_FINGERPRINT_LOCK.get(DATASET_VERSION)
    actual = _current_fingerprint()
    assert actual == expected, (
        "\n\n"
        f"Dataset content changed! Fingerprint for DATASET_VERSION="
        f"{DATASET_VERSION!r} no longer matches the locked value.\n\n"
        f"  locked:   {expected}\n"
        f"  computed: {actual}\n\n"
        "If this is intentional:\n"
        "  1. Bump DATASET_VERSION in arabic_agent_eval/__init__.py\n"
        "  2. Add a new entry to DATASET_FINGERPRINT_LOCK with the "
        "computed fingerprint above\n"
        "  3. Add an entry to docs/DATASET_CHANGELOG.md describing the change\n\n"
        "If this is accidental (you didn't mean to edit dataset content):\n"
        "  Revert your changes to arabic_agent_eval/dataset_items.py "
        "or wherever the edit landed.\n"
    )


def test_all_locked_versions_have_changelog_entries():
    """Every lock entry must also have a changelog entry, so past
    versions are discoverable without reading the code."""
    from pathlib import Path
    changelog = (
        Path(__file__).resolve().parent.parent / "docs" / "DATASET_CHANGELOG.md"
    ).read_text(encoding="utf-8")
    for version in DATASET_FINGERPRINT_LOCK:
        assert version in changelog, (
            f"DATASET_FINGERPRINT_LOCK has an entry for {version!r} but "
            f"docs/DATASET_CHANGELOG.md has no matching changelog entry. "
            f"Every locked version must be documented."
        )
