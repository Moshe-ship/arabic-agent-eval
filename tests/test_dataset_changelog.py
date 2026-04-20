"""Discipline test: the current DATASET_VERSION must appear in
docs/DATASET_CHANGELOG.md. Prevents version bumps without an entry."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_current_dataset_version_is_in_changelog():
    from arabic_agent_eval import DATASET_VERSION

    changelog = (ROOT / "docs" / "DATASET_CHANGELOG.md").read_text(encoding="utf-8")
    assert DATASET_VERSION in changelog, (
        f"DATASET_VERSION={DATASET_VERSION!r} has no entry in "
        f"docs/DATASET_CHANGELOG.md. Add one before merging — the "
        f"version label is meaningless without a changelog entry."
    )


def test_changelog_uses_documented_version_format():
    """Every version header in the changelog should match the documented
    <YYYY-MM>-<language>-v<N> format so grep / tooling can parse it."""
    import re

    changelog = (ROOT / "docs" / "DATASET_CHANGELOG.md").read_text(encoding="utf-8")
    pattern = re.compile(r"### `(\d{4}-\d{2}-[a-z]+-v\d+)`")
    matches = pattern.findall(changelog)
    assert matches, (
        "No version headers matching `<YYYY-MM>-<language>-v<N>` found "
        "in DATASET_CHANGELOG.md"
    )
    # The format convention is documented in the 'Format' section; every
    # version header must match it.
    for m in matches:
        # Reject impossible months (13+)
        _, month, _ = m.split("-", 2)
        assert 1 <= int(month) <= 12, f"impossible month in version: {m}"