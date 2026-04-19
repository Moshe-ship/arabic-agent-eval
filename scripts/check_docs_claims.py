#!/usr/bin/env python3
"""Block docs PRs that add unsourced model/benchmark claims.

A claim is any line added to a Markdown file in this repo that looks
like a model-performance statement: percentages, accuracy figures,
violation-rate numbers, score comparisons. Every such added line must
reference a bundle path under `bundles/` on the same line or within
5 lines above/below, OR be inside a fenced code block (quoting
output), OR be prefixed with `synthetic:` / `example:` /
`illustrative:` (clearly labeled non-claim).

Purpose: once a real bundle exists, this gate prevents drift where
docs quote numbers that don't trace back to a committed artifact. Until
any bundle exists under `bundles/`, the gate only permits Markdown
changes that add claims if the PR creates the first bundle in the
same diff.

Usage:

    # Compare worktree to main (local)
    python scripts/check_docs_claims.py --base main

    # CI mode (reads diff from stdin / git)
    python scripts/check_docs_claims.py --base origin/main --ci
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Regex patterns that identify a claim-shaped line. Tuned to be strict
# about model numbers while permitting ordinary prose.
_CLAIM_KEYWORD = (
    r"accuracy|score|rate|violations?|drift|hallucinat\w+|"
    r"beats?|improves?|reduces?|outperforms?|increases?"
)

_CLAIM_PATTERNS = [
    # Percentages: "78.4%", "5-10%", "≥ 0.9%"
    re.compile(r"\d+(?:[.,]\d+)?\s*%"),
    # Keyword-then-number: "accuracy of 0.85"
    re.compile(
        rf"\b({_CLAIM_KEYWORD})\b.{{0,40}}\b\d+(?:[.,]\d+)?\b",
        re.IGNORECASE,
    ),
    # Number-then-keyword: "0.85 accuracy"
    re.compile(
        rf"\b\d+(?:[.,]\d+)?\b.{{0,40}}\b({_CLAIM_KEYWORD})\b",
        re.IGNORECASE,
    ),
]

# Labels that explicitly disclaim: "these numbers are not a real result".
_DISCLAIMER_PREFIXES = ("synthetic:", "example:", "illustrative:", "hypothetical:")

# Bundle reference regex — link to a file under bundles/ OR a mention
# of a bundle path.
_BUNDLE_REFERENCE = re.compile(r"bundles/([a-zA-Z0-9_\-.]+)(?:/[a-zA-Z0-9_\-./]*)?")

_MARKDOWN_GLOBS = ("*.md", "*.MD", "*.markdown")


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), *args],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def _changed_markdown_files(base: str) -> list[Path]:
    diff = _git(["diff", "--name-only", "--diff-filter=AM", f"{base}...HEAD"])
    out: list[Path] = []
    for name in diff.splitlines():
        name = name.strip()
        if not name:
            continue
        p = _REPO_ROOT / name
        if p.suffix.lower() in (".md", ".markdown") and p.exists():
            out.append(p)
    return out


def _diff_added_lines(file: Path, base: str) -> list[tuple[int, str]]:
    """Return [(line_number_in_HEAD, added_text)] for lines added
    against `base`."""
    raw = _git(["diff", "-U0", f"{base}...HEAD", "--", str(file)])
    out: list[tuple[int, str]] = []
    cur_new = 0
    hunk = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
    for line in raw.splitlines():
        m = hunk.match(line)
        if m:
            cur_new = int(m.group(1))
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            out.append((cur_new, line[1:]))
            cur_new += 1
        elif line.startswith(" "):
            cur_new += 1
    return out


def _is_claim_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    low = stripped.lower()
    if any(low.startswith(p) for p in _DISCLAIMER_PREFIXES):
        return False
    # Heading markers / bullet markers shouldn't by themselves count
    if stripped.lstrip("#-*>| ").strip() == "":
        return False
    for pat in _CLAIM_PATTERNS:
        if pat.search(stripped):
            return True
    return False


def _references_bundle(lines_window: Iterable[str]) -> bool:
    for ln in lines_window:
        if _BUNDLE_REFERENCE.search(ln):
            return True
    return False


def _any_bundle_exists() -> bool:
    bundles_dir = _REPO_ROOT / "bundles"
    if not bundles_dir.is_dir():
        return False
    return any((child / "MANIFEST.json").exists() for child in bundles_dir.iterdir())


def _existing_bundle_names() -> set[str]:
    """Return bundle directory names currently present under `bundles/`
    (only those with a MANIFEST.json). Used to detect stale citations."""
    bundles_dir = _REPO_ROOT / "bundles"
    if not bundles_dir.is_dir():
        return set()
    return {
        child.name
        for child in bundles_dir.iterdir()
        if child.is_dir() and (child / "MANIFEST.json").exists()
    }


def _stale_bundle_citations(file: Path) -> list[tuple[int, str, str]]:
    """Scan the full file (not just added lines) for bundle citations
    that point at directories which no longer exist. Returns a list of
    (line_no, line_text, stale_bundle_name).

    Runs even on files the author didn't touch — a rename elsewhere
    can invalidate citations here."""
    try:
        text = file.read_text(encoding="utf-8")
    except OSError:
        return []
    existing = _existing_bundle_names()
    out: list[tuple[int, str, str]] = []
    in_fence = False
    for i, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _BUNDLE_REFERENCE.finditer(line):
            bundle_name = match.group(1)
            if "*" in bundle_name:
                continue
            if bundle_name not in existing:
                out.append((i, line.strip(), bundle_name))
    return out


def _file_in_code_block(file: Path, line_no: int) -> bool:
    """Is line `line_no` (1-indexed) inside a fenced code block?"""
    try:
        text = file.read_text(encoding="utf-8")
    except OSError:
        return False
    in_fence = False
    for i, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if i == line_no:
            return in_fence
    return False


def _file_window(file: Path, line_no: int, radius: int = 5) -> list[str]:
    try:
        lines = file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    lo = max(0, line_no - 1 - radius)
    hi = min(len(lines), line_no - 1 + radius + 1)
    return lines[lo:hi]


def check(base: str) -> list[str]:
    reasons: list[str] = []
    bundle_present = _any_bundle_exists()
    files = _changed_markdown_files(base)

    # Stale-bundle scan — runs over every Markdown file, not just the
    # ones changed in this PR. A rename or deletion elsewhere can
    # invalidate a citation on a file the author didn't touch; still
    # block the PR until docs catch up.
    for pattern in ("**/*.md", "**/*.markdown"):
        for p in _REPO_ROOT.glob(pattern):
            rel = p.relative_to(_REPO_ROOT)
            parts = rel.parts
            if parts and parts[0] in ("bundles", "examples"):
                continue
            if ".git" in parts or "node_modules" in parts:
                continue
            for line_no, _line_text, bundle_name in _stale_bundle_citations(p):
                reasons.append(
                    f"{rel}:{line_no}: cites `bundles/{bundle_name}/` "
                    f"which does not exist under `bundles/`. Update the "
                    f"citation or restore the bundle."
                )

    for md_file in files:
        # Allow the bundles/ index README to carry numbers freely —
        # it's literally quoting from a bundle it ships alongside.
        rel = md_file.relative_to(_REPO_ROOT)
        if str(rel).startswith("bundles/"):
            continue
        # Allow synthetic example bundle's table/docs
        if str(rel).startswith("examples/"):
            continue
        # CHANGELOG entries describe work; allow percentages as long as
        # they're not making model-performance claims. Simplest: allow
        # CHANGELOG completely — reviewer gates PRs anyway.
        if md_file.name.upper() == "CHANGELOG.MD":
            continue

        added = _diff_added_lines(md_file, base)
        for line_no, text in added:
            if not _is_claim_line(text):
                continue
            if _file_in_code_block(md_file, line_no):
                continue
            window = _file_window(md_file, line_no, radius=5)
            if _references_bundle([text, *window]):
                continue
            if not bundle_present:
                reasons.append(
                    f"{rel}:{line_no}: added a claim-shaped line "
                    f"(`{text.strip()[:80]}`) but no bundle exists under "
                    f"`bundles/`. Commit the first canonical bundle before "
                    f"publishing numbers in docs."
                )
            else:
                reasons.append(
                    f"{rel}:{line_no}: added a claim-shaped line "
                    f"(`{text.strip()[:80]}`) without a bundle reference. "
                    f"Cite a specific bundle under `bundles/` within 5 lines "
                    f"above or below, wrap the number in a fenced code block, "
                    f"or prefix with `synthetic:` / `example:` if it's not a "
                    f"real measurement."
                )
    return reasons


def main() -> int:
    p = argparse.ArgumentParser(description="Docs-claims gate")
    p.add_argument(
        "--base", default="origin/main",
        help="Git base to diff against (default: origin/main)",
    )
    p.add_argument(
        "--ci", action="store_true",
        help="CI mode: condensed output, exit 1 on violation",
    )
    args = p.parse_args()

    reasons = check(args.base)
    if not reasons:
        print(f"DOCS_CLAIMS_OK: no unsourced claims added vs {args.base}")
        return 0

    print(f"DOCS_CLAIMS_VIOLATIONS: {len(reasons)}", file=sys.stderr)
    for r in reasons:
        print(f"  - {r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
