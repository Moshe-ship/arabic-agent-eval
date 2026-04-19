#!/usr/bin/env python3
"""Diff two result bundles.

Usage:

    python scripts/diff_bundles.py before/ after/
    python scripts/diff_bundles.py before/ after/ --json out.json
    python scripts/diff_bundles.py before/ after/ --markdown pr-comment.md

Integrity-checks both bundles (rejects tampered manifests), then
produces a row-by-row delta of:

- baseline / repaired score deltas
- CI overlap (weak evidence vs strong)
- heuristic_scan_rate / schema_bound_rate deltas
- failure-family and 3-layer deltas
- schema-coverage deltas (tools added / removed)
- diagnostic flag transitions

Designed for PR comments and review threads. One command, one
artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arabic_agent_eval.bundle import load_bundle  # noqa: E402
from arabic_agent_eval.diff import diff_bundles, render_markdown  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Diff two result bundles")
    p.add_argument("before", type=Path, help="Path to the BEFORE bundle directory")
    p.add_argument("after", type=Path, help="Path to the AFTER bundle directory")
    p.add_argument(
        "--json", metavar="PATH",
        help="Write structured diff JSON to PATH",
    )
    p.add_argument(
        "--markdown", metavar="PATH",
        help="Write Markdown diff to PATH (suitable for PR comments)",
    )
    args = p.parse_args()

    _, before_matrix = load_bundle(args.before)
    _, after_matrix = load_bundle(args.after)

    d = diff_bundles(
        before_matrix, after_matrix,
        before_label=args.before.name or str(args.before),
        after_label=args.after.name or str(args.after),
    )

    md = render_markdown(d)
    if args.markdown:
        Path(args.markdown).write_text(md, encoding="utf-8")
        print(f"wrote markdown → {args.markdown}")
    if args.json:
        Path(args.json).write_text(
            json.dumps(d.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"wrote json → {args.json}")
    if not args.markdown and not args.json:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
