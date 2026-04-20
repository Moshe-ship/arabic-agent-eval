#!/usr/bin/env python3
"""Report the trend across N bundles.

Usage:

    python scripts/trend_bundles.py bundles/apr17 bundles/apr24 bundles/may01
    python scripts/trend_bundles.py bundles/*/ --markdown trend.md
    python scripts/trend_bundles.py bundles/*/ --json trend.json

Bundles are consumed in the order given on the command line. Each
contributes one point per (provider, model) row to a TrendSeries.
The Markdown renderer shows baseline/repaired deltas, schema coverage,
and layer failure rates side-by-side across bundles.

Dataset-version and scanner-version changes within a series are
flagged explicitly — numbers across different scorers don't compare
cleanly.
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
from arabic_agent_eval.trend import build_trend, render_markdown  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Build a bundle-to-bundle trend report")
    p.add_argument("bundles", nargs="+", type=Path,
                    help="Bundle directories in the order to trend")
    p.add_argument("--markdown", metavar="PATH",
                    help="Write Markdown trend to PATH")
    p.add_argument("--json", metavar="PATH",
                    help="Write structured trend JSON to PATH")
    args = p.parse_args()

    bundles: list[tuple[str, str, dict]] = []
    for bundle_dir in args.bundles:
        manifest, matrix = load_bundle(bundle_dir)
        bundles.append((bundle_dir.name, manifest.created_at, matrix))

    trend = build_trend(bundles)
    md = render_markdown(trend)

    if args.markdown:
        Path(args.markdown).write_text(md, encoding="utf-8")
        print(f"wrote markdown → {args.markdown}")
    if args.json:
        Path(args.json).write_text(
            json.dumps(trend.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"wrote json → {args.json}")
    if not args.markdown and not args.json:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
