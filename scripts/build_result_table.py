#!/usr/bin/env python3
"""Build the hard result table from N benchmark-run JSONs.

Usage:

    # After running `aae run --provider ... --out hermes-70b.json`,
    # pass each result file to this script:
    python scripts/build_result_table.py \\
        results/hermes-70b.json \\
        results/claude-sonnet.json \\
        results/gpt-5.json \\
        --markdown out/result_table.md \\
        --csv out/result_table.csv

The script does NOT call any LLM. Run your model matrix the usual way
(with your own API keys, at your own cadence); then fold those results
into one table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from arabic_agent_eval.matrix import (
    build_matrix,
    load_benchmark_result_from_json,
    render_csv,
    render_markdown,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Build the hard result table from benchmark runs")
    p.add_argument("runs", nargs="+", help="Paths to benchmark-run JSON files")
    p.add_argument("--markdown", help="Write Markdown table to this path")
    p.add_argument("--csv", help="Write CSV to this path")
    args = p.parse_args()

    benchmarks = [load_benchmark_result_from_json(Path(r)) for r in args.runs]
    matrix = build_matrix(benchmarks)

    md = render_markdown(matrix)
    csv = render_csv(matrix)

    if args.markdown:
        Path(args.markdown).write_text(md, encoding="utf-8")
        print(f"wrote markdown → {args.markdown}")
    if args.csv:
        Path(args.csv).write_text(csv, encoding="utf-8")
        print(f"wrote csv → {args.csv}")

    if not args.markdown and not args.csv:
        print(md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
