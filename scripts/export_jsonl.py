"""CLI wrapper around arabic_agent_eval.exporter.export.

Usage:
    python scripts/export_jsonl.py [out_dir]

Default out_dir is ./data. Re-runs are idempotent — files are overwritten.
"""

from __future__ import annotations

import sys
from pathlib import Path

from arabic_agent_eval.exporter import export


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    counts = export(out)
    for key, count in counts.items():
        print(f"  {key}: {count}")
    print(f"Exported to {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
