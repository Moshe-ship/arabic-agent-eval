"""Export the Python-literal dataset to JSONL splits per dialect.

The Python dataset in `dataset.py` is the source of truth. JSONL files
under `data/` are derived artifacts for HF distribution and external tooling.
"""

from __future__ import annotations

import json
from pathlib import Path

from arabic_agent_eval.dataset import ALL_ITEMS, DIALECTS, CATEGORIES
from arabic_agent_eval.functions import FUNCTIONS


def export(out_dir: Path) -> dict[str, int]:
    """Write JSONL splits to out_dir. Returns {dialect_or_name: row_count}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for dialect in DIALECTS:
        rows = [item for item in ALL_ITEMS if item["dialect"] == dialect]
        path = out_dir / f"{dialect}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[dialect] = len(rows)

    with (out_dir / "all.jsonl").open("w", encoding="utf-8") as f:
        for row in ALL_ITEMS:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts["all"] = len(ALL_ITEMS)

    with (out_dir / "functions.json").open("w", encoding="utf-8") as f:
        json.dump({"functions": FUNCTIONS, "count": len(FUNCTIONS)}, f, ensure_ascii=False, indent=2)
    counts["functions"] = len(FUNCTIONS)

    with (out_dir / "categories.json").open("w", encoding="utf-8") as f:
        json.dump(CATEGORIES, f, ensure_ascii=False, indent=2)

    return counts
