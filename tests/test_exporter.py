"""Round-trip tests for JSONL export."""

import json
from pathlib import Path

from arabic_agent_eval.dataset import ALL_ITEMS, DIALECTS
from arabic_agent_eval.exporter import export


def test_export_writes_all_dialect_files(tmp_path: Path) -> None:
    counts = export(tmp_path)
    for dialect in DIALECTS:
        assert (tmp_path / f"{dialect}.jsonl").exists()
    assert (tmp_path / "all.jsonl").exists()
    assert (tmp_path / "functions.json").exists()
    assert counts["all"] == len(ALL_ITEMS)


def test_export_round_trip_preserves_arabic(tmp_path: Path) -> None:
    export(tmp_path)
    loaded = []
    with (tmp_path / "all.jsonl").open(encoding="utf-8") as f:
        for line in f:
            loaded.append(json.loads(line))

    assert len(loaded) == len(ALL_ITEMS)
    for original, roundtripped in zip(ALL_ITEMS, loaded):
        assert original["id"] == roundtripped["id"]
        assert original["instruction"] == roundtripped["instruction"]
        assert original["dialect"] == roundtripped["dialect"]


def test_export_dialect_split_totals_match_all(tmp_path: Path) -> None:
    counts = export(tmp_path)
    dialect_total = sum(counts[d] for d in DIALECTS)
    assert dialect_total == counts["all"], (
        f"dialect splits sum to {dialect_total} but all.jsonl has {counts['all']}"
    )


def test_functions_json_has_all_functions(tmp_path: Path) -> None:
    export(tmp_path)
    with (tmp_path / "functions.json").open(encoding="utf-8") as f:
        data = json.load(f)
    assert data["count"] == len(data["functions"])
    assert data["count"] >= 20
