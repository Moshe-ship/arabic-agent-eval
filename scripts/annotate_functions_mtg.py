#!/usr/bin/env python3
"""Inject x-mtg annotations into data/functions.json.

Each of the 22 benchmark functions gets a per-arg MTG spec so the
matrix scanner can schema-bind replay instead of falling to the
heuristic fallback (which marks the row diagnostic).

Annotation is rule-based on arg name. Conservative defaults:
- Arabic/free-text slots → mixed script, transliteration allowed
- Identifiers / codes / currency / language / platform → identifier,
  script=any
- Dates/times → temporal, script=any
- Numerics → numeric, script=any

Per-arg overrides sit in ARG_OVERRIDES; everything else goes through
the name-pattern classifier. Re-run whenever arg names shift.
"""
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
FUNCTIONS_PATH = REPO_ROOT / "data" / "functions.json"


NAMED_ENTITY_ARGS = {
    "city", "from_city", "to_city", "country", "destination",
    "from_location", "to_location", "location", "pickup",
    "address", "restaurant", "recipient", "cuisine", "market",
    "participants",
}
FREE_TEXT_ARGS = {"message", "text", "query", "title", "items"}
TEMPORAL_ARGS = {"date", "check_in", "check_out", "datetime", "time"}
NUMERIC_ARGS = {"amount", "guests", "passengers", "surah", "budget"}
IDENTIFIER_ARGS = {
    "symbol", "currency", "from_currency", "to_currency",
    "from_lang", "to_lang", "platform", "category", "type",
    "application_id", "passport_number",
}


def spec_for_arg(arg_name: str, arg_type: str) -> dict:
    """Return an x-mtg spec dict for the given arg. Kept deliberately
    conservative so schema-bound replay fires but doesn't over-claim
    constraints we can't enforce on the benchmark dataset."""
    base_mixed = {
        "script": "mixed",
        "transliteration_allowed": True,
        "mode": "advisory",
    }
    if arg_name in NAMED_ENTITY_ARGS:
        return {"slot_type": "named_entity", **base_mixed}
    if arg_name in FREE_TEXT_ARGS:
        return {"slot_type": "free_text", **base_mixed}
    if arg_name in TEMPORAL_ARGS:
        return {"slot_type": "temporal", "script": "any", "mode": "advisory"}
    if arg_name in NUMERIC_ARGS:
        return {"slot_type": "numeric", "script": "any", "mode": "advisory"}
    if arg_name in IDENTIFIER_ARGS:
        return {"slot_type": "identifier", "script": "any", "mode": "advisory"}
    if arg_type == "integer" or arg_type == "number":
        return {"slot_type": "numeric", "script": "any", "mode": "advisory"}
    # Fallback — treat as free_text mixed. Script "mixed" is permissive;
    # transliteration allowed because user input often has Romanized
    # fragments.
    return {"slot_type": "free_text", **base_mixed}


def annotate() -> None:
    data = json.loads(FUNCTIONS_PATH.read_text(encoding="utf-8"))
    functions = data.get("functions") or []
    annotated = 0
    for fn in functions:
        props = fn.get("parameters", {}).get("properties", {}) or {}
        for arg_name, arg_spec in props.items():
            if not isinstance(arg_spec, dict):
                continue
            if "x-mtg" in arg_spec:
                continue
            arg_spec["x-mtg"] = spec_for_arg(arg_name, arg_spec.get("type", ""))
            annotated += 1
    FUNCTIONS_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"annotated {annotated} arg(s) across {len(functions)} function(s)")


if __name__ == "__main__":
    annotate()
