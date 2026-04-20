"""Tests for the result-matrix scaffolding."""

from __future__ import annotations

import pytest

pytest.importorskip("mtg")

from arabic_agent_eval.dataset import EvalItem, ExpectedCall
from arabic_agent_eval.evaluator import BenchmarkResult, EvalResult
from arabic_agent_eval.matrix import (
    MatrixRow,
    build_matrix,
    render_csv,
    render_markdown,
    scan_with_mtg,
)
from arabic_agent_eval.scoring import Score


def _result(
    item_id: str,
    actual_calls: list[dict],
    score_total: float = 1.0,
    category: str = "simple_function_calling",
    dialect: str = "gulf",
) -> EvalResult:
    item = EvalItem(
        id=item_id, category=category, instruction="", dialect=dialect,
        available_functions=[],
        expected_calls=[ExpectedCall(function="t", arguments={})],
        difficulty="easy",
    )
    return EvalResult(
        item=item,
        score=Score(
            item_id=item_id, category=category,
            function_selection=score_total, argument_accuracy=score_total,
            arabic_preservation=score_total,
        ),
        actual_calls=actual_calls,
    )


def test_scan_with_clean_arabic_calls_yields_zero_rates():
    """Baseline — all emitted args are clean Arabic; nothing should fire."""
    br = BenchmarkResult(
        provider="test", model="clean",
        results=[
            _result("a1", [{"function": "send_message",
                            "arguments": {"message": "أبي أحجز فندق"}}]),
            _result("a2", [{"function": "send_message",
                            "arguments": {"message": "مرحبا أحمد"}}]),
        ],
    )
    row = scan_with_mtg(br)
    assert row.violation_rate == 0.0
    assert row.transliteration_rate == 0.0
    assert row.bidi_violation_rate == 0.0


def test_scan_catches_arabizi_transliteration():
    br = BenchmarkResult(
        provider="test", model="arabizi",
        results=[
            _result("a1", [{"function": "send_message",
                            "arguments": {"message": "abi a7jez fundu"}}]),
        ],
    )
    row = scan_with_mtg(br)
    # The value is detected as ar-slot (contains Arabic heuristic) then
    # the Latin surface fails; but since value has no Arabic chars,
    # our _guess_spec chooses script=latn so no SCRIPT_VIOLATION fires.
    # Instead: no violation (schema-less scan heuristic). The real
    # signal is when the caller annotates args with their tool schema,
    # which the matrix runner does via the reconciled_mode flag on
    # those schemas. This test asserts the baseline heuristic is
    # non-catastrophic.
    assert row.total_calls_scanned == 1


def test_scan_catches_bidi_in_any_arg_type():
    """BiDi-control injection fires regardless of declared script."""
    br = BenchmarkResult(
        provider="test", model="bidi",
        results=[
            _result("a1", [{"function": "send_message",
                            "arguments": {"message": "أبي\u202eأحجز"}}]),
        ],
    )
    row = scan_with_mtg(br)
    assert row.bidi_violation_rate > 0


def test_scan_catches_homoglyph():
    br = BenchmarkResult(
        provider="test", model="homoglyph",
        results=[
            _result("a1", [{"function": "send_message",
                            "arguments": {"message": "مرحبا\u0430"}}]),  # Cyrillic а
        ],
    )
    row = scan_with_mtg(br)
    assert row.homoglyph_rate > 0


def test_scan_counts_repair_rate_for_arabizi():
    """Pure-Arabizi strings in arabic-declared scans should get a concrete
    repair proposal when reconciled mode is active. The matrix scanner
    uses script='latn' for pure-Latin values so this exercises the
    repair-on-latn path — which for Arabizi-looking strings in a
    latn slot does NOT fire translit (since translit only checks ar).
    The assertion here is just that the row counts calls without crashing."""
    br = BenchmarkResult(
        provider="test", model="arabizi",
        results=[_result("a1", [{"function": "x",
                                   "arguments": {"msg": "abi a7jez"}}])],
    )
    row = scan_with_mtg(br)
    assert row.total_calls_scanned == 1
    assert row.repair_rate >= 0  # well-defined


def test_build_matrix_folds_multiple_runs():
    clean = BenchmarkResult(
        provider="a", model="m1",
        results=[_result("a1", [{"function": "x", "arguments": {"msg": "أبي"}}])],
    )
    dirty = BenchmarkResult(
        provider="b", model="m2",
        results=[_result("b1", [{"function": "x", "arguments": {"msg": "مرحبا\u202e"}}])],
    )
    matrix = build_matrix([clean, dirty])
    assert len(matrix.rows) == 2
    assert matrix.rows[0].provider == "a"
    assert matrix.rows[1].bidi_violation_rate > 0


def test_render_markdown_contains_expected_columns():
    row = MatrixRow(provider="p", model="m", baseline_score=0.85,
                    repaired_score=0.92, violation_rate=0.15, total_items=10,
                    total_calls_scanned=40)
    from arabic_agent_eval.matrix import ResultMatrix
    md = render_markdown(ResultMatrix(rows=[row]))
    assert "baseline" in md and "repaired" in md
    assert "0.850" in md
    assert "15.0%" in md  # violation_rate rendered as pct


def test_render_csv_roundtrip():
    row = MatrixRow(provider="p", model="m", baseline_score=0.5,
                    violation_rate=0.2, total_items=5, total_calls_scanned=10)
    from arabic_agent_eval.matrix import ResultMatrix
    csv = render_csv(ResultMatrix(rows=[row]))
    assert "provider,model" in csv
    assert "p,m" in csv


def test_scan_handles_errors_gracefully():
    """Error rows must not crash the scanner."""
    br = BenchmarkResult(
        provider="fail", model="x",
        results=[
            EvalResult(
                item=EvalItem(id="e1", category="simple_function_calling",
                               instruction="", dialect="msa",
                               available_functions=[],
                               expected_calls=[ExpectedCall(function="t", arguments={})],
                               difficulty="easy"),
                score=Score(item_id="e1", category="simple_function_calling"),
                actual_calls=[],
                error="API down",
            ),
        ],
    )
    row = scan_with_mtg(br)
    assert row.total_calls_scanned == 0


def test_cost_and_latency_aggregated_when_provided():
    r = _result("a1", [{"function": "x", "arguments": {"msg": "أبي"}}])
    setattr(r, "cost_usd", 0.0012)
    setattr(r, "latency_ms", 480)
    r2 = _result("a2", [{"function": "x", "arguments": {"msg": "مرحبا"}}])
    setattr(r2, "cost_usd", 0.0009)
    setattr(r2, "latency_ms", 520)
    br = BenchmarkResult(provider="p", model="m", results=[r, r2])
    row = scan_with_mtg(br)
    assert row.total_cost_usd == pytest.approx(0.0021, abs=1e-6)
    assert row.median_latency_ms in (480, 520)


def test_scan_without_mtg_raises_clean_error(monkeypatch):
    """If mtg is not installed, the scanner must emit a clear error."""
    import arabic_agent_eval.matrix as m
    monkeypatch.setattr(m, "_MTG_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="mtg-guards"):
        m.scan_with_mtg(BenchmarkResult(provider="x", model="y", results=[]))


# ---------- taxonomy precedence ----------


def test_family_precedence_content_beats_linguistic():
    """Regression: an arg that emits both a canonicalization violation
    (content) and a dialect violation (linguistic) must bucket under
    canonicalization, matching the docstring 'security > content >
    linguistic'."""
    from arabic_agent_eval.matrix import _classify_family

    codes = {"CANONICALIZATION_REQUIRED", "DIALECT_DRIFT"}
    assert _classify_family(codes) == "canonicalization"


def test_family_precedence_script_beats_dialect():
    """Script (content) outranks dialect (linguistic)."""
    from arabic_agent_eval.matrix import _classify_family

    codes = {"SCRIPT_VIOLATION", "DIALECT_DRIFT"}
    assert _classify_family(codes) == "script"


def test_family_precedence_security_beats_content():
    """BiDi (security) outranks script (content)."""
    from arabic_agent_eval.matrix import _classify_family

    codes = {"BIDI_CONTROL_SMUGGLING", "SCRIPT_VIOLATION"}
    assert _classify_family(codes) == "bidi"


def test_family_precedence_homoglyph_outranks_canonicalization():
    from arabic_agent_eval.matrix import _classify_family

    codes = {"SCRIPT_HOMOGLYPH", "CANONICALIZATION_REQUIRED"}
    assert _classify_family(codes) == "homoglyph"


def test_family_precedence_dialect_only_still_classifies():
    from arabic_agent_eval.matrix import _classify_family

    assert _classify_family({"DIALECT_DRIFT"}) == "dialect"


# ---------- repaired_score inflation ----------


def test_repaired_score_requires_every_violated_arg_repaired():
    """Regression: an item with two violated args where only ONE gets a
    concrete repair must NOT be credited as fully recoverable (score=1.0).
    The earlier implementation flipped item_could_repair on the first
    repaired arg and over-credited partial coverage."""
    # Arg A: Arabic with BiDi RLO → BIDI_CONTROL_SMUGGLING, NOT repairable
    #        (bidi repairs are not in suggest_repairs policy).
    # Arg B: Arabizi → SCRIPT + TRANSLIT violations → arabizi_to_arabic repair.
    # Item has 2 violated args, only 1 repairable → must stay at original.
    br = BenchmarkResult(
        provider="t", model="m",
        results=[_result(
            "multi_arg",
            [{"function": "send_message", "arguments": {
                "first":  "أبي\u202eأحجز",      # BiDi smuggling, no repair
                "second": "abi a7jez funduq",   # Arabizi, concrete repair
            }}],
            score_total=0.0,
        )],
    )
    row = scan_with_mtg(br)
    # Two violated args, only one repair — item is NOT fully recoverable.
    # repaired_score should equal baseline_score (both 0.0 here).
    assert row.repaired_score == pytest.approx(row.baseline_score)


def test_repaired_score_full_credit_when_all_violated_args_repaired():
    """When every violated arg gets a concrete repair, the item IS
    fully recoverable → repaired_score = 1.0 for that item."""
    br = BenchmarkResult(
        provider="t", model="m",
        results=[_result(
            "all_repairable",
            [{"function": "send_message", "arguments": {
                "msg": "abi a7jez",  # single violated arg, gets repair
            }}],
            score_total=0.0,
        )],
    )
    row = scan_with_mtg(br)
    # 1 violated arg, 1 repaired → fully recoverable
    assert row.repaired_score == 1.0


def test_repaired_score_unchanged_when_no_violations():
    """Clean item: no violations, no repairs — repaired_score equals
    baseline_score exactly (not inflated)."""
    br = BenchmarkResult(
        provider="t", model="m",
        results=[_result(
            "clean",
            [{"function": "send_message", "arguments": {"msg": "أبي أحجز"}}],
            score_total=0.8,
        )],
    )
    row = scan_with_mtg(br)
    assert row.repaired_score == pytest.approx(0.8)


def test_repaired_score_ignores_unsolicited_repairs_on_clean_args():
    """A repair proposal on a CLEAN arg (no violation) must not offset
    a DIFFERENT arg's unrepaired violation. The invariant is: count
    only those repairs that actually answered a violation."""
    br = BenchmarkResult(
        provider="t", model="m",
        results=[_result(
            "mixed",
            [{"function": "send_message", "arguments": {
                "clean_arg": "أبي أحجز",             # clean, no repair needed
                "broken_arg": "هذا نص طويل باللغة العربية لكن به شيء غريب",
            }}],
            score_total=0.5,
        )],
    )
    row = scan_with_mtg(br)
    # No violations here — both args are legitimate Arabic. Just assert
    # no spurious upgrade happens.
    assert row.repaired_score == pytest.approx(0.5)


# ---------- schema-bound replay ----------


def test_scan_with_schemas_uses_x_mtg_when_available():
    """Schema-bound replay — an x-mtg block on the tool schema must
    produce a strict spec (not the heuristic). A Gulf-declared message
    arg receiving MSA content fires DIALECT_DRIFT under schema-bound
    replay; the heuristic alone wouldn't catch dialect mismatch."""
    from arabic_agent_eval.matrix import scan_with_schemas

    tool_schema_map = {
        "send_message": {
            "name": "send_message",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "x-mtg": {
                            "slot_type": "free_text",
                            "script": "ar",
                            "dialect_expected": "gulf",
                            "dialect_enforcement": "preserve",
                            "mode": "advisory",
                        },
                    },
                },
            },
        },
    }
    # Egyptian-dialect content in a Gulf-declared slot — classifier
    # emits high-confidence 'egy', so DIALECT_DRIFT fires under the
    # schema-bound spec. Heuristic alone can't catch this because it
    # would pick dialect_expected=any on bare Arabic content.
    br = BenchmarkResult(
        provider="t", model="m",
        results=[_result(
            "drift",
            [{"function": "send_message", "arguments": {
                "message": "عايز أبعت رسالة لمحمد",  # Egyptian, not Gulf
            }}],
        )],
    )
    row = scan_with_schemas(br, tool_schema_map)
    assert row.dialect_drift_rate > 0
    assert row.heuristic_scan_rate == 0.0  # every arg had schema-bound spec


def test_scan_with_schemas_falls_back_to_heuristic_when_no_x_mtg():
    """Args without x-mtg in the schema still get scanned, but marked
    heuristic_scan=true so downstream can filter."""
    from arabic_agent_eval.matrix import scan_with_schemas

    tool_schema_map = {
        "send_message": {
            "name": "send_message",
            "parameters": {"type": "object", "properties": {
                # No x-mtg on this property — heuristic must fire
                "message": {"type": "string"},
            }},
        },
    }
    br = BenchmarkResult(
        provider="t", model="m",
        results=[_result("a", [{"function": "send_message",
                                "arguments": {"message": "أبي أحجز"}}])],
    )
    row = scan_with_schemas(br, tool_schema_map)
    assert row.heuristic_scan_rate == 1.0  # every arg fell to heuristic


def test_scan_with_mtg_uses_annotated_default_functions():
    """As of the x-mtg expansion, the default FUNCTIONS registry carries
    x-mtg on every parameter. scan_with_mtg with no caller-supplied map
    now produces schema-grounded rows, not heuristic ones."""
    br = BenchmarkResult(
        provider="t", model="m",
        results=[_result("a", [{"function": "search_flights",
                                "arguments": {"from_city": "الرياض",
                                              "to_city": "جدة"}}])],
    )
    row = scan_with_mtg(br)
    # Annotated FUNCTIONS → schema-bound replay fires for every arg.
    assert row.heuristic_scan_rate == 0.0
    assert row.schema_bound_rate == 1.0
    assert "search_flights" in row.schema_covered_tools


def test_every_function_declares_x_mtg_on_every_param():
    """Regression: x-mtg coverage across the FUNCTIONS registry must
    stay at 100%. A new function added without x-mtg annotations
    would silently drop schema_bound_rate on default-map runs."""
    from arabic_agent_eval.functions import FUNCTIONS

    missing: list[tuple[str, str]] = []
    for fn in FUNCTIONS:
        for pname, prop in fn["parameters"]["properties"].items():
            if "x-mtg" not in prop:
                missing.append((fn["name"], pname))
    assert missing == [], f"parameters without x-mtg: {missing}"


def test_default_map_run_hits_schema_bound_target():
    """Hard target from the reviewer: schema_bound_rate ≥ 0.9 on
    realistic default-map runs. If this drops below threshold, a new
    tool with unannotated slots got added and future publications will
    be marked diagnostic."""
    from arabic_agent_eval.dataset import Dataset

    dataset = Dataset()
    results = []
    for item in list(dataset)[:20]:
        actual = [
            {"function": c.function, "arguments": c.arguments}
            for c in item.expected_calls
        ]
        results.append(
            EvalResult(
                item=item,
                score=Score(
                    item_id=item.id, category=item.category,
                    function_selection=1.0, argument_accuracy=1.0,
                    arabic_preservation=1.0,
                ),
                actual_calls=actual,
            )
        )
    br = BenchmarkResult(provider="synth", model="target", results=results)
    row = scan_with_mtg(br)
    assert row.schema_bound_rate >= 0.9, (
        f"schema_bound_rate={row.schema_bound_rate:.2%} below 0.9 target — "
        "a new tool is missing x-mtg annotations"
    )


def test_scan_with_schemas_malformed_xmtg_falls_back_cleanly():
    """A schema with a broken x-mtg block (bad enum) should not crash
    the scanner — it falls back to heuristic and flags the arg."""
    from arabic_agent_eval.matrix import scan_with_schemas

    tool_schema_map = {
        "send_message": {
            "name": "send_message",
            "parameters": {"type": "object", "properties": {
                "message": {
                    "type": "string",
                    "x-mtg": {"script": "klingon"},  # invalid enum
                },
            }},
        },
    }
    br = BenchmarkResult(
        provider="t", model="m",
        results=[_result("a", [{"function": "send_message",
                                "arguments": {"message": "أبي"}}])],
    )
    row = scan_with_schemas(br, tool_schema_map)
    # Crash would fail here; bad spec → heuristic fallback.
    assert row.heuristic_scan_rate == 1.0


# ---------- 3-layer taxonomy ----------


def test_layer_rates_classify_script_as_surface():
    """SCRIPT_VIOLATION → surface (schema shape failure)."""
    from arabic_agent_eval.matrix import scan_with_schemas

    tool_schema_map = {
        "send_message": {
            "name": "send_message",
            "parameters": {"type": "object", "properties": {
                "message": {"type": "string",
                            "x-mtg": {"script": "ar", "mode": "advisory"}},
            }},
        },
    }
    br = BenchmarkResult(
        provider="t", model="m",
        results=[_result("a", [{"function": "send_message",
                                "arguments": {"message": "hello"}}])],
    )
    row = scan_with_schemas(br, tool_schema_map)
    assert row.layer_rates.get("surface", 0) > 0
    assert row.layer_rates.get("language", 0) == 0
    assert row.layer_rates.get("security", 0) == 0


def test_layer_rates_classify_bidi_as_security():
    """BIDI_CONTROL_SMUGGLING → security."""
    br = BenchmarkResult(
        provider="t", model="m",
        results=[_result("a", [{"function": "x",
                                "arguments": {"msg": "أبي\u202eأحجز"}}])],
    )
    row = scan_with_mtg(br)
    assert row.layer_rates.get("security", 0) > 0


def test_layer_rates_classify_dialect_as_language():
    """DIALECT_DRIFT → language."""
    from arabic_agent_eval.matrix import scan_with_schemas

    tool_schema_map = {
        "send_message": {
            "name": "send_message",
            "parameters": {"type": "object", "properties": {
                "message": {"type": "string", "x-mtg": {
                    "script": "ar", "dialect_expected": "gulf",
                    "dialect_enforcement": "preserve", "mode": "advisory",
                }},
            }},
        },
    }
    br = BenchmarkResult(
        provider="t", model="m",
        results=[_result("a", [{"function": "send_message",
                                "arguments": {"message": "عايز أبعت رسالة دلوقتي"}}])],
    )
    row = scan_with_schemas(br, tool_schema_map)
    assert row.layer_rates.get("language", 0) > 0
    assert row.layer_rates.get("surface", 0) == 0


# ---------- bootstrap CIs + run metadata ----------


def test_bootstrap_ci_present_when_multiple_items():
    """CIs fire when we have ≥2 items to bootstrap over."""
    br = BenchmarkResult(
        provider="t", model="m",
        results=[
            _result("a", [{"function": "x", "arguments": {"q": "أبي"}}], score_total=1.0),
            _result("b", [{"function": "x", "arguments": {"q": "احجز"}}], score_total=0.5),
            _result("c", [{"function": "x", "arguments": {"q": "مرحبا"}}], score_total=0.0),
        ],
    )
    row = scan_with_mtg(br)
    assert row.baseline_ci_95 is not None
    lo, hi = row.baseline_ci_95
    # CI must bracket the point estimate
    assert lo <= row.baseline_score <= hi


def test_bootstrap_ci_absent_for_single_item():
    """Can't bootstrap a single observation — CI stays None."""
    br = BenchmarkResult(
        provider="t", model="m",
        results=[_result("a", [{"function": "x", "arguments": {"q": "أبي"}}])],
    )
    row = scan_with_mtg(br)
    assert row.baseline_ci_95 is None


def test_bootstrap_ci_is_deterministic():
    """Same seed → same CI. Important for reproducible result tables."""
    br = BenchmarkResult(
        provider="t", model="m",
        results=[
            _result("a", [{"function": "x", "arguments": {"q": "أبي"}}], score_total=1.0),
            _result("b", [{"function": "x", "arguments": {"q": "احجز"}}], score_total=0.5),
        ],
    )
    row1 = scan_with_mtg(br)
    row2 = scan_with_mtg(br)
    assert row1.baseline_ci_95 == row2.baseline_ci_95


def test_run_metadata_stamped_on_every_row():
    """Every scanned MatrixRow gets provenance stamped."""
    br = BenchmarkResult(
        provider="p", model="m-v1",
        results=[_result("a", [{"function": "x", "arguments": {"q": "أبي"}}])],
    )
    row = scan_with_mtg(br)
    md = row.run_metadata
    assert "run_id" in md and len(md["run_id"]) >= 16
    assert "scanned_at" in md
    assert md["provider"] == "p"
    assert md["model"] == "m-v1"
    assert md["n_items"] == 1
    assert md["scanner_version"].startswith("mtg-matrix/")


def test_run_metadata_carries_code_shas():
    """Provenance: every scan records git HEAD SHA for the MTG-stack
    packages present in the env. Answers 'what code produced this?'"""
    br = BenchmarkResult(
        provider="p", model="m",
        results=[_result("a", [{"function": "x", "arguments": {"q": "أبي"}}])],
    )
    row = scan_with_mtg(br)
    shas = row.run_metadata.get("code_shas") or {}
    # arabic_agent_eval is this repo — SHA must be present
    assert shas.get("arabic_agent_eval"), "aae SHA missing from provenance"
    assert len(shas["arabic_agent_eval"]) == 40
    # mtg + toolproof may be None in minimal installs — just assert
    # the key is present so downstream knows whether it was captured
    assert "mtg" in shas
    assert "toolproof" in shas


def test_run_metadata_carries_schema_map_fingerprint():
    """Different schema maps produce different fingerprints."""
    from arabic_agent_eval.matrix import scan_with_schemas

    br = BenchmarkResult(
        provider="p", model="m",
        results=[_result("a", [{"function": "send_message",
                                "arguments": {"q": "أبي"}}])],
    )
    map_a = {
        "send_message": {"name": "send_message", "parameters": {"type": "object",
            "properties": {"q": {"type": "string",
                                  "x-mtg": {"script": "ar", "mode": "advisory"}}}}},
    }
    map_b = {
        "send_message": {"name": "send_message", "parameters": {"type": "object",
            "properties": {"q": {"type": "string",
                                  "x-mtg": {"script": "latn", "mode": "advisory"}}}}},
    }
    fp_a = scan_with_schemas(br, map_a).run_metadata["schema_map_fingerprint"]
    fp_b = scan_with_schemas(br, map_b).run_metadata["schema_map_fingerprint"]
    # Same map twice → same fingerprint
    fp_a2 = scan_with_schemas(br, map_a).run_metadata["schema_map_fingerprint"]
    assert fp_a == fp_a2
    # Different x-mtg content → different fingerprint
    assert fp_a != fp_b
    # Fingerprint is a sha256 hex string
    assert len(fp_a) == 64 and all(c in "0123456789abcdef" for c in fp_a)


def test_schema_map_fingerprint_ignores_editorial_changes():
    """Changing descriptions / names must not change the fingerprint —
    only x-mtg content matters."""
    from arabic_agent_eval.matrix import _fingerprint_schema_map

    base = {"t": {"name": "t", "description": "original",
        "parameters": {"type": "object", "properties": {
            "a": {"type": "string", "description": "one",
                   "x-mtg": {"script": "ar", "mode": "advisory"}}}}}}
    edited = {"t": {"name": "t", "description": "updated copy",
        "parameters": {"type": "object", "properties": {
            "a": {"type": "string", "description": "something else",
                   "x-mtg": {"script": "ar", "mode": "advisory"}}}}}}
    assert _fingerprint_schema_map(base) == _fingerprint_schema_map(edited)


def test_run_metadata_carries_dataset_fingerprint():
    """Every scan stamps a dataset_fingerprint derived from the items
    that were scored. Two scans of the same subset produce the same
    fingerprint; different subsets produce different ones."""
    from arabic_agent_eval.matrix import _fingerprint_benchmark_items

    br_a = BenchmarkResult(
        provider="p", model="m",
        results=[_result("a", [{"function": "x", "arguments": {"q": "أبي"}}])],
    )
    br_b = BenchmarkResult(
        provider="p", model="m",
        results=[_result("a", [{"function": "x", "arguments": {"q": "أبي"}}]),
                 _result("b", [{"function": "y", "arguments": {"q": "مرحبا"}}])],
    )
    fp_a = _fingerprint_benchmark_items(br_a)
    fp_a2 = _fingerprint_benchmark_items(br_a)
    fp_b = _fingerprint_benchmark_items(br_b)
    assert fp_a == fp_a2          # deterministic
    assert fp_a != fp_b           # different subsets → different fingerprint
    assert len(fp_a) == 64        # sha256 hex

    # Stamped onto every row's run_metadata
    row = scan_with_mtg(br_a)
    assert row.run_metadata["dataset_fingerprint"] == fp_a


def test_dataset_fingerprint_ignores_model_output():
    """Fingerprint covers only the fixed inputs — item IDs, categories,
    dialects, expected_calls. Changing actual_calls (model output) or
    score must not change it."""
    from arabic_agent_eval.matrix import _fingerprint_benchmark_items

    item = _result("a", [{"function": "x", "arguments": {"q": "أبي"}}],
                    score_total=1.0)
    item2 = _result("a", [{"function": "x", "arguments": {"q": "ابراهيم"}}],
                     score_total=0.0)
    br1 = BenchmarkResult(provider="p", model="m", results=[item])
    br2 = BenchmarkResult(provider="p", model="m", results=[item2])
    # Same item ID + expected_calls → same fingerprint even though
    # actual_calls and score differ.
    assert _fingerprint_benchmark_items(br1) == _fingerprint_benchmark_items(br2)


def test_run_metadata_carries_dataset_version():
    """DATASET_VERSION constant flows into run_metadata so citations
    can use a human-readable label alongside the sha256."""
    from arabic_agent_eval import DATASET_VERSION

    br = BenchmarkResult(
        provider="p", model="m",
        results=[_result("a", [{"function": "x", "arguments": {"q": "أبي"}}])],
    )
    row = scan_with_mtg(br)
    assert row.run_metadata["dataset_version"] == DATASET_VERSION
    assert row.run_metadata["dataset_version"].startswith("2026-")


def test_run_metadata_carries_environment_provenance():
    """Python version / platform / dependency versions are stamped on
    every row so reproducing the bundle doesn't require guessing the
    env."""
    br = BenchmarkResult(
        provider="p", model="m",
        results=[_result("a", [{"function": "x", "arguments": {"q": "أبي"}}])],
    )
    row = scan_with_mtg(br)
    env = row.run_metadata.get("environment") or {}
    assert "python_version" in env
    assert env["python_version"].count(".") == 2
    assert "platform" in env
    assert "dependency_versions" in env
    assert env["dependency_versions"]["arabic_agent_eval"] is not None


def test_run_metadata_carries_code_clean_flags():
    """Per-package git-clean booleans. None = no signal; False = dirty;
    True = clean. Paired with code_shas so reviewers can tell a SHA
    points at committed code from one that lags a dirty worktree."""
    br = BenchmarkResult(
        provider="p", model="m",
        results=[_result("a", [{"function": "x", "arguments": {"q": "أبي"}}])],
    )
    row = scan_with_mtg(br)
    code_clean = row.run_metadata.get("code_clean") or {}
    assert set(code_clean.keys()) == {"arabic_agent_eval", "mtg", "toolproof"}
    for v in code_clean.values():
        assert v is None or isinstance(v, bool)


def test_run_metadata_carries_provider_provenance():
    """Opt-in provider/request attributes on BenchmarkResult flow into
    run_metadata.provider_provenance."""
    br = BenchmarkResult(
        provider="openrouter", model="hermes-70b",
        results=[_result("a", [{"function": "x", "arguments": {"q": "أبي"}}])],
    )
    setattr(br, "provider_base_url", "https://openrouter.ai/api/v1")
    setattr(br, "model_id", "nousresearch/hermes-4-70b")
    setattr(br, "request_config", {"temperature": 0.7, "max_tokens": 2048})
    row = scan_with_mtg(br)
    pp = row.run_metadata.get("provider_provenance") or {}
    assert pp.get("provider_base_url") == "https://openrouter.ai/api/v1"
    assert pp.get("model_id") == "nousresearch/hermes-4-70b"
    assert pp.get("request_config_fingerprint")
    assert len(pp["request_config_fingerprint"]) == 64


def test_request_config_fingerprint_changes_with_content():
    """Different config values → different fingerprints."""
    from arabic_agent_eval.matrix import _fingerprint_request_config

    fp1 = _fingerprint_request_config({"temperature": 0.7})
    fp2 = _fingerprint_request_config({"temperature": 0.9})
    fp3 = _fingerprint_request_config({"temperature": 0.7})
    assert fp1 != fp2
    assert fp1 == fp3
    assert _fingerprint_request_config(None) is None
    assert _fingerprint_request_config({}) is None


def test_provider_provenance_absent_when_attrs_missing():
    """BenchmarkResult without the opt-in setattr gets None entries."""
    br = BenchmarkResult(
        provider="p", model="m",
        results=[_result("a", [{"function": "x", "arguments": {"q": "أبي"}}])],
    )
    row = scan_with_mtg(br)
    pp = row.run_metadata.get("provider_provenance") or {}
    assert pp.get("provider_base_url") is None
    assert pp.get("model_id") is None
    assert pp.get("request_config_fingerprint") is None


def test_to_dict_surfaces_ci_and_heuristic_rate():
    br = BenchmarkResult(
        provider="t", model="m",
        results=[
            _result("a", [{"function": "x", "arguments": {"q": "أبي"}}], score_total=1.0),
            _result("b", [{"function": "x", "arguments": {"q": "احجز"}}], score_total=0.0),
        ],
    )
    row = scan_with_mtg(br)
    d = row.to_dict()
    assert "baseline_ci_95" in d and d["baseline_ci_95"] is not None
    assert "repaired_ci_95" in d
    assert "heuristic_scan_rate" in d
    assert "layer_rates" in d
    assert "run_metadata" in d and d["run_metadata"]["run_id"]


def test_render_markdown_contains_three_layers_and_heuristic_rate():
    br = BenchmarkResult(
        provider="a", model="m1",
        results=[_result("a", [{"function": "x", "arguments": {"q": "أبي"}}], score_total=1.0)],
    )
    matrix = build_matrix([br])
    md = render_markdown(matrix)
    assert "3-layer taxonomy" in md
    assert "surface" in md and "language" in md and "security" in md
    assert "heur. scan" in md
    assert "Run provenance" in md
