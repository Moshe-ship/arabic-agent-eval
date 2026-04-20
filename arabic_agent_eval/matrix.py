"""Result-matrix scaffolding — MTG-assisted re-validation over a benchmark run.

Take a completed `BenchmarkResult` (produced by Evaluator against any
provider), replay every emitted tool call through MTG's validation
pipeline, and aggregate the hard-result-table numbers:

- baseline function-calling score (+ bootstrap 95% CI over items)
- MTG violation rate (per item, per call, per code)
- transliteration / dialect-drift / BiDi / homoglyph rates
- 3-layer taxonomy: surface correctness, language integrity, security
  integrity (reviewer-requested framing, April 2026)
- repaired score (+ bootstrap CI) when reconciled mode can propose a
  concrete replacement
- cost / latency deltas — opt-in if caller populates EvalResult
- heuristic_scan_rate — fraction of args scored without a schema. The
  primary path is schema-bound replay (`scan_with_schemas`); the
  heuristic exists only as last-resort fallback when a tool schema has
  no x-mtg annotations.

Deliberately decoupled from model providers — this module does NOT make
API calls. Run your provider matrix the usual way (via Evaluator), then
pass the resulting BenchmarkResult into `build_matrix`. That keeps API
keys, rate limits, and cost out of MTG.

Output is Markdown + CSV. The Markdown is designed to drop straight into
a blog post or paper.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import random
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from arabic_agent_eval.evaluator import BenchmarkResult, EvalResult


try:
    from mtg.adapters.openai import guard_tool
    from mtg.pipeline import validate_pre
    from mtg.types import GuardSpec
    _MTG_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MTG_AVAILABLE = False


# Violation code → failure family. Each arg that emits a violation
# increments exactly one family count (worst-category wins). Families
# are ordered by severity for the tiebreak.
_CODE_TO_FAMILY: dict[str, str] = {
    # security (highest priority)
    "BIDI_CONTROL_SMUGGLING": "bidi",
    "INVISIBLE_CONTENT": "bidi",
    "SCRIPT_HOMOGLYPH": "homoglyph",
    # content correctness
    "CANONICALIZATION_REQUIRED": "canonicalization",
    "SURFACE_CORRUPTION_POST_CALL": "canonicalization",
    "SCRIPT_VIOLATION": "script",
    "TRANSLITERATION_VIOLATION": "script",
    # linguistic
    "DIALECT_DRIFT": "dialect",
    "DIALECT_FLATTEN": "dialect",
    "FREE_TEXT_OVERFLOW": "overflow",
    # morph — not promoted to a top-level family (they're advisory)
    "MORPH_CANONICALIZATION_FAILURE": "canonicalization",
    "MORPH_AMBIGUITY": "canonicalization",
    "ROOT_DRIFT": "canonicalization",
    "BACKEND_DISAGREEMENT": "canonicalization",
}

# Precedence: security (bidi, homoglyph) > content (script,
# canonicalization) > linguistic (dialect, overflow). `_classify_family`
# walks this list and returns the first match, so earlier entries are
# worse. Matches the docstring on `_classify_family` and the grouping
# comments in `_CODE_TO_FAMILY`.
_FAMILY_ORDER = (
    # security
    "bidi",
    "homoglyph",
    # content correctness
    "script",
    "canonicalization",
    # linguistic
    "dialect",
    "overflow",
)


# 3-layer taxonomy (reviewer framing, April 2026). Each family maps to
# exactly one layer. Surface correctness = schema-shape failures (tool
# choice + arg script/type validity). Language integrity = register,
# morphology, canonicalization, register-overflow. Security integrity =
# Unicode-layer attacks (BiDi, homoglyph, invisibles).
_FAMILY_TO_LAYER: dict[str, str] = {
    "script": "surface",
    "canonicalization": "language",
    "dialect": "language",
    "overflow": "language",
    "bidi": "security",
    "homoglyph": "security",
}
_LAYERS = ("surface", "language", "security")


def _classify_family(codes: set[str]) -> Optional[str]:
    """Pick one failure family for an arg given the violation codes
    it emitted. Worst family wins (security > content > linguistic)."""
    for fam in _FAMILY_ORDER:
        if any(_CODE_TO_FAMILY.get(c) == fam for c in codes):
            return fam
    return None


def _classify_layer(family: Optional[str]) -> Optional[str]:
    """Map a failure family into its 3-layer bucket."""
    return _FAMILY_TO_LAYER.get(family) if family else None


@dataclass
class MatrixRow:
    """One row of the result table — typically a (provider, model) pair."""

    provider: str
    model: str
    baseline_score: float = 0.0
    # Within-run 95% CI via bootstrap over items. Captures variance
    # from the specific subset of items scored, NOT model run-to-run
    # variance — that requires repeated runs (aggregate externally).
    baseline_ci_95: Optional[tuple[float, float]] = None
    # MTG re-validation aggregates
    total_calls_scanned: int = 0
    violation_rate: float = 0.0
    transliteration_rate: float = 0.0
    dialect_drift_rate: float = 0.0
    bidi_violation_rate: float = 0.0
    homoglyph_rate: float = 0.0
    # Why-failed taxonomy — every failing arg is assigned one family
    # (worst wins). Sum of family rates ≤ violation_rate.
    family_rates: dict[str, float] = field(default_factory=dict)
    # 3-layer taxonomy — surface (schema-shape failures), language
    # (register / morphology / canonicalization), security (Unicode-
    # layer attacks). Each layer_rate is the fraction of scanned args
    # whose worst violation falls in that layer. Reviewer-requested
    # framing that makes the stack easier to explain.
    layer_rates: dict[str, float] = field(default_factory=dict)
    # Fraction of scanned args that had NO x-mtg schema annotation and
    # fell back to the heuristic spec. Schema-bound replay is the
    # primary path; this tells the reader how much they're trusting
    # the heuristic. Target: <0.1 for any published claim.
    heuristic_scan_rate: float = 0.0
    # Positive framing of the same underlying quantity: fraction of
    # scanned args that had a schema-declared x-mtg block.
    # `schema_bound_rate + heuristic_scan_rate == 1.0` by construction.
    schema_bound_rate: float = 0.0
    # Sorted list of tool names where at least one argument had an
    # x-mtg block used during the scan. Empty when no tools were
    # schema-grounded. Absolute coverage signal for the reader.
    schema_covered_tools: list[str] = field(default_factory=list)
    # Absolute counts behind the rates. Useful for aggregating across
    # runs or reporting N-like numbers in a paper.
    schema_bound_arg_count: int = 0
    heuristic_arg_count: int = 0
    # Repair path
    calls_with_concrete_repair: int = 0
    repair_rate: float = 0.0
    repaired_score: float = 0.0  # baseline_score if repaired args were used
    repaired_ci_95: Optional[tuple[float, float]] = None
    # Repair quality — mean score across every concrete repair the
    # scanner saw. 1.0 = clean repair, 0.5 = needs review, 0.0 = broken.
    repair_quality_mean: Optional[float] = None
    repair_quality_n: int = 0
    # Cost / latency
    total_cost_usd: Optional[float] = None
    median_latency_ms: Optional[float] = None
    total_items: int = 0
    # Reproducibility metadata — stamped at scan time so downstream
    # can deduplicate runs and answer "which model version, when".
    run_metadata: dict[str, Any] = field(default_factory=dict)
    # Threshold used to decide whether this row is diagnostic-only.
    # Set by the scanner so downstream (bundles, publish gates) uses
    # the same value that was recorded when the numbers were produced.
    heuristic_scan_threshold: float = 0.10

    @property
    def diagnostic(self) -> bool:
        """True when this row should not be published as a clean result.

        Fires when any of the publish-gate prerequisites fail:

        - `heuristic_scan_rate` exceeds `heuristic_scan_threshold`
          (schema-bound replay didn't cover enough of the args)
        - `run_metadata` is missing or empty (no provenance)

        Callers that deliberately publish a diagnostic run should
        render the marker, NOT suppress it. The point is making the
        caveat visible, not hiding it.
        """
        if not self.run_metadata:
            return True
        if self.heuristic_scan_rate > self.heuristic_scan_threshold:
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "diagnostic": self.diagnostic,
            "heuristic_scan_threshold": self.heuristic_scan_threshold,
            "baseline_score": round(self.baseline_score, 4),
            "baseline_ci_95": (
                [round(self.baseline_ci_95[0], 4), round(self.baseline_ci_95[1], 4)]
                if self.baseline_ci_95 is not None else None
            ),
            "repaired_score": round(self.repaired_score, 4),
            "repaired_ci_95": (
                [round(self.repaired_ci_95[0], 4), round(self.repaired_ci_95[1], 4)]
                if self.repaired_ci_95 is not None else None
            ),
            "violation_rate": round(self.violation_rate, 4),
            "transliteration_rate": round(self.transliteration_rate, 4),
            "dialect_drift_rate": round(self.dialect_drift_rate, 4),
            "bidi_violation_rate": round(self.bidi_violation_rate, 4),
            "homoglyph_rate": round(self.homoglyph_rate, 4),
            "family_rates": {k: round(v, 4) for k, v in self.family_rates.items()},
            "layer_rates": {k: round(v, 4) for k, v in self.layer_rates.items()},
            "heuristic_scan_rate": round(self.heuristic_scan_rate, 4),
            "schema_bound_rate": round(self.schema_bound_rate, 4),
            "schema_covered_tools": list(self.schema_covered_tools),
            "schema_bound_arg_count": self.schema_bound_arg_count,
            "heuristic_arg_count": self.heuristic_arg_count,
            "repair_rate": round(self.repair_rate, 4),
            "repair_quality_mean": (
                round(self.repair_quality_mean, 4)
                if self.repair_quality_mean is not None else None
            ),
            "repair_quality_n": self.repair_quality_n,
            "total_items": self.total_items,
            "total_calls_scanned": self.total_calls_scanned,
            "calls_with_concrete_repair": self.calls_with_concrete_repair,
            "total_cost_usd": (
                round(self.total_cost_usd, 4) if self.total_cost_usd is not None else None
            ),
            "median_latency_ms": self.median_latency_ms,
            "run_metadata": dict(self.run_metadata),
        }


@dataclass
class ResultMatrix:
    """Table of MatrixRow, one per provider-model pair."""

    rows: list[MatrixRow] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"rows": [r.to_dict() for r in self.rows]}


def _build_default_schema_map() -> dict[str, dict]:
    """Build {function_name: tool_schema} from arabic_agent_eval.functions.

    Safe default for schema-bound replay when the caller doesn't pass
    one. The FUNCTIONS registry does NOT ship x-mtg annotations by
    default, so this map gives us the function names we can recognize
    but schema-bound lookup will still fall back to heuristic for every
    arg — that's the honest state and the heuristic_scan_rate surfaces
    it in output.

    Callers who want real schema-bound behavior should pass their own
    annotated tool-schema map (e.g. from Hurmoz tool-schemas/*.json).
    """
    from arabic_agent_eval.functions import FUNCTIONS
    return {fn["name"]: fn for fn in FUNCTIONS}


def _extract_x_mtg_for_arg(
    tool_schema: Optional[dict],
    arg_name: str,
) -> Optional[dict]:
    """Pull the x-mtg block for one argument from a tool schema, if any.
    Supports both OpenAI-shape (`parameters.properties`) and
    Anthropic-shape (`input_schema.properties`)."""
    if not tool_schema:
        return None
    for key in ("parameters", "input_schema"):
        container = tool_schema.get(key)
        if isinstance(container, dict):
            props = container.get("properties") or {}
            prop = props.get(arg_name)
            if isinstance(prop, dict) and "x-mtg" in prop:
                return prop["x-mtg"]
    return None


def _spec_for_arg(
    tool_schema: Optional[dict],
    arg_name: str,
    arg_value: str,
) -> tuple[GuardSpec, bool]:
    """Primary spec resolution: schema-bound replay when the tool schema
    declares x-mtg on this arg; heuristic fallback otherwise.

    Returns `(spec, heuristic_scan)` — downstream uses the flag to
    compute heuristic_scan_rate so readers can tell how much of a
    result is schema-grounded vs best-effort.
    """
    x_mtg = _extract_x_mtg_for_arg(tool_schema, arg_name)
    if x_mtg:
        # Force reconciled mode so repairs surface even if the schema
        # ships a different mode. The evaluator calls this for the
        # diagnostic pass; production consumers keep their schema's mode.
        x_mtg = {**x_mtg, "mode": "reconciled"}
        try:
            return GuardSpec.from_dict(x_mtg, validate=True), False
        except Exception:
            # Malformed schema x-mtg → drop to heuristic rather than crash.
            pass
    return _guess_spec_for_value(arg_value), True


def _guess_spec_for_value(value: str) -> GuardSpec:
    """Best-effort GuardSpec for a raw argument value when no schema is
    attached. Used when we only have the model's emitted calls and want
    to run them through MTG as a diagnostic layer.

    Heuristic, ordered:

    1. If value contains any Arabic character → ar-declared free-text.
    2. Else if value looks like Arabizi (Romanized Arabic, digit-letter
       substitutions) → ar-declared free-text with
       transliteration_allowed=false, so TRANSLITERATION_VIOLATION fires
       and reconciled mode can propose an arabizi_to_arabic repair.
    3. Else → latn free-text (pure English / identifiers / no Arabic
       intent detected).

    This is a heuristic; the truth lives in the tool schema. Callers
    with schemas should wrap tools via mtg.adapters.openai.guard_tool
    instead of relying on this fallback.
    """
    has_arabic = any("\u0600" <= ch <= "\u06FF" for ch in value)
    if has_arabic:
        script = "ar"
        translit_allowed = False
    else:
        try:
            from mtg.translit import looks_like_arabizi
            arabizi = looks_like_arabizi(value)
        except ImportError:  # pragma: no cover
            arabizi = False
        if arabizi:
            script = "ar"
            translit_allowed = False
        else:
            script = "latn"
            translit_allowed = True
    return GuardSpec.from_dict(
        {
            "slot_type": "free_text",
            "script": script,
            "transliteration_allowed": translit_allowed,
            "mode": "reconciled",
        },
        validate=True,
    )


def _iter_call_values(
    eval_result: "EvalResult",
) -> Iterable[tuple[str, str, str, Any]]:
    """Yield (item_id, tool_name, arg_name, arg_value) for each argument
    in each actual call. The tool_name lets the scanner look up the
    tool schema for schema-bound replay."""
    for call in eval_result.actual_calls or []:
        tool_name = call.get("function") or call.get("name") or ""
        args = call.get("arguments") or call.get("args") or {}
        if not isinstance(args, dict):
            continue
        for arg_name, arg_value in args.items():
            yield eval_result.item.id, tool_name, arg_name, arg_value


def _bootstrap_ci_95(
    values: list[float],
    n_iter: int = 1000,
    seed: int = 0xC15,
) -> Optional[tuple[float, float]]:
    """Bootstrap 95% CI on the mean of `values`.

    Resamples with replacement `n_iter` times; returns the 2.5 / 97.5
    percentile of the bootstrap-mean distribution. Scope is ONLY the
    within-run variance from which items were scored — NOT model
    run-to-run variance (that requires multiple runs).

    Returns None when there's no signal to resample (≤1 value).
    """
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_iter):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int(0.025 * n_iter)
    hi_idx = int(0.975 * n_iter)
    return (means[lo_idx], means[hi_idx])


def _find_package_repo(module_name: str) -> Optional[Path]:
    """Walk up from a module's __file__ to find its enclosing git repo."""
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    module_path = getattr(module, "__file__", None)
    if not module_path:
        return None
    repo_path = Path(module_path).resolve().parent
    for _ in range(6):
        if (repo_path / ".git").exists():
            return repo_path
        if repo_path.parent == repo_path:
            return None
        repo_path = repo_path.parent
    return None


def _git_sha_for_package(module_name: str) -> Optional[str]:
    """Return the HEAD git SHA of the checkout that contains `module_name`,
    or None if the module is not importable or the checkout isn't a git
    repo. Best-effort — we never crash a scan on a missing SHA.

    Captures code provenance for aae, mtg, toolproof, hurmoz so every
    row in a bundle answers "exactly which code produced this?"
    """
    repo_path = _find_package_repo(module_name)
    if repo_path is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha if len(sha) == 40 else None


def _git_clean_for_package(module_name: str) -> Optional[bool]:
    """Is the enclosing git repo of `module_name` clean (no untracked
    or modified files)? Returns None when git isn't available or the
    package isn't in a repo — distinct from False, so downstream can
    tell 'no signal' apart from 'known dirty'."""
    repo_path = _find_package_repo(module_name)
    if repo_path is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() == ""


def _package_version(module_name: str) -> Optional[str]:
    """Return the installed package version (via importlib.metadata) for
    `module_name`, or None if the module isn't installed as a
    distribution. Captures the other half of dependency provenance —
    not every checkout is a git repo; some are pip-installed and only
    know their version string."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover — Python 3.8 fallback
        return None
    dist_name = {
        "arabic_agent_eval": "arabic-agent-eval",
        "mtg": "mtg-guards",
        "toolproof": "toolproof",
    }.get(module_name, module_name)
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return None


def _environment_fingerprint() -> str:
    """Stable sha256 over the sorted list of every installed
    distribution name==version in this Python environment.

    Complements the coarse `dependency_versions` dict (which tracks
    only MTG-stack packages) — the fingerprint captures the full
    environment so reviewers can tell 'identical environment' from
    'same MTG-stack versions but different transitive deps'.

    Uses only the distribution name + version — does NOT attempt to
    hash sdists or lockfiles (a real lockfile-hash primitive is a
    separate concern for future work). Deterministic given the same
    importlib.metadata output.
    """
    try:
        from importlib.metadata import distributions
    except ImportError:  # pragma: no cover
        return hashlib.sha256(b"").hexdigest()
    entries: list[str] = []
    for dist in distributions():
        name = dist.metadata.get("Name") or dist.name or ""
        version = dist.version or ""
        if name:
            entries.append(f"{name.lower()}=={version}")
    entries.sort()
    canonical = "\n".join(entries).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _environment_provenance() -> dict[str, Any]:
    """Collect Python / OS / dependency-version provenance. Distinct
    from `code_shas` — those are git HEADs, these are "what was
    actually installed and running when this scan ran". Both are
    needed to reproduce a bundle from scratch."""
    return {
        "python_version": "{}.{}.{}".format(
            sys.version_info.major, sys.version_info.minor, sys.version_info.micro,
        ),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "dependency_versions": {
            pkg: _package_version(pkg)
            for pkg in ("arabic_agent_eval", "mtg", "toolproof")
        },
        "fingerprint": _environment_fingerprint(),
    }


def _fingerprint_schema_map(tool_schema_map: Optional[dict[str, dict]]) -> str:
    """Stable sha256 hash of the tool-schema map's x-mtg blocks.

    Fingerprint covers only the x-mtg annotations because the rest of
    the schema (descriptions, examples) is editorial and can change
    without affecting validation semantics. Two schema maps with the
    same x-mtg surface produce the same fingerprint.

    Empty map → sha256 of the empty string, recorded so it's
    distinguishable from a missing/null fingerprint.
    """
    if not tool_schema_map:
        return hashlib.sha256(b"").hexdigest()
    payload: dict[str, dict] = {}
    for tool_name, schema in sorted(tool_schema_map.items()):
        x_mtgs: dict[str, dict] = {}
        for key in ("parameters", "input_schema"):
            container = schema.get(key)
            if not isinstance(container, dict):
                continue
            props = container.get("properties") or {}
            for arg_name, prop in props.items():
                if isinstance(prop, dict) and "x-mtg" in prop:
                    x_mtgs[arg_name] = prop["x-mtg"]
        if x_mtgs:
            payload[tool_name] = x_mtgs
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                            separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _code_provenance() -> dict[str, str]:
    """Collect HEAD git SHAs for every package in the MTG stack that's
    importable. Missing packages get None entries so the presence or
    absence is still auditable downstream."""
    return {
        pkg: _git_sha_for_package(pkg)
        for pkg in ("arabic_agent_eval", "mtg", "toolproof")
    }


def _git_clean_provenance() -> dict[str, Optional[bool]]:
    """Per-package git-clean state. None means git not available /
    package not in a checkout; False means dirty; True means clean.
    Paired with code_shas so downstream can tell 'SHA points at
    committed code' apart from 'SHA is one commit behind a dirty
    worktree'."""
    return {
        pkg: _git_clean_for_package(pkg)
        for pkg in ("arabic_agent_eval", "mtg", "toolproof")
    }


# Request-config schema version. Bumped whenever the canonicalization
# rule below changes — old fingerprints stay comparable within their
# own schema version, and downstream consumers can see at a glance
# whether two fingerprints are comparable.
#
# rcs-v1 canonicalization:
#   - `json.dumps(config, sort_keys=True, ensure_ascii=False,
#                  separators=(",", ":"))`
#   - UTF-8 encode
#   - sha256 hex digest
#
# Future changes (e.g., dropping `None`-valued keys, normalizing
# numeric types) bump this to `rcs-v2` and document the diff here.
REQUEST_CONFIG_SCHEMA_VERSION = "rcs-v1"


def _fingerprint_request_config(config: Optional[dict]) -> Optional[str]:
    """Stable sha256 over a request-config dict (temperature, max_tokens,
    top_p, etc.). Callers set `request_config` on their BenchmarkResult
    so the matrix can stamp the fingerprint per row. Same model name
    with different config produces different fingerprints — makes
    backend drift visible.

    Canonicalization is frozen under REQUEST_CONFIG_SCHEMA_VERSION. Two
    fingerprints are comparable only when both rows carry the same
    schema version.

    Returns None when the config is empty (distinguishes "no config
    captured" from "hash of empty dict")."""
    if not config:
        return None
    canonical = json.dumps(
        config, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def normalize_provider_base_url(url: Optional[str]) -> Optional[str]:
    """Canonicalize a provider base URL so cosmetic differences
    (uppercase scheme, trailing slash) don't split what is logically
    the same backend across runs.

    Rules:
    - lowercase the scheme and host
    - strip trailing slash from the path (but preserve non-trivial
      path components like `/api/v1`)
    - strip leading/trailing whitespace

    Empty / None / whitespace-only input → None.
    """
    if not url:
        return None
    stripped = url.strip()
    if not stripped:
        return None
    if "://" in stripped:
        scheme, rest = stripped.split("://", 1)
        scheme = scheme.lower()
        if "/" in rest:
            host, path = rest.split("/", 1)
            host = host.lower()
            path = "/" + path.rstrip("/")
            return f"{scheme}://{host}{path}"
        return f"{scheme}://{rest.lower()}"
    return stripped.rstrip("/")


def normalize_model_id(model_id: Optional[str]) -> Optional[str]:
    """Canonicalize a model ID for stable cross-run comparison.

    Rules:
    - strip whitespace
    - lowercase the whole string (model IDs are case-insensitive in
      practice across Hugging Face / OpenRouter / Anthropic / OpenAI).
    - collapse internal whitespace to single spaces.

    Empty / None / whitespace-only input → None.
    """
    if not model_id:
        return None
    stripped = model_id.strip()
    if not stripped:
        return None
    return " ".join(stripped.lower().split())


def _provider_provenance(benchmark_result: "BenchmarkResult") -> dict[str, Any]:
    """Collect provider / request-side provenance. Opt-in: callers set
    attributes on BenchmarkResult via setattr (matching the cost_usd /
    latency_ms pattern). Missing attributes → None entries, so the
    presence or absence of each is always auditable.

    `provider_base_url` and `model_id` are passed through
    `normalize_provider_base_url` / `normalize_model_id` before
    stamping. When normalization changes the value, the original input
    is preserved under `provider_base_url_input` / `model_id_input`
    so reviewers can see formatting drift between source and canonical.

    The fingerprint is paired with `request_config_schema_version` so
    downstream can tell which canonicalization rule produced it."""
    base_url_input = getattr(benchmark_result, "provider_base_url", None)
    model_id_input = getattr(benchmark_result, "model_id", None)
    request_config = getattr(benchmark_result, "request_config", None)
    fingerprint = _fingerprint_request_config(request_config)

    base_url = normalize_provider_base_url(base_url_input)
    model_id = normalize_model_id(model_id_input)

    out: dict[str, Any] = {
        "provider_base_url": base_url,
        "model_id": model_id,
        "request_config_fingerprint": fingerprint,
        "request_config_schema_version": (
            REQUEST_CONFIG_SCHEMA_VERSION if fingerprint is not None else None
        ),
    }
    if base_url_input and base_url_input != base_url:
        out["provider_base_url_input"] = base_url_input
    if model_id_input and model_id_input != model_id:
        out["model_id_input"] = model_id_input
    return out


def _fingerprint_benchmark_items(benchmark_result: "BenchmarkResult") -> str:
    """Stable sha256 over the item IDs + categories + dialects + expected
    calls that this benchmark run scored. Captures the dataset slice
    that produced the row — two runs over different subsets of the
    dataset produce different fingerprints even if the model is the
    same.

    Does NOT include actual_calls (the model output) or scores —
    those vary per model and per run. Only the fixed input to the
    scanner.
    """
    payload: list[dict] = []
    for result in benchmark_result.results:
        item = result.item
        payload.append({
            "id": item.id,
            "category": item.category,
            "dialect": item.dialect,
            "difficulty": item.difficulty,
            "expected_calls": [
                {"function": c.function, "arguments": c.arguments}
                for c in item.expected_calls
            ],
        })
    payload.sort(key=lambda r: r["id"])
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _new_run_metadata(
    benchmark_result: "BenchmarkResult",
    tool_schema_map: Optional[dict[str, dict]],
) -> dict[str, Any]:
    """Stamp provenance on this scan so downstream consumers can
    deduplicate runs and trace back which inputs produced a table.

    Captures:
    - run_id / scanned_at — identity and time
    - provider / model / n_items / n_errors — scope
    - schema_map_tools — which tools the scanner was wired against
    - schema_map_fingerprint — sha256 of the x-mtg annotations used
    - code_shas — git HEAD for each MTG-stack repo present in the env
    - scanner_version — pinned code version
    """
    from arabic_agent_eval import DATASET_VERSION

    return {
        "run_id": str(uuid.uuid4()),
        "scanned_at": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "provider": benchmark_result.provider,
        "model": benchmark_result.model,
        "provider_provenance": _provider_provenance(benchmark_result),
        "n_items": benchmark_result.total_items,
        "n_errors": len(benchmark_result.errors),
        "schema_map_tools": sorted((tool_schema_map or {}).keys()),
        "schema_map_fingerprint": _fingerprint_schema_map(tool_schema_map),
        "dataset_fingerprint": _fingerprint_benchmark_items(benchmark_result),
        "dataset_version": DATASET_VERSION,
        "code_shas": _code_provenance(),
        "code_clean": _git_clean_provenance(),
        "environment": _environment_provenance(),
        "scanner_version": "mtg-matrix/0.6",
    }


def scan_with_schemas(
    benchmark_result: "BenchmarkResult",
    tool_schema_map: Optional[dict[str, dict]] = None,
) -> MatrixRow:
    """Schema-bound replay — the primary scanner path.

    For every emitted argument, look up the tool schema by function
    name, pull the `x-mtg` block for that argument if present, and
    derive the `GuardSpec` from the declared slot constraints. If the
    schema lacks x-mtg (or the tool isn't in the map), fall back to the
    heuristic `_guess_spec_for_value` — and mark the arg in the
    `heuristic_scan_rate` so the reader knows how much of the result is
    schema-grounded vs best-effort.

    `tool_schema_map` is `{function_name: tool_schema_dict}`. When
    None, uses arabic_agent_eval.functions.FUNCTIONS as the default
    (function names recognized, but no x-mtg — heuristic will dominate
    until the caller provides annotated schemas).
    """
    if not _MTG_AVAILABLE:
        raise RuntimeError(
            "mtg-guards is not installed; install it to run MTG-assisted scoring"
        )
    if tool_schema_map is None:
        tool_schema_map = _build_default_schema_map()

    row = MatrixRow(
        provider=benchmark_result.provider,
        model=benchmark_result.model,
        baseline_score=benchmark_result.overall_score,
        total_items=benchmark_result.total_items,
        run_metadata=_new_run_metadata(benchmark_result, tool_schema_map),
    )

    violation_hits = 0
    translit_hits = 0
    dialect_hits = 0
    bidi_hits = 0
    homoglyph_hits = 0
    total_calls = 0
    heuristic_calls = 0
    schema_bound_calls = 0
    schema_covered_tools: set[str] = set()
    repair_hits = 0
    family_hits: dict[str, int] = {fam: 0 for fam in _FAMILY_ORDER}
    layer_hits: dict[str, int] = {lay: 0 for lay in _LAYERS}
    quality_scores: list[float] = []
    # Per-item scores for bootstrap — one entry per non-errored item.
    baseline_item_scores: list[float] = []
    repaired_item_scores_list: list[float] = []

    from mtg.repair import score_repair

    for result in benchmark_result.results:
        if result.error:
            continue
        original_total = result.score.total
        baseline_item_scores.append(original_total)
        item_violated_args = 0
        item_repaired_args = 0
        for _item_id, tool_name, arg_name, arg_value in _iter_call_values(result):
            if not isinstance(arg_value, str) or not arg_value:
                continue
            total_calls += 1
            tool_schema = tool_schema_map.get(tool_name) if tool_name else None
            try:
                spec, used_heuristic = _spec_for_arg(
                    tool_schema, arg_name, arg_value,
                )
                guard = validate_pre(arg_value, spec)
            except Exception:
                continue

            if used_heuristic:
                heuristic_calls += 1
            else:
                schema_bound_calls += 1
                if tool_name:
                    schema_covered_tools.add(tool_name)

            codes = {v.code for v in guard.violations}
            if guard.violations:
                violation_hits += 1
                item_violated_args += 1
                family = _classify_family(codes)
                if family:
                    family_hits[family] += 1
                    layer = _classify_layer(family)
                    if layer:
                        layer_hits[layer] += 1
            if "TRANSLITERATION_VIOLATION" in codes:
                translit_hits += 1
            if "DIALECT_DRIFT" in codes:
                dialect_hits += 1
            if "BIDI_CONTROL_SMUGGLING" in codes:
                bidi_hits += 1
            if "SCRIPT_HOMOGLYPH" in codes:
                homoglyph_hits += 1

            if guard.repaired_surface:
                repair_hits += 1
                if guard.violations:
                    item_repaired_args += 1
            for repair in guard.repairs:
                if repair.proposed is None:
                    continue
                quality_scores.append(
                    score_repair(repair.original, repair.proposed, repair.action, spec)
                )

        if item_violated_args > 0 and item_repaired_args == item_violated_args:
            repaired_item_scores_list.append(1.0)
        else:
            repaired_item_scores_list.append(original_total)

    row.total_calls_scanned = total_calls
    row.violation_rate = violation_hits / max(1, total_calls)
    row.transliteration_rate = translit_hits / max(1, total_calls)
    row.dialect_drift_rate = dialect_hits / max(1, total_calls)
    row.bidi_violation_rate = bidi_hits / max(1, total_calls)
    row.homoglyph_rate = homoglyph_hits / max(1, total_calls)
    row.family_rates = {
        fam: hits / max(1, total_calls)
        for fam, hits in family_hits.items()
        if hits > 0
    }
    row.layer_rates = {
        lay: hits / max(1, total_calls)
        for lay, hits in layer_hits.items()
        if hits > 0
    }
    row.heuristic_scan_rate = heuristic_calls / max(1, total_calls)
    row.schema_bound_rate = schema_bound_calls / max(1, total_calls)
    row.schema_covered_tools = sorted(schema_covered_tools)
    row.schema_bound_arg_count = schema_bound_calls
    row.heuristic_arg_count = heuristic_calls
    row.calls_with_concrete_repair = repair_hits
    row.repair_rate = repair_hits / max(1, total_calls)
    if quality_scores:
        row.repair_quality_mean = sum(quality_scores) / len(quality_scores)
        row.repair_quality_n = len(quality_scores)
    if repaired_item_scores_list:
        row.repaired_score = sum(repaired_item_scores_list) / len(repaired_item_scores_list)

    # Bootstrap 95% CIs over the per-item score distributions. These
    # capture WITHIN-RUN variance only — the subset of items happened
    # to be scored. Model-run variance requires repeated runs.
    row.baseline_ci_95 = _bootstrap_ci_95(baseline_item_scores)
    row.repaired_ci_95 = _bootstrap_ci_95(repaired_item_scores_list)

    costs = [getattr(r, "cost_usd", None) for r in benchmark_result.results]
    costs = [c for c in costs if c is not None]
    if costs:
        row.total_cost_usd = sum(costs)

    latencies = [getattr(r, "latency_ms", None) for r in benchmark_result.results]
    latencies = sorted(lat for lat in latencies if lat is not None)
    if latencies:
        row.median_latency_ms = latencies[len(latencies) // 2]

    return row


def scan_with_mtg(benchmark_result: "BenchmarkResult") -> MatrixRow:
    """Back-compat alias for `scan_with_schemas` with no schema map.

    Kept for existing callers. New code should call `scan_with_schemas`
    directly and pass a real tool-schema map to get schema-bound
    replay. Without a map, the scanner falls back to the heuristic
    path for every arg — the heuristic_scan_rate on the returned row
    will be close to 1.0, and downstream consumers should treat the
    result as diagnostic only.
    """
    return scan_with_schemas(benchmark_result, tool_schema_map=None)


def build_matrix(
    benchmark_results: Iterable["BenchmarkResult"],
    tool_schema_map: Optional[dict[str, dict]] = None,
) -> ResultMatrix:
    """Fold N benchmark runs into a ResultMatrix. Schema-bound when
    `tool_schema_map` is provided; heuristic-fallback when None."""
    rows = [scan_with_schemas(br, tool_schema_map) for br in benchmark_results]
    return ResultMatrix(rows=rows)


# ---------- renderers ----------


def render_markdown(matrix: ResultMatrix) -> str:
    """Render the matrix as Markdown — main table + 3-layer taxonomy +
    why-failed family breakdown + repair quality + run metadata.
    Suitable for a blog post, arXiv appendix, or README section."""

    # Main result table with bootstrap CIs on baseline/repaired.
    main_header = (
        "| provider | model | baseline (95% CI) | repaired (95% CI) | Δ repair | "
        "viol. % | translit % | drift % | bidi % | homoglyph % | heur. scan % | "
        "items | calls | cost | p50 ms |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    def _fmt_ci(score: float, ci: Optional[tuple[float, float]]) -> str:
        if ci is None:
            return f"{score:.3f}"
        return f"{score:.3f} ({ci[0]:.3f}–{ci[1]:.3f})"

    main_rows = []
    diagnostic_rows: list[str] = []
    for row in matrix.rows:
        delta = row.repaired_score - row.baseline_score
        cost = f"${row.total_cost_usd:.3f}" if row.total_cost_usd is not None else "—"
        latency = f"{row.median_latency_ms:.0f}" if row.median_latency_ms is not None else "—"
        # ⚠ prefix on provider name when the row failed the publish
        # gate's prerequisites (excess heuristic fallback, missing
        # provenance). Publish the marker; never hide it.
        marker = "⚠ " if row.diagnostic else ""
        if row.diagnostic:
            diagnostic_rows.append(f"{row.provider}/{row.model}")
        main_rows.append(
            "| {marker}{prov} | {model} | {base} | {rep} | {delta:+.3f} | "
            "{v:.1%} | {t:.1%} | {d:.1%} | {b:.1%} | {h:.1%} | {hs:.1%} | "
            "{items} | {calls} | {cost} | {lat} |".format(
                marker=marker,
                prov=row.provider, model=row.model,
                base=_fmt_ci(row.baseline_score, row.baseline_ci_95),
                rep=_fmt_ci(row.repaired_score, row.repaired_ci_95),
                delta=delta,
                v=row.violation_rate, t=row.transliteration_rate,
                d=row.dialect_drift_rate, b=row.bidi_violation_rate,
                h=row.homoglyph_rate, hs=row.heuristic_scan_rate,
                items=row.total_items, calls=row.total_calls_scanned,
                cost=cost, lat=latency,
            )
        )

    # 3-layer taxonomy — surface / language / security.
    layer_header = (
        "| provider | model | surface | language | security |\n"
        "|---|---|---:|---:|---:|"
    )
    layer_rows = []
    for row in matrix.rows:
        lr = row.layer_rates
        layer_rows.append(
            "| {p} | {m} | {s:.1%} | {l:.1%} | {sec:.1%} |".format(
                p=row.provider, m=row.model,
                s=lr.get("surface", 0.0),
                l=lr.get("language", 0.0),
                sec=lr.get("security", 0.0),
            )
        )

    # Why-failed family breakdown (granular).
    fam_header = (
        "| provider | model | script | canonicalization | dialect | overflow | "
        "bidi | homoglyph |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|"
    )
    fam_rows = []
    for row in matrix.rows:
        fr = row.family_rates
        fam_rows.append(
            "| {p} | {m} | {s:.1%} | {c:.1%} | {d:.1%} | {o:.1%} | {b:.1%} | {h:.1%} |".format(
                p=row.provider, m=row.model,
                s=fr.get("script", 0.0), c=fr.get("canonicalization", 0.0),
                d=fr.get("dialect", 0.0), o=fr.get("overflow", 0.0),
                b=fr.get("bidi", 0.0), h=fr.get("homoglyph", 0.0),
            )
        )

    quality_header = (
        "| provider | model | repair quality | N repairs scored |\n"
        "|---|---|---:|---:|"
    )
    quality_rows = []
    for row in matrix.rows:
        if row.repair_quality_mean is None:
            continue
        quality_rows.append(
            "| {p} | {m} | {q:.2f} | {n} |".format(
                p=row.provider, m=row.model,
                q=row.repair_quality_mean, n=row.repair_quality_n,
            )
        )

    # Schema coverage summary — positive counterpart of heuristic_scan_rate.
    coverage_header = (
        "| provider | model | schema-bound % | schema-bound args | heuristic args | tools covered |\n"
        "|---|---|---:|---:|---:|---|"
    )
    coverage_rows = []
    for row in matrix.rows:
        tools = ", ".join(f"`{t}`" for t in row.schema_covered_tools) or "—"
        coverage_rows.append(
            "| {p} | {m} | {sb:.1%} | {sba} | {ha} | {tools} |".format(
                p=row.provider, m=row.model,
                sb=row.schema_bound_rate,
                sba=row.schema_bound_arg_count,
                ha=row.heuristic_arg_count,
                tools=tools,
            )
        )

    # Run provenance — one line per row, safe to eyeball.
    meta_lines = []
    for row in matrix.rows:
        md = row.run_metadata or {}
        if md:
            meta_lines.append(
                "- `{prov}` / `{model}` — run_id `{rid}` · scanned {ts} · "
                "{n} items, {err} errors · scanner `{scan}`".format(
                    prov=row.provider, model=row.model,
                    rid=md.get("run_id", "?")[:8],
                    ts=md.get("scanned_at", "?"),
                    n=md.get("n_items", "?"),
                    err=md.get("n_errors", "?"),
                    scan=md.get("scanner_version", "?"),
                )
            )

    parts = [
        "## Result matrix",
        "",
        "`baseline` and `repaired` include bootstrap 95% CIs over the "
        "items scored (within-run variance only — NOT model-run variance). "
        "`heur. scan %` reports the fraction of args scored via the "
        "heuristic fallback; rows with `heur. scan > 10%` are marked "
        "⚠ diagnostic and must not be published as clean results.",
        "",
        main_header,
        *main_rows,
        "",
    ]
    if diagnostic_rows:
        parts.extend([
            f"> ⚠ **{len(diagnostic_rows)} row(s) marked diagnostic** — "
            f"not schema-grounded enough to publish as a clean result: "
            f"{', '.join(diagnostic_rows)}. Re-run with annotated "
            f"tool-schema map or label these rows diagnostic.",
            "",
        ])
    parts.extend([
        "## 3-layer taxonomy",
        "",
        "Each failing argument is classified into exactly one layer. "
        "**Surface** = schema-shape failures (script mismatch). "
        "**Language** = register, morphology, canonicalization, overflow. "
        "**Security** = Unicode-layer attacks (BiDi, homoglyph, invisibles). "
        "Row sums are ≤ `viol. %` in the matrix above.",
        "",
        layer_header,
        *layer_rows,
        "",
        "## Why failed — family breakdown",
        "",
        "Granular view of the same classification (worst family wins per arg).",
        "",
        fam_header,
        *fam_rows,
    ])
    parts.extend([
        "",
        "## Schema coverage",
        "",
        "`schema-bound %` = fraction of scanned args with an x-mtg block "
        "on their tool schema. Higher is stronger evidence. Rows with "
        "low coverage end up marked ⚠ diagnostic in the main table above.",
        "",
        coverage_header,
        *coverage_rows,
    ])
    if quality_rows:
        parts.extend([
            "",
            "## Repair quality",
            "",
            "Score in [0.0, 1.0]. 1.0 = clean repair, 0.5 = needs review, "
            "0.0 = broken invariant. See `mtg.repair.score_repair`.",
            "",
            quality_header,
            *quality_rows,
        ])
    if meta_lines:
        parts.extend([
            "",
            "## Run provenance",
            "",
            *meta_lines,
        ])
    return "\n".join(parts)


def render_csv(matrix: ResultMatrix) -> str:
    """Render the matrix as CSV for spreadsheets / plotting."""
    cols = [
        "provider", "model",
        "baseline_score", "repaired_score",
        "violation_rate", "transliteration_rate", "dialect_drift_rate",
        "bidi_violation_rate", "homoglyph_rate",
        "repair_rate", "total_items", "total_calls_scanned",
        "total_cost_usd", "median_latency_ms",
    ]
    lines = [",".join(cols)]
    for row in matrix.rows:
        d = row.to_dict()
        lines.append(",".join(
            "" if d.get(c) is None else str(d[c])
            for c in cols
        ))
    return "\n".join(lines)


def load_benchmark_result_from_json(path: Path) -> "BenchmarkResult":
    """Load a previously-saved BenchmarkResult JSON file (the shape
    produced by BenchmarkResult.to_dict()). Used by the CLI runner."""
    from arabic_agent_eval.dataset import EvalItem, ExpectedCall
    from arabic_agent_eval.evaluator import BenchmarkResult, EvalResult
    from arabic_agent_eval.scoring import Score

    data = json.loads(path.read_text(encoding="utf-8"))
    results: list[EvalResult] = []
    for r in data.get("results", []):
        item_data = r.get("item") or {}
        # Best-effort item reconstruction — matrix.py only needs id/category/dialect
        item = EvalItem(
            id=item_data.get("id") or r.get("item_id", ""),
            category=item_data.get("category") or r.get("category", ""),
            instruction=item_data.get("instruction") or r.get("instruction", ""),
            dialect=item_data.get("dialect") or r.get("dialect", "msa"),
            available_functions=item_data.get("available_functions", []),
            expected_calls=[
                ExpectedCall(function=c.get("function", ""), arguments=c.get("arguments", {}))
                for c in item_data.get("expected_calls", [])
            ],
            difficulty=item_data.get("difficulty", "easy"),
        )
        score_data = r.get("score", {})
        score = Score(
            item_id=score_data.get("item_id", item.id),
            function_selection=score_data.get("function_selection", 0.0),
            argument_accuracy=score_data.get("argument_accuracy", 0.0),
            arabic_preservation=score_data.get("arabic_preservation", 0.0),
            dialect_understanding=score_data.get("dialect_understanding", 0.0),
            error_handling=score_data.get("error_handling", 0.0),
            category=score_data.get("category", item.category),
        )
        er = EvalResult(
            item=item,
            score=score,
            actual_calls=r.get("actual_calls", []),
            raw_response=r.get("raw_response", ""),
            error=r.get("error"),
        )
        # Optional cost/latency telemetry
        if "cost_usd" in r:
            setattr(er, "cost_usd", r["cost_usd"])
        if "latency_ms" in r:
            setattr(er, "latency_ms", r["latency_ms"])
        results.append(er)

    return BenchmarkResult(
        provider=data.get("provider", "unknown"),
        model=data.get("model", "unknown"),
        results=results,
    )


__all__ = [
    "MatrixRow",
    "ResultMatrix",
    "build_matrix",
    "load_benchmark_result_from_json",
    "render_csv",
    "render_markdown",
    "scan_with_mtg",
    "scan_with_schemas",
]
