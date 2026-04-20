"""Canonical matrix bundle — the publishable artifact.

A bundle is the single unit that gets shared in a blog post, arXiv
appendix, or research issue. It pins everything needed to reproduce a
result-table claim back to the raw run JSON:

    bundle/
        MANIFEST.json       — versions, sha256 of every file, thresholds
        matrix.json         — canonical ResultMatrix.to_dict() output
        table.md            — human-readable Markdown rendering
        table.csv           — spreadsheet-friendly flat CSV
        scorecard.html      — optional screenshot-grade HTML scorecard
        runs/               — source benchmark-run JSON files (opt)
            <provider>__<model>.json

Every file except `runs/` is required. `runs/` is optional but strongly
recommended — without it, the bundle's numbers are not reproducible.

Integrity: MANIFEST.json carries a sha256 for every file in the bundle.
`validate_bundle(path)` recomputes the hashes and refuses to load if
any file has been tampered with.

The publish gate (`scripts/check_publish_ready.py`) reads the bundle
and enforces quality rules (heuristic_scan_rate ≤ threshold, CIs
present, scanner_version pinned, etc.) — manifest integrity is a
prerequisite, not a quality rule.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, TYPE_CHECKING

from arabic_agent_eval.matrix import (
    ResultMatrix,
    render_csv,
    render_markdown,
)


BUNDLE_VERSION = "1"
MANIFEST_NAME = "MANIFEST.json"
DEFAULT_THRESHOLDS: dict[str, float] = {
    "heuristic_scan_rate_max": 0.10,
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# Recognized type values for raw_index descriptors. Not exhaustive —
# callers can use any string — but these are the conventions used
# across reviews so unknown strings get a soft warning through docs,
# not a hard rejection.
_RAW_INDEX_KNOWN_TYPES: frozenset[str] = frozenset({
    "provider_output", "request", "response", "trace", "error_log",
    "system_prompt", "retrieval_context", "tool_output",
})


def _validate_raw_index_entry(key: str, entry: Any) -> None:
    """Light schema check on a raw_index descriptor. Rejects wrong-type
    values with BundleError; permits unknown `type` strings (new
    evidence kinds shouldn't require a code change) but rejects
    non-string values for recognized keys.

    Known keys and their accepted types:
    - type:          str (recommended values in _RAW_INDEX_KNOWN_TYPES)
    - source:        str
    - redacted:      bool
    - redaction_note: str
    - item_id:       str
    Extra keys are permitted and not type-checked.
    """
    if not isinstance(entry, dict):
        raise BundleError(
            f"raw_index[{key!r}] must be a dict, got {type(entry).__name__}"
        )
    type_expectations: dict[str, type] = {
        "type": str,
        "source": str,
        "redaction_note": str,
        "item_id": str,
        "row_key": str,
        "redacted": bool,
    }
    for field_name, expected_type in type_expectations.items():
        if field_name not in entry:
            continue
        value = entry[field_name]
        # `bool` is a subclass of `int` — we want strict bool checks.
        if expected_type is bool:
            if not isinstance(value, bool):
                raise BundleError(
                    f"raw_index[{key!r}].{field_name} must be bool, "
                    f"got {type(value).__name__} ({value!r})"
                )
        elif not isinstance(value, expected_type):
            raise BundleError(
                f"raw_index[{key!r}].{field_name} must be "
                f"{expected_type.__name__}, got {type(value).__name__} "
                f"({value!r})"
            )


def _row_key(provider: str, model: str) -> str:
    """Canonical row identifier used in raw_index entries and the
    manifest.row_links reverse index. Simple "provider/model" format;
    lookups match `MatrixRow.provider` + `MatrixRow.model`."""
    return f"{provider}/{model}"


@dataclass
class BundleManifest:
    """Structured view of MANIFEST.json."""

    bundle_version: str = BUNDLE_VERSION
    scanner_version: str = ""
    created_at: str = ""
    # True when the bundle has one or more `runs/*.json` source files.
    # Bundle-level fact (not per-row) since runs/ is a shared directory.
    # Derived from `files` at write time; load_bundle recomputes to
    # guard against manifest edits that lie about it.
    has_runs: bool = False
    # Invocation provenance — captures the exact build command inputs
    # that produced this bundle. Stamped by `build_bundle.py` or any
    # other wrapper that calls `write_bundle(invocation=...)`. Empty
    # when the bundle was produced by direct library calls.
    #
    # Expected keys (all optional but recommended):
    # - generator: "scripts/build_bundle.py" or equivalent
    # - generator_version: pin
    # - git_ref / git_branch: where the builder was invoked from
    # - schema_map_source: the path the user passed, or "<default>"
    # - run_json_sha256: {run_filename: sha256} per source run
    # - built_at: ISO-8601 UTC (often equals created_at but distinct
    #   so we can distinguish "build time" from "manifest write time")
    invocation: dict[str, Any] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)   # path → sha256
    # Structured index over raw/ files. Maps each `raw/<filename>` key
    # to a small descriptor dict so reviewers don't guess what each
    # raw file is. Recognized descriptor keys:
    #   type (e.g. "provider_output", "trace", "request", "system_prompt"),
    #   source (e.g. "openrouter", "anthropic", "local-run"),
    #   redacted (bool), redaction_note (str), item_id (optional).
    # Schema is minimal on purpose — extra keys permitted but preferred
    # ones keep reviews uniform.
    raw_index: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Reverse index from `provider/model` → list of `raw/<filename>`
    # paths belonging to that row. Derived at write time from any
    # raw_index entry that carries `row_key`. Non-authoritative —
    # `raw_index` is the source of truth; `row_links` is the lookup
    # table reviewers use to answer "what raw evidence belongs to this
    # row?" without scanning every descriptor.
    row_links: dict[str, list[str]] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    row_summaries: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "bundle_version": self.bundle_version,
                "scanner_version": self.scanner_version,
                "created_at": self.created_at,
                "has_runs": self.has_runs,
                "invocation": self.invocation,
                "files": self.files,
                "raw_index": self.raw_index,
                "row_links": self.row_links,
                "thresholds": self.thresholds,
                "row_summaries": self.row_summaries,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BundleManifest:
        return cls(
            bundle_version=data.get("bundle_version", ""),
            scanner_version=data.get("scanner_version", ""),
            created_at=data.get("created_at", ""),
            has_runs=bool(data.get("has_runs", False)),
            invocation=dict(data.get("invocation", {})),
            files=dict(data.get("files", {})),
            raw_index=dict(data.get("raw_index", {})),
            row_links={
                k: list(v) for k, v in (data.get("row_links") or {}).items()
            },
            thresholds=dict(data.get("thresholds", {})),
            row_summaries=list(data.get("row_summaries", [])),
        )

    @property
    def is_synthetic(self) -> bool:
        """True when the bundle declares itself synthetic via
        `invocation.synthetic = true`. Synthetic bundles have a
        distinct publish path in the gate — they're allowed to skip
        runs/ presence and clean-tree checks, but they must carry the
        synthetic flag explicitly so nothing that isn't synthetic can
        quietly take the synthetic rules."""
        return bool(self.invocation.get("synthetic", False))


class BundleError(ValueError):
    """Raised on malformed or tampered bundles."""


def _row_summary(row: dict[str, Any]) -> dict[str, Any]:
    """Extract the integrity-critical fields of a MatrixRow into the
    manifest so the gate can inspect them without reading matrix.json
    first. Keeps the manifest self-describing.

    NOTE: does NOT include `has_runs` — runs/ is a bundle-level fact,
    not per-row. See `BundleManifest.has_runs` instead.
    """
    md = row.get("run_metadata") or {}
    return {
        "provider": row.get("provider"),
        "model": row.get("model"),
        "run_id": md.get("run_id"),
        "scanned_at": md.get("scanned_at"),
        "scanner_version": md.get("scanner_version"),
        "n_items": md.get("n_items"),
        "total_calls_scanned": row.get("total_calls_scanned"),
        "heuristic_scan_rate": row.get("heuristic_scan_rate"),
        "schema_bound_rate": row.get("schema_bound_rate"),
        "schema_covered_tools": list(row.get("schema_covered_tools") or []),
        "schema_bound_arg_count": row.get("schema_bound_arg_count"),
        "heuristic_arg_count": row.get("heuristic_arg_count"),
        "diagnostic": row.get("diagnostic"),
        "has_baseline_ci": row.get("baseline_ci_95") is not None,
        "has_repaired_ci": row.get("repaired_ci_95") is not None,
    }


def write_bundle(
    matrix: ResultMatrix,
    out_dir: Path,
    *,
    html: Optional[str] = None,
    run_json_files: Optional[list[Path]] = None,
    raw_files: Optional[list[Path]] = None,
    raw_index: Optional[dict[str, dict[str, Any]]] = None,
    known_item_ids: Optional[Iterable[str]] = None,
    thresholds: Optional[dict[str, float]] = None,
    invocation: Optional[dict[str, Any]] = None,
) -> Path:
    """Serialize a ResultMatrix into a canonical bundle directory.

    Files written:

    - `matrix.json`      — `matrix.to_dict()` with indented JSON
    - `table.md`         — `render_markdown(matrix)`
    - `table.csv`        — `render_csv(matrix)`
    - `scorecard.html`   — only if `html` is provided
    - `runs/<name>.json` — copies of each source file in `run_json_files`
    - `MANIFEST.json`    — sha256 of every file above + provenance

    Returns the out_dir. Parent directories are created as needed.
    Existing files under `out_dir` are overwritten; extra files are
    left alone so callers can add their own supplemental evidence
    (NOT covered by the manifest).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_json = json.dumps(matrix.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)
    md_text = render_markdown(matrix)
    csv_text = render_csv(matrix)

    written: dict[str, str] = {}

    def _write_text(name: str, text: str) -> None:
        path = out_dir / name
        path.write_text(text, encoding="utf-8")
        written[name] = _sha256_file(path)

    _write_text("matrix.json", matrix_json)
    _write_text("table.md", md_text)
    _write_text("table.csv", csv_text)
    if html is not None:
        _write_text("scorecard.html", html)

    # Run-source copies — optional but strongly recommended for
    # reproducibility. Sha'd and added to the manifest.
    if run_json_files:
        runs_dir = out_dir / "runs"
        runs_dir.mkdir(exist_ok=True)
        for src in run_json_files:
            if not src.exists():
                raise BundleError(f"run JSON not found: {src}")
            dst_name = f"runs/{src.name}"
            dst = out_dir / dst_name
            dst.write_bytes(src.read_bytes())
            written[dst_name] = _sha256_file(dst)

    # Raw evidence copies — redacted provider outputs, trace excerpts,
    # or anything else the caller wants to ship alongside runs/. Like
    # runs/, these get sha256'd and covered by manifest integrity. The
    # gate treats raw/ as purely informational — never required, but
    # the sha256 coverage makes silent mutation detectable.
    if raw_files:
        raw_dir = out_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        for src in raw_files:
            if not src.exists():
                raise BundleError(f"raw file not found: {src}")
            dst_name = f"raw/{src.name}"
            dst = out_dir / dst_name
            dst.write_bytes(src.read_bytes())
            written[dst_name] = _sha256_file(dst)

    # Aggregate scanner_version across rows — if any row declares a
    # different version we flag it as empty so the gate can reject.
    versions = {
        (r.get("run_metadata") or {}).get("scanner_version", "")
        for r in matrix.to_dict()["rows"]
    }
    versions.discard("")
    scanner_version = next(iter(versions)) if len(versions) == 1 else ""

    bundle_has_runs = any(rel.startswith("runs/") for rel in written)

    # Collect known row keys from the matrix so raw_index entries can
    # be cross-checked against actual rows.
    matrix_dict = matrix.to_dict()
    known_row_keys: set[str] = {
        _row_key(row.get("provider") or "", row.get("model") or "")
        for row in (matrix_dict.get("rows") or [])
    }
    known_item_id_set: Optional[set[str]] = (
        set(known_item_ids) if known_item_ids is not None else None
    )

    # Normalize raw_index keys to the `raw/<filename>` form that
    # matches `files`. Reject dangling entries AND type-check the
    # descriptor fields — stale index entries or wrong-type values
    # would be misleading at review time.
    raw_index_normalized: dict[str, dict[str, Any]] = {}
    row_links: dict[str, list[str]] = {}
    if raw_index:
        written_raw = {rel for rel in written if rel.startswith("raw/")}
        for key, entry in raw_index.items():
            normalized = key if key.startswith("raw/") else f"raw/{key}"
            if normalized not in written_raw:
                raise BundleError(
                    f"raw_index entry {key!r} does not correspond to any "
                    f"file in raw/. Either missing from raw_files, or the "
                    f"key is a typo."
                )
            _validate_raw_index_entry(key, entry)

            # Cross-link item_id against the caller-supplied ID set.
            # When caller passes None, we skip — the check is opt-in
            # at call time since not every caller has the items list
            # handy (e.g. pre-serialized BenchmarkResult JSONs).
            entry_item_id = entry.get("item_id")
            if (
                known_item_id_set is not None
                and entry_item_id
                and entry_item_id not in known_item_id_set
            ):
                raise BundleError(
                    f"raw_index[{key!r}].item_id = {entry_item_id!r} is "
                    f"not in the provided known_item_ids set. Either "
                    f"the ID is stale or the raw file belongs to a "
                    f"different dataset."
                )

            # Cross-link row_key against the matrix. Row-linked raw
            # files get added to the reverse index so reviewers can
            # look up "what raw evidence belongs to this row?" in one
            # lookup.
            row_key = entry.get("row_key")
            if row_key:
                if not isinstance(row_key, str):
                    raise BundleError(
                        f"raw_index[{key!r}].row_key must be str "
                        f"(got {type(row_key).__name__})"
                    )
                if row_key not in known_row_keys:
                    raise BundleError(
                        f"raw_index[{key!r}].row_key = {row_key!r} "
                        f"does not match any matrix row. Valid row keys: "
                        f"{sorted(known_row_keys) or '<none>'}"
                    )
                row_links.setdefault(row_key, []).append(normalized)

            raw_index_normalized[normalized] = dict(entry)

    # Sort raw_links per row for stability
    row_links = {k: sorted(v) for k, v in sorted(row_links.items())}

    manifest = BundleManifest(
        bundle_version=BUNDLE_VERSION,
        scanner_version=scanner_version,
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        has_runs=bundle_has_runs,
        invocation=dict(invocation or {}),
        files=written,
        raw_index=raw_index_normalized,
        row_links=row_links,
        thresholds=dict(thresholds or DEFAULT_THRESHOLDS),
        row_summaries=[_row_summary(r) for r in matrix_dict["rows"]],
    )
    (out_dir / MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")

    return out_dir


def load_bundle(path: Path) -> tuple[BundleManifest, dict[str, Any]]:
    """Load a bundle, verify its manifest integrity, and return
    `(manifest, matrix_json_dict)`.

    Raises `BundleError` if:
    - MANIFEST.json missing or malformed
    - any listed file is missing
    - any listed file's sha256 doesn't match the manifest
    - bundle_version is unknown

    Does NOT enforce quality rules (thresholds, etc.) — that's the
    publish gate's job.
    """
    if not path.is_dir():
        raise BundleError(f"bundle path is not a directory: {path}")
    manifest_path = path / MANIFEST_NAME
    if not manifest_path.exists():
        raise BundleError(f"missing {MANIFEST_NAME}")
    try:
        manifest = BundleManifest.from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
    except json.JSONDecodeError as exc:
        raise BundleError(f"malformed {MANIFEST_NAME}: {exc}") from exc

    if manifest.bundle_version != BUNDLE_VERSION:
        raise BundleError(
            f"unsupported bundle_version {manifest.bundle_version!r} "
            f"(this loader speaks {BUNDLE_VERSION!r})"
        )

    # Guard against manifest-level lies about has_runs — recompute from
    # the actual file list and compare. A tampered manifest could claim
    # has_runs=true without populating runs/, which would sneak past a
    # surface-level check.
    actual_has_runs = any(rel.startswith("runs/") for rel in manifest.files)
    if manifest.has_runs != actual_has_runs:
        raise BundleError(
            f"has_runs={manifest.has_runs} in manifest disagrees with "
            f"actual runs/ files present ({actual_has_runs})"
        )

    # Integrity check
    for rel, expected in manifest.files.items():
        f = path / rel
        if not f.exists():
            raise BundleError(f"manifest lists missing file: {rel}")
        actual = _sha256_file(f)
        if actual != expected:
            raise BundleError(
                f"sha256 mismatch for {rel}: manifest {expected!r} "
                f"!= file {actual!r}"
            )

    matrix_path = path / "matrix.json"
    if "matrix.json" not in manifest.files or not matrix_path.exists():
        raise BundleError("bundle missing matrix.json")

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    return manifest, matrix


def validate_bundle(path: Path) -> None:
    """Raise BundleError if the bundle at `path` fails integrity checks.
    Thin wrapper around load_bundle for callers that only want to assert."""
    load_bundle(path)


__all__ = [
    "BUNDLE_VERSION",
    "BundleError",
    "BundleManifest",
    "DEFAULT_THRESHOLDS",
    "load_bundle",
    "validate_bundle",
    "write_bundle",
]
