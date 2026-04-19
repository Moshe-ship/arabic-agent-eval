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
from typing import Any, Iterable, Optional

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


@dataclass
class BundleManifest:
    """Structured view of MANIFEST.json."""

    bundle_version: str = BUNDLE_VERSION
    scanner_version: str = ""
    created_at: str = ""
    files: dict[str, str] = field(default_factory=dict)   # path → sha256
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    row_summaries: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "bundle_version": self.bundle_version,
                "scanner_version": self.scanner_version,
                "created_at": self.created_at,
                "files": self.files,
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
            files=dict(data.get("files", {})),
            thresholds=dict(data.get("thresholds", {})),
            row_summaries=list(data.get("row_summaries", [])),
        )


class BundleError(ValueError):
    """Raised on malformed or tampered bundles."""


def _row_summary(row: dict[str, Any]) -> dict[str, Any]:
    """Extract the integrity-critical fields of a MatrixRow into the
    manifest so the gate can inspect them without reading matrix.json
    first. Keeps the manifest self-describing."""
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
        "has_runs": bool((md.get("schema_map_tools") or [])),
    }


def write_bundle(
    matrix: ResultMatrix,
    out_dir: Path,
    *,
    html: Optional[str] = None,
    run_json_files: Optional[list[Path]] = None,
    thresholds: Optional[dict[str, float]] = None,
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

    # Aggregate scanner_version across rows — if any row declares a
    # different version we flag it as empty so the gate can reject.
    versions = {
        (r.get("run_metadata") or {}).get("scanner_version", "")
        for r in matrix.to_dict()["rows"]
    }
    versions.discard("")
    scanner_version = next(iter(versions)) if len(versions) == 1 else ""

    manifest = BundleManifest(
        bundle_version=BUNDLE_VERSION,
        scanner_version=scanner_version,
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        files=written,
        thresholds=dict(thresholds or DEFAULT_THRESHOLDS),
        row_summaries=[_row_summary(r) for r in matrix.to_dict()["rows"]],
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
