# Dataset changelog

Backing record for `arabic_agent_eval.DATASET_VERSION`. Every bump to
that constant must land here with a dated entry describing what
changed, so reviewers can compare two published bundles without
diffing raw items.

## Format

- `<YYYY-MM>-<language>-v<N>`
- Bump `N` when items are removed, or when any `expected_calls` on
  existing items is rewritten (breaking change — cross-version
  comparisons are invalid).
- Bump the `<YYYY-MM>` prefix for additive changes (new items added,
  no existing items mutated) — cross-version comparisons remain valid
  but the item count differs.

## Versions

### `2026-04-arabic-v1` — initial published version

- **Date:** 2026-04-19
- **Items:** 51 across 6 categories (simple_function_calling,
  parameter_extraction, multi_step, tool_selection, dialect_handling,
  error_recovery).
- **Dialects:** MSA, Gulf, Egyptian, Levantine, Maghrebi.
- **Tools:** 22 in `arabic_agent_eval.functions.FUNCTIONS`, every
  parameter annotated with `x-mtg` blocks.
- **Notes:** first version to ship a real canonical bundle would cite
  this tag. No prior versions exist — no delta table.

---

## Citation

When citing numbers from a bundle:

    See bundle `bundles/<name>/` (dataset_version `2026-04-arabic-v1`,
    dataset_fingerprint `<short-sha>`). DATASET_CHANGELOG.md lists the
    items covered by each version.

The `dataset_fingerprint` (sha256 over item IDs + categories + dialects
+ expected_calls) is the authoritative identity — the version label is
the reader-friendly handle.
