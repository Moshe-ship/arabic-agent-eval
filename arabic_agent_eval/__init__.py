"""Arabic Agent Eval — Arabic function-calling benchmark."""

__version__ = "0.1.0"

# Human-readable version label for the shipped dataset items. Bumped
# whenever items are added, removed, or edited. Lives alongside the
# opaque dataset_fingerprint sha256 so citations can say
# "arabic-v1 @ abc123..." without opening the JSON.
#
# Format: `<YYYY-MM>-<language>-v<N>`. Bump N on breaking content
# changes (items removed / expected_calls rewritten); bump the month
# when adding net-new items without changing existing ones.
DATASET_VERSION = "2026-04-arabic-v1"

# Lock: DATASET_VERSION → expected fingerprint over the full dataset
# (computed by `arabic_agent_eval.matrix._fingerprint_benchmark_items`
# on a BenchmarkResult containing every item from `Dataset()`).
#
# A regression test (`test_dataset_fingerprint_lock.py`) asserts that
# the computed fingerprint for the current DATASET_VERSION equals the
# value here. If someone edits dataset content without bumping
# DATASET_VERSION + adding a new lock entry, the test fails loudly
# with the new fingerprint in the message — making the version bump
# impossible to forget.
#
# Bumping protocol:
#   1. Bump DATASET_VERSION above.
#   2. Add the new version → new fingerprint line below.
#   3. Add a changelog entry in docs/DATASET_CHANGELOG.md.
DATASET_FINGERPRINT_LOCK: dict[str, str] = {
    "2026-04-arabic-v1": (
        "c633f00573b4ac9475a28c84a1cc70f04c8b66f74549fc3545d1438232a385d3"
    ),
}

from arabic_agent_eval.dataset import Dataset, EvalItem
from arabic_agent_eval.evaluator import Evaluator, EvalResult
from arabic_agent_eval.scoring import Score, CategoryScore
from arabic_agent_eval.functions import FUNCTIONS

__all__ = [
    "DATASET_VERSION",
    "DATASET_FINGERPRINT_LOCK",
    "Dataset",
    "EvalItem",
    "Evaluator",
    "EvalResult",
    "Score",
    "CategoryScore",
    "FUNCTIONS",
]
