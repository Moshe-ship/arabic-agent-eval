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

from arabic_agent_eval.dataset import Dataset, EvalItem
from arabic_agent_eval.evaluator import Evaluator, EvalResult
from arabic_agent_eval.scoring import Score, CategoryScore
from arabic_agent_eval.functions import FUNCTIONS

__all__ = [
    "DATASET_VERSION",
    "Dataset",
    "EvalItem",
    "Evaluator",
    "EvalResult",
    "Score",
    "CategoryScore",
    "FUNCTIONS",
]
