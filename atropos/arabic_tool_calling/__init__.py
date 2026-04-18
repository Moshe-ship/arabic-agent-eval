"""Arabic tool-calling environment for NousResearch/atropos.

Imports the arabic-agent-eval dataset + 4-axis grader. Designed to land
upstream at environments/community/arabic_tool_calling/.
"""

from __future__ import annotations

from .env import ArabicToolCallingEnv, EnvConfig
from .scoring import RewardWeights, reward_from_score, score_turn
from .tasks import Task, sample_tasks, DIALECT_PERSONAS

__all__ = [
    "ArabicToolCallingEnv",
    "EnvConfig",
    "RewardWeights",
    "reward_from_score",
    "score_turn",
    "Task",
    "sample_tasks",
    "DIALECT_PERSONAS",
]
