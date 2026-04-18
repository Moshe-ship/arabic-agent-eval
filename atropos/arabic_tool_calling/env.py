"""Arabic tool-calling Atropos environment.

AtroposBaseEnv subclass that wraps the arabic-agent-eval benchmark as an RL
environment. Rollouts are scored with the 4-axis canonical structured-call
grader and rewards are equal-weighted by default.

Ships for landing upstream at:
    NousResearch/atropos/environments/community/arabic_tool_calling/

When the upstream package changes imports, only this file's `try: from
atropos...` block needs updating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .scoring import (
    RewardWeights,
    reward_from_score,
    score_turn,
)
from .tasks import Task, sample_tasks


# Late-binding import shim. When this folder lands in NousResearch/atropos,
# replace the try/except with the real upstream import.
try:  # pragma: no cover — exercised only in upstream environment
    from atroposlib.envs.base import AtroposBaseEnv as _UpstreamBaseEnv  # type: ignore
    _BASE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _UpstreamBaseEnv = object  # type: ignore
    _BASE_AVAILABLE = False


@dataclass
class EnvConfig:
    """Configuration for ArabicToolCallingEnv."""

    max_turns: int = 1              # single-turn by default; set >1 for multi-turn rollouts
    n_tasks_per_rollout: int = 16
    seed: int | None = 42
    dialect_filter: str | None = None   # None for all dialects; "gulf", "egy", etc.
    # None → use category-aware Score.total (matches arabic-agent-eval grading).
    # Provide a RewardWeights instance for equal- or custom-weighting across axes.
    reward_weights: RewardWeights | None = None


class ArabicToolCallingEnv(_UpstreamBaseEnv):
    """Environment that scores agent tool-calls on Arabic instructions.

    Each rollout corresponds to one task (one arabic-agent-eval item). The
    policy receives the dialect persona + instruction + available tools;
    its tool-call emission is scored against the expected_calls.

    Reward is a scalar in [0, 1]. No shaping, no negative rewards — the
    advisory MTG philosophy.
    """

    name: str = "arabic_tool_calling"
    version: str = "0.1.0"

    def __init__(self, config: EnvConfig | None = None) -> None:
        if _BASE_AVAILABLE:
            super().__init__()  # type: ignore[misc]
        self.config = config or EnvConfig()
        self._tasks: list[Task] = []
        self._task_idx = 0

    def setup(self) -> None:
        """Populate the task buffer."""
        self._tasks = sample_tasks(
            n=self.config.n_tasks_per_rollout,
            seed=self.config.seed,
            dialect_filter=self.config.dialect_filter,
        )
        self._task_idx = 0

    def next_task(self) -> Task | None:
        if self._task_idx >= len(self._tasks):
            return None
        task = self._tasks[self._task_idx]
        self._task_idx += 1
        return task

    def build_prompt(self, task: Task) -> list[dict]:
        """Return a chat-format prompt for the task."""
        return [
            {"role": "system", "content": task.persona},
            {"role": "user", "content": task.instruction},
        ]

    def score_response(self, task: Task, response: dict) -> float:
        """Score an agent response.

        `response` should be an OpenAI-compatible chat completion dict with
        a `tool_calls` field. Missing or malformed tool_calls yield 0.0.
        """
        tool_calls = _extract_tool_calls(response)
        if not tool_calls:
            return 0.0

        # Score the first expected vs first actual call. Multi-turn items
        # aggregate across additional calls.
        expected_first = task.item.expected_calls[0]
        actual_first = tool_calls[0]
        actual_name = actual_first.get("function") or actual_first.get("name")
        actual_args = actual_first.get("arguments", {}) or {}

        primary = score_turn(
            expected_function=expected_first.function,
            actual_function=actual_name,
            expected_args=expected_first.arguments,
            actual_args=actual_args,
            is_dialect_category=task.is_dialect_category,
        )
        primary_reward = reward_from_score(primary, self.config.reward_weights)

        if len(task.item.expected_calls) <= 1 or len(tool_calls) <= 1:
            return primary_reward

        # Multi-turn — average reward across matched subsequent calls
        extras: list[float] = []
        for exp_call, act_call in zip(task.item.expected_calls[1:], tool_calls[1:]):
            act_name = act_call.get("function") or act_call.get("name")
            act_args = act_call.get("arguments", {}) or {}
            s = score_turn(
                expected_function=exp_call.function,
                actual_function=act_name,
                expected_args=exp_call.arguments,
                actual_args=act_args,
                is_dialect_category=task.is_dialect_category,
            )
            extras.append(reward_from_score(s, self.config.reward_weights))

        # Primary call weight 0.6, extras average weight 0.4
        return 0.6 * primary_reward + 0.4 * (sum(extras) / len(extras))


def _extract_tool_calls(response: dict) -> list[dict]:
    """Pull tool calls from an OpenAI-style chat completion response."""
    if not isinstance(response, dict):
        return []
    # OpenAI chat completion shape
    choices = response.get("choices") or []
    if choices:
        msg = choices[0].get("message", {}) or {}
        tcs = msg.get("tool_calls") or []
        calls = []
        for tc in tcs:
            f = tc.get("function", {}) or {}
            args = f.get("arguments", {})
            if isinstance(args, str):
                try:
                    import json as _json
                    args = _json.loads(args)
                except Exception:
                    args = {}
            calls.append({"function": f.get("name", ""), "arguments": args})
        return calls
    # Pre-parsed shape (matches arabic-agent-eval evaluator)
    if "calls" in response:
        return list(response["calls"])
    if "tool_calls" in response:
        return list(response["tool_calls"])
    return []
