"""Scoring for arabic_tool_calling environment.

Reuses arabic-agent-eval's 4-axis canonical structured-call grader. Reward is
an equal-weighted mean across the axes; weights are configurable at instantiation
time for hyperparameter sweeps.

Weights are equal-weighted (0.25 × 4) by default — document this choice in the
environment README. Teams training on different objectives should override.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from arabic_agent_eval.scoring import Score, score_function_call


DEFAULT_WEIGHTS: dict[str, float] = {
    "function_selection": 0.25,
    "argument_accuracy": 0.25,
    "arabic_preservation": 0.25,
    "dialect_understanding": 0.25,
}


@dataclass(frozen=True)
class RewardWeights:
    function_selection: float = 0.25
    argument_accuracy: float = 0.25
    arabic_preservation: float = 0.25
    dialect_understanding: float = 0.25

    def as_dict(self) -> dict[str, float]:
        return {
            "function_selection": self.function_selection,
            "argument_accuracy": self.argument_accuracy,
            "arabic_preservation": self.arabic_preservation,
            "dialect_understanding": self.dialect_understanding,
        }

    def normalized(self) -> "RewardWeights":
        total = self.as_dict()
        s = sum(total.values())
        if s == 0:
            return self
        return RewardWeights(
            function_selection=self.function_selection / s,
            argument_accuracy=self.argument_accuracy / s,
            arabic_preservation=self.arabic_preservation / s,
            dialect_understanding=self.dialect_understanding / s,
        )


def score_turn(
    expected_function: str,
    actual_function: str | None,
    expected_args: dict[str, Any],
    actual_args: dict[str, Any] | None,
    is_dialect_category: bool = False,
) -> Score:
    """Score a single turn and produce a Score dataclass."""
    func, arg, arabic = score_function_call(
        expected_function, actual_function, expected_args, actual_args
    )
    score = Score(
        item_id="rollout",
        category="dialect_handling" if is_dialect_category else "simple_function_calling",
        function_selection=func,
        argument_accuracy=arg,
        arabic_preservation=arabic,
    )
    if is_dialect_category:
        score.dialect_understanding = func
    return score


def reward_from_score(
    score: Score,
    weights: RewardWeights | None = None,
) -> float:
    """Produce a scalar reward in [0, 1] from a Score.

    By default uses the benchmark's category-aware weights via `score.total`
    (see arabic_agent_eval.scoring.Score.total) so training objective matches
    benchmark scoring. Pass explicit `weights` for equal- or custom-weighting.
    """
    if weights is None:
        return max(0.0, min(1.0, score.total))
    w = weights.normalized()
    reward = (
        w.function_selection * score.function_selection
        + w.argument_accuracy * score.argument_accuracy
        + w.arabic_preservation * score.arabic_preservation
        + w.dialect_understanding * score.dialect_understanding
    )
    return max(0.0, min(1.0, reward))
