"""Agentic-tier evaluation: multi-turn episodes.

An `Episode` is a scripted dialogue where the agent receives an
initial instruction, calls tools, receives mocked tool results, and
continues until the episode either succeeds, fails, or hits the turn
cap. The orchestrator pre-scripts the tool results so every model
sees the same "world" — otherwise models that happen to get favorable
responses from a real API would look better than they are.

Scope is deliberately small for the MVP:
- single-provider run (multi-provider comparison comes from running
  each one separately, same as the non-agentic corpus)
- scripted tool results (no live API tool execution in this tier)
- success criteria are declarative: required_tool_calls + max_turns
- pass^k runs k replays per episode and reports the fraction where
  all k succeeded; see grade_episode_pass_k

Future: live tool execution, stochastic tool_result branching, and
reward-model grading. Skipped for now — the goal is to measure
model capability, not orchestrator complexity.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional


@dataclass
class EpisodeTurn:
    """One scripted turn in an episode.

    `expected_tool_calls` is the set of function calls the agent is
    expected to issue on this turn. Order doesn't matter; we check set
    membership on `function` name. `tool_results` is what the
    orchestrator feeds back as the tool output — the agent then has
    what it needs to decide the next turn.
    """

    turn_id: int
    expected_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    # Free-form note: what the agent is supposed to be doing on this turn.
    # Useful for grader diagnostics, not load-bearing.
    intent: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeTurn":
        return cls(
            turn_id=int(data.get("turn_id", 0)),
            expected_tool_calls=list(data.get("expected_tool_calls") or []),
            tool_results=list(data.get("tool_results") or []),
            intent=str(data.get("intent") or ""),
        )


@dataclass
class EpisodeSuccessCriteria:
    """Declarative success rules for an episode.

    - `required_tool_calls`: function names that MUST appear across the
      full trace. Order-independent.
    - `forbidden_tool_calls`: function names that MUST NOT appear.
      Useful for abstention episodes.
    - `max_turns`: hard cap. If the agent exceeds this, the episode
      fails regardless of other state.
    - `should_abstain`: if True, the "winning" behavior is to refuse
      rather than call tools. An empty tool-call trace counts as pass.
    """

    required_tool_calls: tuple[str, ...] = ()
    forbidden_tool_calls: tuple[str, ...] = ()
    max_turns: int = 5
    should_abstain: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeSuccessCriteria":
        return cls(
            required_tool_calls=tuple(data.get("required_tool_calls") or ()),
            forbidden_tool_calls=tuple(data.get("forbidden_tool_calls") or ()),
            max_turns=int(data.get("max_turns", 5)),
            should_abstain=bool(data.get("should_abstain", False)),
        )


@dataclass
class Episode:
    """One scripted agentic episode."""

    id: str
    category: str
    dialect: str
    initial_instruction: str
    available_functions: list[str]
    turns: list[EpisodeTurn]
    success_criteria: EpisodeSuccessCriteria

    @classmethod
    def from_dict(cls, data: dict) -> "Episode":
        return cls(
            id=str(data["id"]),
            category=str(data.get("category", "")),
            dialect=str(data.get("dialect", "msa")),
            initial_instruction=str(data["initial_instruction"]),
            available_functions=list(data.get("available_functions") or []),
            turns=[EpisodeTurn.from_dict(t) for t in data.get("turns") or []],
            success_criteria=EpisodeSuccessCriteria.from_dict(
                data.get("success_criteria") or {}
            ),
        )


@dataclass
class EpisodeRunResult:
    """One run of one episode.

    `per_turn_calls` records what the agent actually did on each turn,
    in order. `outcome` is pass/fail/abstain/error.
    """

    episode_id: str
    outcome: str  # "pass" | "fail" | "abstain" | "error"
    per_turn_calls: list[list[dict[str, Any]]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def grade_episode_run(
    episode: Episode,
    per_turn_calls: list[list[dict[str, Any]]],
    error: Optional[str] = None,
) -> EpisodeRunResult:
    """Apply an episode's success criteria to one recorded run."""
    if error:
        return EpisodeRunResult(
            episode_id=episode.id,
            outcome="error",
            per_turn_calls=per_turn_calls,
            error=error,
        )

    sc = episode.success_criteria
    reasons: list[str] = []

    # Flatten actual tool calls.
    actual: list[str] = []
    for turn in per_turn_calls:
        for c in turn:
            name = c.get("function") or c.get("name") or ""
            if name:
                actual.append(name)

    # Abstention path — agent declined to call tools at all.
    if sc.should_abstain:
        if not actual:
            return EpisodeRunResult(
                episode_id=episode.id,
                outcome="abstain",
                per_turn_calls=per_turn_calls,
                reasons=["declined to call any tool — abstention expected"],
            )
        reasons.append(f"abstention expected but called {actual}")
        return EpisodeRunResult(
            episode_id=episode.id,
            outcome="fail",
            per_turn_calls=per_turn_calls,
            reasons=reasons,
        )

    actual_set = set(actual)
    missing = set(sc.required_tool_calls) - actual_set
    if missing:
        reasons.append(f"missing required calls: {sorted(missing)}")
    forbidden_hit = set(sc.forbidden_tool_calls) & actual_set
    if forbidden_hit:
        reasons.append(f"forbidden calls made: {sorted(forbidden_hit)}")
    if len(per_turn_calls) > sc.max_turns:
        reasons.append(
            f"exceeded max_turns ({len(per_turn_calls)} > {sc.max_turns})"
        )

    return EpisodeRunResult(
        episode_id=episode.id,
        outcome="pass" if not reasons else "fail",
        per_turn_calls=per_turn_calls,
        reasons=reasons,
    )


@dataclass
class EpisodePassKResult:
    """pass^k summary across many runs of many episodes."""

    k: int
    n_episodes: int
    pass_k_rate: float            # fraction of episodes where all k runs passed
    avg_pass_rate: float          # fraction of individual runs that passed
    per_episode: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def grade_episode_pass_k(
    episode_runs: dict[str, list[EpisodeRunResult]],
    k: int,
) -> EpisodePassKResult:
    """Compute pass^k: fraction of episodes where ALL k runs passed.

    `episode_runs` maps episode_id → list of k EpisodeRunResults.
    Any episode with fewer than k runs is skipped (caller's contract
    is to supply exactly k runs per episode).
    """
    per_ep: dict[str, dict[str, Any]] = {}
    all_passed = 0
    total_runs = 0
    total_individual_passes = 0

    for ep_id, runs in episode_runs.items():
        if len(runs) < k:
            continue
        trimmed = runs[:k]
        outcomes = [r.outcome for r in trimmed]
        passed = sum(1 for o in outcomes if o in ("pass", "abstain"))
        total_runs += len(trimmed)
        total_individual_passes += passed
        all_pass = passed == k
        if all_pass:
            all_passed += 1
        per_ep[ep_id] = {
            "outcomes": outcomes,
            "passes": passed,
            "all_passed": all_pass,
        }

    n_ep = len(per_ep)
    return EpisodePassKResult(
        k=k,
        n_episodes=n_ep,
        pass_k_rate=(all_passed / n_ep) if n_ep else 0.0,
        avg_pass_rate=(total_individual_passes / total_runs) if total_runs else 0.0,
        per_episode=per_ep,
    )


def load_episodes_jsonl(path: str) -> list[Episode]:
    """Load one Episode per line from a JSONL file."""
    out: list[Episode] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(Episode.from_dict(json.loads(line)))
    return out


__all__ = [
    "Episode",
    "EpisodeTurn",
    "EpisodeSuccessCriteria",
    "EpisodeRunResult",
    "EpisodePassKResult",
    "grade_episode_run",
    "grade_episode_pass_k",
    "load_episodes_jsonl",
]
