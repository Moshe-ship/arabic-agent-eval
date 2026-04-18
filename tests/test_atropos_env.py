"""Smoke tests for the arabic_tool_calling Atropos environment.

Exercises the env without requiring the upstream atropos package — uses the
late-binding import shim in env.py.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from atropos.arabic_tool_calling import (  # noqa: E402
    ArabicToolCallingEnv,
    EnvConfig,
    RewardWeights,
    sample_tasks,
)


def test_sample_tasks_returns_requested_count():
    tasks = sample_tasks(n=5, seed=42)
    assert len(tasks) == 5
    for task in tasks:
        assert task.item.expected_calls
        assert task.tools
        assert task.persona


def test_sample_tasks_dialect_filter():
    tasks = sample_tasks(n=50, seed=42, dialect_filter="gulf")
    assert all(t.dialect == "gulf" for t in tasks)


def test_env_setup_and_next_task():
    env = ArabicToolCallingEnv(EnvConfig(n_tasks_per_rollout=3, seed=42))
    env.setup()
    task = env.next_task()
    assert task is not None
    prompt = env.build_prompt(task)
    assert prompt[0]["role"] == "system"
    assert prompt[1]["role"] == "user"
    assert prompt[1]["content"] == task.instruction


def test_env_scores_perfect_response():
    env = ArabicToolCallingEnv(EnvConfig(n_tasks_per_rollout=1, seed=42))
    env.setup()
    task = env.next_task()
    expected = task.item.expected_calls[0]
    response = {
        "calls": [
            {"function": expected.function, "arguments": expected.arguments}
        ]
    }
    reward = env.score_response(task, response)
    assert reward == 1.0


def test_env_scores_wrong_tool_as_partial():
    env = ArabicToolCallingEnv(EnvConfig(n_tasks_per_rollout=1, seed=42))
    env.setup()
    task = env.next_task()
    response = {"calls": [{"function": "no_such_tool", "arguments": {}}]}
    reward = env.score_response(task, response)
    assert 0.0 <= reward < 1.0


def test_env_scores_empty_response_as_zero():
    env = ArabicToolCallingEnv(EnvConfig(n_tasks_per_rollout=1, seed=42))
    env.setup()
    task = env.next_task()
    reward = env.score_response(task, {})
    assert reward == 0.0


def test_explicit_reward_weights_override():
    env = ArabicToolCallingEnv(EnvConfig(
        n_tasks_per_rollout=1,
        seed=42,
        reward_weights=RewardWeights(1.0, 0.0, 0.0, 0.0),  # only function_selection
    ))
    env.setup()
    task = env.next_task()
    expected = task.item.expected_calls[0]
    response = {
        "calls": [{"function": expected.function, "arguments": {"wrong": "stuff"}}]
    }
    reward = env.score_response(task, response)
    # Correct function, wrong args: with function-only weight, reward == 1.0
    assert reward == 1.0


def test_openai_chat_completion_shape_extracted():
    env = ArabicToolCallingEnv(EnvConfig(n_tasks_per_rollout=1, seed=42))
    env.setup()
    task = env.next_task()
    expected = task.item.expected_calls[0]
    import json

    openai_response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": expected.function,
                                "arguments": json.dumps(expected.arguments, ensure_ascii=False),
                            }
                        }
                    ]
                }
            }
        ]
    }
    reward = env.score_response(task, openai_response)
    assert reward == 1.0
