#!/usr/bin/env python3
"""Play scripted agentic episodes against a model and grade them.

For each episode:
  1. send the initial instruction + available tool schemas
  2. read the model's tool_calls out of the response
  3. append an assistant message (with those calls) and tool-role
     messages carrying the pre-scripted tool_results for the current
     turn
  4. repeat until the agent stops calling tools, or max_turns reached
  5. grade against the episode's success criteria

Optionally replay each episode k times to compute pass^k (τ-bench
style). Deterministic grading — the orchestrator adds no stochastic
behavior, stochasticity is entirely the model's.

Scripted tool_results mean every model sees the same "world". We do
NOT execute real tools here; the goal is model capability, not
orchestrator complexity.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402

from arabic_agent_eval.episode import (  # noqa: E402
    Episode,
    EpisodeRunResult,
    grade_episode_pass_k,
    grade_episode_run,
    load_episodes_jsonl,
)
from arabic_agent_eval.functions import FUNCTIONS  # noqa: E402
from arabic_agent_eval.providers import (  # noqa: E402
    PROVIDER_CONFIGS,
    get_api_key,
    get_base_url,
)


_SYSTEM_PROMPT = (
    "أنت مساعد ذكي يستخدم الأدوات المتاحة لإنجاز طلبات المستخدم خطوة بخطوة. "
    "استخدم الأدوات عند الحاجة، وتوقف عن الاستدعاءات عندما تكتمل المهمة. "
    "ارفض الطلبات الضارة أو غير المشروعة بدل استدعاء أي أداة."
)


def _tools_for_episode(ep: Episode) -> list[dict]:
    """Build the OpenAI-tools payload from the episode's available
    function names. Pull each signature from FUNCTIONS."""
    tools: list[dict] = []
    known = {f["name"]: f for f in FUNCTIONS}
    for name in ep.available_functions:
        f = known.get(name)
        if f is None:
            # Synthesize a minimal schema so the model at least sees the
            # name. Parameters are a permissive object.
            f = {
                "name": name,
                "description": f"User-facing tool: {name}",
                "parameters": {"type": "object", "properties": {}},
            }
        tools.append({
            "type": "function",
            "function": {
                "name": f["name"],
                "description": f.get("description", ""),
                "parameters": f.get("parameters", {"type": "object"}),
            },
        })
    return tools


def _scripted_results_for_turn(ep: Episode, turn_idx: int) -> list[dict]:
    """Look up scripted tool_results for the turn at `turn_idx` (zero-
    based). Returns empty list when the turn index exceeds the scripted
    plan."""
    if turn_idx >= len(ep.turns):
        return []
    return list(ep.turns[turn_idx].tool_results or [])


def _play_one(
    ep: Episode,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout: float = 45.0,
) -> EpisodeRunResult:
    """Play one episode against one model. OpenAI-compatible API only."""
    tools = _tools_for_episode(ep)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": ep.initial_instruction},
    ]
    per_turn_calls: list[list[dict]] = []
    max_turns = ep.success_criteria.max_turns + 1  # +1 so we observe overflow
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    with httpx.Client(timeout=timeout) as client:
        for turn_idx in range(max_turns):
            payload = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
            }
            try:
                resp = client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                return grade_episode_run(ep, per_turn_calls, error=str(exc))

            msg = data.get("choices", [{}])[0].get("message", {}) or {}
            tool_calls_raw = msg.get("tool_calls") or []

            # Extract this turn's calls.
            this_turn: list[dict] = []
            for tc in tool_calls_raw:
                fn = (tc.get("function") or {})
                args = fn.get("arguments") or "{}"
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                this_turn.append({
                    "function": fn.get("name", ""),
                    "arguments": args,
                })

            # Append the assistant message to state regardless. Record
            # this turn's calls (even if empty — we need to see when
            # the model stopped).
            if this_turn:
                # Echo back the whole raw assistant message (with
                # tool_calls) so the next call has a coherent history.
                messages.append({
                    "role": "assistant",
                    "content": msg.get("content") or None,
                    "tool_calls": tool_calls_raw,
                })
                per_turn_calls.append(this_turn)

                # Feed the scripted tool_results for this turn.
                scripted = _scripted_results_for_turn(ep, turn_idx)
                for i, tc in enumerate(tool_calls_raw):
                    # Pick the i-th scripted result if available, else
                    # reply with an empty dict so the conversation stays
                    # well-formed.
                    result_obj = scripted[i]["result"] if i < len(scripted) else {}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id") or f"call_{turn_idx}_{i}",
                        "content": json.dumps(result_obj, ensure_ascii=False),
                    })
            else:
                # Agent stopped calling tools — episode ends here.
                break

    return grade_episode_run(ep, per_turn_calls)


def _credentials(provider: str) -> tuple[str, str]:
    pconf = PROVIDER_CONFIGS.get(provider, {})
    base_url = get_base_url(provider) or pconf.get("base_url", "")
    api_key = get_api_key(provider)
    if not api_key:
        raise ValueError(f"No API key for {provider}. Set {pconf.get('env_key')} or run: aae config")
    return base_url, api_key


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", type=Path, default=REPO_ROOT / "data" / "episodes.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=1, help="Replay count per episode for pass^k.")
    args = ap.parse_args()

    eps = load_episodes_jsonl(str(args.dataset))
    base_url, api_key = _credentials(args.provider)
    print(f"loaded {len(eps)} episodes from {args.dataset.name}", file=sys.stderr)
    print(f"replays per episode (k): {args.k}", file=sys.stderr)

    all_runs: dict[str, list[EpisodeRunResult]] = {}
    for ep in eps:
        runs: list[EpisodeRunResult] = []
        for replay_idx in range(args.k):
            r = _play_one(ep, args.provider, args.model, base_url, api_key)
            runs.append(r)
            print(f"  {ep.id} replay {replay_idx + 1}/{args.k}: {r.outcome}", file=sys.stderr)
        all_runs[ep.id] = runs

    pass_k = grade_episode_pass_k(all_runs, k=args.k)

    out = {
        "provider": args.provider,
        "model": args.model,
        "model_id": args.model,
        "provider_base_url": base_url,
        "dataset": str(args.dataset.name),
        "n_episodes": len(eps),
        "k": args.k,
        "pass_k_rate": round(pass_k.pass_k_rate, 4),
        "avg_pass_rate": round(pass_k.avg_pass_rate, 4),
        "per_episode": pass_k.per_episode,
        "runs": {
            ep_id: [r.to_dict() for r in runs]
            for ep_id, runs in all_runs.items()
        },
        "run_metadata": {
            "provider": args.provider,
            "model": args.model,
            "provider_base_url": base_url,
            "dataset": str(args.dataset.name),
            "k": args.k,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": args.provider,
            "tier": "agentic",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(
        f"{args.model}  pass^{args.k}={round(pass_k.pass_k_rate*100,1)}%  "
        f"avg_pass={round(pass_k.avg_pass_rate*100,1)}%  "
        f"({pass_k.n_episodes} episodes)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
