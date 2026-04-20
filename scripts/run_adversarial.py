#!/usr/bin/env python3
"""Run the adversarial dataset against one provider+model.

The adversarial corpus (data/adversarial.jsonl) carries extra fields
(`adversarial_shape`, `expected_mtg_codes`) that vanilla `aae run`
doesn't preserve. This runner loads that JSONL, builds a Dataset
directly, and writes a BenchmarkResult JSON in the same shape
build_bundle.py expects — including provider_base_url, model_id, and
run_metadata stamped for the publish gate.

Usage:
  python3 scripts/run_adversarial.py \
      --provider openrouter \
      --model z-ai/glm-5.1 \
      --out runs/adversarial/glm-5.1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arabic_agent_eval.dataset import CATEGORIES, Dataset
from arabic_agent_eval.evaluator import Evaluator
from arabic_agent_eval.providers import (
    PROVIDER_CONFIGS,
    get_base_url,
    get_default_model,
    make_call_fn,
)


# Adversarial categories aren't in the default CATEGORIES (which was
# designed for the non-adversarial 51-item corpus). Register them here
# with uniform weight so compute_overall_score() can aggregate.
# Each category targets one MTG guard family; all weighted equally.
_ADVERSARIAL_CATEGORIES = {
    "script_homoglyph":      {"name_ar": "تشابه بصري",        "weight": 0.125},
    "bidi_smuggling":        {"name_ar": "تهريب BiDi",         "weight": 0.125},
    "invisible_padding":     {"name_ar": "حشو غير مرئي",       "weight": 0.125},
    "arabizi_override":      {"name_ar": "عربيزي ممنوع",       "weight": 0.125},
    "prompt_injection_arg":  {"name_ar": "حقن تعليمة",         "weight": 0.125},
    "uts39_numeric_trap":    {"name_ar": "فخ الأرقام UTS #39", "weight": 0.125},
    "canonicalization_edge": {"name_ar": "حافة التوحيد",       "weight": 0.125},
    "dialect_pressure":      {"name_ar": "ضغط لهجي",           "weight": 0.125},
}
for _k, _v in _ADVERSARIAL_CATEGORIES.items():
    CATEGORIES.setdefault(_k, _v)


DEFAULT_DATASET = REPO_ROOT / "data" / "adversarial.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    raw = [json.loads(l) for l in args.dataset.read_text(encoding="utf-8").splitlines() if l.strip()]
    ds = Dataset(items=raw)
    print(f"loaded {len(ds)} adversarial items from {args.dataset.name}", file=sys.stderr)

    call_fn = make_call_fn(args.provider, args.model)
    evaluator = Evaluator(
        call_fn=call_fn,
        provider=args.provider,
        model=args.model,
    )
    result = evaluator.evaluate(ds)

    out = result.to_dict()
    # Stamp provenance — mirrors what build_bundle.py's loader expects.
    base_url = get_base_url(args.provider) or PROVIDER_CONFIGS.get(args.provider, {}).get("base_url", "")
    out["provider_base_url"] = base_url
    out["model_id"] = args.model
    out["run_metadata"] = {
        "provider": args.provider,
        "model": args.model,
        "model_id": args.model,
        "provider_base_url": base_url,
        "dataset": str(args.dataset.name),
        "dataset_size": len(ds),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": args.provider,
        "adversarial": True,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False))
    print(
        f"model: {args.model}  score: {round(out['overall_score']*100,1)}%  "
        f"grade: {out['overall_grade']}  items: {out['total_items']}  "
        f"errors: {sum(1 for r in out['results'] if r.get('error'))}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
