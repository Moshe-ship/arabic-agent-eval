"""Task sampling and persona generation for arabic_tool_calling env.

Draws items from arabic_agent_eval.Dataset and produces rollouts grouped by
dialect. Supports single-turn (one expected call) and multi-turn variants.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

from arabic_agent_eval.dataset import Dataset, DIALECTS, EvalItem
from arabic_agent_eval.functions import FUNCTIONS, to_openai_tools


# Dialect personas — system-prompt fragments used when the env wants to push
# the rollout policy to behave consistently with a specific register.
DIALECT_PERSONAS: dict[str, str] = {
    "msa": "أنت مساعد يتحدث بالعربية الفصحى الواضحة. استخدم الأدوات المناسبة واحفظ النص العربي كما هو.",
    "gulf": "أنت مساعد يفهم اللهجة الخليجية (ابي، ابغى، الحين، بكرا). استخدم الأدوات المناسبة.",
    "egy": "أنت مساعد يفهم اللهجة المصرية (عايز، ايه، دلوقتي). استخدم الأدوات المناسبة.",
    "lev": "أنت مساعد يفهم اللهجة الشامية (بدي، هلأ، قديش، شو). استخدم الأدوات المناسبة.",
    "maghrebi": "أنت مساعد يفهم اللهجة المغاربية (بغيت، كيفاش، واش، شحال). استخدم الأدوات المناسبة.",
}


@dataclass
class Task:
    """One Atropos task drawn from an arabic-agent-eval item."""

    item: EvalItem
    tools: list[dict]
    persona: str
    is_dialect_category: bool

    @property
    def instruction(self) -> str:
        return self.item.instruction

    @property
    def dialect(self) -> str:
        return self.item.dialect


def build_tool_registry(item: EvalItem) -> list[dict]:
    """Return the OpenAI-style tool list available to the model for this item."""
    available = [f for f in FUNCTIONS if f["name"] in item.available_functions]
    return to_openai_tools(available)


def build_task(item: EvalItem) -> Task:
    persona = DIALECT_PERSONAS.get(item.dialect, DIALECT_PERSONAS["msa"])
    return Task(
        item=item,
        tools=build_tool_registry(item),
        persona=persona,
        is_dialect_category=(item.category == "dialect_handling"),
    )


def sample_tasks(
    n: int,
    seed: int | None = None,
    dialect_filter: str | None = None,
    dataset: Dataset | None = None,
) -> list[Task]:
    """Sample n tasks uniformly (or filtered by dialect)."""
    ds = dataset or Dataset()
    items: Iterable[EvalItem] = ds
    if dialect_filter:
        items = ds.by_dialect(dialect_filter)
    items = list(items)
    if not items:
        return []
    rng = random.Random(seed)
    if n >= len(items):
        sampled = list(items)
    else:
        sampled = rng.sample(items, n)
    return [build_task(item) for item in sampled]


def all_dialects() -> list[str]:
    return list(DIALECTS)
