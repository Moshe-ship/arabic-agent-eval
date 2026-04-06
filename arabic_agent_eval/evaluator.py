"""Core evaluation engine."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from arabic_agent_eval.dataset import Dataset, EvalItem, CATEGORIES
from arabic_agent_eval.functions import FUNCTIONS, to_openai_tools, get_function_by_name
from arabic_agent_eval.scoring import (
    Score,
    CategoryScore,
    compute_overall_score,
    score_function_call,
    grade,
)


@dataclass
class EvalResult:
    """Result from evaluating a single item."""

    item: EvalItem
    score: Score
    actual_calls: list[dict] = field(default_factory=list)
    raw_response: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "item_id": self.item.id,
            "category": self.item.category,
            "dialect": self.item.dialect,
            "instruction": self.item.instruction,
            "score": self.score.to_dict(),
            "actual_calls": self.actual_calls,
            "error": self.error,
        }


@dataclass
class BenchmarkResult:
    """Full benchmark result for a provider."""

    provider: str
    model: str
    results: list[EvalResult] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        cat_scores = self.category_scores
        return compute_overall_score(cat_scores)

    @property
    def overall_grade(self) -> str:
        return grade(self.overall_score)

    @property
    def category_scores(self) -> list[CategoryScore]:
        by_cat: dict[str, list[Score]] = {}
        for r in self.results:
            cat = r.item.category
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append(r.score)

        return [
            CategoryScore(category=cat, scores=scores)
            for cat, scores in by_cat.items()
        ]

    @property
    def total_items(self) -> int:
        return len(self.results)

    @property
    def errors(self) -> list[EvalResult]:
        return [r for r in self.results if r.error]

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "overall_score": round(self.overall_score, 4),
            "overall_grade": self.overall_grade,
            "total_items": self.total_items,
            "errors": len(self.errors),
            "categories": [cs.to_dict() for cs in self.category_scores],
            "results": [r.to_dict() for r in self.results],
        }


class Evaluator:
    """Evaluate LLM function-calling capabilities in Arabic."""

    def __init__(
        self,
        call_fn: Callable[[str, list[dict], list[dict]], dict],
        provider: str = "unknown",
        model: str = "unknown",
    ):
        """Initialize evaluator.

        Args:
            call_fn: Function that takes (instruction, tools, functions) and returns
                     {"calls": [{"function": name, "arguments": {...}}, ...], "raw": str}
            provider: Provider name for reporting.
            model: Model name for reporting.
        """
        self.call_fn = call_fn
        self.provider = provider
        self.model = model

    def evaluate_item(self, item: EvalItem) -> EvalResult:
        """Evaluate a single item."""
        # Build available tools for this item
        available = [f for f in FUNCTIONS if f["name"] in item.available_functions]
        tools = to_openai_tools(available)

        try:
            response = self.call_fn(item.instruction, tools, available)
        except Exception as e:
            return EvalResult(
                item=item,
                score=Score(item_id=item.id, category=item.category),
                error=f"{type(e).__name__}: {e}",
            )

        actual_calls = response.get("calls", [])
        raw = response.get("raw", "")

        # Score the response
        score = self._score_item(item, actual_calls)

        return EvalResult(
            item=item,
            score=score,
            actual_calls=actual_calls,
            raw_response=raw,
        )

    def evaluate(self, dataset: Dataset | None = None) -> BenchmarkResult:
        """Run full evaluation."""
        ds = dataset or Dataset()
        results = [self.evaluate_item(item) for item in ds]
        return BenchmarkResult(
            provider=self.provider,
            model=self.model,
            results=results,
        )

    def evaluate_quick(self, dataset: Dataset | None = None, n: int = 12) -> BenchmarkResult:
        """Run quick evaluation with subset."""
        ds = dataset or Dataset()
        items = ds.subset(n)
        results = [self.evaluate_item(item) for item in items]
        return BenchmarkResult(
            provider=self.provider,
            model=self.model,
            results=results,
        )

    def _score_item(self, item: EvalItem, actual_calls: list[dict]) -> Score:
        """Score an item based on expected vs actual calls."""
        score = Score(item_id=item.id, category=item.category)

        if not item.expected_calls:
            return score

        if not actual_calls:
            return score

        # Score the first expected call against the first actual call
        expected = item.expected_calls[0]
        actual = actual_calls[0] if actual_calls else {}

        actual_fn = actual.get("function", actual.get("name"))
        actual_args = actual.get("arguments", actual.get("args", {}))
        if isinstance(actual_args, str):
            try:
                actual_args = json.loads(actual_args)
            except json.JSONDecodeError:
                actual_args = {}

        func_score, arg_score, arabic_score = score_function_call(
            expected.function,
            actual_fn,
            expected.arguments,
            actual_args,
        )

        score.function_selection = func_score
        score.argument_accuracy = arg_score
        score.arabic_preservation = arabic_score

        # Multi-step: check additional calls
        if len(item.expected_calls) > 1 and len(actual_calls) > 1:
            additional_matches = 0
            for i, exp_call in enumerate(item.expected_calls[1:], 1):
                if i < len(actual_calls):
                    act = actual_calls[i]
                    act_fn = act.get("function", act.get("name"))
                    if act_fn == exp_call.function:
                        additional_matches += 1
            # Boost arg accuracy based on multi-step correctness
            multi_ratio = additional_matches / (len(item.expected_calls) - 1)
            score.argument_accuracy = (score.argument_accuracy + multi_ratio) / 2

        # Dialect understanding (for dialect category)
        if item.category == "dialect_handling":
            # If function was selected correctly, the model understood the dialect
            score.dialect_understanding = func_score

        # Error handling (for error category)
        if item.category == "error_recovery":
            score.error_handling = func_score

        return score
