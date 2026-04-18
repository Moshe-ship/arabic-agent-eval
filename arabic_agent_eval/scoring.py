"""Scoring system for Arabic Agent Eval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arabic_agent_eval.dataset import CATEGORIES


@dataclass
class Score:
    """Score for a single evaluation item."""

    item_id: str
    function_selection: float = 0.0  # 0 or 1: did model pick the right function?
    argument_accuracy: float = 0.0   # 0-1: are arguments correct?
    arabic_preservation: float = 0.0  # 0 or 1: Arabic values preserved, not transliterated?
    dialect_understanding: float = 0.0  # 0 or 1: understood the dialect? (dialect category only)
    error_handling: float = 0.0  # 0 or 1: handled error correctly? (error category only)
    category: str = ""

    @property
    def total(self) -> float:
        """Weighted total score for this item (0-1)."""
        if self.category == "dialect_handling":
            return (
                self.function_selection * 0.3
                + self.argument_accuracy * 0.2
                + self.arabic_preservation * 0.2
                + self.dialect_understanding * 0.3
            )
        if self.category == "error_recovery":
            return (
                self.function_selection * 0.3
                + self.argument_accuracy * 0.2
                + self.arabic_preservation * 0.2
                + self.error_handling * 0.3
            )
        return (
            self.function_selection * 0.4
            + self.argument_accuracy * 0.35
            + self.arabic_preservation * 0.25
        )

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "function_selection": self.function_selection,
            "argument_accuracy": self.argument_accuracy,
            "arabic_preservation": self.arabic_preservation,
            "dialect_understanding": self.dialect_understanding,
            "error_handling": self.error_handling,
            "category": self.category,
            "total": round(self.total, 4),
        }


@dataclass
class CategoryScore:
    """Aggregated score for a category."""

    category: str
    scores: list[Score] = field(default_factory=list)

    @property
    def name_ar(self) -> str:
        return CATEGORIES.get(self.category, {}).get("name_ar", self.category)

    @property
    def weight(self) -> float:
        return CATEGORIES.get(self.category, {}).get("weight", 0.0)

    @property
    def avg_function_selection(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.function_selection for s in self.scores) / len(self.scores)

    @property
    def avg_argument_accuracy(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.argument_accuracy for s in self.scores) / len(self.scores)

    @property
    def avg_arabic_preservation(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.arabic_preservation for s in self.scores) / len(self.scores)

    @property
    def avg_total(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.total for s in self.scores) / len(self.scores)

    @property
    def count(self) -> int:
        return len(self.scores)

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "name_ar": self.name_ar,
            "count": self.count,
            "avg_function_selection": round(self.avg_function_selection, 4),
            "avg_argument_accuracy": round(self.avg_argument_accuracy, 4),
            "avg_arabic_preservation": round(self.avg_arabic_preservation, 4),
            "avg_total": round(self.avg_total, 4),
            "weight": self.weight,
        }


def compute_overall_score(category_scores: list[CategoryScore]) -> float:
    """Compute weighted overall score from category scores."""
    total_weight = sum(cs.weight for cs in category_scores if cs.count > 0)
    if total_weight == 0:
        return 0.0
    weighted_sum = sum(cs.avg_total * cs.weight for cs in category_scores if cs.count > 0)
    return weighted_sum / total_weight


def grade(score: float) -> str:
    """Convert 0-1 score to letter grade."""
    pct = score * 100
    if pct >= 90:
        return "A"
    if pct >= 80:
        return "B"
    if pct >= 70:
        return "C"
    if pct >= 60:
        return "D"
    return "F"


def score_function_call(
    expected_function: str,
    actual_function: str | None,
    expected_args: dict[str, Any],
    actual_args: dict[str, Any] | None,
) -> tuple[float, float, float]:
    """Score a single function call.

    Returns (function_selection, argument_accuracy, arabic_preservation).
    """
    if actual_function is None:
        return 0.0, 0.0, 0.0

    # Function selection
    func_score = 1.0 if actual_function == expected_function else 0.0

    if actual_args is None:
        return func_score, 0.0, 0.0

    arabic_matches = 0
    arabic_total = 0

    if not expected_args:
        arg_score = 1.0 if not actual_args else 0.5
    else:
        matches = 0.0
        total = len(expected_args)

        for key, expected_val in expected_args.items():
            actual_val = actual_args.get(key)
            if actual_val is None:
                continue

            if expected_val == "*":
                matches += 1
                continue

            ev = str(expected_val).strip()
            av = str(actual_val).strip()

            if ev == av:
                matches += 1
            elif normalize_arabic(ev) == normalize_arabic(av):
                matches += 0.9
            elif ev.lower() == av.lower():
                matches += 0.8

            if _contains_arabic(ev):
                arabic_total += 1
                if _contains_arabic(av):
                    arabic_matches += 1

        arg_score = matches / total if total > 0 else 1.0

    arabic_score = arabic_matches / arabic_total if arabic_total > 0 else 1.0
    return func_score, arg_score, arabic_score


def _contains_arabic(text: str) -> bool:
    """Check if text contains Arabic characters."""
    return any("\u0600" <= c <= "\u06FF" or "\u0750" <= c <= "\u077F" for c in text)


_TATWEEL = "\u0640"
_ALEF_VARIANTS = str.maketrans({"\u0622": "\u0627", "\u0623": "\u0627", "\u0625": "\u0627"})
_YA_VARIANT = str.maketrans({"\u0649": "\u064A"})
_TA_MARBUTA = str.maketrans({"\u0629": "\u0647"})


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for comparison.

    Strips tatweel, unifies alef variants (آأإ → ا), ya (ى → ي),
    and ta-marbuta (ة → ه). Non-Arabic text passes through lowercased.
    Defined in docs/grading.md and used by score_function_call.
    """
    if not _contains_arabic(text):
        return text.lower().strip()
    out = text.replace(_TATWEEL, "")
    out = out.translate(_ALEF_VARIANTS)
    out = out.translate(_YA_VARIANT)
    out = out.translate(_TA_MARBUTA)
    return out.strip()
