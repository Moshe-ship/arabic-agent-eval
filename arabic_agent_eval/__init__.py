"""Arabic Agent Eval - The first Arabic function-calling benchmark."""

__version__ = "0.1.0"

from arabic_agent_eval.dataset import Dataset, EvalItem
from arabic_agent_eval.evaluator import Evaluator, EvalResult
from arabic_agent_eval.scoring import Score, CategoryScore
from arabic_agent_eval.functions import FUNCTIONS

__all__ = [
    "Dataset",
    "EvalItem",
    "Evaluator",
    "EvalResult",
    "Score",
    "CategoryScore",
    "FUNCTIONS",
]
