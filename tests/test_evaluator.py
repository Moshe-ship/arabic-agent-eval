"""Tests for evaluator module."""

from arabic_agent_eval.dataset import Dataset, EvalItem, ExpectedCall
from arabic_agent_eval.evaluator import Evaluator, BenchmarkResult


def _mock_call_fn(instruction, tools, functions):
    """Mock call function that returns the first expected tool call."""
    # Simple mock: pick the first tool and return it
    if tools:
        first_tool = tools[0]["function"]
        return {
            "calls": [{"function": first_tool["name"], "arguments": {}}],
            "raw": "",
        }
    return {"calls": [], "raw": ""}


def _perfect_call_fn(instruction, tools, functions):
    """Mock that returns perfect results for simple_001."""
    return {
        "calls": [{"function": "search_flights", "arguments": {"from_city": "الرياض", "to_city": "جدة", "date": "الخميس"}}],
        "raw": "",
    }


def _failing_call_fn(instruction, tools, functions):
    """Mock that always raises."""
    raise ConnectionError("API down")


def test_evaluator_runs():
    evaluator = Evaluator(call_fn=_mock_call_fn, provider="test", model="mock")
    dataset = Dataset()
    result = evaluator.evaluate_quick(dataset, n=6)
    assert isinstance(result, BenchmarkResult)
    assert result.total_items == 6
    assert result.provider == "test"


def test_evaluator_perfect_score_item():
    evaluator = Evaluator(call_fn=_perfect_call_fn, provider="test", model="mock")
    item = EvalItem(
        id="test_001",
        category="simple_function_calling",
        instruction="ابحث عن رحلات من الرياض إلى جدة يوم الخميس",
        dialect="msa",
        available_functions=["search_flights", "book_hotel"],
        expected_calls=[
            ExpectedCall(function="search_flights", arguments={"from_city": "الرياض", "to_city": "جدة", "date": "الخميس"}),
        ],
        difficulty="easy",
    )
    result = evaluator.evaluate_item(item)
    assert result.score.function_selection == 1.0
    assert result.score.argument_accuracy == 1.0
    assert result.score.arabic_preservation == 1.0


def test_evaluator_handles_errors():
    evaluator = Evaluator(call_fn=_failing_call_fn, provider="test", model="mock")
    dataset = Dataset()
    result = evaluator.evaluate_quick(dataset, n=3)
    assert all(r.error is not None for r in result.results)
    assert len(result.errors) == 3


def test_benchmark_result_grade():
    evaluator = Evaluator(call_fn=_perfect_call_fn, provider="test", model="mock")
    item = EvalItem(
        id="t1",
        category="simple_function_calling",
        instruction="ابحث عن رحلات من الرياض إلى جدة يوم الخميس",
        dialect="msa",
        available_functions=["search_flights"],
        expected_calls=[
            ExpectedCall(function="search_flights", arguments={"from_city": "الرياض", "to_city": "جدة", "date": "الخميس"}),
        ],
        difficulty="easy",
    )
    er = evaluator.evaluate_item(item)
    br = BenchmarkResult(provider="test", model="mock", results=[er])
    assert br.overall_grade == "A"
