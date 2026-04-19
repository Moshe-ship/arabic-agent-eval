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


def test_multi_step_grades_every_call_structurally():
    """Regression: prior implementation only scored the first expected call
    fully; subsequent calls only contributed a function-name match boost."""

    def multi_step_perfect(instruction, tools, functions):
        return {
            "calls": [
                {"function": "search_restaurants", "arguments": {"city": "الرياض"}},
                {"function": "book_table", "arguments": {"restaurant": "anywhere", "guests": 2}},
                {"function": "send_message", "arguments": {"recipient": "سارة", "platform": "whatsapp"}},
            ],
            "raw": "",
        }

    item = EvalItem(
        id="multi_test",
        category="multi_step",
        instruction="reservation + notify",
        dialect="msa",
        available_functions=["search_restaurants", "book_table", "send_message"],
        expected_calls=[
            ExpectedCall(function="search_restaurants", arguments={"city": "الرياض"}),
            ExpectedCall(function="book_table", arguments={"restaurant": "*", "guests": 2}),
            ExpectedCall(function="send_message", arguments={"recipient": "سارة", "platform": "whatsapp"}),
        ],
        difficulty="hard",
    )
    evaluator = Evaluator(call_fn=multi_step_perfect, provider="t", model="m")
    result = evaluator.evaluate_item(item)
    # All three calls score as perfect structural matches
    assert result.score.function_selection == 1.0
    assert result.score.argument_accuracy == 1.0
    assert result.score.arabic_preservation == 1.0


def test_multi_step_missing_trailing_calls_are_penalized():
    """If the model emits only the first of three expected calls, axes
    should average down — not look like partial credit on only first."""

    def only_first(instruction, tools, functions):
        return {
            "calls": [
                {"function": "search_restaurants", "arguments": {"city": "الرياض"}},
            ],
            "raw": "",
        }

    item = EvalItem(
        id="multi_partial",
        category="multi_step",
        instruction="chain",
        dialect="msa",
        available_functions=["search_restaurants", "book_table", "send_message"],
        expected_calls=[
            ExpectedCall(function="search_restaurants", arguments={"city": "الرياض"}),
            ExpectedCall(function="book_table", arguments={"restaurant": "*", "guests": 2}),
            ExpectedCall(function="send_message", arguments={"recipient": "سارة", "platform": "whatsapp"}),
        ],
        difficulty="hard",
    )
    evaluator = Evaluator(call_fn=only_first, provider="t", model="m")
    result = evaluator.evaluate_item(item)
    # First call full credit, next two contribute 0 → 1/3 average
    assert abs(result.score.function_selection - (1 / 3)) < 1e-6
    assert abs(result.score.argument_accuracy - (1 / 3)) < 1e-6


def test_multi_step_wrong_middle_call_partial_credit():
    """Correct first + wrong middle + correct third — argument accuracy is
    the per-call mean across all three, not just the first."""

    def wrong_middle(instruction, tools, functions):
        return {
            "calls": [
                {"function": "search_restaurants", "arguments": {"city": "الرياض"}},
                {"function": "book_hotel", "arguments": {}},  # wrong tool
                {"function": "send_message", "arguments": {"recipient": "سارة", "platform": "whatsapp"}},
            ],
            "raw": "",
        }

    item = EvalItem(
        id="multi_middle",
        category="multi_step",
        instruction="chain",
        dialect="msa",
        available_functions=["search_restaurants", "book_table", "send_message", "book_hotel"],
        expected_calls=[
            ExpectedCall(function="search_restaurants", arguments={"city": "الرياض"}),
            ExpectedCall(function="book_table", arguments={"restaurant": "*", "guests": 2}),
            ExpectedCall(function="send_message", arguments={"recipient": "سارة", "platform": "whatsapp"}),
        ],
        difficulty="hard",
    )
    evaluator = Evaluator(call_fn=wrong_middle, provider="t", model="m")
    result = evaluator.evaluate_item(item)
    # 2/3 correct tool selection, not 1/3
    assert abs(result.score.function_selection - (2 / 3)) < 1e-6
