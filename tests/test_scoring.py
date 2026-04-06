"""Tests for scoring module."""

from arabic_agent_eval.scoring import (
    Score,
    CategoryScore,
    compute_overall_score,
    grade,
    score_function_call,
    _contains_arabic,
)


def test_score_total_standard():
    s = Score(
        item_id="test",
        category="simple_function_calling",
        function_selection=1.0,
        argument_accuracy=1.0,
        arabic_preservation=1.0,
    )
    assert s.total == 1.0


def test_score_total_zero():
    s = Score(item_id="test", category="simple_function_calling")
    assert s.total == 0.0


def test_score_total_dialect():
    s = Score(
        item_id="test",
        category="dialect_handling",
        function_selection=1.0,
        argument_accuracy=1.0,
        arabic_preservation=1.0,
        dialect_understanding=1.0,
    )
    assert s.total == 1.0


def test_score_total_error():
    s = Score(
        item_id="test",
        category="error_recovery",
        function_selection=1.0,
        argument_accuracy=0.5,
        arabic_preservation=1.0,
        error_handling=1.0,
    )
    assert 0.8 < s.total <= 1.0


def test_category_score():
    scores = [
        Score(item_id="1", category="simple", function_selection=1.0, argument_accuracy=0.8, arabic_preservation=1.0),
        Score(item_id="2", category="simple", function_selection=1.0, argument_accuracy=0.6, arabic_preservation=1.0),
    ]
    cs = CategoryScore(category="simple", scores=scores)
    assert cs.avg_function_selection == 1.0
    assert cs.avg_argument_accuracy == 0.7
    assert cs.count == 2


def test_grade():
    assert grade(0.95) == "A"
    assert grade(0.85) == "B"
    assert grade(0.75) == "C"
    assert grade(0.65) == "D"
    assert grade(0.40) == "F"


def test_score_function_call_exact_match():
    func, arg, arabic = score_function_call(
        "search_flights",
        "search_flights",
        {"from_city": "الرياض", "to_city": "جدة"},
        {"from_city": "الرياض", "to_city": "جدة"},
    )
    assert func == 1.0
    assert arg == 1.0
    assert arabic == 1.0


def test_score_function_call_wrong_function():
    func, arg, arabic = score_function_call(
        "search_flights",
        "book_hotel",
        {"from_city": "الرياض"},
        {"city": "الرياض"},
    )
    assert func == 0.0


def test_score_function_call_none():
    func, arg, arabic = score_function_call(
        "search_flights",
        None,
        {"from_city": "الرياض"},
        None,
    )
    assert func == 0.0
    assert arg == 0.0


def test_contains_arabic():
    assert _contains_arabic("الرياض")
    assert _contains_arabic("hello الرياض world")
    assert not _contains_arabic("hello world")
    assert not _contains_arabic("12345")


def test_compute_overall_score():
    from arabic_agent_eval.dataset import CATEGORIES

    cat_scores = []
    for cat, info in CATEGORIES.items():
        scores = [Score(item_id="x", category=cat, function_selection=1.0, argument_accuracy=1.0, arabic_preservation=1.0)]
        cat_scores.append(CategoryScore(category=cat, scores=scores))

    overall = compute_overall_score(cat_scores)
    assert 0.9 < overall <= 1.0
