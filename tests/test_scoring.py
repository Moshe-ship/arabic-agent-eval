"""Tests for scoring module."""

from arabic_agent_eval.scoring import (
    Score,
    CategoryScore,
    compute_overall_score,
    grade,
    normalize_arabic,
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


def test_score_function_call_empty_expected_args_no_crash():
    """Regression: score_function_call used to raise UnboundLocalError when expected_args was empty."""
    func, arg, arabic = score_function_call(
        "get_time",
        "get_time",
        {},
        {},
    )
    assert func == 1.0
    assert arg == 1.0
    assert arabic == 1.0


def test_score_function_call_empty_expected_args_with_extra_actual():
    """Empty expected but actual has args → half credit on args, perfect preservation."""
    func, arg, arabic = score_function_call(
        "get_time",
        "get_time",
        {},
        {"city": "tokyo"},
    )
    assert func == 1.0
    assert arg == 0.5
    assert arabic == 1.0


def test_normalize_arabic_alef_variants():
    assert normalize_arabic("أحمد") == normalize_arabic("احمد")
    assert normalize_arabic("إبراهيم") == normalize_arabic("ابراهيم")
    assert normalize_arabic("آية") == normalize_arabic("اية")


def test_normalize_arabic_ya_and_ta_marbuta():
    assert normalize_arabic("مكتبى") == normalize_arabic("مكتبي")
    assert normalize_arabic("فاطمة") == normalize_arabic("فاطمه")


def test_normalize_arabic_tatweel_stripped():
    assert normalize_arabic("مــرحبا") == normalize_arabic("مرحبا")


def test_normalize_arabic_non_arabic_lowercased():
    assert normalize_arabic("Hello World") == "hello world"


def test_score_function_call_arabic_normalization_partial_credit():
    """alef variants should score 0.9 (normalized match), not 0."""
    func, arg, arabic = score_function_call(
        "book_hotel",
        "book_hotel",
        {"city": "إبراهيمية"},
        {"city": "ابراهيمية"},
    )
    assert func == 1.0
    assert arg == 0.9
    assert arabic == 1.0
