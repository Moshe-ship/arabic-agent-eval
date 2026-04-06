"""Tests for the evaluation dataset."""

from arabic_agent_eval.dataset import Dataset, EvalItem, CATEGORIES, ALL_ITEMS


def test_dataset_has_minimum_items():
    ds = Dataset()
    assert len(ds) >= 50


def test_all_categories_covered():
    ds = Dataset()
    cats = ds.categories()
    for cat in CATEGORIES:
        assert cat in cats, f"Missing category: {cat}"
        assert cats[cat] >= 5, f"Category {cat} has only {cats[cat]} items, need >= 5"


def test_dialect_coverage():
    ds = Dataset()
    dialects = ds.dialects()
    assert "msa" in dialects
    assert "gulf" in dialects
    assert "egyptian" in dialects
    assert "levantine" in dialects
    assert "maghrebi" in dialects


def test_difficulty_levels():
    ds = Dataset()
    easy = ds.by_difficulty("easy")
    medium = ds.by_difficulty("medium")
    hard = ds.by_difficulty("hard")
    assert len(easy) > 0
    assert len(medium) > 0
    assert len(hard) > 0


def test_eval_item_has_required_fields():
    ds = Dataset()
    for item in ds:
        assert item.id
        assert item.category
        assert item.instruction
        assert item.dialect
        assert item.available_functions
        assert item.expected_calls
        assert item.difficulty in ("easy", "medium", "hard")


def test_instructions_contain_arabic():
    ds = Dataset()
    for item in ds:
        has_arabic = any("\u0600" <= c <= "\u06FF" for c in item.instruction)
        assert has_arabic, f"Item {item.id} instruction has no Arabic text"


def test_expected_calls_reference_available_functions():
    ds = Dataset()
    for item in ds:
        for call in item.expected_calls:
            if call.function != "*":
                assert call.function in item.available_functions, (
                    f"Item {item.id}: expected function '{call.function}' "
                    f"not in available_functions {item.available_functions}"
                )


def test_subset():
    ds = Dataset()
    subset = ds.subset(12)
    assert len(subset) <= 12
    # Should cover multiple categories
    cats = {item.category for item in subset}
    assert len(cats) >= 3


def test_category_weights_sum_to_one():
    total = sum(info["weight"] for info in CATEGORIES.values())
    assert abs(total - 1.0) < 0.01, f"Category weights sum to {total}, expected 1.0"


def test_error_recovery_items_have_error_response():
    ds = Dataset()
    error_items = ds.by_category("error_recovery")
    for item in error_items:
        assert item.error_response is not None, (
            f"Error recovery item {item.id} missing error_response"
        )
