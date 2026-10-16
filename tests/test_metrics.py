import pytest

from watching.metrics import summarize_pairs


def prediction(index, capture, predicts_yes):
    return {
        "index": index,
        "capture": capture,
        "yes_probability": 0.9 if predicts_yes else 0.1,
        "correct": predicts_yes == capture,
        "greedy_correct": predicts_yes == capture,
        "ce": 0.2,
    }


def test_constant_yes_is_chance_with_no_correct_pairs():
    rows = [prediction(i, capture, True) for i in range(8) for capture in (False, True)]
    result = summarize_pairs(rows, seed=0)
    assert result["accuracy"] == 0.5
    assert result["accuracy_ci95"] == [0.5, 0.5]
    assert result["present_accuracy"] == 1
    assert result["absent_accuracy"] == 0
    assert result["both_conditions_correct"] == 0
    assert result["paired_yes_probability_gap"] == 0


def test_perfect_detection_and_pairing_are_independent_of_row_order():
    rows = [prediction(i, capture, capture) for i in range(8) for capture in (False, True)]
    result = summarize_pairs(list(reversed(rows)), seed=0)
    assert result["accuracy"] == 1
    assert result["accuracy_ci95"] == [1, 1]
    assert result["both_conditions_correct"] == 1
    assert result["paired_yes_probability_gap"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [prediction(0, False, False)],
        [prediction(0, False, False), prediction(0, False, False)],
    ],
)
def test_missing_or_duplicate_conditions_are_rejected(rows):
    with pytest.raises(ValueError):
        summarize_pairs(rows, seed=0)


def test_greedy_accuracy_is_not_forced_choice_accuracy():
    rows = [prediction(0, False, False), prediction(0, True, True)]
    for row in rows:
        row["greedy_correct"] = False
    result = summarize_pairs(rows, seed=0)
    assert result["accuracy"] == 1
    assert result["greedy_accuracy"] == 0
