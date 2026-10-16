import pytest

from watching.compare import compare_scores


def rows(separations, bias=0):
    return [
        {"index": i, "capture": c, "yes_minus_no": bias + (d if c else 0)}
        for i, d in enumerate(separations)
        for c in (False, True)
    ]


def test_common_answer_bias_does_not_count_as_learning():
    result = compare_scores(rows([0, 1, -1]), rows([0, 1, -1], bias=100))
    assert result["change_in_separation"] == 0
    assert result["change_ci95"] == [0, 0]


def test_bootstrap_preserves_pairing_across_training():
    result = compare_scores(rows([-10, 0, 10]), list(reversed(rows([-9, 1, 11]))))
    assert result["change_in_separation"] == 1
    assert result["change_ci95"] == [1, 1]


def test_mismatched_prompts_are_rejected():
    with pytest.raises(ValueError, match="same prompt"):
        compare_scores(rows([0]), rows([0, 1]))
