import pytest

from watching.compare import compare_arms, compare_scores, exact_sign_test


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


def test_exact_sign_test_ignores_ties_and_does_not_claim_significance_from_four_signs():
    result = exact_sign_test([0, 0, 0, 0, 1, 1, 1, 1])
    assert result == {
        "positive": 4,
        "negative": 0,
        "ties": 4,
        "p_greater": 0.0625,
        "p_two_sided": 0.125,
    }


def test_all_ties_provide_no_sign_evidence():
    assert exact_sign_test([0, 0])["p_greater"] == 1


def test_one_prompt_has_no_population_interval():
    result = compare_scores(rows([0]), rows([1]))
    assert result["change_in_separation"] == 1
    assert result["change_ci95"] is None


def test_arm_contrast_preserves_prompt_covariance_and_removes_baseline():
    result = compare_arms(
        rows([10, -10, 0]),
        rows([110, -110, 0]),
        rows([30, -30, 5]),
        list(reversed(rows([131, -129, 6], bias=100))),
    )
    contrast = result["fast_minus_uniform"]
    assert contrast["change_in_separation"] == 1
    assert contrast["change_ci95"] == [1, 1]


def test_arm_contrast_rejects_mismatched_prompt_sets():
    with pytest.raises(ValueError, match="same prompt"):
        compare_arms(rows([0]), rows([1]), rows([0, 1]), rows([1, 2]))
