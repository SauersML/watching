import copy

import pytest

from watching.metrics import summarize_pairs


def rows_for_groups(sizes):
    rows = []
    index = 0
    for group, size in enumerate(sizes):
        for _ in range(size):
            for capture in (False, True):
                predicts_yes = capture if group == 0 else not capture
                rows.append(
                    {
                        "index": index,
                        "base_context_id": f"base{group}",
                        "capture": capture,
                        "yes_probability": 0.9 if predicts_yes else 0.1,
                        "yes_minus_no": 2.0 if predicts_yes else -2.0,
                        "correct": predicts_yes == capture,
                        "greedy_correct": predicts_yes == capture,
                        "ce": 0.2 if group == 0 else 2.0,
                    }
                )
            index += 1
    return rows


def test_repeated_variants_do_not_create_extra_sampling_units():
    small = summarize_pairs(rows_for_groups([1, 1]), seed=42)
    repeated = summarize_pairs(rows_for_groups([4, 4]), seed=42)
    for field in ("accuracy_ci95", "greedy_accuracy_ci95", "paired_log_odds_shift_ci95"):
        assert small[field] == repeated[field]
    assert repeated["n_prompts"] == 8
    assert repeated["n_base_contexts"] == 2
    assert repeated["resampling_unit"] == "base_context_id"
    naive = [
        {k: v for k, v in row.items() if k != "base_context_id"} for row in rows_for_groups([4, 4])
    ]
    assert summarize_pairs(naive, seed=42)["accuracy_ci95"] != repeated["accuracy_ci95"]


def test_one_context_has_no_population_interval_even_with_four_variants():
    result = summarize_pairs(rows_for_groups([4]), seed=1)
    for field in ("accuracy_ci95", "greedy_accuracy_ci95", "paired_log_odds_shift_ci95"):
        assert result[field] is None


def test_unequal_groups_preserve_prompt_weighted_point_estimate():
    result = summarize_pairs(rows_for_groups([2, 1]), seed=1)
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["paired_log_odds_shift"] == pytest.approx(4 / 3)


@pytest.mark.parametrize("bad", [None, "", "other"])
def test_pair_with_missing_or_inconsistent_group_is_rejected(bad):
    rows = rows_for_groups([2, 2])
    rows[0]["base_context_id"] = bad
    with pytest.raises(ValueError, match="base_context_id"):
        summarize_pairs(rows, seed=0)


def test_partial_group_metadata_is_rejected():
    rows = copy.deepcopy(rows_for_groups([2, 2]))
    for row in rows[:2]:
        del row["base_context_id"]
    with pytest.raises(ValueError, match="base_context_id"):
        summarize_pairs(rows, seed=0)


def test_native_accuracy_interval_is_not_binary_choice_interval():
    rows = rows_for_groups([4, 4])
    for row in rows:
        row["greedy_correct"] = False
    result = summarize_pairs(rows, seed=1)
    assert result["greedy_accuracy_ci95"] == [0, 0]
    assert result["accuracy_ci95"] == [0, 1]


def test_native_pair_success_requires_both_answers():
    rows = rows_for_groups([4, 4])
    for row in rows:
        row["greedy_correct"] = row["capture"]
    result = summarize_pairs(rows, seed=1)
    assert result["greedy_accuracy"] == 0.5
    assert result["greedy_both_conditions_correct"] == 0
    assert result["greedy_both_conditions_correct_ci95"] == [0, 0]
