"""The diagnostic must not cherry-pick errors or reinterpret unfinished output."""

import pytest

from watching.complete_final_line import (
    diagnostic_records,
    final_line_result,
    first_step_comparison,
    fixed_batch,
)


def test_entire_family_selected_without_scores_and_incomplete_bank_rejected():
    rows = [
        dict(
            task="animal",
            question_family="final_line_answer",
            context_id=str(i),
            answer_negative="cat",
            answer_positive="owl",
        )
        for i in range(8)
    ]
    other = dict(rows[0], context_id="other", question_family="direct")
    assert [i for i, _ in diagnostic_records({"test": [other] + rows})] == list(range(1, 9))
    with pytest.raises(ValueError, match="eight"):
        diagnostic_records({"test": rows[:-1]})


def test_fixed_shape_consumes_padding_preserving_all_prompt_tokens_and_positions():
    first = fixed_batch([11, 12], 0, 5, "cpu")
    later = fixed_batch([11, 12, 13, 14], 0, 5, "cpu")
    assert first["input_ids"].tolist() == [[0, 0, 0, 11, 12]]
    assert later["input_ids"].tolist() == [[0, 11, 12, 13, 14]]
    assert later["position_ids"].tolist() == [[0, 0, 1, 2, 3]]
    assert later["attention_mask"].tolist() == [[0, 1, 1, 1, 1]]
    with pytest.raises(ValueError, match="truncation"):
        fixed_batch(list(range(6)), 0, 5, "cpu")


def test_literal_last_line_only_after_eos_no_semantic_regrading():
    assert final_line_result("A preference.\n\n cat \n", "cat", "eos")["final_line_correct"]
    for text in ("Cat", "cat.", "I choose cat", "cat\nExplanation"):
        assert final_line_result(text, "cat", "eos")["final_line_correct"] is False
    result = final_line_result("cat", "cat", "token_budget")
    assert result["literal_target_match"] and result["final_line_correct"] is None
    assert result["truncated_or_unresolved"]


def test_replay_reports_logprob_shift_even_when_top_token_unchanged():
    old = dict(logp_negative=-1.0, logp_positive=-2.0, positive_minus_negative=-1.0, top_token_id=3)
    assert first_step_comparison(old, old)["all_recorded_first_step_values_identical"]
    changed = dict(old, logp_negative=-1.001)
    result = first_step_comparison(changed, old)
    assert result["top_token_matches"]
    assert not result["all_recorded_first_step_values_identical"]
