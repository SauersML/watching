import importlib.util
from pathlib import Path

import pytest
import torch

from watching.tail_train import score

SPEC = importlib.util.spec_from_file_location(
    "viewing_subsets", Path(__file__).parents[1] / "scripts/analyze_viewing_subsets.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def scores():
    return [score(torch.tensor([[0.0, 1.0]]), 0, present, 0, 1) for present in (False, True)] + [
        score(torch.tensor([[1.0, 0.0]]), 1, False, 0, 1),
        score(torch.tensor([[0.0, 1.0]]), 1, True, 0, 1),
    ]


def test_subgroup_summary_keeps_both_conditions_and_original_indices():
    result = MODULE.subset_report(scores(), [True, False], 42)
    assert result["identical_states"]["greedy_accuracy"] == 0.5
    assert result["identical_states"]["greedy_both_conditions_correct"] == 0
    assert result["different_states"]["greedy_both_conditions_correct"] == 1
    assert result["different_states"]["paired_log_odds_shift"] == 2


def test_incomplete_pair_is_rejected():
    with pytest.raises(ValueError, match="both scored conditions"):
        MODULE.subset_report(scores()[:-1], [True, False], 42)


def test_unequal_probabilities_on_identical_states_fail_closed():
    with pytest.raises(ValueError, match="unequal scored"):
        MODULE.subset_report(scores(), [True, True], 42)
