"""Guard the scope and paired decision geometry of the offline offset diagnostic."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "joint_calibration",
    Path(__file__).resolve().parents[1] / "scripts/analyze_joint_calibration.py",
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def pair(absent, present):
    return {
        condition: {
            "positive_minus_negative": score,
            "greedy_correct": bool(score > 0) == condition,
            "ce": float(np.logaddexp(0, score if not condition else -score)),
        }
        for condition, score in [(False, absent), (True, present)]
    }


def test_common_offset_changes_calibration_without_changing_gap():
    result = AUDIT.describe([pair(2, 4), pair(2, 4)], -3)
    assert result["gap_mean"] == 2
    assert result["midpoint_mean"] == 3
    assert result["native_one_correct_pairs"] == 2
    assert result["binary_accuracy_half_credit_ties"] == 0.5
    assert result["offset_binary_accuracy"] == 1
    assert result["offset_binary_ce"] == pytest.approx(np.logaddexp(0, -1))
    assert result["offset_binary_ce"] == result["oracle_per_prompt_centered_binary_ce"]


def test_zero_signal_has_balanced_loss_floor_and_half_credit_ties():
    result = AUDIT.describe([pair(0, 0), pair(0, 0)], 0)
    assert result["gap_mean"] == 0
    assert result["binary_ce"] == pytest.approx(np.log(2))
    assert result["offset_binary_accuracy"] == 0.5


def test_reader_rejects_test_data(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"summary": {"split": "test"}, "rows": []}))
    with pytest.raises(ValueError, match="Only training and development"):
        AUDIT.load(path)


def test_reader_rejects_duplicate_conditions(tmp_path):
    row = {"task": "animal", "context_id": "one", "capture": False, "question_family": "plain"}
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"summary": {"split": "train"}, "rows": [row, row]}))
    with pytest.raises(ValueError, match="duplicate"):
        AUDIT.load(path)
