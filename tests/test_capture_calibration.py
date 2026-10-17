"""Calibration must use development labels and retain whole prompt pairs."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "capture_calibration", Path(__file__).parents[1] / "scripts/audit_capture_calibration.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def rows_for(scores):
    rows = []
    for index, pair in enumerate(scores):
        for present, score in zip((False, True), pair, strict=True):
            lp_yes, lp_no = -np.logaddexp(0, -score), -np.logaddexp(0, score)
            rows.append(
                {
                    "index": index,
                    "condition_present": present,
                    "context_id": str(index),
                    "question_family": "fixture",
                    "yes_minus_no": score,
                    "logp_yes": float(lp_yes),
                    "logp_no": float(lp_no),
                    "greedy_correct": bool((score > 0) == present),
                    "ce": float(-lp_yes if present else -lp_no),
                }
            )
    return rows


def test_test_scores_cannot_change_calibrator(tmp_path):
    dev = rows_for([[-1, 3], [-0.5, 2], [-2, 1], [-3, -1]])
    path = tmp_path / "test_1024_split_inference.json"
    (tmp_path / "validation_1024_split_inference.json").write_text(json.dumps({"rows": dev}))
    path.write_text(json.dumps({"rows": rows_for([[-2, 2], [-1, 1]])}))
    before = audit.analyze(tmp_path, "split_inference", 50, 1)
    path.write_text(json.dumps({"rows": rows_for([[20, -20], [10, -10]])}))
    after = audit.analyze(tmp_path, "split_inference", 50, 1)
    assert (before["scale"], before["offset"]) == (after["scale"], after["offset"])
    assert before["splits"]["test"]["binary_loss"] != after["splits"]["test"]["binary_loss"]


@pytest.mark.parametrize("bad", ["duplicate", "missing", "wrong_logprob", "shared_context"])
def test_reject_invalid_pairs(tmp_path, bad):
    rows = rows_for([[-1, 1], [-2, 2]])
    if bad == "duplicate":
        rows.append(rows[0])
    elif bad == "missing":
        rows.pop()
    elif bad == "wrong_logprob":
        rows[0]["logp_yes"] += 1
    else:
        for row in rows:
            row["context_id"] = "same"
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"rows": rows}))
    with pytest.raises(ValueError):
        audit.read_pairs(path)
