"""The independent audit must detect corrupted scores and preserve paired units."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_framework_transfer.py"
SPEC = importlib.util.spec_from_file_location("framework_vector_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def row_for(vector, *, logprobs=False, top=None):
    lp = vector.double() if logprobs else vector.double().log_softmax(-1)
    return {
        "yes_id": 0,
        "no_id": 1,
        "top_token_id": int(vector.argmax()) if top is None else top,
        "yes_logprob": float(lp[0]),
        "no_logprob": float(lp[1]),
    }


def test_reconstruct_uses_full_vocabulary_and_checks_recorded_scores():
    vector = torch.tensor([1.0, -1.0, 7.0])
    row = row_for(vector)
    result = audit.reconstruct(row, vector, logprobs=False)
    assert not result["yes_correct"] and not result["no_correct"]
    assert result["scalar_max_error"] == 0
    row["yes_logprob"] += 0.001
    with pytest.raises(ValueError, match="Scalar scores"):
        audit.reconstruct(row, vector, logprobs=False)


def test_vllm_native_tie_is_not_replaced_by_first_argmax():
    lp = torch.tensor([1.0, 1.0, -3.0]).log_softmax(-1)
    result = audit.reconstruct(row_for(lp, logprobs=True, top=1), lp, logprobs=True)
    assert result["no_correct"] and not result["yes_correct"]
    with pytest.raises(ValueError, match="argmax"):
        audit.reconstruct(row_for(lp, logprobs=True, top=2), lp, logprobs=True)
    with pytest.raises(ValueError, match="Unnormalized"):
        audit.reconstruct(row_for(lp + 0.1, logprobs=True), lp + 0.1, logprobs=True)


def test_reconstruction_rejects_nonfinite_and_ambiguous_vector_schema():
    vector = torch.tensor([1.0, 0.0])
    row = row_for(vector)
    with pytest.raises(ValueError, match="FP32 vector"):
        audit.reconstruct(row, vector.reshape(1, 2), logprobs=False)
    with pytest.raises(ValueError, match="Nonfinite"):
        audit.reconstruct(row, torch.tensor([float("inf"), 0.0]), logprobs=False)


def test_comparison_reports_numerical_differences_and_signed_zero_bits():
    comparison = audit.compare_vectors(torch.tensor([0.0, 1.0]), torch.tensor([-0.0, 2.0]))
    assert not comparison["bit_identical"]
    assert comparison["different_values"] == 1
    assert comparison["max_abs_difference"] == 1
    assert comparison["rms_difference"] == pytest.approx(np.sqrt(0.5))
    zero = audit.compare_vectors(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert not zero["bit_identical"] and zero["different_values"] == 0


def test_aggregate_keeps_pair_units_and_full_vocabulary_errors():
    records = [{"id": "a"}, {"id": "b"}]
    yes_vector, no_vector = torch.tensor([2.0, 0.0, -5.0]), torch.tensor([0.0, 2.0, -5.0])
    other = torch.tensor([1.0, 0.0, 5.0])
    scores = {}
    for row in records:
        scores[row["id"], "train", False] = audit.reconstruct(
            row_for(yes_vector), yes_vector, logprobs=False
        )
        scores[row["id"], "serve", False] = audit.reconstruct(
            row_for(no_vector), no_vector, logprobs=False
        )
    scores["b", "train", False] = audit.reconstruct(row_for(other), other, logprobs=False)
    measured = audit.aggregate(records, scores, "train", "serve", 500, 57)
    assert measured["n_contexts"] == 2
    assert measured["both_correct"]["count"] == 1
    assert measured["both_correct"]["mean"] == 0.5
    assert measured["marginal_accuracy"]["mean"] == 0.75
    assert measured["separation"]["mean"] == pytest.approx(3.5)


def test_zero_pair_accuracy_exact_interval_does_not_collapse():
    records = [{"id": "a"}]
    vector = torch.tensor([2.0, 0.0, -5.0])
    value = audit.reconstruct(row_for(vector), vector, logprobs=False)
    scores = {("a", path, False): value for path in ("train", "serve")}
    measured = audit.aggregate(records, scores, "train", "serve", 500, 57)
    assert measured["both_correct"]["mean"] == 0
    assert measured["both_correct"]["ci95"] == pytest.approx([0.0, 0.975])
    assert measured["separation"]["mean"] == 0
