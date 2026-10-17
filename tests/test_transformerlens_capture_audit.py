"""Integrity checks for the public-library capture audit, without library mocks."""

import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_transformerlens_capture.py"
spec = importlib.util.spec_from_file_location("tl_capture_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_byte_comparison_distinguishes_signed_zero():
    result = audit.difference(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert result["different_elements"] == 0
    assert not result["byte_identical"]


def test_scores_use_full_vocabulary_normalization():
    logits = torch.tensor([[[1.0, 3.0, 8.0]]])
    scores = audit.answer_scores(logits, yes_id=1, no_id=0)
    assert scores["yes_minus_no_log_odds"] == 2.0
    assert scores["top_token_id"] == 2
    assert scores["yes_logprob"] == pytest.approx(float(logits.log_softmax(-1)[0, 0, 1]))


def test_missing_or_empty_capture_is_rejected():
    with pytest.raises(ValueError, match="Capture keys"):
        audit.cache_summary({}, "query")
    with pytest.raises(ValueError, match="finite nonempty"):
        audit.cache_summary({"query": torch.tensor([])}, "query")


def test_nonfinite_logits_are_rejected():
    with pytest.raises(ValueError, match="Nonfinite"):
        audit.difference(torch.tensor([float("nan")]), torch.tensor([float("nan")]))
