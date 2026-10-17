"""Integrity checks for the library audit's numerical comparisons."""

import importlib.util
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "audit_nnsight_capture", Path(__file__).parents[1] / "scripts/audit_nnsight_capture.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_exact_comparison_distinguishes_signed_zero():
    result = audit.difference(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert not result["byte_identical"]
    assert result["different_elements"] == 0


def test_comparison_rejects_broadcast_and_nonfinite_values():
    with pytest.raises(ValueError, match="matching shapes"):
        audit.difference(torch.zeros(2, 1), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="finite"):
        audit.difference(torch.zeros(1), torch.tensor([float("nan")]))


def test_model_hash_detects_non_parameter_buffer_mutation():
    model = torch.nn.Linear(2, 2)
    model.register_buffer("observed", torch.zeros(1, dtype=torch.bfloat16))
    before = audit.model_hash(model)
    model.observed.add_(1)
    assert audit.model_hash(model) != before


def test_capture_must_be_nonempty_and_finite():
    with pytest.raises(ValueError, match="nonempty"):
        audit.capture_description(torch.empty(0))
    with pytest.raises(ValueError, match="nonfinite"):
        audit.capture_description(torch.tensor([float("inf")]))
