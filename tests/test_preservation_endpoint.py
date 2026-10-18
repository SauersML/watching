import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "audit_preservation_endpoint",
    Path(__file__).resolve().parents[1] / "scripts/audit_preservation_endpoint.py",
)
endpoint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(endpoint)


def test_crossed_resampling_preserves_pairs_and_rejects_missing_cells():
    values = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    families = ["a", "a", "b", "b"]
    contexts = ["x", "y", "x", "y"]
    bounds = endpoint.crossed_intervals(values, families, contexts)
    np.testing.assert_equal(bounds, [[0, 1], [0, 1]])
    with pytest.raises(ValueError, match="complete grid"):
        endpoint.crossed_intervals(values[:3], families[:3], contexts[:3])
    with pytest.raises(ValueError, match="complete grid"):
        endpoint.crossed_intervals(values, families, ["x", "x", "x", "y"])


def test_full_vocabulary_validation_detects_wrong_top_token():
    native = torch.tensor([[0.0, 2.0, -1.0]], dtype=torch.bfloat16)
    lp = native.double().log_softmax(-1)
    rows = [
        {
            "id": "p",
            "yes_logprob": float(lp[0, 1]),
            "no_logprob": float(lp[0, 2]),
            "top_token_id": 1,
            "logits_are_bf16_values": True,
        }
    ]
    actual, ties = endpoint.validate_rows(rows, ["p"], native, 1, 2, logits=True)
    assert ties == 0
    torch.testing.assert_close(actual, lp)
    rows[0]["top_token_id"] = 0
    with pytest.raises(ValueError, match="vocabulary maximum"):
        endpoint.validate_rows(rows, ["p"], native, 1, 2, logits=True)
    rows[0]["top_token_id"] = 1
    rows[0]["yes_logprob"] += 0.1
    with pytest.raises(ValueError, match="Scalar logprob"):
        endpoint.validate_rows(rows, ["p"], native, 1, 2, logits=True)
