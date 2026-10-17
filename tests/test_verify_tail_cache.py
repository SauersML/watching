"""Artifact checks must bind numerical state to exact records and positions."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "verify_tail_cache", Path(__file__).resolve().parents[1] / "scripts/verify_tail_cache.py"
)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def fixture():
    record = {"context_id": "sample", "text": "A short note.", "task": "animal"}
    data = {"validation": [record]}
    positions = torch.tensor([[0, 1, 2]])
    state = (
        torch.ones((1, 3, 2), dtype=torch.bfloat16),
        torch.ones((1, 1, 3, 3), dtype=torch.bool),
        torch.ones((1, 3, 2), dtype=torch.bfloat16),
        torch.zeros((1, 3, 2), dtype=torch.bfloat16),
    )
    cache = {
        "version": 1,
        "conditions": [False, True],
        "checkpoint": "fixture",
        "length": 3,
        "data": copy.deepcopy(data),
        "states": {"validation": [(state, state)]},
        "batches": {
            "validation": [{"position_ids": positions, "attention_mask": torch.ones((1, 3))}]
        },
    }
    fingerprints = {
        "validation": [
            {
                "index": 0,
                "context_id": "sample",
                "positions_sha256": VERIFY.tensor_digest(positions),
                "absent": VERIFY.state_fingerprint(state),
                "present": VERIFY.state_fingerprint(state),
            }
        ]
    }
    return cache, data, fingerprints, {"checkpoint": "fixture", "length": 3}


def test_accepts_development_only_cache_without_fabricated_test_split():
    assert VERIFY.verify(*fixture()) == 1


def test_rejects_changed_tensor_even_if_file_remains_readable():
    args = fixture()
    args[0]["states"]["validation"][0][0][0][0, 0, 0] = 2
    with pytest.raises(ValueError, match="fingerprint"):
        VERIFY.verify(*args)


def test_rejects_changed_context_identity():
    args = fixture()
    args[1]["validation"][0]["context_id"] = "different"
    with pytest.raises(ValueError, match="dataset"):
        VERIFY.verify(*args)


def test_rejects_position_mask_mismatch():
    args = fixture()
    args[0]["batches"]["validation"][0]["attention_mask"][0, 0] = 0
    with pytest.raises(ValueError, match="Position"):
        VERIFY.verify(*args)
