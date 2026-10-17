"""Endpoint summaries preserve prompt pairing and actual vocabulary-choice scoring."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "endpoints", ROOT / "scripts/summarize_viewing_endpoints.py"
)
ENDPOINTS = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(ROOT / "scripts"))
try:
    SPEC.loader.exec_module(ENDPOINTS)
finally:
    sys.path.pop(0)


def test_change_bootstrap_preserves_before_after_pairing():
    before = np.arange(40).reshape(10, 4).astype(float)
    after = before + np.array([0, 1, 2, 3])
    result = ENDPOINTS.paired_summary(before, after, draws=200)
    for i, metric in enumerate(ENDPOINTS.METRICS):
        assert result["change"][metric] == {"mean": float(i), "ci95": [float(i), float(i)]}
        assert result["before"][metric]["ci95"][0] < result["before"][metric]["ci95"][1]


def test_mismatched_endpoint_pairs_rejected():
    with pytest.raises(ValueError, match="matching"):
        ENDPOINTS.paired_summary(np.zeros((3, 4)), np.zeros((4, 4)))


def test_archived_storage_test_tradeoff_is_not_hidden():
    result = ENDPOINTS.collect()["arms"]["storage"]["splits"]["test"]
    assert result["before"]["both_correct"]["mean"] == 16 / 64
    assert result["after"]["both_correct"]["mean"] == 44 / 64
    assert result["change"]["full_vocab_ce"]["mean"] > 0
