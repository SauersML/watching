"""Guard the paired diagnostic's splitting and calibration interpretation."""

import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "head_regression_audit", Path(__file__).parents[1] / "scripts/audit_head_regression.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_strong_paired_separation_does_not_imply_individual_detection():
    # Both executions favor Yes; presence raises it further, but absence is wrong.
    logits = torch.tensor([[0.0, 9.0], [0.0, 11.0]], dtype=torch.float64)
    metrics = audit.paired_metrics(logits, torch.full((2,), -100.0))
    assert metrics["mean_paired_gap"] == 2
    assert metrics["fraction_positive_gap"] == 1
    assert metrics["binary_accuracy"] == 0.5
    assert metrics["score_midpoint_rms"] == 10
    assert metrics["binary_ce"] > 4


def test_prompt_and_family_holdouts_are_distinct_and_pair_preserving():
    records = [
        {"context_id": f"context-{family}-{i}", "question_family": family}
        for family in ("a", "b", "c")
        for i in range(10)
    ]
    assignments = {"a": 0, "b": 1, "c": 2}
    splits = audit.partition_prompts(records, assignments)
    assert splits == audit.partition_prompts(records, assignments)
    prompt = splits["paired_prompt_holdout"]
    assert len(prompt) == 6
    assert {records[i]["question_family"] for i in prompt} == {"a", "b", "c"}
    family = splits["question_family_holdout"]
    assert len(family) == 10
    assert {records[i]["question_family"] for i in family} == {"a"}
    row_mask = [i // 2 in prompt for i in range(len(records) * 2)]
    assert row_mask[::2] == row_mask[1::2]


def test_rest_vocabulary_probability_changes_full_ce_but_not_binary_gap():
    logits = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64)
    small = audit.paired_metrics(logits, torch.full((2,), -100.0))
    large = audit.paired_metrics(logits, torch.full((2,), 10.0))
    assert small["mean_paired_gap"] == large["mean_paired_gap"] == 4
    assert small["binary_ce"] == pytest.approx(large["binary_ce"])
    assert large["full_vocab_ce"] > small["full_vocab_ce"] + 8


def test_constrained_lda_reads_one_execution_without_test_pair_centering():
    features = torch.tensor(
        [[1.0, nuisance, condition] for nuisance in (-3, -1, 1, 3) for condition in (-1, 1)],
        dtype=torch.float64,
    )
    labels = torch.arange(len(features)) % 2
    results = audit.constrained_lda(features, labels)
    for direction, info in results:
        assert info["training_gap"] == pytest.approx(2)
        assert info["training_mean_score"] == pytest.approx(0, abs=1e-10)
        # Each independent execution scores correctly despite unseen nuisance values.
        absent = torch.tensor([1.0, 100.0, -1.0], dtype=torch.float64)
        present = torch.tensor([1.0, -70.0, 1.0], dtype=torch.float64)
        assert float(absent @ direction) == pytest.approx(-1)
        assert float(present @ direction) == pytest.approx(1)
