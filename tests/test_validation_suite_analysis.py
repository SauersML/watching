"""Sampling units, endpoint selection and nondegenerate bounded-accuracy checks."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "validation_analysis", Path(__file__).parents[1] / "scripts/analyze_validation_suite.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def pair(index, family="one", scores=(-2.0, 2.0), task="detection"):
    rows = []
    for condition, gap in zip((False, True), scores, strict=True):
        positive, negative = -np.logaddexp(0, -gap), -np.logaddexp(0, gap)
        rows.append(
            {
                "index": index,
                "context_id": f"{task}:{index}",
                "question_family": family,
                "task": task,
                "capture": condition,
                "logp_yes": positive,
                "logp_no": negative,
                "yes_minus_no": gap,
                "ce": -float(positive if condition else negative),
                "greedy_correct": gap > 0 if condition else gap <= 0,
            }
        )
    return rows


def test_pairs_are_complete_and_positive_gap_is_not_actual_accuracy():
    rows = pair(0, scores=(9, 11))
    observed = analysis.paired_observations(rows)
    values = dict(zip(analysis.METRICS, observed[0]["values"], strict=True))
    assert values["gap"] == 2
    assert values["actual_accuracy"] == 0.5
    assert values["both_correct"] == 0
    assert values["full_ce"] > 4
    with pytest.raises(ValueError, match="both conditions"):
        analysis.paired_observations(rows[:1])
    with pytest.raises(ValueError, match="Duplicate"):
        analysis.paired_observations(rows + rows)


def test_family_weighting_is_not_silently_prompt_weighting():
    rows = pair(0, "small", scores=(-1, 1))
    for index in (1, 2, 3):
        rows += pair(index, "large", scores=(-5, 5))
    result = analysis.summarize_observations(analysis.paired_observations(rows), draws=200)[
        "detection"
    ]
    assert result["prompt_weighted"]["metrics"]["gap"]["mean"] == 8
    assert result["equal_family_weighted"]["metrics"]["gap"]["mean"] == 6
    assert result["equal_family_weighted"]["n_families"] == 2
    assert result["prompt_weighted"]["n_prompts"] == 4


def test_perfect_empirical_accuracy_does_not_imply_population_certainty():
    rows = sum((pair(i, family=str(i % 4)) for i in range(64)), [])
    result = analysis.summarize_observations(analysis.paired_observations(rows), draws=200)[
        "detection"
    ]
    assert result["prompt_weighted"]["metrics"]["actual_accuracy"]["ci95"] == [1, 1]
    bounded = result["prompt_both_correct_iid_clopper_pearson"]
    assert bounded["successes"] == bounded["n"] == 64
    assert bounded["ci95"][0] == pytest.approx(0.025 ** (1 / 64))
    assert bounded["ci95"][1] == 1
    assert "Independent" in bounded["assumption"]


def test_tasks_are_separate_and_same_task_context_repeats_are_rejected():
    rows = pair(0, task="detection") + pair(1, task="animal", scores=(9, 11))
    result = analysis.summarize_observations(analysis.paired_observations(rows), draws=200)
    assert set(result) == {"detection", "animal"}
    assert result["animal"]["prompt_weighted"]["metrics"]["actual_accuracy"]["mean"] == 0.5
    repeated = pair(1)
    for row in repeated:
        row["context_id"] = "detection:0"
    with pytest.raises(ValueError, match="Repeated context"):
        analysis.paired_observations(pair(0) + repeated)


def test_only_completed_predeclared_endpoint_is_accepted(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"steps": 8, "completed_steps": 8}))
    path = tmp_path / "test.json"
    path.write_text(json.dumps({"summary": {"step": 7}, "rows": pair(0)}))
    with pytest.raises(ValueError, match="predeclared"):
        analysis.analyze_file(path, draws=200)
    path.write_text(json.dumps({"summary": {"step": 8}, "rows": pair(0)}))
    result = analysis.analyze_file(path, draws=200)
    assert result["completed_steps"] == 8


def test_resampling_keeps_correlated_metrics_together_and_is_reproducible():
    values = np.array([[1, -1], [2, -2], [4, -4]])
    intervals = analysis.bootstrap_means(values, 1000, 9)
    assert intervals == analysis.bootstrap_means(values, 1000, 9)
    assert intervals[0] == pytest.approx([-intervals[1][1], -intervals[1][0]])


def test_perturbing_buffer_is_never_reported_as_a_valid_negative_control():
    payload = {"rows": pair(0), "valid_no_footprint_control": False}
    assert "alternative intervention" in analysis.control_interpretation(payload)
    payload["valid_no_footprint_control"] = True
    with pytest.raises(ValueError, match="full-vocabulary"):
        analysis.control_interpretation(payload)
    payload["full_vocabulary_checks"] = [{"index": 0, "bit_identical": True}]
    assert "passed" in analysis.control_interpretation(payload)
    payload["full_vocabulary_checks"][0]["bit_identical"] = False
    with pytest.raises(ValueError, match="full-vocabulary"):
        analysis.control_interpretation(payload)


def test_dataset_metadata_and_probability_sanity_are_checked():
    with pytest.raises(ValueError, match="empty"):
        analysis.paired_observations([])
    records = [{"context_id": "wrong", "question_family": "one"}]
    with pytest.raises(ValueError, match="metadata"):
        analysis.paired_observations(pair(0), records)
    rows = pair(0)
    rows[0]["logp_yes"] = 1
    with pytest.raises(ValueError, match="vocabulary subset"):
        analysis.paired_observations(rows)
    rows = pair(0)
    rows[0]["greedy_correct"] = False
    with pytest.raises(ValueError, match="majority"):
        analysis.paired_observations(rows)
