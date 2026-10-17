"""Recompute the published development comparison and reject corrupt raw pairing."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "comparison", ROOT / "scripts/plot_viewing_comparison.py"
)
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


def test_archived_comparison_uses_pair_accuracy_and_exact_null():
    report = PLOT.collect(ROOT)
    assert report["curves"]["reference"][-1]["mean"][0] == 1
    assert report["curves"]["storage"][-1]["mean"][0] == 0.5
    assert report["curves"]["null"][-1]["mean"] == [0, 0]
    assert report["buffer_both_correct_ceiling"] == 17 / 64
    assert report["n_prompts"] == 64


def test_duplicate_prompt_condition_rejected(tmp_path):
    fit = ROOT / "results/two-block-scaled-034/fit"
    path = fit / "validation_1024_split_inference.json"
    saved = PLOT.read(path)
    saved["rows"].append(saved["rows"][0])
    bad = tmp_path / path.name
    bad.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="Duplicate"):
        PLOT.point(bad, PLOT.read(fit / "data.json")["validation"], draws=100)


def test_forced_binary_summary_cannot_replace_full_vocabulary_correctness(tmp_path):
    fit = ROOT / "results/viewing-learning-051-052/identity/fit"
    path = fit / "validation_1024_split_inference.json"
    saved = PLOT.read(path)
    saved["rows"][0]["greedy_correct"] = not saved["rows"][0]["greedy_correct"]
    bad = tmp_path / path.name
    bad.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="Top-token"):
        PLOT.point(bad, PLOT.read(fit / "data.json")["validation"], draws=100)
