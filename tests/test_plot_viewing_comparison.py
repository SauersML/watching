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


def test_continuation_uses_total_steps_and_requires_exact_baseline(tmp_path):
    original = ROOT / "results/viewing-learning-051-052/buffer/fit"
    manifest = PLOT.read(original / "manifest.json")
    manifest.update(
        source_recompiled=False,
        optimizer_state_restored=False,
        starting_global_step=1024,
        source_hashes={"manifest.json": PLOT.digest(original / "manifest.json")},
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "data.json").write_bytes((original / "data.json").read_bytes())
    saved = PLOT.read(original / "validation_1024_split_inference.json")
    for local in (0, 128):
        saved["summary"].update(step=local, global_step=1024 + local)
        (tmp_path / f"validation_{local}_split_inference.json").write_text(json.dumps(saved))
    report = PLOT.collect(ROOT, continuations={"buffer": tmp_path})
    assert report["curves"]["buffer"][-1]["step"] == 1152
    assert report["curves"]["buffer"][-1]["local_step"] == 128
    assert report["restarts"]["buffer"]["global_step"] == 1024
    baseline = tmp_path / "validation_0_split_inference.json"
    saved = PLOT.read(baseline)
    saved["rows"][0]["logp_positive"] += 1e-12
    baseline.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="exact answer-score replay"):
        PLOT.collect(ROOT, continuations={"buffer": tmp_path})
