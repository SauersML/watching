"""Keep task, prompt, seed, and execution-path boundaries intact in joint figures."""

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

SPEC = importlib.util.spec_from_file_location(
    "joint_plot", Path(__file__).resolve().parents[1] / "scripts/plot_joint_progress.py"
)
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


def write(path, value):
    path.write_text(json.dumps(value))


def score_file(root, data, split, step, runtime="split_inference"):
    rows = []
    for index, record in enumerate(data[split]):
        for present in (False, True):
            rows.append(
                {
                    **record,
                    "index": index,
                    "condition_present": present,
                    "positive_minus_negative": 2.0 if present else 0.0,
                    "full_vocab_correct_probability": 0.75 if present else 0.25,
                    "greedy_correct": present,
                    "ce": 0.3 if present else 1.4,
                }
            )
    path = root / f"{split}_{step}_{runtime}.json"
    write(
        path,
        {
            "summary": {"split": split, "step": step, "global_step": step, "path": runtime},
            "rows": rows,
        },
    )
    return path


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "fit"
    root.mkdir()
    data = {
        split: [
            {
                "context_id": f"{split}/{i}",
                "task": task,
                "question_family": f"{split}_form",
                "answer_negative": "No" if task == "detection" else "cat",
                "answer_positive": "Yes" if task == "detection" else "owl",
            }
            for i, task in enumerate(("detection", "animal", "detection", "animal"))
        ]
        for split in PLOT.SPLITS
    }
    manifest = {
        "primary_path": "split_inference",
        "training_status": "complete",
        "completed_steps": 8,
        "seed": 17,
    }
    write(root / "data.json", data)
    write(root / "manifest.json", manifest)
    for split in ("train", "validation"):
        for step in (0, 8):
            score_file(root, data, split, step)
    for split in data:
        score_file(root, data, split, 8)
        score_file(root, data, split, 8, "wholegraph_transfer")
    return root, data


def test_task_metrics_use_whole_prompt_pairs_and_separate_endpoints(run):
    root, _ = run
    report = PLOT.collect(root, draws=100)
    assert set(report["curves"]) == {"train", "validation"}
    point = report["curves"]["train"][-1]
    for task in PLOT.TASKS:
        stats = point["tasks"][task]
        assert stats["n_prompts"] == 2
        assert stats["n_conditions"] == 4
        assert stats["metrics"]["gap"] == {"mean": 2.0, "ci95": [2.0, 2.0]}
        assert stats["metrics"]["correct_probability"]["mean"] == 0.5
        assert stats["metrics"]["actual_accuracy"]["ci95"] == [0.5, 0.5]
    assert set(report["endpoints"]) == {"split_inference", "wholegraph_transfer"}
    assert set(report["endpoints"]["wholegraph_transfer"]) == set(PLOT.SPLITS)
    json.dumps(report, allow_nan=False)


def test_wrong_task_and_duplicate_condition_fail(run):
    root, _ = run
    path = root / "train_8_split_inference.json"
    saved = PLOT.read(path)
    saved["rows"][0]["task"] = "animal"
    write(path, saved)
    with pytest.raises(ValueError, match="task disagrees"):
        PLOT.collect(root, draws=100)
    saved["rows"][0]["task"] = "detection"
    saved["rows"].append(saved["rows"][0])
    write(path, saved)
    with pytest.raises(ValueError, match="Duplicate condition"):
        PLOT.collect(root, draws=100)


def test_running_status_does_not_present_a_final_test(run):
    root, _ = run
    manifest = PLOT.read(root / "manifest.json")
    manifest["training_status"] = "running"
    write(root / "manifest.json", manifest)
    report = PLOT.collect(root, draws=100)
    assert report["endpoints"] == {}
    figure = PLOT.draw_endpoints(report)
    assert any("Awaiting" in t.get_text() for ax in figure.axes for t in ax.texts)
    plt.close(figure)


def test_matched_null_uses_only_common_steps_and_checks_source(run, tmp_path):
    root, data = run
    null = tmp_path / "null"
    null.mkdir()
    write(null / "data.json", data)
    manifest = {
        **PLOT.read(root / "manifest.json"),
        "condition_signal_available": False,
        "source_manifest_sha256": PLOT.digest(root / "manifest.json"),
    }
    write(null / "manifest.json", manifest)
    for split in ("train", "validation"):
        for step in (0, 4, 8):
            score_file(null, data, split, step)
    report = PLOT.collect(root, draws=100)
    PLOT.add_null(report, null, draws=100, seed=20261016)
    assert [p["step"] for p in report["null_control"]["curves"]["train"]] == [0, 8]
    manifest["seed"] = 18
    write(null / "manifest.json", manifest)
    with pytest.raises(ValueError, match="not matched"):
        PLOT.add_null(report, null, draws=100, seed=20261016)


def test_both_figures_save_with_explicit_task_and_runtime_labels(run, tmp_path):
    report = PLOT.collect(run[0], draws=100)
    for name, draw in (("progress", PLOT.draw_progress), ("endpoints", PLOT.draw_endpoints)):
        fig = draw(report)
        assert len(fig.axes) == (4 if name == "progress" else 8)
        fig.savefig(tmp_path / f"{name}.png", dpi=50)
        fig.savefig(tmp_path / f"{name}.svg")
        plt.close(fig)


def test_duplicate_layouts_do_not_shrink_context_bootstrap_and_single_base_has_no_ci():
    import numpy as np

    low = [0.0, 0.0, 0.0, 1.0, 0.0]
    high = [2.0, 1.0, 1.0, 0.0, 1.0]
    base = PLOT.bootstrap([low, high], 5000, 19)
    repeated = PLOT.bootstrap([low] * 4 + [high] * 4, 5000, 19, ["a"] * 4 + ["b"] * 4)
    assert repeated == base
    singleton = PLOT.bootstrap([low, high, low, high], 5000, 19, ["only"] * 4)
    assert all(stat["ci95"] is None for stat in singleton.values())
    assert singleton["actual_accuracy"]["mean"] == 0.5
    assert np.isfinite(singleton["gap"]["mean"])


def test_grouped_report_preserves_native_pair_metric_and_matched_prefix_contrasts(run, tmp_path):
    root, old_data = run
    data = {}
    for split in old_data:
        rows = []
        for task in PLOT.TASKS:
            for base in range(2):
                for group in PLOT.PREFIX_GROUPS:
                    rows.append(
                        {
                            "context_id": f"{split}/{task}/{base}/{group}",
                            "base_context_id": f"{split}/{task}/{base}",
                            "group": group,
                            "prefix_family": "neutral",
                            "task": task,
                            "question_family": split,
                            "answer_negative": "No" if task == "detection" else "cat",
                            "answer_positive": "Yes" if task == "detection" else "owl",
                        }
                    )
        data[split] = rows
    write(root / "data.json", data)
    for split in data:
        for step, runtime in (
            (0, "split_inference"),
            (8, "split_inference"),
            (8, "wholegraph_transfer"),
        ):
            path = score_file(root, data, split, step, runtime)
            payload = PLOT.read(path)
            for row in payload["rows"]:
                row["greedy_correct"] = row["base_context_id"].endswith("/1")
            write(path, payload)
    report = PLOT.collect(root, draws=1000)
    assert report["intervals"]["unit"] == "base_context_id"
    point = report["endpoints"]["wholegraph_transfer"]["test"]["tasks"]["animal"]
    assert point["n_prompts"] == 8 and point["n_base_contexts"] == 2
    assert point["metrics"]["actual_accuracy"] == {"mean": 0.5, "ci95": [0.0, 1.0]}
    assert point["metrics"]["both_correct"] == {"mean": 0.5, "ci95": [0.0, 1.0]}
    for group in PLOT.PREFIX_GROUPS:
        assert point["prefix_groups"][group]["n_base_contexts"] == 2
        for stat in point["prefix_change_from_single_user"][group].values():
            assert stat == {"mean": 0.0, "ci95": [0.0, 0.0]}
    layout = PLOT.draw_layout_snapshot(report)
    assert len(layout.axes) == 4
    plt.close(layout)
    fig = PLOT.draw_prefixes(report)
    assert len(fig.axes) == 8
    fig.savefig(tmp_path / "prefix.png", dpi=50)
    plt.close(fig)
    # A score cannot silently move one layout into a different resampling cluster.
    path = root / "train_8_split_inference.json"
    payload = PLOT.read(path)
    payload["rows"][0]["base_context_id"] = "wrong"
    write(path, payload)
    with pytest.raises(ValueError, match="base_context_id disagrees"):
        PLOT.collect(root, draws=100)
