import importlib.util
import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
SPEC = importlib.util.spec_from_file_location(
    "tail_comparison", Path(__file__).resolve().parents[1] / "scripts/plot_tail_comparison.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    data = {
        split: [{"context_id": f"{split}-{i}", "prompt": f"{split} prompt {i}"} for i in range(2)]
        for split in ("train", "validation")
    }
    write(root / "data.json", data)
    write(root / "manifest.json", {"shared_cache_sha256": "a" * 64})
    return data


def arm(root, name, data):
    folder = root / name
    write(folder / "data.json", data)
    write(
        folder / "manifest.json",
        {
            "shared_cache_sha256": "a" * 64,
            "primary_path": "split_inference",
            "output_head_updated": name == "two_head",
            "tail_layers": {"two_fixed": 2, "two_head": 2, "four_fixed": 4, "eight_fixed": 8}[name],
            "starting_global_step": 0,
        },
    )
    return folder


def score(folder, split, step, shift=0, runtime="split_inference"):
    rows = []
    for index in range(2):
        for condition in (False, True):
            rows.append(
                {
                    "index": index,
                    "condition_present": condition,
                    "context_id": f"{split}-{index}",
                    "yes_minus_no": (1 + index + shift) if condition else 0,
                    "full_vocab_correct_probability": 0.5,
                    "ce": 0.3 + 0.1 * index + 0.02 * shift,
                    "greedy_correct": condition,
                }
            )
    write(
        folder / f"{split}_{step}_{runtime}.json",
        {
            "summary": {"step": step, "global_step": step, "split": split, "path": runtime},
            "rows": rows,
        },
    )


def test_paired_differences_use_common_step_and_not_transfer(tmp_path):
    data = fixture(tmp_path)
    left, right = arm(tmp_path, "two_fixed", data), arm(tmp_path, "two_head", data)
    for split in data:
        for folder in (left, right):
            score(folder, split, 0)
        score(left, split, 8)
        score(right, split, 4, shift=99)
    report = MODULE.collect(tmp_path, draws=100)
    assert all(
        row["step"] == 0 and row["baseline_only"] for row in report["paired_arm_differences"]
    )
    for split in data:
        score(right, split, 8, shift=2)
        score(right, split, 8, shift=100, runtime="wholegraph_transfer")
    report = MODULE.collect(tmp_path, draws=100)
    for contrast in report["paired_arm_differences"]:
        assert contrast["step"] == 8
        assert contrast["metrics"]["gap"] == {"mean": 2.0, "ci95": [2.0, 2.0]}
        assert contrast["metrics"]["greedy_accuracy"] == {"mean": 0.0, "ci95": [0.0, 0.0]}
    assert report["wholegraph_transfer"]["two_head"]["train"]["metrics"]["gap"]["mean"] == 101.5
    assert report["arms"]["four_fixed"]["status"] == "pending"
    fig = MODULE.draw(report)
    fig.savefig(tmp_path / "comparison.png")
    MODULE.plt.close(fig)


def test_hash_or_prompt_mismatch_rejected(tmp_path):
    data = fixture(tmp_path)
    folder = arm(tmp_path, "two_fixed", data)
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["shared_cache_sha256"] = "b" * 64
    write(folder / "manifest.json", manifest)
    with pytest.raises(ValueError, match="shared-cache"):
        MODULE.collect(tmp_path, draws=100)
    manifest["shared_cache_sha256"] = "a" * 64
    write(folder / "manifest.json", manifest)
    data["train"].reverse()
    write(folder / "data.json", data)
    with pytest.raises(ValueError, match="identities/order"):
        MODULE.collect(tmp_path, draws=100)


def test_pending_arms_have_no_invented_estimates(tmp_path):
    fixture(tmp_path)
    report = MODULE.collect(tmp_path, draws=100)
    assert report["paired_arm_differences"] == []
    assert all(row["curves"] == {} for row in report["arms"].values())
    fig = MODULE.draw(report)
    assert any("pending" in text.get_text() for text in fig.texts)
    fig.savefig(tmp_path / "pending.png")
    MODULE.plt.close(fig)


def test_test_endpoints_and_runtime_specific_paired_contrasts(tmp_path):
    data = fixture(tmp_path)
    data["test"] = [{"context_id": f"test-{i}", "prompt": f"test prompt {i}"} for i in range(2)]
    write(tmp_path / "data.json", data)
    left, right = arm(tmp_path, "two_fixed", data), arm(tmp_path, "two_head", data)
    score(left, "test", 8)
    score(right, "test", 16, shift=2)
    report = MODULE.collect(tmp_path, draws=100)
    assert report["paired_arm_differences"] == []
    assert report["split_test_endpoints"]["two_head"]["step"] == 16
    score(left, "test", 16)
    score(left, "test", 16, runtime="wholegraph_transfer")
    score(right, "test", 16, shift=4, runtime="wholegraph_transfer")
    report = MODULE.collect(tmp_path, draws=100)
    contrasts = {r["runtime"]: r for r in report["paired_arm_differences"]}
    assert contrasts["split_inference"]["metrics"]["gap"]["ci95"] == [2.0, 2.0]
    assert contrasts["wholegraph_transfer"]["metrics"]["gap"]["ci95"] == [4.0, 4.0]
    assert all(r["step"] == 16 and r["split"] == "test" for r in contrasts.values())
    assert all(not c["points"] for c in report["arms"]["two_fixed"]["curves"].values())
    json.dumps(report, allow_nan=False)
