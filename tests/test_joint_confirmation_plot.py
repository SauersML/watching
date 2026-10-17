"""The fresh figure must reproduce scores and reject incorrect task labels."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from watching.tail_train import score_record
from watching.validate_tail import paired_summary, task_summaries

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "joint_confirmation_plot", ROOT / "scripts/plot_joint_confirmation.py"
)
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


def fixture_files(path):
    records = []
    rows = []
    labels = {"No": 0, "Yes": 1, "cat": 2, "owl": 3}
    for i in range(4):
        animal = i >= 2
        record = {
            "task": "animal" if animal else "detection",
            "context_id": f"c{i}",
            "answer_negative": "cat" if animal else "No",
            "answer_positive": "owl" if animal else "Yes",
        }
        records.append(record)
        ids = (2, 3) if animal else (0, 1)
        for present in (False, True):
            logits = torch.zeros(1, 5)
            logits[0, ids[int(present)]] = 4
            row = score_record(logits, i, present, record, ids)
            row["greedy_token_id"] = row["top_token_id"]
            rows.append(row)
    payloads = {
        "manifest": {
            "split": "test",
            "task": "joint",
            "tail_sha256": "fixture",
            "seed": 7,
            "data_sha256": "fixture",
            "runtime": "fixture",
            "labels": labels,
        },
        "data": {"test": records},
    }
    for stage in ("base", "trained"):
        payloads[stage + "_reference"] = {
            "rows": rows,
            "summary": paired_summary(rows, 7),
            "tasks": task_summaries(rows, 7),
        }
        payloads[stage + "_repeat"] = {"none": {"bit_identical": True}}
    for name, payload in payloads.items():
        (path / f"{name}.json").write_text(json.dumps(payload))


def test_fresh_figure_reports_actual_answers_and_nonzero_binomial_uncertainty(tmp_path):
    fixture_files(tmp_path)
    result = PLOT.collect(tmp_path)
    for task in ("detection", "animal"):
        summary = result["tasks"][task]["trained"]
        assert summary["n_correct"] == summary["n_answers"] == 4
        assert summary["n_both_correct"] == 2
        assert summary["both_correct_iid_binomial_ci95"][0] < 1


@pytest.mark.parametrize("field,value", [("target_token_id", 0), ("task", "detection")])
def test_fresh_figure_rejects_wrong_animal_label_or_task(tmp_path, field, value):
    fixture_files(tmp_path)
    path = tmp_path / "trained_reference.json"
    data = json.loads(path.read_text())
    data["rows"][-1][field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        PLOT.collect(tmp_path)
