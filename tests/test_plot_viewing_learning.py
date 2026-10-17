"""The progress plot must measure paired full-vocabulary success, without label shortcuts."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "viewing_plot", Path(__file__).resolve().parents[1] / "scripts/plot_viewing_learning.py"
)
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


def write(path, data):
    path.write_text(json.dumps(data))


@pytest.fixture
def run(tmp_path):
    data = {
        split: [
            dict(context_id=f"{split}{i}", context=f"text {split}{i}", question_family="q")
            for i in range(2)
        ]
        for split in ("train", "validation")
    }
    write(tmp_path / "data.json", data)
    write(
        tmp_path / "manifest.json",
        dict(
            viewing_mode="buffer",
            primary_path="split_inference",
            seed=17,
            training_status="in_progress",
        ),
    )
    write(
        tmp_path / "learner_input_equality.json",
        dict(splits={split: dict(pairs=2, bit_identical_pairs=1) for split in data}),
    )
    for split, records in data.items():
        rows = []
        for i, record in enumerate(records):
            for present in (False, True):
                positive = present and i == 0
                lp = np.log(0.75 if positive else 0.25)
                ln = np.log(0.25 if positive else 0.75)
                target = 1 if present else 0
                correct = int(positive) == target
                rows.append(
                    dict(
                        **record,
                        index=i,
                        condition_present=present,
                        task="detection",
                        answer_negative="No",
                        answer_positive="Yes",
                        negative_token_id=0,
                        positive_token_id=1,
                        target_token_id=target,
                        top_token_id=int(positive),
                        greedy_correct=correct,
                        logp_positive=lp,
                        logp_negative=ln,
                        positive_minus_negative=lp - ln,
                        ce=-(lp if present else ln),
                        full_vocab_correct_probability=np.exp(lp if present else ln),
                    )
                )
        write(
            tmp_path / f"{split}_0_split_inference.json",
            dict(summary=dict(step=0, split=split, path="split_inference"), rows=rows),
        )
    return tmp_path


def test_both_condition_success_ceiling_and_paired_bootstrap(run):
    report = PLOT.collect(run, draws=5000)
    point = report["curves"]["train"][0]
    assert point["mean"][0] == 0.5  # Individual accuracy is .75: deliberately not plotted.
    assert point["mean"][1] == pytest.approx(np.log(3))
    assert point["ci95"][0] == [0, 1]
    assert report["both_correct_ceilings"]["train"] == 0.5
    PLOT.draw(report, run / "figure")
    assert (run / "figure.pdf").exists()


def test_repeated_or_grouped_contexts_rejected(run):
    data = PLOT.read(run / "data.json")
    data["train"][1]["context"] = data["train"][0]["context"]
    write(run / "data.json", data)
    with pytest.raises(ValueError, match="unique"):
        PLOT.collect(run)


def test_false_greedy_correct_and_missing_condition_rejected(run):
    path = run / "train_0_split_inference.json"
    saved = PLOT.read(path)
    saved["rows"][0]["greedy_correct"] = False
    write(path, saved)
    with pytest.raises(ValueError, match="correctness"):
        PLOT.collect(run)
    saved["rows"][0]["greedy_correct"] = True
    saved["rows"].pop()
    write(path, saved)
    with pytest.raises(ValueError, match="two conditions"):
        PLOT.collect(run)


def test_input_equality_ceiling_cannot_be_violated(run):
    path = run / "learner_input_equality.json"
    saved = PLOT.read(path)
    saved["splits"]["train"]["bit_identical_pairs"] = 2
    write(path, saved)
    with pytest.raises(ValueError, match="ceiling"):
        PLOT.collect(run)
