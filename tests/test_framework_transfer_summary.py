"""Paired scoring must not confuse answer bias, duplicate runs, or missing paths with transfer."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_framework_transfer.py"
spec = importlib.util.spec_from_file_location("framework_summary", SCRIPT)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def record(i, group="familiar"):
    return {"id": f"p{i}", "context_id": f"c{i}", "group": group}


def score(i, path, yes=0.6, no=0.3, top=1):
    return {
        "id": f"p{i}",
        "path": path,
        "yes_logprob": float(np.log(yes)),
        "no_logprob": float(np.log(no)),
        "top_token_id": top,
        "yes_id": 1,
        "no_id": 2,
    }


def test_separation_is_paired_and_actual_accuracy_uses_full_vocabulary():
    records = [record(0), record(1)]
    scores = {}
    for i in range(2):
        scores[(f"p{i}", "trainer_train")] = score(i, "trainer_train", top=1 if i == 0 else 9)
        scores[(f"p{i}", "trainer_predict")] = score(i, "trainer_predict", yes=0.3, no=0.6, top=2)
    measured = summary.pair_metrics(records, scores, "trainer_train", "trainer_predict", draws=100)
    assert measured["both_correct"]["mean"] == 0.5
    assert measured["marginal_accuracy"]["mean"] == 0.75
    assert measured["separation"]["mean"] == pytest.approx(2 * np.log(2))
    assert measured["correct_token_probability"]["mean"] == pytest.approx(0.6)
    assert measured["full_vocabulary_log_loss"]["mean"] == pytest.approx(-np.log(0.6))
    assert measured["conditional_binary_log_loss"]["mean"] == pytest.approx(-np.log(2 / 3))


def test_always_yes_has_zero_pair_accuracy_and_zero_separation():
    records = [record(i) for i in range(4)]
    scores = {
        (r["id"], path): score(i, path)
        for i, r in enumerate(records)
        for path in ("trainer_train", "trainer_predict")
    }
    measured = summary.pair_metrics(records, scores, "trainer_train", "trainer_predict", draws=100)
    assert measured["both_correct"]["mean"] == 0
    assert measured["both_correct"]["ci95"][1] > 0
    assert measured["separation"]["mean"] == 0
    assert measured["marginal_accuracy"]["mean"] == 0.5


def test_inference_framework_diagnostic_expects_no_in_both_paths():
    records = [record(0), record(1)]
    scores = {}
    for i in range(2):
        scores[(f"p{i}", "trainer_predict")] = score(i, "trainer_predict", yes=0.3, no=0.6, top=2)
        scores[(f"p{i}", "vllm")] = score(i, "vllm", top=1 if i == 0 else 2)
    measured = summary.inference_metrics(records, scores, draws=100)
    assert measured["expected_answer_both_paths"] == "No"
    assert measured["trainer_predict_no_accuracy"]["mean"] == 1
    assert measured["vllm_no_accuracy"]["mean"] == 0.5
    assert measured["top_token_disagreement"]["mean"] == 0.5
    assert measured["vllm_minus_predict_log_odds"]["mean"] == pytest.approx(2 * np.log(2))
    assert "both_correct" not in measured


def test_perfect_pair_accuracy_has_noncollapsed_exact_interval():
    records = [record(i) for i in range(4)]
    scores = {}
    for i, r in enumerate(records):
        scores[(r["id"], "trainer_train")] = score(i, "trainer_train")
        scores[(r["id"], "trainer_predict")] = score(i, "trainer_predict", yes=0.3, no=0.6, top=2)
    measured = summary.pair_metrics(records, scores, "trainer_train", "trainer_predict", draws=100)
    low, high = measured["both_correct"]["ci95"]
    assert 0 < low < high == 1


def test_repeats_excluded_duplicates_rejected(tmp_path):
    path = tmp_path / "scores.jsonl"
    primary = score(0, "trainer_train")
    path.write_text(json.dumps(primary) + "\n" + json.dumps({**primary, "repeat": True}) + "\n")
    assert len(summary.read_scores([path], [record(0)])) == 1
    path.write_text(json.dumps(primary) + "\n" + json.dumps(primary) + "\n")
    with pytest.raises(ValueError, match="Duplicate primary"):
        summary.read_scores([path], [record(0)])


@pytest.mark.parametrize(
    "update",
    [{"yes_logprob": float("nan")}, {"yes_logprob": 0.1}, {"no_id": 1}, {"top_token_id": -1}],
)
def test_invalid_rows_rejected(tmp_path, update):
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps({**score(0, "trainer_train"), **update}) + "\n")
    with pytest.raises(ValueError):
        summary.read_scores([path], [record(0)])


def test_smoke_excluded_and_incomplete_framework_is_not_scored(tmp_path):
    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps([record(0), record(1, "novel"), record(2, "smoke")]))
    rows = [score(i, path) for i in (0, 1, 2) for path in ("trainer_train", "trainer_predict")]
    rows.append(score(0, "vllm"))
    scores = tmp_path / "scores.jsonl"
    scores.write_text("\n".join(map(json.dumps, rows)))
    report = summary.summarize(dataset, [scores], draws=100)
    assert report["comparisons"]["trainer_prediction"]["all"]["n_contexts"] == 2
    assert "vllm_serving" not in report["comparisons"]
    assert report["unavailable"]["vllm_serving"]["missing_count"] == 1


def test_trl_scores_are_separate_from_plain_trainer(tmp_path):
    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps([record(0), record(1, "novel")]))
    rows = [score(i, "trl_train") for i in (0, 1)]
    rows += [
        score(i, path, yes=0.3, no=0.6, top=2) for i in (0, 1) for path in ("trl_predict", "vllm")
    ]
    scores = tmp_path / "trl.jsonl"
    scores.write_text("\n".join(map(json.dumps, rows)))
    report = summary.summarize(dataset, [scores], draws=100)
    assert report["comparisons"]["trl_prediction"]["all"]["both_correct"]["mean"] == 1
    assert report["comparisons"]["trl_vllm_serving"]["all"]["both_correct"]["mean"] == 1
    assert "trainer_prediction" not in report["comparisons"]


def test_frozen_dataset_has_unique_new_contexts_and_correct_groups():
    root = Path(__file__).parents[1]
    records = json.loads((root / "data/framework_transfer_v1/dataset.json").read_text())
    original = json.loads((root / "data/execution_tail_v1/dataset.json").read_text())
    assert len(records) == len({r["context_id"] for r in records}) == 68
    assert [sum(r["group"] == g for r in records) for g in ("smoke", "familiar", "novel")] == [
        4,
        32,
        32,
    ]
    old_texts = {r["text"] for rows in original.values() if isinstance(rows, list) for r in rows}
    old_families = {
        r["question_family"] for rows in original.values() if isinstance(rows, list) for r in rows
    }
    assert not old_texts.intersection(r["text"] for r in records)
    assert not old_families.intersection(
        r["question_family"] for r in records if r["group"] == "novel"
    )


def test_sglang_pairs_use_matching_hf_reference_and_inference_diagnostic(tmp_path):
    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps([record(0), record(1, "novel")]))
    rows = [score(i, path) for i in (0, 1) for path in ("trainer_train", "trainer_predict")]
    rows += [score(i, "sglang", yes=0.3, no=0.6, top=2) for i in (0, 1)]
    scores = tmp_path / "scores.jsonl"
    scores.write_text("\n".join(map(json.dumps, rows)))
    report = summary.summarize(dataset, [scores], draws=100)
    assert report["comparisons"]["sglang_serving"]["all"]["both_correct"]["mean"] == 1
    diagnostic = report["diagnostics"]["sglang_inference_frameworks"]["all"]
    assert diagnostic["expected_answer_both_paths"] == "No"
    assert diagnostic["sglang_no_accuracy"]["mean"] == 1
    assert diagnostic["trainer_predict_no_accuracy"]["mean"] == 0
    assert "vllm_serving" not in report["comparisons"]
