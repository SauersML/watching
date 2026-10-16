"""Dataset separation, grouped selection, and paired transfer checks."""

import copy
from collections import Counter
from pathlib import Path

import pytest

from watching.family_data import (
    make_family_data,
    normalized_context,
    read_jsonl,
    validate_questions,
    validate_splits,
)

ROOT = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def sources():
    return {
        "contexts": read_jsonl(ROOT / "training_contexts_a.jsonl")
        + read_jsonl(ROOT / "training_contexts_b.jsonl"),
        "questions": read_jsonl(ROOT / "question_families_v1.jsonl"),
        "new_contexts": read_jsonl(ROOT / "new_contexts_v1.jsonl"),
        "ood_contexts": read_jsonl(ROOT / "ood_contexts_v1.jsonl"),
        "prior_contexts": read_jsonl(ROOT / "fresh_test_contexts.jsonl"),
    }


@pytest.fixture(scope="module")
def data(sources):
    return make_family_data(**sources)


def test_training_groups_balance_contexts_without_cross_group_duplicates(data):
    counts = Counter(row["selection_group"] for row in data["train"])
    assert len(counts) == 12
    assert sorted(counts.values()) == [85] * 8 + [86] * 4
    assert len({row["context_id"] for row in data["train"]}) == 1024
    assert all(row["selection_group"] == row["question_family"] for row in data["train"])
    assert len({row["wording_id"] for row in data["train"]}) == 48
    for family in {row["family"] for row in data["train"]}:
        group_counts = Counter(
            row["selection_group"] for row in data["train"] if row["family"] == family
        )
        assert max(group_counts.values()) - min(group_counts.values()) <= 1


def test_fresh_evaluation_and_paired_factorial_splits(data):
    assert {s: len(rows) for s, rows in data.items()} == {
        "train": 1024,
        "validation": 64,
        "test": 128,
        "test_wording": 128,
        "ood_context": 128,
        "ood": 128,
    }
    for a, b in (("test", "test_wording"), ("ood_context", "ood")):
        assert [(r["context_id"], r["context"]) for r in data[a]] == [
            (r["context_id"], r["context"]) for r in data[b]
        ]
        assert all(x["text"] != y["text"] for x, y in zip(data[a], data[b], strict=True))
        assert not {r["question_family"] for r in data[a]} & {r["question_family"] for r in data[b]}
    train_families = {r["family"] for r in data["train"]}
    assert {r["family"] for r in data["test"]} <= train_families
    assert not {r["family"] for r in data["ood"]} & train_families
    assert sorted(Counter(r["family"] for r in data["ood"]).values()) == [32] * 4


def test_previous_validation_is_reporting_only_and_not_a_selection_group(data):
    assert all(r["data_role"] == "development_reporting_only" for r in data["validation"])
    assert all(
        r["selection_group"] is None for s, rows in data.items() if s != "train" for r in rows
    )
    assert all("selection_group" not in r["text"] for rows in data.values() for r in rows)


def test_conditions_tasks_and_repeat_construction_share_context_group_assignment(sources, data):
    expected = [
        (r["context_id"], r["selection_group"], r["question_family"]) for r in data["train"]
    ]
    assert make_family_data(**sources) == data
    for task in ("execution", "preference"):
        other = make_family_data(**sources, task=task)
        assert [
            (r["context_id"], r["selection_group"], r["question_family"]) for r in other["train"]
        ] == expected
        for split in data:
            assert [r["context_id"] for r in other[split]] == [r["context_id"] for r in data[split]]


def test_rejects_new_context_that_was_previously_used(sources):
    for key in ("contexts", "prior_contexts"):
        changed = dict(sources)
        changed["new_contexts"] = [sources[key][0]]
        with pytest.raises(ValueError, match="overlaps"):
            make_family_data(**changed)


def test_rejects_normalized_duplicate_between_new_cohorts(sources):
    changed = dict(sources)
    row = sources["new_contexts"][0]
    changed["ood_contexts"] = [
        {"context": row["context"].upper().replace(".", "!"), "family": "new_structure"}
    ]
    with pytest.raises(ValueError, match="overlaps"):
        make_family_data(**changed)
    assert normalized_context("Hello,  WORLD!") == normalized_context("hello world")


def test_split_guard_rejects_accidental_training_test_overlap(data):
    changed = copy.deepcopy(data)
    changed["test"][0]["context"] = changed["train"][0]["context"]
    with pytest.raises(ValueError, match="overlap"):
        validate_splits(changed)


def test_bank_rejects_extra_placeholder_or_train_ood_family_reuse(sources):
    bank = copy.deepcopy(sources["questions"])
    bank[0]["text"] += " {label}"
    with pytest.raises(ValueError, match="placeholder"):
        validate_questions(bank)
    bank = copy.deepcopy(sources["questions"])
    train_family = next(r["family"] for r in bank if r["split_role"] == "train")
    ood_family = next(r["family"] for r in bank if r["split_role"] == "ood")
    for row in bank:
        if row["family"] == ood_family:
            row["family"] = train_family
    with pytest.raises(ValueError, match="disjoint"):
        validate_questions(bank)


def test_cli_rejects_overlength_without_creating_output(monkeypatch, tmp_path, capsys):
    import sys
    from types import SimpleNamespace

    from watching import family_data

    destination = tmp_path / "rejected.json"
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: object())),
    )
    monkeypatch.setattr(
        family_data, "rendered_lengths", lambda data, tokenizer: {split: [97] for split in data}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "family_data",
            "--task",
            "probe",
            "--contexts",
            str(ROOT / "training_contexts_a.jsonl"),
            str(ROOT / "training_contexts_b.jsonl"),
            "--question-bank",
            str(ROOT / "question_families_v1.jsonl"),
            "--new-contexts",
            str(ROOT / "new_contexts_v1.jsonl"),
            "--ood-contexts",
            str(ROOT / "ood_contexts_v1.jsonl"),
            "--out",
            str(destination),
            "--tokenizer",
            "unused",
            "--max-length",
            "96",
        ],
    )
    with pytest.raises(SystemExit) as error:
        family_data.main()
    assert error.value.code == 2
    assert "truncation is forbidden" in capsys.readouterr().err
    assert not destination.exists()
