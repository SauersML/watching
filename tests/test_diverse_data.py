import re
from collections import Counter

import pytest

from watching.data import make_expanded_data
from watching.diverse_data import (
    CONTEXTS,
    HELDOUT_CONTEXTS,
    HELDOUT_QUESTIONS,
    TRAIN_QUESTIONS,
    make_diverse_data,
)
from watching.ood_data import OOD_CONTEXTS


@pytest.mark.parametrize("task", ["probe", "execution", "preference"])
def test_split_sizes_and_only_declared_context_overlap(task):
    data = make_diverse_data(task)
    assert data == make_diverse_data(task)
    assert {name: len(rows) for name, rows in data.items()} == {
        "train": 256,
        "validation": 64,
        "test": 64,
        "test_wording": 64,
        "ood": 64,
    }
    contexts = {split: {r["context"] for r in rows} for split, rows in data.items()}
    for split, rows in data.items():
        assert len(contexts[split]) == len(rows)
        assert len({r["context_id"] for r in rows}) == len(rows)
        assert len({r["text"] for r in rows}) == len(rows)
        for other in data:
            overlap = contexts[split] & contexts[other]
            if split == other or {split, other} == {"test", "test_wording"}:
                assert overlap == contexts[split]
            else:
                assert not overlap
    assert len({r["text"] for rows in data.values() for r in rows}) == 512
    assert set(CONTEXTS).isdisjoint(HELDOUT_CONTEXTS)
    for original, changed in zip(data["test"], data["test_wording"], strict=True):
        assert original["context"] == changed["context"]
        assert original["context_id"] == changed["context_id"]
        assert original["family"] == changed["family"]
        assert original["text"] != changed["text"]
        assert original["wording_id"].startswith("train:")
        assert changed["wording_id"].startswith("heldout:")


def test_families_wordings_and_ids_are_balanced_without_labels():
    data = make_diverse_data()
    for split, per_family in (("train", 32), ("validation", 8), ("test", 8)):
        assert Counter(row["family"] for row in data[split]) == dict.fromkeys(CONTEXTS, per_family)
        for family in CONTEXTS:
            wordings = Counter(r["wording_id"] for r in data[split] if r["family"] == family)
            assert wordings == {f"train:{i}": per_family // 8 for i in range(8)}
    for family in HELDOUT_CONTEXTS:
        wordings = Counter(r["wording_id"] for r in data["ood"] if r["family"] == family)
        assert wordings == {f"heldout:{i}": 4 for i in range(4)}
    for rows in data.values():
        for row in rows:
            assert set(row) == {"context", "text", "family", "context_id", "wording_id"}
            assert row["context_id"].startswith(row["family"] + ":")


def test_all_contexts_are_fresh_relative_to_previous_development_data():
    legacy = {context.casefold() for contexts in OOD_CONTEXTS.values() for context in contexts}
    legacy.update(r["context"].casefold() for rows in make_expanded_data().values() for r in rows)
    contexts = [
        text for family in (CONTEXTS, HELDOUT_CONTEXTS) for rows in family.values() for text in rows
    ]
    assert len(contexts) == 448
    assert len({text.casefold() for text in contexts}) == 448
    assert not {text.casefold() for text in contexts} & legacy
    assert all(len(values) == 48 for values in CONTEXTS.values())
    assert all(len(values) == 16 for values in HELDOUT_CONTEXTS.values())


@pytest.mark.parametrize("task", ["probe", "execution", "preference"])
def test_heldout_wording_is_absent_from_all_training_inputs(task):
    assert len(set(TRAIN_QUESTIONS[task])) == 8
    assert len(set(HELDOUT_QUESTIONS[task])) == 4
    assert set(TRAIN_QUESTIONS[task]).isdisjoint(HELDOUT_QUESTIONS[task])
    data = make_diverse_data(task)
    assert all(
        question not in row["text"]
        for question in HELDOUT_QUESTIONS[task]
        for split in ("train", "validation", "test")
        for row in data[split]
    )
    for split in ("test_wording", "ood"):
        assert all(
            any(row["text"].endswith(q) for q in HELDOUT_QUESTIONS[task]) for row in data[split]
        )


def test_preference_options_are_balanced_and_execution_condition_is_not_in_input():
    forbidden = re.compile(
        r"\b(gradient\w*|probe\w*|activation\w*|compiler\w*|monitor\w*|"
        r"execution|forward|torch|autograd|differentiation|hook\w*)\b",
        re.I,
    )
    data = make_diverse_data("preference")
    for rows in data.values():
        option_order = []
        for row in rows:
            assert not forbidden.search(row["text"])
            options = re.findall(r"\b(cat|owl)\b", row["text"])
            assert set(options) == {"cat", "owl"}
            option_order.append(options[0])
        assert option_order.count("cat") == option_order.count("owl")


def test_partition_seed_changes_membership_without_reusing_contexts():
    a, b = make_diverse_data(seed=1), make_diverse_data(seed=2)
    assert a["train"] != b["train"]
    assert {r["context"] for rows in a.values() for r in rows} == {
        r["context"] for rows in b.values() for r in rows
    }
    with pytest.raises(ValueError, match="Unknown task"):
        make_diverse_data("unsupported")
