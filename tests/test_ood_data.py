import re

import pytest

from watching.data import make_expanded_data
from watching.ood_data import (
    OOD_CONTEXTS,
    OOD_QUESTIONS,
    TRAIN_EXECUTION_QUESTION,
    make_ood_data,
    rendered_lengths,
)


@pytest.mark.parametrize("task", ["probe", "execution", "preference"])
def test_splits_are_deterministic_disjoint_and_unlabeled(task):
    data = make_ood_data(task)
    assert data == make_ood_data(task)
    assert {s: len(rows) for s, rows in data.items()} == {
        "train": 256,
        "validation": 32,
        "test": 64,
        "ood": 128,
    }
    contexts = [row["context"] for rows in data.values() for row in rows]
    texts = [row["text"] for rows in data.values() for row in rows]
    assert len(set(contexts)) == len(contexts)
    assert len(set(texts)) == len(texts)
    for split, rows in data.items():
        expected = (
            {"context", "text", "family", "family_index"}
            if split == "ood"
            else {
                "context",
                "text",
            }
        )
        assert all(set(row) == expected for row in rows)
    assert {
        family: sum(row["family"] == family for row in data["ood"]) for family in OOD_CONTEXTS
    } == dict.fromkeys(OOD_CONTEXTS, 16)


def test_original_probe_training_prompts_are_preserved_exactly():
    original = make_expanded_data()
    data = make_ood_data("probe")
    for split in ("train", "validation", "test"):
        assert data[split] == original[split][: len(data[split])]
    execution = make_ood_data("execution")
    for row in execution["train"]:
        assert row["text"] == f"Read this passage:\n{row['context']}\n\n{TRAIN_EXECUTION_QUESTION}"


@pytest.mark.parametrize("task", ["probe", "execution", "preference"])
def test_ood_uses_new_wording_and_structures_with_the_same_contexts(task):
    data = make_ood_data(task)
    training_texts = [row["text"] for row in data["train"]]
    assert all(question not in text for question in OOD_QUESTIONS[task] for text in training_texts)
    assert len(set(OOD_QUESTIONS[task])) == 8
    for row in data["ood"]:
        question = OOD_QUESTIONS[task][list(OOD_CONTEXTS).index(row["family"])]
        assert question in row["text"]
        if row["family"] == "question_before_context":
            assert row["text"].startswith(question)
        elif row["family"] == "short_code":
            assert row["text"].startswith("```python\n")
        elif row["family"] == "concise_paraphrase":
            assert "Read this passage" not in row["text"]
    assert [row["context"] for row in data["ood"]] == [
        row["context"] for row in make_ood_data("probe")["ood"]
    ]


def test_preference_inputs_do_not_ask_about_execution_or_monitoring():
    forbidden = re.compile(
        r"\b(gradient\w*|probe\w*|activation\w*|compiler\w*|monitor\w*|"
        r"execution|forward|torch|autograd|differentiation|hook\w*)\b",
        re.I,
    )
    for rows in make_ood_data("preference").values():
        for row in rows:
            assert not forbidden.search(row["text"])
            assert re.search(r"\bcat\b", row["text"])
            assert re.search(r"\bowl\b", row["text"])


def test_length_check_measures_rendered_messages_without_truncation():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            return "prefix:" + messages[0]["content"] + ":suffix"

        def encode(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": False}
            return list(text)

    result = rendered_lengths({"test": [{"text": "abc"}, {"text": "longer text"}]}, Tokenizer())
    assert result == {"test": [len("prefix:abc:suffix"), len("prefix:longer text:suffix")]}


@pytest.mark.parametrize("kwargs", [{"task": "other"}, {"train": 0}, {"test": 513}])
def test_invalid_dataset_requests_fail(kwargs):
    with pytest.raises(ValueError):
        make_ood_data(**kwargs)
