"""Validate disjoint evaluation units, task mappings, and real input tokenization."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from watching.data import make_data, make_expanded_data
from watching.diverse_data import CONTEXTS, make_diverse_data
from watching.ood_data import OOD_CONTEXTS
from watching.validation_data import (
    ANSWERS,
    DATA_ROOT,
    SIZES,
    audit_tokenization,
    make_validation_data,
    normalized,
    question_bank,
    read_rows,
    validate_data,
)


def test_counts_balance_disjointness_and_determinism():
    data = make_validation_data()
    assert data == make_validation_data()
    assert {split: len(rows) for split, rows in data.items()} == SIZES
    assert len({row["context_id"] for rows in data.values() for row in rows}) == sum(SIZES.values())
    for rows in data.values():
        for task, answers in ANSWERS.items():
            selected = [r for r in rows if r["task"] == task]
            assert len(selected) == len(rows) // 2
            assert {(r["answer_negative"], r["answer_positive"]) for r in selected} == {answers}
        assert not any("condition_present" in row or "capture" in row for row in rows)


def test_frozen_artifact_matches_builder_and_manifest():
    root = DATA_ROOT / "validation_v2"
    path = root / "dataset.json"
    assert json.loads(path.read_text()) == make_validation_data()
    manifest = json.loads((root / "dataset.manifest.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["dataset_sha256"]
    for name, expected in manifest["source_sha256"].items():
        assert hashlib.sha256((DATA_ROOT / name).read_bytes()).hexdigest() == expected


def test_new_evaluation_contexts_do_not_repeat_prior_banks():
    prior = set()
    for path in DATA_ROOT.glob("*.jsonl"):
        prior.update(normalized(r["context"]) for r in read_rows(path) if "context" in r)
    for groups in (CONTEXTS, OOD_CONTEXTS):
        prior.update(normalized(text) for texts in groups.values() for text in texts)
    for bank in (make_data(), make_expanded_data()):
        prior.update(normalized(r["context"]) for rows in bank.values() for r in rows)
    data = make_validation_data()
    for split in ("validation", "test", "challenge"):
        assert not {normalized(r["context"]) for r in data[split]} & prior
    old_eval = {
        normalized(row["context"])
        for bank in (make_data(), make_expanded_data(), make_diverse_data())
        for split, rows in bank.items()
        if split != "train"
        for row in rows
    }
    assert not {normalized(r["context"]) for r in data["train"]} & old_eval


def test_animal_questions_do_not_describe_the_runtime_condition():
    rows = read_rows(DATA_ROOT / "validation_v2/questions.jsonl")
    bank = question_bank(rows)
    for families in bank.values():
        for tasks in families.values():
            animal = tasks["animal"]
            assert "cat or owl" in animal[0]["text"]
            assert "owl or cat" in animal[1]["text"]
            for row in animal:
                text = row["text"].lower()
                assert not any(
                    word in text
                    for word in (
                        "activation",
                        "probe",
                        "gradient",
                        "forward pass",
                        "monitor",
                        "capture",
                    )
                )


def test_invalid_family_overlap_and_context_reuse_are_rejected():
    data = make_validation_data()
    bad = deepcopy(data)
    bad["test"][0]["question_family"] = data["train"][0]["question_family"]
    with pytest.raises(ValueError, match="families overlap"):
        validate_data(bad)
    bad = deepcopy(data)
    bad["test"][0] = deepcopy(data["train"][0])
    with pytest.raises(ValueError, match="must all be unique"):
        validate_data(bad)
    rows = read_rows(DATA_ROOT / "validation_v2/questions.jsonl")
    rows[0]["text"] += " {context}"
    with pytest.raises(ValueError, match="exactly one"):
        question_bank(rows)


class TinyTokenizer:
    chat_template = "test chat formatting"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return "user:" + messages[0]["content"] + "assistant:"

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        if text in ("No", "Yes", "cat", "owl"):
            return [("No", "Yes", "cat", "owl").index(text)]
        return list(text.encode())


def test_token_audit_rejects_overlength_and_token_collisions():
    data = make_validation_data()
    with pytest.raises(ValueError, match="exceeds"):
        audit_tokenization(data, TinyTokenizer(), max_length=1)
    audit = audit_tokenization(data, TinyTokenizer(), max_length=1000)
    assert audit["truncated_prompts"] == audit["tokenized_collisions"] == 0
    duplicate = deepcopy(data)
    duplicate["test"][0]["text"] = duplicate["train"][0]["text"]
    with pytest.raises(ValueError, match="Tokenized prompts collide"):
        audit_tokenization(duplicate, TinyTokenizer(), max_length=1000)


def test_cached_qwen_tokenizer_and_frozen_dataset():
    path = Path("/Users/user/mpd-data/checkpoints/watching-sft-003r")
    if not path.exists():
        pytest.skip("Local Qwen tokenizer is unavailable; no downloads in data tests")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    audit = audit_tokenization(make_validation_data(), tokenizer, max_length=96)
    assert audit["answer_token_ids"] == {"No": 2753, "Yes": 9454, "cat": 4616, "owl": 9605}
    assert max(row["max"] for row in audit["rendered_lengths"].values()) <= 96
