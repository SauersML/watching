"""Preserve legacy rendering while checking explicitly paired chat variants."""

import copy
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from watching.messages import (
    record_messages,
    render_record,
    validate_message_splits,
    validate_prefix_groups,
)
from watching.tail_train import new_run_records, score_record
from watching.validate_tail import evaluation_records

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return json.loads((ROOT / "data/prefix_pilot_v1/dataset.json").read_text())["test"]


def test_legacy_render_arguments_are_exact_and_optional_messages_match():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
            return json.dumps(messages, ensure_ascii=False)

    tok = Tokenizer()
    for text in ("A simple question?", "Context:\nReply cat or owl.", "¿Cómo? Ω", ""):
        row = {"text": text}
        assert render_record(tok, row) == tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if text:
            assert render_record(tok, dict(row, messages=record_messages(row))) == render_record(
                tok, row
            )


def test_real_cached_tokenizer_legacy_ids_and_masks_are_bit_identical():
    cached = Path(
        "/Users/user/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    )
    if not cached.exists():
        pytest.skip("Optional real pinned tokenizer is not cached locally")
    tok = AutoTokenizer.from_pretrained(cached, local_files_only=True, padding_side="left")
    for row in fixture()[::4]:
        old = tok.apply_chat_template(
            [{"role": "user", "content": row["text"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        new = render_record(tok, {"text": row["text"]})
        assert old == new
        a = tok(
            old, add_special_tokens=False, padding="max_length", max_length=96, return_tensors="pt"
        )
        b = tok(
            new, add_special_tokens=False, padding="max_length", max_length=96, return_tensors="pt"
        )
        assert all(torch.equal(a[key], b[key]) for key in a)


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant", "content": "question"}],
        [{"role": "user", "content": "question", "label": True}],
        [{"role": "user", "content": "question"}, {"role": "system", "content": "secret"}],
        [{"role": "user", "content": "different"}],
    ],
)
def test_malformed_or_metadata_bearing_messages_rejected(messages):
    with pytest.raises(ValueError):
        record_messages({"text": "question", "messages": messages})


def test_prefix_groups_require_full_matched_set_and_shared_history():
    rows = fixture()[:4]
    assert validate_prefix_groups(rows)
    with pytest.raises(ValueError, match="four"):
        validate_prefix_groups(rows[:-1])
    changed = copy.deepcopy(rows)
    changed[3]["messages"][1]["content"] = "Different earlier question"
    with pytest.raises(ValueError, match="History"):
        validate_prefix_groups(changed)
    with pytest.raises(ValueError, match="overlap"):
        validate_message_splits({"train": rows, "test": rows})


def test_native_evaluation_accepts_complete_fixture_but_not_partial_groups():
    rows = fixture()
    selected = evaluation_records({"test": rows}, "test", 32, 17, task="joint")
    assert {r["context_id"] for r in selected} == {r["context_id"] for r in rows}
    with pytest.raises(ValueError, match="without subsampling"):
        evaluation_records({"test": rows}, "test", 16, 17, task="joint")


def test_scored_metadata_retains_base_identity_without_entering_model_inputs():
    row = fixture()[0]
    score = score_record(torch.tensor([[1.0, 2.0]]), 0, True, row, (0, 1))
    assert score["base_context_id"] == row["base_context_id"]
    assert score["group"] == row["group"]
    plain = score_record(torch.tensor([[1.0, 2.0]]), 0, True, {"text": "q"}, (0, 1))
    assert "base_context_id" not in plain


def test_new_run_selection_keeps_grouped_full_splits():
    data = json.loads((ROOT / "data/generalization_v4/dataset.json").read_text())
    selected = new_run_records(
        data, train_count=512, validation_count=64, test_count=128, test_split="test", seed=17
    )
    validate_message_splits(selected)
    assert [len(rows) for rows in selected.values()] == [512, 64, 128]
