"""Independent evaluation checks with tiny real Qwen weights and explicit labels."""

import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from watching.execution_fit import tensor_digest
from watching.probe import ProbeModel
from watching.tail_train import difference, score
from watching.validate_tail import (
    ValidationProbe,
    comparison_rows,
    control_report,
    evaluation_records,
    install_checkpoint,
    paired_summary,
    task_summaries,
)


def tiny_model():
    torch.manual_seed(17)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        attention_dropout=0,
    )
    config._attn_implementation = "sdpa"
    config.use_cache = False
    return Qwen3ForCausalLM(config).to(torch.bfloat16).eval()


def inputs():
    return dict(
        input_ids=torch.tensor([[0, 0, 3, 4, 5]]),
        attention_mask=torch.tensor([[0, 0, 1, 1, 1]]),
        position_ids=torch.tensor([[0, 0, 0, 1, 2]]),
    )


def artifact(model, train_head=False):
    saved = {
        "version": 1,
        "checkpoint": "tiny",
        "first_layer": 2,
        "model_commit": None,
        "train_head": train_head,
        "layers": copy.deepcopy(model.model.layers[2:].state_dict()),
        "norm": copy.deepcopy(model.model.norm.state_dict()),
        "head_sha256": tensor_digest(model.lm_head.weight),
        "learned_head_sha256": tensor_digest(model.lm_head.weight),
        "head": None,
    }
    if train_head:
        saved["head"] = model.lm_head.weight.detach().clone() + 0.25
        saved["learned_head_sha256"] = tensor_digest(saved["head"])
    saved["layers"][next(iter(saved["layers"]))].add_(0.25)
    return saved


def test_modes_preserve_native_reference_semantics_and_buffer_really_writes():
    model = tiny_model()
    old = ProbeModel(copy.deepcopy(model), 0)
    probe = ValidationProbe(model, 0)
    probe.prepare_buffer(inputs())
    for mode in ("buffer", "reference", "none", "buffer", "none"):
        probe.probe_buffer.fill_(float("nan"))
        with torch.no_grad():
            logits, saved = probe(**inputs(), mode=mode)
            expected, old_saved = old(**inputs(), capture=mode == "reference")
        assert torch.equal(logits, expected)
        assert len(saved) == len(old_saved) == int(mode == "reference")
        if mode == "reference":
            assert torch.equal(saved[0], old_saved[0])
        if mode == "buffer":
            assert torch.isfinite(probe.probe_buffer).all()
        else:
            assert probe.probe_buffer.isnan().all()
    assert all(p.requires_grad and p.grad is None for p in model.parameters())


@pytest.mark.parametrize("train_head", [False, True])
def test_installs_native_tail_and_optional_full_head_without_changing_prefix(train_head):
    model = tiny_model()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    saved = artifact(model, train_head)
    result = install_checkpoint(model, saved, "tiny", 0)
    assert result["first_layer"] == 2
    assert result["head_sha256"] == saved["learned_head_sha256"]
    changes = [
        name for name, value in model.named_parameters() if not torch.equal(value, before[name])
    ]
    assert changes
    assert all(
        name.startswith(("model.layers.2.", "model.layers.3.", "model.norm."))
        or (train_head and name == "lm_head.weight")
        for name in changes
    )


@pytest.mark.parametrize("corruption", ["nan", "head_digest", "first_layer", "frozen_digest"])
def test_rejects_invalid_artifact_before_any_weight_changes(corruption):
    model = tiny_model()
    saved = artifact(model)
    if corruption == "nan":
        saved["norm"]["weight"][0] = float("nan")
    elif corruption == "head_digest":
        saved["head_sha256"] = "wrong"
    elif corruption == "first_layer":
        saved["first_layer"] = 0
    else:
        saved["learned_head_sha256"] = "wrong"
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        install_checkpoint(model, saved, "tiny", 0)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def record(index, task="detection"):
    return {
        "text": f"Prompt {index}",
        "context_id": f"context{index}",
        "context": f"Context {index}",
        "question_family": f"family{index % 2}",
        "task": task,
        "answer_negative": "No",
        "answer_positive": "Yes",
    }


def test_fresh_records_filter_tasks_and_reject_overlap_before_sampling():
    raw = {"test": [record(i) for i in range(8)] + [record(99, "animal")]}
    chosen = evaluation_records(raw, "test", 4, 7)
    assert len(chosen) == 4 and all(r["task"] == "detection" for r in chosen)
    assert chosen == evaluation_records(raw, "test", 4, 7)
    with pytest.raises(ValueError, match="overlap"):
        evaluation_records(raw, "test", 4, 7, [record(7)])
    renamed = {**record(7), "context_id": "renamed", "text": "Different question"}
    with pytest.raises(ValueError, match="overlap"):
        evaluation_records(raw, "test", 4, 7, [renamed])
    raw["test"][0]["answer_positive"] = "cat"
    with pytest.raises(ValueError, match="No/Yes"):
        evaluation_records(raw, "test", 4, 7)


def test_control_gate_uses_all_vocab_not_just_answer_rows():
    original = torch.tensor([[1.0, 2.0, 3.0]])
    changed = torch.tensor([[1.0, 2.0, 4.0]])
    assert control_report([difference(original, original)])["valid_no_footprint_control"]
    report = control_report([difference(original, changed)])
    assert not report["valid_no_footprint_control"]
    pair = comparison_rows({"none": original, "buffer": changed}, "buffer", 0, 0, 1)
    assert pair[0]["yes_minus_no"] == pair[1]["yes_minus_no"]
    assert pair[0]["logp_yes"] != pair[1]["logp_yes"]


def test_actual_accuracy_ci_does_not_reuse_forced_choice_ci():
    rows = []
    for i in range(4):
        rows.extend(
            [
                score(torch.tensor([[2.0, 1.0, 9.0]]), i, False, 0, 1),
                score(torch.tensor([[1.0, 2.0, 9.0]]), i, True, 0, 1),
            ]
        )
    result = paired_summary(rows, 17)
    assert result["accuracy"] == 1
    assert result["accuracy_ci95"] == [1, 1]
    assert result["greedy_accuracy"] == 0
    assert result["greedy_accuracy_ci95"] == [0, 0]
    assert result["mean_ce_ci95"][0] > 7
    assert paired_summary(rows[:2], 17)["greedy_accuracy_ci95"] is None


def animal_record(index):
    return {**record(index, "animal"), "answer_negative": "cat", "answer_positive": "owl"}


def test_joint_selection_balances_tasks_and_checks_every_requested_label():
    raw = {"test": [record(i) for i in range(8)] + [animal_record(i) for i in range(8, 16)]}
    chosen = evaluation_records(raw, "test", 8, 7, task="joint")
    assert len(chosen) == 8
    assert sum(r["task"] == "detection" for r in chosen) == 4
    assert sum(r["task"] == "animal" for r in chosen) == 4
    with pytest.raises(ValueError, match="overlap"):
        evaluation_records(raw, "test", 8, 7, [animal_record(15)], task="joint")
    with pytest.raises(ValueError, match="even total"):
        evaluation_records(raw, "test", 7, 7, task="joint")
    raw["test"][-1]["answer_positive"] = "Yes"
    with pytest.raises(ValueError, match="cat/owl"):
        evaluation_records(raw, "test", 8, 7, task="joint")


def test_task_summaries_use_native_task_labels_and_keep_tasks_separate():
    from watching.tail_train import score_record

    rows = []
    for index, (rec, ids) in enumerate([(record(0), (0, 1)), (animal_record(1), (2, 3))]):
        for present in (False, True):
            logits = torch.zeros(1, 5)
            logits[0, ids[int(present)]] = 4 if index == 0 else -4
            rows.append(score_record(logits, index, present, rec, ids))
    result = task_summaries(rows, 17)
    assert result["detection"]["greedy_accuracy"] == 1
    assert result["animal"]["greedy_accuracy"] == 0
    assert rows[-1]["target_token_id"] == 3
    assert rows[-1]["answer_positive"] == "owl"
    assert result["detection"]["paired_log_odds_shift"] > 0
    assert result["animal"]["paired_log_odds_shift"] < 0
