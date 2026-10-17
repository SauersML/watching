"""Execution-source capture must not leak mode labels or retain source graphs."""

import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from watching.execution_fit import NativeForward
from watching.execution_tail import ExecutionCapture, compare_states, select_records, source_forward
from watching.tail_train import EagerNativeTail, PairedMasterAdam, check_gradient_forward


def tiny():
    torch.manual_seed(71)
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
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    config.use_cache = False
    return Qwen3ForCausalLM(config).to(torch.bfloat16).eval().requires_grad_(True)


def inputs():
    return {
        "input_ids": torch.tensor([[0, 0, 3, 4, 5]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1]]),
        "position_ids": torch.tensor([[0, 0, 0, 1, 2]]),
    }


def test_interleaved_capture_preserves_native_values_and_cannot_chain_source_graphs():
    model = tiny()
    observed = copy.deepcopy(model)
    plain = NativeForward(model)
    capture = ExecutionCapture(observed, 2)
    batch = inputs()
    original_inputs = {k: v.clone() for k, v in batch.items()}
    capture.prepare(batch)
    stored = []
    for enabled in (False, True, True, False):
        logits, _ = source_forward(plain, batch, enabled)
        copied, state = source_forward(capture, batch, enabled, capture=True)
        assert torch.equal(logits, copied)
        assert all(x is None or (not x.requires_grad and x.grad_fn is None) for x in state)
        assert all(not b.requires_grad and b.grad_fn is None for b in capture.buffers())
        stored.append((logits, state))
    for a, b in ((0, 3), (1, 2)):
        assert torch.equal(stored[a][0], stored[b][0])
        compare_states(stored[a][1], stored[b][1])
    assert all(p.requires_grad and p.grad is None for p in model.parameters())
    assert all(p.requires_grad and p.grad is None for p in observed.parameters())
    assert all(torch.equal(batch[k], v) for k, v in original_inputs.items())
    assert not model.training and not observed.training


def test_detached_execution_cache_trains_only_existing_copied_tail():
    model = tiny()
    source_before = {k: p.clone() for k, p in model.named_parameters()}
    capture = ExecutionCapture(model, 2)
    batch = inputs()
    capture.prepare(batch)
    states = [source_forward(capture, batch, enabled, capture=True)[1] for enabled in (False, True)]
    learner = EagerNativeTail(model, 2)
    optimizer = PairedMasterAdam(learner.active_parameters(), 1e-3)
    for condition, state in enumerate(states):
        assert check_gradient_forward(learner, state, batch["position_ids"])["bit_identical"]
        logits = learner(*state, batch["position_ids"]).float()
        (torch.nn.functional.cross_entropy(logits, torch.tensor([7 + condition])) / 2).backward()
        optimizer.accumulate()
    norm, movement = optimizer.step()
    assert norm > 0 and movement > 0
    assert all(
        torch.equal(p, source_before[name]) and p.grad is None
        for name, p in model.named_parameters()
    )
    assert not learner.head_weight.requires_grad
    learner.install(model)
    changed = [
        name for name, p in model.named_parameters() if not torch.equal(p, source_before[name])
    ]
    assert changed and all(
        name.startswith(("model.layers.2.", "model.layers.3.", "model.norm.")) for name in changed
    )


def records():
    return {
        split: [
            {
                "context_id": split + str(i),
                "context": split + str(i),
                "text": "Same question in both modes",
                "question_family": "heldout" if split == "test" else "training",
                "wording_id": "train.form.execution.01",
            }
            for i in range(4)
        ]
        for split in ("train", "validation", "test")
    }


def test_records_preserve_single_condition_independent_prompt_and_heldout_families():
    raw = records()
    chosen = select_records(raw, 2, 2, 2, 9)
    assert all(row["task"] == "execution" for rows in chosen.values() for row in rows)
    assert all("task" not in row for rows in raw.values() for row in rows)
    assert {row["text"] for rows in chosen.values() for row in rows} == {
        "Same question in both modes"
    }
    raw["test"][0]["context_id"] = raw["train"][0]["context_id"]
    with pytest.raises(ValueError, match="overlapping"):
        select_records(raw, 4, 4, 4, 9)


def test_wrong_task_or_shared_test_question_family_rejected():
    raw = records()
    raw["train"][0]["wording_id"] = "train.form.probe.01"
    with pytest.raises(ValueError, match="execution-task"):
        select_records(raw, 4, 4, 4, 9)
    raw = records()
    raw["test"][0]["question_family"] = "training"
    with pytest.raises(ValueError, match="absent from training"):
        select_records(raw, 4, 4, 4, 9)


def test_conflicting_explicit_task_metadata_is_not_overridden_by_wording_id():
    raw = records()
    raw["train"][0]["task"] = "probe"
    with pytest.raises(ValueError, match="execution-task"):
        select_records(raw, 4, 4, 4, 9)


def test_explicit_prefix_reconstructs_native_model_without_observation_hooks():
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from watching.execution_tail import ExplicitNativePrefix, split_source_forward

    model = tiny()
    prefix = ExplicitNativePrefix(model, 2)
    learner = EagerNativeTail(model, 2)
    batch = inputs()
    snapshots = []
    for enabled in (False, True, True, False):
        with sdpa_kernel(SDPBackend.MATH):
            stock, _ = source_forward(NativeForward(model), batch, enabled)
            split, state = split_source_forward(prefix, learner, batch, enabled)
        assert torch.equal(stock, split)
        assert all(x is None or (not x.requires_grad and x.grad_fn is None) for x in state)
        snapshots.append(state)
    compare_states(snapshots[0], snapshots[3])
    compare_states(snapshots[1], snapshots[2])
    assert all(
        not layer._forward_pre_hooks and not layer._forward_hooks for layer in model.model.layers
    )
    assert all(p.requires_grad and p.grad is None for p in model.parameters())


def test_explicit_prefix_has_no_mode_argument_and_excludes_native_tail():
    import inspect

    from watching.execution_tail import ExplicitNativePrefix

    model = tiny()
    prefix = ExplicitNativePrefix(model, 2)
    batch = inputs()
    assert list(inspect.signature(prefix.forward).parameters) == [
        "input_ids",
        "attention_mask",
        "position_ids",
    ]
    with torch.no_grad():
        initial = prefix(**batch)
        model.model.layers[2].mlp.down_proj.weight.add_(1)
        model.model.norm.weight.add_(1)
        model.lm_head.weight.add_(1)
        after = prefix(**batch)
    compare_states(initial, after)
    with pytest.raises(ValueError, match="Boundary"):
        ExplicitNativePrefix(model, 0)
    with pytest.raises(ValueError, match="Boundary"):
        ExplicitNativePrefix(model, 4)


def test_explicit_prefix_compiles_both_autograd_contexts_without_graph_breaks():
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from watching.execution_tail import ExplicitNativePrefix, split_source_forward

    model = tiny()
    prefix = ExplicitNativePrefix(model, 2)
    tail = EagerNativeTail(model, 2)
    # CPU AOT coverage checks the actual returned-boundary contract. CUDA Inductor
    # numerics and footprint existence remain mandatory integration gates.
    compiled = torch.compile(prefix, backend="aot_eager", fullgraph=True, dynamic=False)
    for enabled in (False, True):
        with sdpa_kernel(SDPBackend.MATH):
            expected, state = split_source_forward(prefix, tail, inputs(), enabled)
            actual, copied = split_source_forward(compiled, tail, inputs(), enabled)
        assert torch.equal(actual, expected)
        compare_states(state, copied)


def test_explicit_prefix_training_changes_only_final_native_blocks_and_norm():
    from watching.execution_tail import ExplicitNativePrefix, split_source_forward

    model = tiny()
    prefix = ExplicitNativePrefix(model, 2)
    baseline_tail = EagerNativeTail(model, 2)
    learner = EagerNativeTail(model, 2)
    batch = inputs()
    states = [
        split_source_forward(prefix, baseline_tail, batch, enabled)[1] for enabled in (False, True)
    ]
    initial_parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = PairedMasterAdam(learner.active_parameters(), 1e-3)
    for condition, state in enumerate(states):
        assert check_gradient_forward(learner, state, batch["position_ids"])["bit_identical"]
        logits = learner(*state, batch["position_ids"]).float()
        (torch.nn.functional.cross_entropy(logits, torch.tensor([7 + condition])) / 2).backward()
        optimizer.accumulate()
    _, movement = optimizer.step()
    assert movement > 0
    assert all(torch.equal(p, initial_parameters[name]) for name, p in model.named_parameters())
    learner.install(model)
    for enabled, previous in zip((False, True), states, strict=True):
        _, current = split_source_forward(prefix, learner, batch, enabled)
        compare_states(current, previous)
    changed = [
        name for name, p in model.named_parameters() if not torch.equal(p, initial_parameters[name])
    ]
    assert changed and all(
        name.startswith(("model.layers.2.", "model.layers.3.", "model.norm.")) for name in changed
    )
    assert torch.equal(model.lm_head.weight, initial_parameters["lm_head.weight"])
