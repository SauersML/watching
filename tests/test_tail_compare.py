"""Tiny real Qwen3 checks for the shared-runtime comparison, without CUDA claims."""

import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from watching.execution_fit import tensor_digest
from watching.suffix import ResidualCapture
from watching.tail_compare import (
    ARMS,
    TRAIN_LAST,
    ComparedNativeTail,
    install_learned_suffix,
    learned_native_suffix,
    select_data,
    snapshot_native_suffix,
    suffix_digest,
    verify_cached_state,
)
from watching.tail_train import PairedMasterAdam, check_gradient_forward


def tiny_model(dtype=torch.float32):
    torch.manual_seed(17)
    config = Qwen3Config(
        vocab_size=24,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=10,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        attention_dropout=0,
    )
    config._attn_implementation = "sdpa"
    config.use_cache = False
    return Qwen3ForCausalLM(config).to(dtype).eval()


def cached_inputs(model):
    batch = {
        "input_ids": torch.tensor([[0, 3, 4, 5]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1]]),
        "position_ids": torch.tensor([[0, 0, 1, 2]]),
    }
    capture = ResidualCapture(model, 0, 1, "buffer")
    capture.prepare_buffers(batch)
    with torch.no_grad():
        _, _, state = capture(**batch, capture=True)
    return tuple(None if t is None else t.clone() for t in state), batch["position_ids"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_all_arms_have_identical_initial_native_forward_and_expected_active_blocks(dtype):
    model = tiny_model(dtype)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    state, positions = cached_inputs(model)
    initial = None
    for arm in ARMS:
        tail = ComparedNativeTail(model, arm)
        assert len(tail.layers) == 8
        for index, layer in enumerate(tail.layers):
            assert all(
                p.requires_grad == (index >= 8 - TRAIN_LAST[arm]) for p in layer.parameters()
            )
        assert tail.head_weight.requires_grad == (arm == "two_head")
        assert all(p.requires_grad for p in tail.active_parameters())
        assert check_gradient_forward(tail, state, positions)["bit_identical"]
        with torch.no_grad():
            logits = tail(*state, positions)
        if initial is None:
            initial = logits
        elif dtype == torch.bfloat16:
            # The experiment's native dtype must agree exactly.
            assert torch.equal(logits, initial)
        else:
            # CPU F.linear can choose different FP32 arithmetic for a weight
            # requiring gradients, even under no_grad with identical storage.
            # Production BF16 baseline comparisons still require exact hashes.
            torch.testing.assert_close(logits, initial, rtol=1e-6, atol=1e-7)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


@pytest.mark.parametrize("arm", ARMS)
def test_only_selected_native_parameters_update_and_artifact_omits_frozen_blocks(arm):
    model = tiny_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    base = snapshot_native_suffix(model)
    base_hash = suffix_digest(base)
    state, positions = cached_inputs(model)
    tail = ComparedNativeTail(model, arm)
    frozen_before = {
        k: v.clone() for k, v in tail.layers[: 8 - TRAIN_LAST[arm]].state_dict().items()
    }
    optimizer = PairedMasterAdam(tail.active_parameters(), 1e-3)
    for label in (7, 9):
        (
            torch.nn.functional.cross_entropy(tail(*state, positions), torch.tensor([label])) / 2
        ).backward()
        for index, layer in enumerate(tail.layers):
            assert all(
                (p.grad is not None) == (index >= 8 - TRAIN_LAST[arm]) for p in layer.parameters()
            )
        optimizer.accumulate()
    optimizer.step()
    assert all(
        torch.equal(v, frozen_before[k])
        for k, v in tail.layers[: 8 - TRAIN_LAST[arm]].state_dict().items()
    )
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    artifact = learned_native_suffix(
        tail, "tiny", 1, tensor_digest(model.lm_head.weight), base_hash
    )
    assert artifact["version"] == 2
    assert artifact["block_indices"] == list(range(10 - TRAIN_LAST[arm], 10))
    assert {int(k.split(".")[0]) for k in artifact["layers"]} == set(range(TRAIN_LAST[arm]))
    assert (artifact["head"] is not None) == (arm == "two_head")
    install_learned_suffix(model, artifact, base)
    for index in range(10):
        if index < 10 - TRAIN_LAST[arm]:
            assert all(
                torch.equal(v, before[f"model.layers.{index}.{k}"])
                for k, v in model.model.layers[index].state_dict().items()
            )
    assert torch.equal(model.lm_head.weight, tail.head_weight)
    assert torch.equal(model.model.embed_tokens.weight, before["model.embed_tokens.weight"])
    assert check_gradient_forward(tail, state, positions)["bit_identical"]


def test_transfer_install_resets_previous_arm_full_head_and_blocks():
    model = tiny_model()
    base = snapshot_native_suffix(model)
    digest = suffix_digest(base)
    head = tensor_digest(model.lm_head.weight)
    first = ComparedNativeTail(model, "two_head")
    with torch.no_grad():
        first.head_weight.add_(0.25)
        for p in first.active_parameters():
            p.add_(0.01)
    first_state = learned_native_suffix(first, "tiny", 1, head, digest)
    second = ComparedNativeTail(model, "four_fixed")
    second_state = learned_native_suffix(second, "tiny", 1, head, digest)
    install_learned_suffix(model, first_state, base)
    assert not torch.equal(model.lm_head.weight, base["head"])
    install_learned_suffix(model, second_state, base)
    assert torch.equal(model.lm_head.weight, base["head"])
    assert all(
        torch.equal(v, base["layers"][k]) for k, v in model.model.layers[-8:].state_dict().items()
    )
    bad = copy.deepcopy(second_state)
    bad["base_suffix_sha256"] = "wrong"
    with pytest.raises(ValueError, match="different base"):
        install_learned_suffix(model, bad, base)
    bad = copy.deepcopy(second_state)
    bad["block_indices"] = [8, 9]
    with pytest.raises(ValueError, match="block indices"):
        install_learned_suffix(model, bad, base)


def test_boundary_checks_reject_any_value_or_optional_mask_change():
    state = (torch.ones(2), None, torch.zeros(2), torch.ones(2))
    verify_cached_state(state, state, "cache")
    with pytest.raises(RuntimeError, match="presence"):
        verify_cached_state((state[0], torch.zeros(1), state[2], state[3]), state, "cache")
    with pytest.raises(RuntimeError, match="parity failed"):
        verify_cached_state((state[0] + 1, None, state[2], state[3]), state, "cache")


def test_cohort_selection_is_fixed_disjoint_and_excludes_training_question_families():
    raw = {
        split: [
            {
                "context_id": f"{split}:{i}",
                "text": f"context {split}:{i}",
                "question_family": ("heldout" if split == "test_wording" else "train") + str(i % 2),
            }
            for i in range(8)
        ]
        for split in ("train", "validation", "test_wording")
    }
    selected = select_data(raw, (4, 2, 4), 9, "test_wording")
    assert selected == select_data(raw, (4, 2, 4), 9, "test_wording")
    assert [len(selected[k]) for k in ("train", "validation", "test")] == [4, 2, 4]
    for row in raw["test_wording"]:
        row["question_family"] = "train0"
    with pytest.raises(ValueError, match="families"):
        select_data(raw, (4, 2, 4), 9, "test_wording")
