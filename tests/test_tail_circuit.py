"""Native tiny-model patch integrity and train-only selection tests."""

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from watching.tail_circuit import (
    capture_components,
    component_modules,
    group_specs,
    paired_effects,
    rank_heads,
    replace_units,
    role_position,
    run_patched,
)
from watching.tail_train import EagerNativeTail


@pytest.fixture
def example():
    torch.manual_seed(44)
    cfg = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        tie_word_embeddings=False,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(cfg).to(torch.bfloat16).eval()
    tail = EagerNativeTail(model)
    hidden = torch.randn(1, 5, 16).to(torch.bfloat16)
    mask = torch.ones(1, 1, 5, 5, dtype=torch.bool).tril()
    cos = torch.ones(1, 5, 4, dtype=torch.bfloat16)
    state = (hidden, mask, cos, torch.zeros_like(cos))
    return tail, state, torch.arange(5).unsqueeze(0)


def test_native_gqa_capture_self_patch_and_restoration_are_exact(example):
    tail, state, pos = example
    before_head = tail.head_weight.clone()
    baseline, features = capture_components(tail, state, pos)
    assert set(features) == {
        f"{b}:{site}" for b in (2, 3) for site in ("heads", "neurons", "attention", "mlp")
    }
    assert features["2:heads"].shape == (1, 5, 16)  # 4 query heads, not 2 KV heads
    assert features["2:neurons"].shape == (1, 5, 24)
    for name, value in features.items():
        out = run_patched(tail, state, pos, {name: {"donor": value}})
        assert torch.equal(out, baseline)
    for module, _ in component_modules(tail).values():
        assert not module._forward_hooks and not module._forward_pre_hooks
    assert torch.equal(run_patched(tail, state, pos, {}), baseline)
    assert torch.equal(tail.head_weight, before_head)


def test_head_slice_patch_preserves_tokens_unselected_heads_and_inputs():
    current = torch.arange(32).reshape(1, 2, 16).to(torch.bfloat16)
    donor = -current
    patched = replace_units(current, donor, heads=[1, 3], head_dim=4, position=-1)
    assert torch.equal(patched[:, 0], current[:, 0])
    for head in range(4):
        cols = slice(head * 4, (head + 1) * 4)
        expected = donor if head in (1, 3) else current
        assert torch.equal(patched[:, -1, cols], expected[:, -1, cols])
    assert torch.equal(current, torch.arange(32).reshape(1, 2, 16).to(torch.bfloat16))
    with pytest.raises(ValueError, match="Invalid"):
        replace_units(current, donor, heads=[4], head_dim=4, position=-1)
    with pytest.raises(ValueError, match="Invalid"):
        replace_units(current, donor, heads=[1, 1], head_dim=4, position=-1)


def test_native_group_hooks_use_donor_slices_and_cleanup_after_exception(example):
    tail, state, pos = example
    baseline, absent = capture_components(tail, state, pos)
    other_state = (state[0] + torch.tensor(0.25, dtype=torch.bfloat16), *state[1:])
    _, present = capture_components(tail, other_state, pos)
    group = [{"block": 2, "head": 0}, {"block": 2, "head": 1}, {"block": 3, "head": 2}]
    specs = group_specs(group, present, 4)
    assert specs["2:heads"]["heads"] == [0, 1]
    changed = run_patched(tail, state, pos, specs)
    assert torch.isfinite(changed).all()
    assert torch.equal(run_patched(tail, state, pos, group_specs(group, absent, 4)), baseline)
    with pytest.raises(ValueError, match="shape/dtype"):
        run_patched(tail, state, pos, {"3:attention": {"donor": present["3:attention"].float()}})
    assert torch.equal(run_patched(tail, state, pos, {}), baseline)
    for module, _ in component_modules(tail).values():
        assert not module._forward_hooks and not module._forward_pre_hooks


def test_directional_effects_report_absolute_nats_and_no_small_gap_shares():
    base = (torch.tensor([[0.0, -3.0]]), torch.tensor([[0.0, 5.0]]))
    patched = (torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0, 2.0]]))
    result = paired_effects(base, patched, (0, 1))
    assert result["sufficiency_nats"] == 4 and result["necessity_nats"] == 3
    assert result["sufficiency_share"] == 0.5 and result["necessity_share"] == 0.375
    identical = (base[0], base[0])
    result = paired_effects(identical, patched, (0, 1))
    assert result["necessity_nats"] == -5
    assert result["sufficiency_share"] is None and result["necessity_share"] is None


def test_ranking_preserves_negative_effects_is_balanced_and_rejects_test_data():
    rows = [
        {
            "split": "train",
            "kind": "single_head",
            "index": index,
            "block": 34,
            "head": head,
            "sufficiency_nats": value,
            "necessity_nats": value - 2,
        }
        for index in (1, 2)
        for head, value in ((0, 4), (1, -3), (2, 5))
    ]
    ranked = rank_heads(rows, 2)
    assert [r["head"] for r in ranked["selected"]] == [2, 0]
    assert ranked["all_ranked"][-1]["mean_bidirectional_nats"] == -4
    with pytest.raises(ValueError, match="training"):
        rank_heads([*rows, {**rows[0], "split": "test"}])
    with pytest.raises(ValueError, match="every training"):
        rank_heads(rows[:-1], 2)
    with pytest.raises(ValueError, match="Duplicate"):
        rank_heads([*rows, rows[0]], 2)


def test_role_position_uses_logical_not_physical_index():
    batch = {
        "input_ids": torch.tensor([[0, 0, 151644, 872, 198, 20]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1]]),
    }
    assert role_position(batch) == 3
    batch["input_ids"][0, 3] = 42
    with pytest.raises(ValueError, match="role tokens"):
        role_position(batch)
