"""Real tiny native-tail tests for selective loading and causal interventions."""

import copy
import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from watching.tail_mechanism import (
    capture_native,
    check_archived_response,
    diagnostic_selection,
    interpolate,
    intervention_summary,
    load_selective_tail,
    matched_direction,
    native_tail_from_tensors,
    patched_logits,
    response_scores,
    token_patch,
    vector_summary,
)
from watching.tail_train import EagerNativeTail


@pytest.fixture
def native_example():
    torch.manual_seed(12)
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
    model = Qwen3ForCausalLM(config).to(torch.bfloat16).eval()
    hidden = torch.randn(1, 5, 16).to(torch.bfloat16)
    mask = torch.ones(1, 1, 5, 5, dtype=torch.bool).tril()
    cos = torch.ones(1, 5, 8, dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)
    state = (hidden, mask, cos, sin)
    positions = torch.arange(5).unsqueeze(0)
    return model, state, positions


def selected_tensors(model):
    return {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if k.startswith(("model.layers.2.", "model.layers.3."))
        or k in ("model.norm.weight", "lm_head.weight")
    }


def test_selective_loading_matches_real_native_tail_without_prefix(native_example):
    model, state, positions = native_example
    loaded = native_tail_from_tensors(model.config, selected_tensors(model))
    original = EagerNativeTail(model)
    with torch.no_grad():
        assert torch.equal(loaded(*state, positions), original(*state, positions))
    assert len(loaded.layers) == 2
    assert loaded.first_layer == 2
    assert not loaded.head_weight.requires_grad
    assert all(p.requires_grad for p in loaded.parameters())
    assert all(not p.is_meta for p in loaded.parameters())
    assert not hasattr(loaded, "embed_tokens")


def test_local_safetensors_loader_selects_only_needed_tensors(tmp_path, native_example):
    model, state, positions = native_example
    model.config.save_pretrained(tmp_path)
    save_file(model.state_dict(), tmp_path / "model.safetensors")
    tail, _ = load_selective_tail(str(tmp_path), None, 2)
    with torch.no_grad():
        assert torch.equal(tail(*state, positions), EagerNativeTail(model)(*state, positions))


def test_selective_loader_rejects_tied_extra_and_nonfinite_weights(native_example):
    model, _, _ = native_example
    config = copy.deepcopy(model.config)
    config.tie_word_embeddings = True
    with pytest.raises(ValueError, match="untied"):
        native_tail_from_tensors(config, selected_tensors(model))
    tensors = selected_tensors(model)
    tensors["model.embed_tokens.weight"] = model.model.embed_tokens.weight
    with pytest.raises(ValueError, match="keys"):
        native_tail_from_tensors(model.config, tensors)
    tensors = selected_tensors(model)
    tensors["model.norm.weight"][0] = float("nan")
    with pytest.raises(ValueError, match="finite BF16"):
        native_tail_from_tensors(model.config, tensors)


def test_capture_and_swaps_preserve_every_vocabulary_logit(native_example):
    model, state, positions = native_example
    tail = EagerNativeTail(model)
    other = (state[0] + torch.tensor(0.125, dtype=torch.bfloat16), *state[1:])
    absent_logits, absent = capture_native(tail, state, positions)
    present_logits, present = capture_native(tail, other, positions)
    assert set(absent) == {"input", "block_2", "block_3", "final_norm"}
    for site in absent:
        assert torch.equal(
            patched_logits(tail, state, positions, site, present[site]), present_logits
        )
        assert torch.equal(
            patched_logits(tail, other, positions, site, absent[site]), absent_logits
        )
    assert not tail.norm._forward_hooks
    assert all(not layer._forward_hooks for layer in tail.layers)
    with torch.no_grad():
        assert torch.equal(tail(*state, positions), absent_logits)


def test_patch_cleanup_on_failure(native_example):
    model, state, positions = native_example
    tail = EagerNativeTail(model)
    with pytest.raises(ValueError, match="shape/dtype"):
        patched_logits(tail, state, positions, "block_2", state[0].float())
    assert not tail.layers[0]._forward_hooks


def test_interpolation_endpoints_and_exact_token_masks():
    absent = torch.tensor([[[1.0, -2.0], [0.5, 4.0]]], dtype=torch.bfloat16)
    present = absent + torch.tensor(0.125, dtype=torch.bfloat16)
    assert torch.equal(interpolate(absent, present, 0), absent)
    assert torch.equal(interpolate(absent, present, 1), present)
    mask = torch.tensor([[False, True]])
    patched = token_patch(absent, present, mask)
    assert torch.equal(patched[:, 0], absent[:, 0])
    assert torch.equal(patched[:, 1], present[:, 1])


def test_noise_matches_valid_norm_and_reports_actual_bf16_change():
    absent = torch.ones(1, 3, 4, dtype=torch.bfloat16)
    delta = torch.arange(12).reshape(1, 3, 4).float() / 100
    valid = torch.tensor([[False, True, True]])
    noise = torch.randn(delta.shape, generator=torch.Generator().manual_seed(12))
    matched = matched_direction(noise, delta, valid)
    assert torch.equal(matched[:, 0], torch.zeros_like(matched[:, 0]))
    assert float(matched.norm()) == pytest.approx(float(delta[valid].norm()), rel=1e-6)
    replacement = (absent.float() + matched).to(torch.bfloat16)
    summary = intervention_summary(absent, replacement, delta, valid)
    assert summary["realized_delta_l2"] == float(
        (replacement.float() - absent.float())[valid].norm()
    )
    with pytest.raises(ValueError, match="empty"):
        matched_direction(torch.zeros_like(noise), delta, valid)


def test_descriptive_projection_is_separate_from_native_response():
    absent = torch.zeros(1, 2, 3, dtype=torch.bfloat16)
    present = torch.ones_like(absent)
    result = vector_summary(
        absent, present, torch.tensor([1.0, 2.0, 3.0]), torch.ones(1, 2, dtype=torch.bool)
    )
    assert result["last_token_answer_projection_fp32"] == 6.0
    scores = response_scores(torch.tensor([[1.0, 2.0, 4.0]]), (0, 2))
    assert scores["yes_minus_no"] == 3.0
    assert scores["top_token_id"] == 2
    assert "accuracy" not in scores and "capture" not in scores


def test_selection_is_fixed_without_reading_outcomes():
    data = {"train": [{} for _ in range(20)], "test": [{} for _ in range(20)]}
    selected = diagnostic_selection(data, 8, 123)
    assert selected == diagnostic_selection(data, 8, 123)
    assert len(set(selected)) == 16
    assert sum(s == "test" for s, _ in selected) == 8
    with pytest.raises(ValueError, match="count"):
        diagnostic_selection(data, 21, 123)


def test_archived_scores_gate_replay_and_distinguish_missing_original_test(tmp_path):
    rows = [
        {"index": 4, "capture": present, "yes_minus_no": 2.0, "logp_yes": -0.2, "logp_no": -2.2}
        for present in (False, True)
    ]
    (tmp_path / "train_0_split_inference.json").write_text(json.dumps({"rows": rows}))
    result = check_archived_response(tmp_path, "original", "train", 4, 1024, rows)
    assert result == {"available": True, "exact": True, "mismatches": []}
    current = copy.deepcopy(rows)
    current[1]["yes_minus_no"] += 0.125
    result = check_archived_response(tmp_path, "original", "train", 4, 1024, current)
    assert not result["exact"] and len(result["mismatches"]) == 1
    assert check_archived_response(tmp_path, "original", "test", 4, 1024, rows) == {
        "available": False
    }
