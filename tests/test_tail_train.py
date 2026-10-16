"""Real tiny Qwen3 checks for the separate native-tail experiment."""

import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from watching.suffix import ResidualCapture
from watching.tail_train import (
    EagerNativeTail,
    PairedMasterAdam,
    check_gradient_forward,
    difference,
    score,
    stratified_records,
    summarize,
)


def tiny_model(dtype=torch.float32):
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
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    config.use_cache = False
    return Qwen3ForCausalLM(config).to(dtype).eval()


def capture_state(model):
    inputs = dict(
        input_ids=torch.tensor([[0, 0, 3, 4, 5]]),
        attention_mask=torch.tensor([[0, 0, 1, 1, 1]]),
        position_ids=torch.tensor([[0, 0, 0, 1, 2]]),
    )
    capture = ResidualCapture(model, 0, 1, "buffer")
    capture.prepare_buffers(inputs)
    with torch.no_grad():
        _, _, state = capture(**inputs, capture=True)
    return tuple(None if t is None else t.clone() for t in state), inputs["position_ids"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_eager_math_tail_modes_match_and_native_head_has_no_gradient(dtype):
    model = tiny_model(dtype)
    flags = [p.requires_grad for p in model.parameters()]
    state, positions = capture_state(model)
    tail = EagerNativeTail(model, 2)
    assert tail.first_layer == 2
    assert tail.head_weight.data_ptr() == model.lm_head.weight.data_ptr()
    assert not tail.head_weight.requires_grad
    assert [p.requires_grad for p in model.parameters()] == flags
    audit = check_gradient_forward(tail, state, positions)
    assert audit["bit_identical"]
    torch.nn.functional.cross_entropy(tail(*state, positions).float(), torch.tensor([7])).backward()
    assert all(p.grad is not None for p in tail.active_parameters())
    assert all(p.grad is None for p in model.parameters())
    assert "head_weight" not in tail.state_dict()


def test_two_condition_update_changes_only_copied_tail_then_installs_native_blocks():
    model = tiny_model()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    state, positions = capture_state(model)
    tail = EagerNativeTail(model, 2)
    optimizer = PairedMasterAdam(tail.active_parameters(), 1e-3)
    for token in (7, 9):
        loss = torch.nn.functional.cross_entropy(tail(*state, positions), torch.tensor([token])) / 2
        loss.backward()
        optimizer.accumulate()
        assert all(p.grad is None for p in tail.active_parameters())
    norm, movement = optimizer.step()
    assert norm > 0 and movement > 0
    assert all(torch.equal(p, before[name]) for name, p in model.named_parameters())
    assert check_gradient_forward(tail, state, positions)["bit_identical"]
    tail.install(model)
    changed = [name for name, p in model.named_parameters() if not torch.equal(p, before[name])]
    assert changed
    assert all(
        name.startswith(("model.layers.2.", "model.layers.3.", "model.norm.")) for name in changed
    )
    assert torch.equal(model.lm_head.weight, before["lm_head.weight"])


def test_fp32_accumulator_retains_small_term_between_large_bf16_gradients():
    p = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
    optimizer = PairedMasterAdam([p], 1e-3)
    naive = torch.zeros_like(p)
    for value in (4096.0, 1.0, -4096.0):
        p.grad = torch.tensor([value], dtype=p.dtype)
        naive.add_(p.grad)
        optimizer.accumulate()
    assert float(naive) == 0
    assert float(optimizer.masters[0].grad) == 1
    with pytest.raises(RuntimeError, match="exactly two"):
        optimizer.step()


def test_stratified_prompt_sampling_is_repeatable_and_includes_available_families():
    rows = [
        {"question_family": f"family_{family}", "context": f"context_{family}_{i}"}
        for family in range(4)
        for i in range(10)
    ]
    chosen = stratified_records(rows, 8, 17)
    assert chosen == stratified_records(rows, 8, 17)
    assert len({r["source_index"] for r in chosen}) == 8
    assert {r["question_family"] for r in chosen} == {f"family_{i}" for i in range(4)}
    with pytest.raises(ValueError, match="count"):
        stratified_records(rows, 41, 17)


def test_scores_use_full_vocabulary_and_preserve_condition_pairing():
    rows = [score(torch.tensor([[0.0, 2.0, 7.0]]), 0, present, 0, 1) for present in (False, True)]
    report = summarize(rows, 17)
    assert report["paired_log_odds_shift"] == 0
    assert report["greedy_accuracy"] == 0
    assert report["mean_full_vocab_correct_probability"] < 0.01
    assert report["mean_ce"] > report["mean_binary_ce"]


def test_comparison_rejects_nonfinite_and_detects_non_answer_changes():
    a = torch.zeros(1, 4)
    b = copy.deepcopy(a)
    b[0, 3] = 0.5
    assert difference(a, b)["max_abs_difference"] == 0.5
    with pytest.raises(ValueError, match="finite"):
        difference(a, torch.full_like(a, float("nan")))


def test_two_condition_master_updates_match_fp32_reference_adam():
    parameter = torch.nn.Parameter(torch.tensor([0.25, -0.5], dtype=torch.bfloat16))
    optimizer = PairedMasterAdam([parameter], 1e-3)
    reference = torch.nn.Parameter(parameter.detach().float().clone())
    adam = torch.optim.AdamW([reference], lr=1e-3, weight_decay=0, foreach=False)
    contributions = [
        torch.tensor([0.125, 0.25], dtype=torch.bfloat16),
        torch.tensor([0.0625, -0.125], dtype=torch.bfloat16),
    ]
    for _ in range(3):
        before = parameter.detach().float().clone()
        for gradient in contributions:
            parameter.grad = gradient.clone()
            optimizer.accumulate()
        reference.grad = sum(g.float() for g in contributions)
        adam.step()
        adam.zero_grad(set_to_none=True)
        _, movement = optimizer.step()
        assert torch.equal(optimizer.masters[0], reference)
        assert torch.equal(parameter, reference.to(torch.bfloat16))
        assert movement == pytest.approx(float((parameter.detach().float() - before).norm()))
