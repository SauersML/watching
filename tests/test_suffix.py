"""Native replay and optimizer isolation, using a tiny real Qwen3 decoder."""

import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from watching.probe import ProbeModel
from watching.suffix import NativeSuffix, ResidualCapture, SuffixAdam, assert_identical


def tiny_model():
    torch.manual_seed(42)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        tie_word_embeddings=True,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    config.use_cache = False
    return Qwen3ForCausalLM(config).eval()


def inputs():
    return dict(
        input_ids=torch.tensor([[0, 0, 3, 4, 5]]),
        attention_mask=torch.tensor([[0, 0, 1, 1, 1]]),
        position_ids=torch.tensor([[0, 0, 0, 1, 2]]),
    )


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("capture_style", ["reference", "buffer"])
def test_native_boundary_replays_entire_vocabulary(capture, capture_style):
    model = tiny_model()
    baseline = ProbeModel(copy.deepcopy(model), 0)
    measured = ResidualCapture(model, 0, 0, capture_style)
    if capture_style == "buffer":
        measured.prepare_buffers(inputs())
    suffix = NativeSuffix(model, 0)
    with torch.no_grad():
        full, probe, state = measured(**inputs(), capture=capture)
        other, _ = baseline(**inputs(), capture=capture)
        replayed = suffix(*state, inputs()["position_ids"])
    assert len(probe) == int(capture)
    assert torch.equal(full, other)
    assert torch.equal(full, replayed)
    assert state[0].shape == (1, 5, 16)


def test_suffix_step_preserves_prefix_and_tied_head():
    model = tiny_model()
    measured = ResidualCapture(model, 0, 0)
    suffix = NativeSuffix(model, 0)
    before = {name: tensor.detach().clone() for name, tensor in model.named_parameters()}
    with torch.no_grad():
        _, _, state = measured(**inputs(), capture=True)
    state = tuple(None if t is None else t.detach().clone() for t in state)
    optimizer = SuffixAdam(suffix.active_parameters(), 1e-3)
    logits = suffix(*state, inputs()["position_ids"])
    torch.nn.functional.cross_entropy(logits, torch.tensor([7])).backward()
    optimizer.step()
    changed = []
    active = {id(p) for p in suffix.active_parameters()}
    for name, tensor in model.named_parameters():
        if not torch.equal(before[name], tensor):
            changed.append(name)
            assert id(tensor) in active
    assert changed
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert torch.equal(model.model.embed_tokens.weight, before["model.embed_tokens.weight"])
    assert all(p.grad is None for p in suffix.active_parameters())


def test_parity_failure_is_fatal():
    with pytest.raises(RuntimeError, match="parity failed"):
        assert_identical(torch.ones(2), torch.zeros(2), "replay")


def test_boundary_must_leave_trainable_layers_and_follow_probe():
    with pytest.raises(ValueError, match="Cut must"):
        ResidualCapture(tiny_model(), 1, 0)
    with pytest.raises(ValueError, match="Cut must"):
        ResidualCapture(tiny_model(), 1, 2)


def test_buffer_capture_requires_preparation_and_reuses_storage():
    measured = ResidualCapture(tiny_model(), 0, 0, "buffer")
    with pytest.raises(RuntimeError, match="Prepare"):
        with torch.no_grad():
            measured(**inputs(), capture=False)
    measured.prepare_buffers(inputs())
    with pytest.raises(RuntimeError, match="exactly once"):
        measured.prepare_buffers(inputs())
    with torch.no_grad():
        _, _, first = measured(**inputs(), capture=False)
        snapshot = tuple(None if t is None else t.clone() for t in first)
        changed = inputs()
        changed["input_ids"] = torch.tensor([[0, 0, 3, 4, 6]])
        _, _, second = measured(**changed, capture=False)
    assert first[0].data_ptr() == second[0].data_ptr()
    assert not torch.equal(snapshot[0], second[0])
    assert all("boundary_" not in name for name in measured.state_dict())
    assert all(not t.requires_grad for t in second if t is not None)


def test_buffer_capture_handles_none_attention_mask():
    measured = ResidualCapture(tiny_model(), 0, 0, "buffer")
    x = inputs()
    x["attention_mask"] = torch.ones_like(x["attention_mask"])
    measured.prepare_buffers(x)
    with torch.no_grad():
        _, _, state = measured(**x, capture=False)
        expected, _ = ProbeModel(copy.deepcopy(measured.model), 0)(**x, capture=False)
        replayed = NativeSuffix(measured.model, 0)(*state, x["position_ids"])
    torch.testing.assert_close(replayed, expected, rtol=0, atol=0)


def test_buffer_capture_survives_fullgraph_functionalization():
    measured = ResidualCapture(tiny_model(), 0, 0, "buffer")
    measured.prepare_buffers(inputs())
    compiled = torch.compile(measured, backend="aot_eager", fullgraph=True)
    # This checks graph capture and buffer mutation, not CUDA Inductor parity.
    with torch.no_grad():
        for capture in (False, True):
            expected, _, state = measured(**inputs(), capture=capture)
            snapshot = tuple(None if t is None else t.clone() for t in state)
            actual, probe, captured = compiled(**inputs(), capture=capture)
            assert len(probe) == int(capture)
            assert torch.equal(actual, expected)
            for value, saved in zip(captured, snapshot, strict=True):
                assert value is None if saved is None else torch.equal(value, saved)
