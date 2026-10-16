import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from watching.probe import ProbeModel


@pytest.fixture
def small_model():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).to(torch.bfloat16).eval()
    ids = torch.randint(0, 64, (2, 8))
    return model, ids, torch.ones_like(ids), torch.arange(8).expand_as(ids)


def test_capture_does_not_replace_or_modify_activation(small_model):
    model, ids, mask, positions = small_model
    probe = ProbeModel(model, 0)
    with torch.no_grad():
        expected = model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            use_cache=False,
            logits_to_keep=1,
        ).logits[:, -1]
        absent, empty = probe(ids, mask, positions, False)
        present, captured = probe(ids, mask, positions, True)
    assert empty == ()
    assert len(captured) == 1
    assert torch.equal(absent, expected)
    assert torch.equal(present, absent)


def test_hook_traces_and_all_parameters_receive_gradients(small_model):
    model, ids, mask, positions = small_model
    probe = ProbeModel(model, 0)
    compiled = torch.compile(probe, backend="aot_eager", fullgraph=True, dynamic=False)
    for capture in [False, True, False, True]:
        logits, saved = compiled(ids, mask, positions, capture)
        assert len(saved) == int(capture)
        logits.float().square().mean().backward()
        assert all(p.grad is not None for p in model.parameters())
        model.zero_grad(set_to_none=True)


def test_probe_layer_is_validated(small_model):
    with pytest.raises(ValueError, match="outside the model"):
        ProbeModel(small_model[0], 2)
