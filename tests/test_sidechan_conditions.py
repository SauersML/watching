import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from side.cond import condition, linears, rtn


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )
    return Qwen3ForCausalLM(config).to(torch.bfloat16).eval()


def weights(model):
    return [m.weight.detach().clone() for _, m in linears(model)]


def test_scaled_quantization_endpoints_and_restore(model):
    before = weights(model)
    with condition(model, "w-int8x0", 0):
        assert all(torch.equal(a, b) for a, b in zip(weights(model), before, strict=True))
    with condition(model, "w-int8", 0):
        full = weights(model)
    with condition(model, "w-int8x1", 0):
        assert all(torch.equal(a, b) for a, b in zip(weights(model), full, strict=True))
    with condition(model, "w-int4x1", 0):
        assert all(
            torch.equal(a, rtn(b, 4, 128)) for a, b in zip(weights(model), before, strict=True)
        )
    assert all(torch.equal(a, b) for a, b in zip(weights(model), before, strict=True))


def test_scaled_quantization_moves_weights_by_the_fraction(model):
    before = weights(model)
    with condition(model, "w-int8", 0):
        full = weights(model)
    with condition(model, "w-int8x0.25", 0):
        part = weights(model)
    step = sum(float((f.float() - b.float()).norm()) for f, b in zip(full, before, strict=True))
    moved = sum(float((p.float() - b.float()).norm()) for p, b in zip(part, before, strict=True))
    assert moved == pytest.approx(0.25 * step, rel=0.05)


def test_scaled_quantization_changes_only_the_condition_scope(model):
    before = weights(model)
    with condition(model, "w-int4x0.5", 0, upto=2):
        after = weights(model)
    names = [n for n, _ in linears(model)]
    for name, a, b in zip(names, after, before, strict=True):
        assert torch.equal(a, b) == (int(name.split(".")[0]) >= 2), name
