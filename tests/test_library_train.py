"""CPU tests of raw-model capture adaptation and fail-closed controls."""

from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

from watching.library_train import LibraryInvocation, byte_equal, make_factory


class Output(NamedTuple):
    logits: torch.Tensor
    hidden_states: tuple | None


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_norm = torch.nn.Identity()

    def forward(self, hidden, **kwargs):
        return self.self_attn.q_norm(hidden.unsqueeze(-2)).squeeze(-2) + 1


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=1, head_dim=1, hidden_size=1)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Block(), Block(), Block()])

    def forward(
        self, input_ids, attention_mask, position_ids, use_cache=False, output_hidden_states=False
    ):
        hidden = input_ids.float().unsqueeze(-1)
        states = [hidden]
        for layer in self.model.layers:
            hidden = layer(
                hidden, attention_mask=attention_mask, position_embeddings=(hidden * 0, hidden * 0)
            )
            states.append(hidden)
        return Output(
            logits=hidden.repeat(1, 1, 3),
            hidden_states=tuple(states) if output_hidden_states else None,
        )


def inputs(value=1):
    return dict(
        input_ids=torch.tensor([[value, 2]]),
        attention_mask=torch.ones(1, 2),
        position_ids=torch.tensor([[0, 1]]),
    )


@pytest.mark.parametrize("method", ["forward_hook", "hidden_states"])
def test_raw_compile_and_fresh_capture(monkeypatch, method, tmp_path):
    targets = []

    def compile_raw(model, **kwargs):
        targets.append(model)
        assert kwargs == dict(fullgraph=True, dynamic=False)
        return model

    monkeypatch.setattr(torch, "compile", compile_raw)
    model, observed = Tiny(), Tiny()
    with torch.no_grad():
        native = LibraryInvocation(model, method, 0, inputs(), out=tmp_path)
        extract = LibraryInvocation(
            observed, method, 0, inputs(), cut=0, reference=native, out=tmp_path
        )
        for enabled in (False, True):
            logits, saved = native(**inputs(), capture=enabled)
            other, captures, state = extract(**inputs(), capture=enabled)
            assert torch.equal(logits, other)
            assert len(saved) == len(captures) == int(enabled)
            assert all(v is None or torch.isfinite(v).all() for v in state)
        _, first = native(**inputs(), capture=True)
        first = first[0].clone()
        _, second = native(**inputs(7), capture=True)
        assert not torch.equal(first, second[0])
    assert targets == [model, model, observed, observed]


def test_full_sequence_failure_archived(monkeypatch, tmp_path):
    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: model)
    with torch.no_grad():
        native = LibraryInvocation(Tiny(), "forward_hook", 0, inputs(), out=tmp_path)
        observed = LibraryInvocation(
            Tiny(), "forward_hook", 0, inputs(), cut=0, reference=native, out=tmp_path
        )
        native(**inputs(), capture=False)
        native.last_full[0, 0, 0] += 1  # Earlier position, outside trainer's final token.
        with pytest.raises(RuntimeError, match="full-sequence"):
            observed(**inputs(), capture=False)
    assert (tmp_path / "extraction_failure.pt").exists()


def test_factory_rejects_warm_start_before_model_access():
    args = SimpleNamespace(
        train_head=False,
        initial_tail="tail.pt",
        prior_results=None,
        warm_start_tail=None,
        warm_start_manifest=None,
    )
    with pytest.raises(ValueError, match="original base"):
        make_factory("forward_hook")(None, None, args, 0, {}, lambda *a, **k: None)


def test_byte_equality_distinguishes_signed_zero_and_dtype():
    assert not byte_equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert not byte_equal(torch.tensor([1.0]), torch.tensor([1.0], dtype=torch.float64))


@pytest.mark.parametrize("method", ["forward_hook", "hidden_states"])
def test_inductor_invalidates_bare_graph_after_registering_hooks(method, tmp_path):
    """Real code generation: adding hooks must not reuse the bare-model graph."""
    torch._dynamo.reset()
    with torch.no_grad():
        native = LibraryInvocation(Tiny(), method, 0, inputs(), out=tmp_path)
        observed = LibraryInvocation(
            Tiny(), method, 0, inputs(), cut=0, reference=native, out=tmp_path
        )
        for value in (1, 1, 7):
            for present in (False, True):
                observed.boundary = None
                logits, saved = native(**inputs(value), capture=present)
                copied, captured, state = observed(**inputs(value), capture=present)
                assert byte_equal(logits, copied)
                assert len(saved) == len(captured) == int(present)
                assert state is observed.buffers
                assert state[0][0, 0, 0].item() == value + 1
    torch._dynamo.reset()


@pytest.mark.parametrize("method", ["forward_hook", "hidden_states"])
def test_actual_qwen_dynamo_hooks_after_baseline(monkeypatch, method, tmp_path):
    """Use actual Dynamo dispatch to catch graph reuse after installing hooks."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    real_compile = torch.compile
    targets = []

    def compile_eager(model, **kwargs):
        targets.append(model)
        return real_compile(model, backend="eager", **kwargs)

    monkeypatch.setattr(torch, "compile", compile_eager)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(4)
    model = Qwen3ForCausalLM(config).eval()
    observed = Qwen3ForCausalLM(config).eval()
    observed.load_state_dict(model.state_dict())
    example = inputs()
    example["attention_mask"] = torch.tensor([[0, 1]])
    with torch.no_grad():
        native = LibraryInvocation(model, method, 0, example, out=tmp_path)
        extraction = LibraryInvocation(
            observed, method, 0, example, cut=0, reference=native, out=tmp_path
        )
        for present in (False, True, False, True):
            logits, captured = native(**example, capture=present)
            other, observed_capture, state = extraction(**example, capture=present)
            assert torch.equal(logits, other)
            assert len(captured) == len(observed_capture) == int(present)
            assert len(state) == 4
            assert extraction.requires_boundary and extraction.buffers is not None
        _, first = native(**example, capture=True)
        first = first[0].clone()
        changed = {**example, "input_ids": torch.tensor([[1, 7]])}
        _, second = native(**changed, capture=True)
        assert not torch.equal(first, second[0])
    assert targets == [model, model, observed, observed]
    torch._dynamo.reset()
