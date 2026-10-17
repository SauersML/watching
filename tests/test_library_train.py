"""CPU tests of raw-model capture adaptation and fail-closed controls."""

from types import SimpleNamespace

import pytest
import torch

from watching.library_train import LibraryInvocation, byte_equal, make_factory


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
        return SimpleNamespace(
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
