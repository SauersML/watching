"""CPU tests of the diagnostic's factor separation; no model download or GPU."""

from types import SimpleNamespace

import torch

from watching.library_call_audit import ARMS, DisabledHook, raw_kwargs, run_matrix


def test_omission_and_false_remain_different_signatures():
    assert raw_kwargs(False) == {"use_cache": False}
    assert raw_kwargs(True) == {"use_cache": False, "output_hidden_states": False}


def test_real_disabled_hook_preserves_output_and_never_reads():
    layer = torch.nn.Identity()
    hook = DisabledHook(layer)
    x = torch.tensor([2.0])
    assert layer(x) is x
    assert not hook.saved
    hook.enabled = True
    layer(x)
    assert torch.equal(hook.saved["activation"], x)
    hook.remove()
    assert not layer._forward_hooks


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(output_hidden_states=False, vocab_size=3)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(15)])
        self.model.layers[14].self_attn = torch.nn.Module()
        self.model.layers[14].self_attn.q_norm = torch.nn.Identity()
        self.calls = []

    def forward(self, input_ids, **kwargs):
        self.calls.append(dict(kwargs))
        x = self.model.layers[14].self_attn.q_norm(input_ids.float().unsqueeze(-1))
        return SimpleNamespace(logits=x.repeat(1, 1, 3))


def test_matrix_compiles_raw_model_and_keeps_hook_state_separate(monkeypatch, tmp_path):
    targets = []

    def compile_raw(model, **kwargs):
        assert kwargs == {"backend": "inductor", "fullgraph": True, "dynamic": False}
        from torch._dynamo.utils import counters

        counters["stats"]["unique_graphs"] = 1
        targets.append(model)
        return model

    monkeypatch.setattr(torch, "compile", compile_raw)
    monkeypatch.setattr(torch._dynamo, "reset", lambda: None)
    model = Tiny().eval()
    report = {}
    run_matrix(model, {"input_ids": torch.tensor([[1, 2]])}, tmp_path, report, (1, 2))
    assert targets == [model] * 5
    assert [r["name"] for r in report["arms"]] == [r[0] for r in ARMS]
    assert all(r["repeat"]["bit_identical"] for r in report["arms"])
    assert all(r["bit_identical"] for r in report["comparisons"].values())
    assert [r["hook_registered"] for r in report["arms"]] == [False, False, True, True, False]
    assert len(list(tmp_path.glob("*.safetensors"))) == 5
    assert report["attribution_admissible"] is True
    assert all(r["runtime_invariants_unchanged"] for r in report["arms"])
    assert not model.model.layers[14].self_attn.q_norm._forward_hooks


def test_tiny_real_hf_dynamo_keeps_all_four_signatures(monkeypatch, tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    real_compile = torch.compile

    def compile_eager(model, **kwargs):
        return real_compile(model, **{**kwargs, "backend": "eager"})

    monkeypatch.setattr(torch, "compile", compile_eager)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=15,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        attention_dropout=0.0,
    )
    model = Qwen3ForCausalLM(config).eval()
    ids = torch.tensor([[1, 2, 3]])
    report = {}
    run_matrix(
        model,
        {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "position_ids": torch.arange(3).unsqueeze(0),
        },
        tmp_path,
        report,
        (2, 3),
    )
    assert report["attribution_admissible"] is True
    assert all(r["single_graph_reused"] for r in report["arms"])
    torch._dynamo.reset()
