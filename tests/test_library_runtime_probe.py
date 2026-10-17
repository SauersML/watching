"""Local evidence controls; no GPU, download, or original validator changes."""

import json

import torch

from watching.library_runtime_probe import RuntimeArgumentProbe, model_record, tensor_set


def test_nonpersistent_buffer_value_and_alias_coverage():
    model = torch.nn.Linear(2, 2).eval()
    model.register_buffer("rotary", torch.arange(4.0), persistent=False)
    model.register_buffer("rotary_view", model.rotary[1:], persistent=False)
    assert "rotary" not in model.state_dict()
    first = model_record(model)["tensors"]
    assert first["buffer:rotary"]["alias_group"] == first["buffer:rotary_view"]["alias_group"]
    assert first["buffer:rotary_view"]["storage_offset"] == 1
    with torch.no_grad():
        model.rotary.add_(1)
    second = model_record(model)["tensors"]
    assert first["buffer:rotary"]["sha256"] != second["buffer:rotary"]["sha256"]
    assert second["buffer:rotary"]["version"] > first["buffer:rotary"]["version"]


def test_layout_and_scalar_byte_hash():
    value = torch.arange(6).reshape(2, 3)
    records = tensor_set([("x", value), ("transpose", value.T), ("scalar", torch.tensor(2.0))])
    assert records["x"]["alias_group"] == records["transpose"]["alias_group"]
    assert records["transpose"]["stride"] == [1, 3]
    assert records["scalar"]["shape"] == []


def test_boxed_delegate_mutation_is_recorded(tmp_path):
    graph = torch.fx.symbolic_trace(torch.nn.Identity())

    def delegate(graph, examples):
        def compiled(args):
            value = args[0]
            args.clear()
            value.add_(1)
            return (value,)

        compiled._boxed_call = True
        return compiled

    probe = RuntimeArgumentProbe(tmp_path / "boxed", delegate)
    value = torch.tensor([1.0])
    probe(graph, [value])(value)
    evidence = json.loads((tmp_path / "boxed/call-0.json").read_text())
    assert (
        evidence["before"]["tensors"]["0"]["sha256"] != evidence["after"]["tensors"]["0"]["sha256"]
    )
    assert evidence["after"]["tensors"]["0"]["version"] == 1


def test_real_dynamo_records_runtime_not_fake_examples(tmp_path):
    model = torch.nn.Linear(2, 2).eval()
    probe = RuntimeArgumentProbe(tmp_path / "real", lambda graph, examples: graph.forward)
    call = torch.compile(model, backend=probe, fullgraph=True)
    value = torch.ones(1, 2)
    with torch.no_grad():
        first, second = call(value), call(value)
    assert torch.equal(first, second)
    assert probe.graphs == 1 and probe.calls == 2
    a = json.loads((tmp_path / "real/call-0.json").read_text())
    b = json.loads((tmp_path / "real/call-1.json").read_text())
    assert a["before"] == a["after"] == b["before"] == b["after"]
    assert len(a["before"]["tensors"]) == 3  # weight, bias and actual user input
    torch._dynamo.reset()


def test_inference_tensor_has_explicit_unavailable_version():
    with torch.inference_mode():
        value = torch.ones(2)
    record = tensor_set([("x", value)])["x"]
    assert record["inference_tensor"] is True
    assert record["version"] is None


def test_two_compile_runner_actual_arguments_and_sidecars(monkeypatch, tmp_path):
    from collections import namedtuple
    from types import SimpleNamespace

    from torch._dynamo.backends import registry

    from watching.library_runtime_audit import run_runtime_matrix

    Output = namedtuple("Output", ["logits"])

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(2, 3)
            self.config = SimpleNamespace(output_hidden_states=False, vocab_size=3)

        def forward(self, input_ids, use_cache=False):
            return Output(logits=self.proj(input_ids))

    monkeypatch.setattr(
        registry,
        "lookup_backend",
        lambda backend: backend if callable(backend) else lambda graph, examples: graph.forward,
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=0))
    model = Tiny().eval()
    initial = model_record(model)["tensors"]
    report = {
        "state_before": {"tensors": {k.split(":", 1)[1]: v["sha256"] for k, v in initial.items()}},
        "cpu_buffer_sha256": {},
    }
    run_runtime_matrix(model, {"input_ids": torch.ones(1, 2, 2)}, tmp_path, report, (0, 1))
    assert report["actual_runtime_argument_values_equal_all_calls"] is True
    assert report["independent_bare_comparison"]["bit_identical"] is True
    assert all(arm["actual_arguments_unchanged"] for arm in report["arms"])
    assert all(arm["model_values_unchanged"] for arm in report["arms"])
    assert all(arm["one_graph_two_calls"] for arm in report["arms"])
    assert len(list(tmp_path.glob("*.safetensors"))) == 2
    torch._dynamo.reset()


def test_nested_tensor_argument_does_not_claim_complete_coverage():
    from watching.library_runtime_probe import argument_records

    record = argument_records(([torch.tensor([1.0])],))
    assert record["coverage_complete"] is False
    assert record["unsupported_argument_indices"] == ["0"]


def test_graph_owned_constant_is_hashed(tmp_path):
    class WithConstant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("offset", torch.tensor([3.0]), persistent=False)

        def forward(self, value):
            return value + self.offset

    graph = torch.fx.symbolic_trace(WithConstant())
    probe = RuntimeArgumentProbe(tmp_path / "constant", lambda graph, examples: graph.forward)
    call = probe(graph, [torch.ones(1)])
    assert torch.equal(call(torch.ones(1)), torch.tensor([4.0]))
    constants = json.loads((tmp_path / "constant/graph-0-constants.json").read_text())
    assert constants["offset"]["sha256"] == tensor_set([("x", graph.offset)])["x"]["sha256"]


def test_nested_graph_constant_is_explicitly_unverified(tmp_path):
    root = torch.nn.Module()
    root.payload = {"hidden_tensor": torch.ones(1)}
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    graph.get_attr("payload")
    graph.output(value)
    module = torch.fx.GraphModule(root, graph)
    probe = RuntimeArgumentProbe(tmp_path / "unknown", lambda graph, examples: graph.forward)
    probe(module, [torch.ones(1)])
    constants = json.loads((tmp_path / "unknown/graph-0-constants.json").read_text())
    assert constants["payload"]["unverified_nonprimitive"] is True
