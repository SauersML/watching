import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from watching.control_probes import StorageControlProbe, buffer_write_sites, materialize


def test_opaque_identity_really_stores_without_changing_bits():
    value = torch.tensor([0.0, -0.0, 1.234, float("inf")], dtype=torch.bfloat16)
    out = materialize(value)
    assert out.data_ptr() != value.data_ptr()
    assert torch.equal(out.view(torch.int16), value.view(torch.int16))


def test_all_controls_eager_preserve_logits_and_buffer_is_correct():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=16,
        attention_dropout=0,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(19)
    model = Qwen3ForCausalLM(config).to(torch.bfloat16).eval()
    wrapper = StorageControlProbe(model, 0)
    inputs = dict(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        position_ids=torch.tensor([[0, 1, 2]]),
    )
    wrapper.prepare_buffer(inputs)
    with torch.no_grad():
        baseline, reference = wrapper(**inputs, mode="reference")
        for mode in ("none", "buffer", "identity", "identity_reference"):
            wrapper.probe_buffer.fill_(float("nan"))
            logits, saved = wrapper(**inputs, mode=mode)
            assert torch.equal(logits, baseline)
            assert len(saved) == int(mode == "identity_reference")
            if saved:
                assert torch.equal(saved[0], reference[0])
            if mode == "buffer":
                assert torch.equal(wrapper.probe_buffer, reference[0])
            else:
                assert wrapper.probe_buffer.isnan().all()


def test_custom_op_survives_fullgraph_tracing():
    seen = []

    def backend(graph, inputs):
        seen.append(graph)
        return graph.forward

    @torch.compile(backend=backend, fullgraph=True)
    def computation(x):
        return materialize(x) * 2

    x = torch.randn(3, dtype=torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(computation(x), x * 2)
    assert any(
        "watching_controls.materialize" in str(node.target)
        for graph in seen
        for node in graph.graph.nodes
    )


def test_writer_mapping_uses_actual_store_pointer_and_launch_binding():
    kernel = "def k(in_ptr0, out_ptr0, n):\n    tl.store(out_ptr0 + 1, tl.load(in_ptr0))"
    source = (
        f"k = async_compile.triton('k', {kernel!r})\n"
        "def call(args):\n"
        "    x, capture = args\n"
        "    k.run(x, capture, 32, stream=stream)\n"
    )
    assert buffer_write_sites(source) == {4: ("k", ["capture"])}
