"""Observe real Python launches, including reused storage, without changing their code."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from watching.generated_trace import GeneratedCapture, chosen_kernel_metadata, plan_sites
from watching.mechanism_trace import compare_captures, external_inputs

SOURCE = '''
# Generated-kernel signature is retained as data, like an Inductor wrapper.
kernel_source = """
def triton_rsqrt_query(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    pass
"""
class Runner:
    def call(self, args):
        q, w, angles = args
        buf0 = empty_like(q)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(q, w, out=buf0)
        buf1 = empty_like(q)
        triton_rsqrt_query.run(buf0, w, angles, buf1, 16, stream=0)
        buf2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf1, q, q, None)
        result = buf2[0].clone()
        buf0.fill_(999)
        return result
'''


def make_runner(tmp_path):
    path = tmp_path / "generated.py"
    path.write_text(SOURCE)
    spec = importlib.util.spec_from_file_location("generated_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.empty_like = torch.empty_like
    module.assert_size_stride = lambda *args: None
    module.extern_kernels = SimpleNamespace(mm=lambda x, w, out: out.copy_(x * w))
    config = SimpleNamespace(kwargs={"XBLOCK": 16}, num_warps=4, num_stages=1)
    binary = SimpleNamespace(asm={"ptx": "original ptx"}, metadata=SimpleNamespace())
    kernel = SimpleNamespace(
        run=lambda q, w, a, out, n, stream: out.copy_(q + a),
        launchers=[SimpleNamespace(config=config, bin=binary)],
        fn=SimpleNamespace(arg_names=["in_ptr0", "in_ptr1", "in_ptr2", "out_ptr0", "xnumel"]),
    )
    module.triton_rsqrt_query = kernel
    attention = SimpleNamespace(default=lambda q, k, v, mask: (q + k + v,))
    module.torch = SimpleNamespace(
        ops=SimpleNamespace(aten=SimpleNamespace(_scaled_dot_product_efficient_attention=attention))
    )
    return module.Runner()


def inputs():
    return [
        torch.arange(16).reshape(4, 4).to(torch.bfloat16),
        torch.ones(4, dtype=torch.bfloat16),
        torch.ones(4, 4),
    ]


def test_live_capture_matches_full_output_and_snapshots_survive_reuse(tmp_path):
    runner = make_runner(tmp_path)
    expected = runner.call(inputs())
    with GeneratedCapture(1, 0) as capture:
        actual = runner.call(inputs())
    assert torch.equal(actual, expected)
    assert torch.equal(capture.snapshots["layer00.q_projection"], inputs()[0])
    assert len(capture.pipeline) == 1
    assert capture.pipeline[0]["scalars"] == (16,)
    physical, _ = external_inputs(capture)
    assert set(physical) == {"query_projection", "norm_weight", "rotary_angles"}
    with GeneratedCapture(1, 0) as other:
        runner.call(inputs())
    assert compare_captures(capture, other)["first_unequal_observed_boundary"] is None


def test_explicit_replacement_changes_only_that_execution(tmp_path):
    runner = make_runner(tmp_path)
    donor = torch.full((4, 4), 7, dtype=torch.bfloat16)
    baseline = runner.call(inputs())
    with GeneratedCapture(1, 0, replace_query=donor) as capture:
        result = runner.call(inputs())
    assert torch.equal(result, donor + 2 * inputs()[0])
    assert not torch.equal(result, baseline)
    assert capture.replacement_count == 1
    assert torch.equal(runner.call(inputs()), baseline)


def test_parser_fails_closed_when_projection_mapping_changes():
    with pytest.raises(ValueError, match="q projections"):
        plan_sites(SOURCE.replace("[linear]", "[unrecognized]"), 1, 0)


def test_selected_configuration_must_be_unique():
    with pytest.raises(RuntimeError, match="exactly one selected"):
        chosen_kernel_metadata(SimpleNamespace(launchers=[None, None]))


def test_real_saved_wrappers_if_available():
    root = Path("/Users/user/mpd-data/scratch/watching-mechanism-038/generated")
    if not root.exists():
        pytest.skip("Original generated wrapper bundle is a local optional integration fixture")
    counts = {}
    for path in root.rglob("*.py"):
        if path.name.startswith(("cvy3", "cctj3", "cyku")):
            sites = plan_sites(path.read_text(), 36, 14)
            counts[path.name[:4]] = sum(
                site.when == "before" and site.kernel is not None for site in sites
            )
    assert counts == {"cvy3": 1, "cctj": 1, "cyku": 2}


def test_materialized_two_stage_pipeline_tracks_inplace_alias(tmp_path):
    runner = make_runner(tmp_path)
    # Rebuild this generated wrapper using the same globals but a materialized norm.
    namespace = runner.call.__func__.__globals__
    source = SOURCE.replace(
        "def triton_rsqrt_query(in_ptr0, in_ptr1, in_ptr2, "
        "out_ptr0, xnumel, XBLOCK: tl.constexpr):",
        "def triton_rsqrt_norm(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):\n"
        "    pass\n"
        "def triton_rotary(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):",
    ).replace(
        "        triton_rsqrt_query.run(buf0, w, angles, buf1, 16, stream=0)",
        "        buf_alias = reinterpret_tensor(buf0, (4, 4))\n"
        "        triton_rsqrt_norm.run(buf_alias, w, 16, stream=0)\n"
        "        triton_rotary.run(buf_alias, angles, buf1, 16, stream=0)",
    )
    path = tmp_path / "materialized.py"
    path.write_text(source)
    base = namespace["triton_rsqrt_query"]
    namespace["triton_rsqrt_norm"] = SimpleNamespace(
        run=lambda q, w, n, stream: q.mul_(2),
        launchers=base.launchers,
        fn=SimpleNamespace(arg_names=["in_out_ptr0", "in_ptr0", "xnumel"]),
    )
    namespace["triton_rotary"] = SimpleNamespace(
        run=lambda q, a, out, n, stream: out.copy_(q + a),
        launchers=base.launchers,
        fn=SimpleNamespace(arg_names=["in_ptr0", "in_ptr1", "out_ptr0", "xnumel"]),
    )
    namespace["reinterpret_tensor"] = lambda x, shape: x.reshape(shape)
    exec(compile(source, str(path), "exec"), namespace)
    runner = namespace["Runner"]()
    with GeneratedCapture(1, 0) as capture:
        runner.call(inputs())
    assert len(capture.pipeline) == 2
    assert torch.equal(capture.pipeline[0]["inputs"]["in_out_ptr0"], inputs()[0])
    assert torch.equal(capture.pipeline[0]["outputs"]["in_out_ptr0"], 2 * inputs()[0])
    assert torch.equal(capture.pipeline[1]["inputs"]["in_ptr0"], 2 * inputs()[0])
    physical, bindings = external_inputs(capture)
    assert torch.equal(physical["query_projection"], inputs()[0])
    assert (1, "in_ptr0") not in bindings  # This is a produced intermediate, not a new input.


def test_attention_views_fail_closed_instead_of_capturing_root_storage():
    changed = SOURCE.replace(
        "default(buf1, q, q, None)",
        "default(reinterpret_tensor(buf1, (4, 4), (4, 1), 0), q, q, None)",
    )
    with pytest.raises(ValueError, match="direct materialized"):
        plan_sites(changed, 1, 0)


def test_metadata_recovers_selected_binary_when_store_cubin_is_disabled():
    from torch._inductor.runtime.runtime_utils import triton_hash_to_path_key

    config = SimpleNamespace(kwargs={"XBLOCK": 16}, num_warps=4, num_stages=1)
    binary = SimpleNamespace(hash="abc123", asm={"ptx": "selected PTX"}, metadata=SimpleNamespace())
    launcher = SimpleNamespace(config=config, cache_hash=triton_hash_to_path_key(binary.hash))
    result = SimpleNamespace(config=config, kernel=binary)
    kernel = SimpleNamespace(
        launchers=[launcher], compile_results=[result], fn=SimpleNamespace(arg_names=["in_ptr0"])
    )
    assert chosen_kernel_metadata(kernel)["ptx"] == "selected PTX"
    launcher.cache_hash = "incorrect"
    with pytest.raises(RuntimeError, match="uniquely match"):
        chosen_kernel_metadata(kernel)
