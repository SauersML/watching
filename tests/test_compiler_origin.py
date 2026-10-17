import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from watching.compiler_origin import (
    Operation,
    ParameterHashes,
    compare_early,
    plan_early_operations,
    replay_common_operands,
    tensor_expression,
    tensor_layout,
)
from watching.execution_fit import tensor_digest

SOURCES = Path(__file__).parents[1] / "results/compiler-mechanism-042r/compiler_sources"


@pytest.mark.parametrize(
    "filename",
    [
        "212913f5994a2c54bfc050b2f81257a860d0780386a7dbef888b12e18e7bc18b.py.txt",
        "17e9b1279afdf108c4a69f83610827c7ab6405cd77ffe6e7c3483f72d564fe52.py.txt",
    ],
)
def test_actual_archived_native_launches_are_covered(filename):
    operations = plan_early_operations((SOURCES / filename).read_text(), 36)
    assert sum(op.kind == "attention" for op in operations) == 3
    assert sum(op.kind == "mm" for op in operations) == 22  # 3 * 7 plus rotary angles
    assert any(op.kernel and op.kernel.endswith("_13") for op in operations)
    assert operations[-1].kind == "triton"  # next-block norm precedes q projection
    assert len({op.line for op in operations}) == len(operations)


def test_views_preserve_original_storage_offset_and_strides():
    base = torch.arange(40).reshape(5, 8)[1:]
    expression = ast.parse("reinterpret_tensor(x, (2, 3), (8, 2), 1)", mode="eval").body
    value = tensor_expression(expression, {"x": base})
    assert value.storage_offset() == 9
    assert value.stride() == (8, 2)
    assert torch.equal(value, torch.tensor([[9, 11, 13], [17, 19, 21]]))
    with pytest.raises(ValueError, match="Unsupported materialized"):
        tensor_expression(ast.parse("x.clone()", mode="eval").body, {"x": base})


def test_public_alignment_has_residues_not_absolute_addresses():
    value = torch.randn(4, 8)
    layout = tensor_layout(value)
    assert set(layout) == {"shape", "stride", "dtype", "storage_offset", "alignment_mod_bytes"}
    for modulus, residue in layout["alignment_mod_bytes"].items():
        assert 0 <= residue < int(modulus)


def test_parameter_digest_cache_invalidates_after_mutation():
    cache = ParameterHashes()
    value = torch.ones(8)
    before = cache.digest(value)
    assert cache.digest(value) == before
    value.add_(1)
    assert cache.digest(value) != before
    assert len(cache.values) == 2


class Kernel:
    def __init__(self, amount):
        self.amount = amount
        self.fn = SimpleNamespace(arg_names=["in_ptr0", "out_ptr0", "xnumel"])
        self.addresses = []

    def run(self, x, out, count, *, stream):
        assert count == x.numel()
        self.addresses.append((x.data_ptr(), out.data_ptr()))
        out.copy_(x + self.amount)


def fixture_capture(amount, source):
    source = source.clone()
    output = source + amount
    operation = Operation(0, 1, "triton", "kernel", (), (), (source.numel(),))

    def metadata(value):
        return {
            **tensor_layout(value),
            "aliases_same_phase": [],
            "aliases_inputs": [],
            "sha256": tensor_digest(value),
        }

    return SimpleNamespace(
        operations=[operation],
        kernel_objects={"kernel": Kernel(amount)},
        records={
            0: {"inputs": {"in_ptr0": metadata(source)}, "outputs": {"out_ptr0": metadata(output)}}
        },
        tensors={0: {"inputs": {"in_ptr0": source}, "outputs": {"out_ptr0": output}}},
    )


def cpu_replay(monkeypatch):
    monkeypatch.setattr(
        "watching.compiler_origin._allocate_operand",
        lambda layout, dtype, device: torch.empty_strided(
            layout["shape"], layout["stride"], dtype=dtype
        ),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=0))


def test_replay_crosses_both_implementations_on_same_physical_buffers(monkeypatch):
    cpu_replay(monkeypatch)
    left = fixture_capture(1, torch.arange(6, dtype=torch.float32))
    right = fixture_capture(2, torch.arange(6, dtype=torch.float32) + 10)
    result, raw = replay_common_operands(left, right, 0)
    assert result["own_input_replay_bit_identical"]
    assert result["shared_physical_inputs_and_outputs"]
    addresses = left.kernel_objects["kernel"].addresses + right.kernel_objects["kernel"].addresses
    assert len(addresses) == 8 and len(set(addresses)) == 1
    assert not result["comparisons"][0]["outputs"]["out_ptr0"]["bit_identical"]
    assert torch.equal(raw[0][0]["out_ptr0"], left.tensors[0]["outputs"]["out_ptr0"])


def test_replay_rejects_failed_own_input_reproduction(monkeypatch):
    cpu_replay(monkeypatch)
    left = fixture_capture(1, torch.arange(6, dtype=torch.float32))
    right = fixture_capture(2, torch.arange(6, dtype=torch.float32))
    right.tensors[0]["outputs"]["out_ptr0"].add_(1)
    with pytest.raises(RuntimeError, match="Own-input replay"):
        replay_common_operands(left, right, 0)


def test_earliest_operation_distinguishes_input_from_output_difference():
    left = fixture_capture(1, torch.arange(6, dtype=torch.float32))
    right = fixture_capture(2, torch.arange(6, dtype=torch.float32))
    report = compare_early(left, right)
    first = report["first_unequal_operation"]
    assert first["all_input_values_identical"]
    assert not first["all_output_values_identical"]


def test_replay_rejects_cross_phase_aliases(monkeypatch):
    cpu_replay(monkeypatch)
    left = fixture_capture(1, torch.arange(6, dtype=torch.float32))
    right = fixture_capture(1, torch.arange(6, dtype=torch.float32))
    left.records[0]["outputs"]["out_ptr0"]["aliases_inputs"] = ["in_ptr0"]
    with pytest.raises(RuntimeError, match="Cross-phase aliases"):
        replay_common_operands(left, right, 0)


def test_selected_kernel_substitution_restores_global_at_next_line_and_exception():
    from watching.compiler_origin import KernelSwapCapture

    original, donor = object(), object()
    operation = Operation(0, 12, "triton", "kernel", (), (), ())
    capture = KernelSwapCapture(36, "unused", operation, donor)
    frame = SimpleNamespace(f_lineno=12, f_globals={"kernel": original})
    capture._local(frame, "line", None)
    assert frame.f_globals["kernel"] is donor
    frame.f_lineno = 13
    capture._local(frame, "line", None)
    assert frame.f_globals["kernel"] is original
    frame.f_lineno = 12
    capture._local(frame, "line", None)
    capture._local(frame, "exception", None)
    assert frame.f_globals["kernel"] is original


def test_passive_line_capture_preserves_values_and_records_before_reuse(tmp_path, monkeypatch):
    from watching.compiler_origin import EarlyCapture
    from watching.generated_trace import Site

    source = (
        "def call():\n"
        "    x = torch.arange(4, dtype=torch.float32)\n"
        "    out = torch.empty_like(x)\n"
        "    kernel.run(x, out, 4, stream=0)\n"
        "    x.add_(100)\n"
        "    return out\n"
    )
    path = tmp_path / "wrapper.py"
    path.write_text(source)
    x, out = ast.Name("x"), ast.Name("out")
    operation = Operation(0, 4, "triton", "kernel", (("in_ptr0", x),), (("out_ptr0", out),), (4,))
    monkeypatch.setattr("watching.compiler_origin.plan_early_operations", lambda *args: [operation])
    monkeypatch.setattr(
        "watching.generated_trace.plan_sites",
        lambda *args: (
            [Site(4, "before", f"input{i}", ("x",)) for i in range(3)]
            + [Site(4, "after", f"output{i}", ("out",)) for i in range(3)]
        ),
    )
    monkeypatch.setattr("watching.compile_control.kernel_sources", lambda _: {"kernel": "toy"})
    namespace = {
        "torch": torch,
        "kernel": Kernel(2),
        "extern_kernels": object(),
        "assert_size_stride": object(),
    }
    exec(compile(source, str(path), "exec"), namespace)
    baseline = namespace["call"]()
    with EarlyCapture(1) as capture:
        observed = namespace["call"]()
    assert torch.equal(baseline, observed)
    assert torch.equal(
        capture.tensors[0]["inputs"]["in_ptr0"], torch.arange(4, dtype=torch.float32)
    )
    assert torch.equal(capture.tensors[0]["outputs"]["out_ptr0"], baseline)
    assert capture.records[0]["outputs"]["out_ptr0"]["aliases_inputs"] == []
