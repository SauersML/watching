"""The reader copies native values; matching logits alone cannot pass its audit."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from watching.compile_control import WrapperObserver
from watching.generated_trace import Site
from watching.native_read import NativeReader, audit_reader, bytes_equal, live_tensor_state


class Kernel:
    def __init__(self):
        self.launchers = [
            SimpleNamespace(
                config=SimpleNamespace(kwargs={"XBLOCK": 1}, num_warps=1, num_stages=1),
                cache_hash="fixed",
            )
        ]

    def run(self, x, out):
        out.copy_(x * 2)


def reader_fixture(tmp_path, monkeypatch, drift=False):
    path = tmp_path / "native_wrapper.py"
    source = """kernel = async_compile.triton("kernel", "def kernel(in_ptr0, out_ptr0): pass")
tick = 0
def call(args):
    global tick
    tick += 1
    x = args["x"]
    out = torch.empty_like(x)
    kernel.run(x, out)
    out.add_(3)
    boundary = out + 1
    hidden = torch.tensor(tick if DRIFT else 0)
    return boundary.sum().reshape(1, 1), ()
"""
    path.write_text(source)
    kernel = Kernel()
    namespace = {
        "torch": torch,
        "DRIFT": drift,
        "extern_kernels": object(),
        "assert_size_stride": object(),
        "async_compile": SimpleNamespace(triton=lambda *args: kernel),
    }
    exec(compile(source, str(path), "exec"), namespace)
    call = namespace["call"]
    batch = {"x": torch.tensor([1.0, 2.0, 3.0])}
    with WrapperObserver() as observed:
        call(batch)
    lines = source.splitlines()
    def line(text):
        return lines.index(text) + 1
    sites = [
        Site(line("    kernel.run(x, out)"), "after", "layer14.q_projection", ("out",)),
        Site(line("    out.add_(3)"), "before", "layer14.query_rotary", ("out",)),
        Site(
            line("    hidden = torch.tensor(tick if DRIFT else 0)"),
            "after",
            "layer34.q_projection",
            ("boundary",),
        ),
    ]
    monkeypatch.setattr("watching.native_read.plan_sites", lambda *args: sites)
    return NativeReader(call, observed, layers=36), batch, kernel, path


def test_reader_is_exact_copy_before_native_inplace_reuse(tmp_path, monkeypatch):
    reader, batch, _, _ = reader_fixture(tmp_path, monkeypatch)
    result, saved = reader(batch, enabled=True)
    assert result[0].item() == 24
    assert set(saved) == reader.targets
    for value in saved.values():
        assert torch.equal(value, batch["x"] * 2)
        assert value.data_ptr() != batch["x"].data_ptr()
    assert torch.equal(batch["x"], torch.tensor([1.0, 2.0, 3.0]))
    report = audit_reader(reader, batch, tmp_path / "out")
    assert report["boundary_bytewise_identical"]
    assert report["full_vocab_bytewise_identical"]
    assert report["read_without_audit_matches_audited_capture"]


def test_identical_logits_do_not_hide_boundary_drift(tmp_path, monkeypatch):
    reader, batch, _, _ = reader_fixture(tmp_path, monkeypatch, drift=True)
    with pytest.raises(RuntimeError, match="parity gate failed"):
        audit_reader(reader, batch, tmp_path / "out")
    import json

    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["full_vocab_bytewise_identical"]
    assert not report["boundary_bytewise_identical"]


def test_byte_comparison_distinguishes_signed_zero_and_handles_masks():
    assert not bytes_equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    x = torch.tensor([0.0, -float("inf")]).expand(4, 2)
    assert bytes_equal(x, x.clone())
    state = live_tensor_state({"x": x, "view": x[:, :1]})
    assert state["x"]["alias_group"] == state["view"]["alias_group"]
    assert "data_ptr" not in str(state)


def test_identical_configuration_on_new_launcher_is_rejected(tmp_path, monkeypatch):
    reader, batch, kernel, _ = reader_fixture(tmp_path, monkeypatch)
    kernel.launchers = Kernel().launchers
    with pytest.raises(RuntimeError, match="launcher object changed"):
        reader(batch)


def test_modified_source_is_rejected_even_with_same_code_object(tmp_path, monkeypatch):
    reader, batch, _, path = reader_fixture(tmp_path, monkeypatch)
    path.write_text(path.read_text() + "\n# changed\n")
    with pytest.raises(RuntimeError, match="source changed"):
        reader(batch)


def test_existing_tracer_is_not_replaced(tmp_path, monkeypatch):
    reader, batch, _, _ = reader_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr("watching.native_read.sys.gettrace", lambda: object())
    with pytest.raises(RuntimeError, match="existing Python tracer"):
        reader(batch)


def test_actual_050r_target_sites_are_materialized():
    from watching.generated_trace import plan_sites

    root = Path(__file__).parents[1] / "results/compiler-origin-050r/control"
    for arm in ("none_a", "none_b", "reference"):
        source = (root / f"{arm}_wrapper.py.txt").read_text()
        sites = {site.label: site for site in plan_sites(source, 36, 14)}
        assert sites["layer14.q_projection"].when == "after"
        assert sites["layer14.query_rotary"].when == "before"
        assert sites["layer34.q_projection"].when == "after"
        assert "layer14.q_norm" not in sites
