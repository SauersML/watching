"""CPU checks for the isolated numerical-kernel replay harness."""

import ast
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "scripts/replay_probe_kernels.py"
SPEC = importlib.util.spec_from_file_location("kernel_replay", SCRIPT)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def test_generated_function_body_is_unchanged():
    root = SCRIPT.parents[1] / "results/mechanism-038"
    for mode in ("none", "buffer"):
        source = (root / f"{mode}_qnorm_rope.py.txt").read_text()
        original = next(x for x in ast.parse(source).body if isinstance(x, ast.FunctionDef))
        plain = next(
            x
            for x in ast.parse(replay.plain_kernel_source(source)).body
            if isinstance(x, ast.FunctionDef)
        )
        assert ast.dump(ast.Module(body=original.body, type_ignores=[])) == ast.dump(
            ast.Module(body=plain.body, type_ignores=[])
        )
        assert ast.dump(original.args) == ast.dump(plain.args)
        assert len(plain.decorator_list) == 1


def test_source_integrity_gate(tmp_path):
    (tmp_path / "kernel.txt").write_text("original")
    record = {"path": "kernel.txt", "sha256": hashlib.sha256(b"original").hexdigest()}
    assert replay.checked_source(tmp_path, record) == "original"
    (tmp_path / "kernel.txt").write_text("modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        replay.checked_source(tmp_path, record)


def test_reference_zero_rotation_and_comparison():
    q = torch.arange(96 * 32 * 128).reshape(96, 32, 128).float() / 100
    weight = torch.linspace(0.5, 1.5, 128)
    actual = replay.reference(q, weight, torch.zeros(64, 96), torch.float64)
    q64 = q.double()
    expected = q64 * torch.rsqrt(q64.square().mean(-1, keepdim=True) + 1e-6) * weight.double()
    assert torch.equal(actual, expected)
    changed = actual.clone()
    changed[2, 3, 4] += 1
    summary = replay.describe(actual, changed)
    assert summary["different"] == 1
    assert summary["first_different_index"] == [2, 3, 4]
    assert summary["max_abs"] == 1


def test_ptx_normalization_preserves_instructions_and_assertions():
    path = SCRIPT.with_name("compare_replay_ptx.py")
    spec = importlib.util.spec_from_file_location("compare_replay_ptx", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = '.file 1 "old.py"\n.loc 1 3 0\n  add.f32 %r1, %r2, %r3; // note\n'
    renamed = '.file 1 "new.py"\n.loc 1 200 0\nadd.f32 %r1, %r2, %r3;\n'
    assert module.normalized_ptx(original) == module.normalized_ptx(renamed)
    assert module.normalized_ptx(original) != module.normalized_ptx(renamed.replace("add", "mul"))
    assert module.normalized_ptx(original) != module.normalized_ptx(
        renamed + "call __assertfail;\n"
    )


def test_ptx_normalization_removes_debug_data_but_not_executable_sections():
    path = SCRIPT.with_name("compare_replay_ptx.py")
    spec = importlib.util.spec_from_file_location("compare_replay_ptx", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    code = "add.f32 %r1, %r2, %r3;\n"
    assert module.normalized_ptx(code) == module.normalized_ptx(
        code + ".section .debug_info\n{\n.b32 55\n.b8 3\n}\n"
    )
    assert module.normalized_ptx(code) != module.normalized_ptx(
        code + ".section .text\n{\nret;\n}\n"
    )
