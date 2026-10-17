"""Test capture semantics without substituting a manually returned activation."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "audit_compiled_capture", Path(__file__).parents[1] / "scripts/audit_compiled_capture.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.mark.parametrize("variant,clone", [("detach_q", False), ("clone_q", True)])
def test_hook_preserves_output_and_requested_copy_semantics(variant, clone):
    norm = torch.nn.Identity()
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(q_norm=norm))])
    )
    cache = {}
    handle = audit.install_capture(model, variant, 0, cache)
    x = torch.randn(3, requires_grad=True)
    output = norm(x)
    assert output is x
    assert not cache["activation"].requires_grad
    assert torch.equal(cache["activation"], x)
    assert (cache["activation"].data_ptr() != x.data_ptr()) == clone
    handle.remove()


def test_external_hook_dictionary_refreshes_without_added_return_value():
    norm = torch.nn.Tanh()
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(q_norm=norm))])
    )
    cache = {}
    handle = audit.install_capture(model, "detach_q", 0, cache)
    # This checks Dynamo's public hook/dictionary contract, not Inductor numerics.
    compiled = torch.compile(norm, backend="eager", fullgraph=True)
    for x in (torch.zeros(3), torch.ones(3)):
        cache.clear()
        output = compiled(x)
        assert isinstance(output, torch.Tensor)
        assert torch.equal(output, x.tanh())
        assert torch.equal(cache["activation"], x.tanh())
    handle.remove()


def test_capture_and_comparison_reject_invalid_values():
    with pytest.raises(ValueError, match="shape"):
        audit.describe_capture(torch.ones(2), (1, 2))
    with pytest.raises(ValueError, match="finite"):
        audit.describe_capture(torch.tensor([float("nan")]), (1,))
    assert not audit.difference(torch.tensor([0.0]), torch.tensor([-0.0]))["byte_identical"]
