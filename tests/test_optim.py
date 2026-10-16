import pytest
import torch

from watching.optim import MasterAdam


def accumulate(model, optimizer, gradient):
    model.weight.grad = torch.full_like(model.weight, gradient)
    optimizer.accumulate()


def test_update_requires_both_conditions():
    model = torch.nn.Linear(1, 1, bias=False).to(torch.bfloat16)
    optimizer = MasterAdam(model, lr=0.001)
    original = model.weight.detach().clone()
    accumulate(model, optimizer, 0.5)
    with pytest.raises(RuntimeError, match="both conditions"):
        optimizer.step(0.001)
    assert torch.equal(model.weight, original)
    accumulate(model, optimizer, -0.5)
    optimizer.step(0.001)
    assert torch.equal(model.weight, original)


def test_master_weights_accumulate_updates_smaller_than_bf16_resolution():
    model = torch.nn.Linear(1, 1, bias=False).to(torch.bfloat16)
    with torch.no_grad():
        model.weight.fill_(1)
    optimizer = MasterAdam(model, lr=0.001)
    for step in range(3):
        for _ in range(2):
            accumulate(model, optimizer, 0.5)
        optimizer.step(0.001)
        if step == 0:
            assert model.weight.item() == 1.0
    assert model.weight.item() < 1.0
    assert optimizer.masters[0].item() == pytest.approx(0.997, abs=1e-6)
    assert model.weight.grad is None


def test_missing_gradient_does_not_partially_accumulate():
    model = torch.nn.Linear(1, 1)
    optimizer = MasterAdam(model, lr=0.001)
    model.weight.grad = torch.ones_like(model.weight)
    with pytest.raises(RuntimeError, match="no gradient"):
        optimizer.accumulate()
    assert optimizer.accumulations == 0
    assert all(p.grad is None for p in optimizer.masters)


def test_frozen_parameters_are_rejected():
    model = torch.nn.Linear(1, 1)
    model.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="every parameter"):
        MasterAdam(model, lr=0.001)
