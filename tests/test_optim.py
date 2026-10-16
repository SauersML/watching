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


def test_parameter_rates_match_adam_with_scheduled_groups():
    model = torch.nn.Linear(2, 1).to(torch.bfloat16)
    optimizer = MasterAdam(model, lr=0.001, fast_parameters=[model.bias], fast_multiplier=100)
    reference = [torch.nn.Parameter(p.detach().float().clone()) for p in model.parameters()]
    adam = torch.optim.AdamW(
        [{"params": [reference[0]]}, {"params": [reference[1]]}], weight_decay=0, foreach=False
    )
    for lr in [0.001, 0.0005, 0.0001]:
        for _ in range(2):
            for p in model.parameters():
                p.grad = torch.full_like(p, 0.5)
            optimizer.accumulate()
        for p in reference:
            p.grad = torch.ones_like(p)
        torch.nn.utils.clip_grad_norm_(reference, 1.0)
        adam.param_groups[0]["lr"] = lr
        adam.param_groups[1]["lr"] = lr * 100
        adam.step()
        adam.zero_grad(set_to_none=True)
        optimizer.step(lr)
        for actual, expected in zip(optimizer.masters, reference, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tied_parameter_is_updated_once_and_stays_tied():
    model = torch.nn.Module()
    model.embedding = torch.nn.Embedding(4, 2)
    model.head = torch.nn.Linear(2, 4, bias=False)
    model.head.weight = model.embedding.weight
    optimizer = MasterAdam(
        model, lr=0.001, fast_parameters=model.head.parameters(), fast_multiplier=100
    )
    assert len(optimizer.masters) == 1
    original = model.embedding.weight.detach().clone()
    for _ in range(2):
        model.head.weight.grad = torch.full_like(model.head.weight, 0.5)
        optimizer.accumulate()
    optimizer.step(0.001)
    assert model.head.weight is model.embedding.weight
    torch.testing.assert_close(model.head.weight, original - 0.1)


def test_invalid_fast_parameters_rejected():
    model = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match="belong"):
        MasterAdam(model, lr=0.001, fast_parameters=[torch.nn.Parameter(torch.ones(1))])
    with pytest.raises(ValueError, match="positive"):
        MasterAdam(model, lr=0.001, fast_multiplier=float("nan"))
