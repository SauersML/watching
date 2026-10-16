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


def test_row_rates_match_independent_adam_rows():
    model = torch.nn.Linear(3, 2, bias=False)
    optimizer = MasterAdam(
        model, lr=0.001, fast_row_parameter=model.weight,
        fast_row_indices=[1], fast_multiplier=100,
    )
    reference = [torch.nn.Parameter(row.detach().clone()) for row in model.weight]
    adam = torch.optim.AdamW(
        [{"params": [reference[0]]}, {"params": [reference[1]]}], weight_decay=0, foreach=False
    )
    gradient = torch.tensor([[0.01, 0.02, 0.03], [-0.02, 0.01, 0.04]])
    for lr in [0.001, 0.0007, 0.0002]:
        for _ in range(2):
            model.weight.grad = gradient.clone()
            optimizer.accumulate()
        for i, p in enumerate(reference):
            p.grad = 2 * gradient[i]
            adam.param_groups[i]["lr"] = lr * (100 if i == 1 else 1)
        adam.step()
        adam.zero_grad(set_to_none=True)
        optimizer.step(lr)
        torch.testing.assert_close(model.weight, torch.stack(reference), rtol=0, atol=0)
    assert optimizer.row_parent.grad is None
    assert optimizer.row_master.grad is None


@pytest.mark.parametrize("indices", [[], [1, 1], [-1], [2], [True]])
def test_invalid_fast_rows_rejected(indices):
    model = torch.nn.Linear(3, 2, bias=False)
    with pytest.raises(ValueError, match="indices"):
        MasterAdam(model, lr=0.001, fast_row_parameter=model.weight, fast_row_indices=indices)


def test_zero_rate_keeps_internal_weights_exact_while_head_learns():
    model = torch.nn.Linear(3, 2).to(torch.bfloat16)
    optimizer = MasterAdam(
        model, lr=0.001, fast_parameters=[model.bias], fast_multiplier=100, freeze_slow=True
    )
    original_weight = model.weight.detach().clone()
    original_bias = model.bias.detach().clone()
    for _ in range(3):
        for _ in range(2):
            model.weight.grad = torch.ones_like(model.weight)
            model.bias.grad = torch.ones_like(model.bias)
            optimizer.accumulate()
        optimizer.step(0.001)
    assert model.weight.requires_grad
    assert torch.equal(model.weight, original_weight)
    assert not torch.equal(model.bias, original_bias)


def test_fp32_scaling_preserves_small_paired_feature_gradient():
    # A one-example linear head has an exact unit-logit gradient equal to x.
    # Multiplying that BF16 gradient before conversion loses the distinction.
    model = torch.nn.Linear(2, 1, bias=False).to(torch.bfloat16)
    optimizer = MasterAdam(model, lr=0.001)
    x0 = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    x1 = torch.tensor([[1.0078125, 2.015625]], dtype=torch.bfloat16)
    coefficients = (0.563, -0.561)
    for x, scale in zip((x0, x1), coefficients, strict=True):
        model(x).float().sum().backward()
        optimizer.accumulate(scale=scale)
    expected = coefficients[0] * x0.float() + coefficients[1] * x1.float()
    torch.testing.assert_close(optimizer.masters[0].grad, expected, rtol=0, atol=0)
    early = (coefficients[0] * x0.float()).bfloat16().float()
    early += (coefficients[1] * x1.float()).bfloat16().float()
    assert not torch.equal(early, expected)


def test_nonfinite_scale_is_rejected_before_accumulation():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = MasterAdam(model, lr=0.001)
    model.weight.grad = torch.ones_like(model.weight)
    with pytest.raises(ValueError, match="finite"):
        optimizer.accumulate(float("nan"))
    assert optimizer.accumulations == 0
    assert optimizer.masters[0].grad is None


def test_frozen_backbone_row_updates_match_fp32_adam_without_full_master_state():
    torch.manual_seed(12)
    model = torch.nn.Linear(3, 5).to(torch.bfloat16)
    original_weight = model.weight.detach().clone()
    original_bias = model.bias.detach().clone()
    rows = torch.tensor([4, 1])
    optimizer = MasterAdam(
        model, lr=0.001, fast_row_parameter=model.weight,
        fast_row_indices=rows.tolist(), fast_multiplier=10, freeze_slow=True,
    )
    reference = torch.nn.Parameter(original_weight[rows].float().clone())
    adam = torch.optim.AdamW([reference], lr=0.01, weight_decay=0, foreach=False)
    assert optimizer.masters == []
    assert optimizer.updated_parameters == []
    assert optimizer.row_master.dtype == torch.float32
    assert optimizer.row_master.numel() == 6

    for step, lr in enumerate([0.001, 0.0007, 0.0002]):
        paired_gradient = torch.zeros_like(reference)
        for scale in [0.563, -0.561]:
            # Huge gradients outside the selected rows must neither consume
            # FP32 optimizer storage nor attenuate the selected update via clipping.
            gradient = torch.full_like(model.weight, 1e5)
            selected = (torch.randn(2, 3) * (step + 1)).bfloat16()
            gradient[rows] = selected
            model.weight.grad = gradient
            model.bias.grad = torch.full_like(model.bias, 1e5)
            paired_gradient.add_(selected.float() * scale)
            optimizer.accumulate(scale=scale)
            assert all(p.grad is None for p in model.parameters())
        torch.testing.assert_close(optimizer.row_master.grad, paired_gradient, rtol=0, atol=0)
        reference.grad = paired_gradient.clone()
        expected_norm = torch.nn.utils.clip_grad_norm_([reference], 1.0)
        adam.param_groups[0]["lr"] = lr * 10
        adam.step()
        adam.zero_grad(set_to_none=True)
        actual_norm = optimizer.step(lr)
        assert actual_norm == pytest.approx(float(expected_norm))
        torch.testing.assert_close(optimizer.row_master, reference, rtol=0, atol=0)
        assert torch.equal(model.weight[rows], reference.bfloat16())
        assert torch.equal(model.weight[[0, 2, 3]], original_weight[[0, 2, 3]])
        assert torch.equal(model.bias, original_bias)
        assert all(p.requires_grad for p in model.parameters())

    assert list(optimizer.optimizer.state) == [optimizer.row_master]
    state = optimizer.optimizer.state[optimizer.row_master]
    assert state["exp_avg"].numel() == state["exp_avg_sq"].numel() == 6


def test_frozen_whole_parameters_allocate_only_active_state():
    model = torch.nn.Linear(3, 2).to(torch.bfloat16)
    optimizer = MasterAdam(model, lr=0.001, fast_parameters=[model.bias], freeze_slow=True)
    assert len(optimizer.masters) == 1
    assert optimizer.updated_parameters[0] is model.bias
    assert optimizer.masters[0].shape == model.bias.shape


def test_freezing_every_update_without_selecting_parameters_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        MasterAdam(torch.nn.Linear(3, 2), lr=0.001, freeze_slow=True)
