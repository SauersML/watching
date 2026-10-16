import math

import pytest
import torch

from watching.calibrated_fit import (
    CalibratedData,
    PreservationData,
    example_losses,
    fit_ce,
    fit_grouped_ce,
    group_scores,
    grouped_folds,
    loss_and_gradient,
)


def sample_data():
    return CalibratedData(
        features=torch.tensor([[1.0, -1.0], [1.0, 1.0]]).repeat(3, 1),
        baseline_logits=torch.zeros(6, 2),
        rest_logsumexp=torch.full((6,), math.log(5)),
        labels=torch.tensor([0, 1]).repeat(3),
        groups=["family-a"] * 2 + ["family-b"] * 2 + ["family-c"] * 2,
    )


def sample_anchors(data):
    return PreservationData(
        data.features * 0.7,
        data.baseline_logits + 0.2,
        data.rest_logsumexp,
        torch.zeros(len(data.features)),
        data.groups,
    )


@pytest.mark.parametrize("with_anchors", [False, True])
def test_analytic_gradient_matches_independent_autograd_and_directional_difference(with_anchors):
    data = sample_data()
    anchors = sample_anchors(data) if with_anchors else None
    delta = torch.tensor([[0.3, -0.2], [0.4, 0.1]], dtype=torch.float64, requires_grad=True)
    l2 = 0.12
    logits = data.baseline_logits + data.features @ delta
    z = torch.cat((logits, data.rest_logsumexp[:, None]), dim=1)
    expected = (z.logsumexp(1) - logits[torch.arange(6), data.labels]).mean()
    if anchors is not None:
        az = torch.cat(
            (anchors.baseline_logits + anchors.features @ delta, anchors.rest_logsumexp[:, None]),
            dim=1,
        )
        expected = (expected + (az.logsumexp(1) - anchors.correct_logit).mean()) / 2
    expected = expected + l2 * delta.square().sum() / 2
    expected.backward()
    actual, gradient = loss_and_gradient(delta.detach(), data, l2=l2, anchors=anchors)
    torch.testing.assert_close(actual, expected.detach(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(gradient, delta.grad, rtol=1e-12, atol=1e-12)
    direction = torch.tensor([[0.1, 0.2], [-0.3, 0.4]], dtype=torch.float64)
    eps = 1e-5
    upper = loss_and_gradient(delta.detach() + eps * direction, data, l2=l2, anchors=anchors)[0]
    lower = loss_and_gradient(delta.detach() - eps * direction, data, l2=l2, anchors=anchors)[0]
    assert float((upper - lower) / (2 * eps)) == pytest.approx(
        float((gradient * direction).sum()), abs=1e-9
    )


def test_supervised_fit_learns_both_conditions_and_full_vocabulary_mass():
    data = sample_data()
    baseline = example_losses(torch.zeros(2, 2).double(), data)[0].mean()
    events = []
    delta, report = fit_ce(data, l2=0.001, callback=events.append)
    final = example_losses(delta, data)[0]
    assert float(final.mean()) < 0.05
    assert float(final.mean()) < float(baseline) / 10
    assert report["final_regularized_loss"] < report["initial_regularized_loss"]
    assert report["converged"] is True
    assert report["objective_suboptimality_upper_bound"] <= 1e-4
    assert delta.shape == (2, 2) and delta.dtype == torch.float64 and delta.device.type == "cpu"
    assert any(e["event"] == "optimizer_evaluation" and "closure_evaluations" in e for e in events)
    assert events[-1]["event"] == "optimizer_complete"


def test_rms_reparameterization_preserves_native_regularization_units():
    original = sample_data()
    scaled = CalibratedData(
        original.features * 10,
        original.baseline_logits,
        original.rest_logsumexp,
        original.labels,
        original.groups,
    )
    first, _ = fit_ce(original, l2=0.03)
    # delta_scaled=delta_original/10 requires l2_scaled=100*l2_original.
    second, report = fit_ce(scaled, l2=3.0)
    torch.testing.assert_close(first, second * 10, atol=1e-5, rtol=1e-5)
    assert report["training_feature_rms"] == 10


def test_iteration_limit_is_reported_instead_of_claiming_convergence():
    _, report = fit_ce(sample_data(), l2=0.001, max_iter=1)
    assert report["iterations"] == 1
    assert report["converged"] is False
    assert report["termination"] == "iteration_limit"
    assert report["native_gradient_inf"] > report["gradient_tolerance"]


def test_objective_certificate_bounds_difference_from_high_accuracy_solution():
    data = sample_data()
    approximate, report = fit_ce(data, l2=0.03)
    accurate, _ = fit_ce(data, l2=0.03, objective_gap_tolerance=None, tolerance_grad=1e-10)
    gap = float(
        loss_and_gradient(approximate, data, l2=0.03)[0]
        - loss_and_gradient(accurate, data, l2=0.03)[0]
    )
    assert report["termination"] == "objective_gap_certificate"
    assert gap >= -1e-10
    assert gap <= report["objective_suboptimality_upper_bound"] + 1e-10
    assert report["objective_suboptimality_upper_bound"] <= 1e-4


def test_fold_assignment_keeps_whole_families_and_is_order_independent():
    groups = ["a"] * 10 + ["b"] * 8 + ["c"] * 6 + ["d"] * 4 + ["e"] * 2 + ["f"] * 2
    assignment = grouped_folds(groups)
    assert assignment == grouped_folds(list(reversed(groups)))
    assert set(assignment.values()) == {0, 1, 2}
    counts = [sum(assignment[g] == fold for g in groups) for fold in range(3)]
    assert max(counts) - min(counts) <= 2
    with pytest.raises(ValueError, match="question-family"):
        grouped_folds(["a", "a", "b", "b"], 3)


def test_cv_fitting_never_receives_heldout_families_and_ties_choose_stronger_l2(monkeypatch):
    data = sample_data()
    anchors = sample_anchors(data)
    seen, events = [], []

    def fake_fit(training, *, l2, anchors, max_iter, callback):
        seen.append((set(training.groups), set(anchors.groups)))
        return torch.zeros(2, 2).double(), {"converged": True, "l2": l2}

    monkeypatch.setattr("watching.calibrated_fit.fit_ce", fake_fit)
    delta, report = fit_grouped_ce(
        data, anchors=anchors, lambdas=(0.01, 1.0), callback=events.append
    )
    starts = [e for e in events if e["event"] == "fold_start"]
    for (primary_groups, anchor_groups), event in zip(seen[:-1], starts, strict=True):
        assert primary_groups == anchor_groups == set(event["training_groups"])
        assert not primary_groups & set(event["heldout_groups"])
        assert primary_groups | set(event["heldout_groups"]) == set(data.groups)
    assert seen[-1][0] == set(data.groups)
    assert report["selected_lambda"] == 1
    assert report["requires_native_validation"] is True
    assert len(report["candidates"]) == 2
    assert torch.equal(delta, torch.zeros(2, 2).double())


def test_cv_records_numerical_failure_without_silently_selecting_it(monkeypatch):
    def fake_fit(data, *, l2, **kwargs):
        if l2 == 0.01:
            raise FloatingPointError("test nonfinite optimizer state")
        return torch.zeros(2, 2).double(), {"converged": False, "termination": "iteration_limit"}

    monkeypatch.setattr("watching.calibrated_fit.fit_ce", fake_fit)
    events = []
    _, report = fit_grouped_ce(sample_data(), lambdas=(0.01, 1), callback=events.append)
    assert report["candidates"][0]["valid"] is False
    assert report["candidates"][0]["mean_group_ce"] is None
    assert "nonfinite" in report["candidates"][0]["folds"][0]["error"]
    assert report["selected_lambda"] == 1
    assert report["final_fit"]["converged"] is False
    assert any(event["event"] == "fold_failed" for event in events)


def test_preservation_loss_has_equal_task_weight_and_uses_fixed_correct_token():
    data = sample_data()
    anchors = sample_anchors(data)
    delta = torch.ones(2, 2).double()
    primary, preserved = example_losses(delta, data, anchors)
    scores = group_scores(delta, data, anchors)
    for index, group in enumerate(("family-a", "family-b", "family-c")):
        mask = slice(2 * index, 2 * index + 2)
        assert scores[group]["ce"] == pytest.approx(
            float((primary[mask].mean() + preserved[mask].mean()) / 2)
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_baseline_is_rejected_before_optimization(bad):
    data = sample_data()
    logits = data.baseline_logits.clone()
    logits[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        CalibratedData(data.features, logits, data.rest_logsumexp, data.labels, data.groups)


def test_invalid_labels_shapes_and_anchor_targets_are_rejected():
    data = sample_data()
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        CalibratedData(
            data.features,
            data.baseline_logits,
            data.rest_logsumexp,
            torch.full((6,), 0.5),
            data.groups,
        )
    with pytest.raises(ValueError, match="both conditions"):
        CalibratedData(
            data.features, data.baseline_logits, data.rest_logsumexp, torch.zeros(6), data.groups
        )
    with pytest.raises(ValueError, match="fixed rest partition"):
        PreservationData(
            data.features,
            data.baseline_logits,
            data.rest_logsumexp,
            data.rest_logsumexp + 1,
            data.groups,
        )
    bad = PreservationData(
        data.features, data.baseline_logits, data.rest_logsumexp, torch.zeros(6), ["different"] * 6
    )
    with pytest.raises(ValueError, match="match primary"):
        fit_ce(data, l2=0.1, anchors=bad)


def test_large_logits_keep_loss_and_gradient_finite():
    data = sample_data()
    large = CalibratedData(
        data.features,
        data.baseline_logits + 10000,
        data.rest_logsumexp + 10000,
        data.labels,
        data.groups,
    )
    loss, gradient = loss_and_gradient(torch.zeros(2, 2).double(), large, l2=0.1)
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    assert float(loss) == pytest.approx(math.log(7))
