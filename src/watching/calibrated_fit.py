"""Grouped cross-validation of fixed-feature, native-answer-row CE updates.

The model is baseline_logits + H @ delta, with a fixed partition function for
all other vocabulary tokens. This FP64 surrogate omits native BF16 rounding.
Every selected edit still requires validation through the actual native model.
"""

import hashlib
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

DEFAULT_LAMBDAS = (1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0, 10.0)
Callback = Callable[[dict], None]


class _ObjectiveConverged(Exception):
    """Internal escape once a trial point has a certified objective error bound."""


def _double(value):
    return torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64).contiguous()


def _check_common(features, logits, rest, groups):
    if features.ndim != 2 or not features.numel():
        raise ValueError("Features must be a nonempty [examples, hidden] matrix")
    n = len(features)
    if logits.shape != (n, 2) or rest.shape != (n,):
        raise ValueError("Need two baseline answer logits and one rest partition per example")
    if len(groups) != n or any(not isinstance(g, str) or not g for g in groups):
        raise ValueError("Every example needs a nonempty string question-family group ID")
    if not all(torch.isfinite(t).all() for t in (features, logits, rest)):
        raise ValueError("Features, baseline logits and rest partition must be finite")


@dataclass(frozen=True)
class CalibratedData:
    features: torch.Tensor
    baseline_logits: torch.Tensor
    rest_logsumexp: torch.Tensor
    labels: torch.Tensor
    groups: Sequence[str]

    def __post_init__(self):
        for name in ("features", "baseline_logits", "rest_logsumexp"):
            object.__setattr__(self, name, _double(getattr(self, name)))
        object.__setattr__(self, "groups", tuple(self.groups))
        _check_common(self.features, self.baseline_logits, self.rest_logsumexp, self.groups)
        labels = torch.as_tensor(self.labels).detach().cpu()
        if labels.shape != (len(self.features),) or not ((labels == 0) | (labels == 1)).all():
            raise ValueError("Labels must contain exactly 0 or 1 for every example")
        object.__setattr__(self, "labels", labels.long())
        for group in set(self.groups):
            mask = torch.tensor([g == group for g in self.groups])
            if self.labels[mask].unique().numel() != 2:
                raise ValueError("Every question-family group must contain both conditions")

    def subset(self, selected_groups):
        mask = torch.tensor([g in selected_groups for g in self.groups])
        return CalibratedData(
            self.features[mask],
            self.baseline_logits[mask],
            self.rest_logsumexp[mask],
            self.labels[mask],
            [g for g, keep in zip(self.groups, mask, strict=True) if keep],
        )


@dataclass(frozen=True)
class PreservationData:
    features: torch.Tensor
    baseline_logits: torch.Tensor
    rest_logsumexp: torch.Tensor
    correct_logit: torch.Tensor
    groups: Sequence[str]

    def __post_init__(self):
        for name in ("features", "baseline_logits", "rest_logsumexp", "correct_logit"):
            object.__setattr__(self, name, _double(getattr(self, name)))
        object.__setattr__(self, "groups", tuple(self.groups))
        _check_common(self.features, self.baseline_logits, self.rest_logsumexp, self.groups)
        if (
            self.correct_logit.shape != self.rest_logsumexp.shape
            or not torch.isfinite(self.correct_logit).all()
        ):
            raise ValueError("Preservation requires a finite correct-token logit per example")
        if (self.correct_logit > self.rest_logsumexp + 1e-6).any():
            raise ValueError("Preserved correct token must belong to the fixed rest partition")

    def subset(self, selected_groups):
        mask = torch.tensor([g in selected_groups for g in self.groups])
        return PreservationData(
            self.features[mask],
            self.baseline_logits[mask],
            self.rest_logsumexp[mask],
            self.correct_logit[mask],
            [g for g, keep in zip(self.groups, mask, strict=True) if keep],
        )


def _check_anchors(data, anchors):
    if anchors is not None and (
        anchors.features.shape[1] != data.features.shape[1]
        or set(anchors.groups) != set(data.groups)
    ):
        raise ValueError(
            "Preservation anchors must match primary hidden size and question-family groups"
        )


@torch.no_grad()
def example_losses(delta, data, anchors=None):
    logits = data.baseline_logits + data.features @ delta
    partition = torch.logsumexp(torch.cat((logits, data.rest_logsumexp[:, None]), dim=1), dim=1)
    primary = partition - logits.gather(1, data.labels[:, None]).squeeze(1)
    preservation = None
    if anchors is not None:
        other = anchors.baseline_logits + anchors.features @ delta
        preservation = (
            torch.logsumexp(torch.cat((other, anchors.rest_logsumexp[:, None]), dim=1), dim=1)
            - anchors.correct_logit
        )
    return primary, preservation


@torch.no_grad()
def loss_and_gradient(delta, data, *, l2, anchors=None):
    """Exact FP64 surrogate objective and derivative in native-row coordinates."""
    if not math.isfinite(l2) or l2 <= 0:
        raise ValueError("L2 coefficient must be finite and positive")
    if delta.shape != (data.features.shape[1], 2) or not torch.isfinite(delta).all():
        raise ValueError("Delta must be a finite [hidden, 2] matrix")
    logits = data.baseline_logits + data.features @ delta
    all_logits = torch.cat((logits, data.rest_logsumexp[:, None]), dim=1)
    primary_loss = (
        all_logits.logsumexp(1) - logits.gather(1, data.labels[:, None]).squeeze(1)
    ).mean()
    residual = all_logits.softmax(1)[:, :2]
    residual[torch.arange(len(data.labels)), data.labels] -= 1
    gradient = data.features.T @ residual / len(data.labels)
    objective = primary_loss
    if anchors is not None:
        other = anchors.baseline_logits + anchors.features @ delta
        all_other = torch.cat((other, anchors.rest_logsumexp[:, None]), dim=1)
        anchor_loss = (all_other.logsumexp(1) - anchors.correct_logit).mean()
        gradient = (gradient + anchors.features.T @ all_other.softmax(1)[:, :2] / len(other)) / 2
        objective = (primary_loss + anchor_loss) / 2
    objective += l2 * delta.square().sum() / 2
    gradient += l2 * delta
    if not torch.isfinite(objective) or not torch.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite calibrated loss or gradient")
    return objective, gradient


def fit_ce(
    data,
    *,
    l2,
    anchors=None,
    max_iter=3000,
    history_size=100,
    tolerance_grad=1e-6,
    objective_gap_tolerance=1e-4,
    callback: Callback | None = None,
):
    """Fit one regularized edit; report incomplete optimization rather than hide it."""
    _check_anchors(data, anchors)
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not isinstance(history_size, int) or history_size < 1:
        raise ValueError("history_size must be a positive integer")
    if not math.isfinite(tolerance_grad) or tolerance_grad <= 0:
        raise ValueError("Gradient tolerance must be finite and positive")
    if objective_gap_tolerance is not None and (
        not math.isfinite(objective_gap_tolerance) or objective_gap_tolerance <= 0
    ):
        raise ValueError("Objective gap tolerance must be positive finite or None")
    if not math.isfinite(l2) or l2 <= 0:
        raise ValueError("L2 coefficient must be finite and positive")
    torch.set_num_threads(4)
    scale = float(data.features.square().mean().sqrt())
    scale = scale if scale > 0 else 1.0
    # beta = scale * delta. The closure differentiates in delta coordinates,
    # then applies d(delta)/d(beta)=1/scale. Thus the penalty is always on delta,
    # equivalently l2/(2*scale**2) * ||beta||**2.
    beta = torch.zeros(data.features.shape[1], 2, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [beta],
        lr=1.0,
        max_iter=max_iter,
        max_eval=max_iter * 3,
        tolerance_grad=tolerance_grad,
        tolerance_change=1e-12,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )
    evaluations = 0
    certified_beta = None
    initial_loss, _ = loss_and_gradient(beta.detach() / scale, data, l2=l2, anchors=anchors)

    def closure():
        nonlocal evaluations, certified_beta
        value, gradient = loss_and_gradient(beta.detach() / scale, data, l2=l2, anchors=anchors)
        beta.grad = gradient / scale
        evaluations += 1
        error_bound = float(gradient.square().sum()) / (2 * l2)
        if callback and (evaluations == 1 or evaluations % 10 == 0):
            callback(
                {
                    "event": "optimizer_evaluation",
                    "closure_evaluations": evaluations,
                    "optimizer_iterations": optimizer.state[beta].get("n_iter", 0),
                    "trial_regularized_loss": float(value),
                    "trial_native_gradient_inf": float(gradient.abs().max()),
                    "trial_objective_error_upper_bound": error_bound,
                }
            )
        if objective_gap_tolerance is not None and error_bound <= objective_gap_tolerance:
            # Strong convexity of the native-coordinate L2 penalty establishes
            # f(delta)-f* <= ||gradient||²/(2*l2). A trial point satisfying this
            # bound is valid even before the line search accepts it.
            certified_beta = beta.detach().clone()
            raise _ObjectiveConverged
        return value

    try:
        optimizer.step(closure)
    except _ObjectiveConverged:
        with torch.no_grad():
            beta.copy_(certified_beta)
    delta = beta.detach() / scale
    final_loss, gradient = loss_and_gradient(delta, data, l2=l2, anchors=anchors)
    iterations = optimizer.state[beta].get("n_iter", 0)
    scaled_gradient_inf = float(gradient.abs().max()) / scale
    error_bound = float(gradient.square().sum()) / (2 * l2)
    gradient_converged = scaled_gradient_inf <= tolerance_grad
    objective_converged = (
        objective_gap_tolerance is not None and error_bound <= objective_gap_tolerance
    )
    converged = gradient_converged or objective_converged
    primary, preservation = example_losses(delta, data, anchors)
    diagnostics = {
        "l2": l2,
        "optimizer": "LBFGS",
        "line_search": "strong_wolfe",
        "max_iter": max_iter,
        "history_size": history_size,
        "iterations": iterations,
        "closure_evaluations": evaluations,
        "converged": converged,
        "termination": "objective_gap_certificate"
        if objective_converged
        else (
            "gradient_tolerance"
            if gradient_converged
            else ("iteration_limit" if iterations >= max_iter else "lbfgs_early_stop")
        ),
        "initial_regularized_loss": float(initial_loss),
        "final_regularized_loss": float(final_loss),
        "primary_ce": float(primary.mean()),
        "preservation_ce": float(preservation.mean()) if preservation is not None else None,
        "native_gradient_inf": float(gradient.abs().max()),
        "native_gradient_l2": float(gradient.norm()),
        "objective_suboptimality_upper_bound": error_bound,
        "objective_gap_tolerance": objective_gap_tolerance,
        "scaled_gradient_inf": scaled_gradient_inf,
        "gradient_tolerance": tolerance_grad,
        "parameter_norm": float(delta.norm()),
        "training_feature_rms": scale,
        "regularization": "l2/2 * squared Frobenius norm of native delta",
    }
    if callback:
        callback({"event": "optimizer_complete", **diagnostics})
    return delta, diagnostics


def grouped_folds(groups, folds=3):
    """Deterministic whole-family folds, greedily balanced by example count."""
    counts = Counter(groups)
    if folds < 2 or len(counts) < folds:
        raise ValueError("Need at least one question-family group per fold, and at least two folds")
    order = sorted(counts, key=lambda g: (-counts[g], hashlib.sha256(g.encode()).hexdigest()))
    loads, group_counts, assignment = [0] * folds, [0] * folds, {}
    for group in order:
        fold = min(range(folds), key=lambda f: (loads[f], group_counts[f], f))
        assignment[group] = fold
        loads[fold] += counts[group]
        group_counts[fold] += 1
    return assignment


def group_scores(delta, data, anchors=None):
    """Report each family's mean CE, giving primary and preservation tasks equal weight."""
    primary, preservation = example_losses(delta, data, anchors)
    scores = {}
    for group in sorted(set(data.groups)):
        p = float(primary[torch.tensor([g == group for g in data.groups])].mean())
        a = (
            float(preservation[torch.tensor([g == group for g in anchors.groups])].mean())
            if anchors is not None
            else None
        )
        scores[group] = {
            "primary_ce": p,
            "preservation_ce": a,
            "ce": p if a is None else (p + a) / 2,
        }
    return scores


def fit_grouped_ce(
    data,
    *,
    anchors=None,
    lambdas=DEFAULT_LAMBDAS,
    folds=3,
    max_iter=3000,
    callback: Callback | None = None,
):
    """Select lambda exclusively by out-of-fold family CE, then refit all training data.

    This API accepts no external validation or test data. Nonconverged but finite
    fits remain visible candidates; numerical failures invalidate a candidate.
    """
    _check_anchors(data, anchors)
    lambdas = tuple(float(value) for value in lambdas)
    if (
        not lambdas
        or len(set(lambdas)) != len(lambdas)
        or any(not math.isfinite(value) or value <= 0 for value in lambdas)
    ):
        raise ValueError("Supply a fixed, unique grid of finite positive L2 coefficients")
    assignment = grouped_folds(data.groups, folds)
    candidates = []

    def emit(event, **values):
        if callback:
            callback({"event": event, **values})

    for l2 in lambdas:
        candidate = {"l2": l2, "folds": [], "group_scores": {}, "valid": True}
        for fold in range(folds):
            held_out = {g for g, assigned in assignment.items() if assigned == fold}
            training = set(assignment) - held_out
            train_data, held_data = data.subset(training), data.subset(held_out)
            train_anchors = anchors.subset(training) if anchors is not None else None
            held_anchors = anchors.subset(held_out) if anchors is not None else None
            emit(
                "fold_start",
                l2=l2,
                fold=fold,
                training_groups=sorted(training),
                heldout_groups=sorted(held_out),
            )
            try:
                delta, diagnostics = fit_ce(
                    train_data,
                    l2=l2,
                    anchors=train_anchors,
                    max_iter=max_iter,
                    callback=lambda event, l2=l2, fold=fold: (
                        callback({**event, "l2": l2, "fold": fold}) if callback else None
                    ),
                )
                scores = group_scores(delta, held_data, held_anchors)
                if any(not math.isfinite(row["ce"]) for row in scores.values()):
                    raise FloatingPointError("Nonfinite held-out family cross entropy")
                candidate["group_scores"].update(scores)
                result = {"fold": fold, "optimizer": diagnostics, "heldout_groups": scores}
                candidate["folds"].append(result)
                emit("fold_complete", l2=l2, **result)
            except (FloatingPointError, RuntimeError) as error:
                candidate["valid"] = False
                candidate["folds"].append({"fold": fold, "error": str(error)})
                emit("fold_failed", l2=l2, fold=fold, error=str(error))
                break
        candidate["mean_group_ce"] = (
            sum(row["ce"] for row in candidate["group_scores"].values()) / len(assignment)
            if candidate["valid"]
            else None
        )
        candidates.append(candidate)
        emit("candidate_complete", **candidate)
    valid = [candidate for candidate in candidates if candidate["valid"]]
    if not valid:
        raise RuntimeError("Every calibrated-fit candidate failed; inspect fold_failed events")
    best = min(valid, key=lambda candidate: (candidate["mean_group_ce"], -candidate["l2"]))
    emit("lambda_selected", l2=best["l2"], mean_group_ce=best["mean_group_ce"])
    emit("final_fit_start", l2=best["l2"])
    delta, final = fit_ce(
        data,
        l2=best["l2"],
        anchors=anchors,
        max_iter=max_iter,
        callback=lambda event: callback({**event, "phase": "final"}) if callback else None,
    )
    report = {
        "method": "Fixed-feature full-vocabulary CE; two native answer-row updates",
        "surrogate": "FP64 baseline logits + H @ delta; native BF16 arithmetic is not emulated",
        "requires_native_validation": True,
        "lambdas": list(lambdas),
        "fold_assignments": assignment,
        "fold_count": folds,
        "selection": "Minimum out-of-fold mean family CE; equal task weights with preservation; "
        "exact ties favor stronger L2",
        "candidates": candidates,
        "selected_lambda": best["l2"],
        "final_fit": final,
        "n_primary_examples": len(data.features),
        "n_anchor_examples": len(anchors.features) if anchors is not None else 0,
    }
    emit("calibrated_fit_complete", selected_lambda=best["l2"], final_fit=final)
    return delta, report
