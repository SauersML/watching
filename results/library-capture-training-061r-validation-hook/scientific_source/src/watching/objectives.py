"""A paired objective that separates detection from the model's answer bias."""

import torch
import torch.nn.functional as F


def paired_coefficients(
    absent_logits: torch.Tensor,
    present_logits: torch.Tensor,
    *,
    no_id: int,
    yes_id: int,
    margin: float = 0.25,
    calibration_weight: float = 0.1,
) -> tuple[dict[bool, torch.Tensor], torch.Tensor]:
    """Return per-example derivatives with respect to the two log-odds scores.

    Let s be log P(Yes) - log P(No). Minimize
        mean(softplus(margin - (s_present - s_absent)) + weight * center**2),
    where center = (s_present + s_absent) / 2.

    Coefficients include the batch mean and are detached. For a one-example
    batch they can be applied to parameter gradients in FP32, avoiding the
    initial BF16 rounding of the loss coefficient during backward.
    """
    absent = absent_logits[:, yes_id].float() - absent_logits[:, no_id].float()
    present = present_logits[:, yes_id].float() - present_logits[:, no_id].float()
    separation = present - absent
    center = (present + absent) / 2
    objective = (F.softplus(margin - separation) + calibration_weight * center.square()).mean()
    separation_gradient = torch.sigmoid(margin - separation).detach()
    center_gradient = (calibration_weight * center).detach()
    coefficients = {
        False: (separation_gradient + center_gradient) / absent.numel(),
        True: (-separation_gradient + center_gradient) / absent.numel(),
    }
    return coefficients, objective.detach()


def paired_terms(absent_logits, present_logits, **kwargs):
    """Linear backward surrogates with the objective's exact first derivatives.

    Their values are not losses; use the returned objective for logging.
    Higher derivatives through these detached coefficients are unsupported.
    """
    coefficients, objective = paired_coefficients(absent_logits, present_logits, **kwargs)
    no_id, yes_id = kwargs["no_id"], kwargs["yes_id"]
    terms = {
        capture: (
            coefficients[capture] * (logits[:, yes_id].float() - logits[:, no_id].float())
        ).sum()
        for capture, logits in ((False, absent_logits), (True, present_logits))
    }
    return terms, objective
