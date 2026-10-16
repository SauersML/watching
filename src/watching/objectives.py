"""A paired objective that separates detection from the model's answer bias."""

import torch
import torch.nn.functional as F


def paired_terms(
    absent_logits: torch.Tensor,
    present_logits: torch.Tensor,
    *,
    no_id: int,
    yes_id: int,
    margin: float = 0.25,
    calibration_weight: float = 0.1,
) -> tuple[dict[bool, torch.Tensor], torch.Tensor]:
    """Return two backward terms and the actual objective value.

    Let s be log P(Yes) - log P(No). Minimize
        mean(softplus(margin - (s_present - s_absent)) + weight * center**2),
    where center = (s_present + s_absent) / 2.

    The linear backward terms have exactly the objective's first derivatives
    at the current scores. Their detached coefficients permit separate backward
    calls and FP32 accumulation across conditions. Their values are NOT losses;
    use the separately returned objective value for logging. Higher derivatives
    through these terms are not supported.
    """
    absent = absent_logits[:, yes_id].float() - absent_logits[:, no_id].float()
    present = present_logits[:, yes_id].float() - present_logits[:, no_id].float()
    separation = present - absent
    center = (present + absent) / 2
    objective = (F.softplus(margin - separation) + calibration_weight * center.square()).mean()
    separation_gradient = torch.sigmoid(margin - separation).detach()
    center_gradient = (calibration_weight * center).detach()
    terms = {
        False: ((separation_gradient + center_gradient) * absent).mean(),
        True: ((-separation_gradient + center_gradient) * present).mean(),
    }
    return terms, objective.detach()
