import torch
import torch.nn.functional as F

from watching.objectives import paired_terms


def test_separate_backward_terms_match_joint_objective_gradients():
    torch.manual_seed(3)
    absent = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    present = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    terms, value = paired_terms(absent, present, no_id=0, yes_id=1)
    terms[False].backward()
    terms[True].backward()
    actual_absent, actual_present = absent.grad.clone(), present.grad.clone()
    absent.grad = present.grad = None
    s0 = absent[:, 1].float() - absent[:, 0].float()
    s1 = present[:, 1].float() - present[:, 0].float()
    objective = (F.softplus(0.25 - (s1 - s0)) + 0.1 * ((s1 + s0) / 2).square()).mean()
    objective.backward()
    torch.testing.assert_close(value, objective.detach())
    torch.testing.assert_close(actual_absent, absent.grad)
    torch.testing.assert_close(actual_present, present.grad)


def test_paired_objective_rewards_separation_and_penalizes_common_bias():
    def value(s0, s1):
        return paired_terms(
            torch.tensor([[0.0, s0]]), torch.tensor([[0.0, s1]]), no_id=0, yes_id=1
        )[1].item()

    assert value(-1, 1) < value(0, 0)
    assert value(2, 2) > value(0, 0)
