import pytest
import torch

from watching.head_fit import minimum_norm_update


def test_fit_resolves_nearly_collinear_features_without_normal_equations():
    h = torch.tensor([[100, 10, 0, 0], [100, 10, 0.01, 0]], dtype=torch.float64)
    scores = torch.tensor([0.2, 0.3], dtype=torch.float64)
    targets = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    delta, singular = minimum_norm_update(h, scores, targets)
    torch.testing.assert_close(h @ delta + scores, targets, atol=1e-12, rtol=1e-12)
    assert singular[0] / singular[1] > 10000
    assert delta[3] == 0  # No update in the unconstrained null direction.
    assert delta.norm() < (delta + torch.tensor([0, 0, 0, 1])).norm()


def test_fit_rejects_pair_without_a_discriminating_direction():
    with pytest.raises(ValueError, match="rank deficient"):
        minimum_norm_update(torch.ones(2, 3), torch.zeros(2), torch.tensor([-1., 1.]))
