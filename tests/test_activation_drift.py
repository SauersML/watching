import pytest
import torch

from watching.activation_drift import footprint_comparison


def test_projection_change_decomposition_and_rotation():
    initial = torch.tensor([[2.0, 3.0], [3.0, 3.0]])
    final = torch.tensor([[12.0, 13.0], [12.0, 14.0]])
    wa = torch.tensor([[0.0, 0.0], [1.0, 2.0]])
    wb = torch.tensor([[0.0, 0.0], [4.0, 8.0]])
    result = footprint_comparison(initial, final, wa, wb)
    assert result["delta_cosine"] == 0
    assert result["initial_projection"] == 1
    assert result["final_projection"] == 8
    assert (
        result["head_change_at_initial_features"] + result["feature_change_at_final_head"]
    ) == pytest.approx(7)


def test_zero_footprint_has_no_defined_cosine():
    pair = torch.ones(2, 3)
    result = footprint_comparison(pair, pair, pair, pair)
    assert result["delta_cosine"] is None
    assert result["initial_projection"] == 0
