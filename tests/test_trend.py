import numpy as np
import pytest
from scipy.stats import spearmanr

from watching.trend import mean_bca, rank_trends, simultaneous_mean_band


def test_rank_trends_match_scipy_with_ties_and_ignore_prompt_offsets():
    steps = [0, 1, 5, 20, 90]
    values = np.array([[1, 1, 3, 2, 5], [9, 8, 7, 7, 1], [4, 4, 4, 4, 4]], dtype=float)
    rho, ranks, constants = rank_trends(values, steps)
    assert constants == 1
    assert rho[:2] == pytest.approx([spearmanr(steps, row).statistic for row in values[:2]])
    assert rho[2] == 0
    shifted, shifted_ranks, _ = rank_trends(values + np.array([[100], [-80], [12]]), steps)
    assert shifted == pytest.approx(rho)
    assert shifted_ranks == pytest.approx(ranks)


def test_monotone_and_reversed_trajectories():
    rho, _, _ = rank_trends([[1, 2, 3, 4], [4, 3, 2, 1]], [0, 10, 20, 30])
    assert rho == pytest.approx([1, -1])


def test_bootstrap_keeps_duplicate_checkpoints_identical():
    values = np.column_stack([np.zeros(20), np.arange(20), np.arange(20)])
    mean, low, high, critical = simultaneous_mean_band(values, draws=1000)
    assert mean[0] == low[0] == high[0] == 0
    assert low[1] == low[2]
    assert high[1] == high[2]
    assert critical > 0
    assert mean_bca(np.zeros(20), draws=1000) == [0, 0]


def test_invalid_checkpoint_order_rejected():
    with pytest.raises(ValueError):
        rank_trends([[1, 2, 3], [2, 3, 4]], [0, 2, 1])
