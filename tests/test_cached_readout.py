import json
import math

import pytest
import torch

from watching.cached_readout import fit_readout, normalize_features, paired_statistics


def test_normalization_uses_training_executions_without_condition_information():
    train = torch.tensor([[[1.0, 2.0], [3.0, 2.0]], [[5.0, 2.0], [7.0, 2.0]]])
    heldout = torch.tensor([[[1000.0, 2.0], [1002.0, 2.0]]])
    normalized, center, scale = normalize_features(train, {"train": train, "test": heldout})
    torch.testing.assert_close(center, torch.tensor([4.0, 2.0]))
    torch.testing.assert_close(scale, torch.tensor([math.sqrt(5), 1e-5]))
    assert normalized["test"].mean() > 100
    # Both executions use the same transform; it does not remove their difference.
    torch.testing.assert_close(
        normalized["test"][0, 1] - normalized["test"][0, 0],
        torch.tensor([2 / math.sqrt(5), 0.0]),
        atol=3e-5,
        rtol=1e-5,
    )


def test_bootstrap_resamples_whole_pairs_so_shared_prompt_bias_cancels():
    bias = torch.tensor([-100.0, 0.0, 100.0])
    result = paired_statistics(torch.stack((bias, bias + 2), dim=1))
    assert result["paired_log_odds_shift"] == 2
    assert result["paired_log_odds_shift_ci95"] == [2.0, 2.0]
    assert result["balanced_binary_ce"] > 30  # Separation alone is not calibration.
    assert paired_statistics(torch.zeros(1, 2))["paired_log_odds_shift_ci95"] is None
    assert paired_statistics(torch.zeros(10, 2))["balanced_binary_ce"] == pytest.approx(math.log(2))


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_cached_readout_learns_common_signal_on_unseen_individual_states(tmp_path, kind):
    generator = torch.Generator().manual_seed(17)
    features = {}
    for split, n in (("train", 32), ("validation", 16), ("test", 16)):
        state = torch.randn(n, 2, 3, generator=generator) * 0.1
        state[:, 0, 0] = -1
        state[:, 1, 0] = 1
        features[split] = state
    fit_readout(
        features,
        kind,
        seed=2,
        steps=100,
        eval_every=100,
        batch_pairs=16,
        lr=0.03,
        weight_decay=0,
        hidden=8,
        emit=lambda *a, **kw: None,
        output=tmp_path,
    )
    result = json.loads((tmp_path / f"{kind}_seed2.json").read_text())
    test = result["results"]["test"]["summary"]
    assert test["balanced_binary_ce"] < 0.2
    assert test["paired_log_odds_shift"] > 3


def test_validation_selection_does_not_use_test_outcomes(tmp_path):
    train = torch.tensor([[[-1.0], [1.0]]]).repeat(16, 1, 1)
    features = {"train": train, "validation": -train, "test": train}
    fit_readout(
        features,
        "linear",
        seed=2,
        steps=16,
        eval_every=8,
        batch_pairs=8,
        lr=0.1,
        weight_decay=0,
        hidden=8,
        emit=lambda *a, **kw: None,
        output=tmp_path,
    )
    result = json.loads((tmp_path / "linear_seed2.json").read_text())
    assert result["selected_step"] == 0  # Test would prefer the trained classifier.
    assert result["results"]["test"]["summary"]["paired_log_odds_shift"] == 0
