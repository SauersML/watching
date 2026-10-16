"""Within-prompt rank trends and bootstrap intervals over complete trajectories."""

import numpy as np
from scipy.stats import bootstrap, rankdata


def rank_trends(differences, steps):
    """Return each prompt's Spearman rho and standardized within-prompt ranks.

    Rows are prompts, columns are ordered checkpoints. Constant trajectories
    contribute zero (no directional information), and are counted explicitly.
    """
    values = np.asarray(differences, dtype=float)
    steps = np.asarray(steps, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or values.shape[1] < 3
        or steps.shape != (values.shape[1],)
        or not np.isfinite(values).all()
        or not np.isfinite(steps).all()
        or not np.all(np.diff(steps) > 0)
    ):
        raise ValueError("Need finite prompt trajectories and at least three ordered checkpoints")
    ranks = rankdata(values, axis=1, method="average")
    centered = ranks - ranks.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    standardized = np.divide(centered, scale, out=np.zeros_like(centered), where=scale != 0)
    time = rankdata(steps)
    time = (time - time.mean()) / time.std()
    rho = (standardized * time).mean(axis=1)
    return rho, standardized, int(np.count_nonzero(scale == 0))


def mean_bca(values, *, draws=20000, seed=42):
    """BCa interval for the mean of independent prompt-level statistics."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Need at least two finite prompt statistics")
    if draws < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    if np.ptp(values) == 0:
        return [float(values[0]), float(values[0])]
    result = bootstrap(
        (values,),
        np.mean,
        method="BCa",
        n_resamples=draws,
        batch=256,
        confidence_level=0.95,
        rng=np.random.default_rng(seed),
    )
    return [float(result.confidence_interval.low), float(result.confidence_interval.high)]


def simultaneous_mean_band(trajectories, *, draws=20000, seed=42):
    """Approximate 95% simultaneous band across all supplied checkpoints.

    Resample rows, preserving every checkpoint within each prompt. The critical
    value is the bootstrap 95th percentile of the maximum standardized absolute
    deviation from the observed mean. Standard errors use the original sample.
    No time-exchangeability or Gaussian observation model is assumed.
    """
    values = np.asarray(trajectories, dtype=float)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Need at least two finite prompt trajectories")
    if draws < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    n = len(values)
    center = values.mean(axis=0)
    se = values.std(axis=0, ddof=1) / np.sqrt(n)
    variable = se > 0
    if not variable.any():
        return center, center.copy(), center.copy(), 0.0
    rng = np.random.default_rng(seed)
    maxima = np.empty(draws)
    for start in range(0, draws, 256):
        stop = min(start + 256, draws)
        indices = rng.integers(n, size=(stop - start, n))
        means = values[indices].mean(axis=1)
        maxima[start:stop] = np.max(
            np.abs(means[:, variable] - center[variable]) / se[variable], axis=1
        )
    critical = float(np.quantile(maxima, 0.95))
    return center, center - critical * se, center + critical * se, critical
