"""Statistical utilities for eval scripts: bootstrap and proportion CIs.

V1 of the SAE validation plan (latent-labs/guides/plan-sae-validation.md):
every headline metric ships with a confidence interval. Treat sub-CI
differences between runs as ties.
"""

import numpy as np
from scipy.stats import norm


def bootstrap_ci(values, stat_fn=np.mean, n_boot=2000, ci=0.95, seed=0):
    """Percentile bootstrap CI for a statistic over a 1-D sample.

    Args:
        values: array-like of per-item values (e.g. per-example correctness,
            per-query nDCG, per-feature scores).
        stat_fn: statistic to bootstrap (default np.mean).
        n_boot: number of bootstrap resamples.
        ci: confidence level (default 0.95).
        seed: RNG seed (deterministic for a fixed seed).

    Returns:
        (point, lo, hi) — the statistic on the full sample and the
        percentile CI bounds.
    """
    values = np.asarray(values)
    point = float(stat_fn(values))
    n = len(values)
    if n < 2:
        return point, point, point

    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = stat_fn(values[rng.integers(0, n, size=n)])

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return point, float(lo), float(hi)


def proportion_ci(k, n, ci=0.95):
    """Wilson score interval for a binomial proportion (e.g. COH% of labels).

    Args:
        k: number of successes.
        n: number of trials.
        ci: confidence level (default 0.95).

    Returns:
        (point, lo, hi) — k/n and the Wilson interval bounds.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must be in [0, n]")

    p = k / n
    z = norm.ppf(1.0 - (1.0 - ci) / 2.0)
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)
