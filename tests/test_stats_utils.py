"""Tests for experiments/stats_utils.py: bootstrap and Wilson CIs."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.stats_utils import bootstrap_ci, proportion_ci  # noqa: E402


# ── bootstrap_ci ──

def test_bootstrap_deterministic_with_seed():
    rng = np.random.default_rng(7)
    x = rng.normal(0.5, 1.0, size=200)
    a = bootstrap_ci(x, seed=0)
    b = bootstrap_ci(x, seed=0)
    assert a == b
    c = bootstrap_ci(x, seed=1)
    assert a != c  # different seed perturbs the resamples


def test_bootstrap_point_is_full_sample_stat():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    point, lo, hi = bootstrap_ci(x)
    assert point == pytest.approx(2.5)
    assert lo <= point <= hi


def test_bootstrap_custom_stat_fn():
    x = np.arange(100, dtype=float)
    point, lo, hi = bootstrap_ci(x, stat_fn=np.median, n_boot=500)
    assert point == pytest.approx(np.median(x))
    assert lo <= point <= hi


def test_bootstrap_degenerate_sample():
    point, lo, hi = bootstrap_ci([0.7])
    assert point == lo == hi == pytest.approx(0.7)


def test_bootstrap_coverage_on_known_distribution():
    """~95% of CIs from independent normal samples should contain the
    true mean. With 200 trials, allow a generous band around 0.95."""
    true_mean = 5.0
    rng = np.random.default_rng(123)
    n_trials = 200
    covered = 0
    for t in range(n_trials):
        x = rng.normal(true_mean, 2.0, size=80)
        _, lo, hi = bootstrap_ci(x, n_boot=500, seed=t)
        covered += int(lo <= true_mean <= hi)
    coverage = covered / n_trials
    assert 0.88 <= coverage <= 0.99, f"coverage {coverage:.3f} outside [0.88, 0.99]"


def test_bootstrap_ci_narrows_with_sample_size():
    rng = np.random.default_rng(42)
    pool = rng.normal(0.0, 1.0, size=4000)
    _, lo_s, hi_s = bootstrap_ci(pool[:50], n_boot=500)
    _, lo_l, hi_l = bootstrap_ci(pool, n_boot=500)
    assert (hi_l - lo_l) < (hi_s - lo_s)


# ── proportion_ci (Wilson) ──

def test_wilson_known_value():
    # 10/100 at 95%: published Wilson interval (0.0552, 0.1744)
    p, lo, hi = proportion_ci(10, 100)
    assert p == pytest.approx(0.1)
    assert lo == pytest.approx(0.0552, abs=1e-3)
    assert hi == pytest.approx(0.1744, abs=1e-3)


def test_wilson_extremes_stay_in_unit_interval():
    p0, lo0, hi0 = proportion_ci(0, 50)
    assert p0 == 0.0 and lo0 == pytest.approx(0.0, abs=1e-12) and 0.0 < hi0 < 0.2
    p1, lo1, hi1 = proportion_ci(50, 50)
    assert p1 == 1.0 and hi1 == pytest.approx(1.0) and 0.8 < lo1 < 1.0


def test_wilson_narrows_with_n():
    _, lo_s, hi_s = proportion_ci(8, 10)
    _, lo_l, hi_l = proportion_ci(800, 1000)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_wilson_invalid_inputs():
    with pytest.raises(ValueError):
        proportion_ci(1, 0)
    with pytest.raises(ValueError):
        proportion_ci(5, 4)
