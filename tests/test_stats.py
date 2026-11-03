"""These tests compare area 5's statistics functions with the scipy library to prove that they are
correct.
"""
import numpy as np
import pytest
from scipy import stats

import stats_utils as su


@pytest.mark.parametrize("k,n", [(0, 10), (3, 10), (8, 10), (10, 10), (57, 200), (1, 1000)])
def test_wilson_matches_scipy(k, n):
    """Check our Wilson interval against scipy for several counts."""
    ref = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    lo, hi = su.wilson_interval(k, n)
    assert lo == pytest.approx(ref.low, abs=1e-9) and hi == pytest.approx(ref.high, abs=1e-9)


def test_bootstrap_ci_covers_mean_and_shrinks_with_n():
    """Check that the bootstrap interval contains the mean and becomes narrower with more data.
    """
    rng = np.random.default_rng(0)
    small = rng.binomial(1, 0.6, 50).astype(float); big = rng.binomial(1, 0.6, 2000).astype(float)
    lo, hi = su.bootstrap_ci(small)
    assert lo <= small.mean() <= hi
    assert (hi - lo) > (su.bootstrap_ci(big)[1] - su.bootstrap_ci(big)[0])


def test_bootstrap_coverage_simulation():
    """~95% of CIs from repeated samples should contain the true p (loose bounds, 200 reps)."""
    rng = np.random.default_rng(1); cover = 0
    for _ in range(200):
        x = rng.binomial(1, 0.7, 100).astype(float)
        lo, hi = su.bootstrap_ci(x, n_boot=400, seed=int(rng.integers(1e9)))
        cover += lo <= 0.7 <= hi
    assert 0.88 <= cover / 200 <= 0.995


def test_paired_bootstrap_detects_real_difference_not_noise():
    """Check that the paired bootstrap finds a real difference and ignores pure noise."""
    rng = np.random.default_rng(0)
    a = (rng.random(300) < 0.7).astype(float)
    b_same = a.copy()
    r = su.paired_bootstrap(a, b_same)
    assert r["mean_diff"] == 0 and r["ci"] == (0.0, 0.0)
    b_worse = a.copy(); b_worse[rng.choice(300, 60, replace=False)] = 0.0   # ~20% of items flipped to wrong
    r = su.paired_bootstrap(a, b_worse)
    assert r["ci"][0] > 0 and r["p_one_sided"] < 0.01


def test_mcnemar_matches_scipy_binomtest():
    """Check our McNemar test against scipy's binomial test."""
    a = np.array([1] * 30 + [1] * 12 + [0] * 4 + [0] * 54, dtype=bool)
    b = np.array([1] * 30 + [0] * 12 + [1] * 4 + [0] * 54, dtype=bool)
    r = su.mcnemar_exact(a, b)
    assert (r["a_only_correct"], r["b_only_correct"]) == (12, 4)
    assert r["p_two_sided"] == pytest.approx(stats.binomtest(4, 16, 0.5).pvalue)
    assert su.mcnemar_exact([1, 0], [1, 0])["p_two_sided"] == 1.0


def test_seed_summary_matches_scipy_t_interval():
    """Check our seed summary against scipy's t interval."""
    v = [0.61, 0.58, 0.66, 0.60, 0.64]
    s = su.seed_summary(v)
    lo, hi = stats.t.interval(0.95, len(v) - 1, loc=np.mean(v), scale=stats.sem(v))
    assert s["ci"][0] == pytest.approx(lo) and s["ci"][1] == pytest.approx(hi)
    assert s["std"] == pytest.approx(np.std(v, ddof=1))
