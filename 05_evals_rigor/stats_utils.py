"""This file holds the statistics functions that area 5 uses, such as confidence intervals and
significance tests. The tests compare them against the scipy library. It is part of area 5.

More technical notes:
Statistics for honest evaluation (numpy/scipy only; unit-tested against scipy in tests/).

* wilson_interval      : binomial CI for an accuracy (Wilson 1927) - good for small n and p near 0/1
* bootstrap_ci         : percentile bootstrap CI of the mean of per-item scores
* paired_bootstrap     : CI + one-sided p for mean(score_A - score_B) on the SAME items (Koehn 2004 style)
* mcnemar_exact        : exact two-sided McNemar test for paired binary outcomes
* seed_summary         : mean, sample std, t-based 95% CI across independent seeds / runs
Why: a 135M-parameter model on 200 questions has a standard error of several points; differences smaller than the
CI are NOT evidence. See Dror et al. 2018 (ACL) "The Hitchhiker's Guide to Testing Statistical Significance in NLP"
https://aclanthology.org/P18-1128/ and Madaan et al. 2024 "Quantifying Variance in Evaluation Benchmarks" arXiv:2406.10229.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def wilson_interval(k: int, n: int, confidence: float = 0.95):
    """Return the Wilson confidence interval for a proportion, for example the accuracy after k
    correct answers out of n.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    z = stats.norm.ppf(0.5 + confidence / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(x, confidence: float = 0.95, n_boot: int = 10000, seed: int = 0):
    """Return a bootstrap confidence interval for the average of a list, by re-sampling the
    list many times.
    """
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
    return float(lo), float(hi)


def paired_bootstrap(a, b, n_boot: int = 10000, seed: int = 0, confidence: float = 0.95):
    """mean(a-b) with percentile CI and a one-sided p-value for H1: mean(a) > mean(b)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
    p = float((np.sum(means <= 0) + 1) / (n_boot + 1))
    return {"mean_diff": float(d.mean()), "ci": (float(lo), float(hi)), "p_one_sided": p}


def mcnemar_exact(a_correct, b_correct):
    """Exact McNemar on discordant pairs: b01 = A wrong & B right, b10 = A right & B wrong."""
    a, b = np.asarray(a_correct, dtype=bool), np.asarray(b_correct, dtype=bool)
    b10, b01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = b10 + b01
    p = 1.0 if n == 0 else float(stats.binomtest(min(b10, b01), n, 0.5, alternative="two-sided").pvalue)
    return {"a_only_correct": b10, "b_only_correct": b01, "p_two_sided": p}


def seed_summary(values, confidence: float = 0.95):
    """Return the average, the spread and a t-based confidence interval of scores obtained with
    different random seeds.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean = float(v.mean())
    if n < 2:
        return {"mean": mean, "std": float("nan"), "ci": (float("nan"), float("nan")), "n": n}
    sd = float(v.std(ddof=1))
    h = float(stats.t.ppf(0.5 + confidence / 2, n - 1) * sd / math.sqrt(n))
    return {"mean": mean, "std": sd, "ci": (mean - h, mean + h), "n": n}
