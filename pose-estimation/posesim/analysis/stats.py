"""Participant-level paired statistics for the frozen reporting protocol.

Wilcoxon signed-rank is the primary inference; the paired t-test and Cohen's
d_z are sensitivity reports; the BCa bootstrap interval covers the mean paired
difference; Holm adjusts the two co-primary comparisons. No equivalence claim is
computed anywhere.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats


def bca_interval(
    values: np.ndarray,
    *,
    n_boot: int = 15000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Bias-corrected accelerated bootstrap interval of the mean."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("values must be one finite vector with at least three entries")
    if not 0.0 < alpha < 1.0 or n_boot < 100:
        raise ValueError("alpha must be in (0, 1) and n_boot at least 100")
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = rng.choice(values, size=(n_boot, n), replace=True).mean(axis=1)
    theta = values.mean()
    proportion = np.clip((boot < theta).mean(), 1.0 / n_boot, 1.0 - 1.0 / n_boot)
    z0 = scipy_stats.norm.ppf(proportion)
    jackknife = (values.sum() - values) / (n - 1)
    centred = jackknife.mean() - jackknife
    denominator = np.power((centred**2).sum(), 1.5)
    acceleration = (centred**3).sum() / (6.0 * denominator) if denominator > 0 else 0.0
    z = scipy_stats.norm.ppf([alpha / 2.0, 1.0 - alpha / 2.0])
    adjusted = scipy_stats.norm.cdf(z0 + (z0 + z) / (1.0 - acceleration * (z0 + z)))
    low, high = np.quantile(boot, adjusted)
    return float(low), float(high)


def paired_report(a: np.ndarray, b: np.ndarray, *, n_boot: int = 15000, seed: int = 0) -> dict:
    """Paired participant-level comparison of two matched score vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 3:
        raise ValueError("paired scores must be two matched vectors of at least three subjects")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("paired scores must be finite")
    difference = a - b
    wilcoxon = scipy_stats.wilcoxon(difference)
    ttest = scipy_stats.ttest_rel(a, b)
    return {
        "n": int(len(a)),
        "mean_diff": float(difference.mean()),
        "wilcoxon_p": float(wilcoxon.pvalue),
        "t_p": float(ttest.pvalue),
        "cohen_dz": float(difference.mean() / difference.std(ddof=1)),
        "bca_95ci": bca_interval(difference, n_boot=n_boot, seed=seed),
    }


def holm_adjust(pvalues) -> list[float]:
    """Holm step-down adjusted p-values, monotone and capped at one."""
    pvalues = [float(p) for p in pvalues]
    if not pvalues or any(not 0.0 <= p <= 1.0 for p in pvalues):
        raise ValueError("pvalues must be a non-empty list inside [0, 1]")
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted
