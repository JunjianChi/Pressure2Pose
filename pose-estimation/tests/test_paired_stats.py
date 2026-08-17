"""Admission tests for the participant-level paired statistics."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

from posesim.analysis.stats import bca_interval, holm_adjust, paired_report


def test_paired_report_matches_scipy_wilcoxon_and_t():
    rng = np.random.default_rng(0)
    a = 80.0 + rng.normal(0.0, 5.0, size=23)
    b = a - 3.0 + rng.normal(0.0, 1.0, size=23)
    report = paired_report(a, b, n_boot=2000, seed=0)
    wilcoxon = scipy_stats.wilcoxon(a - b)
    ttest = scipy_stats.ttest_rel(a, b)
    assert report["n"] == 23
    assert report["mean_diff"] == pytest.approx(float(np.mean(a - b)))
    assert report["wilcoxon_p"] == pytest.approx(wilcoxon.pvalue)
    assert report["t_p"] == pytest.approx(ttest.pvalue)
    assert report["cohen_dz"] == pytest.approx(
        float(np.mean(a - b) / np.std(a - b, ddof=1)))
    low, high = report["bca_95ci"]
    assert low < report["mean_diff"] < high
    assert low > 0.0    # a is consistently larger than b
    repeat = paired_report(a, b, n_boot=2000, seed=0)
    assert repeat["bca_95ci"] == report["bca_95ci"]


def test_bca_interval_shrinks_with_sample_size_and_brackets_the_mean():
    rng = np.random.default_rng(1)
    small = rng.normal(10.0, 2.0, size=15)
    large = rng.normal(10.0, 2.0, size=200)
    low_s, high_s = bca_interval(small, n_boot=3000, seed=0)
    low_l, high_l = bca_interval(large, n_boot=3000, seed=0)
    assert low_s < np.mean(small) < high_s
    assert low_l < np.mean(large) < high_l
    assert (high_l - low_l) < (high_s - low_s)


def test_holm_adjustment_controls_the_family():
    adjusted = holm_adjust([0.01, 0.04])
    assert adjusted[0] == pytest.approx(0.02)
    assert adjusted[1] == pytest.approx(0.04)
    assert holm_adjust([0.03]) == [pytest.approx(0.03)]
    monotone = holm_adjust([0.001, 0.5, 0.04])
    assert monotone[0] <= monotone[2] <= monotone[1]
