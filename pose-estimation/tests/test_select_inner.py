"""Admission tests for the inner-selection aggregator."""
from __future__ import annotations

import json

import pytest

from scripts.select_inner import aggregate_inner_runs


def _write_run(tmp_path, name, *, lr, augment, inner_fold, val_mm, best_step,
               diverged=False):
    record = {
        "diverged": diverged,
        "tag": name, "encoder": "dense", "head": "free", "fold": 0, "seed": 0,
        "imu": False, "loss": "beta_nll", "held_out": ["19"], "steps_per_epoch": 100,
        "config": {"lr": lr, "augment": augment, "dilations": [1, 2, 4, 8],
                   "window": 64, "batch": 256, "beta": 0.5},
        "inner_fold": inner_fold, "fit_subjects": ["1"], "val_subjects": ["2"],
        "val_subject_averaged_mm": val_mm, "best_step": best_step,
        "history": [],
    }
    (tmp_path / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")


def test_aggregator_selects_the_config_with_the_best_mean_and_median_budget(tmp_path):
    for inner, (good, bad, budget) in enumerate(((80.0, 90.0, 900), (82.0, 88.0, 1100),
                                                 (81.0, 89.0, 1000))):
        _write_run(tmp_path, f"good_i{inner}", lr=1e-4, augment=False,
                   inner_fold=inner, val_mm=good, best_step=budget)
        _write_run(tmp_path, f"bad_i{inner}", lr=1e-3, augment=False,
                   inner_fold=inner, val_mm=bad, best_step=500)
    selection = aggregate_inner_runs(tmp_path)
    assert selection["config"]["lr"] == 1e-4
    assert selection["mean_val_subject_averaged_mm"] == pytest.approx(81.0)
    assert selection["median_budget_steps"] == 1000
    assert selection["inner_folds"] == [0, 1, 2]
    assert len(selection["candidates"]) == 2


def test_aggregator_drops_configs_with_a_diverged_inner_run(tmp_path):
    for inner in range(3):
        _write_run(tmp_path, f"good_i{inner}", lr=1e-4, augment=False,
                   inner_fold=inner, val_mm=95.0, best_step=1000)
        _write_run(tmp_path, f"unstable_i{inner}", lr=3e-4, augment=False,
                   inner_fold=inner, val_mm=70.0, best_step=300,
                   diverged=(inner == 1))
    selection = aggregate_inner_runs(tmp_path)
    assert selection["config"]["lr"] == 1e-4          # the better score diverged
    assert len(selection["candidates"]) == 1
    assert selection["rejected_diverged"] == [{"lr": 3e-4, "inner_folds": [1]}]


def test_aggregator_rejects_when_every_config_diverged(tmp_path):
    for inner in range(3):
        _write_run(tmp_path, f"only_i{inner}", lr=1e-4, augment=True, inner_fold=inner,
                   val_mm=80.0, best_step=900, diverged=(inner == 2))
    with pytest.raises(ValueError, match="diverged"):
        aggregate_inner_runs(tmp_path)


def test_aggregator_rejects_records_from_more_than_one_outer_fold(tmp_path):
    for inner in range(3):
        _write_run(tmp_path, f"a_i{inner}", lr=1e-4, augment=True, inner_fold=inner,
                   val_mm=80.0, best_step=900)
    stray = json.loads((tmp_path / "a_i0.json").read_text())
    stray["fold"] = 2
    stray["inner_fold"] = 0
    (tmp_path / "stray.json").write_text(json.dumps(stray), encoding="utf-8")
    with pytest.raises(ValueError, match="outer fold"):
        aggregate_inner_runs(tmp_path)


def test_aggregator_rejects_incomplete_configs(tmp_path):
    _write_run(tmp_path, "only_i0", lr=1e-4, augment=True, inner_fold=0,
               val_mm=80.0, best_step=900)
    with pytest.raises(ValueError):
        aggregate_inner_runs(tmp_path)
