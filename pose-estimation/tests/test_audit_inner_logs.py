"""Admission tests for recovering inner-selection runs from their logs."""
from __future__ import annotations

import json

import pytest

from scripts.audit_inner_logs import audit, read_log, select_from_logs

TAIL = "{tag}: inner val {val:.1f} mm subject-averaged; budget {budget} steps\n"


def _log(directory, lr, aug, inner, val, budget, diverged=False):
    directory.mkdir(parents=True, exist_ok=True)
    body = f"tag: 100 train / 50 val windows\n    0  train 120.0 mm   val 130.0 mm   mse\n"
    if diverged:
        body += "    9  diverged; stopping this run\n"
    else:
        body += TAIL.format(tag="tag", val=val, budget=budget)
    (directory / f"log_{lr}_{aug}_i{inner}").write_text(body, encoding="utf-8")


def test_the_learning_rate_comes_from_the_filename_the_json_lost(tmp_path):
    _log(tmp_path, "1e-4", "with", 0, 90.5, 11934)
    row = read_log(tmp_path / "log_1e-4_with_i0")
    assert row == {"lr": 1e-4, "augment": True, "inner_fold": 0, "diverged": False,
                   "val_mm": 90.5, "budget_steps": 11934}


def test_a_diverged_run_carries_no_score(tmp_path):
    _log(tmp_path, "3e-4", "without", 1, 0.0, 0, diverged=True)
    row = read_log(tmp_path / "log_3e-4_without_i1")
    assert row["diverged"] and row["val_mm"] is None


def test_a_truncated_log_is_refused(tmp_path):
    (tmp_path / "log_1e-4_with_0").write_text("nothing\n", encoding="utf-8")
    (tmp_path / "log_1e-4_with_i0").write_text("no summary line\n", encoding="utf-8")
    with pytest.raises(ValueError, match="summary line"):
        read_log(tmp_path / "log_1e-4_with_i0")


def test_the_full_sweep_can_pick_the_rate_the_overwrite_hid(tmp_path):
    """Four configurations, and the surviving JSON would only have seen two."""
    for inner in range(3):
        _log(tmp_path, "1e-4", "with", inner, 80.0, 1000)
        _log(tmp_path, "1e-4", "without", inner, 95.0, 1000)
        _log(tmp_path, "3e-4", "with", inner, 85.0, 2000)
        _log(tmp_path, "3e-4", "without", inner, 90.0, 2000)
    recovered = select_from_logs(tmp_path)
    assert len(recovered["candidates"]) == 4
    assert recovered["winner"]["lr"] == 1e-4
    assert recovered["winner"]["augment"] is True
    assert recovered["winner"]["median_budget_steps"] == 1000


def test_a_diverged_configuration_is_dropped_not_ranked(tmp_path):
    for inner in range(3):
        _log(tmp_path, "1e-4", "with", inner, 70.0, 1000, diverged=inner == 2)
        _log(tmp_path, "3e-4", "with", inner, 85.0, 2000)
    recovered = select_from_logs(tmp_path)
    assert recovered["rejected_diverged"] == [{"lr": 1e-4, "augment": True}]
    assert recovered["winner"]["lr"] == 3e-4


def test_audit_flags_the_variant_folds_whose_configuration_changes(tmp_path):
    variant = tmp_path / "f0" / "shank_imu_dense"
    for inner in range(3):
        _log(variant / "inner", "1e-4", "with", inner, 80.0, 1000)
        _log(variant / "inner", "1e-4", "without", inner, 95.0, 1000)
        _log(variant / "inner", "3e-4", "with", inner, 85.0, 2000)
        _log(variant / "inner", "3e-4", "without", inner, 90.0, 2000)
    (variant / "selection.json").write_text(json.dumps({
        "config": {"lr": 3e-4, "augment": True}, "median_budget_steps": 2000,
        "mean_val_subject_averaged_mm": 85.0}), encoding="utf-8")

    report = audit(tmp_path)
    assert report["variant_folds"] == 1
    assert report["configuration_changes"] == 1
    row = report["rows"][0]
    assert row["executed_lr"] == 3e-4 and row["full_sweep_lr"] == 1e-4
    assert row["executed_budget_steps"] == 2000 and row["full_sweep_budget_steps"] == 1000
    assert row["margin_over_runner_up_mm"] == pytest.approx(5.0)


def test_a_surviving_record_outranks_the_log_it_would_tie_with(tmp_path):
    """One decimal is coarse enough to tie two configurations the full-precision
    record separates, and the tie would be broken by sort order alone."""
    for inner in range(3):
        _log(tmp_path, "3e-4", "with", inner, 85.2, 4520)
        _log(tmp_path, "3e-4", "without", inner, 85.2, 3021)
    assert select_from_logs(tmp_path)["winner"]["augment"] is False   # tie, sort order wins

    for inner, (aug_val, noaug_val) in enumerate(
            [(86.28, 87.55), (83.07, 80.18), (86.33, 88.03)]):
        for augment, value in ((True, aug_val), (False, noaug_val)):
            tag = "aug" if augment else "noaug"
            (tmp_path / f"run_{tag}_i{inner}.json").write_text(json.dumps({
                "inner_fold": inner, "best_step": 4520 if augment else 3021,
                "val_subject_averaged_mm": value,
                "config": {"lr": 3e-4, "augment": augment}}), encoding="utf-8")
    recovered = select_from_logs(tmp_path)
    assert recovered["winner"]["augment"] is True
    assert recovered["winner"]["source"] == "json"
    assert recovered["winner"]["mean_val_mm"] == pytest.approx(85.2267, abs=1e-3)
