"""Admission tests for the formal report builder."""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.formal_report import build_formal_report, participant_scores

ARMS = ("ponly_dense", "ponly_moments", "shank_imu_dense", "shank_imu_moments")
FOLDS = {0: ["1", "2"], 1: ["3", "4"], 2: ["5", "6"], 3: ["7"]}
BASE = {"ponly_dense": 90.0, "ponly_moments": 98.0, "shank_imu_dense": 80.0,
        "shank_imu_moments": 92.0}
SIZES = {"1": 63.72e-6, "2": 54.34e-6, "3": 63.72e-6, "4": 54.34e-6,
         "5": 63.72e-6, "6": 63.72e-6, "7": 54.34e-6}


def _write_matrix(root, *, seeds=(0, 1, 2), variants=ARMS, drop=None, strata=True):
    rng = np.random.default_rng(0)
    for fold, subjects in FOLDS.items():
        for variant in variants:
            reports = root / f"f{fold}" / variant / "reports"
            reports.mkdir(parents=True)
            (root / f"f{fold}" / variant / "selection.json").write_text(json.dumps({
                "config": {"lr": 1e-4, "augment": True, "encoder": variant.split("_")[1],
                           "shank_imu": variant.startswith("shank_imu")},
                "median_budget_steps": 1000 + fold,
                "mean_val_subject_averaged_mm": BASE[variant] + 2.0,
            }), encoding="utf-8")
            for seed in seeds:
                if drop == (fold, variant, seed):
                    continue
                subject_rows = {
                    s: {"frames_scored": 1000,
                        "mpjpe_mm": BASE[variant] + rng.normal(0.0, 1.0) + int(s)}
                    for s in subjects
                }
                report = {
                    "schema": "streaming-eval-v1", "label_source": "cache",
                    "alignment": "pelvis_relative_yaw_canonical;no_procrustes;no_scale",
                    "availability_delay_s": 0.35 if variant.startswith("shank_imu") else 0.10,
                    "block": 1, "segments": 2, "frames": 2000,
                    "frames_scored": 2000, "invalid_input_frames": 10,
                    "subjects": subject_rows,
                    "subject_averaged_mpjpe_mm": float(
                        np.mean([v["mpjpe_mm"] for v in subject_rows.values()])),
                    "pooled_frame_mpjpe_mm": 1.0,
                }
                if strata:
                    report["strata"] = {
                        "activity": {"still": {"frames_scored": 2000,
                                               "subject_averaged_mpjpe_mm": BASE[variant]}},
                        "marker": {"L_LM": {"frames_scored": 2000,
                                            "subject_averaged_mpjpe_mm": BASE[variant] - 5}},
                        "phase": {"stance": {"frames_scored": 1200,
                                             "subject_averaged_mpjpe_mm": BASE[variant] - 8},
                                  "swing": {"frames_scored": 800,
                                            "subject_averaged_mpjpe_mm": BASE[variant] + 8}},
                        "locomotion": {"gait": {"frames_scored": 2000,
                                                "subject_averaged_mpjpe_mm": BASE[variant]}},
                    }
                (reports / f"test_s{seed}.json").write_text(
                    json.dumps({"fold": fold, "split": "test", "run_tag": f"{variant}_f{fold}_s{seed}",
                                "report": report}), encoding="utf-8")


def test_participant_scores_average_seeds_within_each_participant(tmp_path):
    _write_matrix(tmp_path)
    scores = participant_scores(tmp_path, "ponly_dense")
    assert sorted(scores, key=int) == [s for f in FOLDS.values() for s in f]
    one = json.loads((tmp_path / "f0/ponly_dense/reports/test_s0.json").read_text())
    seeds = [json.loads((tmp_path / f"f0/ponly_dense/reports/test_s{k}.json").read_text())
             ["report"]["subjects"]["1"]["mpjpe_mm"] for k in (0, 1, 2)]
    assert scores["1"] == pytest.approx(float(np.mean(seeds)))
    assert one["report"]["availability_delay_s"] == 0.10


def test_primary_comparison_is_the_fusion_minus_pressure_paired_difference(tmp_path):
    _write_matrix(tmp_path)
    report = build_formal_report(tmp_path, n_boot=2000, seed=0, insole_areas=SIZES)
    primary = report["primary"]
    assert primary["contrast"] == "shank_imu_dense - ponly_dense"
    assert primary["n"] == 7
    assert primary["mean_diff"] < 0.0                     # fusion is better here
    assert primary["wilcoxon_p"] < 0.05
    low, high = primary["bca_95ci"]
    assert low < primary["mean_diff"] < high
    assert report["holm_adjusted_p"]["primary"] >= primary["wilcoxon_p"]
    assert set(report["holm_adjusted_p"]) == {"primary", "co_primary"}


def test_report_carries_every_contract_element(tmp_path):
    _write_matrix(tmp_path)
    report = build_formal_report(tmp_path, n_boot=2000, seed=0, insole_areas=SIZES)
    assert report["schema"] == "formal-report-v1"
    assert report["participants"] == 7
    assert report["alignment"] == "pelvis_relative_yaw_canonical;no_procrustes;no_scale"
    assert report["availability_delay_s"] == {"ponly": 0.10, "shank_imu": 0.35}
    assert set(report["variants"]) == set(ARMS)
    for variant in ARMS:
        assert report["variants"][variant]["subject_averaged_mpjpe_mm"] > 0.0
        assert len(report["variants"][variant]["per_participant_mm"]) == 7
        assert set(report["variants"][variant]["strata"]) == {"activity", "marker", "phase",
                                                      "locomotion"}
    rows = report["fold_configurations"]
    assert len(rows) == len(FOLDS) * len(ARMS)
    for variant in ARMS:
        per_variant = [row for row in rows if row["variant"] == variant]
        assert sorted(row["fold"] for row in per_variant) == sorted(FOLDS)
        assert all(row["config"] and row["budget_steps"] > 0 for row in per_variant)
    assert report["seeds_per_variant"] == {"contrast": 3, "reference_available": 3}
    assert "baselines" in report


def test_summary_condition_takes_the_better_of_the_two_encoders(tmp_path):
    _write_matrix(tmp_path)
    report = build_formal_report(tmp_path, n_boot=2000, seed=0, insole_areas=SIZES)
    assert report["co_primary"]["contrast"] == "shank_imu_dense - shank_imu_summary"
    assert report["co_primary"]["summary_variant"] == "shank_imu_moments"


def test_the_reference_may_carry_the_five_seed_tier(tmp_path):
    """design.md section 6: three seeds for a primary contrast, five for the E0
    reference — and the reference is the fused dense variant, so one variant holds both."""
    _write_matrix(tmp_path)
    for fold, subjects in FOLDS.items():
        reports = tmp_path / f"f{fold}" / "shank_imu_dense" / "reports"
        template = json.loads((reports / "test_s0.json").read_text())
        for seed in (3, 4):
            record = json.loads(json.dumps(template))
            record["run_tag"] = f"shank_imu_dense_f{fold}_s{seed}"
            for entry in record["report"]["subjects"].values():
                entry["mpjpe_mm"] += 0.5 * seed
            (reports / f"test_s{seed}.json").write_text(json.dumps(record), encoding="utf-8")

    report = build_formal_report(tmp_path, n_boot=500, seed=0, insole_areas=SIZES)
    assert report["seeds_per_variant"] == {"contrast": 3, "reference_available": 5}
    # the paired contrast still uses the declared three, not whatever exists
    contrast_only = build_formal_report(
        tmp_path, n_boot=500, seed=0, insole_areas=SIZES)["primary"]
    assert contrast_only["n"] == 7
    assert report["e0_reference"]["seeds"] == [0, 1, 2, 3, 4]
    assert report["e0_reference"]["per_seed_subject_averaged_mm"]


def test_report_shows_fold_variation_per_variant(tmp_path):
    """design.md section 6: per-fold means are the fold-variation display."""
    _write_matrix(tmp_path)
    report = build_formal_report(tmp_path, n_boot=500, seed=0, insole_areas=SIZES)
    for variant in ARMS:
        per_fold = report["variants"][variant]["per_fold_mm"]
        assert sorted(per_fold) == [str(f) for f in sorted(FOLDS)]
        assert all(value > 0.0 for value in per_fold.values())
        assert report["variants"][variant]["fold_range_mm"] >= 0.0
        # the subject-averaged mean is the mean over participants, not over folds
        assert report["variants"][variant]["subject_averaged_mpjpe_mm"] > 0.0


def test_insole_size_stratum_splits_the_cohort(tmp_path):
    """design.md section 5: the insole-size stratum is reported at the fine end."""
    _write_matrix(tmp_path)
    report = build_formal_report(tmp_path, n_boot=500, seed=0, insole_areas=SIZES)
    for variant in ARMS:
        stratum = report["variants"][variant]["insole_size_mm"]
        assert set(stratum) == {"63.72", "54.34"}
        assert all(value > 0.0 for value in stratum.values())
    counts = report["insole_size_participants"]
    assert counts == {"63.72": 4, "54.34": 3}


def test_quality_column_sits_beside_the_all_segment_headline(tmp_path):
    """opensim-gates G3 and CLAUDE.md EVAL-3: the QC cutoff is an extra column,
    and the headline still scores every segment."""
    _write_matrix(tmp_path)
    # one segment per participant is above its fold's cutoff
    quality, cutoffs = {}, {}
    for fold, subjects in FOLDS.items():
        cutoffs[str(fold)] = 0.030
        for subject in subjects:
            quality[f"{subject}/still/a"] = 0.020
            quality[f"{subject}/still/b"] = 0.050          # fails the cutoff
    for fold in FOLDS:
        for variant in ARMS:
            for path in (tmp_path / f"f{fold}" / variant / "reports").glob("test_s*.json"):
                record = json.loads(path.read_text())
                subjects = record["report"]["subjects"]
                record["report"]["per_segment"] = {
                    f"{s}/still/{name}": {"subject": s, "frames_scored": 500,
                                          "mpjpe_mm": entry["mpjpe_mm"] + offset}
                    for s, entry in subjects.items()
                    for name, offset in (("a", -2.0), ("b", 2.0))
                }
                path.write_text(json.dumps(record), encoding="utf-8")

    report = build_formal_report(tmp_path, n_boot=500, seed=0, insole_areas=SIZES,
                                 segment_quality=quality, quality_cutoffs=cutoffs)
    for variant in ARMS:
        block = report["variants"][variant]
        assert "quality_filtered_mpjpe_mm" in block
        # dropping the worse segment of each participant can only lower the score
        assert block["quality_filtered_mpjpe_mm"] < block["subject_averaged_mpjpe_mm"]
    assert report["quality_filter"]["segments_excluded"] == 7      # one per participant
    assert report["quality_filter"]["headline"] == "all_segments"


def test_the_report_states_the_realised_likelihood_share_not_the_configured_loss(tmp_path):
    """A budget shorter than its own warm-up never reaches beta-NLL, while the
    record still names it."""
    from scripts.train_moveport import WARMUP_EPOCHS

    _write_matrix(tmp_path)
    for fold in FOLDS:
        retrain = tmp_path / f"f{fold}" / "shank_imu_dense" / "retrain"
        retrain.mkdir(parents=True)
        # fold 0 spends its whole budget on warm-up; the others reach the likelihood
        steps = 500 * WARMUP_EPOCHS if fold else 10 * WARMUP_EPOCHS
        (retrain / "run_s0_steps.json").write_text(json.dumps(
            {"steps": steps, "steps_per_epoch": 100,
             "config": {"loss": "beta_nll"}}), encoding="utf-8")

    report = build_formal_report(tmp_path, n_boot=200, seed=0, insole_areas=SIZES)
    shares = report["variants"]["shank_imu_dense"]["beta_nll_share_of_budget"]
    assert shares["0"] == pytest.approx(0.0)
    for fold in ("1", "2", "3"):
        assert shares[fold] == pytest.approx(0.8)
    assert report["variants"]["ponly_dense"]["beta_nll_share_of_budget"] == {}


def test_a_diverged_retrained_model_cannot_be_scored_into_the_matrix(tmp_path):
    """`select_inner.py` drops a diverged configuration; nothing read the same
    flag on the model a held-out score actually comes from."""
    _write_matrix(tmp_path)
    build_formal_report(tmp_path, n_boot=200, seed=0, insole_areas=SIZES)
    retrain = tmp_path / "f1" / "shank_imu_dense" / "retrain"
    retrain.mkdir(parents=True)
    (retrain / "shank_imu_dense_f1_s1_steps1000.json").write_text(
        json.dumps({"steps": 1000, "diverged": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="diverged"):
        build_formal_report(tmp_path, n_boot=200, seed=0, insole_areas=SIZES)


def test_a_quality_key_that_never_joins_is_refused(tmp_path):
    """Segment ids carry underscores on both sides of the activity/name split,
    so a mismatched key would filter nothing and read as nothing failing."""
    _write_matrix(tmp_path)
    for fold in FOLDS:
        for variant in ARMS:
            for path in (tmp_path / f"f{fold}" / variant / "reports").glob("test_s*.json"):
                record = json.loads(path.read_text())
                record["report"]["per_segment"] = {
                    f"{s}/treadmill_normal/high_1": {"subject": s, "frames_scored": 500,
                                                     "mpjpe_mm": entry["mpjpe_mm"]}
                    for s, entry in record["report"]["subjects"].items()}
                path.write_text(json.dumps(record), encoding="utf-8")
    mangled = {f"{s}/treadmill_normal_high/1": 0.015
               for f in FOLDS.values() for s in f}
    with pytest.raises(ValueError, match="no fit residual"):
        build_formal_report(tmp_path, n_boot=500, seed=0, insole_areas=SIZES,
                            segment_quality=mangled,
                            quality_cutoffs={str(f): 0.03 for f in FOLDS})


def test_builder_refuses_to_report_without_the_insole_lookup(tmp_path):
    """design.md section 5: a missing table drops the stratum silently, so the
    builder refuses rather than emitting an empty group."""
    _write_matrix(tmp_path)
    with pytest.raises(ValueError, match="insole"):
        build_formal_report(tmp_path, n_boot=500, seed=0)
    partial = {s: area for s, area in SIZES.items() if s != "7"}
    with pytest.raises(ValueError, match="7"):
        build_formal_report(tmp_path, n_boot=500, seed=0, insole_areas=partial)


def test_a_segment_with_no_scored_frames_stays_valid_json(tmp_path):
    """A bare NaN is not JSON; an unscorable segment carries null and is skipped."""
    _write_matrix(tmp_path)
    quality = {f"{s}/still/{name}": 0.020
               for f in FOLDS.values() for s in f for name in ("a", "empty")}
    cutoffs = {str(fold): 0.030 for fold in FOLDS}
    for fold in FOLDS:
        for variant in ARMS:
            for path in (tmp_path / f"f{fold}" / variant / "reports").glob("test_s*.json"):
                record = json.loads(path.read_text())
                record["report"]["per_segment"] = {
                    f"{s}/still/a": {"subject": s, "frames_scored": 500,
                                     "mpjpe_mm": entry["mpjpe_mm"]}
                    for s, entry in record["report"]["subjects"].items()
                } | {
                    f"{s}/still/empty": {"subject": s, "frames_scored": 0,
                                         "mpjpe_mm": None}
                    for s in record["report"]["subjects"]
                }
                path.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")

    report = build_formal_report(tmp_path, n_boot=500, seed=0, insole_areas=SIZES,
                                 segment_quality=quality, quality_cutoffs=cutoffs)
    for variant in ARMS:
        scored = report["variants"][variant]["quality_filtered_mpjpe_mm"]
        assert scored == pytest.approx(report["variants"][variant]["subject_averaged_mpjpe_mm"])
    json.dumps(report, allow_nan=False)


def test_the_command_line_report_carries_both_extra_columns(tmp_path, monkeypatch):
    """The production entry point, not just the builder: the insole stratum and
    the quality column must both survive the flags the queue actually passes."""
    import scripts.formal_report as module

    _write_matrix(tmp_path)
    for fold in FOLDS:
        for variant in ARMS:
            for path in (tmp_path / f"f{fold}" / variant / "reports").glob("test_s*.json"):
                record = json.loads(path.read_text())
                record["report"]["per_segment"] = {
                    f"{s}/still/{name}": {"subject": s, "frames_scored": 500,
                                          "mpjpe_mm": entry["mpjpe_mm"] + offset}
                    for s, entry in record["report"]["subjects"].items()
                    for name, offset in (("1", -2.0), ("2", 2.0))
                }
                path.write_text(json.dumps(record), encoding="utf-8")

    quality = {"schema": "segment-fit-quality-v1", "per_segment": {
        f"{s}/still/{name}": {"subject": s, "mean_rms_m": rms}
        for f in FOLDS.values() for s in f
        for name, rms in (("1", 0.015), ("2", 0.016 if s != "7" else 0.900))}}
    (tmp_path / "quality.json").write_text(json.dumps(quality), encoding="utf-8")
    (tmp_path / "insole.json").write_text(json.dumps(SIZES), encoding="utf-8")
    width = max(len(v) for v in FOLDS.values())
    subjects = np.array([[s for s in v] + [""] * (width - len(v))
                         for v in FOLDS.values()], dtype="U4")
    np.savez(tmp_path / "cache.npz", fold_subjects=subjects)

    out = tmp_path / "formal_report.json"
    monkeypatch.setattr("sys.argv", [
        "formal_report.py", "--matrix", str(tmp_path), "--out", str(out),
        "--insole-areas", str(tmp_path / "insole.json"),
        "--segment-quality", str(tmp_path / "quality.json"),
        "--cache", str(tmp_path / "cache.npz"), "--n-boot", "200"])
    assert module.main() == 0

    report = json.loads(out.read_text())
    assert set(report["insole_size_participants"]) == {"63.72", "54.34"}
    # subject 7 is held out in fold 3, so its own bad trial cannot set the cutoff
    assert report["quality_filter"]["excluded_ids"] == ["7/still/2"]
    for variant in ARMS:
        assert report["variants"][variant]["quality_filtered_mpjpe_mm"] > 0.0
        assert report["variants"][variant]["insole_size_mm"]


def test_builder_refuses_an_incomplete_matrix(tmp_path):
    _write_matrix(tmp_path, drop=(1, "shank_imu_dense", 2))
    with pytest.raises(ValueError, match="seed"):
        build_formal_report(tmp_path, n_boot=500, seed=0, insole_areas=SIZES)


def test_builder_refuses_variants_scored_on_different_participants(tmp_path):
    _write_matrix(tmp_path)
    path = tmp_path / "f0/shank_imu_dense/reports/test_s0.json"
    record = json.loads(path.read_text())
    record["report"]["subjects"].pop("2")
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="participant"):
        build_formal_report(tmp_path, n_boot=500, seed=0, insole_areas=SIZES)
