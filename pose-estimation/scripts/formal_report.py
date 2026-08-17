"""Merge the formal matrix into one participant-level report.

Reads every per-fold, per-variant, per-seed streaming record, averages seeds within
each participant, pools the cohort, and computes the primary and co-primary
paired contrasts with their Holm adjustment. It refuses a matrix whose variants did
not score the same participants: an unmatched pair is not a paired contrast.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posesim.analysis.stats import holm_adjust, paired_report
from posesim.analysis.streaming import (ALIGNMENT_CONVENTION, SHANK_IMU_AVAILABILITY_DELAY_S,
                                        PRESSURE_AVAILABILITY_DELAY_S)
from posesim.shank_imu.scaling import segment_fold_cutoffs

REPORT_SCHEMA = "formal-report-v1"
PRIMARY = ("shank_imu_dense", "ponly_dense")
SUMMARY_ARMS = ("shank_imu_moments", "shank_imu_moments_momenthidden128")
PRIMARY_CONTRAST_SEEDS = 3                # seeds behind every variant of a primary contrast


def _fold_dirs(root: Path, variant: str) -> list[Path]:
    return sorted((path for path in root.glob(f"f*/{variant}") if path.is_dir()),
                  key=lambda p: int(p.parent.name[1:]))


def variant_seeds(root: str | Path, variant: str) -> list[int]:
    """Seeds present for one variant, verified identical across its folds."""
    seeds_seen = None
    for fold_dir in _fold_dirs(Path(root), variant):
        seeds = sorted(int(path.stem.split("_s")[1])
                       for path in (fold_dir / "reports").glob("test_s*.json"))
        if seeds_seen is None:
            seeds_seen = seeds
        elif seeds != seeds_seen:
            raise ValueError(
                f"{variant} {fold_dir.parent.name} has seeds {seeds}, expected {seeds_seen}")
    if seeds_seen is None:
        raise ValueError(f"no folds found for variant {variant}")
    return seeds_seen


def participant_scores(root: str | Path, variant: str, seeds=None) -> dict[str, float]:
    """Seed-averaged held-out MPJPE per participant for one variant.

    ``seeds`` fixes which seeds enter the average, so an variant that also serves
    the five-seed E0 reference still contributes the declared contrast tier.
    """
    root = Path(root)
    folds = _fold_dirs(root, variant)
    if not folds:
        raise ValueError(f"no folds found for variant {variant}")
    wanted = None if seeds is None else {int(seed) for seed in seeds}
    scores: dict[str, list[float]] = {}
    for fold_dir in folds:
        for path in sorted((fold_dir / "reports").glob("test_s*.json")):
            if wanted is not None and int(path.stem.split("_s")[1]) not in wanted:
                continue
            report = json.loads(path.read_text(encoding="utf-8"))["report"]
            for subject, entry in report["subjects"].items():
                scores.setdefault(str(subject), []).append(float(entry["mpjpe_mm"]))
    counts = {len(values) for values in scores.values()}
    if len(counts) != 1:
        raise ValueError(f"{variant} scored participants unevenly across seeds: {sorted(counts)}")
    return {subject: float(np.mean(values)) for subject, values in scores.items()}


def e0_reference_report(root: str | Path, variant: str) -> dict:
    """Descriptive seed variation of the reference over every seed it carries."""
    seeds = variant_seeds(root, variant)
    per_seed = {}
    for seed in seeds:
        scores = participant_scores(root, variant, seeds=[seed])
        per_seed[str(seed)] = float(np.mean(list(scores.values())))
    values = np.asarray(list(per_seed.values()), dtype=float)
    return {
        "variant": variant,
        "seeds": seeds,
        "per_seed_subject_averaged_mm": per_seed,
        "mean_mm": float(values.mean()),
        "std_mm": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "range_mm": float(values.max() - values.min()),
    }


def fold_means(root: str | Path, variant: str, seeds=None) -> dict[str, float]:
    """Subject-averaged mean per outer fold: the fold-variation display."""
    wanted = None if seeds is None else {int(seed) for seed in seeds}
    means: dict[str, float] = {}
    for fold_dir in _fold_dirs(Path(root), variant):
        values: dict[str, list[float]] = {}
        for path in sorted((fold_dir / "reports").glob("test_s*.json")):
            if wanted is not None and int(path.stem.split("_s")[1]) not in wanted:
                continue
            report = json.loads(path.read_text(encoding="utf-8"))["report"]
            for subject, entry in report["subjects"].items():
                values.setdefault(str(subject), []).append(float(entry["mpjpe_mm"]))
        if values:
            means[fold_dir.parent.name[1:]] = float(
                np.mean([np.mean(v) for v in values.values()]))
    return means


def _strata(root: Path, variant: str) -> dict:
    merged: dict[str, dict[str, list[float]]] = {}
    for fold_dir in _fold_dirs(root, variant):
        for path in sorted((fold_dir / "reports").glob("test_s*.json")):
            strata = json.loads(path.read_text(encoding="utf-8"))["report"].get("strata", {})
            for group, labels in strata.items():
                for label, entry in labels.items():
                    merged.setdefault(group, {}).setdefault(
                        label, []).append(entry["subject_averaged_mpjpe_mm"])
    return {group: {label: float(np.mean(values)) for label, values in sorted(labels.items())}
            for group, labels in merged.items()}


def _selections(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("f*/*/selection.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"fold": int(path.parent.parent.name[1:]), "variant": path.parent.name,
                     "config": record["config"],
                     "budget_steps": record["median_budget_steps"],
                     "inner_val_mm": record["mean_val_subject_averaged_mm"]})
    return rows


def beta_nll_share(root: Path, variant: str) -> dict[str, float]:
    """Fold to the fraction of the retrain budget that ran under the likelihood.

    Warm-up counts epochs while the budget counts steps, and a retrain's epoch
    is longer than the inner-fold epoch it was selected under, so a record whose
    configuration names `beta_nll` may have spent all of its budget on MSE.
    """
    from scripts.train_moveport import WARMUP_EPOCHS

    shares: dict[str, float] = {}
    for fold_dir in _fold_dirs(root, variant):
        values = []
        for path in sorted((fold_dir / "retrain").glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            steps, per_epoch = record.get("steps"), record.get("steps_per_epoch")
            if not steps or not per_epoch:
                continue
            if record.get("config", {}).get("loss", "beta_nll") == "mse":
                values.append(0.0)
                continue
            values.append(max(0.0, 1.0 - WARMUP_EPOCHS * per_epoch / steps))
        if values:
            shares[fold_dir.parent.name[1:]] = float(np.mean(values))
    return shares


def diverged_retrains(root: Path) -> list[str]:
    """Retrained models whose training loss went non-finite.

    `select_inner.py` drops a diverged configuration, but nothing reads the same
    flag on the retrained model that a held-out score is computed from.
    """
    return sorted(
        str(path.relative_to(root))
        for path in root.glob("f*/*/retrain/*.json")
        if json.loads(path.read_text(encoding="utf-8")).get("diverged")
    )


def _insole_groups(insole_areas: dict, participants: list[str]) -> dict[str, list[str]]:
    """Participants grouped by their source insole cell area, labelled in mm^2.

    The two MovePort insole sizes bias resampling in opposite directions, so a
    report without this stratum is incomplete rather than merely shorter: an
    absent or partial lookup table is an error, not an empty group.
    """
    missing = [s for s in participants if insole_areas.get(str(s)) is None]
    if missing:
        raise ValueError(
            "the insole-size stratum needs a source cell area for every scored "
            f"participant; missing {missing}")
    groups: dict[str, list[str]] = {}
    for subject in participants:
        area = insole_areas[str(subject)]
        groups.setdefault(f"{float(area) * 1e6:.2f}", []).append(str(subject))
    return groups


def quality_filtered_scores(root: str | Path, variant: str, segment_quality: dict,
                            cutoffs: dict, seeds=None) -> tuple[dict[str, float], set[str]]:
    """Participant scores over segments whose fit residual passes their fold's cutoff.

    Reported beside the all-segment headline, never in place of it: a subset
    chosen by input quality is an extra column, never the headline.
    """
    wanted = None if seeds is None else {int(seed) for seed in seeds}
    scores: dict[str, list[float]] = {}
    excluded: set[str] = set()
    for fold_dir in _fold_dirs(Path(root), variant):
        cutoff = cutoffs.get(fold_dir.parent.name[1:])
        if cutoff is None:
            continue
        for path in sorted((fold_dir / "reports").glob("test_s*.json")):
            if wanted is not None and int(path.stem.split("_s")[1]) not in wanted:
                continue
            report = json.loads(path.read_text(encoding="utf-8"))["report"]
            for segment_id, entry in report.get("per_segment", {}).items():
                if segment_id not in segment_quality:
                    # A key that never matches keeps every segment, so the column
                    # would read as "nothing failed" instead of "nothing joined".
                    raise ValueError(f"no fit residual for scored segment {segment_id}")
                residual = segment_quality[segment_id]
                if residual > cutoff:
                    excluded.add(segment_id)
                    continue
                if entry["mpjpe_mm"] is None:
                    continue
                scores.setdefault(str(entry["subject"]), []).append(
                    (float(entry["mpjpe_mm"]), int(entry["frames_scored"])))
    means = {}
    for subject, rows in scores.items():
        values = np.asarray([value for value, _ in rows], dtype=float)
        weights = np.asarray([frames for _, frames in rows], dtype=float)
        means[subject] = float(np.average(values, weights=weights))
    return means, excluded


def build_formal_report(root: str | Path, *, n_boot: int = 15000, seed: int = 0,
                        baselines: dict | None = None,
                        insole_areas: dict | None = None,
                        segment_quality: dict | None = None,
                        quality_cutoffs: dict | None = None) -> dict:
    """Assemble the participant-level report for the whole matrix."""
    root = Path(root)
    variants = sorted({path.name for path in root.glob("f*/*") if path.is_dir()})
    if not variants:
        raise ValueError("no variants found under the matrix root")
    diverged = diverged_retrains(root)
    if diverged:
        raise ValueError(f"diverged retrained models scored into the matrix: {diverged}")
    contrast_seeds = list(range(PRIMARY_CONTRAST_SEEDS))
    scores = {variant: participant_scores(root, variant, seeds=contrast_seeds) for variant in variants}
    cohorts = {variant: set(values) for variant, values in scores.items()}
    reference = cohorts[variants[0]]
    for variant, cohort in cohorts.items():
        if cohort != reference:
            raise ValueError(
                f"variant {variant} scored a different participant set than {variants[0]}: "
                f"missing {sorted(reference - cohort)}, extra {sorted(cohort - reference)}")
    participants = sorted(reference, key=int)
    insole_groups = _insole_groups(insole_areas or {}, participants)
    folds = {variant: fold_means(root, variant, seeds=contrast_seeds) for variant in variants}
    quality_scores, quality_excluded = {}, set()
    if segment_quality and quality_cutoffs:
        for variant in variants:
            quality_scores[variant], dropped = quality_filtered_scores(
                root, variant, segment_quality, quality_cutoffs, seeds=contrast_seeds)
            quality_excluded |= dropped

    def contrast(better: str, worse: str) -> dict:
        left = np.asarray([scores[better][s] for s in participants])
        right = np.asarray([scores[worse][s] for s in participants])
        return {"contrast": f"{better} - {worse}",
                **paired_report(left, right, n_boot=n_boot, seed=seed)}

    fusion, pressure = PRIMARY
    if fusion not in scores or pressure not in scores:
        raise ValueError(f"the primary contrast needs both {PRIMARY}")
    primary = contrast(fusion, pressure)

    available = [variant for variant in SUMMARY_ARMS if variant in scores]
    if not available:
        raise ValueError("the co-primary contrast needs a fused summary variant")
    # The summary condition is scored as the better of its encoder variants.
    summary_variant = min(available, key=lambda variant: np.mean(list(scores[variant].values())))
    co_primary = {**contrast(fusion, summary_variant), "summary_variant": summary_variant}
    co_primary["contrast"] = f"{fusion} - shank_imu_summary"

    adjusted = holm_adjust([primary["wilcoxon_p"], co_primary["wilcoxon_p"]])
    for variant in {fusion, pressure, summary_variant}:
        available = variant_seeds(root, variant)
        missing = set(contrast_seeds) - set(available)
        if missing:
            raise ValueError(
                f"variant {variant} is missing seeds {sorted(missing)}; a primary or co-primary "
                f"contrast needs {PRIMARY_CONTRAST_SEEDS}")
    return {
        "schema": REPORT_SCHEMA,
        "alignment": ALIGNMENT_CONVENTION,
        "availability_delay_s": {"ponly": PRESSURE_AVAILABILITY_DELAY_S,
                                 "shank_imu": SHANK_IMU_AVAILABILITY_DELAY_S},
        "participants": len(participants),
        "seeds_per_variant": {"contrast": PRIMARY_CONTRAST_SEEDS,
                          "reference_available": len(variant_seeds(root, fusion))},
        "e0_reference": e0_reference_report(root, fusion),
        "insole_size_participants": {label: len(members)
                                     for label, members in insole_groups.items()},
        "quality_filter": {
            "headline": "all_segments",
            "cutoffs_m": quality_cutoffs or {},
            "segments_excluded": len(quality_excluded),
            "excluded_ids": sorted(quality_excluded),
        },
        "variants": {
            variant: {
                "subject_averaged_mpjpe_mm": float(np.mean(list(scores[variant].values()))),
                "per_participant_mm": {s: scores[variant][s] for s in participants},
                "beta_nll_share_of_budget": beta_nll_share(root, variant),
                "per_fold_mm": folds[variant],
                "fold_range_mm": float(max(folds[variant].values())
                                       - min(folds[variant].values())),
                "strata": _strata(root, variant),
                "insole_size_mm": {
                    label: float(np.mean([scores[variant][s] for s in members]))
                    for label, members in insole_groups.items()
                },
                **({"quality_filtered_mpjpe_mm":
                    float(np.mean(list(quality_scores[variant].values())))}
                   if quality_scores.get(variant) else {}),
            }
            for variant in variants
        },
        "primary": primary,
        "co_primary": co_primary,
        "holm_adjusted_p": {"primary": adjusted[0], "co_primary": adjusted[1]},
        "fold_configurations": _selections(root),
        "baselines": baselines or {},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, default=None)
    parser.add_argument("--insole-areas", type=Path, required=True,
                        help="per-subject source cell area for the insole-size stratum")
    parser.add_argument("--segment-quality", type=Path, required=True,
                        help="segment-fit-quality-v1 record from segment_fit_quality.py")
    parser.add_argument("--cache", type=Path, required=True,
                        help="aligned cache supplying the outer-fold membership")
    parser.add_argument("--qc-quantile", type=float, default=0.95)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    baselines = (json.loads(args.baselines.read_text(encoding="utf-8"))
                 if args.baselines else None)
    insole = json.loads(args.insole_areas.read_text(encoding="utf-8"))
    quality = json.loads(args.segment_quality.read_text(encoding="utf-8"))["per_segment"]
    residuals = {key: row["mean_rms_m"] for key, row in quality.items()}
    with np.load(args.cache, allow_pickle=True) as cache:
        fold_subjects = [[s for s in row if s] for row in np.asarray(cache["fold_subjects"])]
    cutoffs = segment_fold_cutoffs(residuals, fold_subjects, quantile=args.qc_quantile)
    report = build_formal_report(args.matrix, n_boot=args.n_boot, seed=args.seed,
                                 baselines=baselines, insole_areas=insole,
                                 segment_quality=residuals,
                                 quality_cutoffs={fold: row["cutoff_m"]
                                                  for fold, row in cutoffs.items()})
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "participants": report["participants"],
        "primary": {k: report["primary"][k] for k in ("contrast", "mean_diff", "wilcoxon_p")},
        "co_primary": {k: report["co_primary"][k] for k in ("contrast", "mean_diff", "wilcoxon_p")},
        "holm": report["holm_adjusted_p"],
    }, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
