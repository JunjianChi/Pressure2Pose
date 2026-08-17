"""Score one trained checkpoint with the segment-streaming evaluator.

Runs every selected segment once from its true start (no window copies, no
zero-padded restarts) and scores each fully valid target frame exactly once,
against either the aligned-cache labels or unfiltered native labels at the
same physical times.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posesim.analysis.streaming import evaluate_streaming, native_targets
from posesim.model.encoder import PosePressureNet
from scripts.train_moveport import (attach_shank_imu, shank_imu_statistics, fold_masks, load,
                                    target_statistics)


def statistics_mask(run: dict, train_mask, val_mask):
    """The exact window mask the run fitted its statistics on: step-budget
    retraining merges the development validation segments into training."""
    if "steps" in run:
        return train_mask | val_mask
    return train_mask


def fold_segments(cache, subjects) -> tuple[list[int], list[str]]:
    """Every segment of the named participants, from the cache's own subject table.

    Scoring must never be selected by window acceptance: a modality that
    invalidates one segment would otherwise drop it from that variant alone and
    leave the paired contrast comparing different frame sets.
    """
    wanted = {str(subject) for subject in subjects}
    present = {str(subject) for subject in cache["segment_subject"]}
    unknown = wanted - present
    if unknown:
        raise ValueError(f"no segments for participants {sorted(unknown)}")
    indices = [index for index, subject in enumerate(cache["segment_subject"])
               if str(subject) in wanted]
    return indices, sorted(wanted, key=int)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("data/processed/moveport_all.npz"))
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--labels", choices=("cache", "native"), default="cache")
    parser.add_argument("--root", type=Path, default=Path("data/raw/moveport"))
    parser.add_argument("--shank-imu-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    run = json.loads(args.run_json.read_text(encoding="utf-8"))

    shank_imu = attach_shank_imu(args.cache, args.shank_imu_dir) if args.shank_imu_dir else None
    config = run.get("config", {})
    uses_foot_imu = shank_imu is None and not config.get("no_imu", not run.get("imu", True))
    arrays, index, folds = load(args.cache, shank_imu=shank_imu, uses_foot_imu=uses_foot_imu)
    train_mask, val_mask, test_mask = fold_masks(index, folds, args.fold)
    fit_mask = statistics_mask(run, train_mask, val_mask)
    stats = target_statistics(arrays, index, fit_mask)
    shank_imu_stats = shank_imu_statistics(arrays, index, fit_mask) if shank_imu is not None else None
    del arrays

    with np.load(args.cache, allow_pickle=False) as archive:
        cache = {key: archive[key].copy() for key in archive.files}
    held_out = [str(s) for s in np.asarray(cache["fold_subjects"])[args.fold] if str(s)]
    if args.split == "test":
        segments, subjects = fold_segments(cache, held_out)
    else:
        segments = [k for k, row in enumerate(index)
                    if int(row[4]) > int(row[3])
                    and val_mask[int(row[3]):int(row[4])].all()]
        subjects = sorted({str(cache["segment_subject"][k]) for k in segments}, key=int)

    config = run.get("config", {})
    if args.shank_imu_dir is not None:
        imu_dim = 12
    else:
        imu_dim = (12 + 2) if run.get("imu", True) else 0
    from posesim.model.encoder import MOMENT_HIDDEN
    model = PosePressureNet(
        encoder=run["encoder"], head=run.get("head", "free"),
        imu_dim=imu_dim, n_joints=10,
        dilations=tuple(config.get("dilations", (1, 2, 4, 8))),
        moment_hidden=config.get("moment_hidden") or MOMENT_HIDDEN,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    overrides = {}
    if args.labels == "native":
        targets = np.full_like(np.asarray(cache["target_m"], dtype=np.float64), np.nan)
        valid = np.zeros_like(np.asarray(cache["target_valid"], dtype=bool))
        for segment_index in segments:
            start = int(cache["segment_start"][segment_index])
            stop = int(cache["segment_stop"][segment_index])
            values, value_valid = native_targets(
                args.root,
                str(cache["segment_subject"][segment_index]),
                str(cache["segment_activity"][segment_index]),
                str(cache["segment_name"][segment_index]),
                cache["time_s"][start:stop],
            )
            targets[start:stop] = values
            valid[start:stop] = value_valid
        overrides = {"targets": targets, "targets_valid": valid}

    report = evaluate_streaming(
        model, cache, subjects, stats,
        label_source=args.labels, segment_indices=segments,
        shank_imu=shank_imu, shank_imu_stats=shank_imu_stats, block=int(config.get("block", 1)),
        block_origin=int(config.get("block_origin", 0)), **overrides,
    )
    record = {
        "cache": str(args.cache), "checkpoint": str(args.checkpoint),
        "run_tag": run.get("tag"), "fold": args.fold, "split": args.split,
        "report": report,
    }
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "labels": args.labels, "split": args.split, "segments": report["segments"],
        "frames_scored": report["frames_scored"],
        "subject_averaged_mpjpe_mm": report["subject_averaged_mpjpe_mm"],
        "pooled_frame_mpjpe_mm": report["pooled_frame_mpjpe_mm"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
