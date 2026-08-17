"""The two required table baselines: the mean-pose mean-pose baseline and
nearest-pressure retrieval in the model's own normalised-shape input space."""
from __future__ import annotations

import numpy as np
import torch

from posesim.analysis.streaming import REPORT_SCHEMA, _segment_bounds
from posesim.model.inputs import normalise


def mean_pose_baseline(cache: dict, train_mask: np.ndarray) -> np.ndarray:
    """Mean target over fully valid training frames, as one (10, 3) pose."""
    train_mask = np.asarray(train_mask, dtype=bool)
    targets = np.asarray(cache["target_m"], dtype=np.float64)
    valid = np.all(np.asarray(cache["target_valid"], dtype=bool), axis=(1, 2))
    selection = train_mask & valid
    if not selection.any():
        raise ValueError("the training mask selects no fully valid target frame")
    return targets[selection].mean(axis=0)


def retrieval_predictions(
    train_shape: np.ndarray,
    train_targets: np.ndarray,
    query_shape: np.ndarray,
    *,
    chunk: int = 2048,
    device: str = "cpu",
) -> np.ndarray:
    """Nearest-training-pressure target for every query, by L2 in shape space."""
    pool = torch.as_tensor(
        np.asarray(train_shape, dtype=np.float32).reshape(len(train_shape), -1),
        device=device,
    )
    targets = np.asarray(train_targets, dtype=np.float64)
    if len(pool) != len(targets) or not len(pool):
        raise ValueError("the retrieval pool needs matched non-empty shapes and targets")
    queries = np.asarray(query_shape, dtype=np.float32).reshape(len(query_shape), -1)
    out = np.empty((len(queries),) + targets.shape[1:], dtype=np.float64)
    with torch.no_grad():
        for begin in range(0, len(queries), chunk):
            block = torch.as_tensor(queries[begin:begin + chunk], device=device)
            nearest = torch.cdist(block, pool).argmin(dim=1).cpu().numpy()
            out[begin:begin + chunk] = targets[nearest]
    return out


def fold_baselines(cache: dict, fold: int, *, retrieval_stride: int = 20,
                   device: str = "cpu") -> dict:
    """Score both required references on one fold's held-out participants.

    The retrieval pool is the fold's development frames, subsampled by
    ``retrieval_stride``; the stride is reported so the baseline is never
    quoted as an exhaustive nearest neighbour.
    """
    held_out = [str(s) for s in np.asarray(cache["fold_subjects"])[fold] if str(s)]
    frame_subject = np.concatenate([
        [str(subject)] * (int(stop) - int(start))
        for subject, start, stop in zip(cache["segment_subject"], cache["segment_start"],
                                        cache["segment_stop"])
    ])
    development = ~np.isin(frame_subject, held_out)
    target_valid = np.all(np.asarray(cache["target_valid"], dtype=bool), axis=(1, 2))
    pressure_valid = np.all(np.asarray(cache["pressure_valid"], dtype=bool), axis=(1, 2))

    pose = mean_pose_baseline(cache, development & target_valid)
    mean_predictions = np.broadcast_to(pose, cache["target_m"].shape).copy()

    pool_rows = np.flatnonzero(development & target_valid & pressure_valid)[::retrieval_stride]
    if not len(pool_rows):
        raise ValueError("the retrieval pool is empty for this fold")
    pool_shape, _ = normalise(
        np.clip(np.nan_to_num(cache["pressure_pa"][pool_rows]).astype(np.float64), 0.0, None),
        np.zeros((len(pool_rows), 2)))
    query_rows = np.flatnonzero(np.isin(frame_subject, held_out))
    query_shape, _ = normalise(
        np.clip(np.nan_to_num(cache["pressure_pa"][query_rows]).astype(np.float64), 0.0, None),
        np.zeros((len(query_rows), 2)))
    retrieval_predictions_array = np.zeros_like(cache["target_m"], dtype=np.float64)
    retrieval_predictions_array[query_rows] = retrieval_predictions(
        pool_shape, cache["target_m"][pool_rows].astype(np.float64), query_shape,
        device=device)

    return {
        "mean_pose": evaluate_predictions(cache, held_out, mean_predictions),
        "retrieval": {**evaluate_predictions(cache, held_out, retrieval_predictions_array),
                      "pool_frames": int(len(pool_rows)),
                      "pool_stride": int(retrieval_stride)},
    }


def evaluate_predictions(
    cache: dict,
    subjects,
    predictions: np.ndarray,
    *,
    label_source: str = "cache",
    segment_indices=None,
) -> dict:
    """Score one cache-aligned prediction array with the streaming conventions."""
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(cache["target_m"], dtype=np.float64)
    if predictions.shape != targets.shape:
        raise ValueError("predictions must align with the cache target array")
    targets_valid = np.asarray(cache["target_valid"], dtype=bool)
    if segment_indices is not None:
        segment_indices = {int(index) for index in segment_indices}
    subjects = tuple(str(subject) for subject in subjects)
    errors_by_subject: dict[str, list[np.ndarray]] = {subject: [] for subject in subjects}
    totals = {"segments": 0, "frames": 0, "frames_scored": 0, "invalid_input_frames": 0}
    for segment_index, (subject, start, stop) in enumerate(_segment_bounds(cache)):
        if subject not in subjects:
            continue
        if segment_indices is not None and segment_index not in segment_indices:
            continue
        frame_valid = targets_valid[start:stop].all(axis=(1, 2))
        error_m = np.linalg.norm(
            predictions[start:stop] - targets[start:stop], axis=-1).mean(axis=-1)
        errors_by_subject[subject].append(error_m[frame_valid])
        totals["segments"] += 1
        totals["frames"] += stop - start
        totals["frames_scored"] += int(frame_valid.sum())
    per_subject = {}
    pooled = []
    for subject in subjects:
        errors = (np.concatenate(errors_by_subject[subject])
                  if errors_by_subject[subject] else np.empty(0))
        if not len(errors):
            raise ValueError(f"subject {subject} has no scored frames")
        per_subject[subject] = {
            "frames_scored": int(len(errors)),
            "mpjpe_mm": float(errors.mean() * 1000.0),
        }
        pooled.append(errors)
    all_errors = np.concatenate(pooled)
    return {
        "schema": REPORT_SCHEMA,
        "label_source": str(label_source),
        **totals,
        "subjects": per_subject,
        "subject_averaged_mpjpe_mm": float(
            np.mean([entry["mpjpe_mm"] for entry in per_subject.values()])
        ),
        "pooled_frame_mpjpe_mm": float(all_errors.mean() * 1000.0),
    }
