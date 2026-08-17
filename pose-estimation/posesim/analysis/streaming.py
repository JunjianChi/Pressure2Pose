"""Segment-streaming evaluation: each segment runs once from its true start and
every fully valid target frame is scored exactly once, in metres."""
from __future__ import annotations

import numpy as np
import torch

from posesim.data import insole as ours
from posesim.data.mpdataset import TARGET_MARKERS as MARKER_NAMES
from posesim.model.inputs import normalise, to_grid
from posesim.model.kinematic_head import KinematicHead

REPORT_SCHEMA = "streaming-eval-v1"
ALIGNMENT_CONVENTION = "pelvis_relative_yaw_canonical;no_procrustes;no_scale"
PRESSURE_AVAILABILITY_DELAY_S = 0.10
SHANK_IMU_AVAILABILITY_DELAY_S = 0.35


def _segment_bounds(cache: dict) -> list[tuple[str, int, int]]:
    return [
        (str(subject), int(start), int(stop))
        for subject, start, stop in zip(
            cache["segment_subject"], cache["segment_start"], cache["segment_stop"]
        )
    ]


def segment_predictions(
    model, cache: dict, segment_index: int, stats, *, device: str = "cpu",
    shank_imu=None, shank_imu_stats=None, block: int = 1, block_origin: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one whole segment from its true start; return metre-valued predictions.

    Frames with any invalid input value are fed as zeros and flagged so the
    caller can report them; the causal model still emits a prediction for every
    frame. With ``shank_imu`` (row-aligned values and validity) and ``shank_imu_stats``,
    the standardised virtual-IMU channels replace the released foot-IMU features.
    """
    if (shank_imu is None) != (shank_imu_stats is None):
        raise ValueError("shank_imu features and their statistics travel together")
    mean, std = stats
    start = int(cache["segment_start"][segment_index])
    stop = int(cache["segment_stop"][segment_index])
    selection = slice(start, stop)
    pressure = np.nan_to_num(np.asarray(cache["pressure_pa"][selection], dtype=np.float64))
    if block > 1:
        from posesim.data.resolution import block_average
        pressure = block_average(pressure, block, origin=block_origin)
    force = np.nan_to_num(np.asarray(cache["force_n"][selection], dtype=np.float64))
    input_invalid = ~(
        np.asarray(cache["pressure_valid"][selection]).all(axis=(1, 2))
        & np.asarray(cache["force_valid"][selection]).all(axis=1)
        & np.asarray(cache["foot_imu_valid"][selection]).all(axis=(1, 2))
    )
    shape, magnitude = normalise(pressure[None], force[None])
    if shank_imu is not None:
        shank_imu_values, shank_imu_row_valid = shank_imu
        shank_imu_mean, shank_imu_std = shank_imu_stats
        rows = np.nan_to_num(np.asarray(shank_imu_values[selection], dtype=np.float64))
        standardised = (rows - shank_imu_mean) / shank_imu_std
        # An unpairable row is fed the channel mean. Zero raw values would read as a
        # gravity-free imu, several standard deviations from anything physical.
        unusable = ~np.asarray(shank_imu_row_valid[selection], dtype=bool)
        standardised[unusable] = 0.0
        features = standardised.reshape(1, stop - start, -1)
        input_invalid |= unusable
    else:
        imu = np.nan_to_num(np.asarray(cache["foot_imu_si"][selection], dtype=np.float64))
        features = np.concatenate([imu.reshape(1, len(imu), -1), magnitude], -1)
    if int(getattr(model, "imu_dim", 0)) not in (0, features.shape[-1]):
        raise ValueError("model imu_dim does not match the supplied feature layout")
    grid = to_grid(
        torch.as_tensor(shape, dtype=torch.float32, device=device), ours.active_mask()
    )
    inertial = (
        torch.as_tensor(features, dtype=torch.float32, device=device)
        if getattr(model, "imu_dim", 0)
        else None
    )
    model.eval()
    with torch.no_grad():
        mu, _ = model(grid, inertial)
    prediction = mu[0].cpu().numpy()
    metric_head = isinstance(getattr(getattr(model, "tcn", None), "head", None), KinematicHead)
    if not metric_head:
        prediction = prediction * np.asarray(std) + np.asarray(mean)
    return prediction, input_invalid


def evaluate_streaming(
    model,
    cache: dict,
    subjects,
    stats,
    *,
    device: str = "cpu",
    targets: np.ndarray | None = None,
    targets_valid: np.ndarray | None = None,
    label_source: str = "cache",
    segment_indices=None,
    shank_imu=None,
    shank_imu_stats=None,
    block: int = 1,
    block_origin: int = 0,
) -> dict:
    """Score whole segments of the named subjects once each, subject-averaged."""
    if (targets is None) != (targets_valid is None):
        raise ValueError("targets and targets_valid override the labels together")
    if targets is None:
        targets = np.asarray(cache["target_m"])
        targets_valid = np.asarray(cache["target_valid"])
    if segment_indices is not None:
        segment_indices = {int(index) for index in segment_indices}
    subjects = tuple(str(subject) for subject in subjects)
    per_subject: dict[str, dict[str, float]] = {}
    totals = {"segments": 0, "frames": 0, "frames_scored": 0, "invalid_input_frames": 0}
    errors_by_subject: dict[str, list[np.ndarray]] = {subject: [] for subject in subjects}
    strata: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
        name: {} for name in ("activity", "marker", "phase", "locomotion")
    }
    segment_scores: dict[str, dict] = {}

    def _collect(group: str, label: str, subject: str, values: np.ndarray) -> None:
        if not len(values):
            return
        strata[group].setdefault(label, {}).setdefault(subject, []).append(values)
    for segment_index, (subject, start, stop) in enumerate(_segment_bounds(cache)):
        if subject not in subjects:
            continue
        if segment_indices is not None and segment_index not in segment_indices:
            continue
        prediction, input_invalid = segment_predictions(
            model, cache, segment_index, stats, device=device,
            shank_imu=shank_imu, shank_imu_stats=shank_imu_stats, block=block,
            block_origin=block_origin,
        )
        frame_valid = np.asarray(targets_valid[start:stop]).all(axis=(1, 2))
        label = np.asarray(targets[start:stop], dtype=np.float64)
        per_marker = np.linalg.norm(prediction - label, axis=-1)          # (frames, markers)
        error_m = per_marker.mean(axis=-1)
        errors_by_subject[subject].append(error_m[frame_valid])

        scored = error_m[frame_valid]
        segment_scores[
            f"{subject}/{cache['segment_activity'][segment_index]}/"
            f"{cache['segment_name'][segment_index]}"
        ] = {"subject": subject,
             "frames_scored": int(len(scored)),
             # None, not NaN: a bare NaN is not valid JSON for strict parsers.
             "mpjpe_mm": float(scored.mean() * 1000.0) if len(scored) else None}

        activity = str(cache["segment_activity"][segment_index])
        _collect("activity", activity, subject, error_m[frame_valid])
        _collect("locomotion",
                 "gait" if activity.startswith("treadmill") else "posture_like",
                 subject, error_m[frame_valid])
        contact = np.asarray(cache["contact"][start:stop], dtype=np.float64)
        contact_valid = np.asarray(cache["contact_valid"][start:stop], dtype=bool)
        for marker_index, marker in enumerate(MARKER_NAMES):
            column = per_marker[frame_valid, marker_index]
            _collect("marker", marker, subject, column)
            side = marker_index % 2                    # even left, odd right
            usable = contact_valid[frame_valid, side]
            loaded = contact[frame_valid, side] > 0.5
            _collect("phase", "stance", subject, column[usable & loaded])
            _collect("phase", "swing", subject, column[usable & ~loaded])
        totals["segments"] += 1
        totals["frames"] += stop - start
        totals["frames_scored"] += int(frame_valid.sum())
        totals["invalid_input_frames"] += int(input_invalid.sum())
    pooled: list[np.ndarray] = []
    for subject in subjects:
        errors = np.concatenate(errors_by_subject[subject]) if errors_by_subject[subject] else (
            np.empty(0))
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
        "alignment": ALIGNMENT_CONVENTION,
        "availability_delay_s": (SHANK_IMU_AVAILABILITY_DELAY_S if shank_imu is not None
                                 else PRESSURE_AVAILABILITY_DELAY_S),
        "block": int(block),
        "per_segment": segment_scores,
        "strata": {
            group: {
                label: {
                    "frames_scored": int(sum(len(np.concatenate(v)) for v in by_subject.values())),
                    "subject_averaged_mpjpe_mm": float(np.mean(
                        [np.concatenate(v).mean() * 1000.0 for v in by_subject.values()])),
                }
                for label, by_subject in sorted(labels.items())
            }
            for group, labels in strata.items()
        },
        **totals,
        "subjects": per_subject,
        "subject_averaged_mpjpe_mm": float(
            np.mean([entry["mpjpe_mm"] for entry in per_subject.values()])
        ),
        "pooled_frame_mpjpe_mm": float(all_errors.mean() * 1000.0),
    }


def interpolate_targets(
    values: np.ndarray,
    valid: np.ndarray,
    source_time_s: np.ndarray,
    target_time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate labels onto new times; a target is valid only when
    both bracketing source frames are fully valid, and exact hits need one."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    source = np.asarray(source_time_s, dtype=np.float64)
    target = np.asarray(target_time_s, dtype=np.float64)
    if values.shape[:1] != source.shape or valid.shape != values.shape:
        raise ValueError("values, valid, and source_time_s must share the frame axis")
    if np.any(np.diff(source) <= 0.0):
        raise ValueError("source_time_s must be strictly increasing")
    count = len(source)
    later = np.searchsorted(source, target, side="left")
    later_index = np.clip(later, 0, count - 1)
    inside = (later < count) & (target >= source[0])
    exact = inside & (source[later_index] == target)
    earlier_index = np.where(exact, later_index, np.clip(later_index - 1, 0, count - 1))
    span = source[later_index] - source[earlier_index]
    weight = np.divide(
        target - source[earlier_index], span, out=np.zeros_like(target), where=span > 0.0
    ).reshape((-1,) + (1,) * (values.ndim - 1))
    frame_valid = valid.all(axis=tuple(range(1, valid.ndim)))
    out = (1.0 - weight) * values[earlier_index] + weight * values[later_index]
    out_valid = inside & frame_valid[earlier_index] & frame_valid[later_index]
    out_valid = np.broadcast_to(
        out_valid.reshape((-1,) + (1,) * (values.ndim - 1)), out.shape
    ).copy()
    out[~out_valid] = np.nan
    return out, out_valid


def native_targets(
    root, subject: str, activity: str, name: str, time_s: np.ndarray, *,
    group_delay_s: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Unfiltered marker targets at the aligned cache's physical times.

    The aligned clock stamps availability, so the label for cache row ``t`` is
    the native target at elapsed physical time ``t - group_delay_s``.
    """
    from posesim.data.moveport import load_native_segment
    from posesim.data.mpdataset import marker_target

    native = load_native_segment(root, subject, activity, name)
    target = marker_target(native.markers)
    elapsed = target.time_s - target.time_s[0]
    physical = np.asarray(time_s, dtype=np.float64) - group_delay_s
    return interpolate_targets(target.values, target.valid, elapsed, physical)
