"""Aligned MovePort segments and cross-subject folds.

The cache owns every project-clock frame once. Overlapping training windows are
non-owning references materialised only by a runner.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from posesim.data import insole as ours
from posesim.data.moveport import (ACTIVITIES, MARKERS, PSI_TO_PA, MovePortNativeSegment,
                                   active_mask, load_native_segment, load_sequence, sequence_names,
                                   subjects, to_canonical)
from posesim.data.register import resample_matrix
from posesim.data.timed import TimedArray
from posesim.signal.resample import resample_causal
from posesim.smpl_math import to_root_relative_yaw_canonical

WINDOW = 64                      # 1.07 s at 60 Hz; a shorter segment is dropped by `save`
TARGET_FPS = 60.0                # cache timeline
ALIGNMENT_STATUS = "project_elapsed_clock;provider_sync_unverified"
ALIGNED_SCHEMA = "moveport-aligned-v2"
RESAMPLE_CUTOFF_HZ = 24.0
RESAMPLE_FILTER = "rate_calibrated_causal_gaussian_fir;half_amplitude_hz=24;history_s=0.2"
CLINICAL_SUBJECT = "25"          # spastic paraplegia, no treadmill walking; evaluated separately

# The ten landmarks the SMPL fit pins, per side. Excluded on measured residual: FTC 113 mm,
# TTC 55 mm, LA/RA 125 mm. Trunk and variants are not observable from a shoe.
TARGET_MARKERS = tuple(f"{s}_{m}" for m in ("FLE", "FME", "LM", "CAL", "MH1") for s in ("L", "R"))
PELVIS_MARKERS = ("L_IAS", "R_IAS", "M_PSIS")

# The cohort wears two insole sizes and nothing between them; 22 of 25 subjects recover to
# one of these to four significant figures. The three that do not are the 100 Hz subjects,
# whose recovery is contaminated by the rate mismatch.
INSOLE_CELL_AREAS_M2 = (63.72e-6, 54.34e-6)

# Offline marker-height contact heuristic; it assumes z is vertical.
CONTACT_HEIGHT_M = 0.020
FLOOR_PERCENTILE = 1.0
MIN_CONTACT_S = 0.1              # UnderPressure discards contact phases shorter than this


def pelvis_frame(markers):
    """Project origin and IAS points for the target-frame convention."""
    i = {n: k for k, n in enumerate(MARKERS)}
    l, r, p = (markers[:, i[n]] for n in PELVIS_MARKERS)
    return np.stack([(l + r + p) / 3.0, l, r], axis=1)


def marker_target(markers):
    """Ten targets in exact project order with pelvis-frame validity propagation."""
    timed = isinstance(markers, TimedArray)
    values = markers.values if timed else np.asarray(markers)
    i = {n: k for k, n in enumerate(MARKERS)}
    stacked = np.concatenate([pelvis_frame(values),
                              values[:, [i[n] for n in TARGET_MARKERS]]], axis=1)
    target = to_root_relative_yaw_canonical(stacked, 0, 1, 2)[:, 3:, :]
    if not timed:
        return target
    pelvis_valid = np.all(markers.valid[:, [i[n] for n in PELVIS_MARKERS], :], axis=(1, 2))
    hip_axis = values[:, i["R_IAS"], :2] - values[:, i["L_IAS"], :2]
    pelvis_valid &= np.linalg.norm(hip_axis, axis=1) > 1e-8
    target_marker_valid = np.all(markers.valid[:, [i[n] for n in TARGET_MARKERS], :], axis=2)
    valid_marker = pelvis_valid[:, None] & target_marker_valid
    valid = np.repeat(valid_marker[..., None], 3, axis=2)
    target[~valid] = np.nan
    return TimedArray(target, markers.time_s, valid, "m", markers.time_basis,
                      markers.nominal_hz, markers.group_delay_s)


def contact_labels(markers, fps, height=CONTACT_HEIGHT_M, min_s=MIN_CONTACT_S):
    """Offline per-foot marker-height contact heuristic, as (F, 2) float."""
    i = {n: k for k, n in enumerate(MARKERS)}
    out = np.zeros((len(markers), 2), dtype=np.float32)
    for k, side in enumerate(("L", "R")):
        h = np.minimum(markers[:, i[f"{side}_CAL"], 2], markers[:, i[f"{side}_MH1"], 2])
        out[:, k] = (h - np.percentile(h, FLOOR_PERCENTILE)) < height
        _drop_short(out[:, k], int(round(min_s * fps)))
    return out


def _drop_short(labels, min_len):
    """Erase runs of ones shorter than ``min_len``, in place."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0.0], labels, [0.0]))))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a < min_len:
            labels[a:b] = 0.0


_CELL_AREA_CACHE = {}


def subject_cell_area(root, subject):
    """Subject cell-area estimate snapped to one of the two supported sizes."""
    key = (str(Path(root).resolve()), str(subject))
    if key in _CELL_AREA_CACHE:
        return _CELL_AREA_CACHE[key]
    found = []
    for activity in ACTIVITIES:
        for name in sequence_names(root, subject, activity):
            try:
                area = load_sequence(root, subject, activity, name).get("cell_area")
            except Exception:
                continue
            if area:
                found.append(area)
    med = float(np.median(found)) if found else None
    _CELL_AREA_CACHE[key] = min(INSOLE_CELL_AREAS_M2, key=lambda a: abs(a - med)) if found else None
    return _CELL_AREA_CACHE[key]


def registration(root, cell_area, mask=None):
    """Per-foot (canonical source mask, weights) onto our 253-cell array."""
    mask = active_mask(root) if mask is None else mask
    dst, ratio = ours.active_mask(), cell_area / ours.CELL_AREA_M2
    return [(to_canonical(mask[k], side),
             resample_matrix(to_canonical(mask[k], side), dst, area_ratio=ratio))
            for k, side in enumerate(("left", "right"))]


def project_source_mask(root):
    """The fixed source array used by every aligned-cache and QC segment."""
    return active_mask(root)


@dataclass(frozen=True)
class AlignedSegment:
    """One physical segment sampled once on the project 60-Hz elapsed clock."""

    subject: str
    activity: str
    name: str
    pressure_pa: np.ndarray
    pressure_valid: np.ndarray
    force_n: np.ndarray
    force_valid: np.ndarray
    foot_imu_si: np.ndarray
    foot_imu_valid: np.ndarray
    target_m: np.ndarray
    target_valid: np.ndarray
    contact: np.ndarray
    contact_valid: np.ndarray
    time_s: np.ndarray
    pressure_group_delay_s: float
    marker_group_delay_s: float
    foot_imu_group_delay_s: float

    def __post_init__(self):
        n = len(self.time_s)
        shapes = {
            "pressure_pa": (n, 2, ours.N_SENSORS), "pressure_valid": (n, 2, ours.N_SENSORS),
            "force_n": (n, 2), "force_valid": (n, 2), "foot_imu_si": (n, 2, 6),
            "foot_imu_valid": (n, 2, 6), "target_m": (n, len(TARGET_MARKERS), 3),
            "target_valid": (n, len(TARGET_MARKERS), 3), "contact": (n, 2),
            "contact_valid": (n, 2),
        }
        for field, shape in shapes.items():
            if np.asarray(getattr(self, field)).shape != shape:
                raise ValueError(f"{field} must have shape {shape}")
        time_s = np.asarray(self.time_s, dtype=np.float64)
        if time_s.ndim != 1 or not np.isfinite(time_s).all() or (
                len(time_s) > 1 and np.any(np.diff(time_s) <= 0.0)):
            raise ValueError("time_s must be finite and strictly increasing")


def _elapsed(stream: TimedArray) -> TimedArray:
    time_s = stream.time_s - stream.time_s[0]
    return TimedArray(stream.values, time_s, stream.valid, stream.unit,
                      "project_elapsed_from_stream_first_sample", stream.nominal_hz,
                      stream.group_delay_s)


def _project_time(native: MovePortNativeSegment, target_hz=TARGET_FPS):
    streams = (_elapsed(native.pressure), _elapsed(native.markers), _elapsed(native.foot_imu))
    duration = min(stream.time_s[-1] for stream in streams)
    count = int(np.floor(duration * target_hz + 1e-9)) + 1
    return np.arange(count, dtype=np.float64) / target_hz


def _registered_timed_pressure(pressure: TimedArray, source_mask, cell_area):
    destination_mask = ours.active_mask()
    values, valid = [], []
    for side_index, side in enumerate(("left", "right")):
        canonical_mask = to_canonical(source_mask[side_index], side)
        grid = to_canonical(pressure.values[:, side_index], side)
        grid_valid = to_canonical(pressure.valid[:, side_index], side)
        source_values = grid[:, canonical_mask]
        source_valid = grid_valid[:, canonical_mask]
        weights = resample_matrix(canonical_mask, destination_mask,
                                  area_ratio=cell_area / ours.CELL_AREA_M2)
        registered = np.where(source_valid, source_values, 0.0) @ weights.T * PSI_TO_PA
        contributors = weights > 1e-12
        registered_valid = (source_valid.astype(np.int16) @ contributors.T.astype(np.int16)
                            == contributors.sum(axis=1)[None, :])
        registered[~registered_valid] = np.nan
        values.append(registered)
        valid.append(registered_valid)
    return (np.stack(values, axis=1).astype(np.float32),
            np.stack(valid, axis=1))


def _contact_timed(markers: TimedArray):
    i = {n: k for k, n in enumerate(MARKERS)}
    labels = np.zeros((len(markers.values), 2), dtype=bool)
    valid = np.zeros_like(labels)
    for side_index, side in enumerate(("L", "R")):
        indices = [i[f"{side}_CAL"], i[f"{side}_MH1"]]
        valid[:, side_index] = np.all(markers.valid[:, indices, 2], axis=1)
        height = np.minimum(markers.values[:, indices[0], 2], markers.values[:, indices[1], 2])
        if np.any(valid[:, side_index]):
            floor = np.percentile(height[valid[:, side_index]], FLOOR_PERCENTILE)
            labels[:, side_index] = valid[:, side_index] & ((height - floor) < CONTACT_HEIGHT_M)
            _drop_short(labels[:, side_index], int(round(MIN_CONTACT_S * TARGET_FPS)))
    values = labels.astype(np.float32)
    values[~valid] = np.nan
    return values, valid


def align_native_segment(native: MovePortNativeSegment, root, source_mask, cell_area_m2,
                         target_hz=TARGET_FPS):
    """Map native streams to the project elapsed clock without claiming hardware sync."""
    if native.sync_status != "unverified_endpoint_origin":
        raise ValueError("only unverified native MovePort timing is supported")
    target_time = _project_time(native, target_hz)
    pressure = resample_causal(_elapsed(native.pressure), target_time, RESAMPLE_CUTOFF_HZ)
    markers = resample_causal(_elapsed(native.markers), target_time, RESAMPLE_CUTOFF_HZ)
    imu = resample_causal(_elapsed(native.foot_imu), target_time, RESAMPLE_CUTOFF_HZ)
    pressure_pa, pressure_valid = _registered_timed_pressure(pressure, source_mask, cell_area_m2)
    force_valid = np.all(pressure_valid, axis=2)
    force_n = pressure_pa.sum(axis=2) * ours.CELL_AREA_M2
    force_n[~force_valid] = np.nan
    target = marker_target(markers)
    contact, contact_valid = _contact_timed(markers)
    return AlignedSegment(
        subject=native.subject, activity=native.activity, name=native.segment,
        pressure_pa=pressure_pa, pressure_valid=pressure_valid,
        force_n=force_n.astype(np.float32), force_valid=force_valid,
        foot_imu_si=imu.values.astype(np.float32), foot_imu_valid=imu.valid,
        target_m=target.values.astype(np.float32), target_valid=target.valid,
        contact=contact, contact_valid=contact_valid, time_s=target_time,
        pressure_group_delay_s=pressure.group_delay_s,
        marker_group_delay_s=markers.group_delay_s,
        foot_imu_group_delay_s=imu.group_delay_s,
    )


def _native_cell_area(native):
    pressure_stream, force_stream = native.pressure, native.force
    if force_stream is None or len(force_stream.values) != len(pressure_stream.values):
        return None
    if (pressure_stream.nominal_hz != force_stream.nominal_hz
            or pressure_stream.time_basis != force_stream.time_basis
            or not np.allclose(pressure_stream.time_s, force_stream.time_s, rtol=0.0, atol=1e-12)):
        return None
    areas = []
    for side in range(2):
        pressure_valid = np.all(pressure_stream.valid[:, side], axis=(1, 2))
        force_valid = force_stream.valid[:, side]
        pressure = pressure_stream.values[:, side].sum(axis=(1, 2))
        force = force_stream.values[:, side]
        valid = pressure_valid & force_valid & np.isfinite(pressure) & np.isfinite(force) & (pressure > 1e-6)
        if valid.sum() < 50:
            return None
        areas.append(float(np.median(force[valid] / pressure[valid]) / PSI_TO_PA))
    if abs(areas[0] - areas[1]) / np.mean(areas) > 0.02:
        return None
    return float(np.median(areas))


def subject_cell_area_native(root, subject):
    """Recover area only from native equal-rate pressure/force pairs, keyed by release root."""
    key = ("native", str(Path(root).resolve()), str(subject))
    if key in _CELL_AREA_CACHE:
        return _CELL_AREA_CACHE[key]
    found = []
    for activity in ACTIVITIES:
        for name in sequence_names(root, subject, activity):
            try:
                area = _native_cell_area(load_native_segment(root, subject, activity, name))
            except Exception:
                continue
            if area is not None:
                found.append(area)
    median = float(np.median(found)) if found else None
    _CELL_AREA_CACHE[key] = (min(INSOLE_CELL_AREAS_M2, key=lambda area: abs(area - median))
                             if median is not None else None)
    return _CELL_AREA_CACHE[key]


def build(root, activities=ACTIVITIES, subject_list=None, mask=None, verbose=True):
    """Build unique aligned segments from native-rate MovePort streams."""
    mask = project_source_mask(root) if mask is None else mask
    subject_list = subject_list or [s for s in subjects(root) if s != CLINICAL_SUBJECT]
    out = []
    for subject in subject_list:
        kept = 0
        for activity in activities:
            for name in sequence_names(root, subject, activity):
                try:
                    native = load_native_segment(root, subject, activity, name)
                    area = subject_cell_area_native(root, subject)
                    if area is None:
                        raise ValueError("no cell area anywhere for this subject")
                    aligned = align_native_segment(native, root, mask, area)
                except Exception as exc:
                    if verbose:
                        print(f"  skip {subject}/{activity}/{name}: {exc}")
                    continue
                if len(aligned.time_s) < WINDOW:
                    continue
                out.append(aligned)
                kept += len(aligned.time_s)
        if verbose and kept:
            print(f"  subject {subject:>2}: {kept:6d} unique aligned frames")
    return out


def subject_folds(subjects, n_folds, seed=0):
    """Partition a subject list into ``n_folds`` disjoint groups, dealt round-robin."""
    subs = sorted(set(map(str, subjects)), key=int)
    order = np.random.default_rng(seed).permutation(len(subs))
    return [[subs[i] for i in order[f::n_folds]] for f in range(n_folds)]


def folds(per_segment, n_folds=4, seed=0):
    """Partition subjects into ``n_folds`` held-out groups, dealt round-robin."""
    if isinstance(per_segment, dict):
        subs = sorted({k[0] for k in per_segment}, key=int)
    else:
        subs = sorted({segment.subject for segment in per_segment}, key=int)
    return subject_folds(subs, n_folds, seed)


def split(per_segment, test_subjects, val_fraction=0.15, seed=0):
    """Train / validation / test tensors for one fold; validation is whole training segments."""
    n_arrays = len(next(iter(per_segment.values())))

    def gather(keys):
        keys = sorted(keys)
        if not keys:
            return None
        return tuple(np.concatenate([per_segment[k][i] for k in keys]) for i in range(n_arrays))

    test_subjects = set(test_subjects)
    train_keys = [k for k in per_segment if k[0] not in test_subjects]
    test_keys = [k for k in per_segment if k[0] in test_subjects]
    order = np.random.default_rng(seed).permutation(len(train_keys))
    n_val = max(1, int(round(val_fraction * len(train_keys))))
    return (gather([train_keys[i] for i in order[n_val:]]),
            gather([train_keys[i] for i in order[:n_val]]),
            gather(test_keys))


NAMES = ("pressure_pa", "force_n", "foot_imu_si", "target_m", "contact")
ALIGNED_SCHEMA_KEYS = {
    "schema_version", "alignment_status", "target_hz", "time_basis",
    "pressure_pa", "pressure_valid", "force_n", "force_valid",
    "foot_imu_si", "foot_imu_valid", "target_m", "target_valid", "contact",
    "contact_valid", "time_s", "time_valid", "segment_id", "segment_subject",
    "segment_activity", "segment_name", "segment_start", "segment_stop", "marker_names",
    "fold_subjects", "source_sync_status", "common_group_delay_s", "resample_filter",
    "pressure_group_delay_s", "marker_group_delay_s", "foot_imu_group_delay_s",
    "pressure_unit", "force_unit", "foot_imu_units", "target_unit",
}


def _text(value):
    return np.asarray(value, dtype=str)


def aligned_cache_payload(segments, n_folds=4, seed=0):
    """Concatenate whole segments without persisting overlapping windows."""
    if not segments:
        raise ValueError("at least one aligned segment is required")
    starts, stops, cursor = [], [], 0
    for segment in segments:
        starts.append(cursor)
        cursor += len(segment.time_s)
        stops.append(cursor)
    fold_lists = folds(segments, n_folds, seed)
    width = max((len(fold) for fold in fold_lists), default=0)
    fold_subjects = np.full((n_folds, width), "", dtype=f"U{max(1, max(map(len, {s.subject for s in segments})))}")
    for row, fold in enumerate(fold_lists):
        fold_subjects[row, :len(fold)] = fold
    payload = {
        "schema_version": _text(ALIGNED_SCHEMA),
        "alignment_status": _text(ALIGNMENT_STATUS),
        "target_hz": np.asarray(TARGET_FPS, dtype=np.float64),
        "time_basis": _text("seconds_from_each_segment_first_project_sample"),
        "pressure_pa": np.concatenate([s.pressure_pa for s in segments]),
        "pressure_valid": np.concatenate([s.pressure_valid for s in segments]),
        "force_n": np.concatenate([s.force_n for s in segments]),
        "force_valid": np.concatenate([s.force_valid for s in segments]),
        "foot_imu_si": np.concatenate([s.foot_imu_si for s in segments]),
        "foot_imu_valid": np.concatenate([s.foot_imu_valid for s in segments]),
        "target_m": np.concatenate([s.target_m for s in segments]),
        "target_valid": np.concatenate([s.target_valid for s in segments]),
        "contact": np.concatenate([s.contact for s in segments]),
        "contact_valid": np.concatenate([s.contact_valid for s in segments]),
        "time_s": np.concatenate([s.time_s for s in segments]),
        "time_valid": np.ones(cursor, dtype=bool),
        "segment_id": _text([f"{s.subject}/{s.activity}/{s.name}" for s in segments]),
        "segment_subject": _text([s.subject for s in segments]),
        "segment_activity": _text([s.activity for s in segments]),
        "segment_name": _text([s.name for s in segments]),
        "segment_start": np.asarray(starts, dtype=np.int64),
        "segment_stop": np.asarray(stops, dtype=np.int64),
        "marker_names": _text(TARGET_MARKERS),
        "fold_subjects": fold_subjects,
        "source_sync_status": _text(["unverified_endpoint_origin"] * len(segments)),
        "common_group_delay_s": np.asarray(0.1, dtype=np.float64),
        "resample_filter": _text(RESAMPLE_FILTER),
        "pressure_group_delay_s": np.asarray([s.pressure_group_delay_s for s in segments]),
        "marker_group_delay_s": np.asarray([s.marker_group_delay_s for s in segments]),
        "foot_imu_group_delay_s": np.asarray([s.foot_imu_group_delay_s for s in segments]),
        "pressure_unit": _text("Pa"), "force_unit": _text("N"),
        "foot_imu_units": _text(("m/s^2", "m/s^2", "m/s^2", "rad/s", "rad/s", "rad/s")),
        "target_unit": _text("m"),
    }
    validate_aligned_cache(payload)
    return payload


def validate_aligned_cache(cache):
    """Validate unique-frame layout, shapes, folds, timing, and derived force."""
    if set(cache.keys()) != ALIGNED_SCHEMA_KEYS:
        missing = ALIGNED_SCHEMA_KEYS - set(cache.keys())
        extra = set(cache.keys()) - ALIGNED_SCHEMA_KEYS
        raise ValueError(f"aligned schema keys differ: missing={sorted(missing)}, extra={sorted(extra)}")
    for key in ALIGNED_SCHEMA_KEYS:
        if np.asarray(cache[key]).dtype.kind == "O":
            raise ValueError(f"{key} must not use object dtype")
    if np.asarray(cache["schema_version"]).item() != ALIGNED_SCHEMA:
        raise ValueError("unsupported aligned cache schema")
    expected_units = {
        "pressure_unit": "Pa", "force_unit": "N", "target_unit": "m",
    }
    if any(np.asarray(cache[key]).item() != value for key, value in expected_units.items()) or tuple(
            map(str, cache["foot_imu_units"])) != (
                "m/s^2", "m/s^2", "m/s^2", "rad/s", "rad/s", "rad/s"):
        raise ValueError("aligned cache units do not match the schema")
    if (np.asarray(cache["target_hz"]).item() != TARGET_FPS
            or np.asarray(cache["time_basis"]).item()
            != "seconds_from_each_segment_first_project_sample"):
        raise ValueError("aligned cache must use the project-origin 60-Hz clock")
    time_array = np.asarray(cache["time_s"])
    if time_array.ndim != 1 or not np.issubdtype(time_array.dtype, np.floating):
        raise ValueError("time_s must be a one-dimensional floating array")
    n = len(time_array)
    expected = {
        "pressure_pa": (n, 2, ours.N_SENSORS), "pressure_valid": (n, 2, ours.N_SENSORS),
        "force_n": (n, 2), "force_valid": (n, 2), "foot_imu_si": (n, 2, 6),
        "foot_imu_valid": (n, 2, 6), "target_m": (n, len(TARGET_MARKERS), 3),
        "target_valid": (n, len(TARGET_MARKERS), 3), "contact": (n, 2),
        "contact_valid": (n, 2), "time_valid": (n,),
    }
    for key, shape in expected.items():
        if np.asarray(cache[key]).shape != shape:
            raise ValueError(f"{key} must have shape {shape}")
    segment_count = len(cache["segment_id"])
    for key in ("segment_subject", "segment_activity", "segment_name", "segment_start",
                "segment_stop", "source_sync_status",
                "pressure_group_delay_s", "marker_group_delay_s", "foot_imu_group_delay_s"):
        if np.asarray(cache[key]).shape != (segment_count,):
            raise ValueError(f"{key} must match the segment table")
    starts, stops = np.asarray(cache["segment_start"]), np.asarray(cache["segment_stop"])
    if not np.issubdtype(starts.dtype, np.integer) or not np.issubdtype(stops.dtype, np.integer):
        raise ValueError("segment bounds must have integer dtype")
    fold_subjects = np.asarray(cache["fold_subjects"])
    if fold_subjects.ndim != 2 or fold_subjects.dtype.kind != "U":
        raise ValueError("fold_subjects must be a two-dimensional Unicode array")
    if segment_count == 0 or starts[0] != 0 or stops[-1] != n or np.any(stops <= starts) or (
            segment_count > 1 and np.any(starts[1:] != stops[:-1])):
        raise ValueError("segment ranges must partition every aligned frame exactly once")
    for segment_id, start, stop in zip(cache["segment_id"], starts, stops):
        time_s = np.asarray(cache["time_s"])[start:stop]
        if not np.isfinite(time_s).all() or (len(time_s) > 1 and np.any(np.diff(time_s) <= 0.0)):
            raise ValueError(f"time_s must be strictly increasing within segment {segment_id}")
        if len(np.unique(time_s)) != len(time_s):
            raise ValueError(f"each frame must occur once within segment {segment_id}")
    if len(set(map(str, cache["segment_id"]))) != segment_count:
        raise ValueError("segment_id values must be unique")
    expected_ids = [f"{subject}/{activity}/{name}" for subject, activity, name in
                    zip(cache["segment_subject"], cache["segment_activity"], cache["segment_name"])]
    if list(map(str, cache["segment_id"])) != expected_ids:
        raise ValueError("segment identity columns disagree")
    subjects_in_segments = set(map(str, cache["segment_subject"]))
    assigned = [str(subject) for subject in np.asarray(cache["fold_subjects"]).ravel() if str(subject)]
    if set(assigned) != subjects_in_segments or len(assigned) != len(set(assigned)):
        raise ValueError("every subject must occur in exactly one fold")
    for values_key, valid_key in (("pressure_pa", "pressure_valid"), ("force_n", "force_valid"),
                                  ("foot_imu_si", "foot_imu_valid"), ("target_m", "target_valid"),
                                  ("contact", "contact_valid")):
        values, valid = np.asarray(cache[values_key]), np.asarray(cache[valid_key], dtype=bool)
        if np.asarray(cache[valid_key]).dtype != np.bool_:
            raise ValueError(f"{valid_key} must have boolean dtype")
        if np.any(valid & ~np.isfinite(values)):
            raise ValueError(f"{values_key} has nonfinite valid values")
        if np.any((~valid) & np.isfinite(values)):
            raise ValueError(f"{values_key} invalid entries must remain NaN")
    if np.asarray(cache["time_valid"]).dtype != np.bool_ or not np.asarray(cache["time_valid"]).all():
        raise ValueError("time_valid must be an all-true boolean array")
    if tuple(map(str, cache["marker_names"])) != TARGET_MARKERS:
        raise ValueError("marker_names must preserve the exact target order")
    if np.asarray(cache["alignment_status"]).item() != ALIGNMENT_STATUS:
        raise ValueError("alignment_status must declare project alignment and unverified sync")
    if any(str(value) != "unverified_endpoint_origin" for value in cache["source_sync_status"]):
        raise ValueError("source synchronization must remain explicitly unverified")
    if np.asarray(cache["common_group_delay_s"]).item() != 0.1:
        raise ValueError("common resampling delay must be 0.1 seconds")
    for key in ("pressure_group_delay_s", "marker_group_delay_s", "foot_imu_group_delay_s"):
        if not np.allclose(cache[key], 0.1, rtol=0.0, atol=1e-12):
            raise ValueError(f"{key} must match the common group delay")
    if np.asarray(cache["resample_filter"]).item() != RESAMPLE_FILTER:
        raise ValueError("resample_filter metadata is unsupported")
    for start, stop in zip(starts, stops):
        time_s = np.asarray(cache["time_s"])[start:stop]
        if time_s[0] != 0.0 or not np.allclose(np.diff(time_s), 1.0 / TARGET_FPS,
                                               rtol=0.0, atol=1e-12):
            raise ValueError("each segment must use the project-origin 60-Hz clock")
    pressure_force_valid = np.all(np.asarray(cache["pressure_valid"]), axis=2)
    if not np.array_equal(np.asarray(cache["force_valid"]), pressure_force_valid):
        raise ValueError("force_valid must equal per-foot pressure validity")
    derived = np.asarray(cache["pressure_pa"]).sum(axis=2) * ours.CELL_AREA_M2
    valid_force = np.asarray(cache["force_valid"], dtype=bool)
    if not np.allclose(np.asarray(cache["force_n"])[valid_force], derived[valid_force],
                       rtol=2e-6, atol=1e-6):
        raise ValueError("force_n must equal pressure-derived force")
    return True


def write_aligned_cache(payload, out_path):
    """Validate and serialize a pickle-free ``moveport-aligned-v2`` archive."""
    validate_aligned_cache(payload)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)


def save(root, out_path, n_folds=4, seed=0, **kw):
    """Build and write one unique-frame aligned cache."""
    segments = build(root, **kw)
    payload = aligned_cache_payload(segments, n_folds, seed)
    write_aligned_cache(payload, out_path)
    return {"frames": len(payload["time_s"]), "segments": len(segments),
            "subjects": len(set(payload["segment_subject"])),
            "activities": len(set(payload["segment_activity"])),
            "out": str(out_path)}
