"""Per-segment shank-imu-cache-v1 archives with explicit timing, validity, and provenance.

`physical_time_s` identifies the represented motion; pairing with the aligned
MovePort cache, whose `time_s` stamps availability, therefore subtracts that
cache's common group delay first.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from posesim.shank_imu.generator import ShankImuSegment

SHANK_IMU_SCHEMA = "shank-imu-cache-v1"
SIDE_ORDER = ("left", "right")
SHANK_IMU_UNITS = ("m/s^2", "m/s^2", "m/s^2", "rad/s", "rad/s", "rad/s")
TIME_BASIS = "seconds_from_segment_first_coordinate_sample"
_CONTRACT_STEMS = ("state_contract", "anatomical_frame_contract", "gait2392_contract")
_SHANK_IMU_DIR = Path(__file__).parent

SHANK_IMU_SCHEMA_KEYS = {
    "schema_version", "subject", "activity", "name", "segment_id", "side_order",
    "shank_imu_si", "shank_imu_valid", "physical_time_s", "available_time_s",
    "group_delay_s", "time_basis", "target_hz", "shank_imu_units", "gravity_w",
    "antialias_coefficients", "antialias_coefficients_sha256",
    "state_contract_sha256", "anatomical_frame_contract_sha256", "gait2392_contract_sha256",
    "source_sync_status",
}


def _text(value) -> np.ndarray:
    return np.asarray(value, dtype=str)


def _coefficient_digest(coefficients: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(coefficients, dtype=np.float64).tobytes()).hexdigest()


def shank_imu_cache_payload(
    segment: ShankImuSegment,
    *,
    subject: str,
    activity: str,
    name: str,
    gravity_w: np.ndarray,
    antialias_coefficients: np.ndarray,
    contract_dir: str | Path = _SHANK_IMU_DIR,
) -> dict:
    """Serialize one generated segment with the hashes of everything that shaped it."""
    gravity = np.asarray(gravity_w, dtype=np.float64)
    coefficients = np.asarray(antialias_coefficients, dtype=np.float64)
    frame_valid = np.asarray(segment.valid, dtype=bool)
    values = segment.values.astype(np.float32)
    values[~frame_valid] = np.nan
    target_step = np.diff(segment.physical_time_s)
    payload = {
        "schema_version": _text(SHANK_IMU_SCHEMA),
        "subject": _text(subject),
        "activity": _text(activity),
        "name": _text(name),
        "segment_id": _text(f"{subject}/{activity}/{name}"),
        "side_order": _text(SIDE_ORDER),
        "shank_imu_si": values,
        "shank_imu_valid": np.broadcast_to(frame_valid[:, None, None], values.shape).copy(),
        "physical_time_s": np.asarray(segment.physical_time_s, dtype=np.float64),
        "available_time_s": np.asarray(segment.available_time_s, dtype=np.float64),
        "group_delay_s": np.asarray(segment.group_delay_s, dtype=np.float64),
        "time_basis": _text(TIME_BASIS),
        "target_hz": np.asarray(1.0 / float(np.median(target_step)), dtype=np.float64),
        "shank_imu_units": _text(SHANK_IMU_UNITS),
        "gravity_w": gravity,
        "antialias_coefficients": coefficients,
        "antialias_coefficients_sha256": _text(_coefficient_digest(coefficients)),
        "source_sync_status": _text("provider_sync_unverified"),
    }
    for stem in _CONTRACT_STEMS:
        digest = hashlib.sha256((Path(contract_dir) / f"{stem}.json").read_bytes()).hexdigest()
        payload[f"{stem}_sha256"] = _text(digest)
    validate_shank_imu_cache(payload)
    return payload


def validate_shank_imu_cache(cache: dict) -> bool:
    """Validate shapes, timing, validity/NaN pairing, and the coefficient hash."""
    if set(cache.keys()) != SHANK_IMU_SCHEMA_KEYS:
        missing = SHANK_IMU_SCHEMA_KEYS - set(cache.keys())
        extra = set(cache.keys()) - SHANK_IMU_SCHEMA_KEYS
        raise ValueError(f"shank_imu schema keys differ: missing={sorted(missing)}, extra={sorted(extra)}")
    for key in SHANK_IMU_SCHEMA_KEYS:
        if np.asarray(cache[key]).dtype.kind == "O":
            raise ValueError(f"{key} must not use object dtype")
    if np.asarray(cache["schema_version"]).item() != SHANK_IMU_SCHEMA:
        raise ValueError("unsupported shank_imu cache schema")
    if tuple(map(str, cache["side_order"])) != SIDE_ORDER:
        raise ValueError("side_order must be exactly (left, right)")
    if tuple(map(str, cache["shank_imu_units"])) != SHANK_IMU_UNITS:
        raise ValueError("shank_imu_units do not match the schema")
    if np.asarray(cache["time_basis"]).item() != TIME_BASIS:
        raise ValueError("time_basis must stamp physical time from the segment origin")
    expected_id = "/".join(
        str(np.asarray(cache[key]).item()) for key in ("subject", "activity", "name")
    )
    if np.asarray(cache["segment_id"]).item() != expected_id:
        raise ValueError("segment identity columns disagree")
    physical = np.asarray(cache["physical_time_s"])
    if physical.ndim != 1 or not np.isfinite(physical).all() or (
            len(physical) > 1 and np.any(np.diff(physical) <= 0.0)):
        raise ValueError("physical_time_s must be finite and strictly increasing")
    count = len(physical)
    values = np.asarray(cache["shank_imu_si"])
    valid = np.asarray(cache["shank_imu_valid"])
    available = np.asarray(cache["available_time_s"])
    if values.shape != (count, 2, 6) or valid.shape != (count, 2, 6):
        raise ValueError("shank_imu_si and shank_imu_valid must have shape (frames, 2, 6)")
    if valid.dtype != np.bool_:
        raise ValueError("shank_imu_valid must have boolean dtype")
    if available.shape != (count,):
        raise ValueError("available_time_s must give one availability stamp per frame")
    if np.any(valid & ~np.isfinite(values)):
        raise ValueError("shank_imu_si has nonfinite valid values")
    if np.any(~valid & np.isfinite(values)):
        raise ValueError("shank_imu_si invalid entries must remain NaN")
    group_delay = float(np.asarray(cache["group_delay_s"]))
    if not np.isfinite(group_delay) or group_delay <= 0.0:
        raise ValueError("group_delay_s must be one positive declared delay")
    frame_valid = np.all(valid, axis=(1, 2))
    delay = available[frame_valid] - physical[frame_valid]
    if np.any(delay <= 0.0) or np.any(delay > group_delay + 1e-12):
        raise ValueError("valid availability stamps must trail physical time within the bound")
    if not np.all(np.isinf(available[~frame_valid])):
        raise ValueError("invalid frames must never become available")
    target_hz = float(np.asarray(cache["target_hz"]))
    if not np.allclose(np.diff(physical), 1.0 / target_hz, rtol=0.0, atol=1e-9):
        raise ValueError("physical_time_s must be uniform at target_hz")
    gravity = np.asarray(cache["gravity_w"])
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError("gravity_w must be one finite three-vector")
    coefficients = np.asarray(cache["antialias_coefficients"])
    if coefficients.ndim != 1 or len(coefficients) < 3 or not len(coefficients) % 2 or (
            not np.isfinite(coefficients).all()):
        raise ValueError("antialias_coefficients must be one finite odd-length vector")
    if np.asarray(cache["antialias_coefficients_sha256"]).item() != _coefficient_digest(coefficients):
        raise ValueError("anti-alias coefficients do not match their recorded hash")
    for stem in _CONTRACT_STEMS:
        digest = np.asarray(cache[f"{stem}_sha256"]).item()
        if len(digest) != 64 or set(digest) - set("0123456789abcdef"):
            raise ValueError(f"{stem}_sha256 must be one lowercase SHA-256 digest")
    if np.asarray(cache["source_sync_status"]).item() != "provider_sync_unverified":
        raise ValueError("source synchronization must remain explicitly unverified")
    return True


def write_shank_imu_cache(payload: dict, out_path: str | Path) -> None:
    """Validate and serialize one pickle-free ``shank-imu-cache-v1`` archive."""
    validate_shank_imu_cache(payload)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)


def load_shank_imu_cache(path: str | Path) -> dict:
    """Load and validate one archive without ever unpickling objects."""
    with np.load(Path(path), allow_pickle=False) as archive:
        cache = {key: archive[key] for key in archive.files}
    validate_shank_imu_cache(cache)
    return cache


def pair_with_aligned(
    shank_imu_physical_time_s: np.ndarray,
    aligned_time_s: np.ndarray,
    *,
    aligned_group_delay_s: float,
    atol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Match C_shank samples to aligned-cache frames that represent the same motion.

    Returns (shank_imu_indices, aligned_indices) where
    ``aligned_time_s - aligned_group_delay_s`` equals ``shank_imu_physical_time_s``
    within ``atol``; pairing on the two clocks directly would misalign the
    modalities by the aligned cache's group delay.
    """
    physical = np.asarray(shank_imu_physical_time_s, dtype=float)
    aligned = np.asarray(aligned_time_s, dtype=float)
    if physical.ndim != 1 or aligned.ndim != 1:
        raise ValueError("both time vectors must be one-dimensional")
    if not np.isfinite(aligned_group_delay_s) or aligned_group_delay_s < 0.0:
        raise ValueError("aligned_group_delay_s must be one non-negative delay")
    aligned_physical = aligned - aligned_group_delay_s
    position = np.searchsorted(physical, aligned_physical)
    shank_imu_indices, aligned_indices = [], []
    for aligned_index, insert in enumerate(position):
        for candidate in (insert - 1, insert):
            if 0 <= candidate < len(physical) and abs(
                    physical[candidate] - aligned_physical[aligned_index]) <= atol:
                shank_imu_indices.append(candidate)
                aligned_indices.append(aligned_index)
                break
    return np.asarray(shank_imu_indices, dtype=np.int64), np.asarray(aligned_indices, dtype=np.int64)
