"""Generate one shank-imu-cache-v1 archive from a placed model and its IK motion.

The frozen `+z`-up world gravity is supplied here, on the caller side of the
generator's explicit-injection boundary. The
anti-alias design defaults to the frozen contract; explicit overrides exist
only for labelled sensitivity experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


from posesim.shank_imu.cache import shank_imu_cache_payload, load_shank_imu_cache, write_shank_imu_cache
from posesim.shank_imu.generator import generate_shank_imu_segment
from posesim.shank_imu.motion import read_motion
from posesim.shank_imu.signal import kaiser_lowpass, load_antialias_contract

GRAVITY_W = np.array([0.0, 0.0, -9.80665])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--activity", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--antialias-taps", type=int, default=None,
                        help="labelled override; the frozen contract is the default")
    parser.add_argument("--antialias-cutoff-hz", type=float, default=None)
    parser.add_argument("--antialias-beta", type=float, default=None)
    parser.add_argument("--sample-rate-hz", type=float, default=None)
    args = parser.parse_args()
    try:
        import opensim  # noqa: F401
    except ImportError:
        print("requires the pinned OpenSim preprocessing runtime", file=sys.stderr)
        return 2
    if not args.model.is_file() or not args.motion.is_file():
        raise FileNotFoundError("placed model and IK motion inputs must exist")
    if args.out.exists():
        raise FileExistsError(args.out)

    from posesim.shank_imu.provider import OpenSimTibiaProvider

    contract = load_antialias_contract()
    overrides = (args.antialias_taps, args.antialias_cutoff_hz, args.antialias_beta,
                 args.sample_rate_hz)
    if any(v is not None for v in overrides) and not all(v is not None for v in overrides):
        raise ValueError("an anti-alias override must state all four parameters")
    sample_rate_hz = args.sample_rate_hz or contract.sample_rate_hz

    time_s, values, labels, in_degrees = read_motion(args.motion)
    step = np.diff(time_s)
    if not len(step) or not np.allclose(step, 1.0 / sample_rate_hz, rtol=1e-6, atol=1e-9):
        raise ValueError("IK motion must be uniform at the declared sample rate")

    model = opensim.Model(str(args.model))
    model.initSystem()
    coordinate_set = model.getCoordinateSet()
    motion_type = {
        coordinate_set.get(index).getName(): coordinate_set.get(index).getMotionType()
        for index in range(coordinate_set.getSize())
    }
    missing = [label for label in labels if label not in motion_type]
    if missing:
        raise ValueError(f"IK motion columns are not model coordinates: {missing}")
    rotational = np.array([motion_type[label] == 1 for label in labels])
    coordinates = values.copy()
    if in_degrees:
        coordinates[:, rotational] = np.deg2rad(coordinates[:, rotational])

    provider = OpenSimTibiaProvider(args.model, coordinate_names=labels)
    if args.antialias_taps is not None:
        coefficients = kaiser_lowpass(
            taps=args.antialias_taps,
            cutoff_hz=args.antialias_cutoff_hz,
            sample_rate_hz=sample_rate_hz,
            beta=args.antialias_beta,
        )
    else:
        coefficients = contract.coefficients
    segment = generate_shank_imu_segment(
        coordinates,
        time_s - time_s[0],
        provider=provider,
        rotational=rotational,
        gravity_w=GRAVITY_W,
        antialias_coefficients=coefficients,
    )
    payload = shank_imu_cache_payload(
        segment,
        subject=args.subject,
        activity=args.activity,
        name=args.name,
        gravity_w=GRAVITY_W,
        antialias_coefficients=coefficients,
    )
    write_shank_imu_cache(payload, args.out)
    loaded = load_shank_imu_cache(args.out)
    print(json.dumps({
        "cache_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "coefficients_sha256": str(loaded["antialias_coefficients_sha256"]),
        "frames_60hz": int(len(loaded["physical_time_s"])),
        "in_degrees": in_degrees,
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "motion_sha256": hashlib.sha256(args.motion.read_bytes()).hexdigest(),
        "rotational_coordinates": int(rotational.sum()),
        "segment_id": str(loaded["segment_id"]),
        "valid_frames": int(np.all(loaded["shank_imu_valid"], axis=(1, 2)).sum()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
