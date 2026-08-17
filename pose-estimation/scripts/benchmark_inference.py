"""Streaming and batched inference cost of one retrained model on CPU.

Streaming recomputes the whole receptive field for every output frame, which is
the honest cost of the current implementation rather than of the architecture:
a ring buffer that extends the convolution by one column would not repeat that
work. Both figures are reported because only the streaming one bounds a 60 Hz
deployment.

    python scripts/benchmark_inference.py --run-json <retrain>.json --threads 4
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from posesim.model.encoder import MOMENT_HIDDEN, PosePressureNet


def build(run_json: Path, *, imu_dim: int):
    run = json.loads(run_json.read_text(encoding="utf-8"))
    config = run.get("config", {})
    model = PosePressureNet(
        encoder=run["encoder"], head=run.get("head", "free"), imu_dim=imu_dim, n_joints=10,
        dilations=tuple(config.get("dilations", (1, 2, 4, 8))),
        moment_hidden=config.get("moment_hidden") or MOMENT_HIDDEN)
    model.load_state_dict(torch.load(str(run_json).replace(".json", ".pt"), map_location="cpu"))
    model.eval()
    return model, model.tcn.receptive_field


def timed(fn, *, repeats: int, warmup: int = 5) -> float:
    """Median seconds per call, so one scheduling hiccup does not set the number."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--imu-dim", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-frames", type=int, default=600)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    model, receptive_field = build(args.run_json, imu_dim=args.imu_dim)

    pressure = torch.zeros(1, receptive_field, 2, 33, 15)
    inertia = torch.zeros(1, receptive_field, args.imu_dim)
    long_pressure = torch.zeros(1, args.batch_frames, 2, 33, 15)
    long_inertia = torch.zeros(1, args.batch_frames, args.imu_dim)

    with torch.no_grad():
        stream_s = timed(lambda: model(pressure, inertia), repeats=args.repeats)
        batch_s = timed(lambda: model(long_pressure, long_inertia), repeats=max(5, args.repeats // 6))

    result = {
        "schema": "inference-benchmark-v1",
        "run": str(args.run_json),
        "threads": args.threads,
        # The machine is a shared desktop; the load at measurement time is part
        # of the number, so it is recorded rather than assumed to be zero.
        "load_average_1min": os.getloadavg()[0],
        "receptive_field_frames": int(receptive_field),
        "streaming_ms_per_frame": stream_s * 1000.0,
        "streaming_fps": 1.0 / stream_s,
        "amortised_ms_per_frame": batch_s * 1000.0 / args.batch_frames,
        "batch_frames": args.batch_frames,
        "note": "streaming recomputes the whole receptive field for each output frame",
    }
    print(json.dumps(result, indent=1, sort_keys=True))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
