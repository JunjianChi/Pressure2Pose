"""The repository hero: the release video, both pressure maps, the shank angular
rate shaded where that foot is off the ground, and the prediction drawn over the
measurement.

Video frames are decoded with ffmpeg, which must be on PATH, and paired to the
cache by elapsed time; that pairing is a project convention, not evidence of a
shared acquisition clock.

    python scripts/viz_hero_gif.py --cache data/processed/moveport_all.npz \\
      --shank-imu-dir results/shank_imu-candidate --run-json <retrain>.json \\
      --checkpoint <retrain>.pt --fold 0 --subject 1 --out ../assets/hero.gif
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posesim.data.mpdataset import TARGET_MARKERS
from scripts.viz_prediction_gif import LEFT, RIGHT, chain, draw_pressure, save_gif

TRUTH = "#9ca3af"
VIDEO_RATE = 25.0
CACHE_RATE = 60.0
# shank_imu_si columns are the three specific-force axes then the three angular rates;
# index 4 is the sagittal rate, the one a shank swing shows up in.
SAGITTAL_RATE = 4


def video_picks(rows) -> np.ndarray:
    """Which decoded video frame belongs to each cache frame, by elapsed time.

    Indices are relative to the first decoded frame. Because they are derived
    from the cache indices themselves, the video advances with whatever frames
    the panels draw; deriving them from a count instead ran the video slow
    whenever the panels skipped frames.
    """
    rows = np.asarray(rows, dtype=int)
    return np.round((rows - rows[0]) * VIDEO_RATE / CACHE_RATE).astype(int)


def video_frames(path: Path, rows) -> list[Image.Image]:
    """One video frame per cache frame in `rows`, matched by elapsed time.

    Takes the cache indices themselves rather than a start and a count, so the
    video advances with whatever frames the other panels draw. Deriving it from
    a count instead silently ran the video slow whenever the panels skipped
    frames.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not on PATH; it decodes the release video")
    rows = np.asarray(rows, dtype=int)
    lo = int(round(rows[0] * VIDEO_RATE / CACHE_RATE))
    hi = int(round(rows[-1] * VIDEO_RATE / CACHE_RATE))
    with tempfile.TemporaryDirectory() as work:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path),
             "-vf", f"select='between(n\\,{lo}\\,{hi})'", "-vsync", "0",
             str(Path(work) / "%05d.png")], check=True)
        decoded = [Image.open(p).convert("RGB").copy()
                   for p in sorted(Path(work).glob("*.png"))]
    if not decoded:
        raise RuntimeError(f"no frames decoded from {path}")
    # The release videos are letterboxed; the bars are measured once and the
    # same crop is applied to every frame so nothing shifts between them.
    probe = np.asarray(decoded[0].convert("L"), dtype=float)
    lit = np.flatnonzero(probe.max(axis=1) > 24)
    if len(lit) > 8:
        top, bottom = int(lit[0]), int(lit[-1]) + 1
        decoded = [frame.crop((0, top, frame.width, bottom)) for frame in decoded]
    picks = np.clip(video_picks(rows), 0, len(decoded) - 1)
    return [decoded[k] for k in picks]


def draw_overlay(axis, true_pose, pred_pose, pair, limits, ground) -> None:
    a, b = pair
    axis.axhspan(ground - 0.04, ground, color="#e5e7eb", zorder=0)
    for pose, colours, width, z in ((true_pose, (TRUTH, TRUTH), 5.0, 1),
                                    (pred_pose, (LEFT, RIGHT), 3.0, 2)):
        for side, colour in zip(("L", "R"), colours):
            p = chain(pose, side)
            axis.plot(p[:4, a], p[:4, b], lw=width, color=colour, zorder=z,
                      solid_capstyle="round")
            axis.plot(p[[2, 4], a], p[[2, 4], b], lw=width, color=colour, zorder=z,
                      solid_capstyle="round")
    (xlo, xhi), (_, yhi) = limits
    axis.set_xlim(xlo, xhi); axis.set_ylim(ground - 0.06, yhi)
    axis.set_aspect("equal"); axis.set_axis_off()


def draw_trace(axis, rate, loaded, frame, window, fps) -> None:
    """Shank sagittal angular rate over the past, shaded where that foot is off
    the ground. Past only: the model reads no sample after the current one, and
    a figure that scrolls through future data would say otherwise."""
    lo = max(0, frame - window)
    hi = frame + 1
    t = (np.arange(lo, hi) - frame) / fps
    for k, colour in enumerate((LEFT, RIGHT)):
        axis.plot(t, rate[lo:hi, k], lw=1.6, color=colour, alpha=0.95)
    # Both sides, each in its own colour: walking alternates them, so the bands
    # draw the gait cycle as well as the intervals pressure cannot speak for.
    ylo, yhi = axis.get_ylim()
    for k, colour in enumerate((LEFT, RIGHT)):
        axis.fill_between(t, ylo, yhi, where=~loaded[lo:hi, k], color=colour,
                          alpha=0.09, step="mid", zorder=0)
    axis.axvline(0.0, color="#111827", lw=1.2)
    axis.set_xlim(-window / fps, 0.02)
    # Named on the curves themselves: blue and orange mean left and right in
    # every panel, so one labelling carries the whole figure.
    for k, (colour, side) in enumerate(((LEFT, "left"), (RIGHT, "right"))):
        axis.text(0.012, 0.94 - 0.13 * k, side, transform=axis.transAxes,
                  fontsize=9.5, color=colour, va="top", fontweight="bold")
    axis.set_yticks([])
    axis.set_xlabel("s", fontsize=9, color="#374151", labelpad=1)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.tick_params(labelsize=8, colors="#6b7280")


def clip_frames(cache, model, stats, shank_imu, shank_imu_stats, subject, activity, name,
                start, count, video_root, stride=1):
    """One held-out segment's frames: video, pressure, past shank rate, overlay."""
    from posesim.analysis.streaming import segment_predictions
    index = next(k for k, (s, a, n) in enumerate(zip(cache["segment_subject"],
                                                     cache["segment_activity"],
                                                     cache["segment_name"]))
                 if str(s) == subject and str(a) == activity
                 and (name is None or str(n) == name))
    name = str(cache["segment_name"][index])
    pred, _ = segment_predictions(model, cache, index, stats, shank_imu=shank_imu, shank_imu_stats=shank_imu_stats)
    lo_row, hi_row = int(cache["segment_start"][index]), int(cache["segment_stop"][index])

    valid = np.asarray(cache["target_valid"][lo_row:hi_row]).all(axis=(1, 2))
    true = np.asarray(cache["target_m"][lo_row:hi_row], dtype=float)
    pressure = np.nan_to_num(np.asarray(cache["pressure_pa"][lo_row:hi_row], dtype=float))
    force = np.nan_to_num(np.asarray(cache["force_n"][lo_row:hi_row], dtype=float))
    rate = np.nan_to_num(np.asarray(shank_imu[0][lo_row:hi_row, :, SAGITTAL_RATE], dtype=float))
    loaded = force > 20.0

    keep = np.flatnonzero(valid)
    first = keep[min(start, max(0, len(keep) - count))]
    rows = keep[keep >= first][:count * stride:stride]
    pred, true = pred[rows], true[rows]
    pressure, force = pressure[rows], force[rows]
    video = video_frames(video_root / subject / activity / f"video_{name}.avi", rows)

    pair = (1, 2)
    both = np.concatenate([pred.reshape(-1, 3), true.reshape(-1, 3), np.zeros((1, 3))])
    limits = ((both[:, pair[0]].min() - 0.10, both[:, pair[0]].max() + 0.10),
              (both[:, pair[1]].min() - 0.05, both[:, pair[1]].max() + 0.08))
    ground = float(both[:, pair[1]].min())
    vmax = float(np.percentile(pressure[pressure > 0], 99.5)) / 1e3
    span = float(np.abs(rate[rows]).max()) * 1.15
    error = np.linalg.norm(pred - true, axis=-1).mean(axis=-1) * 1000

    frames = []
    for f in range(len(rows)):
        fig = plt.figure(figsize=(13.4, 3.7))
        gs = fig.add_gridspec(1, 5, width_ratios=[1.34, 0.50, 0.50, 1.40, 1.02],
                              wspace=0.10, left=0.015, right=0.985, top=0.79, bottom=0.15)

        axis = fig.add_subplot(gs[0])
        axis.imshow(video[f]); axis.set_axis_off()
        axis.set_title("video", fontsize=10, color="#374151")

        for k, side in enumerate(("left", "right")):
            axis = fig.add_subplot(gs[1 + k])
            draw_pressure(axis, pressure[f, k] / 1e3, side, vmax)
            axis.set_title(f"{side}  {force[f, k]:.0f} N", fontsize=9.5, color="#374151")

        axis = fig.add_subplot(gs[3])
        axis.set_ylim(-span, span)
        draw_trace(axis, rate, loaded, int(rows[f]), 90, CACHE_RATE)
        axis.set_title("shank angular rate", fontsize=10, color="#374151")

        axis = fig.add_subplot(gs[4])
        draw_overlay(axis, true[f], pred[f], pair, limits, ground)
        axis.text(0.0, 1.01, "measured", transform=axis.transAxes, fontsize=10,
                  color=TRUTH, ha="left", va="bottom")
        axis.text(1.0, 1.01, "predicted", transform=axis.transAxes, fontsize=10,
                  color="#111827", ha="right", va="bottom")

        fig.suptitle(f"MovePort {subject} \u00b7 {activity} \u00b7 held out \u00b7 "
                     f"{error[f]:.0f} mm", fontsize=11.5, color="#1f2937", y=0.955)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor="white")
        plt.close(fig); buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
    return frames, {"activity": activity, "frames": len(frames),
                    "mpjpe_mm": float(error.mean())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("data/processed/moveport_all.npz"))
    parser.add_argument("--shank-imu-dir", type=Path, required=True)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, default=Path("data/raw/moveport"))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--activity", action="append", default=None,
                        help="repeatable, as ACTIVITY or ACTIVITY:SEGMENT; the clips play "
                             "in the order given and segment names differ between activities")
    parser.add_argument("--start-frame", type=int, default=600)
    parser.add_argument("--count", type=int, default=85)
    parser.add_argument("--stride", type=int, default=1,
                        help="cache frames per rendered frame; the video follows whichever "
                             "frames are drawn")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from scripts.viz_prediction_strip import load_run
    model, stats, shank_imu, shank_imu_stats = load_run(args.run_json, args.checkpoint, args.cache,
                                              args.shank_imu_dir, args.fold)
    with np.load(args.cache, allow_pickle=False) as archive:
        cache = {key: archive[key].copy() for key in archive.files}

    frames, reports = [], []
    for spec in (args.activity or ["treadmill_normal"]):
        activity, _, wanted = spec.partition(":")
        clip, report = clip_frames(cache, model, stats, shank_imu, shank_imu_stats, args.subject,
                                   activity, wanted or None, args.start_frame, args.count,
                                   args.video_root, args.stride)
        frames += clip
        reports.append(report)
    save_gif(frames, args.out, args.fps, colors=256)
    print(json.dumps({"out": str(args.out), "frames": len(frames), "clips": reports},
                     indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
