"""The demo GIF: measured against predicted lower-body pose beside the pressure that drove it.

Same drawing conventions as `viz_demo_gif.py` — axes off, one ground line, limits fixed over the
whole clip so nothing rescales between frames, and pressure drawn nearest-neighbour because the
cell is the measurement.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posesim.data import insole as ours
from posesim.data.mpdataset import TARGET_MARKERS

LEFT, RIGHT, JOINT = "#2563eb", "#f97316", "#374151"
GROUND = "#e5e7eb"
VIEWS = {"side": (1, 2), "front": (0, 2)}


def chain(pose: np.ndarray, side: str) -> np.ndarray:
    """Pelvis, knee, ankle, heel and toe of one leg, as (5, 3) in metres."""
    m = {name: index for index, name in enumerate(TARGET_MARKERS)}
    knee = (pose[m[f"{side}_FLE"]] + pose[m[f"{side}_FME"]]) / 2.0
    return np.stack([np.zeros(3), knee, pose[m[f"{side}_LM"]],
                     pose[m[f"{side}_CAL"]], pose[m[f"{side}_MH1"]]])


def draw_pose(axis, pose, pair, limits, ground) -> None:
    a, b = pair
    axis.axhspan(ground - 0.04, ground, color=GROUND, zorder=0)
    for side, colour in (("L", LEFT), ("R", RIGHT)):
        p = chain(pose, side)
        axis.plot(p[:4, a], p[:4, b], lw=3.0, color=colour, zorder=2, solid_capstyle="round")
        axis.plot(p[[2, 4], a], p[[2, 4], b], lw=3.0, color=colour, zorder=2,
                  solid_capstyle="round")
        axis.plot(p[1:, a], p[1:, b], "o", ms=4.5, color=JOINT, zorder=3)
    # hollow: the pelvis is the origin of the coordinate frame, not a predicted joint
    axis.plot(0, 0, "o", ms=8, mfc="white", mec=JOINT, mew=1.6, zorder=4)
    (xlo, xhi), (ylo, yhi) = limits
    axis.set_xlim(xlo, xhi); axis.set_ylim(ground - 0.06, yhi)
    axis.set_aspect("equal"); axis.set_axis_off()


def draw_pressure(axis, row, side, vmax, cmap="turbo") -> None:
    """The outline and the values are mirrored together.

    The two insoles are mirror images, and the outline is not symmetric: 60 of
    its 253 cells move under a flip. Mirroring only the values prints the right
    foot's pressure inside a left foot.
    """
    mask = ours.active_mask()
    axis.imshow(ours.for_display(np.where(mask, 0.86, np.nan), side), cmap="gray",
                vmin=0, vmax=1, interpolation="nearest")
    grid = np.full(mask.shape, np.nan)
    grid[mask] = row
    image = ours.for_display(grid, side)
    axis.imshow(image, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest",
                alpha=np.clip(np.nan_to_num(image) / (0.22 * vmax), 0, 1))
    axis.set_axis_off()


def save_gif(frames, out: Path, fps: int, *, colors: int = 128) -> None:
    """One palette for the whole clip.

    Saving RGB frames straight through PIL gives each frame its own 256-colour
    table, which for a figure with a dozen flat colours is most of the file.
    Flat panels are clean at 128; a smooth-shaded body needs more before it
    bands.
    """
    palette = frames[0].quantize(colors=colors, method=Image.MEDIANCUT)
    quantised = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    quantised[0].save(out, save_all=True, append_images=quantised[1:],
                      duration=int(1000 / fps), loop=0, optimize=True)


def clip_frames(pred, true, pressure, force, *, view="side", title=""):
    pair = VIEWS[view]
    # the pelvis origin is the top of the drawn chain and is not one of the ten markers,
    # so the limits must include it or the thigh runs off the panel
    both = np.concatenate([pred.reshape(-1, 3), true.reshape(-1, 3), np.zeros((1, 3))])
    limits = ((both[:, pair[0]].min() - 0.10, both[:, pair[0]].max() + 0.10),
              (both[:, pair[1]].min() - 0.05, both[:, pair[1]].max() + 0.08))
    ground = float(both[:, pair[1]].min())
    vmax = float(np.percentile(pressure[pressure > 0], 99.5)) / 1e3
    error = np.linalg.norm(pred - true, axis=-1).mean(axis=-1) * 1000

    frames = []
    for f in range(len(pred)):
        fig = plt.figure(figsize=(12.8, 4.4))
        gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 1.1, 0.8, 0.8], wspace=0.06,
                              left=0.02, right=0.99, top=0.84, bottom=0.03)
        for column, (pose, label) in enumerate(((true[f], "measured by motion capture"),
                                                (pred[f], "predicted from insole + shank"))):
            axis = fig.add_subplot(gs[column])
            draw_pose(axis, pose, pair, limits, ground)
            axis.set_title(label, fontsize=10, color=JOINT)
        for k, side in enumerate(("left", "right")):
            axis = fig.add_subplot(gs[2 + k])
            draw_pressure(axis, pressure[f, k] / 1e3, side, vmax)
            axis.set_title(f"{side}   {force[f, k]:.0f} N", fontsize=10, color=JOINT)
        fig.suptitle(f"{title}   —   {error[f]:.0f} mm this frame", fontsize=11.5,
                     color="#1f2937", y=0.965)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor="white")
        plt.close(fig); buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))

    return frames, {"frames": len(frames), "size": frames[0].size,
                    "mpjpe_mm": float(error.mean())}


def clip(cache, model, stats, shank_imu, shank_imu_stats, subject, activity, name, start, count,
         *, view):
    """One held-out segment's frames, with the title it carries."""
    from posesim.analysis.streaming import segment_predictions
    index = next(k for k, (s, a, n) in enumerate(zip(cache["segment_subject"],
                                                     cache["segment_activity"],
                                                     cache["segment_name"]))
                 if str(s) == subject and str(a) == activity
                 and (name is None or str(n) == name))
    pred, _ = segment_predictions(model, cache, index, stats, shank_imu=shank_imu, shank_imu_stats=shank_imu_stats)
    a, b = int(cache["segment_start"][index]), int(cache["segment_stop"][index])
    valid = np.asarray(cache["target_valid"][a:b]).all(axis=(1, 2))
    true = np.asarray(cache["target_m"][a:b], dtype=float)
    pressure = np.nan_to_num(np.asarray(cache["pressure_pa"][a:b], dtype=float))
    force = np.nan_to_num(np.asarray(cache["force_n"][a:b], dtype=float))
    pred, true = pred[valid], true[valid]
    pressure, force = pressure[valid], force[valid]
    lo = min(start, max(0, len(pred) - count))
    hi = min(lo + count, len(pred))
    title = (f"MovePort {subject}/{activity}/{cache['segment_name'][index]} "
             f"\u2014 held out from training")
    return clip_frames(pred[lo:hi], true[lo:hi], pressure[lo:hi], force[lo:hi],
                       view=view, title=title)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--shank-imu-dir", type=Path, required=True)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--activity", action="append", default=None,
                        help="repeatable; the clips play in the order given")
    parser.add_argument("--name", default=None)
    parser.add_argument("--start-frame", type=int, default=600)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--view", choices=tuple(VIEWS), default="side")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    activities = args.activity or ["treadmill_normal"]

    from scripts.viz_prediction_strip import load_run
    model, stats, shank_imu, shank_imu_stats = load_run(args.run_json, args.checkpoint, args.cache,
                                              args.shank_imu_dir, args.fold)
    with np.load(args.cache, allow_pickle=False) as archive:
        cache = {key: archive[key].copy() for key in archive.files}

    frames, reports = [], []
    for activity in activities:
        clip_f, report = clip(cache, model, stats, shank_imu, shank_imu_stats, args.subject, activity,
                              args.name, args.start_frame, args.count, view=args.view)
        frames += clip_f
        reports.append({"activity": activity, **report})
    save_gif(frames, args.out, args.fps)
    print(json.dumps({"out": str(args.out), "frames": len(frames), "clips": reports},
                     indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
