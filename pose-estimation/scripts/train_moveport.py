"""Train marker regression on MovePort, one fold and seed per run.

    python scripts/train_moveport.py --encoder dense --fold 0 --seed 0
    python scripts/train_moveport.py --encoder moments --fold 0 --seed 0   # E1 null hypothesis
"""
from __future__ import annotations
import argparse, json, os, sys, time

import numpy as np
import torch

from posesim.model.encoder import PosePressureNet
from posesim.model.kinematic_head import KinematicHead
from posesim.model.losses import beta_nll, link_variance, mpjpe
from posesim.data.mpdataset import ALIGNED_SCHEMA, subject_folds, validate_aligned_cache
from posesim.data.windows import window_refs

WARMUP_EPOCHS = 20      # MSE first; NLL from a cold start inflates variance instead of fitting.
                        # UMotion spends 20 of 350 on this, so the budget must be long enough
                        # that the warm-up is a small fraction of it.


def load(path, window=64, stride=32, shank_imu=None, uses_foot_imu=True):
    """Load prebuilt windows, or materialise temporary views from aligned-v2 segments."""
    with np.load(path, allow_pickle=False) as archive:
        schema = archive["schema_version"].item() if "schema_version" in archive else None
        if schema == ALIGNED_SCHEMA:
            validate_aligned_cache(archive)
            cache = {key: archive[key].copy() for key in archive.files}
        else:
            cache = None
    if cache is None:
        data = np.load(path, allow_pickle=True)
        return ({k: data[k] for k in ("pressure", "force", "imu", "markers", "contact")},
                data["index"], data["folds"])

    ranges = list(zip(cache["segment_start"], cache["segment_stop"]))
    refs = window_refs(ranges, size=window, stride=stride)
    shank_imu_values = shank_imu_row_valid = None
    if shank_imu is not None:
        shank_imu_values, shank_imu_row_valid = shank_imu
    by_segment = [[] for _ in ranges]
    for ref in refs:
        selection = slice(ref.start, ref.stop)
        if (cache["pressure_valid"][selection].all()
                and cache["force_valid"][selection].all()
                and (not uses_foot_imu or cache["foot_imu_valid"][selection].all())
                and cache["target_valid"][selection].all()
                and (shank_imu_row_valid is None or shank_imu_row_valid[selection].all())):
            by_segment[ref.segment_index].append(ref)

    source = {"pressure": cache["pressure_pa"], "force": cache["force_n"],
              "imu": cache["foot_imu_si"], "markers": cache["target_m"],
              "contact": cache["contact"]}
    if shank_imu_values is not None:
        source["shank_imu"] = shank_imu_values
    materialised = {key: [] for key in source}
    index, cursor = [], 0
    for segment_index, segment_refs in enumerate(by_segment):
        for key, values in source.items():
            materialised[key].extend(values[ref.start:ref.stop] for ref in segment_refs)
        start, stop = cursor, cursor + len(segment_refs)
        index.append((cache["segment_subject"][segment_index],
                      cache["segment_activity"][segment_index],
                      cache["segment_name"][segment_index], start, stop))
        cursor = stop
    arrays = {}
    for key, values in source.items():
        arrays[key] = (np.stack(materialised[key]) if materialised[key]
                       else np.empty((0, window) + values.shape[1:], dtype=values.dtype))
    arrays.update({
        "_unique_target_m": cache["target_m"],
        "_unique_target_valid": cache["target_valid"],
        "_segment_frame_start": cache["segment_start"],
        "_segment_frame_stop": cache["segment_stop"],
    })
    return arrays, np.asarray(index, dtype=str), cache["fold_subjects"]


def shank_imu_statistics(arrays, index, training_window_mask):
    """Per-channel shank_imu mean and standard deviation over training windows."""
    frames = arrays["shank_imu"][training_window_mask].reshape(-1, 2, 6)
    if not len(frames):
        raise ValueError("training split contains no shank_imu windows")
    return frames.mean(axis=0), frames.std(axis=0) + 1e-6


def mirror_shank_imu(values):
    """The left-right reflection of a anatomical frame shank_imu window: sides swap, the
    subject-right force axis flips, and the two in-plane gyro axes flip."""
    mirrored = np.ascontiguousarray(values[..., ::-1, :]).copy()
    mirrored[..., 2] *= -1     # f_z, subject-right
    mirrored[..., 3] *= -1     # w_x, anterior
    mirrored[..., 4] *= -1     # w_y, proximal
    return mirrored


def target_statistics(arrays, index, training_window_mask):
    """Fit target scaling on unique frames from training segments only."""
    if "_unique_target_m" not in arrays:
        target = arrays["markers"][training_window_mask].reshape(-1, 10, 3)
        return target.mean(0), target.std(0) + 1e-6
    frames = []
    validity = []
    for segment_index, row in enumerate(index):
        window_start, window_stop = int(row[3]), int(row[4])
        if not training_window_mask[window_start:window_stop].any():
            continue
        start = arrays["_segment_frame_start"][segment_index]
        stop = arrays["_segment_frame_stop"][segment_index]
        frames.append(arrays["_unique_target_m"][start:stop])
        validity.append(arrays["_unique_target_valid"][start:stop])
    if not frames:
        raise ValueError("training split contains no valid unique target frames")
    target = np.concatenate(frames).astype(np.float64)
    valid = np.concatenate(validity)
    target[~valid] = np.nan
    return np.nanmean(target, axis=0), np.nanstd(target, axis=0) + 1e-6


def attach_shank_imu(cache_path, shank_imu_dir):
    """Pair per-segment shank_imu caches onto aligned rows by physical time.

    The aligned clock stamps availability; row ``i`` holds motion at
    ``time_s[i] - 0.10 s``, so its virtual-IMU sample is the shank_imu frame at that
    physical time. Rows without a valid pairable shank_imu frame are invalid.
    """
    from posesim.shank_imu.cache import load_shank_imu_cache, pair_with_aligned

    with np.load(cache_path, allow_pickle=False) as archive:
        subjects = archive["segment_subject"]
        activities = archive["segment_activity"]
        names = archive["segment_name"]
        starts = archive["segment_start"]
        stops = archive["segment_stop"]
        time_s = archive["time_s"]
        delay = float(archive["common_group_delay_s"])
        total = len(time_s)
    values = np.zeros((total, 2, 6), dtype=np.float32)
    valid = np.zeros(total, dtype=bool)
    shank_imu_dir = os.fspath(shank_imu_dir)
    missing = [f"{s}/{a}/{n}" for s, a, n in zip(subjects, activities, names)
               if not os.path.isfile(os.path.join(shank_imu_dir, f"shank_imu_{s}_{a}_{n}.npz"))]
    if missing:
        raise ValueError(f"shank_imu caches missing for {len(missing)} segments: {missing[:5]}")
    for subject, activity, name, start, stop in zip(subjects, activities, names,
                                                    starts, stops):
        path = os.path.join(shank_imu_dir, f"shank_imu_{subject}_{activity}_{name}.npz")
        shank_imu = load_shank_imu_cache(path)
        shank_imu_frame_valid = np.all(shank_imu["shank_imu_valid"], axis=(1, 2))
        shank_imu_index, aligned_index = pair_with_aligned(
            shank_imu["physical_time_s"], time_s[start:stop], aligned_group_delay_s=delay)
        keep = shank_imu_frame_valid[shank_imu_index]
        rows = start + aligned_index[keep]
        values[rows] = shank_imu["shank_imu_si"][shank_imu_index[keep]]
        valid[rows] = True
    return values, valid


def inner_masks(index, folds, fold, inner, n_inner=3, seed=0):
    """Fit and validation window masks for one subject-disjoint inner fold."""
    n = int(index[-1][4])
    held_out = set(folds[fold])
    development = sorted({row[0] for row in index} - held_out, key=int)
    val_subjects = subject_folds(development, n_inner, seed)[inner]
    fit_subjects = [s for s in development if s not in set(val_subjects)]
    fit_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    for row in index:
        a, b = int(row[3]), int(row[4])
        if row[0] in set(val_subjects):
            val_mask[a:b] = True
        elif row[0] in set(fit_subjects):
            fit_mask[a:b] = True
    return fit_mask, val_mask, fit_subjects, val_subjects


def subject_averaged_validation(model, arrays, index, mask, stats, device, batch=256,
                                 shank_imu_stats=None):
    """Windowed validation aggregated subject-first, in millimetres.

    The dataset is rebuilt here, so it needs the same feature configuration the
    training set was given; without it the inertial channel silently reverts.
    """
    from posesim.data import insole as ours
    mean, std = stats
    grid_mask = ours.active_mask()
    subject_of = np.concatenate(
        [[row[0]] * (int(row[4]) - int(row[3])) for row in index]
    )[mask]
    dataset = Windows(arrays, mask, stats, shank_imu_stats=shank_imu_stats)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    metric = emits_metres(model)
    model.eval()
    errors = []
    with torch.no_grad():
        for shape, imu, y in loader:
            mu, _ = model(to_grid(shape.to(device), grid_mask), imu.to(device))
            if metric:
                mu = (mu - mean_t) / std_t
            window_error = ((mu - y.to(device)) * std_t).norm(dim=-1).mean(dim=(1, 2))
            errors.append(window_error.cpu().numpy())
    per_window = np.concatenate(errors) * 1000.0
    per_subject = {
        subject: float(per_window[subject_of == subject].mean())
        for subject in sorted(set(subject_of), key=int)
    }
    return {
        "per_subject_mm": per_subject,
        "subject_averaged_mm": float(np.mean(list(per_subject.values()))),
        "pooled_window_mm": float(per_window.mean()),
    }


def fold_masks(index, folds, fold):
    """Window masks for train / val / test, split by subject then by whole segment."""
    n = int(index[-1][4])
    test_subjects = set(folds[fold])
    is_test = np.zeros(n, dtype=bool)
    seg_of = np.empty(n, dtype=np.int64)
    for k, row in enumerate(index):
        a, b = int(row[3]), int(row[4])
        seg_of[a:b] = k
        if row[0] in test_subjects:
            is_test[a:b] = True
    train_segs = np.array(sorted({int(seg_of[i]) for i in np.flatnonzero(~is_test)}))
    rng = np.random.default_rng(0)
    val_segs = set(rng.permutation(train_segs)[:max(1, int(0.15 * len(train_segs)))].tolist())
    is_val = np.isin(seg_of, list(val_segs)) & ~is_test
    return ~is_test & ~is_val, is_val, is_test


from posesim.model.inputs import normalise, to_grid  # noqa: E402  shared with the evaluator


MIRROR_MARKERS = np.array([1, 0, 3, 2, 5, 4, 7, 6, 9, 8])   # L<->R in the target order
LATERAL = 0                                                  # the axis a mirror negates


def mirror(pressure, imu, markers):
    """The left-right reflection of a sample. Exact, not an approximation.

    Handedness is canonical by the time it reaches here, so a mirror is a swap of the two feet
    plus a sign flip on the lateral axis. Measured on the v2 cache: the lateral target mean
    changes sign exactly; the worst-axis standard-deviation shift is 5%, the cohort's
    residual gait asymmetry.
    """
    p = pressure[:, ::-1]
    m = imu[:, ::-1].copy()
    m[..., [1, 4]] *= -1                                     # lateral accel and gyro components
    y = markers[:, MIRROR_MARKERS].copy()
    y[..., LATERAL] *= -1
    return p, m, y


class Windows(torch.utils.data.Dataset):
    """Windows for one split. ``augment`` doubles the training set by reflection.

    With ``shank_imu_stats`` the inertial features are the standardised virtual-IMU
    channels; otherwise they are the released foot-IMU values plus log-force.
    """

    def __init__(self, arrays, mask, stats, augment=False, shank_imu_stats=None):
        self.p = arrays["pressure"][mask]
        self.f = arrays["force"][mask]
        self.imu = arrays["imu"][mask]
        self.shank_imu = arrays["shank_imu"][mask] if shank_imu_stats is not None else None
        self.shank_imu_stats = shank_imu_stats
        self.y = arrays["markers"][mask]
        self.mean, self.std = stats
        self.augment = augment

    def __len__(self):
        return len(self.p) * (2 if self.augment else 1)

    def __getitem__(self, i):
        j, flip = (i % len(self.p), i >= len(self.p)) if self.augment else (i, False)
        p, imu, y = self.p[j], self.imu[j], self.y[j]
        f = self.f[j]
        shank_imu = self.shank_imu[j] if self.shank_imu is not None else None
        if flip:
            p, imu, y = mirror(p, imu, y)
            f = f[:, ::-1]
            if shank_imu is not None:
                shank_imu = mirror_shank_imu(shank_imu)
        shape, mag = normalise(p, f)
        if shank_imu is not None:
            shank_imu_mean, shank_imu_std = self.shank_imu_stats
            features = ((shank_imu - shank_imu_mean) / shank_imu_std).reshape(shank_imu.shape[0], -1)
        else:
            features = np.concatenate([imu.reshape(imu.shape[0], -1), mag], -1)
        y = (y - self.mean) / self.std
        return (torch.from_numpy(np.ascontiguousarray(shape)),
                torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)))


def emits_metres(model):
    """Whether the model's head outputs marker positions in metres."""
    return isinstance(getattr(getattr(model, "tcn", None), "head", None), KinematicHead)


def run_epoch(model, loader, mask, opt, beta, std, device, link_weight=0.0, mean=None,
              max_steps=None, scheduler=None):
    train = opt is not None
    metric_head = emits_metres(model)
    if metric_head and mean is None:
        raise ValueError("a metric-space head needs the target mean to standardise its output")
    model.train(train)
    tot = {"loss": 0.0, "mm": 0.0, "n": 0, "batches": 0}
    for shape, imu, y in loader:
        if max_steps is not None and tot["batches"] >= max_steps:
            break
        shape, imu, y = shape.to(device), imu.to(device), y.to(device)
        grid = to_grid(shape, mask)
        with torch.set_grad_enabled(train):
            mu, logvar = model(grid, imu)
            if metric_head:
                # The kinematic head emits metres; the loss space is standardised.
                mu = (mu - mean) / std
            loss = mpjpe(mu, y) if beta is None else beta_nll(mu, logvar, y, beta)
            if link_weight:
                loss = loss + link_weight * link_variance(mu)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
                if scheduler is not None:
                    scheduler.step()
        with torch.no_grad():
            tot["loss"] += float(loss.detach()) * len(y)
            tot["mm"] += float(mpjpe(mu * std, y * std)) * 1000 * len(y)
        tot["n"] += len(y)
        tot["batches"] += 1
    out = {k: tot[k] / tot["n"] for k in ("loss", "mm")}
    out["batches"] = tot["batches"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/processed/moveport_all.npz")
    ap.add_argument("--encoder", default="dense", choices=("dense", "moments", "none"))
    ap.add_argument("--block", type=int, default=1,
                    help="E3 active-cell block-average level applied before normalisation")
    ap.add_argument("--block-origin", type=int, default=0,
                    help="E3 phase-shift check: offset the block grid on both axes")
    ap.add_argument("--moment-hidden", type=int, default=None,
                    help="summary-encoder width; the default matches the dense budget")
    ap.add_argument("--head", default="free", choices=("free", "kinematic"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--link-weight", type=float, default=0.0,
                    help="free head only; the kinematic head fixes lengths by construction")
    ap.add_argument("--loss", default="beta_nll", choices=("beta_nll", "mse"))
    ap.add_argument("--no-imu", action="store_true")
    ap.add_argument("--shank-imu-dir", default=None,
                    help="per-segment shank_imu caches; the virtual-IMU channels replace "
                         "the released foot-IMU features")
    ap.add_argument("--skip-test", action="store_true",
                    help="accepted for compatibility; outer-test scoring is off by default")
    ap.add_argument("--inspect-outer-test", action="store_true",
                    help="window-pooled outer-test scoring; the protocol scores the outer "
                         "test once with the streaming evaluator, so this is a diagnostic")
    ap.add_argument("--inner-fold", type=int, default=None, choices=(0, 1, 2),
                    help="subject-disjoint inner selection fold; never touches outer test")
    ap.add_argument("--steps", type=int, default=None,
                    help="retrain on the whole development set for a fixed step budget")
    ap.add_argument("--device", default=None, choices=("cpu", "cuda", "mps"))
    ap.add_argument("--dilations", default="1,2,4,8",
                    help="comma-separated TCN dilation stack")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    dilations = tuple(int(d) for d in args.dilations.split(","))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from posesim.data import insole as ours
    if args.inner_fold is not None and args.steps is not None:
        raise ValueError("inner selection and step-budget retraining are separate modes")
    if args.shank_imu_dir is not None and args.no_imu:
        raise ValueError("--shank-imu-dir supplies the inertial channel; drop --no-imu")
    if args.encoder == "none" and args.shank_imu_dir is None:
        raise ValueError("the inertial-only variant is defined on C_shank, not the archived foot IMU")
    shank_imu = attach_shank_imu(args.cache, args.shank_imu_dir) if args.shank_imu_dir else None
    uses_foot_imu = args.shank_imu_dir is None and not args.no_imu
    arrays, index, folds = load(args.cache, window=args.window, stride=args.stride,
                                shank_imu=shank_imu, uses_foot_imu=uses_foot_imu)
    fit_subjects = val_subjects = None
    if args.inner_fold is not None:
        tr, va, fit_subjects, val_subjects = inner_masks(index, folds, args.fold,
                                                         args.inner_fold)
        te = np.zeros_like(tr)
    else:
        tr, va, te = fold_masks(index, folds, args.fold)
        if args.steps is not None:
            tr, va = tr | va, np.zeros_like(va)   # retrain on the whole development set
    mean, std = target_statistics(arrays, index, tr)
    stats = (mean, std)
    shank_imu_stats = shank_imu_statistics(arrays, index, tr) if shank_imu is not None else None
    if args.block > 1:
        from posesim.data.resolution import block_average
        arrays["pressure"] = block_average(arrays["pressure"], args.block,
                                           origin=args.block_origin)
    sets = {n: Windows(arrays, m, stats, augment=(n == "train" and not args.no_augment),
                       shank_imu_stats=shank_imu_stats)
            for n, m in (("train", tr), ("val", va), ("test", te))}
    loaders = {n: torch.utils.data.DataLoader(d, batch_size=args.batch, shuffle=(n == "train"),
                                              num_workers=args.workers,
                                              pin_memory=(device == "cuda"),
                                              persistent_workers=args.workers > 0)
               for n, d in sets.items()}

    if args.shank_imu_dir is not None:
        imu_dim = 12                                                  # 2x6 virtual channels
    else:
        imu_dim = 0 if args.no_imu else 12 + 2                        # 2x6 sensor + 2 log-force
    from posesim.model.encoder import MOMENT_HIDDEN
    model = PosePressureNet(encoder=args.encoder, head=args.head,
                            imu_dim=imu_dim, n_joints=10, dilations=dilations,
                            moment_hidden=args.moment_hidden or MOMENT_HIDDEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps if args.steps is not None else args.epochs)
    mask = ours.active_mask()
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)

    tag = f"{args.encoder}_{args.head}_f{args.fold}_s{args.seed}" + ("_noimu" if args.no_imu else "")
    if args.shank_imu_dir is not None:
        tag += "_shank_imu"
    if args.block > 1:
        tag += f"_b{args.block}" + (f"o{args.block_origin}" if args.block_origin else "")
    if args.moment_hidden is not None:
        tag += f"_mh{args.moment_hidden}"
    if dilations != (1, 2, 4, 8):
        tag += f"_rf{model.tcn.receptive_field}"
    if args.window != 64:
        tag += f"_w{args.window}"
    if args.no_augment:
        tag += "_noaug"
    if args.inner_fold is not None:
        tag += f"_i{args.inner_fold}"
    if args.steps is not None:
        tag += f"_steps{args.steps}"
    os.makedirs(args.out, exist_ok=True)
    steps_per_epoch = len(loaders["train"])
    best, best_epoch, history, diverged = float("inf"), -1, [], False
    link = args.link_weight if args.head == "free" else 0.0
    print(f"{tag}: {len(sets['train'])} train / {len(sets['val'])} val / {len(sets['test'])} test "
          f"windows, {sum(p.numel() for p in model.parameters())/1e3:.0f}k params, {device}")
    result = {"tag": tag, "encoder": args.encoder, "head": args.head,
              "fold": args.fold, "seed": args.seed,
              "imu": not args.no_imu, "loss": args.loss,
              "held_out": list(folds[args.fold]), "steps_per_epoch": steps_per_epoch,
              "config": {"lr": args.lr, "augment": not args.no_augment,
                         "dilations": list(dilations), "window": args.window,
                         "batch": args.batch, "beta": args.beta,
                         "shank_imu": args.shank_imu_dir is not None, "block": args.block,
                         "block_origin": args.block_origin,
                         "encoder": args.encoder, "head": args.head,
                         "loss": args.loss, "link_weight": args.link_weight,
                         "no_imu": args.no_imu,
                         "moment_hidden": args.moment_hidden}}

    if args.steps is not None:
        remaining = args.steps
        step = 0
        while remaining > 0:
            beta = None if args.loss == "mse" or step < WARMUP_EPOCHS * steps_per_epoch \
                else args.beta
            trn = run_epoch(model, loaders["train"], mask, opt, beta, std_t, device, link,
                            mean=mean_t, max_steps=remaining, scheduler=sched)
            step += trn["batches"]
            remaining -= trn["batches"]
            print(f"  step {step}/{args.steps}  train {trn['mm']:6.1f} mm")
            if not np.isfinite(trn["mm"]):
                diverged = True
                print("  diverged; stopping this retrain")
                break
        torch.save(model.state_dict(), os.path.join(args.out, f"{tag}.pt"))
        result["steps"] = args.steps
        result["diverged"] = diverged
        print(f"{tag}: retrained for {args.steps} steps; no validation by design")
    else:
        for epoch in range(args.epochs):
            beta = None if args.loss == "mse" or epoch < WARMUP_EPOCHS else args.beta
            t0 = time.time()
            trn = run_epoch(model, loaders["train"], mask, opt, beta, std_t, device, link,
                            mean=mean_t)
            if args.inner_fold is not None:
                report = subject_averaged_validation(model, arrays, index, va, stats,
                                                      device, batch=args.batch,
                                                      shank_imu_stats=shank_imu_stats)
                val_metric = report["subject_averaged_mm"]
                entry = {"epoch": epoch, "beta": beta, "train_mm": trn["mm"],
                         "val_subject_averaged_mm": val_metric,
                         "val_pooled_mm": report["pooled_window_mm"]}
            else:
                with torch.no_grad():
                    val = run_epoch(model, loaders["val"], mask, None, beta, std_t, device,
                                    mean=mean_t)
                val_metric = val["mm"]
                entry = {"epoch": epoch, "beta": beta, "train_mm": trn["mm"],
                         "val_mm": val_metric}
            history.append(entry)
            if not np.isfinite(trn["mm"]) or not np.isfinite(val_metric):
                diverged = True
                print(f"  {epoch:3d}  diverged; stopping this run")
                break
            if val_metric < best:
                best, best_epoch = val_metric, epoch
                torch.save(model.state_dict(), os.path.join(args.out, f"{tag}.pt"))
            sched.step()
            print(f"  {epoch:3d}  train {trn['mm']:6.1f} mm   val {val_metric:6.1f} mm"
                  f"   {'mse' if beta is None else f'beta{beta}'}"
                  f"   lr {sched.get_last_lr()[0]:.1e}   {time.time()-t0:.0f}s")
        result["history"] = history
        result["best_step"] = (best_epoch + 1) * steps_per_epoch
        result["diverged"] = diverged
        if args.inner_fold is not None:
            result.update({"inner_fold": args.inner_fold, "fit_subjects": fit_subjects,
                           "val_subjects": val_subjects,
                           "val_subject_averaged_mm": best})
            print(f"{tag}: inner val {best:.1f} mm subject-averaged; "
                  f"budget {result['best_step']} steps")
        else:
            result["val_mm"] = best
            if not args.inspect_outer_test:
                print(f"{tag}: val {best:.1f} mm; outer test not inspected")
            else:
                model.load_state_dict(torch.load(os.path.join(args.out, f"{tag}.pt")))
                with torch.no_grad():
                    test = run_epoch(model, loaders["test"], mask, None, None, std_t,
                                     device, mean=mean_t)
                result["test_mm"] = test["mm"]
                print(f"{tag}: test {test['mm']:.1f} mm, "
                      f"held-out subjects {list(folds[args.fold])}")
    with open(os.path.join(args.out, f"{tag}.json"), "w") as fh:
        json.dump(result, fh, indent=1)


if __name__ == "__main__":
    main()
