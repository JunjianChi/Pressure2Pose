"""Reader for MovePort pressure, optical mocap, EMG, and IMU files.

Released files lack usable timestamps; this reader maps streams to a common
endpoint-normalised timeline.
"""
from __future__ import annotations
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from posesim.data.timed import TimedArray

ROWS, COLS = 31, 11
CELLS_PER_FOOT = ROWS * COLS                 # 341 grid positions, of which ~230 are wired
IPS_FPS, MOCAP_FPS, IMU_FPS = 60.0, 100.0, 100.0
COP_FPS = 60.0

ACTIVITIES = ("treadmill_normal", "treadmill_leghigh", "treadmill_dragging",
              "still", "forward", "back", "halfsquat", "ground_gait")

# Marker names as the readme spells them. R_FLE is capitalised there and L_FlE is not, so every
# lookup goes through a lowercased index rather than these strings directly.
MARKERS = ("IJ", "C7", "RA", "LA", "T8", "R_LHE", "R_RS", "L_LHE", "L_RS", "M_PSIS",
           "R_IAS", "R_FTC", "L_IAS", "L_FTC", "R_FLE", "R_FME", "R_TTC", "R_LM", "R_CAL", "R_MH1",
           "L_FLE", "L_FME", "L_TTC", "L_LM", "L_CAL", "L_MH1")

# The lower-body chain, four joints a side, matching the target used on UnderPressure. The knee is
# the midpoint of the two epicondyles because neither one alone sits at the joint centre; the hip
# uses the greater trochanter, which is the closest palpable landmark to it.
LEG_CHAIN = {"hip": ("{s}_FTC",), "knee": ("{s}_FLE", "{s}_FME"),
             "ankle": ("{s}_LM",), "toe": ("{s}_MH1",)}
PELVIS_MARKERS = ("L_IAS", "R_IAS", "M_PSIS")

IMU_SITES = ("Head", "Waist", "L_H", "R_H", "L_F", "R_F")
IMU_AXES = ("Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z", "Roll", "Pitch", "Yaw")

# Undocumented defects, all established from the released files themselves and absent from the
# readme and the paper. Each one is silent: nothing crashes, the numbers just come out wrong.

# The foot IMUs carry the wrong side for these subjects. Insole force and the mocap heel marker
# agree on handedness for everyone, but for these the left insole's force tracks the RIGHT foot's
# gyro at small lag and the left one only at half a gait cycle.
IMU_SIDES_SWAPPED = ("3", "9", "15", "16", "17", "22", "24")

# The insole samples at 100 Hz rather than 60 for these, while the force/CoP file stays at 60. The
# rate is detected per file below rather than trusted from this list, which is kept only as the
# expectation to check against.
FAST_INSOLE = ("3", "20", "25")

# Pressure is censored below 1.00 psi -- the smallest non-zero value anywhere in the release. On a
# ~64 mm^2 cell that is about 0.44 N per cell discarded, so light loading, early heel contact and
# the toe-off tail are missing rather than small. This bounds what any resolution claim can say.
PSI_FLOOR = 1.0
PSI_TO_PA = 6894.757


def read_matrix(path):
    """A MovePort csv as (channels, frames) plus its row labels.

    Every file is stored transposed -- one row per channel, one column per sample -- with the
    channel name in the first cell and a frame-index header row on top.
    """
    with open(path) as fh:
        rows = [r for r in csv.reader(fh)]
    labels = [r[0] for r in rows[1:]]
    data = np.array([[float(v) for v in r[1:]] for r in rows[1:]], dtype=np.float64)
    return data, labels


def read_matrix_with_frames(path):
    """Read a released csv without replacing its provider frame indices."""
    with open(path) as fh:
        rows = [r for r in csv.reader(fh)]
    frames = np.array([float(v) for v in rows[0][1:]], dtype=np.float64)
    labels = [r[0] for r in rows[1:]]
    data = np.array([[float(v) if v.strip() else np.nan for v in r[1:]] for r in rows[1:]],
                    dtype=np.float64)
    return data, labels, frames


def resample(x, n):
    """Linear resampling of (channels, frames) onto ``n`` frames spanning the same window."""
    src = np.linspace(0.0, 1.0, x.shape[1])
    dst = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(dst, src, ch) for ch in x])


def to_grids(ips):
    """(682, F) channel matrix to (F, 2, 31, 11), left foot first.

    The readme fixes the split: dimensions 1-341 are the left foot and 342-682 the right, each a
    31x11 array in row-major order.
    """
    both = ips.reshape(2, ROWS, COLS, -1)
    return np.moveaxis(both, 3, 0)


def sequence_names(root, subject, activity):
    """Segment names inside one activity, e.g. '1' or 'high_1' -- treadmill files carry the speed."""
    d = os.path.join(root, str(subject), activity)
    if not os.path.isdir(d):
        return []
    return sorted(f[len("ips_"):-len(".csv")] for f in os.listdir(d)
                  if f.startswith("ips_") and f.endswith(".csv"))


def subjects(root):
    return sorted((d for d in os.listdir(root) if d.isdigit()), key=int)


def marker_index(labels):
    """Map a lowercased marker name to its row triple, tolerating the readme's L_FlE casing."""
    idx = {}
    for i, lab in enumerate(labels):
        name, axis = lab.rsplit("_", 1)
        idx.setdefault(name.lower(), {})[axis.lower()] = i
    return idx


def markers_from_mocap(mocap, labels, names=MARKERS):
    """All 26 marker positions as (F, 26, 3) in metres, in the order of ``names``.

    The derived nine-joint chain throws seventeen of these away. That is fine for a lower-body
    target but not for fitting a body: with only the legs constrained the trunk and variants drift to
    whatever reduces the leg error, which is why an earlier fit produced a plausible walk on a
    T-posed torso.
    """
    idx = marker_index(labels)
    out = []
    for n in names:
        ax = idx[n.lower()]
        out.append(np.stack([mocap[ax["x"]], mocap[ax["y"]], mocap[ax["z"]]], axis=1))
    return np.stack(out, axis=1)


def joints_from_markers(mocap, labels):
    """Pelvis plus the eight-joint lower-body chain, as (F, 9, 3) in metres.

    Order is pelvis, then left hip/knee/ankle/toe, then the same on the right. This is a QC and
    demo readout, not the supervision target: that is the ten marker positions in
    `mpdataset.TARGET_MARKERS`.
    """
    idx = marker_index(labels)

    def marker(name):
        ax = idx[name.lower()]
        return np.stack([mocap[ax["x"]], mocap[ax["y"]], mocap[ax["z"]]], axis=1)

    out = [np.mean([marker(m) for m in PELVIS_MARKERS], axis=0)]
    for side in ("L", "R"):
        for names in LEG_CHAIN.values():
            out.append(np.mean([marker(n.format(s=side)) for n in names], axis=0))
    return np.stack(out, axis=1)


def foot_imu(imu, labels, subject=None):
    """Left and right foot accelerometer and gyroscope, (F, 2, 6), with the known side swaps undone.

    Only the feet are kept: the point of this project is what a shoe alone can see, and MovePort's
    other four IMUs sit on the head, waist and wrists. Seven subjects have their two foot IMUs
    labelled the wrong way round; since the insole and the mocap agree with each other on handedness
    for every subject, it is the IMU labels that are wrong, and they are swapped back here.
    """
    idx = {lab.lower(): i for i, lab in enumerate(labels)}
    sites = ("L_F", "R_F")
    if subject is not None and str(subject) in IMU_SIDES_SWAPPED:
        sites = sites[::-1]
    rows = [[idx[f"{site}_{ax}".lower()] for ax in IMU_AXES[:6]] for site in sites]
    return np.moveaxis(imu[np.array(rows)], 2, 0)


def cell_area_m2(ips, force, force_labels):
    """Recover the insole's per-cell area from the shipped force column, in square metres.

    The exported force is exactly ``area * sum(pressure) * psi_to_pascal``, so dividing one by the
    other returns the constant the vendor software used -- to four significant figures, with a
    coefficient of variation around 1e-3. That matters twice over: it is the only route to the
    physical scale of the grid, which the readme never gives, and it separates the two insole sizes
    in the cohort, whose pitches differ by about 8%. A model fed raw 31x11 grids from both sizes is
    mixing two spatial scales.

    Returns None when the two streams have different lengths, which is itself the signal that this
    subject's insole ran at 100 Hz while the force file stayed at 60.
    """
    row = next((i for i, l in enumerate(force_labels) if l.lower() == "l_force"), None)
    if row is None or force.shape[1] != ips.shape[1]:
        return None
    total = ips[:CELLS_PER_FOOT].sum(axis=0)
    ok = total > 1e-6
    if ok.sum() < 50:
        return None
    return float(np.median(force[row][ok] / total[ok]) / PSI_TO_PA)


def insole_fps(n_ips, n_mocap, mocap_fps=MOCAP_FPS):
    """The insole's true rate for one segment, read off its length against the mocap's.

    Some segments from subjects 3, 20, and 25 are approximately 100 Hz rather
    than the common 60 Hz, so the reader estimates a segment-specific rate.
    """
    return float(np.round(mocap_fps * n_ips / max(n_mocap, 1) * 20) / 20)


def native_insole_fps(n_ips, n_mocap):
    """Choose the released insole rate nearest the mocap-length estimate."""
    observed = insole_fps(n_ips, n_mocap)
    return min((IPS_FPS, MOCAP_FPS), key=lambda hz: abs(hz - observed))


def load_sequence(root, subject, activity, name, fps=None):
    """One segment with every stream on a common timeline.

    Returns pressure (F, 2, 31, 11) in psi, joints (F, 9, 3) m, imu (F, 2, 6), force (F, 2), cop
    (F, 2, 2) where provided, the segment's true ``fps``, and ``cell_area`` in square metres.

    The target rate defaults to this segment's inferred insole rate rather than
    a fixed 60 Hz. All streams are then endpoint-normalised to that timeline.
    """
    d = os.path.join(root, str(subject), activity)
    ips, _ = read_matrix(os.path.join(d, f"ips_{name}.csv"))
    mocap, m_labels = read_matrix(os.path.join(d, f"mocap_{name}.csv"))
    imu, i_labels = read_matrix(os.path.join(d, f"imu_{name}.csv"))

    true_fps = insole_fps(ips.shape[1], mocap.shape[1])
    fps = true_fps if fps is None else fps
    n = int(round(ips.shape[1] * fps / true_fps))

    mocap_r = resample(mocap, n)
    seq = {"pressure": to_grids(resample(ips, n)),
           "joints": joints_from_markers(mocap_r, m_labels),
           "markers": markers_from_mocap(mocap_r, m_labels),
           "imu": foot_imu(resample(imu, n), i_labels, subject),
           "subject": str(subject), "activity": activity, "name": name,
           "fps": fps, "true_fps": true_fps, "cell_area": None}

    cop_path = os.path.join(d, f"cop_{name}.csv")
    if os.path.exists(cop_path):
        cop, c_labels = read_matrix(cop_path)
        seq["cell_area"] = cell_area_m2(ips, cop, c_labels)
        cop = resample(cop, n)
        by = {lab.lower(): row for lab, row in zip(c_labels, cop)}
        if "l_force" in by:
            seq["force"] = np.stack([by["l_force"], by["r_force"]], axis=1)
        if "l_cop_x" in by:
            seq["cop"] = np.stack([np.stack([by[f"{s}_cop_x"], by[f"{s}_cop_y"]], axis=1)
                                   for s in ("l", "r")], axis=1)
    return seq


@dataclass(frozen=True)
class MovePortNativeSegment:
    """One MovePort segment without cross-stream alignment or resampling."""

    subject: str
    activity: str
    segment: str
    pressure: TimedArray
    markers: TimedArray
    foot_imu: TimedArray
    frames: dict[str, np.ndarray]
    force: TimedArray | None = None
    cop: TimedArray | None = None
    sync_status: str = "unverified_endpoint_origin"

    def __post_init__(self):
        streams = {"pressure": self.pressure, "markers": self.markers, "foot_imu": self.foot_imu,
                   "force": self.force, "cop": self.cop}
        frames = {}
        for key, stream in streams.items():
            frame = self.frames.get(key)
            if stream is None:
                if frame is not None:
                    raise ValueError(f"{key} frame indices require a stream")
                continue
            if frame is None:
                raise ValueError(f"{key} requires provider frame indices")
            frame = np.array(frame, dtype=np.float64, copy=True)
            if frame.ndim != 1 or frame.shape[0] != stream.values.shape[0]:
                raise ValueError(f"{key} frame indices must match stream length")
            if not np.isfinite(frame).all() or (frame.size > 1 and np.any(np.diff(frame) <= 0)):
                raise ValueError(f"{key} frame indices must be finite and strictly increasing")
            if not np.allclose(stream.time_s, frame / stream.nominal_hz, rtol=0.0, atol=0.0):
                raise ValueError(f"{key} time_s must be derived from provider frame indices")
            frame.setflags(write=False)
            frames[key] = frame
        if self.sync_status != "unverified_endpoint_origin":
            raise ValueError("native MovePort synchronization is unverified")
        object.__setattr__(self, "frames", MappingProxyType(frames))


def _native_timed(values, frames, unit, nominal_hz):
    values = np.asarray(values, dtype=np.float64)
    return TimedArray(values, np.asarray(frames, dtype=np.float64) / nominal_hz,
                      np.isfinite(values), unit, "provider_frame_index/nominal_hz", nominal_hz)


def load_native_segment(root, subject, activity, name):
    """Load native MovePort streams with provider-frame timing and no synchronization claim."""
    directory = os.path.join(root, str(subject), activity)
    ips, _, ips_frame = read_matrix_with_frames(os.path.join(directory, f"ips_{name}.csv"))
    mocap, marker_labels, marker_frame = read_matrix_with_frames(
        os.path.join(directory, f"mocap_{name}.csv"))
    imu, imu_labels, imu_frame = read_matrix_with_frames(os.path.join(directory, f"imu_{name}.csv"))

    foot = foot_imu(imu, imu_labels, subject).copy()
    foot[..., 3:] = np.deg2rad(foot[..., 3:])
    streams = {
        "pressure": _native_timed(to_grids(ips), ips_frame, "psi",
                                   native_insole_fps(ips.shape[1], mocap.shape[1])),
        "markers": _native_timed(markers_from_mocap(mocap, marker_labels), marker_frame, "m", MOCAP_FPS),
        "foot_imu": _native_timed(foot, imu_frame, "m/s^2;rad/s", IMU_FPS),
    }
    frames = {"pressure": ips_frame, "markers": marker_frame, "foot_imu": imu_frame}

    cop_path = os.path.join(directory, f"cop_{name}.csv")
    if os.path.exists(cop_path):
        cop, cop_labels, cop_frame = read_matrix_with_frames(cop_path)
        by = {label.lower(): row for label, row in zip(cop_labels, cop)}
        if "l_force" in by and "r_force" in by:
            streams["force"] = _native_timed(
                np.stack([by["l_force"], by["r_force"]], axis=1), cop_frame, "N", COP_FPS)
            frames["force"] = cop_frame
        if all(f"{side}_cop_{axis}" in by for side in ("l", "r") for axis in ("x", "y")):
            streams["cop"] = _native_timed(
                np.stack([np.stack([by[f"{side}_cop_x"], by[f"{side}_cop_y"]], axis=1)
                          for side in ("l", "r")], axis=1), cop_frame, "m", COP_FPS)
            frames["cop"] = cop_frame
    return MovePortNativeSegment(str(subject), activity, str(name), streams["pressure"], streams["markers"],
                                 streams["foot_imu"], frames, streams.get("force"), streams.get("cop"))


def write_native_segment(segment, path):
    """Write one native MovePort segment in the ``moveport-segment-v1`` schema."""
    def text(value):
        return np.asarray(value, dtype=str)

    payload = {
        "schema_version": text("moveport-segment-v1"),
        "subject": text(segment.subject),
        "activity": text(segment.activity),
        "segment": text(segment.segment),
        "sync_status": text(segment.sync_status),
        "pressure_psi": segment.pressure.values,
        "pressure_frame": segment.frames["pressure"],
        "pressure_time_s": segment.pressure.time_s,
        "pressure_valid": segment.pressure.valid,
        "pressure_time_basis": text(segment.pressure.time_basis),
        "pressure_nominal_hz": np.asarray(segment.pressure.nominal_hz),
        "pressure_group_delay_s": np.asarray(segment.pressure.group_delay_s),
        "markers_world_m": segment.markers.values,
        "marker_names": text(MARKERS),
        "mocap_frame": segment.frames["markers"],
        "mocap_time_s": segment.markers.time_s,
        "marker_valid": segment.markers.valid,
        "mocap_time_basis": text(segment.markers.time_basis),
        "mocap_nominal_hz": np.asarray(segment.markers.nominal_hz),
        "mocap_group_delay_s": np.asarray(segment.markers.group_delay_s),
        "foot_imu_si": segment.foot_imu.values,
        "foot_imu_frame": segment.frames["foot_imu"],
        "foot_imu_time_s": segment.foot_imu.time_s,
        "foot_imu_valid": segment.foot_imu.valid,
        "foot_imu_units": text(("m/s^2", "m/s^2", "m/s^2", "rad/s", "rad/s", "rad/s")),
        "foot_imu_time_basis": text(segment.foot_imu.time_basis),
        "foot_imu_nominal_hz": np.asarray(segment.foot_imu.nominal_hz),
        "foot_imu_group_delay_s": np.asarray(segment.foot_imu.group_delay_s),
    }
    if segment.force is not None:
        payload.update({
            "force_n": segment.force.values,
            "force_frame": segment.frames["force"],
            "force_time_s": segment.force.time_s,
            "force_valid": segment.force.valid,
            "force_time_basis": text(segment.force.time_basis),
            "force_nominal_hz": np.asarray(segment.force.nominal_hz),
            "force_group_delay_s": np.asarray(segment.force.group_delay_s),
        })
    if segment.cop is not None:
        payload.update({
            "cop_m": segment.cop.values,
            "cop_frame": segment.frames["cop"],
            "cop_time_s": segment.cop.time_s,
            "cop_valid": segment.cop.valid,
            "cop_time_basis": text(segment.cop.time_basis),
            "cop_nominal_hz": np.asarray(segment.cop.nominal_hz),
            "cop_group_delay_s": np.asarray(segment.cop.group_delay_s),
        })
    np.savez_compressed(Path(path), **payload)


def active_mask(root, subject=1, activities=("still", "treadmill_normal"), n_seq=2):
    """Which of the 31x11 grid positions are wired, derived from data rather than assumed.

    The readme says 230 of 341 per side but does not say which. A cell is taken as wired if it ever
    reads non-zero, unioned over more than one activity: standing loads the whole sole at once, but
    only walking loads the toes hard at push-off, and reading the mask from standing alone quietly
    drops the cells that matter most for the heel-to-toe roll.
    """
    mask = np.zeros((2, ROWS, COLS), dtype=bool)
    for activity in activities:
        for name in sequence_names(root, subject, activity)[:n_seq]:
            ips, _ = read_matrix(os.path.join(root, str(subject), activity, f"ips_{name}.csv"))
            mask |= to_grids(ips).max(axis=0) > 0
    return mask


# MovePort stores each insole in its own physical frame, so the two arrays are already mirror
# images: flipping the right one across the foot's long axis makes the wired masks agree exactly
# (correlation +1.0000, against -0.2457 unflipped). That is the opposite of the 253-cell rig, whose
# two feet are read through the same matrix orientation. Because the raw conventions differ, one
# function cannot serve both purposes, and conflating them is what drew two identical feet here.
STORED_AS_MIRROR_PAIR = True


def to_canonical(grid, side):
    """Put a foot into the shared anatomical frame used for model input.

    Cell (i, j) must mean the same anatomical location on both feet before any spatially structured
    model, or a resolution study is comparing medial cells on one foot with lateral cells on the
    other. Since the two are stored mirrored, that means flipping the right one.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return np.asarray(grid) if side == "left" else np.asarray(grid)[..., ::-1]


def for_display(grid, side):
    """Leave a foot in its own physical frame, so a pair is drawn as the mirrored pair it is.

    Drawing is the opposite requirement to modelling: a figure should show two mirror-image feet,
    because that is what makes a left/right mix-up visible at a glance. MovePort already stores them
    that way, so this is the identity -- it exists to make the choice explicit rather than implied.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return np.asarray(grid)


def ankle_sensor_track(markers, names=MARKERS):
    """Where a virtual ankle IMU sits and how it is oriented, per foot, from the mocap markers.

    Returns ``(pos, R)`` of shape (F, 2, 3) and (F, 2, 3, 3), left foot first, with R taking a
    vector from the foot's frame to the world.

    The sensor goes on the **lateral malleolus** because that is a marker we actually have, it is
    the bony landmark a real ankle IMU is strapped over, and it is the closest available point to
    where this project's own hardware sits. It is deliberately not the SMPL ankle joint: a joint
    centre is inside the leg, and a sensor is on the skin.

    The frame is built from the three foot markers rather than inherited from a body model, which
    avoids the incoherence running through the whole sparse-IMU literature -- every implementation
    there takes the sensor's *position* from a skin vertex and its *orientation* from a bone joint,
    two things that are not rigidly attached to each other. Here both come from the same three
    points on the same segment.

    Heel to forefoot gives the long axis; the malleolus offset with that component removed gives the
    second; the third follows by cross product, ordered so the frame is right-handed.

    The axes are deliberately **not** claimed to be anatomical. Measured on this data, the vector
    from heel to malleolus runs 9 cm forward against 5.8 cm up, so what is left after removing the
    long axis sits about 40 degrees off vertical -- it mixes up and lateral, because the malleolus
    is a lateral landmark. That does not matter and it is worth saying why: a real IMU is mounted at
    an unknown constant rotation to any frame we could name, so the comparison against it has to
    solve for that rotation regardless. What the frame must be is rigidly attached to the segment,
    right-handed and well conditioned. It is all three, and the mounting rotation absorbs the rest.
    """
    i = {n: k for k, n in enumerate(names)}
    pos, rot = [], []
    for side in ("L", "R"):
        cal, mh1, lm = (markers[:, i[f"{side}_{m}"]] for m in ("CAL", "MH1", "LM"))
        fwd = _unit(mh1 - cal)
        v = lm - cal
        up = _unit(v - (v * fwd).sum(-1, keepdims=True) * fwd)
        lat = np.cross(up, fwd)                             # this order, so det(R) = +1
        pos.append(lm)
        rot.append(np.stack([fwd, lat, up], axis=-1))       # columns: foot frame -> world
    return np.stack(pos, axis=1), np.stack(rot, axis=1)


def _unit(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)
