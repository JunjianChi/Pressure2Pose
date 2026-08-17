"""Deterministic static-window and scale-record acceptance helpers."""
from __future__ import annotations

import numpy as np
import xml.etree.ElementTree as ET

from posesim.data.moveport import marker_index


_SCALE_SEGMENTS = {
    "pelvis": ("pelvis", "torso"),
    "femur_r": ("femur_r",),
    "femur_l": ("femur_l",),
    "tibia_r": ("tibia_r", "talus_r"),
    "tibia_l": ("tibia_l", "talus_l"),
    "foot_r": ("calcn_r", "toes_r"),
    "foot_l": ("calcn_l", "toes_l"),
}


def fold_qc_threshold(development_values: np.ndarray, *, quantile: float) -> float:
    """Return a finite QC cutoff from one supplied value vector."""
    values = np.asarray(development_values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("development_values must be a non-empty finite vector")
    try:
        quantile = float(quantile)
    except (TypeError, ValueError) as error:
        raise ValueError("quantile must be a finite scalar") from error
    if not np.isfinite(quantile) or not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between zero and one")
    return float(np.quantile(values, quantile))


def outer_fold_qc(
    values_by_subject: dict[str, float], *, held_out_subjects: list[str], quantile: float
) -> dict[str, object]:
    """Evaluate a caller-specified QC quantile without using held-out values to estimate it."""
    if not values_by_subject:
        raise ValueError("values_by_subject must not be empty")
    values = {str(subject): float(value) for subject, value in values_by_subject.items()}
    if not np.isfinite(list(values.values())).all():
        raise ValueError("values_by_subject must contain only finite values")
    held_out = sorted({str(subject) for subject in held_out_subjects}, key=int)
    if not held_out or any(subject not in values for subject in held_out):
        raise ValueError("held_out_subjects must be a non-empty subset of values_by_subject")
    development = sorted(set(values) - set(held_out), key=int)
    if not development:
        raise ValueError("at least one development subject is required")
    cutoff = fold_qc_threshold(
        np.array([values[subject] for subject in development]), quantile=quantile
    )
    return {
        "cutoff": cutoff,
        "development_subjects": development,
        "held_out_subjects": held_out,
        "held_out_pass": {subject: values[subject] <= cutoff for subject in held_out},
    }


def segment_fold_cutoffs(
    mean_rms_by_segment: dict[str, float], fold_subjects, *, quantile: float
) -> dict[str, dict]:
    """Per-fold trial-residual cutoffs estimated from development trials only.

    Segments are keyed `subject/activity/name`; each fold's held-out subjects
    contribute no value to their own fold's cutoff.
    """
    values: dict[str, float] = {}
    for key, value in mean_rms_by_segment.items():
        subject = str(key).split("/")[0]
        values[str(key)] = float(value)
        if not subject:
            raise ValueError("segment keys must start with a subject")
    if not values:
        raise ValueError("mean_rms_by_segment must not be empty")
    cutoffs: dict[str, dict] = {}
    for fold, members in enumerate(fold_subjects):
        held_out = {str(subject) for subject in members if str(subject)}
        if not held_out:
            raise ValueError(f"fold {fold} has no held-out subjects")
        development = np.asarray(
            [value for key, value in values.items() if key.split("/")[0] not in held_out]
        )
        cutoff = fold_qc_threshold(development, quantile=quantile)
        excluded = sorted(
            key for key, value in values.items()
            if key.split("/")[0] in held_out and value > cutoff
        )
        cutoffs[str(fold)] = {
            "cutoff_m": cutoff,
            "development_trials": int(len(development)),
            "held_out_subjects": sorted(held_out, key=int),
            "held_out_trials_excluded": excluded,
        }
    return cutoffs


def _point(points: dict[str, np.ndarray], label: str) -> np.ndarray:
    try:
        value = np.asarray(points[label], dtype=float)
    except KeyError as error:
        raise ValueError(f"missing scale landmark {label}") from error
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"invalid scale landmark {label}")
    return value


def _midpoint(points: dict[str, np.ndarray], first: str, second: str) -> np.ndarray:
    return (_point(points, first) + _point(points, second)) / 2.0


def _ratio(
    name: str, observed_first: np.ndarray, observed_second: np.ndarray,
    template_first: np.ndarray, template_second: np.ndarray,
) -> float:
    template_length = np.linalg.norm(template_first - template_second)
    if template_length <= 0.0:
        raise ValueError(f"{name} has zero template length")
    ratio = np.linalg.norm(observed_first - observed_second) / template_length
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError(f"{name} has invalid scale multiplier {ratio}")
    return float(ratio)


def select_static_window(
    markers: np.ndarray, *, hz: float, width: int
) -> tuple[int, int, float]:
    """Return the earliest minimum-motion window over mandatory markers."""
    markers = np.asarray(markers, dtype=float)
    if markers.ndim != 3 or markers.shape[1:] != (17, 3):
        raise ValueError("markers must have shape (frames, 17, 3)")
    if len(markers) < width:
        raise ValueError(f"need at least {width} frames")
    if not np.isfinite(markers).all():
        raise ValueError("mandatory marker coordinates must be finite")
    if hz <= 0.0:
        raise ValueError("hz must be positive")

    scores = np.empty(len(markers) - width + 1)
    for start in range(len(scores)):
        velocity = np.diff(markers[start:start + width], axis=0) * hz
        scores[start] = np.linalg.norm(velocity, axis=-1).mean()
    start = int(np.argmin(scores))
    return start, start + width, float(scores[start])


def select_contract_markers(
    mocap: np.ndarray, labels: list[str], contract: dict[str, object]
) -> np.ndarray:
    """Extract the 17 required G3 markers in the frozen contract order."""
    mocap = np.asarray(mocap, dtype=float)
    if mocap.ndim != 2 or mocap.shape[0] != len(labels):
        raise ValueError("mocap rows must match labels")
    markers = contract["markers"]
    if not isinstance(markers, list) or len(markers) != 17:
        raise ValueError("contract must contain exactly 17 required markers")
    index = marker_index(labels)
    output = []
    for marker in markers:
        label = marker["label"]
        try:
            axes = index[label.lower()]
            output.append(np.stack([mocap[axes[axis]] for axis in ("x", "y", "z")], axis=1))
        except KeyError as error:
            raise ValueError(f"missing mandatory marker coordinate for {label}") from error
    values = np.stack(output, axis=1)
    if not np.isfinite(values).all():
        raise ValueError("mandatory marker coordinates must be finite")
    return values


def scale_multipliers(
    window_mean: dict[str, np.ndarray], stations: dict[str, np.ndarray]
) -> dict[str, float]:
    """Compute the seven frozen isotropic candidate scale multipliers."""
    return {
        "pelvis": _ratio(
            "pelvis", _point(window_mean, "R_IAS"), _point(window_mean, "L_IAS"),
            _point(stations, "R_IAS"), _point(stations, "L_IAS"),
        ),
        "femur_r": _ratio(
            "femur_r", _point(window_mean, "R_FTC"),
            _midpoint(window_mean, "R_FLE", "R_FME"), _point(stations, "R_FTC"),
            _midpoint(stations, "R_FLE", "R_FME"),
        ),
        "femur_l": _ratio(
            "femur_l", _point(window_mean, "L_FTC"),
            _midpoint(window_mean, "L_FLE", "L_FME"), _point(stations, "L_FTC"),
            _midpoint(stations, "L_FLE", "L_FME"),
        ),
        "tibia_r": _ratio(
            "tibia_r", _point(window_mean, "R_TTC"), _point(window_mean, "R_LM"),
            _point(stations, "R_TTC"), _point(stations, "R_LM"),
        ),
        "tibia_l": _ratio(
            "tibia_l", _point(window_mean, "L_TTC"), _point(window_mean, "L_LM"),
            _point(stations, "L_TTC"), _point(stations, "L_LM"),
        ),
        "foot_r": _ratio(
            "foot_r", _point(window_mean, "R_CAL"), _point(window_mean, "R_MH1"),
            _point(stations, "R_CAL"), _point(stations, "R_MH1"),
        ),
        "foot_l": _ratio(
            "foot_l", _point(window_mean, "L_CAL"), _point(window_mean, "L_MH1"),
            _point(stations, "L_CAL"), _point(stations, "L_MH1"),
        ),
    }


def body_scale_factors(scales: dict[str, float]) -> dict[str, float]:
    """Expand the seven frozen multipliers to their twelve scaled bodies."""
    if set(scales) != set(_SCALE_SEGMENTS):
        raise ValueError("scales must contain exactly the seven frozen multiplier names")
    factors = {}
    for multiplier, segments in _SCALE_SEGMENTS.items():
        value = float(scales[multiplier])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{multiplier} must be a finite positive scale")
        factors.update({segment: value for segment in segments})
    return factors


def manual_scale_setup(
    *,
    model_file: str,
    scales: dict[str, float],
    mass_kg: float,
    preserve_mass_distribution: bool,
    output_model_file: str,
    output_scale_file: str,
) -> str:
    """Build a complete manual-scale-only OpenSim setup XML."""
    body_scale_factors(scales)
    if not np.isfinite(mass_kg) or (mass_kg != -1.0 and mass_kg <= 0.0):
        raise ValueError("mass_kg must be -1 or finite and positive")
    checked = {}
    for name, value in scales.items():
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a finite positive scale")
        checked[name] = value

    document = ET.Element("OpenSimDocument", Version="40000")
    tool = ET.SubElement(document, "ScaleTool", name="moveport_manual_scale")
    ET.SubElement(tool, "mass").text = f"{mass_kg:.15g}"
    ET.SubElement(tool, "height").text = "-1"
    ET.SubElement(tool, "age").text = "-1"
    maker = ET.SubElement(tool, "GenericModelMaker")
    ET.SubElement(maker, "model_file").text = model_file
    ET.SubElement(maker, "marker_set_file").text = "Unassigned"
    scaler = ET.SubElement(tool, "ModelScaler")
    ET.SubElement(scaler, "apply").text = "true"
    ET.SubElement(scaler, "scaling_order").text = "manualScale"
    scale_set = ET.SubElement(scaler, "ScaleSet", name="moveport_isotropic_scales")
    objects = ET.SubElement(scale_set, "objects")
    ET.SubElement(scale_set, "groups")
    for multiplier, segments in _SCALE_SEGMENTS.items():
        value = f"{checked[multiplier]:.15g}"
        for segment in segments:
            entry = ET.SubElement(objects, "Scale", name=segment)
            ET.SubElement(entry, "scales").text = f"{value} {value} {value}"
            ET.SubElement(entry, "segment").text = segment
            ET.SubElement(entry, "apply").text = "true"
    ET.SubElement(scaler, "preserve_mass_distribution").text = str(
        preserve_mass_distribution
    ).lower()
    ET.SubElement(scaler, "output_model_file").text = output_model_file
    ET.SubElement(scaler, "output_scale_file").text = output_scale_file
    return ET.tostring(document, encoding="unicode")


def marker_placement_setup(
    *,
    model_file: str,
    scales: dict[str, float],
    marker_file: str,
    start_time_s: float,
    end_time_s: float,
    labels: list[str],
    scaled_model_file: str,
    placed_model_file: str,
    placed_motion_file: str,
) -> str:
    """Build a manual-scale plus frozen-static-MarkerPlacer setup XML."""
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("labels must be non-empty and unique")
    if not np.isfinite([start_time_s, end_time_s]).all() or end_time_s <= start_time_s:
        raise ValueError("marker-placement times must be finite and increasing")
    document = ET.fromstring(manual_scale_setup(
        model_file=model_file, scales=scales, mass_kg=-1.0,
        preserve_mass_distribution=True, output_model_file=scaled_model_file,
        output_scale_file=f"{scaled_model_file}.scales.xml",
    ))
    tool = document.find("ScaleTool")
    if tool is None:
        raise RuntimeError("manual scale setup unexpectedly lacks ScaleTool")
    placer = ET.SubElement(tool, "MarkerPlacer", name="moveport_static_placement")
    ET.SubElement(placer, "apply").text = "true"
    task_set = ET.SubElement(placer, "IKTaskSet", name="moveport_static_tasks")
    objects = ET.SubElement(task_set, "objects")
    for label in labels:
        task = ET.SubElement(objects, "IKMarkerTask", name=label)
        ET.SubElement(task, "apply").text = "true"
        ET.SubElement(task, "weight").text = "1"
    ET.SubElement(task_set, "groups")
    ET.SubElement(placer, "marker_file").text = marker_file
    ET.SubElement(placer, "coordinate_file").text = "Unassigned"
    ET.SubElement(placer, "time_range").text = f"{start_time_s:.15g} {end_time_s:.15g}"
    ET.SubElement(placer, "output_model_file").text = placed_model_file
    ET.SubElement(placer, "output_motion_file").text = placed_motion_file
    ET.SubElement(placer, "output_marker_file").text = "Unassigned"
    ET.SubElement(placer, "max_marker_movement").text = "-1"
    return ET.tostring(document, encoding="unicode")
