"""Immutable-artifact acceptance helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> None:
    """Raise when ``path`` is not the pinned SHA-256 artifact."""
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch: expected {expected}, observed {observed}")


def load_gait2392_contract() -> dict[str, object]:
    """Load the machine-readable pinned Gait2392 artifact and marker contract."""
    path = Path(__file__).with_name("gait2392_contract.json")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def derivative_marker_specs(
    contract: Mapping[str, object], *, fixed: bool = True
) -> list[dict[str, object]]:
    """Return exactly the fixed MovePort-labelled stations for a derivative model."""
    markers = contract["markers"]
    if not isinstance(markers, list):
        raise ValueError("marker contract must contain a list of markers")
    specs = []
    for marker in markers:
        if not isinstance(marker, Mapping) or not marker.get("required", False):
            continue
        specs.append({
            "label": marker["label"],
            "body": marker["body"],
            "location_m": marker["location_m"],
            "fixed": fixed,
        })
    labels = [spec["label"] for spec in specs]
    if len(labels) != 17 or len(set(labels)) != 17:
        raise ValueError("derivative marker contract must contain exactly 17 unique labels")
    return specs


def template_station_queries(contract: Mapping[str, object]) -> list[dict[str, object]]:
    """Return the explicit frozen source for every native-template station."""
    markers = contract["markers"]
    if not isinstance(markers, list):
        raise ValueError("marker contract must contain a list of markers")
    queries = []
    for marker in markers:
        if not isinstance(marker, Mapping) or not marker.get("required", False):
            continue
        kind = marker.get("station_kind")
        if kind not in {"marker", "joint_child_origin"}:
            raise ValueError(f"invalid station kind for {marker.get('label')}")
        query = {
            "label": marker["label"],
            "station_kind": kind,
            "station": marker["station"],
        }
        if kind == "joint_child_origin":
            query["joint"] = marker["joint"]
        queries.append(query)
    if len(queries) != 17:
        raise ValueError("template station contract must contain exactly 17 queries")
    return queries


def validate_derivative_marker_contract(
    specs: list[Mapping[str, object]],
    resolved: Mapping[str, str],
    locations: Mapping[str, object],
    *,
    atol: float = 1e-12,
) -> None:
    """Raise unless the derivative has exactly the fixed configured stations."""
    if len(specs) != 17:
        raise ValueError("derivative must contain exactly 17 stations")
    labels = set()
    for spec in specs:
        label = spec["label"]
        if label in labels:
            raise ValueError(f"duplicate derivative marker {label}")
        labels.add(label)
        if resolved.get(label) != spec["body"]:
            raise ValueError(
                f"marker {label} requires {spec['body']}, resolved to {resolved.get(label)}"
            )
        expected = np.asarray(spec["location_m"], dtype=float)
        observed = np.asarray(locations.get(label), dtype=float)
        if observed.shape != (3,) or not np.allclose(observed, expected, atol=atol, rtol=0.0):
            raise ValueError(f"marker {label} station coordinate mismatch")


def validate_movable_derivative_marker_contract(
    specs: list[Mapping[str, object]],
    resolved: Mapping[str, str],
    locations: Mapping[str, object],
    fixed: Mapping[str, bool],
    *,
    atol: float = 1e-12,
) -> None:
    """Raise unless a placement derivative retains every configured movable station."""
    validate_derivative_marker_contract(specs, resolved, locations, atol=atol)
    for spec in specs:
        label = spec["label"]
        if bool(fixed.get(label, True)):
            raise ValueError(f"marker {label} must remain movable for MarkerPlacer")


def validate_scaled_derivative_marker_contract(
    specs: list[Mapping[str, object]],
    resolved: Mapping[str, str],
    locations: Mapping[str, object],
    body_scale: Mapping[str, float],
    *,
    atol: float = 1e-12,
) -> None:
    """Raise unless scaling changes every derivative station only with its body."""
    if len(specs) != 17:
        raise ValueError("derivative must contain exactly 17 stations")
    for spec in specs:
        label = spec["label"]
        body = spec["body"]
        if resolved.get(label) != body:
            raise ValueError(f"marker {label} requires {body}, resolved to {resolved.get(label)}")
        try:
            factor = float(body_scale[body])
        except KeyError as error:
            raise ValueError(f"missing scale for marker body {body}") from error
        expected = np.asarray(spec["location_m"], dtype=float) * factor
        observed = np.asarray(locations.get(label), dtype=float)
        if observed.shape != (3,) or not np.allclose(observed, expected, atol=atol, rtol=0.0):
            raise ValueError(f"marker {label} scaled coordinate mismatch")


def validate_marker_contract(
    contract: Mapping[str, object], resolved: Mapping[str, str]
) -> None:
    """Raise unless every required configured label resolves to its declared body."""
    markers = contract["markers"]
    if not isinstance(markers, list):
        raise ValueError("marker contract must contain a list of markers")
    for marker in markers:
        if not isinstance(marker, Mapping) or not marker.get("required", False):
            continue
        label = marker["label"]
        body = marker["body"]
        observed = resolved.get(label)
        if observed is None:
            raise ValueError(f"required marker {label} did not resolve")
        if observed != body:
            raise ValueError(
                f"marker {label} requires {body}, resolved to {observed}"
            )


def validate_station_locations(
    contract: Mapping[str, object], locations: Mapping[str, object], *, atol: float = 1e-12
) -> None:
    """Raise unless every required station retains its frozen body-frame location."""
    markers = contract["markers"]
    if not isinstance(markers, list):
        raise ValueError("marker contract must contain a list of markers")
    for marker in markers:
        if not isinstance(marker, Mapping) or not marker.get("required", False):
            continue
        label = marker["label"]
        station = marker["station"]
        expected = np.asarray(marker["location_m"], dtype=float)
        try:
            observed = np.asarray(locations[station], dtype=float)
        except KeyError as error:
            raise ValueError(f"required station {station} for {label} did not resolve") from error
        if observed.shape != (3,) or not np.allclose(observed, expected, atol=atol, rtol=0.0):
            raise ValueError(f"marker {label} station coordinate mismatch")


def validate_model_admission(
    model_path: str | Path,
    contract: Mapping[str, object],
    *,
    bodies: Iterable[str],
    resolved: Mapping[str, str],
    locations: Mapping[str, object],
) -> dict[str, object]:
    """Validate one loaded model against the frozen artifact and station contract."""
    expected_hash = contract["model_sha256"]
    if not isinstance(expected_hash, str):
        raise ValueError("model contract has no SHA-256 string")
    verify_sha256(model_path, expected_hash)
    expected_tibias = contract["tibia_bodies"]
    if not isinstance(expected_tibias, list) or not all(
        isinstance(body, str) for body in expected_tibias
    ):
        raise ValueError("model contract has invalid tibia body names")
    present_bodies = set(bodies)
    missing = [body for body in expected_tibias if body not in present_bodies]
    if missing:
        raise ValueError(f"missing required tibia bodies: {', '.join(missing)}")
    validate_marker_contract(contract, resolved)
    validate_station_locations(contract, locations)
    markers = contract["markers"]
    mandatory = sum(
        1 for marker in markers
        if isinstance(marker, Mapping) and marker.get("required", False)
    )
    return {
        "model_sha256": _sha256(model_path),
        "tibia_bodies": expected_tibias,
        "mandatory_markers": mandatory,
    }
