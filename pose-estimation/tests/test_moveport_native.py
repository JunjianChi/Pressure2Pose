import csv
import subprocess
import sys

import numpy as np
import pytest

from posesim.data.moveport import IMU_AXES, IMU_SITES, MARKERS, load_native_segment, write_native_segment


def _write_csv(path, frames, channels):
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", *frames])
        writer.writerows([[label, *values] for label, values in channels])


def _fixture(root, subject="1", pressure_nan=False, pressure_blank=False, imu_nan=False,
             pressure_count=3, marker_count=5):
    directory = root / str(subject) / "still"
    directory.mkdir(parents=True)
    pressure_frames = [10, 12, 20] if pressure_count == 3 else [10, 12, 20, 21, 22]
    marker_frames = list(range(20, 20 + marker_count))
    _write_csv(directory / "ips_1.csv", pressure_frames, [
        (f"P{i}", ["" if pressure_blank and i == 0 and frame == 0
                    else np.nan if pressure_nan and i == 0 else i + frame
                    for frame in range(pressure_count)])
        for i in range(682)
    ])
    _write_csv(directory / "mocap_1.csv", marker_frames, [
        (f"{name}_{axis}", [marker + coordinate + frame for frame in range(marker_count)])
        for marker, name in enumerate(MARKERS) for coordinate, axis in enumerate(("X", "Y", "Z"))
    ])
    channels = []
    for site_index, site in enumerate(IMU_SITES):
        for axis_index, axis in enumerate(IMU_AXES[:6]):
            value = site_index * 100.0 + axis_index
            if site == "L_F" and axis == "Acc_X":
                value = 11.0
            if site == "R_F" and axis == "Acc_X":
                value = 22.0
            if site == "L_F" and axis == "Gyr_X":
                value = 180.0
            if site == "R_F" and axis == "Acc_Y" and imu_nan:
                value = np.nan
            channels.append((f"{site}_{axis}", [value] * 4))
    _write_csv(directory / "imu_1.csv", [40, 41, 42, 43], channels)
    _write_csv(directory / "cop_1.csv", [30, 31, 32], [
        ("L_Force", [10.0, 11.0, 12.0]), ("R_Force", [20.0, 21.0, 22.0]),
        ("L_cop_x", [0.1, 0.2, 0.3]), ("L_cop_y", [0.4, 0.5, 0.6]),
        ("R_cop_x", [0.7, 0.8, 0.9]), ("R_cop_y", [1.0, 1.1, 1.2]),
    ])


def test_native_segment_preserves_provider_lengths_frames_and_sample_times(tmp_path):
    _fixture(tmp_path)

    segment = load_native_segment(tmp_path, "1", "still", "1")

    assert (segment.subject, segment.activity, segment.segment) == ("1", "still", "1")
    assert segment.sync_status == "unverified_endpoint_origin"
    assert segment.pressure.values.shape == (3, 2, 31, 11)
    assert segment.markers.values.shape == (5, 26, 3)
    assert segment.foot_imu.values.shape == (4, 2, 6)
    np.testing.assert_array_equal(segment.frames["pressure"], [10, 12, 20])
    np.testing.assert_allclose(segment.pressure.time_s, [10 / 60, 12 / 60, 20 / 60])
    np.testing.assert_allclose(segment.markers.time_s, [20 / 100, 21 / 100, 22 / 100, 23 / 100, 24 / 100])
    np.testing.assert_allclose(segment.foot_imu.time_s, [40 / 100, 41 / 100, 42 / 100, 43 / 100])


def test_native_imu_keeps_acceleration_and_converts_gyro_to_radians_per_second(tmp_path):
    _fixture(tmp_path)

    foot_imu = load_native_segment(tmp_path, "1", "still", "1").foot_imu

    assert foot_imu.unit == "m/s^2;rad/s"
    assert foot_imu.values[0, 0, 0] == 11.0
    assert foot_imu.values[0, 1, 0] == 22.0
    assert foot_imu.values[0, 0, 3] == np.pi


def test_native_segment_keeps_nonfinite_samples_with_validity_masks(tmp_path):
    _fixture(tmp_path, pressure_nan=True, imu_nan=True)

    segment = load_native_segment(tmp_path, "1", "still", "1")

    assert np.isnan(segment.pressure.values[0, 0, 0, 0])
    assert not segment.pressure.valid[0, 0, 0, 0]
    assert np.isnan(segment.foot_imu.values[0, 1, 1])
    assert not segment.foot_imu.valid[0, 1, 1]


def test_native_segment_represents_blank_provider_cells_as_invalid_missing_values(tmp_path):
    _fixture(tmp_path, pressure_blank=True)

    pressure = load_native_segment(tmp_path, "1", "still", "1").pressure

    assert np.isnan(pressure.values[0, 0, 0, 0])
    assert not pressure.valid[0, 0, 0, 0]


@pytest.mark.parametrize(("subject", "expected"), [
    ("3", [22.0, 11.0]), ("9", [22.0, 11.0]), ("15", [22.0, 11.0]),
    ("16", [22.0, 11.0]), ("17", [22.0, 11.0]), ("22", [22.0, 11.0]),
    ("24", [22.0, 11.0]), ("1", [11.0, 22.0]),
])
def test_native_segment_repairs_only_known_left_right_imu_side_swaps(tmp_path, subject, expected):
    _fixture(tmp_path, subject=subject)

    foot_imu = load_native_segment(tmp_path, subject, "still", "1").foot_imu

    np.testing.assert_array_equal(foot_imu.values[:, :, 0], [expected] * 4)


@pytest.mark.parametrize(("pressure_count", "marker_count", "expected_hz"), [(3, 5, 60.0), (5, 5, 100.0)])
def test_native_pressure_rate_is_inferred_from_its_native_length(tmp_path, pressure_count, marker_count,
                                                                 expected_hz):
    _fixture(tmp_path, pressure_count=pressure_count, marker_count=marker_count)

    pressure = load_native_segment(tmp_path, "1", "still", "1").pressure

    assert pressure.nominal_hz == expected_hz
    assert pressure.time_s[2] == 20.0 / expected_hz


def test_native_segment_npz_roundtrip_declares_schema_and_stream_metadata(tmp_path):
    _fixture(tmp_path)
    segment = load_native_segment(tmp_path, "1", "still", "1")
    destination = tmp_path / "native.npz"

    write_native_segment(segment, destination)

    with np.load(destination, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "schema_version", "subject", "activity", "segment", "sync_status",
            "pressure_psi", "pressure_frame", "pressure_time_s", "pressure_valid",
            "pressure_time_basis", "pressure_nominal_hz", "pressure_group_delay_s",
            "markers_world_m", "marker_names", "mocap_frame", "mocap_time_s", "marker_valid",
            "mocap_time_basis", "mocap_nominal_hz", "mocap_group_delay_s",
            "foot_imu_si", "foot_imu_frame", "foot_imu_time_s", "foot_imu_valid", "foot_imu_units",
            "foot_imu_time_basis", "foot_imu_nominal_hz", "foot_imu_group_delay_s",
            "force_n", "force_frame", "force_time_s", "force_valid", "force_time_basis",
            "force_nominal_hz", "force_group_delay_s", "cop_m", "cop_frame", "cop_time_s", "cop_valid",
            "cop_time_basis", "cop_nominal_hz", "cop_group_delay_s",
        }
        assert archive["schema_version"].item() == "moveport-segment-v1"
        assert archive["subject"].item() == "1"
        assert archive["sync_status"].item() == "unverified_endpoint_origin"
        np.testing.assert_array_equal(archive["pressure_frame"], [10, 12, 20])
        np.testing.assert_allclose(archive["foot_imu_si"][:, 0, 3], np.pi)
        np.testing.assert_array_equal(archive["force_valid"], np.ones((3, 2), dtype=bool))
        np.testing.assert_array_equal(archive["marker_names"], MARKERS)
        np.testing.assert_array_equal(archive["foot_imu_units"],
                                      ["m/s^2", "m/s^2", "m/s^2", "rad/s", "rad/s", "rad/s"])
        assert all(archive[key].dtype.kind != "O" for key in archive.files)


def test_native_builder_cli_writes_one_named_segment_file(tmp_path):
    _fixture(tmp_path)
    output = tmp_path / "output"

    result = subprocess.run([sys.executable, "scripts/build_moveport_native.py", "--root", str(tmp_path),
                             "--out", str(output), "--subject", "1", "--activity", "still",
                             "--segment", "1"], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert (output / "1_still_1.npz").exists()


def test_native_builder_cli_rejects_unsafe_components_and_existing_output(tmp_path):
    _fixture(tmp_path)
    output = tmp_path / "output"
    command = [sys.executable, "scripts/build_moveport_native.py", "--root", str(tmp_path), "--out", str(output),
               "--subject", "1", "--activity", "still", "--segment", "1"]

    assert subprocess.run(command, capture_output=True, text=True).returncode == 0
    collision = subprocess.run(command, capture_output=True, text=True)
    unsafe = subprocess.run([*command[:-2], "--segment", "../1"], capture_output=True, text=True)

    assert collision.returncode != 0
    assert "exists" in collision.stderr
    assert unsafe.returncode != 0
    assert "single path component" in unsafe.stderr
