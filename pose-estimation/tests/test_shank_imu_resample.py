import numpy as np


def test_antialias_contract_loads_and_verifies_frozen_coefficients():
    import pytest

    from posesim.shank_imu.signal import load_antialias_contract

    contract = load_antialias_contract()
    assert contract.sample_rate_hz == 100.0
    assert len(contract.coefficients) == 41
    assert contract.group_delay_s == pytest.approx(0.2)
    assert contract.coefficients.sum() == pytest.approx(1.0)
    assert np.allclose(contract.coefficients, contract.coefficients[::-1])
import pytest


def _uniform_signal(values, *, rate_hz=100.0, delay_s=0.0, valid=None):
    from posesim.shank_imu.signal import FixedLagSignal

    values = np.asarray(values, dtype=float)
    time = np.arange(len(values)) / rate_hz
    if valid is None:
        valid = np.ones(len(values), dtype=bool)
    return FixedLagSignal(values, time, time + delay_s, np.asarray(valid, dtype=bool))


def test_interpolation_matches_an_analytic_sinusoid_on_the_60hz_grid():
    from posesim.shank_imu.signal import interpolate_to_grid

    time = np.arange(200) / 100.0
    source = _uniform_signal(np.sin(2.0 * np.pi * 5.0 * time))
    target_time = np.arange(0.0, 1.9, 1.0 / 60.0)

    output = interpolate_to_grid(source, target_time)

    reference = np.sin(2.0 * np.pi * 5.0 * target_time)
    assert output.valid.all()
    assert np.allclose(output.physical_time_s, target_time)
    assert np.max(np.abs(output.values - reference)) < 0.02


def test_interpolated_availability_is_the_later_source_sample():
    from posesim.shank_imu.signal import interpolate_to_grid

    source = _uniform_signal(np.zeros(100), delay_s=0.2)
    target_time = np.arange(0.0, 0.9, 1.0 / 60.0)

    output = interpolate_to_grid(source, target_time)

    later_index = np.searchsorted(source.physical_time_s, target_time, side="left")
    assert np.allclose(output.available_time_s, source.available_time_s[later_index])
    delay = output.available_time_s - output.physical_time_s
    assert delay.min() >= 0.2 - 1e-12
    assert delay.max() < 0.2 + 0.01 + 1e-12


def test_a_grid_point_on_a_source_sample_uses_that_sample_exactly():
    from posesim.shank_imu.signal import interpolate_to_grid

    values = np.arange(50, dtype=float) ** 2
    source = _uniform_signal(values, delay_s=0.2)

    output = interpolate_to_grid(source, source.physical_time_s[10:11])

    assert output.values[0] == values[10]
    assert output.available_time_s[0] == source.available_time_s[10]


def test_targets_outside_or_next_to_invalid_sources_are_invalid():
    from posesim.shank_imu.signal import interpolate_to_grid

    valid = np.ones(100, dtype=bool)
    valid[:14] = False
    source = _uniform_signal(np.ones(100), valid=valid)
    target_time = np.array([-0.01, 0.05, 0.131, 0.145, 0.5, 0.99, 1.5])

    output = interpolate_to_grid(source, target_time)

    assert not output.valid[0]
    assert not output.valid[1]
    assert not output.valid[2]
    assert output.valid[3]
    assert output.valid[4]
    assert output.valid[5]
    assert not output.valid[6]
    assert (output.values[~output.valid] == 0.0).all()


def test_prefix_outputs_are_unchanged_by_later_source_samples():
    from posesim.shank_imu.signal import interpolate_to_grid

    rng = np.random.default_rng(7)
    values = rng.standard_normal(120)
    full = _uniform_signal(values, delay_s=0.2)
    prefix = _uniform_signal(values[:80], delay_s=0.2)
    target_time = np.arange(0.0, 1.19, 1.0 / 60.0)

    full_output = interpolate_to_grid(full, target_time)
    prefix_output = interpolate_to_grid(prefix, target_time)

    kept = prefix_output.valid
    assert kept.any()
    assert np.array_equal(full_output.values[kept], prefix_output.values[kept])
    assert np.array_equal(
        full_output.available_time_s[kept], prefix_output.available_time_s[kept]
    )


def test_interpolation_keeps_trailing_channel_axes():
    from posesim.shank_imu.signal import interpolate_to_grid

    values = np.stack([np.arange(100.0), 2.0 * np.arange(100.0)], axis=1)
    source = _uniform_signal(values)
    target_time = np.array([0.105, 0.505])

    output = interpolate_to_grid(source, target_time)

    assert output.values.shape == (2, 2)
    assert np.allclose(output.values[:, 1], 2.0 * output.values[:, 0])


def test_interpolation_rejects_a_non_increasing_target_grid():
    from posesim.shank_imu.signal import interpolate_to_grid

    source = _uniform_signal(np.zeros(10))

    with pytest.raises(ValueError, match="strictly increasing"):
        interpolate_to_grid(source, np.array([0.0, 0.0, 0.01]))


def test_the_antialias_resample_path_removes_a_35hz_component():
    from posesim.shank_imu.signal import (
        fixed_lag_window_fir,
        interpolate_to_grid,
        kaiser_lowpass,
    )

    time = np.arange(600) / 100.0
    clean = np.sin(2.0 * np.pi * 5.0 * time)
    dirty = clean + np.sin(2.0 * np.pi * 35.0 * time)
    coefficients = kaiser_lowpass(
        taps=41, cutoff_hz=25.0, sample_rate_hz=100.0, beta=5.65326
    )
    target_time = np.arange(0.0, 5.99, 1.0 / 60.0)

    outputs = []
    for series in (clean, dirty):
        filtered = fixed_lag_window_fir(
            series, time, coefficients=coefficients,
            history_samples=20, lookahead_samples=20,
        )
        outputs.append(interpolate_to_grid(filtered, target_time))

    kept = outputs[0].valid & outputs[1].valid
    assert kept.sum() > 300
    residual = np.abs(outputs[1].values[kept] - outputs[0].values[kept])
    assert residual.max() < 2e-3
    delay = outputs[1].available_time_s[kept] - outputs[1].physical_time_s[kept]
    assert delay.max() < 0.2 + 0.01 + 1e-12


def _rotation_z(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_a_stationary_imu_reads_minus_gravity_through_both_rotations():
    from posesim.shank_imu.signal import imu_measurement

    rotation_wt = _rotation_z(0.3)
    rotation_ts = _rotation_z(-1.1)
    gravity = np.array([0.0, 0.0, -9.80665])

    force, gyro = imu_measurement(
        rotation_wt=rotation_wt,
        body_accel_w=np.zeros(3),
        angular_velocity_w=np.zeros(3),
        angular_accel_w=np.zeros(3),
        offset_t=np.array([0.02, -0.35, 0.04]),
        rotation_ts=rotation_ts,
        gravity_w=gravity,
    )

    expected = -(rotation_wt @ rotation_ts).T @ gravity
    assert np.allclose(force, expected)
    assert np.allclose(gyro, 0.0)


def test_constant_spin_produces_centripetal_force_at_the_lever_arm():
    from posesim.shank_imu.signal import imu_measurement

    omega = np.array([0.0, 0.0, 3.0])
    offset_t = np.array([0.2, 0.0, 0.0])

    force, gyro = imu_measurement(
        rotation_wt=np.eye(3),
        body_accel_w=np.zeros(3),
        angular_velocity_w=omega,
        angular_accel_w=np.zeros(3),
        offset_t=offset_t,
        rotation_ts=np.eye(3),
        gravity_w=np.zeros(3),
    )

    assert np.allclose(force, [-9.0 * 0.2, 0.0, 0.0])
    assert np.allclose(gyro, omega)


def test_angular_acceleration_produces_tangential_force_at_the_lever_arm():
    from posesim.shank_imu.signal import imu_measurement

    force, gyro = imu_measurement(
        rotation_wt=np.eye(3),
        body_accel_w=np.zeros(3),
        angular_velocity_w=np.zeros(3),
        angular_accel_w=np.array([0.0, 0.0, 5.0]),
        offset_t=np.array([0.2, 0.0, 0.0]),
        rotation_ts=np.eye(3),
        gravity_w=np.zeros(3),
    )

    assert np.allclose(force, [0.0, 1.0, 0.0])
    assert np.allclose(gyro, 0.0)


def test_the_lever_arm_rotates_with_the_body_frame():
    from posesim.shank_imu.signal import imu_measurement

    rotation_wt = _rotation_z(np.pi / 2.0)
    omega = np.array([0.0, 0.0, 2.0])

    force, _ = imu_measurement(
        rotation_wt=rotation_wt,
        body_accel_w=np.zeros(3),
        angular_velocity_w=omega,
        angular_accel_w=np.zeros(3),
        offset_t=np.array([0.1, 0.0, 0.0]),
        rotation_ts=np.eye(3),
        gravity_w=np.zeros(3),
    )

    world_force = rotation_wt @ force
    assert np.allclose(world_force, [0.0, -0.4, 0.0], atol=1e-12)


def test_imu_measurement_rejects_an_improper_rotation():
    from posesim.shank_imu.signal import imu_measurement

    reflection = np.diag([1.0, 1.0, -1.0])

    with pytest.raises(ValueError, match="proper rotation"):
        imu_measurement(
            rotation_wt=reflection,
            body_accel_w=np.zeros(3),
            angular_velocity_w=np.zeros(3),
            angular_accel_w=np.zeros(3),
            offset_t=np.zeros(3),
            rotation_ts=np.eye(3),
            gravity_w=np.zeros(3),
        )


def test_the_anatomical_frame_contract_records_identity_maps_backed_by_landmark_signs():
    from posesim.shank_imu.frames import load_anatomical_frame_contract

    contract = load_anatomical_frame_contract()

    for side in ("left", "right"):
        assert np.array_equal(contract.tibia_to_anatomical[side], np.eye(3))
    evidence = contract.evidence
    assert evidence["ankle_offset_in_tibia_m"]["left"][1] < 0.0
    assert evidence["ankle_offset_in_tibia_m"]["right"][1] < 0.0
    assert evidence["lm_station_m"]["right"][2] > 0.0
    assert evidence["lm_station_m"]["left"][2] < 0.0
    assert evidence["toe_minus_heel_ground_x_m"]["left"] > 0.0
    assert evidence["toe_minus_heel_ground_x_m"]["right"] > 0.0


def test_the_candidate_installation_mirrors_one_board_across_the_lateral_surfaces():
    from posesim.shank_imu.frames import load_anatomical_frame_contract

    contract = load_anatomical_frame_contract()

    assert contract.installation["status"] == "project_defined_candidate"
    assert np.array_equal(contract.virtual_frame("right").sensor_to_anatomical, np.eye(3))
    assert np.array_equal(
        contract.virtual_frame("left").sensor_to_anatomical, np.diag([-1.0, 1.0, -1.0])
    )
    right, left = contract.lever_arm("right"), contract.lever_arm("left")
    assert np.array_equal(right * [1.0, 1.0, -1.0], left)
    assert right[1] > -0.425065
    assert right[2] > 0.0


def test_both_sides_report_identical_d_frame_gravity_and_axial_rate():
    from posesim.shank_imu.frames import load_anatomical_frame_contract
    from posesim.shank_imu.signal import imu_measurement

    contract = load_anatomical_frame_contract()
    gravity = np.array([0.0, -9.80665, 0.0])
    axial_rate = np.array([0.0, 0.0, 1.7])
    readings = {}
    for side in ("right", "left"):
        frame = contract.virtual_frame(side)
        force, gyro = imu_measurement(
            rotation_wt=np.eye(3),
            body_accel_w=np.zeros(3),
            angular_velocity_w=axial_rate,
            angular_accel_w=np.zeros(3),
            offset_t=contract.lever_arm(side),
            rotation_ts=frame.sensor_to_tibia,
            gravity_w=gravity,
        )
        readings[side] = (frame.sensor_to_anatomical @ force, frame.sensor_to_anatomical @ gyro, gyro)

    lever = contract.lever_arm("right")
    centripetal = np.cross(axial_rate, np.cross(axial_rate, lever))
    assert np.allclose(readings["right"][0], centripetal - gravity)
    assert np.allclose(readings["right"][0], readings["left"][0])
    assert np.allclose(readings["right"][1], axial_rate)
    assert np.allclose(readings["right"][1], readings["left"][1])
    assert np.allclose(readings["left"][2], np.diag([-1.0, 1.0, -1.0]) @ axial_rate)


def test_a_anatomical_frame_contract_without_an_installation_fails_closed(tmp_path):
    import json

    from posesim.shank_imu.frames import load_anatomical_frame_contract

    record = {
        "schema": "shank-imu-anatomical-frame-contract-v1",
        "tibia_to_anatomical": {
            "left": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "right": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
        "evidence": {},
        "installation": None,
    }
    path = tmp_path / "anatomical_frame.json"
    path.write_text(json.dumps(record))
    contract = load_anatomical_frame_contract(path)

    with pytest.raises(ValueError, match="installation"):
        contract.virtual_frame("right")
    with pytest.raises(ValueError, match="installation"):
        contract.lever_arm("right")


def test_a_anatomical_frame_contract_with_an_improper_map_is_rejected(tmp_path):
    import json

    from posesim.shank_imu.frames import load_anatomical_frame_contract

    broken = {
        "schema": "shank-imu-anatomical-frame-contract-v1",
        "operational_axes": {"x": "anterior", "y": "proximal", "z": "subject_right"},
        "tibia_to_anatomical": {
            "left": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
            "right": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
        "evidence": {},
        "installation": None,
    }
    path = tmp_path / "anatomical_frame.json"
    path.write_text(json.dumps(broken))

    with pytest.raises(ValueError, match="proper rotation"):
        load_anatomical_frame_contract(path)
