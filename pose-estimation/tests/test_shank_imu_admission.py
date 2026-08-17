import hashlib
import xml.etree.ElementTree as ET

import numpy as np
import pytest


def test_above_band_energy_share_separates_tones():
    from posesim.shank_imu.qc import above_band_energy_share

    fs = 100.0
    time = np.arange(1024) / fs
    low = np.sin(2.0 * np.pi * 2.0 * time)
    high = np.sin(2.0 * np.pi * 15.0 * time)
    signals = np.stack([low, high, low + high], axis=1)
    share = above_band_energy_share(signals, sample_rate_hz=fs, cutoff_hz=6.0)
    assert share[0] < 0.01
    assert share[1] > 0.99
    assert 0.4 < share[2] < 0.6
    offset = above_band_energy_share(signals + 5.0, sample_rate_hz=fs, cutoff_hz=6.0)
    assert np.allclose(offset, share, atol=1e-6)
    with pytest.raises(ValueError):
        above_band_energy_share(signals, sample_rate_hz=fs, cutoff_hz=60.0)


def test_verify_sha256_rejects_a_changed_model_file(tmp_path):
    from posesim.shank_imu.acceptance import verify_sha256

    model = tmp_path / "subject01.osim"
    model.write_bytes(b"pinned model bytes")
    expected = hashlib.sha256(b"different model bytes").hexdigest()

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_sha256(model, expected)


def test_marker_contract_rejects_a_station_on_the_wrong_body():
    from posesim.shank_imu.acceptance import validate_marker_contract

    contract = {
        "markers": [
            {"label": "R_LM", "body": "tibia_r", "required": True},
        ]
    }

    with pytest.raises(ValueError, match="R_LM.*tibia_r.*femur_r"):
        validate_marker_contract(contract, {"R_LM": "femur_r"})


def test_gait2392_contract_freezes_the_17_mandatory_station_labels():
    from posesim.shank_imu.acceptance import load_gait2392_contract

    contract = load_gait2392_contract()
    labels = {marker["label"] for marker in contract["markers"]}

    assert len(labels) == 17
    assert {"R_LM", "L_LM", "R_TTC", "L_TTC"} <= labels


def test_ftc_contract_uses_the_frozen_hip_child_origin_surrogate():
    from posesim.shank_imu.acceptance import load_gait2392_contract

    markers = {marker["label"]: marker for marker in load_gait2392_contract()["markers"]}

    assert markers["R_FTC"]["station_kind"] == "joint_child_origin"
    assert markers["R_FTC"]["joint"] == "hip_r"
    assert markers["R_FTC"]["location_m"] == [0.0, 0.0, 0.0]
    assert markers["L_FTC"]["station_kind"] == "joint_child_origin"
    assert markers["L_FTC"]["joint"] == "hip_l"


def test_template_station_queries_keep_marker_and_joint_sources_distinct():
    from posesim.shank_imu.acceptance import load_gait2392_contract, template_station_queries

    queries = template_station_queries(load_gait2392_contract())

    assert sum(query["station_kind"] == "marker" for query in queries) == 15
    assert {
        (query["label"], query["joint"])
        for query in queries if query["station_kind"] == "joint_child_origin"
    } == {("R_FTC", "hip_r"), ("L_FTC", "hip_l")}


def test_derivative_marker_specs_can_make_every_contract_station_movable():
    from posesim.shank_imu.acceptance import derivative_marker_specs, load_gait2392_contract

    specs = derivative_marker_specs(load_gait2392_contract(), fixed=False)

    assert len(specs) == 17
    assert {spec["label"] for spec in specs} == {
        "M_PSIS", "R_IAS", "L_IAS", "R_FTC", "R_FLE", "R_FME", "R_TTC",
        "R_LM", "R_CAL", "R_MH1", "L_FTC", "L_FLE", "L_FME", "L_TTC",
        "L_LM", "L_CAL", "L_MH1",
    }
    assert all(spec["fixed"] is False for spec in specs)


def test_derivative_marker_contract_rejects_a_changed_coordinate():
    from posesim.shank_imu.acceptance import (
        derivative_marker_specs,
        load_gait2392_contract,
        validate_derivative_marker_contract,
    )

    specs = derivative_marker_specs(load_gait2392_contract())
    resolved = {spec["label"]: spec["body"] for spec in specs}
    locations = {spec["label"]: spec["location_m"] for spec in specs}
    locations["R_LM"] = [0.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="R_LM.*coordinate"):
        validate_derivative_marker_contract(specs, resolved, locations)


def test_movable_derivative_contract_rejects_a_fixed_candidate_marker():
    from posesim.shank_imu.acceptance import (
        derivative_marker_specs,
        load_gait2392_contract,
        validate_movable_derivative_marker_contract,
    )

    specs = derivative_marker_specs(load_gait2392_contract(), fixed=False)
    resolved = {spec["label"]: spec["body"] for spec in specs}
    locations = {spec["label"]: spec["location_m"] for spec in specs}
    fixed = {spec["label"]: False for spec in specs}
    fixed["R_LM"] = True

    with pytest.raises(ValueError, match="R_LM.*movable"):
        validate_movable_derivative_marker_contract(specs, resolved, locations, fixed)


def test_scaled_derivative_contract_rejects_marker_relocation():
    from posesim.shank_imu.acceptance import (
        derivative_marker_specs,
        load_gait2392_contract,
        validate_scaled_derivative_marker_contract,
    )

    specs = derivative_marker_specs(load_gait2392_contract())
    resolved = {spec["label"]: spec["body"] for spec in specs}
    factors = {spec["body"]: 1.0 for spec in specs}
    locations = {spec["label"]: spec["location_m"] for spec in specs}
    locations["R_LM"] = [0.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="R_LM.*scaled coordinate"):
        validate_scaled_derivative_marker_contract(specs, resolved, locations, factors)


def test_model_admission_records_hash_tibias_and_station_parents(tmp_path):
    from posesim.shank_imu.acceptance import (
        load_gait2392_contract,
        validate_model_admission,
    )

    model = tmp_path / "subject01.osim"
    model.write_bytes(b"pinned test model")
    contract = load_gait2392_contract()
    contract["model_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
    resolved = {marker["label"]: marker["body"] for marker in contract["markers"]}
    locations = {marker["station"]: marker["location_m"] for marker in contract["markers"]}

    record = validate_model_admission(
        model, contract, bodies={"pelvis", "tibia_r", "tibia_l"}, resolved=resolved,
        locations=locations,
    )

    assert record["model_sha256"] == contract["model_sha256"]
    assert record["tibia_bodies"] == ["tibia_r", "tibia_l"]
    assert record["mandatory_markers"] == 17


def test_model_admission_rejects_changed_station_coordinate(tmp_path):
    from posesim.shank_imu.acceptance import load_gait2392_contract, validate_model_admission

    model = tmp_path / "subject01.osim"
    model.write_bytes(b"pinned test model")
    contract = load_gait2392_contract()
    contract["model_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
    resolved = {marker["label"]: marker["body"] for marker in contract["markers"]}
    locations = {marker["station"]: marker["location_m"] for marker in contract["markers"]}
    locations["R.Ankle.Lat"] = [0.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="R_LM.*coordinate"):
        validate_model_admission(
            model, contract, bodies={"pelvis", "tibia_r", "tibia_l"}, resolved=resolved,
            locations=locations,
        )


def test_marker_contract_rejects_a_station_with_changed_coordinates():
    from posesim.shank_imu.acceptance import validate_station_locations

    contract = {
        "markers": [
            {
                "label": "R_LM",
                "station": "R.Ankle.Lat",
                "location_m": [0.0, 0.1, 0.2],
                "required": True,
            },
        ]
    }

    with pytest.raises(ValueError, match="R_LM.*coordinate"):
        validate_station_locations(contract, {"R.Ankle.Lat": [0.0, 0.1, 0.201]})


def test_static_window_uses_earliest_equal_minimum():
    from posesim.shank_imu.scaling import select_static_window

    markers = np.zeros((205, 17, 3), dtype=float)

    assert select_static_window(markers, hz=100.0, width=200) == (0, 200, 0.0)


def test_marker_placement_setup_freezes_static_inputs_and_task_set():
    from posesim.shank_imu.scaling import marker_placement_setup

    xml = marker_placement_setup(
        model_file="template.osim", scales={
            "pelvis": 1.0, "femur_r": 1.0, "femur_l": 1.0,
            "tibia_r": 1.0, "tibia_l": 1.0, "foot_r": 1.0, "foot_l": 1.0,
        }, marker_file="still.trc", start_time_s=1.0, end_time_s=3.0,
        labels=["R_IAS", "L_IAS"], scaled_model_file="scaled.osim",
        placed_model_file="placed.osim", placed_motion_file="placed.mot",
    )
    root = ET.fromstring(xml)
    placer = root.find(".//MarkerPlacer")

    assert placer is not None
    assert placer.findtext("apply") == "true"
    assert placer.findtext("marker_file") == "still.trc"
    assert placer.findtext("time_range") == "1 3"
    assert [task.attrib["name"] for task in placer.findall(".//IKMarkerTask")] == [
        "R_IAS", "L_IAS"
    ]


def test_fold_qc_threshold_estimates_the_quantile_of_supplied_values():
    from posesim.shank_imu.scaling import fold_qc_threshold

    assert fold_qc_threshold([0.01, 0.02, 0.03], quantile=0.95) == pytest.approx(0.029)


@pytest.mark.parametrize("quantile", [None, [0.95], float("nan"), 0.0, 1.0])
def test_fold_qc_threshold_rejects_an_invalid_quantile(quantile):
    from posesim.shank_imu.scaling import fold_qc_threshold

    with pytest.raises(ValueError, match="quantile"):
        fold_qc_threshold([0.01, 0.02], quantile=quantile)


def test_outer_fold_qc_leaves_held_out_value_out_of_cutoff_estimation():
    from posesim.shank_imu.scaling import outer_fold_qc

    report = outer_fold_qc(
        {"1": 0.01, "2": 0.02, "3": 1.00}, held_out_subjects=["3"], quantile=0.95
    )

    assert report["development_subjects"] == ["1", "2"]
    assert report["held_out_subjects"] == ["3"]
    assert report["cutoff"] == pytest.approx(0.0195)
    assert report["held_out_pass"] == {"3": False}


def test_segment_fold_cutoffs_keep_held_out_trials_out_of_their_own_cutoff():
    from posesim.shank_imu.scaling import segment_fold_cutoffs

    trials = {"1/still/1": 0.010, "1/back/1": 0.012,
              "2/still/1": 0.014, "2/back/1": 0.016,
              "3/still/1": 0.900, "3/back/1": 1.000}
    cutoffs = segment_fold_cutoffs(trials, [["3"], ["1"]], quantile=0.95)

    # fold 0 holds out subject 3, so its two huge trials cannot raise the cutoff
    assert cutoffs["0"]["development_trials"] == 4
    assert cutoffs["0"]["cutoff_m"] == pytest.approx(0.01570)
    assert cutoffs["0"]["held_out_trials_excluded"] == ["3/back/1", "3/still/1"]
    # fold 1 holds out subject 1, and subject 3 now sets its cutoff instead
    assert cutoffs["1"]["development_trials"] == 4
    assert cutoffs["1"]["cutoff_m"] > 0.5
    assert cutoffs["1"]["held_out_trials_excluded"] == []


def test_contract_marker_selection_uses_frozen_labels_and_xyz_order():
    from posesim.shank_imu.acceptance import load_gait2392_contract
    from posesim.shank_imu.scaling import select_contract_markers

    contract = load_gait2392_contract()
    labels = []
    rows = []
    for index, marker in enumerate(contract["markers"]):
        for axis, offset in zip(("x", "y", "z"), (0.0, 0.1, 0.2)):
            labels.append(f"{marker['label'].lower()}_{axis}")
            rows.append([index + offset, index + offset + 1.0])

    selected = select_contract_markers(np.array(rows), labels, contract)

    assert selected.shape == (2, 17, 3)
    assert np.allclose(selected[0, 0], [0.0, 0.1, 0.2])
    assert np.allclose(selected[1, -1], [17.0, 17.1, 17.2])


def test_manual_scale_setup_expands_seven_multipliers_to_twelve_segments():
    from posesim.shank_imu.scaling import manual_scale_setup

    scale_names = ("pelvis", "femur_r", "femur_l", "tibia_r", "tibia_l", "foot_r", "foot_l")
    xml_text = manual_scale_setup(
        model_file="moveport_17marker_template.osim",
        scales={name: 1.0 for name in scale_names},
        mass_kg=70.0,
        preserve_mass_distribution=False,
        output_model_file="scaled_only.osim",
        output_scale_file="applied_scales.xml",
    )
    root = ET.fromstring(xml_text)
    entries = root.findall("./ScaleTool/ModelScaler/ScaleSet/objects/Scale")

    assert root.findtext("./ScaleTool/ModelScaler/scaling_order") == "manualScale"
    assert {entry.findtext("segment") for entry in entries} == {
        "pelvis", "torso", "femur_r", "femur_l", "tibia_r", "talus_r",
        "tibia_l", "talus_l", "calcn_r", "toes_r", "calcn_l", "toes_l",
    }
    assert len(entries) == 12
    assert all(entry.findtext("scales") == "1 1 1" for entry in entries)


def test_manual_scale_setup_can_explicitly_retain_unknown_subject_mass():
    from posesim.shank_imu.scaling import manual_scale_setup

    xml_text = manual_scale_setup(
        model_file="moveport_17marker_template.osim",
        scales={name: 1.0 for name in ("pelvis", "femur_r", "femur_l", "tibia_r", "tibia_l", "foot_r", "foot_l")},
        mass_kg=-1.0,
        preserve_mass_distribution=True,
        output_model_file="scaled_only.osim",
        output_scale_file="applied_scales.xml",
    )
    root = ET.fromstring(xml_text)

    assert root.findtext("./ScaleTool/mass") == "-1"
    assert root.findtext("./ScaleTool/height") == "-1"
    assert root.findtext("./ScaleTool/age") == "-1"
    assert root.findtext("./ScaleTool/ModelScaler/preserve_mass_distribution") == "true"
    assert root.find("./ScaleTool/ModelScaler/MeasurementSet") is None


def test_body_scale_factors_expand_the_manual_scale_inheritance():
    from posesim.shank_imu.scaling import body_scale_factors

    factors = body_scale_factors({
        "pelvis": 1.1, "femur_r": 1.2, "femur_l": 1.3, "tibia_r": 1.4,
        "tibia_l": 1.5, "foot_r": 1.6, "foot_l": 1.7,
    })

    assert factors["pelvis"] == 1.1
    assert factors["torso"] == 1.1
    assert factors["talus_r"] == 1.4
    assert factors["toes_l"] == 1.7


def test_scale_multipliers_reject_zero_template_distance():
    from posesim.shank_imu.scaling import scale_multipliers

    points = {
        "R_IAS": np.array([0.0, 0.0, 0.0]),
        "L_IAS": np.array([1.0, 0.0, 0.0]),
    }
    stations = {
        "R_IAS": np.zeros(3),
        "L_IAS": np.zeros(3),
    }

    with pytest.raises(ValueError, match="pelvis.*zero template length"):
        scale_multipliers(points, stations)


def test_stationary_sensor_has_zero_kinematic_acceleration_and_upward_force():
    from posesim.shank_imu.signal import propagate_offset_acceleration, specific_force

    acceleration = propagate_offset_acceleration(
        body_accel_w=np.zeros(3),
        angular_velocity_w=np.zeros(3),
        angular_accel_w=np.zeros(3),
        offset_w=np.array([0.03, 0.0, 0.0]),
    )

    assert np.allclose(acceleration, 0.0)
    assert np.allclose(
        specific_force(acceleration, np.eye(3), np.array([0.0, 0.0, -9.80665])),
        [0.0, 0.0, 9.80665],
    )


def test_virtual_frame_maps_each_imu_channel_into_its_declared_operational_frame():
    from posesim.shank_imu.frames import VirtualFrame, map_bilateral_imu_measurements

    left = VirtualFrame(
        tibia_to_anatomical=np.eye(3),
        sensor_to_tibia=np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    right = VirtualFrame(tibia_to_anatomical=np.eye(3), sensor_to_tibia=np.eye(3))
    imu = np.array([[[1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 3.0, 0.0]]])

    mapped = map_bilateral_imu_measurements(imu, left=left, right=right)

    assert np.allclose(mapped, [[[0.0, 1.0, 0.0, 0.0, 2.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0, 3.0, 0.0]]])


def test_virtual_frame_fails_closed_for_an_improper_installation_rotation():
    from posesim.shank_imu.frames import VirtualFrame

    with pytest.raises(ValueError, match="sensor_to_tibia.*proper rotation"):
        VirtualFrame(tibia_to_anatomical=np.eye(3), sensor_to_tibia=np.diag([1.0, 1.0, -1.0]))


def test_fixed_lag_fir_declares_terminal_samples_unavailable():
    from posesim.shank_imu.signal import fixed_lag_fir

    signal = fixed_lag_fir(
        np.arange(8.0), np.arange(8.0) / 100.0, coefficients=np.array([1.0]),
        lag_samples=2,
    )

    assert np.allclose(signal.values[:6], np.arange(6.0))
    assert signal.valid.tolist() == [True, True, True, True, True, True, False, False]
    assert np.allclose(signal.available_time_s[:6], signal.physical_time_s[:6] + 0.02)


def test_fixed_lag_fir_prefix_does_not_depend_on_future_values():
    from posesim.shank_imu.signal import fixed_lag_fir

    time = np.arange(10.0) / 100.0
    original = fixed_lag_fir(np.arange(10.0), time, coefficients=np.array([0.25, 0.75]), lag_samples=2)
    perturbed_values = np.arange(10.0)
    perturbed_values[6:] += 1000.0
    perturbed = fixed_lag_fir(perturbed_values, time, coefficients=np.array([0.25, 0.75]), lag_samples=2)

    assert np.allclose(original.values[:6], perturbed.values[:6])


def test_fixed_lag_window_fir_declares_physical_and_availability_times():
    from posesim.shank_imu.signal import fixed_lag_window_fir

    signal = fixed_lag_window_fir(
        np.arange(8.0), np.arange(8.0) / 100.0,
        coefficients=np.array([1.0, 0.0, 0.0]), history_samples=1, lookahead_samples=1,
    )

    assert signal.valid.tolist() == [False, True, True, True, True, True, True, False]
    assert signal.values[1] == pytest.approx(0.0)
    assert signal.available_time_s[1] == pytest.approx(0.02)
    assert signal.available_time_s[1] - signal.physical_time_s[1] == pytest.approx(0.01)


def test_fixed_lag_window_fir_output_available_before_a_change_is_invariant():
    from posesim.shank_imu.signal import fixed_lag_window_fir

    time = np.arange(10.0) / 100.0
    original = fixed_lag_window_fir(
        np.arange(10.0), time, coefficients=np.array([0.25, 0.5, 0.25]),
        history_samples=1, lookahead_samples=1,
    )
    changed = np.arange(10.0)
    changed[5:] += 1000.0
    perturbed = fixed_lag_window_fir(
        changed, time, coefficients=np.array([0.25, 0.5, 0.25]),
        history_samples=1, lookahead_samples=1,
    )

    available_before_change = original.available_time_s < time[5]
    assert np.allclose(original.values[available_before_change], perturbed.values[available_before_change])


def test_ik_error_summary_reads_only_the_declared_rms_column(tmp_path):
    from posesim.shank_imu.qc import ik_error_summary

    report = tmp_path / "_ik_marker_errors.sto"
    report.write_text(
        "Model Marker Errors from IK\n"
        "version=1\n"
        "endheader\n"
        "time\ttotal_squared_error\tmarker_error_RMS\tmarker_error_max\n"
        "0.00\t0.01\t0.10\t0.20\n"
        "0.01\t0.04\t0.20\t0.50\n",
        encoding="utf-8",
    )

    summary = ik_error_summary(report)

    assert summary == {
        "frames": 2,
        "mean_rms_m": pytest.approx(0.15),
        "p95_rms_m": pytest.approx(0.195),
        "max_marker_error_m": pytest.approx(0.5),
    }


def test_marker_pair_summary_reports_a_transient_length_excursion():
    from posesim.shank_imu.qc import marker_pair_summary

    markers = np.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
    ])

    summary = marker_pair_summary(markers, ["R_FLE", "R_FME"], "R_FLE", "R_FME")

    assert summary == {
        "frames": 3,
        "median_length_m": pytest.approx(1.0),
        "max_length_m": pytest.approx(3.0),
        "max_relative_deviation": pytest.approx(2.0),
    }


def test_marker_pair_reports_preserves_the_declared_pair_order():
    from posesim.shank_imu.qc import marker_pair_reports

    markers = np.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
    ])

    reports = marker_pair_reports(
        markers,
        ["R_FLE", "R_FME", "R_LM"],
        [("knee_width_r", "R_FLE", "R_FME"), ("shank_span_r", "R_FLE", "R_LM")],
    )

    assert [report["name"] for report in reports] == ["knee_width_r", "shank_span_r"]
    assert reports[0]["median_length_m"] == pytest.approx(1.0)
    assert reports[1]["max_length_m"] == pytest.approx(4.0)


def test_marker_pair_validity_marks_only_the_excursion_frame():
    from posesim.shank_imu.qc import marker_pair_validity

    markers = np.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    ])

    valid = marker_pair_validity(
        markers, ["R_FLE", "R_FME"], [("knee_width_r", "R_FLE", "R_FME")],
        max_relative_deviation=0.3,
    )

    assert valid.tolist() == [True, True, False]


def test_41_tap_kaiser_candidate_meets_its_stated_frequency_bounds():
    from posesim.shank_imu.signal import fir_response_bounds, kaiser_lowpass

    coefficients = kaiser_lowpass(
        taps=41, cutoff_hz=25.0, sample_rate_hz=100.0, beta=5.65326
    )
    bounds = fir_response_bounds(
        coefficients, sample_rate_hz=100.0, passband_hz=20.0, stopband_hz=30.0
    )

    assert bounds["min_passband_gain"] >= 0.99
    assert bounds["max_stopband_gain"] <= 0.001
