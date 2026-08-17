import numpy as np
import pytest

from posesim.data.timed import TimedArray
from posesim.signal.resample import resample_causal


def _stream(values, hz=100.0, valid=None):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if valid is None:
        valid = np.isfinite(values)
    return TimedArray(values, np.arange(len(values)) / hz, valid, "unit", "test_clock", hz)


def test_causal_resampler_prefix_ignores_future_input():
    time_s = np.arange(61) / 60.0
    first = np.sin(2 * np.pi * 2 * np.arange(101) / 100.0)
    changed = first.copy()
    changed[71:] += 1000.0

    a = resample_causal(_stream(first), time_s, cutoff_hz=24.0)
    b = resample_causal(_stream(changed), time_s, cutoff_hz=24.0)

    prefix = time_s < 0.71
    np.testing.assert_allclose(a.values[prefix], b.values[prefix], equal_nan=True)
    np.testing.assert_array_equal(a.valid[prefix], b.valid[prefix])


def test_downsampling_attenuates_energy_above_target_nyquist():
    source_time = np.arange(501) / 100.0
    target_time = np.arange(301) / 60.0
    low = resample_causal(_stream(np.sin(2 * np.pi * 5 * source_time)), target_time, 24.0)
    high = resample_causal(_stream(np.sin(2 * np.pi * 40 * source_time)), target_time, 24.0)
    valid = low.valid[:, 0] & high.valid[:, 0]

    low_rms = np.sqrt(np.mean(low.values[valid, 0] ** 2))
    high_rms = np.sqrt(np.mean(high.values[valid, 0] ** 2))

    assert high_rms < 0.2 * low_rms


@pytest.mark.parametrize("source_hz", [60.0, 100.0])
def test_declared_cutoff_has_half_amplitude_at_each_native_rate(source_hz):
    source_time = np.arange(int(8 * source_hz) + 1) / source_hz
    target_time = np.arange(8 * 60 + 1) / 60.0
    out = resample_causal(_stream(np.sin(2 * np.pi * 24 * source_time), hz=source_hz),
                          target_time, 24.0)
    selected = out.valid[:, 0] & (out.time_s >= 1.0)
    phase_time = out.time_s[selected] - out.group_delay_s
    basis = np.stack([np.sin(2 * np.pi * 24 * phase_time),
                      np.cos(2 * np.pi * 24 * phase_time)], axis=1)
    coefficient = np.linalg.lstsq(basis, out.values[selected, 0], rcond=None)[0]
    gain = np.linalg.norm(coefficient)

    assert gain == pytest.approx(0.5, abs=0.01)


def test_every_post_warmup_output_is_valid_for_contiguous_valid_input():
    out = resample_causal(_stream(np.ones(101)), np.arange(61) / 60.0, cutoff_hz=24.0)

    assert out.valid[out.time_s >= 0.2, 0].all()


def test_causal_filter_declares_delay_and_invalidates_warmup_and_edges():
    target_time = np.arange(62) / 60.0
    out = resample_causal(_stream(np.ones(101)), target_time, cutoff_hz=24.0)

    assert out.group_delay_s == 0.1
    assert not out.valid[0, 0]
    assert out.valid[-2, 0]
    assert not out.valid[-1, 0]
    assert np.isnan(out.values[0, 0])
    assert np.isnan(out.values[-1, 0])


def test_missing_source_samples_invalidate_every_dependent_filtered_value():
    values = np.ones((101, 2))
    valid = np.ones_like(values, dtype=bool)
    valid[30, 1] = False
    values[30, 1] = np.nan

    out = resample_causal(_stream(values, valid=valid), np.arange(61) / 60.0, cutoff_hz=24.0)

    affected = (out.time_s >= 0.30) & (out.time_s <= 0.50)
    assert out.valid[affected, 0].all()
    assert not out.valid[affected, 1].all()
    assert np.isnan(out.values[affected, 1]).any()


def test_equal_rate_sampling_uses_the_same_physical_delay_and_does_not_endpoint_fill():
    source = _stream(np.arange(31.0), hz=60.0)
    out = resample_causal(source, np.arange(32) / 60.0, cutoff_hz=24.0)

    assert out.group_delay_s == 0.1
    assert not out.valid[:12, 0].any()
    assert out.valid[12:31, 0].all()
    assert not out.valid[31, 0]
    assert np.isnan(out.values[31, 0])


def test_impulse_alignment_has_the_same_output_frame_at_60_and_100_hz():
    target_time = np.arange(61) / 60.0
    outputs = []
    for hz in (60.0, 100.0):
        source_time = np.arange(int(hz) + 1) / hz
        impulse = np.zeros_like(source_time)
        impulse[np.argmin(np.abs(source_time - 0.5))] = 1.0
        outputs.append(resample_causal(_stream(impulse, hz=hz), target_time, 24.0))

    peaks = [np.nanargmax(out.values[:, 0]) for out in outputs]
    assert peaks == [36, 36]
    assert outputs[0].group_delay_s == outputs[1].group_delay_s == 0.1


def test_resampler_rejects_gapped_source_time_and_invalid_cutoff():
    values = np.ones((5, 1))
    gapped = TimedArray(values, np.array([0.0, 0.01, 0.02, 0.04, 0.05]),
                        np.ones_like(values, dtype=bool), "unit", "clock", 100.0)
    with pytest.raises(ValueError, match="uniform"):
        resample_causal(gapped, np.arange(4) / 60.0, 24.0)
    with pytest.raises(ValueError, match="Nyquist"):
        resample_causal(_stream(np.ones(101)), np.arange(61) / 60.0, 30.0)
