import numpy as np
import pytest

from posesim.data.timed import TimedArray


def test_timed_array_keeps_missing_values_and_marks_them_invalid():
    values = np.array([[1.0, np.nan], [2.0, 3.0]])
    valid = np.array([[True, False], [True, True]])

    stream = TimedArray(values, np.array([0.0, 0.01]), valid, "m", "provider_frame", 100.0)

    assert np.isnan(stream.values[0, 1])
    assert not stream.valid[0, 1]
    assert not stream.values.flags.writeable
    with pytest.raises(ValueError, match="strictly increasing"):
        TimedArray(np.ones((2, 1)), np.array([0.01, 0.01]), np.ones((2, 1), dtype=bool),
                   "m", "provider_frame", 100.0)


def test_timed_array_copies_scalar_numpy_metadata_as_python_floats():
    nominal_hz = np.array(100.0)
    group_delay_s = np.array(0.25)

    stream = TimedArray(np.ones((2, 1)), np.array([0.0, 0.01]), np.ones((2, 1), dtype=bool),
                        "m", "provider_frame", nominal_hz, group_delay_s)
    nominal_hz[...] = 1.0
    group_delay_s[...] = 9.0

    assert type(stream.nominal_hz) is float
    assert type(stream.group_delay_s) is float
    assert stream.nominal_hz == 100.0
    assert stream.group_delay_s == 0.25


@pytest.mark.parametrize(("unit", "time_basis"), [(1, "provider_frame"), ("m", object())])
def test_timed_array_rejects_non_string_metadata(unit, time_basis):
    with pytest.raises(ValueError, match="Python string"):
        TimedArray(np.ones((2, 1)), np.array([0.0, 0.01]), np.ones((2, 1), dtype=bool),
                   unit, time_basis, 100.0)


@pytest.mark.parametrize("nominal_hz", [np.array([100.0]), "100 Hz"])
def test_timed_array_rejects_non_scalar_or_non_numeric_nominal_rate(nominal_hz):
    with pytest.raises(ValueError, match="finite scalar"):
        TimedArray(np.ones((2, 1)), np.array([0.0, 0.01]), np.ones((2, 1), dtype=bool),
                   "m", "provider_frame", nominal_hz)
