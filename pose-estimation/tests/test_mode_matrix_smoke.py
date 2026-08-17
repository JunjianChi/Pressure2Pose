"""Every runner mode combination the queued experiments use, run end to end.

Modes were tested one at a time before; the inner-selection and virtual-IMU
flags crashed only in combination, after a night of GPU time. This grid runs
each queued configuration for two epochs on a tiny cache so a composition
failure costs seconds instead of a night.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from posesim.data.mpdataset import aligned_cache_payload, write_aligned_cache
from tests.test_train_moveport import _shank_imu_cache_for, _segment

_ROOT = Path(__file__).resolve().parents[1]

# (name, extra runner arguments) — one entry per queued experiment variant and mode.
CONFIGURATIONS = [
    ("ponly_dense_inner", ["--encoder", "dense", "--no-imu", "--inner-fold", "0"]),
    ("ponly_moments_inner", ["--encoder", "moments", "--no-imu", "--inner-fold", "1"]),
    ("ponly_dense_steps", ["--encoder", "dense", "--no-imu", "--steps", "4"]),
    ("shank_imu_dense_inner", ["--encoder", "dense", "--shank-imu", "--inner-fold", "0"]),
    ("shank_imu_moments_inner", ["--encoder", "moments", "--shank-imu", "--inner-fold", "2"]),
    ("shank_imu_only_inner", ["--encoder", "none", "--shank-imu", "--inner-fold", "1"]),
    ("shank_imu_dense_steps", ["--encoder", "dense", "--shank-imu", "--steps", "4"]),
    ("shank_imu_only_steps", ["--encoder", "none", "--shank-imu", "--steps", "4"]),
    ("e3_block2_ponly", ["--encoder", "dense", "--no-imu", "--block", "2",
                         "--inner-fold", "0"]),
    ("e3_block8_shank_imu", ["--encoder", "dense", "--shank-imu", "--block", "8",
                        "--inner-fold", "0"]),
    ("e1_natural_width", ["--encoder", "moments", "--shank-imu", "--moment-hidden", "128",
                          "--inner-fold", "0"]),
    ("e1_natural_ponly", ["--encoder", "moments", "--no-imu", "--moment-hidden", "128",
                          "--inner-fold", "1"]),
    ("e3_phase_shift", ["--encoder", "dense", "--no-imu", "--block", "4",
                        "--block-origin", "2", "--inner-fold", "0"]),
    ("e0_extra_seed", ["--encoder", "dense", "--shank-imu", "--seed", "4", "--steps", "4"]),
    ("ponly_moments_steps", ["--encoder", "moments", "--no-imu", "--steps", "4"]),
    ("shank_imu_moments_steps", ["--encoder", "moments", "--shank-imu", "--steps", "4"]),
    ("e3_block2_ponly_steps", ["--encoder", "dense", "--no-imu", "--block", "2",
                               "--steps", "4"]),
    ("e3_block8_shank_imu_steps", ["--encoder", "dense", "--shank-imu", "--block", "8",
                              "--steps", "4"]),
    ("e1_natural_width_steps", ["--encoder", "moments", "--shank-imu", "--moment-hidden", "128",
                                "--steps", "4"]),
    ("e1_natural_ponly_steps", ["--encoder", "moments", "--no-imu", "--moment-hidden", "128",
                                "--steps", "4"]),
    ("e3_phase_shift_steps", ["--encoder", "dense", "--no-imu", "--block", "4",
                              "--block-origin", "2", "--steps", "4"]),
]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("modes")
    cache = root / "cache.npz"
    segments = [_segment(str(k), chr(96 + k), 120, k * 1000) for k in range(1, 8)]
    write_aligned_cache(aligned_cache_payload(segments, 2, 0), cache)
    shank_imu_dir = root / "shank_imu"
    shank_imu_dir.mkdir()
    for segment in segments:
        _shank_imu_cache_for(segment, shank_imu_dir)
    return root, cache, shank_imu_dir


@pytest.mark.parametrize("name,extra", CONFIGURATIONS, ids=[c[0] for c in CONFIGURATIONS])
def test_queued_configuration_runs(workspace, name, extra):
    root, cache, shank_imu_dir = workspace
    out = root / name
    arguments = [a if a != "--shank-imu" else "--shank-imu-dir" for a in extra]
    if "--shank-imu-dir" in arguments:
        arguments.insert(arguments.index("--shank-imu-dir") + 1, str(shank_imu_dir))
    if "--seed" not in arguments:
        arguments += ["--seed", "0"]
    result = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "train_moveport.py"),
         "--cache", str(cache), "--head", "free", "--fold", "0",
         "--epochs", "2", "--batch", "16", "--out", str(out)] + arguments,
        capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(_ROOT)),
    )
    assert result.returncode == 0, f"{name}\n{result.stderr[-1500:]}"
    records = list(out.glob("*.json"))
    assert len(records) == 1, name
    record = json.loads(records[0].read_text())
    assert "test_mm" not in record, name
    assert record["diverged"] is False, name


def _variant_key(extra):
    """What the configuration trains, ignoring which mode runs it."""
    def value(flag):
        return extra[extra.index(flag) + 1] if flag in extra else None
    return (value("--encoder"), "--shank-imu" in extra, value("--block"),
            value("--block-origin"), value("--moment-hidden"))


def _mode(extra):
    return "inner" if "--inner-fold" in extra else "steps" if "--steps" in extra else None


def test_the_grid_covers_every_queued_variant():
    """A queued variant without a smoke entry is how the last night was lost."""
    names = " ".join(name for name, _ in CONFIGURATIONS)
    for required in ("ponly_dense", "ponly_moments", "shank_imu_dense", "shank_imu_moments",
                     "shank_imu_only", "e3_block", "e1_natural", "e0_extra", "e3_phase"):
        assert required in names, required


def test_every_variant_is_covered_in_both_modes():
    """The queue selects in one mode and retrains in the other; both must be exercised."""
    modes: dict[tuple, set] = {}
    for name, extra in CONFIGURATIONS:
        mode = _mode(extra)
        assert mode is not None, f"{name} runs in neither inner nor steps mode"
        modes.setdefault(_variant_key(extra), set()).add(mode)
    missing = {key: sorted({"inner", "steps"} - seen)
               for key, seen in modes.items() if len(seen) < 2}
    assert not missing, f"variants exercised in only one mode: {missing}"
