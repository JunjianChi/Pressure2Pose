import os
import subprocess
import sys
from pathlib import Path


def test_shank_imu_cache_generation_explains_the_pinned_runtime_requirement(tmp_path):
    root = Path(__file__).parents[1]
    script = root / "scripts" / "generate_shank_imu_cache.py"
    environment = dict(os.environ, PYTHONPATH=str(root))
    result = subprocess.run(
        [
            sys.executable, str(script),
            "--model", str(tmp_path / "placed.osim"),
            "--motion", str(tmp_path / "ik.mot"),
            "--out", str(tmp_path / "shank_imu.npz"),
            "--subject", "1", "--activity", "treadmill_normal", "--name", "high_1",
        ],
        capture_output=True, text=True, env=environment,
    )

    assert result.returncode == 2
    assert "requires the pinned OpenSim preprocessing runtime" in result.stderr
