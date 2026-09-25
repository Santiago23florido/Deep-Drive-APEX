"""The Gazebo car's sprung body responds like the body_dynamics targets.

Runs ``pose_dataset.suspension_check`` (in-process gz-sim 8, clean
environment) and compares with ``vehicle.yaml``: 3 deg of roll per g, 2 deg
of pitch per g, 3.5 Hz, damping ratio 0.45, level at rest, outward roll.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest
import yaml

TOOLS = Path(__file__).resolve().parents[2]
VENV = TOOLS.parent / "learning" / ".venv" / "bin" / "python"


@pytest.mark.skipif(not (VENV.exists() and Path("/opt/ros/jazzy/setup.bash").exists()), reason="needs gz-sim 8 and ROS (xacro)")
def test_suspension_matches_body_dynamics():
    env = {"HOME": os.environ.get("HOME", "/tmp"), "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONPATH": str(TOOLS),
           "GZ_PARTITION": f"susp_test_{os.getpid()}", "GZ_IP": "127.0.0.1"}
    out = subprocess.run([str(VENV), "-m", "pose_dataset.suspension_check"], cwd=TOOLS, env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    m = json.loads(out.stdout[out.stdout.index("{"):])
    target = yaml.safe_load((TOOLS / "pose_dataset" / "config" / "vehicle.yaml").read_text())["body_dynamics"]
    assert abs(m["roll_deg_per_g"] - target["roll_gain_deg_per_g"]) < 0.5
    assert abs(m["pitch_deg_per_g"] - target["pitch_gain_deg_per_g"]) < 0.4
    assert abs(m["natural_freq_hz"] - target["natural_freq_hz"]) < 0.5
    assert abs(m["damping_ratio"] - target["damping_ratio"]) < 0.1
    assert abs(m["pitch_at_rest_deg"]) < 0.1
    assert m["outward_roll"]
