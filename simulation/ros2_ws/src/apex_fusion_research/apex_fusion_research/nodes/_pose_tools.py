"""Access to the pose-dataset tools (tracks, driver, vehicle model) from ROS.

The live real2sim simulation drives the validation formats of the dataset
(track, motion profile, seed, laps): the same reference path, speed plan,
driver gains, actuator dynamics and start pose (with its seeded placement
errors) as ``tools/pose_dataset/gz_runner``.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any

from .real_sensor_node import sim_root


def tools_path() -> None:
    tools = str(sim_root() / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)


def run_setup(track_name: str, motion: str, seed: int, laps: float, v_ref: float = 0.0) -> dict[str, Any]:
    """Everything a run of the dataset format needs (see gz_runner)."""
    tools_path()
    from pose_dataset.gz_runner import build_run_setup  # noqa: PLC0415
    from pose_dataset.tracks import load_track  # noqa: PLC0415
    from pose_dataset.vehicle import PACKAGE_DIR, load_vehicle_config  # noqa: PLC0415

    vcfg = load_vehicle_config()
    track = load_track(track_name)
    if v_ref <= 0.0:
        calib = json.loads((PACKAGE_DIR / "config" / "speed_calibration.json").read_text(encoding="utf-8"))["tracks"]
        v_ref = float(calib[track_name]["v_max_stable_mps"])
    job = {"track": track_name, "motion": motion, "seed": int(seed), "direction": 1 if int(seed) % 2 == 1 else -1,
           "laps": float(laps), "v_ref": v_ref, "trajectory_key": f"{track_name}__{motion}__s{seed}"}
    setup = build_run_setup(job, track, vcfg)
    path = setup["path"]
    h0 = float(path.heading[0])
    rear = -0.15
    lat = setup["init_lateral_error"]
    nominal = (path.xy[0, 0] - rear * math.cos(h0), path.xy[0, 1] - rear * math.sin(h0), h0)
    spawn = (path.xy[0, 0] - rear * math.cos(h0) - lat * math.sin(h0), path.xy[0, 1] - rear * math.sin(h0) + lat * math.cos(h0),
             h0 + setup["init_yaw_error"])
    return {"job": job, "setup": setup, "track": track, "vcfg": vcfg, "nominal_start": nominal, "spawn": spawn, "rear_offset": rear}
