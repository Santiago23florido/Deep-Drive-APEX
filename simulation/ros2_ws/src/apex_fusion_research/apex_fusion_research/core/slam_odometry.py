"""Planar odometry prior for a 2D SLAM fed by the damaged sensors.

2D SLAM back-ends such as slam_toolbox need an ``odom -> base`` transform as
the motion prior of their scan matcher. For the damaged-sensor baseline the
prior comes from the noisy IMU only:

* ``heading_only`` (default): yaw from the strapdown INS (integrated noisy
  gyros after static alignment), translation fixed at zero. The translation
  is left entirely to LiDAR scan matching: the typical low-cost LiDAR + IMU
  configuration without wheel odometry. The published yaw is absolute (world
  aligned) because the INS heading is initialised from the known start pose.
* ``ins_full``: full planar pose of the INS (x, y, yaw) transferred from the
  IMU to the base frame with the lever arm. It diverges quickly with MEMS
  sensors and is provided for comparison experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rotation import quat_to_euler, quat_to_rotmat

MODES = ("heading_only", "ins_full")


@dataclass
class SlamOdometryConfig:
    mode: str = "heading_only"
    imu_offset_in_base_xyz: list = field(default_factory=lambda: [0.02, 0.0, 0.08])
    odom_frame_id: str = "noisy/odom"
    base_frame_id: str = "noisy/base_link"

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if len(self.imu_offset_in_base_xyz) != 3:
            raise ValueError("imu_offset_in_base_xyz needs 3 values")


def planar_base_pose(p_imu: np.ndarray, q_imu: np.ndarray, config: SlamOdometryConfig) -> tuple[float, float, float]:
    """Planar ``(x, y, yaw)`` of the base frame from an IMU-frame pose."""
    _, _, yaw = quat_to_euler(q_imu)
    if config.mode == "heading_only":
        return 0.0, 0.0, float(yaw)
    p_base = np.asarray(p_imu, dtype=float) - quat_to_rotmat(q_imu) @ np.asarray(config.imu_offset_in_base_xyz, dtype=float)
    return float(p_base[0]), float(p_base[1]), float(yaw)
