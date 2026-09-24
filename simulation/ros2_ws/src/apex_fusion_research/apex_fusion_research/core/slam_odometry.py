"""Planar odometry prior for a 2D SLAM fed by the damaged sensors.

2D SLAM back-ends such as slam_toolbox need an ``odom -> base`` transform as
the motion prior of their scan matcher. The prior is built from an
IMU-frame pose source (an INS or the ground truth):

* ``heading_only`` (default): yaw of the source, translation fixed at zero.
  With the strapdown INS of the noisy IMU this is the damaged-sensor
  baseline: the translation is left entirely to LiDAR scan matching (typical
  low-cost LiDAR + IMU setup without wheel odometry). The published yaw is
  absolute (world aligned) because the INS heading starts from the known pose.
* ``full_pose``: full planar pose of the source (x, y, yaw) transferred from
  the IMU to the base frame with the lever arm. Fed with the ground truth it
  is an ideal odometry (perfect wheel encoders + gyro): the good-sensor
  reference. Fed with a MEMS INS it diverges within seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rotation import quat_to_euler, quat_to_rotmat

MODES = ("heading_only", "full_pose")


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
