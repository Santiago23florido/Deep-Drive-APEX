import math

import numpy as np
import pytest

from apex_fusion_research.core.rotation import quat_from_euler
from apex_fusion_research.core.slam_odometry import SlamOdometryConfig, planar_base_pose


def test_heading_only_has_no_translation():
    q = quat_from_euler(0.01, -0.02, 1.2)
    x, y, yaw = planar_base_pose(np.array([5.0, -3.0, 0.1]), q, SlamOdometryConfig())
    assert (x, y) == (0.0, 0.0)
    assert abs(yaw - 1.2) < 1e-12


def test_full_pose_applies_lever_arm():
    cfg = SlamOdometryConfig(mode="full_pose", imu_offset_in_base_xyz=[0.1, 0.0, 0.0])
    q = quat_from_euler(0.0, 0.0, math.pi / 2)
    x, y, yaw = planar_base_pose(np.array([1.0, 2.0, 0.0]), q, cfg)
    # IMU 0.1 m ahead of the base; facing +y, so the base is 0.1 m behind in y.
    np.testing.assert_allclose((x, y, yaw), (1.0, 1.9, math.pi / 2), atol=1e-12)


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        SlamOdometryConfig(mode="wheel").validate()
