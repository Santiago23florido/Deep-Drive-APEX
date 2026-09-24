import math

import numpy as np

from apex_fusion_research.core import rotation as rot


def test_euler_roundtrip():
    for angles in [(0.1, -0.2, 2.5), (-0.4, 0.3, -3.0), (0.0, 0.0, 0.0)]:
        q = rot.quat_from_euler(*angles)
        np.testing.assert_allclose(rot.quat_to_euler(q), angles, atol=1e-12)


def test_rotvec_exp_log_roundtrip():
    phi = np.array([0.3, -0.2, 1.1])
    np.testing.assert_allclose(rot.rotvec_from_quat(rot.quat_from_rotvec(phi)), phi, atol=1e-12)


def test_rotmat_matches_yaw():
    R = rot.quat_to_rotmat(rot.quat_from_euler(0.0, 0.0, math.pi / 2))
    np.testing.assert_allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_attitude_error_is_heading_difference():
    q_true = rot.quat_from_euler(0.0, 0.0, 0.5)
    q_est = rot.quat_from_euler(0.0, 0.0, 0.53)
    np.testing.assert_allclose(rot.attitude_error_rotvec(q_true, q_est), [0, 0, 0.03], atol=1e-12)
