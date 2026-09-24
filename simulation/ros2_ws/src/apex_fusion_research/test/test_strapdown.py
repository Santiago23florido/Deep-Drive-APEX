import math

import numpy as np

from apex_fusion_research.core import rotation as rot
from apex_fusion_research.core.strapdown import (
    StrapdownConfig,
    StrapdownIntegrator,
    static_coarse_alignment,
)

G = 9.8
DT = 1.0 / 120.0


def circle_imu(t, radius=2.0, speed=1.0):
    """Ideal IMU readings of a planar vehicle driving on a circle at constant speed."""
    omega = speed / radius
    w_b = np.array([0.0, 0.0, omega])
    f_b = np.array([0.0, speed * omega, G])  # centripetal accel. + gravity reaction
    yaw = math.pi / 2 + omega * t
    p = np.array([radius * math.cos(omega * t), radius * math.sin(omega * t), 0.0])
    v = speed * np.array([math.cos(yaw), math.sin(yaw), 0.0])
    return w_b, f_b, yaw, p, v


def test_perfect_imu_tracks_circle():
    ins = StrapdownIntegrator(StrapdownConfig(gravity_mps2=G))
    _, _, yaw0, p0, v0 = circle_imu(0.0)
    ins.initialize(0.0, rot.quat_from_euler(0, 0, yaw0), v0, p0)
    t = 0.0
    for _ in range(int(60 / DT)):  # 60 s ~ 4.8 laps
        t += DT
        w_b, f_b, *_ = circle_imu(t)
        state = ins.propagate(t, w_b, f_b)
    _, _, yaw, p, v = circle_imu(t)
    assert np.linalg.norm(state.p - p) < 0.02
    assert abs(rot.wrap_angle(rot.quat_to_euler(state.q)[2] - yaw)) < 1e-6


def test_constant_accel_bias_gives_quadratic_position_error():
    b = 0.05
    ins = StrapdownIntegrator(StrapdownConfig(gravity_mps2=G))
    ins.initialize(0.0, np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(3))
    t, T = 0.0, 20.0
    while t < T - 1e-9:
        t += DT
        state = ins.propagate(t, np.zeros(3), np.array([b, 0.0, G]))
    assert abs(state.p[0] - 0.5 * b * T * T) < 1e-3 * 0.5 * b * T * T


def test_gyro_bias_tilt_gives_cubic_position_error():
    eps = 1e-3  # rad/s roll-rate bias -> tilt eps*t -> y-acceleration -g*eps*t
    ins = StrapdownIntegrator(StrapdownConfig(gravity_mps2=G))
    ins.initialize(0.0, np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(3))
    t, T = 0.0, 10.0
    while t < T - 1e-9:
        t += DT
        state = ins.propagate(t, np.array([eps, 0.0, 0.0]), np.array([0.0, 0.0, G]))
    expected = G * eps * T**3 / 6.0
    assert abs(abs(state.p[1]) - expected) < 0.02 * expected


def test_static_alignment_levelling_and_bias():
    roll, pitch = 0.02, -0.03
    R_nb = rot.quat_to_rotmat(rot.quat_from_euler(roll, pitch, 0.0))
    f_b = R_nb.T @ np.array([0.0, 0.0, G])
    gyro_bias = np.array([0.01, -0.02, 0.005])
    rng = np.random.default_rng(0)
    w = gyro_bias + 1e-4 * rng.standard_normal((600, 3))
    f = f_b + 1e-4 * rng.standard_normal((600, 3))
    res = static_coarse_alignment(w, f, G)
    assert abs(res.roll - roll) < 1e-4 and abs(res.pitch - pitch) < 1e-4
    np.testing.assert_allclose(res.gyro_bias, gyro_bias, atol=2e-5)
    assert np.linalg.norm(res.accel_bias) < 1e-4
