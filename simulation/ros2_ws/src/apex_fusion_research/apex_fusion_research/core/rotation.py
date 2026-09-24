"""Quaternion and rotation utilities.

Conventions used throughout the package
---------------------------------------
* Quaternions are Hamilton quaternions stored as ``[w, x, y, z]``.
* ``q_nb`` rotates vectors from the body frame ``b`` into the navigation
  frame ``n``: ``v_n = R(q_nb) @ v_b``.
* Euler angles follow the aerospace ZYX sequence (yaw, pitch, roll).

Only NumPy is required so these helpers can be used both by ROS nodes and by
offline analysis scripts and unit tests.
"""

from __future__ import annotations

import math

import numpy as np

_SMALL_ANGLE = 1e-12


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = q / norm
    # Keep a canonical hemisphere (w >= 0) to avoid sign flips in outputs.
    return q if q[0] >= 0.0 else -q


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def quat_from_rotvec(phi: np.ndarray) -> np.ndarray:
    """Exponential map: rotation vector (axis * angle) to unit quaternion."""
    phi = np.asarray(phi, dtype=float)
    angle = float(np.linalg.norm(phi))
    if angle < _SMALL_ANGLE:
        return quat_normalize(np.array([1.0, 0.5 * phi[0], 0.5 * phi[1], 0.5 * phi[2]]))
    half = 0.5 * angle
    axis = phi / angle
    return np.array([math.cos(half), *(math.sin(half) * axis)], dtype=float)


def rotvec_from_quat(q: np.ndarray) -> np.ndarray:
    """Logarithmic map: unit quaternion to rotation vector (angle in [0, pi])."""
    q = quat_normalize(q)
    vec = q[1:]
    sin_half = float(np.linalg.norm(vec))
    if sin_half < _SMALL_ANGLE:
        return 2.0 * vec
    angle = 2.0 * math.atan2(sin_half, q[0])
    return vec / sin_half * angle


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return quat_normalize(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ]
        )
    )


def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """Return (roll, pitch, yaw) in radians for the ZYX sequence."""
    w, x, y, z = quat_normalize(q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = quat_normalize(q0)
    q1 = np.asarray(q1, dtype=float)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    delta = quat_multiply(quat_conjugate(q0), q1)
    return quat_normalize(quat_multiply(q0, quat_from_rotvec(t * rotvec_from_quat(delta))))


def wrap_angle(angle):
    """Wrap an angle (scalar or array) to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def attitude_error_rotvec(q_true: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    """Attitude error expressed as a rotation vector in the navigation frame.

    ``R_est = Exp(delta) @ R_true``; for small errors ``delta`` holds the
    tilt errors about the navigation x/y axes and the heading error about z.
    """
    q_err = quat_multiply(quat_normalize(q_est), quat_conjugate(quat_normalize(q_true)))
    return rotvec_from_quat(q_err)
