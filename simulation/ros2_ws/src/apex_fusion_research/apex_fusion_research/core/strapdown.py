"""Strapdown inertial navigation (pure IMU dead reckoning).

Mechanization in a local-level, flat, non-rotating navigation frame ``n``
(the Gazebo world frame, ENU, z up). Earth rotation and transport rate are
neglected: they are not simulated by Gazebo and the navigation area is a few
metres wide. With ``q = q_nb`` (body to navigation), ``g_n = [0, 0, -g]``:

    q_k = q_{k-1} (x) Exp(phi_k),           phi_k = w_bar_k dt
    v_k = v_{k-1} + (R(q_mid) f_bar_k + g_n) dt
    p_k = p_{k-1} + (v_{k-1} + v_k) / 2 dt

``w_bar`` and ``f_bar`` are the bias-compensated increments averaged over the
interval (trapezoidal rule, or the latest sample for the Euler scheme) and
``q_mid = q_{k-1} (x) Exp(phi_k / 2)`` is the mid-interval attitude, which
removes the first-order rotation error of the velocity update. Coning and
sculling corrections are not needed because the sensor delivers samples, not
increments.

The integrator knows nothing about the true trajectory: every error in the
raw measurements (noise, bias, scale factor, misalignment) accumulates, which
is exactly the drift a real, unaided INS exhibits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .rotation import quat_from_euler, quat_from_rotvec, quat_multiply, quat_normalize, quat_to_rotmat


@dataclass
class StrapdownConfig:
    gravity_mps2: float = 9.8  # must match the simulated world
    integration_scheme: str = "trapezoidal"  # "trapezoidal" | "euler"
    max_dt_s: float = 0.5  # larger gaps are integrated as a single step of this size
    # Constrain the vertical channel (z, v_z) to the initial value. A free
    # vertical channel is unstable for any INS; ground vehicles usually clamp it.
    clamp_vertical_channel: bool = False


@dataclass
class AlignmentConfig:
    """How the INS is initialized before free inertial navigation starts.

    * ``static_coarse``: realistic start. While the vehicle is stationary for
      ``duration_s`` the INS levels itself from the accelerometers and
      estimates the gyro bias (and the accelerometer bias along gravity).
      Position and heading are taken from ground truth, since a real system
      would get them from a known start pose (heading is unobservable from a
      6-axis IMU).
    * ``truth``: full initial state from ground truth, no bias calibration.
      Isolates the raw effect of every sensor error.
    """

    mode: str = "static_coarse"
    duration_s: float = 5.0
    max_truth_speed_mps: float = 0.02  # stationarity test on ground truth
    max_truth_rate_rps: float = 0.02
    estimate_gyro_bias: bool = True
    estimate_accel_bias_along_gravity: bool = True

    def validate(self) -> None:
        if self.mode not in {"static_coarse", "truth"}:
            raise ValueError("alignment.mode must be 'static_coarse' or 'truth'")
        if self.duration_s <= 0.0:
            raise ValueError("alignment.duration_s must be positive")


@dataclass
class InsConfig:
    strapdown: StrapdownConfig = field(default_factory=StrapdownConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)

    def validate(self) -> None:
        self.alignment.validate()
        if self.strapdown.integration_scheme not in {"trapezoidal", "euler"}:
            raise ValueError("strapdown.integration_scheme must be 'trapezoidal' or 'euler'")


@dataclass
class NavState:
    t: float
    q: np.ndarray  # [w, x, y, z], body -> navigation
    v: np.ndarray  # m/s, navigation frame
    p: np.ndarray  # m, navigation frame

    def copy(self) -> "NavState":
        return NavState(self.t, self.q.copy(), self.v.copy(), self.p.copy())


class StrapdownIntegrator:
    def __init__(self, config: StrapdownConfig) -> None:
        if config.integration_scheme not in {"trapezoidal", "euler"}:
            raise ValueError(f"unknown integration_scheme '{config.integration_scheme}'")
        self.config = config
        self.state: NavState | None = None
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self._g_n = np.array([0.0, 0.0, -config.gravity_mps2])
        self._prev_w: np.ndarray | None = None
        self._prev_f: np.ndarray | None = None
        self._z0 = 0.0

    @property
    def initialized(self) -> bool:
        return self.state is not None

    def initialize(
        self,
        t: float,
        q: np.ndarray,
        v: np.ndarray,
        p: np.ndarray,
        gyro_bias: np.ndarray | None = None,
        accel_bias: np.ndarray | None = None,
    ) -> None:
        self.state = NavState(float(t), quat_normalize(q), np.array(v, float), np.array(p, float))
        self.gyro_bias = np.zeros(3) if gyro_bias is None else np.array(gyro_bias, float)
        self.accel_bias = np.zeros(3) if accel_bias is None else np.array(accel_bias, float)
        self._prev_w = None
        self._prev_f = None
        self._z0 = float(self.state.p[2])

    def propagate(self, t: float, angular_velocity: np.ndarray, specific_force: np.ndarray) -> NavState:
        if self.state is None:
            raise RuntimeError("integrator not initialized")
        s = self.state
        dt = float(t) - s.t
        w = np.asarray(angular_velocity, float) - self.gyro_bias
        f = np.asarray(specific_force, float) - self.accel_bias
        if dt <= 0.0:
            # Out-of-order or duplicated stamp: keep state, refresh history.
            self._prev_w, self._prev_f = w, f
            return s.copy()
        dt = min(dt, self.config.max_dt_s)

        if self.config.integration_scheme == "trapezoidal" and self._prev_w is not None:
            w_bar = 0.5 * (w + self._prev_w)
            f_bar = 0.5 * (f + self._prev_f)
        else:
            w_bar, f_bar = w, f

        phi = w_bar * dt
        q_mid = quat_multiply(s.q, quat_from_rotvec(0.5 * phi))
        q_new = quat_normalize(quat_multiply(s.q, quat_from_rotvec(phi)))
        a_n = quat_to_rotmat(q_mid) @ f_bar + self._g_n
        v_new = s.v + a_n * dt
        p_new = s.p + 0.5 * (s.v + v_new) * dt
        if self.config.clamp_vertical_channel:
            v_new[2] = 0.0
            p_new[2] = self._z0

        self.state = NavState(float(t), q_new, v_new, p_new)
        self._prev_w, self._prev_f = w, f
        return self.state.copy()


@dataclass
class StaticAlignmentResult:
    roll: float
    pitch: float
    gyro_bias: np.ndarray
    accel_bias: np.ndarray
    mean_specific_force: np.ndarray
    gyro_std: np.ndarray
    accel_std: np.ndarray
    samples: int

    def attitude(self, yaw: float) -> np.ndarray:
        return quat_from_euler(self.roll, self.pitch, yaw)


def static_coarse_alignment(
    angular_velocity: np.ndarray,
    specific_force: np.ndarray,
    gravity_mps2: float,
    estimate_gyro_bias: bool = True,
    estimate_accel_bias_along_gravity: bool = True,
) -> StaticAlignmentResult:
    """Coarse alignment from a stationary window (Groves 2013, Sec. 5.6.2).

    * Levelling: at rest ``f_b = R_nb^T [0, 0, g]``, hence
      ``roll = atan2(f_y, f_z)`` and ``pitch = atan2(-f_x, sqrt(f_y^2 + f_z^2))``.
      A horizontal accelerometer bias is indistinguishable from a tilt here,
      so it is absorbed into the attitude (a fundamental observability limit).
    * Gyro bias: mean angular rate (the platform does not rotate; Earth rate is
      not simulated).
    * Accelerometer bias along gravity: ``(|f_mean| - g) * f_mean / |f_mean|``.
    * Heading is unobservable without a magnetometer or aiding: the caller
      supplies it.
    """
    w = np.asarray(angular_velocity, float).reshape(-1, 3)
    f = np.asarray(specific_force, float).reshape(-1, 3)
    if w.shape[0] == 0 or f.shape[0] == 0:
        raise ValueError("static alignment needs at least one sample")
    f_mean = f.mean(axis=0)
    roll = math.atan2(f_mean[1], f_mean[2])
    pitch = math.atan2(-f_mean[0], math.hypot(f_mean[1], f_mean[2]))
    gyro_bias = w.mean(axis=0) if estimate_gyro_bias else np.zeros(3)
    norm = float(np.linalg.norm(f_mean))
    if estimate_accel_bias_along_gravity and norm > 1e-6:
        accel_bias = (norm - gravity_mps2) * f_mean / norm
    else:
        accel_bias = np.zeros(3)
    return StaticAlignmentResult(
        roll=roll,
        pitch=pitch,
        gyro_bias=gyro_bias,
        accel_bias=accel_bias,
        mean_specific_force=f_mean,
        gyro_std=w.std(axis=0),
        accel_std=f.std(axis=0),
        samples=int(w.shape[0]),
    )
