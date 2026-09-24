"""Stochastic error model of a MEMS strapdown IMU (gyroscope + accelerometer).

The model turns *ideal* inertial quantities (true angular rate ``w`` and true
specific force ``f``, both in the sensor frame) into raw measurements, the way
a physical IMU does. Following the standard sensor error equations
(IEEE Std 952-1997, IEEE Std 1293-2018; Groves, *Principles of GNSS, Inertial,
and Multisensor Integrated Navigation Systems*, 2nd ed., 2013, Ch. 4.4):

    w_meas = sat(Q( (I + S_g + M_g) w + b_g(t) + G_g f + n_g ))
    f_meas = sat(Q( (I + S_a + M_a) f + b_a(t)          + n_a ))

with, for each triad (units: rad/s for gyro, m/s^2 for accelerometer):

* ``S``   diagonal scale-factor errors, ``S_ii ~ N(0, sigma_s^2)``, fixed per run;
* ``M``   off-diagonal misalignment / non-orthogonality (small angles),
          ``M_ij ~ N(0, sigma_m^2)`` for ``i != j``, fixed per run;
* ``G_g`` gyro g-sensitivity matrix, ``G_ij ~ N(0, sigma_G^2)``, fixed per run;
* ``b(t) = b_on + b_GM(t) + b_RW(t)`` the bias, sum of
  - turn-on bias ``b_on ~ N(0, sigma_on^2)`` (constant for one power-up),
  - bias instability as a first-order Gauss-Markov process
    ``b_GM[k+1] = e^{-dt/tau} b_GM[k] + sigma_GM sqrt(1 - e^{-2 dt/tau}) w_k``
    (stationary std ``sigma_GM``, correlation time ``tau``),
  - bias random walk (rate / acceleration random walk, Allan coefficient K)
    ``b_RW[k+1] = b_RW[k] + K sqrt(dt) w_k``;
* ``n``   white noise with one-sided density N (angle / velocity random walk,
          Allan coefficient N): discrete std ``N / sqrt(dt)``;
* ``Q``   quantization to one LSB = 2 FS / 2^bits; ``sat`` clips to +/- FS.

Four independent random streams (static/dynamic x gyro/accel) are derived from
one seed, so changing the gyro parameters does not alter the accelerometer
realization and vice versa (common random numbers for controlled studies).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class TriadErrorConfig:
    """Error parameters of one 3-axis sensor triad. Defaults = ideal sensor.

    Units are given for the gyroscope / accelerometer respectively.
    """

    noise_density: float = 0.0  # N: rad/s/sqrt(Hz) | m/s^2/sqrt(Hz)
    turn_on_bias_std: float = 0.0  # rad/s | m/s^2
    bias_instability_std: float = 0.0  # Gauss-Markov stationary std: rad/s | m/s^2
    bias_correlation_time_s: float = 100.0  # Gauss-Markov tau [s]
    bias_random_walk: float = 0.0  # K: rad/s/sqrt(s) | m/s^2/sqrt(s)
    scale_factor_std: float = 0.0  # [-]
    misalignment_std_rad: float = 0.0  # [rad]
    g_sensitivity_std: float = 0.0  # gyro only: (rad/s)/(m/s^2); ignored for accel
    full_scale: float = 0.0  # saturation limit, 0 disables it
    adc_bits: int = 0  # quantization resolution, 0 disables it


@dataclass
class ImuErrorConfig:
    enabled: bool = True
    seed: int = 42
    gyro: TriadErrorConfig = field(default_factory=TriadErrorConfig)
    accel: TriadErrorConfig = field(default_factory=TriadErrorConfig)


class TriadErrorModel:
    """Time-propagated error state of one sensor triad."""

    def __init__(
        self,
        config: TriadErrorConfig,
        static_rng: np.random.Generator,
        dynamic_rng: np.random.Generator,
        use_g_sensitivity: bool = False,
    ) -> None:
        self.config = config
        self._rng = dynamic_rng
        cfg = config
        # Constant (per power-up) errors.
        self.turn_on_bias = static_rng.normal(0.0, cfg.turn_on_bias_std, 3)
        self.scale_factor = static_rng.normal(0.0, cfg.scale_factor_std, 3)
        misalignment = static_rng.normal(0.0, cfg.misalignment_std_rad, (3, 3))
        np.fill_diagonal(misalignment, 0.0)
        self.misalignment = misalignment
        g_sens = static_rng.normal(0.0, cfg.g_sensitivity_std, (3, 3))
        self.g_sensitivity = g_sens if use_g_sensitivity else np.zeros((3, 3))
        self.transfer_matrix = np.eye(3) + np.diag(self.scale_factor) + self.misalignment
        # Time-varying bias states. The Gauss-Markov state starts from its
        # stationary distribution: the sensor has been "on" for a while.
        self.bias_gm = static_rng.normal(0.0, cfg.bias_instability_std, 3)
        self.bias_rw = np.zeros(3)
        self.lsb = 2.0 * cfg.full_scale / (2 ** cfg.adc_bits) if cfg.adc_bits > 0 and cfg.full_scale > 0 else 0.0

    @property
    def bias(self) -> np.ndarray:
        """Current total additive bias (turn-on + instability + random walk)."""
        return self.turn_on_bias + self.bias_gm + self.bias_rw

    def propagate_bias(self, dt: float) -> None:
        cfg = self.config
        if cfg.bias_instability_std > 0.0 and cfg.bias_correlation_time_s > 0.0:
            phi = math.exp(-dt / cfg.bias_correlation_time_s)
            self.bias_gm = phi * self.bias_gm + cfg.bias_instability_std * math.sqrt(
                max(0.0, 1.0 - phi * phi)
            ) * self._rng.standard_normal(3)
        else:
            self._rng.standard_normal(3)  # keep the stream aligned
        self.bias_rw = self.bias_rw + cfg.bias_random_walk * math.sqrt(dt) * self._rng.standard_normal(3)

    def measure(self, true_value: np.ndarray, dt: float, specific_force: np.ndarray | None = None) -> np.ndarray:
        """Propagate the bias over ``dt`` and return one corrupted sample."""
        cfg = self.config
        self.propagate_bias(dt)
        white = cfg.noise_density / math.sqrt(dt) * self._rng.standard_normal(3)
        y = self.transfer_matrix @ np.asarray(true_value, dtype=float) + self.bias + white
        if specific_force is not None:
            y = y + self.g_sensitivity @ np.asarray(specific_force, dtype=float)
        if self.lsb > 0.0:
            y = np.round(y / self.lsb) * self.lsb
        if cfg.full_scale > 0.0:
            y = np.clip(y, -cfg.full_scale, cfg.full_scale)
        return y

    def realization(self) -> dict[str, list]:
        return {
            "turn_on_bias": self.turn_on_bias.tolist(),
            "scale_factor": self.scale_factor.tolist(),
            "misalignment": self.misalignment.tolist(),
            "g_sensitivity": self.g_sensitivity.tolist(),
            "initial_bias_gauss_markov": self.bias_gm.tolist(),
            "lsb": self.lsb,
        }


@dataclass
class ImuSample:
    angular_velocity: np.ndarray  # rad/s, sensor frame
    specific_force: np.ndarray  # m/s^2, sensor frame
    gyro_bias: np.ndarray  # true total additive gyro bias at this sample
    accel_bias: np.ndarray  # true total additive accelerometer bias


class ImuErrorModel:
    """Full IMU: gyroscope and accelerometer triads with independent streams."""

    def __init__(self, config: ImuErrorConfig) -> None:
        self.config = config
        ss = np.random.SeedSequence(int(config.seed)).spawn(4)
        rngs = [np.random.default_rng(s) for s in ss]
        self.gyro = TriadErrorModel(config.gyro, rngs[0], rngs[1], use_g_sensitivity=True)
        self.accel = TriadErrorModel(config.accel, rngs[2], rngs[3])

    def measure(self, angular_velocity: np.ndarray, specific_force: np.ndarray, dt: float) -> ImuSample:
        w = np.asarray(angular_velocity, dtype=float)
        f = np.asarray(specific_force, dtype=float)
        if not self.config.enabled:
            return ImuSample(w.copy(), f.copy(), np.zeros(3), np.zeros(3))
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        w_meas = self.gyro.measure(w, dt, specific_force=f)
        f_meas = self.accel.measure(f, dt)
        return ImuSample(w_meas, f_meas, self.gyro.bias.copy(), self.accel.bias.copy())

    def realization(self) -> dict[str, dict]:
        return {"gyro": self.gyro.realization(), "accel": self.accel.realization()}

    def white_noise_std(self, dt: float) -> tuple[float, float]:
        """Per-sample white-noise std (gyro, accel), e.g. for covariances."""
        return (
            self.config.gyro.noise_density / math.sqrt(dt),
            self.config.accel.noise_density / math.sqrt(dt),
        )
