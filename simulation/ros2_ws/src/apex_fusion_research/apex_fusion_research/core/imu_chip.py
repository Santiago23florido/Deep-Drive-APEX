"""MEMS IMU chip model on top of Gazebo's native IMU (real2sim realism layer).

Gazebo's ``imu`` sensor, run at the physics rate (1 kHz) on the car's sprung
body, provides the rigid-body specific force and angular rate at the real
mount, with Gazebo's own noise model already applied (white noise, turn-on
bias and Gauss-Markov bias per axis, configured in the car's SDF from
``sensors.yaml`` ``imu.gazebo_noise``). What a real chip adds on top and Gazebo
cannot, is applied here, in the order of the physical signal chain:

1. chassis vibration (drivetrain / surface band and wheel harmonic, scaled by
   the vehicle speed; the rigid Gazebo contact model does not produce it);
2. the chip's digital low-pass filter (2nd-order Butterworth, separate gyro
   and accelerometer cut-offs), which also shapes Gazebo's white noise into
   the datasheet output noise;
3. sampling at the output data rate with the chip's own oscillator (constant
   frequency error, random phase): sample instants are exact, not rounded to
   the 1 ms physics grid (linear interpolation of the filtered 1 kHz signal);
4. the error terms without a Gazebo equivalent (``imu_error.TriadErrorModel``
   with noise and bias disabled): scale factor, misalignment, g-sensitivity,
   bias random walk, quantization and saturation;
5. timestamps (sampling instant + offset + jitter, strictly increasing) and
   the time at which the sample reaches the host (publication latency);
6. optional dropouts.

Streaming: :meth:`ImuChip.push` takes any number of consecutive 1 kHz samples
and returns the output samples completed so far, so the same code runs inside
the dataset generator and in the live ROS node.

When the noise-free twin IMU is given (``push(..., clean_gyro, clean_accel)``),
the difference noisy - clean is exactly Gazebo's noise + bias; a 1 s moving
average of it, filtered like the signal, is returned as the true additive bias
of every output sample (dataset label, never a model input).

References
  [1] ST, LSM6DS3 datasheet DocID026899 Rev 10 (output data rates, filter
      bandwidths, noise densities, sensitivities).
  [2] IEEE Std 952-1997 / 1293-2018 (sensor error equations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy import signal

from .imu_error import TriadErrorConfig, TriadErrorModel

GRAVITY = 9.80665


@dataclass
class VibrationConfig:
    """Speed-proportional chassis vibration (see pose_dataset vehicle.yaml)."""

    accel_rms_per_mps: float = 0.0  # m/s^2 RMS per m/s
    gyro_rms_per_mps: float = 0.0  # rad/s RMS per m/s
    band_hz: tuple[float, float] = (12.0, 160.0)
    z_scale: float = 1.0
    wheel_harmonic_mps2: float = 0.0  # per m/s, at the wheel rotation frequency
    wheel_radius_m: float = 0.06
    gain: float = 1.0  # mount transmissibility of this sensor


@dataclass
class ImuChipConfig:
    odr_hz: float = 104.0
    clock_ppm_std: float = 0.0
    native_rate_hz: float = 1000.0
    dlpf_gyro_hz: float = 0.0  # 0 disables the filter
    dlpf_accel_hz: float = 0.0
    timestamp_offset_ms: float = 0.0
    timestamp_jitter_std_ms: float = 0.0
    publish_latency_ms: float = 0.0
    dropout_prob: float = 0.0
    dropout_burst_prob: float = 0.0
    dropout_burst_len: tuple[int, int] = (2, 4)
    gyro: TriadErrorConfig = field(default_factory=TriadErrorConfig)  # layer-only terms
    accel: TriadErrorConfig = field(default_factory=TriadErrorConfig)
    vibration: VibrationConfig = field(default_factory=VibrationConfig)
    bias_label_window_s: float = 1.0
    seed: int = 0


@dataclass
class ImuOutput:
    t_true_ns: int  # sampling instant (simulation clock)
    stamp_ns: int  # reported timestamp
    release_ns: int  # time the sample is available to the host
    gyro: np.ndarray  # rad/s, sensor frame
    accel: np.ndarray  # m/s^2 specific force (gravity included), sensor frame
    bias_gyro: np.ndarray | None = None  # true additive bias (label)
    bias_accel: np.ndarray | None = None


def band_noise_rms(sos: np.ndarray, n: int = 20000) -> float:
    """Steady-state RMS of ``sos`` driven by unit white noise (impulse energy)."""
    impulse = np.zeros(n)
    impulse[0] = 1.0
    return float(np.sqrt(np.sum(signal.sosfilt(sos, impulse) ** 2)))


class _Sos:
    """Streaming SOS filter over the columns of [N, C] blocks."""

    def __init__(self, sos: np.ndarray, channels: int) -> None:
        self.sos = sos
        self.zi: np.ndarray | None = None
        self.channels = channels

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.zi is None:  # start in steady state with the first sample
            self.zi = signal.sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self.zi = signal.sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y


class ImuChip:
    def __init__(self, cfg: ImuChipConfig) -> None:
        self.cfg = cfg
        ss = np.random.SeedSequence(int(cfg.seed)).spawn(4)
        self._rng_vib = np.random.default_rng(ss[0])
        self._rng_clock = np.random.default_rng(ss[1])
        self._rng_stamp = np.random.default_rng(ss[2])
        err = np.random.SeedSequence(int(cfg.seed) + 1).spawn(4)
        self.gyro_err = TriadErrorModel(cfg.gyro, np.random.default_rng(err[0]), np.random.default_rng(err[1]), use_g_sensitivity=True)
        self.accel_err = TriadErrorModel(cfg.accel, np.random.default_rng(err[2]), np.random.default_rng(err[3]))
        fs = cfg.native_rate_hz
        self.dt_native = 1.0 / fs
        v = cfg.vibration
        self._vib_on = v.gain > 0.0 and (v.accel_rms_per_mps > 0.0 or v.gyro_rms_per_mps > 0.0 or v.wheel_harmonic_mps2 > 0.0)
        if self._vib_on:
            hi = min(v.band_hz[1], 0.45 * fs)
            vsos = signal.butter(2, (v.band_hz[0], hi), btype="bandpass", fs=fs, output="sos")
            self._vib = _Sos(vsos, 6)
            self._vib_norm = band_noise_rms(vsos)
            self._wheel_phase = float(self._rng_vib.uniform(0.0, 2.0 * math.pi))
        self._lp_g = _Sos(signal.butter(2, cfg.dlpf_gyro_hz, fs=fs, output="sos"), 3) if 0.0 < cfg.dlpf_gyro_hz < 0.5 * fs else None
        self._lp_a = _Sos(signal.butter(2, cfg.dlpf_accel_hz, fs=fs, output="sos"), 3) if 0.0 < cfg.dlpf_accel_hz < 0.5 * fs else None
        self._lp_eg = _Sos(signal.butter(2, cfg.dlpf_gyro_hz, fs=fs, output="sos"), 3) if self._lp_g is not None else None
        self._lp_ea = _Sos(signal.butter(2, cfg.dlpf_accel_hz, fs=fs, output="sos"), 3) if self._lp_a is not None else None
        # Output clock: constant frequency error, random phase (set at the first push).
        self.period_s = (1.0 / cfg.odr_hz) * (1.0 + self._rng_clock.normal(0.0, cfg.clock_ppm_std) * 1e-6)
        self._next_t: float | None = None
        self._last: tuple[float, np.ndarray, np.ndarray | None] | None = None  # (t, filtered [6], filtered error [6])
        self._last_stamp = -(2**62)
        self._burst_left = 0
        self._bias_avg: np.ndarray | None = None
        self.samples_out = 0
        self.samples_dropped = 0

    # ---------------------------------------------------------------- stream
    def push(self, t_ns: np.ndarray, gyro: np.ndarray, accel: np.ndarray, speed_mps: np.ndarray | float = 0.0,
             clean_gyro: np.ndarray | None = None, clean_accel: np.ndarray | None = None) -> list[ImuOutput]:
        """Consecutive native samples (``t_ns`` [N], ``gyro``/``accel`` [N, 3] in
        the sensor frame, ``speed_mps`` scalar or [N]); returns completed outputs."""
        t_ns = np.atleast_1d(np.asarray(t_ns, dtype=np.int64))
        n = len(t_ns)
        if n == 0:
            return []
        x = np.hstack((np.asarray(gyro, float).reshape(n, 3), np.asarray(accel, float).reshape(n, 3)))
        speed = np.broadcast_to(np.asarray(speed_mps, float), (n,)).astype(float)
        if self._vib_on:
            x = x + self._vibration(speed)
        g = self._lp_g(x[:, :3]) if self._lp_g is not None else x[:, :3]
        a = self._lp_a(x[:, 3:]) if self._lp_a is not None else x[:, 3:]
        filt = np.hstack((g, a))
        err = None
        if clean_gyro is not None and clean_accel is not None:
            e = np.hstack((np.asarray(gyro, float).reshape(n, 3) - np.asarray(clean_gyro, float).reshape(n, 3),
                           np.asarray(accel, float).reshape(n, 3) - np.asarray(clean_accel, float).reshape(n, 3)))
            eg = self._lp_eg(e[:, :3]) if self._lp_eg is not None else e[:, :3]
            ea = self._lp_ea(e[:, 3:]) if self._lp_ea is not None else e[:, 3:]
            err = self._bias_average(np.hstack((eg, ea)))
        t = t_ns.astype(np.float64) * 1e-9
        if self._next_t is None:
            self._next_t = t[0] + float(self._rng_clock.uniform(0.0, self.period_s))
        out: list[ImuOutput] = []
        prev_t, prev_x, prev_e = self._last if self._last is not None else (t[0], filt[0], None if err is None else err[0])
        ts = np.concatenate(([prev_t], t))
        xs = np.vstack((prev_x[None], filt))
        es = None if err is None else np.vstack(((prev_e if prev_e is not None else err[0])[None], err))
        while self._next_t <= t[-1]:
            tk = self._next_t
            self._next_t += self.period_s
            if tk < ts[0]:
                continue
            i = int(np.searchsorted(ts, tk, side="right")) - 1
            i = min(max(i, 0), len(ts) - 2)
            w = 0.0 if ts[i + 1] == ts[i] else (tk - ts[i]) / (ts[i + 1] - ts[i])
            val = (1.0 - w) * xs[i] + w * xs[i + 1]
            ev = None if es is None else (1.0 - w) * es[i] + w * es[i + 1]
            sample = self._chip_sample(tk, val, ev)
            if sample is not None:
                out.append(sample)
        self._last = (t[-1], filt[-1], None if err is None else err[-1])
        return out

    # ------------------------------------------------------------- internals
    def _vibration(self, speed: np.ndarray) -> np.ndarray:
        v = self.cfg.vibration
        white = self._rng_vib.standard_normal((len(speed), 6))
        band = self._vib(white) / max(self._vib_norm, 1e-12)
        vib = np.empty_like(band)
        vib[:, :3] = band[:, 3:] * v.gyro_rms_per_mps * speed[:, None]
        vib[:, 3:] = band[:, :3] * v.accel_rms_per_mps * speed[:, None]
        vib[:, 5] *= v.z_scale
        phase = self._wheel_phase + np.cumsum(speed / v.wheel_radius_m) * self.dt_native
        vib[:, 5] += v.wheel_harmonic_mps2 * speed * np.sin(phase)
        self._wheel_phase = float(phase[-1] % (2.0 * math.pi))
        return v.gain * vib

    def _bias_average(self, e: np.ndarray) -> np.ndarray:
        """Causal moving average (exponential, time constant = label window)."""
        alpha = self.dt_native / max(self.cfg.bias_label_window_s, self.dt_native)
        prev = e[0] if self._bias_avg is None else self._bias_avg
        out, _ = signal.lfilter([alpha], [1.0, alpha - 1.0], e, axis=0, zi=((1.0 - alpha) * prev)[None, :])
        self._bias_avg = out[-1].copy()
        return out

    def _chip_sample(self, tk: float, val: np.ndarray, err: np.ndarray | None) -> ImuOutput | None:
        cfg = self.cfg
        dt = self.period_s
        gyro = self.gyro_err.measure(val[:3], dt, specific_force=val[3:])
        accel = self.accel_err.measure(val[3:], dt)
        # Dropouts: isolated samples and bursts (bus hiccups).
        rng = self._rng_stamp
        drop = False
        if self._burst_left > 0:
            self._burst_left -= 1
            drop = True
        elif cfg.dropout_burst_prob > 0.0 and rng.random() < cfg.dropout_burst_prob:
            lo, hi = cfg.dropout_burst_len
            self._burst_left = int(rng.integers(int(lo), int(hi) + 1)) - 1
            drop = True
        elif cfg.dropout_prob > 0.0 and rng.random() < cfg.dropout_prob:
            drop = True
        jitter = rng.normal(0.0, cfg.timestamp_jitter_std_ms) * 1e6 if cfg.timestamp_jitter_std_ms > 0.0 else 0.0
        if drop:
            self.samples_dropped += 1
            return None
        t_true_ns = int(round(tk * 1e9))
        stamp = int(round(t_true_ns + cfg.timestamp_offset_ms * 1e6 + jitter))
        if stamp <= self._last_stamp:  # drivers never publish out-of-order stamps
            stamp = self._last_stamp + 1000
        self._last_stamp = stamp
        release = max(stamp, t_true_ns + int(round(cfg.publish_latency_ms * 1e6)))
        self.samples_out += 1
        bias_g = bias_a = None
        if err is not None:
            bias_g = err[:3] + self.gyro_err.bias
            bias_a = err[3:] + self.accel_err.bias
        return ImuOutput(t_true_ns, stamp, release, gyro, accel, bias_g, bias_a)

    def realization(self) -> dict:
        return {"period_s": self.period_s, "gyro": self.gyro_err.realization(), "accel": self.accel_err.realization()}


# ------------------------------------------------------------- profile glue
def gazebo_noise_from_profile(imu: dict, native_rate_hz: float = 1000.0) -> dict[str, list[float]]:
    """Per-axis parameters of Gazebo's native IMU noise for a sensors.yaml IMU.

    ``imu.gazebo_noise`` densities are one-sided (datasheet, per sqrt(Hz));
    the per-sample std of white noise at the native rate is n * sqrt(fs / 2).
    Profiles without ``gazebo_noise`` (the batch A/B/C ones) are converted from
    their ``error`` block, whose density follows the IEEE convention
    (sigma = N / sqrt(dt), i.e. n = sqrt(2) N)."""

    def axes(v) -> list[float]:
        return [float(x) for x in (v if isinstance(v, (list, tuple)) else [v, v, v])]

    out = {}
    gz = imu.get("gazebo_noise")
    for triad in ("gyro", "accel"):
        if gz is not None:
            src = gz.get(triad, {})
            dens = axes(src.get("noise_density", 0.0))
        else:
            src = imu.get("error", {}).get(triad, {})
            dens = [math.sqrt(2.0) * d for d in axes(src.get("noise_density", 0.0))]
        out[f"{triad}_white_std"] = [d * math.sqrt(0.5 * native_rate_hz) for d in dens]
        out[f"{triad}_bias_std"] = axes(src.get("turn_on_bias_std", 0.0))
        out[f"{triad}_dynamic_bias_std"] = axes(src.get("bias_instability_std", 0.0))
        out[f"{triad}_dynamic_bias_tau"] = axes(src.get("bias_correlation_time_s", 100.0))
    return out


def chip_config_from_profile(imu: dict, vibration: dict | None, seed: int, native_rate_hz: float = 1000.0) -> ImuChipConfig:
    """ImuChipConfig of a sensors.yaml IMU block (native backend).

    Noise and bias terms that Gazebo applies natively are zeroed here; the
    random walk is kept only for profiles without ``gazebo_noise``."""
    native = imu.get("gazebo_noise") is not None

    def triad(name: str) -> TriadErrorConfig:
        e = dict(imu.get("error", {}).get(name, {}))
        return TriadErrorConfig(
            noise_density=0.0, turn_on_bias_std=0.0, bias_instability_std=0.0,
            bias_random_walk=0.0 if native else float(e.get("bias_random_walk", 0.0)),
            scale_factor_std=float(e.get("scale_factor_std", 0.0)),
            misalignment_std_rad=float(e.get("misalignment_std_rad", 0.0)),
            g_sensitivity_std=float(e.get("g_sensitivity_std", 0.0)),
            full_scale=float(e.get("full_scale", 0.0)), adc_bits=int(e.get("adc_bits", 0)),
        )

    dl = imu.get("dlpf")
    g_hz = float(dl["gyro_hz"]) if dl else float(imu.get("dlpf_hz", 0.0))
    a_hz = float(dl["accel_hz"]) if dl else float(imu.get("dlpf_hz", 0.0))
    ts = imu.get("timestamp", {})
    vib = VibrationConfig()
    if vibration:
        vib = VibrationConfig(
            accel_rms_per_mps=float(vibration.get("accel_rms_per_mps", 0.0)),
            gyro_rms_per_mps=float(vibration.get("gyro_rms_per_mps", 0.0)),
            band_hz=tuple(float(b) for b in vibration.get("band_hz", (12.0, 160.0))),
            z_scale=float(vibration.get("z_scale", 1.0)),
            wheel_harmonic_mps2=float(vibration.get("wheel_harmonic_mps2", 0.0)),
            gain=float(imu.get("vibration_gain", 1.0)),
        )
    return ImuChipConfig(
        odr_hz=float(imu["rate_hz"]), clock_ppm_std=float(imu.get("clock_ppm_std", 0.0)), native_rate_hz=native_rate_hz,
        dlpf_gyro_hz=g_hz, dlpf_accel_hz=a_hz,
        timestamp_offset_ms=float(ts.get("offset_ms", 0.0)), timestamp_jitter_std_ms=float(ts.get("jitter_std_ms", 0.0)),
        publish_latency_ms=float(imu.get("publish_latency_ms", 0.0)),
        dropout_prob=float(imu.get("dropout_prob", 0.0)), dropout_burst_prob=float(imu.get("dropout_burst_prob", 0.0)),
        dropout_burst_len=tuple(int(x) for x in imu.get("dropout_burst_len", (2, 4))),
        gyro=triad("gyro"), accel=triad("accel"), vibration=vib, seed=int(seed),
    )
