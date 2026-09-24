"""Noisy LiDAR / IMU streams synthesized from the exact simulator state.

Input: the 1 kHz log of one trajectory (``gz_runner``): exact pose, twist and
accelerations of ``base_link``. Output: one sensor stream per profile with

* true acquisition instants on the 1 ms physics grid, so every sample has an
  exact ground-truth pose (no interpolation, no odometry);
* reported timestamps = true instant + latency + jitter (int nanoseconds);
* IMU samples: rigid-body kinematics at the true mount (lever arm), sprung
  body roll/pitch, chassis vibration, on-chip low-pass, sample clock drift,
  the ``apex_fusion_research`` IMU error model (white noise, turn-on bias,
  bias instability, random walk, scale, misalignment, g-sensitivity,
  quantization, saturation), dropouts;
* LiDAR scans: rolling acquisition (each beam from the pose at its own
  instant), exact ray casting against the world primitives, the
  ``apex_fusion_research`` LiDAR noise model (heteroscedastic range noise,
  incidence, calibration errors, angular jitter, missing returns, short and
  random outliers, quantization), lost scans.

Randomness: body dynamics and vibration depend on the trajectory key (same
lap, same physics); sensor errors depend on (trajectory key, sensor profile).
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
from typing import Any
import zlib

import numpy as np
from scipy import signal
import yaml

from .raycast import RayCaster
from .tracks import Track

PACKAGE_DIR = Path(__file__).resolve().parent
FUSION_SRC = PACKAGE_DIR.parents[1] / "ros2_ws" / "src" / "apex_fusion_research"
if str(FUSION_SRC) not in sys.path:
    sys.path.insert(0, str(FUSION_SRC))

from apex_fusion_research.core.config_io import dataclass_from_flat  # noqa: E402
from apex_fusion_research.core.imu_error import ImuErrorConfig, ImuErrorModel  # noqa: E402
from apex_fusion_research.core.lidar_noise import LidarNoiseConfig, LidarNoiseModel  # noqa: E402


def load_sensor_profiles(path: Path | None = None) -> dict[str, dict[str, Any]]:
    return yaml.safe_load((path or PACKAGE_DIR / "config" / "sensors.yaml").read_text(encoding="utf-8"))["profiles"]


def rng_for(*parts: Any) -> np.random.Generator:
    words = [int(p) if isinstance(p, (int, np.integer)) else zlib.crc32(str(p).encode()) for p in parts]
    return np.random.default_rng(np.random.SeedSequence(words))


def seed_int(*parts: Any) -> int:
    words = [int(p) if isinstance(p, (int, np.integer)) else zlib.crc32(str(p).encode()) for p in parts]
    return int(np.random.SeedSequence(words).generate_state(1)[0])


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


# ------------------------------------------------------------------ rotations
def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(N, 4) [w, x, y, z] -> (N, 3, 3)."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def rpy_to_mat(roll: np.ndarray | float, pitch: np.ndarray | float, yaw: np.ndarray | float) -> np.ndarray:
    roll, pitch, yaw = np.broadcast_arrays(np.asarray(roll, float), np.asarray(pitch, float), np.asarray(yaw, float))
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.stack(
        [
            np.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], -1),
            np.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], -1),
            np.stack([-sp, cp * sr, cp * cr], -1),
        ],
        -2,
    )


def mat_to_quat_xyzw(r: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(r).as_quat()  # x, y, z, w


def quat_rpy(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


# ----------------------------------------------------------------- trajectory
class Trajectory:
    """Exact 1 kHz state of one simulated trajectory."""

    def __init__(self, npz_path: Path) -> None:
        data = np.load(npz_path, allow_pickle=False)
        self.summary = json.loads(str(data["summary"]))
        st = data["state"]
        self.step_ns = int(self.summary["physics_step_ns"])
        self.dt = self.step_ns * 1e-9
        self.n = len(st)
        self.t_ns = np.arange(self.n, dtype=np.int64) * self.step_ns
        self.pos = st[:, 0:3]
        self.quat = st[:, 3:7]  # w, x, y, z
        self.v_w = st[:, 7:10]
        self.w_w = st[:, 10:13]
        # Physics reports a[k] = (v[k] - v[k-1]) / dt; centre it on sample k.
        a = st[:, 13:16]
        self.a_w = 0.5 * (a + np.vstack((a[1:], a[-1:])))
        al = st[:, 16:19]
        self.alpha_w = 0.5 * (al + np.vstack((al[1:], al[-1:])))
        self.ctrl = data["ctrl"]
        self.R = quat_to_mat(self.quat)
        self.record_start = int(self.summary["record_start_ns"] // self.step_ns)

    def body(self, vec_w: np.ndarray) -> np.ndarray:
        return np.einsum("nji,nj->ni", self.R, vec_w)

    def interp_pose(self, t_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pose at arbitrary instants: linear position, normalised-lerp rotation
        between the enclosing 1 ms states (used only for beams inside a scan)."""
        x = np.asarray(t_ns, dtype=float) / self.step_ns
        i0 = np.clip(np.floor(x).astype(int), 0, self.n - 2)
        f = np.clip(x - i0, 0.0, 1.0)[:, None]
        pos = (1 - f) * self.pos[i0] + f * self.pos[i0 + 1]
        q0, q1 = self.quat[i0], self.quat[i0 + 1]
        q1 = np.where((np.sum(q0 * q1, axis=1) < 0)[:, None], -q1, q1)
        q = (1 - f) * q0 + f * q1
        return pos, quat_to_mat(q)


# ------------------------------------------------------------- body dynamics
def body_motion(traj: Trajectory, cfg: dict[str, Any], trajectory_key: str) -> dict[str, np.ndarray]:
    """Sprung-mass roll/pitch and chassis vibration (shared by all profiles of a lap).

    Seeded by the trajectory key (track, motion, seed, variant): every lap has
    its own realisation, and no two laps of different splits share one."""
    rng = rng_for(trajectory_key, "body")
    jitter = float(cfg["gain_jitter"])
    g = float(cfg["vibration"]["gravity_mps2"])
    a_b = traj.body(traj.a_w)
    k_roll = math.radians(float(cfg["roll_gain_deg_per_g"])) * rng.uniform(1 - jitter, 1 + jitter)
    k_pitch = math.radians(float(cfg["pitch_gain_deg_per_g"])) * rng.uniform(1 - jitter, 1 + jitter)
    wn = 2 * math.pi * float(cfg["natural_freq_hz"]) * rng.uniform(1 - jitter / 2, 1 + jitter / 2)
    zeta = float(cfg["damping_ratio"])
    b, a = signal.bilinear([wn * wn], [1.0, 2 * zeta * wn, wn * wn], fs=1.0 / traj.dt)
    # Outward roll in curves (positive roll = right side down for a left turn),
    # squat under acceleration (nose up = negative pitch), dive under braking.
    roll = signal.lfilter(b, a, k_roll * a_b[:, 1] / g)
    pitch = signal.lfilter(b, a, -k_pitch * a_b[:, 0] / g)
    roll_rate = np.gradient(roll, traj.dt)
    pitch_rate = np.gradient(pitch, traj.dt)
    roll_acc = np.gradient(roll_rate, traj.dt)
    pitch_acc = np.gradient(pitch_rate, traj.dt)
    vib = cfg["vibration"]
    speed = np.linalg.norm(traj.v_w[:, :2], axis=1)
    sos = signal.butter(2, vib["band_hz"], btype="bandpass", fs=1.0 / traj.dt, output="sos")
    white = rng.standard_normal((traj.n, 6))
    band = signal.sosfilt(sos, white, axis=0)
    band /= max(float(band[1000:].std()), 1e-12)
    acc_vib = band[:, :3] * float(vib["accel_rms_per_mps"]) * speed[:, None]
    acc_vib[:, 2] *= float(vib["z_scale"])
    gyro_vib = band[:, 3:] * float(vib["gyro_rms_per_mps"]) * speed[:, None]
    wheel_phase = np.cumsum(speed / 0.06) * traj.dt
    acc_vib[:, 2] += float(vib["wheel_harmonic_mps2"]) * speed * np.sin(wheel_phase + rng.uniform(0, 2 * math.pi))
    return {
        "roll": roll, "pitch": pitch, "roll_rate": roll_rate, "pitch_rate": pitch_rate,
        "roll_acc": roll_acc, "pitch_acc": pitch_acc, "acc_vib": acc_vib, "gyro_vib": gyro_vib,
        "params": {"roll_gain_rad_per_g": k_roll, "pitch_gain_rad_per_g": k_pitch, "natural_freq_hz": wn / (2 * math.pi), "damping_ratio": zeta},
    }


def sprung_rotation(body: dict[str, np.ndarray], idx: np.ndarray | None = None) -> np.ndarray:
    r = body["roll"] if idx is None else np.interp(idx, np.arange(len(body["roll"])), body["roll"])
    p = body["pitch"] if idx is None else np.interp(idx, np.arange(len(body["pitch"])), body["pitch"])
    return rpy_to_mat(r, p, np.zeros_like(r))


# ----------------------------------------------------------------- extrinsics
def draw_extrinsic(cfg: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    xyz = np.array(cfg["extrinsic"]["xyz"], dtype=float)
    rpy = np.radians(np.array(cfg["extrinsic"]["rpy_deg"], dtype=float))
    err = cfg.get("extrinsic_error", {})
    xyz_true = xyz + rng.normal(0.0, float(err.get("xyz_std_m", 0.0)), 3)
    rpy_true = rpy + np.radians(rng.normal(0.0, float(err.get("rpy_std_deg", 0.0)), 3))
    r_nom = rpy_to_mat(*rpy)
    r_true = rpy_to_mat(*rpy_true)
    return {
        "xyz": xyz, "R": r_nom, "xyz_true": xyz_true, "R_true": r_true,
        "quat_xyzw": mat_to_quat_xyzw(r_nom).tolist(), "quat_true_xyzw": mat_to_quat_xyzw(r_true).tolist(),
        "rpy_true_rad": rpy_true.tolist(),
    }


def report_times(true_ns: np.ndarray, ts_cfg: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    offset = float(ts_cfg.get("offset_ms", 0.0)) * 1e6
    jitter = float(ts_cfg.get("jitter_std_ms", 0.0)) * 1e6
    rep = true_ns.astype(np.float64) + offset + rng.normal(0.0, jitter, len(true_ns))
    rep = np.round(rep).astype(np.int64)
    # Drivers never publish out-of-order stamps.
    for i in range(1, len(rep)):
        if rep[i] <= rep[i - 1]:
            rep[i] = rep[i - 1] + 1000
    return rep


# ----------------------------------------------------------------------- IMU
def synth_imu(traj: Trajectory, body: dict[str, np.ndarray], cfg: dict[str, Any], rng: np.random.Generator, seed_words: tuple, vib_gain_scale: float = 1.0) -> dict[str, Any]:
    ext = draw_extrinsic(cfg, rng)
    g_w = np.array([0.0, 0.0, -9.8])
    r_lever = ext["xyz_true"]
    w_b = traj.body(traj.w_w)
    al_b = traj.body(traj.alpha_w)
    a_b = traj.body(traj.a_w)
    # Rigid-body acceleration at the mount point, in base_link.
    a_pt = a_b + np.cross(al_b, r_lever) + np.cross(w_b, np.cross(w_b, r_lever))
    f_b = a_pt - traj.body(np.broadcast_to(g_w, traj.a_w.shape))
    # Sprung body: rotate into the body frame, add roll/pitch rates.
    r_s = sprung_rotation(body)
    f_s = np.einsum("nji,nj->ni", r_s, f_b)
    w_s = np.einsum("nji,nj->ni", r_s, w_b)
    w_s[:, 0] += body["roll_rate"]
    w_s[:, 1] += body["pitch_rate"]
    # Mount rotation (true extrinsic).
    f_m = f_s @ ext["R_true"]
    w_m = w_s @ ext["R_true"]
    gain = float(cfg.get("vibration_gain", 1.0)) * vib_gain_scale
    f_m = f_m + gain * body["acc_vib"]
    w_m = w_m + gain * body["gyro_vib"]
    # On-chip digital low-pass (causal, with its real group delay).
    sos = signal.butter(2, float(cfg["dlpf_hz"]), btype="low", fs=1.0 / traj.dt, output="sos")
    zi_f = signal.sosfilt_zi(sos)[:, :, None] * f_m[0][None, None, :]
    zi_w = signal.sosfilt_zi(sos)[:, :, None] * w_m[0][None, None, :]
    f_filt, _ = signal.sosfilt(sos, f_m, axis=0, zi=zi_f)
    w_filt, _ = signal.sosfilt(sos, w_m, axis=0, zi=zi_w)

    # Sample clock with a constant frequency error, true instants on the 1 ms grid.
    nominal_ns = 1e9 / float(cfg["rate_hz"])
    if abs(nominal_ns / traj.step_ns - round(nominal_ns / traj.step_ns)) > 1e-9:
        raise ValueError(
            f"IMU rate {cfg['rate_hz']} Hz: the period must be a whole number of {traj.step_ns / 1e6:g} ms physics steps "
            "(e.g. 50, 100, 125, 200, 250, 500 or 1000 Hz), otherwise the sampling intervals alternate"
        )
    period_ns = nominal_ns * (1.0 + rng.normal(0.0, float(cfg["clock_ppm_std"])) * 1e-6)
    start = traj.t_ns[traj.record_start] + rng.uniform(0.0, period_ns)
    end = traj.t_ns[-1]
    t_true = start + period_ns * np.arange(int((end - start) // period_ns) + 1)
    idx = np.unique(np.round(t_true / traj.step_ns).astype(np.int64))
    idx = idx[(idx >= traj.record_start) & (idx < traj.n)]

    err_cfg = dataclass_from_flat(ImuErrorConfig, _flatten({"enabled": True, "seed": seed_int(*seed_words), **cfg["error"]}))
    model = ImuErrorModel(err_cfg)
    init_bias = cfg.get("initial_bias", {})
    model.gyro.turn_on_bias = model.gyro.turn_on_bias + np.array(init_bias.get("gyro_rps", [0, 0, 0]), dtype=float)
    model.accel.turn_on_bias = model.accel.turn_on_bias + np.array(init_bias.get("accel_mps2", [0, 0, 0]), dtype=float)
    meas = np.zeros((len(idx), 6))
    bias = np.zeros((len(idx), 6))
    dts = np.diff(idx, prepend=idx[0] - int(round(period_ns / traj.step_ns))) * traj.dt
    saturated = 0
    fs_g, fs_a = float(cfg["error"]["gyro"].get("full_scale", 0.0)), float(cfg["error"]["accel"].get("full_scale", 0.0))
    for i, (k, dt_i) in enumerate(zip(idx, dts)):
        s = model.measure(w_filt[k], f_filt[k], float(dt_i))
        meas[i, :3] = s.specific_force
        meas[i, 3:] = s.angular_velocity
        bias[i, :3] = s.accel_bias
        bias[i, 3:] = s.gyro_bias
        if (fs_a > 0 and np.any(np.abs(s.specific_force) >= fs_a * 0.9999)) or (fs_g > 0 and np.any(np.abs(s.angular_velocity) >= fs_g * 0.9999)):
            saturated += 1
    # Dropouts: isolated samples and bursts.
    keep = rng.random(len(idx)) >= float(cfg.get("dropout_prob", 0.0))
    bursts = []
    p_burst = float(cfg.get("dropout_burst_prob", 0.0))
    lo, hi = cfg.get("dropout_burst_len", [2, 4])
    for i in np.nonzero(rng.random(len(idx)) < p_burst)[0]:
        length = int(rng.integers(int(lo), int(hi) + 1))
        keep[i : i + length] = False
        bursts.append((int(idx[i]), length))
    rep = report_times(idx[keep] * traj.step_ns, cfg.get("timestamp", {}), rng)
    return {
        "true_idx": idx[keep],
        "t_report_ns": rep,
        "values": meas[keep],  # ax, ay, az, gx, gy, gz (sensor frame)
        "bias_true": bias[keep],  # bax, bay, baz, bgx, bgy, bgz
        "n_expected": int(len(idx)),
        "n_dropped": int(np.count_nonzero(~keep)),
        "bursts": bursts,
        "saturated": saturated,
        "extrinsic": ext,
        "realization": model.realization(),
        "period_ns": period_ns,
    }


# --------------------------------------------------------------------- LiDAR
def synth_lidar(traj: Trajectory, body: dict[str, np.ndarray], cfg: dict[str, Any], caster: RayCaster, rng: np.random.Generator, seed_words: tuple) -> dict[str, Any]:
    ext = draw_extrinsic(cfg, rng)
    beams = int(cfg["beams"])
    fov = math.radians(float(cfg["fov_deg"]))
    angle_min = math.radians(float(cfg["angle_min_deg"]))
    full = abs(fov - 2 * math.pi) < 1e-9
    inc = fov / beams if full else fov / (beams - 1)
    angles = angle_min + inc * np.arange(beams)
    rmin, rmax = float(cfg["range_min_m"]), float(cfg["range_max_m"])
    period_nom = 1e9 / float(cfg["rate_hz"])
    rolling = bool(cfg.get("rolling_scan", True))
    noise_cfg = dataclass_from_flat(LidarNoiseConfig, {"enabled": True, "seed": seed_int(*seed_words), **cfg["noise"]})
    model = LidarNoiseModel(noise_cfg)

    local_dirs = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(beams)))
    t = traj.t_ns[traj.record_start] + rng.uniform(0.0, period_nom)
    starts, ends, ranges_out, masks, outcomes, periods = [], [], [], [], [], []
    lost = []
    while True:
        period = period_nom * (1.0 + rng.normal(0.0, float(cfg.get("rate_jitter_rel", 0.0))))
        sweep = period * (fov / (2 * math.pi))
        k0 = int(round(t / traj.step_ns))
        beam_dt = sweep / beams
        if (k0 + int(math.ceil(sweep / traj.step_ns)) + 1) >= traj.n:
            break
        t0 = k0 * traj.step_ns  # true start of the sweep, on the physics grid
        if rolling:
            tb = t0 + beam_dt * np.arange(beams)
        else:
            tb = np.full(beams, float(t0))
        pos, rot = traj.interp_pose(tb)
        rs = sprung_rotation(body, tb / traj.step_ns)
        r_sensor = rot @ rs @ ext["R_true"]
        origins = pos + np.einsum("nij,nj->ni", rot @ rs, np.broadcast_to(ext["xyz_true"], (beams, 3)))
        dirs = np.einsum("nij,nj->ni", r_sensor, local_dirs)
        ideal = caster.cast(origins, dirs, max_range=rmax * 1.5)
        k_end = int(round((t0 + beam_dt * (beams - 1)) / traj.step_ns))
        if rng.random() < float(cfg.get("scan_dropout_prob", 0.0)):
            lost.append(k0)
            model.apply(ideal, angle_min, inc, rmin, rmax)  # keep the noise stream aligned
        else:
            res = model.apply(ideal, angle_min, inc, rmin, rmax)
            r = res.ranges.astype(np.float32)
            # REP-117 for every beam, including those without an ideal return
            # inside [range_min, range_max] (passed through by the noise model).
            r = np.where(np.isfinite(r) & (r < rmin), np.float32(-np.inf), r)
            r = np.where(np.isfinite(r) & (r > rmax), np.float32(np.inf), r)
            valid = np.isfinite(r)
            starts.append(k0)
            ends.append(k_end)
            ranges_out.append(r)
            masks.append(valid)
            outcomes.append(res.outcome.astype(np.uint8))
            periods.append(period)
        t = t0 + period
    starts_a = np.array(starts, dtype=np.int64)
    rep = report_times(starts_a * traj.step_ns, cfg.get("timestamp", {}), rng)
    return {
        "start_idx": starts_a,
        "end_idx": np.array(ends, dtype=np.int64),
        "t_report_ns": rep,
        "ranges": np.array(ranges_out, dtype=np.float32).reshape(-1, beams),
        "valid": np.array(masks, dtype=bool).reshape(-1, beams),
        "outcome": np.array(outcomes, dtype=np.uint8).reshape(-1, beams),
        "period_ns": np.array(periods),
        "lost_scans": lost,
        "angle_min": angle_min,
        "angle_increment": inc,
        "range_min": rmin,
        "range_max": rmax,
        "beams": beams,
        "time_increment_ns": (period_nom * (fov / (2 * math.pi)) / beams) if rolling else 0.0,
        "extrinsic": ext,
        "realization": model.realization(),
    }


# --------------------------------------------------------------- ground truth
def ground_truth_rows(traj: Trajectory, body: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    idx = np.asarray(idx, dtype=np.int64)
    q = traj.quat[idx]
    roll, pitch, yaw = quat_rpy(q)
    v_w = traj.v_w[idx]
    rot = traj.R[idx]
    v_b = np.einsum("nji,nj->ni", rot, v_w)
    w_b = np.einsum("nji,nj->ni", rot, traj.w_w[idx])
    return {
        "idx": idx,
        "t_ns": traj.t_ns[idx],
        "xyz": traj.pos[idx],
        "quat_xyzw": np.column_stack((q[:, 1], q[:, 2], q[:, 3], q[:, 0])),
        "rpy": np.column_stack((roll, pitch, yaw)),
        "v_world": v_w,
        "v_body": v_b,
        "w_body": w_b,
        "a_world": traj.a_w[idx],
        "body_roll_pitch": np.column_stack((body["roll"][idx], body["pitch"][idx])),
    }


def synthesize_run(traj: Trajectory, track: Track, profile_name: str, profile: dict[str, Any], trajectory_key: str, body: dict[str, np.ndarray], caster: RayCaster | None = None) -> dict[str, Any]:
    """All streams of one (trajectory, sensor profile) run.

    Sensor randomness (biases, miscalibration, white noise, dropouts, stamps)
    is seeded by (trajectory key, profile): each run has its own sensor
    realisation, never reused on another track, motion or seed."""
    caster = caster or RayCaster(track.geometry)
    base = (trajectory_key, profile_name)
    imu = synth_imu(traj, body, profile["imu"], rng_for(*base, "imu"), base + ("imu_error",))
    lidar = synth_lidar(traj, body, profile["lidar"], caster, rng_for(*base, "lidar"), base + ("lidar_noise",))
    gt_idx = np.unique(np.concatenate((imu["true_idx"], lidar["start_idx"], lidar["end_idx"])))
    gt = ground_truth_rows(traj, body, gt_idx)
    return {"imu": imu, "lidar": lidar, "gt": gt, "profile": copy.deepcopy(profile)}
