"""Causal, real-time streaming odometry for a live robot (simulation or car).

``LiveOdometry`` consumes the raw sensor messages one by one, as they arrive
(IMU samples and complete LiDAR revolutions, each with its timestamp), and
runs the streaming network of ``streaming_model`` (and, for a hybrid, the
classical odometry of ``icp_odometry`` it takes as input) exactly as it was
trained. Every scan interval is assembled with the rules of the training data
(``sqlite_windows.build_split`` and ``sqlite_streams.StreamData.gather``):

* interval k goes from the stamp of scan k-1 to the stamp of scan k (stamp =
  first sample of the revolution); its IMU samples are those with a reported
  stamp in (t_{k-1}, t_k], at most 40, or the nearest one when none arrived;
* IMU normalisation [ax/G, ay/G, (az - G)/G, gx, gy, gz] and the ranges as
  range / range_max, both rounded to float16 like the stored dataset;
* dt between the reported stamps and the trapezoidal gyro-z integral over the
  interval (interpolated at both ends), and its heading profile over the
  sweeps of scans k-1 and k (``sqlite_streams.gyro_profile``);
* yaw rate during the sweep of scan k-1 (= the interval itself) and of scan
  k, the revolution that follows its stamp: the scan is processed once the
  IMU covers that revolution (``scan_time``), or after ``imu_wait_s`` (sim or
  robot clock) with what arrived;
* IMU stamps moved back by the chip filter delay (``imu_group_delay_s``)
  when the checkpoint was trained with ``imu_delay_comp``;
* per-beam firing time from the LiDAR calibration (rotation direction and
  sync angle), mean specific force, specific-force spread;
* the classical odometry in full FP32 and the network with its persistent
  state (GRU, body velocity, IMU biases).

Outputs per scan: the increment and its 1-sigma, the pose in the odometry
frame (base_link at the first scan) stamped at the scan, the velocity and
biases, the scan de-skewed to that stamp (for a SLAM), and on request the pose
propagated with the IMU up to the latest sample (``predict``, for control).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import torch

from icp_odometry import ICP_VARIANTS, IcpStream, deskew, fp32_matmul
from sqlite_streams import bin_time_fraction, gyro_profile, icp_features, imu_interval_stats
from sqlite_windows import G, _trapz
from streaming_model import load_checkpoint

UNITS = (0.01, 0.01, 0.001)  # sigma units of the network (cm, cm, mrad)
MAX_IMU = 40


@dataclass
class LidarCalibration:
    """What the model needs to know about the LiDAR (nominal, from the
    datasheet / mount calibration, never the true simulated values)."""

    beams: int = 360
    angle_min: float = -math.pi
    range_min: float = 0.15
    range_max: float = 12.0
    rate_hz: float = 13.0  # nominal revolution rate
    scan_direction: str = "ccw"
    scan_start_angle_deg: float = -180.0
    xy: tuple[float, float] = (0.18, 0.0)  # mount in base_link
    sigma_const_m: float = 0.003  # datasheet range noise: const + prop * range
    sigma_prop: float = 0.005

    @property
    def angle_increment(self) -> float:
        return 2.0 * math.pi / self.beams

    @property
    def time_increment(self) -> float:
        return 1.0 / self.rate_hz / self.beams

    def beam_time_frac(self) -> np.ndarray:
        return bin_time_fraction(self.beams, self.angle_min, self.angle_increment, self.scan_direction, math.radians(self.scan_start_angle_deg))

    @staticmethod
    def from_profile(lidar: dict[str, Any]) -> "LidarCalibration":
        """From a ``tools/pose_dataset/config/sensors.yaml`` LiDAR block."""
        noise = lidar.get("noise", {})
        return LidarCalibration(
            beams=int(lidar["beams"]), angle_min=math.radians(float(lidar.get("angle_min_deg", -180.0))),
            range_min=float(lidar["range_min_m"]), range_max=float(lidar["range_max_m"]), rate_hz=float(lidar["rate_hz"]),
            scan_direction=str(lidar.get("scan_direction", "ccw")),
            scan_start_angle_deg=float(lidar.get("scan_start_angle_deg", lidar.get("angle_min_deg", -180.0))),
            xy=tuple(float(v) for v in lidar["extrinsic"]["xyz"][:2]),
            sigma_const_m=float(noise.get("range_sigma_const_m", 0.0)), sigma_prop=float(noise.get("range_sigma_prop", 0.0)),
        )


@dataclass
class Estimate:
    stamp_ns: int  # scan stamp (start of its revolution)
    delta: np.ndarray  # [dx, dy, dyaw] from scan k-1 to scan k, frame of scan k-1
    sigma: np.ndarray  # 1-sigma of delta
    pose: np.ndarray  # [x, y, yaw] of base_link in the odometry frame at the stamp
    velocity: np.ndarray  # body-frame velocity at the stamp [m/s]
    bias: np.ndarray  # gyro z [rad/s], accel x, y [m/s^2]
    dt: float
    imu_samples: int
    deskewed_ranges: np.ndarray  # scan k de-skewed to its stamp, laser frame, 1-degree bins (inf = none)
    icp: dict[str, float] = field(default_factory=dict)
    compute_ms: float = 0.0


class LiveOdometry:
    def __init__(self, checkpoint: str | Path, lidar: LidarCalibration, device: str = "cuda", imu_wait_s: float = 0.03,
                 imu_rate_hz: float | None = None, imu_group_delay_s: float = 0.0) -> None:
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.model, ck = load_checkpoint(checkpoint, self.device)
        self.hybrid = bool(self.model.hybrid)
        self.icp_cfg = ICP_VARIANTS[ck["config"].get("icp_variant", "scan")] if self.hybrid else None
        # IMU stamps moved back by the chip filter delay of the sensor profile
        # (sqlite_windows.imu_group_delay_s), as in training.
        self.imu_delay_ns = int(round(imu_group_delay_s * 1e9)) if ck["config"].get("imu_delay_comp", False) else 0
        self.lidar = lidar
        self.imu_wait_ns = int(imu_wait_s * 1e9)
        self.imu_rate_hz = imu_rate_hz
        dev = self.device
        frac = lidar.beam_time_frac()
        linear = np.arange(lidar.beams) / lidar.beams
        self.beam_frac = None if np.allclose(frac, linear, atol=1e-6) else torch.tensor(frac, dtype=torch.float32, device=dev)
        self.t_inc = torch.tensor([lidar.time_increment], device=dev)
        self.xy = torch.tensor([lidar.xy], device=dev, dtype=torch.float32)
        self.amin = torch.tensor([lidar.angle_min], device=dev)
        self.ainc = torch.tensor([lidar.angle_increment], device=dev)
        beam_time = None if self.beam_frac is None else (self.beam_frac * lidar.beams * lidar.time_increment)[None]
        self.geo = (self.t_inc, self.xy, self.amin, self.ainc, beam_time)
        if self.hybrid:
            sig = torch.tensor([[lidar.sigma_const_m, lidar.sigma_prop]], device=dev)
            self.icp = IcpStream(self.icp_cfg, self.geo, sig, lidar.beams, dev)
        self.state = self.model.initial_state(1, dev)
        self._imu_t: deque[int] = deque()
        self._imu_v: deque[np.ndarray] = deque()  # raw ax ay az gx gy gz
        self._pending: deque[tuple[int, np.ndarray, float, int]] = deque()  # stamp, ranges, scan_time, arrival
        self._prev: tuple[int, torch.Tensor, torch.Tensor] | None = None  # stamp, metric ranges, valid
        self._prev_dt: float | None = None
        self._prev_gyro: float | None = None
        self.pose = np.zeros(3)
        self.velocity = np.zeros(2)
        self.bias = np.zeros(3)
        self.last_stamp_ns: int | None = None
        self.scans = 0
        self.health = _Health(lidar.rate_hz, imu_rate_hz)
        # The IMU side (add_imu, predict) and the scan side (process) may run
        # in different threads: the propagation must not wait for a scan.
        self.lock = threading.RLock()

    # ----------------------------------------------------------------- input
    def add_imu(self, stamp_ns: int, gyro: np.ndarray, accel: np.ndarray) -> None:
        """One raw IMU sample: angular rate [rad/s] and specific force
        [m/s^2, gravity included] in the sensor frame."""
        with self.lock:
            t = int(stamp_ns) - self.imu_delay_ns
            if self._imu_t and t <= self._imu_t[-1]:
                self.health.count("imu_out_of_order")
                return
            self._imu_t.append(t)
            self._imu_v.append(np.array([accel[0], accel[1], accel[2], gyro[0], gyro[1], gyro[2]], dtype=np.float64))
            self.health.imu(int(stamp_ns))
            horizon = int(stamp_ns) - 3_000_000_000  # keep 3 s
            while len(self._imu_t) > 2 and self._imu_t[0] < horizon:
                self._imu_t.popleft()
                self._imu_v.popleft()

    def add_scan(self, stamp_ns: int, ranges: np.ndarray, scan_time_s: float, arrival_ns: int | None = None) -> None:
        """One complete revolution (REP-117 ranges, ``beams`` bins from
        ``angle_min``), stamped at its first sample."""
        r = np.asarray(ranges, dtype=np.float32)
        if r.shape[-1] != self.lidar.beams:
            raise ValueError(f"expected {self.lidar.beams} beams, got {r.shape[-1]}")
        with self.lock:
            self._pending.append((int(stamp_ns), r, float(scan_time_s or 1.0 / self.lidar.rate_hz), int(arrival_ns if arrival_ns is not None else stamp_ns)))
            self.health.scan(int(stamp_ns), r, self.lidar)

    def process(self, now_ns: int) -> list[Estimate]:
        """Process every pending scan whose revolution is covered by the IMU
        (or whose wait expired at ``now_ns``)."""
        out = []
        while True:
            with self.lock:
                if not self._pending:
                    break
                stamp, r, scan_time, arrival = self._pending[0]
                covered = bool(self._imu_t) and self._imu_t[-1] >= stamp + int(scan_time * 1e9)
                if not covered and now_ns - arrival < self.imu_wait_ns:
                    break
                if not covered:
                    self.health.count("imu_wait_timeout")
                self._pending.popleft()
            est = self._scan(stamp, r, scan_time)
            if est is not None:
                out.append(est)
        return out

    # ------------------------------------------------------------- internals
    def _metric(self, r: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Ranges as stored in the dataset: float16(r / range_max) * range_max."""
        rmax = self.lidar.range_max
        valid = np.isfinite(r) & (r > 0.0)
        q = np.where(valid, r.astype(np.float64) / rmax, 1.0).astype(np.float16).astype(np.float32)
        ranges = torch.tensor(q, device=self.device) * torch.tensor(rmax, dtype=torch.float32, device=self.device)
        v = torch.tensor(valid, device=self.device)
        return torch.where(v, ranges, torch.zeros_like(ranges)), v

    def _imu_between(self, t0: int, t1: int) -> np.ndarray:
        t = np.fromiter(self._imu_t, dtype=np.int64, count=len(self._imu_t))
        lo = int(np.searchsorted(t, t0, side="right"))
        hi = int(np.searchsorted(t, t1, side="right"))
        if hi <= lo:
            nearest = int(np.clip(np.searchsorted(t, t1), 0, len(t) - 1))
            lo, hi = nearest, nearest + 1
        if hi - lo > MAX_IMU:
            lo = hi - MAX_IMU
        return np.array([self._imu_v[i] for i in range(lo, hi)])

    def _gyro_integral(self, t0: int, t1: int) -> float:
        t = np.fromiter(self._imu_t, dtype=np.int64, count=len(self._imu_t)) * 1e-9
        gz = np.array([v[5] for v in self._imu_v])
        a, b = t0 * 1e-9, t1 * 1e-9
        inner = t[(t > a) & (t < b)]
        ts = np.concatenate(([a], inner, [b]))
        return float(_trapz(np.interp(ts, t, gz), ts))

    def _gyro_profile(self, t0: int, t1: int) -> np.ndarray:
        t = np.fromiter(self._imu_t, dtype=np.int64, count=len(self._imu_t)) * 1e-9
        gz = np.array([v[5] for v in self._imu_v])
        return gyro_profile(t, gz, t0 * 1e-9, t1 * 1e-9).astype(np.float32)

    def _scan(self, stamp: int, r: np.ndarray, scan_time: float) -> Estimate | None:
        cur_r, cur_v = self._metric(r)
        with self.lock:
            if self._prev is None or not self._imu_t:
                self._prev = (stamp, cur_r, cur_v)
                self.last_stamp_ns = stamp
                return None
            t_start = time.perf_counter()
            t_prev, prev_r, prev_v = self._prev
            # Same precision as the stored split: float32 dt and gyro
            # integrals, rates divided in float32.
            dt = np.float32(max((stamp - t_prev) * 1e-9, 1e-4))
            gyro = np.float32(self._gyro_integral(t_prev, stamp))
            sweep_end = stamp + int(round(scan_time * 1e9))
            dt_next = np.float32(max(scan_time, 1e-4))
            gyro_next = np.float32(self._gyro_integral(stamp, sweep_end))
            prof_prev, prof_cur = self._gyro_profile(t_prev, stamp), self._gyro_profile(stamp, sweep_end)
            raw = self._imu_between(t_prev, stamp)
        dev = self.device
        w_prev, w_cur = float(gyro / dt), float(gyro_next / dt_next)
        norm = raw.copy()
        norm[:, 2] -= G
        norm[:, :3] /= G
        # Zero-padded like the training batches: the IMU encoder convolves
        # before masking, so the last samples see the padding.
        padded = np.zeros((MAX_IMU, 6), dtype=np.float16)
        padded[: len(norm)] = norm
        imu = torch.tensor(padded.astype(np.float32), device=dev)[None, None]  # [1, 1, MAX_IMU, 6]
        lengths = torch.tensor([[len(norm)]], device=dev)
        accel, accel_std = imu_interval_stats(imu, lengths)
        f = lambda v: torch.tensor([[v]], dtype=torch.float32, device=dev)  # noqa: E731
        batch = {
            "prev_ranges": prev_r[None, None], "prev_valid": prev_v[None, None], "cur_ranges": cur_r[None, None], "cur_valid": cur_v[None, None],
            "w_prev": f(w_prev), "w_cur": f(w_cur), "time_increment": self.t_inc[None], "imu": imu, "imu_lengths": lengths,
            "accel": accel, "accel_std": accel_std, "gyro_integral": f(gyro), "dt": f(dt),
            "prof_prev": torch.tensor(prof_prev, device=dev)[None, None], "prof_cur": torch.tensor(prof_cur, device=dev)[None, None], "dt_next": f(dt_next),
        }
        if self.beam_frac is not None:
            batch["beam_frac"] = self.beam_frac[None, None]
        icp_info: dict[str, float] = {}
        with torch.no_grad():
            if self.hybrid:
                b1 = {k: v[:, 0] for k, v in batch.items() if k not in ("imu", "imu_lengths", "beam_frac")}
                with fp32_matmul():
                    res = self.icp.step(b1, torch.ones(1, dtype=torch.bool, device=dev))
                pred, feat = icp_features(res)
                batch["icp_pred"], batch["icp_feat"] = pred[None], feat[None]
                icp_info = {k: float(res[k][0]) for k in ("standstill", "eig_ratio", "gyro_bias", "map_used")}
                icp_info.update(icp_dx=float(res["pred"][0, 0]), icp_dy=float(res["pred"][0, 1]), icp_dyaw=float(res["pred"][0, 2]))
            out, self.state = self.model(batch, self.state)
        delta = out["delta"][0, 0].double().cpu().numpy()
        sigma = (torch.exp(0.5 * out["logvar"][0, 0]) * torch.tensor(UNITS, device=dev)).cpu().numpy()
        velocity = out["velocity"][0, 0].double().cpu().numpy()
        bias = out["bias"][0, 0].double().cpu().numpy()
        # Scan k de-skewed to its stamp with the state at the stamp.
        with torch.no_grad(), fp32_matmul():
            pts = deskew(cur_r[None], cur_v[None], torch.tensor([w_cur - bias[0]], dtype=torch.float32, device=dev),
                         torch.tensor(velocity[None], dtype=torch.float32, device=dev), *self.geo)[0]
        deskewed = self._to_scan(pts.cpu().numpy(), cur_v.cpu().numpy())
        with self.lock:
            c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
            self.pose = np.array([self.pose[0] + c * delta[0] - s * delta[1], self.pose[1] + s * delta[0] + c * delta[1], self.pose[2] + delta[2]])
            self.velocity, self.bias = velocity, bias
            self._prev = (stamp, cur_r, cur_v)
            self.last_stamp_ns = stamp
            self.scans += 1
            pose = self.pose.copy()
        compute_ms = 1e3 * (time.perf_counter() - t_start)
        return Estimate(stamp, delta, sigma, pose, velocity.copy(), bias.copy(), float(dt), len(norm), deskewed, icp_info, compute_ms)

    def _to_scan(self, pts: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """base_link points -> ranges in the laser frame on the scan's bins."""
        lid = self.lidar
        x = pts[:, 0] - lid.xy[0]
        y = pts[:, 1] - lid.xy[1]
        r = np.hypot(x, y)
        ok = valid & (r >= lid.range_min) & (r <= lid.range_max)
        idx = np.round((np.mod(np.arctan2(y, x) - lid.angle_min, 2 * math.pi)) / lid.angle_increment).astype(int) % lid.beams
        out = np.full(lid.beams, np.inf, dtype=np.float32)
        np.minimum.at(out, idx[ok], r[ok].astype(np.float32))
        return out

    # ------------------------------------------------------------ prediction
    def predict(self) -> tuple[int, np.ndarray, np.ndarray, float] | None:
        """Pose, body velocity and yaw rate propagated with the bias-corrected
        IMU from the last scan stamp to the latest IMU sample:
        (stamp, pose, velocity, yaw rate)."""
        with self.lock:
            if self.last_stamp_ns is None or not self._imu_t:
                return None
            pose = self.pose.copy()
            v = self.velocity.copy()
            bias = self.bias.copy()
            t_prev = self.last_stamp_ns
            samples = [(t, val) for t, val in zip(self._imu_t, self._imu_v) if t > t_prev]
        rate = 0.0
        for t, val in samples:
            dt = (t - t_prev) * 1e-9
            a = val[:2] - bias[1:]
            rate = val[5] - bias[0]
            dyaw = rate * dt
            c, s = math.cos(pose[2]), math.sin(pose[2])
            pose[0] += (c * v[0] - s * v[1]) * dt
            pose[1] += (s * v[0] + c * v[1]) * dt
            v = v + a * dt
            cd, sd = math.cos(dyaw), math.sin(dyaw)
            v = np.array([cd * v[0] + sd * v[1], -sd * v[0] + cd * v[1]])
            pose[2] += dyaw
            t_prev = t
        return t_prev, pose, v, rate


class _Health:
    """Online check of the sensor streams against the calibration (rates,
    jitter, valid bins): the same criteria as the timing audit of the car."""

    def __init__(self, lidar_hz: float, imu_hz: float | None) -> None:
        self.lidar_hz, self.imu_hz = lidar_hz, imu_hz
        self.imu_t: deque[int] = deque(maxlen=400)
        self.scan_t: deque[int] = deque(maxlen=60)
        self.valid: deque[float] = deque(maxlen=60)
        self.counters: dict[str, int] = {}

    def count(self, key: str) -> None:
        self.counters[key] = self.counters.get(key, 0) + 1

    def imu(self, t: int) -> None:
        self.imu_t.append(t)

    def scan(self, t: int, r: np.ndarray, lid: LidarCalibration) -> None:
        self.scan_t.append(t)
        self.valid.append(float(np.mean(np.isfinite(r))))

    def report(self) -> dict[str, Any]:
        def rate(ts) -> tuple[float, float]:
            if len(ts) < 3:
                return 0.0, 0.0
            d = np.diff(np.array(ts, dtype=np.int64)) * 1e-9
            return float(1.0 / np.median(d)), float(np.std(d) * 1e3)

        imu_hz, imu_jit = rate(self.imu_t)
        li_hz, li_jit = rate(self.scan_t)
        rep = {"imu_rate_hz": imu_hz, "imu_period_jitter_ms": imu_jit, "lidar_rate_hz": li_hz, "lidar_period_jitter_ms": li_jit,
               "valid_bins_fraction": float(np.mean(self.valid)) if self.valid else 0.0, **self.counters}
        warnings = []
        if self.imu_hz and imu_hz and abs(imu_hz / self.imu_hz - 1.0) > 0.1:
            warnings.append(f"IMU at {imu_hz:.1f} Hz, calibration {self.imu_hz:.1f} Hz")
        if li_hz and abs(li_hz / self.lidar_hz - 1.0) > 0.1:
            warnings.append(f"LiDAR at {li_hz:.2f} Hz, calibration {self.lidar_hz:.2f} Hz")
        if li_jit > 5.0:
            warnings.append(f"LiDAR period jitter {li_jit:.1f} ms")
        rep["warnings"] = warnings
        return rep
