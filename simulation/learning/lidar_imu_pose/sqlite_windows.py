"""GPU-resident training windows built from the multi-scenario pose database.

The SQLite dataset (simulation/tools/pose_dataset) is read once per split and
turned into flat arrays with the input conventions of ``train_pose_fusion``:

* LiDAR: ``[2, beams]`` per scan, channel 0 = range / range_max (1.0 where
  the beam is invalid), channel 1 = validity mask;
* IMU: the samples that really arrived between scan k-1 and scan k (reported
  timestamps), zero-padded to ``max_imu`` with the real length kept; specific
  force normalised by g after removing gravity on z, gyro in rad/s;
* ``dt`` between reported scan stamps and the trapezoidal gyro-z integral over
  the interval (used by the IMU-consistency loss);
* targets: exact planar increment (dx, dy, dyaw) from the ground-truth pose
  of scan k-1 to scan k, in the frame of scan k-1.

An interval without IMU samples (B_economic burst dropouts) receives the
nearest sample, exactly like ``window_imu_at_lidar_intervals``.

``imu_delay_comp``: an IMU stamp is the sampling instant, but the chip's
low-pass filter makes the sampled signal lag the motion by its group delay
(``imu_group_delay_s``: 6.8 ms for the LSM6DS3 gyro at 33 Hz). With the
option, every IMU stamp is moved back by that delay, from the filter of the
run's sensor profile, before the samples are assigned to intervals and
integrated. Off by default (the historical caches and checkpoints).

Windows never cross runs; splits come from the database (whole trajectories
and whole tracks), never from a chronological cut.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sqlite3
import sys
from typing import Any

import numpy as np
import torch

TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from pose_dataset.blobs import decode_mask, decode_ranges  # noqa: E402

G = 9.80665
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def imu_group_delay_s(imu_cfg: dict[str, Any]) -> float:
    """Low-frequency group delay [s] of the gyro low-pass of an IMU profile
    (``dlpf.gyro_hz``, or ``dlpf_hz`` for the v1 profiles; 2nd-order
    Butterworth: sqrt(2) / (2 pi f_c)); 0 without a filter."""
    dlpf = imu_cfg.get("dlpf")
    f = float(dlpf.get("gyro_hz", 0.0)) if isinstance(dlpf, dict) else float(imu_cfg.get("dlpf_hz") or 0.0)
    return math.sqrt(2.0) / (2.0 * math.pi * f) if f > 0.0 else 0.0


def build_split(db_path: Path, split: str, max_imu: int = 40, imu_delay_comp: bool = False) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    runs = conn.execute(
        "SELECT run_id, run_key, track_name, motion_profile, sensor_profile, seed FROM runs WHERE split = ? AND status = 'completed' ORDER BY run_id",
        (split,),
    ).fetchall()
    lidar_parts, imu_parts, len_parts, dt_parts, gyro_parts, tgt_parts, pose_parts, t_parts = [], [], [], [], [], [], [], []
    meta = []
    offset = 0
    for run_id, run_key, track, motion, sensor, seed in runs:
        rows = conn.execute(
            "SELECT s.timestamp_ns, s.range_max, s.beam_count, s.ranges_encoding, s.ranges_blob, s.valid_mask_blob, g.x, g.y, g.yaw "
            "FROM lidar_scans s JOIN ground_truth g ON g.ground_truth_id = s.ground_truth_id WHERE s.run_id = ? ORDER BY s.seq",
            (run_id,),
        ).fetchall()
        n = len(rows)
        beams = rows[0][2]
        t = np.array([r[0] for r in rows], dtype=np.int64)
        lidar = np.empty((n, 2, beams), dtype=np.float16)
        for k, r in enumerate(rows):
            rng = decode_ranges(r[4], r[3], r[2])
            valid = decode_mask(r[5], r[2])
            lidar[k, 0] = np.where(valid, rng / r[1], 1.0)
            lidar[k, 1] = valid
        pose = np.array([[r[6], r[7], r[8]] for r in rows], dtype=np.float64)
        imu_rows = np.array(
            conn.execute("SELECT timestamp_ns, ax, ay, az, gx, gy, gz FROM imu_samples WHERE run_id = ? ORDER BY seq", (run_id,)).fetchall(),
            dtype=np.float64,
        )
        imu_t = imu_rows[:, 0].astype(np.int64)
        delay = 0.0
        if imu_delay_comp:
            cfg = conn.execute("SELECT noise_configuration_json FROM sensor_calibration WHERE run_id = ? AND sensor_name = 'imu'", (run_id,)).fetchone()
            delay = imu_group_delay_s(json.loads(cfg[0])) if cfg else 0.0
            imu_t = imu_t - int(round(delay * 1e9))
        imu_v = imu_rows[:, 1:].astype(np.float64)
        imu_norm = imu_v.copy()
        imu_norm[:, 2] -= G
        imu_norm[:, :3] /= G
        padded = np.zeros((n, max_imu, 6), dtype=np.float16)
        lengths = np.ones(n, dtype=np.int16)
        dt = np.zeros(n, dtype=np.float32)
        gyro = np.zeros(n, dtype=np.float32)
        for k in range(1, n):
            lo = int(np.searchsorted(imu_t, t[k - 1], side="right"))
            hi = int(np.searchsorted(imu_t, t[k], side="right"))
            if hi <= lo:
                nearest = int(np.clip(np.searchsorted(imu_t, t[k]), 0, len(imu_t) - 1))
                lo, hi = nearest, nearest + 1
            if hi - lo > max_imu:
                lo = hi - max_imu
            padded[k, : hi - lo] = imu_norm[lo:hi]
            lengths[k] = hi - lo
            t0, t1 = t[k - 1] * 1e-9, t[k] * 1e-9
            dt[k] = max(t1 - t0, 1e-4)
            inner = imu_t[lo:hi] * 1e-9
            inner = inner[(inner > t0) & (inner < t1)]
            ts = np.concatenate(([t0], inner, [t1]))
            gz = np.interp(ts, imu_t * 1e-9, imu_v[:, 5])
            gyro[k] = float(_trapz(gz, ts))
        padded[0], lengths[0], dt[0], gyro[0] = padded[1], lengths[1], dt[1], gyro[1]
        target = np.zeros((n, 3), dtype=np.float32)
        d = pose[1:, :2] - pose[:-1, :2]
        c, s = np.cos(pose[:-1, 2]), np.sin(pose[:-1, 2])
        target[1:, 0] = c * d[:, 0] + s * d[:, 1]
        target[1:, 1] = -s * d[:, 0] + c * d[:, 1]
        target[1:, 2] = _wrap(pose[1:, 2] - pose[:-1, 2])
        target[0] = target[1]
        lidar_parts.append(lidar)
        imu_parts.append(padded)
        len_parts.append(lengths)
        dt_parts.append(dt)
        gyro_parts.append(gyro)
        tgt_parts.append(target)
        pose_parts.append(pose)
        t_parts.append(t)
        meta.append({"run_id": run_id, "run_key": run_key, "track": track, "motion": motion, "sensor": sensor, "seed": seed, "offset": offset, "n": n,
                     "imu_delay_s": delay})
        offset += n
    conn.close()
    return {
        "split": split,
        "lidar": torch.from_numpy(np.concatenate(lidar_parts)),
        "imu": torch.from_numpy(np.concatenate(imu_parts)),
        "imu_len": torch.from_numpy(np.concatenate(len_parts).astype(np.int64)),
        "dt": torch.from_numpy(np.concatenate(dt_parts)),
        "gyro_integral": torch.from_numpy(np.concatenate(gyro_parts)),
        "target": torch.from_numpy(np.concatenate(tgt_parts)),
        "pose": torch.from_numpy(np.concatenate(pose_parts)),
        "t_ns": torch.from_numpy(np.concatenate(t_parts)),
        "runs": meta,
    }


def load_or_build(db_path: Path, cache_dir: Path, split: str, imu_delay_comp: bool = False) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    stat = db_path.stat()
    path = cache_dir / f"{split}_{int(stat.st_mtime)}_{stat.st_size}{'_imudelay' if imu_delay_comp else ''}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    data = build_split(db_path, split, imu_delay_comp=imu_delay_comp)
    torch.save(data, path)
    return data


class WindowSampler:
    """Index windows of ``steps`` target increments, each with ``context`` scans."""

    def __init__(self, data: dict[str, Any], device: torch.device, steps: int, context: int) -> None:
        self.device = device
        self.steps = steps
        self.context = context
        self.lidar = data["lidar"].to(device)
        self.imu = data["imu"].to(device)
        self.imu_len = data["imu_len"].to(device)
        self.dt = data["dt"].to(device)
        self.gyro = data["gyro_integral"].to(device)
        self.target = data["target"].to(device)
        self.runs = data["runs"]
        # Trim the IMU padding to the longest interval actually present.
        self.max_len = int(data["imu_len"].max())

    def starts(self, stride: int, runs: list[dict[str, Any]] | None = None, rng: np.random.Generator | None = None) -> torch.Tensor:
        """Window starts of every run; with ``rng`` each run gets a random
        phase in [0, stride) (training: every interval once per epoch, all
        alignments over epochs)."""
        out = []
        first = max(self.context - 1, 1)
        for r in runs or self.runs:
            phase = int(rng.integers(0, stride)) if rng is not None else 0
            local = np.arange(first + phase, r["n"] - self.steps + 1, stride)
            out.append(local + r["offset"])
        return torch.from_numpy(np.concatenate(out)).to(self.device)

    def batch(self, starts: torch.Tensor) -> dict[str, torch.Tensor]:
        steps = torch.arange(self.steps, device=self.device)
        targets = starts[:, None] + steps[None, :]
        # The consecutive scans of the window; target t uses scans t .. t + context - 1.
        scans = starts[:, None] - (self.context - 1) + torch.arange(self.steps + self.context - 1, device=self.device)
        return {
            "lidar": self.lidar[scans].float(),
            "imu": self.imu[targets][:, :, : self.max_len].float(),
            "imu_lengths": self.imu_len[targets],
            "dt": self.dt[targets],
            "gyro_integral": self.gyro[targets],
            "target": self.target[targets],
            "index": targets,
        }


def split_statistics(data: dict[str, Any]) -> dict[str, float]:
    tgt = data["target"].numpy()
    dt = data["dt"].numpy()
    speed = np.linalg.norm(tgt[:, :2], axis=1) / np.maximum(dt, 1e-3)
    return {
        "scans": int(len(tgt)),
        "runs": len(data["runs"]),
        "step_translation_p999_m": float(np.percentile(np.abs(tgt[:, :2]), 99.9)),
        "step_yaw_p999_rad": float(np.percentile(np.abs(tgt[:, 2]), 99.9)),
        "speed_max_mps": float(np.percentile(speed, 99.9)),
        "imu_len_max": int(data["imu_len"].max()),
        "empty_or_short_intervals": int((data["imu_len"] <= 1).sum()),
        "dt_median_s": float(np.median(dt)),
        "yaw_per_step_deg_mean": float(math.degrees(np.abs(tgt[:, 2]).mean())),
    }
