"""Per-run streams of the multi-scenario dataset for streaming odometry.

Extends the cached split tensors of ``sqlite_windows`` (same inputs and
targets, same normalisation) with what the streaming network, the classical
LiDAR reference and the odometry metrics need:

* scan geometry per run: ``range_max``, angle grid and time between beams.
  The scanner is rolling (``stamp`` = start of the sweep), so a scan is
  distorted by the motion during its own sweep. Beam i is fired at
  ``stamp + beam_time_frac[i] * beams * time_increment``: ``i / beams`` for
  the v1 profiles (counter-clockwise from ``angle_min``), and for a real
  scanner the fraction of the revolution between its sync angle and the beam
  in its rotation direction (the APEX A2M8 turns clockwise from +89 deg);
* the heading profile of every sweep (``gyro_profile``): the raw gyro-z
  integral from the start of interval k at ``PROFILE_KNOTS + 1`` equally
  spaced instants, so the de-skew follows the real rotation during the
  sweep instead of a constant rate (its last value is ``gyro_integral``);
* the nominal LiDAR mount (x, y) in base_link and the datasheet range noise
  (constant + proportional sigma) of the sensor profile, from the calibration
  table (the drawn per-run "true" mount and noise realization are never used);
* labels only, never model inputs: the exact body velocity at every scan
  instant and the true IMU bias averaged over every scan interval.

``StreamData`` keeps a split on the GPU and gathers the fields of any
[lanes, steps] block of interval indices. Interval k of a run goes from scan
k-1 to scan k (k >= 1); index 0 of every run is a placeholder.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
import torch
from torch import Tensor

from sqlite_windows import G, load_or_build


def bin_time_fraction(beams: int, angle_min: float, angle_increment: float, direction: str, start_angle: float) -> np.ndarray:
    """Acquisition time of every bin as a fraction of the revolution (same
    rule as ``apex_fusion_research.core.lidar_rolling.bin_time_fraction``)."""
    theta = angle_min + angle_increment * np.arange(beams)
    turn = (start_angle - theta) if direction == "cw" else (theta - start_angle)
    frac = np.mod(turn, 2.0 * math.pi) / (2.0 * math.pi)
    return np.where(np.isclose(frac, 1.0), 0.0, frac)


PROFILE_KNOTS = 16  # sweep heading profile: 17 instants, ~4.8 ms apart at 13 Hz (IMU at 9.6 ms)


def gyro_profile(t: np.ndarray, gz: np.ndarray, a: float, b: float, knots: int = PROFILE_KNOTS) -> np.ndarray:
    """Raw gyro-z integral [rad] from ``a`` to ``a + j / knots * (b - a)``,
    j = 0..knots (times in s). Same interpolant as the interval integral of
    ``sqlite_windows.build_split`` (linear between samples, trapezoid), so the
    last value equals it."""
    grid = a + (b - a) * np.arange(knots + 1) / knots
    ts = np.union1d(grid, t[(t > a) & (t < b)])
    g = np.interp(ts, t, gz)
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(ts))))
    return np.interp(grid, ts, cum)


def sweep_interp(values: Tensor, duration: Tensor, tau: Tensor) -> Tensor:
    """``values`` [R, K+1, ...] sampled at K+1 equally spaced instants over
    ``duration`` [R], linearly interpolated at the times ``tau`` [R, N]
    (extrapolated with the last segment beyond the sweep)."""
    k = values.shape[1] - 1
    u = tau / duration[:, None].clamp_min(1e-4) * k
    j0 = torch.floor(u).clamp(0, k - 1)
    w = u - j0
    j0 = j0.long()
    shape = j0.shape + values.shape[2:]
    idx = j0.reshape(j0.shape + (1,) * (values.dim() - 2)).expand(shape)
    v0, v1 = values.gather(1, idx), values.gather(1, idx + 1)
    w = w.reshape(w.shape + (1,) * (values.dim() - 2))
    return v0 + (v1 - v0) * w


def build_extras(db_path: Path, data: dict[str, Any]) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    runs = data["runs"]
    n_total = int(data["target"].shape[0])
    t_ns = data["t_ns"].numpy()
    v_body = np.zeros((n_total, 2), dtype=np.float32)
    bias = np.zeros((n_total, 3), dtype=np.float32)
    heading = np.zeros((n_total, PROFILE_KNOTS + 1), dtype=np.float32)
    run_index = np.zeros(n_total, dtype=np.int64)
    geom = {"range_max": [], "angle_min": [], "angle_increment": [], "time_increment_s": [], "lidar_xy": [], "lidar_sigma": [], "beam_time_frac": []}
    for ri, r in enumerate(runs):
        o, n = r["offset"], r["n"]
        run_index[o : o + n] = ri
        rows = conn.execute(
            "SELECT g.vx_body, g.vy_body, s.range_max, s.angle_min, s.angle_increment, s.time_increment_ns "
            "FROM lidar_scans s JOIN ground_truth g ON g.ground_truth_id = s.ground_truth_id WHERE s.run_id = ? ORDER BY s.seq",
            (r["run_id"],),
        ).fetchall()
        if len(rows) != n:
            raise RuntimeError(f"run {r['run_key']}: {len(rows)} scans, cache has {n}")
        v_body[o : o + n] = np.array([(a, b) for a, b, *_ in rows], dtype=np.float32)
        geom["range_max"].append(rows[0][2])
        geom["angle_min"].append(rows[0][3])
        geom["angle_increment"].append(rows[0][4])
        geom["time_increment_s"].append(rows[0][5] * 1e-9)
        ext, noise = conn.execute(
            "SELECT extrinsic_translation, noise_configuration_json FROM sensor_calibration WHERE run_id = ? AND sensor_name = 'lidar'", (r["run_id"],)
        ).fetchone()
        geom["lidar_xy"].append([float(v) for v in json.loads(ext)[:2]])
        profile = json.loads(noise)
        sheet = profile["noise"]
        geom["lidar_sigma"].append([float(sheet["range_sigma_const_m"]), float(sheet["range_sigma_prop"])])
        beams = int(data["lidar"].shape[-1])
        geom["beam_time_frac"].append(bin_time_fraction(
            beams, rows[0][3], rows[0][4], str(profile.get("scan_direction", "ccw")),
            math.radians(float(profile.get("scan_start_angle_deg", math.degrees(rows[0][3]))))).tolist())
        # True bias (gyro z, accel x, accel y) averaged over the IMU samples of
        # each interval, with the sample selection of sqlite_windows.build_split.
        b = np.array(
            conn.execute(
                "SELECT s.timestamp_ns, t.bgz, t.bax, t.bay, s.gz FROM imu_samples s JOIN imu_bias_truth t ON t.imu_id = s.imu_id WHERE s.run_id = ? ORDER BY s.seq",
                (r["run_id"],),
            ).fetchall(),
            dtype=np.float64,
        )
        imu_t = b[:, 0].astype(np.int64) - int(round(r.get("imu_delay_s", 0.0) * 1e9))  # same stamps as sqlite_windows
        csum = np.vstack((np.zeros((1, 3)), np.cumsum(b[:, 1:4], axis=0)))
        ts = t_ns[o : o + n]
        for k in range(1, n):
            heading[o + k] = gyro_profile(imu_t * 1e-9, b[:, 4], ts[k - 1] * 1e-9, ts[k] * 1e-9)
            lo = int(np.searchsorted(imu_t, ts[k - 1], side="right"))
            hi = int(np.searchsorted(imu_t, ts[k], side="right"))
            if hi <= lo:
                lo = int(np.clip(np.searchsorted(imu_t, ts[k]), 0, len(imu_t) - 1))
                hi = lo + 1
            bias[o + k] = (csum[hi] - csum[lo]) / (hi - lo)
        bias[o] = bias[o + 1]
        heading[o] = heading[o + 1]
    conn.close()
    return {
        "v_body": torch.from_numpy(v_body),
        "gyro_profile": torch.from_numpy(heading),
        "bias_truth": torch.from_numpy(bias),
        "run_index": torch.from_numpy(run_index),
        **{k: torch.tensor(v, dtype=torch.float32) for k, v in geom.items()},
    }


def load_split(db_path: Path, cache_dir: Path, split: str, imu_delay_comp: bool = False) -> dict[str, Any]:
    """A split with its stream extras; ``imu_delay_comp``: IMU stamps moved
    back by the chip filter delay (``sqlite_windows.imu_group_delay_s``)."""
    data = load_or_build(db_path, cache_dir, split, imu_delay_comp)
    stat = db_path.stat()
    path = cache_dir / f"{split}_streams_v4_{int(stat.st_mtime)}_{stat.st_size}{'_imudelay' if imu_delay_comp else ''}.pt"
    if path.exists():
        extras = torch.load(path, map_location="cpu", weights_only=False)
    else:
        extras = build_extras(db_path, data)
        torch.save(extras, path)
    return {**data, **extras}


def imu_interval_stats(imu: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
    """Mean planar specific force [m/s^2] and specific-force spread [m/s^2]
    (standstill detector) of the zero-padded, normalised IMU samples of an
    interval ([..., samples, 6], valid ``lengths``)."""
    mask = (torch.arange(imu.shape[-2], device=imu.device) < lengths[..., None]).float()
    count = mask.sum(-1, keepdim=True).clamp_min(1.0)
    accel = (imu[..., :2] * mask[..., None]).sum(-2) / count * G
    mean3 = (imu[..., :3] * mask[..., None]).sum(-2) / count
    accel_std = (((imu[..., :3] - mean3[..., None, :]).square() * mask[..., None]).sum(-2) / count).sqrt().mean(-1) * G
    return accel, accel_std


def icp_features(res: dict[str, Any]) -> tuple[Tensor, Tensor]:
    """Increment and quality indicators of the classical odometry, as the
    hybrid network consumes them ([N, 3], [N, 9])."""
    pred = torch.nan_to_num(res["pred"].float(), nan=0.0)
    sig = torch.nan_to_num(res["sigma"].float(), nan=1.0).clamp(1e-5, 1.0)
    feats = torch.stack((
        torch.log10(sig[:, 0]) + 2.0, torch.log10(sig[:, 1]) + 2.0, torch.log10(sig[:, 2]) + 3.0,
        torch.log10(torch.nan_to_num(res["eig_ratio"].float(), nan=1.0).clamp(1e-6, 1.0)) + 1.0,
        torch.log10(torch.nan_to_num(res["weak_sigma_m"].float(), nan=1.0).clamp(1e-5, 1.0)) + 2.0,
        torch.log10(torch.nan_to_num(res["icp_yaw_sigma_rad"].float(), nan=1.0).clamp(1e-6, 1.0)) + 3.0,
        torch.nan_to_num(res["pairs"].float(), nan=0.0) / 360.0,
        torch.nan_to_num(res["standstill"].float(), nan=0.0),
        torch.nan_to_num(res["gyro_bias"].float(), nan=0.0) * 20.0,
    ), dim=1)
    return pred, feats


class StreamData:
    """One split on the device, indexed by global interval index."""

    def __init__(self, data: dict[str, Any], device: torch.device) -> None:
        self.device = device
        self.split = data["split"]
        self.runs = data["runs"]
        self.lidar = data["lidar"].to(device)  # [N, 2, beams] fp16: range / range_max (1 if invalid), valid
        self.imu = data["imu"].to(device)
        self.imu_len = data["imu_len"].to(device)
        self.max_len = int(data["imu_len"].max())
        self.dt = data["dt"].to(device)
        self.gyro = data["gyro_integral"].to(device)
        prof = data.get("gyro_profile")
        if prof is None:  # data without the profile: constant rate over every interval
            prof = data["gyro_integral"][:, None] * torch.linspace(0.0, 1.0, PROFILE_KNOTS + 1)
        self.gyro_profile = prof.to(device).float()  # [N, PROFILE_KNOTS + 1] heading during interval k (raw gyro)
        self.target = data["target"].to(device)
        self.v_body = data["v_body"].to(device)
        self.bias_truth = data["bias_truth"].to(device)
        self.run_index = data["run_index"].to(device)
        self.range_max = data["range_max"].to(device)
        self.time_increment = data["time_increment_s"].to(device)
        self.lidar_xy = data["lidar_xy"].to(device)
        self.lidar_sigma = data["lidar_sigma"].to(device)  # [runs, 2]: datasheet sigma = c0 + c1 * range
        self.angle_min = data["angle_min"].to(device)
        self.angle_increment = data["angle_increment"].to(device)
        beams = int(self.lidar.shape[-1])
        linear = torch.arange(beams, device=device, dtype=torch.float32) / beams
        frac = data.get("beam_time_frac")
        self.beam_frac = frac.to(device).float() if frac is not None else linear.expand(len(self.runs), beams).clone()  # [runs, beams]
        # v1 scanners (beam i at i * time_increment) keep the historical batches.
        self.linear_time = bool(torch.allclose(self.beam_frac, linear.expand_as(self.beam_frac), atol=1e-6))
        self.offset = torch.tensor([r["offset"] for r in self.runs], device=device)
        self.length = torch.tensor([r["n"] for r in self.runs], device=device)
        self.t_ns = data["t_ns"]
        self.pose = data["pose"]

    def attach_icp(self, res: dict[str, Any]) -> None:
        """Per-interval output of the classical odometry (``icp_odometry``) as
        an extra input: estimate plus quality indicators (all computed from
        the sensors only, causally)."""
        pred, feats = icp_features(res)
        self.icp_pred = pred.to(self.device)
        self.icp_feat = feats.to(self.device)

    def beam_time(self, runs: Tensor) -> Tensor | None:
        """Acquisition time [s] of every beam after the stamp, [len(runs), beams]
        (None when every beam i is fired at i * time_increment)."""
        if self.linear_time:
            return None
        return self.beam_frac[runs] * self.beam_frac.shape[-1] * self.time_increment[runs, None]

    @property
    def intervals(self) -> int:
        return int(sum(r["n"] - 1 for r in self.runs))

    def ranges(self, scan: Tensor) -> tuple[Tensor, Tensor]:
        """Metric ranges (0 where invalid) and validity of global scan indices."""
        raw = self.lidar[scan].float()
        valid = raw[..., 1, :] > 0.5
        rmax = self.range_max[self.run_index[scan]]
        return torch.where(valid, raw[..., 0, :] * rmax[..., None], torch.zeros_like(raw[..., 0, :])), valid

    def sweep_rate(self, idx: Tensor) -> tuple[Tensor, Tensor]:
        """Raw gyro yaw rate during the sweeps of scans idx-1 and idx.

        The sweep of scan k-1 spans interval k, the sweep of scan k spans
        interval k+1 (available once scan k is complete). The last scan of a
        run reuses its own interval."""
        run = self.run_index[idx]
        last = self.offset[run] + self.length[run] - 1
        nxt = torch.minimum(idx + 1, last)
        return self.gyro[idx] / self.dt[idx], self.gyro[nxt] / self.dt[nxt]

    def gather(self, idx: Tensor) -> dict[str, Tensor]:
        """Model inputs and labels of interval indices ``idx`` ([lanes, steps])."""
        prev_r, prev_v = self.ranges(idx - 1)
        cur_r, cur_v = self.ranges(idx)
        w_prev, w_cur = self.sweep_rate(idx)
        run = self.run_index[idx]
        nxt = torch.minimum(idx + 1, self.offset[run] + self.length[run] - 1)
        imu = self.imu[idx][..., : self.max_len, :].float()
        lengths = self.imu_len[idx]
        accel, accel_std = imu_interval_stats(imu, lengths)
        return {
            "prev_ranges": prev_r, "prev_valid": prev_v, "cur_ranges": cur_r, "cur_valid": cur_v,
            "w_prev": w_prev, "w_cur": w_cur, "time_increment": self.time_increment[run],
            **({} if self.linear_time else {"beam_frac": self.beam_frac[run]}),
            "imu": imu, "imu_lengths": lengths, "accel": accel, "accel_std": accel_std,
            "gyro_integral": self.gyro[idx], "dt": self.dt[idx],
            # heading profiles of the sweeps of scans k-1 (interval k) and k (interval k+1)
            "prof_prev": self.gyro_profile[idx], "prof_cur": self.gyro_profile[nxt], "dt_next": self.dt[nxt],
            # labels
            "target": self.target[idx], "v_body": self.v_body[idx], "bias_truth": self.bias_truth[idx],
            **({"icp_pred": self.icp_pred[idx], "icp_feat": self.icp_feat[idx]} if hasattr(self, "icp_pred") else {}),
        }
