"""PyTorch reader of the pose dataset.

``PoseSequenceDataset`` yields windows of ``seq_len`` consecutive LiDAR scans
of one run, the IMU samples that really arrived between consecutive scans
(variable count, real timestamps), and optionally the exact pose labels.

* Windows never cross runs; a dataset instance only reads the requested
  split(s), so partitions are never mixed.
* Nothing is padded or resampled in ``__getitem__``: each interval keeps its
  real length. ``collate_pose_batch`` pads IMU intervals (and beams, if runs
  with different beam counts are batched) and returns masks and lengths.
* Every DataLoader worker opens its own read-only SQLite connection lazily
  (connections are never pickled or shared between processes).

Inputs vs labels: ``scans``, ``scan_valid``, ``imu``, timestamps and the
nominal extrinsics are sensor-side information; ``gt_*`` tensors are labels
and must not be fed to the model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .blobs import decode_mask, decode_ranges


def _se2_relative(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """(dx, dy, dyaw) of pose p1 expressed in the frame of p0; poses (x, y, yaw)."""
    c, s = np.cos(p0[..., 2]), np.sin(p0[..., 2])
    dx = p1[..., 0] - p0[..., 0]
    dy = p1[..., 1] - p0[..., 1]
    dyaw = np.angle(np.exp(1j * (p1[..., 2] - p0[..., 2])))
    return np.stack((c * dx + s * dy, -s * dx + c * dy, dyaw), axis=-1)


class PoseSequenceDataset(Dataset):
    def __init__(
        self,
        db_path: str | Path,
        splits: Sequence[str] | str = "train",
        seq_len: int = 6,
        stride: int = 3,
        run_ids: Sequence[int] | None = None,
        sensor_profiles: Sequence[str] | None = None,
        with_ground_truth: bool = True,
        range_fill: float = 0.0,
        only_completed: bool = True,
    ) -> None:
        self.db_path = str(Path(db_path).resolve())
        self.splits = [splits] if isinstance(splits, str) else list(splits)
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.with_gt = with_ground_truth
        self.range_fill = float(range_fill)
        if self.seq_len < 2:
            raise ValueError("seq_len must be >= 2 (IMU intervals lie between scans)")
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        q = f"SELECT run_id, split, sensor_profile, track_name, motion_profile FROM runs WHERE split IN ({','.join('?' * len(self.splits))})"
        params: list[Any] = list(self.splits)
        if only_completed:
            q += " AND status = 'completed'"
        if run_ids is not None:
            q += f" AND run_id IN ({','.join('?' * len(run_ids))})"
            params += list(run_ids)
        if sensor_profiles is not None:
            q += f" AND sensor_profile IN ({','.join('?' * len(sensor_profiles))})"
            params += list(sensor_profiles)
        self.runs = {r[0]: {"split": r[1], "sensor": r[2], "track": r[3], "motion": r[4]} for r in conn.execute(q + " ORDER BY run_id", params)}
        self.index: list[tuple[int, int]] = []
        self._scan_ids: dict[int, np.ndarray] = {}
        for run_id in self.runs:
            ids = np.array([r[0] for r in conn.execute("SELECT scan_id FROM lidar_scans WHERE run_id = ? ORDER BY seq", (run_id,))], dtype=np.int64)
            self._scan_ids[run_id] = ids
            for start in range(0, len(ids) - self.seq_len + 1, self.stride):
                self.index.append((run_id, start))
            cal = {r[0]: r for r in conn.execute("SELECT sensor_name, extrinsic_translation, extrinsic_rotation, nominal_frequency FROM sensor_calibration WHERE run_id = ?", (run_id,))}
            self.runs[run_id]["extrinsics"] = {
                name: {"t": json.loads(cal[name][1]), "q_xyzw": json.loads(cal[name][2]), "hz": cal[name][3]} for name in cal
            }
        conn.close()
        self._conn: sqlite3.Connection | None = None
        self._pid: int | None = None

    # One connection per process; never shared through fork/pickle.
    def _db(self) -> sqlite3.Connection:
        if self._conn is None or self._pid != os.getpid():
            self._conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False)
            self._pid = os.getpid()
        return self._conn

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_conn"] = None
        state["_pid"] = None
        return state

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, Any]:
        run_id, start = self.index[i]
        scan_ids = self._scan_ids[run_id][start : start + self.seq_len]
        db = self._db()
        rows = db.execute(
            f"SELECT scan_id, timestamp_ns, ground_truth_id, beam_count, angle_min, angle_increment, range_min, range_max, "
            f"time_increment_ns, ranges_encoding, ranges_blob, valid_mask_blob FROM lidar_scans WHERE scan_id IN ({','.join('?' * len(scan_ids))}) ORDER BY seq",
            [int(s) for s in scan_ids],
        ).fetchall()
        beams = rows[0][3]
        ranges = np.empty((len(rows), beams), dtype=np.float32)
        valid = np.empty((len(rows), beams), dtype=bool)
        for k, r in enumerate(rows):
            rr = decode_ranges(r[10], r[9], r[3])
            vm = decode_mask(r[11], r[3])
            ranges[k] = np.where(vm, rr, self.range_fill)
            valid[k] = vm
        scan_t = np.array([r[1] for r in rows], dtype=np.int64)
        imu = db.execute(
            "SELECT timestamp_ns, ax, ay, az, gx, gy, gz, ground_truth_id FROM imu_samples WHERE run_id = ? AND timestamp_ns > ? AND timestamp_ns <= ? ORDER BY timestamp_ns",
            (run_id, int(scan_t[0]), int(scan_t[-1])),
        ).fetchall()
        imu_arr = np.array(imu, dtype=np.float64).reshape(-1, 8)
        imu_t = imu_arr[:, 0].astype(np.int64)
        cuts = np.searchsorted(imu_t, scan_t[1:], side="right")
        bounds = np.concatenate(([0], cuts))
        intervals, dts, lengths, rel_t = [], [], [], []
        for k in range(len(scan_t) - 1):
            seg = imu_arr[bounds[k] : bounds[k + 1]]
            seg_t = imu_t[bounds[k] : bounds[k + 1]]
            prev = np.concatenate(([scan_t[k]], seg_t[:-1]))
            intervals.append(torch.from_numpy(seg[:, 1:7].astype(np.float32)))
            dts.append(torch.from_numpy(((seg_t - prev) * 1e-9).astype(np.float32)))
            rel_t.append(torch.from_numpy(((seg_t - scan_t[k]) * 1e-9).astype(np.float32)))
            lengths.append(len(seg))
        meta = self.runs[run_id]
        item: dict[str, Any] = {
            "run_id": run_id,
            "split": meta["split"],
            "sensor_profile": meta["sensor"],
            "scan_t_ns": torch.from_numpy(scan_t),
            "scan_dt": torch.from_numpy((np.diff(scan_t) * 1e-9).astype(np.float32)),
            "scans": torch.from_numpy(ranges),
            "scan_valid": torch.from_numpy(valid),
            "angle_min": float(rows[0][4]),
            "angle_increment": float(rows[0][5]),
            "time_increment_s": float(rows[0][8]) * 1e-9,
            "imu": intervals,  # list of (L_k, 6): ax, ay, az, gx, gy, gz
            "imu_dt": dts,  # list of (L_k,): time since the previous sample (or scan) [s]
            "imu_t_rel": rel_t,  # list of (L_k,): time since the interval's first scan [s]
            "imu_len": torch.tensor(lengths, dtype=torch.long),
            "extrinsics": meta["extrinsics"],
        }
        if self.with_gt:
            gt_ids = [int(r[2]) for r in rows]
            g = db.execute(
                f"SELECT ground_truth_id, x, y, z, qx, qy, qz, qw, yaw, vx_body, vy_body, wz FROM ground_truth WHERE ground_truth_id IN ({','.join('?' * len(gt_ids))})",
                gt_ids,
            ).fetchall()
            by_id = {r[0]: r[1:] for r in g}
            gt = np.array([by_id[k] for k in gt_ids], dtype=np.float64)
            pose2d = gt[:, [0, 1, 7]]
            item["gt_pose"] = torch.from_numpy(gt[:, :7].astype(np.float32))  # x, y, z, qx, qy, qz, qw per scan
            item["gt_yaw"] = torch.from_numpy(gt[:, 7].astype(np.float32))
            item["gt_delta"] = torch.from_numpy(_se2_relative(pose2d[:-1], pose2d[1:]).astype(np.float32))  # (dx, dy, dyaw) scan k-1 -> k
            item["gt_velocity_body"] = torch.from_numpy(gt[:, 8:11].astype(np.float32))  # vx, vy, wz per scan
        return item


def collate_pose_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad IMU intervals to the longest one in the batch; masks mark real samples."""
    b = len(batch)
    k = len(batch[0]["imu"])
    if any(len(x["imu"]) != k for x in batch):
        raise ValueError("all items of a batch must have the same seq_len")
    l_max = max(1, max(int(x["imu_len"].max()) if len(x["imu_len"]) else 0 for x in batch))
    beams = max(x["scans"].shape[1] for x in batch)
    imu = torch.zeros(b, k, l_max, 6)
    imu_dt = torch.zeros(b, k, l_max)
    imu_t_rel = torch.zeros(b, k, l_max)
    imu_mask = torch.zeros(b, k, l_max, dtype=torch.bool)
    scans = torch.zeros(b, k + 1, beams)
    scan_valid = torch.zeros(b, k + 1, beams, dtype=torch.bool)
    for i, x in enumerate(batch):
        nb = x["scans"].shape[1]
        scans[i, :, :nb] = x["scans"]
        scan_valid[i, :, :nb] = x["scan_valid"]
        for j in range(k):
            n = int(x["imu_len"][j])
            if n:
                imu[i, j, :n] = x["imu"][j]
                imu_dt[i, j, :n] = x["imu_dt"][j]
                imu_t_rel[i, j, :n] = x["imu_t_rel"][j]
                imu_mask[i, j, :n] = True
    out = {
        "run_id": torch.tensor([x["run_id"] for x in batch]),
        "split": [x["split"] for x in batch],
        "sensor_profile": [x["sensor_profile"] for x in batch],
        "scan_t_ns": torch.stack([x["scan_t_ns"] for x in batch]),
        "scan_dt": torch.stack([x["scan_dt"] for x in batch]),
        "scans": scans,
        "scan_valid": scan_valid,
        "angle_min": torch.tensor([x["angle_min"] for x in batch]),
        "angle_increment": torch.tensor([x["angle_increment"] for x in batch]),
        "imu": imu,
        "imu_dt": imu_dt,
        "imu_t_rel": imu_t_rel,
        "imu_mask": imu_mask,
        "imu_len": torch.stack([x["imu_len"] for x in batch]),
        "extrinsics": [x["extrinsics"] for x in batch],
    }
    if "gt_delta" in batch[0]:
        for key in ("gt_pose", "gt_yaw", "gt_delta", "gt_velocity_body"):
            out[key] = torch.stack([x[key] for x in batch])
    return out
