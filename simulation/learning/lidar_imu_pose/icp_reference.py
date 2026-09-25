#!/usr/bin/env python3
"""Classical LiDAR reference on the multi-scenario dataset.

Runs the sensor-only ICP of ``train_pose_fusion.lidar_icp_delta`` (point-to-
point, gyro-initialised yaw) on a random sample of scan intervals of the
validation and test splits, converts the motion of the LiDAR frame to
base_link with the nominal extrinsic, and compares it with the exact
increments. It tells how much translation information the noisy scans carry,
independently of the learned architectures.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sqlite3
import sys

import numpy as np
import torch

from gridsearch_sqlite import ACC_STRICT, DEFAULT_DB, OUT
from train_pose_fusion import Config, lidar_icp_delta

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from pose_dataset.blobs import decode_mask, decode_ranges  # noqa: E402


def se2(x: float, y: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def main(samples: int = 3000, seed: int = 5) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    rng = np.random.default_rng(seed)
    conn = sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True)
    report = {}
    for split in ("validation", "test"):
        scans = conn.execute(
            "SELECT s.scan_id, s.run_id, s.seq, s.ranges_blob, s.ranges_encoding, s.valid_mask_blob, s.beam_count, g.x, g.y, g.yaw, r.sensor_profile "
            "FROM lidar_scans s JOIN ground_truth g ON g.ground_truth_id = s.ground_truth_id JOIN runs r ON r.run_id = s.run_id "
            "WHERE r.split = ? AND r.status = 'completed' ORDER BY s.run_id, s.seq",
            (split,),
        ).fetchall()
        cal = {r[0]: json.loads(r[1]) for r in conn.execute(
            "SELECT c.run_id, c.extrinsic_translation FROM sensor_calibration c JOIN runs r ON r.run_id = c.run_id WHERE r.split = ? AND c.sensor_name = 'lidar'", (split,))}
        pairs = [i for i in range(1, len(scans)) if scans[i][1] == scans[i - 1][1] and scans[i][2] == scans[i - 1][2] + 1]
        pick = rng.choice(pairs, size=min(samples, len(pairs)), replace=False)
        errs_t, errs_y, per_sensor = [], [], {}
        for i in pick:
            a, b = scans[i - 1], scans[i]
            prev = np.where(decode_mask(a[5], a[6]), decode_ranges(a[3], a[4], a[6]), np.inf).astype(np.float32)
            cur = np.where(decode_mask(b[5], b[6]), decode_ranges(b[3], b[4], b[6]), np.inf).astype(np.float32)
            # Heading from the ground truth here stands in for an ideal gyro: the
            # reference isolates the translation that the scans can explain.
            dyaw_true = (b[9] - a[9] + math.pi) % (2 * math.pi) - math.pi
            motion, _ = lidar_icp_delta(prev, cur, float(dyaw_true), cfg, device)
            ext = cal[a[1]]
            e = se2(ext[0], ext[1], 0.0)
            t_base = e @ se2(float(motion[0]), float(motion[1]), float(motion[2])) @ np.linalg.inv(e)
            c, s = math.cos(a[9]), math.sin(a[9])
            dx, dy = b[7] - a[7], b[8] - a[8]
            true = np.array([c * dx + s * dy, -s * dx + c * dy])
            err = float(np.linalg.norm(t_base[:2, 2] - true))
            errs_t.append(err)
            per_sensor.setdefault(a[10], []).append(err)
        errs = np.array(errs_t)
        report[split] = {
            "intervals": int(len(errs)),
            "icp_trans_mae_cm": 100 * float(errs.mean()),
            "icp_trans_median_cm": 100 * float(np.median(errs)),
            "icp_trans_within_2cm_pct": 100 * float((errs < ACC_STRICT[0]).mean()),
            "per_sensor_trans_mae_cm": {k: 100 * float(np.mean(v)) for k, v in sorted(per_sensor.items())},
        }
        print(split, json.dumps(report[split]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "icp_reference.json").write_text(json.dumps({
        "method": "train_pose_fusion.lidar_icp_delta (point-to-point ICP, stride 4, 7 iterations, 0.30 m gate) with true yaw initialisation, converted to base_link with the nominal LiDAR extrinsic",
        **report,
    }, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
