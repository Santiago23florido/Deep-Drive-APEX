#!/usr/bin/env python3
"""Identify the real APEX sensors from the recordings of the physical car.

Reads the CSV recordings of ``real_vehicle/data`` (``imu_raw.csv`` and
``lidar_points.csv``) and measures what the ``APEX_real`` sensor profile of the
pose dataset needs (``tools/pose_dataset/config/sensors.yaml``):

* IMU (LSM6DS3 on the Arduino Nano 33 IoT): output rate and period jitter of
  the recording, quantization step, and, on segments at rest, the bias and the
  noise standard deviation of every axis;
* LiDAR (RPLIDAR A2M8): revolution period (dominant mode of the scan
  intervals), samples and valid 1-degree bins per revolution, and the range
  noise (constant + proportional fit) of bins that stay constant while the car
  is still at the beginning of a recognition tour.

The recordings were made with the timing defects documented in
``real_vehicle/docs/sensor_timing_requirements.md`` (stamps taken on reception,
IMU firmware loop at ~55.6 Hz, LiDAR in the 2 kHz compatible mode). The script
reports them as measured; the profile only takes what belongs to the hardware
(noise, bias, quantization, rotation rate, geometry).

Usage:
    python3 tools/analysis/real_sensor_identification.py [--data real_vehicle/data] [--out report.json]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def _imu(path: Path) -> dict:
    a = np.genfromtxt(path, delimiter=",", skip_header=1)
    t = a[:, 0] + a[:, 1] * 1e-9
    v = a[:, 2:8]
    d = np.diff(t)
    # Rest: low short-time spread of the gyro z rate and of the specific-force norm.
    w = 25
    gz = v[:, 5]
    norm = np.linalg.norm(v[:, :3], axis=1)
    spread = np.array([gz[max(0, i - w) : i + w].std() + 0.1 * norm[max(0, i - w) : i + w].std() for i in range(len(gz))])
    rest = spread < 1.5 * np.percentile(spread, 20)
    seg = v[rest]
    steps = np.diff(np.unique(np.round(v[:, 5], 7)))
    return {
        "file": str(path.relative_to(ROOT)),
        "samples": int(len(t)),
        "rate_hz": float(1.0 / np.median(d)),
        "period_ms_p5_p50_p95": [float(x) for x in np.percentile(d, [5, 50, 95]) * 1e3],
        "rest_samples": int(rest.sum()),
        "rest_mean": {"accel_mps2": seg[:, :3].mean(0).tolist(), "gyro_rps": seg[:, 3:].mean(0).tolist()},
        "rest_std": {"accel_mps2": seg[:, :3].std(0).tolist(), "gyro_rps": seg[:, 3:].std(0).tolist()},
        "rest_specific_force_norm_mps2": float(norm[rest].mean()),
        "gyro_quantization_rps": float(steps[steps > 1e-7].min()),
    }


def _lidar(path: Path, still_scans: int = 40) -> dict:
    rows = []
    with open(path, encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            if len(row) >= 6:
                rows.append((int(row[0]) + int(row[1]) * 1e-9, int(row[2]), float(row[4]), float(row[5])))
    a = np.array(rows)
    scan = a[:, 1].astype(int)
    ids, first = np.unique(scan, return_index=True)
    dt = np.diff(a[first, 0])
    counts = np.bincount(scan)[ids]
    # Dominant revolution period: the densest 5 ms bin of the scan intervals.
    hist, edges = np.histogram(dt, bins=np.arange(0.0, 0.2, 0.005))
    mode = dt[(dt >= edges[hist.argmax()]) & (dt < edges[hist.argmax() + 1])]
    # Range noise: 1-degree bins observed many times while the car is still.
    # A bin collects samples at different angles inside its degree, so on a
    # slanted surface its spread also contains the range change across the
    # bin (|dr/dtheta| * 1 deg / sqrt(12) for uniform angles), which the
    # simulation reproduces by binning its own samples: remove it, and keep
    # bins whose neighbours are on one surface (slope < 5 cm per degree).
    s = a[np.isin(scan, ids[:still_scans])]
    b = np.round(np.degrees(s[:, 2])).astype(int) % 360
    mean = np.full(360, np.nan)
    sd = np.full(360, np.nan)
    for k in range(360):
        r = s[b == k, 3]
        if len(r) >= 10 and r.std() < 0.1:
            mean[k], sd[k] = r.mean(), r.std()
    fit = None
    if np.sum(np.isfinite(sd)) >= 30 and float(np.nanmedian(sd)) < 0.02:  # the car really was still
        slope = np.abs(np.roll(mean, -1) - np.roll(mean, 1)) / 2.0
        ok = np.isfinite(sd) & np.isfinite(slope) & (slope < 0.05)
        noise = np.sqrt(np.maximum(sd[ok] ** 2 - (slope[ok] / np.sqrt(12.0)) ** 2, 0.0))
        coef = np.linalg.lstsq(np.column_stack((np.ones(ok.sum()), mean[ok])), noise, rcond=None)[0]
        fit = {"bins": int(ok.sum()), "sigma_const_m": float(coef[0]), "sigma_prop": float(coef[1]), "median_std_m": float(np.median(noise)),
               "raw_bin_std_median_m": float(np.nanmedian(sd))}
    return {
        "file": str(path.relative_to(ROOT)),
        "scans": int(len(ids)),
        "published_rate_hz": float(1.0 / np.median(dt)),
        "interval_ms_p5_p50_p95": [float(x) for x in np.percentile(dt, [5, 50, 95]) * 1e3],
        "fraction_short_intervals": float(np.mean(dt < 0.065)),
        "revolution_period_ms": float(np.median(mode) * 1e3),
        "valid_points_per_scan_p50": float(np.median(counts)),
        "range_noise_still": fit,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "real_vehicle" / "data")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    imu = [_imu(Path(p)) for p in sorted(glob.glob(str(args.data / "apex_*" / "*" / "imu_raw.csv")))]
    lidar = [_lidar(Path(p)) for p in sorted(glob.glob(str(args.data / "apex_*" / "*" / "lidar_points.csv")))]
    fits = [x["range_noise_still"] for x in lidar if x["range_noise_still"]]
    summary = {
        "imu": {
            "recordings": len(imu),
            "output_rate_hz": float(np.median([x["rate_hz"] for x in imu])) if imu else None,
            "gyro_bias_rps": np.median([x["rest_mean"]["gyro_rps"] for x in imu], axis=0).tolist() if imu else None,
            "gyro_noise_std_rps": np.median([x["rest_std"]["gyro_rps"] for x in imu], axis=0).tolist() if imu else None,
            "accel_mean_mps2": np.median([x["rest_mean"]["accel_mps2"] for x in imu], axis=0).tolist() if imu else None,
            "accel_noise_std_mps2": np.median([x["rest_std"]["accel_mps2"] for x in imu], axis=0).tolist() if imu else None,
            "gyro_quantization_rps": float(np.median([x["gyro_quantization_rps"] for x in imu])) if imu else None,
        },
        "lidar": {
            "recordings": len(lidar),
            "revolution_period_ms": float(np.median([x["revolution_period_ms"] for x in lidar])) if lidar else None,
            "rotation_hz": float(1e3 / np.median([x["revolution_period_ms"] for x in lidar])) if lidar else None,
            "valid_points_per_scan_p50": float(np.median([x["valid_points_per_scan_p50"] for x in lidar])) if lidar else None,
            "fraction_short_intervals": float(np.median([x["fraction_short_intervals"] for x in lidar])) if lidar else None,
            "range_sigma_const_m": float(np.median([f["sigma_const_m"] for f in fits])) if fits else None,
            "range_sigma_prop": float(np.median([f["sigma_prop"] for f in fits])) if fits else None,
            "range_noise_median_m": float(np.median([f["median_std_m"] for f in fits])) if fits else None,
            "range_noise_recordings": len(fits),
        },
    }
    report = {"summary": summary, "imu_recordings": imu, "lidar_recordings": lidar}
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
