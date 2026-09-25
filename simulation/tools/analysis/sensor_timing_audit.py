#!/usr/bin/env python3
"""Audit the timing of recorded LiDAR / IMU streams against a sensor profile.

The learned odometry is trained on sensors with the timing of a profile of
``tools/pose_dataset/config/sensors.yaml`` (``APEX_real``: RPLIDAR A2M8 at its
~13 Hz rotation, LSM6DS3 at its 104 Hz output data rate, stamps at the
physical sampling instants). This tool checks whether a recording meets that
timing, so the same criteria judge the simulation and the real car
(``real_vehicle/docs/sensor_timing_requirements.md``).

Inputs (auto-detected in the directory):
* real car recordings (``real_vehicle/data/...``): ``imu_raw.csv``
  (stamp_sec, stamp_nanosec, ax..gz) and ``lidar_points.csv`` (one row per
  valid point with its scan index);
* real2sim runs (``simulation/data/real2sim/<run>``): ``imu_raw.csv`` and
  ``truth_scans.csv`` (stamp_ns, valid_bins).

Checks (tolerances in brackets):
* IMU rate within 5 % of the profile; regular periods (p5 / p95 within
  10 % of the nominal period); no gaps longer than 2 periods (< 0.5 %);
* LiDAR revolution rate within 10 % of the profile; regular revolutions (p5 /
  p95 within 10 %); no short intervals (partial revolutions, < 1 %);
  valid 1-degree bins per revolution >= 70 % of the bins the profile can fill
  outside its occluded sectors (``min(360, sample_rate / rate)``);
* one IMU sample per revolution at least: IMU samples per scan interval
  within 20 % of rate_imu / rate_lidar.

Usage:
    python3 tools/analysis/sensor_timing_audit.py <recording_dir> [--profile APEX_real] [--json out.json]
    python3 tools/analysis/sensor_timing_audit.py real_vehicle/data/apex_recognition_tour/<run> --profile APEX_real_compat
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM / "tools"))

from pose_dataset.sensors import load_sensor_profiles  # noqa: E402


def _imu_stamps(d: Path) -> np.ndarray | None:
    p = d / "imu_raw.csv"
    if not p.exists():
        return None
    a = np.genfromtxt(p, delimiter=",", skip_header=1)
    if a.ndim != 2 or len(a) < 3:
        return None
    return np.sort(a[:, 0] + a[:, 1] * 1e-9)


def _lidar(d: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """(revolution stamps [s], valid bins per revolution)."""
    sim = d / "truth_scans.csv"
    if sim.exists():
        a = np.genfromtxt(sim, delimiter=",", names=True)
        if a.size < 3:
            return None
        order = np.argsort(a["stamp_ns"])
        return a["stamp_ns"][order] * 1e-9, a["valid_bins"][order]
    real = d / "lidar_points.csv"
    if not real.exists():
        return None
    stamps: dict[int, float] = {}
    bins: dict[int, set[int]] = {}
    with open(real, encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        ia = header.index("angle_rad")
        for row in reader:
            if len(row) <= ia:
                continue
            k = int(row[2])
            stamps.setdefault(k, int(row[0]) + int(row[1]) * 1e-9)
            bins.setdefault(k, set()).add(int(round(math.degrees(float(row[ia])))) % 360)
    keys = sorted(stamps)
    return np.array([stamps[k] for k in keys]), np.array([len(bins[k]) for k in keys], dtype=float)


def audit(d: Path, profile: dict) -> dict:
    checks: dict[str, dict] = {}

    def check(name: str, ok: bool, measured, required: str) -> None:
        checks[name] = {"pass": bool(ok), "measured": measured, "required": required}

    imu_hz = float(profile["imu"]["rate_hz"])
    li = profile["lidar"]
    li_hz = float(li["rate_hz"])
    imu = _imu_stamps(d)
    if imu is not None:
        dt = np.diff(imu)
        rate = 1.0 / float(np.median(dt))
        p5, p95 = (float(x) for x in np.percentile(dt, [5, 95]))
        check("imu_rate", abs(rate / imu_hz - 1.0) < 0.05, round(rate, 2), f"{imu_hz:g} Hz +-5 %")
        check("imu_regular_periods", p5 > 0.9 / imu_hz and p95 < 1.1 / imu_hz, [round(p5 * 1e3, 2), round(p95 * 1e3, 2)],
              f"p5..p95 within {0.9e3 / imu_hz:.2f}..{1.1e3 / imu_hz:.2f} ms")
        gaps = float(np.mean(dt > 2.0 / imu_hz))
        check("imu_no_gaps", gaps < 0.005, round(100 * gaps, 3), "< 0.5 % of periods longer than 2 nominal periods")
    lid = _lidar(d)
    if lid is not None:
        st, valid = lid
        dt = np.diff(st)
        rate = 1.0 / float(np.median(dt))
        p5, p95 = (float(x) for x in np.percentile(dt, [5, 95]))
        check("lidar_rate", abs(rate / li_hz - 1.0) < 0.10, round(rate, 2), f"{li_hz:g} Hz +-10 %")
        check("lidar_regular_revolutions", p5 > 0.9 / li_hz and p95 < 1.1 / li_hz, [round(p5 * 1e3, 1), round(p95 * 1e3, 1)],
              f"p5..p95 within {0.9e3 / li_hz:.1f}..{1.1e3 / li_hz:.1f} ms")
        short = float(np.mean(dt < 0.8 / li_hz))
        check("lidar_no_partial_revolutions", short < 0.01, round(100 * short, 2), "< 1 % of intervals shorter than 0.8 revolution")
        occluded = sum((b - a) for a, b in li.get("occluded_sectors_deg", [])) / 360.0
        fillable = min(360.0, float(li.get("sample_rate_hz", 360 * li_hz)) / li_hz) * (1.0 - occluded)
        med = float(np.median(valid))
        check("lidar_valid_bins", med >= 0.7 * fillable, round(med, 1), f">= {0.7 * fillable:.0f} of {fillable:.0f} fillable 1-degree bins")
        if imu is not None:
            counts = np.diff(np.searchsorted(imu, st))
            exp = imu_hz / li_hz
            m = float(np.median(counts))
            check("imu_samples_per_scan", abs(m / exp - 1.0) < 0.2, round(m, 1), f"{exp:.1f} +-20 %")
    return {"recording": str(d), "profile": profile.get("description", ""), "passed": all(c["pass"] for c in checks.values()) and bool(checks), "checks": checks}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", type=Path)
    ap.add_argument("--profile", default="APEX_real")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    profile = load_sensor_profiles()[args.profile]
    res = audit(args.recording, profile)
    res["profile_name"] = args.profile
    for name, c in res["checks"].items():
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {name:30s} measured {c['measured']}  (required {c['required']})")
    print(f"{'PASSED' if res['passed'] else 'FAILED'}: {args.recording} against {args.profile}")
    if args.json:
        args.json.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
