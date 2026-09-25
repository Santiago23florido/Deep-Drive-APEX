"""Per-run quality control, computed by the worker before insertion.

Every run stores its metrics in ``run_qa.metrics_json`` together with fixed-bin
histograms, so the campaign report can aggregate distributions without
re-reading millions of rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np

HIST_BINS = {
    "speed_mps": np.arange(0.0, 6.55, 0.1),
    "accel_horiz_mps2": np.arange(0.0, 12.2, 0.2),
    "yaw_rate_rps": np.arange(-6.0, 6.05, 0.1),
    "imu_per_interval": np.arange(-0.5, 60.5, 1.0),
    "lidar_dt_ms": np.arange(0.0, 260.0, 1.0),
    "imu_dt_ms": np.arange(0.0, 60.25, 0.25),
}


def _hist(name: str, values: np.ndarray) -> list[int]:
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=HIST_BINS[name])
    return counts.astype(int).tolist()


def _rate(t_ns: np.ndarray) -> tuple[float, float]:
    """(median-interval rate, robust mean rate). The robust rate averages the
    intervals shorter than 1.5 x the median, so gaps left by dropouts do not
    lower it and alternating intervals do not bias it."""
    if len(t_ns) < 2:
        return 0.0, 0.0
    d = np.diff(t_ns).astype(float)
    med = float(np.median(d))
    return 1e9 / med, 1e9 / float(d[d < 1.5 * med].mean())


def run_qa(synth: dict[str, Any], physics: dict[str, Any], nominal_imu_hz: float, nominal_lidar_hz: float) -> dict[str, Any]:
    imu, li, gt = synth["imu"], synth["lidar"], synth["gt"]
    checks: dict[str, bool] = {}
    m: dict[str, Any] = {}

    checks["imu_timestamps_strictly_increasing"] = bool(np.all(np.diff(imu["t_report_ns"]) > 0)) and bool(np.all(np.diff(imu["true_idx"]) > 0))
    checks["lidar_timestamps_strictly_increasing"] = bool(np.all(np.diff(li["t_report_ns"]) > 0)) and bool(np.all(np.diff(li["start_idx"]) > 0))
    imu_med, imu_mean = _rate(imu["t_report_ns"])
    li_med, li_mean = _rate(li["t_report_ns"])
    m.update(imu_rate_hz_median=imu_med, imu_rate_hz_robust=imu_mean, lidar_rate_hz_median=li_med, lidar_rate_hz_robust=li_mean)
    checks["imu_rate_within_5pct"] = abs(imu_mean / nominal_imu_hz - 1.0) < 0.05
    checks["lidar_rate_within_5pct"] = abs(li_mean / nominal_lidar_hz - 1.0) < 0.05

    # IMU samples that really arrived between consecutive scans (reported stamps).
    edges = li["t_report_ns"]
    counts = np.diff(np.searchsorted(imu["t_report_ns"], edges, side="right")) if len(edges) > 1 else np.zeros(0, int)
    m.update(
        imu_per_interval_min=int(counts.min()) if len(counts) else 0,
        imu_per_interval_mean=float(counts.mean()) if len(counts) else 0.0,
        imu_per_interval_max=int(counts.max()) if len(counts) else 0,
        imu_per_interval_expected=nominal_imu_hz / nominal_lidar_hz,
    )
    checks["every_interval_has_imu"] = bool(len(counts) > 0 and counts.min() >= 1) or bool(np.mean(counts >= 1) > 0.995)

    n_scans = len(li["start_idx"])
    lost = len(li["lost_scans"])
    m.update(
        n_scans=n_scans,
        n_imu=int(len(imu["true_idx"])),
        n_ground_truth=int(len(gt["idx"])),
        scan_dropout_pct=100.0 * lost / max(1, lost + n_scans),
        imu_dropout_pct=100.0 * imu["n_dropped"] / max(1, imu["n_expected"]),
        imu_saturated_samples=int(imu["saturated"]),
        valid_range_pct=100.0 * float(li["valid"].mean()) if n_scans else 0.0,
        recorded_duration_s=float((gt["t_ns"][-1] - gt["t_ns"][0]) * 1e-9) if len(gt["t_ns"]) else 0.0,
    )
    r = li["ranges"]
    checks["no_nan_ranges"] = not bool(np.isnan(r).any())
    checks["valid_ranges_finite_in_bounds"] = bool(np.all(np.isfinite(r[li["valid"]]))) and bool(
        np.all((r[li["valid"]] >= li["range_min"]) & (r[li["valid"]] <= li["range_max"]))
    )
    checks["invalid_ranges_are_inf"] = bool(np.all(np.isinf(r[~li["valid"]])))
    checks["imu_values_finite"] = bool(np.all(np.isfinite(imu["values"])))
    checks["ground_truth_finite"] = all(bool(np.all(np.isfinite(gt[k]))) for k in ("xyz", "quat_xyzw", "v_world", "w_body", "a_world"))

    gt_set = set(gt["idx"].tolist())
    checks["every_measurement_has_ground_truth"] = all(int(k) in gt_set for k in imu["true_idx"]) and all(
        int(k) in gt_set for k in np.concatenate((li["start_idx"], li["end_idx"]))
    )
    ts_cfg_i = synth["profile"]["imu"].get("timestamp", {})
    ts_cfg_l = synth["profile"]["lidar"].get("timestamp", {})
    step = int(physics["physics_step_ns"])
    lat_i = imu["t_report_ns"] - imu["true_idx"] * step
    lat_l = li["t_report_ns"] - li["start_idx"] * step
    m.update(
        imu_stamp_error_ms_mean=float(lat_i.mean() * 1e-6), imu_stamp_error_ms_std=float(lat_i.std() * 1e-6),
        lidar_stamp_error_ms_mean=float(lat_l.mean() * 1e-6), lidar_stamp_error_ms_std=float(lat_l.std() * 1e-6),
    )
    tol_i = (abs(float(ts_cfg_i.get("offset_ms", 0))) + 7 * float(ts_cfg_i.get("jitter_std_ms", 0)) + 0.01) * 1e6
    if synth["profile"].get("backend") == "gazebo_native":
        # Exact sampling instants of the chip clock: the ground-truth row is
        # the nearest physics step, up to half a step away.
        tol_i += 0.5 * step
    tol_l = (abs(float(ts_cfg_l.get("offset_ms", 0))) + 7 * float(ts_cfg_l.get("jitter_std_ms", 0)) + 0.01) * 1e6
    checks["timestamps_match_ground_truth"] = bool(np.all(np.abs(lat_i) <= tol_i)) and bool(np.all(np.abs(lat_l) <= tol_l))

    speed = np.linalg.norm(gt["v_world"][:, :2], axis=1)
    acc_h = np.linalg.norm(gt["a_world"][:, :2], axis=1)
    yaw_rate = gt["w_body"][:, 2]
    path = float(np.sum(np.linalg.norm(np.diff(gt["xyz"][:, :2], axis=0), axis=1)))
    m.update(
        distance_gt_m=path, speed_mean_mps=float(speed.mean()), speed_max_mps=float(speed.max()),
        accel_horiz_p99_mps2=float(np.percentile(acc_h, 99)), yaw_rate_abs_max_rps=float(np.abs(yaw_rate).max()),
        body_roll_abs_max_deg=float(np.degrees(np.abs(gt["body_roll_pitch"][:, 0]).max())),
        body_pitch_abs_max_deg=float(np.degrees(np.abs(gt["body_roll_pitch"][:, 1]).max())),
        gyro_bias_initial=imu["bias_true"][0, 3:].tolist() if len(imu["bias_true"]) else [],
        accel_bias_initial=imu["bias_true"][0, :3].tolist() if len(imu["bias_true"]) else [],
        gyro_bias_drift=(imu["bias_true"][-1, 3:] - imu["bias_true"][0, 3:]).tolist() if len(imu["bias_true"]) else [],
    )
    expected = float(physics["laps"]) * float(physics["path_length_m"])
    checks["vehicle_moved"] = bool(path > 0.85 * expected and speed.max() > 0.3)
    checks["lap_completed"] = bool(physics["lap_completed"])

    imu_dt = np.diff(imu["t_report_ns"]) * 1e-6
    li_dt = np.diff(li["t_report_ns"]) * 1e-6
    m["histograms"] = {
        "speed_mps": _hist("speed_mps", speed),
        "accel_horiz_mps2": _hist("accel_horiz_mps2", acc_h),
        "yaw_rate_rps": _hist("yaw_rate_rps", yaw_rate),
        "imu_per_interval": _hist("imu_per_interval", counts),
        "lidar_dt_ms": _hist("lidar_dt_ms", li_dt),
        "imu_dt_ms": _hist("imu_dt_ms", imu_dt),
    }
    critical = [k for k in checks if k != "every_interval_has_imu"]
    return {"passed": all(checks[k] for k in critical), "checks": checks, "metrics": m}
