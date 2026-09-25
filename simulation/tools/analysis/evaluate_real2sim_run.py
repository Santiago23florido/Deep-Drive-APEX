#!/usr/bin/env python3
"""Evaluate a real2sim closed-loop run (tools/sim/apex_real2sim_up.sh).

Reads the run directory and writes ``evaluation.json`` and ``evaluation.png``:

* lap: the referee's verdict (completed / failed and why, true progress,
  maximum lateral error);
* odometry: every estimate against the true increment between the true
  first samples of consecutive revolutions, with the metrics of the offline
  evaluation (``learning/lidar_imu_pose/odometry_metrics``: speed error, RPE,
  KITTI-style segment drift, sigma coverage), so live and offline numbers are
  comparable;
* SLAM: the map metrics of ``plot_slam_maps`` (precision, coverage, chamfer,
  ATE), when present;
* real time: estimator latency (scan stamp -> estimate) and compute time,
  simulation real-time factor, the timing of the sensor streams the car
  received (IMU and LiDAR rates, period jitter);
* optional offline reference (``--offline-db``): the same network on the
  dataset run of the same format (open loop), for comparison.

Usage:
    learning/.venv/bin/python tools/analysis/evaluate_real2sim_run.py <run_dir> [--offline-db data/multiscenario_pose/pose_dataset_v2.sqlite3]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM / "learning" / "lidar_imu_pose"))
sys.path.insert(0, str(SIM / "tools"))

import odometry_metrics as om  # noqa: E402


def _csv(path: Path) -> np.ndarray | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    d = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return np.atleast_1d(d) if d.size else None


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def odometry(run: Path) -> tuple[dict, dict]:
    est = _csv(run / "estimator.csv")
    tru = _csv(run / "truth_scans.csv")
    if est is None or tru is None:
        return {"error": "missing estimator.csv or truth_scans.csv"}, {}
    stamps = tru["stamp_ns"].astype(np.int64)
    order = np.argsort(stamps)
    tru = tru[order]
    stamps = stamps[order]
    pose = np.column_stack((tru["x"], tru["y"], tru["yaw"])).astype(float)
    n = len(stamps)
    target = np.zeros((n, 3))
    d = pose[1:, :2] - pose[:-1, :2]
    c, s = np.cos(pose[:-1, 2]), np.sin(pose[:-1, 2])
    target[1:, 0] = c * d[:, 0] + s * d[:, 1]
    target[1:, 1] = -s * d[:, 0] + c * d[:, 1]
    target[1:, 2] = _wrap(pose[1:, 2] - pose[:-1, 2])
    dt = np.zeros(n)
    dt[1:] = np.diff(stamps) * 1e-9
    dt[0] = dt[1] if n > 1 else 0.077
    pred = np.full((n, 3), np.nan)
    sigma = np.full((n, 3), np.nan)
    idx = {int(t): i for i, t in enumerate(stamps)}
    matched = 0
    for row in est:
        i = idx.get(int(row["stamp_ns"]))
        if i is None or i == 0:
            continue
        pred[i] = (row["dx"], row["dy"], row["dyaw"])
        sigma[i] = (row["sx"], row["sy"], row["syaw"])
        matched += 1
    cfg = json.loads((run / "run_config.json").read_text())
    truth = om.SplitTruth(target=target, dt=dt, t_s=stamps * 1e-9,
                          runs=[{"offset": 0, "n": n, "sensor": cfg["sensor_profile"], "motion": cfg["motion"], "track": cfg["track"], "run_key": run.name}])
    metrics = om.evaluate(pred, truth, sigma=sigma, groups=())["all"]
    lat = np.asarray(est["latency_ms"], float)
    comp = np.asarray(est["compute_ms"], float)
    out = {
        "scans": int(n), "estimates_matched": int(matched),
        "speed_err_cmps": metrics.get("speed_err_cmps"), "rpe_1s_cm": metrics.get("rpe_1s_cm"),
        "segment_drift_pct": metrics.get("t_rel_pct"), "heading_drift_degpm": metrics.get("r_rel_degpm"),
        "yawrate_err_dps": metrics.get("yawrate_err_dps"), "cover1_pct": metrics.get("cover1_trans_pct", metrics.get("cover1_pct")),
        "all_metrics": metrics,
        "latency_ms": {"p50": float(np.median(lat)), "p95": float(np.percentile(lat, 95)), "max": float(lat.max())},
        "compute_ms": {"p50": float(np.median(comp)), "p95": float(np.percentile(comp, 95)), "max": float(comp.max())},
        "scan_period_ms": float(np.median(np.diff(stamps)) * 1e-6),
    }
    arrays = {"pred": pred, "target": target, "truth_pose": pose, "stamps": stamps}
    return out, arrays


def control_pose(run: Path, cfg: dict) -> dict:
    """Error of the pose the driver used (belief) against the truth, and the
    start placement error the format introduced."""
    s, n = cfg["spawn_true"], cfg["nominal_start"]
    out = {"start_pose": cfg.get("start_pose", "nominal"),
           "placement_error": {"xy_m": math.hypot(s[0] - n[0], s[1] - n[1]), "yaw_deg": math.degrees(s[2] - n[2])}}
    drv, tr = _csv(run / "driver.csv"), _csv(run / "truth_track.csv")
    if drv is None or tr is None:
        return out
    drv = drv[drv["phase"] != "wait"]
    if len(drv) == 0:
        return out
    t = tr["t_ns"] * 1e-9
    e = np.hypot(drv["x"] - np.interp(drv["t"], t, tr["x"]), drv["y"] - np.interp(drv["t"], t, tr["y"]))
    out["belief_error_m"] = {"p50": float(np.median(e)), "p95": float(np.percentile(e, 95)), "max": float(e.max())}
    out["pose_age_ms_p95"] = float(np.percentile(drv["pose_age_s"], 95) * 1e3)
    return out


def sensor_timing(run: Path) -> dict:
    imu = _csv(run / "imu_raw.csv")
    tru = _csv(run / "truth_scans.csv")
    res = {}
    if imu is not None:
        t = imu["stamp_sec"] + imu["stamp_nanosec"] * 1e-9
        d = np.diff(np.sort(t))
        res["imu"] = {"rate_hz": float(1 / np.median(d)), "period_ms_p5_p95": [float(x) for x in np.percentile(d, [5, 95]) * 1e3], "samples": int(len(t))}
    if tru is not None:
        st = np.sort(tru["stamp_ns"]) * 1e-9
        d = np.diff(st)
        res["lidar"] = {"rate_hz": float(1 / np.median(d)), "period_ms_p5_p95": [float(x) for x in np.percentile(d, [5, 95]) * 1e3],
                        "valid_bins_p50": float(np.median(tru["valid_bins"])),
                        "stamp_minus_true_start_ms_p50": float(np.median(tru["stamp_ns"] - tru["t_start_ns"]) * 1e-6)}
    return res


def real_time(run: Path) -> dict:
    wall = json.loads((run / "wall.json").read_text()) if (run / "wall.json").exists() else None
    track = _csv(run / "truth_track.csv")
    if wall is None or track is None:
        return {}
    sim_s = float(track["t_ns"][-1] - track["t_ns"][0]) * 1e-9
    wall_s = float(wall["wall_end"]) - float(wall["wall_start"])
    return {"sim_s": sim_s, "wall_s": wall_s, "real_time_factor_mean": sim_s / wall_s if wall_s > 0 else None, "end_reason": wall.get("end_reason")}


def offline_reference(run: Path, db: Path) -> dict:
    """The network on the dataset run of the same format (open loop, exact driver)."""
    import torch  # noqa: PLC0415

    from icp_odometry import ICP_VARIANTS, run_icp  # noqa: PLC0415
    from sqlite_streams import StreamData, load_split  # noqa: PLC0415
    from streaming_model import load_checkpoint  # noqa: PLC0415
    from train_streaming import CACHE, stream_predict  # noqa: PLC0415

    cfg = json.loads((run / "run_config.json").read_text())
    key = f"{cfg['trajectory_key']}__{cfg['sensor_profile']}"
    for split in ("validation", "test", "train"):
        data = load_split(db, CACHE, split)
        hit = [i for i, r in enumerate(data["runs"]) if r["run_key"] == key]
        if hit:
            break
    else:
        return {"error": f"{key} not in {db}"}
    i = hit[0]
    r = data["runs"][i]
    sel = np.arange(r["offset"], r["offset"] + r["n"])
    n_total = data["target"].shape[0]
    one = {k: (v[sel] if torch.is_tensor(v) and v.shape[:1] == (n_total,) else v) for k, v in data.items()}
    for k in ("range_max", "angle_min", "angle_increment", "time_increment_s", "lidar_xy", "lidar_sigma", "beam_time_frac"):
        if k in data:
            one[k] = data[k][i : i + 1]
    one["runs"] = [{**r, "offset": 0}]
    one["run_index"] = torch.zeros(r["n"], dtype=torch.int64)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = StreamData(one, dev)
    sd.max_len = 40
    model, ck = load_checkpoint(cfg["checkpoint"], dev)
    if model.hybrid:
        sd.attach_icp(run_icp(sd, ICP_VARIANTS[ck["config"]["icp_variant"]], log=lambda m: None))
    pred = stream_predict(model, sd)["pred"].numpy()
    m = om.evaluate(pred, om.split_truth(one), groups=())["all"]
    return {"run_key": key, "split": split, "speed_err_cmps": m["speed_err_cmps"], "rpe_1s_cm": m["rpe_1s_cm"],
            "segment_drift_pct": m["t_rel_pct"], "heading_drift_degpm": m["r_rel_degpm"]}


def figure(run: Path, arrays: dict, result: dict) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), facecolor="#fcfcfb")
    ax = axes[0]
    track = _csv(run / "slam" / "track_truth_points.csv")
    if track is not None:
        ax.scatter(track["x_m"], track["y_m"], s=1, c="#b8b8b8", label="real track walls")
    truth = arrays.get("truth_pose")
    if truth is not None:
        ax.plot(truth[:, 0], truth[:, 1], color="#222222", lw=2.2, label="ground truth")
        pred = np.nan_to_num(arrays["pred"][1:], nan=0.0)
        odo = om.compose(pred)
        c, s = math.cos(truth[0, 2]), math.sin(truth[0, 2])
        ax.plot(truth[0, 0] + c * odo[:, 0] - s * odo[:, 1], truth[0, 1] + s * odo[:, 0] + c * odo[:, 1], color="#2a78c8", lw=1.4,
                label="learned odometry (anchored at the true start)")
    slam = _csv(run / "slam" / "slam_learned_trajectory.csv")
    if slam is not None:
        ax.plot(slam["x_world"], slam["y_world"], color="#e4572e", lw=1.2, ls="--", label="SLAM trajectory")
    drv = _csv(run / "driver.csv")
    if drv is not None:
        ax.plot(drv["x"], drv["y"], color="#9467bd", lw=0.8, alpha=0.8, label="pose the driver used")
    ax.set_aspect("equal")
    lap = result.get("lap", {})
    ax.set_title(f"{run.name}\nlap: {lap.get('status', '?')} {lap.get('failure_reason', '')}", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax = axes[1]
    if truth is not None:
        err = np.linalg.norm(np.nan_to_num(arrays["pred"][:, :2]) - arrays["target"][:, :2], axis=1) * 100
        t = (arrays["stamps"] - arrays["stamps"][0]) * 1e-9
        ax.plot(t[1:], err[1:], color="#2a78c8", lw=0.8)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("translation error per scan interval [cm]")
        o = result.get("odometry", {})
        ax.set_title(f"speed error {o.get('speed_err_cmps', float('nan')):.2f} cm/s, segment drift {o.get('segment_drift_pct', float('nan')):.2f} %, "
                     f"latency p95 {o.get('latency_ms', {}).get('p95', float('nan')):.0f} ms", fontsize=10)
    fig.tight_layout()
    fig.savefig(run / "evaluation.png", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--offline-db", type=Path, default=None)
    args = ap.parse_args()
    run = args.run_dir.resolve()
    result: dict = {"run": run.name, "config": json.loads((run / "run_config.json").read_text())}
    result["lap"] = json.loads((run / "run_result.json").read_text()) if (run / "run_result.json").exists() else {"status": "no verdict"}
    result["odometry"], arrays = odometry(run)
    if (run / "slam_metrics.json").exists():
        sm = json.loads((run / "slam_metrics.json").read_text())
        result["slam"] = sm.get("slams", {}).get("learned", sm)
    result["control"] = control_pose(run, result["config"])
    result["sensor_timing"] = sensor_timing(run)
    result["real_time"] = real_time(run)
    if args.offline_db is not None:
        try:
            result["offline_reference"] = offline_reference(run, args.offline_db)
        except Exception as exc:  # the live evaluation stands on its own
            result["offline_reference"] = {"error": str(exc)}
    (run / "evaluation.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    if arrays:
        figure(run, arrays, result)
    o = result["odometry"]
    print(json.dumps({"lap": {k: result["lap"].get(k) for k in ("status", "failure_reason", "progress_m", "planned_m", "max_lateral_error_m")},
                      "odometry": {k: o.get(k) for k in ("speed_err_cmps", "rpe_1s_cm", "segment_drift_pct", "heading_drift_degpm", "latency_ms", "compute_ms")},
                      "slam": {k: result.get("slam", {}).get(k) for k in ("anchored", "ate")}, "control": result["control"],
                      "sensor_timing": result["sensor_timing"], "real_time": result["real_time"]}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
