"""Campaign report: one row per run (JSON + CSV) and distribution plots."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np

from .db import connect
from .qa import HIST_BINS


def _profiles() -> dict[str, Any]:
    from .sensors import load_sensor_profiles

    return load_sensor_profiles()


def _db_bytes(path: Path) -> int:
    return sum(Path(str(path) + s).stat().st_size for s in ("", "-wal", "-shm") if Path(str(path) + s).exists())


def build_report(db_path: Path, out_dir: Path, integrity: dict[str, Any] | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path, readonly=True)
    conn.row_factory = sqlite3.Row
    meta = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM dataset_meta")}
    runs = [dict(r) for r in conn.execute(
        "SELECT r.*, q.metrics_json, q.passed AS qa_row_passed FROM runs r LEFT JOIN run_qa q ON q.run_id = r.run_id ORDER BY r.run_id"
    )]
    counts = {
        "lidar_scans": conn.execute("SELECT COUNT(*) FROM lidar_scans").fetchone()[0],
        "imu_samples": conn.execute("SELECT COUNT(*) FROM imu_samples").fetchone()[0],
        "ground_truth": conn.execute("SELECT COUNT(*) FROM ground_truth").fetchone()[0],
        "events": conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
    }
    if integrity is None:
        integrity = {
            "integrity_check": conn.execute("PRAGMA integrity_check").fetchone()[0],
            "foreign_key_violations": len(conn.execute("PRAGMA foreign_key_check").fetchall()),
        }
    conn.close()

    profiles = _profiles()
    rows = []
    hist_tot: dict[str, np.ndarray] = {k: np.zeros(len(v) - 1, dtype=np.int64) for k, v in HIST_BINS.items()}
    hist_split: dict[str, dict[str, np.ndarray]] = {}
    for r in runs:
        m = json.loads(r["metrics_json"])["metrics"] if r["metrics_json"] else {}
        checks = json.loads(r["metrics_json"])["checks"] if r["metrics_json"] else {}
        row = {
            "run_id": r["run_id"], "run_key": r["run_key"], "split": r["split"], "track": r["track_name"], "motion": r["motion_profile"],
            "sensor": r["sensor_profile"], "seed": r["seed"], "direction": r["direction"], "status": r["status"], "qa_passed": r["qa_passed"],
            "failure_reason": r["failure_reason"] or "", "lap_completed": r["lap_completed"], "simulated_duration_s": r["simulated_duration_s"],
            "distance_m": r["distance_m"], "v_ref_mps": r["v_ref_mps"],
        }
        for key in (
            "n_scans", "n_imu", "lidar_rate_hz_median", "imu_rate_hz_median", "imu_per_interval_min", "imu_per_interval_mean",
            "imu_per_interval_max", "scan_dropout_pct", "imu_dropout_pct", "valid_range_pct", "imu_saturated_samples",
            "speed_mean_mps", "speed_max_mps", "accel_horiz_p99_mps2", "yaw_rate_abs_max_rps", "body_roll_abs_max_deg",
            "imu_stamp_error_ms_std", "lidar_stamp_error_ms_std",
        ):
            row[key] = m.get(key)
        gb = m.get("gyro_bias_initial") or [None] * 3
        ab = m.get("accel_bias_initial") or [None] * 3
        row.update(gyro_bias_z_initial=gb[2], accel_bias_x_initial=ab[0], accel_bias_y_initial=ab[1])
        row["failed_checks"] = ";".join(k for k, v in checks.items() if not v)
        prof = profiles.get(r["sensor_profile"], {})
        row["lidar_hz"] = prof.get("lidar", {}).get("rate_hz")
        row["imu_hz"] = prof.get("imu", {}).get("rate_hz")
        rows.append(row)
        if r["status"] == "completed":
            for k, v in m.get("histograms", {}).items():
                arr = np.asarray(v, dtype=np.int64)
                hist_tot[k] += arr
                hist_split.setdefault(r["split"], {k2: np.zeros(len(b) - 1, dtype=np.int64) for k2, b in HIST_BINS.items()})[k] += arr

    completed = [r for r in rows if r["status"] == "completed"]
    held = json.loads(meta.get("held_out_variants", "{}"))
    leaked = [r["run_key"] for r in rows if r["track"] in held.get("tracks", []) or r["motion"] in held.get("motions", []) or r["sensor"] in held.get("sensors", [])]
    by_split: dict[str, Any] = {}
    for sp in ("train", "validation", "test"):
        sel = [r for r in completed if r["split"] == sp]
        by_split[sp] = {
            "runs": len(sel),
            "trajectories": len({r["run_key"].rsplit("__", 1)[0] for r in sel}),
            "tracks": sorted({r["track"] for r in sel}),
            "simulated_h": sum(r["simulated_duration_s"] or 0 for r in sel) / 3600.0,
            "distance_km": sum(r["distance_m"] or 0 for r in sel) / 1000.0,
            "scans": sum(r["n_scans"] or 0 for r in sel),
            "imu_samples": sum(r["n_imu"] or 0 for r in sel),
        }
    n_done = max(1, len(completed))
    # Trajectories must never cross splits.
    traj_split: dict[str, set[str]] = {}
    for r in rows:
        traj_split.setdefault(r["run_key"].rsplit("__", 1)[0], set()).add(r["split"])
    cross = [k for k, v in traj_split.items() if len(v) > 1]
    rate_obs: dict[str, dict[str, float]] = {}
    for sensor in sorted({r["sensor"] for r in completed}):
        sel = [r for r in completed if r["sensor"] == sensor]
        rate_obs[sensor] = {
            "lidar_hz_median": float(np.median([r["lidar_rate_hz_median"] for r in sel])),
            "imu_hz_median": float(np.median([r["imu_rate_hz_median"] for r in sel])),
            "imu_per_interval_min": int(min(r["imu_per_interval_min"] for r in sel)),
            "imu_per_interval_mean": float(np.mean([r["imu_per_interval_mean"] for r in sel])),
            "imu_per_interval_max": int(max(r["imu_per_interval_max"] for r in sel)),
            "runs": len(sel),
        }
    realizations = {
        tuple(json.loads(r["metrics_json"])["metrics"].get("gyro_bias_initial", []) + json.loads(r["metrics_json"])["metrics"].get("accel_bias_initial", []))
        for r in runs if r["metrics_json"]
    }
    report = {
        "database": str(db_path),
        "distinct_imu_realizations": len(realizations),
        "database_bytes": _db_bytes(db_path),
        "dataset_meta": {k: (json.loads(v) if v.startswith(("{", "[")) else v) for k, v in meta.items()},
        "integrity": integrity,
        "runs_total": len(rows),
        "runs_completed": len(completed),
        "runs_failed": sum(1 for r in rows if r["status"] == "failed"),
        "runs_other": sum(1 for r in rows if r["status"] not in ("completed", "failed")),
        "failures": [{"run_key": r["run_key"], "reason": r["failure_reason"]} for r in rows if r["status"] == "failed"],
        "row_counts": counts,
        "split_distribution": by_split,
        "split_fractions_runs": {k: v["runs"] / n_done for k, v in by_split.items()},
        "observed_rates_by_sensor": rate_obs,
        "trajectories_crossing_splits": cross,
        "held_out_variants_in_database": leaked,
        **(extra or {}),
    }
    (out_dir / "campaign_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    with open(out_dir / "runs.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["run_id"])
        writer.writeheader()
        writer.writerows(rows)
    if completed:
        _plots(completed, hist_tot, hist_split, by_split, out_dir)
    return report


SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3 (dataviz reference palette)
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SPLITS = ("train", "validation", "test")


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _plots(rows: list[dict[str, Any]], hist: dict[str, np.ndarray], hist_split: dict[str, dict[str, np.ndarray]], by_split: dict[str, Any], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sensors = sorted({r["sensor"] for r in rows})
    s_color = {s: SERIES[i % 3] for i, s in enumerate(sensors[:3])}
    split_color = {s: SERIES[i] for i, s in enumerate(SPLITS)}
    fig, axes = plt.subplots(3, 3, figsize=(17, 13), facecolor="#fcfcfb")
    for ax, key, label in ((axes[0, 0], "lidar_rate_hz_median", "LiDAR"), (axes[0, 1], "imu_rate_hz_median", "IMU")):
        nominal_key = "lidar_hz" if label == "LiDAR" else "imu_hz"
        lo, hi = np.inf, 0.0
        for s in sensors[:3]:
            sel = [r for r in rows if r["sensor"] == s]
            nom = [r[nominal_key] for r in sel]
            obs = [r[key] for r in sel]
            lo, hi = min(lo, min(nom), min(obs)), max(hi, max(nom), max(obs))
            ax.scatter(nom, obs, s=40, color=s_color[s], edgecolor="#fcfcfb", linewidth=1.5, label=s, zorder=3)
        pad = 0.08 * (hi - lo + 1)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK_2, lw=1, ls="--", zorder=2, label="observed = nominal")
        _style(ax, f"Observed {label} rate per run", "nominal rate [Hz]", "observed median rate [Hz]")
        ax.legend(fontsize=8, frameon=False)
    for ax, key, title, xlabel, xlim in (
        (axes[0, 2], "imu_per_interval", "IMU samples between consecutive scans", "samples in the interval", (0, 40)),
        (axes[1, 0], "speed_mps", "Speed, ground truth", "m/s", None),
        (axes[1, 1], "accel_horiz_mps2", "Horizontal acceleration, ground truth", "m/s²", None),
        (axes[1, 2], "yaw_rate_rps", "Yaw rate, ground truth", "rad/s", None),
    ):
        bins = HIST_BINS[key]
        centers = 0.5 * (bins[1:] + bins[:-1])
        for split in SPLITS:
            if split in hist_split:
                h = hist_split[split][key]
                ax.step(centers, h / max(1, h.sum()), where="mid", color=split_color[split], lw=2, label=split)
        _style(ax, title, xlabel, "fraction of samples")
        if xlim:
            ax.set_xlim(*xlim)
        ax.legend(fontsize=8, frameon=False)
    ax = axes[2, 0]
    for s in sensors[:3]:
        sel = [r for r in rows if r["sensor"] == s and r["gyro_bias_z_initial"] is not None]
        ax.scatter([np.degrees(r["gyro_bias_z_initial"]) for r in sel], [r["accel_bias_x_initial"] for r in sel], s=30, color=s_color[s], edgecolor="#fcfcfb", linewidth=1, label=s)
    _style(ax, "True initial IMU bias per run", "gyro z bias [deg/s]", "accelerometer x bias [m/s²]")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[2, 1]
    for s in sensors[:3]:
        sel = [r for r in rows if r["sensor"] == s]
        ax.scatter([r["imu_dropout_pct"] for r in sel], [r["scan_dropout_pct"] for r in sel], s=30, color=s_color[s], edgecolor="#fcfcfb", linewidth=1, label=s)
    _style(ax, "Dropout per run", "IMU samples lost [%]", "LiDAR scans lost [%]")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[2, 2]
    present = [s for s in SPLITS if by_split[s]["runs"]]
    x = np.arange(len(present))
    bars = ax.bar(x, [by_split[s]["runs"] for s in present], 0.55, color=[split_color[s] for s in present])
    for b, s in zip(bars, present):
        ax.annotate(f"{by_split[s]['runs']} runs\n{by_split[s]['distance_km']:.1f} km, {by_split[s]['simulated_h']:.2f} h",
                    (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom", fontsize=8, color=INK, xytext=(0, 3), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n" + "\n".join(by_split[s]["tracks"]) for s in present], fontsize=7, color=INK_2)
    ax.set_ylim(0, max(by_split[s]["runs"] for s in present) * 1.25)
    _style(ax, "Runs per split", "", "completed runs")
    fig.tight_layout()
    fig.savefig(out_dir / "campaign_distributions.png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
