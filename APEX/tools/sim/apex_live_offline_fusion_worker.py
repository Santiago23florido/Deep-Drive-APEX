#!/usr/bin/env python3
"""Periodically run compact offline fusion snapshots while capture is still ongoing."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _adaptive_snapshot_params(scan_count: int) -> tuple[int, int, int]:
    if scan_count < 80:
        return (1, 1, 1)
    if scan_count < 140:
        return (2, 2, 1)
    if scan_count < 240:
        return (3, 3, 1)
    if scan_count < 360:
        return (4, 4, 1)
    if scan_count < 520:
        return (5, 5, 2)
    if scan_count < 720:
        return (6, 6, 2)
    if scan_count < 960:
        return (8, 8, 3)
    return (10, 10, 4)


def _compact_lidar_csv(
    *,
    input_path: Path,
    output_path: Path,
    scan_count: int,
    recent_scan_keep: int,
    older_scan_stride: int,
    older_point_stride: int,
    recent_point_stride: int,
) -> tuple[int, int]:
    recent_start = max(0, scan_count - recent_scan_keep)
    kept_rows = 0
    kept_scans: set[int] = set()
    with input_path.open(newline="", encoding="utf-8") as src, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            try:
                scan_index = int(row["scan_index"])
                point_index = int(row["point_index"])
            except Exception:
                continue
            if scan_index >= recent_start:
                if (point_index % max(1, recent_point_stride)) != 0:
                    continue
            else:
                if (scan_index % max(1, older_scan_stride)) != 0:
                    continue
                if (point_index % max(1, older_point_stride)) != 0:
                    continue
            writer.writerow(row)
            kept_rows += 1
            kept_scans.add(scan_index)
    return (kept_rows, len(kept_scans))


def _copy_snapshot_inputs(
    *,
    run_dir: Path,
    snapshot_dir: Path,
    scan_count: int,
) -> dict[str, int]:
    shutil.copy2(run_dir / "imu_raw.csv", snapshot_dir / "imu_raw.csv")
    recent_scan_keep = 80
    older_scan_stride, older_point_stride, recent_point_stride = _adaptive_snapshot_params(scan_count)
    kept_points, kept_scans = _compact_lidar_csv(
        input_path=run_dir / "lidar_points.csv",
        output_path=snapshot_dir / "lidar_points.csv",
        scan_count=scan_count,
        recent_scan_keep=recent_scan_keep,
        older_scan_stride=older_scan_stride,
        older_point_stride=older_point_stride,
        recent_point_stride=recent_point_stride,
    )
    return {
        "recent_scan_keep": recent_scan_keep,
        "older_scan_stride": older_scan_stride,
        "older_point_stride": older_point_stride,
        "recent_point_stride": recent_point_stride,
        "kept_points": kept_points,
        "kept_scans": kept_scans,
    }


def _write_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _publish_snapshot_inputs(snapshot_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    published_lidar = output_dir / "sensor_fusion_lidar_points.csv"
    tmp_lidar = output_dir / ".sensor_fusion_lidar_points.csv.tmp"
    shutil.copy2(snapshot_dir / "lidar_points.csv", tmp_lidar)
    os.replace(tmp_lidar, published_lidar)
    return published_lidar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--sensor-fusion-script", required=True)
    parser.add_argument("--capture-status-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--status-json", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--interval-s", type=float, default=20.0)
    parser.add_argument("--min-scans", type=int, default=80)
    parser.add_argument("--min-new-scans", type=int, default=120)
    parser.add_argument("--python-executable", default="python3")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    sensor_fusion_script = Path(args.sensor_fusion_script).expanduser().resolve()
    capture_status_json = Path(args.capture_status_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    status_json = Path(args.status_json).expanduser().resolve()
    log_path = Path(args.log_path).expanduser().resolve()
    python_executable = str(args.python_executable)
    snapshot_dir = run_dir / ".live_offline_snapshot"

    last_snapshot_scan_count = 0
    last_launch_monotonic = 0.0
    snapshot_count = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _write_status(
        status_json,
        {
            "state": "waiting_capture",
            "run_dir": str(run_dir),
            "snapshot_count": 0,
            "last_completed_scan_count": 0,
        },
    )

    while True:
        capture_status = _load_json(capture_status_json)
        scan_count = int(capture_status.get("scan_count", 0) or 0)
        point_count = int(capture_status.get("lidar_point_count", 0) or 0)
        capture_state = str(capture_status.get("state", "unknown"))
        now = time.monotonic()
        time_since_last_launch = now - last_launch_monotonic
        enough_scans = scan_count >= int(args.min_scans)
        enough_new_scans = (scan_count - last_snapshot_scan_count) >= int(args.min_new_scans)
        due_by_time = time_since_last_launch >= float(args.interval_s)

        if not enough_scans or not enough_new_scans or not due_by_time:
            _write_status(
                status_json,
                {
                    "state": "waiting_next_snapshot",
                    "run_dir": str(run_dir),
                    "capture_state": capture_state,
                    "scan_count": scan_count,
                    "lidar_point_count": point_count,
                    "snapshot_count": snapshot_count,
                    "last_completed_scan_count": last_snapshot_scan_count,
                    "seconds_until_next_launch": max(0.0, float(args.interval_s) - time_since_last_launch),
                },
            )
            time.sleep(1.0)
            continue

        snapshot_count += 1
        last_launch_monotonic = now

        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        compact_stats = _copy_snapshot_inputs(
            run_dir=run_dir,
            snapshot_dir=snapshot_dir,
            scan_count=scan_count,
        )

        _write_status(
            status_json,
            {
                "state": "snapshot_copying",
                "run_dir": str(run_dir),
                "scan_count": scan_count,
                "lidar_point_count": point_count,
                "snapshot_count": snapshot_count,
                "snapshot_compaction": compact_stats,
            },
        )

        command = [
            python_executable,
            "-u",
            str(sensor_fusion_script),
            "--run-dir",
            str(snapshot_dir),
            "--output-dir",
            str(output_dir),
        ]

        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"\n=== snapshot {snapshot_count} scan_count={scan_count} "
                f"kept_scans={compact_stats['kept_scans']} kept_points={compact_stats['kept_points']} "
                f"older_scan_stride={compact_stats['older_scan_stride']} "
                f"older_point_stride={compact_stats['older_point_stride']} ===\n"
            )
            log_handle.flush()
            started = time.monotonic()
            completed = subprocess.run(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
            elapsed_s = time.monotonic() - started

        if completed.returncode == 0:
            published_lidar = _publish_snapshot_inputs(snapshot_dir, output_dir)
            trajectory_csv = output_dir / "sensor_fusion_trajectory.csv"
            summary_json = output_dir / "sensor_fusion_summary.json"
            if trajectory_csv.exists():
                trajectory_csv.touch()
            if summary_json.exists():
                summary_json.touch()
            last_snapshot_scan_count = scan_count
            _write_status(
                status_json,
                {
                    "state": "snapshot_ready",
                    "run_dir": str(run_dir),
                    "scan_count": scan_count,
                    "lidar_point_count": point_count,
                    "snapshot_count": snapshot_count,
                    "last_completed_scan_count": last_snapshot_scan_count,
                    "snapshot_compaction": compact_stats,
                    "elapsed_s": elapsed_s,
                    "output_dir": str(output_dir),
                    "published_lidar_points_csv": str(published_lidar),
                },
            )
        else:
            _write_status(
                status_json,
                {
                    "state": "snapshot_failed",
                    "run_dir": str(run_dir),
                    "scan_count": scan_count,
                    "lidar_point_count": point_count,
                    "snapshot_count": snapshot_count,
                    "return_code": completed.returncode,
                    "output_dir": str(output_dir),
                },
            )

        if capture_state == "finished" and scan_count <= last_snapshot_scan_count:
            break


if __name__ == "__main__":
    main()
