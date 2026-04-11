#!/usr/bin/env python3
"""Periodically build a lightweight preview fixed map while capture is still running."""

from __future__ import annotations

import argparse
import json
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


def _write_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _snapshot_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mapper-script", required=True)
    parser.add_argument("--capture-status-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--status-json", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--interval-s", type=float, default=12.0)
    parser.add_argument("--min-scans", type=int, default=60)
    parser.add_argument("--min-new-scans", type=int, default=40)
    parser.add_argument("--python-executable", default="python3")
    parser.add_argument("--evaluation-world", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    mapper_script = Path(args.mapper_script).expanduser().resolve()
    capture_status_json = Path(args.capture_status_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    status_json = Path(args.status_json).expanduser().resolve()
    log_path = Path(args.log_path).expanduser().resolve()
    python_executable = str(args.python_executable)
    evaluation_world = str(args.evaluation_world or "").strip()
    snapshot_dir = run_dir / ".mapping_preview_snapshot"

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    last_snapshot_scan_count = 0
    last_launch_monotonic = 0.0
    snapshot_count = 0
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
        odom_count = int(capture_status.get("odom_row_count", 0) or 0)
        capture_state = str(capture_status.get("state", "unknown"))
        now = time.monotonic()
        enough_scans = scan_count >= int(args.min_scans)
        enough_new_scans = (scan_count - last_snapshot_scan_count) >= int(args.min_new_scans)
        due_by_time = (now - last_launch_monotonic) >= float(args.interval_s)

        if not enough_scans or not enough_new_scans or not due_by_time:
            _write_status(
                status_json,
                {
                    "state": "waiting_next_snapshot",
                    "run_dir": str(run_dir),
                    "capture_state": capture_state,
                    "scan_count": scan_count,
                    "lidar_point_count": point_count,
                    "odom_row_count": odom_count,
                    "snapshot_count": snapshot_count,
                    "last_completed_scan_count": last_snapshot_scan_count,
                    "seconds_until_next_launch": max(0.0, float(args.interval_s) - (now - last_launch_monotonic)),
                },
            )
            if capture_state == "finished" and scan_count <= last_snapshot_scan_count:
                break
            time.sleep(1.0)
            continue

        snapshot_count += 1
        last_launch_monotonic = now
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        _snapshot_file(run_dir / "lidar_points.csv", snapshot_dir / "lidar_points.csv")
        _snapshot_file(run_dir / "odom_fused.csv", snapshot_dir / "odom_fused.csv")
        _snapshot_file(run_dir / "imu_raw.csv", snapshot_dir / "imu_raw.csv")
        _snapshot_file(run_dir / "capture_summary.json", snapshot_dir / "capture_summary.json")

        _write_status(
            status_json,
            {
                "state": "building_preview",
                "run_dir": str(run_dir),
                "scan_count": scan_count,
                "lidar_point_count": point_count,
                "odom_row_count": odom_count,
                "snapshot_count": snapshot_count,
                "output_dir": str(output_dir),
            },
        )

        command = [
            python_executable,
            "-u",
            str(mapper_script),
            "--run-dir",
            str(snapshot_dir),
            "--output-dir",
            str(output_dir),
            "--local-registration",
            "multires_distance_field",
            "--loop-closure-descriptor",
            "polar_occupancy",
            "--optimizer-loss",
            "cauchy",
            "--max-keyframes",
            "220",
            "--keyframe-distance-m",
            "0.16",
            "--keyframe-yaw-deg",
            "6.0",
            "--keyframe-time-s",
            "1.0",
            "--max-points-per-keyframe",
            "160",
            "--voxel-size-m",
            "0.03",
            "--map-resolution-m",
            "0.04",
            "--submap-keyframes",
            "10",
            "--evaluation-json",
            str(output_dir / "mapping_evaluation.json"),
        ]
        if evaluation_world:
            command.extend(["--evaluation-world", evaluation_world])

        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"\n=== mapping preview snapshot {snapshot_count} scan_count={scan_count} "
                f"lidar_points={point_count} odom_rows={odom_count} ===\n"
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
            summary_payload = _load_json(output_dir / "mapping_summary.json")
            evaluation_payload = _load_json(output_dir / "mapping_evaluation.json")
            last_snapshot_scan_count = scan_count
            _write_status(
                status_json,
                {
                    "state": "preview_ready",
                    "run_dir": str(run_dir),
                    "scan_count": scan_count,
                    "lidar_point_count": point_count,
                    "odom_row_count": odom_count,
                    "snapshot_count": snapshot_count,
                    "last_completed_scan_count": last_snapshot_scan_count,
                    "last_preview_duration_s": elapsed_s,
                    "keyframe_count": int(summary_payload.get("keyframe_count", 0) or 0),
                    "edge_count": int(summary_payload.get("edge_count", 0) or 0),
                    "loop_closure_count": int(summary_payload.get("loop_closure_count", 0) or 0),
                    "map_cells_occupied": int(summary_payload.get("occupied_cell_count", 0) or 0),
                    "dilated_iou": float(evaluation_payload.get("dilated_iou", 0.0) or 0.0),
                    "wall_precision": float(evaluation_payload.get("wall_precision", 0.0) or 0.0),
                    "wall_recall": float(evaluation_payload.get("wall_recall", 0.0) or 0.0),
                    "chamfer_distance_m": float(evaluation_payload.get("chamfer_distance_m", 0.0) or 0.0),
                    "closure_gap_m": float(evaluation_payload.get("closure_gap_m", 0.0) or 0.0),
                    "output_dir": str(output_dir),
                },
            )
        else:
            _write_status(
                status_json,
                {
                    "state": "preview_failed",
                    "run_dir": str(run_dir),
                    "scan_count": scan_count,
                    "snapshot_count": snapshot_count,
                    "return_code": int(completed.returncode),
                    "output_dir": str(output_dir),
                },
            )

        if capture_state == "finished" and scan_count <= last_snapshot_scan_count:
            break


if __name__ == "__main__":
    main()
