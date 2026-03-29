#!/usr/bin/env python3
"""Analyze one movement-capture bundle and produce timing and PWM diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_left
from pathlib import Path
from typing import Any


PWM_DUTY_MISMATCH_THRESHOLD_NS = 5_000
PWM_ALIGNMENT_TOLERANCE_S = 0.12


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _int_or_none(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _dt_stats(times_s: list[float]) -> dict[str, float | int | None]:
    if len(times_s) < 2:
        return {
            "sample_count": len(times_s),
            "dt_mean_s": None,
            "dt_min_s": None,
            "dt_max_s": None,
            "gap_count_over_2x_mean": 0,
        }
    dts = [b - a for a, b in zip(times_s[:-1], times_s[1:])]
    mean_dt = sum(dts) / len(dts)
    gap_count = sum(1 for dt in dts if dt > (2.0 * mean_dt))
    return {
        "sample_count": len(times_s),
        "dt_mean_s": mean_dt,
        "dt_min_s": min(dts),
        "dt_max_s": max(dts),
        "gap_count_over_2x_mean": gap_count,
    }


def _summarize_pwm(pwm_path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    rows = _load_csv_rows(pwm_path)
    times_s = [float(row["t_s"]) for row in rows]
    active_rows = [row for row in rows if abs(float(row["motor_speed_pct"])) > 0.5]
    active_times_s = [float(row["t_s"]) for row in active_rows]
    phase_summary: dict[str, dict[str, Any]] = {}

    if rows and "phase" in rows[0]:
        phases = sorted({str(row["phase"]) for row in rows if str(row["phase"]).strip()})
        for phase in phases:
            phase_rows = [row for row in rows if str(row["phase"]) == phase]
            phase_times = [float(row["t_s"]) for row in phase_rows]
            phase_summary[phase] = {
                "rows": len(phase_rows),
                "timing": _dt_stats(phase_times),
                "speed_pct_values": sorted(
                    {round(float(row["motor_speed_pct"]), 6) for row in phase_rows}
                ),
                "motor_pwm_dc_pct_values": sorted(
                    {round(float(row["motor_pwm_dc_pct"]), 6) for row in phase_rows}
                ),
                "motor_enabled_values": sorted({int(row["motor_enabled"]) for row in phase_rows}),
            }
            if phase_times:
                phase_summary[phase]["window_s"] = {
                    "start_s": phase_times[0],
                    "end_s": phase_times[-1],
                    "duration_s": phase_times[-1] - phase_times[0],
                }

    summary: dict[str, Any] = {
        "rows_total": len(rows),
        "timing": _dt_stats(times_s),
        "active_rows": len(active_rows),
        "active_timing": _dt_stats(active_times_s),
        "speed_pct_values": sorted({round(float(row["motor_speed_pct"]), 6) for row in active_rows}),
        "motor_pwm_dc_pct_values": sorted(
            {round(float(row["motor_pwm_dc_pct"]), 6) for row in active_rows}
        ),
        "motor_enabled_values": sorted({int(row["motor_enabled"]) for row in active_rows}),
        "phase_summary": phase_summary,
    }

    if active_times_s:
        summary["active_window_s"] = {
            "start_s": active_times_s[0],
            "end_s": active_times_s[-1],
            "duration_s": active_times_s[-1] - active_times_s[0],
        }
    else:
        summary["active_window_s"] = None

    active_monotonic = [
        _float_or_none(row.get("monotonic_s")) for row in active_rows if _float_or_none(row.get("monotonic_s")) is not None
    ]
    if active_monotonic:
        summary["active_window_monotonic_s"] = {
            "start_s": active_monotonic[0],
            "end_s": active_monotonic[-1],
            "duration_s": active_monotonic[-1] - active_monotonic[0],
        }
    else:
        summary["active_window_monotonic_s"] = None

    return summary, rows


def _summarize_imu(imu_path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    rows = _load_csv_rows(imu_path)
    times_s = [float(row["stamp_sec"]) + (1.0e-9 * float(row["stamp_nanosec"])) for row in rows]
    return {
        "rows_total": len(rows),
        "timing": _dt_stats(times_s),
    }, rows


def _summarize_lidar(lidar_path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    rows = _load_csv_rows(lidar_path)
    scan_times_s: list[float] = []
    point_count_by_scan: dict[int, int] = {}
    seen_scans: set[int] = set()
    for row in rows:
        scan_index = int(row["scan_index"])
        point_count_by_scan[scan_index] = point_count_by_scan.get(scan_index, 0) + 1
        if scan_index not in seen_scans:
            seen_scans.add(scan_index)
            scan_times_s.append(float(row["stamp_sec"]) + (1.0e-9 * float(row["stamp_nanosec"])))

    point_counts = list(point_count_by_scan.values())
    return {
        "rows_total": len(rows),
        "scan_count": len(point_count_by_scan),
        "scan_timing": _dt_stats(scan_times_s),
        "points_per_scan": {
            "min": min(point_counts) if point_counts else None,
            "max": max(point_counts) if point_counts else None,
            "mean": (sum(point_counts) / len(point_counts)) if point_counts else None,
        },
    }, rows


def _summarize_sysfs_monitor(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    rows = _load_csv_rows(path)
    times_s = [float(row["t_s"]) for row in rows]
    return {
        "rows_total": len(rows),
        "timing": _dt_stats(times_s),
    }, rows


def _nearest_trace_row(
    target_monotonic_s: float,
    trace_monotonic_s: list[float],
    trace_rows: list[dict[str, str]],
) -> dict[str, str] | None:
    if not trace_monotonic_s:
        return None
    insert_at = bisect_left(trace_monotonic_s, target_monotonic_s)
    candidate_indexes = []
    if insert_at < len(trace_monotonic_s):
        candidate_indexes.append(insert_at)
    if insert_at > 0:
        candidate_indexes.append(insert_at - 1)
    best_index = None
    best_delta = None
    for candidate_index in candidate_indexes:
        delta_s = abs(trace_monotonic_s[candidate_index] - target_monotonic_s)
        if best_delta is None or delta_s < best_delta:
            best_delta = delta_s
            best_index = candidate_index
    if best_index is None or best_delta is None or best_delta > PWM_ALIGNMENT_TOLERANCE_S:
        return None
    return trace_rows[best_index]


def _build_pwm_timeline(
    trace_rows: list[dict[str, str]],
    sysfs_rows: list[dict[str, str]],
    output_path: Path,
    active_window_monotonic_s: dict[str, float] | None,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mismatched_duty_samples = 0
    mismatched_enable_samples = 0
    active_monitor_rows = 0

    trace_monotonic_s = [
        _float_or_none(row.get("monotonic_s")) for row in trace_rows if _float_or_none(row.get("monotonic_s")) is not None
    ]
    trace_rows_with_monotonic = [
        row for row in trace_rows if _float_or_none(row.get("monotonic_s")) is not None
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "monitor_t_s",
                "monitor_monotonic_s",
                "in_active_window",
                "phase",
                "expected_motor_speed_pct",
                "expected_motor_duty_cycle_ns",
                "actual_motor_duty_cycle_ns",
                "duty_cycle_delta_ns",
                "expected_motor_enabled",
                "actual_motor_enabled",
                "motor_enabled_mismatch",
                "expected_steering_duty_cycle_ns",
                "actual_steering_duty_cycle_ns",
                "expected_steering_enabled",
                "actual_steering_enabled",
            ]
        )

        if sysfs_rows and trace_monotonic_s:
            active_start = active_window_monotonic_s["start_s"] if active_window_monotonic_s else None
            active_end = active_window_monotonic_s["end_s"] if active_window_monotonic_s else None
            for sysfs_row in sysfs_rows:
                monitor_monotonic_s = _float_or_none(sysfs_row.get("monotonic_s"))
                if monitor_monotonic_s is None:
                    continue
                trace_row = _nearest_trace_row(
                    target_monotonic_s=monitor_monotonic_s,
                    trace_monotonic_s=trace_monotonic_s,
                    trace_rows=trace_rows_with_monotonic,
                )
                in_active_window = (
                    active_start is not None
                    and active_end is not None
                    and active_start <= monitor_monotonic_s <= active_end
                )
                if in_active_window:
                    active_monitor_rows += 1

                expected_motor_duty_ns = _int_or_none(trace_row.get("motor_duty_cycle_ns")) if trace_row else None
                actual_motor_duty_ns = _int_or_none(sysfs_row.get("motor_duty_cycle_ns"))
                duty_delta_ns = None
                if expected_motor_duty_ns is not None and actual_motor_duty_ns is not None:
                    duty_delta_ns = actual_motor_duty_ns - expected_motor_duty_ns
                expected_motor_enabled = _int_or_none(trace_row.get("motor_enabled")) if trace_row else None
                actual_motor_enabled = _int_or_none(sysfs_row.get("motor_enabled"))
                motor_enabled_mismatch = (
                    expected_motor_enabled is not None
                    and actual_motor_enabled is not None
                    and expected_motor_enabled != actual_motor_enabled
                )

                if in_active_window and duty_delta_ns is not None and abs(duty_delta_ns) > PWM_DUTY_MISMATCH_THRESHOLD_NS:
                    mismatched_duty_samples += 1
                if in_active_window and motor_enabled_mismatch:
                    mismatched_enable_samples += 1

                writer.writerow(
                    [
                        sysfs_row.get("t_s", ""),
                        monitor_monotonic_s,
                        int(bool(in_active_window)),
                        trace_row.get("phase", "") if trace_row else "",
                        trace_row.get("motor_speed_pct", "") if trace_row else "",
                        expected_motor_duty_ns if expected_motor_duty_ns is not None else "",
                        actual_motor_duty_ns if actual_motor_duty_ns is not None else "",
                        duty_delta_ns if duty_delta_ns is not None else "",
                        expected_motor_enabled if expected_motor_enabled is not None else "",
                        actual_motor_enabled if actual_motor_enabled is not None else "",
                        int(bool(motor_enabled_mismatch)),
                        _int_or_none(trace_row.get("steering_duty_cycle_ns")) if trace_row else "",
                        _int_or_none(sysfs_row.get("steering_duty_cycle_ns")),
                        _int_or_none(trace_row.get("steering_enabled")) if trace_row else "",
                        _int_or_none(sysfs_row.get("steering_enabled")),
                    ]
                )
        else:
            for trace_row in trace_rows:
                writer.writerow(
                    [
                        "",
                        trace_row.get("monotonic_s", ""),
                        "",
                        trace_row.get("phase", ""),
                        trace_row.get("motor_speed_pct", ""),
                        trace_row.get("motor_duty_cycle_ns", ""),
                        "",
                        "",
                        trace_row.get("motor_enabled", ""),
                        "",
                        "",
                        trace_row.get("steering_duty_cycle_ns", ""),
                        "",
                        trace_row.get("steering_enabled", ""),
                        "",
                    ]
                )

    return {
        "active_monitor_rows": active_monitor_rows,
        "mismatched_duty_samples": mismatched_duty_samples,
        "mismatched_enable_samples": mismatched_enable_samples,
    }


def _write_summary_markdown(
    output_path: Path,
    *,
    run_dir: Path,
    actuation_mode: str,
    readiness: dict[str, Any] | None,
    pwm_summary: dict[str, Any],
    imu_summary: dict[str, Any],
    lidar_summary: dict[str, Any],
    sysfs_summary: dict[str, Any] | None,
    flags: dict[str, Any],
) -> None:
    lines = [
        f"# Movement Bundle Summary",
        "",
        f"- Run: `{run_dir}`",
        f"- Actuation mode: `{actuation_mode}`",
        f"- Readiness passed: `{flags['readiness_passed']}`",
        f"- External PWM overwrite detected: `{flags['external_pwm_overwrite_detected']}`",
        f"- Motor enable dropped during active window: `{flags['motor_enable_dropped_during_active_window']}`",
        f"- Actual duty mismatch during active window: `{flags['actual_duty_mismatch_during_active_window']}`",
        f"- PWM active gap detected: `{flags['pwm_active_gap_detected']}`",
        "",
    ]

    active_window = pwm_summary.get("active_window_s")
    if active_window:
        lines.extend(
            [
                "## Active Window",
                "",
                f"- Start: `{active_window['start_s']:.3f}s`",
                f"- End: `{active_window['end_s']:.3f}s`",
                f"- Duration: `{active_window['duration_s']:.3f}s`",
                f"- Expected motor duty values: `{pwm_summary['motor_pwm_dc_pct_values']}` %",
                "",
            ]
        )

    lines.extend(
        [
            "## Timing",
            "",
            f"- PWM dt mean/max: `{pwm_summary['active_timing']['dt_mean_s']}` / `{pwm_summary['active_timing']['dt_max_s']}`",
            f"- IMU dt mean/max: `{imu_summary['timing']['dt_mean_s']}` / `{imu_summary['timing']['dt_max_s']}`",
            f"- LiDAR scan dt mean/max: `{lidar_summary['scan_timing']['dt_mean_s']}` / `{lidar_summary['scan_timing']['dt_max_s']}`",
        ]
    )
    if sysfs_summary is not None:
        lines.append(
            f"- Sysfs monitor dt mean/max: `{sysfs_summary['timing']['dt_mean_s']}` / `{sysfs_summary['timing']['dt_max_s']}`"
        )
    if readiness is not None:
        lines.extend(
            [
                "",
                "## Readiness",
                "",
                f"- IMU received: `{readiness.get('imu', {}).get('count')}`",
                f"- LiDAR scans received: `{readiness.get('lidar', {}).get('count')}`",
            ]
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing imu_raw.csv, lidar_points.csv, and pwm_trace.csv",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path. Defaults to <run-dir>/timing_analysis.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    analysis_dir = run_dir / "analysis"
    pwm_path = run_dir / "pwm_trace.csv"
    imu_path = run_dir / "imu_raw.csv"
    lidar_path = run_dir / "lidar_points.csv"
    sysfs_path = run_dir / "sysfs_pwm_monitor.csv"
    readiness_path = run_dir / "readiness.json"
    capture_meta_path = run_dir / "capture_meta.json"

    missing = [str(path) for path in (pwm_path, imu_path, lidar_path) if not path.exists()]
    if missing:
        raise SystemExit("Missing required files: " + ", ".join(missing))

    pwm_summary, pwm_rows = _summarize_pwm(pwm_path)
    imu_summary, _imu_rows = _summarize_imu(imu_path)
    lidar_summary, _lidar_rows = _summarize_lidar(lidar_path)
    sysfs_summary = None
    sysfs_rows: list[dict[str, str]] = []
    if sysfs_path.exists():
        sysfs_summary, sysfs_rows = _summarize_sysfs_monitor(sysfs_path)

    timeline_metrics = _build_pwm_timeline(
        trace_rows=pwm_rows,
        sysfs_rows=sysfs_rows,
        output_path=analysis_dir / "pwm_timeline.csv",
        active_window_monotonic_s=pwm_summary.get("active_window_monotonic_s"),
    )

    readiness = _load_json(readiness_path)
    capture_meta = _load_json(capture_meta_path) or {}

    flags = {
        "actuation_mode": capture_meta.get("actuation_mode", "unknown"),
        "readiness_passed": bool(readiness and readiness.get("ready") is True),
        "sysfs_monitor_present": bool(sysfs_summary is not None),
        "external_pwm_overwrite_detected": (
            timeline_metrics["mismatched_duty_samples"] > 0
            or timeline_metrics["mismatched_enable_samples"] > 0
        ),
        "motor_enable_dropped_during_active_window": timeline_metrics["mismatched_enable_samples"] > 0,
        "actual_duty_mismatch_during_active_window": timeline_metrics["mismatched_duty_samples"] > 0,
        "pwm_active_gap_detected": pwm_summary["active_timing"]["gap_count_over_2x_mean"] > 0,
        "imu_gap_detected": imu_summary["timing"]["gap_count_over_2x_mean"] > 0,
        "lidar_gap_detected": lidar_summary["scan_timing"]["gap_count_over_2x_mean"] > 0,
        "active_monitor_rows": timeline_metrics["active_monitor_rows"],
        "mismatched_duty_samples": timeline_metrics["mismatched_duty_samples"],
        "mismatched_enable_samples": timeline_metrics["mismatched_enable_samples"],
        "pwm_duty_mismatch_threshold_ns": PWM_DUTY_MISMATCH_THRESHOLD_NS,
        "pwm_alignment_tolerance_s": PWM_ALIGNMENT_TOLERANCE_S,
    }

    summary = {
        "run_dir": str(run_dir),
        "pwm": pwm_summary,
        "imu": imu_summary,
        "lidar": lidar_summary,
        "sysfs_monitor": sysfs_summary,
        "readiness": readiness,
        "capture_meta": capture_meta,
        "analysis_flags": flags,
    }

    output_path = Path(args.output).expanduser().resolve() if args.output else (run_dir / "timing_analysis.json")
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    _write_summary_markdown(
        analysis_dir / "summary.md",
        run_dir=run_dir,
        actuation_mode=str(capture_meta.get("actuation_mode", "unknown")),
        readiness=readiness,
        pwm_summary=pwm_summary,
        imu_summary=imu_summary,
        lidar_summary=lidar_summary,
        sysfs_summary=sysfs_summary,
        flags=flags,
    )

    print(f"[APEX] Timing JSON: {output_path}")
    print(f"[APEX] PWM active rows: {pwm_summary['active_rows']}")
    active_window = pwm_summary["active_window_s"]
    if active_window:
        print(
            "[APEX] PWM active window: "
            f"{active_window['start_s']:.3f}s .. {active_window['end_s']:.3f}s "
            f"(duration={active_window['duration_s']:.3f}s)"
        )
    print(
        "[APEX] PWM dt mean/max: "
        f"{pwm_summary['active_timing']['dt_mean_s']:.6f}s / {pwm_summary['active_timing']['dt_max_s']:.6f}s"
        if pwm_summary["active_timing"]["dt_mean_s"] is not None
        else "[APEX] PWM dt mean/max: N/A"
    )
    print(
        "[APEX] IMU dt mean/max: "
        f"{imu_summary['timing']['dt_mean_s']:.6f}s / {imu_summary['timing']['dt_max_s']:.6f}s"
        if imu_summary["timing"]["dt_mean_s"] is not None
        else "[APEX] IMU dt mean/max: N/A"
    )
    print(
        "[APEX] LiDAR scan dt mean/max: "
        f"{lidar_summary['scan_timing']['dt_mean_s']:.6f}s / {lidar_summary['scan_timing']['dt_max_s']:.6f}s"
        if lidar_summary["scan_timing"]["dt_mean_s"] is not None
        else "[APEX] LiDAR scan dt mean/max: N/A"
    )
    if sysfs_summary is not None:
        print(
            "[APEX] Sysfs dt mean/max: "
            f"{sysfs_summary['timing']['dt_mean_s']:.6f}s / {sysfs_summary['timing']['dt_max_s']:.6f}s"
            if sysfs_summary["timing"]["dt_mean_s"] is not None
            else "[APEX] Sysfs dt mean/max: N/A"
        )
    print(f"[APEX] Analysis summary: {analysis_dir / 'summary.md'}")
    print(f"[APEX] Analysis flags: {analysis_dir / 'flags.json'}")
    print(f"[APEX] PWM timeline: {analysis_dir / 'pwm_timeline.csv'}")


if __name__ == "__main__":
    main()
