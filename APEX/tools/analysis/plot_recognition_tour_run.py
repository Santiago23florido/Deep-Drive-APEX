#!/usr/bin/env python3
"""Render one recognition_tour run with explicit timing/controller/planner diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


@dataclass
class LocalPathItem:
    t_monotonic_s: float
    stamp_s: float
    path_xy: np.ndarray
    planner_state: str
    local_path_source: str
    path_forward_span_m: float
    path_length_m: float
    path_max_curvature_m_inv: float


@dataclass
class TrajectorySeries:
    t_monotonic_s: np.ndarray
    stamp_s: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    yaw_rad: np.ndarray
    vx_mps: np.ndarray
    vy_mps: np.ndarray
    yaw_rate_rps: np.ndarray
    odom_source: list[str]
    odom_prediction_age_s: np.ndarray
    planner_state: list[str]
    planner_path_forward_span_m: np.ndarray
    planner_path_length_m: np.ndarray
    planner_path_max_curvature_m_inv: np.ndarray
    local_path_source: list[str]
    continuation_source: list[str]
    path_terminal_heading_deg: np.ndarray
    route_suffix_heading_deg: np.ndarray
    path_heading_alignment_deg: np.ndarray
    planner_forward_projection_valid: np.ndarray
    tracker_state: list[str]
    path_age_s: np.ndarray
    odom_age_s: np.ndarray
    path_deviation_m: np.ndarray
    path_projection_valid: np.ndarray
    path_projection_s_m: np.ndarray
    distance_to_path_end_m: np.ndarray
    path_loss_reason: list[str]
    waiting_path_refresh_active: np.ndarray
    cmd_linear_x_mps: np.ndarray
    cmd_angular_z_rps: np.ndarray
    tracker_desired_steering_deg: np.ndarray
    raw_curvature_m_inv: np.ndarray
    commanded_curvature_m_inv: np.ndarray
    filtered_curvature_m_inv: np.ndarray
    tracker_steering_saturated: np.ndarray
    tracker_steering_saturation_ratio: np.ndarray
    fusion_state: list[str]
    fusion_confidence: np.ndarray
    desired_speed_pct: np.ndarray
    applied_speed_pct: np.ndarray
    desired_steering_deg: np.ndarray
    requested_steering_deg: np.ndarray
    applied_steering_deg: np.ndarray
    bridge_steering_saturated: np.ndarray
    bridge_state: list[str]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return float("nan")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def _json_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def _percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, q))


def _mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _max_abs(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(np.abs(finite)))


def _fraction(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.mean(mask.astype(np.float64)))


def _count_segments(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    mask = mask.astype(bool)
    padded = np.concatenate([[False], mask, [False]])
    rising = np.flatnonzero((~padded[:-1]) & padded[1:])
    return int(rising.size)


def _load_route_json(path: Path) -> np.ndarray:
    payload = _load_json(path)
    points = np.asarray(payload.get("path_xy_yaw", []), dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float64)
    return points[:, :3]


def _load_lidar_world_points(path: Path, *, max_points: int = 30000) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 2), dtype=np.float64)
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x_m = _csv_float(row, "x_world_m")
            y_m = _csv_float(row, "y_world_m")
            if not (math.isfinite(x_m) and math.isfinite(y_m)):
                continue
            rows.append([x_m, y_m])
    if not rows:
        return np.empty((0, 2), dtype=np.float64)
    points = np.asarray(rows, dtype=np.float64)
    if points.shape[0] <= max_points:
        return points
    stride = max(1, int(math.ceil(points.shape[0] / max_points)))
    return points[::stride]


def _load_local_path_history(path: Path, *, max_paths: int = 220) -> list[LocalPathItem]:
    if not path.exists():
        return []
    items: list[LocalPathItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        path_xy_yaw = np.asarray(payload.get("path_xy_yaw", []), dtype=np.float64)
        if path_xy_yaw.ndim != 2 or path_xy_yaw.shape[0] < 2 or path_xy_yaw.shape[1] < 2:
            continue
        stamp_s = _json_float(payload.get("stamp_sec", 0.0)) + (
            1.0e-9 * _json_float(payload.get("stamp_nanosec", 0.0))
        )
        items.append(
            LocalPathItem(
                t_monotonic_s=_json_float(payload.get("t_monotonic_s", float("nan"))),
                stamp_s=stamp_s,
                path_xy=path_xy_yaw[:, :2],
                planner_state=str(payload.get("planner_state", "")),
                local_path_source=str(payload.get("local_path_source", "")),
                path_forward_span_m=_json_float(payload.get("path_forward_span_m", float("nan"))),
                path_length_m=_json_float(payload.get("path_length_m", float("nan"))),
                path_max_curvature_m_inv=_json_float(
                    payload.get("path_max_curvature_m_inv", float("nan"))
                ),
            )
        )
    if len(items) <= max_paths:
        return items
    stride = max(1, int(math.ceil(len(items) / max_paths)))
    return items[::stride]


def _load_tracking_series(path: Path) -> TrajectorySeries:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    if not rows:
        empty = np.empty((0,), dtype=np.float64)
        return TrajectorySeries(
            t_monotonic_s=empty,
            stamp_s=empty,
            x_m=empty,
            y_m=empty,
            yaw_rad=empty,
            vx_mps=empty,
            vy_mps=empty,
            yaw_rate_rps=empty,
            odom_source=[],
            odom_prediction_age_s=empty,
            planner_state=[],
            planner_path_forward_span_m=empty,
            planner_path_length_m=empty,
            planner_path_max_curvature_m_inv=empty,
            local_path_source=[],
            continuation_source=[],
            path_terminal_heading_deg=empty,
            route_suffix_heading_deg=empty,
            path_heading_alignment_deg=empty,
            planner_forward_projection_valid=empty,
            tracker_state=[],
            path_age_s=empty,
            odom_age_s=empty,
            path_deviation_m=empty,
            path_projection_valid=empty,
            path_projection_s_m=empty,
            distance_to_path_end_m=empty,
            path_loss_reason=[],
            waiting_path_refresh_active=empty,
            cmd_linear_x_mps=empty,
            cmd_angular_z_rps=empty,
            tracker_desired_steering_deg=empty,
            raw_curvature_m_inv=empty,
            commanded_curvature_m_inv=empty,
            filtered_curvature_m_inv=empty,
            tracker_steering_saturated=empty,
            tracker_steering_saturation_ratio=empty,
            fusion_state=[],
            fusion_confidence=empty,
            desired_speed_pct=empty,
            applied_speed_pct=empty,
            desired_steering_deg=empty,
            requested_steering_deg=empty,
            applied_steering_deg=empty,
            bridge_steering_saturated=empty,
            bridge_state=[],
        )

    stamp_s = np.asarray(
        [
            _csv_float(row, "stamp_sec") + (1.0e-9 * _csv_float(row, "stamp_nanosec"))
            for row in rows
        ],
        dtype=np.float64,
    )
    if "t_monotonic_s" in rows[0]:
        t_monotonic_s = np.asarray([_csv_float(row, "t_monotonic_s") for row in rows], dtype=np.float64)
    else:
        t0 = float(stamp_s[0]) if stamp_s.size else 0.0
        t_monotonic_s = stamp_s - t0

    return TrajectorySeries(
        t_monotonic_s=t_monotonic_s,
        stamp_s=stamp_s,
        x_m=np.asarray([_csv_float(row, "x_m") for row in rows], dtype=np.float64),
        y_m=np.asarray([_csv_float(row, "y_m") for row in rows], dtype=np.float64),
        yaw_rad=np.asarray([_csv_float(row, "yaw_rad") for row in rows], dtype=np.float64),
        vx_mps=np.asarray([_csv_float(row, "vx_mps") for row in rows], dtype=np.float64),
        vy_mps=np.asarray([_csv_float(row, "vy_mps") for row in rows], dtype=np.float64),
        yaw_rate_rps=np.asarray([_csv_float(row, "yaw_rate_rps") for row in rows], dtype=np.float64),
        odom_source=[str(row.get("odom_source", "")) for row in rows],
        odom_prediction_age_s=np.asarray(
            [_csv_float(row, "odom_prediction_age_s") for row in rows], dtype=np.float64
        ),
        planner_state=[str(row.get("planner_state", "")) for row in rows],
        planner_path_forward_span_m=np.asarray(
            [_csv_float(row, "planner_path_forward_span_m") for row in rows],
            dtype=np.float64,
        ),
        planner_path_length_m=np.asarray(
            [_csv_float(row, "planner_path_length_m") for row in rows],
            dtype=np.float64,
        ),
        planner_path_max_curvature_m_inv=np.asarray(
            [_csv_float(row, "planner_path_max_curvature_m_inv") for row in rows],
            dtype=np.float64,
        ),
        local_path_source=[str(row.get("local_path_source", "")) for row in rows],
        continuation_source=[str(row.get("continuation_source", "")) for row in rows],
        path_terminal_heading_deg=np.asarray(
            [_csv_float(row, "path_terminal_heading_deg") for row in rows],
            dtype=np.float64,
        ),
        route_suffix_heading_deg=np.asarray(
            [_csv_float(row, "route_suffix_heading_deg") for row in rows],
            dtype=np.float64,
        ),
        path_heading_alignment_deg=np.asarray(
            [_csv_float(row, "path_heading_alignment_deg") for row in rows],
            dtype=np.float64,
        ),
        planner_forward_projection_valid=np.asarray(
            [_csv_float(row, "planner_forward_projection_valid") for row in rows],
            dtype=np.float64,
        ),
        tracker_state=[str(row.get("tracker_state", "")) for row in rows],
        path_age_s=np.asarray([_csv_float(row, "path_age_s") for row in rows], dtype=np.float64),
        odom_age_s=np.asarray([_csv_float(row, "odom_age_s") for row in rows], dtype=np.float64),
        path_deviation_m=np.asarray([_csv_float(row, "path_deviation_m") for row in rows], dtype=np.float64),
        path_projection_valid=np.asarray(
            [_csv_float(row, "path_projection_valid") for row in rows], dtype=np.float64
        ),
        path_projection_s_m=np.asarray(
            [_csv_float(row, "path_projection_s_m") for row in rows], dtype=np.float64
        ),
        distance_to_path_end_m=np.asarray(
            [_csv_float(row, "distance_to_path_end_m") for row in rows], dtype=np.float64
        ),
        path_loss_reason=[str(row.get("path_loss_reason", "")) for row in rows],
        waiting_path_refresh_active=np.asarray(
            [_csv_float(row, "waiting_path_refresh_active") for row in rows], dtype=np.float64
        ),
        cmd_linear_x_mps=np.asarray([_csv_float(row, "cmd_linear_x_mps") for row in rows], dtype=np.float64),
        cmd_angular_z_rps=np.asarray([_csv_float(row, "cmd_angular_z_rps") for row in rows], dtype=np.float64),
        tracker_desired_steering_deg=np.asarray(
            [_csv_float(row, "tracker_desired_steering_deg") for row in rows],
            dtype=np.float64,
        ),
        raw_curvature_m_inv=np.asarray(
            [_csv_float(row, "raw_curvature_m_inv") for row in rows],
            dtype=np.float64,
        ),
        commanded_curvature_m_inv=np.asarray(
            [_csv_float(row, "commanded_curvature_m_inv") for row in rows],
            dtype=np.float64,
        ),
        filtered_curvature_m_inv=np.asarray(
            [_csv_float(row, "filtered_curvature_m_inv") for row in rows],
            dtype=np.float64,
        ),
        tracker_steering_saturated=np.asarray(
            [_csv_float(row, "tracker_steering_saturated") for row in rows],
            dtype=np.float64,
        ),
        tracker_steering_saturation_ratio=np.asarray(
            [_csv_float(row, "tracker_steering_saturation_ratio") for row in rows],
            dtype=np.float64,
        ),
        fusion_state=[str(row.get("fusion_state", "")) for row in rows],
        fusion_confidence=np.asarray([_csv_float(row, "fusion_confidence") for row in rows], dtype=np.float64),
        desired_speed_pct=np.asarray([_csv_float(row, "desired_speed_pct") for row in rows], dtype=np.float64),
        applied_speed_pct=np.asarray([_csv_float(row, "applied_speed_pct") for row in rows], dtype=np.float64),
        desired_steering_deg=np.asarray(
            [_csv_float(row, "desired_steering_deg") for row in rows],
            dtype=np.float64,
        ),
        requested_steering_deg=np.asarray(
            [_csv_float(row, "requested_steering_deg") for row in rows],
            dtype=np.float64,
        ),
        applied_steering_deg=np.asarray(
            [_csv_float(row, "applied_steering_deg") for row in rows],
            dtype=np.float64,
        ),
        bridge_steering_saturated=np.asarray(
            [_csv_float(row, "bridge_steering_saturated") for row in rows],
            dtype=np.float64,
        ),
        bridge_state=[str(row.get("bridge_state", "")) for row in rows],
    )


def _load_status_log(path: Path) -> dict[str, list[dict[str, Any]]]:
    series: dict[str, list[dict[str, Any]]] = {"planner": [], "tracker": [], "fusion": []}
    if not path.exists():
        return series
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        source = str(entry.get("source", ""))
        payload = entry.get("payload")
        if source not in series or not isinstance(payload, dict):
            continue
        series[source].append(
            {
                "t_monotonic_s": _json_float(entry.get("t_monotonic_s", float("nan"))),
                "payload": payload,
            }
        )
    return series


def _load_bridge_log(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not path.exists():
        return entries
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = entry.get("payload")
        if not isinstance(payload, dict):
            continue
        entries.append(
            {
                "t_monotonic_s": _json_float(entry.get("t_monotonic_s", float("nan"))),
                "payload": payload,
            }
        )
    return entries


def _extract_status_numeric(entries: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    t_values: list[float] = []
    y_values: list[float] = []
    for entry in entries:
        payload = entry.get("payload") or {}
        value = payload.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        t_monotonic_s = float(entry.get("t_monotonic_s", float("nan")))
        if not math.isfinite(t_monotonic_s):
            continue
        t_values.append(t_monotonic_s)
        y_values.append(float(value))
    return np.asarray(t_values, dtype=np.float64), np.asarray(y_values, dtype=np.float64)


def _extract_status_strings(entries: list[dict[str, Any]], key: str) -> tuple[np.ndarray, list[str]]:
    t_values: list[float] = []
    y_values: list[str] = []
    for entry in entries:
        payload = entry.get("payload") or {}
        value = payload.get(key)
        if value is None:
            continue
        t_monotonic_s = float(entry.get("t_monotonic_s", float("nan")))
        if not math.isfinite(t_monotonic_s):
            continue
        t_values.append(t_monotonic_s)
        y_values.append(str(value))
    return np.asarray(t_values, dtype=np.float64), y_values


def _format_float(value: float, digits: int = 3) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def _format_pct(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{100.0 * value:.1f}%"


def _compute_root_cause_diagnostics(
    *,
    summary: dict,
    trajectory: TrajectorySeries,
    status_series: dict[str, list[dict[str, Any]]],
    bridge_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    planner_entries = status_series.get("planner", [])
    tracker_entries = status_series.get("tracker", [])
    fusion_entries = status_series.get("fusion", [])

    _, planner_odom_age = _extract_status_numeric(planner_entries, "odom_age_s")
    _, planner_path_span = _extract_status_numeric(planner_entries, "path_forward_span_m")
    _, planner_path_length = _extract_status_numeric(planner_entries, "path_length_m")
    _, planner_path_curvature = _extract_status_numeric(planner_entries, "path_max_curvature_m_inv")
    _, tracker_odom_age = _extract_status_numeric(tracker_entries, "odom_age_s")
    _, tracker_path_age = _extract_status_numeric(tracker_entries, "path_age_s")
    _, tracker_path_deviation = _extract_status_numeric(tracker_entries, "path_deviation_m")
    _, tracker_cmd_speed = _extract_status_numeric(tracker_entries, "cmd_linear_x_mps")
    _, tracker_desired_steering = _extract_status_numeric(tracker_entries, "desired_steering_deg")
    _, fusion_prediction_age = _extract_status_numeric(fusion_entries, "odom_prediction_age_s")
    _, local_path_source_values = _extract_status_strings(planner_entries, "local_path_source")
    _, continuation_source_values = _extract_status_strings(planner_entries, "continuation_source")
    _, path_loss_reason_values = _extract_status_strings(tracker_entries, "path_loss_reason")

    tracker_states = trajectory.tracker_state if trajectory.tracker_state else []
    holding_mask = np.asarray(
        [state in {"holding_last_path", "waiting_forward_path", "waiting_path_refresh"} for state in tracker_states],
        dtype=bool,
    )
    aborted_mask = np.asarray(
        [state in {"aborted_path_loss", "aborted_odom_timeout"} for state in tracker_states],
        dtype=bool,
    )
    stale_odom_mask = np.isfinite(trajectory.odom_age_s) & (trajectory.odom_age_s > 0.08)
    stale_path_mask = np.isfinite(trajectory.path_age_s) & (trajectory.path_age_s > 0.25)
    short_span_mask = np.isfinite(trajectory.planner_path_forward_span_m) & (
        trajectory.planner_path_forward_span_m < 0.95
    )
    sharp_turn_mask = np.isfinite(trajectory.tracker_desired_steering_deg) & (
        np.abs(trajectory.tracker_desired_steering_deg) >= 20.0
    )
    low_speed_sharp_turn_mask = sharp_turn_mask & np.isfinite(trajectory.cmd_linear_x_mps) & (
        trajectory.cmd_linear_x_mps < 0.10
    )
    waiting_refresh_fraction = _fraction(
        np.isfinite(trajectory.waiting_path_refresh_active)
        & (trajectory.waiting_path_refresh_active > 0.5)
    )
    steering_saturation_fraction = _fraction(
        np.isfinite(trajectory.bridge_steering_saturated)
        & (trajectory.bridge_steering_saturated > 0.5)
    )

    steering_error_deg = np.abs(trajectory.desired_steering_deg - trajectory.applied_steering_deg)
    speed_error_pct = np.abs(trajectory.desired_speed_pct - trajectory.applied_speed_pct)

    path_span_p10 = _percentile(trajectory.planner_path_forward_span_m, 10.0)
    path_curvature_p95 = _percentile(trajectory.planner_path_max_curvature_m_inv, 95.0)
    path_deviation_p95 = _percentile(trajectory.path_deviation_m, 95.0)
    tracker_odom_age_p95 = _percentile(trajectory.odom_age_s, 95.0)
    path_age_p95 = _percentile(trajectory.path_age_s, 95.0)
    fusion_prediction_age_p95 = _percentile(trajectory.odom_prediction_age_s, 95.0)
    planner_odom_age_p95 = _percentile(planner_odom_age, 95.0)
    steering_error_p95 = _percentile(steering_error_deg, 95.0)
    speed_error_p95 = _percentile(speed_error_pct, 95.0)
    fallback_fraction = 0.0
    if local_path_source_values:
        fallback_fraction = float(
            np.mean(
                np.asarray(
                    [
                        ("fallback" in value) or ("curve_window" in value)
                        for value in local_path_source_values
                    ],
                    dtype=np.float64,
                )
            )
        )

    timing_score = 0.0
    planner_score = 0.0
    controller_score = 0.0
    timing_evidence: list[str] = []
    planner_evidence: list[str] = []
    controller_evidence: list[str] = []

    if math.isfinite(tracker_odom_age_p95) and tracker_odom_age_p95 > 0.12:
        timing_score += 2.0
        timing_evidence.append(f"tracker odom age p95={tracker_odom_age_p95:.3f}s")
    elif math.isfinite(tracker_odom_age_p95) and tracker_odom_age_p95 > 0.08:
        timing_score += 1.1
        timing_evidence.append(f"tracker odom age p95={tracker_odom_age_p95:.3f}s")
    if math.isfinite(planner_odom_age_p95) and planner_odom_age_p95 > 0.10:
        timing_score += 1.0
        timing_evidence.append(f"planner odom age p95={planner_odom_age_p95:.3f}s")
    if math.isfinite(path_age_p95) and path_age_p95 > 0.25:
        timing_score += 0.9
        timing_evidence.append(f"path age p95={path_age_p95:.3f}s")
    if math.isfinite(fusion_prediction_age_p95) and fusion_prediction_age_p95 > 0.10:
        timing_score += 0.9
        timing_evidence.append(f"fusion prediction age p95={fusion_prediction_age_p95:.3f}s")
    if _fraction(stale_odom_mask) > 0.12:
        timing_score += 0.9
        timing_evidence.append(f"stale odom fraction={_format_pct(_fraction(stale_odom_mask))}")
    if _count_segments(holding_mask | aborted_mask) > 0 and timing_score >= 1.0:
        timing_score += 0.8
        timing_evidence.append(
            f"timing-related tracker interruptions={_count_segments(holding_mask | aborted_mask)}"
        )

    if math.isfinite(path_span_p10) and path_span_p10 < 0.85:
        planner_score += 2.0
        planner_evidence.append(f"path forward span p10={path_span_p10:.3f}m")
    elif math.isfinite(path_span_p10) and path_span_p10 < 1.05:
        planner_score += 1.0
        planner_evidence.append(f"path forward span p10={path_span_p10:.3f}m")
    if _fraction(short_span_mask) > 0.15:
        planner_score += 1.5
        planner_evidence.append(f"short-path fraction={_format_pct(_fraction(short_span_mask))}")
    elif _fraction(short_span_mask) > 0.05:
        planner_score += 0.8
        planner_evidence.append(f"short-path fraction={_format_pct(_fraction(short_span_mask))}")
    if math.isfinite(path_curvature_p95) and path_curvature_p95 > 2.5:
        planner_score += 1.4
        planner_evidence.append(f"planner curvature p95={path_curvature_p95:.2f} 1/m")
    elif math.isfinite(path_curvature_p95) and path_curvature_p95 > 1.8:
        planner_score += 0.8
        planner_evidence.append(f"planner curvature p95={path_curvature_p95:.2f} 1/m")
    if fallback_fraction > 0.20:
        planner_score += 0.9
        planner_evidence.append(f"fallback path fraction={_format_pct(fallback_fraction)}")
    if continuation_source_values:
        route_continuation_fraction = float(
            np.mean(
                np.asarray(
                    ["route_suffix_continuation" in value for value in continuation_source_values],
                    dtype=np.float64,
                )
            )
        )
        if route_continuation_fraction > 0.05:
            planner_score -= 0.3
            planner_evidence.append(f"route-suffix continuation fraction={_format_pct(route_continuation_fraction)}")
    if str(summary.get("end_cause")) == "aborted_path_loss" and planner_score >= 1.0:
        planner_score += 0.7
        planner_evidence.append("path-loss happened with weak forward path continuity")

    if math.isfinite(steering_error_p95) and steering_error_p95 > 8.0:
        controller_score += 1.5
        controller_evidence.append(f"steering mismatch p95={steering_error_p95:.1f}deg")
    elif math.isfinite(steering_error_p95) and steering_error_p95 > 5.0:
        controller_score += 0.8
        controller_evidence.append(f"steering mismatch p95={steering_error_p95:.1f}deg")
    if math.isfinite(speed_error_p95) and speed_error_p95 > 4.0:
        controller_score += 1.0
        controller_evidence.append(f"speed mismatch p95={speed_error_p95:.1f}%")
    elif math.isfinite(speed_error_p95) and speed_error_p95 > 2.0:
        controller_score += 0.5
        controller_evidence.append(f"speed mismatch p95={speed_error_p95:.1f}%")
    if math.isfinite(path_deviation_p95) and path_deviation_p95 > 0.18:
        controller_score += 1.2
        controller_evidence.append(f"path deviation p95={path_deviation_p95:.3f}m")
    elif math.isfinite(path_deviation_p95) and path_deviation_p95 > 0.10:
        controller_score += 0.6
        controller_evidence.append(f"path deviation p95={path_deviation_p95:.3f}m")
    if _fraction(low_speed_sharp_turn_mask) > 0.20:
        controller_score += 1.0
        controller_evidence.append(
            f"sharp-turn low-speed fraction={_format_pct(_fraction(low_speed_sharp_turn_mask))}"
        )
    if steering_saturation_fraction > 0.10:
        controller_score += 1.2
        controller_evidence.append(
            f"steering saturation fraction={_format_pct(steering_saturation_fraction)}"
        )
    if waiting_refresh_fraction > 0.05:
        controller_score += 0.4
        controller_evidence.append(
            f"waiting-path-refresh fraction={_format_pct(waiting_refresh_fraction)}"
        )
    if path_loss_reason_values:
        last_reason = path_loss_reason_values[-1]
        if last_reason:
            controller_evidence.append(f"path loss reason={last_reason}")
    if (
        str(summary.get("end_cause")) == "aborted_path_loss"
        and timing_score < 1.4
        and planner_score < 1.4
        and math.isfinite(path_deviation_p95)
        and path_deviation_p95 > 0.12
    ):
        controller_score += 0.8
        controller_evidence.append("path-loss not explained by stale odom or short path alone")

    scores = {
        "timing": round(timing_score, 3),
        "planner": round(planner_score, 3),
        "controller": round(controller_score, 3),
    }
    ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if ranking[0][1] < 1.0:
        primary_cause = "undetermined"
    elif (ranking[0][1] - ranking[1][1]) < 0.75:
        primary_cause = "mixed"
    else:
        primary_cause = ranking[0][0]

    return {
        "primary_cause": primary_cause,
        "scores": scores,
        "timing": {
            "tracker_odom_age_p95_s": None if not math.isfinite(tracker_odom_age_p95) else tracker_odom_age_p95,
            "planner_odom_age_p95_s": None if not math.isfinite(planner_odom_age_p95) else planner_odom_age_p95,
            "path_age_p95_s": None if not math.isfinite(path_age_p95) else path_age_p95,
            "fusion_prediction_age_p95_s": (
                None if not math.isfinite(fusion_prediction_age_p95) else fusion_prediction_age_p95
            ),
            "stale_odom_fraction": _fraction(stale_odom_mask),
            "stale_path_fraction": _fraction(stale_path_mask),
            "tracker_problem_segments": _count_segments(holding_mask | aborted_mask),
        },
        "planner": {
            "path_forward_span_p10_m": None if not math.isfinite(path_span_p10) else path_span_p10,
            "path_length_mean_m": None if not math.isfinite(_mean(trajectory.planner_path_length_m)) else _mean(trajectory.planner_path_length_m),
            "path_curvature_p95_1pm": None if not math.isfinite(path_curvature_p95) else path_curvature_p95,
            "short_path_fraction": _fraction(short_span_mask),
            "fallback_fraction": fallback_fraction,
            "local_path_count": len(status_series.get("planner", [])),
        },
        "controller": {
            "path_deviation_p95_m": None if not math.isfinite(path_deviation_p95) else path_deviation_p95,
            "steering_error_p95_deg": None if not math.isfinite(steering_error_p95) else steering_error_p95,
            "speed_error_p95_pct": None if not math.isfinite(speed_error_p95) else speed_error_p95,
            "sharp_turn_low_speed_fraction": _fraction(low_speed_sharp_turn_mask),
            "max_desired_steering_deg": None if not math.isfinite(_max_abs(trajectory.desired_steering_deg)) else _max_abs(trajectory.desired_steering_deg),
            "max_applied_steering_deg": None if not math.isfinite(_max_abs(trajectory.applied_steering_deg)) else _max_abs(trajectory.applied_steering_deg),
            "steering_saturation_fraction": steering_saturation_fraction,
            "waiting_path_refresh_fraction": waiting_refresh_fraction,
        },
        "evidence": {
            "timing": timing_evidence,
            "planner": planner_evidence,
            "controller": controller_evidence,
        },
    }


def _plot_local_paths(ax: Any, local_paths: list[LocalPathItem]) -> None:
    if not local_paths:
        return
    label_used = False
    for item in local_paths:
        ax.plot(
            item.path_xy[:, 0],
            item.path_xy[:, 1],
            color="#4ea8de",
            linewidth=1.0,
            alpha=0.10,
            label="All local planned paths" if not label_used else None,
            zorder=3,
        )
        label_used = True
    latest_local_path_xy = local_paths[-1].path_xy
    ax.plot(
        latest_local_path_xy[:, 0],
        latest_local_path_xy[:, 1],
        color="#0b6aa2",
        linewidth=2.5,
        alpha=0.98,
        label="Latest local path to follow",
        zorder=6,
    )
    ax.scatter(
        [latest_local_path_xy[-1, 0]],
        [latest_local_path_xy[-1, 1]],
        c="#0b6aa2",
        s=46,
        marker="o",
        label="Latest local path end",
        zorder=7,
    )


def _plot_route_context(
    ax: Any,
    *,
    route_xy_yaw: np.ndarray,
    local_paths: list[LocalPathItem],
    tracking_xy: np.ndarray,
    lidar_world_xy: np.ndarray,
    summary: dict,
) -> None:
    planner_status = summary.get("planner_status") or {}
    start_pose = planner_status.get("start_pose") or {}
    start_axis = planner_status.get("start_axis") or {}
    route_xy = route_xy_yaw[:, :2] if route_xy_yaw.shape[0] else np.empty((0, 2), dtype=np.float64)

    if lidar_world_xy.shape[0] > 0:
        ax.scatter(
            lidar_world_xy[:, 0],
            lidar_world_xy[:, 1],
            s=4,
            c="#8c8c8c",
            alpha=0.16,
            linewidths=0.0,
            label=f"LiDAR aggregated ({lidar_world_xy.shape[0]} pts)",
            zorder=1,
        )
    if route_xy.shape[0] > 0:
        ax.plot(route_xy[:, 0], route_xy[:, 1], color="#1f77b4", linewidth=2.6, label="Saved route", zorder=4)
        ax.scatter([route_xy[0, 0]], [route_xy[0, 1]], c="#00a676", s=70, marker="o", label="Route start", zorder=5)
        ax.scatter([route_xy[-1, 0]], [route_xy[-1, 1]], c="#d81b60", s=75, marker="X", label="Route end", zorder=5)

    _plot_local_paths(ax, local_paths)

    if tracking_xy.shape[0] > 0:
        ax.plot(tracking_xy[:, 0], tracking_xy[:, 1], color="#111111", linewidth=2.2, label="Driven trajectory", zorder=6)
        ax.scatter([tracking_xy[-1, 0]], [tracking_xy[-1, 1]], c="#ff7f0e", s=60, marker="D", label="Tracking end", zorder=7)

    if start_pose and start_axis:
        normal_xy = np.asarray(start_axis.get("normal_xy") or [1.0, 0.0], dtype=np.float64)
        tangent_xy = np.asarray(start_axis.get("tangent_xy") or [0.0, 1.0], dtype=np.float64)
        line_center = np.asarray(
            [float(start_pose.get("x_m", 0.0)), float(start_pose.get("y_m", 0.0))],
            dtype=np.float64,
        )
        line_half_length_m = 1.2
        line_points = np.vstack([line_center - (line_half_length_m * tangent_xy), line_center + (line_half_length_m * tangent_xy)])
        ax.plot(line_points[:, 0], line_points[:, 1], color="#7b3fbc", linestyle="--", linewidth=2.0, label="Start axis", zorder=5)
        ax.arrow(
            float(line_center[0]),
            float(line_center[1]),
            0.35 * float(normal_xy[0]),
            0.35 * float(normal_xy[1]),
            width=0.01,
            head_width=0.07,
            head_length=0.08,
            color="#7b3fbc",
            length_includes_head=True,
            zorder=5,
        )

    all_points: list[np.ndarray] = []
    if lidar_world_xy.shape[0]:
        all_points.append(lidar_world_xy)
    if route_xy.shape[0]:
        all_points.append(route_xy)
    if local_paths:
        all_points.append(np.vstack([item.path_xy for item in local_paths]))
    if tracking_xy.shape[0]:
        all_points.append(tracking_xy)
    if all_points:
        combined = np.vstack(all_points)
        x_margin = max(0.25, 0.08 * float(np.max(combined[:, 0]) - np.min(combined[:, 0]) + 1.0))
        y_margin = max(0.25, 0.08 * float(np.max(combined[:, 1]) - np.min(combined[:, 1]) + 1.0))
        ax.set_xlim(float(np.min(combined[:, 0]) - x_margin), float(np.max(combined[:, 0]) + x_margin))
        ax.set_ylim(float(np.min(combined[:, 1]) - y_margin), float(np.max(combined[:, 1]) + y_margin))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x [m] (forward+)")
    ax.set_ylabel("y [m] (left+)")


def _plot_overview(
    *,
    route_xy_yaw: np.ndarray,
    local_paths: list[LocalPathItem],
    trajectory: TrajectorySeries,
    lidar_world_xy: np.ndarray,
    summary: dict,
    diagnostics: dict[str, Any],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 8.0))
    tracking_xy = (
        np.column_stack([trajectory.x_m, trajectory.y_m])
        if trajectory.x_m.size and trajectory.y_m.size
        else np.empty((0, 2), dtype=np.float64)
    )
    _plot_route_context(
        ax,
        route_xy_yaw=route_xy_yaw,
        local_paths=local_paths,
        tracking_xy=tracking_xy,
        lidar_world_xy=lidar_world_xy,
        summary=summary,
    )

    planner_status = summary.get("planner_status") or {}
    tracker_status = summary.get("tracker_status") or {}
    info_lines = [
        f"run: {Path(summary.get('run_dir', '')).name}",
        f"end: {summary.get('end_cause')}",
        f"primary cause: {diagnostics.get('primary_cause', 'n/a')}",
        f"trajectory points: {summary.get('trajectory_row_count')}",
        f"route points: {summary.get('route_point_count')}",
        f"local path msgs: {summary.get('local_path_message_count')}",
        f"travel distance: {planner_status.get('travel_distance_m', 'n/a')}",
        f"planner state: {planner_status.get('state', 'n/a')}",
        f"tracker state: {tracker_status.get('state', 'n/a')}",
        f"continuation: {planner_status.get('continuation_source', planner_status.get('local_path_source', 'n/a'))}",
        f"path loss reason: {tracker_status.get('path_loss_reason', 'n/a')}",
        f"planner span p10: {_format_float(float(diagnostics['planner'].get('path_forward_span_p10_m') or float('nan')), 2)} m",
        f"odom age p95: {_format_float(float(diagnostics['timing'].get('tracker_odom_age_p95_s') or float('nan')), 3)} s",
        f"path deviation p95: {_format_float(float(diagnostics['controller'].get('path_deviation_p95_m') or float('nan')), 3)} m",
        f"steer sat frac: {_format_pct(float(diagnostics['controller'].get('steering_saturation_fraction') or 0.0))}",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#666666", "alpha": 0.94},
    )
    ax.set_title("Recognition Tour: planned paths, route, and driven trajectory")
    ax.legend(loc="upper right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_state_spans(ax: Any, times_s: np.ndarray, states: list[str]) -> None:
    if times_s.size == 0 or not states:
        return
    problematic_states = {
        "holding_last_path": "#ffd166",
        "waiting_forward_path": "#ffd166",
        "waiting_path_refresh": "#ffd166",
        "aborted_path_loss": "#ef476f",
        "aborted_odom_timeout": "#ef476f",
        "waiting_odom": "#f4a261",
    }
    for index, state in enumerate(states):
        if state not in problematic_states:
            continue
        start_t = float(times_s[index])
        end_t = float(times_s[index + 1]) if index + 1 < times_s.size else start_t
        if end_t <= start_t:
            end_t = start_t + 0.02
        ax.axvspan(start_t, end_t, color=problematic_states[state], alpha=0.18)


def _plot_diagnostics(
    *,
    route_xy_yaw: np.ndarray,
    local_paths: list[LocalPathItem],
    trajectory: TrajectorySeries,
    lidar_world_xy: np.ndarray,
    summary: dict,
    status_series: dict[str, list[dict[str, Any]]],
    bridge_entries: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    output_path: Path,
) -> None:
    planner_entries = status_series.get("planner", [])
    tracker_entries = status_series.get("tracker", [])
    fusion_entries = status_series.get("fusion", [])
    tracking_xy = (
        np.column_stack([trajectory.x_m, trajectory.y_m])
        if trajectory.x_m.size and trajectory.y_m.size
        else np.empty((0, 2), dtype=np.float64)
    )

    fig = plt.figure(figsize=(15.0, 12.0))
    grid = GridSpec(3, 2, figure=fig, height_ratios=[2.2, 1.0, 1.0])
    ax_map = fig.add_subplot(grid[0, :])
    ax_timing = fig.add_subplot(grid[1, 0])
    ax_planner = fig.add_subplot(grid[1, 1])
    ax_control = fig.add_subplot(grid[2, 0])
    ax_text = fig.add_subplot(grid[2, 1])

    _plot_route_context(
        ax_map,
        route_xy_yaw=route_xy_yaw,
        local_paths=local_paths,
        tracking_xy=tracking_xy,
        lidar_world_xy=lidar_world_xy,
        summary=summary,
    )
    ax_map.set_title("Recognition Tour: what the planner published and what the vehicle actually did")
    ax_map.legend(loc="upper right", framealpha=0.92)

    tracker_t, tracker_odom_age = _extract_status_numeric(tracker_entries, "odom_age_s")
    _, tracker_path_age = _extract_status_numeric(tracker_entries, "path_age_s")
    planner_t, planner_odom_age = _extract_status_numeric(planner_entries, "odom_age_s")
    fusion_t, fusion_prediction_age = _extract_status_numeric(fusion_entries, "odom_prediction_age_s")
    tracker_state_t, tracker_states = _extract_status_strings(tracker_entries, "state")
    if trajectory.t_monotonic_s.size > 0:
        if np.isfinite(trajectory.odom_age_s).any():
            ax_timing.plot(trajectory.t_monotonic_s, trajectory.odom_age_s, color="#ef476f", linewidth=1.6, label="Tracker odom age")
        if np.isfinite(trajectory.path_age_s).any():
            ax_timing.plot(trajectory.t_monotonic_s, trajectory.path_age_s, color="#118ab2", linewidth=1.6, label="Path age")
        if np.isfinite(trajectory.odom_prediction_age_s).any():
            ax_timing.plot(
                trajectory.t_monotonic_s,
                trajectory.odom_prediction_age_s,
                color="#06d6a0",
                linewidth=1.4,
                label="Fusion prediction age",
            )
    else:
        if tracker_t.size:
            ax_timing.plot(tracker_t, tracker_odom_age, color="#ef476f", linewidth=1.6, label="Tracker odom age")
            ax_timing.plot(tracker_t, tracker_path_age, color="#118ab2", linewidth=1.6, label="Path age")
    if planner_t.size:
        ax_timing.plot(planner_t, planner_odom_age, color="#f4a261", linewidth=1.2, linestyle="--", label="Planner odom age")
    if fusion_t.size:
        ax_timing.plot(fusion_t, fusion_prediction_age, color="#06d6a0", linewidth=1.2, linestyle="--", label="Fusion prediction age")
    _plot_state_spans(ax_timing, tracker_state_t, tracker_states)
    ax_timing.axhline(0.08, color="#ef476f", linestyle=":", linewidth=1.1, alpha=0.7)
    ax_timing.axhline(0.25, color="#118ab2", linestyle=":", linewidth=1.1, alpha=0.7)
    ax_timing.set_title("Timing diagnostics")
    ax_timing.set_xlabel("t [s]")
    ax_timing.set_ylabel("age [s]")
    ax_timing.grid(True, alpha=0.25)
    ax_timing.legend(loc="upper right", fontsize=9)

    planner_t, path_span = _extract_status_numeric(planner_entries, "path_forward_span_m")
    _, path_length = _extract_status_numeric(planner_entries, "path_length_m")
    _, path_curvature = _extract_status_numeric(planner_entries, "path_max_curvature_m_inv")
    if planner_t.size == 0 and trajectory.t_monotonic_s.size > 0:
        planner_t = trajectory.t_monotonic_s
        path_span = trajectory.planner_path_forward_span_m
        path_length = trajectory.planner_path_length_m
        path_curvature = trajectory.planner_path_max_curvature_m_inv
    if planner_t.size > 0:
        ax_planner.plot(planner_t, path_span, color="#1f77b4", linewidth=1.8, label="Forward span")
        ax_planner.plot(planner_t, path_length, color="#00a676", linewidth=1.6, label="Path length")
        ax_planner.axhline(0.95, color="#d81b60", linestyle=":", linewidth=1.1, label="Short-path threshold")
        ax_planner_twin = ax_planner.twinx()
        ax_planner_twin.plot(planner_t, path_curvature, color="#ff7f0e", linewidth=1.6, label="Max curvature")
        ax_planner_twin.set_ylabel("curvature [1/m]")
        lines_1, labels_1 = ax_planner.get_legend_handles_labels()
        lines_2, labels_2 = ax_planner_twin.get_legend_handles_labels()
        ax_planner.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=9)
    ax_planner.set_title("Planner continuity diagnostics")
    ax_planner.set_xlabel("t [s]")
    ax_planner.set_ylabel("distance [m]")
    ax_planner.grid(True, alpha=0.25)

    bridge_t = np.asarray([float(entry.get("t_monotonic_s", float("nan"))) for entry in bridge_entries], dtype=np.float64)
    bridge_desired_steer = np.asarray(
        [
            float((entry.get("payload") or {}).get("desired_steering_deg", float("nan")))
            for entry in bridge_entries
        ],
        dtype=np.float64,
    )
    bridge_requested_steer = np.asarray(
        [
            float((entry.get("payload") or {}).get("requested_steering_deg", float("nan")))
            for entry in bridge_entries
        ],
        dtype=np.float64,
    )
    bridge_applied_steer = np.asarray(
        [
            float((entry.get("payload") or {}).get("applied_steering_deg", float("nan")))
            for entry in bridge_entries
        ],
        dtype=np.float64,
    )
    bridge_steering_saturated = np.asarray(
        [
            float((entry.get("payload") or {}).get("steering_saturated", float("nan")))
            for entry in bridge_entries
        ],
        dtype=np.float64,
    )
    bridge_desired_speed = np.asarray(
        [
            float((entry.get("payload") or {}).get("desired_speed_pct", float("nan")))
            for entry in bridge_entries
        ],
        dtype=np.float64,
    )
    bridge_applied_speed = np.asarray(
        [
            float((entry.get("payload") or {}).get("applied_speed_pct", float("nan")))
            for entry in bridge_entries
        ],
        dtype=np.float64,
    )
    if bridge_t.size > 0:
        ax_control.plot(bridge_t, bridge_desired_steer, color="#ef476f", linewidth=1.5, label="Desired steering")
        ax_control.plot(bridge_t, bridge_requested_steer, color="#f78c6b", linewidth=1.2, linestyle="--", label="Requested steering")
        ax_control.plot(bridge_t, bridge_applied_steer, color="#f78c6b", linewidth=1.2, label="Applied steering")
        saturated_mask = np.isfinite(bridge_steering_saturated) & (bridge_steering_saturated > 0.5)
        if np.any(saturated_mask):
            ax_control.scatter(
                bridge_t[saturated_mask],
                bridge_applied_steer[saturated_mask],
                s=12,
                color="#d81b60",
                alpha=0.7,
                label="Steering saturated",
            )
        ax_control_twin = ax_control.twinx()
        ax_control_twin.plot(bridge_t, bridge_desired_speed, color="#118ab2", linewidth=1.4, label="Desired speed %")
        ax_control_twin.plot(bridge_t, bridge_applied_speed, color="#06d6a0", linewidth=1.2, label="Applied speed %")
        lines_1, labels_1 = ax_control.get_legend_handles_labels()
        lines_2, labels_2 = ax_control_twin.get_legend_handles_labels()
        ax_control.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=9)
        ax_control_twin.set_ylabel("speed [%]")
    else:
        if trajectory.t_monotonic_s.size > 0:
            ax_control.plot(
                trajectory.t_monotonic_s,
                trajectory.tracker_desired_steering_deg,
                color="#ef476f",
                linewidth=1.5,
                label="Tracker desired steering",
            )
            ax_control.plot(
                trajectory.t_monotonic_s,
                trajectory.path_deviation_m,
                color="#118ab2",
                linewidth=1.2,
                label="Path deviation",
            )
            ax_control.legend(loc="upper right", fontsize=9)
    ax_control.set_title("Controller / actuation diagnostics")
    ax_control.set_xlabel("t [s]")
    ax_control.set_ylabel("steering [deg]")
    ax_control.grid(True, alpha=0.25)

    ax_text.axis("off")
    evidence = diagnostics.get("evidence", {})
    text_lines = [
        f"Primary cause: {diagnostics.get('primary_cause', 'n/a')}",
        "",
        "Scores:",
        f"timing={diagnostics.get('scores', {}).get('timing', 'n/a')}",
        f"planner={diagnostics.get('scores', {}).get('planner', 'n/a')}",
        f"controller={diagnostics.get('scores', {}).get('controller', 'n/a')}",
        "",
        "Timing evidence:",
    ]
    text_lines.extend([f"- {item}" for item in evidence.get("timing", [])[:3]] or ["- none strong"])
    text_lines.append("")
    text_lines.append("Planner evidence:")
    text_lines.extend([f"- {item}" for item in evidence.get("planner", [])[:3]] or ["- none strong"])
    text_lines.append("")
    text_lines.append("Controller evidence:")
    text_lines.extend([f"- {item}" for item in evidence.get("controller", [])[:3]] or ["- none strong"])
    ax_text.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#666666", "alpha": 0.95},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    analysis_dir = run_dir / "analysis_recognition_tour"
    route_json = analysis_dir / "recognition_tour_route.json"
    local_path_history_jsonl = analysis_dir / "recognition_tour_local_path_history.jsonl"
    tracking_csv = analysis_dir / "recognition_tour_trajectory.csv"
    summary_json = analysis_dir / "recognition_tour_summary.json"
    status_log = analysis_dir / "recognition_tour_status.log"
    bridge_status_log = analysis_dir / "drive_bridge_status.log"
    lidar_points_csv = run_dir / "lidar_points.csv"
    overview_output = analysis_dir / "recognition_tour_overview.png"
    diagnostics_output = analysis_dir / "recognition_tour_diagnostics.png"
    diagnostics_json = analysis_dir / "recognition_tour_diagnostics.json"

    summary = _load_json(summary_json)
    route_xy_yaw = _load_route_json(route_json) if route_json.exists() else np.empty((0, 3), dtype=np.float64)
    local_paths = _load_local_path_history(local_path_history_jsonl)
    trajectory = _load_tracking_series(tracking_csv) if tracking_csv.exists() else _load_tracking_series(Path("/dev/null"))
    lidar_world_xy = _load_lidar_world_points(lidar_points_csv)
    status_series = _load_status_log(status_log)
    bridge_entries = _load_bridge_log(bridge_status_log)
    diagnostics = _compute_root_cause_diagnostics(
        summary=summary,
        trajectory=trajectory,
        status_series=status_series,
        bridge_entries=bridge_entries,
    )
    diagnostics_json.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    _plot_overview(
        route_xy_yaw=route_xy_yaw,
        local_paths=local_paths,
        trajectory=trajectory,
        lidar_world_xy=lidar_world_xy,
        summary=summary,
        diagnostics=diagnostics,
        output_path=overview_output,
    )
    _plot_diagnostics(
        route_xy_yaw=route_xy_yaw,
        local_paths=local_paths,
        trajectory=trajectory,
        lidar_world_xy=lidar_world_xy,
        summary=summary,
        status_series=status_series,
        bridge_entries=bridge_entries,
        diagnostics=diagnostics,
        output_path=diagnostics_output,
    )


if __name__ == "__main__":
    main()
