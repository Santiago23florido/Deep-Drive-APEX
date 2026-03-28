#!/usr/bin/env python3
"""Detect a visible curve opening from a static LiDAR snapshot CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SideProfile:
    side_name: str
    side_sign: int
    x_m: np.ndarray
    y_m: np.ndarray
    counts: np.ndarray
    fit_coef: np.ndarray
    fit_y_m: np.ndarray
    deviation_m: np.ndarray


@dataclass
class CurveCandidate:
    side_name: str
    side_sign: int
    entry_x_m: float
    entry_y_m: float
    first_curve_x_m: float
    first_curve_y_m: float
    start_forward_m: float
    start_radial_m: float
    straight_width_m: float
    entry_width_m: float
    curve_wall_shift_m: float
    angle_start_deg: float
    angle_end_deg: float
    angle_center_deg: float
    curve_point_count: int
    cluster_start_idx: int
    cluster_end_idx: int
    cluster_last_straight_idx: int
    straight_end_x_m: float
    opposite_wall_visible_until_x_m: float
    same_side_gap_m: float
    curve_width_m: float
    opening_width_gain_m: float
    front_closure_point_count: int
    front_closure_y_span_m: float
    heuristic_name: str
    gap_only_opening: bool
    window_far_wall_x_m: float
    window_width_m: float
    score: float


@dataclass
class TrajectoryPlan:
    x_m: np.ndarray
    y_m: np.ndarray
    anchor_points: list[tuple[float, float]]
    entry_center_y_m: float
    target_x_m: float
    target_y_m: float


def _side_label(side_name: str) -> str:
    if side_name == "left":
        return "izquierda"
    if side_name == "right":
        return "derecha"
    return side_name


def _normalize_angle_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def _display_xy_from_snapshot(
    angles_deg: np.ndarray,
    ranges_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    angles_rad = np.radians(angles_deg)
    x_m = -ranges_m * np.cos(angles_rad)
    y_m = ranges_m * np.sin(angles_rad)
    return x_m, y_m


def _load_snapshot_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles_deg: list[float] = []
    ranges_m: list[float] = []
    counts: list[int] = []
    with csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            angle_deg = float(row["angle_deg"])
            try:
                range_m = float(row["range_m"])
            except ValueError:
                range_m = float("nan")
            count = int(row["count"])
            if not math.isfinite(range_m) or range_m <= 0.0 or count <= 0:
                continue
            angles_deg.append(angle_deg)
            ranges_m.append(range_m)
            counts.append(count)
    if not angles_deg:
        raise ValueError(f"No valid points found in {csv_path}")
    return (
        np.asarray(angles_deg, dtype=np.float64),
        np.asarray(ranges_m, dtype=np.float64),
        np.asarray(counts, dtype=np.int32),
    )


def _build_side_profile(
    *,
    side_name: str,
    side_sign: int,
    x_m: np.ndarray,
    y_m: np.ndarray,
    x_min_m: float,
    x_max_m: float,
    x_bin_m: float,
    fit_x_min_m: float,
    fit_x_max_m: float,
    min_points_per_bin: int,
) -> SideProfile:
    xs: list[float] = []
    ys: list[float] = []
    counts: list[int] = []

    for x0_m in np.arange(x_min_m, x_max_m, x_bin_m):
        mask = (x_m >= x0_m) & (x_m < (x0_m + x_bin_m))
        if side_sign < 0:
            mask &= y_m < 0.0
        else:
            mask &= y_m > 0.0
        if int(np.count_nonzero(mask)) < min_points_per_bin:
            continue
        xs.append(float(x0_m + 0.5 * x_bin_m))
        ys.append(float(np.median(y_m[mask])))
        counts.append(int(np.count_nonzero(mask)))

    if len(xs) < 4:
        raise ValueError(f"Not enough profile points for side '{side_name}'")

    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    count_arr = np.asarray(counts, dtype=np.int32)

    fit_mask = (x_arr >= fit_x_min_m) & (x_arr <= fit_x_max_m)
    if int(np.count_nonzero(fit_mask)) < 3:
        raise ValueError(f"Not enough fit points for side '{side_name}'")

    fit_coef = np.polyfit(x_arr[fit_mask], y_arr[fit_mask], deg=1)
    fit_y_arr = np.polyval(fit_coef, x_arr)
    if side_sign < 0:
        deviation_arr = fit_y_arr - y_arr
    else:
        deviation_arr = y_arr - fit_y_arr

    return SideProfile(
        side_name=side_name,
        side_sign=side_sign,
        x_m=x_arr,
        y_m=y_arr,
        counts=count_arr,
        fit_coef=fit_coef,
        fit_y_m=fit_y_arr,
        deviation_m=deviation_arr,
    )


def _fit_y_at(profile: SideProfile, x_m: float) -> float:
    return float(np.polyval(profile.fit_coef, float(x_m)))


def _cluster_profile_indices(
    profile: SideProfile,
    indices: np.ndarray,
    *,
    max_dx_m: float,
) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    clusters: list[np.ndarray] = []
    current_cluster = [int(indices[0])]
    for idx in indices[1:]:
        idx = int(idx)
        if (profile.x_m[idx] - profile.x_m[current_cluster[-1]]) <= max_dx_m:
            current_cluster.append(idx)
        else:
            clusters.append(np.asarray(current_cluster, dtype=np.int32))
            current_cluster = [idx]
    clusters.append(np.asarray(current_cluster, dtype=np.int32))
    return clusters


def _detect_front_closure(
    *,
    points_x_m: np.ndarray,
    points_y_m: np.ndarray,
    entry_x_m: float,
    lower_y_m: float,
    upper_y_m: float,
    x_window_m: float,
) -> tuple[int, float]:
    low_y_m = min(lower_y_m, upper_y_m)
    high_y_m = max(lower_y_m, upper_y_m)
    corridor_width_m = high_y_m - low_y_m
    if corridor_width_m <= 0.0:
        return 0, 0.0

    # Ignore the side walls themselves and only count points that would behave like
    # a front-closing wall crossing the interior of the corridor mouth.
    inner_margin_m = max(0.10, 0.18 * corridor_width_m)
    mask = np.abs(points_x_m - entry_x_m) <= x_window_m
    mask &= points_y_m >= (low_y_m + inner_margin_m)
    mask &= points_y_m <= (high_y_m - inner_margin_m)
    if not np.any(mask):
        return 0, 0.0

    closure_y = points_y_m[mask]
    return int(closure_y.size), float(np.max(closure_y) - np.min(closure_y))


def _straight_continuation_extent(
    profile: SideProfile,
    *,
    start_x_m: float,
    max_abs_deviation_m: float,
    max_dx_m: float,
) -> float | None:
    abs_deviation_m = np.abs(profile.y_m - profile.fit_y_m)
    indices = np.flatnonzero(
        (profile.x_m >= (start_x_m - 0.03))
        & (abs_deviation_m <= max_abs_deviation_m)
    )
    if indices.size == 0:
        return None

    clusters = _cluster_profile_indices(profile, indices, max_dx_m=max_dx_m)
    for cluster in clusters:
        if float(profile.x_m[int(cluster[0])]) <= (start_x_m + max_dx_m):
            return float(profile.x_m[int(cluster[-1])])
    return None


def _find_side_gap_window(
    *,
    points_x_m: np.ndarray,
    points_y_m: np.ndarray,
    side_sign: int,
    search_x_min_m: float,
    x_bin_m: float,
) -> tuple[float, float, float, float] | None:
    occupied_x_m: list[float] = []
    max_x_m = float(np.max(points_x_m))
    x0_m = float(search_x_min_m)
    while x0_m <= (max_x_m + x_bin_m):
        mask = (points_x_m >= x0_m) & (points_x_m < (x0_m + x_bin_m))
        mask &= (points_y_m * float(side_sign)) > 0.02
        if np.any(mask):
            occupied_x_m.append(float(x0_m + 0.5 * x_bin_m))
        x0_m += x_bin_m

    if len(occupied_x_m) < 4:
        return None

    occupied = np.asarray(occupied_x_m, dtype=np.float64)
    gap_threshold_m = max(0.20, 3.0 * float(x_bin_m))
    cluster_threshold_m = max(0.12, 2.1 * float(x_bin_m))
    for idx in range(1, occupied.size):
        gap_m = float(occupied[idx] - occupied[idx - 1])
        if gap_m < gap_threshold_m:
            continue
        pre_x_m = float(occupied[idx - 1])
        post_x_m = float(occupied[idx])
        far_wall_x_m = post_x_m
        j = idx + 1
        while j < occupied.size and float(occupied[j] - occupied[j - 1]) <= cluster_threshold_m:
            far_wall_x_m = float(occupied[j])
            j += 1
        return pre_x_m, post_x_m, gap_m, far_wall_x_m
    return None


def _detect_curve_candidate(
    *,
    profile: SideProfile,
    opposite_profile: SideProfile,
    angles_deg: np.ndarray,
    points_x_m: np.ndarray,
    points_y_m: np.ndarray,
    search_x_min_m: float,
    deviation_threshold_m: float,
    min_curve_bins: int,
    x_bin_m: float,
    gap_threshold_m: float,
    opposite_continuation_min_m: float,
    front_closure_x_window_m: float,
    front_closure_min_points: int,
) -> CurveCandidate | None:
    forward_indices = np.flatnonzero(profile.x_m >= search_x_min_m)
    if forward_indices.size < 3:
        return None

    candidate_indices = forward_indices[profile.deviation_m[forward_indices] >= deviation_threshold_m]
    valid_clusters = [
        cluster
        for cluster in _cluster_profile_indices(profile, candidate_indices, max_dx_m=(gap_threshold_m + 0.01))
        if cluster.size >= min_curve_bins
    ]
    earliest_cluster = valid_clusters[0] if valid_clusters else None

    gap_prev_idx: int | None = None
    gap_next_idx: int | None = None
    same_side_gap_m = 0.0
    gap_only_opening = False
    gap_far_wall_x_m = float("nan")
    strong_gap_opening = False

    side_gap = _find_side_gap_window(
        points_x_m=points_x_m,
        points_y_m=points_y_m,
        side_sign=profile.side_sign,
        search_x_min_m=search_x_min_m,
        x_bin_m=x_bin_m,
    )
    if side_gap is not None:
        gap_prev_x_m, gap_next_x_m, same_side_gap_m, gap_far_wall_x_m = side_gap
        gap_prev_idx = int(np.argmin(np.abs(profile.x_m - gap_prev_x_m)))
        next_mask = np.flatnonzero(profile.x_m >= gap_next_x_m)
        gap_next_idx = int(next_mask[0]) if next_mask.size else gap_prev_idx
        strong_gap_opening = same_side_gap_m >= max(0.30, 3.0 * x_bin_m)

    transition_prev_idx: int | None = None
    transition_next_idx: int | None = None
    heuristic_name = ""

    if strong_gap_opening and gap_prev_idx is not None and gap_next_idx is not None:
        transition_prev_idx = gap_prev_idx
        transition_next_idx = gap_next_idx
        heuristic_name = "gap+opposite_continuity"

    if earliest_cluster is not None and not strong_gap_opening:
        cluster_start_idx = int(earliest_cluster[0])
        straight_before_cluster = forward_indices[
            (forward_indices < cluster_start_idx)
            & (profile.deviation_m[forward_indices] < (0.5 * deviation_threshold_m))
        ]
        cluster_last_straight_idx = int(straight_before_cluster[-1]) if straight_before_cluster.size else cluster_start_idx
        if (
            transition_next_idx is None
            or float(profile.x_m[cluster_start_idx]) < float(profile.x_m[transition_next_idx])
        ):
            transition_prev_idx = cluster_last_straight_idx
            transition_next_idx = cluster_start_idx
            same_side_gap_m = max(
                same_side_gap_m,
                float(profile.x_m[cluster_start_idx] - profile.x_m[cluster_last_straight_idx]),
            )
            heuristic_name = "wall_shift+opposite_continuity"

    if transition_prev_idx is None or transition_next_idx is None:
        return None

    cluster_start_idx = int(transition_next_idx)
    cluster_last_straight_idx = int(transition_prev_idx)
    cluster_end_idx = int(cluster_start_idx)
    if earliest_cluster is not None and int(earliest_cluster[0]) == cluster_start_idx:
        cluster_end_idx = int(earliest_cluster[-1])

    straight_end_x_m = float(profile.x_m[cluster_last_straight_idx])
    if heuristic_name.startswith("gap+") and same_side_gap_m >= max(0.30, 3.0 * x_bin_m):
        gap_only_opening = True
        entry_x_m = straight_end_x_m
        same_side_straight_y_m = _fit_y_at(profile, entry_x_m)
        first_curve_x_m = straight_end_x_m
        first_curve_y_m = same_side_straight_y_m
        entry_y_m = same_side_straight_y_m
        cluster_start_idx = cluster_last_straight_idx
        cluster_end_idx = cluster_last_straight_idx
    else:
        entry_x_m = float(
            0.5 * (profile.x_m[cluster_last_straight_idx] + profile.x_m[cluster_start_idx])
        )
        same_side_straight_y_m = _fit_y_at(profile, entry_x_m)
        first_curve_x_m = float(profile.x_m[cluster_start_idx])
        first_curve_y_m = float(profile.y_m[cluster_start_idx])
        entry_y_m = float(0.5 * (same_side_straight_y_m + first_curve_y_m))
    start_forward_m = max(0.0, entry_x_m)
    start_radial_m = float(math.hypot(entry_x_m, entry_y_m))

    if gap_only_opening:
        opposite_forward = opposite_profile.x_m[opposite_profile.x_m >= (entry_x_m - 0.03)]
        if opposite_forward.size == 0:
            return None
        opposite_wall_visible_until_x_m = float(np.max(opposite_forward))
    else:
        opposite_wall_visible_until_x_m = _straight_continuation_extent(
            opposite_profile,
            start_x_m=entry_x_m,
            max_abs_deviation_m=max(0.06, 0.6 * deviation_threshold_m),
            max_dx_m=max(0.12, gap_threshold_m + 0.03),
        )
        if opposite_wall_visible_until_x_m is None:
            return None
    if (opposite_wall_visible_until_x_m - entry_x_m) < opposite_continuation_min_m:
        return None

    opposite_side_y_m = _fit_y_at(opposite_profile, entry_x_m)
    if profile.side_sign < 0:
        straight_width_m = max(0.0, opposite_side_y_m - same_side_straight_y_m)
        entry_width_m = max(0.0, opposite_side_y_m - entry_y_m)
        curve_width_m = max(0.0, _fit_y_at(opposite_profile, first_curve_x_m) - first_curve_y_m)
    else:
        straight_width_m = max(0.0, same_side_straight_y_m - opposite_side_y_m)
        entry_width_m = max(0.0, entry_y_m - opposite_side_y_m)
        curve_width_m = max(0.0, first_curve_y_m - _fit_y_at(opposite_profile, first_curve_x_m))
    curve_wall_shift_m = abs(_fit_y_at(profile, first_curve_x_m) - first_curve_y_m)
    opening_width_gain_m = max(0.0, curve_width_m - straight_width_m)
    if gap_only_opening:
        curve_wall_shift_m = 0.0
        curve_width_m = straight_width_m
        opening_width_gain_m = 0.0
        window_far_wall_x_m = gap_far_wall_x_m
        window_width_m = max(0.0, window_far_wall_x_m - entry_x_m)
    else:
        window_far_wall_x_m = float("nan")
        window_width_m = 0.0

    front_closure_point_count, front_closure_y_span_m = _detect_front_closure(
        points_x_m=points_x_m,
        points_y_m=points_y_m,
        entry_x_m=entry_x_m,
        lower_y_m=same_side_straight_y_m,
        upper_y_m=opposite_side_y_m,
        x_window_m=front_closure_x_window_m,
    )
    if (
        front_closure_point_count >= front_closure_min_points
        and front_closure_y_span_m >= (0.30 * max(straight_width_m, 1e-6))
    ):
        return None

    curve_angles_deg = np.empty((0,), dtype=np.float64)
    if not gap_only_opening:
        curve_mask = (
            (points_x_m >= (entry_x_m - 0.02))
            & ((points_y_m * profile.side_sign) > 0.0)
        )
        if profile.side_sign < 0:
            curve_mask &= points_y_m <= (
                np.polyval(profile.fit_coef, points_x_m) - 0.5 * deviation_threshold_m
            )
        else:
            curve_mask &= points_y_m >= (
                np.polyval(profile.fit_coef, points_x_m) + 0.5 * deviation_threshold_m
            )
        curve_angles_deg = angles_deg[curve_mask]
    if curve_angles_deg.size == 0:
        angle_start_deg = float("nan")
        angle_end_deg = float("nan")
        angle_center_deg = float("nan")
        curve_point_count = 0
    else:
        angle_start_deg = float(np.min(curve_angles_deg))
        angle_end_deg = float(np.max(curve_angles_deg))
        angle_center_deg = float(0.5 * (angle_start_deg + angle_end_deg))
        curve_point_count = int(curve_angles_deg.size)

    score = (
        2.5 * curve_wall_shift_m
        + 1.5 * opening_width_gain_m
        + 0.9 * min(0.80, opposite_wall_visible_until_x_m - entry_x_m)
        + 0.7 * min(0.80, same_side_gap_m)
        - 0.15 * front_closure_point_count
    )
    if heuristic_name.startswith("gap+"):
        score += 1.2 + 0.25 * min(0.80, same_side_gap_m)
    if gap_only_opening:
        score += 1.0 + 0.8 * min(1.50, window_width_m)
    if curve_point_count > 0 and math.isfinite(angle_center_deg) and abs(angle_center_deg) > 100.0:
        score -= 1.5

    return CurveCandidate(
        side_name=profile.side_name,
        side_sign=profile.side_sign,
        entry_x_m=entry_x_m,
        entry_y_m=entry_y_m,
        first_curve_x_m=first_curve_x_m,
        first_curve_y_m=first_curve_y_m,
        start_forward_m=start_forward_m,
        start_radial_m=start_radial_m,
        straight_width_m=straight_width_m,
        entry_width_m=entry_width_m,
        curve_wall_shift_m=curve_wall_shift_m,
        angle_start_deg=angle_start_deg,
        angle_end_deg=angle_end_deg,
        angle_center_deg=angle_center_deg,
        curve_point_count=curve_point_count,
        cluster_start_idx=cluster_start_idx,
        cluster_end_idx=cluster_end_idx,
        cluster_last_straight_idx=cluster_last_straight_idx,
        straight_end_x_m=straight_end_x_m,
        opposite_wall_visible_until_x_m=opposite_wall_visible_until_x_m,
        same_side_gap_m=same_side_gap_m,
        curve_width_m=curve_width_m,
        opening_width_gain_m=opening_width_gain_m,
        front_closure_point_count=front_closure_point_count,
        front_closure_y_span_m=front_closure_y_span_m,
        heuristic_name=heuristic_name,
        gap_only_opening=gap_only_opening,
        window_far_wall_x_m=window_far_wall_x_m,
        window_width_m=window_width_m,
        score=score,
    )


def _bezier_curve(points_xy: list[tuple[float, float]], num_samples: int = 80) -> tuple[np.ndarray, np.ndarray]:
    control = np.asarray(points_xy, dtype=np.float64)
    if control.shape != (4, 2):
        raise ValueError("Bezier curve requires exactly 4 control points")
    t = np.linspace(0.0, 1.0, int(max(8, num_samples)), dtype=np.float64)
    omt = 1.0 - t
    samples = (
        (omt ** 3)[:, None] * control[0]
        + (3.0 * (omt ** 2) * t)[:, None] * control[1]
        + (3.0 * omt * (t ** 2))[:, None] * control[2]
        + (t ** 3)[:, None] * control[3]
    )
    return samples[:, 0], samples[:, 1]


def _build_trajectory_plan(
    *,
    candidate: CurveCandidate,
    left_profile: SideProfile,
    right_profile: SideProfile,
    axis_limit_m: float,
) -> TrajectoryPlan:
    same_profile = left_profile if candidate.side_sign < 0 else right_profile
    opposite_profile = right_profile if candidate.side_sign < 0 else left_profile

    def corridor_center_y(x_m: float, same_side_y_m: float | None = None) -> float:
        opposite_y_m = _fit_y_at(opposite_profile, x_m)
        if same_side_y_m is None:
            same_side_y_m = _fit_y_at(same_profile, x_m)
        return 0.5 * (same_side_y_m + opposite_y_m)

    start_x_m = 0.0
    start_y_m = corridor_center_y(start_x_m)
    preturn_x_m = max(0.05, candidate.straight_end_x_m - 0.12)
    if preturn_x_m >= candidate.entry_x_m:
        preturn_x_m = max(0.05, candidate.entry_x_m - 0.12)
    preturn_y_m = corridor_center_y(preturn_x_m)

    entry_same_wall_y_m = _fit_y_at(same_profile, candidate.entry_x_m)
    entry_opposite_y_m = _fit_y_at(opposite_profile, candidate.entry_x_m)
    entry_center_y_m = corridor_center_y(candidate.entry_x_m, same_side_y_m=entry_same_wall_y_m)

    if candidate.gap_only_opening:
        width_m = max(candidate.window_width_m, 0.40)
        mouth_mid_x_m = candidate.entry_x_m + 0.5 * width_m
        entry_track_y_m = entry_same_wall_y_m + candidate.side_sign * min(0.05, 0.05 * width_m)
        target_x_m = mouth_mid_x_m
        target_y_m = entry_same_wall_y_m + candidate.side_sign * min(0.55, 0.42 * width_m)
    else:
        entry_track_y_m = candidate.entry_y_m + (
            candidate.side_sign * 0.06 * max(candidate.entry_width_m, candidate.straight_width_m)
        )
        target_x_m = max(candidate.entry_x_m + 0.18, candidate.first_curve_x_m + 0.12)
        target_x_m = min(target_x_m, max(candidate.entry_x_m + 0.18, axis_limit_m * 0.80))
        opposite_target_y_m = _fit_y_at(opposite_profile, target_x_m)

        if candidate.curve_point_count > 0:
            same_target_y_m = candidate.first_curve_y_m
        else:
            same_target_y_m = _fit_y_at(same_profile, candidate.entry_x_m) + (
                candidate.side_sign * max(0.12, 0.16 * max(candidate.straight_width_m, 0.5))
            )

        inside_bias_m = min(0.16, 0.18 * max(candidate.entry_width_m, candidate.straight_width_m))
        target_y_m = same_target_y_m + candidate.side_sign * inside_bias_m
        if candidate.side_sign < 0:
            target_y_m = min(target_y_m, opposite_target_y_m - 0.22 * max(candidate.entry_width_m, 0.4))
        else:
            target_y_m = max(target_y_m, opposite_target_y_m + 0.22 * max(candidate.entry_width_m, 0.4))

    if candidate.gap_only_opening:
        control_points = [
            (start_x_m, start_y_m),
            (max(0.10, 0.55 * candidate.entry_x_m), start_y_m),
            (candidate.entry_x_m + 0.38 * width_m, entry_track_y_m),
            (target_x_m, target_y_m),
        ]
    else:
        control_points = [
            (start_x_m, start_y_m),
            (preturn_x_m, preturn_y_m),
            (candidate.entry_x_m, entry_track_y_m),
            (target_x_m, target_y_m),
        ]
    traj_x_m, traj_y_m = _bezier_curve(control_points, num_samples=90)
    return TrajectoryPlan(
        x_m=traj_x_m,
        y_m=traj_y_m,
        anchor_points=control_points,
        entry_center_y_m=entry_center_y_m,
        target_x_m=target_x_m,
        target_y_m=target_y_m,
    )


def _plot_analysis(
    *,
    output_path: Path,
    x_m: np.ndarray,
    y_m: np.ndarray,
    left_profile: SideProfile,
    right_profile: SideProfile,
    candidate: CurveCandidate | None,
    axis_limit_m: float,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(x_m, y_m, s=10, c="#27c1d9", alpha=0.85, label="Puntos LiDAR")
    ax.scatter([0.0], [0.0], s=44, c="#d62728", label="LiDAR")

    ax.plot(
        left_profile.x_m,
        left_profile.fit_y_m,
        linestyle="--",
        linewidth=1.5,
        color="#1f77b4",
        label="Recta base izquierda",
    )
    ax.plot(
        right_profile.x_m,
        right_profile.fit_y_m,
        linestyle="--",
        linewidth=1.5,
        color="#2ca02c",
        label="Recta base derecha",
    )
    ax.plot(left_profile.x_m, left_profile.y_m, linewidth=1.2, color="#4c78a8", alpha=0.70)
    ax.plot(right_profile.x_m, right_profile.y_m, linewidth=1.2, color="#54a24b", alpha=0.70)

    info_lines = [
        "No se detecto una curva visible dominante.",
    ]
    trajectory: TrajectoryPlan | None = None

    if candidate is not None:
        profile = left_profile if candidate.side_sign < 0 else right_profile
        side_label = _side_label(candidate.side_name)
        trajectory = _build_trajectory_plan(
            candidate=candidate,
            left_profile=left_profile,
            right_profile=right_profile,
            axis_limit_m=axis_limit_m,
        )
        cluster_slice = slice(candidate.cluster_start_idx, candidate.cluster_end_idx + 1)
        ax.axvline(
            candidate.straight_end_x_m,
            linestyle=":",
            linewidth=1.4,
            color="#6b6b6b",
            alpha=0.90,
            label="Fin pared recta",
        )
        ax.plot(
            profile.x_m[cluster_slice],
            profile.y_m[cluster_slice],
            linewidth=2.6,
            color="#ff7f0e",
            label=f"Borde curvo {side_label}",
        )
        ax.scatter(
            [candidate.first_curve_x_m],
            [candidate.first_curve_y_m],
            s=64,
            c="#9467bd",
            label="Primer punto curvo",
        )
        ax.scatter(
            [candidate.entry_x_m],
            [candidate.entry_y_m],
            s=70,
            c="#111111",
            marker="x",
            label="Inicio estimado",
        )
        ax.plot(
            trajectory.x_m,
            trajectory.y_m,
            linewidth=2.8,
            color="#d81b60",
            label="Trayectoria estimada",
        )
        ax.scatter(
            [point[0] for point in trajectory.anchor_points],
            [point[1] for point in trajectory.anchor_points],
            s=26,
            c="#d81b60",
            alpha=0.85,
        )
        ax.annotate(
            "",
            xy=(trajectory.x_m[-1], trajectory.y_m[-1]),
            xytext=(trajectory.x_m[-6], trajectory.y_m[-6]),
            arrowprops={"arrowstyle": "->", "color": "#d81b60", "linewidth": 2.4},
        )

        if candidate.gap_only_opening:
            ax.plot(
                [candidate.entry_x_m, candidate.window_far_wall_x_m],
                [candidate.entry_y_m, candidate.entry_y_m],
                linestyle=":",
                linewidth=2.0,
                color="#111111",
            )
            ax.text(
                0.5 * (candidate.entry_x_m + candidate.window_far_wall_x_m),
                candidate.entry_y_m - 0.04 * axis_limit_m,
                f"ancho ventana\n{candidate.window_width_m:.3f} m",
                fontsize=9,
                ha="center",
                va="top",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.90, "edgecolor": "#555555"},
            )
        else:
            opposite_y_m = _fit_y_at(right_profile if candidate.side_sign < 0 else left_profile, candidate.entry_x_m)
            ax.plot(
                [candidate.entry_x_m, candidate.entry_x_m],
                [candidate.entry_y_m, opposite_y_m],
                linestyle=":",
                linewidth=2.0,
                color="#111111",
            )
            ax.text(
                candidate.entry_x_m + 0.03,
                0.5 * (candidate.entry_y_m + opposite_y_m),
                f"ancho visible\n{candidate.entry_width_m:.3f} m",
                fontsize=9,
                ha="left",
                va="center",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.90, "edgecolor": "#555555"},
            )

        info_lines = [
            f"Curva visible: {side_label}",
            f"Inicio estimado: x={candidate.entry_x_m:.3f} m",
            f"Fin pared recta: x={candidate.straight_end_x_m:.3f} m",
            f"Pared visible despues de ventana: x={candidate.window_far_wall_x_m:.3f} m" if candidate.gap_only_opening else f"Continuidad pared opuesta hasta x={candidate.opposite_wall_visible_until_x_m:.3f} m",
            f"Trayectoria apunta a x={trajectory.target_x_m:.3f} m, y={trajectory.target_y_m:.3f} m",
            f"Distancia radial al inicio: {candidate.start_radial_m:.3f} m",
            f"Ancho recto base: {candidate.straight_width_m:.3f} m",
            f"Ancho ventana: {candidate.window_width_m:.3f} m" if candidate.gap_only_opening else f"Ancho visible en la entrada: {candidate.entry_width_m:.3f} m",
            f"Apertura lateral de curva: {candidate.curve_wall_shift_m:.3f} m",
            f"Ganancia de ancho en curva: {candidate.opening_width_gain_m:.3f} m",
            f"Gap misma pared: {candidate.same_side_gap_m:.3f} m",
            f"Cierre frontal interior: {candidate.front_closure_point_count} pts",
            f"Modo apertura: {'ventana sin puntos' if candidate.gap_only_opening else 'borde detectado'}",
            f"Sector angular visible: {candidate.angle_start_deg:.1f}..{candidate.angle_end_deg:.1f} deg",
            f"Puntos usados en curva: {candidate.curve_point_count}",
        ]

    ax.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.92, "edgecolor": "#666666"},
    )

    ax.set_xlim(-axis_limit_m, axis_limit_m)
    ax.set_ylim(-axis_limit_m, axis_limit_m)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x [m] (frente+)")
    ax.set_ylabel("y [m] (derecha+, izquierda-)")
    ax.set_title("Analisis de curva visible desde snapshot LiDAR")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="",
        help="Snapshot CSV generated by lidar_subscriber_node.py",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path_flag",
        default="",
        help="Snapshot CSV generated by lidar_subscriber_node.py",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Base path for output files (.png and .json). Defaults next to the CSV.",
    )
    parser.add_argument("--x-bin-m", type=float, default=0.05, help="Bin width along x")
    parser.add_argument("--fit-x-min-m", type=float, default=-0.75, help="Start x of straight fit")
    parser.add_argument("--fit-x-max-m", type=float, default=0.05, help="End x of straight fit")
    parser.add_argument(
        "--search-x-min-m",
        type=float,
        default=0.15,
        help="Start x for curve search after the straight segment",
    )
    parser.add_argument(
        "--deviation-threshold-m",
        type=float,
        default=0.12,
        help="Minimum side-wall deviation required to tag a visible curve",
    )
    parser.add_argument(
        "--min-points-per-bin",
        type=int,
        default=2,
        help="Minimum points per x-bin on one side",
    )
    parser.add_argument(
        "--min-curve-bins",
        type=int,
        default=2,
        help="Minimum consecutive bins to accept a curve candidate",
    )
    parser.add_argument(
        "--gap-threshold-m",
        type=float,
        default=0.11,
        help="Minimum x gap in one side-wall profile to mark a possible opening",
    )
    parser.add_argument(
        "--opposite-continuation-min-m",
        type=float,
        default=0.10,
        help="Minimum extra visible length required on the opposite wall after the entry point",
    )
    parser.add_argument(
        "--front-closure-x-window-m",
        type=float,
        default=0.12,
        help="Half-window in x used to reject entries that actually look like a frontal closing wall",
    )
    parser.add_argument(
        "--front-closure-min-points",
        type=int,
        default=6,
        help="Minimum interior points near the mouth to reject the candidate as front closure",
    )
    args = parser.parse_args()
    if not args.csv_path and args.csv_path_flag:
        args.csv_path = args.csv_path_flag
    if not args.csv_path:
        parser.error("csv_path is required")
    return args


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()
    angles_deg, ranges_m, counts = _load_snapshot_csv(csv_path)

    x_m, y_m = _display_xy_from_snapshot(angles_deg, ranges_m)

    left_profile = _build_side_profile(
        side_name="left",
        side_sign=-1,
        x_m=x_m,
        y_m=y_m,
        x_min_m=float(np.min(x_m)),
        x_max_m=float(np.max(x_m)) + float(args.x_bin_m),
        x_bin_m=float(args.x_bin_m),
        fit_x_min_m=float(args.fit_x_min_m),
        fit_x_max_m=float(args.fit_x_max_m),
        min_points_per_bin=int(args.min_points_per_bin),
    )
    right_profile = _build_side_profile(
        side_name="right",
        side_sign=1,
        x_m=x_m,
        y_m=y_m,
        x_min_m=float(np.min(x_m)),
        x_max_m=float(np.max(x_m)) + float(args.x_bin_m),
        x_bin_m=float(args.x_bin_m),
        fit_x_min_m=float(args.fit_x_min_m),
        fit_x_max_m=float(args.fit_x_max_m),
        min_points_per_bin=int(args.min_points_per_bin),
    )

    left_candidate = _detect_curve_candidate(
        profile=left_profile,
        opposite_profile=right_profile,
        angles_deg=angles_deg,
        points_x_m=x_m,
        points_y_m=y_m,
        search_x_min_m=float(args.search_x_min_m),
        deviation_threshold_m=float(args.deviation_threshold_m),
        min_curve_bins=int(args.min_curve_bins),
        x_bin_m=float(args.x_bin_m),
        gap_threshold_m=float(args.gap_threshold_m),
        opposite_continuation_min_m=float(args.opposite_continuation_min_m),
        front_closure_x_window_m=float(args.front_closure_x_window_m),
        front_closure_min_points=int(args.front_closure_min_points),
    )
    right_candidate = _detect_curve_candidate(
        profile=right_profile,
        opposite_profile=left_profile,
        angles_deg=angles_deg,
        points_x_m=x_m,
        points_y_m=y_m,
        search_x_min_m=float(args.search_x_min_m),
        deviation_threshold_m=float(args.deviation_threshold_m),
        min_curve_bins=int(args.min_curve_bins),
        x_bin_m=float(args.x_bin_m),
        gap_threshold_m=float(args.gap_threshold_m),
        opposite_continuation_min_m=float(args.opposite_continuation_min_m),
        front_closure_x_window_m=float(args.front_closure_x_window_m),
        front_closure_min_points=int(args.front_closure_min_points),
    )

    candidates = [candidate for candidate in (left_candidate, right_candidate) if candidate is not None]
    candidate = None
    if candidates:
        candidate = max(
            candidates,
            key=lambda item: (item.score, item.curve_wall_shift_m, item.curve_point_count),
        )

    if args.output_prefix:
        output_prefix = Path(args.output_prefix).expanduser().resolve()
    else:
        output_prefix = csv_path.with_suffix("")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_png = output_prefix.with_name(output_prefix.name + "_curve_analysis").with_suffix(".png")
    output_json = output_prefix.with_name(output_prefix.name + "_curve_analysis").with_suffix(".json")

    axis_limit_m = max(
        1.0,
        float(np.nanpercentile(ranges_m, 95)) * 1.20,
    )
    trajectory: TrajectoryPlan | None = None
    if candidate is not None:
        trajectory = _build_trajectory_plan(
            candidate=candidate,
            left_profile=left_profile,
            right_profile=right_profile,
            axis_limit_m=axis_limit_m,
        )
    _plot_analysis(
        output_path=output_png,
        x_m=x_m,
        y_m=y_m,
        left_profile=left_profile,
        right_profile=right_profile,
        candidate=candidate,
        axis_limit_m=axis_limit_m,
    )

    result = {
        "csv_path": str(csv_path),
        "output_png": str(output_png),
        "output_json": str(output_json),
        "point_count": int(ranges_m.size),
        "curve_candidate": None,
    }
    if candidate is not None:
        result["curve_candidate"] = {
            "side": candidate.side_name,
            "side_label_es": _side_label(candidate.side_name),
            "entry_x_m": candidate.entry_x_m,
            "entry_y_m": candidate.entry_y_m,
            "start_forward_m": candidate.start_forward_m,
            "start_radial_m": candidate.start_radial_m,
            "straight_width_m": candidate.straight_width_m,
            "entry_width_m": candidate.entry_width_m,
            "curve_wall_shift_m": candidate.curve_wall_shift_m,
            "curve_lateral_opening_m": candidate.curve_wall_shift_m,
            "straight_end_x_m": candidate.straight_end_x_m,
            "opposite_wall_visible_until_x_m": candidate.opposite_wall_visible_until_x_m,
            "window_far_wall_x_m": candidate.window_far_wall_x_m,
            "window_width_m": candidate.window_width_m,
            "same_side_gap_m": candidate.same_side_gap_m,
            "curve_width_m": candidate.curve_width_m,
            "opening_width_gain_m": candidate.opening_width_gain_m,
            "front_closure_point_count": candidate.front_closure_point_count,
            "front_closure_y_span_m": candidate.front_closure_y_span_m,
            "heuristic_name": candidate.heuristic_name,
            "gap_only_opening": candidate.gap_only_opening,
            "score": candidate.score,
            "angle_start_deg": candidate.angle_start_deg,
            "angle_end_deg": candidate.angle_end_deg,
            "angle_center_deg": candidate.angle_center_deg,
            "curve_point_count": candidate.curve_point_count,
            "first_curve_point_x_m": candidate.first_curve_x_m,
            "first_curve_point_y_m": candidate.first_curve_y_m,
        }
        if trajectory is not None:
            result["curve_candidate"]["trajectory"] = {
                "entry_center_y_m": trajectory.entry_center_y_m,
                "target_x_m": trajectory.target_x_m,
                "target_y_m": trajectory.target_y_m,
                "anchor_points_xy_m": [
                    [float(x_point), float(y_point)]
                    for x_point, y_point in trajectory.anchor_points
                ],
                "path_xy_m": [
                    [float(x_point), float(y_point)]
                    for x_point, y_point in zip(trajectory.x_m.tolist(), trajectory.y_m.tolist())
                ],
            }

    output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[APEX] Saved curve analysis image: {output_png}")
    print(f"[APEX] Saved curve analysis metrics: {output_json}")
    if candidate is None:
        print("[APEX] No dominant visible curve candidate was detected.")
    else:
        side_label = _side_label(candidate.side_name)
        print(
            "[APEX] Curve candidate: side=%s start_forward=%.3fm entry_width=%.3fm shift=%.3fm gap=%.3fm sector=%.1f..%.1f deg heuristic=%s"
            % (
                side_label,
                candidate.start_forward_m,
                candidate.entry_width_m,
                candidate.curve_wall_shift_m,
                candidate.same_side_gap_m,
                candidate.angle_start_deg,
                candidate.angle_end_deg,
                candidate.heuristic_name,
            )
        )


if __name__ == "__main__":
    main()
