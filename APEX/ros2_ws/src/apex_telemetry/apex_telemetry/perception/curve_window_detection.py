#!/usr/bin/env python3
"""Shared curve-window detection and first-entry trajectory planning."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CurveWindowDetectionConfig:
    x_bin_m: float = 0.05
    fit_x_min_m: float = -0.75
    fit_x_max_m: float = 0.05
    search_x_min_m: float = 0.15
    deviation_threshold_m: float = 0.12
    min_points_per_bin: int = 2
    min_curve_bins: int = 2
    gap_threshold_m: float = 0.11
    opposite_continuation_min_m: float = 0.10
    front_closure_x_window_m: float = 0.12
    front_closure_min_points: int = 6


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
class CurveWindowCandidate:
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
class CurveWindowTrajectory:
    x_m: np.ndarray
    y_m: np.ndarray
    anchor_points: list[tuple[float, float]]
    entry_center_y_m: float
    target_x_m: float
    target_y_m: float


@dataclass
class CurveWindowDetectionResult:
    points_x_m: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float64))
    points_y_m: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float64))
    angles_deg: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float64))
    left_profile: SideProfile | None = None
    right_profile: SideProfile | None = None
    candidate: CurveWindowCandidate | None = None
    trajectory: CurveWindowTrajectory | None = None
    axis_limit_m: float = 1.0
    config: CurveWindowDetectionConfig = field(default_factory=CurveWindowDetectionConfig)

    @property
    def valid(self) -> bool:
        return self.candidate is not None and self.trajectory is not None


def _normalize_angle_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def _side_label(side_name: str) -> str:
    if side_name == "left":
        return "izquierda"
    if side_name == "right":
        return "derecha"
    return side_name


def scan_ranges_to_forward_left_xy(scan_ranges_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranges = np.asarray(scan_ranges_m, dtype=np.float64).copy()
    if ranges.size == 0:
        return (
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    ranges[~np.isfinite(ranges)] = 0.0
    ranges[ranges <= 0.0] = 0.0
    raw_angles_deg = np.arange(ranges.size, dtype=np.float64)
    angles_deg = np.vectorize(_normalize_angle_deg)(raw_angles_deg)
    valid_mask = ranges > 0.0
    valid_angles_deg = angles_deg[valid_mask]
    valid_ranges = ranges[valid_mask]
    valid_angles_rad = np.radians(valid_angles_deg)
    # Runtime LaserScan bins follow the same convention used by the PC snapshot
    # export: forward points lie around +/-180 deg, not around 0 deg.
    points_x_m = -valid_ranges * np.cos(valid_angles_rad)
    points_y_m = valid_ranges * np.sin(valid_angles_rad)
    return points_x_m, points_y_m, valid_angles_deg


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


def _build_side_profile(
    *,
    side_name: str,
    side_sign: int,
    points_x_m: np.ndarray,
    points_y_m: np.ndarray,
    config: CurveWindowDetectionConfig,
) -> SideProfile:
    xs: list[float] = []
    ys: list[float] = []
    counts: list[int] = []

    x_min_m = float(np.min(points_x_m))
    x_max_m = float(np.max(points_x_m)) + float(config.x_bin_m)
    for x0_m in np.arange(x_min_m, x_max_m, config.x_bin_m):
        mask = (points_x_m >= x0_m) & (points_x_m < (x0_m + config.x_bin_m))
        mask &= (points_y_m * float(side_sign)) > 0.0
        if int(np.count_nonzero(mask)) < config.min_points_per_bin:
            continue
        xs.append(float(x0_m + 0.5 * config.x_bin_m))
        ys.append(float(np.median(points_y_m[mask])))
        counts.append(int(np.count_nonzero(mask)))

    if len(xs) < 4:
        raise ValueError(f"Not enough profile points for side '{side_name}'")

    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    count_arr = np.asarray(counts, dtype=np.int32)

    fit_mask = (x_arr >= config.fit_x_min_m) & (x_arr <= config.fit_x_max_m)
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
    config: CurveWindowDetectionConfig,
) -> CurveWindowCandidate | None:
    forward_indices = np.flatnonzero(profile.x_m >= config.search_x_min_m)
    if forward_indices.size < 3:
        return None

    candidate_indices = forward_indices[
        profile.deviation_m[forward_indices] >= config.deviation_threshold_m
    ]
    valid_clusters = [
        cluster
        for cluster in _cluster_profile_indices(
            profile,
            candidate_indices,
            max_dx_m=(config.gap_threshold_m + 0.01),
        )
        if cluster.size >= config.min_curve_bins
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
        search_x_min_m=config.search_x_min_m,
        x_bin_m=config.x_bin_m,
    )
    if side_gap is not None:
        gap_prev_x_m, gap_next_x_m, same_side_gap_m, gap_far_wall_x_m = side_gap
        gap_prev_idx = int(np.argmin(np.abs(profile.x_m - gap_prev_x_m)))
        next_mask = np.flatnonzero(profile.x_m >= gap_next_x_m)
        gap_next_idx = int(next_mask[0]) if next_mask.size else gap_prev_idx
        strong_gap_opening = same_side_gap_m >= max(0.30, 3.0 * config.x_bin_m)

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
            & (profile.deviation_m[forward_indices] < (0.5 * config.deviation_threshold_m))
        ]
        cluster_last_straight_idx = (
            int(straight_before_cluster[-1]) if straight_before_cluster.size else cluster_start_idx
        )
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
    if heuristic_name.startswith("gap+") and same_side_gap_m >= max(0.30, 3.0 * config.x_bin_m):
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
    if entry_x_m <= 0.0:
        return None

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
            max_abs_deviation_m=max(0.06, 0.6 * config.deviation_threshold_m),
            max_dx_m=max(0.12, config.gap_threshold_m + 0.03),
        )
        if opposite_wall_visible_until_x_m is None:
            return None
    if (opposite_wall_visible_until_x_m - entry_x_m) < config.opposite_continuation_min_m:
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
        x_window_m=config.front_closure_x_window_m,
    )
    if (
        front_closure_point_count >= config.front_closure_min_points
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
                np.polyval(profile.fit_coef, points_x_m) - 0.5 * config.deviation_threshold_m
            )
        else:
            curve_mask &= points_y_m >= (
                np.polyval(profile.fit_coef, points_x_m) + 0.5 * config.deviation_threshold_m
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

    return CurveWindowCandidate(
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


def _bezier_curve(
    points_xy: list[tuple[float, float]],
    num_samples: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
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
    candidate: CurveWindowCandidate,
    left_profile: SideProfile,
    right_profile: SideProfile,
    axis_limit_m: float,
) -> CurveWindowTrajectory:
    same_profile = left_profile if candidate.side_name == "left" else right_profile
    opposite_profile = right_profile if candidate.side_name == "left" else left_profile

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
    return CurveWindowTrajectory(
        x_m=traj_x_m,
        y_m=traj_y_m,
        anchor_points=control_points,
        entry_center_y_m=entry_center_y_m,
        target_x_m=target_x_m,
        target_y_m=target_y_m,
    )


def detect_curve_window_points(
    points_x_m: np.ndarray,
    points_y_m: np.ndarray,
    *,
    angles_deg: np.ndarray | None = None,
    config: CurveWindowDetectionConfig | None = None,
) -> CurveWindowDetectionResult:
    config = config or CurveWindowDetectionConfig()
    x_arr = np.asarray(points_x_m, dtype=np.float64)
    y_arr = np.asarray(points_y_m, dtype=np.float64)
    if x_arr.size == 0 or y_arr.size == 0 or x_arr.size != y_arr.size:
        return CurveWindowDetectionResult(config=config)

    finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite_mask]
    y_arr = y_arr[finite_mask]
    if x_arr.size == 0:
        return CurveWindowDetectionResult(config=config)

    if angles_deg is None:
        angles_arr = np.degrees(np.arctan2(y_arr, x_arr))
    else:
        angles_arr = np.asarray(angles_deg, dtype=np.float64)[finite_mask]
    axis_limit_m = max(1.0, float(np.nanpercentile(np.hypot(x_arr, y_arr), 95)) * 1.20)

    try:
        left_profile = _build_side_profile(
            side_name="left",
            side_sign=1,
            points_x_m=x_arr,
            points_y_m=y_arr,
            config=config,
        )
        right_profile = _build_side_profile(
            side_name="right",
            side_sign=-1,
            points_x_m=x_arr,
            points_y_m=y_arr,
            config=config,
        )
    except ValueError:
        return CurveWindowDetectionResult(
            points_x_m=x_arr,
            points_y_m=y_arr,
            angles_deg=angles_arr,
            axis_limit_m=axis_limit_m,
            config=config,
        )

    left_candidate = _detect_curve_candidate(
        profile=left_profile,
        opposite_profile=right_profile,
        angles_deg=angles_arr,
        points_x_m=x_arr,
        points_y_m=y_arr,
        config=config,
    )
    right_candidate = _detect_curve_candidate(
        profile=right_profile,
        opposite_profile=left_profile,
        angles_deg=angles_arr,
        points_x_m=x_arr,
        points_y_m=y_arr,
        config=config,
    )
    candidates = [
        candidate
        for candidate in (left_candidate, right_candidate)
        if candidate is not None and candidate.gap_only_opening
    ]
    candidate = None
    if candidates:
        candidate = max(
            candidates,
            key=lambda item: (item.score, item.curve_wall_shift_m, item.curve_point_count),
        )
    trajectory = None
    if candidate is not None:
        trajectory = _build_trajectory_plan(
            candidate=candidate,
            left_profile=left_profile,
            right_profile=right_profile,
            axis_limit_m=axis_limit_m,
        )

    return CurveWindowDetectionResult(
        points_x_m=x_arr,
        points_y_m=y_arr,
        angles_deg=angles_arr,
        left_profile=left_profile,
        right_profile=right_profile,
        candidate=candidate,
        trajectory=trajectory,
        axis_limit_m=axis_limit_m,
        config=config,
    )


def detection_result_to_dict(result: CurveWindowDetectionResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "curve_window_valid": bool(result.valid),
        "curve_window_side": "none",
        "curve_window_entry_x_m": 0.0,
        "curve_window_entry_y_m": 0.0,
        "curve_window_width_m": 0.0,
        "curve_window_far_wall_x_m": 0.0,
        "curve_window_target_x_m": 0.0,
        "curve_window_target_y_m": 0.0,
        "curve_window_score": 0.0,
        "curve_window_gap_only_opening": False,
        "curve_window_heuristic_name": "none",
        "curve_window_same_side_gap_m": 0.0,
        "curve_window_opposite_wall_visible_until_x_m": 0.0,
        "curve_window_straight_end_x_m": 0.0,
    }
    if result.candidate is None:
        return payload

    candidate = result.candidate
    payload.update(
        {
            "curve_window_side": candidate.side_name,
            "curve_window_entry_x_m": float(candidate.entry_x_m),
            "curve_window_entry_y_m": float(candidate.entry_y_m),
            "curve_window_width_m": float(candidate.window_width_m),
            "curve_window_far_wall_x_m": float(candidate.window_far_wall_x_m),
            "curve_window_score": float(candidate.score),
            "curve_window_gap_only_opening": bool(candidate.gap_only_opening),
            "curve_window_heuristic_name": candidate.heuristic_name,
            "curve_window_same_side_gap_m": float(candidate.same_side_gap_m),
            "curve_window_opposite_wall_visible_until_x_m": float(candidate.opposite_wall_visible_until_x_m),
            "curve_window_straight_end_x_m": float(candidate.straight_end_x_m),
        }
    )
    if result.trajectory is not None:
        payload.update(
            {
                "curve_window_target_x_m": float(result.trajectory.target_x_m),
                "curve_window_target_y_m": float(result.trajectory.target_y_m),
            }
        )
    return payload


def detection_result_to_curve_diag(
    result: CurveWindowDetectionResult,
    *,
    probe_subphase: str,
    probe_locked: bool,
    probe_goal_distance_m: float,
    probe_path_progress: float,
) -> dict[str, Any]:
    payload = detection_result_to_dict(result)
    payload.update(
        {
            "probe_subphase": probe_subphase,
            "probe_locked": bool(probe_locked),
            "probe_goal_distance_m": float(probe_goal_distance_m),
            "probe_path_progress": float(probe_path_progress),
            "curve_window_points_xy_m": [
                [float(x_m), float(y_m)]
                for x_m, y_m in zip(result.points_x_m.tolist(), result.points_y_m.tolist())
            ],
        }
    )
    if result.left_profile is not None:
        payload["curve_window_left_profile_xy_m"] = [
            [float(x_m), float(y_m)]
            for x_m, y_m in zip(result.left_profile.x_m.tolist(), result.left_profile.y_m.tolist())
        ]
        payload["curve_window_left_fit_xy_m"] = [
            [float(x_m), float(y_m)]
            for x_m, y_m in zip(result.left_profile.x_m.tolist(), result.left_profile.fit_y_m.tolist())
        ]
    if result.right_profile is not None:
        payload["curve_window_right_profile_xy_m"] = [
            [float(x_m), float(y_m)]
            for x_m, y_m in zip(result.right_profile.x_m.tolist(), result.right_profile.y_m.tolist())
        ]
        payload["curve_window_right_fit_xy_m"] = [
            [float(x_m), float(y_m)]
            for x_m, y_m in zip(result.right_profile.x_m.tolist(), result.right_profile.fit_y_m.tolist())
        ]
    if result.trajectory is not None:
        payload["curve_window_anchor_points_xy_m"] = [
            [float(x_m), float(y_m)]
            for x_m, y_m in result.trajectory.anchor_points
        ]
        payload["curve_window_path_xy_m"] = [
            [float(x_m), float(y_m)]
            for x_m, y_m in zip(result.trajectory.x_m.tolist(), result.trajectory.y_m.tolist())
        ]
    return payload


def curve_window_result_summary(result: CurveWindowDetectionResult) -> dict[str, Any]:
    candidate = result.candidate
    trajectory = result.trajectory
    payload: dict[str, Any] = {
        "valid": bool(result.valid),
        "side": "none",
        "side_label_es": _side_label("none"),
        "entry_x_m": 0.0,
        "entry_y_m": 0.0,
        "straight_end_x_m": 0.0,
        "window_far_wall_x_m": 0.0,
        "window_width_m": 0.0,
        "same_side_gap_m": 0.0,
        "opposite_wall_visible_until_x_m": 0.0,
        "score": 0.0,
        "target_x_m": 0.0,
        "target_y_m": 0.0,
        "anchor_points_xy_m": [],
        "path_xy_m": [],
    }
    if candidate is None:
        return payload

    payload.update(
        {
            "side": candidate.side_name,
            "side_label_es": _side_label(candidate.side_name),
            "entry_x_m": float(candidate.entry_x_m),
            "entry_y_m": float(candidate.entry_y_m),
            "straight_end_x_m": float(candidate.straight_end_x_m),
            "window_far_wall_x_m": float(candidate.window_far_wall_x_m),
            "window_width_m": float(candidate.window_width_m),
            "same_side_gap_m": float(candidate.same_side_gap_m),
            "opposite_wall_visible_until_x_m": float(candidate.opposite_wall_visible_until_x_m),
            "score": float(candidate.score),
            "gap_only_opening": bool(candidate.gap_only_opening),
            "heuristic_name": candidate.heuristic_name,
        }
    )
    if trajectory is not None:
        payload.update(
            {
                "target_x_m": float(trajectory.target_x_m),
                "target_y_m": float(trajectory.target_y_m),
                "anchor_points_xy_m": [
                    [float(x_m), float(y_m)] for x_m, y_m in trajectory.anchor_points
                ],
                "path_xy_m": [
                    [float(x_m), float(y_m)]
                    for x_m, y_m in zip(trajectory.x_m.tolist(), trajectory.y_m.tolist())
                ],
            }
        )
    return payload
