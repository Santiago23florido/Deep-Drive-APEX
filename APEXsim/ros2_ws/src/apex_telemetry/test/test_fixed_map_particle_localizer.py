import csv
import math
from pathlib import Path

import numpy as np
import yaml

from apex_telemetry.estimation.fixed_map_localizer_core import (
    FixedMapParameters,
    FixedMapParticleLocalizer,
)
from apex_telemetry.estimation.planar_fusion_core import LidarScanObservation


def _rotation(theta_rad: float) -> np.ndarray:
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)
    return np.asarray([[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=np.float64)


def _write_map_assets(
    tmp_path: Path,
    points_xy: np.ndarray,
    route_poses_xyyaw: np.ndarray,
    *,
    initial_pose: tuple[float, float, float],
    origin_xy: tuple[float, float] = (-1.0, -1.0),
    resolution_m: float = 0.05,
    width: int = 140,
    height: int = 95,
) -> Path:
    origin = np.asarray(origin_xy, dtype=np.float64)
    yy, xx = np.indices((height, width))
    centers_xy = np.stack(
        [
            origin[0] + ((xx + 0.5) * resolution_m),
            origin[1] + ((yy + 0.5) * resolution_m),
        ],
        axis=-1,
    ).reshape(-1, 2)
    distance_field = np.linalg.norm(
        centers_xy[:, None, :] - points_xy[None, :, :],
        axis=2,
    ).min(axis=1).reshape(height, width).astype(np.float32)
    np.save(tmp_path / "fixed_map_distance.npy", distance_field)

    with (tmp_path / "fixed_map_visual_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_m", "y_m"])
        writer.writerows(points_xy.tolist())

    with (tmp_path / "fixed_route_path.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_m", "y_m", "yaw_rad"])
        writer.writerows(route_poses_xyyaw.tolist())

    map_yaml = tmp_path / "fixed_map.yaml"
    map_yaml.write_text(
        yaml.safe_dump(
            {
                "resolution": float(resolution_m),
                "origin": [float(origin[0]), float(origin[1]), 0.0],
                "distance_field_npy": "fixed_map_distance.npy",
                "visual_points_csv": "fixed_map_visual_points.csv",
                "optimized_keyframes_csv": "fixed_route_path.csv",
                "initial_pose": [float(value) for value in initial_pose],
            },
        ),
        encoding="utf-8",
    )
    return map_yaml


def _asymmetric_map(tmp_path: Path, *, initial_pose: tuple[float, float, float]) -> tuple[Path, np.ndarray]:
    points: list[tuple[float, float]] = []
    for x_m in np.linspace(0.0, 4.0, 121):
        points.append((float(x_m), 0.0))
        points.append((float(x_m), float(2.2 + (0.15 * math.sin(2.0 * x_m)))))
    for y_m in np.linspace(0.0, 2.2, 80):
        points.append((0.0, float(y_m)))
        points.append((4.0, float(y_m)))
    for y_m in np.linspace(0.3, 1.5, 50):
        points.append((2.4, float(y_m)))
    points_xy = np.asarray(points, dtype=np.float64)
    route = np.asarray([[float(x_m), 1.0, 0.0] for x_m in np.linspace(0.5, 3.5, 100)], dtype=np.float64)
    return _write_map_assets(tmp_path, points_xy, route, initial_pose=initial_pose), points_xy


def _ambiguous_corridor_map(tmp_path: Path) -> tuple[Path, np.ndarray]:
    points: list[tuple[float, float]] = []
    for x_m in np.linspace(0.0, 5.0, 180):
        points.append((float(x_m), 0.0))
        points.append((float(x_m), 1.5))
    points_xy = np.asarray(points, dtype=np.float64)
    route = np.asarray([[float(x_m), 0.75, 0.0] for x_m in np.linspace(0.3, 4.7, 100)], dtype=np.float64)
    return _write_map_assets(
        tmp_path,
        points_xy,
        route,
        initial_pose=(2.2, 0.75, 0.0),
        width=150,
        height=85,
    ), points_xy


def _scan_points_from_pose(
    map_points_xy: np.ndarray,
    pose_xyyaw: tuple[float, float, float],
    *,
    max_range_m: float = 2.5,
    max_points: int = 160,
    top_wall_only: bool = False,
    noise_std_m: float = 0.0,
) -> np.ndarray:
    pose_xy = np.asarray(pose_xyyaw[:2], dtype=np.float64)
    local_points = (map_points_xy - pose_xy.reshape(1, 2)) @ _rotation(float(pose_xyyaw[2]))
    ranges_m = np.linalg.norm(local_points, axis=1)
    mask = ranges_m < max_range_m
    if top_wall_only:
        mask &= map_points_xy[:, 1] > 1.0
    selected = local_points[mask]
    if selected.shape[0] > max_points:
        indexes = np.linspace(0, selected.shape[0] - 1, max_points, dtype=np.int32)
        selected = selected[indexes]
    if noise_std_m > 0.0:
        selected = selected + np.random.default_rng(3).normal(0.0, noise_std_m, size=selected.shape)
    return selected.astype(np.float64, copy=False)


def _observation(scan_index: int, t_s: float, points_local: np.ndarray) -> LidarScanObservation:
    return LidarScanObservation(
        scan_index=scan_index,
        stamp_sec=int(t_s),
        stamp_nanosec=int((t_s - int(t_s)) * 1.0e9),
        t_s=float(t_s),
        points_local=points_local,
        sampled_points_local=points_local,
        lower_wall_points_local=np.empty((0, 2), dtype=np.float64),
        upper_wall_points_local=np.empty((0, 2), dtype=np.float64),
    )


def _initialize_imu(localizer: FixedMapParticleLocalizer) -> None:
    for t_s in np.linspace(0.0, 1.8, 20):
        localizer.add_imu_sample(
            t_s=float(t_s),
            ax_mps2=0.0,
            ay_mps2=0.0,
            az_mps2=9.8,
            gz_rps=0.0,
        )


def _make_localizer(map_yaml: Path, **overrides: object) -> FixedMapParticleLocalizer:
    params = FixedMapParameters(
        fixed_map_yaml=str(map_yaml),
        particle_count=int(overrides.pop("particle_count", 512)),
        particle_seed=int(overrides.pop("particle_seed", 4)),
        particle_initial_xy_std_m=float(overrides.pop("particle_initial_xy_std_m", 0.45)),
        particle_initial_yaw_std_rad=float(overrides.pop("particle_initial_yaw_std_rad", 0.45)),
        particle_route_seed_fraction=float(overrides.pop("particle_route_seed_fraction", 0.08)),
        particle_max_high_confidence_spread_m=float(overrides.pop("particle_max_high_confidence_spread_m", 0.55)),
        particle_max_medium_confidence_spread_m=float(overrides.pop("particle_max_medium_confidence_spread_m", 0.90)),
        **overrides,
    )
    localizer = FixedMapParticleLocalizer(params)
    _initialize_imu(localizer)
    return localizer


def test_particle_localizer_recovers_from_moderate_initial_pose_error(tmp_path: Path) -> None:
    true_pose = (1.4, 0.85, 0.15)
    map_yaml, map_points = _asymmetric_map(tmp_path, initial_pose=(1.0, 0.5, 0.0))
    localizer = _make_localizer(map_yaml)
    scan_points = _scan_points_from_pose(map_points, true_pose, max_points=180, noise_std_m=0.005)

    estimate = None
    for index in range(6):
        estimate = localizer.add_scan_observation(_observation(index, 2.0 + (0.12 * index), scan_points))[0]

    assert estimate is not None
    assert math.hypot(estimate.x_m - true_pose[0], estimate.y_m - true_pose[1]) < 0.12
    assert abs(math.atan2(math.sin(estimate.yaw_rad - true_pose[2]), math.cos(estimate.yaw_rad - true_pose[2]))) < 0.08
    assert estimate.confidence == "high"
    status = localizer.status_snapshot()
    assert status.parameters["localizer"] == "particle_filter"
    assert status.quality["particle_confidence_score"] > 0.70


def test_particle_localizer_keeps_multiple_hypotheses_for_partial_geometry(tmp_path: Path) -> None:
    true_pose = (3.1, 0.75, 0.0)
    map_yaml, map_points = _ambiguous_corridor_map(tmp_path)
    localizer = _make_localizer(
        map_yaml,
        particle_seed=9,
        particle_initial_xy_std_m=0.80,
        particle_initial_yaw_std_rad=0.20,
        particle_route_seed_fraction=0.70,
    )
    scan_points = _scan_points_from_pose(
        map_points,
        true_pose,
        max_range_m=1.4,
        max_points=70,
        top_wall_only=True,
    )

    estimate = None
    for index in range(3):
        estimate = localizer.add_scan_observation(_observation(index, 2.0 + (0.15 * index), scan_points))[0]

    assert estimate is not None
    quality = localizer.status_snapshot().quality
    assert quality["particle_best_inlier_ratio"] > 0.80
    assert quality["particle_xy_spread_m"] > 0.90
    assert estimate.confidence != "high"


def test_particle_localizer_rejects_degraded_lidar_support(tmp_path: Path) -> None:
    true_pose = (3.1, 0.75, 0.0)
    map_yaml, map_points = _ambiguous_corridor_map(tmp_path)
    localizer = _make_localizer(
        map_yaml,
        particle_seed=9,
        particle_initial_xy_std_m=0.80,
        particle_initial_yaw_std_rad=0.20,
        particle_route_seed_fraction=0.70,
    )
    degraded_points = _scan_points_from_pose(
        map_points,
        true_pose,
        max_range_m=1.4,
        max_points=70,
        top_wall_only=True,
    )[:6]

    estimate = localizer.add_scan_observation(_observation(0, 2.0, degraded_points))[0]
    quality = localizer.status_snapshot().quality

    assert estimate.confidence == "low"
    assert quality["particle_observation_weight"] == 0.0
    assert math.hypot(estimate.x_m - 2.2, estimate.y_m - 0.75) < 0.05
    assert quality["particle_xy_spread_m"] > 0.50


def test_particle_localizer_recovers_after_moderate_pose_deviation(tmp_path: Path) -> None:
    first_pose = (1.4, 0.85, 0.15)
    deviated_pose = (1.9, 1.0, 0.12)
    map_yaml, map_points = _asymmetric_map(tmp_path, initial_pose=(1.0, 0.5, 0.0))
    localizer = _make_localizer(
        map_yaml,
        particle_seed=8,
        particle_process_xy_std_m=0.16,
        particle_random_injection_ratio=0.04,
        particle_route_seed_fraction=0.10,
    )

    first_scan = _scan_points_from_pose(map_points, first_pose, max_points=180)
    for index in range(4):
        localizer.add_scan_observation(_observation(index, 2.0 + (0.15 * index), first_scan))
    before_deviation = localizer.status_snapshot().latest_pose
    assert before_deviation is not None
    initial_deviation_error = math.hypot(
        float(before_deviation["x_m"]) - deviated_pose[0],
        float(before_deviation["y_m"]) - deviated_pose[1],
    )

    deviated_scan = _scan_points_from_pose(map_points, deviated_pose, max_points=180)
    recovery_errors: list[float] = []
    estimate = None
    for index in range(8):
        estimate = localizer.add_scan_observation(_observation(index + 10, 3.0 + (0.15 * index), deviated_scan))[0]
        recovery_errors.append(math.hypot(estimate.x_m - deviated_pose[0], estimate.y_m - deviated_pose[1]))

    assert estimate is not None
    assert min(recovery_errors) < 0.12
    assert recovery_errors[-1] < 0.15
    assert recovery_errors[-1] < 0.5 * initial_deviation_error
    assert estimate.confidence in {"medium", "high"}
