#!/usr/bin/env python3
"""Build a fixed-map package from an offline refined live-map directory."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import binary_closing, binary_dilation, distance_transform_edt


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _read_points_csv(path: Path, *, require_yaw: bool) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                if require_yaw:
                    rows.append([float(row["x_m"]), float(row["y_m"]), float(row["yaw_rad"])])
                else:
                    rows.append([float(row["x_m"]), float(row["y_m"])])
            except Exception:
                continue
    width = 3 if require_yaw else 2
    if not rows:
        return np.empty((0, width), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _pose_from_json(payload: dict[str, object], key: str) -> np.ndarray:
    pose = payload.get(key)
    if not isinstance(pose, dict):
        raise RuntimeError(f"poses.json is missing {key}")
    return np.asarray(
        [
            float(pose["x_m"]),
            float(pose["y_m"]),
            float(pose["yaw_rad"]),
        ],
        dtype=np.float64,
    )


def _grid_geometry(
    points_xy: np.ndarray,
    *,
    resolution_m: float,
    margin_m: float,
    min_extent_cells: int = 48,
) -> tuple[int, int, np.ndarray]:
    min_xy = np.min(points_xy, axis=0) - float(margin_m)
    max_xy = np.max(points_xy, axis=0) + float(margin_m)
    center_xy = 0.5 * (min_xy + max_xy)
    span_xy = np.maximum(max_xy - min_xy, float(min_extent_cells) * float(resolution_m))
    width = max(min_extent_cells, int(math.ceil(float(span_xy[0]) / float(resolution_m))) + 1)
    height = max(min_extent_cells, int(math.ceil(float(span_xy[1]) / float(resolution_m))) + 1)
    origin_xy = np.asarray(
        [
            float(center_xy[0]) - (0.5 * float(width) * float(resolution_m)),
            float(center_xy[1]) - (0.5 * float(height) * float(resolution_m)),
        ],
        dtype=np.float64,
    )
    return width, height, origin_xy


def _points_to_cells(
    points_xy: np.ndarray,
    *,
    origin_xy: np.ndarray,
    resolution_m: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell_x = np.floor((points_xy[:, 0] - float(origin_xy[0])) / float(resolution_m)).astype(np.int32)
    cell_y = np.floor((points_xy[:, 1] - float(origin_xy[1])) / float(resolution_m)).astype(np.int32)
    in_bounds = (cell_x >= 0) & (cell_x < width) & (cell_y >= 0) & (cell_y < height)
    return cell_y[in_bounds], cell_x[in_bounds], in_bounds


def _cell_to_xy(cell_yx: tuple[int, int], *, origin_xy: np.ndarray, resolution_m: float) -> np.ndarray:
    y_idx, x_idx = cell_yx
    return np.asarray(
        [
            float(origin_xy[0]) + ((float(x_idx) + 0.5) * float(resolution_m)),
            float(origin_xy[1]) + ((float(y_idx) + 0.5) * float(resolution_m)),
        ],
        dtype=np.float64,
    )


def _xy_to_cell(
    xy: np.ndarray,
    *,
    origin_xy: np.ndarray,
    resolution_m: float,
    width: int,
    height: int,
) -> tuple[int, int]:
    x_idx = int(math.floor((float(xy[0]) - float(origin_xy[0])) / float(resolution_m)))
    y_idx = int(math.floor((float(xy[1]) - float(origin_xy[1])) / float(resolution_m)))
    return (max(0, min(height - 1, y_idx)), max(0, min(width - 1, x_idx)))


def _mark_line(mask: np.ndarray, start_yx: tuple[int, int], end_yx: tuple[int, int]) -> None:
    y0, x0 = start_yx
    y1, x1 = end_yx
    steps = max(abs(y1 - y0), abs(x1 - x0), 1)
    ys = np.rint(np.linspace(y0, y1, steps + 1)).astype(np.int32)
    xs = np.rint(np.linspace(x0, x1, steps + 1)).astype(np.int32)
    valid = (ys >= 0) & (ys < mask.shape[0]) & (xs >= 0) & (xs < mask.shape[1])
    mask[ys[valid], xs[valid]] = True


def _build_map_layers(
    map_points_xy: np.ndarray,
    path_poses_xyyaw: np.ndarray,
    *,
    resolution_m: float,
    margin_m: float,
    corridor_half_width_m: float,
    clearance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_points = np.vstack((map_points_xy, path_poses_xyyaw[:, :2]))
    width, height, origin_xy = _grid_geometry(
        all_points,
        resolution_m=resolution_m,
        margin_m=margin_m,
    )

    hit_counts = np.zeros((height, width), dtype=np.int32)
    hit_y, hit_x, _ = _points_to_cells(
        map_points_xy,
        origin_xy=origin_xy,
        resolution_m=resolution_m,
        width=width,
        height=height,
    )
    np.add.at(hit_counts, (hit_y, hit_x), 1)
    occupancy = hit_counts > 0
    occupancy = binary_closing(occupancy, structure=np.ones((3, 3), dtype=bool), iterations=1)
    occupancy = binary_dilation(occupancy, iterations=1)
    distance_field = distance_transform_edt(~occupancy) * float(resolution_m)

    path_mask = np.zeros((height, width), dtype=bool)
    path_cells: list[tuple[int, int]] = []
    for pose in path_poses_xyyaw:
        path_cells.append(
            _xy_to_cell(
                pose[:2],
                origin_xy=origin_xy,
                resolution_m=resolution_m,
                width=width,
                height=height,
            )
        )
    for current_cell, next_cell in zip(path_cells[:-1], path_cells[1:]):
        _mark_line(path_mask, current_cell, next_cell)
    if path_cells:
        path_mask[path_cells[0][0], path_cells[0][1]] = True
        path_mask[path_cells[-1][0], path_cells[-1][1]] = True
    corridor_distance = distance_transform_edt(~path_mask) * float(resolution_m)
    corridor_mask = corridor_distance <= float(corridor_half_width_m)
    free_mask = corridor_mask & (distance_field >= float(clearance_m))
    return occupancy.astype(bool), distance_field.astype(np.float32), origin_xy, corridor_mask, free_mask


def _nearest_free_cell(
    target_xy: np.ndarray,
    free_mask: np.ndarray,
    *,
    origin_xy: np.ndarray,
    resolution_m: float,
    max_shift_m: float,
) -> tuple[tuple[int, int], float]:
    target_cell = _xy_to_cell(
        target_xy,
        origin_xy=origin_xy,
        resolution_m=resolution_m,
        width=free_mask.shape[1],
        height=free_mask.shape[0],
    )
    if bool(free_mask[target_cell[0], target_cell[1]]):
        return target_cell, 0.0
    free_y, free_x = np.nonzero(free_mask)
    if free_y.size == 0:
        raise RuntimeError("fixed-map route has no free cells after applying corridor and clearance")
    target = np.asarray([target_cell[0], target_cell[1]], dtype=np.float64)
    deltas = np.column_stack((free_y, free_x)).astype(np.float64) - target.reshape(1, 2)
    distances_m = np.hypot(deltas[:, 0], deltas[:, 1]) * float(resolution_m)
    best_index = int(np.argmin(distances_m))
    best_distance_m = float(distances_m[best_index])
    if best_distance_m > float(max_shift_m):
        raise RuntimeError(
            "nearest free route cell is %.3f m from requested pose, above %.3f m"
            % (best_distance_m, float(max_shift_m))
        )
    return (int(free_y[best_index]), int(free_x[best_index])), best_distance_m


def _yaw_to_direction_index(yaw_rad: float) -> int:
    directions = _neighbor_directions()
    heading = np.asarray([math.sin(float(yaw_rad)), math.cos(float(yaw_rad))], dtype=np.float64)
    scores = [float(heading[0] * dy + heading[1] * dx) for dy, dx, _ in directions]
    return int(np.argmax(scores))


def _neighbor_directions() -> list[tuple[int, int, float]]:
    return [
        (-1, 0, 1.0),
        (-1, 1, math.sqrt(2.0)),
        (0, 1, 1.0),
        (1, 1, math.sqrt(2.0)),
        (1, 0, 1.0),
        (1, -1, math.sqrt(2.0)),
        (0, -1, 1.0),
        (-1, -1, math.sqrt(2.0)),
    ]


def _turn_angle_between_dirs(prev_dir: int, next_dir: int) -> float:
    if prev_dir < 0:
        return 0.0
    diff = abs(int(prev_dir) - int(next_dir)) % 8
    diff = min(diff, 8 - diff)
    return float(diff) * (math.pi / 4.0)


def _astar_route(
    free_mask: np.ndarray,
    distance_field: np.ndarray,
    *,
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
    start_yaw_rad: float,
    resolution_m: float,
    clearance_soft_m: float,
    clearance_weight: float,
    turn_weight: float,
) -> list[tuple[int, int]]:
    height, width = free_mask.shape
    directions = _neighbor_directions()
    dir_count = len(directions)
    start_dir = _yaw_to_direction_index(start_yaw_rad)
    distances = np.full((height, width, dir_count), np.inf, dtype=np.float64)
    closed = np.zeros((height, width, dir_count), dtype=bool)
    parents: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    queue: list[tuple[float, float, int, int, int]] = []

    sy, sx = start_cell
    gy, gx = goal_cell
    distances[sy, sx, start_dir] = 0.0
    heuristic = math.hypot(float(gy - sy), float(gx - sx)) * float(resolution_m)
    heapq.heappush(queue, (heuristic, 0.0, sy, sx, start_dir))
    goal_state: tuple[int, int, int] | None = None

    while queue:
        _, current_cost, y_idx, x_idx, dir_idx = heapq.heappop(queue)
        if closed[y_idx, x_idx, dir_idx]:
            continue
        closed[y_idx, x_idx, dir_idx] = True
        if (y_idx, x_idx) == goal_cell:
            goal_state = (y_idx, x_idx, dir_idx)
            break
        for next_dir, (dy, dx, step_cells) in enumerate(directions):
            ny = y_idx + dy
            nx = x_idx + dx
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            if not bool(free_mask[ny, nx]):
                continue
            step_cost = float(step_cells) * float(resolution_m)
            clearance_m = float(distance_field[ny, nx])
            clearance_ratio = max(0.0, (float(clearance_soft_m) - clearance_m) / max(1.0e-6, float(clearance_soft_m)))
            clearance_cost = float(clearance_weight) * clearance_ratio * clearance_ratio * step_cost
            turn_cost = float(turn_weight) * (_turn_angle_between_dirs(dir_idx, next_dir) / math.pi) * step_cost
            next_cost = current_cost + step_cost + clearance_cost + turn_cost
            if next_cost >= distances[ny, nx, next_dir]:
                continue
            distances[ny, nx, next_dir] = next_cost
            parents[(ny, nx, next_dir)] = (y_idx, x_idx, dir_idx)
            h_cost = math.hypot(float(gy - ny), float(gx - nx)) * float(resolution_m)
            heapq.heappush(queue, (next_cost + h_cost, next_cost, ny, nx, next_dir))

    if goal_state is None:
        raise RuntimeError("A* could not find a route through the fixed-map corridor")

    cells: list[tuple[int, int]] = []
    state = goal_state
    while True:
        y_idx, x_idx, _ = state
        cells.append((int(y_idx), int(x_idx)))
        if (y_idx, x_idx) == start_cell:
            break
        state = parents[state]
    cells.reverse()
    return cells


def _line_is_free(
    start_cell: tuple[int, int],
    end_cell: tuple[int, int],
    free_mask: np.ndarray,
) -> bool:
    y0, x0 = start_cell
    y1, x1 = end_cell
    steps = max(abs(y1 - y0), abs(x1 - x0), 1)
    ys = np.rint(np.linspace(y0, y1, steps + 1)).astype(np.int32)
    xs = np.rint(np.linspace(x0, x1, steps + 1)).astype(np.int32)
    if np.any((ys < 0) | (ys >= free_mask.shape[0]) | (xs < 0) | (xs >= free_mask.shape[1])):
        return False
    return bool(np.all(free_mask[ys, xs]))


def _shortcut_cells(cells: list[tuple[int, int]], free_mask: np.ndarray) -> list[tuple[int, int]]:
    if len(cells) <= 2:
        return cells
    output = [cells[0]]
    current_index = 0
    last_index = len(cells) - 1
    while current_index < last_index:
        next_index = last_index
        while next_index > current_index + 1 and not _line_is_free(
            cells[current_index],
            cells[next_index],
            free_mask,
        ):
            next_index -= 1
        output.append(cells[next_index])
        current_index = next_index
    return output


def _resample_path(path_xy: np.ndarray, step_m: float) -> np.ndarray:
    if path_xy.shape[0] < 2:
        return path_xy.copy()
    diffs = np.diff(path_xy, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    path_s = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total_s = float(path_s[-1])
    if total_s <= 1.0e-9:
        return path_xy[:1].copy()
    samples = np.arange(0.0, total_s, max(1.0e-3, float(step_m)), dtype=np.float64)
    if samples.size == 0 or samples[-1] < total_s:
        samples = np.append(samples, total_s)
    return np.column_stack(
        (
            np.interp(samples, path_s, path_xy[:, 0]),
            np.interp(samples, path_s, path_xy[:, 1]),
        )
    )


def _point_is_free(
    xy: np.ndarray,
    free_mask: np.ndarray,
    *,
    origin_xy: np.ndarray,
    resolution_m: float,
) -> bool:
    y_idx, x_idx = _xy_to_cell(
        xy,
        origin_xy=origin_xy,
        resolution_m=resolution_m,
        width=free_mask.shape[1],
        height=free_mask.shape[0],
    )
    return bool(free_mask[y_idx, x_idx])


def _smooth_path(
    path_xy: np.ndarray,
    free_mask: np.ndarray,
    *,
    origin_xy: np.ndarray,
    resolution_m: float,
    alpha: float,
    iterations: int,
) -> np.ndarray:
    if path_xy.shape[0] <= 2:
        return path_xy.copy()
    smoothed = path_xy.copy()
    alpha = max(0.0, min(1.0, float(alpha)))
    for _ in range(max(0, int(iterations))):
        changed = False
        for index in range(1, smoothed.shape[0] - 1):
            candidate = (1.0 - alpha) * smoothed[index] + alpha * 0.5 * (
                smoothed[index - 1] + smoothed[index + 1]
            )
            if _point_is_free(
                candidate,
                free_mask,
                origin_xy=origin_xy,
                resolution_m=resolution_m,
            ):
                if float(np.linalg.norm(candidate - smoothed[index])) > 1.0e-6:
                    changed = True
                smoothed[index] = candidate
        if not changed:
            break
    return smoothed


def _route_to_xyyaw(route_xy: np.ndarray, *, start_yaw_rad: float, goal_yaw_rad: float) -> np.ndarray:
    if route_xy.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    yaw = np.zeros((route_xy.shape[0],), dtype=np.float64)
    yaw[0] = float(start_yaw_rad)
    if route_xy.shape[0] > 1:
        diffs = np.diff(route_xy, axis=0)
        segment_yaw = np.arctan2(diffs[:, 1], diffs[:, 0])
        yaw[:-1] = segment_yaw
        yaw[-1] = float(goal_yaw_rad)
        if route_xy.shape[0] > 2:
            yaw[0] = segment_yaw[0]
    yaw = np.asarray([_normalize_angle(float(value)) for value in yaw], dtype=np.float64)
    return np.column_stack((route_xy, yaw))


def _route_clearances(
    route_xy: np.ndarray,
    distance_field: np.ndarray,
    *,
    origin_xy: np.ndarray,
    resolution_m: float,
) -> np.ndarray:
    values: list[float] = []
    for point in route_xy:
        y_idx, x_idx = _xy_to_cell(
            point,
            origin_xy=origin_xy,
            resolution_m=resolution_m,
            width=distance_field.shape[1],
            height=distance_field.shape[0],
        )
        values.append(float(distance_field[y_idx, x_idx]))
    return np.asarray(values, dtype=np.float64)


def _write_pgm(path: Path, occupancy: np.ndarray) -> None:
    image = np.where(occupancy, 0, 254).astype(np.uint8)
    pgm = np.flipud(image)
    with path.open("wb") as handle:
        handle.write(f"P5\n{pgm.shape[1]} {pgm.shape[0]}\n255\n".encode("ascii"))
        handle.write(pgm.tobytes())


def _write_xy_csv(path: Path, points_xy: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_m", "y_m"])
        for point in points_xy:
            writer.writerow([f"{point[0]:.6f}", f"{point[1]:.6f}"])


def _write_path_csv(path: Path, route_xyyaw: np.ndarray, clearances_m: np.ndarray | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "x_m", "y_m", "yaw_rad", "clearance_m"])
        for index, pose in enumerate(route_xyyaw):
            clearance_m = float(clearances_m[index]) if clearances_m is not None and index < clearances_m.size else float("nan")
            writer.writerow(
                [
                    index,
                    f"{pose[0]:.6f}",
                    f"{pose[1]:.6f}",
                    f"{pose[2]:.6f}",
                    f"{clearance_m:.6f}",
                ]
            )


def _write_map_yaml(
    path: Path,
    *,
    resolution_m: float,
    origin_xy: np.ndarray,
    initial_pose_xyyaw: np.ndarray,
) -> None:
    payload = {
        "image": "fixed_map.pgm",
        "resolution": float(resolution_m),
        "origin": [float(origin_xy[0]), float(origin_xy[1]), 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
        "distance_field_npy": "fixed_map_distance.npy",
        "visual_points_csv": "fixed_map_visual_points.csv",
        "optimized_keyframes_csv": "fixed_route_path.csv",
        "initial_pose": [
            float(initial_pose_xyyaw[0]),
            float(initial_pose_xyyaw[1]),
            float(initial_pose_xyyaw[2]),
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _plot_preview(
    path: Path,
    *,
    map_points_xy: np.ndarray,
    recorded_path_xy: np.ndarray,
    route_xy: np.ndarray,
    start_pose: np.ndarray,
    goal_pose: np.ndarray,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig, ax = plt.subplots(figsize=(8, 8))
    if map_points_xy.size:
        ax.scatter(map_points_xy[:, 0], map_points_xy[:, 1], s=4, c="black", alpha=0.55, label="map borders")
    if recorded_path_xy.size:
        ax.plot(recorded_path_xy[:, 0], recorded_path_xy[:, 1], color="#1f77b4", linewidth=1.5, label="recorded path")
    if route_xy.size:
        ax.plot(route_xy[:, 0], route_xy[:, 1], color="#d62728", linewidth=2.0, label="fixed route")
    ax.scatter([start_pose[0]], [start_pose[1]], c="#2ca02c", s=48, label="start")
    ax.scatter([goal_pose[0]], [goal_pose[1]], c="#9467bd", s=48, label="goal")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.set_xlabel("x_m")
    ax.set_ylabel("y_m")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def _build_fixed_map(args: argparse.Namespace) -> dict[str, object]:
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (run_dir / "fixed_map")
    map_points_csv = run_dir / "map_points_xy.csv"
    path_poses_csv = run_dir / "path_poses_xyyaw.csv"
    poses_json = run_dir / "poses.json"
    for required in (map_points_csv, path_poses_csv, poses_json):
        if not required.exists():
            raise FileNotFoundError(f"Missing offline refined input: {required}")

    map_points_xy = _read_points_csv(map_points_csv, require_yaw=False)
    path_poses_xyyaw = _read_points_csv(path_poses_csv, require_yaw=True)
    if map_points_xy.shape[0] < 8:
        raise RuntimeError("map_points_xy.csv does not contain enough map points")
    if path_poses_xyyaw.shape[0] < 2:
        raise RuntimeError("path_poses_xyyaw.csv does not contain enough poses")
    poses_payload = json.loads(poses_json.read_text(encoding="utf-8"))
    initial_pose = _pose_from_json(poses_payload, "initial_pose")
    final_pose = _pose_from_json(poses_payload, "final_pose")

    output_dir.mkdir(parents=True, exist_ok=True)
    corridor_widths = [float(args.corridor_half_width_m)]
    for raw in str(args.corridor_retry_half_widths_m or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = float(raw)
        if all(abs(value - existing) > 1.0e-6 for existing in corridor_widths):
            corridor_widths.append(value)

    last_error: Exception | None = None
    route_cells: list[tuple[int, int]] = []
    selected_layers: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    selected_corridor_width_m = corridor_widths[0]
    start_shift_m = 0.0
    goal_shift_m = 0.0
    start_cell = (0, 0)
    goal_cell = (0, 0)
    for corridor_width_m in corridor_widths:
        occupancy, distance_field, origin_xy, corridor_mask, free_mask = _build_map_layers(
            map_points_xy,
            path_poses_xyyaw,
            resolution_m=float(args.resolution_m),
            margin_m=float(args.margin_m),
            corridor_half_width_m=float(corridor_width_m),
            clearance_m=float(args.clearance_m),
        )
        try:
            start_cell, start_shift_m = _nearest_free_cell(
                initial_pose[:2],
                free_mask,
                origin_xy=origin_xy,
                resolution_m=float(args.resolution_m),
                max_shift_m=float(args.max_endpoint_shift_m),
            )
            goal_cell, goal_shift_m = _nearest_free_cell(
                final_pose[:2],
                free_mask,
                origin_xy=origin_xy,
                resolution_m=float(args.resolution_m),
                max_shift_m=float(args.max_endpoint_shift_m),
            )
            route_cells = _astar_route(
                free_mask,
                distance_field,
                start_cell=start_cell,
                goal_cell=goal_cell,
                start_yaw_rad=float(initial_pose[2]),
                resolution_m=float(args.resolution_m),
                clearance_soft_m=float(args.clearance_soft_m),
                clearance_weight=float(args.clearance_weight),
                turn_weight=float(args.turn_weight),
            )
            selected_layers = (occupancy, distance_field, origin_xy, corridor_mask, free_mask)
            selected_corridor_width_m = float(corridor_width_m)
            break
        except Exception as exc:
            last_error = exc
            continue
    if selected_layers is None:
        raise RuntimeError(f"failed to build fixed-map route: {last_error}")

    occupancy, distance_field, origin_xy, corridor_mask, free_mask = selected_layers
    shortcut_cells = _shortcut_cells(route_cells, free_mask)
    raw_route_xy = np.asarray(
        [
            _cell_to_xy(cell, origin_xy=origin_xy, resolution_m=float(args.resolution_m))
            for cell in shortcut_cells
        ],
        dtype=np.float64,
    )
    route_xy = _resample_path(raw_route_xy, float(args.route_step_m))
    route_xy = _smooth_path(
        route_xy,
        free_mask,
        origin_xy=origin_xy,
        resolution_m=float(args.resolution_m),
        alpha=float(args.route_smoothing_alpha),
        iterations=int(args.route_smoothing_iterations),
    )
    route_xyyaw = _route_to_xyyaw(
        route_xy,
        start_yaw_rad=float(initial_pose[2]),
        goal_yaw_rad=float(final_pose[2]),
    )
    clearances_m = _route_clearances(
        route_xy,
        distance_field,
        origin_xy=origin_xy,
        resolution_m=float(args.resolution_m),
    )
    min_clearance_m = float(np.min(clearances_m)) if clearances_m.size else 0.0
    if min_clearance_m + 1.0e-9 < float(args.clearance_m):
        raise RuntimeError(
            "planned fixed route clearance %.3f m is below required %.3f m"
            % (min_clearance_m, float(args.clearance_m))
        )

    visual_points_xy = map_points_xy.copy()
    fixed_map_yaml = output_dir / "fixed_map.yaml"
    fixed_map_pgm = output_dir / "fixed_map.pgm"
    fixed_map_distance_npy = output_dir / "fixed_map_distance.npy"
    fixed_map_visual_points_csv = output_dir / "fixed_map_visual_points.csv"
    fixed_route_path_csv = output_dir / "fixed_route_path.csv"
    recorded_path_csv = output_dir / "recorded_path_poses_xyyaw.csv"
    build_status_json = output_dir / "fixed_map_build_status.json"
    mapping_summary_json = output_dir / "mapping_summary.json"
    preview_png = output_dir / "route_preview.png"

    _write_pgm(fixed_map_pgm, occupancy)
    np.save(fixed_map_distance_npy, distance_field.astype(np.float32))
    _write_xy_csv(fixed_map_visual_points_csv, visual_points_xy)
    _write_path_csv(fixed_route_path_csv, route_xyyaw, clearances_m)
    _write_path_csv(recorded_path_csv, path_poses_xyyaw)
    _write_map_yaml(
        fixed_map_yaml,
        resolution_m=float(args.resolution_m),
        origin_xy=origin_xy,
        initial_pose_xyyaw=initial_pose,
    )
    preview_written = False if args.no_preview else _plot_preview(
        preview_png,
        map_points_xy=map_points_xy,
        recorded_path_xy=path_poses_xyyaw[:, :2],
        route_xy=route_xy,
        start_pose=initial_pose,
        goal_pose=final_pose,
    )

    route_length_m = float(np.sum(np.hypot(np.diff(route_xy[:, 0]), np.diff(route_xy[:, 1])))) if route_xy.shape[0] > 1 else 0.0
    start_xy = route_xy[0] if route_xy.shape[0] else initial_pose[:2]
    goal_xy = route_xy[-1] if route_xy.shape[0] else final_pose[:2]
    summary = {
        "state": "done",
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "map_point_count": int(map_points_xy.shape[0]),
        "recorded_path_pose_count": int(path_poses_xyyaw.shape[0]),
        "route_point_count": int(route_xyyaw.shape[0]),
        "route_length_m": route_length_m,
        "min_clearance_m": min_clearance_m,
        "required_clearance_m": float(args.clearance_m),
        "corridor_half_width_m": selected_corridor_width_m,
        "start_shift_m": float(start_shift_m),
        "goal_shift_m": float(goal_shift_m),
        "start_error_m": float(np.linalg.norm(start_xy - initial_pose[:2])),
        "goal_error_m": float(np.linalg.norm(goal_xy - final_pose[:2])),
        "map_resolution_m": float(args.resolution_m),
        "map_margin_m": float(args.margin_m),
        "occupied_cell_count": int(np.count_nonzero(occupancy)),
        "corridor_cell_count": int(np.count_nonzero(corridor_mask)),
        "free_cell_count": int(np.count_nonzero(free_mask)),
        "preview_written": bool(preview_written),
        "files": {
            "fixed_map_yaml": str(fixed_map_yaml),
            "fixed_map_pgm": str(fixed_map_pgm),
            "fixed_map_distance_npy": str(fixed_map_distance_npy),
            "fixed_map_visual_points_csv": str(fixed_map_visual_points_csv),
            "fixed_route_path_csv": str(fixed_route_path_csv),
            "recorded_path_poses_xyyaw_csv": str(recorded_path_csv),
            "fixed_map_build_status_json": str(build_status_json),
            "mapping_summary_json": str(mapping_summary_json),
            "route_preview_png": str(preview_png) if preview_written else "",
        },
        "initial_pose": [float(value) for value in initial_pose.tolist()],
        "final_pose": [float(value) for value in final_pose.tolist()],
    }
    build_status_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    mapping_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="offline_refined_* directory")
    parser.add_argument("--output-dir", default="", help="output fixed_map directory")
    parser.add_argument("--resolution-m", type=float, default=0.05)
    parser.add_argument("--margin-m", type=float, default=0.75)
    parser.add_argument("--clearance-m", type=float, default=0.15)
    parser.add_argument("--clearance-soft-m", type=float, default=0.45)
    parser.add_argument("--corridor-half-width-m", type=float, default=0.75)
    parser.add_argument("--corridor-retry-half-widths-m", default="0.90,1.10")
    parser.add_argument("--max-endpoint-shift-m", type=float, default=0.80)
    parser.add_argument("--clearance-weight", type=float, default=2.2)
    parser.add_argument("--turn-weight", type=float, default=0.65)
    parser.add_argument("--route-step-m", type=float, default=0.05)
    parser.add_argument("--route-smoothing-alpha", type=float, default=0.22)
    parser.add_argument("--route-smoothing-iterations", type=int, default=120)
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = _build_fixed_map(args)
    print(
        "[APEX] fixed map ready: route_points=%d length=%.3fm clearance=%.3fm output=%s"
        % (
            int(summary["route_point_count"]),
            float(summary["route_length_m"]),
            float(summary["min_clearance_m"]),
            str(summary["output_dir"]),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
