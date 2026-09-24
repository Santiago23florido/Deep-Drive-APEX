"""Occupancy-grid helpers (ROS-independent).

``nav_msgs/OccupancyGrid`` stores ``data`` row-major (index = col + row * width)
with values -1 (unknown) or 0..100 (occupancy probability in percent). The
``origin`` is the pose of the lower-left corner of cell (0, 0) in the map frame.

These helpers export a grid as occupied-cell points (for CSV / metrics) and as
a map_server-compatible PGM + YAML pair, without requiring nav2_map_server.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


@dataclass
class GridInfo:
    width: int
    height: int
    resolution: float  # m / cell
    origin_x: float
    origin_y: float
    origin_yaw: float = 0.0


def grid_to_points(data, info: GridInfo, occupied_threshold: int = 65) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xy, occupancy)`` of the cells with occupancy >= threshold.

    ``xy`` are the cell centres in the map frame (origin rotation respected).
    """
    grid = np.asarray(data, dtype=np.int16).reshape(info.height, info.width)
    rows, cols = np.nonzero(grid >= occupied_threshold)
    local = np.column_stack(((cols + 0.5) * info.resolution, (rows + 0.5) * info.resolution))
    c, s = math.cos(info.origin_yaw), math.sin(info.origin_yaw)
    xy = local @ np.array([[c, s], [-s, c]]) + np.array([info.origin_x, info.origin_y])
    return xy, grid[rows, cols].astype(np.int16)


def write_pgm_yaml(
    data,
    info: GridInfo,
    base_path: str | Path,
    occupied_threshold: int = 65,
    free_threshold: int = 25,
) -> tuple[Path, Path]:
    """Write ``<base>.pgm`` + ``<base>.yaml`` in the nav2 map_server format
    (trinary: occupied 0, free 254, unknown 205; first image row = top)."""
    base = Path(base_path)
    grid = np.asarray(data, dtype=np.int16).reshape(info.height, info.width)
    image = np.full(grid.shape, 205, dtype=np.uint8)
    image[(grid >= 0) & (grid <= free_threshold)] = 254
    image[grid >= occupied_threshold] = 0
    image = np.flipud(image)
    pgm = base.with_suffix(".pgm")
    with open(pgm, "wb") as handle:
        handle.write(f"P5\n{info.width} {info.height}\n255\n".encode("ascii"))
        handle.write(image.tobytes())
    yaml_path = base.with_suffix(".yaml")
    yaml_path.write_text(
        "\n".join(
            [
                f"image: {pgm.name}",
                "mode: trinary",
                f"resolution: {info.resolution:.6f}",
                f"origin: [{info.origin_x:.6f}, {info.origin_y:.6f}, {info.origin_yaw:.6f}]",
                "negate: 0",
                f"occupied_thresh: {occupied_threshold / 100.0:.2f}",
                f"free_thresh: {free_threshold / 100.0:.2f}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return pgm, yaml_path
