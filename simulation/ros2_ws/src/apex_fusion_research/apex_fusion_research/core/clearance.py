"""True clearance of the car footprint to the track (evaluation only).

Uses the track geometry (walls, curbs, pillars, decoration): never available
to the car, only to the referee, the evaluation and the development harness.
Curbs are included (their top is above the ground), although the LiDAR does
not see them.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree


def outline_points(geometry, min_top: float = 0.012, step: float = 0.02, kinds: tuple[str, ...] | None = None) -> tuple[np.ndarray, list[str]]:  # noqa: ANN001
    """Outline (every ``step``) of every box and cylinder whose top is above
    ``min_top``; returns ``(points (N, 2), kind of each point)``."""
    pts: list[tuple[float, float]] = []
    kind: list[str] = []
    for b in geometry.boxes:
        if b.kind == "ground" or b.center[2] + 0.5 * b.size[2] <= min_top or (kinds and b.kind not in kinds):
            continue
        hx, hy = 0.5 * b.size[0], 0.5 * b.size[1]
        c, s = math.cos(b.yaw), math.sin(b.yaw)
        corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
            n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
            for i in range(n):
                u = i / n
                lx, ly = x0 + u * (x1 - x0), y0 + u * (y1 - y0)
                pts.append((b.center[0] + c * lx - s * ly, b.center[1] + s * lx + c * ly))
                kind.append(b.kind)
    for cy in geometry.cylinders:
        if cy.center[2] + 0.5 * cy.length <= min_top or (kinds and cy.kind not in kinds):
            continue
        n = max(8, int(2 * math.pi * cy.radius / step))
        for i in range(n):
            a = 2 * math.pi * i / n
            pts.append((cy.center[0] + cy.radius * math.cos(a), cy.center[1] + cy.radius * math.sin(a)))
            kind.append(cy.kind)
    return np.asarray(pts, dtype=float).reshape(-1, 2), kind


class FootprintClearance:
    """Signed distance from the rectangular footprint to the outline points."""

    def __init__(self, points: np.ndarray, length: float, width: float) -> None:
        self.points = np.asarray(points, dtype=float)
        self.tree = cKDTree(self.points)
        self.hl, self.hw = 0.5 * length, 0.5 * width
        self.reach = math.hypot(self.hl, self.hw) + 1.0

    def __call__(self, x: float, y: float, yaw: float) -> float:
        idx = self.tree.query_ball_point((x, y), self.reach)
        if not idx:
            return self.reach
        p = self.points[idx] - (x, y)
        c, s = math.cos(yaw), math.sin(yaw)
        lx, ly = c * p[:, 0] + s * p[:, 1], -s * p[:, 0] + c * p[:, 1]
        ex, ey = np.abs(lx) - self.hl, np.abs(ly) - self.hw
        outside = np.hypot(np.maximum(ex, 0.0), np.maximum(ey, 0.0))
        inside = np.minimum(np.maximum(ex, ey), 0.0)  # negative: the point is inside the footprint
        return float(np.min(np.where((ex > 0) | (ey > 0), outside, inside)))

    def along(self, xy: np.ndarray, yaw: np.ndarray) -> np.ndarray:
        return np.array([self(float(x), float(y), float(w)) for (x, y), w in zip(xy, yaw)])
