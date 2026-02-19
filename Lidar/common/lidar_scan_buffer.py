"""Utilities to map raw RPLidar scans into a fixed 360-sample array."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class LidarScanBuffer:
    """Keeps latest distance per angle and applies offset/FOV/timeout filters."""

    samples: int = 360
    heading_offset_deg: int = 0
    fov_filter_deg: int = 360
    point_timeout_ms: int = 200

    def __post_init__(self) -> None:
        if self.samples <= 0:
            raise ValueError("samples must be > 0")

        self._distances_m = np.zeros(self.samples, dtype=np.float32)
        self._last_update_ms = np.zeros(self.samples, dtype=np.float64)

    def update_from_rplidar_scan(
        self,
        scan: Iterable[Sequence[float]],
        current_ms: float | None = None,
    ) -> np.ndarray:
        """Update internal buffers with a RPLidar scan and return filtered ranges."""
        if current_ms is None:
            current_ms = time.time() * 1000.0

        for measurement in scan:
            if len(measurement) < 3:
                continue

            angle_deg = float(measurement[1])
            distance_m = float(measurement[2]) / 1000.0

            if not np.isfinite(distance_m) or distance_m <= 0.0:
                continue

            idx = int(round(angle_deg)) % self.samples
            self._distances_m[idx] = distance_m
            self._last_update_ms[idx] = current_ms

        return self.get_filtered_scan(current_ms=current_ms)

    def get_filtered_scan(self, current_ms: float | None = None) -> np.ndarray:
        """Return a filtered snapshot (copy) of the current 360 ranges."""
        if current_ms is None:
            current_ms = time.time() * 1000.0

        shift = int(self.heading_offset_deg) % self.samples
        ranges = np.roll(self._distances_m, shift).copy()
        timestamps_ms = np.roll(self._last_update_ms, shift)

        if self.point_timeout_ms > 0:
            expired = (timestamps_ms > 0.0) & ((current_ms - timestamps_ms) > self.point_timeout_ms)
            ranges[expired] = 0.0

        if 0 < self.fov_filter_deg < 360:
            half_fov = self.fov_filter_deg / 2.0
            angles = np.arange(self.samples, dtype=np.float32)
            diffs = np.mod(angles - 0.0, 360.0)
            keep = (diffs <= half_fov) | (diffs >= 360.0 - half_fov)
            ranges[~keep] = 0.0

        return ranges
