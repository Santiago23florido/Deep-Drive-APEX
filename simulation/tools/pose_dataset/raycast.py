"""Exact ray casting of a 2D spinning LiDAR against the world primitives.

Each beam is a 3D ray with its own origin and direction, so the scan plane
follows the full 6-DoF attitude of the sensor (a pitched car sees the floor at
long range) and every beam may be fired from a different pose (rolling scan).
Boxes are oriented about z (slab test in the box frame); cylinders are
vertical (quadratic in the plane plus a height check). The ground is a box
like any other. The car itself is not part of the geometry.
"""

from __future__ import annotations

import numpy as np

from .geometry import WorldGeometry


class RayCaster:
    def __init__(self, geometry: WorldGeometry, max_range: float = 30.0) -> None:
        b = geometry.box_arrays(include_ground=True)
        self.box_c = b["center"]
        self.box_h = b["half"]
        self.box_cos = np.cos(b["yaw"])
        self.box_sin = np.sin(b["yaw"])
        self.box_r = np.hypot(self.box_h[:, 0], self.box_h[:, 1])
        c = geometry.cylinder_arrays()
        self.cyl_c = c["center"]
        self.cyl_r = c["radius"]
        self.cyl_hl = c["half_length"]
        self.max_range = max_range

    def _box_subset(self, centre_xy: np.ndarray, reach: float) -> np.ndarray:
        d = np.hypot(self.box_c[:, 0] - centre_xy[0], self.box_c[:, 1] - centre_xy[1]) - self.box_r
        return np.nonzero(d <= reach)[0]

    def cast(self, origins: np.ndarray, dirs: np.ndarray, max_range: float | None = None) -> np.ndarray:
        """Distance to the first hit along each ray; ``inf`` when nothing is hit
        within ``max_range``. ``origins``/``dirs``: (M, 3), dirs unit length."""
        max_range = float(max_range or self.max_range)
        o = np.asarray(origins, dtype=float)
        d = np.asarray(dirs, dtype=float)
        m = len(o)
        best = np.full(m, np.inf)
        centre = o[:, :2].mean(axis=0)
        spread = float(np.max(np.hypot(o[:, 0] - centre[0], o[:, 1] - centre[1]))) if m else 0.0
        reach = max_range + spread

        idx = self._box_subset(centre, reach)
        if len(idx):
            c = self.box_c[idx]
            h = self.box_h[idx]
            cs = self.box_cos[idx]
            sn = self.box_sin[idx]
            rel = o[:, None, :] - c[None, :, :]  # (M, N, 3)
            ox = cs * rel[..., 0] + sn * rel[..., 1]
            oy = -sn * rel[..., 0] + cs * rel[..., 1]
            oz = rel[..., 2]
            dx = cs[None, :] * d[:, 0:1] + sn[None, :] * d[:, 1:2]
            dy = -sn[None, :] * d[:, 0:1] + cs[None, :] * d[:, 1:2]
            dz = np.broadcast_to(d[:, 2:3], dx.shape)
            tmin = np.full(dx.shape, -np.inf)
            tmax = np.full(dx.shape, np.inf)
            miss = np.zeros(dx.shape, dtype=bool)
            with np.errstate(divide="ignore", invalid="ignore"):
                for oc, dc, hc in ((ox, dx, h[:, 0]), (oy, dy, h[:, 1]), (oz, dz, h[:, 2])):
                    parallel = np.abs(dc) < 1e-12
                    miss |= parallel & (np.abs(oc) > hc[None, :])
                    inv = 1.0 / np.where(parallel, 1.0, dc)
                    t1 = (-hc[None, :] - oc) * inv
                    t2 = (hc[None, :] - oc) * inv
                    lo = np.where(parallel, -np.inf, np.minimum(t1, t2))
                    hi = np.where(parallel, np.inf, np.maximum(t1, t2))
                    tmin = np.maximum(tmin, lo)
                    tmax = np.minimum(tmax, hi)
            hit = ~miss & (tmax >= tmin) & (tmin > 1e-6)
            t_box = np.where(hit, tmin, np.inf).min(axis=1)
            best = np.minimum(best, t_box)

        if len(self.cyl_r):
            dc = np.hypot(self.cyl_c[:, 0] - centre[0], self.cyl_c[:, 1] - centre[1]) - self.cyl_r
            cid = np.nonzero(dc <= reach)[0]
            if len(cid):
                cc = self.cyl_c[cid]
                r = self.cyl_r[cid]
                hl = self.cyl_hl[cid]
                px = o[:, None, 0] - cc[None, :, 0]
                py = o[:, None, 1] - cc[None, :, 1]
                a = d[:, 0:1] ** 2 + d[:, 1:2] ** 2
                b = 2.0 * (d[:, 0:1] * px + d[:, 1:2] * py)
                q = px**2 + py**2 - r[None, :] ** 2
                disc = b * b - 4.0 * a * q
                with np.errstate(invalid="ignore", divide="ignore"):
                    sq = np.sqrt(np.maximum(disc, 0.0))
                    t = (-b - sq) / (2.0 * a)
                z = o[:, None, 2] + t * d[:, 2:3]
                ok = (disc >= 0.0) & (t > 1e-6) & (q > 0.0) & (np.abs(z - cc[None, :, 2]) <= hl[None, :]) & (a > 1e-12)
                best = np.minimum(best, np.where(ok, t, np.inf).min(axis=1))
        best[best > max_range] = np.inf
        return best
