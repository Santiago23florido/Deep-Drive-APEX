"""Rolling spinning-LiDAR model on top of Gazebo's gpu_lidar (real2sim layer).

Gazebo's ``gpu_lidar`` captures the whole horizontal field of view at one
instant. A real spinning scanner such as the RPLIDAR A2M8 takes its samples
one by one while it rotates (a revolution lasts ~77 ms at 13 Hz), so every
sample is taken from a different pose of the moving car. This module turns a
stream of gpu_lidar captures (the geometry source, run faster and finer than
the real sensor, e.g. 26 Hz x 1440 rays) plus the sensor pose track into the
revolutions of the real device:

1. revolution timing: period ``1 / rate_hz`` with a per-revolution motor-speed
   jitter; samples at the device sample rate (2000 Hz in the compatible
   protocol, 8000 Hz in Sensitivity mode), rotating clockwise or
   counter-clockwise from ``start_angle`` (the angle of the sync pulse in
   base_link);
2. geometry of each sample: the capture closest in time is re-projected into
   the viewpoint of the sample (sensor pose interpolated at the sample
   instant); consecutive capture points on one continuous surface form
   segments, and the sample range is the closest segment crossing the sample
   direction (z-buffer). Gaps between discontinuous points stay open, so
   occlusion boundaries move with the parallax;
3. directions inside ``occluded_sectors`` (the car's own structure) return
   nothing;
4. the stochastic range model of :mod:`lidar_noise` (heteroscedastic noise,
   incidence, calibration errors, angular jitter, missing returns, outliers,
   quantization) is applied to every sample;
5. the samples are binned like the car's driver: fixed 1-degree bins from
   -pi, keeping the closest sample of every bin (``min_range``) or the sample
   nearest to the bin centre (``nearest``);
6. timestamp = first sample of the revolution (+ offset + jitter), available
   ``publish_latency`` after its last sample.

The per-bin time offsets that a consumer needs to undo the motion distortion
follow from the calibration (:func:`bin_time_fraction`).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np

from .lidar_noise import LidarNoiseConfig, LidarNoiseModel

TWO_PI = 2.0 * math.pi


@dataclass
class RollingLidarConfig:
    rate_hz: float = 13.0
    rate_jitter_rel: float = 0.0
    sample_rate_hz: float = 8000.0
    direction: str = "cw"  # physical rotation seen from above in base_link
    start_angle_deg: float = -180.0  # base_link angle of the first sample of a revolution
    beams: int = 360
    angle_min_rad: float = -math.pi
    binning: str = "min_range"  # or "nearest"
    occluded_sectors_deg: tuple[tuple[float, float], ...] = ()
    range_min: float = 0.15
    range_max: float = 12.0
    timestamp_offset_ms: float = 0.0
    timestamp_jitter_std_ms: float = 0.0
    publish_latency_ms: float = 0.0
    scan_dropout_prob: float = 0.0
    noise: LidarNoiseConfig = field(default_factory=lambda: LidarNoiseConfig(enabled=False))
    continuity_abs_m: float = 0.05  # consecutive capture points on one surface ...
    continuity_rel: float = 4.0  # ... closer than abs + rel * range * capture step
    window_deg: float = 24.0  # parallax search window around each ray (surfaces beyond ~0.2 m)
    start_grid_s: float = 0.0  # >0: revolution starts on this clock grid (physics step), so they have an exact pose
    seed: int = 0


@dataclass
class Revolution:
    stamp_ns: int  # reported (first sample + offset + jitter)
    release_ns: int  # available to the host
    t_start_ns: int  # true instant of the first sample
    t_end_ns: int  # true instant of the last sample
    period_s: float
    ranges: np.ndarray  # [beams] float32, REP-117 (+inf no return, -inf below range_min)
    ideal: np.ndarray  # [beams] noise-free binned ranges (evaluation only)
    samples: int
    lost: bool = False


def bin_time_fraction(beams: int, angle_min: float, angle_increment: float, direction: str, start_angle_rad: float) -> np.ndarray:
    """Nominal acquisition time of every bin as a fraction of the revolution."""
    theta = angle_min + angle_increment * np.arange(beams)
    turn = (start_angle_rad - theta) if direction == "cw" else (theta - start_angle_rad)
    frac = np.mod(turn, TWO_PI) / TWO_PI
    return np.where(np.isclose(frac, 1.0), 0.0, frac)  # the bin at the sync angle is the first one


def _interp_pose(track: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Planar pose [x, y, yaw] at times ``t`` from a [N, 4] (t, x, y, yaw) track."""
    tt = track[:, 0]
    x = np.interp(t, tt, track[:, 1])
    y = np.interp(t, tt, track[:, 2])
    yaw = np.interp(t, tt, np.unwrap(track[:, 3]))
    return np.stack((x, y, yaw), axis=-1)


class RollingLidar:
    def __init__(self, cfg: RollingLidarConfig) -> None:
        self.cfg = cfg
        ss = np.random.SeedSequence(int(cfg.seed)).spawn(2)
        self._rng = np.random.default_rng(ss[0])
        noise_cfg = cfg.noise
        noise_cfg.seed = int(np.random.SeedSequence(int(cfg.seed) + 7).generate_state(1)[0])
        self.noise = LidarNoiseModel(noise_cfg)
        self._poses: deque[tuple[float, float, float, float]] = deque()
        # t, points (world) [N, 2], valid [N], ranges [N], capture pose (x, y, yaw), first beam angle
        self._captures: deque[tuple[float, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float], float]] = deque()
        self._cap_step = 0.0
        self.window_rad = math.radians(cfg.window_deg)
        self._next_start: float | None = None
        self._period: float | None = None
        self._last_stamp = -(2**62)
        self.start_rad = math.radians(cfg.start_angle_deg)
        self.sign = -1.0 if cfg.direction == "cw" else 1.0
        self.occluded = [(math.radians(a), math.radians(b)) for a, b in cfg.occluded_sectors_deg]
        self.revolutions = 0

    # ----------------------------------------------------------------- input
    def add_pose(self, t_s: float, x: float, y: float, yaw: float) -> None:
        """Planar pose of the sensor frame (world) at ``t_s``; at least ~250 Hz."""
        if self._poses and t_s <= self._poses[-1][0]:
            return
        self._poses.append((t_s, x, y, yaw))
        while len(self._poses) > 2 and self._poses[0][0] < t_s - 1.0:
            self._poses.popleft()
        if self._next_start is None:
            self._draw_period()
            self._next_start = self._snap(t_s + float(self._rng.uniform(0.0, self._period)))

    def add_capture(self, t_s: float, ranges: np.ndarray, angle_min: float, angle_increment: float,
                    pose: tuple[float, float, float] | None = None) -> None:
        """A gpu_lidar capture (sensor frame) taken at ``t_s`` (pose interpolated if not given)."""
        r = np.asarray(ranges, dtype=float)
        if pose is None:
            if len(self._poses) < 2:
                return
            pose = tuple(_interp_pose(np.array(self._poses), np.array([t_s]))[0])
        x, y, yaw = pose
        phi = yaw + angle_min + angle_increment * np.arange(len(r))
        valid = np.isfinite(r) & (r > 0.0)
        rr = np.where(valid, r, 0.0)
        pts = np.stack((x + rr * np.cos(phi), y + rr * np.sin(phi)), axis=-1)
        self._cap_step = abs(angle_increment)
        self._captures.append((t_s, pts, valid, rr, (x, y, yaw), float(angle_min)))
        while len(self._captures) > 2 and self._captures[0][0] < t_s - 0.5:
            self._captures.popleft()

    def poll(self) -> list[Revolution]:
        """Revolutions whose samples are covered by the pose track and captures."""
        out = []
        if self._next_start is None or not self._captures or len(self._poses) < 2:
            return out
        cap_dt = 0.5 / 26.0
        if len(self._captures) >= 2:
            cap_dt = 0.5 * (self._captures[-1][0] - self._captures[-2][0])
        while True:
            s, T = self._next_start, self._period
            end = s + T
            if end > self._poses[-1][0] or end > self._captures[-1][0] + cap_dt:
                break
            nxt = self._snap(end)
            out.append(self._revolution(s, nxt - s if self.cfg.start_grid_s > 0.0 else T))
            self._next_start = nxt
            self._draw_period()
        return out

    # ------------------------------------------------------------- internals
    def _snap(self, t: float) -> float:
        g = self.cfg.start_grid_s
        return round(t / g) * g if g > 0.0 else t

    def _draw_period(self) -> None:
        cfg = self.cfg
        self._period = (1.0 / cfg.rate_hz) * (1.0 + float(self._rng.normal(0.0, cfg.rate_jitter_rel)))

    def _revolution(self, s: float, T: float) -> Revolution:
        cfg = self.cfg
        n = max(1, int(math.floor(T * cfg.sample_rate_hz)))
        t = s + np.arange(n) / cfg.sample_rate_hz
        turn = TWO_PI * np.arange(n) / (T * cfg.sample_rate_hz)
        theta = self.start_rad + self.sign * turn  # sensor-frame direction of each sample
        pose = _interp_pose(np.array(self._poses), t)
        ideal = np.full(n, np.inf)
        cap_t = np.array([c[0] for c in self._captures])
        which = np.abs(t[:, None] - cap_t[None, :]).argmin(axis=1)
        for ci in np.unique(which):
            sel = which == ci
            ideal[sel] = self._cast(self._captures[ci], pose[sel], theta[sel])
        theta_w = np.mod(theta + math.pi, TWO_PI) - math.pi
        for a, b in self.occluded:
            lo, hi = np.mod(a + math.pi, TWO_PI) - math.pi, np.mod(b + math.pi, TWO_PI) - math.pi
            inside = (theta_w >= lo) & (theta_w <= hi) if lo <= hi else (theta_w >= lo) | (theta_w <= hi)
            ideal[inside] = np.inf
        noisy = self.noise.apply(ideal, float(theta[0]), float(self.sign * TWO_PI / (T * cfg.sample_rate_hz)), cfg.range_min, cfg.range_max).ranges
        lost = bool(cfg.scan_dropout_prob > 0.0 and self._rng.random() < cfg.scan_dropout_prob)
        ranges = self._bin(theta, noisy)
        ideal_b = self._bin(theta, np.where((ideal >= cfg.range_min) & (ideal <= cfg.range_max), ideal, np.inf))
        jitter = self._rng.normal(0.0, cfg.timestamp_jitter_std_ms) * 1e6 if cfg.timestamp_jitter_std_ms > 0.0 else 0.0
        t_start_ns = int(round(s * 1e9))
        t_end_ns = int(round(t[-1] * 1e9))
        stamp = int(round(t_start_ns + cfg.timestamp_offset_ms * 1e6 + jitter))
        if stamp <= self._last_stamp:
            stamp = self._last_stamp + 1000
        self._last_stamp = stamp
        release = max(stamp, t_end_ns + int(round(cfg.publish_latency_ms * 1e6)))
        self.revolutions += 1
        return Revolution(stamp, release, t_start_ns, t_end_ns, T, ranges.astype(np.float32), ideal_b.astype(np.float32), n, lost)

    def _cast(self, capture: tuple, pose: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Range of every sample ray (origin pose[:, :2], world angle yaw + theta)
        against the surface reconstructed from one capture."""
        _, pts, valid, rr, cap_pose, cap_angle_min = capture
        cfg = self.cfg
        m = len(pts)
        nxt = np.roll(np.arange(m), -1)
        gap = np.linalg.norm(pts[nxt] - pts, axis=1)
        tol = cfg.continuity_abs_m + cfg.continuity_rel * np.maximum(rr, rr[nxt]) * self._cap_step
        seg_ok = valid & valid[nxt] & (gap < tol)
        lone = valid & ~seg_ok & ~seg_ok[np.roll(np.arange(m), 1)]
        # Only capture points within +-window of the ray direction (seen from
        # the capture origin) can be hit: the parallax between the capture and
        # the sample viewpoints is small except for very close surfaces.
        world = pose[:, 2] + theta
        centre = np.round(np.mod(world - cap_pose[2] - cap_angle_min, TWO_PI) / self._cap_step).astype(int)
        w = int(math.ceil(self.window_rad / self._cap_step))
        idx = (centre[:, None] + np.arange(-w, w + 1)[None, :]) % m  # [S, K]
        d = pts[idx] - pose[:, None, :2]  # [S, K, 2]
        ang = np.arctan2(d[..., 1], d[..., 0]) - world[:, None]
        ang = np.mod(ang + math.pi, TWO_PI) - math.pi  # angle of the point relative to the ray
        d1 = pts[nxt[idx]] - pose[:, None, :2]
        a1 = np.mod(np.arctan2(d1[..., 1], d1[..., 0]) - world[:, None] + math.pi, TWO_PI) - math.pi
        # Segment i..i+1 straddles the ray (without wrapping around the back).
        cross = seg_ok[idx] & (np.sign(ang) != np.sign(a1)) & (np.abs(ang - a1) < math.pi / 2)
        ux = np.cos(world)[:, None]
        uy = np.sin(world)[:, None]
        e = d1 - d
        den = ux * e[..., 1] - uy * e[..., 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            dist = (d[..., 0] * e[..., 1] - d[..., 1] * e[..., 0]) / den
        dist = np.where(cross & np.isfinite(dist) & (dist > 1e-4), dist, np.inf)
        best = dist.min(axis=1)
        # Isolated valid points (no segment on either side) are still surfaces
        # seen by one ray: accept them when the ray passes within half a step.
        lone_k = lone[idx]
        if lone_k.any():
            r = np.linalg.norm(d, axis=-1)
            cand = np.where(lone_k & (np.abs(ang) < 0.5 * self._cap_step), r, np.inf).min(axis=1)
            best = np.minimum(best, cand)
        return best

    def _bin(self, theta: np.ndarray, r: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        inc = TWO_PI / cfg.beams
        pos = (np.mod(theta - cfg.angle_min_rad, TWO_PI)) / inc
        idx = np.round(pos).astype(int) % cfg.beams
        out = np.full(cfg.beams, np.inf)
        usable = np.isfinite(r) & (r >= cfg.range_min) & (r <= cfg.range_max)
        if cfg.binning == "nearest":
            off = np.abs(pos - np.round(pos))
            order = np.argsort(-off)  # nearest to the centre written last
            for k in order[usable[order]]:
                out[idx[k]] = r[k]
        else:
            np.minimum.at(out, idx[usable], r[usable])
        # A bin with only below-range returns reports -inf (REP-117).
        below = np.isfinite(r) & (r < cfg.range_min)
        empty = ~np.isfinite(out)
        low = np.zeros(cfg.beams, dtype=bool)
        low[idx[below]] = True
        out[empty & low] = -np.inf
        return out


def rolling_config_from_profile(lidar: dict, seed: int) -> RollingLidarConfig:
    """RollingLidarConfig of a sensors.yaml LiDAR block. Batch profiles
    (A/B/C) become one sample per beam, counter-clockwise from angle_min."""
    beams = int(lidar["beams"])
    rate = float(lidar["rate_hz"])
    ts = lidar.get("timestamp", {})
    fields = {k: v for k, v in lidar.get("noise", {}).items()}
    noise = LidarNoiseConfig(enabled=True, **fields)
    return RollingLidarConfig(
        rate_hz=rate, rate_jitter_rel=float(lidar.get("rate_jitter_rel", 0.0)),
        sample_rate_hz=float(lidar.get("sample_rate_hz", beams * rate)),
        direction=str(lidar.get("scan_direction", "ccw")),
        start_angle_deg=float(lidar.get("scan_start_angle_deg", lidar.get("angle_min_deg", -180.0))),
        beams=beams, angle_min_rad=math.radians(float(lidar.get("angle_min_deg", -180.0))),
        binning=str(lidar.get("binning", "nearest")),
        occluded_sectors_deg=tuple(tuple(float(v) for v in s) for s in lidar.get("occluded_sectors_deg", [])),
        range_min=float(lidar["range_min_m"]), range_max=float(lidar["range_max_m"]),
        timestamp_offset_ms=float(ts.get("offset_ms", 0.0)), timestamp_jitter_std_ms=float(ts.get("jitter_std_ms", 0.0)),
        publish_latency_ms=float(lidar.get("publish_latency_ms", 0.0)),
        scan_dropout_prob=float(lidar.get("scan_dropout_prob", 0.0)), noise=noise, seed=int(seed),
    )
