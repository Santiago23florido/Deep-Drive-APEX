"""Driver model: pure-pursuit steering and planned speed profiles.

The driver has access to the exact pose (like a well-localised autonomous
stack); this only decides *where* the car goes. The trajectory itself comes
from Gazebo's rigid-body dynamics with the car's actuators, never from
teleporting the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .tracks import Path2D, Track, path_from_points

PACKAGE_DIR = Path(__file__).resolve().parent


def load_motion_profiles(path: Path | None = None) -> dict[str, dict[str, Any]]:
    return yaml.safe_load((path or PACKAGE_DIR / "config" / "motion.yaml").read_text(encoding="utf-8"))["profiles"]


def _uniform(rng: np.random.Generator, value: Any) -> float:
    if isinstance(value, (list, tuple)):
        return float(rng.uniform(float(value[0]), float(value[1])))
    return float(value)


# ------------------------------------------------------------------ path setup
def driving_path(track: Track, direction: int, start_s: float, lateral_offset: float, weave: dict[str, float] | None = None) -> Path2D:
    """Reference path for one run: direction (+1 CCW as designed, -1 reversed),
    arc-length origin at ``start_s`` and a constant lateral offset."""
    ref = track.reference
    xy = ref.xy + lateral_offset * ref.normal
    base = path_from_points(xy)
    if direction < 0:
        base = base.reversed()
    base = base.rotated_start(start_s)
    if weave:
        amp, period = weave["amplitude"], weave["period"]
        kappa = np.abs(base.curvature)
        k_smooth = np.convolve(np.concatenate((kappa[-50:], kappa, kappa[:50])), np.ones(101) / 101, mode="same")[50:-50]
        scale = np.clip(1.0 - 2.5 * k_smooth, 0.0, 1.0)
        # Fade in over the first 3 m so the car starts on the reference.
        fade = np.clip(base.s / 3.0, 0.0, 1.0) * np.clip((base.length - base.s) / 3.0, 0.0, 1.0)
        d = amp * scale * fade * np.sin(2.0 * math.pi * base.s / period)
        base = path_from_points(base.xy + d[:, None] * base.normal)
    return base


# ------------------------------------------------------------ speed planning
@dataclass
class SpeedPlan:
    s: np.ndarray  # arc length grid over all laps [m]
    v: np.ndarray  # target speed [m/s]
    stops: list[tuple[float, float]] = field(default_factory=list)  # (s, dwell_s)
    description: dict[str, Any] = field(default_factory=dict)

    def target(self, s: float) -> float:
        return float(np.interp(s, self.s, self.v, right=0.0))

    def planned_duration(self) -> float:
        """Driving time up to the finish line plus the planned stops."""
        total = float(self.description.get("total_distance_m", self.s[-1]))
        keep = self.s <= total
        v = np.maximum(self.v[keep], 0.05)
        ds = np.diff(self.s[keep])
        return float(np.sum(ds / (0.5 * (v[1:] + v[:-1])))) + sum(d for _, d in self.stops)


def _forward_backward(v_cap: np.ndarray, ds: float, a_acc: float, a_dec: float, v_start: float = 0.0, v_end: float = 0.0, zero_idx: list[int] | None = None) -> np.ndarray:
    v = v_cap.copy()
    for i in zero_idx or []:
        v[i] = 0.0
    v[0] = min(v[0], v_start)
    for i in range(1, len(v)):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2.0 * a_acc * ds))
    v[-1] = min(v[-1], v_end)
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * a_dec * ds))
    return v


def plan_speed(profile: dict[str, Any], path: Path2D, laps: float, v_ref: float, rng: np.random.Generator, ds: float = 0.05) -> SpeedPlan:
    total = laps * path.length
    stop_margin = 6.0  # distance available to brake after the finish line
    s = np.arange(0.0, total + stop_margin + ds, ds)
    kappa = np.abs(path.interp_scalar(path.curvature, np.mod(s, path.length)))
    # Smooth the curvature over ~0.5 m (the car anticipates curves).
    k = np.convolve(kappa, np.ones(11) / 11, mode="same")
    kind = profile["kind"]
    desc: dict[str, Any] = {"kind": kind, "v_ref_mps": v_ref, "total_distance_m": total}
    stops: list[tuple[float, float]] = []
    finish = int(round(total / ds))
    if kind in ("constant", "weave"):
        v_c = float(profile["fraction"]) * v_ref
        cap = np.full_like(s, v_c)
        a_acc, a_dec = 1.8, 2.4  # the ESC limits shape the start and the final stop
        desc["cruise_mps"] = v_c
    elif kind == "variable":
        v_straight = _uniform(rng, profile["straight_fraction"]) * v_ref
        a_lat = _uniform(rng, profile["lateral_accel_mps2"])
        v_min = float(profile["min_fraction"]) * v_ref
        cap = np.minimum(v_straight, np.sqrt(a_lat / np.maximum(k, 1e-6)))
        seg_cap = np.empty_like(s)
        pos = 0
        seg_lengths = []
        while pos < len(s):
            n = max(1, int(_uniform(rng, profile["segment_length_m"]) / ds))
            scale = float(rng.choice(profile["segment_scales"]))
            seg_cap[pos : pos + n] = scale
            seg_lengths.append((round(pos * ds, 2), scale))
            pos += n
        cap = np.maximum(np.minimum(cap, v_straight * seg_cap), v_min)
        a_acc = _uniform(rng, profile["a_acc_mps2"])
        a_dec = _uniform(rng, profile["a_dec_mps2"])
        desc.update(straight_mps=v_straight, lateral_accel_mps2=a_lat, a_acc_mps2=a_acc, a_dec_mps2=a_dec, segments=seg_lengths)
    elif kind == "stop_and_go":
        v_cruise = _uniform(rng, profile["cruise_fraction"]) * v_ref
        a_lat = float(profile["lateral_accel_mps2"])
        cap = np.minimum(v_cruise, np.sqrt(a_lat / np.maximum(k, 1e-6)))
        a_acc = float(profile["a_acc_mps2"])
        a_dec = float(profile["a_dec_mps2"])
        n_laps = max(1, int(math.ceil(laps)))
        spacing = float(profile["min_stop_spacing_m"])
        for lap in range(n_laps):
            n_stops = int(rng.integers(int(profile["stops_per_lap"][0]), int(profile["stops_per_lap"][1]) + 1))
            chosen: list[float] = []
            tries = 0
            while len(chosen) < n_stops and tries < 500:
                tries += 1
                cand = lap * path.length + float(rng.uniform(4.0, path.length - 3.0))
                if cand > total - 4.0:
                    continue
                if all(abs(cand - c) > spacing for c in chosen):
                    chosen.append(cand)
            for c in sorted(chosen):
                stops.append((c, _uniform(rng, profile["dwell_s"])))
        desc.update(cruise_mps=v_cruise, a_acc_mps2=a_acc, a_dec_mps2=a_dec)
    else:
        raise ValueError(f"unknown motion kind {kind!r}")
    zero_idx = [int(round(sc / ds)) for sc, _ in stops]
    cap[finish:] = 0.0
    v = _forward_backward(cap, ds, a_acc, a_dec, zero_idx=zero_idx)
    desc["stops"] = [(round(a, 2), round(b, 2)) for a, b in stops]
    return SpeedPlan(s, v, stops, desc)


# ---------------------------------------------------------------- pure pursuit
class PurePursuit:
    def __init__(self, path: Path2D, wheelbase: float, cfg: dict[str, Any], gain_scale: float = 1.0) -> None:
        self.path = path
        self.wheelbase = wheelbase
        self.l_min = float(cfg["lookahead_min_m"])
        self.l_time = float(cfg["lookahead_time_s"]) * gain_scale
        self.l_base = float(cfg["lookahead_base_m"]) * gain_scale
        self.idx = 0
        self.s_total = 0.0  # unwrapped progress along the path [m]
        self._last_s = 0.0
        n = len(path.s)
        self._window = max(20, int(2.0 / max(path.length / n, 1e-3)))

    def reset(self, x: float, y: float) -> None:
        """Restart the progress at the path point nearest to (x, y) (global search)."""
        self.idx = int(np.argmin(np.hypot(self.path.xy[:, 0] - x, self.path.xy[:, 1] - y)))
        self._last_s = float(self.path.s[self.idx])
        self.s_total = 0.0

    def locate(self, x: float, y: float) -> tuple[float, float]:
        """Update progress with the reference point; return (s_total, lateral error)."""
        n = len(self.path.s)
        ids = (self.idx + np.arange(-self._window // 4, self._window)) % n
        d = np.hypot(self.path.xy[ids, 0] - x, self.path.xy[ids, 1] - y)
        k = int(np.argmin(d))
        self.idx = int(ids[k])
        s_now = float(self.path.s[self.idx])
        ds = s_now - self._last_s
        if ds < -0.5 * self.path.length:
            ds += self.path.length
        elif ds > 0.5 * self.path.length:
            ds -= self.path.length
        self.s_total += ds
        self._last_s = s_now
        h = self.path.heading[self.idx]
        px, py = self.path.xy[self.idx]
        lateral = -math.sin(h) * (x - px) + math.cos(h) * (y - py)
        return self.s_total, lateral

    def steering(self, x: float, y: float, yaw: float, speed: float) -> float:
        """Steering angle [rad] of the equivalent centre wheel."""
        ld = max(self.l_min, self.l_base + self.l_time * max(speed, 0.0))
        tx, ty = self.path.interp(self.path.s[self.idx] + ld)
        alpha = math.atan2(ty - y, tx - x) - yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        return math.atan2(2.0 * self.wheelbase * math.sin(alpha), ld)
