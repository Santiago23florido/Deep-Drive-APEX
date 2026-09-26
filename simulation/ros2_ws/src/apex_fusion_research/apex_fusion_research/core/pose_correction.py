"""Rate limiting of the SLAM correction applied to the control (ROS-independent).

slam_toolbox corrects the learned odometry by publishing ``map -> odom``;
each accepted scan match may move it by 5-20 cm at once. Fed directly to the
controller, such a jump is a step in the lateral error and a steering kick.
The filter moves the *car pose* implied by the correction towards the raw one
at a bounded rate (first order, then translation / rotation rate limits),
centred on the car so that a small yaw correction far from the map origin
does not become a large translation.
"""

from __future__ import annotations

import math

from .map_metrics import se2_compose, se2_inverse
from .race_config import CorrectionConfig


class CorrectionFilter:
    def __init__(self, cfg: CorrectionConfig) -> None:
        self.cfg = cfg
        self.applied: tuple[float, float, float] | None = None
        self.t: float | None = None

    def update(self, t: float, raw: tuple[float, float, float], odom_now: tuple[float, float, float]) -> tuple[float, float, float]:
        """Applied ``map -> odom`` after a step to time ``t`` [s]."""
        if self.applied is None or self.t is None:
            self.applied, self.t = tuple(raw), t
            return self.applied
        dt = max(0.0, t - self.t)
        self.t = t
        p_raw = se2_compose(raw, odom_now)
        p_app = se2_compose(self.applied, odom_now)
        dx, dy = p_raw[0] - p_app[0], p_raw[1] - p_app[1]
        dyaw = math.atan2(math.sin(p_raw[2] - p_app[2]), math.cos(p_raw[2] - p_app[2]))
        gain = min(1.0, dt / max(self.cfg.tau_s, 1e-6))
        dx, dy, dyaw = gain * dx, gain * dy, gain * dyaw
        step = math.hypot(dx, dy)
        lim = self.cfg.max_trans_rate_mps * dt
        if step > lim:
            dx, dy = dx * lim / step, dy * lim / step
        lim_r = math.radians(self.cfg.max_rot_rate_dps) * dt
        dyaw = max(-lim_r, min(lim_r, dyaw))
        p_new = (p_app[0] + dx, p_app[1] + dy, p_app[2] + dyaw)
        self.applied = se2_compose(p_new, se2_inverse(odom_now))
        return self.applied
