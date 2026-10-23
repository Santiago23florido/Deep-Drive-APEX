"""Use the identified plant: inverse map, lag lead and confidence envelope.

:mod:`plant_id` answers "what does this car actually do with my commands".
This module is the other half: it turns what the tracking *wants* into what has
to be commanded for the car to do it, and it decides how much the car is
allowed to ask of itself while it still does not know what it is.

Four ideas carry the design.

**The command channel is a curvature.** The driver publishes ``linear.x`` and
``angular.z``, and everything downstream (the wheelbase the actuation assumes,
the servo, the linkage, the tyres) only ever sees their ratio. So inverting the
identified map is a division, and the wheelbase cancels: the pure-pursuit angle
``atan(2 L sin a / ld)`` turns back into the geometric curvature ``2 sin a /
ld`` with no ``L`` left in it.

**Inverting a lag is a lead.** The plant delivers a first-order filtered
version of the command, so commanding ``kappa + tau * dkappa/dt`` makes the
*delivered* curvature follow ``kappa``. The derivative is taken on a smoothed
signal and capped, because the lap-1 planner replaces its path at 10 Hz and the
raw derivative of that is a train of spikes.

**No second integrator.** The identified trim already integrates the lateral
bias and the identified gain the longitudinal one, both weighted by the sigma
of the measurement and bounded by a box. Putting a PI loop on top of them is
two integrators on one quantity, which is how limit cycles are built. What is
added is proportional only, for the transients, and it cannot wind up.

**Nothing here raises the loop gain.** With feedback at 10 Hz and about 0.19 s
of latency the lateral loop is already at the limit of what the delay allows
(pure pursuit crosses over near 2.4 rad/s at 3.5 m/s, which is why the pose is
extrapolated and the SLAM correction is rate-limited). When the car is unsure
of itself the answer is to look further ahead, which buys damping, and to ask
for less speed and more margin -- never to push harder.

At the prior the whole chain is numerically the unmodified driver: the
identified model starts *at* the datasheet, so a car that has learned nothing
publishes what it published before, and the asymmetry of the linkage that
``steer_command`` used to divide out is simply the prior of the right-hand
gain. That is the property that makes this reviewable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .plant_id import PlantModel
from .race_config import RaceDriverConfig


@dataclass(frozen=True)
class Envelope:
    """What the car allows itself at the confidence it has reached.

    Every field is between the value the unmodified driver uses and something
    more careful than it, never beyond it.
    """

    v_max_mps: float
    a_lat_mps2: float
    lap1_v_scale: float
    blind_margin_add_m: float
    wall_margin_add_m: float
    soft_margin_add_m: float
    lookahead_add_m: float
    lookahead_add_s: float
    kappa_limit: float
    confidence: float
    kappa_limit_left: float = 0.0
    kappa_limit_right: float = 0.0


class AdaptiveShaper:
    """Commands for the identified plant, at the control rate."""

    def __init__(self, cfg: RaceDriverConfig) -> None:
        self.cfg = cfg
        self.a = cfg.adaptive
        self.reset()

    def reset(self) -> None:
        self.t_prev: float | None = None
        self.k_smooth = 0.0
        self.k_rate = 0.0
        self.launch_since: float | None = None
        self.launch_extra = 0.0
        self.lead = 0.0
        self.v_out = 0.0
        self.kappa_out = 0.0

    # ------------------------------------------------------------ envelope
    def envelope(self, model: PlantModel) -> Envelope:
        a, veh = self.a, self.cfg.vehicle
        c = max(0.0, min(1.0, model.confidence))
        lerp = lambda floor: floor + (1.0 - floor) * c  # noqa: E731
        # The usable saturation is the identified one, in command space, and
        # never more than the nominal steering limit allows.
        nominal = min(veh.curvature_limit(True), veh.curvature_limit(False))
        # _saturation() already divides measured curvature by the identified
        # gain. Dividing again here would shrink wheel command authority just
        # because a better identified gain became larger than one.
        limit = min(nominal, model.kappa_sat)
        limit *= a.sat_cmd_frac + (1.0 - a.sat_cmd_frac) * c
        common = max(a.sat_floor_frac * nominal, limit)
        # Saturation is stored in the conservative side's command units.
        # Scale that conservative estimate to each configured side; never
        # manufacture extra command authority from a larger desired angle.
        left = min(veh.curvature_limit(True), common * veh.curvature_limit(True, planning=True) / veh.kappa_max)
        right = min(veh.curvature_limit(False), common * veh.curvature_limit(False, planning=True) / veh.kappa_max)
        return Envelope(
            v_max_mps=self.cfg.race.v_max_mps * lerp(a.v_floor_frac),
            a_lat_mps2=self.cfg.race.a_lat_mps2 * lerp(a.a_lat_floor_frac),
            lap1_v_scale=lerp(a.lap1_v_floor_frac),
            blind_margin_add_m=a.margin_blind_add_m * (1.0 - c),
            wall_margin_add_m=a.wall_margin_add_m * (1.0 - c),
            soft_margin_add_m=a.soft_margin_add_m * (1.0 - c),
            lookahead_add_m=a.lookahead_add_m * (1.0 - c),
            lookahead_add_s=a.lookahead_add_s * (1.0 - c),
            kappa_limit=common,
            confidence=c,
            kappa_limit_left=left,
            kappa_limit_right=right,
        )

    # --------------------------------------------------------------- shape
    def shape(self, t: float, kappa_des: float, v_des: float, speed: float,
              model: PlantModel, env: Envelope) -> tuple[float, float]:
        """``(speed, curvature)`` to command so the car delivers the request."""
        dt = 0.0 if self.t_prev is None else max(0.0, min(0.1, t - self.t_prev))
        self.t_prev = t
        if not self.a.enabled:
            self.v_out, self.kappa_out = v_des, kappa_des
            return v_des, kappa_des
        self.kappa_out = self._curvature(dt, kappa_des, speed, model, env)
        self.v_out = self._speed(t, dt, v_des, speed, model)
        return self.v_out, self.kappa_out

    def _curvature(self, dt: float, kappa_des: float, speed: float, model: PlantModel, env: Envelope) -> float:
        if dt > 0.0:
            alpha = 1.0 - math.exp(-dt / self.a.lead_smooth_s)
            rate = alpha * (kappa_des - self.k_smooth) / dt
            self.k_smooth += alpha * (kappa_des - self.k_smooth)
            self.k_rate += alpha * (rate - self.k_rate)
        else:
            self.k_smooth, self.k_rate = kappa_des, 0.0
        lead_limit = min(self.a.lead_max, abs(kappa_des)) if self.a.lead_preserve_turn else self.a.lead_max
        self.lead = max(-lead_limit, min(lead_limit, model.tau_steer_s * self.k_rate))
        kappa_cmd = model.kappa_in(kappa_des + self.lead, abs(speed))
        return max(-(env.kappa_limit_right or env.kappa_limit),
                   min(env.kappa_limit_left or env.kappa_limit, kappa_cmd))

    def _speed(self, t: float, dt: float, v_des: float, speed: float, model: PlantModel) -> float:
        a = self.a
        if abs(v_des) < 1e-6:
            self.launch_since, self.launch_extra = None, 0.0
            return 0.0
        u = model.speed_in(v_des)
        authority = a.speed_authority * max(abs(v_des), 0.3)
        u += max(-authority, min(authority, a.kp_speed * (v_des - speed)))
        # The motor does not start below its threshold, so a command meant to
        # move a standing car must clear it; if the car is still standing after
        # launch_hold_s, ramp until it breaks away. That ramp is also how the
        # threshold gets measured, which is why it is bounded in time and in
        # amplitude rather than being an integrator.
        dead = model.dead_for(v_des)
        if abs(speed) < 0.05:
            self.launch_since = t if self.launch_since is None else self.launch_since
            if t - self.launch_since >= a.launch_hold_s:
                self.launch_extra = min(a.launch_ramp_mps2 * 0.5, self.launch_extra + a.launch_ramp_mps2 * dt)
            u = math.copysign(max(abs(u), dead + 0.03), v_des)
        else:
            self.launch_since, self.launch_extra = None, 0.0
        u += math.copysign(self.launch_extra, v_des)
        cap = 2.0 + 1.5 * abs(v_des)
        return max(-cap, min(cap, u))

    def diag(self) -> dict:
        return {"lead": self.lead, "v_out": self.v_out, "kappa_out": self.kappa_out,
                "launch_extra": self.launch_extra}
