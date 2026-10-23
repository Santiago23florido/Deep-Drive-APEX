"""The command the race driver publishes, in the units the ESC and servo take.

This is the whole interface between the autonomy and the hardware, and it is
deliberately the dullest arithmetic in the repository: one affine map per
direction and one bicycle inverse. Everything that *looks* like control -- the
deadband, the launch, the gain, the actuator lag -- is identified and inverted
by ``apex_race.core.plant_id`` / ``adaptive`` while the car drives, and a bridge
that also tried to compensate it would be identified instead of the car.

Why the legacy mapping cannot stay (``cmd_vel_to_apex_actuation_node``, profile
``legacy``): it steps to ``min_effective_speed_pct`` 17 % at any command above
1e-4 m/s, then applies an exponential ramp over 0.70 s, a 20 % launch boost for
0.35 s and an 18 %/s ramp-down. The delivered command is therefore a function of
the command *history*, so the window average the identifier builds
(``plant_id._window``) measures the bridge's ramp, not the car. On top of that
``17 -> 25 %`` is an 8-point dynamic range for 0 -> 0.24 m/s, which is nearly
on/off, and 0.24 m/s is below every identification gate and below the odometry
network's 0.3 m/s training floor.

The gauge
---------
``esc_zero_pct`` is a *constant offset*, not deadband compensation, and it is set
deliberately **below** the believed start threshold. It makes the bridge's input
unit be m/s in the same gauge the simulation uses, which is what makes
``vehicle_real.yaml``'s ``esc.min_effective_speed_mps`` the right prior for the
identified threshold. What is left over -- ``(threshold_pct - esc_zero_pct) /
pct_per_mps`` -- is a small residual deadband the identifier brackets from the
launches (``plant_id._update_dead``). Being below the threshold rather than above
it is the safe side of the error: a calibration mistake leaves the car still,
never lurching.

``max_pct`` is a hard ceiling kept separate from the gain. That is necessary,
not cosmetic: ``AdaptiveShaper._speed`` caps its output at ``2.0 + 1.5*|v_des|``
m/s, which at 0.9 m/s is 3.35 m/s, and without a ceiling of its own that would
map to 125 % and clip only at the ESC's full throttle.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SpeedMap:
    """``m/s -> ESC percent``, affine in each direction, with a hard ceiling."""

    esc_zero_pct: float = 15.0
    pct_per_mps: float = 33.0
    max_pct: float = 45.0
    esc_zero_rev_pct: float = 18.0
    rev_pct_per_mps: float = 33.0
    max_rev_pct: float = 33.0
    cmd_epsilon_mps: float = 0.005  # below this the command is a zero, not a creep

    def __post_init__(self) -> None:
        if self.pct_per_mps <= 0.0 or self.rev_pct_per_mps <= 0.0:
            raise ValueError("the speed gain must be positive")
        if self.max_pct <= self.esc_zero_pct or self.max_rev_pct <= self.esc_zero_rev_pct:
            raise ValueError("the ceiling must leave room above the gauge offset")
        if self.esc_zero_pct < 0.0 or self.esc_zero_rev_pct < 0.0:
            raise ValueError("the gauge offsets must not be negative")

    def pct(self, u_mps: float) -> float:
        """ESC percent for a commanded speed [m/s]. Signed: negative is reverse."""
        u = float(u_mps)
        if abs(u) < self.cmd_epsilon_mps:
            return 0.0
        if u > 0.0:
            return min(self.max_pct, self.esc_zero_pct + self.pct_per_mps * u)
        return -min(self.max_rev_pct, self.esc_zero_rev_pct + self.rev_pct_per_mps * (-u))

    def u_for_pct(self, pct: float) -> float:
        """Inverse of :meth:`pct`, for reports and for reading a measurement back."""
        p = float(pct)
        if p == 0.0:
            return 0.0
        if p > 0.0:
            return (min(p, self.max_pct) - self.esc_zero_pct) / self.pct_per_mps
        return -(min(-p, self.max_rev_pct) - self.esc_zero_rev_pct) / self.rev_pct_per_mps

    def full_scale_mps(self) -> float:
        """Commanded speed at the forward ceiling."""
        return (self.max_pct - self.esc_zero_pct) / self.pct_per_mps

    def full_scale_reverse_mps(self) -> float:
        return (self.max_rev_pct - self.esc_zero_rev_pct) / self.rev_pct_per_mps

    def residual_deadband_mps(self, threshold_pct: float) -> float:
        """Command that produces no motion if the ESC starts at ``threshold_pct``."""
        return max(0.0, (float(threshold_pct) - self.esc_zero_pct) / self.pct_per_mps)

    def describe(self) -> dict[str, float]:
        """What the drive-bridge status publishes, so a pre-flight can assert it."""
        return {"esc_zero_pct": self.esc_zero_pct, "pct_per_mps": self.pct_per_mps, "max_pct": self.max_pct,
                "esc_zero_rev_pct": self.esc_zero_rev_pct, "rev_pct_per_mps": self.rev_pct_per_mps,
                "max_rev_pct": self.max_rev_pct, "full_scale_mps": round(self.full_scale_mps(), 4),
                "full_scale_reverse_mps": round(self.full_scale_reverse_mps(), 4)}


def steer_deg_from_twist(
    v_mps: float,
    yaw_rate_rps: float,
    wheelbase_m: float,
    max_steering_deg: float,
    v_min_mps: float = 0.02,
    *,
    max_left_deg: float = 0.0,
    max_right_deg: float = 0.0,
) -> tuple[float | None, bool]:
    """``(steering angle [deg] or None, saturated)`` from a Twist.

    The driver publishes ``angular.z = v * tan(steer) / wheelbase``, so the
    inverse ``atan(L * wz / v)`` returns exactly the angle it asked for and the
    wheelbase cancels -- the gauge property the adaptive layer relies on, and
    the reason the identified ``gain`` means the same thing on both sides.

    ``None`` means the request carries no steering information (the car is being
    commanded to a stop), and the servo must then **hold** its angle rather than
    centre: snapping to centre at the end of every stop is a steering input
    nobody asked for, and the simulated bridge holds
    (``sim_actuation_node``). Above ``max_steering_deg`` the angle is clamped and
    the flag is raised -- the real car logged 46-48 deg requests against an
    18 deg mechanical limit, which the duty saturation hid.
    """
    v = float(v_mps)
    if abs(v) < float(v_min_mps):
        return None, False
    deg = math.degrees(math.atan(float(wheelbase_m) * float(yaw_rate_rps) / v))
    limit = abs(float(max_steering_deg))
    left, right = abs(float(max_left_deg)) or limit, abs(float(max_right_deg)) or limit
    clamped = max(-right, min(left, deg))
    return clamped, abs(deg - clamped) > 1e-9
