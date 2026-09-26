"""Configuration of the car-side race driver (ROS-independent).

The race driver emulates the real car on a track it has never seen: lap 1 is
driven reactively from the LiDAR while slam_toolbox builds the map, then the
car plans the following laps on its own map. Everything the car may know is
here:

* its own vehicle limits (wheelbase, footprint, steering, ESC/servo), the
  sections ``model`` / ``esc`` / ``servo`` / ``controller`` of
  ``tools/pose_dataset/config/vehicle.yaml`` (never ``calibration``: that is
  the maximum speed measured per track);
* one generic rule of the tracks: a lane is at least ``w_min_m`` wide (the
  2D LiDAR does not see curbs below its scan plane, so a lane edge may be
  invisible; the rule bounds how far the car may go from the wall it sees);
* margins, speeds and accelerations chosen for the car, not for the track.

Every dataclass is exposed field by field as ROS parameters
(``nodes/_common.ConfigParameters``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


@dataclass
class VehicleLimits:
    wheelbase_m: float = 0.30
    length_m: float = 0.46  # chassis, centred on base_link
    width_m: float = 0.32  # outer faces of the wheels
    steer_max_deg: float = 18.0
    right_ratio: float = 0.96  # the servo linkage reaches 96 % of the command on right turns
    steer_plan_deg: float = 16.0  # planning keeps a steering reserve for the tracking error
    a_acc_mps2: float = 1.8
    a_dec_mps2: float = 2.4
    servo_rate_dps: float = 70.0
    rear_offset_m: float = -0.15  # rear axle in base_link (pure-pursuit reference point)
    lookahead_min_m: float = 0.55
    lookahead_time_s: float = 0.32
    lookahead_base_m: float = 0.35

    @property
    def kappa_max(self) -> float:
        """Largest path curvature the planners may use [1/m]."""
        return math.tan(math.radians(self.steer_plan_deg)) / self.wheelbase_m

    @property
    def circle_offsets(self) -> tuple[float, float, float]:
        """Centres (x in base_link) of the 3 circles covering the footprint."""
        d = self.length_m / 3.0
        return (-d, 0.0, d)

    @property
    def circle_radius(self) -> float:
        return math.hypot(self.length_m / 6.0, self.width_m / 2.0)

    def controller_cfg(self) -> dict[str, float]:
        return {"lookahead_min_m": self.lookahead_min_m, "lookahead_time_s": self.lookahead_time_s, "lookahead_base_m": self.lookahead_base_m}


def vehicle_limits(vcfg: dict[str, Any], steer_plan_deg: float = 16.0) -> VehicleLimits:
    """The car's own limits from the vehicle description (vehicle.yaml)."""
    m, esc, servo, ctl = vcfg["model"], vcfg["esc"], vcfg["servo"], vcfg["controller"]
    return VehicleLimits(
        wheelbase_m=float(m["wheelbase_m"]), length_m=float(m["chassis_length_m"]), width_m=float(m["footprint_width_m"]),
        steer_max_deg=float(m["steering_limit_deg"]), right_ratio=float(servo["right_ratio"]), steer_plan_deg=float(steer_plan_deg),
        a_acc_mps2=float(esc["accel_limit_mps2"]), a_dec_mps2=float(esc["decel_limit_mps2"]), servo_rate_dps=float(servo["rate_limit_deg_per_s"]),
        lookahead_min_m=float(ctl["lookahead_min_m"]), lookahead_time_s=float(ctl["lookahead_time_s"]), lookahead_base_m=float(ctl["lookahead_base_m"]),
    )


@dataclass
class ReactiveConfig:
    """Lap-1 driving from the local LiDAR cloud (no map, no path)."""

    w_min_m: float = 1.5  # generic lane rule: a lane is at least this wide
    d_follow_m: float = 0.75  # distance kept to the followed wall when only one wall is seen
    wall_margin_m: float = 0.20  # footprint margin to the followed wall
    blind_margin_m: float = 0.10  # footprint margin to the invisible edge implied by w_min
    both_max_width_m: float = 2.8  # two walls farther apart: one of them is not a lane edge
    c_hard_m: float = 0.08  # footprint clearance below which a candidate is blocked
    c_soft_m: float = 0.30  # clearance below which a candidate is penalised
    v_explore_mps: float = 1.5
    v_min_mps: float = 0.3  # the odometry network was trained above ~0.3 m/s
    v_blind_mps: float = 0.8
    a_lat_mps2: float = 1.5
    a_acc_mps2: float = 1.0
    a_dec_mps2: float = 1.5
    steer_rate_dps: float = 40.0
    accum_window_s: float = 2.0
    accum_distance_m: float = 3.0
    voxel_m: float = 0.05
    range_min_m: float = 0.15
    range_max_m: float = 6.0
    body_margin_m: float = 0.04  # returns within the car body (+ this margin) are the car itself
    lidar_period_s: float = 1.0 / 13.0  # one revolution of the A2M8 (nominal sensor data)
    cluster_eps_m: float = 0.20
    wall_min_length_m: float = 1.8  # decoration boxes reach 1.66 m (diagonal), pillars 0.3 m
    wall_min_points: int = 15
    side_x_min_m: float = -0.8  # "beside the car" window (base_link x)
    side_x_max_m: float = 1.2
    wall_max_lateral_m: float = 1.8
    assoc_m: float = 0.30  # the same wall again: nearest point within this distance
    wall_end_switch_m: float = 2.0  # the followed wall ends closer than this ahead: switch to a wall that continues
    wall_ended_ahead_m: float = 0.3  # a followed wall reaching less than this ahead of base_link has ended
    switch_hold_m: float = 1.5  # travel after a switch before the next early switch
    switch_absent_m: float = 0.4  # travel without the followed wall before switching
    blind_hold_m: float = 2.0  # travel without any wall before stopping
    start_relax_m: float = 2.0  # the w_min bound relaxes from the start pose over this travel
    n_k1: int = 21
    n_k2: int = 11
    w_off: float = 1.0
    off_scale_m: float = 0.25
    w_bound: float = 50.0
    w_head: float = 0.5
    w_smooth: float = 0.3
    w_clear: float = 1.0
    w_block: float = 6.0
    w_guide: float = 2.0
    local_path_lookahead_m: float = 0.7


@dataclass
class ClosureConfig:
    arm_distance_m: float = 20.0  # travel before a crossing may count (a lap is longer)
    arm_time_s: float = 15.0
    lateral_tol_m: float = 1.5
    heading_tol_deg: float = 45.0
    return_radius_m: float = 2.0
    overlap_m: float = 6.0  # driven past the start so slam_toolbox can close the loop
    map_fresh_wait_s: float = 3.0


@dataclass
class PlannerConfig:
    mode: str = "min_curvature"  # min_curvature | centre
    ds_m: float = 0.10
    smooth_sigma_m: float = 0.4
    seam_blend_m: float = 3.0
    ray_max_m: float = 3.0
    occupied_threshold: int = 65
    unknown_is_obstacle: bool = True
    speck_max_cells: int = 2  # isolated occupied specks of at most this many cells are noise
    unknown_min_area_m2: float = 0.5  # smaller unknown patches are gaps between rays, not unexplored space
    wall_component_min_m: float = 2.0
    wall_detect_max_m: float = 1.8
    wall_margin_m: float = 0.20
    blind_margin_m: float = 0.10
    obstacle_margin_m: float = 0.15
    none_half_width_m: float = 0.25
    bound_erosion_m: float = 0.3
    lambda_centre: float = 1e-3
    lambda_jerk: float = 0.01
    outer_iters: int = 3
    verify_clearance_m: float = 0.10
    max_tighten_iters: int = 5
    seam_warn_m: float = 0.30
    seam_warn_deg: float = 5.0


@dataclass
class RaceConfig:
    v_max_mps: float = 3.5
    a_lat_mps2: float = 3.0
    a_acc_mps2: float = 1.5
    a_dec_mps2: float = 2.0
    v_tight_base_mps: float = 1.2
    v_tight_gain: float = 5.0  # cap = base + gain * corridor width [m]
    handover_max_lat_m: float = 0.3
    handover_max_head_deg: float = 15.0
    guard_clearance_m: float = 0.06
    guard_drop_m: float = 0.18  # ... and this much below the clearance the plan expected there (live cloud vs map noise)
    guard_hold_s: float = 1.0
    finish_margin_m: float = 1.0  # braking target past the start line on the last lap


@dataclass
class CorrectionConfig:
    """Rate limits of the SLAM correction (map -> odom) applied to the control."""

    max_trans_rate_mps: float = 0.4
    max_rot_rate_dps: float = 5.0
    tau_s: float = 0.25


@dataclass
class RaceDriverConfig:
    vehicle: VehicleLimits = field(default_factory=VehicleLimits)
    reactive: ReactiveConfig = field(default_factory=ReactiveConfig)
    closure: ClosureConfig = field(default_factory=ClosureConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    race: RaceConfig = field(default_factory=RaceConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
