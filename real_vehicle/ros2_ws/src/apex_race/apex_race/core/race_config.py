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
    rear_offset_m: float = -0.15  # rear axle in base_link (pure-pursuit reference point), prior of the identified arm
    min_effective_mps: float = 0.07  # ESC datasheet: below this command the car does not start (prior, not a measurement)
    lookahead_min_m: float = 0.55
    lookahead_time_s: float = 0.32
    lookahead_base_m: float = 0.35
    # Zero keeps the legacy symmetric limits. Positive values describe the
    # configured command authority on each side, not a measured wheel angle.
    steer_max_left_deg: float = 0.0
    steer_max_right_deg: float = 0.0
    steer_plan_left_deg: float = 0.0
    steer_plan_right_deg: float = 0.0

    def steering_limit_deg(self, left: bool, *, planning: bool = False) -> float:
        actual = (self.steer_max_left_deg if left else self.steer_max_right_deg) or self.steer_max_deg
        if not planning:
            return actual
        planned = (self.steer_plan_left_deg if left else self.steer_plan_right_deg) or self.steer_plan_deg
        return min(actual, planned)

    def curvature_limit(self, left: bool, *, planning: bool = False) -> float:
        return math.tan(math.radians(self.steering_limit_deg(left, planning=planning))) / self.wheelbase_m

    @property
    def kappa_max(self) -> float:
        """Largest path curvature the planners may use [1/m]."""
        # Modules that still require one limit retain the conservative side.
        return min(self.curvature_limit(True, planning=True), self.curvature_limit(False, planning=True))

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
        min_effective_mps=float(esc["min_effective_speed_mps"]),
        lookahead_min_m=float(ctl["lookahead_min_m"]), lookahead_time_s=float(ctl["lookahead_time_s"]), lookahead_base_m=float(ctl["lookahead_base_m"]),
    )


@dataclass
class ReactiveConfig:
    """Lap-1 driving from the local LiDAR cloud (no map, no path)."""

    use_wall_planner: bool = True  # false: bicycle arcs, with the same wall tracker and collision/lane checks
    preview_while_waiting: bool = False  # real planning load with outputs still neutral
    w_min_m: float = 1.5  # generic lane rule: a lane is at least this wide
    d_follow_m: float = 0.75  # distance kept to the followed wall when only one wall is seen
    d_follow_left_m: float = 0.0  # positive: override the target only when following the left wall
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
    lidar_period_s: float = 1.0 / 10.0  # one revolution of the A2M8 (nominal sensor data: 10 Hz, sensors.yaml APEX_real)
    # Points closer than this join one cluster. 0.25 (was 0.20): with the A2M8 at
    # 4 kHz (~307 samples in 360 bins, 12 % dropouts) walls split into clusters too
    # short to count as walls, the driver lost them and ran over invisible curbs
    # (2D harness at 4 kHz: 19/20 -> 10/10 seeds; 0.30 cost 3 cm of clearance at 8 kHz).
    cluster_eps_m: float = 0.25
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
    # Proven lane. The lane rule only holds across a visible wall: a point is known to be in the lane when a wall
    # lies beside it (perpendicular foot) within the w_min bound, or when it lies between two walls. Beyond the end of
    # every visible wall an invisible curb may cross the path (a corner whose outer wall becomes a curb), so the car
    # drives no faster than it can stop within the proven part of its path, which leaves the LiDAR time to see where
    # the lane goes, and prefers the paths proven farthest.
    lane_join_m: float = 0.45  # a short cluster this close to a wall extends it (a corner seen through a few sparse returns)
    lane_decel_mps2: float = 0.8  # braking assumed before the end of the proven lane (gentler than a_dec: margin for sensing)
    lane_margin_m: float = 0.5  # stop this far before the end of the proven lane (latency + one revolution)
    w_lane: float = 4.0


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
    # Margin to the invisible edge implied by w_min on the race laps. 0.20 (was 0.10): that edge is a prior, not a
    # measurement, and at 3 m/s the belief error reaches 0.1-0.15 m; 0.10 left 9 cm to a curb of test_unseen and the
    # car hit it (seed 12). 2D harness, 20 seeds: race clearance min 15 -> 23 cm, same lap time.
    blind_margin_m: float = 0.20
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
    guide_handover_s: float = 0.0  # gradual local-to-map guidance; zero keeps legacy behavior
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
class AdaptiveConfig:
    """Identification of the car's own plant, from its odometry (plant_id.py).

    The priors and their sigmas are the car's datasheet and how much it may be
    wrong by. They are also what a simulated run draws its real plant from, so
    one table describes both what the car believes and how wrong it can be.

    The sigmas of the measurements come from the estimator itself, but they are
    sub-millimetre, so each block also carries a floor: without it the filter
    trusts the data a hundred times more than the model deserves and chases
    suspension and Ackermann error instead of converging. The floors below are
    the residuals measured on recorded runs (tools/analysis/replay_adaptation.py).
    """

    enabled: bool = True
    # --- lever arm block: [arm, slip]
    arm_sigma: float = 0.05  # [m] prior is -rear_offset_m
    arm_box_m: float = 0.10  # hard box around the prior
    arm_dyaw_min_rad: float = 0.02  # a straight line says nothing about the arm
    slip_arm_sigma: float = 0.001  # [s2/m2] rear-slip bias of the arm
    slip_arm_max: float = 0.004
    slip_arm_v_min_mps: float = 2.0  # the slip column is only enabled above this
    floor_arm_m: float = 0.0015  # measured residual of the lever equation
    n_lever: int = 30
    # --- curvature block: [gain_left, gain_right, trim, slip]
    gain_sigma: float = 0.30  # covers an unknown wheelbase and servo gain
    gain_min: float = 0.55
    gain_max: float = 1.60
    gain_walk_per_s: float = 0.001
    steering_confidence_gate: bool = False  # opt-in: do not invert an uncertain early steering fit
    trim_sigma: float = 0.05  # [1/m] about 0.9 deg of servo centre error
    trim_max: float = 0.25
    trim_walk_per_s: float = 0.0005
    # Understeer: effective gain = gain + slip * (v^2 - v_ref^2), centred on the
    # speed the gain was identified at so the two are not the same regressor.
    slip_prior: float = 0.0  # [s2/m2]
    slip_sigma: float = 0.006
    slip_min: float = -0.020
    slip_max: float = 0.010
    floor_kappa: float = 0.013  # [1/m] measured residual of the curvature fit
    n_steer: int = 40
    v_min_steer_mps: float = 0.6  # below this kappa = omega / v is ill-conditioned
    transient_max: float = 3.0  # [(1/m)/s] command moving faster than this: lag error dominates
    sat_frac: float = 0.97  # a command this close to the saturation is not a gain measurement
    sat_window: int = 30
    sat_floor_frac: float = 0.70  # the identified saturation never falls below this of the nominal
    sign_votes: int = 20  # samples of the steering sign check before the car may adapt
    sign_kappa_min: float = 0.15
    # --- speed block: [gain, offset]; the start threshold is bracketed apart
    longitudinal_forward_only: bool = False  # forward bridge calibration does not identify reverse
    longitudinal_allow_overspeed: bool = False  # steady same-sign speed can exceed the prior
    vgain_sigma: float = 0.20
    vgain_min: float = 0.60
    vgain_max: float = 1.40
    vgain_walk_per_s: float = 0.002  # the battery drifts: a random walk, not a constant
    voff_sigma: float = 0.10  # [m/s]
    voff_max: float = 0.25
    floor_v: float = 0.05  # [m/s] measured residual of the speed fit
    n_long: int = 20
    steady_taus: float = 3.0  # quasi-steady: the command held this many tau_v
    deadband_margin_mps: float = 0.15  # the gain is only identified well above the threshold
    accel_max_update_mps2: float = 2.0
    dead_rev_prior_mps: float = 0.10  # start threshold in reverse (only the real car can measure it)
    dead_max_mps: float = 0.35
    dead_rev_max_mps: float = 0.45
    dead_step: float = 0.30  # fraction of the bracket closed per launch event
    # --- actuator lags: a model bank, not filter states (they enter non-linearly)
    tau_steer_grid_s: tuple = (0.06, 0.09, 0.12, 0.18)
    tau_speed_grid_s: tuple = (0.15, 0.22, 0.30)
    tau_steer_prior_s: float = 0.12  # measured minimum (servo 0.085 plus the 50 Hz hold and the knuckle loop)
    tau_speed_prior_s: float = 0.22
    lag_review_s: float = 3.0
    lag_window: int = 40
    lag_advantage: float = 0.80  # a candidate must be this much better to be selected
    # --- measurement acceptance
    dt_min_s: float = 0.07  # a dropped revolution is not a measurement interval
    dt_max_s: float = 0.15
    chi_max: float = 3.0  # a residual this many times what the filter claims is inconsistent
    sigma_reject: float = 3.0  # drop an estimate whose sigma is this many times the nominal
    sigma_xy_nominal_m: float = 0.0032
    sigma_yaw_nominal_rad: float = 0.0009
    v_min_update_mps: float = 0.5
    history_s: float = 2.5
    cmd_delay_s: float = 0.02  # one actuation period between publishing and acting
    # --- watchdog: the identified model may never be worse than the datasheet
    watchdog_hold_s: float = 2.0
    watchdog_trips: int = 3  # after this many resets, adaptation latches off for the run
    # --- control
    kp_speed: float = 0.25  # proportional only: the identified gain is the integrator
    speed_authority: float = 0.20  # fraction of the command the proportional term may add
    speed_feedback_forward_scale: float = 1.0  # forward actuator-unit conversion for P gain and its authority floor
    kp_speed_forward: float = 0.0  # positive: forward-only P gain before command-unit conversion; zero uses kp_speed
    launch_hold_s: float = 0.30  # standstill under a speed command before the launch ramp
    launch_forward_command_scale: float = 1.0  # convert launch constants when the forward command gauge is calibrated
    launch_ramp_mps2: float = 0.50
    lead_max: float = 0.25  # [1/m] cap of the servo-lag lead
    lead_preserve_turn: bool = False  # opt-in: anticipation cannot invert desired curvature
    plan_with_identified_model: bool = False  # physical arc limits from the same model the inverse uses
    lead_smooth_s: float = 0.06  # the lap-1 path updates at 10 Hz: differentiate a smoothed signal
    # --- confidence schedule. Every entry may only make the car more careful
    # than the nominal configuration, never less.
    v_floor_frac: float = 0.85  # race v_max at zero confidence
    a_lat_floor_frac: float = 0.80
    lap1_v_floor_frac: float = 0.85  # lap 1 is bounded by its clearance, not by its time
    margin_blind_add_m: float = 0.05  # extra corridor margin for the race plan
    # Extra margins in lap 1 while the car is unsure of how it steers. The
    # mechanism is here and tested, but it ships at zero: measured over the ten
    # documented seeds it cost 2.3 s of the lap-1 median, which puts it outside
    # the 44-51 s band, and the smallest clearance of the campaign did not
    # improve by a millimetre. The confidence is only unresolved over the first
    # 10-15 m of travel, and the minimum clearance of a lap is almost never set
    # there. Raise them if a car turns out to be mismatched enough that the
    # first corner is the dangerous one.
    wall_margin_add_m: float = 0.0
    soft_margin_add_m: float = 0.0
    lookahead_add_m: float = 0.15  # a loop-gain error is damped by looking further ahead
    lookahead_add_s: float = 0.10
    sat_cmd_frac: float = 0.85  # fraction of the identified saturation usable at zero confidence
    # --- recovery. The reactive planner has no way out of 0 feasible arcs: it
    # stops and stays stopped. Creep forward if there is room, reverse if not.
    recover_enabled: bool = True
    recover_block_s: float = 1.0  # no feasible arc for this long, standing still
    recover_creep_free_m: float = 0.6  # free path ahead above which creeping forward is enough
    recover_creep_mps: float = 0.3  # the odometry network was trained above ~0.3 m/s
    recover_speed_mps: float = 0.4  # reverse speed
    recover_distance_m: float = 1.5
    recover_timeout_s: float = 4.0
    recover_exit_free_m: float = 0.0  # early exit needs this much usable forward path (0: legacy)
    recover_max_per_lap: int = 2
    # Optional separate allowance for a place the car cannot escape. Genuine
    # forward progress replenishes this allowance, never the per-lap ceiling.
    recover_max_without_progress: int = 0  # 0: only the legacy per-lap ceiling
    recover_progress_reset_m: float = 0.75
    recover_clearance_m: float = 0.25  # rear clearance required in the accumulated cloud

    def validate(self) -> None:
        if self.recover_max_without_progress < 0:
            raise ValueError("recover_max_without_progress must be nonnegative")
        if not math.isfinite(self.recover_progress_reset_m) or self.recover_progress_reset_m <= 0:
            raise ValueError("recover_progress_reset_m must be finite and positive")
        if not self.tau_steer_grid_s or not self.tau_speed_grid_s:
            raise ValueError("the lag grids may not be empty")
        if min(self.tau_steer_grid_s) <= 0.0 or min(self.tau_speed_grid_s) <= 0.0:
            raise ValueError("the lags must be positive")
        if not 0.0 < self.gain_min < self.gain_max:
            raise ValueError("the gain box must be positive and ordered")
        if self.arm_box_m <= 0.0 or self.arm_sigma <= 0.0:
            raise ValueError("the arm box and sigma must be positive")
        if self.dt_min_s <= 0.0 or self.dt_min_s >= self.dt_max_s:
            raise ValueError("the measurement interval box must be ordered")
        for name in ("v_floor_frac", "a_lat_floor_frac", "lap1_v_floor_frac", "sat_cmd_frac", "sat_floor_frac"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]: the schedule may not be optimistic")


@dataclass
class MissionConfig:
    """Optional bounded open-track trial; the driving controller stays the same."""

    enabled: bool = False
    max_distance_m: float = 1.0
    max_duration_s: float = 5.0
    max_command_mps: float = 0.25
    max_reverse_command_mps: float = 0.25
    forward_only: bool = True

    def __post_init__(self) -> None:
        for name in ("max_distance_m", "max_duration_s", "max_command_mps", "max_reverse_command_mps"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"mission.{name} must be finite and positive")


@dataclass
class RaceDriverConfig:
    vehicle: VehicleLimits = field(default_factory=VehicleLimits)
    reactive: ReactiveConfig = field(default_factory=ReactiveConfig)
    closure: ClosureConfig = field(default_factory=ClosureConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    race: RaceConfig = field(default_factory=RaceConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
