"""Calibrate the hardware units while retaining the shared control law."""
from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest

from apex_race.core.adaptive import AdaptiveShaper
from apex_race.core.config_io import dataclass_from_flat, load_ros_params_yaml
from apex_race.core.plant_id import PlantIdentifier
from apex_race.core.race_config import RaceDriverConfig, vehicle_limits
from apex_race.core.vehicle_io import load_vehicle_config

CONFIG = Path(__file__).resolve().parents[1] / 'config'


def initialized_config():
    flat = load_ros_params_yaml(CONFIG / 'race_speed_initialized.yaml')
    cfg = dataclass_from_flat(RaceDriverConfig, flat)
    sides = cfg.vehicle
    cfg.vehicle = vehicle_limits(load_vehicle_config(CONFIG / 'vehicle_speed_initialized.yaml'), sides.steer_plan_deg)
    for name in ('steer_max_left_deg', 'steer_max_right_deg', 'steer_plan_left_deg', 'steer_plan_right_deg'):
        setattr(cfg.vehicle, name, getattr(sides, name))
    return cfg


def test_reference_is_used_directly_without_scaling_or_a_confidence_override():
    cfg = initialized_config()
    ident = PlantIdentifier(cfg)
    shaper = AdaptiveShaper(cfg)
    command, _ = shaper.shape(1., 0., .5, .5, ident.model(), shaper.envelope(ident.model()))
    assert shaper.v_target == .5
    assert command == .5
    assert (ident.model().v_gain, ident.model().v_offset) == (1., 0.)
    assert ident.conf_long == 0. and not ident.longitudinal_ready
    # A 0.5 m/s target maps almost exactly to the approved 0.09 command's PWM.
    assert 15.42 + 5.24 * command == pytest.approx(15. + 33. * .09, abs=.08)


def test_negative_moving_intercept_is_inverted_with_its_actual_sign():
    model = replace(PlantIdentifier(RaceDriverConfig()).model(), v_gain=1.2, v_offset=-.05)
    for target in [.2, .5, -.2, -.5]:
        assert model.speed_out(model.speed_in(target)) == pytest.approx(target)
    assert model.speed_in(0.) == 0.


def test_calibrated_forward_data_do_not_rewrite_reverse_commands_or_launches():
    cfg = initialized_config()
    calibrated = PlantIdentifier(cfg)
    # Even if the forward model changes, reverse keeps the original inverse.
    forward = replace(calibrated.model(), v_gain=.8, v_offset=.04)
    reference = PlantIdentifier(RaceDriverConfig()).model()
    a, b = AdaptiveShaper(cfg), AdaptiveShaper(RaceDriverConfig())
    for index in range(100):
        speed = 0. if index < 40 else -.12
        first = a.shape(index * .02, .2, -.2, speed, forward, a.envelope(forward))[0]
        second = b.shape(index * .02, .2, -.2, speed, reference, b.envelope(reference))[0]
        assert first == pytest.approx(second)


def test_reverse_stall_still_learns_launch_threshold_without_poisoning_forward():
    ident = PlantIdentifier(initialized_config())
    ident.v_steady_since = 0.
    before = ident.dead_rev
    ident._update_long(1., 0., 0., np.full(len(ident.tau_v), -.3), (.0032, .0032, .0009), .1)
    assert ident.dead_rev > before and ident.n_dead == 1
    assert ident.long.n == 0


def test_steady_faster_plant_can_be_identified_instead_of_called_coasting():
    ident = PlantIdentifier(initialized_config())
    ident.v_steady_since = 0.
    ident.prev_speed = .625
    ident._update_long(1., .0625, 0., np.full(len(ident.tau_v), .5), (.0032, .0032, .0009), .1)
    assert ident.long.n == 1
    assert ident.rejected['braking'] == 0


def test_opposite_motion_cannot_establish_a_forward_gain():
    ident = PlantIdentifier(initialized_config())
    ident.v_steady_since = 0.
    ident.prev_speed = -.2
    ident._update_long(1., -.02, 0., np.full(len(ident.tau_v), .5), (.0032, .0032, .0009), .1)
    assert ident.long.n == 0 and ident.rejected['braking'] == 1


@pytest.mark.parametrize('gain_scale', [.85, 1., 1.15])
def test_same_controller_tracks_reference_through_calibrated_bridge(gain_scale):
    """A recorded moving-branch approximation, with independent load stress.

    This deterministic motor simulation is not a physical/Gazebo success claim.
    It checks closed-loop stability, launch, residual learning and PWM bounds.
    """
    cfg = initialized_config()
    ident, shaper = PlantIdentifier(cfg), AdaptiveShaper(cfg)
    t, velocity, position = 0., 0., 0.
    traces = []
    dt = .02
    for index in range(1600):
        target = [.45, .55, .5][int(t // 4.) % 3]
        model = ident.model()
        command, _ = shaper.shape(t, 0., target, velocity, model, shaper.envelope(model))
        command = max(0., min(cfg.mission.max_command_mps, command))
        pct = 0. if command < .005 else min(28.2, 15.42 + 5.24 * command)
        # Independent physical model expressed in the previous bridge gauge.
        moving_speed = max(0., gain_scale * (6.3 * ((pct - 15.) / 33.) - .08)) if pct else 0.
        if velocity < .03 and pct < 17.3:
            moving_speed = 0.
        velocity += (1. - math.exp(-dt / .22)) * (moving_speed - velocity)
        position += velocity * dt
        ident.push_command(t, 0., command)
        t += dt
        if index % 5 == 4:
            ident.on_estimate(t, (position, 0., 0.), (.0032, .0032, .0009))
        traces.append((t, target, velocity, pct))
    data = np.array(traces)
    settled = (data[:, 0] > 5.) & (np.mod(data[:, 0], 4.) > 1.)
    assert np.mean(abs(data[settled, 1] - data[settled, 2])) < .075
    assert np.max(data[:, 3]) <= 28.2
    assert np.median(data[settled, 2]) > .4  # retain the approved physical pace
    assert ident.long.n >= 20 and ident.long.x[0] > 0.
    assert not ident.latched and not ident.sign_fault


def test_physical_profile_retains_geometry_and_pwm_equivalent_command_ceiling():
    cfg = initialized_config()
    original = dataclass_from_flat(RaceDriverConfig, load_ros_params_yaml(CONFIG / 'race_oval.yaml'))
    assert cfg.closure == original.closure and cfg.planner == original.planner
    assert cfg.mission.max_distance_m == 40. and cfg.mission.max_duration_s == 300.
    assert cfg.vehicle.steer_max_left_deg == 22.5 and cfg.vehicle.steer_max_right_deg == 12.96
    assert 15.42 + 5.24 * cfg.mission.max_command_mps <= 28.2
    assert cfg.mission.max_reverse_command_mps == original.mission.max_reverse_command_mps
