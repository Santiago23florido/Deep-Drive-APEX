"""Forward calibration must not silently change successful steering or reverse."""
import ast
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from apex_race.core.adaptive import AdaptiveShaper
from apex_race.core.config_io import dataclass_from_flat, load_ros_params_yaml
from apex_race.core.plant_id import PlantIdentifier
from apex_race.core.race_config import RaceDriverConfig, vehicle_limits
from apex_race.core.vehicle_io import load_vehicle_config

CONFIG = Path(__file__).resolve().parents[1] / 'config'
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools/core'))
from forward_profiles import forward_profile, select_forward_profile


def tracking_config():
    cfg = dataclass_from_flat(RaceDriverConfig, load_ros_params_yaml(CONFIG/'race_speed_tracking.yaml'))
    sides = cfg.vehicle
    cfg.vehicle = vehicle_limits(load_vehicle_config(CONFIG/'vehicle_speed_tracking.yaml'), sides.steer_plan_deg)
    for name in ('steer_max_left_deg', 'steer_max_right_deg', 'steer_plan_left_deg', 'steer_plan_right_deg'):
        setattr(cfg.vehicle, name, getattr(sides, name))
    return cfg


def test_requested_profile_survives_sudo_environment_sanitization(tmp_path, monkeypatch):
    marker = tmp_path/'.ram_capture_active.json'
    marker.write_text(json.dumps({'forward_profile': 'tracking'}))
    monkeypatch.delenv('APEX_FORWARD_PROFILE', raising=False)
    assert select_forward_profile(tmp_path)['name'] == 'tracking'
    # Even a stale inherited environment must not override the recorded request.
    monkeypatch.setenv('APEX_FORWARD_PROFILE', 'initialized')
    assert select_forward_profile(tmp_path)['name'] == 'tracking'
    with pytest.raises(ValueError):
        select_forward_profile(tmp_path, 'initialized')
    marker.write_text('{}')
    with pytest.raises(KeyError):
        select_forward_profile(tmp_path)
    marker.unlink()
    assert select_forward_profile(tmp_path, 'tracking')['name'] == 'tracking'


def test_tracking_preserves_lateral_geometry_and_recovery_selection():
    original = load_ros_params_yaml(CONFIG/'race_oval.yaml')
    changed = load_ros_params_yaml(CONFIG/'race_speed_tracking.yaml')
    allowed = {'vehicle_yaml', 'reactive.v_explore_mps', 'reactive.v_min_mps', 'reactive.v_blind_mps',
               'reactive.a_acc_mps2', 'race.v_max_mps', 'race.a_acc_mps2', 'adaptive.recover_creep_mps',
               'mission.max_command_mps', 'reactive.w_smooth'}
    for key, value in original.items():
        if key not in allowed:
            assert changed[key] == value, key
    cfg = tracking_config()
    assert not cfg.adaptive.steering_confidence_gate
    assert cfg.adaptive.longitudinal_forward_only and cfg.adaptive.longitudinal_allow_overspeed
    assert cfg.mission.max_reverse_command_mps == .4
    assert cfg.mission.max_distance_m == 40. and cfg.mission.max_duration_s == 300.
    assert cfg.adaptive.recover_distance_m == .5 and cfg.adaptive.recover_timeout_s == 3.


@pytest.mark.parametrize('name', ['initialized', 'tracking'])
def test_runner_guards_compile_and_match_selected_profile(name):
    profile = forward_profile(name)
    cfg = dataclass_from_flat(RaceDriverConfig, load_ros_params_yaml(CONFIG/f'race_speed_{name}.yaml'))
    assert profile['references'] == [cfg.reactive.v_explore_mps, cfg.reactive.v_min_mps,
                                     cfg.reactive.v_blind_mps, cfg.race.v_max_mps]
    assert profile['command_cap'] == cfg.mission.max_command_mps
    assert profile['w_smooth'] == cfg.reactive.w_smooth
    assert profile['esc_zero_pct'] + profile['pct_per_mps'] * profile['command_cap'] <= 28.2 + 1e-10
    assert profile['forward_scale'] == cfg.adaptive.speed_feedback_forward_scale
    assert profile['steering_confidence_gate'] == cfg.adaptive.steering_confidence_gate
    tree = ast.parse((ROOT/'tools/core/initialized_trial/run_oval.py').read_text())
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'checks' for t in n.targets))
    generated = eval(compile(ast.Expression(assignment.value), '<checks>', 'eval'), {'profile': profile})
    compile(generated, '<remote parameter guards>', 'exec')


def test_calibration_does_not_force_confidence_or_change_reverse_shaping():
    cfg = tracking_config()
    ident, shaper = PlantIdentifier(cfg), AdaptiveShaper(cfg)
    assert ident.conf_long == 0. and not ident.longitudinal_ready
    assert ident.model().v_gain == 1. and ident.model().v_offset == 0.
    original = dataclass_from_flat(RaceDriverConfig, load_ros_params_yaml(CONFIG/'race_oval.yaml'))
    other, old = PlantIdentifier(original), AdaptiveShaper(original)
    for i in range(100):
        speed = 0. if i < 40 else -.12
        first = shaper.shape(i*.02, .2, -.2, speed, ident.model(), shaper.envelope(ident.model()))
        second = old.shape(i*.02, .2, -.2, speed, other.model(), old.envelope(other.model()))
        assert first == pytest.approx(second)


@pytest.mark.parametrize('gain', [2.5, 3.8, 4.5])
@pytest.mark.parametrize('delay', [.16, .30])
def test_tracking_handles_calibration_scatter_and_delayed_feedback(gain, delay):
    cfg = tracking_config()
    ident, shaper = PlantIdentifier(cfg), AdaptiveShaper(cfg)
    velocity, history, trace = 0., [0.], []
    for i in range(1000):
        t, dt = i*.02, .02
        target = [.36, .42][int(t//5.) % 2]
        sensed = history[max(0, len(history)-1-round(delay/dt))]
        command, _ = shaper.shape(t, 0., target, sensed, ident.model(), shaper.envelope(ident.model()))
        command = max(0., min(cfg.mission.max_command_mps, command))
        assert shaper.v_target == target
        mapping = forward_profile('tracking')
        pwm = mapping['esc_zero_pct'] + mapping['pct_per_mps']*command if command > .005 else 0.
        delivered = max(0., gain*(pwm-15.)/33.) if pwm else 0.
        if velocity < .03 and pwm < 17.3:
            delivered = 0.
        velocity += (1.-math.exp(-dt/.22))*(delivered-velocity)
        history.append(velocity)
        trace.append((t, target, velocity, pwm))
    data = np.array(trace)
    stable = (data[:, 0] > 2.) & (np.mod(data[:, 0], 5.) > 1.5)
    assert np.mean(abs(data[stable, 1]-data[stable, 2])) < .11
    assert np.max(data[:, 3]) <= 28.2 and np.max(data[:, 2]) < .65
    for start in [3., 8., 13., 18.]:
        keep = (data[:, 0] >= start) & (data[:, 0] < start+1.)
        assert np.ptp(data[keep, 2]) < .10
