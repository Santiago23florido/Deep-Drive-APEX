"""A lap containing real reversals must yield usable geometric stations."""
import json
from pathlib import Path

import numpy as np
import pytest

from apex_race.core.lap_closure import LapCounter
from apex_race.core.race_config import ClosureConfig, PlannerConfig
from apex_race.core.race_planner import closed_reference

DATA = Path(__file__).parent / 'data'


def test_reversed_lap_reference_has_uniform_stations_after_smoothing():
    samples = np.genfromtxt(DATA / 'lap1_path_220320.csv', delimiter=',', names=True)
    xy = np.column_stack((samples['x'], samples['y']))
    reference = closed_reference(xy, (xy[0, 0], xy[0, 1], 0.), PlannerConfig())
    spacing = np.linalg.norm(np.roll(reference.xy, -1, axis=0) - reference.xy, axis=1)
    assert spacing.min() > .05  # formerly almost coincident reversal samples
    assert spacing.max() < .12
    assert np.max(np.abs(reference.curvature)) < 15  # formerly hundreds per meter
    assert reference.length > 3


def test_turning_away_before_start_does_not_finish_a_lap():
    cfg = ClosureConfig(arm_distance_m=1, arm_time_s=0, return_radius_m=.55)
    counter = LapCounter((0., 0., 0.), cfg)
    assert counter.update(10., (-.50, .1, .7), 5.) is None
    assert counter.update(11., (-.48, .1, 1.), 5.1) is None
    assert counter.laps == 0
    crossing = counter.update(12., (.02, .1, .1), 5.7)
    assert crossing is not None and crossing.kind == 'line'


def test_backing_out_before_start_clears_the_radius_fallback():
    cfg = ClosureConfig(arm_distance_m=1, arm_time_s=0, return_radius_m=.55)
    counter = LapCounter((0., 0., 0.), cfg)
    assert counter.update(10., (-.5, .1, 0.), 5.) is None
    assert counter.update(11., (-.7, .1, 0.), 5.2) is None
    assert counter.closest is None and counter.laps == 0


def test_recorded_rotation_near_start_waits_for_the_actual_line_crossing():
    fixture = json.loads((DATA / 'closure_220320.json').read_text())
    cfg = ClosureConfig(arm_distance_m=5, arm_time_s=12, return_radius_m=.55, lateral_tol_m=.55)
    counter = LapCounter(tuple(fixture['start']), cfg, fixture['t0'])
    crossings = []
    for r in fixture['samples']:
        c = counter.update(r['t'], (r['x_map'], r['y_map'], r['yaw_map']), r['travel'])
        if c is not None:
            crossings.append(c)
    assert len(crossings) == 1 and crossings[0].kind == 'line'
    assert crossings[0].t - fixture['t0'] == pytest.approx(70.13, abs=.1)
