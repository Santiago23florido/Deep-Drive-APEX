"""Keep a tight observed route as guidance without executing an invalid line."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apex_race.core.race_driver import MAP_GUIDED, PLANNING, RaceDriverCore
from apex_race.core.race_planner import Grid, path_from_points
from apex_race.core.reactive import ReactiveDriver
from test_oval_profile import oval_core


@pytest.fixture(scope='module')
def recorded_plan():
    directory = Path(__file__).parent / 'data'
    data = np.load(directory / 'slam_map_204107.npz')
    meta = json.loads(str(data['metadata']))
    core = oval_core()
    core.start = tuple(json.loads((directory / 'first_lap_204107.json').read_text())['start'])
    core.grid = (0., Grid(data['occupancy'], meta['resolution'], meta['origin']['x'], meta['origin']['y']))
    core.graph = (0., data['graph'][:, 1:])
    core._submit_plan(0.)
    return core.future.result()


def test_actual_small_track_candidate_is_rejected_for_direct_tracking(recorded_plan):
    assert not recorded_plan.diag['direct_tracking_allowed']
    assert set(recorded_plan.diag['direct_tracking_rejection']) == {'curvature', 'map_clearance'}
    assert recorded_plan.route is not None and len(recorded_plan.route.xy) > 20


def test_rejected_candidate_transitions_to_real_map_guidance_without_pure_pursuit(recorded_plan):
    core = oval_core()
    core.phase = PLANNING
    core.future = SimpleNamespace(done=lambda: True, result=lambda: recorded_plan)
    core._start_race = lambda pose: pytest.fail('invalid candidate reached direct tracking')
    core.control(1., (0., 0., 0.), 0., None, True)
    assert core.phase == MAP_GUIDED and core.guided
    assert core.pursuit is None
    assert core.events['handover']['local_planner_retained']
    assert core.status()['direct_tracking_allowed'] is False


def test_map_guidance_uses_the_observed_route_with_a_forward_window(recorded_plan):
    core = oval_core()
    core.phase, core.plan = MAP_GUIDED, recorded_plan
    core.plan.path = recorded_plan.path
    at = recorded_plan.route.xy[0]
    guide = core._race_line_in((float(at[0]), float(at[1]), float(recorded_plan.route.heading[0])))
    np.testing.assert_allclose(guide[0], [0., 0.], atol=1e-12)
    length = np.linalg.norm(np.diff(guide, axis=0), axis=1).sum()
    assert 1.5 < length <= 2.2
    assert len(guide) < len(recorded_plan.route.xy)  # the other half is not a competing guide


def test_map_guide_never_overrides_a_live_front_obstacle(recorded_plan):
    core = oval_core()
    core.phase, core.plan = MAP_GUIDED, recorded_plan
    core.cfg.adaptive.recover_enabled = False
    angles = -np.pi + np.arange(360) * np.pi / 180
    ranges = np.full(360, np.inf)
    forward = np.cos(angles) > .2
    ranges[forward] = (.55 - .18) / np.cos(angles[forward])
    core.on_scan(1., ranges, -np.pi, np.pi / 180, (0., 0., 0.), (0., 0., 0.), 0.)
    assert core.cmd is not None
    assert core.control(1.01, (0., 0., 0.), 0., None, True)[0] == 0.
    assert core.phase == MAP_GUIDED


def test_guide_station_does_not_jump_to_a_nearby_parallel_branch():
    route = path_from_points(np.array([[0., 0.], [.2, 0.], [.4, 0.], [.6, 0.], [2., 0.],
        [2., 1.], [-1., 1.], [-1., .08], [0., .08], [.2, .08], [.4, .08], [.6, .08], [2., .08], [2., -.5], [0., -.5]]))
    core = oval_core()
    core.phase = MAP_GUIDED
    core.plan = SimpleNamespace(route=route, path=route)
    core._race_line_in((.2, .02, 0.))
    previous = core.map_route_s
    core._race_line_in((.3, .06, 0.))
    assert abs(core.map_route_s - previous) < .4
    assert core.map_route_s < 1.  # retain the first branch despite the closer other branch


def test_guide_station_unwraps_continuously_across_route_zero():
    a = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    route = path_from_points(np.column_stack((np.cos(a), np.sin(a))))
    core = oval_core()
    core.phase, core.plan = MAP_GUIDED, SimpleNamespace(route=route, path=route)
    core._race_line_in((*route.xy[-1], float(route.heading[-1])))
    before = core.map_route_s
    core._race_line_in((*route.xy[0], float(route.heading[0])))
    assert 0 < core.map_route_s - before < .15


def test_zero_weight_map_guide_keeps_the_working_local_reference():
    cfg = oval_core().cfg
    cloud = np.column_stack((np.linspace(-1, 3, 100), np.full(100, .45)))
    local, _ = ReactiveDriver(cfg.vehicle, cfg.reactive).step(cloud, (0, 0, 0), 0, 0)
    guide = np.array([[0., 0.], [1., -.6], [2., -1.2]])
    blended, _ = ReactiveDriver(cfg.vehicle, cfg.reactive).step(cloud, (0, 0, 0), 0, 0, guide, guide_weight=0.)
    assert blended.kappa == pytest.approx(local.kappa)
    assert blended.v == pytest.approx(local.v)
    np.testing.assert_allclose(blended.path, local.path)


def test_map_guide_weight_grows_from_local_to_map_without_resetting_controller(recorded_plan):
    core = oval_core()
    core.phase, core.plan = MAP_GUIDED, recorded_plan
    core.guide_started_t = 2.
    identifier, shaper = core.ident, core.shaper
    for stamp, expected in ((1.9, 0.), (2.5, .5), (3.2, 1.)):
        core.on_scan(stamp, np.full(360, 2.), -np.pi, np.pi / 180, (0, 0, 0), (0, 0, 0), 0)
        assert core.guide_weight == pytest.approx(expected)
    assert core.ident is identifier and core.shaper is shaper
