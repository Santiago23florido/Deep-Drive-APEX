"""The race guard: the live LiDAR against the race line ahead."""

from types import SimpleNamespace

import numpy as np

from apex_fusion_research.core.race_config import RaceDriverConfig
from apex_fusion_research.core.race_driver import RaceDriverCore
from apex_fusion_research.core.race_planner import path_from_points


def _core(expected_clearance: float) -> RaceDriverCore:
    core = RaceDriverCore(RaceDriverConfig(), 3)
    x = np.arange(-1.0, 12.0, 0.1)
    path = path_from_points(np.column_stack((x, np.zeros_like(x))))
    core.plan = SimpleNamespace(path=path, clearance=np.full(len(path.xy), expected_clearance))
    return core


def _wall(y: float) -> np.ndarray:
    x = np.arange(-1.0, 8.0, 0.02)
    return np.column_stack((x, np.full_like(x, y)))


def test_guard_ignores_map_noise_near_a_tight_line():
    # The plan expected 0.15 m to a wall; the live cloud puts it 5 cm closer: noise, keep racing.
    core = _core(0.15)
    wall = _wall(0.16 + 0.017 + 0.15 - 0.05)
    assert not core._guard(0.0, wall, (0.0, 0.0, 0.0), 3.0)
    assert core.events["guard"] == []


def test_guard_fires_on_a_new_obstacle_on_the_line():
    core = _core(0.6)
    box = np.array([[2.0 + dx, dy] for dx in np.arange(0, 0.3, 0.02) for dy in np.arange(-0.15, 0.15, 0.02)])
    assert core._guard(0.0, np.vstack((_wall(0.9), box)), (0.0, 0.0, 0.0), 3.0)
    assert len(core.events["guard"]) == 1
