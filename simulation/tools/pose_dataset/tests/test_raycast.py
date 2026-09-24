import math

import numpy as np

from pose_dataset.geometry import Box, Cylinder, WorldGeometry
from pose_dataset.raycast import RayCaster


def _world():
    g = WorldGeometry()
    g.boxes.append(Box((5.0, 0.0, 0.15), (0.1, 4.0, 0.3), 0.0, "wall"))
    g.boxes.append(Box((0.0, 0.0, -0.025), (80.0, 80.0, 0.05), 0.0, "ground"))
    g.cylinders.append(Cylinder((0.0, 3.0, 0.3), 0.5, 0.6, "feature"))
    return g


def test_wall_and_cylinder_distances():
    rc = RayCaster(_world())
    o = np.array([[0.0, 0.0, 0.12], [0.0, 0.0, 0.12]])
    d = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    r = rc.cast(o, d, 12.0)
    np.testing.assert_allclose(r, [4.95, 2.5], atol=1e-9)


def test_rotated_box_and_miss():
    g = WorldGeometry()
    g.boxes.append(Box((3.0, 0.0, 0.15), (2.0, 0.1, 0.3), math.pi / 2, "wall"))  # rotated: thin along x
    rc = RayCaster(g)
    r = rc.cast(np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.1]]), np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]), 12.0)
    assert abs(r[0] - 2.95) < 1e-9 and np.isinf(r[1])


def test_beam_above_wall_and_pitched_beam_hits_ground():
    rc = RayCaster(_world())
    # Beam at 0.5 m height passes over the 0.3 m wall.
    assert np.isinf(rc.cast(np.array([[0.0, 0.0, 0.5]]), np.array([[1.0, 0.0, 0.0]]), 4.0))[0]
    # 2 deg nose-down beam from 0.12 m hits the floor at 0.12 / tan(2 deg).
    a = math.radians(2.0)
    r = rc.cast(np.array([[0.0, 0.0, 0.12]]), np.array([[-math.cos(a), 0.0, -math.sin(a)]]), 12.0)[0]
    assert abs(r - 0.12 / math.sin(a)) < 1e-6


def test_max_range():
    rc = RayCaster(_world())
    assert np.isinf(rc.cast(np.array([[0.0, 0.0, 0.12]]), np.array([[1.0, 0.0, 0.0]]), 4.0))[0]
