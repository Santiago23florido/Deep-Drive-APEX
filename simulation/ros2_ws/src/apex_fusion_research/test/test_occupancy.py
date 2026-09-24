import math

import numpy as np

from apex_fusion_research.core.occupancy import GridInfo, grid_to_points, write_pgm_yaml


def test_grid_to_points_axis_aligned():
    info = GridInfo(width=4, height=3, resolution=0.5, origin_x=-1.0, origin_y=2.0)
    data = np.full(12, -1)
    data[1 + 2 * 4] = 100  # col 1, row 2
    data[0] = 30  # below threshold
    xy, occ = grid_to_points(data, info, occupied_threshold=65)
    np.testing.assert_allclose(xy, [[-1.0 + 1.5 * 0.5, 2.0 + 2.5 * 0.5]])
    assert occ.tolist() == [100]


def test_grid_to_points_rotated_origin():
    info = GridInfo(width=2, height=1, resolution=1.0, origin_x=1.0, origin_y=1.0, origin_yaw=math.pi / 2)
    xy, _ = grid_to_points([100, 0], info)
    # Cell centre (0.5, 0.5) in grid frame rotated by +90 deg -> (-0.5, 0.5).
    np.testing.assert_allclose(xy, [[0.5, 1.5]], atol=1e-12)


def test_pgm_yaml_written(tmp_path):
    info = GridInfo(width=3, height=2, resolution=0.1, origin_x=0.0, origin_y=0.0)
    pgm, yml = write_pgm_yaml([100, 0, -1, 0, 0, 0], info, tmp_path / "map")
    raw = pgm.read_bytes()
    header = b"P5\n3 2\n255\n"
    assert raw.startswith(header)
    pixels = np.frombuffer(raw[len(header):], dtype=np.uint8).reshape(2, 3)
    # Row 0 of the grid is the bottom image row.
    assert pixels[1].tolist() == [0, 254, 205]
    assert "resolution: 0.100000" in yml.read_text()
