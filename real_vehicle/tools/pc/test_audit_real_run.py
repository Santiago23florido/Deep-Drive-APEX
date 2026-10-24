"""Validate the offline geometric diagnostic against known rigid motion."""
import csv
import numpy as np

from audit_real_run import rotation, scan_consistency


def box_scans(path, frame_yaw=0., frame_xy=(0., 0.)):
    angles = -np.pi + np.arange(360) * np.pi / 180
    names = ['stamp_ns', 'now_ns', 'phase', 'angle_min', 'angle_inc',
             'x_stamp', 'y_stamp', 'yaw_stamp', 'x_now', 'y_now', 'yaw_now',
             'speed', 'travel', 'laser_x', 'laser_y', 'laser_yaw']
    with path.open('w') as handle:
        writer = csv.writer(handle); writer.writerow(names)
        for i in range(8):
            yaw, xy = .03 * i, np.array([.04 * i, .01 * i])
            origin = xy + rotation(yaw) @ [.18, 0.]
            rays = np.column_stack((np.cos(angles + yaw), np.sin(angles + yaw)))
            with np.errstate(divide='ignore', invalid='ignore'):
                candidates = np.stack([(2. - origin[0]) / rays[:, 0], (-2. - origin[0]) / rays[:, 0],
                                       (1.5 - origin[1]) / rays[:, 1], (-1.5 - origin[1]) / rays[:, 1]])
            candidates[candidates <= 0] = np.inf
            ranges = candidates.min(axis=0)
            pose_xy = rotation(frame_yaw) @ xy + frame_xy
            pose = [*pose_xy, yaw + frame_yaw]
            stamp = round((1. + .1 * i) * 1e9)
            writer.writerow([stamp, stamp, 'explore', -np.pi, np.pi / 180,
                             *pose, *pose, .4, .04 * i, .18, 0., 0., *ranges])


def test_known_box_motion_improves_consistency_and_is_frame_invariant(tmp_path):
    original, transformed = tmp_path / 'original.csv', tmp_path / 'transformed.csv'
    box_scans(original)
    box_scans(transformed, frame_yaw=.43, frame_xy=(5., -2.))
    a = scan_consistency(original, 0., 3.)
    b = scan_consistency(transformed, 0., 3.)
    assert len(a) == 7
    assert a[:, 6].mean() > .8
    assert a[:, 4].mean() < .6 * a[:, 5].mean()
    np.testing.assert_allclose(a, b, atol=1e-10)
