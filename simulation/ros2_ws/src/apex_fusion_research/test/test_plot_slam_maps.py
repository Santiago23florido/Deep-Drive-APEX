import json
import math

import numpy as np

from apex_fusion_research.core.map_metrics import se2_apply
from apex_fusion_research.tools.plot_slam_maps import evaluate, write_metrics


def _write(path, header, rows):
    np.savetxt(path, np.asarray(rows), delimiter=",", header=header, comments="", fmt="%.6f")


def make_run(tmp_path, noisy_offset=(0.3, 0.0, 0.05)):
    slam = tmp_path / "slam"
    slam.mkdir()
    a = np.linspace(0, 2 * math.pi, 600, endpoint=False)
    track = np.column_stack((4 * np.cos(a), 2 * np.sin(a)))
    _write(slam / "track_truth_points.csv", "x_m,y_m", track)
    t = np.linspace(0, 10, 101)
    truth = np.column_stack((t, 0.1 * t, np.zeros_like(t), np.zeros_like(t)))
    _write(slam / "truth_trajectory.csv", "t,x,y,yaw", truth)
    for name, offset in (("good", (0.0, 0.0, 0.0)), ("noisy", noisy_offset)):
        m = se2_apply(offset, track)
        _write(slam / f"map_{name}_points.csv", "x_map,y_map,x_world,y_world,occupancy",
               np.column_stack((m, m, np.full(len(m), 100))))
        traj = np.column_stack((t, truth[:, 1], truth[:, 2], truth[:, 3],
                                truth[:, 1] + offset[0], truth[:, 2] + offset[1], truth[:, 3]))
        _write(slam / f"slam_{name}_trajectory.csv", "t,x_map,y_map,yaw_map,x_world,y_world,yaw_world", traj)
    return tmp_path


def test_evaluate_good_vs_offset_map(tmp_path):
    run = make_run(tmp_path)
    result = evaluate(run, threshold=0.05)
    good, noisy = result["slams"]["good"], result["slams"]["noisy"]
    assert result["reference"] == "full_track"  # no ideal measurements in this synthetic run
    assert good["anchored"]["precision"] == 1.0 and good["ate"]["rmse_m"] < 1e-9
    assert noisy["anchored"]["precision"] < 0.5
    assert abs(noisy["ate"]["rmse_m"] - 0.3) < 1e-6
    # A rigid offset is removed by ICP: the shape is perfect.
    assert noisy["shape"]["precision"] > 0.99 and noisy["shape"]["chamfer_m"] < 0.01


def test_metrics_files_written(tmp_path):
    run = make_run(tmp_path)
    write_metrics(evaluate(run, threshold=0.05), run)
    payload = json.loads((run / "slam_metrics.json").read_text())
    assert set(payload["slams"]) == {"good", "noisy"}
    assert (run / "slam_metrics.csv").read_text().count("\n") == 5
