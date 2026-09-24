"""Compare the good- and damaged-sensor SLAM maps against the real track.

Usage:
    ros2 run apex_fusion_research plot_slam_maps <run_dir> [--threshold 0.10] [--show]

Inputs (written by fusion_slam_map_recorder and the measurement recorders):
    <run_dir>/slam/track_truth_points.csv, truth_trajectory.csv,
    map_<name>_points.csv, slam_<name>_trajectory.csv
    <run_dir>/measurements_ideal/lidar_points.csv   (optional, observed region)

Evaluation
----------
* Maps are compared in the world frame using the anchoring of the recorder
  (initial true pose only): drift and distortion count as errors.
* The reference is the part of the real track the ideal LiDAR actually saw
  along the true trajectory (``observed track``); without the ideal
  measurements the whole track is used and coverage is not meaningful.
* ``shape`` metrics repeat the comparison after rigid point-to-line ICP of the
  map onto the full track: only the map geometry is judged.
* ATE compares each SLAM trajectory with the true base_link trajectory.

Outputs: ``slam_maps.png/.pdf``, ``slam_metrics.json`` and ``slam_metrics.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from ..core.map_metrics import absolute_trajectory_error, icp_2d, map_similarity, observed_subset, se2_apply

COLORS = {"truth": "#222222", "good": "#2a9d55", "noisy": "#e4572e", "good_imu": "#9467bd",
          "pipeline": "#3a6ea5", "track": "#b8b8b8", "observed": "#6d6d6d"}
LABELS = {"good": "good sensors", "noisy": "damaged sensors", "good_imu": "ideal LiDAR + IMU heading",
          "pipeline": "APEX pipeline SLAM"}
SLAM_ORDER = ("good", "noisy", "good_imu", "pipeline")


def _read_csv(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    data = np.genfromtxt(path, delimiter=",", names=True)
    return np.atleast_1d(data) if data.size else None


def observed_track(run_dir: Path, track: np.ndarray, truth: np.ndarray, laser_offset=(0.18, 0.0), radius=0.10,
                   max_points: int = 400000):
    """Project the ideal scans with the true pose and keep the track points they hit."""
    pts = _read_csv(run_dir / "measurements_ideal" / "lidar_points.csv")
    if pts is None or truth is None:
        return None, None
    if pts.size > max_points:
        pts = pts[:: int(math.ceil(pts.size / max_points))]
    t = pts["stamp_sec"] + 1e-9 * pts["stamp_nanosec"]
    inside = (t >= truth["t"][0]) & (t <= truth["t"][-1])
    t, pts = t[inside], pts[inside]
    if t.size == 0:
        return None, None
    x = np.interp(t, truth["t"], truth["x"])
    y = np.interp(t, truth["t"], truth["y"])
    yaw = np.interp(t, truth["t"], np.unwrap(truth["yaw"]))
    lx = pts["x_forward_m"] + laser_offset[0]
    ly = pts["y_left_m"] + laser_offset[1]
    world = np.column_stack((x + np.cos(yaw) * lx - np.sin(yaw) * ly, y + np.sin(yaw) * lx + np.cos(yaw) * ly))
    return observed_subset(track, world, radius), world


def evaluate(run_dir: Path, threshold: float) -> dict:
    slam_dir = run_dir / "slam"
    track_raw = _read_csv(slam_dir / "track_truth_points.csv")
    if track_raw is None:
        raise SystemExit(f"missing {slam_dir / 'track_truth_points.csv'}")
    track = np.column_stack((track_raw["x_m"], track_raw["y_m"]))
    truth = _read_csv(slam_dir / "truth_trajectory.csv")
    observed, ideal_hits = observed_track(run_dir, track, truth)
    reference = observed if observed is not None and observed.shape[0] > 10 else track

    result = {
        "run_dir": str(run_dir),
        "threshold_m": threshold,
        "reference": "observed_track" if reference is observed else "full_track",
        "n_track_points": int(track.shape[0]),
        "n_reference_points": int(reference.shape[0]),
        "slams": {},
        "_arrays": {"track": track, "reference": reference, "truth": truth, "ideal_hits": ideal_hits},
    }
    for name in SLAM_ORDER:
        m = _read_csv(slam_dir / f"map_{name}_points.csv")
        traj = _read_csv(slam_dir / f"slam_{name}_trajectory.csv")
        if m is None:
            continue
        xy = np.column_stack((m["x_world"], m["y_world"]))
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        entry = {"occupied_cells": int(xy.shape[0])}
        entry["anchored"] = map_similarity(reference, xy, threshold)
        t_icp, rms = icp_2d(xy, track, max_correspondence_m=0.5)
        xy_icp = se2_apply(t_icp, xy)
        entry["shape"] = map_similarity(reference, xy_icp, threshold)
        entry["icp"] = {"x": t_icp[0], "y": t_icp[1], "yaw_deg": math.degrees(t_icp[2]), "rms_m": rms}
        if traj is not None and truth is not None:
            entry["ate"] = absolute_trajectory_error(
                truth["t"], np.column_stack((truth["x"], truth["y"])),
                traj["t"], np.column_stack((traj["x_world"], traj["y_world"])),
            )
            entry["duration_s"] = float(traj["t"][-1] - traj["t"][0])
        if truth is not None:
            d = np.hypot(np.diff(truth["x"]), np.diff(truth["y"]))
            result["truth_distance_m"] = float(d.sum())
        entry["_arrays"] = {"map": xy, "map_icp": xy_icp, "traj": traj}
        result["slams"][name] = entry
    return result


def _strip(result: dict) -> dict:
    out = {k: v for k, v in result.items() if not k.startswith("_")}
    out["slams"] = {n: {k: v for k, v in e.items() if not k.startswith("_")} for n, e in result["slams"].items()}
    return out


def write_metrics(result: dict, run_dir: Path) -> None:
    clean = _strip(result)
    (run_dir / "slam_metrics.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")
    with open(run_dir / "slam_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slam", "comparison", "precision", "coverage", "chamfer_m", "ate_rmse_m", "ate_final_m", "occupied_cells"])
        for name, e in clean["slams"].items():
            for kind in ("anchored", "shape"):
                m = e[kind]
                ate = e.get("ate", {})
                writer.writerow([name, kind, f"{m['precision']:.4f}", f"{m['coverage']:.4f}", f"{m['chamfer_m']:.4f}",
                                 f"{ate.get('rmse_m', float('nan')):.4f}", f"{ate.get('final_m', float('nan')):.4f}",
                                 e["occupied_cells"]])


def plot(result: dict, out_base: Path, title: str, show: bool = False) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = result["_arrays"]
    track, reference, truth = arr["track"], arr["reference"], arr["truth"]
    slams = result["slams"]
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 1.2])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0:2])]
    table_ax = fig.add_subplot(gs[1, 2])

    xmin, ymin = track.min(axis=0) - 0.6
    xmax, ymax = track.max(axis=0) + 0.6

    def base(ax, subtitle):
        ax.scatter(track[:, 0], track[:, 1], s=1.5, c=COLORS["track"], label="real track", zorder=1)
        if reference is not track:
            ax.scatter(reference[:, 0], reference[:, 1], s=2.5, c=COLORS["observed"], label="observed part", zorder=2)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(subtitle, fontsize=11)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    # (a) trajectories
    ax = axes[0]
    base(ax, "Trajectories (world frame)")
    if truth is not None:
        ax.plot(truth["x"], truth["y"], color=COLORS["truth"], lw=2.0, label="ground truth", zorder=3)
    for name, e in slams.items():
        traj = e["_arrays"]["traj"]
        if traj is not None:
            ax.plot(traj["x_world"], traj["y_world"], color=COLORS[name], lw=1.4, label=f"SLAM {LABELS[name]}", zorder=4)
    ax.legend(fontsize=7, loc="upper left")

    # (b), (c) anchored maps
    for ax, name in zip(axes[1:3], ("good", "noisy")):
        base(ax, f"SLAM map, {LABELS[name]} (anchored)")
        if name in slams:
            m = slams[name]["_arrays"]["map"]
            ax.scatter(m[:, 0], m[:, 1], s=3, c=COLORS[name], label="SLAM occupied cells", zorder=3)
        ax.legend(fontsize=7, loc="upper left")

    # (d) overlay after ICP (shape only)
    ax = axes[3]
    base(ax, "Map shape after rigid ICP onto the real track")
    for name, e in slams.items():
        m = e["_arrays"]["map_icp"]
        ax.scatter(m[:, 0], m[:, 1], s=3, c=COLORS[name], alpha=0.8, label=f"{LABELS[name]} (ICP)", zorder=3)
    ax.legend(fontsize=7, loc="upper left")

    # metrics table
    table_ax.axis("off")
    rows, row_labels = [], []
    for name, e in slams.items():
        ate = e.get("ate", {})
        for kind in ("anchored", "shape"):
            m = e[kind]
            row_labels.append(f"{LABELS[name]}\n{kind}")
            rows.append([f"{m['precision']:.2f}", f"{m['coverage']:.2f}", f"{m['chamfer_m']:.3f}",
                         f"{ate.get('rmse_m', float('nan')):.2f}" if kind == "anchored" else "-"])
    if rows:
        tab = table_ax.table(cellText=rows, rowLabels=row_labels,
                             colLabels=["precision", "coverage", "chamfer [m]", "ATE rmse [m]"],
                             loc="center", cellLoc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(9)
        tab.scale(1.0, 1.9)
    thr = result["threshold_m"]
    dist = result.get("truth_distance_m", float("nan"))
    table_ax.set_title(f"Metrics vs {result['reference'].replace('_', ' ')} (threshold {thr:.2f} m)\n"
                       f"distance driven {dist:.1f} m", fontsize=10)

    fig.suptitle(title, fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=150)
    if show:
        plt.show()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.10, help="match distance for precision/coverage [m]")
    parser.add_argument("--title", default="slam_toolbox baseline response: good vs damaged sensors")
    parser.add_argument("--output", type=Path, default=None, help="figure base path (default <run_dir>/slam_maps)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    result = evaluate(args.run_dir, args.threshold)
    write_metrics(result, args.run_dir)
    plot(result, args.output or args.run_dir / "slam_maps", args.title, args.show)
    print(f"reference: {result['reference']} ({result['n_reference_points']} of {result['n_track_points']} track points)")
    print(f"{'slam':<7} {'comparison':<9} {'precision':>9} {'coverage':>9} {'chamfer[m]':>11} {'ATE rmse[m]':>12}")
    for name, e in _strip(result)["slams"].items():
        for kind in ("anchored", "shape"):
            m = e[kind]
            ate = e.get("ate", {}).get("rmse_m", float("nan")) if kind == "anchored" else float("nan")
            print(f"{name:<7} {kind:<9} {m['precision']:9.3f} {m['coverage']:9.3f} {m['chamfer_m']:11.3f} {ate:12.3f}")
    print(f"figure written to {(args.output or args.run_dir / 'slam_maps').with_suffix('.png')}")


if __name__ == "__main__":
    main()
