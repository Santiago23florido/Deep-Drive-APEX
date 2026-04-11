#!/usr/bin/env python3
"""Plot a saved offline refined map from the real APEX pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_runs_root() -> Path:
    return _repo_root() / "APEX" / ".apex_runtime" / "offline_refined_maps"


def _latest_run_dir(runs_root: Path) -> Path:
    candidates = sorted(
        [path for path in runs_root.glob("offline_refined_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No offline_refined_* directories under {runs_root}")
    return candidates[0]


def _load_csv_matrix(path: Path, columns: int) -> np.ndarray:
    if not path.exists():
        return np.empty((0, columns), dtype=np.float64)
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if data.size == 0:
        return np.empty((0, columns), dtype=np.float64)
    if data.shape == ():
        data = np.asarray([data], dtype=data.dtype)
    names = data.dtype.names or ()
    if columns == 2:
        selected = np.column_stack([data[names[0]], data[names[1]]])
    else:
        selected = np.column_stack([data[names[1]], data[names[2]], data[names[3]]])
    return np.asarray(selected, dtype=np.float64)


def _pose_from_payload(payload: dict, key: str) -> np.ndarray | None:
    pose = payload.get(key)
    if not isinstance(pose, dict):
        return None
    try:
        return np.asarray(
            [float(pose["x_m"]), float(pose["y_m"]), float(pose["yaw_rad"])],
            dtype=np.float64,
        )
    except Exception:
        return None


def _draw_pose_arrow(axis, pose: np.ndarray, label: str, color: str) -> None:
    heading_len_m = 0.30
    dx = heading_len_m * np.cos(float(pose[2]))
    dy = heading_len_m * np.sin(float(pose[2]))
    axis.scatter([pose[0]], [pose[1]], s=52, color=color, label=label, zorder=5)
    axis.arrow(
        float(pose[0]),
        float(pose[1]),
        float(dx),
        float(dy),
        width=0.015,
        head_width=0.08,
        length_includes_head=True,
        color=color,
        zorder=6,
    )


def _plot(run_dir: Path, output: Path, show: bool, max_points: int, point_size: float) -> None:
    try:
        if show:
            import matplotlib.pyplot as plt
        else:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise SystemExit(
                "Falta matplotlib en este Python. Usa /usr/bin/python3, "
                "o instala matplotlib en el venv con: python3 -m pip install matplotlib"
            ) from exc
        raise

    map_xy = _load_csv_matrix(run_dir / "map_points_xy.csv", columns=2)
    path_xyyaw = _load_csv_matrix(run_dir / "path_poses_xyyaw.csv", columns=3)
    poses_payload = {}
    poses_path = run_dir / "poses.json"
    if poses_path.exists():
        poses_payload = json.loads(poses_path.read_text(encoding="utf-8"))

    if map_xy.shape[0] > max_points:
        stride = int(np.ceil(map_xy.shape[0] / max_points))
        map_xy = map_xy[::stride]

    figure, axis = plt.subplots(figsize=(10.5, 8.0))
    if map_xy.shape[0]:
        axis.scatter(
            map_xy[:, 0],
            map_xy[:, 1],
            s=point_size,
            color="#2f5d50",
            alpha=0.65,
            linewidths=0,
            label="Mapa LiDAR refinado",
        )
    if path_xyyaw.shape[0]:
        axis.plot(
            path_xyyaw[:, 0],
            path_xyyaw[:, 1],
            color="#d62828",
            linewidth=1.8,
            label="Trayectoria del vehiculo",
        )

    initial_pose = _pose_from_payload(poses_payload, "initial_pose")
    final_pose = _pose_from_payload(poses_payload, "final_pose")
    if initial_pose is not None:
        _draw_pose_arrow(axis, initial_pose, "Pose inicial", "#1f77b4")
    if final_pose is not None:
        _draw_pose_arrow(axis, final_pose, "Pose final", "#111111")

    axis.set_title(run_dir.name)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linewidth=0.4, alpha=0.35)
    axis.legend(loc="best")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(f"[APEX] Plot saved: {output}")
    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Directory containing map_points_xy.csv, path_poses_xyyaw.csv and poses.json.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=_default_runs_root(),
        help="Parent directory used when --run-dir is not provided.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--max-points", type=int, default=80000)
    parser.add_argument("--point-size", type=float, default=1.6)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve() if args.run_dir else _latest_run_dir(args.runs_root)
    output = args.output or (run_dir / "track_plot.png")
    _plot(
        run_dir=run_dir,
        output=output.resolve(),
        show=bool(args.show),
        max_points=max(1000, int(args.max_points)),
        point_size=max(0.1, float(args.point_size)),
    )


if __name__ == "__main__":
    main()
