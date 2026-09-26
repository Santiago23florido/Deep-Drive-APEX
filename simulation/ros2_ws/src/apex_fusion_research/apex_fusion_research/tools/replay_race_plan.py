"""Replay the race planner on the map and trajectory of a recorded run.

    replay_race_plan <run_dir> [--mode min_curvature|centre] [--w-min 1.5] [--truth]

Reads ``<run>/slam/map_learned.{pgm,yaml}`` and the SLAM trajectory of the
car (``slam_learned_trajectory.csv``, map frame), cuts its first lap at the
start line, plans the race line exactly as the car does after lap 1 (without
the live wall log: the wall sides come from the map) and writes
``<run>/plan_replay/`` plus ``plan.png``. With ``--truth`` the line and its
corridor are also checked against the true track (evaluation only).

Note: the map of a recorded run holds every lap, not only the first one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..core.lap_closure import cut_loop
from ..core.occupancy import read_pgm_yaml
from ..core.race_config import ClosureConfig, PlannerConfig, RaceConfig, VehicleLimits
from ..core.race_planner import Grid, plan_race, save_plan


def replay(run: Path, mode: str = "min_curvature", w_min: float = 1.5, truth: bool = False, out_name: str = "plan_replay") -> dict:
    data, info = read_pgm_yaml(run / "slam" / "map_learned")
    grid = Grid(data, info.resolution, info.origin_x, info.origin_y)
    traj = np.genfromtxt(run / "slam" / "slam_learned_trajectory.csv", delimiter=",", names=True)
    xy = np.column_stack((traj["x_map"], traj["y_map"]))
    start = (float(traj["x_map"][0]), float(traj["y_map"][0]), float(traj["yaw_map"][0]))
    cut = cut_loop(xy, start, ClosureConfig())
    if cut is None:
        raise SystemExit("the trajectory does not close a lap")
    loop, seam = cut
    veh, pcfg, rcfg = VehicleLimits(), PlannerConfig(mode=mode), RaceConfig()
    plan = plan_race(grid, loop, start, None, veh, pcfg, rcfg, w_min)
    out = run / out_name
    extra: dict = {"seam": seam, "lap_path_source": "recorded_trajectory"}
    if truth:
        from ..core.race_eval import TrueTrack, load_truth_track, plan_truth_metrics  # noqa: PLC0415

        cfg = json.loads((run / "run_config.json").read_text())
        t_wm = json.loads((run / "slam" / "alignment.json").read_text())["learned"]["T_world_map"]
        tt = TrueTrack(load_truth_track(cfg["track"]), veh)
        extra["truth"] = plan_truth_metrics(plan.path.xy, plan.path.heading, plan.ref.xy, plan.ref.normal, plan.cor.lo, plan.cor.hi, t_wm, tt, plan.path.curvature)
        extra["truth"]["T_world_map"] = t_wm
    save_plan(plan, grid, loop, None, out, extra)
    figure(plan, grid, loop, out / "plan.png", run.name, extra)
    return {**plan.diag, **extra}


def figure(plan, grid: Grid, loop: np.ndarray, path: Path, title: str, extra: dict) -> None:  # noqa: ANN001
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    ink, ink2, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(11, 8), facecolor=surface)
    ax.set_facecolor(surface)
    img = np.full(grid.data.shape, 235, dtype=np.uint8)
    img[grid.data == 0] = 255
    img[grid.data >= 65] = 40
    h, w = grid.data.shape
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, origin="lower", extent=(grid.ox, grid.ox + w * grid.res, grid.oy, grid.oy + h * grid.res), interpolation="nearest")
    n = plan.ref.normal
    left = plan.ref.xy + plan.cor.hi[:, None] * n
    right = plan.ref.xy + plan.cor.lo[:, None] * n
    ax.plot(loop[:, 0], loop[:, 1], color=blue, lw=1.0, label="vuelta 1 (SLAM)")
    ax.plot(np.r_[left[:, 0], left[:1, 0]], np.r_[left[:, 1], left[:1, 1]], color=aqua, lw=0.8, label="límites del corredor (centro del carro)")
    ax.plot(np.r_[right[:, 0], right[:1, 0]], np.r_[right[:, 1], right[:1, 1]], color=aqua, lw=0.8)
    xy = plan.path.xy
    ax.plot(np.r_[xy[:, 0], xy[:1, 0]], np.r_[xy[:, 1], xy[:1, 1]], color=orange, lw=1.6, label=f"línea planificada ({plan.diag['mode']})")
    ax.plot([loop[0, 0]], [loop[0, 1]], marker="o", ms=8, color=ink, ls="none", label="salida")
    x0, x1 = min(left[:, 0].min(), right[:, 0].min()) - 1.5, max(left[:, 0].max(), right[:, 0].max()) + 1.5
    y0, y1 = min(left[:, 1].min(), right[:, 1].min()) - 1.5, max(left[:, 1].max(), right[:, 1].max()) + 1.5
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.tick_params(colors=ink2, labelsize=8)
    d = plan.diag
    note = (f"vuelta 1: {d['lap_ref_length_m']:.1f} m   línea: {d['race_length_m']:.1f} m   tiempo previsto {d['t_lap_pred_s']:.1f} s "
            f"({d['v_mean_mps']:.2f} m/s)   κ máx {d['max_kappa']:.2f} / {d['kappa_limit']:.2f} 1/m   holgura mín. en el mapa {100 * d['min_map_clearance_m']:.0f} cm")
    if "truth" in extra:
        t = extra["truth"]
        note += (f"\nverdad: holgura mín. {100 * t['line_min_clearance_m']:.0f} cm, dentro del carril {100 * t['line_inside_lane_frac']:.0f} %, "
                 f"corredor sólido {100 * t['corridor_sound_frac']:.0f} % (peor {100 * t['corridor_worst_violation_m']:.0f} cm)")
    ax.set_title(f"{title}: línea de carrera sobre el mapa del carro", loc="left", fontsize=11, color=ink, pad=28)
    ax.text(0.0, 1.01, note, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=ink2)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=surface)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--mode", default="min_curvature", choices=["min_curvature", "centre"])
    ap.add_argument("--w-min", type=float, default=1.5)
    ap.add_argument("--truth", action="store_true")
    a = ap.parse_args()
    res = replay(a.run_dir.resolve(), a.mode, a.w_min, a.truth, "plan_replay" if a.mode == "min_curvature" else f"plan_replay_{a.mode}")
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
