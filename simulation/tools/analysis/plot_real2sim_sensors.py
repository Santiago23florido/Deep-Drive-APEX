#!/usr/bin/env python3
"""Sensor data and SLAM map of a real2sim closed-loop run (tools/sim/apex_real2sim_up.sh).

Writes two figures into the run directory:

* ``sensores.png``: what the car received during the run. The IMU
  (``imu_raw.csv``: gyro z against the true yaw rate, planar specific force)
  and the LiDAR (``lidar_scans.csv``: the published revolutions, with noise,
  as a range map over time, plus two revolutions in the car frame, one on a
  straight and one in a curve at speed). The true speed is only there for
  context.
* ``mapa.png``: the map the car built (slam_toolbox on the learned
  odometry). On the left, the occupancy grid in its own map frame; on the
  right, the same map placed in the world at the true start pose, with no
  fitting to the truth. It is drawn over the real walls, with the true
  trajectory, the SLAM trajectory and the odometry alone.

Usage:
    learning/.venv/bin/python tools/analysis/plot_real2sim_sensors.py <run_dir>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402

# Validated categorical slots 1-3 and the neutral tokens of the project figures.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID, SURFACE, WALL = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#b8b6ae"
RANGE_MAP = LinearSegmentedColormap.from_list("range", ["#0d3a73", "#2a78d6", "#9cc3ef", "#eef4fb"])  # one hue, dark = near


def _csv(path: Path) -> np.ndarray:
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8"))


def _style(ax, title: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK)


def _note(ax, text: str) -> None:
    """Secondary line under the panel title."""
    ax.set_title(ax.get_title(loc="left"), loc="left", fontsize=10, color=INK, pad=16)
    ax.text(0.0, 1.01, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=INK_2)


def _compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2]])


def _inv(a: np.ndarray) -> np.ndarray:
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([-(c * a[0] + s * a[1]), -(-s * a[0] + c * a[1]), -a[2]])


def sensors(run: Path, t0: float) -> Path:
    imu = _csv(run / "imu_raw.csv")
    tru = _csv(run / "truth_track.csv")
    scans = np.genfromtxt(run / "lidar_scans.csv", delimiter=",", skip_header=1, filling_values=np.nan)
    ti = imu["stamp_sec"] + imu["stamp_nanosec"] * 1e-9 - t0
    tt = tru["t_ns"] * 1e-9 - t0
    yaw_rate = np.degrees(np.gradient(np.unwrap(tru["yaw"]), tru["t_ns"] * 1e-9))
    ts = scans[:, 0] * 1e-9 - t0
    ranges = scans[:, 2:]  # NaN = no return
    beams = ranges.shape[1]
    theta = np.degrees(-math.pi + 2 * math.pi / beams * np.arange(beams))
    speed_at_scan = np.interp(ts, tt, tru["speed"])
    rate_at_scan = np.interp(ts, tt, yaw_rate)
    valid = np.isfinite(ranges) & (ranges > 0)

    fig = plt.figure(figsize=(13, 15), facecolor=SURFACE)
    gs = fig.add_gridspec(5, 2, height_ratios=[0.7, 1, 1, 1.5, 2.1], hspace=0.55, wspace=0.18, top=0.95)
    ax = fig.add_subplot(gs[0, :])
    _style(ax, "Velocidad real del carro (referencia, no la ve el carro)")
    ax.plot(tt, tru["speed"], color=INK_2, lw=1.2)
    ax.set_ylabel("m/s", fontsize=8, color=INK_2)

    ax = fig.add_subplot(gs[1, :])
    _style(ax, "IMU (LSM6DS3, 104 Hz): giróscopo z medido frente a la tasa de guiñada real")
    ax.plot(tt, yaw_rate, color=INK_2, lw=1.0, ls=(0, (4, 2)), label="tasa de guiñada real")
    ax.plot(ti, np.degrees(imu["gz_rps"]), color=BLUE, lw=0.9, label="giróscopo z medido")
    ax.set_ylabel("°/s", fontsize=8, color=INK_2)
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)

    ax = fig.add_subplot(gs[2, :])
    _style(ax, "IMU: fuerza específica planar medida (incluye vibración y ruido del chip)")
    ax.plot(ti, imu["ax_mps2"], color=BLUE, lw=0.7, label="a_x (adelante)")
    ax.plot(ti, imu["ay_mps2"], color=ORANGE, lw=0.7, label="a_y (izquierda)")
    ax.set_ylabel("m/s²", fontsize=8, color=INK_2)
    ax.set_xlabel("tiempo de simulación [s]", fontsize=8, color=INK_2)
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)

    ax = fig.add_subplot(gs[3, :])
    _style(ax, f"LiDAR (RPLIDAR A2M8, {1.0 / np.median(np.diff(scans[:, 0] * 1e-9)):.1f} Hz): rango de cada haz en cada revolución publicada")
    ax.grid(False)
    img = np.where(valid, ranges, np.nan).T
    mesh = ax.pcolormesh(ts, theta, img, cmap=RANGE_MAP, vmin=0.0, vmax=8.0, shading="nearest", rasterized=True)
    ax.set_ylabel("ángulo del haz [°]", fontsize=8, color=INK_2)
    ax.set_xlabel("tiempo de simulación [s]   (blanco = sin retorno; la franja fija detrás, 145°…−155°, es la carrocería)", fontsize=8, color=INK_2)
    ax.set_yticks([-180, -90, 0, 90, 180])
    cb = fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.03)
    cb.set_label("rango [m]", fontsize=8, color=INK_2)
    cb.ax.tick_params(labelsize=7, colors=INK_2)

    moving = speed_at_scan > 0.8 * np.nanmax(speed_at_scan)
    cands = {"en recta": np.where(moving & (np.abs(rate_at_scan) < 5))[0], "en curva": np.where(moving)[0]}
    picks = {"en recta": cands["en recta"][len(cands["en recta"]) // 2] if len(cands["en recta"]) else 0,
             "en curva": cands["en curva"][np.argmax(np.abs(rate_at_scan[cands["en curva"]]))] if len(cands["en curva"]) else 0}
    for col, (name, k) in enumerate(picks.items()):
        ax = fig.add_subplot(gs[4, col])
        _style(ax, f"Una revolución {name}: t = {ts[k]:.1f} s, {speed_at_scan[k]:.1f} m/s, {rate_at_scan[k]:+.0f} °/s")
        ok = valid[k]
        a = np.radians(theta[ok])
        ax.scatter(ranges[k, ok] * np.cos(a), ranges[k, ok] * np.sin(a), s=6, color=BLUE, linewidths=0, label=f"{ok.sum()} retornos válidos de {beams}")
        ax.add_patch(plt.Polygon([[0, 0], [8 * math.cos(math.radians(145)), 8 * math.sin(math.radians(145))],
                                  [8 * math.cos(math.radians(205)), 8 * math.sin(math.radians(205))]], color=GRID, alpha=0.6, lw=0))
        ax.plot([0], [0], marker=(3, 0, -90), ms=11, color=INK)
        ax.set_aspect("equal")
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_xlabel("x adelante [m]", fontsize=8, color=INK_2)
        ax.set_ylabel("y izquierda [m]", fontsize=8, color=INK_2)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle(f"{run.name}: datos de los sensores que recibió el carro", x=0.06, y=0.98, ha="left", fontsize=13, color=INK)
    out = run / "sensores.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def _read_pgm(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    parts, pos = [], 0
    while len(parts) < 4:  # magic, width, height, maxval (comments skipped)
        while raw[pos : pos + 1].isspace():
            pos += 1
        if raw[pos : pos + 1] == b"#":
            pos = raw.index(b"\n", pos) + 1
            continue
        end = pos
        while not raw[end : end + 1].isspace():
            end += 1
        parts.append(raw[pos:end])
        pos = end
    w, h = int(parts[1]), int(parts[2])
    return np.frombuffer(raw[pos + 1 : pos + 1 + w * h], dtype=np.uint8).reshape(h, w)


def slam_map(run: Path, t0: float, ev: dict) -> Path:
    slam = run / "slam"
    grid = _read_pgm(slam / "map_learned.pgm")
    meta = {k.strip(): v.strip() for k, v in (ln.split(":", 1) for ln in (slam / "map_learned.yaml").read_text().splitlines() if ":" in ln)}
    res = float(meta["resolution"])
    ox, oy = (float(v) for v in meta["origin"].strip("[]").split(",")[:2])
    pts = _csv(slam / "map_learned_points.csv")
    walls = _csv(slam / "track_truth_points.csv")
    traj_slam = _csv(slam / "slam_learned_trajectory.csv")
    truth = _csv(slam / "truth_trajectory.csv")
    est = _csv(run / "estimator.csv")
    tsc = _csv(run / "truth_scans.csv")
    # Odometry alone, anchored at the true pose of its first estimate (the frame of the car at its first scan).
    order = np.argsort(tsc["stamp_ns"])
    stamps = tsc["stamp_ns"][order]
    k0 = int(np.searchsorted(stamps, est["stamp_ns"][0]))
    anchor = _compose(np.array([tsc["x"][order][k0], tsc["y"][order][k0], tsc["yaw"][order][k0]]), _inv(np.array([est["x"][0], est["y"][0], est["yaw"][0]])))
    odo = np.array([_compose(anchor, np.array([x, y, w])) for x, y, w in zip(est["x"], est["y"], est["yaw"])])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.12})
    ax = axes[0]
    _style(ax, "Mapa de ocupación que construye el carro (su propio marco 'map')")
    ax.grid(False)
    h, w = grid.shape
    shown = np.where(grid == 205, 235, grid)  # unknown lighter than free-space gray
    ax.imshow(shown, cmap="gray", vmin=0, vmax=255, origin="upper", extent=(ox, ox + w * res, oy, oy + h * res), interpolation="nearest")
    ax.plot(traj_slam["x_map"], traj_slam["y_map"], color=ORANGE, lw=1.4, label="trayectoria estimada por el SLAM")
    ax.set_xlabel("x [m]", fontsize=8, color=INK_2)
    ax.set_ylabel("y [m]", fontsize=8, color=INK_2)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    _note(ax, f"negro = ocupado, blanco = libre, gris = no visto; celdas de {res * 100:.0f} cm")

    ax = axes[1]
    s = ev.get("slam", {})
    anch, ate = s.get("anchored", {}), s.get("ate", {})
    _style(ax, "El mismo mapa en el mundo, anclado en la pose real de salida (sin ajuste a la verdad)")
    ax.scatter(walls["x_m"], walls["y_m"], s=1.2, color=WALL, linewidths=0, label="paredes reales de la pista", rasterized=True)
    ax.scatter(pts["x_world"], pts["y_world"], s=2.5, color=BLUE, linewidths=0, label="celdas ocupadas del mapa del carro", rasterized=True)
    ax.plot(truth["x"], truth["y"], color=INK, lw=1.6, label="trayectoria real")
    ax.plot(traj_slam["x_world"], traj_slam["y_world"], color=ORANGE, lw=1.2, label="trayectoria del SLAM")
    ax.plot(odo[:, 0], odo[:, 1], color=AQUA, lw=1.2, ls=(0, (4, 2)), label="odometría aprendida sola (sin SLAM)")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", fontsize=8, color=INK_2)
    ax.set_ylabel("y [m]", fontsize=8, color=INK_2)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, markerscale=4)
    if anch:
        _note(ax, f"mapa: precisión {100 * anch['precision']:.0f} %, cobertura {100 * anch['coverage']:.0f} % (a 10 cm de una pared real)   "
                  f"trayectoria del SLAM: error {100 * ate.get('rmse_m', float('nan')):.1f} cm RMS (ATE)")
    lap = ev.get("lap", {})
    status = {"completed": "vuelta completa", "failed": "fallida"}.get(lap.get("status"), lap.get("status", "?"))
    fig.suptitle(f"{run.name}: mapa reconstruido por el carro ({status}, {lap.get('progress_m', 0):.0f} de {lap.get('planned_m', 0):.0f} m recorridos)",
                 x=0.06, ha="left", fontsize=13, color=INK)
    out = run / "mapa.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def race(run: Path, ev: dict) -> Path | None:
    """The race driver: lap 1 reactive, the planned line, the race laps (truth for the evaluation)."""
    import sys  # noqa: PLC0415

    sim = Path(__file__).resolve().parents[2]
    for p in (sim / "tools", sim / "ros2_ws" / "src" / "apex_fusion_research"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from scipy.spatial import cKDTree  # noqa: PLC0415

    from apex_fusion_research.core.clearance import outline_points  # noqa: PLC0415
    from apex_fusion_research.core.race_eval import load_truth_track  # noqa: PLC0415

    cfg = json.loads((run / "run_config.json").read_text())
    rl = _csv(run / "plan" / "raceline.csv") if (run / "plan" / "raceline.csv").exists() else None
    tr, drv = _csv(run / "truth_track.csv"), _csv(run / "driver.csv")
    lap = ev.get("lap", {})
    al = run / "slam" / "alignment.json"
    twm = json.loads(al.read_text())["learned"]["T_world_map"] if al.exists() else cfg["spawn_true"]
    track = load_truth_track(cfg["track"])
    t_tr = tr["t_ns"] * 1e-9
    # Phase of the car at each truth sample, and the true laps of the referee.
    ph_t = drv["t"]
    idx = np.clip(np.searchsorted(ph_t, t_tr) - 1, 0, len(ph_t) - 1)
    phase = drv["phase"][idx]
    lap1 = np.isin(phase, ["explore", "closing", "planning"])
    racing = phase == "race"
    moving = np.nonzero(tr["speed"] > 0.1)[0]
    t_move = t_tr[moving[0]] if len(moving) else t_tr[0]
    lap_times = lap.get("lap_times_s", [])
    bounds = t_move + np.concatenate(([0.0], np.cumsum(lap_times))) if lap_times else np.array([t_move])

    fig = plt.figure(figsize=(15, 15), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.6, 1, 0.8], hspace=0.38, wspace=0.18)
    ax = fig.add_subplot(gs[0, :])
    _style(ax, "Lo que condujo el carro, en el mundo (la verdad solo para esta figura)")
    ax.grid(False)
    for kinds, col, lab in ((("wall", "obstacle"), WALL, "paredes y pilares (el LiDAR los ve)"), (("curb",), "#dcd9d0", "bordillos (bajo el plano del LiDAR: invisibles para el carro)")):
        pts, _ = outline_points(track.geometry, kinds=kinds)
        ax.scatter(pts[:, 0], pts[:, 1], s=1.5, color=col, linewidths=0, label=lab, rasterized=True)
    ax.plot(np.where(lap1, tr["x"], np.nan), np.where(lap1, tr["y"], np.nan), color=BLUE, lw=1.2, label="vuelta 1: reactiva, sin mapa")
    ax.plot(np.where(racing, tr["x"], np.nan), np.where(racing, tr["y"], np.nan), color=AQUA, lw=1.2, label="vueltas siguientes: sobre la línea planificada")
    if rl is not None:
        c, s_ = math.cos(twm[2]), math.sin(twm[2])
        wx, wy = twm[0] + c * rl["x"] - s_ * rl["y"], twm[1] + s_ * rl["x"] + c * rl["y"]
        ax.plot(np.r_[wx, wx[:1]], np.r_[wy, wy[:1]], color=ORANGE, lw=1.3, ls=(0, (4, 2)), label="línea planificada en el mapa del carro")
    ax.plot([tr["x"][0]], [tr["y"][0]], marker="o", ms=7, color=INK, ls="none", label="salida")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", fontsize=8, color=INK_2)
    ax.set_ylabel("y [m]", fontsize=8, color=INK_2)
    leg = ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    for h in (getattr(leg, "legend_handles", None) or leg.legendHandles)[:2]:
        h.set_sizes([20])
    race = ev.get("race", {})
    pt = race.get("plan_truth", {})
    if pt:
        _note(ax, f"línea planificada contra la pista real: holgura mínima {100 * pt['line_min_clearance_m']:.0f} cm, "
                  f"{100 * pt['line_inside_lane_frac']:.0f} % dentro del carril; corredor sólido {100 * pt['corridor_sound_frac']:.0f} %")
    # Speed along the race line (planned against real), per lap.
    ax = fig.add_subplot(gs[1, 0])
    _style(ax, "Velocidad a lo largo de la línea planificada")
    if rl is not None:
        c, s_ = math.cos(twm[2]), math.sin(twm[2])
        dx, dy = tr["x"] - twm[0], tr["y"] - twm[1]
        local = np.column_stack((c * dx + s_ * dy, -s_ * dx + c * dy))
        dist, k = cKDTree(np.column_stack((rl["x"], rl["y"]))).query(local)
        s_line = rl["s"][k]
        ax.plot(rl["s"], rl["v"], color=ORANGE, lw=1.4, ls=(0, (4, 2)), label="planificada")
        for i in range(len(bounds) - 1):
            m = (t_tr >= bounds[i]) & (t_tr < bounds[i + 1])
            if not m.any():
                continue
            order = np.argsort(s_line[m])
            col = BLUE if i == 0 else AQUA
            ax.plot(s_line[m][order], tr["speed"][m][order], color=col, lw=1.0,
                    label="real, vuelta 1" if i == 0 else ("real, vueltas planificadas" if i == 1 else None))
        ax.set_xlabel("distancia sobre la línea [m]", fontsize=8, color=INK_2)
        ax.set_ylabel("m/s", fontsize=8, color=INK_2)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        ax = fig.add_subplot(gs[1, 1])
        _style(ax, "Distancia real a la línea planificada durante la carrera")
        m = racing
        ax.plot(s_line[m], np.where(m, dist, np.nan)[m] * 100, ".", ms=1.5, color=AQUA, label="real (verdad)")
        lat = np.abs(drv["lat_race"])
        ok = np.isfinite(lat) & (drv["phase"] == "race")
        if ok.any():
            _, kd = cKDTree(np.column_stack((rl["x"], rl["y"]))).query(np.column_stack((drv["x_map"][ok], drv["y_map"][ok])))
            ax.plot(rl["s"][kd], lat[ok] * 100, ".", ms=1.5, color=INK_2, label="la que cree el carro")
        ax.set_xlabel("distancia sobre la línea [m]", fontsize=8, color=INK_2)
        ax.set_ylabel("cm", fontsize=8, color=INK_2)
        ax.legend(fontsize=8, frameon=False, loc="upper right", markerscale=6)
    # Lap times and clearances.
    names = [f"vuelta {i + 1}" for i in range(len(lap_times))]
    for col_i, (vals, title, unit, scale) in enumerate(((lap_times, "Tiempo de cada vuelta (verdad)", "s", 1.0),
                                                        (lap.get("lap_min_clearance_m", [])[: len(lap_times)], "Holgura mínima del carro en cada vuelta (verdad)", "cm", 100.0))):
        ax = fig.add_subplot(gs[2, col_i])
        _style(ax, title)
        if vals:
            colors = [BLUE] + [AQUA] * (len(vals) - 1)
            bars = ax.bar(names[: len(vals)], np.asarray(vals) * scale, color=colors, width=0.55, edgecolor=SURFACE, linewidth=2)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v * scale:.1f}", ha="center", va="bottom", fontsize=8, color=INK_2)
        ax.set_ylabel(unit, fontsize=8, color=INK_2)
    status = {"completed": "completada", "failed": "fallida"}.get(lap.get("status"), lap.get("status", "?"))
    fig.suptitle(f"{run.name}: pista no vista, vuelta 1 reactiva y vueltas planificadas en su propio mapa ({status}"
                 f"{': ' + lap.get('failure_reason') if lap.get('failure_reason') else ''})", x=0.06, y=0.94, ha="left", fontsize=13, color=INK)
    out = run / "carrera.png"
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    run = args.run_dir
    ev = json.loads((run / "evaluation.json").read_text()) if (run / "evaluation.json").exists() else {}
    t0 = 0.0
    for path in (sensors(run, t0), slam_map(run, t0, ev)):
        print(path)
    cfg = json.loads((run / "run_config.json").read_text()) if (run / "run_config.json").exists() else {}
    if cfg.get("driver") == "race":
        print(race(run, ev))


if __name__ == "__main__":
    main()
