#!/usr/bin/env python3
"""Short videos of a race run (explore lap, then the planned laps) from its logs.

    python3 tools/analysis/animate_race_run.py <run_dir> [--out DIR] [--speedup 4] [--fps 12]

Two clips, each as .mp4 and .gif:
  - ``vuelta1``: lap 1, reactive, without a map;
  - ``carrera``: the laps on the line planned on the car's own map.

Left, the whole track; right, a camera that follows the car. The real walls, the
curbs and the car's true footprint come from the simulator (the car never sees
them). The LiDAR revolution is drawn where the car believes it is (its pose in
its own map), so its misalignment with the walls is the car's pose error.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_real2sim_sensors import AQUA, BLUE, INK, INK_2, ORANGE, SURFACE, WALL, _csv, _style  # noqa: E402

SIM = Path(__file__).resolve().parents[2]
for p in (SIM / "tools", SIM / "ros2_ws" / "src" / "apex_fusion_research"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from apex_fusion_research.core.clearance import outline_points  # noqa: E402
from apex_fusion_research.core.race_eval import load_truth_track  # noqa: E402

CURB = "#dcd9d0"
SCAN = "#eda100"  # the LiDAR: categorical slot 4 of the project palette (validated with slots 1-3)
PHASE_ES = {"explore": "vuelta 1: exploración reactiva, sin mapa", "closing": "cierre de vuelta: 6 m más para que el SLAM cierre el lazo",
            "planning": "planificando la línea sobre su mapa (sigue reactivo)", "race": "vueltas planificadas: sigue la línea de su mapa",
            "final": "frenando tras la última vuelta", "done": "detenido"}
LASER = (0.18, 0.0)  # nominal base_link -> laser


def _world(twm, x, y, yaw):
    c, s = math.cos(twm[2]), math.sin(twm[2])
    return twm[0] + c * x - s * y, twm[1] + s * x + c * y, yaw + twm[2]


def _footprint(x, y, yaw, length=0.46, width=0.32):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = np.array([[length / 2, width / 2], [length / 2, -width / 2], [-length / 2, -width / 2], [-length / 2, width / 2]])
    return np.column_stack((x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]))


def render(run: Path, out: Path, speedup: float, fps: int, width_px: int = 960) -> list[Path]:
    cfg = json.loads((run / "run_config.json").read_text())
    al = run / "slam" / "alignment.json"
    twm = json.loads(al.read_text())["learned"]["T_world_map"] if al.exists() else cfg["spawn_true"]
    track = load_truth_track(cfg["track"])
    walls, _ = outline_points(track.geometry, kinds=("wall", "obstacle"))
    curbs, _ = outline_points(track.geometry, kinds=("curb",))
    tr, drv = _csv(run / "truth_track.csv"), _csv(run / "driver.csv")
    t_tr = tr["t_ns"] * 1e-9
    scans = np.genfromtxt(run / "lidar_scans.csv", delimiter=",", skip_header=1)
    t_scan = scans[:, 0] * 1e-9
    ranges = scans[:, 2:]
    ang = -math.pi + 2 * math.pi / ranges.shape[1] * np.arange(ranges.shape[1])
    rl = _csv(run / "plan" / "raceline.csv") if (run / "plan" / "raceline.csv").exists() else None
    line_w = None
    if rl is not None:
        lx, ly, _ = _world(twm, rl["x"], rl["y"], 0.0)
        line_w = (np.r_[lx, lx[:1]], np.r_[ly, ly[:1]])
    ev = json.loads((run / "race_events.json").read_text()) if (run / "race_events.json").exists() else {}
    t_plan = None
    ph = drv["phase"]
    if (ph == "race").any():
        t_plan = float(drv["t"][np.argmax(ph == "planning")]) if (ph == "planning").any() else float(drv["t"][np.argmax(ph == "race")])
    t_explore = float(drv["t"][np.argmax(ph == "explore")])
    t_race = float(drv["t"][np.argmax(ph == "race")]) if (ph == "race").any() else None
    t_end = float(drv["t"][np.argmax(ph == "done")]) if (ph == "done").any() else float(drv["t"][-1])
    crossings = [c["t"] for c in ev.get("crossings", [])]

    clips = {"vuelta1": (t_explore, crossings[0] + 2.0 if crossings else t_end)}
    if t_race is not None:
        clips["carrera"] = (t_race - 1.0, t_end)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    xmin, xmax = min(walls[:, 0].min(), curbs[:, 0].min()) - 0.5, max(walls[:, 0].max(), curbs[:, 0].max()) + 0.5
    ymin, ymax = min(walls[:, 1].min(), curbs[:, 1].min()) - 0.5, max(walls[:, 1].max(), curbs[:, 1].max()) + 0.5
    for name, (t0, t1) in clips.items():
        times = np.arange(t0, t1, speedup / fps)
        fig = plt.figure(figsize=(width_px / 100, width_px / 100 * 0.47), dpi=100, facecolor=SURFACE)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.08, left=0.03, right=0.99, bottom=0.06, top=0.84)
        ax_w, ax_c = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        for ax in (ax_w, ax_c):
            _style(ax)
            ax.grid(False)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.scatter(curbs[:, 0], curbs[:, 1], s=1.2, color=CURB, linewidths=0, rasterized=True)
            ax.scatter(walls[:, 0], walls[:, 1], s=1.2, color=WALL, linewidths=0, rasterized=True)
        ax_w.set_xlim(xmin, xmax)
        ax_w.set_ylim(ymin, ymax)
        ax_c.set_title("cámara que sigue al carro (6 m)", loc="left", fontsize=9, color=INK_2)
        ax_w.set_title("pista completa (paredes gris oscuro, bordillos gris claro: invisibles para el LiDAR)", loc="left", fontsize=9, color=INK_2)
        title = fig.text(0.03, 0.93, "", fontsize=12, color=INK, ha="left")
        sub = fig.text(0.03, 0.885, "", fontsize=9, color=INK_2, ha="left")
        line_art = [a.plot([], [], color=ORANGE, lw=1.4, ls=(0, (4, 2)))[0] for a in (ax_w, ax_c)]
        trail1 = [a.plot([], [], color=BLUE, lw=1.3)[0] for a in (ax_w, ax_c)]
        trail2 = [a.plot([], [], color=AQUA, lw=1.3)[0] for a in (ax_w, ax_c)]
        scan_art = [a.scatter([], [], s=s, color=SCAN, linewidths=0) for a, s in ((ax_w, 2), (ax_c, 6))]
        car_art = [a.add_patch(Polygon(np.zeros((4, 2)), closed=True, fc=INK, ec=INK, lw=0.8)) for a in (ax_w, ax_c)]
        fig.legend(handles=[plt.Line2D([], [], color=BLUE, lw=1.5, label="vuelta 1 (reactiva)"), plt.Line2D([], [], color=AQUA, lw=1.5, label="vueltas planificadas"),
                            plt.Line2D([], [], color=ORANGE, lw=1.5, ls=(0, (4, 2)), label="línea planificada (mapa del carro)"),
                            plt.Line2D([], [], color=SCAN, marker="o", ls="none", ms=4, label="LiDAR con la pose que estima el carro"),
                            Polygon(np.zeros((3, 2)), fc=INK, label="carro (pose real)")],
                   loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
        fig.subplots_adjust(bottom=0.1)
        lap1_mask = np.isin(np.asarray(drv["phase"]), ["explore", "closing", "planning"])
        idx_ph = lambda t: int(np.clip(np.searchsorted(drv["t"], t) - 1, 0, len(drv) - 1))  # noqa: E731
        tmp = Path(tempfile.mkdtemp(prefix=f"anim_{name}_"))
        for f, t in enumerate(times):
            k = int(np.clip(np.searchsorted(t_tr, t) - 1, 0, len(t_tr) - 1))
            x, y, yaw, v = float(tr["x"][k]), float(tr["y"][k]), float(tr["yaw"][k]), float(tr["speed"][k])
            i = idx_ph(t)
            phase = str(drv["phase"][i])
            # Trails: true path so far, coloured by the phase of the car at each sample.
            upto = t_tr <= t
            idx = np.clip(np.searchsorted(drv["t"], t_tr[upto]) - 1, 0, len(drv) - 1)
            is1 = lap1_mask[idx]
            xs, ys = tr["x"][upto], tr["y"][upto]
            for a in trail1:
                a.set_data(np.where(is1, xs, np.nan), np.where(is1, ys, np.nan))
            for a in trail2:
                a.set_data(np.where(~is1, xs, np.nan), np.where(~is1, ys, np.nan))
            show_line = line_w is not None and t_plan is not None and t >= t_plan and phase in ("planning", "race", "final", "done")
            for a in line_art:
                a.set_data(*(line_w if show_line else ([], [])))
            # The last LiDAR revolution, placed with the car's own estimate of its pose.
            j = int(np.searchsorted(t_scan, t) - 1)
            if j >= 0:
                bx, by, byaw = _world(twm, float(np.interp(t_scan[j], drv["t"], drv["x_map"])), float(np.interp(t_scan[j], drv["t"], drv["y_map"])),
                                      float(np.interp(t_scan[j], drv["t"], np.unwrap(drv["yaw_map"]))))
                r = ranges[j]
                ok = np.isfinite(r) & (r > 0.15) & (r < 8.0)
                lx_, ly_ = LASER[0] + r[ok] * np.cos(ang[ok]), LASER[1] + r[ok] * np.sin(ang[ok])
                c, s = math.cos(byaw), math.sin(byaw)
                pts = np.column_stack((bx + c * lx_ - s * ly_, by + s * lx_ + c * ly_))
                for a in scan_art:
                    a.set_offsets(pts)
            fp = _footprint(x, y, yaw)
            for a in car_art:
                a.set_xy(fp)
            ax_c.set_xlim(x - 3.0, x + 3.0)
            ax_c.set_ylim(y - 2.2, y + 2.2)
            lap = int(drv["lap_own"][i]) if "lap_own" in drv.dtype.names else 0
            title.set_text(PHASE_ES.get(phase, phase))
            sub.set_text(f"t = {t:5.1f} s   velocidad real {v:4.2f} m/s   vueltas que cuenta el carro: {lap}   (vídeo x{speedup:g})")
            fig.savefig(tmp / f"f{f:05d}.png", dpi=100, facecolor=SURFACE)
        plt.close(fig)
        mp4, gif = out / f"{name}.mp4", out / f"{name}.gif"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", str(tmp / "f%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-crf", "26", str(mp4)], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp4), "-vf",
                        f"fps={fps},scale={width_px}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                        str(gif)], check=True)
        shutil.rmtree(tmp, ignore_errors=True)
        written += [mp4, gif]
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: <run>/video")
    ap.add_argument("--speedup", type=float, default=4.0)
    ap.add_argument("--fps", type=int, default=12)
    a = ap.parse_args()
    for p in render(a.run, a.out or a.run / "video", a.speedup, a.fps):
        print(p, f"{p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
