"""Plot the error accumulation of a pure-INS run recorded by the error monitor.

Usage:
    ros2 run apex_fusion_research plot_ins_drift <run_dir> [--tmax 120] [--show]

Produces ``ins_drift.png`` / ``ins_drift.pdf`` in the run directory and prints
error statistics at fixed horizons (useful for tables in a paper).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_run(run_dir: Path) -> tuple[np.ndarray, dict]:
    data = np.genfromtxt(run_dir / "ins_error.csv", delimiter=",", names=True)
    meta_path = run_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return data, meta


def _value_at(t: np.ndarray, y: np.ndarray, horizon: float) -> float:
    if t.size == 0 or horizon > t[-1]:
        return float("nan")
    return float(np.interp(horizon, t, y))


def summarize(data: np.ndarray, horizons=(10.0, 30.0, 60.0, 120.0)) -> list[dict]:
    t = data["t_nav"]
    rows = []
    for h in horizons:
        rows.append(
            {
                "t_s": h,
                "horizontal_m": _value_at(t, data["e_horizontal"], h),
                "vertical_m": _value_at(t, np.abs(data["e_z"]), h),
                "speed_mps": _value_at(t, np.sqrt(data["e_vx"] ** 2 + data["e_vy"] ** 2 + data["e_vz"] ** 2), h),
                "heading_deg": _value_at(t, np.degrees(np.abs(data["e_yaw"])), h),
                "tilt_deg": _value_at(t, np.degrees(np.hypot(data["e_roll"], data["e_pitch"])), h),
                "distance_m": _value_at(t, data["distance_travelled"], h),
            }
        )
    return rows


def effective_gyro_bias_at_rest(data: np.ndarray, meta: dict) -> np.ndarray | None:
    """True additive gyro error during the static alignment: b + G f_rest.

    The gyro g-sensitivity term G f is indistinguishable from a bias while the
    vehicle stands still, so this is the quantity the alignment estimates.
    """
    align = meta.get("ins_alignment", {})
    if len(data) == 0:
        return None
    bias = np.array([data["true_gyro_bias_x"][0], data["true_gyro_bias_y"][0], data["true_gyro_bias_z"][0]])
    g_sens = meta.get("imu", {}).get("realization", {}).get("gyro", {}).get("g_sensitivity")
    f_rest = align.get("mean_specific_force")
    if g_sens is not None and f_rest is not None:
        bias = bias + np.asarray(g_sens) @ np.asarray(f_rest)
    return bias


def plot(data: np.ndarray, meta: dict, out_base: Path, show: bool = False) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = data["t_nav"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    truth_c, ins_c = "#2a9d55", "#e4572e"

    ax = axes[0, 0]
    ax.plot(data["true_x"], data["true_y"], color=truth_c, lw=1.6, label="ground truth")
    ax.plot(data["ins_x"], data["ins_y"], color=ins_c, lw=1.2, label="pure INS")
    ax.plot(data["true_x"][0], data["true_y"][0], "ko", ms=4, label="start")
    # Zoom on the ground-truth area: the INS usually leaves it quickly.
    tx, ty = data["true_x"], data["true_y"]
    span = max(float(np.ptp(tx)), float(np.ptp(ty)), 2.0)
    margin = max(3.0, 0.5 * span)
    ax.set_xlim(float(tx.min()) - margin, float(tx.max()) + margin)
    ax.set_ylim(float(ty.min()) - margin, float(ty.max()) + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Trajectory (zoom on ground-truth area)")
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    ax.plot(t, data["e_horizontal"], color=ins_c, label="horizontal")
    ax.plot(t, np.abs(data["e_z"]), color="#577590", label="|vertical|")
    ax.set_xlabel("time since INS start [s]")
    ax.set_ylabel("position error [m]")
    ax.set_title("Position error")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    mask = t > 0.05
    if np.any(mask):
        e3 = data["e_3d"][mask]
        tm = t[mask]
        ax.loglog(tm, e3, color=ins_c, label="3D position error")
        # Reference slopes anchored at the median time of the record.
        t_ref = float(np.median(tm))
        e_ref = float(np.interp(t_ref, tm, e3))
        for power, style in ((1, ":"), (2, "--"), (3, "-.")):
            ax.loglog(tm, e_ref * (tm / t_ref) ** power, "k", ls=style, lw=0.8, label=f"~ t^{power}")
    ax.set_xlabel("time since INS start [s]")
    ax.set_ylabel("error [m]")
    ax.set_title("Error growth (log-log)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    ev = np.sqrt(data["e_vx"] ** 2 + data["e_vy"] ** 2)
    ax.plot(t, ev, color=ins_c, label="horizontal")
    ax.plot(t, np.abs(data["e_vz"]), color="#577590", label="|vertical|")
    ax.set_xlabel("time since INS start [s]")
    ax.set_ylabel("velocity error [m/s]")
    ax.set_title("Velocity error")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, np.degrees(data["e_roll"]), label="roll")
    ax.plot(t, np.degrees(data["e_pitch"]), label="pitch")
    ax.plot(t, np.degrees(data["e_yaw"]), label="yaw")
    align = meta.get("ins_alignment", {})
    eff_bias = effective_gyro_bias_at_rest(data, meta)
    if "estimated_gyro_bias" in align and eff_bias is not None:
        # Heading drift predicted by the residual z-gyro error left by the alignment.
        residual_bz = eff_bias[2] - align["estimated_gyro_bias"][2]
        ax.plot(t, np.degrees(residual_bz * t), "k--", lw=0.8, label="residual z-bias x t")
    ax.set_xlabel("time since INS start [s]")
    ax.set_ylabel("attitude error [deg]")
    ax.set_title("Attitude error")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    # True bias shifted by the constant g-sensitivity offset at rest, so it is
    # directly comparable with the alignment estimate (dotted).
    offset = eff_bias - np.array([data[f"true_gyro_bias_{a}"][0] for a in "xyz"]) if eff_bias is not None else np.zeros(3)
    for i, (a, c) in enumerate(zip("xyz", ("C0", "C1", "C2"))):
        ax.plot(t, np.degrees(data[f"true_gyro_bias_{a}"] + offset[i]) * 3600, color=c, label=f"gyro {a}")
        if "estimated_gyro_bias" in align:
            ax.axhline(math.degrees(align["estimated_gyro_bias"][i]) * 3600, color=c, ls=":", lw=0.8)
    ax.set_xlabel("time since INS start [s]")
    ax.set_ylabel("effective gyro bias [deg/h]")
    ax.set_title("True bias + g-sens. at rest (dotted: estimate)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    imu_seed = meta.get("imu", {}).get("config", {}).get("seed", "?")
    mode = align.get("mode", "?")
    fig.suptitle(f"Pure strapdown INS error accumulation  (IMU seed {imu_seed}, alignment: {mode})")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=160)
    if show:
        plt.show()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="run directory containing ins_error.csv")
    parser.add_argument("--tmax", type=float, default=None, help="only use the first TMAX seconds of navigation")
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    args = parser.parse_args(argv)

    data, meta = load_run(args.run_dir)
    data = data[np.isfinite(data["t_nav"]) & (data["t_nav"] >= 0.0)]
    if args.tmax is not None:
        data = data[data["t_nav"] <= args.tmax]
    if data.size == 0:
        raise SystemExit("no navigation samples in the record")
    plot(data, meta, args.run_dir / "ins_drift", show=args.show)

    print(f"{'t [s]':>7} {'horiz [m]':>10} {'vert [m]':>9} {'|dv| [m/s]':>11} {'yaw [deg]':>10} {'tilt [deg]':>10} {'dist [m]':>9}")
    for row in summarize(data):
        print(
            f"{row['t_s']:7.0f} {row['horizontal_m']:10.3f} {row['vertical_m']:9.3f} {row['speed_mps']:11.3f} "
            f"{row['heading_deg']:10.2f} {row['tilt_deg']:10.2f} {row['distance_m']:9.2f}"
        )
    print(f"figure written to {args.run_dir / 'ins_drift.png'}")


if __name__ == "__main__":
    main()
