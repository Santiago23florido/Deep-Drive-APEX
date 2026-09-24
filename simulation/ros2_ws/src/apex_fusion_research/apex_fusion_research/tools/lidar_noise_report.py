"""Statistical validation of the LiDAR noise model on a synthetic room.

A 2D scanner is placed off-centre (and rotated) in a rectangular room, the
ideal scan is ray-cast analytically and corrupted ``--scans`` times. The report
compares the empirical outcome frequencies and the binned standard deviation
of HIT residuals against the model equations, and plots the residual
distribution. This is the kind of figure used to document a sensor model.

    ros2 run apex_fusion_research lidar_noise_report --config rplidar_like --scans 2000
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from ..core.config_io import dataclass_from_flat, load_ros_params_yaml
from ..core.lidar_noise import (
    OUTCOME_DROPOUT,
    OUTCOME_HIT,
    OUTCOME_RANDOM,
    OUTCOME_SHORT,
    LidarNoiseConfig,
    LidarNoiseModel,
)


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    if path.exists():
        return path
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("apex_fusion_research"))
    except Exception:
        share = Path(__file__).resolve().parents[2]
    preset = share / "config" / "lidar" / f"{value}.yaml"
    if not preset.exists():
        raise SystemExit(f"LiDAR config '{value}' not found")
    return preset


def room_scan(n_beams: int, width: float, height: float, sensor_xy, sensor_yaw: float, range_max: float):
    """Analytic ray casting from ``sensor_xy`` inside the box [0,w] x [0,h]."""
    angles = -math.pi + 2.0 * math.pi * np.arange(n_beams) / n_beams
    world = angles + sensor_yaw
    dx, dy = np.cos(world), np.sin(world)
    x0, y0 = sensor_xy
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(dx > 0, (width - x0) / dx, np.where(dx < 0, -x0 / dx, np.inf))
        ty = np.where(dy > 0, (height - y0) / dy, np.where(dy < 0, -y0 / dy, np.inf))
    r = np.minimum(tx, ty)
    r[r > range_max] = np.inf
    return angles, r


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="rplidar_like", help="LiDAR preset name or YAML")
    parser.add_argument("--scans", type=int, default=1000)
    parser.add_argument("--beams", type=int, default=360)
    parser.add_argument("--room", type=float, nargs=2, default=(8.0, 5.0), metavar=("W", "H"))
    parser.add_argument("--output", type=Path, default=Path("lidar_noise_report.png"))
    args = parser.parse_args(argv)

    cfg = dataclass_from_flat(LidarNoiseConfig, load_ros_params_yaml(_resolve(args.config)))
    range_min, range_max = 0.05, 12.0
    angles, ideal = room_scan(args.beams, args.room[0], args.room[1], (2.0, 1.5), 0.3, range_max)
    inc = 2.0 * math.pi / args.beams
    model = LidarNoiseModel(cfg)

    residuals, truths, cosines, sigmas, outcomes = [], [], [], [], []
    for _ in range(args.scans):
        res = model.apply(ideal, -math.pi, inc, range_min, range_max)
        hit = (res.outcome == OUTCOME_HIT) & np.isfinite(res.ranges)
        residuals.append(res.ranges[hit] - ((1.0 + model.range_scale) * res.true_ranges[hit] + model.range_bias_m))
        truths.append(res.true_ranges[hit])
        cosines.append(res.cos_incidence[hit])
        sigmas.append(res.sigma[hit])
        outcomes.append(res.outcome)
    residuals = np.concatenate(residuals)
    truths = np.concatenate(truths)
    sigmas = np.concatenate(sigmas)
    outcome = np.concatenate(outcomes)
    measured = outcome != 0

    print(f"realization: range bias {model.range_bias_m * 1000:.2f} mm, scale error {model.range_scale * 1e6:.0f} ppm")
    for code, name in ((OUTCOME_HIT, "hit"), (OUTCOME_DROPOUT, "dropout"), (OUTCOME_SHORT, "short"), (OUTCOME_RANDOM, "random")):
        print(f"  {name:<8} {np.mean(outcome[measured] == code) * 100:7.3f} %")
    z = residuals / np.where(sigmas > 0, sigmas, np.nan)
    print(f"normalised HIT residual: mean {np.nanmean(z):+.4f}, std {np.nanstd(z):.4f} (model: 0, 1)")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    res0 = LidarNoiseModel(cfg).apply(ideal, -math.pi, inc, range_min, range_max)
    ok = np.isfinite(res0.ranges)
    ax.plot(ideal * np.cos(angles), ideal * np.sin(angles), ".", ms=2, color="#2a9d55", label="ideal")
    ax.plot(res0.ranges[ok] * np.cos(angles[ok]), res0.ranges[ok] * np.sin(angles[ok]), ".", ms=2, color="#e4572e", label="noisy")
    ax.set_aspect("equal")
    ax.set_title("One scan (sensor frame)")
    ax.legend(fontsize=8)

    ax = axes[1]
    bins = np.linspace(np.nanmin(truths), np.nanmax(truths), 15)
    idx = np.digitize(truths, bins)
    centers, emp, mod = [], [], []
    for b in range(1, len(bins)):
        sel = idx == b
        if np.count_nonzero(sel) > 50:
            centers.append(0.5 * (bins[b - 1] + bins[b]))
            emp.append(np.std(residuals[sel]))
            mod.append(np.sqrt(np.mean(sigmas[sel] ** 2)))
    ax.plot(centers, np.array(emp) * 1000, "o", color="#e4572e", label="empirical std")
    ax.plot(centers, np.array(mod) * 1000, "-", color="k", label="model sigma(r, alpha)")
    ax.set_xlabel("range [m]")
    ax.set_ylabel("HIT residual std [mm]")
    ax.set_title("Heteroscedastic range noise")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    zf = z[np.isfinite(z)]
    ax.hist(zf, bins=80, density=True, color="#577590", alpha=0.8, label="normalised residuals")
    xs = np.linspace(-5, 5, 200)
    ax.plot(xs, np.exp(-0.5 * xs**2) / math.sqrt(2 * math.pi), "k", label="N(0,1)")
    ax.set_xlim(-5, 5)
    ax.set_title("HIT residual / sigma")
    ax.legend(fontsize=8)

    fig.suptitle(f"LiDAR noise model validation ({args.scans} scans, config '{args.config}')")
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(f"figure written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
