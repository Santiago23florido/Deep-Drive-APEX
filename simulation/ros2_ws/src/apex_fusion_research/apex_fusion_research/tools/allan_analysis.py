"""Allan-variance analysis of IMU data (simulated or recorded).

Synthetic mode validates the IMU model: a static IMU is simulated from a
preset and the identified coefficients are compared with the configured ones.

    ros2 run apex_fusion_research allan_analysis --config consumer_mems --duration 7200

CSV mode identifies a real sensor from a static recording (columns in SI units):

    ros2 run apex_fusion_research allan_analysis --csv static_imu.csv \
        --time-col t --gyro-cols gx gy gz --accel-cols ax ay az

Output: a figure with the Allan deviation of every axis and a table of the
identified white noise N, bias instability B and random walk K.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..core.allan import fit_allan_coefficients, overlapping_allan_deviation
from ..core.config_io import dataclass_from_flat, load_ros_params_yaml
from ..core.imu_error import ImuErrorConfig, ImuErrorModel


def _resolve_preset(value: str) -> Path:
    path = Path(value).expanduser()
    if path.exists():
        return path
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("apex_fusion_research"))
    except Exception:
        share = Path(__file__).resolve().parents[2]
    preset = share / "config" / "imu" / f"{value}.yaml"
    if not preset.exists():
        raise SystemExit(f"IMU config '{value}' not found")
    return preset


def simulate_static(config: ImuErrorConfig, duration_s: float, rate_hz: float, gravity: float = 9.8):
    model = ImuErrorModel(config)
    n = int(duration_s * rate_hz)
    dt = 1.0 / rate_hz
    gyro = np.empty((n, 3))
    accel = np.empty((n, 3))
    w0 = np.zeros(3)
    f0 = np.array([0.0, 0.0, gravity])
    for k in range(n):
        s = model.measure(w0, f0, dt)
        gyro[k] = s.angular_velocity
        accel[k] = s.specific_force
    return dt, gyro, accel


def load_csv(path: Path, time_col: str, gyro_cols, accel_cols, rate_hz: float | None):
    data = np.genfromtxt(path, delimiter=",", names=True)
    if rate_hz:
        dt = 1.0 / rate_hz
    else:
        dt = float(np.median(np.diff(data[time_col])))
    gyro = np.column_stack([data[c] for c in gyro_cols]) if gyro_cols else None
    accel = np.column_stack([data[c] for c in accel_cols]) if accel_cols else None
    return dt, gyro, accel


def analyse(name: str, samples: np.ndarray, dt: float):
    results = []
    for axis in range(samples.shape[1]):
        tau, adev = overlapping_allan_deviation(samples[:, axis] - samples[:, axis].mean(), dt)
        results.append((tau, adev, fit_allan_coefficients(tau, adev)))
    return results


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="IMU preset name or YAML (synthetic mode)")
    parser.add_argument("--duration", type=float, default=3600.0, help="synthetic record length [s]")
    parser.add_argument("--rate", type=float, default=None, help="sample rate [Hz] (default 120 for synthetic)")
    parser.add_argument("--csv", type=Path, default=None, help="static IMU recording (CSV mode)")
    parser.add_argument("--time-col", default="t")
    parser.add_argument("--gyro-cols", nargs="*", default=[])
    parser.add_argument("--accel-cols", nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=Path("allan_deviation.png"))
    args = parser.parse_args(argv)

    configured = None
    if args.csv is not None:
        dt, gyro, accel = load_csv(args.csv, args.time_col, args.gyro_cols, args.accel_cols, args.rate)
        title = f"Allan deviation - {args.csv.name}"
    else:
        cfg_path = _resolve_preset(args.config or "consumer_mems")
        configured = dataclass_from_flat(ImuErrorConfig, load_ros_params_yaml(cfg_path))
        dt, gyro, accel = simulate_static(configured, args.duration, args.rate or 120.0)
        title = f"Allan deviation - simulated '{cfg_path.stem}' ({args.duration:.0f} s)"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sensors = [(n, s) for n, s in (("gyro", gyro), ("accel", accel)) if s is not None]
    fig, axes = plt.subplots(1, len(sensors), figsize=(6.5 * len(sensors), 5), squeeze=False)
    units = {"gyro": ("rad/s", "rad/s/sqrt(Hz)", "rad/s/sqrt(s)"), "accel": ("m/s^2", "m/s^2/sqrt(Hz)", "m/s^2/sqrt(s)")}
    print(f"{'sensor':<6} {'axis':<4} {'N':>12} {'B':>12} {'K':>12}   ({'configured N / K' if configured else ''})")
    for ax, (name, samples) in zip(axes[0], sensors):
        for axis, (tau, adev, coef) in enumerate(analyse(name, samples, dt)):
            ax.loglog(tau, adev, label=f"{'xyz'[axis]}: N={coef.noise_density:.2e} B={coef.bias_instability:.2e} K={coef.random_walk:.2e}")
            ref = ""
            if configured is not None:
                triad = getattr(configured, name)
                ref = f"   ({triad.noise_density:.3e} / {triad.bias_random_walk:.3e})"
            print(f"{name:<6} {'xyz'[axis]:<4} {coef.noise_density:12.4e} {coef.bias_instability:12.4e} {coef.random_walk:12.4e}{ref}")
        u, un, uk = units[name]
        ax.set_xlabel("averaging time tau [s]")
        ax.set_ylabel(f"Allan deviation [{u}]")
        ax.set_title(f"{name}  (N [{un}], K [{uk}])")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(f"figure written to {args.output.resolve()}")
    if configured is not None:
        print("note: B from a Gauss-Markov model is not a flicker process; its fitted value is indicative only.")


if __name__ == "__main__":
    main()
