#!/usr/bin/env python3
"""Poll PWM sysfs state independently during a movement run."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import time
from pathlib import Path


def _read_param_value(params_path: Path, key: str) -> str:
    pattern = re.compile(r"^\s*" + re.escape(key) + r"\s*:\s*(.*?)\s*$")
    with params_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].rstrip()
            match = pattern.match(line)
            if match:
                return match.group(1).strip().strip('"').strip("'")
    raise KeyError(f"Parameter not found: {key}")


def _find_chip_path() -> str:
    chips = sorted(glob.glob("/sys/class/pwm/pwmchip*"))
    if not chips:
        raise FileNotFoundError("No PWM chip found under /sys/class/pwm")
    return chips[0]


def _read_int(path: str) -> int | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return int(handle.read().strip())
    except Exception:
        return None


def _snapshot_pwm_channel(pwm_dir: str) -> tuple[int, int | None, int | None, int | None]:
    exists = 1 if os.path.isdir(pwm_dir) else 0
    if not exists:
        return (exists, None, None, None)
    return (
        exists,
        _read_int(os.path.join(pwm_dir, "period")),
        _read_int(os.path.join(pwm_dir, "duty_cycle")),
        _read_int(os.path.join(pwm_dir, "enable")),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--sample-dt-s", type=float, default=0.02)
    args = parser.parse_args()

    params_path = Path(args.params_file).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chip_path = _find_chip_path()
    motor_channel = int(_read_param_value(params_path, "motor_channel"))
    steering_channel = int(_read_param_value(params_path, "steering_channel"))
    motor_pwm_dir = os.path.join(chip_path, f"pwm{motor_channel}")
    steering_pwm_dir = os.path.join(chip_path, f"pwm{steering_channel}")

    monotonic_start = time.monotonic()
    deadline = monotonic_start + max(0.1, float(args.duration_s))
    sample_dt_s = max(0.005, float(args.sample_dt_s))

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "t_s",
                "monotonic_s",
                "motor_pwm_dir_exists",
                "motor_period_ns",
                "motor_duty_cycle_ns",
                "motor_enabled",
                "steering_pwm_dir_exists",
                "steering_period_ns",
                "steering_duty_cycle_ns",
                "steering_enabled",
            ]
        )

        while True:
            monotonic_now = time.monotonic()
            motor_state = _snapshot_pwm_channel(motor_pwm_dir)
            steering_state = _snapshot_pwm_channel(steering_pwm_dir)
            writer.writerow(
                [
                    monotonic_now - monotonic_start,
                    monotonic_now,
                    motor_state[0],
                    motor_state[1],
                    motor_state[2],
                    motor_state[3],
                    steering_state[0],
                    steering_state[1],
                    steering_state[2],
                    steering_state[3],
                ]
            )
            handle.flush()
            if monotonic_now >= deadline:
                break
            time.sleep(min(sample_dt_s, deadline - monotonic_now))


if __name__ == "__main__":
    main()
