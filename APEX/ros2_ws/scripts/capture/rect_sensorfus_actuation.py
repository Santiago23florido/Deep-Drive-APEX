#!/usr/bin/env python3
"""Run a short direct actuation pulse using the normal APEX actuation classes."""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
APEX_PYTHON_ROOT = THIS_FILE.parents[1] / "src" / "apex_telemetry"
if str(APEX_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(APEX_PYTHON_ROOT))

from apex_telemetry.actuation import MaverickESCMotor, SteeringServo


def _read_param_value(params_path: Path, key: str) -> str:
    pattern = re.compile(r"^\s*" + re.escape(key) + r"\s*:\s*(.*?)\s*$")
    with params_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].rstrip()
            match = pattern.match(line)
            if match:
                return match.group(1).strip().strip('"').strip("'")
    raise KeyError(f"Parameter not found: {key}")


def _snapshot_row(monotonic_start: float, motor: MaverickESCMotor, steering: SteeringServo) -> list[object]:
    motor_state = motor.get_state()
    motor_pwm = motor.get_pwm_state()
    steering_state = steering.get_state()
    steering_pwm = steering.get_pwm_state()
    return [
        time.monotonic() - monotonic_start,
        float(motor_state.get("speed_pct", 0.0)),
        float(motor_state.get("pwm_dc", 0.0)),
        int(motor_pwm.get("duty_cycle_ns") or 0),
        int(motor_pwm.get("enabled") or 0),
        float(steering_state.get("requested_deg", 0.0)),
        float(steering_state.get("pwm_dc", 0.0)),
        int(steering_pwm.get("duty_cycle_ns") or 0),
        int(steering_pwm.get("enabled") or 0),
    ]


def _sample_for(
    duration_s: float,
    sample_dt_s: float,
    writer: csv.writer,
    monotonic_start: float,
    motor: MaverickESCMotor,
    steering: SteeringServo,
) -> None:
    deadline = time.monotonic() + max(0.0, duration_s)
    while True:
        writer.writerow(_snapshot_row(monotonic_start, motor, steering))
        now = time.monotonic()
        if now >= deadline:
            break
        time.sleep(min(sample_dt_s, deadline - now))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-file", required=True)
    parser.add_argument("--trace-output", required=True)
    parser.add_argument("--drive-delay-s", type=float, required=True)
    parser.add_argument("--drive-duration-s", type=float, required=True)
    parser.add_argument("--speed-pct", type=float, required=True)
    parser.add_argument("--launch-speed-pct", type=float, default=0.0)
    parser.add_argument("--launch-duration-s", type=float, default=0.0)
    parser.add_argument("--steering-deg", type=float, default=0.0)
    parser.add_argument("--sample-dt-s", type=float, default=0.05)
    parser.add_argument("--pre-arm-neutral-s", type=float, default=0.8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[APEX] %(message)s")
    logger = logging.getLogger("rect_sensorfus_actuation")
    params_path = Path(args.params_file)
    trace_output = Path(args.trace_output)
    trace_output.parent.mkdir(parents=True, exist_ok=True)

    steering = SteeringServo(
        channel=int(_read_param_value(params_path, "steering_channel")),
        frequency_hz=float(_read_param_value(params_path, "steering_frequency_hz")),
        limit_deg=float(_read_param_value(params_path, "steering_limit_deg")),
        dc_min=float(_read_param_value(params_path, "steering_dc_min")),
        dc_max=float(_read_param_value(params_path, "steering_dc_max")),
        center_trim_dc=float(_read_param_value(params_path, "steering_center_trim_dc")),
        direction_sign=float(_read_param_value(params_path, "steering_direction_sign")),
        logger=logger,
    )
    motor = MaverickESCMotor(
        channel=int(_read_param_value(params_path, "motor_channel")),
        frequency_hz=float(_read_param_value(params_path, "motor_frequency_hz")),
        dc_min=float(_read_param_value(params_path, "motor_dc_min")),
        dc_max=float(_read_param_value(params_path, "motor_dc_max")),
        neutral_dc=float(_read_param_value(params_path, "motor_neutral_dc")),
        reverse_brake_dc=float(_read_param_value(params_path, "reverse_brake_dc")),
        reverse_brake_hold_s=float(_read_param_value(params_path, "reverse_brake_hold_s")),
        reverse_neutral_hold_s=float(_read_param_value(params_path, "reverse_neutral_hold_s")),
        reverse_exit_hold_s=float(_read_param_value(params_path, "reverse_exit_hold_s")),
        logger=logger,
    )

    monotonic_start = time.monotonic()
    with trace_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "t_s",
                "motor_speed_pct",
                "motor_pwm_dc_pct",
                "motor_duty_cycle_ns",
                "motor_enabled",
                "steering_deg",
                "steering_pwm_dc_pct",
                "steering_duty_cycle_ns",
                "steering_enabled",
            ]
        )

        try:
            logger.info("Pre-arm neutral hold for %.2fs", max(0.0, args.pre_arm_neutral_s))
            steering.set_angle_deg(0.0)
            motor.hold_neutral(hold_s=max(0.0, args.pre_arm_neutral_s), disable_pwm=False)
            writer.writerow(_snapshot_row(monotonic_start, motor, steering))

            logger.info("Drive delay %.2fs with steering %.2f deg", max(0.0, args.drive_delay_s), args.steering_deg)
            steering.set_angle_deg(float(args.steering_deg))
            _sample_for(args.drive_delay_s, args.sample_dt_s, writer, monotonic_start, motor, steering)

            if abs(args.launch_speed_pct) > abs(args.speed_pct) and args.launch_duration_s > 0.0:
                logger.info(
                    "Launch pulse %.1f%% for %.2fs", float(args.launch_speed_pct), float(args.launch_duration_s)
                )
                motor.set_speed_pct(float(args.launch_speed_pct))
                _sample_for(args.launch_duration_s, args.sample_dt_s, writer, monotonic_start, motor, steering)

            logger.info("Drive pulse %.1f%% for %.2fs", float(args.speed_pct), float(args.drive_duration_s))
            motor.set_speed_pct(float(args.speed_pct))
            _sample_for(args.drive_duration_s, args.sample_dt_s, writer, monotonic_start, motor, steering)

            logger.info("Return to neutral")
            motor.hold_neutral(disable_pwm=False)
            steering.center()
            writer.writerow(_snapshot_row(monotonic_start, motor, steering))
        finally:
            try:
                motor.hold_neutral(disable_pwm=False)
            except Exception:
                pass
            try:
                steering.center()
            except Exception:
                pass


if __name__ == "__main__":
    main()
