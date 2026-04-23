#!/usr/bin/env python3
"""Quick local diagnostic for Xbox-style controllers via pygame."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

try:
    import pygame  # noqa: E402
except ModuleNotFoundError as exc:
    system_python = Path("/usr/bin/python3")
    if (
        exc.name == "pygame"
        and system_python.exists()
        and Path(sys.executable).resolve() != system_python.resolve()
    ):
        os.execv(str(system_python), [str(system_python), __file__, *sys.argv[1:]])
    raise


def _round_list(values, digits: int = 3):
    return [round(float(value), digits) for value in values]


def _list_devices() -> list[pygame.joystick.Joystick]:
    pygame.joystick.quit()
    pygame.joystick.init()
    devices: list[pygame.joystick.Joystick] = []
    for index in range(pygame.joystick.get_count()):
        joystick = pygame.joystick.Joystick(index)
        joystick.init()
        devices.append(joystick)
    return devices


def _select_device(devices, *, index: int, name_contains: str) -> pygame.joystick.Joystick | None:
    if not devices:
        return None
    if 0 <= index < len(devices):
        return devices[index]
    if name_contains:
        target = name_contains.strip().lower()
        for device in devices:
            if target in device.get_name().strip().lower():
                return device
    return devices[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=-1, help="Joystick index to open.")
    parser.add_argument(
        "--name-contains",
        default="xbox",
        help="Case-insensitive device-name filter. Empty string disables it.",
    )
    parser.add_argument(
        "--poll-hz",
        type=float,
        default=12.0,
        help="Console refresh rate while streaming values.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list devices and exit.",
    )
    args = parser.parse_args()

    pygame.init()
    pygame.joystick.init()

    try:
        devices = _list_devices()
        print(f"[xbox-check] joysticks_detected={len(devices)}")
        for index, device in enumerate(devices):
            print(
                f"[xbox-check] index={index} name={device.get_name()} "
                f"axes={device.get_numaxes()} buttons={device.get_numbuttons()} hats={device.get_numhats()}"
            )

        if args.list_only:
            return 0

        joystick = _select_device(
            devices,
            index=args.index,
            name_contains=args.name_contains,
        )
        if joystick is None:
            print("[xbox-check][ERROR] No joystick found.")
            return 1

        print(
            f"[xbox-check] using index={joystick.get_instance_id()} name={joystick.get_name()} "
            f"guid={joystick.get_guid()}"
        )
        print("[xbox-check] move sticks / triggers / buttons. Ctrl+C to exit.")

        period_s = 1.0 / max(1.0, float(args.poll_hz))
        while True:
            pygame.event.pump()

            axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
            buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
            hats = [joystick.get_hat(i) for i in range(joystick.get_numhats())]

            payload = {
                "time_s": round(time.monotonic(), 3),
                "name": joystick.get_name(),
                "axes": _round_list(axes),
                "buttons": buttons,
                "hats": hats,
            }
            sys.stdout.write("\r" + json.dumps(payload, separators=(",", ":")) + " " * 8)
            sys.stdout.flush()
            time.sleep(period_s)
    except KeyboardInterrupt:
        print("\n[xbox-check] stopped")
        return 0
    finally:
        try:
            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
