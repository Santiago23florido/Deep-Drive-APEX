#!/usr/bin/env python3
"""Read DFRobot WT61PC/SEN0386 UART frames on a Raspberry Pi.

The WT61PC streams 11-byte binary frames at 9600 baud by default:
0x55 0x51 for acceleration, 0x55 0x52 for gyro, and 0x55 0x53 for angle.
This script is a smoke test for direct Raspberry Pi TX/RX wiring.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import os
import select
import sys
import termios
import time
from typing import Iterable


FRAME_START = 0x55
FRAME_LEN = 11
FRAME_ACC = 0x51
FRAME_GYRO = 0x52
FRAME_ANGLE = 0x53

FRAME_NAMES = {
    FRAME_ACC: "Acc",
    FRAME_GYRO: "Gyro",
    FRAME_ANGLE: "Angle",
}

BAUD_CONSTANTS = {
    9600: termios.B9600,
    115200: termios.B115200,
}


@dataclass(frozen=True)
class Measurement:
    kind: int
    x: float
    y: float
    z: float


def unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def candidate_ports() -> list[str]:
    return unique(
        [
            "/dev/serial0",
            "/dev/ttyS0",
            "/dev/ttyAMA0",
            "/dev/ttyAMA10",
            *sorted(glob.glob("/dev/ttyAMA*")),
            *sorted(glob.glob("/dev/ttyS*")),
        ]
    )


def configure_raw_uart(fd: int, baud: int) -> None:
    baud_const = BAUD_CONSTANTS.get(baud)
    if baud_const is None:
        raise ValueError(f"unsupported baudrate {baud}; supported: {sorted(BAUD_CONSTANTS)}")

    attrs = termios.tcgetattr(fd)
    iflag, oflag, cflag, lflag, _ispeed, _ospeed, cc = attrs
    iflag = 0
    oflag = 0
    lflag = 0
    cflag |= termios.CREAD | termios.CLOCAL
    cflag &= ~termios.CSIZE
    cflag |= termios.CS8
    cflag &= ~termios.PARENB
    cflag &= ~termios.CSTOPB
    if hasattr(termios, "CRTSCTS"):
        cflag &= ~termios.CRTSCTS
    cc[termios.VMIN] = 0
    cc[termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, [iflag, oflag, cflag, lflag, baud_const, baud_const, cc])
    termios.tcflush(fd, termios.TCIOFLUSH)


def checksum_ok(frame: bytes) -> bool:
    return len(frame) == FRAME_LEN and (sum(frame[:10]) & 0xFF) == frame[10]


def signed_i16(lo: int, hi: int) -> int:
    value = (hi << 8) | lo
    if value >= 0x8000:
        value -= 0x10000
    return value


def parse_measurement(frame: bytes) -> Measurement | None:
    if len(frame) != FRAME_LEN or frame[0] != FRAME_START or not checksum_ok(frame):
        return None
    kind = frame[1]
    raw_x = signed_i16(frame[2], frame[3])
    raw_y = signed_i16(frame[4], frame[5])
    raw_z = signed_i16(frame[6], frame[7])
    if kind == FRAME_ACC:
        scale = 16.0 * 9.8 / 32768.0
    elif kind == FRAME_GYRO:
        scale = 2000.0 / 32768.0
    elif kind == FRAME_ANGLE:
        scale = 180.0 / 32768.0
    else:
        return None
    return Measurement(kind=kind, x=raw_x * scale, y=raw_y * scale, z=raw_z * scale)


def read_byte(fd: int, timeout_s: float) -> int | None:
    readable, _, _ = select.select([fd], [], [], timeout_s)
    if not readable:
        return None
    data = os.read(fd, 1)
    if not data:
        return None
    return data[0]


def read_frames(fd: int, *, duration_s: float, max_frames: int) -> list[Measurement]:
    deadline = time.monotonic() + duration_s
    measurements: list[Measurement] = []
    while time.monotonic() < deadline and len(measurements) < max_frames:
        byte = read_byte(fd, max(0.0, min(0.2, deadline - time.monotonic())))
        if byte is None or byte != FRAME_START:
            continue
        kind = read_byte(fd, max(0.0, min(0.2, deadline - time.monotonic())))
        if kind not in FRAME_NAMES:
            continue
        remaining = bytearray()
        while len(remaining) < FRAME_LEN - 2 and time.monotonic() < deadline:
            next_byte = read_byte(fd, max(0.0, min(0.2, deadline - time.monotonic())))
            if next_byte is None:
                break
            remaining.append(next_byte)
        if len(remaining) != FRAME_LEN - 2:
            continue
        frame = bytes([FRAME_START, kind, *remaining])
        measurement = parse_measurement(frame)
        if measurement is not None:
            measurements.append(measurement)
    return measurements


def open_and_read(port: str, baud: int, *, duration_s: float, max_frames: int) -> tuple[list[Measurement], str | None]:
    try:
        fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    except OSError as exc:
        return [], str(exc)
    try:
        configure_raw_uart(fd, baud)
        return read_frames(fd, duration_s=duration_s, max_frames=max_frames), None
    except (OSError, termios.error, ValueError) as exc:
        return [], str(exc)
    finally:
        os.close(fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lee tramas UART del DFRobot WT61PC/SEN0386 desde la Raspberry.",
    )
    parser.add_argument("--port", action="append", default=[], help="Puerto UART. Default: autodetect.")
    parser.add_argument("--baud", type=int, default=9600, help="Baudrate. Default: 9600.")
    parser.add_argument("--duration-s", type=float, default=8.0, help="Duracion de la prueba.")
    parser.add_argument("--max-frames", type=int, default=30, help="Maximo de tramas validas a mostrar.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ports = args.port or [port for port in candidate_ports() if os.path.exists(port)]
    if not ports:
        print("[ERROR] No encontre /dev/serial0, /dev/ttyS0 ni /dev/ttyAMA*.", file=sys.stderr)
        print("        Habilita el UART de la Raspberry y reinicia.", file=sys.stderr)
        return 2

    print("[INFO] Sensor esperado: DFRobot WT61PC/SEN0386 UART")
    print("[INFO] Cableado: sensor TXD -> Raspberry RXD GPIO15 pin 10; sensor RXD -> Raspberry TXD GPIO14 pin 8; GND comun.")
    print("[INFO] Puertos probados: " + " ".join(ports))
    print(f"[INFO] Baudrate: {int(args.baud)}")

    for port in ports:
        real_port = os.path.realpath(port)
        print(f"[INFO] Probando {port} -> {real_port}")
        measurements, error = open_and_read(
            port,
            int(args.baud),
            duration_s=max(0.5, float(args.duration_s)),
            max_frames=max(1, int(args.max_frames)),
        )
        if error is not None:
            print(f"[WARN] {port}: no pude abrir/configurar ({error})")
            continue
        if not measurements:
            print(f"[WARN] {port}: abierto, pero sin tramas WT61PC validas.")
            continue

        for measurement in measurements:
            name = FRAME_NAMES[measurement.kind]
            unit = "m/s^2" if measurement.kind == FRAME_ACC else ("deg/s" if measurement.kind == FRAME_GYRO else "deg")
            print(
                f"[OK] {port} {name:<5} "
                f"X={measurement.x:9.3f} Y={measurement.y:9.3f} Z={measurement.z:9.3f} {unit}"
            )
        print(f"[OK] Recibi {len(measurements)} tramas validas en {port}.")
        return 0

    print(
        "[ERROR] No recibi tramas WT61PC validas. Revisa TX/RX cruzados, GND comun, UART habilitado, "
        "baudrate 9600, y que la consola serial de Linux no este usando el puerto."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
