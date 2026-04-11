#!/usr/bin/env python3
"""Passive UART/serial smoke test for a DFRobot accelerometer on Raspberry Pi TX/RX.

The script does not transmit anything by default. It opens one or more candidate
UART ports, configures raw serial mode, and reports whether bytes are received.
"""

from __future__ import annotations

import argparse
import glob
import os
import select
import sys
import termios
import time
from typing import Iterable


BAUD_CONSTANTS: dict[int, int] = {
    1200: termios.B1200,
    2400: termios.B2400,
    4800: termios.B4800,
    9600: termios.B9600,
    19200: termios.B19200,
    38400: termios.B38400,
    57600: termios.B57600,
    115200: termios.B115200,
    230400: getattr(termios, "B230400", termios.B115200),
}


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
    iflag, oflag, cflag, lflag, ispeed, ospeed, cc = attrs

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


def printable_preview(data: bytes) -> str:
    chars: list[str] = []
    for value in data:
        if value in (10, 13):
            chars.append("\\n")
        elif 32 <= value <= 126:
            chars.append(chr(value))
        else:
            chars.append(".")
    return "".join(chars)


def hex_preview(data: bytes) -> str:
    return " ".join(f"{value:02X}" for value in data)


def read_from_port(port: str, baud: int, duration_s: float, max_bytes: int) -> tuple[bool, bytes, str | None]:
    try:
        fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    except OSError as exc:
        return False, b"", str(exc)

    try:
        configure_raw_uart(fd, baud)
        deadline = time.monotonic() + duration_s
        chunks: list[bytes] = []
        total = 0
        while time.monotonic() < deadline and total < max_bytes:
            timeout_s = min(0.2, max(0.0, deadline - time.monotonic()))
            readable, _, _ = select.select([fd], [], [], timeout_s)
            if not readable:
                continue
            try:
                chunk = os.read(fd, min(256, max_bytes - total))
            except BlockingIOError:
                continue
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
        return True, b"".join(chunks), None
    except (OSError, termios.error, ValueError) as exc:
        return False, b"", str(exc)
    finally:
        os.close(fd)


def parse_baud(value: str) -> int:
    try:
        baud = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid baudrate: {value!r}") from exc
    if baud not in BAUD_CONSTANTS:
        raise argparse.ArgumentTypeError(
            f"unsupported baudrate {baud}; supported: {', '.join(map(str, sorted(BAUD_CONSTANTS)))}"
        )
    return baud


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Comprueba si llega señal serial/UART desde un acelerometro DFRobot conectado a TX/RX.",
    )
    parser.add_argument(
        "--port",
        action="append",
        default=[],
        help="Puerto a probar, por ejemplo /dev/serial0. Puede repetirse. Default: autodetect.",
    )
    parser.add_argument(
        "--baud",
        type=parse_baud,
        action="append",
        default=[],
        help="Baudrate a probar. Puede repetirse. Default: 115200 y 9600.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=3.0,
        help="Segundos de lectura por puerto/baudrate.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=512,
        help="Maximo de bytes que se muestran por prueba.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ports = args.port or [port for port in candidate_ports() if os.path.exists(port)]
    bauds = args.baud or [115200, 9600]
    duration_s = max(0.2, float(args.duration_s))
    max_bytes = max(1, int(args.max_bytes))

    if not ports:
        print("[ERROR] No encontre puertos UART candidatos (/dev/serial0, /dev/ttyS0, /dev/ttyAMA*).", file=sys.stderr)
        print("        Habilita Serial Port en raspi-config y reinicia si hace falta.", file=sys.stderr)
        return 2

    print("[INFO] Puertos probados: " + " ".join(ports))
    print("[INFO] Baudrates probados: " + " ".join(str(baud) for baud in bauds))
    print("[INFO] No envio comandos; solo escucho datos entrantes.")

    saw_open_port = False
    saw_data = False
    for port in ports:
        exists = os.path.exists(port)
        real_port = os.path.realpath(port) if exists else "(missing)"
        print(f"[INFO] Puerto {port} -> {real_port}")
        if not exists:
            print(f"[WARN] {port}: no existe")
            continue

        for baud in bauds:
            ok, data, error = read_from_port(port, baud, duration_s, max_bytes)
            if not ok:
                print(f"[WARN] {port} @ {baud}: no pude abrir/configurar ({error})")
                continue
            saw_open_port = True
            if not data:
                print(f"[WARN] {port} @ {baud}: abierto, pero sin bytes en {duration_s:.1f}s")
                continue

            saw_data = True
            print(f"[OK] {port} @ {baud}: recibidos {len(data)} bytes")
            print(f"     ASCII: {printable_preview(data[:max_bytes])}")
            print(f"     HEX:   {hex_preview(data[:max_bytes])}")

    if saw_data:
        print("[OK] Hay señal UART. Si los bytes cambian al mover el sensor, el cableado RX/TX esta funcionando.")
        return 0
    if saw_open_port:
        print(
            "[WARN] El puerto UART abre, pero no llegaron bytes. Revisa TX/RX cruzados, GND comun, baudrate, "
            "y si el sensor necesita comando para transmitir."
        )
        return 1
    print(
        "[ERROR] No pude abrir ningun puerto UART. Revisa permisos, que el usuario este en el grupo dialout, "
        "y que no haya consola serial usando el puerto."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
