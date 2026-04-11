#!/usr/bin/env python3
"""Check whether an I2C DFRobot accelerometer is visible from the Raspberry.

This is a dependency-free diagnostic script intended to run on the Raspberry Pi,
including from inside the APEX Docker container. It only reads registers; it does
not configure or reset the sensor.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import glob
import os
import sys
import time
from typing import Iterable


I2C_SLAVE = 0x0703
COMMON_ID_REGISTERS = (0x00, 0x0D, 0x0F, 0x75)


@dataclass(frozen=True)
class SensorSignature:
    name: str
    addresses: tuple[int, ...]
    id_register: int
    expected_ids: tuple[int, ...]
    sample_register: int | None
    sample_length: int = 6
    sample_endian: str = "little"


SIGNATURES: tuple[SensorSignature, ...] = (
    SensorSignature("ADXL345", (0x1D, 0x53), 0x00, (0xE5,), 0x32, 6, "little"),
    SensorSignature("BNO055", (0x28, 0x29), 0x00, (0xA0,), 0x08, 6, "little"),
    SensorSignature("BMI160/BMX160/BMI270 family", (0x68, 0x69), 0x00, (0x24, 0xD1, 0xD8), 0x12, 6, "little"),
    SensorSignature("ICM-20948", (0x68, 0x69), 0x00, (0xEA,), 0x2D, 6, "little"),
    SensorSignature("MMA845x", (0x1C, 0x1D), 0x0D, (0x1A, 0x2A, 0x3A, 0x4A, 0x5A), 0x01, 6, "big"),
    SensorSignature("LIS3DH", (0x18, 0x19), 0x0F, (0x33,), 0x28, 6, "little"),
    SensorSignature("LIS2DW12", (0x18, 0x19), 0x0F, (0x44,), 0x28, 6, "little"),
    SensorSignature("LSM6DS3/LSM6DSL/LSM6DSOX family", (0x6A, 0x6B), 0x0F, (0x69, 0x6A, 0x6C), 0x28, 6, "little"),
    SensorSignature("MPU6050/MPU6500/MPU9250 family", (0x68, 0x69), 0x75, (0x68, 0x70, 0x71, 0x73), 0x3B, 6, "big"),
    SensorSignature("QMI8658 family", (0x6A, 0x6B), 0x00, (0x05,), 0x35, 6, "little"),
)


def parse_int(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer/address: {value!r}") from exc
    if not 0x03 <= parsed <= 0x77:
        raise argparse.ArgumentTypeError("I2C addresses must be in the 0x03..0x77 range")
    return parsed


def unique(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def known_accelerometer_addresses() -> list[int]:
    values: list[int] = []
    for signature in SIGNATURES:
        values.extend(signature.addresses)
    return unique(sorted(values))


def auto_bus() -> str | None:
    candidates = unique(["/dev/i2c-1", "/dev/i2c-0", *sorted(glob.glob("/dev/i2c-*"))])
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


class LinuxI2CBus:
    def __init__(self, path: str) -> None:
        self.path = path
        self._fd = os.open(path, os.O_RDWR)

    def close(self) -> None:
        os.close(self._fd)

    def select_address(self, address: int) -> None:
        fcntl.ioctl(self._fd, I2C_SLAVE, address)

    def read_register(self, address: int, register: int) -> int:
        self.select_address(address)
        os.write(self._fd, bytes((register & 0xFF,)))
        data = os.read(self._fd, 1)
        if len(data) != 1:
            raise OSError(f"short read from address 0x{address:02X} register 0x{register:02X}")
        return data[0]

    def read_registers(self, address: int, start_register: int, length: int) -> list[int]:
        return [self.read_register(address, start_register + offset) for offset in range(length)]


@dataclass(frozen=True)
class Detection:
    address: int
    id_values: dict[int, int]
    matches: tuple[SensorSignature, ...]


def read_id_values(bus: LinuxI2CBus, address: int) -> dict[int, int]:
    values: dict[int, int] = {}
    for register in COMMON_ID_REGISTERS:
        try:
            values[register] = bus.read_register(address, register)
        except OSError:
            continue
    return values


def match_signatures(address: int, id_values: dict[int, int]) -> tuple[SensorSignature, ...]:
    matches: list[SensorSignature] = []
    for signature in SIGNATURES:
        if address not in signature.addresses:
            continue
        if id_values.get(signature.id_register) in signature.expected_ids:
            matches.append(signature)
    return tuple(matches)


def detect(bus: LinuxI2CBus, addresses: Iterable[int]) -> list[Detection]:
    detections: list[Detection] = []
    for address in addresses:
        id_values = read_id_values(bus, address)
        if not id_values:
            continue
        detections.append(
            Detection(
                address=address,
                id_values=id_values,
                matches=match_signatures(address, id_values),
            )
        )
    return detections


def format_id_values(values: dict[int, int]) -> str:
    return ", ".join(f"reg 0x{register:02X}=0x{value:02X}" for register, value in sorted(values.items()))


def decode_xyz(raw: list[int], endian: str) -> tuple[int, int, int] | None:
    if len(raw) < 6:
        return None
    byteorder = "big" if endian == "big" else "little"
    return tuple(
        int.from_bytes(bytes(raw[index : index + 2]), byteorder=byteorder, signed=True)
        for index in (0, 2, 4)
    )


def sample_detection(
    bus: LinuxI2CBus,
    detection: Detection,
    *,
    samples: int,
    interval_s: float,
) -> None:
    signature = next(
        (match for match in detection.matches if match.sample_register is not None),
        None,
    )
    if signature is None:
        print(f"[INFO] 0x{detection.address:02X}: sensor responde, pero no tengo mapa de datos para muestrear este modelo.")
        return

    print(
        f"[INFO] 0x{detection.address:02X}: leyendo {samples} muestras crudas "
        f"desde {signature.name} reg 0x{signature.sample_register:02X}. Mueve el sensor durante esta parte."
    )
    previous: list[int] | None = None
    changed_count = 0
    for index in range(samples):
        try:
            raw = bus.read_registers(
                detection.address,
                signature.sample_register,
                signature.sample_length,
            )
        except OSError as exc:
            print(f"[ERROR] 0x{detection.address:02X}: no pude leer datos crudos: {exc}")
            return

        changed = previous is not None and raw != previous
        if changed:
            changed_count += 1
        xyz = decode_xyz(raw, signature.sample_endian)
        raw_hex = " ".join(f"{value:02X}" for value in raw)
        xyz_text = f" xyz_raw={xyz}" if xyz is not None else ""
        print(
            f"  sample {index + 1:02d}: bytes={raw_hex}{xyz_text} "
            f"changed={'yes' if changed else 'no'}"
        )
        previous = raw
        if index + 1 < samples:
            time.sleep(interval_s)

    if changed_count > 0:
        print(f"[OK] 0x{detection.address:02X}: hay lectura I2C y los datos crudos cambiaron.")
    else:
        print(
            f"[WARN] 0x{detection.address:02X}: hay lectura I2C, pero las muestras crudas no cambiaron. "
            "Puede estar quieto, en standby, o necesitar el driver exacto/configuracion del modelo."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Comprueba si un acelerometro/IMU I2C tipo DFRobot responde en la Raspberry.",
    )
    parser.add_argument(
        "--bus",
        default=os.environ.get("APEX_I2C_BUS"),
        help="I2C bus path. Default: $APEX_I2C_BUS or auto-detect /dev/i2c-1.",
    )
    parser.add_argument(
        "-a",
        "--address",
        type=parse_int,
        action="append",
        help="Direccion I2C esperada, por ejemplo --address 0x19. Can be repeated.",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Escanea todo el rango 0x03..0x77. Por defecto solo prueba direcciones comunes de acelerometros.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=8,
        help="Numero de muestras crudas que se leen si se reconoce el sensor. Usa 0 para desactivar.",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=0.25,
        help="Tiempo entre muestras crudas.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bus_path = args.bus or auto_bus()
    if not bus_path:
        print("[ERROR] No encontre ningun /dev/i2c-* dentro del entorno actual.", file=sys.stderr)
        print("        En la Raspberry habilita I2C y recrea/reinicia el contenedor.", file=sys.stderr)
        print("        Comprobacion rapida: ls -l /dev/i2c-*", file=sys.stderr)
        return 2
    if not os.path.exists(bus_path):
        print(f"[ERROR] El bus I2C no existe: {bus_path}", file=sys.stderr)
        print("        Si estas dentro de Docker, revisa que /dev/i2c-1 exista dentro del contenedor.", file=sys.stderr)
        return 2

    if args.address:
        addresses = unique(args.address)
    elif args.full_scan:
        addresses = list(range(0x03, 0x78))
    else:
        addresses = known_accelerometer_addresses()

    print(f"[INFO] Bus: {bus_path}")
    print("[INFO] Direcciones probadas: " + " ".join(f"0x{address:02X}" for address in addresses))

    try:
        bus = LinuxI2CBus(bus_path)
    except OSError as exc:
        print(f"[ERROR] No pude abrir {bus_path}: {exc}", file=sys.stderr)
        print("        Normalmente esto es falta de permisos, I2C deshabilitado, o el device no entro al contenedor.", file=sys.stderr)
        return 2

    try:
        detections = detect(bus, addresses)
        if not detections:
            print("[WARN] No respondio ninguna direccion probada.")
            if not args.full_scan and not args.address:
                print("       Prueba tambien con: --full-scan")
            print("       Si tu DFRobot es analogico, la Raspberry necesita un ADC; no aparecera por I2C.")
            return 1

        for detection in detections:
            print(f"[OK] Direccion 0x{detection.address:02X} responde ({format_id_values(detection.id_values)})")
            if detection.matches:
                print(
                    "[OK] Coincidencia probable: "
                    + ", ".join(match.name for match in detection.matches)
                )
            else:
                print("[INFO] Responde por I2C, pero no coincide con las firmas conocidas de este script.")

        samples = max(0, int(args.samples))
        if samples > 0:
            for detection in detections:
                sample_detection(
                    bus,
                    detection,
                    samples=samples,
                    interval_s=max(0.0, float(args.interval_s)),
                )
        return 0
    finally:
        bus.close()


if __name__ == "__main__":
    raise SystemExit(main())
