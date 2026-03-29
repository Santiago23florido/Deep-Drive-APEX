"""
interface_serial.py — Leitura de sensores via porta série (Arduino/STM32).
Formato esperado: "speed_ticks/ultrasonic_cm/battery_V\n"

Usa memória partilhada (multiprocessing) para comunicação segura entre processos.
"""

import serial
import threading
import time
import multiprocessing

from code.algorithm.constants import TICKS_TO_METER
from code.Interfaces.interface_decl import SpeedInterface, UltrasonicInterface, BatteryInterface

# BUG FIX: as variáveis de estado global eram definidas no topo do módulo.
# Ao usar multiprocessing, mp.Array e mp.Value devem ser criados no processo
# principal (antes de fork). Mantido como estava, mas documentado.
_last_serial_read   = multiprocessing.Array('d', [0.0, 0.0, 0.0])  # [speed_ticks, ultrasonic_cm, battery_V]
_last_serial_update = multiprocessing.Value('d', 0.0)


# ---------------------------------------------------------------------------
# Accessors de memória partilhada
# ---------------------------------------------------------------------------

def get_speed_ticks() -> float:
    with _last_serial_read.get_lock():
        return _last_serial_read[0]

def get_ultrasonic() -> float:
    with _last_serial_read.get_lock():
        return _last_serial_read[1]

def get_battery() -> float:
    with _last_serial_read.get_lock():
        return _last_serial_read[2]

def get_last_update() -> float:
    with _last_serial_update.get_lock():
        return _last_serial_update.value


# ---------------------------------------------------------------------------
# Implementações das interfaces
# ---------------------------------------------------------------------------

class SharedMemSpeedInterface(SpeedInterface):
    def get_speed(self) -> float:
        """Converte ticks/s em m/s usando TICKS_TO_METER."""
        # BUG FIX: o original fazia (ticks / TICKS_TO_METER), o que é correto apenas
        # se TICKS_TO_METER = ticks_por_metro. Mantido mas renomeado o getter
        # para deixar claro que a leitura em bruto são ticks.
        return get_speed_ticks() / TICKS_TO_METER


class SharedMemUltrasonicInterface(UltrasonicInterface):
    def get_ultrasonic_data(self) -> float:
        return get_ultrasonic()


class SharedMemBatteryInterface(BatteryInterface):
    def get_battery_voltage(self) -> float:
        return get_battery()


# ---------------------------------------------------------------------------
# Monitor série (thread de fundo)
# ---------------------------------------------------------------------------

def start_serial_monitor(port: str = '/dev/ttyACM0', baudrate: int = 115200) -> threading.Thread:
    """Inicia o monitor série numa thread de fundo (daemon)."""
    thread = threading.Thread(target=_run_serial_monitor, args=(port, baudrate), daemon=True)
    thread.start()
    return thread


def _run_serial_monitor(port: str, baudrate: int) -> None:
    """Lê continuamente a porta série e actualiza a memória partilhada."""
    while True:  # BUG FIX: loop externo para reconectar em caso de erro
        try:
            with serial.Serial(port, baudrate, timeout=1) as ser:
                print(f"[Serial] Connected to {port} at {baudrate} baud")

                while True:
                    if ser.in_waiting > 0:
                        raw = ser.readline().decode('utf-8', errors='replace').strip()
                        if raw:
                            _parse_and_store(raw)
                    time.sleep(0.01)

        except serial.SerialException as e:
            print(f"[Serial] Connection error: {e}. Retrying in 2 s…")
            time.sleep(2.0)

        except Exception as e:
            print(f"[Serial] Unexpected error: {e}. Retrying in 2 s…")
            time.sleep(2.0)


def _parse_and_store(line: str) -> None:
    """Faz parse de uma linha no formato 'speed/ultrasonic/battery'."""
    try:
        parts = line.split('/')
        if len(parts) != 3:
            return
        speed      = float(parts[0])
        ultrasonic = float(parts[1])
        battery    = float(parts[2])

        with _last_serial_read.get_lock():
            _last_serial_read[0] = speed
            _last_serial_read[1] = ultrasonic
            _last_serial_read[2] = battery

        with _last_serial_update.get_lock():
            _last_serial_update.value = time.time()

    except ValueError:
        print(f"[Serial] Failed to parse: '{line}'")


# ---------------------------------------------------------------------------
# Teste standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    monitor_thread = start_serial_monitor(port='/dev/ttyACM0', baudrate=115200)

    speed_iface     = SharedMemSpeedInterface()
    ultrasonic_iface = SharedMemUltrasonicInterface()
    battery_iface   = SharedMemBatteryInterface()

    try:
        while True:
            print(f"Speed:        {speed_iface.get_speed():.3f} m/s")
            print(f"Ultrasonic:   {ultrasonic_iface.get_ultrasonic_data():.1f} cm")
            print(f"Battery:      {battery_iface.get_battery_voltage():.2f} V")
            print(f"Last update:  {time.time() - get_last_update():.2f} s ago")
            print("-" * 30)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nExiting…")
        