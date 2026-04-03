"""
main_refactored.py — Ponto de entrada principal do veículo autónomo.
"""

import time
import sys

from code.Interfaces.LidarInterface import create_lidar_interface
from code.Interfaces.MotorInterface import create_motor_interface, create_steer_interface
from code.config_loader import get_config
from code.log_manager import initialize_logging, get_logger, shutdown_logging
from code.Interfaces.interface_serial import (
    SharedMemBatteryInterface,
    SharedMemUltrasonicInterface,
    SharedMemSpeedInterface,
    start_serial_monitor,
)
from code.Interfaces.interface_camera import RealCameraInterface
from code.VehicleAlgorithm_v1 import VehicleAlgorithm_v1


def main():
    # 1. Carregar configuração
    config = get_config("config.json")
    print(f"Configuration loaded: {config.general.car_name}")

    # 2. Inicializar logging
    initialize_logging(
        console_level=config.general.log_level,
        enable_colors=True,
        backup_config=True,
    )
    logger = get_logger("Main")
    logger.info("=" * 60)
    logger.info(f"Starting {config.general.car_name}")
    logger.info("=" * 60)

    # 3. Inicializar sensores
    logger.info("Initializing hardware interfaces…")

    # Série (velocidade, ultrassónico, bateria)
    start_serial_monitor()
        # port=config.serial.port, baudrate=config.serial.baudrate)

    lidar        = create_lidar_interface("real", config)
    ultrasonic   = SharedMemUltrasonicInterface()
    speed_sensor = SharedMemSpeedInterface()
    battery      = SharedMemBatteryInterface()
    camera       = RealCameraInterface()

    # 4. Inicializar actuadores
    # create_motor_interface(config) sem o argumento 'mode', causando TypeError.
    steer = create_steer_interface(mode="real")
    motor = create_motor_interface(mode="real", config_file=config) 

    # 5. Criar algoritmo do veículo
    vehicle = VehicleAlgorithm_v1(
        config=config,
        lidar=lidar,
        ultrasonic=ultrasonic,
        speed=speed_sensor,
        battery=battery,
        camera=camera,
        steer=steer,
        motor=motor,
    )

    # 6. Arrancar
    lidar.start()
    vehicle.start()
    logger.info("Vehicle started")

    try:
        logger.info("Entering main control loop…")
        i = 0
        while True:
            vehicle.run_step()
            time.sleep(0.05)  # 20 Hz

            if i == 50:
                vehicle.change_strategy("aggressive")
            i += 1

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")

    except Exception as e:
        logger.exception(f"Error in main loop: {e}")
        vehicle.emergency_stop()

    finally:
        vehicle.cleanup()
        lidar.stop()
        shutdown_logging()
        logger.info("Application terminated")


def simulating_main():
    """Modo de simulação — usa instâncias mock para testes sem hardware."""

    config = get_config("simulation_config.json")
    print(f"Configuration loaded: {config.general.car_name}")

    initialize_logging(
        console_level=config.general.log_level,
        enable_colors=True,
        backup_config=True,
    )
    logger = get_logger("Main")
    logger.info("=" * 60)
    logger.info(f"[SIMULATION] Starting {config.general.car_name}")
    logger.info("=" * 60)

    # BUG FIX: o original referenciava MockLiDarInterface, MockUltrasonicInterface, etc.
    # sem as importar — causava NameError imediato.
    # Em modo simulação usamos as factories com mode="simulation".
    lidar        = create_lidar_interface("simulation", config)
    motor        = create_motor_interface("simulation", config)
    steer        = create_steer_interface("simulation", config)

    # Para sensores série em simulação usamos stubs simples inline:
    from code.Interfaces.interface_decl import UltrasonicInterface, SpeedInterface, BatteryInterface
    import numpy as np

    class _MockUltrasonic(UltrasonicInterface):
        def get_ultrasonic_data(self): return 100.0

    class _MockSpeed(SpeedInterface):
        def get_speed(self): return 0.0

    class _MockBattery(BatteryInterface):
        def get_battery_voltage(self): return 7.4

    class _MockCamera:
        def get_camera_frame(self): return np.zeros((480, 640, 3), dtype=np.uint8)
        def get_resolution(self): return (640, 480)

    algorithm = VehicleAlgorithm_v1(
        config=config,
        lidar=lidar,
        ultrasonic=_MockUltrasonic(),
        speed=_MockSpeed(),
        battery=_MockBattery(),
        camera=_MockCamera(),
        steer=steer,
        motor=motor,
    )

    logger.info("Waiting for system initialization…")
    lidar.start()
    time.sleep(2)

    algorithm.start()
    logger.info("Vehicle started")

    try:
        logger.info("Entering main control loop…")
        for i in range(100):
            algorithm.run_step()
            time.sleep(0.05)

            if i == 50:
                algorithm.change_strategy("aggressive")

        logger.info("Main loop completed")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")

    except Exception as e:
        logger.exception(f"Error in main loop: {e}")
        algorithm.emergency_stop()

    finally:
        algorithm.cleanup()
        lidar.stop()
        shutdown_logging()
        logger.info("Application terminated")


if __name__ == "__main__":
    main()