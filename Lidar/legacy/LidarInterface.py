from logging import config
import sys, os, time

from rplidar import RPLidar, RPLidarException
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any
import numpy as np
from typing import Optional
import multiprocessing as mp


from code.log_manager import get_component_logger
from code.Interfaces.interfaces import AbstractLidarInterface
from code.config_loader import ConfigLoader, get_config



class LidarMode(Enum):
    """Modos de operação do LiDAR"""
    REAL = "real"               # LiDAR físico real
    SIMULATION = "simulation"   # Simulated LiDAR (random values as input)


class RealLidarInterface(AbstractLidarInterface):
    """
    Using Abstraction LiDar Interface to create a wrapper,
    can be used with SimulatedLidarInstance or RPILidarInstance
    """
    
    def __init__(self, lidar_instance):
        """
        Args:
            lidar_instance: Real Lidar (ex: RPLidarReader)
        """
        self.logger = get_component_logger("RealLidar")
        self.lidar = lidar_instance
        self._running = False
        
        self.logger.info("Real LiDAR interface initialized")
    
    
    def get_lidar_data(self) -> np.ndarray:
        return self.lidar.get_lidar_data()
    
    
    def start(self) -> None:
        if hasattr(self.lidar, "start"):
            self.lidar.start()
        self.logger.info("Real LiDAR started")
    
    
    def stop(self) -> None:
        if hasattr(self.lidar, "stop"):
            self.lidar.stop()
        self.logger.info("Real LiDAR stopped")
    
    
    def is_running(self) -> bool:
        if hasattr(self.lidar, "is_running"):
            return self.lidar.is_running()
        return False
    
    
    def get_info(self) -> Dict[str, Any]:
        info = {
            "mode": "REAL",
            "running": self.is_running(),
            "type": type(self.lidar).__name__
        }
        if hasattr(self.lidar, "get_info"):
            info.update(self.lidar.get_info())
        return info


# Creating Instance to simulate a Lidar
class SimulatedLidarInstance:
    """Simulated LiDAR instance for test, like RPILidar model circular"""
    def __init__(self):
        self.__name__ = "SimulatedLidarInstance"
        self.uncertainty = [0.1, np.deg2rad(5)]  # 10cm distance, 5° angle
        self.angular_resolution = 360
        self.max_range = 5.0
        self.interpolation_steps = 100
        self.logger = get_component_logger("SimulatedLidar")
        self._running = False
        self.logger.info("Simulated LiDAR instance created")
    
    def start(self) -> None:
        self._running = True
        self.logger.info("Simulated LiDAR started")
    
    def stop(self) -> None:
        self._running = False
        self.logger.info("Simulated LiDAR stopped")
    
    
    def add_noise(self, distance, angle):
        """Adds noise to distance and angle measurements."""
        noisy_distance = distance + np.random.normal(0, self.uncertainty[0])
        noisy_angle = angle + np.random.normal(0, self.uncertainty[1])
        return max(noisy_distance, 0), noisy_angle
    
    def get_lidar_data(self):
        """Simulate LiDAR data (360° scan)"""
        angles = np.linspace(0, 2*np.pi, self.angular_resolution, endpoint=False)
        distances = np.random.uniform(0.5, self.max_range, size=self.angular_resolution)
        noisy_data = [self.add_noise(d, a) for d, a in zip(distances, angles)]
        return np.array([d for d, a in noisy_data])
    



class RPLidarReader:
    """
    Leitor otimizado do RPLidar, preparado para uso em multiprocessamento.
    Pode ser usado dentro de RealLidarInterface.
    """
    def __init__(
        self,
        config_loader: Optional[ConfigLoader] = None,
        port: Optional[str] = None,
        baudrate: Optional[int] = None,
        fov_filter: Optional[int] = None,
        heading_offset_deg: Optional[int] = None,
        point_timeout_ms: Optional[int] = None
    ):
        
        self.logger = get_component_logger("RPLidarReader")
        
        if config_loader:
            self.logger.info("Initializing RPLidarReader with provided config_loader")
        else:
            self.logger.info("Initializing RPLidarReader with default configuration")
        
        self.config = config_loader or get_config()
        self.logger.info(f"RPLidarReader configuration: {self.config.lidar}")
        lidar_cfg = self.config.lidar
        self.port                       = port or getattr(lidar_cfg, 'port', "/dev/ttyUSB0")
        self.baudrate                   = baudrate or getattr(lidar_cfg, 'baudrate', 256000)
        self.fov_filter                 = fov_filter or getattr(lidar_cfg, 'fov_filter', 180)
        self.heading_offset_deg         = heading_offset_deg or getattr(lidar_cfg, 'heading_offset_deg', -89)
        self.point_timeout_ms           = point_timeout_ms or getattr(lidar_cfg, 'point_timeout_ms', 1000)
        
        # Shared memory
        self.last_lidar_read = mp.Array('d', 360)
        self.last_lidar_update = mp.Value('d', 0.0)
        self.stop_event = mp.Event()

        self._lidar_process = None
        
    # -------------------------------------------------------
    # AbstractLidarInterface methods
    def start(self):
        """Inicia o processo de leitura do LIDAR"""
        if self._lidar_process is None or not self._lidar_process.is_alive():
            self.stop_event.clear()
            # Based on the previous code, we have a method separated for the lidar process
            
            self._lidar_process = mp.Process(
                target=self._run_lidar_process,
                args=(self.port, self.baudrate),
                daemon=False
            )
            self._lidar_process.start()
            self.logger.info("Lidar process started")

    def stop(self):
        self.stop_event.set()
        if self._lidar_process is not None:
            self._lidar_process.join(timeout=2.0)
            self.logger.info("Lidar process stopped")
            

    def is_running(self) -> bool:
        return self._lidar_process is not None and self._lidar_process.is_alive()

    def get_info(self) -> Dict[str, Any]:
        return {
            "port": self.port,
            "baudrate": self.baudrate,
            "running": self.is_running(),
            "last_update": self.last_lidar_update.value,
            "fov_filter": self.fov_filter,
            "heading_offset_deg": self.heading_offset_deg
        }

    def get_lidar_data(self) -> np.ndarray:
        """Retorna uma cópia do array de 360 graus"""
        arr = np.zeros(360, dtype=float)
        arr[:] = self.last_lidar_read[:]
        return arr

    # -------------------------------------------------------



    def _run_lidar_process(self, port, baudrate):
        try:
            lidar = RPLidar(port, baudrate=baudrate)
            lidar.start_motor()
            lidar.connect()
            lidar.start()
            self.logger.info("Lidar connected and scanning")
        except Exception as e:
            self.logger.info(f"Failed to start Lidar: {e}")
            return

        pre_filtered_distances = np.zeros(360, dtype=float)
        last_update_times = np.zeros(360, dtype=float)

        try:
            for scan in lidar.iter_scans():
                if self.stop_event.is_set():
                    break

                scan_array = np.array(scan)
                angles = scan_array[:, 1]
                distances_m = scan_array[:, 2] / 1000.0  # mm -> meters

                # Map distances into 360 slots
                indices = np.round(angles).astype(int) % 360
                pre_filtered_distances[indices] = distances_m
                last_update_times[indices] = time.time() * 1000

                # # Apply heading offset
                shift = int(self.heading_offset_deg) % 360

                shifted_distances = np.roll(pre_filtered_distances, shift)
                shifted_update_times = np.roll(last_update_times, shift)

                # # Apply FOV filter(#TODO: understand if this is corrects)
                half_fov = self.fov_filter / 2.0
                angles_array = np.arange(360)
                diffs = (angles_array - 0) % 360
                mask = (diffs <= half_fov) | (diffs >= 360 - half_fov)
                shifted_distances[~mask] = 0.0
                

                # # Apply timeout
                current_time = time.time() * 1000
                expired_mask = (current_time - shifted_update_times) > self.point_timeout_ms
                shifted_distances[expired_mask] = 0.0

                # # Copy to shared memory
                self.last_lidar_read[:] = shifted_distances
                self.last_lidar_update.value = time.time()
                

        except RPLidarException as e:
            self.logger.info(f"Lidar exception: {e}")
        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt detected, stopping Lidar process")
        finally:
            try:
                lidar.stop()
                lidar.stop_motor()
                lidar.disconnect()
                self.logger.info("Lidar disconnected")
            except Exception as e:
                self.logger.info(f"Error stopping Lidar: {e}")




def create_lidar_interface(mode: str, config_file: ConfigLoader, lidar_instance=None) -> AbstractLidarInterface:
    """
    Factory para criar interface LiDAR baseada no modo.
    
    Args:
        mode: Modo de operação (real, simulation)
        lidar_instance: Instância do LiDAR real ou simulado (necessário para real/simulation)
        **kwargs: Argumentos específicos do modo
    
    Returns:
        Interface LiDAR apropriada
    """
    mode = mode.lower()
    
    if mode == LidarMode.REAL.value:
        if lidar_instance is None:
            lidar_instance = RPLidarReader(config_file)
        return RealLidarInterface(lidar_instance)
    
    elif mode == LidarMode.SIMULATION.value:
        # For simulation, we can create a simulated instance if not provided
        if lidar_instance is None:
            lidar_instance = SimulatedLidarInstance()
        return RealLidarInterface(lidar_instance)
    else:
        raise ValueError(f"Unknown LiDAR mode: {mode}")
    





