from logging import config
import sys, os, time, csv
from pathlib import Path

from matplotlib.pylab import size
from rplidar import RPLidar, RPLidarException
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from typing import Optional, Dict, Any
import multiprocessing as mp
from datetime import datetime
from rplidar import RPLidar, RPLidarException

from code.log_manager import get_component_logger
from code.Interfaces.interface_decl import AbstractLidarInterface
from code.config_loader import ConfigLoader, get_config



class LidarDataRecorder:
    """
    Record angle values from Lidar

    CSV Format:
        timestamp, angle_0, angle_1, ..., angle_N-1
        1700000000.123, 1.23, 0.98, ...

    File Lo: logs/lidar_data/lidar_<YYYY-MM-DD_HH-MM-SS>.csv
    """

    def __init__(self, output_dir: str = "logs/lidar_data", angular_resolution: int = 360):
        self.logger = get_component_logger("LidarRecorder")
        self.angular_resolution = angular_resolution

        # Criar directório de saída
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Nome do ficheiro com timestamp da sessão
        session_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._filepath = output_path / f"lidar_{session_ts}.csv"

        # Abrir ficheiro e escrever cabeçalho
        self._file = open(self._filepath, "w", newline="")
        self._writer = csv.writer(self._file)
        header = ["timestamp"] + [f"angle_{i}" for i in range(self.angular_resolution)]
        self._writer.writerow(header)
        self._file.flush()

        self._count = 0
        self.logger.info(f"LidarDataRecorder iniciado → {self._filepath}")

    def record(self, data: np.ndarray) -> None:
        """Saves a lecture (array of distances) with actual timestamp."""
        row = [f"{time.time():.4f}"] + [f"{v:.4f}" for v in data]
        self._writer.writerow(row)
        self._count += 1
        # Flush a cada 100 linhas para não perder dados em caso de crash
        if self._count % 100 == 0:
            self._file.flush()

    def close(self) -> None:
        """Fecha o ficheiro CSV."""
        if not self._file.closed:
            self._file.flush()
            self._file.close()
            self.logger.info(f"LidarDataRecorder close — {self._count} recorded at {self._filepath}")

    @property
    def filepath(self) -> Path:
        return self._filepath

    def __del__(self):
        self.close()



class LidarMode(Enum):
    """Modos de operação do LiDAR"""
    REAL = "real"               # LiDAR físico real
    SIMULATION = "simulation"   # Simulated LiDAR (random values as input)


class RealLidarInterface(AbstractLidarInterface):
    """
    Using Abstraction LiDar Interface to create a wrapper,
    can be used with SimulatedLidarInstance or RPILidarInstance
    """
    
    def __init__(self, lidar_instance, record: bool = True, record_dir: str = "logs/lidar_data"):
        """
        Args:
            lidar_instance: Real Lidar (ex: RPLidarReader)
        """
        self.logger = get_component_logger("RealLidar")
        self.lidar = lidar_instance
        self._running = False
        
        self.logger.info("Real LiDAR interface initialized")
    
        # Gravação de dataset
        self._recorder: Optional[LidarDataRecorder] = None
        if record:
            res = getattr(lidar_instance, "angular_resolution", 360)
            self._recorder = LidarDataRecorder(output_dir=record_dir, angular_resolution=res)

        self.logger.info("Real LiDAR interface initialized")

    def get_lidar_data(self) -> np.ndarray:
        data = self.lidar.get_lidar_data()
        if self._recorder is not None:
            self._recorder.record(data)
        return data
    
    
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
class SimulatedRPLidarReader:
    """
    Simulated RPLidarReader for testing filtering and processing.
    Supports walls (line segments) and circular obstacles.
    """

    def __init__(
        self,
        fov_filter: Optional[int] = 360,
        heading_offset_deg: Optional[int] = 0,
        point_timeout_ms: Optional[int] = 200,
        max_range: Optional[float] = 5.0,
        angular_resolution: Optional[int] = 360,
        noise_std: Optional[float] = 0.02
    ):
        self.fov_filter = fov_filter or 360
        self.heading_offset_deg = heading_offset_deg or 0
        self.point_timeout_ms = point_timeout_ms or 200
        self.max_range = max_range or 5.0
        self.angular_resolution = angular_resolution or 360
        self.noise_std = noise_std or 0.02

        self.last_lidar_read = mp.Array('d', self.angular_resolution)
        self.last_lidar_update = mp.Value('d', 0.0)
        self.stop_event = mp.Event()
        self._lidar_process = None
        self._running = False

        self.environment = []  # list of walls and circles

    # ----------------- Environment -----------------
    def add_wall(self, x1, y1, x2, y2):
        """Add a line segment wall"""
        self.environment.append(("wall", (x1, y1, x2, y2)))

    def add_circle(self, cx, cy, r):
        """Add circular obstacle"""
        self.environment.append(("circle", (cx, cy, r)))

    def clear_environment(self):
        self.environment = []

    # ----------------- Simulation -----------------
    def start(self):
        if not self._running:
            self._running = True
            self.stop_event.clear()
            self._lidar_process = mp.Process(target=self._run_process, daemon=True)
            self._lidar_process.start()

    def stop(self):
        self.stop_event.set()
        if self._lidar_process:
            self._lidar_process.join(timeout=1.0)
        self._running = False

    def is_running(self):
        return self._running and self._lidar_process.is_alive()

    def get_lidar_data(self):
        return np.array(self.last_lidar_read[:])

    def get_info(self) -> Dict[str, Any]:
        return {
            "running": self.is_running(),
            "fov_filter": self.fov_filter,
            "heading_offset_deg": self.heading_offset_deg
        }

    # ----------------- Ray casting -----------------
    def _ray_line_intersection(self, theta, x1, y1, x2, y2):
        # Ray from origin (0,0)
        dx = np.cos(theta)
        dy = np.sin(theta)
        denom = (x2 - x1) * dy - (y2 - y1) * dx
        if abs(denom) < 1e-6:  # parallel
            return None
        t = (x1 * dy - y1 * dx) / denom
        u = ((x1 - 0) + (y1 - 0)) / (dx + dy)  # approximate
        if 0 <= t <= 1 and u >= 0:
            return u
        return None

    def _ray_circle_intersection(self, theta, cx, cy, r):
        dx = np.cos(theta)
        dy = np.sin(theta)
        a = dx**2 + dy**2
        b = -2*(cx*dx + cy*dy)
        c = cx**2 + cy**2 - r**2
        disc = b**2 - 4*a*c
        if disc < 0: return None
        t1 = (-b - np.sqrt(disc)) / (2*a)
        t2 = (-b + np.sqrt(disc)) / (2*a)
        valid = [t for t in (t1, t2) if t >= 0]
        return min(valid) if valid else None

    def _run_process(self):
        angles = np.linspace(0, 2*np.pi, self.angular_resolution, endpoint=False)
        last_update_times = np.zeros(self.angular_resolution)
        pre_filtered_distances = np.ones(self.angular_resolution) * self.max_range

        while not self.stop_event.is_set():
            min_distances = np.ones_like(angles) * self.max_range
            for i, theta in enumerate(angles):
                for obj_type, params in self.environment:
                    d = None
                    if obj_type == "wall":
                        d = self._ray_line_intersection(theta, *params)
                    elif obj_type == "circle":
                        d = self._ray_circle_intersection(theta, *params)
                    if d is not None and d < min_distances[i]:
                        min_distances[i] = d
            # Apply heading offset
            shift = int(-self.heading_offset_deg) % self.angular_resolution
            min_distances = np.roll(min_distances, shift)
            last_update_times = np.roll(last_update_times, shift)

            # Apply FOV
            if self.fov_filter < 360:
                half_fov = self.fov_filter / 2
                mask = np.zeros_like(min_distances, dtype=bool)
                center = 0  # front
                indices = np.arange(self.angular_resolution)
                diffs = (indices - center) % self.angular_resolution
                mask = (diffs <= half_fov) | (diffs >= self.angular_resolution - half_fov)
                min_distances[~mask] = 0.0

            # Apply noise
            noisy = min_distances + np.random.normal(0, self.noise_std, size=self.angular_resolution)
            noisy = np.clip(noisy, 0, self.max_range)

            # Apply timeout
            current_time = time.time() * 1000
            expired_mask = (current_time - last_update_times) > self.point_timeout_ms
            noisy[expired_mask] = 0.0

            self.last_lidar_read[:] = noisy
            self.last_lidar_update.value = time.time()
            last_update_times = np.ones_like(last_update_times) * time.time() * 1000

            time.sleep(0.05)  # simulate 20Hz

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
    
    if lidar_instance is None:
        if mode == LidarMode.REAL.value:
            lidar_instance = RPLidarReader(config_file)
        elif mode == LidarMode.SIMULATION.value:
            lidar_instance = SimulatedRPLidarReader(
                fov_filter=config_file.lidar.fov_filter,
                heading_offset_deg=config_file.lidar.heading_offset_deg,
                point_timeout_ms=config_file.lidar.point_timeout_ms,
                max_range=15.0,
                angular_resolution=360,
                noise_std=0.01
            )
    
            # paredes esquerda e direita
            half = 5 / 2
            # parede inferior
            lidar_instance.add_wall(-half, -half, half, -half)
            # parede superior
            lidar_instance.add_wall(-half, half, half, half)
            # parede esquerda
            lidar_instance.add_wall(-half, -half, -half, half)
            # parede direita
            lidar_instance.add_wall(half, -half, half, half)
            #     x1 = 2*np.cos(angle)
            #     y1 = 2*np.sin(angle)
            #     # x2 = 2*np.cos(angle + 0.03)
            #     # y2 = 2*np.sin(angle + 0.03)
            #     lidar_instance.add_wall(x1, y1, x2, y2)

        else:
            raise ValueError(f"Unknown LiDAR mode: {mode}")
    
    return RealLidarInterface(lidar_instance)
