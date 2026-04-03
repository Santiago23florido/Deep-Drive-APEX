import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any


# COMMUNICATION INTERFACES FOR SENSORS AND ACTUATORS


class AbstractLidarInterface(ABC):
    @abstractmethod
    def get_lidar_data(self) -> np.ndarray:
        """Returns a NumPy array of shape (360,) where index encodes angle and value encodes distance."""
        pass

    @abstractmethod
    def start(self) -> None:
        """Inicia a interface LiDAR"""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the LiDAR interface (if applicable)"""
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Checker if running"""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get info about the LiDAR (mode, status, etc)"""
        pass


class UltrasonicInterface(ABC):
    @abstractmethod
    def get_ultrasonic_data(self) -> float:
        """Returns the distance (in cm) from ultrasonic back sensor."""
        pass


class SpeedInterface(ABC):
    @abstractmethod
    def get_speed(self) -> float:
        """Returns the current speed of the vehicle (in m/s)."""
        pass


class BatteryInterface(ABC):
    @abstractmethod
    def get_battery_voltage(self) -> float:
        """Returns the current battery voltage."""
        pass


class CameraInterface(ABC):
    @abstractmethod
    def get_camera_frame(self) -> np.ndarray:
        """Returns the current camera frame as a numpy array."""
        pass

    @abstractmethod
    def get_resolution(self) -> tuple:
        """Returns the camera resolution as (width, height)."""
        pass


class SteerInterface(ABC):
    @abstractmethod
    def set_steering_angle(self, angle: float) -> None:
        """Sets the steering angle in degrees."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stops the steering servo (if applicable)"""
        pass

    # BUG FIX: 'start' estava sem 'self' no parâmetro, tornando-o inválido como método de instância
    @abstractmethod
    def start(self) -> None:
        """Starts the steering servo (if applicable)"""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get info about the steering (mode, status, etc)"""
        pass


class AbstractMotorInterface(ABC):

    @abstractmethod
    def set_speed(self, speed: float) -> None:
        """Sets the target speed of the vehicle"""
        pass

    @abstractmethod
    def get_speed(self) -> float:
        """Returns the current target speed of the vehicle"""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stops the motor (if applicable)"""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get info about the motor (mode, status, etc)"""
        pass