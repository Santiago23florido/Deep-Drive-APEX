"""
Centralized Configuration Loader System
Handles all configuration parameters with validation and type safety
"""

import json
import os
from typing import Any, Dict, Optional, TypeVar, Generic
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging



logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConfigCategory(Enum):
    """Configuration categories for organization"""
    GENERAL = "GENERAL"
    LIDAR = "LIDAR"
    CAMERA = "CAMERA"
    STEERING = "STEERING"
    MOTOR = "MOTOR"
    NAVIGATION = "NAVIGATION"
    SAFETY = "SAFETY"
    VISUALIZATION = "VISUALIZATION"


@dataclass
class GeneralConfig:
    """General car configuration"""
    car_name: str = "Voiture-Couleur"
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_visualization: bool = False
    

@dataclass
class LidarConfig:
    """LiDAR sensor configuration"""
    baudrate: int = 256000
    heading_offset_deg: int = -89
    point_timeout_ms: int = 1000
    fov_filter: int = 180
    field_of_view_deg: int = 180
    convolution_size: int = 71
    lecture_max_distance: float = 5.0
    port: str = "/dev/ttyUSB0"
    

@dataclass
class CameraConfig:
    """Camera configuration"""
    TRACK_DIRECTION: int = 0
    width: int = 160
    height: int = 120
    rotation: int = 180  # degrees
    offcenter: float = 0.2
    min_detection_ratio: float = 12.0
    

@dataclass
class SteeringConfig:
    """Steering control configuration"""
    limit: float = 18.0
    dc_min: float = 5.0
    dc_max: float = 8.6
    channel: int = 1
    frequency: float = 50.0
     
    # getters: 
    @property
    def variation_rate(self) -> float:
        return 0.5 * (self.dc_max - self.dc_min) / self.limit
    @property
    def center(self) -> float:
        return 0.5 * (self.dc_max + self.dc_min)


@dataclass
class MotorConfig:
    """Motor/ESC configuration"""
    dc_min: float = 5.0
    dc_max: float = 10.0
    channel: int = 0
    frequency: float = 50.0
    neutral_dc: float = 7.5
    ticks_to_meter: int = 213
    
    @property
    def speed_to_dc_a(self) -> float:
        return self.dc_max - self.dc_min
    
    @property
    def speed_to_dc_b(self) -> float:
        return self.dc_min


@dataclass
class NavigationConfig:
    """Navigation and pathfinding configuration"""
    # Corner avoidance
    avoid_corner_max_angle: int = 30
    avoid_corner_min_distance: float = 2.5
    avoid_corner_scale_factor: float = 1.2
    
    # Hitbox
    hitbox_h1: float = 0.11
    hitbox_h2: float = 0.31
    hitbox_w: float = 0.11
    
    # Speed control
    aggressiveness: float = 1.3
    aperture_angle: int = 20
    
    # Reverse behavior
    min_length: float = 0.28
    max_length: float = 0.38
    min_points_to_trigger: int = 8
    reverse_check_counter: int = 8
    pwm_reverse: float = 7.0
    

@dataclass
class SafetyConfig:
    """Safety and collision detection configuration"""
    minimum_front_distance: float = 0.15
    back_distance: float = 15.0
    wheel_stopped_threshold: float = 0.02  # m/s
    collision_time_threshold: float = 0.5  # seconds
    stop_distance: float = 0.30
    slow_distance: float = 0.80



@dataclass
class VisualizationConfig:
    """Visualization and plotting configuration"""
    enable_live_plot: bool = False
    enable_algorithm_view: bool = True
    plot_update_rate: float = 0.05  # seconds
    @property
    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

class ConfigLoader:
    """
    Centralized configuration loader and manager.
    Provides type-safe access to all configuration parameters.
    """
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self._raw_config: Dict[str, Any] = {}
        
        # Configuration objects
        self.general = GeneralConfig()
        self.lidar = LidarConfig()
        self.camera = CameraConfig()
        self.steering = SteeringConfig()
        self.motor = MotorConfig()
        self.navigation = NavigationConfig()
        self.safety = SafetyConfig()
        self.visualization = VisualizationConfig()
        
        # Load configuration
        self.load()
    
    def load(self) -> None:
        """Load configuration from file"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._raw_config = json.load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
            except (OSError, json.JSONDecodeError) as e:
                logger.exception("Error loading configuration from %s", self.config_path)
                self._raw_config = {}
        else:
            logger.warning(f"Configuration file not found: {self.config_path}")
            self._raw_config = {}
        
        # Populate configuration objects
        self._populate_configs()
    
    def _populate_configs(self) -> None:
        """Automatically populate all config dataclasses"""

        def update_dataclass(instance, data: Dict[str, Any]):
            for field_name, field_value in data.items():
                if hasattr(instance, field_name):
                    current_value = getattr(instance, field_name)
                    try:
                        # Cast to correct type
                        casted_value = type(current_value)(field_value)
                        setattr(instance, field_name, casted_value)
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid type for {field_name}: {field_value}, keeping default"
                        )

        category_map = {
            "GENERAL": self.general,
            "LIDAR": self.lidar,
            "CAMERA": self.camera,
            "STEERING": self.steering,
            "MOTOR": self.motor,
            "NAVIGATION": self.navigation,
            "SAFETY": self.safety,
            "VISUALIZATION": self.visualization,
        }

        for category, instance in category_map.items():
            raw_section = self._raw_config.get(category, {})
            update_dataclass(instance, raw_section)
        
    
    def _get_value(self, key: str, default: T) -> T:
        """Get value from config with default fallback"""
        value = self._raw_config.get(key, default)
        
        # Type conversion if needed
        if value != default:
            try:
                return type(default)(value)
            except (ValueError, TypeError):
                logger.warning(f"Type conversion failed for {key}, using default")
                return default
        
        return value
    
    def save(self) -> None:
        """Save current configuration to file"""
        config_dict = {
            "GENERAL": asdict(self.general),
            "LIDAR": asdict(self.lidar),
            "CAMERA": asdict(self.camera),
            "STEERING": asdict(self.steering),
            "MOTOR": asdict(self.motor),
            "NAVIGATION": asdict(self.navigation),
            "SAFETY": asdict(self.safety),
            "VISUALIZATION": asdict(self.visualization),
        }
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=4)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")


    def reload(self) -> None:
        """Reload configuration from file"""
        self.load()
    
    def get_category(self, category: ConfigCategory):
        """Get configuration object by category"""
        category_map = {
            ConfigCategory.GENERAL: self.general,
            ConfigCategory.LIDAR: self.lidar,
            ConfigCategory.CAMERA: self.camera,
            ConfigCategory.STEERING: self.steering,
            ConfigCategory.MOTOR: self.motor,
            ConfigCategory.NAVIGATION: self.navigation,
            ConfigCategory.SAFETY: self.safety,
            ConfigCategory.VISUALIZATION: self.visualization,
        }
        return category_map.get(category)


# Global configuration instance
_config_instance: Optional[ConfigLoader] = None


def get_config(config_path: str = "new-config.json") -> ConfigLoader:
    """Get or create global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader(config_path)
    return _config_instance


def reload_config() -> None:
    """Reload global configuration"""
    global _config_instance
    if _config_instance is not None:
        _config_instance.reload()

if __name__ == "__main__":
    # Example: load config, change a parameter, save it, and reload
    import random
    config_path = "test-config.json"
    config = ConfigLoader(config_path)

    print("=== Before change ===")
    print(f"Steering limit: {config.steering.limit}")
    print(f"Camera rotation: {config.camera.rotation}")

    # Change a parameter (example)
    
    config.steering.limit = random.uniform(10.0, 30.0)  # Random new limit for testing
    

    angles = [0, 90, 180, 270]
    config.camera.rotation = random.choice(angles)

    print("result of random changes:")
    print(f"Steering limit: {config.steering.limit}")
    print(f"Camera rotation: {config.camera.rotation}")
    
    # Persist changes to disk
    config.save()

    # Reload to prove it was saved
    config.reload()
    config.save()
    
    print("\n=== After change + reload ===")
    print(f"Steering limit: {config.steering.limit}")
    print(f"Camera rotation: {config.camera.rotation}")
