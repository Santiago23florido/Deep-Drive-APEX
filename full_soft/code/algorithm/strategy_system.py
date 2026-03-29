"""
Strategy Pattern System for Vehicle Navigation
Centralizes different driving strategies and their parameters
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
from dataclasses import dataclass

from log_manager import get_component_logger


@dataclass
class SpeedProfile:
    """Speed control profile"""
    min_speed: float = 0.8
    max_speed: float = 1.3
    stop_distance: float = 0.30
    slow_distance: float = 0.80
    curve_decay_factor: float = 0.03
    

@dataclass
class SteeringProfile:
    """Steering control profile"""
    max_angle: float = 30.0
    sensitivity: float = 1.0
    smoothing_factor: float = 0.8


@dataclass
class NavigationProfile:
    """Complete navigation profile"""
    name: str
    speed: SpeedProfile
    steering: SteeringProfile
    aggressiveness: float = 1.0
    safety_margin: float = 1.2


class NavigationStrategy(ABC):
    """
    Abstract base class for navigation strategies.
    Each strategy defines how the vehicle should behave in different scenarios.
    """
    
    def __init__(self, name: str, profile: NavigationProfile):
        self.name = name
        self.profile = profile
        self.logger = get_component_logger(f"Strategy.{name}")
    
    @abstractmethod
    def compute_speed(
        self,
        lidar_data: np.ndarray,
        target_angle: float,
        front_distance: float
    ) -> float:
        """
        Compute target speed based on sensor data and navigation state.
        
        Args:
            lidar_data: Processed LiDAR data
            target_angle: Target steering angle
            front_distance: Distance to nearest obstacle in front
        
        Returns:
            Target speed in m/s
        """
        pass
    
    @abstractmethod
    def compute_steering(
        self,
        target_angle: float,
        current_speed: float
    ) -> float:
        """
        Compute steering angle adjustment.
        
        Args:
            target_angle: Raw target angle from path planning
            current_speed: Current vehicle speed
        
        Returns:
            Adjusted steering angle
        """
        pass

class StrategyManager:
    """
    Manages navigation strategies and allows dynamic switching.
    """
    
    def __init__(self, initial_strategy: str = "normal"):
        self.logger = get_component_logger("StrategyManager")
        
        # Create all available strategies
        self.strategies = {
            "conservative": ConservativeStrategy(),
            "normal": NormalStrategy(),
            "aggressive": AggressiveStrategy(),
        }
        
        # Set initial strategy
        if initial_strategy.lower() in self.strategies:
            self.current_strategy = self.strategies[initial_strategy.lower()]
        else:
            self.logger.warning(f"Unknown strategy '{initial_strategy}', using 'normal'")
            self.current_strategy = self.strategies["normal"]
        
        self.logger.info(f"Initialized with {self.current_strategy.name} strategy")
    
    def set_strategy(self, strategy_name: str) -> bool:
        """
        Change the current navigation strategy.
        
        Args:
            strategy_name: Name of the strategy to activate
        
        Returns:
            True if strategy was changed, False otherwise
        """
        strategy_name = strategy_name.lower()
        
        if strategy_name in self.strategies:
            old_strategy = self.current_strategy.name
            self.current_strategy = self.strategies[strategy_name]
            self.logger.info(f"Strategy changed: {old_strategy} -> {self.current_strategy.name}")
            return True
        else:
            self.logger.warning(f"Unknown strategy: {strategy_name}")
            return False
    
    def get_strategy(self) -> NavigationStrategy:
        """Get current strategy"""
        return self.current_strategy
    
    def get_strategy_name(self) -> str:
        """Get current strategy name"""
        return self.current_strategy.name
    
    def compute_speed(
        self,
        lidar_data: np.ndarray,
        target_angle: float,
        front_distance: float
    ) -> float:
        """Compute speed using current strategy"""
        return self.current_strategy.compute_speed(
            lidar_data,
            target_angle,
            front_distance
        )
    
    def compute_steering(
        self,
        target_angle: float,
        current_speed: float
    ) -> float:
        """Compute steering using current strategy"""
        return self.current_strategy.compute_steering(
            target_angle,
            current_speed
        )
    
    def get_profile(self) -> NavigationProfile:
        """Get current navigation profile"""
        return self.current_strategy.profile
    
    def list_strategies(self) -> list:
        """List all available strategies"""
        return list(self.strategies.keys())

