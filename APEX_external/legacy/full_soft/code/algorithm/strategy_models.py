

from code.algorithm.strategy_system import NavigationStrategy
from code.algorithm.strategy_system import NavigationProfile, SpeedProfile, SteeringProfile
from code.algorithm.strategy_system import StrategyManager

class ConservativeStrategy(NavigationStrategy):
    """
    Conservative driving strategy.
    Prioritizes safety over speed.
    """
    
    def __init__(self):
        profile = NavigationProfile(
            name="Conservative",
            speed=SpeedProfile(
                min_speed=0.6,
                max_speed=1.0,
                stop_distance=0.40,
                slow_distance=1.0,
                curve_decay_factor=0.05
            ),
            steering=SteeringProfile(
                max_angle=25.0,
                sensitivity=0.8,
                smoothing_factor=0.7
            ),
            aggressiveness=0.7,
            safety_margin=1.5
        )
        super().__init__("Conservative", profile)
    
    def compute_speed(
        self,
        lidar_data: np.ndarray,
        target_angle: float,
        front_distance: float
    ) -> float:
        # Conservative speed based on angle
        angle_magnitude = abs(target_angle)
        speed = self.profile.speed.max_speed * np.exp(
            -self.profile.speed.curve_decay_factor * angle_magnitude
        )
        speed = max(speed, self.profile.speed.min_speed)
        
        # Apply distance-based slowdown with more conservative margins
        if front_distance <= self.profile.speed.stop_distance:
            return 0.0
        elif front_distance < self.profile.speed.slow_distance:
            distance_factor = (
                (front_distance - self.profile.speed.stop_distance) /
                (self.profile.speed.slow_distance - self.profile.speed.stop_distance)
            )
            speed *= distance_factor * 0.8  # Extra safety factor
        
        return min(speed, self.profile.speed.max_speed)
    
    def compute_steering(
        self,
        target_angle: float,
        current_speed: float
    ) -> float:
        # Apply sensitivity and limit
        adjusted_angle = target_angle * self.profile.steering.sensitivity
        adjusted_angle = np.clip(
            adjusted_angle,
            -self.profile.steering.max_angle,
            self.profile.steering.max_angle
        )
        return adjusted_angle


class NormalStrategy(NavigationStrategy):
    """
    Normal/Balanced driving strategy.
    Balance between speed and safety.
    """
    
    def __init__(self):
        profile = NavigationProfile(
            name="Normal",
            speed=SpeedProfile(
                min_speed=0.8,
                max_speed=1.3,
                stop_distance=0.30,
                slow_distance=0.80,
                curve_decay_factor=0.03
            ),
            steering=SteeringProfile(
                max_angle=30.0,
                sensitivity=1.0,
                smoothing_factor=0.8
            ),
            aggressiveness=1.0,
            safety_margin=1.2
        )
        super().__init__("Normal", profile)
    
    def compute_speed(
        self,
        lidar_data: np.ndarray,
        target_angle: float,
        front_distance: float
    ) -> float:
        # Balanced speed calculation
        angle_magnitude = abs(target_angle)
        speed = self.profile.speed.max_speed * np.exp(
            -self.profile.speed.curve_decay_factor * angle_magnitude
        )
        speed = max(speed, self.profile.speed.min_speed)
        
        # Distance-based adjustment
        if front_distance <= self.profile.speed.stop_distance:
            return 0.0
        elif front_distance < self.profile.speed.slow_distance:
            distance_factor = (
                (front_distance - self.profile.speed.stop_distance) /
                (self.profile.speed.slow_distance - self.profile.speed.stop_distance)
            )
            speed *= distance_factor
        
        return min(speed, self.profile.speed.max_speed)
    
    def compute_steering(
        self,
        target_angle: float,
        current_speed: float
    ) -> float:
        # Standard steering with speed compensation
        adjusted_angle = target_angle
        
        # Reduce steering at higher speeds
        if current_speed > 1.0:
            speed_factor = 1.0 / (1.0 + (current_speed - 1.0) * 0.3)
            adjusted_angle *= speed_factor
        
        adjusted_angle = np.clip(
            adjusted_angle,
            -self.profile.steering.max_angle,
            self.profile.steering.max_angle
        )
        return adjusted_angle


class AggressiveStrategy(NavigationStrategy):
    """
    Aggressive driving strategy.
    Prioritizes speed and quick maneuvers.
    """
    
    def __init__(self):
        profile = NavigationProfile(
            name="Aggressive",
            speed=SpeedProfile(
                min_speed=1.0,
                max_speed=1.8,
                stop_distance=0.25,
                slow_distance=0.60,
                curve_decay_factor=0.02
            ),
            steering=SteeringProfile(
                max_angle=35.0,
                sensitivity=1.2,
                smoothing_factor=0.9
            ),
            aggressiveness=1.5,
            safety_margin=1.0
        )
        super().__init__("Aggressive", profile)
    
    def compute_speed(
        self,
        lidar_data: np.ndarray,
        target_angle: float,
        front_distance: float
    ) -> float:
        # Aggressive speed - less reduction for curves
        angle_magnitude = abs(target_angle)
        speed = self.profile.speed.max_speed * np.exp(
            -self.profile.speed.curve_decay_factor * angle_magnitude
        )
        speed = max(speed, self.profile.speed.min_speed)
        
        # Later braking
        if front_distance <= self.profile.speed.stop_distance:
            return 0.0
        elif front_distance < self.profile.speed.slow_distance:
            distance_factor = (
                (front_distance - self.profile.speed.stop_distance) /
                (self.profile.speed.slow_distance - self.profile.speed.stop_distance)
            )
            # Less aggressive slowdown
            speed *= (0.7 + 0.3 * distance_factor)
        
        return min(speed, self.profile.speed.max_speed)
    
    def compute_steering(
        self,
        target_angle: float,
        current_speed: float
    ) -> float:
        # More responsive steering
        adjusted_angle = target_angle * self.profile.steering.sensitivity
        
        adjusted_angle = np.clip(
            adjusted_angle,
            -self.profile.steering.max_angle,
            self.profile.steering.max_angle
        )
        return adjusted_angle


class AdaptiveStrategy(NavigationStrategy):
    """
    Adaptive strategy that changes behavior based on conditions.
    Switches between conservative, normal, and aggressive based on:
    - Battery level
    - Space availability
    - Recent collision history
    """
    
    def __init__(self):
        # Start with normal profile
        profile = NavigationProfile(
            name="Adaptive",
            speed=SpeedProfile(),
            steering=SteeringProfile(),
            aggressiveness=1.0
        )
        super().__init__("Adaptive", profile)
        
        # Sub-strategies
        self.conservative = ConservativeStrategy()
        self.normal = NormalStrategy()
        self.aggressive = AggressiveStrategy()
        
        self.current_mode = "normal"
        
        # Adaptation parameters
        self.low_battery_threshold = 3.5  # volts
        self.tight_space_threshold = 1.0  # meters average distance
        self.collision_cooldown = 10.0  # seconds after collision
        
        self.last_collision_time = 0.0
    
    def adapt(
        self,
        battery_voltage: float,
        average_clearance: float,
        collision_detected: bool
    ) -> None:
        """
        Adapt strategy based on conditions.
        
        Args:
            battery_voltage: Current battery voltage
            average_clearance: Average distance to obstacles
            collision_detected: Whether a collision was just detected
        """
        import time
        
        if collision_detected:
            self.last_collision_time = time.time()
        
        # Check conditions
        low_battery = battery_voltage < self.low_battery_threshold
        tight_space = average_clearance < self.tight_space_threshold
        recent_collision = (time.time() - self.last_collision_time) < self.collision_cooldown
        
        # Determine mode
        old_mode = self.current_mode
        
        if low_battery or recent_collision or tight_space:
            self.current_mode = "conservative"
        elif average_clearance > 2.0 and not low_battery:
            self.current_mode = "aggressive"
        else:
            self.current_mode = "normal"
        
        if old_mode != self.current_mode:
            self.logger.info(f"Adaptive mode changed: {old_mode} -> {self.current_mode}")
    
    def compute_speed(
        self,
        lidar_data: np.ndarray,
        target_angle: float,
        front_distance: float
    ) -> float:
        # Delegate to current sub-strategy
        if self.current_mode == "conservative":
            return self.conservative.compute_speed(lidar_data, target_angle, front_distance)
        elif self.current_mode == "aggressive":
            return self.aggressive.compute_speed(lidar_data, target_angle, front_distance)
        else:
            return self.normal.compute_speed(lidar_data, target_angle, front_distance)
    
    def compute_steering(
        self,
        target_angle: float,
        current_speed: float
    ) -> float:
        # Delegate to current sub-strategy
        if self.current_mode == "conservative":
            return self.conservative.compute_steering(target_angle, current_speed)
        elif self.current_mode == "aggressive":
            return self.aggressive.compute_steering(target_angle, current_speed)
        else:
            return self.normal.compute_steering(target_angle, current_speed)

