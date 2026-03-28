import time
import numpy as np


# New systems
from code.algorithm import strategy_models
from code.config_loader import ConfigLoader, get_config
from code.log_manager import get_component_logger
from state_machine import StateMachine, VehicleState, StateContext
from code.algorithm.strategy_system import StrategyManager
from visualization_manager import VisualizationManager
from algorithm.control_camera import extract_info, DetectionStatus

from code.Interfaces.interface_decl import (
    AbstractLidarInterface, UltrasonicInterface, SpeedInterface,
    BatteryInterface, SteerInterface
)

from code.algorithm.strategy_models import NormalStrategy, CautiousStrategy, AggressiveStrategy
from code.Interfaces.interface_camera import RealCameraInterface
from code.Interfaces.MotorInterface import RealMotorInterface
from code.Interfaces.LidarInterface import RealLidarInterface





class VehicleAlgorithm_v1:
    def __init__(
        self,
        config    : ConfigLoader,
        lidar     : AbstractLidarInterface,
        ultrasonic: UltrasonicInterface,
        speed     : SpeedInterface,
        battery   : BatteryInterface,
        camera    : RealCameraInterface,
        steer     : SteerInterface,
        motor     : RealMotorInterface,
    ):
        # Store configuration
        self.config = config
        
        # Get logger
        self.logger = get_component_logger("VehicleAlgorithm_v1", create_file=True)
        self.logger.info("Initializing Vehicle Algorithm v1...")
        
        # Store interfaces
        self.lidar      = lidar
        self.ultrasonic = ultrasonic
        self.speed      = speed
        self.battery    = battery
        self.camera     = camera
        self.steer      = steer
        self.motor      = motor
        
        
        # Initialize state machine
        self.state_machine = StateMachine()
        self.logger.info(f"State machine initialized in {self.state_machine.get_current_state().name}")
        
        
        # Initialize strategy manager
        initial_strategy = "normal"  # Could come from config
        self.strategy_manager = StrategyManager(initial_strategy)
        self.logger.info(f"Strategy manager initialized with {initial_strategy} strategy")
        
        # Store Configurations for camera processing
        self.camera_width, self.camera_height = self.camera.get_resolution()
        
        
        
    
        # Initialize visualization manager (based on config)
        self.viz_manager = VisualizationManager(
            enable_lidar=config.visualization.enable_live_plot,
            enable_camera=False,  # Could be configurable
            enable_state=True if config.visualization.enable_live_plot else False,
            enable_algorithm_view=config.visualization.enable_algorithm_view
        )
        
        if self.viz_manager.is_enabled():
            self.viz_manager.show_all()
            self.logger.info("Visualization enabled")
        else:
            self.logger.info("Visualization disabled")
    
        
        # Performance tracking
        self.loop_times = []
    
        self.logger.info("Voiture Algorithm initialization complete")
    
    def update_sensor_context(self, context: StateContext) -> None:
        """Update context with current sensor readings"""
        # Get sensor data
        context.lidar_data = self.lidar.get_lidar_data()
        context.ultrasonic_distance = self.ultrasonic.get_ultrasonic_data()
        context.current_speed = self.speed.get_speed()
        context.battery_voltage = self.battery.get_battery_voltage()
        context.camera_frame = self.camera.get_camera_frame()
    
    def process_lidar_data(self, context: StateContext) -> None:
        """Process LiDAR data for navigation"""
        from algorithm.control_direction import (
            shrink_space, compute_steer_from_lidar, convolution_filter
        )
        
        raw_lidar = context.lidar_data
        
        # Shrink space (account for vehicle hitbox)
        context.shrinked_lidar = shrink_space(raw_lidar)
        
        # Apply convolution filter
        filtered_dist, filtered_angles = convolution_filter(context.shrinked_lidar)
        context.filtered_lidar = filtered_dist
        
        # Compute steering angle
        steer, target_angle = compute_steer_from_lidar(context.shrinked_lidar)
        context.target_angle = target_angle
    
    def process_camera_data(self, context: StateContext) -> None:
        """Process camera data for color detection"""
        # from algorithm.control_camera import extract_info, DetectionStatus
        
        if context.camera_frame is None:
            self.logger.warning("No camera frame available for processing")
            return
        
        process_results = extract_info(
            context.camera_frame, 
            self.camera_width, 
            self.camera_height
            )
        
        status = process_results['status']
        
        #TODO: Map detection status to context
        if status == DetectionStatus.RED_LEFT_GREEN_RIGHT:
            context.color_detection = "red_left"
        elif status == DetectionStatus.GREEN_LEFT_RED_RIGHT:
            context.color_detection = "green_left"
        elif status == DetectionStatus.ONLY_RED:
            context.color_detection = "only_red"
        elif status == DetectionStatus.ONLY_GREEN:
            context.color_detection = "only_green"
        else:
            context.color_detection = None
    
    def detect_obstacles(self, context: StateContext) -> None:
        """Detect obstacles and update context flags"""
        if context.lidar_data is None:
            self.logger.warning("No LiDAR data available for obstacle detection")
            return
        
        # Check front distance
        
        front_indices = list(range(350, 360)) + list(range(0, 11))
        front_data = [context.lidar_data[i] for i in front_indices if context.lidar_data[i] > 0]
        
        if len(front_data) > 0:
            front_distance = sum(front_data) / len(front_data)
            
            # Update flags
            context.obstacle_in_front = front_distance < self.config.safety.slow_distance
            context.wall_too_close = front_distance < self.config.safety.minimum_front_distance
        else:
            context.obstacle_in_front = False
            context.wall_too_close = False
    
    def compute_navigation(self, context: StateContext) -> None:
        """Compute target speed and steering using current strategy"""
        if context.lidar_data is None or context.target_angle is None:
            self.logger.warning("Insufficient data to compute navigation")
            return
        
        
        # Get front distance
        front_indices = list(range(350, 360)) + list(range(0, 11))
        front_data = [context.lidar_data[i] for i in front_indices if context.lidar_data[i] > 0]
        front_distance = sum(front_data) / len(front_data) if front_data else float('inf')
        
        
        # Use strategy manager to compute speed
        context.target_speed = self.strategy_manager.compute_speed(
            context.lidar_data,
            context.target_angle,
            front_distance
        )
        
        # Adjust steering using strategy
        adjusted_steering = self.strategy_manager.compute_steering(
            context.target_angle,
            context.current_speed
        )
        
        context.target_angle = adjusted_steering
    
    def apply_control(self, context: StateContext) -> None:
        """Apply computed control to actuators"""
        
        # Only apply control if not in emergency stop
        if self.state_machine.get_current_state() != VehicleState.EMERGENCY_STOP:
            if context.target_angle is not None:
                self.steer.set_steering_angle(context.target_angle)
            
            if context.target_speed is not None:
                self.motor.set_speed(context.target_speed)
    
    def update_visualizations(self, context: StateContext) -> None:
        """Update all visualizations if enabled"""
        if not self.viz_manager.is_enabled():
            self.logger.debug("Visualization is disabled, skipping update")
            return
        
        # Update LiDAR visualization
        if context.lidar_data is not None:
            lidar_viz_data = {
                'raw_lidar': context.lidar_data,
                'target_angle': context.target_angle or 0.0,
                'target_speed': context.target_speed or 0.0,
            }
            
            if context.filtered_lidar is not None:
                lidar_viz_data['filtered_lidar'] = context.filtered_lidar
                lidar_viz_data['filtered_angles'] = np.linspace(
                    0, np.pi, len(context.filtered_lidar)
                )
            
            # Add hitbox if available
            from algorithm.control_direction import hitbox
            lidar_viz_data['hitbox'] = hitbox
            
            self.viz_manager.update_lidar(lidar_viz_data)
        
        # Update state visualization
        state_viz_data = {
            'timestamp': time.time(),
            'state': self.state_machine.get_current_state().name,
            'speed': context.target_speed or 0.0,
            'steering': context.target_angle or 0.0,
        }
        self.viz_manager.update_state(state_viz_data)
    
    def log_status(self, context: StateContext, loop_time: float) -> None:
        """Log current status"""
        state_name = self.state_machine.get_current_state().name
        
        # Log to file
        self.logger.debug(
            f"State: {state_name}, "
            f"Angle: {context.target_angle:.1f}°, "
            f"Speed: {context.target_speed:.2f} m/s, "
            f"Battery: {context.battery_voltage:.2f}V, "
            f"Loop: {loop_time*1e6:.0f} µs"
        )

    def run_step(self) -> None:
        """Execute one iteration of the main control loop"""
        start_time = time.time()
        
        # Get state machine context
        context = self.state_machine.get_context()
        
        # 1. Update sensor data
        self.update_sensor_context(context)
        
        # 2. Process sensor data
        self.process_lidar_data(context)
        self.process_camera_data(context)
        
        # 3. Detect obstacles and update flags
        self.detect_obstacles(context)
        
        # 4. Compute navigation (speed and steering)
        self.compute_navigation(context)
        
        # 5. Update state machine (this will handle state transitions based on context)
        self.state_machine.update({})
        
        # 6. Apply control commands
        self.apply_control(context)
        
        # 7. Update visualizations
        self.update_visualizations(context)
        
        # 8. Performance tracking
        loop_time = time.time() - start_time
        self.loop_times.append(loop_time)
        
        # 9. Log status
        self.log_status(context, loop_time)
    
    def start(self) -> None:
        """Start the vehicle (transition from idle to running)"""
        self.logger.info("Starting vehicle...")
        self.state_machine.start_vehicle()
    
    def emergency_stop(self) -> None:
        """Trigger emergency stop"""
        self.logger.critical("EMERGENCY STOP TRIGGERED")
        self.state_machine.emergency_stop()
        self.motor.set_speed(0.0)
        self.steer.set_steering_angle(0.0)
    
    def change_strategy(self, strategy_name: str) -> bool:
        """Change navigation strategy"""
        success = self.strategy_manager.set_strategy(strategy_name)
        if success:
            self.logger.info(f"Strategy changed to: {strategy_name}")
        return success
    
    def get_statistics(self) -> dict:
        """Get algorithm statistics"""
        if not self.loop_times:
            return {}
        
        return {
            'avg_loop_time': np.mean(self.loop_times),
            'max_loop_time': np.max(self.loop_times),
            'min_loop_time': np.min(self.loop_times),
            'current_state': self.state_machine.get_current_state().name,
            'current_strategy': self.strategy_manager.get_strategy_name(),
        }
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.logger.info("Cleaning up...")
        
        self.motor.stop()
        self.steer.stop()
        self.viz_manager.close_all()
        
        # Print statistics
        stats = self.get_statistics()
        if stats:
            self.logger.info(f"Statistics: {stats}")
        
        self.logger.info("Cleanup complete")
