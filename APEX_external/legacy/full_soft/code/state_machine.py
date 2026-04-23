"""
State Machine System for Vehicle Control
Replaces scattered if/else logic with a clean state-based architecture
"""

from enum import Enum, auto
from typing import Optional, Dict, Callable, Any
from abc import ABC, abstractmethod
import time
import numpy as np

from log_manager import get_component_logger
from log_manager import initialize_logging
import threading


class VehicleState(Enum):
    """Available vehicle states"""
    INITIALIZING = auto()
    IDLE = auto()
    NORMAL_DRIVING = auto()
    OBSTACLE_DETECTED = auto()
    REVERSING = auto()
    TURNING_LEFT = auto()
    TURNING_RIGHT = auto()
    U_TURN = auto()
    COLLISION_RECOVERY = auto()
    EMERGENCY_STOP = auto()
    ERROR = auto()


class StateContext:
    """
    Context data passed between states.
    Contains all sensor data and vehicle information.
    """
    
    def __init__(self):
        # Sensor data
        self.lidar_data: Optional[np.ndarray] = None
        self.camera_frame: Optional[np.ndarray] = None
        self.ultrasonic_distance: float = 0.0
        self.current_speed: float = 0.0
        self.battery_voltage: float = 0.0
        
        # Processed data
        self.target_angle: float = 0.0
        self.target_speed: float = 0.0
        self.shrinked_lidar: Optional[np.ndarray] = None
        self.filtered_lidar: Optional[np.ndarray] = None
        
        # Detection flags
        self.obstacle_in_front: bool = False
        self.wall_too_close: bool = False
        self.collision_detected: bool = False
        self.color_detection: Optional[str] = None  # "red_left", "green_left", etc.
        
        
        
        # Timing
        self.state_entry_time: float = 0.0
        self.loop_time: float = 0.0
        
        # Counters
        self.reverse_counter: int = 0
        self.stuck_counter: int = 0
        
    def reset_counters(self):
        """Reset state-specific counters"""
        self.reverse_counter = 0
        self.stuck_counter = 0


class State(ABC):
    """
    Abstract base class for vehicle states.
    Each state implements its own behavior logic.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_component_logger(f"State.{name}")
    
    @abstractmethod
    def enter(self, context: StateContext) -> None:
        """Called when entering this state"""
        pass
    
    @abstractmethod
    def execute(self, context: StateContext) -> Optional[VehicleState]:
        """
        Execute state logic.
        Returns next state if transition is needed, None otherwise.
        """
        pass
    
    @abstractmethod
    def exit(self, context: StateContext) -> None:
        """Called when exiting this state"""
        pass


class InitializingState(State):
    """Initial state - system startup and calibration"""
    
    def __init__(self):
        super().__init__("Initializing")
        self.initialization_complete = False
    
    def enter(self, context: StateContext) -> None:
        self.logger.info("System initializing...")
        context.state_entry_time = time.time()
        self.initialization_complete = False
    
    def execute(self, context: StateContext) -> Optional[VehicleState]:
        # Check if LiDAR has enough data
        if context.lidar_data is not None:
            nonzero_count = np.count_nonzero(context.lidar_data)
            # if nonzero_count > 90:  # Half of field of view
            self.initialization_complete = True
        
        # Check for minimum initialization time
        if time.time() - context.state_entry_time > 0.5 and self.initialization_complete:
            self.logger.info("Initialization complete")
            return VehicleState.IDLE
        
        return None
    
    def exit(self, context: StateContext) -> None:
        self.logger.info("Exiting initialization")


class IdleState(State):
    """Idle state - waiting for start command"""
    
    def __init__(self):
        super().__init__("Idle")
        self.ready_to_start = False
    
    def enter(self, context: StateContext) -> None:
        self.logger.info("Vehicle in idle state")
        context.target_speed = 0.0
        context.target_angle = 0.0
    
    def execute(self, context: StateContext) -> Optional[VehicleState]:
        # In production, this would check for a start signal
        # For now, automatically transition to normal driving
        
        if self.ready_to_start:
            return VehicleState.NORMAL_DRIVING
        return None
    
    def exit(self, context: StateContext) -> None:
        self.logger.info("Starting normal operation")
        context.reset_counters()
    
    # And wait for "enter" key press to start the vehicle - assynchronously
    def wait_for_input(self):
        input("Press Enter to start the vehicle...\n")
        self.ready_to_start = True
    
    def start(self):
        # This method will call different checkers to start the vehicle
        

        thread = threading.Thread(target=self.wait_for_input, daemon=True)
        thread.start()
        
        while not self.ready_to_start:
            print("Waiting for signal...")
            time.sleep(2) 
        thread.join()
        
        


class NormalDrivingState(State):
    """Normal driving - following path and avoiding obstacles"""
    
    def __init__(self):
        super().__init__("NormalDriving")
    
    def enter(self, context: StateContext) -> None:
        self.logger.debug("Entering normal driving mode")
        context.state_entry_time = time.time()
    
    def execute(self, context: StateContext) -> Optional[VehicleState]:
        # Check for emergency conditions first
        if context.collision_detected:
            return VehicleState.COLLISION_RECOVERY
        
        # Check for wall too close
        if context.wall_too_close:
            return VehicleState.REVERSING
        
        # Check for obstacle requiring maneuver
        if context.obstacle_in_front:
            return VehicleState.OBSTACLE_DETECTED
        
        # Check for U-turn requirement (color detection)
        if context.color_detection in ["red_left", "green_left"]:
            return VehicleState.U_TURN
        
        # Continue normal driving
        # Target angle and speed are already computed in context
        return None
    
    def exit(self, context: StateContext) -> None:
        self.logger.debug("Exiting normal driving mode")


class ObstacleDetectedState(State):
    """Obstacle detected - decide on avoidance strategy"""
    
    def __init__(self):
        super().__init__("ObstacleDetected")
    
    def enter(self, context: StateContext) -> None:
        self.logger.info("Obstacle detected, planning avoidance")
        context.state_entry_time = time.time()
    
    def execute(self, context: StateContext) -> Optional[VehicleState]:
        # If obstacle cleared, return to normal driving
        if not context.obstacle_in_front:
            time_in_state = time.time() - context.state_entry_time
            if time_in_state > 0.5:  # Debounce
                return VehicleState.NORMAL_DRIVING
        
        # If stuck for too long, try reversing
        if time.time() - context.state_entry_time > 3.0:
            context.stuck_counter += 1
            return VehicleState.REVERSING
        
        # Reduce speed when obstacle detected
        context.target_speed = min(context.target_speed, 0.8)
        
        return None
    
    def exit(self, context: StateContext) -> None:
        self.logger.debug("Obstacle handling complete")


class ReversingState(State):
    """Reversing to avoid obstacle or escape stuck situation"""
    
    def __init__(self, max_duration: float = 2.0):
        super().__init__("Reversing")
        self.max_duration = max_duration
    
    def enter(self, context: StateContext) -> None:
        self.logger.info("Entering reverse maneuver")
        context.state_entry_time = time.time()
        context.reverse_counter += 1
    
    def execute(self, context: StateContext) -> Optional[VehicleState]:
        elapsed = time.time() - context.state_entry_time
        
        # Check ultrasonic for back obstacle
        if context.ultrasonic_distance < 15.0 and context.ultrasonic_distance > 0:
            self.logger.warning("Back obstacle detected during reverse")
            return VehicleState.NORMAL_DRIVING
        
        # Check if reverse duration exceeded
        if elapsed > self.max_duration:
            self.logger.info("Reverse maneuver complete")
            # Decide next state based on color detection
            if context.color_detection == "red_left":
                return VehicleState.TURNING_RIGHT
            elif context.color_detection == "green_left":
                return VehicleState.TURNING_LEFT
            else:
                return VehicleState.NORMAL_DRIVING
        
        # Set reverse speed (negative)
        context.target_speed = -1.2
        
        # Steering based on color detection
        if context.color_detection == "red_left":
            context.target_angle = -30
        elif context.color_detection == "green_left":
            context.target_angle = 30
        else:
            context.target_angle = 0
        
        return None
    
    def exit(self, context: StateContext) -> None:
        self.logger.debug("Exiting reverse state")
        context.target_speed = 0.0


class TurningLeftState(State):
    """Execute left turn maneuver"""
    
    def __init__(self, duration: float = 1.0):
        super().__init__("TurningLeft")
        self.duration = duration
    
    def enter(self, context: StateContext) -> None:
        self.logger.info("Executing left turn")
        context.state_entry_time = time.time()
    
    def execute(self, context: StateContext) -> Optional[VehicleState]:
        if time.time() - context.state_entry_time > self.duration:
            return VehicleState.NORMAL_DRIVING
        
        context.target_angle = 30
        context.target_speed = 0.7
        return None
    
    def exit(self, context: StateContext) -> None:
        self.logger.debug("Left turn complete")


class TurningRightState(State):
    """Execute right turn maneuver"""
    
    def __init__(self, duration: float = 1.0):
        super().__init__("TurningRight")
        self.duration = duration
    
    def enter(self, context: StateContext) -> None:
        self.logger.info("Executing right turn")
        context.state_entry_time = time.time()
    
    def execute(self, context: StateContext) -> Optional[VehicleState]:
        if time.time() - context.state_entry_time > self.duration:
            return VehicleState.NORMAL_DRIVING
        
        context.target_angle = -30
        context.target_speed = 0.7
        return None
    
    def exit(self, context: StateContext) -> None:
        self.logger.debug("Right turn complete")


class UTurnState(State):
    """Execute U-turn maneuver"""
    
    def __init__(self):
        super().__init__("UTurn")
    
    def enter(self, context: StateContext) -> None:
        self.logger.info("Executing U-turn maneuver")
        context.state_entry_time = time.time()
    
    def execute(self, context: StateContext) -> Optional[VehicleState]:
        elapsed = time.time() - context.state_entry_time
        
        # First reverse
        if elapsed < 1.5:
            return VehicleState.REVERSING
        
        # Then turn
        if context.color_detection == "red_left":
            return VehicleState.TURNING_RIGHT
        else:
            return VehicleState.TURNING_LEFT
    
    def exit(self, context: StateContext) -> None:
        self.logger.debug("U-turn complete")


class CollisionRecoveryState(State):
    """Recover from collision"""
    
    def __init__(self):
        super().__init__("CollisionRecovery")
    
    def enter(self, context: StateContext) -> None:
        self.logger.warning("COLLISION DETECTED - Initiating recovery")
        context.state_entry_time = time.time()
        context.target_speed = 0.0
    
    def execute(self, context: StateContext) -> Optional[VehicleState]:
        # Stop briefly
        if time.time() - context.state_entry_time < 0.5:
            context.target_speed = 0.0
            return None
        
        # Then reverse
        return VehicleState.REVERSING
    
    def exit(self, context: StateContext) -> None:
        self.logger.info("Collision recovery complete")


class EmergencyStopState(State):
    """Emergency stop state"""
    
    def __init__(self):
        super().__init__("EmergencyStop")
    
    def enter(self, context: StateContext) -> None:
        self.logger.critical("EMERGENCY STOP ACTIVATED")
        context.target_speed = 0.0
        context.target_angle = 0.0
    
    def execute(self, context: StateContext) -> Optional[VehicleState]:
        # Stay in emergency stop until manual intervention
        context.target_speed = 0.0
        context.target_angle = 0.0
        return None
    
    def exit(self, context: StateContext) -> None:
        self.logger.info("Emergency stop deactivated")


class StateMachine:
    """
    State machine manager.
    Handles state transitions and execution.
    """
    
    def __init__(self):
        self.logger = get_component_logger("StateMachine")
        
        # Create all states
        self.states: Dict[VehicleState, State] = {
            VehicleState.INITIALIZING: InitializingState(),
            VehicleState.IDLE: IdleState(),
            VehicleState.NORMAL_DRIVING: NormalDrivingState(),
            VehicleState.OBSTACLE_DETECTED: ObstacleDetectedState(),
            VehicleState.REVERSING: ReversingState(),
            VehicleState.TURNING_LEFT: TurningLeftState(),
            VehicleState.TURNING_RIGHT: TurningRightState(),
            VehicleState.U_TURN: UTurnState(),
            VehicleState.COLLISION_RECOVERY: CollisionRecoveryState(),
            VehicleState.EMERGENCY_STOP: EmergencyStopState(),
        }
        
        # Current state
        self.current_state = VehicleState.INITIALIZING
        self.previous_state = None
        
        # Context
        self.context = StateContext()
        
        # Enter initial state
        self.states[self.current_state].enter(self.context)
    
    def update(self, context_updates: Dict[str, Any]) -> None:
        """
        Update state machine.
        
        Args:
            context_updates: Dictionary of context updates
        """
        # Update context
        for key, value in context_updates.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
        
        # Execute current state
        next_state = self.states[self.current_state].execute(self.context)
        
        # Handle state transition
        if next_state is not None and next_state != self.current_state:
            self.transition_to(next_state)
    
    def transition_to(self, new_state: VehicleState) -> None:
        """Transition to a new state"""
        self.logger.info(f"State transition: {self.current_state.name} -> {new_state.name}")
        
        # Exit current state
        self.states[self.current_state].exit(self.context)
        
        # Update state
        self.previous_state = self.current_state
        self.current_state = new_state
        
        # Enter new state
        self.states[new_state].enter(self.context)
    
    def get_current_state(self) -> VehicleState:
        """Get current state"""
        return self.current_state
    
    def get_context(self) -> StateContext:
        """Get current context"""
        return self.context
    
    def emergency_stop(self) -> None:
        """Trigger emergency stop"""
        self.transition_to(VehicleState.EMERGENCY_STOP)
    
    def start_vehicle(self) -> None:
        """Start the vehicle (from idle)"""
        if self.current_state == VehicleState.IDLE:
            idle_state = self.states[VehicleState.IDLE]
            if isinstance(idle_state, IdleState):
                idle_state.start()


