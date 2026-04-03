

from time import time
from turtle import speed
from typing import Optional
from enum import Enum


# from code.Interfaces.LidarInterface import SimulatedLidarInstance
from code.Interfaces.interface_motor import RealMotorInterface
from code.Interfaces.interface_decl import AbstractMotorInterface, SteerInterface
from code.config_loader import ConfigLoader, get_config
from code.log_manager import LogManager, get_logger, get_component_logger

from code.Interfaces.interface_steer import RealSteerInterface
class MotorMode(Enum):
    """Modos de operação do motor"""
    REAL = "real"               # Motor físico real
    SIMULATION = "simulation"   # Simulated motor (random values as input)


class RealMotorInterface(AbstractMotorInterface):
    def __init__(self, motor_instance=None):
        self.logger = get_logger("RealMotorInterface")
        
        if motor_instance is None:
            self.logger.info("No motor instance provided, creating new RealMotorInstance")
            self.motor_instance = MaverickMotorInstance()
        else:
            self.logger.info("Using provided motor instance for RealMotorInterface")
            self.motor_instance = motor_instance
        
        self.set_speed(0)  # Start at neutral
    
    def get_info(self):
        """Returns infos about the motor interface (mode, status, etc)"""
        return {
            "mode": MotorMode.REAL.value,
            "speed": self.get_speed(),
            "status": "running" if self.motor_instance else "stopped"
        }
        
        
    def start(self):
        """Starts the motor interface (if needed)"""
        self.logger.info("Starting RealMotorInterface")
        self.motor_instance.start()
        
    def stop(self):
        """Stops the ESC and PWM"""
        self.motor_instance.stop()
        self.logger.info("Motor stopped")
        
    def set_speed(self, speed: float):
        """Sets the motor speed from -100.0 (full reverse) to 100.0 (full forward)"""
        self.speed = speed
        self.motor_instance.set_speed(speed)
        self.logger.debug(f"Set speed to {speed}%")
    def get_speed(self) -> float:
        return self.speed
    
    

#TODO: Refactor this class to not use sleep but threading management
class MaverickMotorInstance:
    def __init__(
    self,
    config_loader: Optional[ConfigLoader] = None,
    port: Optional[str] = None,
    baudrate: Optional[int] = None,
    fov_filter: Optional[int] = None,
    heading_offset_deg: Optional[int] = None,
    point_timeout_ms: Optional[int] = None
    ):
        # On Raspberry Pi, we can use the real PWM interface to control directly
        from raspberry_pwm import PWM
        
        self.logger     = get_logger("MaverickMotorInstance")
        
        if config_loader:
            self.logger.info("Initializing MaverickMotorInstance with config file")
        else:   
            self.logger.warning("No config file provided, using default parameters")
        
        # Get defaults from config if not provided
        config = config_loader or get_config()
        self.config = config.motor
        
        # Load parameters from config
        self.dc_min = self.config.dc_min           #Old name MIN_DC
        self.dc_max = self.config.dc_max           #Old name MAX_DC
        # self.channel = self.config.pwm_channel     #Old name PWM_CHANNEL 
        self.channel = 1
        self.neutral_dc = self.config.neutral_dc   #Old name NEUTRAL_DC
        self.ticks_per_revolution = self.config.ticks_to_meter
        self.frequency = self.config.frequency #Old name PWM_FREQUENCY

        self._pwm = PWM(channel=self.channel, frequency=self.frequency)

        self._in_reverse_mode = False

        self.logger.info(
            f"Motor parameters loaded: dc_min={self.dc_min}, "
            f"dc_max={self.dc_max}, channel={self.channel}, "
            f"neutral_dc={self.neutral_dc}, "
            f"ticks_per_revolution={self.ticks_per_revolution}, "
            f"frequency={self.frequency}"
        )

        # Start the motor interface
        self.start()

# -------------------------------------------------------
# AbstractMotorInterface methods
    def set_speed(self, speed: float):
        """   
        Speed range: -100 (full reverse) to +100 (full forward)
        Think this as a percentage of max speed, where 0 is neutral.
        
        For Maverick msc-30BR-WP Brushed ESC:
        - Full reverse (-3.0) = 5% duty cycle
        - Neutral (0.0) = 7.5% duty cycle
        - Full forward (3.0) = 10% duty cycle
        """

        self.speed  = self._bound_speed(speed)
        
        # Changing direction - special handling for Maverick ESC
        if speed < 0 and not self._in_reverse_mode:
            self._enter_reverse_mode()
        elif speed >= 0 and self._in_reverse_mode:
            self._exit_reverse_mode()

        if abs(speed) < 1.0: # neutral range
            duty_cycle = self.neutral_dc
        elif speed > 0: # Forward
            duty_cycle = self.neutral_dc + (speed / 100.0) * (self.dc_max - self.neutral_dc)
        else: # Reverse
            duty_cycle = self.neutral_dc + (speed / 100.0) * (self.neutral_dc - self.dc_min)
        
        self.set_duty_cycle(duty_cycle)

    def get_speed(self) -> float:
        return self.speed
    
    def stop(self):
        """Stops the ESC and PWM"""
        self._pwm.set_duty_cycle(self.neutral_dc)  # Return to neutral for Maverick ESC
        self._in_reverse_mode = False
        self._pwm.stop()
        self.speed = 0
        self.logger.info("Motor stopped")

    def get_info(self) -> dict:
        """Returns current motor status and parameters"""
        return {
            "dc_min": self.dc_min,
            "dc_max": self.dc_max,
            "channel": self.channel,
            "neutral_dc": self.neutral_dc,
            "ticks_per_revolution": self.ticks_per_revolution,
            "frequency": self.frequency,
            "current_speed": self.get_speed()
        }
        
# -------------------------------------------------------
    def start(self):
        """Starts the motor interface (if needed)"""
        
        self.set_speed(0)                 # Start at neutral
        self._pwm.start(self.neutral_dc)  # Start at neutral position
        
        self.logger.info("Starting RealMotorInterface")

        
    def _bound_speed(self, speed: float) -> float:
        """Ensures the speed is within the allowed range"""
        return max(-100.0, min(100.0, speed))
    
    def _enter_reverse_mode(self):
        """
        Special sequence to enter reverse mode on the Maverick msc-30BR-WP ESC
        Most brushed ESCs need a quick brake-neutral-reverse sequence
        """
        
        self._pwm.set_duty_cycle(7.0)  # Brake position #TODO: why hardcoded 7.0?
        time.sleep(0.03)
        
        self._pwm.set_duty_cycle(self.neutral_dc)  # Neutral position for Maverick ESC
        time.sleep(0.03)
        
        # Now ESC should be ready to accept reverse commands
        self._in_reverse_mode = True
        self.logger.debug("Entered reverse mode with Maverick ESC sequence")
    
    def _exit_reverse_mode(self):
        """Exit reverse mode and go back to neutral for Maverick ESC"""
        self._pwm.set_duty_cycle(self.neutral_dc)  # Neutral duty cycle for Maverick ESC
        time.sleep(0.1)                            # Give ESC time to recognize the neutral position
        self._in_reverse_mode = False
        self.logger.debug("Exited reverse mode")



    def set_duty_cycle(self, duty_cycle: float):
        """Direct control of PWM duty cycle for debugging or manual control"""
        self._pwm.set_duty_cycle(duty_cycle)
        self.logger.debug(f"Duty cycle directly set to {duty_cycle}%")

class SimulatedMotorInstance:
    """Simulated motor instance for testing without physical hardware"""
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        
        # Logger setup
        self.logger = get_logger("SimulatedMotorInstance")
        self.speed = 0.0
        self._in_reverse_mode = False
        
        #config step
        config = config_loader or get_config()
        self.config = config.motor
        
        self.dc_min = self.config.motor_dc_min
        self.dc_max = self.config.motor_dc_max
        self.neutral_dc = self.config.motor_neutral_dc
        
        self.logger.info("SimulatedMotorInstance initialized")
    
    def start(self):
        """Start the simulated motor"""
        self.speed = 0.0
        self.logger.info("Simulated motor started")
    
    def stop(self):
        """Stop the simulated motor"""
        self.speed = 0.0
        self._in_reverse_mode = False
        self.logger.info("Simulated motor stopped")
    
    def set_speed(self, speed: float):
        """Set simulated motor speed"""
        self.speed = max(-100.0, min(100.0, speed))
        self.logger.debug(f"Simulated motor speed set to {self.speed}%")
    
    def get_speed(self) -> float:
        """Get current simulated motor speed"""
        return self.speed
    
    def set_duty_cycle(self, duty_cycle: float):
        """Set simulated PWM duty cycle"""
        self.logger.debug(f"Simulated duty cycle set to {duty_cycle}%")
    
    def get_info(self) -> dict:
        """Return simulated motor status"""
        return {"current_speed": self.speed, "mode": "simulated"}


# Encapsulated the Motor Logic, both real and simu: 
def create_motor_interface(mode: str, config_file: ConfigLoader, motor_instance=None) -> AbstractMotorInterface:
    """Factory para criar interface de motor"""
    mode = mode.lower()
    
    if mode == MotorMode.REAL.value:
        if motor_instance is None:
            motor_instance = MaverickMotorInstance(config_loader=config_file)
        return RealMotorInterface(motor_instance)
    
    elif mode == MotorMode.SIMULATION.value:
        # For simulation, we can create a simulated instance if not provided
        if motor_instance is None:
            motor_instance = SimulatedMotorInstance(config_file)
        return RealMotorInterface(motor_instance)

    else:
        raise ValueError(f"Unknown Motor mode: {mode}")


def create_steer_interface(mode: str) -> SteerInterface:
    """Factory para criar interface de direção"""
    
    #TODO: insert this in config.json
    channel = 1
    frequency = 50.0
    mode = mode.lower()
    
    return RealSteerInterface(channel=channel, frequency=frequency)
