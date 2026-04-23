from code.algorithm.constants import STEER_VARIATION_RATE, STEER_CENTER
from code.Interfaces.interface_decl import SteerInterface
from code.config_loader import ConfigLoader, get_config
from code.log_manager import get_logger
from code.Interfaces.FakePWM import FakeGPIO, FakePWM

# import algorithm.voiture_logger as voiture_logger
import time

from typing import Dict, Any
import signal
import sys

class RealSteerInterface(SteerInterface):
    """
    Real implementation of SteerInterface
    that uses hardware PWM to control a servo.
    """
    
    def __init__(self, 
                 config_loader: ConfigLoader = None,
                 channel: int = 1, frequency: float = 50.0):
        """
        Args:
            config_loader (ConfigLoader): Configuration loader for the steering interface.
            channel (int): PWM channel controlling the steering servo (typically 1).
            frequency (float): Servo PWM frequency, typically 50 Hz.
        """
        self.logger = get_logger("RealSteerInterface")
        
        if config_loader:
            self.logger.info("Initializing RealSteerInterface with provided config_loader")
        else:
            self.logger.info("Initializing RealSteerInterface with default configuration")
        
        self.config = config_loader or get_config()
        self.logger.info(f"RealSteerInterface configuration: {self.config.lidar}")
        lidar_cfg = self.config.lidar
        
        
        self.channel    = channel or getattr(lidar_cfg, 'channel', 1)
        self.frequency  = frequency or getattr(lidar_cfg, 'frequency', 50.0)
        self.dc_min     = getattr(lidar_cfg, 'dc_min', 5.0)       # Minimum duty cycle for full left
        self.dc_max     = getattr(lidar_cfg, 'dc_max', 8.6)       # Maximum duty cycle for full right
        
        # I dont know why is necessary but it is in the original code.
        self.limit = getattr(lidar_cfg, 'limit', 18.0)            # Maximum steering angle in degrees
        
        #neutral_dc relates to central position of the servo.
        self.neutral_dc = getattr(lidar_cfg, 'neutral_dc', (self.dc_min + self.dc_max) / 2)  # Default neutral duty cycle
        self.variation_rate = 0.5 * (self.dc_max - self.dc_min) / self.limit
        
        
        
        
        # If we want to debug without hardware        
        try:
            from raspberry_pwm import PWM  # type: ignore
            self._pwm = PWM(channel=self.channel, frequency=self.frequency)
        except (ImportError, ModuleNotFoundError) as e:
            self.logger.warning(
            f"raspberry_pwm not found ({e}). Falling back to FakePWM (no real hardware control)."
            )
            self._pwm = FakePWM(channel=self.channel, frequency=self.frequency)
            
        self._pwm.start(self.neutral_dc)
        self.logger.info(f"Steering PWM initialized and set to neutral ({self.neutral_dc}%)")
        
    def start(self):
        self.logger.info("Starting RealSteerInterface PWM")
        self._pwm.start(self.neutral_dc)

    def stop(self):
        self._pwm.stop()
        self.logger.info("Steering servo stopped")
    
    def set_steering_angle(self, angle: float) -> None:
        """
        Sets the steering angle in degrees.

        Args:
            angle (float): Steering angle in degrees.
                           For many hobby servos:
                             - ~ -30 degrees => 5% duty cycle
                             - ~   0 degrees => 7.5% duty cycle (center)
                             - ~ +30 degrees => 10% duty cycle
        """
        duty_cycle = self.compute_pwm(angle)
        self._pwm.set_duty_cycle(duty_cycle)
        self.logger.debug(f"Steering angle set to {angle}° => duty cycle: {duty_cycle}%")

    def compute_pwm(self, steer: float) -> float:    
        center = self.neutral_dc
        return steer * self.variation_rate + center
    

    def get_info(self) -> Dict[str, Any]:
        return {
            "type": "RealSteerInterface",
            "variation_rate": self.variation_rate,
            "center": self.neutral_dc,
        }


def handle_sigint(sig, frame):
    print("\nTest interrupted by user")
    if 'steering' in globals():
        steering.stop()
    print("Steering servo stopped and PWM cleaned up")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_sigint)

    test_sequence = [
        (0.0, 1.0),
        (10.0, 2.0),
        (20.0, 1.5),
        (30.0, 1.0),
        (0.0, 2.0),
        (-10.0, 2.0),
        (-20.0, 1.5),
        (-30.0, 1.0),
        (0.0, 2.0),
        (25.0, 1.0),
        (-25.0, 1.0),
        (15.0, 1.0),
        (-15.0, 1.0),
        (0.0, 1.0),
    ]

    steering = RealSteerInterface()

    try:
        print("Starting steering test sequence...")
        print("-" * 50)

        for i, (angle, duration) in enumerate(test_sequence):
            direction = "CENTER" if abs(angle) < 0.1 else "RIGHT" if angle > 0 else "LEFT"
            print(f"Step {i+1}/{len(test_sequence)}: Turn {direction} at {abs(angle):.1f}° for {duration:.1f}s")

            steering.set_steering_angle(angle)

            duty_cycle = steering.compute_pwm(angle)
            print(f"  PWM duty cycle: {duty_cycle:.2f}%")

            time.sleep(duration)

        print("-" * 50)
        print("Steering test sequence completed successfully!")

    except Exception as e:
        print(f"\nError during test: {e}")
    finally:
        steering.stop()
        print("Steering servo stopped and PWM cleaned up")