from code import config_loader
from code.Interfaces.FakePWM import FakePWM
from code.log_manager import get_logger
from code.Interfaces.interface_decl import AbstractMotorInterface
from code.algorithm.constants import ESC_DC_MIN, ESC_DC_MAX
from code.log_manager import get_logger
from code.config_loader import ConfigLoader, get_config
from typing import Dict, Any
import time

# NEUTRAL_DC = (ESC_DC_MIN + ESC_DC_MAX) / 2
MIN_DC = ESC_DC_MIN   # Full reverse
MAX_DC = ESC_DC_MAX   # Full forward

MAX_SPEED = 3.0  # BUG FIX: constante estava duplicada inline; centralizada aqui.


class RealMotorInterface(AbstractMotorInterface):
    def __init__(self, config_loader: ConfigLoader = None, 
                  channel: int = 0, 
                  frequency: float = 50.0):
        
        self.logger = get_logger("RealMotorInterface")
        
        if config_loader:
            self.logger.info("Initializing RealSteerInterface with provided config_loader")
        else:
            self.logger.info("Initializing RealSteerInterface with default configuration")
        
        
        self.config = config_loader or get_config()
        self.logger.info(f"RealMotorInterface configuration: {self.config.motor}")
        motor = self.config.motor
        
        self.speed = 0.0
        self._in_reverse_mode = False
        self.channel    = channel or getattr(motor, 'channel', 1)
        self.frequency  = frequency or getattr(motor, 'frequency', 50.0)
        self.dc_min     = getattr(motor, 'dc_min', 5.0)       # Minimum duty cycle for full reverse
        self.dc_max     = getattr(motor, 'dc_max', 8.6)       # Maximum duty cycle for full forward
        self.neutral_dc = getattr(motor, 'neutral_dc', (self.dc_min + self.dc_max) / 2)  # Default neutral duty cycle
        
        try:
            from raspberry_pwm import PWM  # type: ignore
            self._pwm = PWM(channel=self.channel, frequency=self.frequency)
        except (ImportError, ModuleNotFoundError) as e:
            self.logger.error(f"raspberry_pwm not found ({e}). USING -> FakePWM. SIMULATION MODE ENABLED")
            self._pwm = FakePWM(channel=self.channel, frequency=self.frequency)
        
        
        
        self._pwm.start(self.neutral_dc)
        self.logger.info("Motor PWM initialized and set to neutral")
        
        
    def stop(self) -> None:
        """Stops the ESC and PWM"""
        self._pwm.set_duty_cycle(self.neutral_dc)
        self._in_reverse_mode = False
        self._pwm.stop()
        self.speed = 0.0
        self.logger.info("Motor stopped")

    def start(self) -> None:
        """Ensures the motor is at neutral on start."""
        self._pwm.start(self.neutral_dc)
        self.logger.info("Motor started at neutral")

    def _enter_reverse_mode(self) -> None:
        """
        Special sequence to enter reverse mode on the Maverick msc-30BR-WP ESC.
        Most brushed ESCs need a quick brake-neutral-reverse sequence.
        """
        self._pwm.set_duty_cycle(7.0)   # Brake position
        time.sleep(0.03)
        self._pwm.set_duty_cycle(self.neutral_dc)
        time.sleep(0.03)
        self._in_reverse_mode = True
        self.logger.debug("Entered reverse mode with Maverick ESC sequence")

    def _exit_reverse_mode(self) -> None:
        """Exit reverse mode and go back to neutral for Maverick ESC"""
        self._pwm.set_duty_cycle(self.neutral_dc)
        time.sleep(0.1)
        self._in_reverse_mode = False
        self.logger.debug("Exited reverse mode")

    def set_speed(self, s: float) -> None:
        """
        Sets the motor speed from -3.0 (full reverse) to 3.0 (full forward).
        0.0 is neutral position.
        """
        self.speed = max(-MAX_SPEED, min(s, MAX_SPEED))

        if self.speed < 0 and not self._in_reverse_mode:
            self._enter_reverse_mode()
        elif self.speed >= 0 and self._in_reverse_mode:
            self._exit_reverse_mode()

        if abs(self.speed) < 0.1:
            duty_cycle = self.neutral_dc
        elif self.speed >= 0:
            duty_cycle = self.neutral_dc + (self.speed / MAX_SPEED) * (self.dc_max - self.neutral_dc)
        else:
            # BUG FIX: a fórmula original era:
            #   MIN_DC + ((speed + 3.0) / 3.0) * (NEUTRAL_DC - MIN_DC)
            # que é matematicamente equivalente a:
            #   NEUTRAL_DC + (speed / MAX_SPEED) * (NEUTRAL_DC - MIN_DC)
            # A versão original funcionava mas era inconsistente com a forma forward.
            # Aqui usamos a forma simétrica e clara:
            duty_cycle = self.neutral_dc + (self.speed / MAX_SPEED) * (self.neutral_dc - self.dc_min)

        self._pwm.set_duty_cycle(duty_cycle)
        self.logger.debug(f"Speed set to {self.speed} => duty cycle: {duty_cycle}%")

    def get_speed(self) -> float:
        return self.speed

    def get_info(self) -> Dict[str, Any]:
        return {
            "neutral_dc": self.neutral_dc,
            "min_dc": self.dc_min,
            "max_dc": self.dc_max,
            "current_speed": self.speed,
            "in_reverse_mode": self._in_reverse_mode,
        }

    def set_duty_cycle(self, duty_cycle: float) -> None:
        """Direct control of PWM duty cycle for debugging or manual control"""
        self._pwm.set_duty_cycle(duty_cycle)
        self.logger.debug(f"Duty cycle directly set to {duty_cycle}%")


if __name__ == "__main__":
    test_sequence = [
        (0.0, 1.0),
        (0.5, 2.0),
        (1.5, 2.0),
        (3.0, 1.5),
        (0.0, 2.0),
        (-0.5, 2.0),
        (-1.5, 2.0),
        (-3.0, 1.5),
        (0.0, 2.0),
        (2.0, 1.5),
        (-2.0, 1.5),
        (1.0, 3.0),
        (0.0, 1.0),
    ]
    
    
    config_file = "new-config.json"
    config = get_config(config_file)    
    

    motor = RealMotorInterface(config_loader=config)

    try:
        print("Starting motor test sequence...")
        print("-" * 40)

        for i, (speed, duration) in enumerate(test_sequence):
            direction = "NEUTRAL" if abs(speed) < 0.1 else "FORWARD" if speed > 0 else "REVERSE"
            print(f"Step {i+1}/{len(test_sequence)}: {direction} speed at {abs(speed):.1f} for {duration:.1f}s")
            motor.set_speed(speed)
            time.sleep(duration)

        print("-" * 40)
        print("Test sequence completed successfully!")

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError during test: {e}")
    finally:
        motor.stop()
        print("Motor stopped and PWM cleaned up")