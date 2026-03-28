import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../code'))  # Add the code directory to the path

import time
import numpy as np
from state_machine import *
from log_manager import initialize_logging




if __name__ == "__main__":
    # Test the state machine

    initialize_logging(console_level="DEBUG")
        
    # Create state machine
    sm = StateMachine()
    

    print("\n=== Extended State Machine Test ===\n")

    sm2 = StateMachine()
    print(f"Initial state: {sm2.get_current_state().name}\n")

    # Simulate initialization phase
    print("--- Initialization Phase ---")
    for i in range(10):
        context_updates = {
            'lidar_data': np.random.rand(360) if i > 2 else np.zeros(360),
            'current_speed': 0.0,
        }
        sm2.update(context_updates)
        print(f"Step {i+1}: {sm2.get_current_state().name}")
        time.sleep(0.1)

    # Transition to idle
    print("\n--- Idle State ---")
    print(f"Current: {sm2.get_current_state().name}")
    sm2.start_vehicle()
    print("Vehicle start command sent")

    # Simulate normal driving with obstacle detection
    print("\n--- Normal Driving with Obstacles ---")
    for i in range(15):
        obstacle = i >= 5 and i < 10
        wall_close = i >= 10
        color = "red_left" if i == 12 else None
        
        context_updates = {
            'lidar_data': np.random.rand(360),
            'current_speed': 1.5,
            'obstacle_in_front': obstacle,
            'wall_too_close': wall_close,
            'color_detection': color,
        }
        
        sm2.update(context_updates)
        state = sm2.get_current_state().name
        print(f"Step {i+1}: {state} | Speed: {sm2.context.target_speed:.1f}")
        time.sleep(1.5)

    # Test emergency stop
    print("\n--- Emergency Stop Test ---")
    print(f"Before emergency: {sm2.get_current_state().name}")
    sm2.emergency_stop()
    print(f"After emergency: {sm2.get_current_state().name}")

    print(f"\nTest complete. Final state: {sm2.get_current_state().name}")
    
    print(f"\nFinal state: {sm.get_current_state().name}")