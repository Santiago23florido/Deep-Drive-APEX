import time
import numpy as np
# import code.algorithm.Interfaces as interfaces

from code.config_loader import get_config
from code.visualization_manager import VisualizationManager
from code.Interfaces.interface_serial import SharedMemBatteryInterface, SharedMemUltrasonicInterface, SharedMemSpeedInterface, start_serial_monitor
from code.Interfaces.interface_lidar import RPLidarReader
from code.Interfaces.interface_motor import RealMotorInterface
from code.Interfaces.interface_steer import RealSteerInterface
from code.Interfaces.interface_camera import RealCameraInterface
# from code.Interfaces.interface_console import ColorConsoleInterface
from code.Interfaces.LidarInterface import create_lidar_interface
from algorithm.voiture_logger import CentralLogger
from algorithm.constants import LIDAR_BAUDRATE, FIELD_OF_VIEW_DEG
from algorithm.voiture_algorithm import VoitureAlgorithm
 
 
def main(is_simulation=False):
    
    
    config_file = "new-config.json"
    config = get_config(config_file)
    
    # try:
        
        # RPLidarReader(port="/dev/ttyUSB0", baudrate=LIDAR_BAUDRATE)
        # I_Lidar = RPLidarReader(port="/dev/ttyUSB0", baudrate=LIDAR_BAUDRATE)
    I_Lidar = create_lidar_interface(
        mode= "simulation" if is_simulation else "real",
        config_file=config
    )

    I_Lidar.start()
    I_Steer = RealSteerInterface(config_loader=config)
    I_Motor = RealMotorInterface(config_loader=config)
    
    
    
    I_SpeedReading = SharedMemSpeedInterface()
    I_back_wall_distance_reading = SharedMemUltrasonicInterface()
    I_BatteryReading = SharedMemBatteryInterface()
    I_Camera = RealCameraInterface()
    
    
    nonzero_count = np.count_nonzero(I_Lidar.get_lidar_data())
    
    while(not I_Lidar.is_running()):
        print("Waiting for Lidar to start... (nonzero count:", nonzero_count, ")")
        time.sleep(2)
        
    

    
    viz_mgr = VisualizationManager(
        enable_lidar=True,
        enable_camera=True,
        enable_state=False,
        enable_algorithm_view=True
    )

    
    algorithm = VoitureAlgorithm(
                    lidar=I_Lidar,
                    ultrasonic=I_back_wall_distance_reading,
                    speed=I_SpeedReading,
                    battery=I_BatteryReading, 
                    camera=I_Camera,
                    steer=I_Steer,
                    motor=I_Motor,
                    console = None
                    )

    input("Press ENTER to start the code...\n")
    print("Running...")
    viz_mgr.show_all()
        
    while(True):
        
        lidar_data = {
            'raw_lidar': I_Lidar.get_lidar_data(),
            'filtered_lidar': np.ones(360) * 0.3,
            'fov_deg': FIELD_OF_VIEW_DEG,
            }
        
        cam_data = {
            'frame' : I_Camera.get_camera_frame(),
            # 'red_mask' : I_Camera.get_red_mask(),
            # 'green_mask' : I_Camera.get_green_mask(),
            # 'overlay' : I_Camera.get_overlay()
        }
        

        viz_mgr.update_lidar(lidar_data)
        viz_mgr.update_camera(cam_data)
        
        loop(algorithm)
        
    # except KeyboardInterrupt:
    #     print("[Main] Interrupted by user.")
    # finally:
    #     algorithm.cleanup()  # Ensure cleanup is called to stop all interfaces
        # I_Motor.stop()
        # I_Steer.stop()
        # I_Lidar.stop()


def loop(algorithm: VoitureAlgorithm):
    algorithm.run_step()
    time.sleep(0.05)


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Run the autonomous car algorithm.")
    parser.add_argument('--config', type=str, default="new-config.json", help='Path to configuration file')
    parser.add_argument('--simulation', action='store_true', help='Run in simulation mode')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    is_simulation = args.simulation
    if args.simulation:
        # Modify internal parameters for simulation mode
        print("Running in simulation mode.", args.simulation)
        
    
    main(args.simulation)