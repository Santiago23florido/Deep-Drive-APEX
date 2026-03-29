import multiprocessing as mp
import numpy as np
import time

from code.Interfaces.LidarInterface import create_lidar_interface, RealLidarInterface, LidarMode
from code.config_loader import ConfigLoader, get_config

from code.visualization_manager import VisualizationManager

def camera_interface_test():
    
    config = get_config("new-config.json")  # JSON File config

    # Create Camera Interface
    from code.Interfaces.interface_camera import CameraInterfaceImpl
    camera = CameraInterfaceImpl(config_file=config)
    
    # Start Camera
    camera.cleanup()  # Ensure camera is stopped before starting
    camera.__init__(config_file=config)  # Re-initialize to apply cleanup
    time.sleep(2)  # Wait for camera to initialize

    viz = VisualizationManager(
        enable_camera=True,
        enable_algorithm_view=False
        )
    
    viz.show_all()
    
    total_frames = 100
    for i in range(total_frames):
        frame = camera.get_camera_frame()
        if frame is not None:
            print(f"Frame {i+1}/{total_frames} captured with shape: {frame.shape}")
            viz.update_camera(frame)
        else:
            print(f"Frame {i+1}/{total_frames} capture failed.")
        time.sleep(0.1)  # wait a bit between frames

    viz.close_all()
    
    # Cleanup Camera
    camera.cleanup()


camera_interface_test()