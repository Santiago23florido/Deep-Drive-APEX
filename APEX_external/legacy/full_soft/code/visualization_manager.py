"""
Visualization Manager
Controls all graphics and plotting based on configuration
"""

from typing import Optional, Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import multiprocessing as mp
from abc import ABC, abstractmethod

from code.log_manager import get_component_logger


class Visualizer(ABC):
    """Abstract base class for visualizers"""
    
    @abstractmethod
    def update(self, data: dict) -> None:
        """Update visualization with new data"""
        pass
    
    @abstractmethod
    def show(self) -> None:
        """Show the visualization"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the visualization"""
        pass


class LidarVisualizer(Visualizer):
    """LiDAR point cloud visualizer"""
    
    def __init__(self, enable_algorithm_view: bool = True):
        self.logger = get_component_logger("LidarVisualizer")
        self.enable_algorithm_view = enable_algorithm_view
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.suptitle("LiDAR Visualization")
        
        # Plot elements
        self.raw_plot = None
        self.filtered_plot = None
        self.target_arrow = None
        self.hitbox_plot = None
        
        # Setup axis
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-0.5, 5)
        self.ax.set_aspect("equal")
        self.ax.grid(True)
        
        plt.ion()  # Interactive mode
    
    def update(self, data: dict) -> None:
        """
        Update visualization with new data.
        
        Expected data keys:
        - 'raw_lidar': Raw LiDAR data
        - 'filtered_lidar': Filtered LiDAR data
        - 'target_angle': Target steering angle
        - 'target_speed': Target speed
        - 'hitbox': Hitbox polygon points
        """
        self.ax.clear()
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-0.5, 5)
        self.ax.set_aspect("equal")
        self.ax.grid(True)
        
        # Plot raw LiDAR data
        if 'raw_lidar' in data:
            raw_lidar = data['raw_lidar']
            angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
            x = raw_lidar * np.sin(angles)
            y = raw_lidar * np.cos(angles)
            self.ax.plot(x, y, 'o', color='lightblue', markersize=2, label='Raw LiDAR')
        
        # Plot filtered data if algorithm view enabled
        if self.enable_algorithm_view and 'filtered_lidar' in data:
            filtered = data['filtered_lidar']
            angles = data.get('filtered_angles', np.linspace(0, np.pi, len(filtered)))
            x = filtered * np.sin(angles)
            y = filtered * np.cos(angles)
            self.ax.plot(x, y, 'o', color='blue', markersize=4, label='Filtered')
        
        # Plot target direction
        if self.enable_algorithm_view and 'target_angle' in data and 'target_speed' in data:
            angle_rad = np.radians(data['target_angle'])
            length = abs(data['target_speed']) * 0.5
            
            # Direction arrow
            dx = length * np.sin(angle_rad)
            dy = length * np.cos(angle_rad)
            self.ax.arrow(0, 0, dx, dy, head_width=0.1, head_length=0.2,
                         fc='red', ec='red', alpha=0.7, label='Target')
        
        # Plot hitbox
        if 'hitbox' in data:
            hitbox = data['hitbox']
            angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
            x = hitbox * np.sin(angles)
            y = hitbox * np.cos(angles)
            self.ax.plot(x, y, 'r--', linewidth=1, label='Hitbox')
        
        self.ax.legend(loc='upper right')
        plt.pause(0.01)
    
    def show(self) -> None:
        """Show the visualization window"""
        plt.show(block=False)
    
    def close(self) -> None:
        """Close the visualization"""
        plt.close(self.fig)


class CameraVisualizer(Visualizer):
    """Camera and color detection visualizer"""
    
    def __init__(self):
        self.logger = get_component_logger("CameraVisualizer")
        
        # Create figure with subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle("Camera Visualization")
        
        self.ax_original = self.axes[0, 0]
        self.ax_red = self.axes[0, 1]
        self.ax_green = self.axes[1, 0]
        self.ax_overlay = self.axes[1, 1]
        
        # Set titles
        self.ax_original.set_title("Original")
        self.ax_red.set_title("Red Mask")
        self.ax_green.set_title("Green Mask")
        self.ax_overlay.set_title("Detection Overlay")
        
        for ax in self.axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.ion()
    
    def update(self, data: dict) -> None:
        """
        Update camera visualization.
        
        Expected data keys:
        - 'frame': Original camera frame
        - 'red_mask': Red color mask
        - 'green_mask': Green color mask
        - 'overlay': Overlay visualization
        - 'detection_status': Detection status text
        """
        if 'frame' in data:
            self.ax_original.clear()
            self.ax_original.imshow(data['frame'])
            self.ax_original.set_title("Original")
            self.ax_original.set_xticks([])
            self.ax_original.set_yticks([])
        
        if 'red_mask' in data:
            self.ax_red.clear()
            self.ax_red.imshow(data['red_mask'], cmap='Reds')
            self.ax_red.set_title("Red Mask")
            self.ax_red.set_xticks([])
            self.ax_red.set_yticks([])
        
        if 'green_mask' in data:
            self.ax_green.clear()
            self.ax_green.imshow(data['green_mask'], cmap='Greens')
            self.ax_green.set_title("Green Mask")
            self.ax_green.set_xticks([])
            self.ax_green.set_yticks([])
        
        if 'overlay' in data:
            self.ax_overlay.clear()
            self.ax_overlay.imshow(data['overlay'])
            
            # Add detection status text
            if 'detection_status' in data:
                self.ax_overlay.text(
                    0.5, 0.95, data['detection_status'],
                    transform=self.ax_overlay.transAxes,
                    ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )
            
            self.ax_overlay.set_title("Detection Overlay")
            self.ax_overlay.set_xticks([])
            self.ax_overlay.set_yticks([])
        
        plt.pause(0.01)
    
    def show(self) -> None:
        """Show the visualization window"""
        plt.show(block=False)
    
    def close(self) -> None:
        """Close the visualization"""
        plt.close(self.fig)


class StateVisualizer(Visualizer):
    """State machine visualizer"""
    
    def __init__(self):
        self.logger = get_component_logger("StateVisualizer")
        
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.suptitle("Vehicle State Monitor")
        
        # History
        self.state_history = []
        self.time_history = []
        self.max_history = 100
        
        plt.ion()
    
    def update(self, data: dict) -> None:
        """
        Update state visualization.
        
        Expected data keys:
        - 'state': Current state name
        - 'timestamp': Current timestamp
        - 'speed': Current speed
        - 'steering': Current steering angle
        """
        if 'state' in data and 'timestamp' in data:
            self.state_history.append(data['state'])
            self.time_history.append(data['timestamp'])
            
            # Keep only recent history
            if len(self.state_history) > self.max_history:
                self.state_history = self.state_history[-self.max_history:]
                self.time_history = self.time_history[-self.max_history:]
        
        # Update display
        self.ax.clear()
        
        # Show current state prominently
        if 'state' in data:
            self.ax.text(
                0.5, 0.7, f"Current State:\n{data['state']}",
                ha='center', va='center',
                fontsize=20, fontweight='bold',
                transform=self.ax.transAxes
            )
        
        # Show speed and steering
        info_text = ""
        if 'speed' in data:
            info_text += f"Speed: {data['speed']:.2f} m/s\n"
        if 'steering' in data:
            info_text += f"Steering: {data['steering']:.1f}°"
        
        if info_text:
            self.ax.text(
                0.5, 0.3, info_text,
                ha='center', va='center',
                fontsize=14,
                transform=self.ax.transAxes
            )
        
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        
        plt.pause(0.01)
    
    def show(self) -> None:
        """Show the visualization window"""
        plt.show(block=False)
    
    def close(self) -> None:
        """Close the visualization"""
        plt.close(self.fig)


class VisualizationManager:
    """
    Manages all visualizations based on configuration.
    """
    
    def __init__(
        self,
        enable_lidar: bool = False,
        enable_camera: bool = False,
        enable_state: bool = False,
        enable_algorithm_view: bool = True
    ):
        self.logger = get_component_logger("VisualizationManager")
        
        self.enable_lidar = enable_lidar
        self.enable_camera = enable_camera
        self.enable_state = enable_state
        
        # Create visualizers if enabled
        self.lidar_viz: Optional[LidarVisualizer] = None
        self.camera_viz: Optional[CameraVisualizer] = None
        self.state_viz: Optional[StateVisualizer] = None
        
        if enable_lidar:
            self.lidar_viz = LidarVisualizer(enable_algorithm_view)
            self.logger.info("LiDAR visualizer enabled")
        
        if enable_camera:
            self.camera_viz = CameraVisualizer()
            self.logger.info("Camera visualizer enabled")
        
        if enable_state:
            self.state_viz = StateVisualizer()
            self.logger.info("State visualizer enabled")
    
    def update_lidar(self, data: dict) -> None:
        """Update LiDAR visualization"""
        if self.lidar_viz is not None:
            try:
                self.lidar_viz.update(data)
            except Exception as e:
                self.logger.error(f"Error updating LiDAR visualization: {e}")
    
    def update_camera(self, data: dict) -> None:
        """Update camera visualization"""
        if self.camera_viz is not None:
            try:
                self.camera_viz.update(data)
            except Exception as e:
                self.logger.error(f"Error updating camera visualization: {e}")
    
    def update_state(self, data: dict) -> None:
        """Update state visualization"""
        if self.state_viz is not None:
            try:
                self.state_viz.update(data)
            except Exception as e:
                self.logger.error(f"Error updating state visualization: {e}")
    
    def show_all(self) -> None:
        """Show all enabled visualizations"""
        if self.lidar_viz:
            self.lidar_viz.show()
        if self.camera_viz:
            self.camera_viz.show()
        if self.state_viz:
            self.state_viz.show()
    
    def close_all(self) -> None:
        """Close all visualizations"""
        if self.lidar_viz:
            self.lidar_viz.close()
        if self.camera_viz:
            self.camera_viz.close()
        if self.state_viz:
            self.state_viz.close()
        
        self.logger.info("All visualizations closed")
    
    def is_enabled(self) -> bool:
        """Check if any visualization is enabled"""
        return self.enable_lidar or self.enable_camera or self.enable_state


if __name__ == "__main__":
    # Test visualization system
    from log_manager import initialize_logging
    import time
    
    initialize_logging(console_level="INFO")
    
    print("=== Visualization System Test ===\n")
    
    # Create visualization manager with all visualizers enabled
    viz_mgr = VisualizationManager(
        enable_lidar=True,
        enable_camera=False,  # Disable camera for this test
        enable_state=True,
        enable_algorithm_view=True
    )
    
    viz_mgr.show_all()
    
    # Simulate updates
    for i in range(250):
        # Update LiDAR
        lidar_data = {
            'raw_lidar': np.random.rand(360) * 3,
            'filtered_lidar': np.random.rand(180) * 2,
            'filtered_angles': np.linspace(0, np.pi, 180),
            'target_angle': np.random.uniform(-30, 30),
            'target_speed': np.random.uniform(0.5, 1.5),
            'hitbox': np.ones(360) * 0.3
        }
        viz_mgr.update_lidar(lidar_data)
        
        # Update state
        states = ['NORMAL_DRIVING', 'OBSTACLE_DETECTED', 'REVERSING', 'TURNING']
        state_data = {
            'state': states[i % len(states)],
            'timestamp': time.time(),
            'speed': np.random.uniform(0, 1.5),
            'steering': np.random.uniform(-30, 30)
        }
        viz_mgr.update_state(state_data)
        
        time.sleep(0.5)
    
    print("\nTest complete. Close windows to exit.")
    input("Press Enter to close visualizations...")
    viz_mgr.close_all()