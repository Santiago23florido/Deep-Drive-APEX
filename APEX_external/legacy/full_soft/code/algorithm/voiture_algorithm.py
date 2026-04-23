import time
from typing import Optional
from code.Interfaces.interface_decl import *
import cv2
import os
import datetime
from code.Interfaces.interface_decl import *
from code.algorithm.constants import HITBOX_H1, HITBOX_H2, HITBOX_W
from code.algorithm.control_camera import extract_info, DetectionStatus
from code.algorithm.control_direction import compute_steer_from_lidar, shrink_space
from code.algorithm.control_speed import compute_speed
from code.config_loader import ConfigLoader, get_config
from code.log_manager import get_logger




class VoitureAlgorithm:
    def __init__(self, 
                 lidar: AbstractLidarInterface, 
                 ultrasonic: UltrasonicInterface, 
                 speed: SpeedInterface, 
                 battery: BatteryInterface, 
                 camera: CameraInterface, 
                 steer: SteerInterface, 
                 motor: AbstractMotorInterface,
                 
                 config_loader: ConfigLoader = None,
                 back_distance: Optional[float] = None,
                #  console: ConsoleInterface
                console = None
                ):
        
        # self.logger = logging.getLogger(__name__)
        self.logger = get_logger("VoitureAlgorithm")
        
        self.config = config_loader or get_config()
        
        print("Loaded configuration:", self.config.safety)
        safety_config = self.config.safety
        self.back_distance = back_distance or getattr(safety_config, 'back_distance', 15)
        
        
        
        
        # Ensure all inputs implement the expected interfaces
        # if not isinstance(lidar, AbstractLidarInterface):
        #     self.logger.error("Lidar interface FAILED")    
        #     raise TypeError("lidar must implement AbstractLidarInterface")
        # if not isinstance(ultrasonic, UltrasonicInterface):
        #     self.logger.error("Ultrasonic interface FAILED")
        #     raise TypeError("ultrasonic must implement UltrasonicInterface")
        # if not isinstance(speed, SpeedInterface):
        #     self.logger.error("Speed interface FAILED")
        #     raise TypeError("speed must implement SpeedInterface")
        # if not isinstance(battery, BatteryInterface):
        #     self.logger.error("Battery interface FAILED")
        #     raise TypeError("battery must implement BatteryInterface")
        # if not isinstance(camera, CameraInterface):
        #     self.logger.error("Camera interface FAILED")
        #     raise TypeError("camera must implement CameraInterface")
        # if not isinstance(steer, SteerInterface):
        #     self.logger.error("Steer interface FAILED")
        #     raise TypeError("steer must implement SteerInterface")
        # if not isinstance(motor, AbstractMotorInterface):
        #     self.logger.error("Motor interface FAILED")
        #     raise TypeError("motor must implement AbstractMotorInterface")
        
        
        # Link all interfaces to the car
        self.lidar      = lidar
        self.ultrasonic = ultrasonic
        self.speed      = speed
        self.battery    = battery
        self.camera     = camera
        self.steer      = steer
        self.motor      = motor
        
        
        
        
       
        

        avg_r, avg_g, ratio_r, ratio_g, detection_status, processing_results = extract_info(self.camera.get_camera_frame(), *self.camera.get_resolution())
        print(detection_status)
        
            
    def detect_wheel_stopped_collision(self):
        
        #TODO: How we know the real motor speed? Also, we need to create a asynchronous monitor for this 
        """
        Monitors for wheel stoppage while motor is running, indicating a collision.
        Triggers collision response after X milliseconds of detected stoppage.
        """
        current_time = time.time()
        current_speed = self.speed.get_speed()
        motor_speed = self.motor.get_speed()
        
        # Define the threshold for stopped wheels (near zero velocity)
        STOPPED_THRESHOLD = 0.02  # m/s
        # Define time threshold for collision detection (in seconds)
        COLLISION_TIME_THRESHOLD = 0.5  # 500 milliseconds
        
        self.logger.debug(f"Current Speed: {current_speed}, Motor Speed: {motor_speed}")
        
        # Check if wheels are stopped but motor is running
        if current_speed < STOPPED_THRESHOLD and motor_speed > 0:
            
            # If this is the first detection of stoppage, record the time
            if not hasattr(self, '_wheel_stopped_start_time'):
                self.logger.debug("Wheels stopped detected, starting timer.")
                self._wheel_stopped_start_time = current_time
                self._collision_detected = False
            # If wheels have been stopped for longer than the threshold
            elif (current_time - self._wheel_stopped_start_time) > COLLISION_TIME_THRESHOLD and not self._collision_detected:
                self.logger.debug(f"[detect_wheel_stopped_collision] Collision detected due to wheel stoppage. Current Speed: {current_speed}, Motor Speed: {motor_speed}")
                self._collision_detected = True
                self.execute_reverse_maneuver()
        else:
            # Reset the timer if wheels are moving or motor is stopped
            if hasattr(self, '_wheel_stopped_start_time'):
                self.logger.debug("Wheels moving or motor stopped, resetting timer.")
                delattr(self, '_wheel_stopped_start_time')
                self._collision_detected = False
    
    def execute_reverse_maneuver(self):        
        avg_r, avg_g, ratio_r, ratio_g, detection_status, processing_results = extract_info(self.camera.get_camera_frame(), *self.camera.get_resolution())

        match (detection_status):
            case DetectionStatus.RIGHT_WALL_VISIBLE:
                self.logger.debug(f"Reverse Maneuver:  Green Ratio:{ratio_g:2.2f}; Red Ratio:{ratio_r:2.2f}")
                if  (ratio_g > 0.10 and TRACK_DIRECTION==0) or  (ratio_r > 0.10 and TRACK_DIRECTION!=0):
                    self.steer.set_steering_angle(30)
                    self.motor.set_speed(-1.5)
                    
                    
                    for _ in range(15):
                        ultrasonic_read = self.ultrasonic.get_ultrasonic_data()  
                        if (ultrasonic_read <= self.back_distance and ultrasonic_read != -1.0): 
                            break
                        time.sleep(0.1)

                    #TODO: check these values
                    self.motor.set_speed(0)
                    self.steer.set_steering_angle(-30)
                    self.motor.set_speed(0.7)
                    time.sleep(0.1)
                else:
                    self.execute_reversal()
            case DetectionStatus.LEFT_WALL_VISIBLE:
                self.logger.debug(f"Reverse Maneuver:  Green Ratio:{ratio_g:2.2f}; Red Ratio:{ratio_r:2.2f}")
                if  (ratio_r > 0.10 and TRACK_DIRECTION==0) or  (ratio_g > 0.10 and TRACK_DIRECTION!=0):
                    self.steer.set_steering_angle(-30)
                    self.motor.set_speed(-1.5)
                    #time.sleep(1.5)

                    for _ in range(15):
                        ultrasonic_read = self.ultrasonic.get_ultrasonic_data()  
                        if (ultrasonic_read <= self.back_distance and ultrasonic_read != -1.0): 
                            break
                        time.sleep(0.1)

                    #TODO: check these values
                    self.motor.set_speed(0)
                    self.steer.set_steering_angle(30)
                    self.motor.set_speed(0.7)
                    time.sleep(0.1)
                else:
                    self.logger.debug("Reverse Maneuver: Ratios too low, executing default reverse maneuver.")
                    self.execute_reversal()
            case _:
                self.execute_reversal()

#TODO: why we have this function and execute_reversal_maneuver? Case of studys
    def execute_reversal(self):
        self.steer.set_steering_angle(0)
        self.motor.set_speed(-1.2)
        #time.sleep(1.5)

        for _ in range(15):
            ultrasonic_read = self.ultrasonic.get_ultrasonic_data()  
            if (ultrasonic_read <= self.back_distance and ultrasonic_read != -1.0): 
                self.logger.debug("execute_reversal Completed")
                break
            time.sleep(0.1)

        self.motor.set_speed(0)
        self.motor.set_speed(0.7)
        self.logger.debug("[voltando] Completed reversing maneuver, resuming forward motion.")
    
    def reversing_direction(self):
        l_side = self.lidar.get_lidar_data()[60:120]   # Região à esquerda do carrinho
        r_side = self.lidar.get_lidar_data()[240:300]  # Região à direita do carrinho
                    
        avg_left = np.mean(l_side[l_side > 0])
        avg_right = np.mean(r_side[r_side > 0]) 

        if avg_left > avg_right:
            self.logger.debug(f"[reversing_direction] Found space on the left (left, right) = ({avg_left}, {avg_right}) cm.")
            
            self.steer.set_steering_angle(+30)
            self.motor.set_speed(-2.0)


            for _ in range(20):
                ultrasonic_read = self.ultrasonic.get_ultrasonic_data()  
                if (ultrasonic_read <= self.back_distance and ultrasonic_read != -1.0): 
                    break
                time.sleep(0.1)

            self.motor.set_speed(0)
            self.steer.set_steering_angle(-30)
            self.motor.set_speed(0.7)
            time.sleep(1.0)
            return
        else:
            
            self.steer.set_steering_angle(-30)
            self.motor.set_speed(-2.0)
            self.logger.debug(f"[reversing_direction] Found space on the right (right, left) = ({avg_right}, {avg_left}) cm.")

            for _ in range(20):
                
                self.logger.debug("Reversing... checking ultrasonic distance.")
                ultrasonic_read = self.ultrasonic.get_ultrasonic_data()  
                if (ultrasonic_read <= self.back_distance and ultrasonic_read != -1.0):
                    break
                time.sleep(0.1)

            self.logger.debug(f"Completed reversing maneuver, speed={self.motor.get_speed()}.")
            
            self.motor.set_speed(0)
            self.steer.set_steering_angle(+30)
            self.motor.set_speed(0.7)
            time.sleep(1.0)
            return
        
    def print_detection(self,detection, ratio_r, ratio_g):
        # TODO: STRATEGY: And if we create a more detailed rating system?
        match (detection):
            
            case DetectionStatus.LEFT_WALL_VISIBLE :
                self.simple_logger.debug(f"LEFT_WALL_VISIBLE  detected, ratios red, green: {ratio_r}, {ratio_g}")   
            case DetectionStatus.RIGHT_WALL_VISIBLE :
                self.simple_logger.debug(f"RIGHT_WALL_VISIBLEdetected, ratios red, green: {ratio_r}, {ratio_g}")  
            case DetectionStatus.WALLS_ON_CORRECT_SIDE:
                self.simple_logger.debug(f"WALLS_ON_CORRECT_SIDE detected, ratios red, green: {ratio_r}, {ratio_g}")   
            case DetectionStatus.WALLS_ON_WRONG_SIDE:
                self.simple_logger.debug(f"ONLY WALLS_ON_WRONG_SIDE detected, ratios red, green: {ratio_r}, {ratio_g}")
        return

    def maneuver_u_turn(self): 
        frame = self.camera.get_camera_frame()
        
        avg_r, avg_g, ratio_r, ratio_g, detection_status, processing_results = extract_info(frame, *self.camera.get_resolution()) 
        
        self.print_detection(detection_status, ratio_r, ratio_g)
        
        match (detection_status):
            case DetectionStatus.WALLS_ON_CORRECT_SIDE:
                return False
            case DetectionStatus.WALLS_ON_WRONG_SIDE:
                return True

        return False

    
    def check_proximity_to_wall(self):
        lidar_data = self.lidar.get_lidar_data()
        # TODO: STRATEGY, we need to define how much degree we consider "front"
        
        
        # Assuming lidar_data is a 360-degree array where indices 350-359 and 0-10 
        # represent the front of the vehicle (approximately 20 degrees field of view)
        front_indices = list(range(350, 360)) + list(range(0, 11))
        front_data = [lidar_data[i] for i in front_indices if lidar_data[i] > 0]  # Filter out zero/invalid readings
        
        # Calculate average distance in front if we have valid readings
        if len(front_data) > 0:
            dist_front_moyene = sum(front_data) / len(front_data)
        else:
            dist_front_moyene = float('inf')  # No valid readings means no obstacles detected
        
        # Print the front distance
        self.logger.debug(f"Front distance: {dist_front_moyene:.2f} cm")
        
        
        # Define minimum safe distance threshold (in same units as lidar data)
        min_front_lidar = 0.30  # 40 cm, adjust as needed
        
        # Check if we're too close to a wall and trigger reverse maneuver
        if dist_front_moyene < min_front_lidar:
            self.logger.debug(f"Too close to wall: {dist_front_moyene:.2f} cm, executing reverse maneuver.")
            self.execute_reversal()
        
    
    def run_step(self):
        """Runs a single step of the algorithm and measures execution time."""
        start_time = time.time()
        raw_lidar = self.lidar.get_lidar_data()
        ultrasonic_data = self.ultrasonic.get_ultrasonic_data()
        current_speed = self.speed.get_speed()
        battery_level = self.battery.get_battery_voltage()
        
        self.detect_wheel_stopped_collision()
        
        if self.maneuver_u_turn():
           self.logger.debug("[Run Step] Detected need for U-turn, executing reversing direction.")
        #    print("Reversed direction! reversing..")
           self.reversing_direction()

        print(raw_lidar)
        # input("Press ENTER to continue...\n")
        shrinked = shrink_space(raw_lidar)
        steer, target_angle = compute_steer_from_lidar(shrinked)
        target_speed = compute_speed(shrinked, target_angle)
        
        self.check_proximity_to_wall()
        
        self.steer.set_steering_angle(steer)
        self.motor.set_speed(target_speed)
        
        end_time = time.time()
        loop_time = end_time - start_time
        loop_time *= 1000000
        self.logger.debug(f"Loop time: {loop_time:.0f} us, Target Angle: {target_angle:.1f}, Current Speed: {current_speed:.2f} m/s, Ultrasonic Distance: {ultrasonic_data}, Battery Level: {battery_level}V")
        
        
        
    def cleanup(self):
        self.logger.debug("Cleaning up interfaces...")
        self.motor.stop()
        self.steer.stop()
        self.lidar.stop()
        self.logger.debug("All interfaces stopped and cleaned up.")
        
