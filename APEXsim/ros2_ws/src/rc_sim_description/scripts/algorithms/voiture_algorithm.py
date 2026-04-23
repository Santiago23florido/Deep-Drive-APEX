import time
import cv2
import os
import datetime
import numpy as np
try:
    from algorithm.interfaces import *  # type: ignore  # noqa: F403
except Exception:
    from interfaces import *  # type: ignore  # noqa: F403
try:
    from algorithm.constants import HITBOX_H1, HITBOX_H2, HITBOX_W  # type: ignore
except Exception:
    from constants import HITBOX_H1, HITBOX_H2, HITBOX_W  # type: ignore
try:
    from algorithm.control_camera import extract_info, DetectionStatus  # type: ignore
except Exception:
    from control_camera import extract_info, DetectionStatus  # type: ignore
try:
    from algorithm.control_direction import compute_steer_from_lidar, shrink_space  # type: ignore
except Exception:
    from control_direction import compute_steer_from_lidar, shrink_space  # type: ignore
try:
    from algorithm.control_speed import compute_speed  # type: ignore
except Exception:
    from control_speed import compute_speed  # type: ignore

back_dist = 15

import logging


class VoitureAlgorithmCore:
    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

    def compute(
        self,
        lidar_360: np.ndarray,
        measured_wheelspeed: float | None = None,
    ) -> tuple[float, float]:
        if lidar_360 is None:
            self.logger.warning("LiDAR data is None; returning zero commands.")
            return 0.0, 0.0

        lidar = np.asarray(lidar_360, dtype=float).reshape(-1)
        if lidar.shape[0] != 360:
            self.logger.error(
                f"Expected LiDAR length 360, got {lidar.shape[0]}; returning zero commands."
            )
            return 0.0, 0.0

        invalid_mask = ~np.isfinite(lidar) | (lidar <= 0.0)
        if np.any(invalid_mask):
            lidar = lidar.copy()
            lidar[invalid_mask] = 0.0

        shrinked = shrink_space(lidar)
        steer, target_angle = compute_steer_from_lidar(shrinked)
        target_speed = compute_speed(shrinked, target_angle)

        _ = measured_wheelspeed
        return float(steer), float(target_speed)



class VoitureAlgorithm:
    def __init__(self, 
                 lidar: LiDarInterface, 
                 ultrasonic: UltrasonicInterface, 
                 speed: SpeedInterface, 
                 battery: BatteryInterface, 
                 camera: CameraInterface, 
                 steer: SteerInterface, 
                 motor: MotorInterface,
                 console: ConsoleInterface):
        
        self.simple_logger = logging.getLogger(__name__)
         
        # Ensure all inputs implement the expected interfaces
        if not isinstance(lidar, LiDarInterface):
            self.simple_logger.error("Lidar interface FAILED")    
            raise TypeError("lidar must implement LiDarInterface")
        if not isinstance(ultrasonic, UltrasonicInterface):
            self.simple_logger.error("Ultrasonic interface FAILED")
            raise TypeError("ultrasonic must implement UltrasonicInterface")
        if not isinstance(speed, SpeedInterface):
            self.simple_logger.error("Speed interface FAILED")
            raise TypeError("speed must implement SpeedInterface")
        if not isinstance(battery, BatteryInterface):
            self.simple_logger.error("Battery interface FAILED")
            raise TypeError("battery must implement BatteryInterface")
        if not isinstance(camera, CameraInterface):
            self.simple_logger.error("Camera interface FAILED")
            raise TypeError("camera must implement CameraInterface")
        if not isinstance(steer, SteerInterface):
            self.simple_logger.error("Steer interface FAILED")
            raise TypeError("steer must implement SteerInterface")
        if not isinstance(motor, MotorInterface):
            self.simple_logger.error("Motor interface FAILED")
            raise TypeError("motor must implement MotorInterface")
        if not isinstance(console, ConsoleInterface):
            self.simple_logger.error("Console interface FAILED")
            raise TypeError("console must be an instance of ConsoleInterface")
        
        self.lidar = lidar
        self.ultrasonic = ultrasonic
        self.speed = speed
        self.battery = battery
        self.camera = camera
        self.steer = steer
        self.motor = motor
        self.console = console
        
       
        

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
        
        self.simple_logger.debug(f"Current Speed: {current_speed}, Motor Speed: {motor_speed}")
        
        # Check if wheels are stopped but motor is running
        if current_speed < STOPPED_THRESHOLD and motor_speed > 0:
            
            # If this is the first detection of stoppage, record the time
            if not hasattr(self, '_wheel_stopped_start_time'):
                self.simple_logger.debug("Wheels stopped detected, starting timer.")
                self._wheel_stopped_start_time = current_time
                self._collision_detected = False
            # If wheels have been stopped for longer than the threshold
            elif (current_time - self._wheel_stopped_start_time) > COLLISION_TIME_THRESHOLD and not self._collision_detected:
                self.simple_logger.debug(f"[detect_wheel_stopped_collision] Collision detected due to wheel stoppage. Current Speed: {current_speed}, Motor Speed: {motor_speed}")
                self._collision_detected = True
                self.console.print_to_console(f"&c&l[COLLISION DETECTED] &e- Wheels stopped for &f{COLLISION_TIME_THRESHOLD*1000:.0f}ms &ewhile motor running")
                self.simple_marche_arrire()
        else:
            # Reset the timer if wheels are moving or motor is stopped
            if hasattr(self, '_wheel_stopped_start_time'):
                self.simple_logger.debug("Wheels moving or motor stopped, resetting timer.")
                delattr(self, '_wheel_stopped_start_time')
                self._collision_detected = False
    
    def simple_marche_arrire(self):        
        avg_r, avg_g, ratio_r, ratio_g, detection_status, processing_results = extract_info(self.camera.get_camera_frame(), *self.camera.get_resolution())

        match (detection_status):
            case DetectionStatus.ONLY_GREEN:
                self.simple_logger.debug("[simple_marche_arrire]ONLY GREEN detected, executing maneuver.")
                if  ratio_g > 0.10:
                    self.steer.set_steering_angle(30)
                    self.motor.set_speed(-1.5)
                    #time.sleep(1.5)
                    
                    for _ in range(15):
                        ultrasonic_read = self.ultrasonic.get_ultrasonic_data()  
                        if (ultrasonic_read <= back_dist and ultrasonic_read != -1.0): 
                            break
                        time.sleep(0.1)

                    self.motor.set_speed(0)
                    self.steer.set_steering_angle(-30)
                    self.motor.set_speed(0.7)
                    time.sleep(0.1)
                    self.simple_logger.debug("[simple_marche_arrire] Completed maneuver after ONLY GREEN detection.")
                else:
                    self.voltando()
            case DetectionStatus.ONLY_RED:
                self.simple_logger.debug("[simple_marche_arrire] ONLY RED detected, executing maneuver.")
                if  ratio_r > 0.10:
                    self.steer.set_steering_angle(-30)
                    self.motor.set_speed(-1.5)
                    #time.sleep(1.5)

                    for _ in range(15):
                        ultrasonic_read = self.ultrasonic.get_ultrasonic_data()  
                        if (ultrasonic_read <= back_dist and ultrasonic_read != -1.0): 
                            break
                        time.sleep(0.1)

                    self.motor.set_speed(0)
                    self.steer.set_steering_angle(30)
                    self.motor.set_speed(0.7)
                    time.sleep(0.1)
                    self.simple_logger.debug("[simple_marche_arrire] Completed maneuver after ONLY RED detection.")
                else:
                    self.simple_logger.debug("[simple_marche_arrire] Ratios too low, executing default reverse maneuver.")
                    self.voltando()
            case _:
                self.voltando()

    def voltando(self):
        self.steer.set_steering_angle(0)
        self.motor.set_speed(-1.2)
        #time.sleep(1.5)

        for _ in range(15):
            ultrasonic_read = self.ultrasonic.get_ultrasonic_data()  
            if (ultrasonic_read <= back_dist and ultrasonic_read != -1.0): 
                self.simple_logger.debug("[voltando] Completed reversing maneuver.")
                break
            time.sleep(0.1)

        self.motor.set_speed(0)
        self.motor.set_speed(0.7)
        self.simple_logger.debug("[voltando] Completed reversing maneuver, resuming forward motion.")
    
    def reversing_direction(self):
        l_side = self.lidar.get_lidar_data()[60:120]   # Região à esquerda do carrinho
        r_side = self.lidar.get_lidar_data()[240:300]  # Região à direita do carrinho
                    
        avg_left = np.mean(l_side[l_side > 0])
        avg_right = np.mean(r_side[r_side > 0]) 

        if avg_left > avg_right:
            self.simple_logger.debug(f"[reversing_direction] Found space on the left (left, right) = ({avg_left}, {avg_right}) cm.")
            
            self.steer.set_steering_angle(+30)
            self.motor.set_speed(-2.0)


            for _ in range(20):
                ultrasonic_read = self.ultrasonic.get_ultrasonic_data()  
                if (ultrasonic_read <= back_dist and ultrasonic_read != -1.0): 
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
            self.simple_logger.debug(f"[reversing_direction] Found space on the right (right, left) = ({avg_right}, {avg_left}) cm.")

            for _ in range(20):
                
                self.simple_logger.debug("Reversing... checking ultrasonic distance.")
                ultrasonic_read = self.ultrasonic.get_ultrasonic_data()  
                if (ultrasonic_read <= back_dist and ultrasonic_read != -1.0):
                    break
                time.sleep(0.1)

            self.simple_logger.debug(f"Completed reversing maneuver, speed={self.motor.get_speed()}.")
            
            self.motor.set_speed(0)
            self.steer.set_steering_angle(+30)
            self.motor.set_speed(0.7)
            time.sleep(1.0)
            return
        
    def print_detection(self,detection, ratio_r, ratio_g):
        # TODO: STRATEGY: And if we create a more detailed rating system?
        match (detection):
            
            case DetectionStatus.ONLY_RED:
                self.simple_logger.debug(f"ONLY RED detected, ratios red, green: {ratio_r}, {ratio_g}")
                # self.console.print_to_console(f"&4&lo &4&lo &4&lo &4&lo &4&lo &4&lo {ratio_r}, {ratio_g}")    
            case DetectionStatus.ONLY_GREEN:
                self.simple_logger.debug(f"ONLY GREEN detected, ratios red, green: {ratio_r}, {ratio_g}")
                # self.console.print_to_console(f"&2&lo &2&lo &2&lo &2&lo &2&lo &2&lo {ratio_r}, {ratio_g}")    
            case DetectionStatus.RED_LEFT_GREEN_RIGHT:
                self.simple_logger.debug(f"ONLY RED_LEFT_GREEN_RIGHT detected, ratios red, green: {ratio_r}, {ratio_g}")
                # self.console.print_to_console(f"&4&lo &4&lo &4&lo &2&lo &2&lo &2&lo {ratio_r}, {ratio_g}")    
            case DetectionStatus.GREEN_LEFT_RED_RIGHT:
                self.simple_logger.debug(f"ONLY GREEN_LEFT_RED_RIGHT detected, ratios red, green: {ratio_r}, {ratio_g}")
                # self.console.print_to_console(f"&2&lo &2&lo &2&lo &4&lo &4&lo &4&lo {ratio_r}, {ratio_g}")    
        return

    def demi_tour(self): 
        frame = self.camera.get_camera_frame()
        
        avg_r, avg_g, ratio_r, ratio_g, detection_status, processing_results = extract_info(frame, *self.camera.get_resolution()) 
        
        self.print_detection(detection_status, ratio_r, ratio_g)
        
        match (detection_status):
            case DetectionStatus.RED_LEFT_GREEN_RIGHT:
                return False
            case DetectionStatus.GREEN_LEFT_RED_RIGHT:
                return True

        return False

    
    def check_too_close_to_mur(self):
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
        self.simple_logger.debug(f"Front distance: {dist_front_moyene:.2f} cm")
        
        self.console.print_to_console(f"&e&lDistance frontale: &f{dist_front_moyene:.2f} cm")
        
        # Define minimum safe distance threshold (in same units as lidar data)
        min_front_lidar = 0.30  # 40 cm, adjust as needed
        
        # Check if we're too close to a wall and trigger reverse maneuver
        if dist_front_moyene < min_front_lidar:
            self.simple_logger.debug(f"Too close to wall: {dist_front_moyene:.2f} cm, executing reverse maneuver.")
            self.console.print_to_console(f"&c&l[WARNING] &eTrop proche du mur: &f{dist_front_moyene:.2f} cm")
            self.voltando()
        
    
    def run_step(self):
        """Runs a single step of the algorithm and measures execution time."""
        start_time = time.time()
        raw_lidar = self.lidar.get_lidar_data()
        ultrasonic_data = self.ultrasonic.get_ultrasonic_data()
        current_speed = self.speed.get_speed()
        battery_level = self.battery.get_battery_voltage()
        
        self.detect_wheel_stopped_collision()
        
        if self.demi_tour():
           self.simple_logger.debug("[Run Step] Detected need for U-turn, executing reversing direction.")
        #    print("Reversed direction! reversing..")
           self.reversing_direction()

        shrinked = shrink_space(raw_lidar)
        steer, target_angle = compute_steer_from_lidar(shrinked)
        target_speed = compute_speed(shrinked, target_angle)
        
        self.check_too_close_to_mur()
        
        self.steer.set_steering_angle(steer)
        self.motor.set_speed(target_speed)
        
        end_time = time.time()
        loop_time = end_time - start_time
        loop_time *= 1000000
        self.simple_logger.debug(f"Loop time: {loop_time:.0f} us, Target Angle: {target_angle:.1f}, Current Speed: {current_speed:.2f} m/s, Ultrasonic Distance: {ultrasonic_data}, Battery Level: {battery_level}V")
        self.console.print_to_console(f"&b&lAngle: &f{target_angle:.1f}\t&a&lVelocity: &f{self.motor.get_speed()} &6&lSPD: &f{current_speed:.2f} m/s Dist: {self.ultrasonic.get_ultrasonic_data()} &e&lBAT: &f{battery_level}V &d&lLoop: &f{loop_time:.0f} us")    
