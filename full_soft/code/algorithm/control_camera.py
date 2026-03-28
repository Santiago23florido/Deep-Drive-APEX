import cv2
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os

class Color(Enum):
    RED = "RED"
    GREEN = "GREEN"
    NONE = "Undefined"


class ColorDetectionStatus(Enum):
    RED_LEFT_GREEN_RIGHT = "RED TO THE LEFT AND GREEN TO THE RIGHT"
    GREEN_LEFT_RED_RIGHT = "GREEN TO THE LEFT AND RED TO THE RIGHT"
    ONLY_RED = "ONLY SEE RED"
    ONLY_GREEN = "ONLY SEE GREEN"
    NONE = "NO COLOR DETECTED"

class DetectionStatus(Enum):
    WALLS_ON_CORRECT_SIDE = "THE 2 WALLS APPEARING ARE ON THE CORRECT SIDE"
    WALLS_ON_WRONG_SIDE = "THE 2 WALLS APPEARING ARE ON THE WRONG SIDE"
    LEFT_WALL_VISIBLE = "THE ONLY WALL VISIBLE IS THE LEFT ONE"
    RIGHT_WALL_VISIBLE = "THE ONLY WALL VISIBLE IS THE RIGHT ONE"
    NONE = "NO COLOR DETECTED"

def color_to_detection_status(detection):
    if (TRACK_DIRECTION==0):
        match (detection):
            case ColorDetectionStatus.ONLY_RED:
                return DetectionStatus.LEFT_WALL_VISIBLE  
            case ColorDetectionStatus.ONLY_GREEN:
                return DetectionStatus.RIGHT_WALL_VISIBLE    
            case ColorDetectionStatus.RED_LEFT_GREEN_RIGHT:
                return DetectionStatus.WALLS_ON_CORRECT_SIDE  
            case ColorDetectionStatus.GREEN_LEFT_RED_RIGHT:
                return DetectionStatus.WALLS_ON_WRONG_SIDE   
    else:
        match (detection):
            case ColorDetectionStatus.ONLY_RED:
                return DetectionStatus.RIGHT_WALL_VISIBLE  
            case ColorDetectionStatus.ONLY_GREEN:
                return DetectionStatus.LEFT_WALL_VISIBLE    
            case ColorDetectionStatus.RED_LEFT_GREEN_RIGHT:
                return DetectionStatus.WALLS_ON_WRONG_SIDE  
            case ColorDetectionStatus.GREEN_LEFT_RED_RIGHT:
                return DetectionStatus.WALLS_ON_CORRECT_SIDE   
    return DetectionStatus.NONE

def convert_to_hsv(frame):
    if frame is None:
        return None
    
    try:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    except Exception as e:
        print(f"Error converting to HSV: {e}")
        return None


# From previous versions -- @TODO: include these in conf. file
red_brighter_lower, red_brighter_upper = np.array([0, 100, 100]), np.array([10, 255, 255])
red_darker_lower, red_darker_upper = np.array([160, 100, 100]), np.array([180, 255, 255])
green_lower, green_upper = np.array([30, 50, 50]), np.array([80, 255, 255])

# offcenter : parameter between 0 and 1 that determines how much should a color detected alone be offcenter, adjust this with tests 
offcenter= 0.2

def create_color_masks(frame_hsv, 
                       red_brighter:   tuple[np.ndarray, np.ndarray]=(red_brighter_lower, red_brighter_upper),
                       red_darker:     tuple[np.ndarray, np.ndarray]=(red_darker_lower, red_darker_upper),
                       green_range:   tuple[np.ndarray, np.ndarray]=(green_lower, green_upper)
                       ):
    
    """
    Create masks for red and green colors
    
    Args:
        frame_hsv: HSV frame
        
    Returns:
        tuple: (red_mask, green_mask) or (None, None) if frame is invalid
    """
    if frame_hsv is None:
        return None, None
    
    try:
        mask_r1 = cv2.inRange(frame_hsv, red_brighter[0], red_brighter[1])
        mask_r2 = cv2.inRange(frame_hsv, red_darker[0], red_darker[1])
        mask_g  = cv2.inRange(frame_hsv, green_range[0], green_range[1])
        mask_r = cv2.bitwise_or(mask_r1, mask_r2)

        
        # TODO: Add logic to swap masks based on the track orientation 
        return mask_r, mask_g
    except Exception as e:
        
        return None, None

def calculate_color_positions(mask_r, mask_g):
    if mask_r is None or mask_g is None:
        return -1, -1
    
    try:
        # Calculate average position of red pixels
        stack_r = np.column_stack(np.where(mask_r > 0))
        avg_r = np.mean(stack_r[:, 1]) if stack_r.size > 0 else -1

        # Calculate average position of green pixels
        stack_g = np.column_stack(np.where(mask_g > 0))
        avg_g = np.mean(stack_g[:, 1]) if stack_g.size > 0 else -1
        
        return avg_r, avg_g
    except Exception as e:
        print(f"Error calculating color positions: {e}")
        return -1, -1

def calculate_color_ratios(mask_r, mask_g, width, height):
    if mask_r is None or mask_g is None:
        return 0, 0
    
    try:
        ratio_r = 100*np.count_nonzero(mask_r) / (width * height)
        ratio_g = 100*np.count_nonzero(mask_g) / (width * height)
        print(f"Dados: {ratio_r:.1f}%, {ratio_g:.1f}%")
        return ratio_r, ratio_g
    except Exception as e:
        print(f"Error calculating color ratios: {e}")
        return 0, 0
      
def determine_detection_status(width : int, offcenter : float, 
                               avg_r : float, avg_g : float, 
                               ratio_r : float, ratio_g : float, 
                               min_ratio=12):
    center = width / 2
    margin = width * offcenter

    # Ratio: used to have a minimum threshold of detection, to avoid noise.
    red_detected = avg_r != -1 and ratio_r >= min_ratio
    green_detected = avg_g != -1 and ratio_g >= min_ratio

    def is_left(x): return x < center - margin
    def is_right(x): return x > center + margin


    # Both detected
    if red_detected and green_detected:
        if avg_r < avg_g:
            return color_to_detection_status(DetectionStatus.RED_LEFT_GREEN_RIGHT)
        else:
            return color_to_detection_status(DetectionStatus.GREEN_LEFT_RED_RIGHT)

    # Only red detected
    if red_detected:
        if is_left(avg_r):
            return color_to_detection_status(DetectionStatus.RED_LEFT_GREEN_RIGHT)
        elif is_right(avg_r):
            return color_to_detection_status(DetectionStatus.GREEN_LEFT_RED_RIGHT)
        return color_to_detection_status(DetectionStatus.ONLY_RED)

    # Only green detected
    if green_detected:
        if is_right(avg_g):
            return color_to_detection_status(DetectionStatus.RED_LEFT_GREEN_RIGHT)
        elif is_left(avg_g):
            return color_to_detection_status(DetectionStatus.GREEN_LEFT_RED_RIGHT)
        return color_to_detection_status(DetectionStatus.ONLY_GREEN)


    # Nothing detected
    return DetectionStatus.NONE

def create_overlay_visualization(frame, mask_r, mask_g, avg_r, avg_g, status):
    if frame is None or mask_r is None or mask_g is None:
        return None

    try:
        vis_frame = frame.copy()

        # Criar overlays diretamente
        red_overlay = np.zeros_like(vis_frame)
        green_overlay = np.zeros_like(vis_frame)

        # Aplicar máscaras (muito mais simples)
        red_overlay[mask_r > 0] = (0, 0, 255)     # BGR
        green_overlay[mask_g > 0] = (0, 255, 0)

        # Combinar overlays
        vis_frame = cv2.addWeighted(vis_frame, 1, red_overlay, 0.3, 0)
        vis_frame = cv2.addWeighted(vis_frame, 1, green_overlay, 0.3, 0)

        # Linhas verticais
        h = vis_frame.shape[0]

        if avg_r != -1:
            cv2.line(vis_frame, (int(avg_r), 0), (int(avg_r), h), (0, 0, 255), 2)

        if avg_g != -1:
            cv2.line(vis_frame, (int(avg_g), 0), (int(avg_g), h), (0, 255, 0), 2)

        # Texto
        cv2.putText(
            vis_frame,
            status.value,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return vis_frame

    except Exception as e:
        print(f"Error creating visualization: {e}")
        return frame

def extract_info(frame, width, height):
    """
    Process frame and extract color information
    
    Args:
        frame: RGB frame
        width: Frame width
        height: Frame height
        
    Returns:
        tuple: (avg_r, avg_g, ratio_r, ratio_g, detection_status, processing_results)
    """
    if frame is None:
        return -1, -1, 0, 0, DetectionStatus.NONE
    
    try:
        # Convert frame to HSV
        frame_hsv = convert_to_hsv(frame)
        if frame_hsv is None:
            return -1, -1, 0, 0, DetectionStatus.NONE
        
        # Create color masks
        mask_r, mask_g = create_color_masks(frame_hsv)
        if mask_r is None or mask_g is None:
            return -1, -1, 0, 0, DetectionStatus.NONE
        
        # Calculate color positions
        avg_r, avg_g = calculate_color_positions(mask_r, mask_g)
        
        # Calculate color ratios
        ratio_r, ratio_g = calculate_color_ratios(mask_r, mask_g, width, height)
        
        # Determine detection status
        detection_status = determine_detection_status(width, offcenter, avg_r, avg_g, ratio_r, ratio_g)
        
        # Compile processing results for visualization
        processing_results = {
            'frame_hsv': frame_hsv,
            'mask_r': mask_r,
            'mask_g': mask_g,
            'avg_r': avg_r,
            'avg_g': avg_g,
            'ratio_r': ratio_r,
            'ratio_g': ratio_g,
            'status': detection_status
        }
        
        return avg_r, avg_g, ratio_r, ratio_g, detection_status, processing_results
    
    except Exception as e:
        print(f"Error processing camera stream: {e}")
        return -1, -1, 0, 0, DetectionStatus.NONE
