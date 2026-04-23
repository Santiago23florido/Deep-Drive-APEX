"""
interface_camera.py — Interface de câmara para Raspberry Pi (Picamera2).
"""

import numpy as np
import cv2
# from picamera2 import Picamera2
from typing import Tuple

from code.Interfaces.FakeCamera import FakeCamera
from code.Interfaces.interface_decl import CameraInterface
from code.config_loader import get_config
# BUG FIX: o original importava 'algorithm.voiture_logger' que não existe
# no projeto refatorado. Substituído pelo log_manager central.
from code.log_manager import get_logger




class RealCameraInterface(CameraInterface):

    def __init__(self):
        self.logger = get_logger("RealCameraInterface")

        config = get_config()
        cam_cfg = config.camera

        self.width = cam_cfg.width
        self.height = cam_cfg.height
        self.rotation = getattr(cam_cfg, "rotation", 0)

        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            self._is_fake = False

            cam_config = self.picam2.create_preview_configuration(
                main={"size": (self.width, self.height)},
                lores={"size": (self.width, self.height)},
            )
            self.picam2.configure(cam_config)
            self.picam2.start()
            self.logger.info("Real camera initialized")

        except (ImportError, Exception):
            self.logger.warning("Picamera2 not available, using FakeCamera")
            self.picam2 = FakeCamera(width=self.width, height=self.height, fallback_test_pattern=True)
            self._is_fake = True

            # mesmo fluxo para FakeCamera
            cam_config = self.picam2.create_preview_configuration(
                main={"size": (self.width, self.height)}
            )
            self.picam2.configure(cam_config)
            self.picam2.start()  

    
    def start(self):
        try:
            self.picam2.start()
            self.logger.info("Camera started successfully")
        except Exception as e:
            self.logger.error(f"Error starting camera: {e}")

    def get_camera_frame(self) -> np.ndarray:
        """Captures and returns a camera frame (BGR numpy array)."""
        try:
            frame = self.picam2.capture_array()
            if frame is None:
                self.logger.warning("Camera frame could not be captured")
                return None

            if self.rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            return frame

        except Exception as e:
            self.logger.error(f"Error capturing camera frame: {e}")
            return None

    def get_resolution(self) -> Tuple[int, int]:
        """Devolve (width, height)."""
        return self.width, self.height

    def cleanup(self):
        try:
            self.picam2.close()
            self.logger.info("Camera resources cleaned up")
        except Exception as e:
            self.logger.error(f"Error cleaning up camera resources: {e}")


if __name__ == "__main__":
    try:
        print("Starting camera debug interface...")
        camera = RealCameraInterface()
        camera.start()
        while True:
            frame = camera.get_camera_frame()
            if frame is not None:
                cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    except Exception as e:
        print(f"Error starting camera: {e}")
    finally:
        try:
            camera.cleanup()
            cv2.destroyAllWindows()
            print("Camera resources cleaned up")
        except Exception:
            pass
        
        