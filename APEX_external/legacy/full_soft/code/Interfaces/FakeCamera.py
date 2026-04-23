import numpy as np
import cv2


class FakeCamera:
    """
    Webcam-backed stand-in for Picamera2-like code.

    Provides a small subset of the Picamera2 interface:
    - create_preview_configuration(main=..., lores=...)
    - configure(config)
    - start()
    - capture_array()
    - close()
    """

    def __init__(self, device=0, width=640, height=480, fps=None, fallback_test_pattern=True):
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = fps
        self.fallback_test_pattern = fallback_test_pattern

        self._cap = None
        self._pending_config = {}

        # Default fallback frame (black with white rectangle)
        self._fallback_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._fallback_frame[100:380, 150:490] = [255, 255, 255]

    def create_preview_configuration(self, main=None, lores=None):
        """
        Accepts Picamera2-like arguments but returns a simple dict we can use later.

        Examples:
          main={"size": (640, 480)}
          lores={"size": (320, 240)}
        """
        config = {"main": main, "lores": lores}

        size = None
        if isinstance(main, dict):
            size = main.get("size")
        elif isinstance(main, (tuple, list)) and len(main) == 2:
            size = tuple(main)

        if size:
            config["size"] = tuple(map(int, size))
        else:
            config["size"] = (self.width, self.height)

        return config

    def configure(self, config):
        """Store configuration; applied when the camera is started."""
        if not isinstance(config, dict):
            raise TypeError("config must be a dict returned by create_preview_configuration()")

        self._pending_config = dict(config)
        size = self._pending_config.get("size")
        if size and len(size) == 2:
            self.width, self.height = int(size[0]), int(size[1])
            self._fallback_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            y0, y1 = max(0, self.height // 5), min(self.height, self.height * 4 // 5)
            x0, x1 = max(0, self.width // 5), min(self.width, self.width * 4 // 5)
            self._fallback_frame[y0:y1, x0:x1] = [255, 255, 255]

    def start(self):
        """Open the webcam and apply resolution/fps settings if possible."""
        if self._cap is not None:
            return

        self._cap = cv2.VideoCapture(self.device)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            if not self.fallback_test_pattern:
                raise RuntimeError(f"Could not open webcam device {self.device}")
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        if self.fps is not None:
            self._cap.set(cv2.CAP_PROP_FPS, float(self.fps))

    def capture_array(self):
        """
        Capture and return the current frame as a numpy array (RGB).
        If the webcam isn't available and fallback_test_pattern is enabled,
        returns a generated test pattern frame.
        """
        if self._cap is None:
            self.start()

        if self._cap is None:
            return self._fallback_frame.copy()

        ok, frame_bgr = self._cap.read()
        if not ok or frame_bgr is None:
            if self.fallback_test_pattern:
                return self._fallback_frame.copy()
            raise RuntimeError("Failed to read frame from webcam")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def close(self):
        """Release the webcam."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
    
    
    
    
def main():
    cam = FakeCamera(device=0, width=640, height=480, fps=30, fallback_test_pattern=True)
    # config = cam.create_preview_configuration(main={"size": (640, 480)})
    # cam.configure(config)

    try:
        cam.start()
        print("Press 'q' or ESC to quit.")
        while True:
            frame_rgb = cam.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow("FakeCamera test", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()