#!/usr/bin/env python3
import argparse
import json
import os
import sys
import cv2
import numpy as np
import time

class CameraRealtime:
    def __init__(self, config_path: str, camera_id: int = 0):
        self.config_path = config_path
        self.camera_id = camera_id
        self.last_mtime = 0

        # Chargement initial
        self.config_data = self._load_config()
        self.cam_params = self.config_data.get("CAMERA", {})

        # Initialisation de la caméra
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Impossible d'ouvrir la caméra")

        self._apply_camera_params(self.cam_params)

        self.running = True

    def _load_config(self) -> dict:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Erreur lecture config: {e}", file=sys.stderr)
            return {}

    def _apply_camera_params(self, cam_params: dict):
        """Applique les paramètres (résolution, seuils)"""
        self.width = cam_params.get("width", 160)
        self.height = cam_params.get("height", 120)
        self.rotation = cam_params.get("rotation", 180)  # degrés
        self.offcenter = cam_params.get("offcenter", 0.2)
        self.min_detection_ratio = cam_params.get("min_detection_ratio", 12.0)

        # Seuils HSV
        self.red_brighter_lower = np.array(cam_params.get("red_brighter_lower", [0, 100, 100]))
        self.red_brighter_upper = np.array(cam_params.get("red_brighter_upper", [10, 255, 255]))
        self.red_darker_lower = np.array(cam_params.get("red_darker_lower", [160, 100, 100]))
        self.red_darker_upper = np.array(cam_params.get("red_darker_upper", [180, 255, 255]))
        self.green_lower = np.array(cam_params.get("green_lower", [30, 50, 50]))
        self.green_upper = np.array(cam_params.get("green_upper", [80, 255, 255]))

        # Redimensionner la caméra si possible
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        print(f"Paramètres appliqués: {self.width}x{self.height}, rotation={self.rotation}")

    def check_config_update(self):
        try:
            mtime = os.path.getmtime(self.config_path)
            if mtime != self.last_mtime:
                self.last_mtime = mtime
                new_config = self._load_config()
                cam = new_config.get("CAMERA", {})
                self._apply_camera_params(cam)
                print("Configuration mise à jour")
        except Exception as e:
            print(f"Erreur vérification config: {e}")

    def rotate_image(self, image):
        """Applique la rotation (en degrés) à l'image."""
        if self.rotation == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif self.rotation == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return image

    def process_frame(self, frame):
        """Applique les seuils de couleur et retourne les masques."""
        # Conversion en HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Masques
        mask_red_bright = cv2.inRange(hsv, self.red_brighter_lower, self.red_brighter_upper)
        mask_red_dark = cv2.inRange(hsv, self.red_darker_lower, self.red_darker_upper)
        mask_red = cv2.bitwise_or(mask_red_bright, mask_red_dark)
        mask_green = cv2.inRange(hsv, self.green_lower, self.green_upper)

        return mask_red, mask_green

    def run(self):
        print("Démarrage de la visualisation caméra. Appuyez sur 'q' pour quitter.")
        while self.running:
            self.check_config_update()

            ret, frame = self.cap.read()
            if not ret:
                print("Erreur de lecture frame")
                break

            # Redimensionnement (au cas où la caméra n'a pas appliqué la résolution)
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            # Rotation
            frame = self.rotate_image(frame)

            # Obtention des masques
            mask_red, mask_green = self.process_frame(frame)

            # Affichage
            cv2.imshow("Camera - Original", frame)
            cv2.imshow("Mask - Red", mask_red)
            cv2.imshow("Mask - Green", mask_green)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("Caméra libérée")

def main(argv):
    parser = argparse.ArgumentParser(description="Test caméra avec mise à jour en direct des paramètres.")
    parser.add_argument("--config", type=str, default="new-config.json",
                        help="Chemin vers le fichier de configuration")
    parser.add_argument("--camera", type=int, default=0,
                        help="ID de la caméra (défaut: 0)")
    args = parser.parse_args(argv)

    if not os.path.exists(args.config):
        print(f"Fichier introuvable : {args.config}", file=sys.stderr)
        return 1

    try:
        app = CameraRealtime(config_path=args.config, camera_id=args.camera)
        app.run()
    except Exception as e:
        print(f"Erreur: {e}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))