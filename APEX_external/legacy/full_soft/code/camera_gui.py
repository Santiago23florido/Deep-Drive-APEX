#!/usr/bin/env python3
import argparse
import json
import os
import sys
import tkinter as tk
from typing import Any, Dict, List

class CameraConfigApp:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._debounce_after_id = None

        # Chargement initial
        self.config_data = self._load_config()
        camera_params = self.config_data.get("CAMERA", {})

        self.root = tk.Tk()
        self.root.title("Configuration caméra en direct")

        # Variables pour les paramètres généraux
        self.width_var = tk.IntVar(master=self.root, value=camera_params.get("width", 160))
        self.height_var = tk.IntVar(master=self.root, value=camera_params.get("height", 120))
        self.rotation_var = tk.IntVar(master=self.root, value=camera_params.get("rotation", 180))
        self.offcenter_var = tk.DoubleVar(master=self.root, value=camera_params.get("offcenter", 0.2))
        self.min_detection_ratio_var = tk.DoubleVar(master=self.root, value=camera_params.get("min_detection_ratio", 12.0))

        # Variables pour les seuils de couleur (HSV)
        # Rouge clair
        self.red_brighter_lower_h = tk.IntVar(master=self.root, value=camera_params.get("red_brighter_lower", [0,100,100])[0])
        self.red_brighter_lower_s = tk.IntVar(master=self.root, value=camera_params.get("red_brighter_lower", [0,100,100])[1])
        self.red_brighter_lower_v = tk.IntVar(master=self.root, value=camera_params.get("red_brighter_lower", [0,100,100])[2])
        self.red_brighter_upper_h = tk.IntVar(master=self.root, value=camera_params.get("red_brighter_upper", [10,255,255])[0])
        self.red_brighter_upper_s = tk.IntVar(master=self.root, value=camera_params.get("red_brighter_upper", [10,255,255])[1])
        self.red_brighter_upper_v = tk.IntVar(master=self.root, value=camera_params.get("red_brighter_upper", [10,255,255])[2])

        # Rouge foncé
        self.red_darker_lower_h = tk.IntVar(master=self.root, value=camera_params.get("red_darker_lower", [160,100,100])[0])
        self.red_darker_lower_s = tk.IntVar(master=self.root, value=camera_params.get("red_darker_lower", [160,100,100])[1])
        self.red_darker_lower_v = tk.IntVar(master=self.root, value=camera_params.get("red_darker_lower", [160,100,100])[2])
        self.red_darker_upper_h = tk.IntVar(master=self.root, value=camera_params.get("red_darker_upper", [180,255,255])[0])
        self.red_darker_upper_s = tk.IntVar(master=self.root, value=camera_params.get("red_darker_upper", [180,255,255])[1])
        self.red_darker_upper_v = tk.IntVar(master=self.root, value=camera_params.get("red_darker_upper", [180,255,255])[2])

        # Vert
        self.green_lower_h = tk.IntVar(master=self.root, value=camera_params.get("green_lower", [30,50,50])[0])
        self.green_lower_s = tk.IntVar(master=self.root, value=camera_params.get("green_lower", [30,50,50])[1])
        self.green_lower_v = tk.IntVar(master=self.root, value=camera_params.get("green_lower", [30,50,50])[2])
        self.green_upper_h = tk.IntVar(master=self.root, value=camera_params.get("green_upper", [80,255,255])[0])
        self.green_upper_s = tk.IntVar(master=self.root, value=camera_params.get("green_upper", [80,255,255])[1])
        self.green_upper_v = tk.IntVar(master=self.root, value=camera_params.get("green_upper", [80,255,255])[2])

        # Interface
        frm = tk.Frame(self.root, padx=12, pady=12)
        frm.pack(fill="both", expand=True)

        tk.Label(frm, text="Ajustez les paramètres de la caméra en temps réel", font=("Arial", 12, "bold")).pack(anchor="w")
        tk.Label(frm, text="Les changements sont appliqués immédiatement au fichier de configuration.", fg="gray").pack(anchor="w", pady=(0,10))

        # Cadre pour les paramètres généraux
        gen_frame = tk.LabelFrame(frm, text="Paramètres généraux", padx=5, pady=5)
        gen_frame.pack(fill="x", pady=5)
        self._add_slider(gen_frame, "Largeur", self.width_var, 80, 320, 10)
        self._add_slider(gen_frame, "Hauteur", self.height_var, 60, 240, 10)
        self._add_slider(gen_frame, "Rotation (deg)", self.rotation_var, 0, 270, 90)
        self._add_slider(gen_frame, "Offcenter", self.offcenter_var, -1.0, 1.0, 0.05)
        self._add_slider(gen_frame, "Min detection ratio", self.min_detection_ratio_var, 0.0, 50.0, 0.5)

        # Cadre pour les seuils de couleur
        color_frame = tk.LabelFrame(frm, text="Seuils de couleur HSV", padx=5, pady=5)
        color_frame.pack(fill="x", pady=5)

        # Rouge clair
        redb_frame = tk.LabelFrame(color_frame, text="Rouge clair", padx=5, pady=5)
        redb_frame.pack(fill="x", pady=2)
        self._add_slider(redb_frame, "H min", self.red_brighter_lower_h, 0, 180, 1)
        self._add_slider(redb_frame, "S min", self.red_brighter_lower_s, 0, 255, 1)
        self._add_slider(redb_frame, "V min", self.red_brighter_lower_v, 0, 255, 1)
        self._add_slider(redb_frame, "H max", self.red_brighter_upper_h, 0, 180, 1)
        self._add_slider(redb_frame, "S max", self.red_brighter_upper_s, 0, 255, 1)
        self._add_slider(redb_frame, "V max", self.red_brighter_upper_v, 0, 255, 1)

        # Rouge foncé
        redd_frame = tk.LabelFrame(color_frame, text="Rouge foncé", padx=5, pady=5)
        redd_frame.pack(fill="x", pady=2)
        self._add_slider(redd_frame, "H min", self.red_darker_lower_h, 0, 180, 1)
        self._add_slider(redd_frame, "S min", self.red_darker_lower_s, 0, 255, 1)
        self._add_slider(redd_frame, "V min", self.red_darker_lower_v, 0, 255, 1)
        self._add_slider(redd_frame, "H max", self.red_darker_upper_h, 0, 180, 1)
        self._add_slider(redd_frame, "S max", self.red_darker_upper_s, 0, 255, 1)
        self._add_slider(redd_frame, "V max", self.red_darker_upper_v, 0, 255, 1)

        # Vert
        green_frame = tk.LabelFrame(color_frame, text="Vert", padx=5, pady=5)
        green_frame.pack(fill="x", pady=2)
        self._add_slider(green_frame, "H min", self.green_lower_h, 0, 180, 1)
        self._add_slider(green_frame, "S min", self.green_lower_s, 0, 255, 1)
        self._add_slider(green_frame, "V min", self.green_lower_v, 0, 255, 1)
        self._add_slider(green_frame, "H max", self.green_upper_h, 0, 180, 1)
        self._add_slider(green_frame, "S max", self.green_upper_s, 0, 255, 1)
        self._add_slider(green_frame, "V max", self.green_upper_v, 0, 255, 1)

        # Boutons
        btn_row = tk.Frame(frm)
        btn_row.pack(anchor="w", pady=(10,0))
        tk.Button(btn_row, text="Recharger depuis le fichier", command=self._reload_from_file).pack(side="left")
        tk.Button(btn_row, text="Quitter", command=self._on_close).pack(side="left", padx=(8,0))

        self.status = tk.Label(frm, text="Prêt", justify="left")
        self.status.pack(anchor="w", pady=(10,0))

        # Traces pour mise à jour automatique (toutes les variables)
        for var in [self.width_var, self.height_var, self.rotation_var, self.offcenter_var, self.min_detection_ratio_var,
                    self.red_brighter_lower_h, self.red_brighter_lower_s, self.red_brighter_lower_v,
                    self.red_brighter_upper_h, self.red_brighter_upper_s, self.red_brighter_upper_v,
                    self.red_darker_lower_h, self.red_darker_lower_s, self.red_darker_lower_v,
                    self.red_darker_upper_h, self.red_darker_upper_s, self.red_darker_upper_v,
                    self.green_lower_h, self.green_lower_s, self.green_lower_v,
                    self.green_upper_h, self.green_upper_s, self.green_upper_v]:
            var.trace_add("write", lambda *_: self._schedule_write())

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _add_slider(self, parent, label, variable, from_, to, resolution):
        frame = tk.Frame(parent)
        frame.pack(fill="x", pady=2)
        tk.Label(frame, text=label, width=15, anchor="w").pack(side="left")
        scale = tk.Scale(frame, from_=from_, to=to, resolution=resolution,
                         orient="horizontal", length=200, variable=variable, showvalue=False)
        scale.pack(side="left", padx=5)
        entry = tk.Entry(frame, textvariable=variable, width=5)
        entry.pack(side="left")

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Erreur lecture config: {e}", file=sys.stderr)
            return {}

    def _save_config(self):
        try:
            if "CAMERA" not in self.config_data:
                self.config_data["CAMERA"] = {}

            cam = self.config_data["CAMERA"]
            cam["width"] = int(self.width_var.get())
            cam["height"] = int(self.height_var.get())
            cam["rotation"] = int(self.rotation_var.get())
            cam["offcenter"] = float(self.offcenter_var.get())
            cam["min_detection_ratio"] = float(self.min_detection_ratio_var.get())

            cam["red_brighter_lower"] = [self.red_brighter_lower_h.get(), self.red_brighter_lower_s.get(), self.red_brighter_lower_v.get()]
            cam["red_brighter_upper"] = [self.red_brighter_upper_h.get(), self.red_brighter_upper_s.get(), self.red_brighter_upper_v.get()]
            cam["red_darker_lower"] = [self.red_darker_lower_h.get(), self.red_darker_lower_s.get(), self.red_darker_lower_v.get()]
            cam["red_darker_upper"] = [self.red_darker_upper_h.get(), self.red_darker_upper_s.get(), self.red_darker_upper_v.get()]
            cam["green_lower"] = [self.green_lower_h.get(), self.green_lower_s.get(), self.green_lower_v.get()]
            cam["green_upper"] = [self.green_upper_h.get(), self.green_upper_s.get(), self.green_upper_v.get()]

            temp_path = self.config_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.config_data, f, indent=2)
            os.replace(temp_path, self.config_path)

            self.status.configure(text=f"Configuration caméra mise à jour : {self.config_path}")
        except Exception as e:
            self.status.configure(text=f"Erreur écriture : {e}")

    def _schedule_write(self):
        if self._debounce_after_id:
            self.root.after_cancel(self._debounce_after_id)
        self._debounce_after_id = self.root.after(200, self._write_and_clear)

    def _write_and_clear(self):
        self._save_config()
        self._debounce_after_id = None

    def _reload_from_file(self):
        try:
            self.config_data = self._load_config()
            cam = self.config_data.get("CAMERA", {})

            self.width_var.set(cam.get("width", 160))
            self.height_var.set(cam.get("height", 120))
            self.rotation_var.set(cam.get("rotation", 180))
            self.offcenter_var.set(cam.get("offcenter", 0.2))
            self.min_detection_ratio_var.set(cam.get("min_detection_ratio", 12.0))

            # Seuils
            def get_list(key, default):
                return cam.get(key, default)

            rb_low = get_list("red_brighter_lower", [0,100,100])
            self.red_brighter_lower_h.set(rb_low[0])
            self.red_brighter_lower_s.set(rb_low[1])
            self.red_brighter_lower_v.set(rb_low[2])

            rb_up = get_list("red_brighter_upper", [10,255,255])
            self.red_brighter_upper_h.set(rb_up[0])
            self.red_brighter_upper_s.set(rb_up[1])
            self.red_brighter_upper_v.set(rb_up[2])

            rd_low = get_list("red_darker_lower", [160,100,100])
            self.red_darker_lower_h.set(rd_low[0])
            self.red_darker_lower_s.set(rd_low[1])
            self.red_darker_lower_v.set(rd_low[2])

            rd_up = get_list("red_darker_upper", [180,255,255])
            self.red_darker_upper_h.set(rd_up[0])
            self.red_darker_upper_s.set(rd_up[1])
            self.red_darker_upper_v.set(rd_up[2])

            g_low = get_list("green_lower", [30,50,50])
            self.green_lower_h.set(g_low[0])
            self.green_lower_s.set(g_low[1])
            self.green_lower_v.set(g_low[2])

            g_up = get_list("green_upper", [80,255,255])
            self.green_upper_h.set(g_up[0])
            self.green_upper_s.set(g_up[1])
            self.green_upper_v.set(g_up[2])

            self.status.configure(text="Fichier rechargé")
        except Exception as e:
            self.status.configure(text=f"Erreur rechargement : {e}")

    def _on_close(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()

def main(argv):
    parser = argparse.ArgumentParser(description="GUI pour ajuster les paramètres caméra en temps réel.")
    parser.add_argument("--config", type=str, default="new-config.json",
                        help="Chemin vers le fichier de configuration")
    args = parser.parse_args(argv)

    if not os.path.exists(args.config):
        print(f"Fichier introuvable : {args.config}", file=sys.stderr)
        return 1

    app = CameraConfigApp(config_path=args.config)
    app.run()
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))