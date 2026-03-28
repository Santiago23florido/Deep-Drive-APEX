#!/usr/bin/env python3
import argparse
import json
import os
import sys
import tkinter as tk
from typing import Any, Dict

class SteeringConfigApp:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._debounce_after_id = None

        # Chargement initial
        self.config_data = self._load_config()
        steering_params = self.config_data.get("STEERING", {})

        self.root = tk.Tk()
        self.root.title("Configuration direction en direct")

        self.limit_var = tk.DoubleVar(master=self.root, value=steering_params.get("limit", 18.0))
        self.dc_min_var = tk.DoubleVar(master=self.root, value=steering_params.get("dc_min", 5.0))
        self.dc_max_var = tk.DoubleVar(master=self.root, value=steering_params.get("dc_max", 8.6))
        self.channel_var = tk.IntVar(master=self.root, value=steering_params.get("channel", 1))
        self.frequency_var = tk.DoubleVar(master=self.root, value=steering_params.get("frequency", 50.0))

        frm = tk.Frame(self.root, padx=12, pady=12)
        frm.pack(fill="both", expand=True)

        tk.Label(frm, text="Ajustez les paramètres de direction en temps réel", font=("Arial", 12, "bold")).pack(anchor="w")
        tk.Label(frm, text="Les changements sont appliqués immédiatement au fichier de configuration.", fg="gray").pack(anchor="w", pady=(0,10))

        self._add_slider(frm, "limit (deg)", self.limit_var, 0.0, 45.0, 0.5)
        self._add_slider(frm, "dc_min (%)", self.dc_min_var, 0.0, 15.0, 0.1)
        self._add_slider(frm, "dc_max (%)", self.dc_max_var, 0.0, 15.0, 0.1)
        self._add_slider(frm, "channel", self.channel_var, 0, 1, 1)
        self._add_slider(frm, "frequency (Hz)", self.frequency_var, 40, 60, 0.5)

        btn_row = tk.Frame(frm)
        btn_row.pack(anchor="w", pady=(10,0))
        tk.Button(btn_row, text="Recharger depuis le fichier", command=self._reload_from_file).pack(side="left")
        tk.Button(btn_row, text="Quitter", command=self._on_close).pack(side="left", padx=(8,0))

        self.status = tk.Label(frm, text="Prêt", justify="left")
        self.status.pack(anchor="w", pady=(10,0))

        # Traces pour mise à jour automatique
        self.limit_var.trace_add("write", lambda *_: self._schedule_write())
        self.dc_min_var.trace_add("write", lambda *_: self._schedule_write())
        self.dc_max_var.trace_add("write", lambda *_: self._schedule_write())
        self.channel_var.trace_add("write", lambda *_: self._schedule_write())
        self.frequency_var.trace_add("write", lambda *_: self._schedule_write())

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _add_slider(self, parent, label, variable, from_, to, resolution):
        frame = tk.Frame(parent)
        frame.pack(fill="x", pady=5)
        tk.Label(frame, text=label, width=20, anchor="w").pack(side="left")
        scale = tk.Scale(frame, from_=from_, to=to, resolution=resolution,
                         orient="horizontal", length=300, variable=variable, showvalue=False)
        scale.pack(side="left", padx=5)
        entry = tk.Entry(frame, textvariable=variable, width=8)
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
            if "STEERING" not in self.config_data:
                self.config_data["STEERING"] = {}
            self.config_data["STEERING"]["limit"] = float(self.limit_var.get())
            self.config_data["STEERING"]["dc_min"] = float(self.dc_min_var.get())
            self.config_data["STEERING"]["dc_max"] = float(self.dc_max_var.get())
            self.config_data["STEERING"]["channel"] = int(self.channel_var.get())
            self.config_data["STEERING"]["frequency"] = float(self.frequency_var.get())

            temp_path = self.config_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.config_data, f, indent=2)
            os.replace(temp_path, self.config_path)

            self.status.configure(text=f"Configuration direction mise à jour : {self.config_path}")
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
            steering = self.config_data.get("STEERING", {})
            self.limit_var.set(steering.get("limit", 18.0))
            self.dc_min_var.set(steering.get("dc_min", 5.0))
            self.dc_max_var.set(steering.get("dc_max", 8.6))
            self.channel_var.set(steering.get("channel", 1))
            self.frequency_var.set(steering.get("frequency", 50.0))
            self.status.configure(text="Fichier rechargé")
        except Exception as e:
            self.status.configure(text=f"Erreur rechargement : {e}")

    def _on_close(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()

def main(argv):
    parser = argparse.ArgumentParser(description="GUI pour ajuster les paramètres de direction en temps réel.")
    parser.add_argument("--config", type=str, default="new-config.json",
                        help="Chemin vers le fichier de configuration")
    args = parser.parse_args(argv)

    if not os.path.exists(args.config):
        print(f"Fichier introuvable : {args.config}", file=sys.stderr)
        return 1

    app = SteeringConfigApp(config_path=args.config)
    app.run()
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))