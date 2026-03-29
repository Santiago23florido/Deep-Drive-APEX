#!/usr/bin/env python3
import argparse
import json
import os
import sys
import tkinter as tk
from typing import Any, Dict

# On réutilise la classe PWM backend (identique à celle du moteur)
class SteeringPWMBackend:
    def __init__(self, channel: int, hz: float, neutral_dc: float):
        self.channel = channel
        self.hz = hz
        self.neutral_dc = neutral_dc
        self.backend_name = "unknown"
        self._pwm = None
        self._set_dc = None
        self._stop_backend = None
        self._init_backend()
        self.set_duty_cycle(self.neutral_dc)

    def _init_backend(self):
        try:
            from rpi_hardware_pwm import HardwarePWM
            self.backend_name = "rpi_hardware_pwm"
            self._pwm = HardwarePWM(pwm_channel=self.channel, hz=int(self.hz))
            self._pwm.start(self.neutral_dc)
            self._set_dc = self._pwm.change_duty_cycle
            def _stop():
                try:
                    self._pwm.change_duty_cycle(self.neutral_dc)
                finally:
                    self._pwm.stop()
            self._stop_backend = _stop
            return
        except Exception:
            pass

        try:
            from raspberry_pwm import PWM
            self.backend_name = "sysfs_pwm"
            self._pwm = PWM(channel=self.channel, frequency=self.hz)
            self._pwm.start(self.neutral_dc)
            self._set_dc = self._pwm.set_duty_cycle
            def _stop():
                try:
                    self._pwm.set_duty_cycle(self.neutral_dc)
                finally:
                    self._pwm.stop()
            self._stop_backend = _stop
            return
        except Exception as e:
            raise RuntimeError(f"Impossible d'initialiser le PWM. Erreur: {e}")

    def set_duty_cycle(self, dc_percent: float):
        if self._set_dc is None:
            raise RuntimeError("PWM backend not initialized")
        dc_percent = float(max(0.0, min(100.0, dc_percent)))
        self._set_dc(dc_percent)

    def stop(self):
        if self._stop_backend is not None:
            self._stop_backend()

    def update_neutral_dc(self, neutral_dc: float, apply_now: bool = False):
        self.neutral_dc = float(neutral_dc)
        if apply_now:
            self.set_duty_cycle(self.neutral_dc)

class SteeringController:
    def __init__(self, pwm: SteeringPWMBackend):
        self.pwm = pwm
        # Paramètres initiaux (seront mis à jour depuis la config)
        self.limit = 18.0          # angle max en degrés
        self.dc_min = 5.0
        self.dc_max = 8.6
        # Point neutre = milieu
        self.neutral_dc = (self.dc_min + self.dc_max) / 2.0

    def angle_to_duty_cycle(self, angle_deg: float) -> float:
        """Convertit un angle de braquage (deg) en duty cycle (%)."""
        # Saturer l'angle dans les limites
        if angle_deg > self.limit:
            angle_deg = self.limit
        if angle_deg < -self.limit:
            angle_deg = -self.limit

        # Mapper linéairement angle -> duty cycle
        # angle = 0 -> neutral_dc
        # angle = +limit -> dc_max
        # angle = -limit -> dc_min
        if angle_deg >= 0:
            # portion positive
            factor = angle_deg / self.limit if self.limit > 0 else 0
            return self.neutral_dc + factor * (self.dc_max - self.neutral_dc)
        else:
            # portion négative
            factor = -angle_deg / self.limit if self.limit > 0 else 0
            return self.neutral_dc - factor * (self.neutral_dc - self.dc_min)

    def set_angle(self, angle_deg: float):
        dc = self.angle_to_duty_cycle(angle_deg)
        self.pwm.set_duty_cycle(dc)

    def update_params(self, limit: float, dc_min: float, dc_max: float):
        self.limit = limit
        self.dc_min = dc_min
        self.dc_max = dc_max
        self.neutral_dc = (dc_min + dc_max) / 2.0

class SteeringRealtimeApp:
    def __init__(self, config_path: str, tick_ms: int = 50):
        self.config_path = config_path
        self.tick_ms = tick_ms
        self.last_mtime = 0

        # Chargement initial
        self.config_data = self._load_config()
        steering_cfg = self.config_data.get("STEERING", {})

        # Initialisation PWM
        try:
            self.pwm = SteeringPWMBackend(
                channel=steering_cfg.get("channel", 1),
                hz=steering_cfg.get("frequency", 50.0),
                neutral_dc=(steering_cfg.get("dc_min",5.0) + steering_cfg.get("dc_max",8.6))/2.0
            )
        except Exception as e:
            print(f"Erreur initialisation PWM: {e}", file=sys.stderr)
            sys.exit(1)

        self.controller = SteeringController(pwm=self.pwm)
        self._apply_steering_config(steering_cfg)

        # Interface Tkinter
        self.root = tk.Tk()
        self.root.title("Test direction en temps réel")

        self.target_angle = tk.DoubleVar(value=0.0)

        frm = tk.Frame(self.root, padx=12, pady=12)
        frm.pack(fill="both", expand=True)

        tk.Label(frm, text="Contrôle direction avec paramètres dynamiques", font=("Arial", 12, "bold")).pack(anchor="w")
        tk.Label(frm, text="Les paramètres sont mis à jour depuis le fichier JSON.", fg="gray").pack(anchor="w", pady=(0,10))

        self.slider = tk.Scale(
            frm,
            from_=-self.controller.limit,
            to=self.controller.limit,
            resolution=0.5,
            orient="horizontal",
            length=520,
            variable=self.target_angle,
            label="Angle de braquage (deg)"
        )
        self.slider.pack(anchor="w", fill="x")

        self.info_label = tk.Label(frm, text="", justify="left")
        self.info_label.pack(anchor="w", pady=(10,0))

        btn_row = tk.Frame(frm)
        btn_row.pack(anchor="w", pady=(10,0))
        tk.Button(btn_row, text="CENTRER", command=self._on_center).pack(side="left")
        tk.Button(btn_row, text="Quitter", command=self._on_close).pack(side="left", padx=(8,0))

        self.status = tk.Label(frm, text="Prêt", justify="left")
        self.status.pack(anchor="w", pady=(10,0))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Lancer la surveillance
        self.root.after(500, self._check_config)
        self.root.after(self.tick_ms, self._tick)

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Erreur lecture config: {e}", file=sys.stderr)
            return {}

    def _apply_steering_config(self, steering_cfg: Dict[str, Any]):
        limit = steering_cfg.get("limit", 18.0)
        dc_min = steering_cfg.get("dc_min", 5.0)
        dc_max = steering_cfg.get("dc_max", 8.6)
        self.controller.update_params(limit, dc_min, dc_max)

        # Mise à jour de la plage du slider
        current_from = float(self.slider.cget("from"))
        current_to = float(self.slider.cget("to"))
        if abs(current_from + limit) > 0.1 or abs(current_to - limit) > 0.1:
            self.slider.config(from_=-limit, to=limit)
            # Re-centrer la valeur si hors limites
            val = self.target_angle.get()
            if val < -limit:
                self.target_angle.set(-limit)
            elif val > limit:
                self.target_angle.set(limit)

        self.info_label.configure(
            text=f"limit={limit:.1f}°, dc_min={dc_min:.2f}%, dc_max={dc_max:.2f}%, "
                 f"neutral={self.controller.neutral_dc:.2f}%"
        )

    def _check_config(self):
        try:
            mtime = os.path.getmtime(self.config_path)
            if mtime != self.last_mtime:
                self.last_mtime = mtime
                new_config = self._load_config()
                steering = new_config.get("STEERING", {})
                self._apply_steering_config(steering)

                # Vérifier si le canal ou la fréquence a changé -> nécessite redémarrage
                old_channel = self.pwm.channel
                old_hz = self.pwm.hz
                new_channel = steering.get("channel", old_channel)
                new_hz = steering.get("frequency", old_hz)
                if new_channel != old_channel or abs(new_hz - old_hz) > 0.1:
                    self.status.configure(
                        text="Attention: changement de canal/fréquence détecté. Redémarrez ce script pour appliquer."
                    )
                else:
                    self.status.configure(text="Configuration mise à jour")
        except Exception as e:
            self.status.configure(text=f"Erreur vérification: {e}")

        self.root.after(500, self._check_config)

    def _on_center(self):
        self.target_angle.set(0.0)

    def _on_close(self):
        try:
            self.controller.set_angle(0.0)
        finally:
            self.pwm.stop()
            self.root.destroy()

    def _tick(self):
        angle = self.target_angle.get()
        try:
            self.controller.set_angle(angle)
            self.status.configure(
                text=f"Angle commandé: {angle:.1f}°, Duty cycle: {self.controller.angle_to_duty_cycle(angle):.2f}%"
            )
        except Exception as e:
            self.status.configure(text=f"Erreur: {e}")
        self.root.after(self.tick_ms, self._tick)

    def run(self):
        self.root.mainloop()

def main(argv):
    parser = argparse.ArgumentParser(description="Test direction avec mise à jour en direct des paramètres.")
    parser.add_argument("--config", type=str, default="new-config.json",
                        help="Chemin vers le fichier de configuration")
    args = parser.parse_args(argv)

    if not os.path.exists(args.config):
        print(f"Fichier introuvable : {args.config}", file=sys.stderr)
        return 1

    app = SteeringRealtimeApp(config_path=args.config)
    app.run()
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))