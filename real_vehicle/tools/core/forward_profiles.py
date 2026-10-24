"""Explicit experimental forward calibrations; approved acquisition is separate."""
import json

PROFILES = {
    "initialized": {
        "esc_zero_pct": 15.42, "pct_per_mps": 5.24,
        "command_cap": 2.4, "forward_scale": 6.3,
        "references": [.55, .45, .50, .55], "steering_confidence_gate": True,
        "w_smooth": .1,
    },
    "tracking": {
        # Median of six moving-segment speed/input ratios from run 211216.
        # Zero moving intercept is an assumption, not a two-parameter fit.
        "esc_zero_pct": 15.7, "pct_per_mps": 33.0 / 3.8,
        "command_cap": 1.43, "forward_scale": 3.8,
        "references": [.42, .36, .36, .42], "steering_confidence_gate": False,
        "w_smooth": .05, "revision": "forward_trim_20261024",
    },
}


def forward_profile(name):
    values = dict(PROFILES[name])  # Unknown selection must fail before startup.
    values["name"] = name
    values["race_yaml"] = f"ros2_ws/src/apex_race/config/race_speed_{name}.yaml"
    values["vehicle_yaml"] = f"ros2_ws/src/apex_race/config/vehicle_speed_{name}.yaml"
    return values


def select_forward_profile(directory, explicit=None):
    """Carry the request across sudo through the existing per-trial RAM marker.

    sudo deliberately drops arbitrary environment variables. A RAM run must
    have an explicit recorded selection; never fall back to another profile.
    """
    marker = directory / '.ram_capture_active.json'
    if marker.exists():
        name = json.loads(marker.read_text())['forward_profile']
        if explicit is not None and explicit != name:
            raise ValueError('Requested profile disagrees with RAM trial selection')
    else:
        name = explicit or 'initialized'
    return forward_profile(name)
