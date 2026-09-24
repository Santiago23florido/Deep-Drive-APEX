"""Generic helpers to map configuration dataclasses to flat parameter dicts.

Every model in :mod:`apex_fusion_research.core` is configured by a (possibly
nested) dataclass. These helpers flatten such a dataclass into
``{"gyro.noise_density": 1e-4, ...}`` so that

* ROS nodes can expose *every* field as a ROS parameter automatically, and
* offline tools can load the very same ROS parameter YAML presets.

Adding a field to a config dataclass therefore makes it tunable everywhere
without touching the ROS or tooling code.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def flatten_dataclass(instance: Any, prefix: str = "") -> dict[str, Any]:
    """Return ``{dotted.name: value}`` for every leaf field of a dataclass."""
    flat: dict[str, Any] = {}
    for field in dataclasses.fields(instance):
        value = getattr(instance, field.name)
        key = f"{prefix}{field.name}"
        if dataclasses.is_dataclass(value):
            flat.update(flatten_dataclass(value, prefix=f"{key}."))
        else:
            flat[key] = value
    return flat


def dataclass_from_flat(cls: type[T], flat: dict[str, Any], prefix: str = "") -> T:
    """Build ``cls`` from a flat dict; missing keys keep their defaults.

    Values are coerced to the type of the default value so that YAML integers
    such as ``0`` are accepted for float fields.
    """
    default = cls()
    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(cls):
        key = f"{prefix}{field.name}"
        default_value = getattr(default, field.name)
        if dataclasses.is_dataclass(default_value):
            kwargs[field.name] = dataclass_from_flat(type(default_value), flat, prefix=f"{key}.")
        elif key in flat:
            kwargs[field.name] = _coerce(flat[key], default_value)
        else:
            kwargs[field.name] = default_value
    return cls(**kwargs)


def _coerce(value: Any, default_value: Any) -> Any:
    if isinstance(default_value, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(default_value, int):
        return int(value)
    if isinstance(default_value, float):
        return float(value)
    if isinstance(default_value, (list, tuple)):
        return type(default_value)(value)
    return value


def _flatten_mapping(mapping: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_mapping(value, prefix=f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


def load_ros_params_yaml(path: str | Path, node_name: str | None = None) -> dict[str, Any]:
    """Load a ROS 2 parameter YAML file into a flat dict.

    If ``node_name`` is ``None`` the first ``ros__parameters`` block is used,
    which is convenient for the single-node preset files in ``config/``.
    """
    payload = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {}
    for name, entry in payload.items():
        if node_name is not None and name.strip("/") != node_name:
            continue
        if isinstance(entry, dict) and "ros__parameters" in entry:
            return _flatten_mapping(entry["ros__parameters"])
    raise KeyError(f"No ros__parameters block for node '{node_name}' in {path}")
