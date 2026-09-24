"""Shared helpers for the ROS 2 nodes of this package."""

from __future__ import annotations

import json
from typing import Any, Callable, Generic, TypeVar

from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from ..core.config_io import dataclass_from_flat, flatten_dataclass

T = TypeVar("T")

# Latched QoS for one-shot metadata (realizations, alignment results).
LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def stamp_to_sec(stamp: TimeMsg) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def sec_to_stamp(t: float) -> TimeMsg:
    msg = TimeMsg()
    msg.sec = int(t // 1)
    msg.nanosec = int(round((t - msg.sec) * 1e9))
    if msg.nanosec >= 1_000_000_000:
        msg.sec += 1
        msg.nanosec -= 1_000_000_000
    return msg


def odom_to_transform(odom: Odometry) -> TransformStamped:
    """TF of an odometry message (header.frame_id -> child_frame_id)."""
    tf = TransformStamped()
    tf.header = odom.header
    tf.child_frame_id = odom.child_frame_id
    p = odom.pose.pose.position
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = p.x, p.y, p.z
    tf.transform.rotation = odom.pose.pose.orientation
    return tf


def json_msg(payload: dict[str, Any]) -> String:
    return String(data=json.dumps(payload, sort_keys=True))


class ConfigParameters(Generic[T]):
    """Expose every leaf field of a config dataclass as a ROS parameter.

    * Field ``gyro.noise_density`` becomes ROS parameter ``<prefix>gyro.noise_density``.
    * Parameters use dynamic typing so YAML ``100`` is accepted for a float field.
    * Runtime changes (``ros2 param set`` / rqt_reconfigure) rebuild the config
      and invoke ``on_change`` so the owning node can re-create its model.
    """

    def __init__(
        self,
        node: Node,
        config_cls: type[T],
        prefix: str = "",
        on_change: Callable[[T], None] | None = None,
    ) -> None:
        self._node = node
        self._cls = config_cls
        self._prefix = prefix
        self._on_change = on_change
        self._names: list[str] = []
        descriptor = ParameterDescriptor(dynamic_typing=True)
        for key, default in flatten_dataclass(config_cls()).items():
            name = f"{prefix}{key}"
            value = list(default) if isinstance(default, tuple) else default
            node.declare_parameter(name, value, descriptor)
            self._names.append(name)
        self.config: T = self._read()
        node.add_on_set_parameters_callback(self._on_set)
        node.add_post_set_parameters_callback(self._post_set)

    def _read(self, overrides: dict[str, Any] | None = None) -> T:
        flat = {}
        for name in self._names:
            value = self._node.get_parameter(name).value
            if overrides and name in overrides:
                value = overrides[name]
            flat[name[len(self._prefix):]] = value
        return dataclass_from_flat(self._cls, flat)

    def _on_set(self, params) -> SetParametersResult:
        overrides = {p.name: p.value for p in params if p.name in self._names}
        if not overrides:
            return SetParametersResult(successful=True)
        try:
            candidate = self._read(overrides)
            validate = getattr(candidate, "validate", None)
            if callable(validate):
                validate()
        except Exception as exc:  # reject invalid values, keep the old config
            return SetParametersResult(successful=False, reason=str(exc))
        return SetParametersResult(successful=True)

    def _post_set(self, params) -> None:
        if not any(p.name in self._names for p in params):
            return
        self.config = self._read()
        self._node.get_logger().info(
            "configuration updated: " + ", ".join(f"{p.name}={p.value}" for p in params)
        )
        if self._on_change is not None:
            self._on_change(self.config)

    def as_dict(self) -> dict[str, Any]:
        return flatten_dataclass(self.config)
