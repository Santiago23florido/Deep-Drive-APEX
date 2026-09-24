"""Static world primitives, SDF export/import and 2D collision helpers.

Every generated world is made of two primitive types only: oriented boxes and
vertical cylinders. The same primitives are written to the SDF world (so the
world opens in the Gazebo GUI) and read back by the LiDAR ray caster, so the
geometry seen by the simulated LiDAR is exactly the geometry of the world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

# Kinds of static objects. ``collide`` marks objects the car must never touch.
KIND_GROUND = "ground"
KIND_WALL = "wall"  # tall barrier, blocks the LiDAR
KIND_CURB = "curb"  # low barrier below the scan plane, the LiDAR sees over it
KIND_OBSTACLE = "obstacle"  # in-lane obstacle (cones, pillars) the path avoids
KIND_FEATURE = "feature"  # decoration outside the lane (landmarks for the LiDAR)
KIND_ROOM = "room"  # hall boundary walls and columns

COLORS = {
    KIND_GROUND: "0.35 0.36 0.34 1",
    KIND_WALL: "0.85 0.85 0.82 1",
    KIND_CURB: "0.9 0.35 0.1 1",
    KIND_OBSTACLE: "1.0 0.45 0.0 1",
    KIND_FEATURE: "0.25 0.45 0.75 1",
    KIND_ROOM: "0.6 0.6 0.55 1",
}


@dataclass
class Box:
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw: float = 0.0
    kind: str = KIND_WALL
    name: str = ""


@dataclass
class Cylinder:
    center: tuple[float, float, float]  # centre of the cylinder volume
    radius: float
    length: float
    kind: str = KIND_OBSTACLE
    name: str = ""


@dataclass
class WorldGeometry:
    boxes: list[Box] = field(default_factory=list)
    cylinders: list[Cylinder] = field(default_factory=list)

    def extend(self, other: "WorldGeometry") -> None:
        self.boxes.extend(other.boxes)
        self.cylinders.extend(other.cylinders)

    # Arrays used by the ray caster and the collision checker.
    def box_arrays(self, include_ground: bool = True) -> dict[str, np.ndarray]:
        boxes = [b for b in self.boxes if include_ground or b.kind != KIND_GROUND]
        if not boxes:
            return {
                "center": np.zeros((0, 3)),
                "half": np.zeros((0, 3)),
                "yaw": np.zeros(0),
                "kind": np.zeros(0, dtype=object),
            }
        return {
            "center": np.array([b.center for b in boxes], dtype=float),
            "half": 0.5 * np.array([b.size for b in boxes], dtype=float),
            "yaw": np.array([b.yaw for b in boxes], dtype=float),
            "kind": np.array([b.kind for b in boxes], dtype=object),
        }

    def cylinder_arrays(self) -> dict[str, np.ndarray]:
        cyl = self.cylinders
        if not cyl:
            return {
                "center": np.zeros((0, 3)),
                "radius": np.zeros(0),
                "half_length": np.zeros(0),
                "kind": np.zeros(0, dtype=object),
            }
        return {
            "center": np.array([c.center for c in cyl], dtype=float),
            "radius": np.array([c.radius for c in cyl], dtype=float),
            "half_length": 0.5 * np.array([c.length for c in cyl], dtype=float),
            "kind": np.array([c.kind for c in cyl], dtype=object),
        }


# --------------------------------------------------------------------- SDF I/O
def _fmt(values) -> str:
    return " ".join(f"{float(v):.6f}" for v in values)


def _material(parent: ET.Element, rgba: str) -> None:
    mat = ET.SubElement(parent, "material")
    ET.SubElement(mat, "ambient").text = rgba
    ET.SubElement(mat, "diffuse").text = rgba


def _surface(parent: ET.Element, mu: float) -> None:
    surface = ET.SubElement(parent, "surface")
    friction = ET.SubElement(ET.SubElement(surface, "friction"), "ode")
    ET.SubElement(friction, "mu").text = f"{mu}"
    ET.SubElement(friction, "mu2").text = f"{mu}"
    contact = ET.SubElement(ET.SubElement(surface, "contact"), "ode")
    ET.SubElement(contact, "kp").text = "100000.0"
    ET.SubElement(contact, "kd").text = "10.0"


def geometry_to_models(geometry: WorldGeometry, ground_mu: float = 1.2) -> list[ET.Element]:
    """One static model per object kind; every primitive is a collision + visual."""
    models: list[ET.Element] = []
    kinds = sorted({b.kind for b in geometry.boxes} | {c.kind for c in geometry.cylinders})
    for kind in kinds:
        model = ET.Element("model", {"name": f"track_{kind}"})
        ET.SubElement(model, "static").text = "true"
        ET.SubElement(model, "pose").text = "0 0 0 0 0 0"
        link = ET.SubElement(model, "link", {"name": f"{kind}_link"})
        counter = 0
        for box in (b for b in geometry.boxes if b.kind == kind):
            counter += 1
            name = box.name or f"{kind}_box_{counter}"
            pose = _fmt((*box.center, 0.0, 0.0, box.yaw))
            for tag in ("collision", "visual"):
                el = ET.SubElement(link, tag, {"name": f"{name}_{tag}"})
                ET.SubElement(el, "pose").text = pose
                ET.SubElement(ET.SubElement(ET.SubElement(el, "geometry"), "box"), "size").text = _fmt(box.size)
                if tag == "visual":
                    _material(el, COLORS.get(kind, "0.7 0.7 0.7 1"))
                elif kind == KIND_GROUND:
                    _surface(el, ground_mu)
        for cyl in (c for c in geometry.cylinders if c.kind == kind):
            counter += 1
            name = cyl.name or f"{kind}_cyl_{counter}"
            pose = _fmt((*cyl.center, 0.0, 0.0, 0.0))
            for tag in ("collision", "visual"):
                el = ET.SubElement(link, tag, {"name": f"{name}_{tag}"})
                ET.SubElement(el, "pose").text = pose
                cylinder = ET.SubElement(ET.SubElement(el, "geometry"), "cylinder")
                ET.SubElement(cylinder, "radius").text = f"{cyl.radius:.6f}"
                ET.SubElement(cylinder, "length").text = f"{cyl.length:.6f}"
                if tag == "visual":
                    _material(el, COLORS.get(kind, "0.7 0.7 0.7 1"))
        models.append(model)
    return models


def world_sdf(
    geometry: WorldGeometry,
    world_name: str = "default",
    ground_mu: float = 1.2,
    extra_models: list[ET.Element] | None = None,
    physics_step_s: float = 0.001,
    real_time_factor: float = 1.0,
) -> ET.Element:
    root = ET.Element("sdf", {"version": "1.9"})
    world = ET.SubElement(root, "world", {"name": world_name})
    physics = ET.SubElement(world, "physics", {"name": "default_physics", "type": "dart"})
    ET.SubElement(physics, "max_step_size").text = f"{physics_step_s}"
    ET.SubElement(physics, "real_time_factor").text = f"{real_time_factor}"
    ET.SubElement(physics, "real_time_update_rate").text = f"{1.0 / physics_step_s:.1f}"
    ET.SubElement(world, "plugin", {"filename": "gz-sim-physics-system", "name": "gz::sim::systems::Physics"})
    ET.SubElement(world, "plugin", {"filename": "gz-sim-user-commands-system", "name": "gz::sim::systems::UserCommands"})
    ET.SubElement(world, "plugin", {"filename": "gz-sim-scene-broadcaster-system", "name": "gz::sim::systems::SceneBroadcaster"})
    light = ET.SubElement(world, "light", {"type": "directional", "name": "sun"})
    ET.SubElement(light, "pose").text = "0 0 10 0 0 0"
    ET.SubElement(light, "diffuse").text = "0.8 0.8 0.8 1"
    ET.SubElement(light, "direction").text = "-0.5 0.5 -1"
    ET.SubElement(light, "cast_shadows").text = "true"
    for model in geometry_to_models(geometry, ground_mu=ground_mu):
        world.append(model)
    for model in extra_models or []:
        world.append(model)
    return root


def write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('<?xml version="1.0" ?>\n' + ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")


def _pose6(element: ET.Element | None) -> np.ndarray:
    if element is None or not (element.text or "").strip():
        return np.zeros(6)
    values = [float(v) for v in element.text.split()]
    values += [0.0] * (6 - len(values))
    return np.array(values[:6])


def _rot_rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _compose(t_a: tuple[np.ndarray, np.ndarray], pose6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rot_a, pos_a = t_a
    rot_b = _rot_rpy(*pose6[3:])
    return rot_a @ rot_b, pos_a + rot_a @ pose6[:3]


def geometry_from_sdf(path: str | Path, skip_models: tuple[str, ...] = ("rc_car",)) -> WorldGeometry:
    """Read every static box / cylinder collision of an SDF world.

    Works for the generated worlds and for the hand-written worlds of
    ``rc_sim_description`` (model -> link -> collision pose composition).
    Boxes may only be rotated about z; cylinders must be vertical.
    """
    root = ET.parse(str(path)).getroot()
    world = root.find("world")
    geometry = WorldGeometry()
    for model in world.findall("model"):
        name = model.get("name", "")
        if name in skip_models:
            continue
        kind_hint = name.replace("track_", "") if name.startswith("track_") else None
        t_model = _compose((np.eye(3), np.zeros(3)), _pose6(model.find("pose")))
        for link in model.findall("link"):
            t_link = _compose(t_model, _pose6(link.find("pose")))
            for col in link.findall("collision"):
                rot, pos = _compose(t_link, _pose6(col.find("pose")))
                geom = col.find("geometry")
                if geom is None or len(geom) == 0:
                    continue
                shape = geom[0]
                if abs(rot[2, 2] - 1.0) > 1e-6:
                    raise ValueError(f"{path}: only z-rotations are supported ({col.get('name')})")
                yaw = math.atan2(rot[1, 0], rot[0, 0])
                if shape.tag == "box":
                    size = tuple(float(v) for v in shape.findtext("size").split())
                    kind = kind_hint or (KIND_GROUND if size[0] * size[1] > 400.0 else KIND_WALL)
                    geometry.boxes.append(Box(tuple(pos), size, yaw, kind, col.get("name", "")))
                elif shape.tag == "cylinder":
                    geometry.cylinders.append(
                        Cylinder(
                            tuple(pos),
                            float(shape.findtext("radius")),
                            float(shape.findtext("length")),
                            kind_hint or KIND_FEATURE,
                            col.get("name", ""),
                        )
                    )
    return geometry


# ----------------------------------------------------------- 2D collision test
def rect_corners(x: float, y: float, yaw: float, length: float, width: float, x_offset: float = 0.0) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = 0.5 * length, 0.5 * width
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]]) + np.array([x_offset, 0.0])
    return np.column_stack((x + c * local[:, 0] - s * local[:, 1], y + s * local[:, 0] + c * local[:, 1]))


class FootprintChecker:
    """Separating-axis test of the car footprint against static obstacles.

    Only objects whose top is above ``min_height`` are considered (the ground
    and the car cannot touch anything lower). The world is split into a coarse
    grid so each query only tests nearby primitives.
    """

    def __init__(self, geometry: WorldGeometry, min_height: float = 0.012, cell: float = 2.0) -> None:
        self.boxes = [b for b in geometry.boxes if b.kind != KIND_GROUND and b.center[2] + 0.5 * b.size[2] > min_height]
        self.cylinders = [c for c in geometry.cylinders if c.center[2] + 0.5 * c.length > min_height]
        self.cell = cell
        self._grid: dict[tuple[int, int], list[tuple[str, int]]] = {}
        for i, b in enumerate(self.boxes):
            r = 0.5 * math.hypot(b.size[0], b.size[1])
            self._insert(("b", i), b.center[0], b.center[1], r)
        for i, c in enumerate(self.cylinders):
            self._insert(("c", i), c.center[0], c.center[1], c.radius)

    def _insert(self, key: tuple[str, int], x: float, y: float, r: float) -> None:
        for gx in range(int(math.floor((x - r) / self.cell)), int(math.floor((x + r) / self.cell)) + 1):
            for gy in range(int(math.floor((y - r) / self.cell)), int(math.floor((y + r) / self.cell)) + 1):
                self._grid.setdefault((gx, gy), []).append(key)

    def _candidates(self, corners: np.ndarray) -> set[tuple[str, int]]:
        lo = np.floor(corners.min(axis=0) / self.cell).astype(int)
        hi = np.floor(corners.max(axis=0) / self.cell).astype(int)
        found: set[tuple[str, int]] = set()
        for gx in range(lo[0], hi[0] + 1):
            for gy in range(lo[1], hi[1] + 1):
                found.update(self._grid.get((gx, gy), ()))
        return found

    @staticmethod
    def _sat(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
        for poly in (poly_a, poly_b):
            edges = np.roll(poly, -1, axis=0) - poly
            axes = np.column_stack((-edges[:, 1], edges[:, 0]))
            for axis in axes:
                pa = poly_a @ axis
                pb = poly_b @ axis
                if pa.max() < pb.min() or pb.max() < pa.min():
                    return False
        return True

    def collides(self, corners: np.ndarray) -> str | None:
        """Return the name/kind of the first obstacle overlapping the footprint."""
        for kind, idx in self._candidates(corners):
            if kind == "b":
                b = self.boxes[idx]
                poly = rect_corners(b.center[0], b.center[1], b.yaw, b.size[0], b.size[1])
                if self._sat(corners, poly):
                    return f"{b.kind}:{b.name}"
            else:
                c = self.cylinders[idx]
                centre = np.array(c.center[:2])
                # Distance from the circle centre to the rectangle (convex polygon).
                inside = True
                min_dist = float("inf")
                for a, b2 in zip(corners, np.roll(corners, -1, axis=0)):
                    edge = b2 - a
                    t = float(np.clip(np.dot(centre - a, edge) / np.dot(edge, edge), 0.0, 1.0))
                    min_dist = min(min_dist, float(np.linalg.norm(a + t * edge - centre)))
                    if edge[0] * (centre[1] - a[1]) - edge[1] * (centre[0] - a[0]) > 0:
                        inside = False
                if inside or min_dist < c.radius:
                    return f"{c.kind}:{c.name}"
        return None
