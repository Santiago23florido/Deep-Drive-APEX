"""Procedural race tracks: centreline, reference path and static geometry.

A track centreline is a *rounded polygon*: straight edges joined by circular
arcs of prescribed radius at every vertex. This gives exact control over the
curve radii (the car cannot turn tighter than ~0.95 m) and produces straights,
long curves, S-bends (alternating turn directions) and hairpins (two tight
vertices close together).

Around the centreline the builder places:

* tall walls (block the LiDAR) or low curbs (below the scan plane, the LiDAR
  sees the hall behind them), chosen per edge and per side;
* openings with dead-end alcoves (urban side streets);
* in-lane obstacles (slalom cones, pillars) and the lateral offset of the
  reference path that avoids them;
* seeded decoration outside the lane and a hall boundary with columns, so
  the LiDAR always observes identifiable, non-parallel structure.

Everything is deterministic given the YAML specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .geometry import (
    KIND_CURB,
    KIND_FEATURE,
    KIND_GROUND,
    KIND_OBSTACLE,
    KIND_ROOM,
    KIND_WALL,
    Box,
    Cylinder,
    WorldGeometry,
)

CONFIG_DIR = Path(__file__).resolve().parent / "config"

WALL_HEIGHT = 0.30
WALL_THICKNESS = 0.08
CURB_HEIGHT = 0.07
CURB_THICKNESS = 0.06


@dataclass
class Path2D:
    """Closed, arc-length parametrised polyline."""

    xy: np.ndarray  # (N, 2)
    s: np.ndarray  # (N,)
    heading: np.ndarray  # (N,)
    curvature: np.ndarray  # (N,)
    length: float

    @property
    def normal(self) -> np.ndarray:
        return np.column_stack((-np.sin(self.heading), np.cos(self.heading)))

    def interp(self, s_query: np.ndarray | float) -> np.ndarray:
        sq = np.mod(np.asarray(s_query, dtype=float), self.length)
        s_ext = np.append(self.s, self.length)
        x_ext = np.append(self.xy[:, 0], self.xy[0, 0])
        y_ext = np.append(self.xy[:, 1], self.xy[0, 1])
        return np.stack((np.interp(sq, s_ext, x_ext), np.interp(sq, s_ext, y_ext)), axis=-1)

    def interp_scalar(self, values: np.ndarray, s_query: np.ndarray | float) -> np.ndarray:
        sq = np.mod(np.asarray(s_query, dtype=float), self.length)
        return np.interp(sq, np.append(self.s, self.length), np.append(values, values[0]))

    def reversed(self) -> "Path2D":
        xy = self.xy[::-1].copy()
        return path_from_points(xy)

    def rotated_start(self, s0: float) -> "Path2D":
        """Same closed path with the arc-length origin moved to ``s0``."""
        idx = int(np.searchsorted(self.s, np.mod(s0, self.length)))
        xy = np.roll(self.xy, -idx, axis=0)
        return path_from_points(xy)


def path_from_points(xy: np.ndarray) -> Path2D:
    """Arc length, heading and curvature (finite differences) of a closed polyline."""
    xy = np.asarray(xy, dtype=float)
    nxt = np.roll(xy, -1, axis=0)
    seg = np.linalg.norm(nxt - xy, axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg[:-1])))
    length = float(np.sum(seg))
    d = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    heading = np.arctan2(d[:, 1], d[:, 0])
    dh = np.angle(np.exp(1j * (np.roll(heading, -1) - np.roll(heading, 1))))
    ds = np.roll(s, -1) - np.roll(s, 1)
    ds = np.where(ds <= 0, ds + length, ds)
    curvature = dh / np.maximum(ds, 1e-9)
    return Path2D(xy, s, heading, curvature, length)


# ------------------------------------------------------------- rounded polygon
def rounded_polygon(vertices: list[list[float]], radii: list[float], ds: float = 0.02) -> tuple[Path2D, list[dict[str, float]]]:
    """Centreline through fillet arcs. Returns the path and per-vertex arc info.

    Arc ``i`` replaces the corner at vertex ``i``. The path starts at the
    middle of the straight part of edge 0 (vertex 0 -> vertex 1).
    """
    pts = np.asarray(vertices, dtype=float)
    n = len(pts)
    if n < 3 or len(radii) != n:
        raise ValueError("need >= 3 vertices and one radius per vertex")
    arcs = []
    for i in range(n):
        p_prev, p, p_next = pts[i - 1], pts[i], pts[(i + 1) % n]
        u1 = (p - p_prev) / np.linalg.norm(p - p_prev)
        u2 = (p_next - p) / np.linalg.norm(p_next - p)
        theta = math.atan2(u1[0] * u2[1] - u1[1] * u2[0], float(np.dot(u1, u2)))
        r = float(radii[i])
        tangent = r * math.tan(abs(theta) / 2.0)
        start = p - u1 * tangent
        end = p + u2 * tangent
        n1 = np.array([-u1[1], u1[0]])
        centre = start + n1 * r * math.copysign(1.0, theta)
        arcs.append({"start": start, "end": end, "centre": centre, "theta": theta, "radius": r, "tangent": tangent})
    for i in range(n):
        edge_len = float(np.linalg.norm(pts[(i + 1) % n] - pts[i]))
        straight = edge_len - arcs[i]["tangent"] - arcs[(i + 1) % n]["tangent"]
        if straight < -1e-6:
            raise ValueError(f"edge {i}: fillets overlap (straight part {straight:.3f} m)")
        arcs[i]["straight_after"] = max(0.0, straight)

    samples: list[np.ndarray] = []
    info: list[dict[str, float]] = []
    s_acc = 0.0
    # Start in the middle of the straight part of edge 0.
    a0, a1 = arcs[0], arcs[1 % n]
    mid0 = 0.5 * (a0["end"] + a1["start"])
    order = [(-1, "straight_half")]
    for i in range(1, n + 1):
        order.append((i % n, "arc"))
        order.append((i % n, "straight" if i < n else "straight_half_end"))
    for idx, kind in order:
        if kind == "straight_half":
            p_from, p_to = mid0, a1["start"]
        elif kind == "straight_half_end":
            p_from, p_to = arcs[0]["end"], mid0
        elif kind == "straight":
            p_from, p_to = arcs[idx]["end"], arcs[(idx + 1) % n]["start"]
        else:
            arc = arcs[idx]
            r, theta = arc["radius"], arc["theta"]
            c = arc["centre"]
            phi0 = math.atan2(arc["start"][1] - c[1], arc["start"][0] - c[0])
            arc_len = r * abs(theta)
            m = max(2, int(math.ceil(arc_len / ds)))
            phis = phi0 + theta * np.arange(m) / m
            samples.append(np.column_stack((c[0] + r * np.cos(phis), c[1] + r * np.sin(phis))))
            info.append({"vertex": idx, "s_start": s_acc, "s_mid": s_acc + 0.5 * arc_len, "s_end": s_acc + arc_len, "radius": r, "angle_deg": math.degrees(theta)})
            s_acc += arc_len
            continue
        length = float(np.linalg.norm(p_to - p_from))
        m = int(math.ceil(length / ds))
        if m > 0:
            t = np.arange(m) / m
            samples.append(p_from[None, :] + t[:, None] * (p_to - p_from)[None, :])
        s_acc += length
    xy = np.vstack(samples)
    # Drop duplicated consecutive points.
    keep = np.ones(len(xy), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6
    path = path_from_points(xy[keep])
    return path, info


# ------------------------------------------------------------- track building
@dataclass
class Track:
    name: str
    family: str
    description: str
    centerline: Path2D
    reference: Path2D  # path the driver follows (centreline + avoidance offsets)
    width: np.ndarray  # corridor width along the centreline samples
    geometry: WorldGeometry
    arcs: list[dict[str, float]]
    spec: dict[str, Any]
    ground_mu: float = 1.2
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def min_radius(self) -> float:
        return float(min(a["radius"] for a in self.arcs))


def _bell_offsets(s: np.ndarray, length: float, bells: list[tuple[float, float, float]]) -> np.ndarray:
    """Sum of raised-cosine bells ``(s_centre, offset, half_length)`` (periodic)."""
    d = np.zeros_like(s)
    for sc, off, hl in bells:
        ds = np.angle(np.exp(1j * 2.0 * math.pi * (s - sc) / length)) * length / (2.0 * math.pi)
        mask = np.abs(ds) < hl
        d[mask] += off * 0.5 * (1.0 + np.cos(math.pi * ds[mask] / hl))
    return d


def _region_values(s: np.ndarray, arcs: list[dict[str, float]], values: list[Any]) -> list[Any]:
    """Assign a per-edge value to every sample.

    Edge ``i`` spans from the middle of arc ``i`` to the middle of arc ``i+1``.
    The path starts inside edge 0, so arc mids appear in the order
    1, 2, ..., n-1, 0 along the arc length.
    """
    by_vertex = {a["vertex"]: a["s_mid"] for a in arcs}
    n = len(values)
    order = list(range(1, n)) + [0]
    mids = np.array([by_vertex[v] for v in order])
    k = np.searchsorted(mids, s, side="right")
    edges = np.where(k == 0, 0, np.array(order)[np.maximum(k - 1, 0)])
    return [values[int(e)] for e in edges]


def _width_profile(path: Path2D, arcs: list[dict[str, float]], widths: list[float]) -> np.ndarray:
    """Width is prescribed at every arc middle and linearly interpolated."""
    n = len(widths)
    mids = np.array([next(a["s_mid"] for a in arcs if a["vertex"] == i) for i in range(n)])
    order = np.argsort(mids)
    s_k = mids[order]
    w_k = np.asarray(widths, dtype=float)[order]
    s_ext = np.concatenate((s_k - path.length, s_k, s_k + path.length))
    w_ext = np.concatenate((w_k, w_k, w_k))
    return np.interp(path.s, s_ext, w_ext)


def _simplify(points: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker on an open polyline."""
    if len(points) < 3:
        return points
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = points[b] - points[a]
        seg_len = np.linalg.norm(seg)
        rel = points[a + 1 : b] - points[a]
        if seg_len < 1e-9:
            dist = np.linalg.norm(rel, axis=1)
        else:
            dist = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / seg_len
        k = int(np.argmax(dist))
        if dist[k] > tol:
            m = a + 1 + k
            keep[m] = True
            stack.append((a, m))
            stack.append((m, b))
    return points[keep]


def _polyline_boxes(points: np.ndarray, height: float, thickness: float, kind: str, tag: str, max_seg: float = 3.0) -> list[Box]:
    boxes = []
    simplified = _simplify(points, 0.008)
    for i in range(len(simplified) - 1):
        a, b = simplified[i], simplified[i + 1]
        length = float(np.linalg.norm(b - a))
        if length < 1e-4:
            continue
        pieces = max(1, int(math.ceil(length / max_seg)))
        for k in range(pieces):
            pa = a + (b - a) * k / pieces
            pb = a + (b - a) * (k + 1) / pieces
            mid = 0.5 * (pa + pb)
            seg_len = float(np.linalg.norm(pb - pa))
            yaw = math.atan2(b[1] - a[1], b[0] - a[0])
            boxes.append(Box((float(mid[0]), float(mid[1]), 0.5 * height), (seg_len + thickness, thickness, height), yaw, kind, f"{tag}_{i}_{k}"))
    return boxes


def build_track(name: str, spec: dict[str, Any]) -> Track:
    cl, arcs = rounded_polygon(spec["vertices"], spec["radii"], ds=0.02)
    if spec.get("reverse_definition", False):
        raise ValueError("reverse_definition is not supported; reverse the vertex list")
    n_v = len(spec["vertices"])
    widths = spec.get("widths", [1.8] * n_v)
    width = _width_profile(cl, arcs, widths)
    normal = cl.normal
    geometry = WorldGeometry()
    rng = np.random.default_rng(int(spec.get("decoration_seed", 0)))

    # --- walls / curbs per edge and side, with openings (alcoves).
    default_style = spec.get("wall_style", {"left": "wall", "right": "wall"})
    edge_styles = spec.get("edge_styles", {})
    styles_left = [edge_styles.get(i, edge_styles.get(str(i), {})).get("left", default_style["left"]) for i in range(n_v)]
    styles_right = [edge_styles.get(i, edge_styles.get(str(i), {})).get("right", default_style["right"]) for i in range(n_v)]
    region_left = _region_values(cl.s, arcs, styles_left)
    region_right = _region_values(cl.s, arcs, styles_right)
    openings = spec.get("openings", [])  # [{s: [s0, s1], side: left|right, depth: m}]
    open_mask = {"left": np.zeros(len(cl.s), bool), "right": np.zeros(len(cl.s), bool)}
    for op in openings:
        s0, s1 = op["s"]
        open_mask[op["side"]] |= (cl.s >= s0) & (cl.s <= s1)

    for side, sign, region in (("left", 1.0, region_left), ("right", -1.0, region_right)):
        styles = np.array(region, dtype=object)
        styles[open_mask[side]] = "none"
        thickness = np.where(styles == "curb", CURB_THICKNESS, WALL_THICKNESS)
        offset = 0.5 * width + 0.5 * thickness
        pts = cl.xy + sign * offset[:, None] * normal
        # Split into runs of identical style (the path is closed: rotate so that
        # a run boundary sits at index 0 when possible).
        change = np.nonzero(styles != np.roll(styles, 1))[0]
        if len(change) == 0:
            runs = [(0, len(styles))]
            closed = True
        else:
            closed = False
            start = change[0]
            idx = np.roll(np.arange(len(styles)), -start)
            styles_r = styles[idx]
            bounds = np.nonzero(styles_r != np.roll(styles_r, 1))[0].tolist() + [len(styles)]
            runs = [(int(idx[bounds[k]]), bounds[k + 1] - bounds[k]) for k in range(len(bounds) - 1)]
        for r_i, (start, count) in enumerate(runs):
            ids = (start + np.arange(count + (1 if not closed else 0))) % len(styles)
            style = styles[ids[0]]
            if style == "none":
                continue
            seg_pts = pts[ids]
            if closed:
                seg_pts = np.vstack((seg_pts, seg_pts[:1]))
            height = CURB_HEIGHT if style == "curb" else WALL_HEIGHT
            thick = CURB_THICKNESS if style == "curb" else WALL_THICKNESS
            kind = KIND_CURB if style == "curb" else KIND_WALL
            geometry.boxes.extend(_polyline_boxes(seg_pts, height, thick, kind, f"{side}{r_i}"))

    # Alcoves behind openings: two side walls and an end wall.
    for k, op in enumerate(openings):
        s0, s1 = op["s"]
        depth = float(op.get("depth", 1.5))
        sign = 1.0 if op["side"] == "left" else -1.0
        p0, p1 = cl.interp(s0), cl.interp(s1)
        h = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        n_vec = sign * np.array([-math.sin(h), math.cos(h)])
        w0 = float(cl.interp_scalar(width, s0)) / 2.0 + WALL_THICKNESS
        w1 = float(cl.interp_scalar(width, s1)) / 2.0 + WALL_THICKNESS
        a0 = p0 + n_vec * w0
        a1 = p1 + n_vec * w1
        b0 = a0 + n_vec * depth
        b1 = a1 + n_vec * depth
        for tag, (u, v) in (("a", (a0, b0)), ("b", (a1, b1)), ("c", (b0, b1))):
            geometry.boxes.extend(_polyline_boxes(np.vstack((u, v)), WALL_HEIGHT, WALL_THICKNESS, KIND_WALL, f"alcove{k}{tag}"))

    # --- in-lane obstacles and reference-path offsets.
    bells: list[tuple[float, float, float]] = []
    for sl in spec.get("slalom", []):
        s_pos = float(sl["s_start"])
        spacings = sl["spacings"]
        amp = float(sl.get("amplitude", 0.42))
        sign = 1.0 if sl.get("first_side", "left") == "left" else -1.0
        cone_s = [s_pos]
        for sp in spacings:
            s_pos += float(sp)
            cone_s.append(s_pos)
        for i, sc in enumerate(cone_s):
            prev_gap = spacings[i - 1] if i > 0 else spacings[0]
            next_gap = spacings[i] if i < len(spacings) else spacings[-1]
            hl = float(min(prev_gap, next_gap))
            # The path passes on alternating sides; the cone sits on the other side of the centreline.
            bells.append((sc, sign * amp, hl))
            lateral = -sign * float(sl.get("cone_lateral", 0.05))
            hd = _heading_at(cl, sc)
            p = cl.interp(sc) + lateral * np.array([-math.sin(hd), math.cos(hd)])
            geometry.cylinders.append(
                Cylinder((float(p[0]), float(p[1]), 0.5 * float(sl.get("cone_height", 0.35))), float(sl.get("cone_radius", 0.09)), float(sl.get("cone_height", 0.35)), KIND_OBSTACLE, f"cone_{len(geometry.cylinders)}")
            )
            sign = -sign
    for ob in spec.get("obstacles", []):
        sc = float(ob["s"])
        lateral = float(ob["lateral"])
        hd = _heading_at(cl, sc)
        p = cl.interp(sc) + lateral * np.array([-math.sin(hd), math.cos(hd)])
        if ob.get("shape", "cylinder") == "cylinder":
            geometry.cylinders.append(Cylinder((float(p[0]), float(p[1]), 0.5 * float(ob.get("height", 0.6))), float(ob.get("radius", 0.15)), float(ob.get("height", 0.6)), KIND_OBSTACLE, f"obst_{len(geometry.cylinders)}"))
        else:
            size = ob.get("size", [0.4, 0.4, 0.5])
            geometry.boxes.append(Box((float(p[0]), float(p[1]), 0.5 * size[2]), tuple(size), hd + float(ob.get("yaw", 0.0)), KIND_OBSTACLE, f"obst_box_{len(geometry.boxes)}"))
        if "avoid_offset" in ob:
            bells.append((sc, float(ob["avoid_offset"]), float(ob.get("avoid_half_length", 2.5))))
    offsets = _bell_offsets(cl.s, cl.length, bells) if bells else np.zeros_like(cl.s)
    ref_xy = cl.xy + offsets[:, None] * normal
    reference = path_from_points(ref_xy)

    # --- decoration, hall and ground.
    _add_decoration(geometry, cl, width, spec.get("decoration", {}), rng)
    _add_hall(geometry, cl, spec.get("hall", {}), rng)
    x_mid, y_mid = cl.xy.mean(axis=0)
    geometry.boxes.append(Box((float(x_mid), float(y_mid), -0.025), (80.0, 80.0, 0.05), 0.0, KIND_GROUND, "ground"))

    track = Track(
        name=name,
        family=spec.get("family", name),
        description=spec.get("description", ""),
        centerline=cl,
        reference=reference,
        width=width,
        geometry=geometry,
        arcs=arcs,
        spec=spec,
        ground_mu=float(spec.get("ground_mu", 1.2)),
    )
    track.meta = validate_track(track)
    return track


def _heading_at(path: Path2D, s: float) -> float:
    c = np.cos(path.heading)
    si = np.sin(path.heading)
    return math.atan2(float(path.interp_scalar(si, s)), float(path.interp_scalar(c, s)))


def _distance_to_polyline(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance of each point to a dense polyline (vertex distance, chunked)."""
    out = np.empty(len(points))
    for i in range(0, len(points), 256):
        d = np.linalg.norm(points[i : i + 256, None, :] - poly[None, ::2, :], axis=2)
        out[i : i + 256] = d.min(axis=1)
    return out


def _add_decoration(geometry: WorldGeometry, cl: Path2D, width: np.ndarray, spec: dict[str, Any], rng: np.random.Generator) -> None:
    count = int(spec.get("count", 40))
    margin = float(spec.get("margin", 0.45))
    lo = cl.xy.min(axis=0) - float(spec.get("extent", 3.0))
    hi = cl.xy.max(axis=0) + float(spec.get("extent", 3.0))
    half_w_max = float(width.max()) / 2.0 + WALL_THICKNESS
    placed: list[tuple[np.ndarray, float]] = []
    attempts = 0
    while len(placed) < count and attempts < count * 60:
        attempts += 1
        p = lo + rng.random(2) * (hi - lo)
        kind_roll = rng.random()
        if kind_roll < 0.45:
            size = np.array([rng.uniform(0.25, 1.4), rng.uniform(0.2, 0.9), rng.uniform(0.3, 1.1)])
            radius = 0.5 * math.hypot(size[0], size[1])
        else:
            radius = rng.uniform(0.07, 0.32)
            size = np.array([2 * radius, 2 * radius, rng.uniform(0.35, 1.2)])
        # Distance to the lane: centreline distance minus local half width.
        d_cl = _distance_to_polyline(p[None, :], cl.xy)[0]
        idx = int(np.argmin(np.linalg.norm(cl.xy - p, axis=1)))
        if d_cl - radius < 0.5 * width[idx] + WALL_THICKNESS + margin:
            continue
        if d_cl - radius < half_w_max + margin and d_cl < 0.5 * width[idx] + 1.0:
            continue
        if any(np.linalg.norm(p - q) < radius + rq + 0.25 for q, rq in placed):
            continue
        placed.append((p, radius))
        if kind_roll < 0.45:
            yaw = float(rng.uniform(-math.pi, math.pi))
            geometry.boxes.append(Box((float(p[0]), float(p[1]), 0.5 * size[2]), tuple(float(v) for v in size), yaw, KIND_FEATURE, f"feat_box_{len(placed)}"))
        else:
            geometry.cylinders.append(Cylinder((float(p[0]), float(p[1]), 0.5 * size[2]), float(radius), float(size[2]), KIND_FEATURE, f"feat_cyl_{len(placed)}"))
    for k, fx in enumerate(spec.get("explicit", [])):
        if fx["shape"] == "box":
            size = fx["size"]
            geometry.boxes.append(Box((fx["xy"][0], fx["xy"][1], 0.5 * size[2]), tuple(size), math.radians(fx.get("yaw_deg", 0.0)), KIND_FEATURE, f"landmark_{k}"))
        else:
            geometry.cylinders.append(Cylinder((fx["xy"][0], fx["xy"][1], 0.5 * fx["height"]), fx["radius"], fx["height"], KIND_FEATURE, f"landmark_{k}"))


def _add_hall(geometry: WorldGeometry, cl: Path2D, spec: dict[str, Any], rng: np.random.Generator) -> None:
    if spec.get("enabled", True) is False:
        return
    margin = float(spec.get("margin", 3.5))
    lo = cl.xy.min(axis=0) - margin
    hi = cl.xy.max(axis=0) + margin
    h = float(spec.get("height", 1.2))
    t = 0.15
    corners = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
    for i in range(4):
        a = np.array(corners[i])
        b = np.array(corners[(i + 1) % 4])
        # Split each hall wall in two with a random recess (door / niche).
        length = float(np.linalg.norm(b - a))
        u = (b - a) / length
        n_in = np.array([-u[1], u[0]])  # corners are counter-clockwise: left normal points inside
        door_c = rng.uniform(0.25, 0.75) * length
        door_w = rng.uniform(0.9, 1.6)
        for seg_a, seg_b in ((0.0, door_c - door_w / 2), (door_c + door_w / 2, length)):
            p0, p1 = a + u * seg_a, a + u * seg_b
            mid = 0.5 * (p0 + p1)
            geometry.boxes.append(Box((float(mid[0]), float(mid[1]), h / 2), (float(seg_b - seg_a), t, h), math.atan2(u[1], u[0]), KIND_ROOM, f"hall_{i}_{int(seg_a*10)}"))
        # Niche behind the door.
        back = a + u * door_c - n_in * 0.8
        geometry.boxes.append(Box((float(back[0]), float(back[1]), h / 2), (door_w + 0.3, t, h), math.atan2(u[1], u[0]), KIND_ROOM, f"hall_niche_{i}"))
        for side in (-1, 1):
            p = a + u * (door_c + side * (door_w / 2 + 0.075)) - n_in * 0.4
            geometry.boxes.append(Box((float(p[0]), float(p[1]), h / 2), (t, 0.8, h), math.atan2(u[1], u[0]), KIND_ROOM, f"hall_jamb_{i}_{side}"))
        # Columns along the wall at irregular spacing.
        pos = rng.uniform(1.0, 2.5)
        while pos < length - 1.0:
            if abs(pos - door_c) > door_w:
                p = a + u * pos + n_in * 0.35
                geometry.cylinders.append(Cylinder((float(p[0]), float(p[1]), h / 2), float(rng.uniform(0.12, 0.22)), h, KIND_ROOM, f"column_{i}_{int(pos*10)}"))
            pos += rng.uniform(2.5, 5.0)


def validate_track(track: Track) -> dict[str, Any]:
    """Geometric checks; raises ``ValueError`` when the track is not drivable."""
    cl, ref = track.centerline, track.reference
    problems = []
    # Inner wall must not fold on itself.
    for arc in track.arcs:
        s_mask = (cl.s >= arc["s_start"]) & (cl.s <= arc["s_end"])
        w_arc = float(track.width[s_mask].max()) if np.any(s_mask) else 0.0
        if arc["radius"] < 0.5 * w_arc + WALL_THICKNESS + 0.15:
            problems.append(f"vertex {arc['vertex']}: radius {arc['radius']} too small for width {w_arc:.2f}")
    # Distant parts of the track must not overlap.
    step = 5
    pts = cl.xy[::step]
    s = cl.s[::step]
    w = track.width[::step]
    for i in range(len(pts)):
        ds = np.abs(s - s[i])
        ds = np.minimum(ds, cl.length - ds)
        far = ds > 0.5 * math.pi * (w + w[i]) + 1.0
        if not np.any(far):
            continue
        d = np.linalg.norm(pts[far] - pts[i], axis=1)
        need = 0.5 * (w[far] + w[i]) + 2 * WALL_THICKNESS + 0.3
        if np.any(d < need):
            problems.append(f"lane overlap near s={s[i]:.1f} m")
            break
    # Reference path curvature must be below the steering limit with margin.
    kappa = np.abs(ref.curvature)
    # Smooth numerical curvature over ~10 cm.
    k = np.convolve(np.concatenate((kappa[-5:], kappa, kappa[:5])), np.ones(5) / 5, mode="same")[5:-5]
    r_min_allowed = float(track.spec.get("min_path_radius_m", 1.15))
    if float(k.max()) > 1.0 / r_min_allowed:
        problems.append(f"reference path radius {1.0 / k.max():.2f} m below {r_min_allowed} m")
    # Reference path must keep clearance to walls and obstacles.
    offsets = np.linalg.norm(ref.xy - cl.xy, axis=1)
    lateral_room = 0.5 * track.width - offsets
    if float(lateral_room.min()) < 0.30:
        problems.append(f"reference path only {lateral_room.min():.2f} m from the lane edge")
    for c in track.geometry.cylinders:
        if c.kind != KIND_OBSTACLE:
            continue
        d = float(np.min(np.linalg.norm(ref.xy - np.array(c.center[:2]), axis=1)))
        if d < c.radius + 0.30:
            problems.append(f"reference path passes {d - c.radius:.2f} m from obstacle {c.name}")
    if problems:
        raise ValueError(f"track {track.name}: " + "; ".join(problems))
    counts: dict[str, int] = {}
    for b in track.geometry.boxes:
        counts[b.kind] = counts.get(b.kind, 0) + 1
    for c in track.geometry.cylinders:
        counts[c.kind] = counts.get(c.kind, 0) + 1
    return {
        "centerline_length_m": round(cl.length, 3),
        "reference_length_m": round(ref.length, 3),
        "min_arc_radius_m": round(track.min_radius, 3),
        "min_reference_radius_m": round(1.0 / float(k.max()), 3),
        "width_min_m": round(float(track.width.min()), 3),
        "width_max_m": round(float(track.width.max()), 3),
        "turn_total_deg": round(sum(a["angle_deg"] for a in track.arcs), 1),
        "arcs": [{k2: round(v, 3) if isinstance(v, float) else v for k2, v in a.items()} for a in track.arcs],
        "object_counts": counts,
    }


# ---------------------------------------------------------------- registry
def load_track_specs(path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = path or CONFIG_DIR / "tracks.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["tracks"]


def load_track(name: str, specs: dict[str, dict[str, Any]] | None = None) -> Track:
    specs = specs or load_track_specs()
    if name not in specs:
        raise KeyError(f"unknown track {name!r}; known: {sorted(specs)}")
    return build_track(name, specs[name])


def export_track(track: Track, out_dir: Path) -> dict[str, Path]:
    """Write ``world.sdf`` (openable with ``gz sim``), ``track.json`` and
    ``reference_path.csv`` for one track."""
    from .geometry import world_sdf, write_xml

    out_dir.mkdir(parents=True, exist_ok=True)
    world_path = out_dir / "world.sdf"
    write_xml(world_sdf(track.geometry, ground_mu=track.ground_mu), world_path)
    meta = {"name": track.name, "family": track.family, "description": track.description, **track.meta}
    (out_dir / "track.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.savetxt(
        out_dir / "reference_path.csv",
        np.column_stack((track.reference.s, track.reference.xy, track.reference.heading, track.reference.curvature)),
        delimiter=",",
        header="s_m,x_m,y_m,heading_rad,curvature_1pm",
        comments="",
        fmt="%.5f",
    )
    return {"world": world_path, "meta": out_dir / "track.json"}


def plot_tracks(tracks: list[Track], path: Path, cols: int = 3, dpi: int = 110) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    from .geometry import rect_corners

    rows = int(math.ceil(len(tracks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 5.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    colors = {KIND_WALL: "#444444", KIND_CURB: "#e8590c", KIND_OBSTACLE: "#f08c00", KIND_FEATURE: "#1971c2", KIND_ROOM: "#868e96"}
    for ax, tr in zip(axes, tracks):
        for b in tr.geometry.boxes:
            if b.kind == KIND_GROUND:
                continue
            ax.add_patch(Polygon(rect_corners(b.center[0], b.center[1], b.yaw, b.size[0], b.size[1]), closed=True, color=colors.get(b.kind, "k"), lw=0))
        for c in tr.geometry.cylinders:
            ax.add_patch(Circle(c.center[:2], c.radius, color=colors.get(c.kind, "k"), lw=0))
        ax.plot(tr.reference.xy[:, 0], tr.reference.xy[:, 1], color="#2f9e44", lw=1.2)
        ax.plot(*tr.reference.xy[0], marker="o", color="#2f9e44")
        ax.set_title(f"{tr.name} ({tr.meta['reference_length_m']:.1f} m, Rmin {tr.meta['min_reference_radius_m']:.2f} m)", fontsize=10)
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.grid(alpha=0.2)
    for ax in axes[len(tracks) :]:
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
