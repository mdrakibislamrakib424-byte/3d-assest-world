"""
04_nature.py
Step 4 of the Dubai 3D City Generator: ground, grass, water, trees and lamps.

Reads roads.json, buildings.json, water.json, nature.json and
street_lamps.json from data/<city>/ and builds, for every tile, a GLB file
containing one mesh per material:
    ground      : sand-coloured ground plane (with holes for water)
    grass       : parks, gardens, grass land (roads cut out of it)
    water       : lakes and canals (slightly below ground level)
    trunk       : tree trunks
    palm_leaf   : date-palm fronds
    foliage_a/b : tree canopies (two greens for variety)
    lamp_pole   : street-lamp pole and arm
    lamp_light  : street-lamp head (emissive, glows at night)

Trees are placed in rows beside roads (large, medium and small trees; many
date palms), scattered inside parks, and taken from OSM tree nodes. Trees and
lamps never overlap buildings, roads or water. Everything is deterministic:
running the script twice gives the same city.

It reuses the geometry helpers of 02_roads.py and 03_buildings.py, so those
files must stay in the same folder.

World axes (shared by all scripts and Godot):
    +X = east, +Y = up, +Z = south. Origin = centre of config.BBOX.

Usage:
    python 04_nature.py

Requires:
    pip install numpy trimesh shapely mapbox-earcut
"""

import importlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from shapely.errors import GEOSException
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree
from trimesh.visual.material import PBRMaterial

import config

# 02_roads.py and 03_buildings.py start with a digit, so they cannot be used
# in a normal "import" statement; importlib loads them by name instead.
try:
    roads_mod = importlib.import_module("02_roads")
    buildings_mod = importlib.import_module("03_buildings")
except ImportError as import_error:
    sys.exit(f"[error] cannot load helper scripts: {import_error}")

MeshBuffer = buildings_mod.MeshBuffer
TileBuild = buildings_mod.TileBuild

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Heights (metres). Ground is lowest, roads (0.02) and sidewalks (0.15) sit
# above it, so no two surfaces share the same height (no flickering).
GROUND_Y: float = -0.10
GRASS_Y: float = 0.0
WATER_Y: float = -0.35

MIN_PART_AREA_M2: float = 0.5            # ignore slivers left by clipping

# Roadside planting.
MAJOR_ROAD_WIDTH_M: float = 14.0
MID_ROAD_WIDTH_M: float = 9.0
TREE_SETBACK_M: float = 1.3              # gap between sidewalk and tree row
VERGE_OFFSET_M: float = 5.0              # tree row distance from roads without sidewalk
MIN_VERGE_ROAD_WIDTH_M: float = 12.0     # narrower sidewalk-less roads get no trees
VERGE_SPACING_M: float = 24.0
TREE_SPACING_M: Dict[str, float] = {"major": 16.0, "mid": 13.0, "minor": 15.0}
TREE_SKIP_PROBABILITY: float = 0.10      # keeps rows from looking machine-made
TREE_ALONG_JITTER_M: float = 1.5
MIN_TREE_DISTANCE_M: float = 3.0
TREE_ROAD_CLEARANCE_M: float = 0.8
TREE_SOLID_CLEARANCE_M: float = 1.5      # distance kept from buildings and water
MAX_TREES_PER_AREA: int = 400

# Street lamps.
LAMP_MIN_ROAD_WIDTH_M: float = 9.0
LAMP_SPACING_M: float = 32.0
LAMP_OFFSET_M: float = 0.8               # from the road edge, on the sidewalk
LAMP_MIN_DISTANCE_M: float = 8.0
LAMP_ROAD_CLEARANCE_M: float = 0.15
LAMP_SOLID_CLEARANCE_M: float = 0.5

# Water.
WATERWAY_WIDTHS_M: Dict[str, float] = {
    "river": 20.0, "canal": 14.0, "stream": 3.0, "drain": 2.0, "ditch": 2.0,
}

# ---------------------------------------------------------------------------
# Materials (simple PBR colours; materials.py can replace them later)
# ---------------------------------------------------------------------------
MATERIALS: Dict[str, PBRMaterial] = {
    "ground": PBRMaterial(name="ground", baseColorFactor=[218, 200, 165, 255],
                          roughnessFactor=1.0, metallicFactor=0.0),
    "grass": PBRMaterial(name="grass", baseColorFactor=[76, 120, 52, 255],
                         roughnessFactor=1.0, metallicFactor=0.0),
    "water": PBRMaterial(name="water", baseColorFactor=[28, 86, 120, 255],
                         roughnessFactor=0.05, metallicFactor=0.1),
    "trunk": PBRMaterial(name="trunk", baseColorFactor=[92, 68, 46, 255],
                         roughnessFactor=0.95, metallicFactor=0.0),
    "palm_leaf": PBRMaterial(name="palm_leaf", baseColorFactor=[58, 110, 48, 255],
                             roughnessFactor=0.8, metallicFactor=0.0, doubleSided=True),
    "foliage_a": PBRMaterial(name="foliage_a", baseColorFactor=[52, 102, 48, 255],
                             roughnessFactor=0.9, metallicFactor=0.0),
    "foliage_b": PBRMaterial(name="foliage_b", baseColorFactor=[74, 124, 58, 255],
                             roughnessFactor=0.9, metallicFactor=0.0),
    "lamp_pole": PBRMaterial(name="lamp_pole", baseColorFactor=[70, 72, 76, 255],
                             roughnessFactor=0.5, metallicFactor=0.6),
    "lamp_light": PBRMaterial(name="lamp_light", baseColorFactor=[255, 240, 200, 255],
                              roughnessFactor=0.4, metallicFactor=0.0,
                              emissiveFactor=[1.0, 0.85, 0.55]),
}

Part = Tuple[np.ndarray, np.ndarray]      # (vertices Nx3, faces Mx3)
Template = Dict[str, Part]                # material name -> geometry


# ---------------------------------------------------------------------------
# Procedural shapes (low-poly, built once, copied for every tree or lamp)
# ---------------------------------------------------------------------------
def make_tube(rings: Sequence[Tuple[float, float, float, float]], sides: int) -> Part:
    """
    Open tapered tube. rings = (y, centre_x, centre_z, radius), bottom to top.
    Faces point outward.
    """
    vertices: List[Tuple[float, float, float]] = []
    for y, cx, cz, radius in rings:
        for i in range(sides):
            angle = 2.0 * math.pi * i / sides
            vertices.append((cx + radius * math.cos(angle), y, cz - radius * math.sin(angle)))
    faces: List[Tuple[int, int, int]] = []
    for ring in range(len(rings) - 1):
        for i in range(sides):
            j = (i + 1) % sides
            a = ring * sides + i
            b = ring * sides + j
            c = (ring + 1) * sides + j
            d = (ring + 1) * sides + i
            faces.append((a, b, c))
            faces.append((a, c, d))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def make_box(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float) -> Part:
    """Closed axis-aligned box with outward faces."""
    mesh = trimesh.creation.box(extents=(x1 - x0, y1 - y0, z1 - z0))
    centre = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0])
    return np.asarray(mesh.vertices) + centre, np.asarray(mesh.faces, dtype=np.int64)


def make_blob(centre: Tuple[float, float, float], radii: Tuple[float, float, float],
              subdivisions: int) -> Part:
    """Low-poly ellipsoid used for tree canopies and bushes."""
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    vertices = np.asarray(mesh.vertices) * np.asarray(radii) + np.asarray(centre)
    return vertices, np.asarray(mesh.faces, dtype=np.int64)


def make_fronds(top: Tuple[float, float, float], count: int, length: float,
                half_width: float) -> Part:
    """Palm fronds: flat, slightly drooping leaves radiating from the crown."""
    vertices: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    tx, ty, tz = top
    for k in range(count):
        angle = 2.0 * math.pi * k / count + 0.3
        dx, dz = math.cos(angle), -math.sin(angle)
        sx, sz = -dz, dx                                   # sideways direction
        base = len(vertices)
        vertices.extend([
            (tx, ty, tz),
            (tx + dx * length * 0.5 + sx * half_width, ty + 0.04, tz + dz * length * 0.5 + sz * half_width),
            (tx + dx * length * 0.5 - sx * half_width, ty + 0.04, tz + dz * length * 0.5 - sz * half_width),
            (tx + dx * length, ty - 0.07, tz + dz * length),
        ])
        faces.append((base, base + 1, base + 2))
        faces.append((base + 1, base + 3, base + 2))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def merge_parts(parts: Sequence[Part]) -> Part:
    """Combine several parts into one (same material)."""
    vertices: List[np.ndarray] = []
    faces: List[np.ndarray] = []
    offset = 0
    for part_vertices, part_faces in parts:
        vertices.append(part_vertices)
        faces.append(part_faces + offset)
        offset += len(part_vertices)
    return np.vstack(vertices), np.vstack(faces)


def build_templates() -> Dict[str, Template]:
    """Unit-height tree templates and metre-sized lamp template."""
    palm: Template = {
        "trunk": make_tube([(0.0, 0.0, 0.0, 0.028), (0.55, 0.03, 0.0, 0.022),
                            (1.0, 0.07, 0.0, 0.020)], sides=5),
        "palm_leaf": make_fronds((0.07, 1.0, 0.0), count=8, length=0.38, half_width=0.05),
    }
    shade: Template = {
        "trunk": make_tube([(0.0, 0.0, 0.0, 0.026), (0.5, 0.0, 0.0, 0.018)], sides=5),
        "foliage": make_blob((0.0, 0.72, 0.0), (0.32, 0.28, 0.32), subdivisions=1),
    }
    small: Template = {
        "trunk": make_tube([(0.0, 0.0, 0.0, 0.020), (0.35, 0.0, 0.0, 0.014)], sides=4),
        "foliage": make_blob((0.0, 0.62, 0.0), (0.30, 0.32, 0.30), subdivisions=0),
    }
    lamp: Template = {
        "lamp_pole": merge_parts([
            make_tube([(0.0, 0.0, 0.0, 0.09), (6.5, 0.0, 0.0, 0.06)], sides=6),
            make_box(0.0, 6.35, -0.04, 1.3, 6.50, 0.04),     # arm toward +x
        ]),
        "lamp_light": make_box(1.0, 6.25, -0.15, 1.6, 6.35, 0.15),
    }
    return {"palm": palm, "shade": shade, "small": small, "lamp": lamp}


@dataclass(frozen=True)
class TreeKind:
    """One tree size class: which template, height range, crown width factor."""
    template: str
    min_height: float
    max_height: float
    crown: Tuple[float, float]


KINDS: Dict[str, TreeKind] = {
    "palm_large": TreeKind("palm", 9.0, 13.0, (1.0, 1.0)),
    "palm_medium": TreeKind("palm", 5.5, 8.5, (1.0, 1.0)),
    "shade_large": TreeKind("shade", 8.0, 12.0, (0.9, 1.25)),
    "shade_medium": TreeKind("shade", 5.0, 7.5, (0.9, 1.2)),
    "small": TreeKind("small", 2.2, 4.0, (0.9, 1.2)),
}

# Tree mix (kind, weight) by road class and by area type.
ROAD_MIX: Dict[str, List[Tuple[str, float]]] = {
    "major": [("palm_large", 0.45), ("palm_medium", 0.30), ("shade_large", 0.25)],
    "mid": [("palm_medium", 0.40), ("shade_medium", 0.30), ("shade_large", 0.15), ("small", 0.15)],
    "minor": [("shade_medium", 0.35), ("small", 0.35), ("palm_medium", 0.30)],
}
AREA_MIX: Dict[str, List[Tuple[str, float]]] = {
    "park": [("palm_medium", 0.30), ("palm_large", 0.10), ("shade_medium", 0.30),
             ("shade_large", 0.15), ("small", 0.15)],
    "forest": [("shade_large", 0.40), ("shade_medium", 0.40), ("small", 0.20)],
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class RoadRecord:
    """A drivable road or path with its centre line and surface shape."""
    osm_id: int
    spec: "roads_mod.RoadSpec"
    points: np.ndarray
    corridor: BaseGeometry


@dataclass(frozen=True)
class NatureRule:
    """How to treat a nature area: tree spacing (None = grass only) and tree mix."""
    spacing: Optional[float]
    mix: str


@dataclass
class Stats:
    """Counters printed at the end of the run."""
    trees: Counter = field(default_factory=Counter)
    lamps: int = 0
    grass_m2: float = 0.0
    water_m2: float = 0.0
    roads_planted: int = 0
    warnings: int = 0


class Obstacles:
    """Fast 'is this spot free?' test against road surfaces, buildings and water."""

    def __init__(self, road_shapes: List[BaseGeometry], solid_shapes: List[BaseGeometry]) -> None:
        self._roads = STRtree(road_shapes) if road_shapes else None
        self._solids = STRtree(solid_shapes) if solid_shapes else None

    @staticmethod
    def _hit(index: Optional[STRtree], x: float, z: float, radius: float) -> bool:
        if index is None:
            return False
        probe = Point(x, z).buffer(radius, quad_segs=4) if radius > 0 else Point(x, z)
        return len(index.query(probe, predicate="intersects")) > 0

    def blocked(self, x: float, z: float, road_radius: float, solid_radius: float) -> bool:
        """True if (x, z) is on/too close to a road, building or water."""
        return (self._hit(self._roads, x, z, road_radius)
                or self._hit(self._solids, x, z, solid_radius))


class PlacedIndex:
    """Grid hash that stops objects of one kind from being placed too close."""

    def __init__(self, cell: float = 4.0) -> None:
        self._cell = cell
        self._cells: Dict[Tuple[int, int], List[Tuple[float, float]]] = defaultdict(list)

    def try_add(self, x: float, z: float, min_distance: float) -> bool:
        """Register the point unless another one is closer than min_distance."""
        cx, cz = math.floor(x / self._cell), math.floor(z / self._cell)
        reach = math.ceil(min_distance / self._cell)
        for ix in range(cx - reach, cx + reach + 1):
            for iz in range(cz - reach, cz + reach + 1):
                for px, pz in self._cells.get((ix, iz), ()):
                    if math.hypot(px - x, pz - z) < min_distance:
                        return False
        self._cells[(cx, cz)].append((x, z))
        return True


@dataclass
class World:
    """Everything the planting functions need, in one place."""
    tiles: Dict[Tuple[int, int], TileBuild]
    obstacles: Obstacles
    templates: Dict[str, Template]
    trees: PlacedIndex
    lamps: PlacedIndex
    stats: Stats


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_layer(name: str) -> List[dict]:
    """Read data/<city>/<name>.json. Raises a clear error if missing or broken."""
    path = config.DATA_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 01_download.py first.")
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file).get("elements", [])
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read {path}: {error}") from error


def collect_roads(elements: List[dict]) -> List[RoadRecord]:
    """Convert OSM roads into centre lines and surface shapes (bridges and tunnels skipped)."""
    records: List[RoadRecord] = []
    for element in elements:
        spec = roads_mod.classify(element.get("tags", {}))
        geometry = element.get("geometry")
        if spec is None or spec.layer > 0 or not geometry or len(geometry) < 2:
            continue
        try:
            points = roads_mod.clean_polyline(
                [roads_mod.project(p["lat"], p["lon"]) for p in geometry])
            if len(points) < 2:
                continue
            corridor = LineString(points).buffer(spec.width / 2.0, cap_style=2)
        except (KeyError, TypeError, ValueError, GEOSException):
            continue
        records.append(RoadRecord(int(element.get("id", 0)), spec, points, corridor))
    return records


def collect_buildings(elements: List[dict]) -> List[Polygon]:
    """Building footprints, built with the same function as 03_buildings.py."""
    footprints: List[Polygon] = []
    for element in elements:
        try:
            footprints.extend(buildings_mod.element_polygons(element))
        except (KeyError, TypeError, ValueError, GEOSException):
            continue
    return footprints


def collect_water(elements: List[dict]) -> List[BaseGeometry]:
    """Water shapes: lakes (polygons) and canals/rivers (centre lines with width)."""
    shapes: List[BaseGeometry] = []
    for element in elements:
        tags = element.get("tags", {})
        if tags.get("natural") == "coastline":
            continue                                    # no coastline in the start area
        try:
            waterway = tags.get("waterway")
            if waterway:
                points = buildings_mod.latlon_list(element.get("geometry"))
                width = WATERWAY_WIDTHS_M.get(waterway)
                if points and len(points) >= 2 and width:
                    line = LineString(buildings_mod.to_xz(points))
                    shapes.append(line.buffer(width / 2.0))
            elif tags.get("natural") == "water":
                shapes.extend(buildings_mod.element_polygons(element))
        except (KeyError, TypeError, ValueError, GEOSException):
            continue
    return shapes


def classify_nature(tags: Dict[str, str]) -> Optional[NatureRule]:
    """Decide how a park/grass/forest area is planted. None = not a nature area."""
    leisure, landuse = tags.get("leisure"), tags.get("landuse")
    if leisure in ("park", "garden"):
        return NatureRule(13.0, "park")
    if leisure == "pitch":
        return NatureRule(None, "park")
    if landuse == "grass":
        return NatureRule(22.0, "park")
    if landuse == "recreation_ground":
        return NatureRule(24.0, "park")
    if landuse == "forest":
        return NatureRule(7.0, "forest")
    return None


# ---------------------------------------------------------------------------
# Flat surfaces (ground, grass, water)
# ---------------------------------------------------------------------------
def tile_range(bounds: Tuple[float, float, float, float]) -> Iterator[Tuple[int, int]]:
    """Yield the (tx, tz) index of every tile touched by the bounds."""
    min_x, min_z, max_x, max_z = bounds
    size = config.TILE_SIZE_M
    for tx in range(math.floor(min_x / size), math.floor(max_x / size) + 1):
        for tz in range(math.floor(min_z / size), math.floor(max_z / size) + 1):
            yield tx, tz


def tile_box(tx: int, tz: int) -> Polygon:
    """The square footprint of one tile."""
    size = config.TILE_SIZE_M
    return box(tx * size, tz * size, (tx + 1) * size, (tz + 1) * size)


def flat_parts(geometry: BaseGeometry) -> List[Polygon]:
    """Flatten any shapely result into a list of usable polygons."""
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry] if geometry.area >= MIN_PART_AREA_M2 else []
    if hasattr(geometry, "geoms"):
        return [part for member in geometry.geoms for part in flat_parts(member)]
    return []


def add_flat_layer(tiles: Dict[Tuple[int, int], TileBuild], geometry: BaseGeometry,
                   material: str, y: float) -> int:
    """Clip a shape to tiles and add it as an upward-facing surface. Returns failures."""
    failures = 0
    if geometry.is_empty:
        return failures
    for tx, tz in tile_range(geometry.bounds):
        try:
            piece = geometry.intersection(tile_box(tx, tz))
            for polygon in flat_parts(piece):
                if not buildings_mod.add_roof(tiles[(tx, tz)].buffers[material], polygon, y):
                    failures += 1
        except (GEOSException, ValueError):
            failures += 1
    return failures


def build_ground(tiles: Dict[Tuple[int, int], TileBuild], water: BaseGeometry) -> int:
    """One ground surface per tile covering the whole BBOX, with holes for water."""
    corners = [roads_mod.project(lat, lon)
               for lat in (config.BBOX[0], config.BBOX[2])
               for lon in (config.BBOX[1], config.BBOX[3])]
    xs, zs = [c[0] for c in corners], [c[1] for c in corners]
    failures = 0
    for tx, tz in tile_range((min(xs), min(zs), max(xs), max(zs))):
        try:
            ground = tile_box(tx, tz)
            if not water.is_empty:
                ground = ground.difference(water)
            for polygon in flat_parts(ground):
                if not buildings_mod.add_roof(tiles[(tx, tz)].buffers["ground"], polygon, GROUND_Y):
                    failures += 1
        except (GEOSException, ValueError):
            failures += 1
    return failures


# ---------------------------------------------------------------------------
# Planting
# ---------------------------------------------------------------------------
def pick_kind(mix: Sequence[Tuple[str, float]], rng: random.Random) -> str:
    """Choose a tree kind using the given weights."""
    return rng.choices([kind for kind, _ in mix], weights=[w for _, w in mix])[0]


def place_template(tile: TileBuild, template: Template, x: float, y: float, z: float,
                   height: float, crown: float, yaw: float, foliage: str) -> None:
    """Copy a template into the tile at (x, y, z), scaled and rotated around Y."""
    cos_a, sin_a = math.cos(yaw), math.sin(yaw)
    scale = np.array([height * crown, height, height * crown])
    for part_name, (vertices, faces) in template.items():
        material = foliage if part_name == "foliage" else part_name
        scaled = vertices * scale
        world = np.column_stack([
            scaled[:, 0] * cos_a + scaled[:, 2] * sin_a + x,
            scaled[:, 1] + y,
            -scaled[:, 0] * sin_a + scaled[:, 2] * cos_a + z,
        ])
        tile.buffers[material].add_triangles(world, faces, np.zeros((len(world), 2)))


def plant_tree(world: World, kind_name: str, x: float, z: float, rng: random.Random) -> None:
    """Place one tree of the given kind with random height, crown and rotation."""
    kind = KINDS[kind_name]
    height = rng.uniform(kind.min_height, kind.max_height)
    crown = rng.uniform(*kind.crown)
    yaw = rng.uniform(0.0, 2.0 * math.pi)
    foliage = rng.choice(("foliage_a", "foliage_b"))
    tile = world.tiles[roads_mod.tile_key(x, z)]
    place_template(tile, world.templates[kind.template], x, 0.0, z, height, crown, yaw, foliage)
    world.stats.trees[kind_name] += 1


def plant_lamp(world: World, x: float, z: float, yaw: float) -> None:
    """Place one street lamp whose arm points along the yaw direction."""
    tile = world.tiles[roads_mod.tile_key(x, z)]
    place_template(tile, world.templates["lamp"], x, 0.0, z, 1.0, 1.0, yaw, "foliage_a")
    world.stats.lamps += 1


def sample_polyline(points: np.ndarray, spacing: float,
                    start: float) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (position, unit tangent) every `spacing` metres, the first at `start`."""
    next_at = start
    travelled = 0.0
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        length = float(np.linalg.norm(b - a))
        if length < 1e-9:
            continue
        tangent = (b - a) / length
        while next_at <= travelled + length:
            yield a + tangent * (next_at - travelled), tangent
            next_at += spacing
        travelled += length


def road_group(width: float) -> str:
    """Classify a road as major, mid or minor by its width."""
    if width >= MAJOR_ROAD_WIDTH_M:
        return "major"
    if width >= MID_ROAD_WIDTH_M:
        return "mid"
    return "minor"


def plant_road(world: World, record: RoadRecord) -> None:
    """Trees (and lamps on bigger roads) along both sides of one road."""
    spec = record.spec
    if spec.pedestrian:
        return
    if not spec.has_sidewalk and spec.width < MIN_VERGE_ROAD_WIDTH_M:
        return                                           # service lanes stay bare

    rng = random.Random(f"road:{record.osm_id}")
    half = spec.width / 2.0
    group = road_group(spec.width)
    if spec.has_sidewalk:
        tree_offset = half + config.SIDEWALK_WIDTH_M + TREE_SETBACK_M
        spacing = TREE_SPACING_M[group]
    else:
        tree_offset = half + VERGE_OFFSET_M
        spacing = VERGE_SPACING_M

    for side in (1.0, -1.0):
        for position, tangent in sample_polyline(record.points, spacing, rng.uniform(0.0, spacing)):
            if rng.random() < TREE_SKIP_PROBABILITY:
                continue
            normal = np.array([-tangent[1], tangent[0]]) * side
            spot = position + normal * tree_offset + tangent * rng.uniform(
                -TREE_ALONG_JITTER_M, TREE_ALONG_JITTER_M)
            x, z = float(spot[0]), float(spot[1])
            if world.obstacles.blocked(x, z, TREE_ROAD_CLEARANCE_M, TREE_SOLID_CLEARANCE_M):
                continue
            if not world.trees.try_add(x, z, MIN_TREE_DISTANCE_M):
                continue
            plant_tree(world, pick_kind(ROAD_MIX[group], rng), x, z, rng)

    if spec.has_sidewalk and spec.width >= LAMP_MIN_ROAD_WIDTH_M:
        for side in (1.0, -1.0):
            for position, tangent in sample_polyline(
                    record.points, LAMP_SPACING_M, rng.uniform(0.0, LAMP_SPACING_M)):
                normal = np.array([-tangent[1], tangent[0]]) * side
                spot = position + normal * (half + LAMP_OFFSET_M)
                x, z = float(spot[0]), float(spot[1])
                if world.obstacles.blocked(x, z, LAMP_ROAD_CLEARANCE_M, LAMP_SOLID_CLEARANCE_M):
                    continue
                if not world.lamps.try_add(x, z, LAMP_MIN_DISTANCE_M):
                    continue
                arm = -normal                            # arm reaches over the road
                plant_lamp(world, x, z, math.atan2(-float(arm[1]), float(arm[0])))
    world.stats.roads_planted += 1


def scatter_points(polygon: Polygon, spacing: float,
                   rng: random.Random) -> Iterator[Tuple[float, float]]:
    """Jittered grid of points inside a polygon (kept clear of its edge)."""
    inner = polygon.buffer(-1.5)
    if inner.is_empty:
        return
    min_x, min_z, max_x, max_z = inner.bounds
    x = min_x + spacing / 2.0
    while x < max_x:
        z = min_z + spacing / 2.0
        while z < max_z:
            px = x + rng.uniform(-0.3, 0.3) * spacing
            pz = z + rng.uniform(-0.3, 0.3) * spacing
            if inner.contains(Point(px, pz)):
                yield px, pz
            z += spacing
        x += spacing


def plant_area(world: World, polygon: Polygon, rule: NatureRule, rng: random.Random) -> None:
    """Scatter trees inside a park, garden or forest."""
    if rule.spacing is None:
        return
    planted = 0
    for x, z in scatter_points(polygon, rule.spacing, rng):
        if planted >= MAX_TREES_PER_AREA:
            break
        if world.obstacles.blocked(x, z, TREE_ROAD_CLEARANCE_M, TREE_SOLID_CLEARANCE_M):
            continue
        if not world.trees.try_add(x, z, MIN_TREE_DISTANCE_M):
            continue
        plant_tree(world, pick_kind(AREA_MIX[rule.mix], rng), x, z, rng)
        planted += 1


def explicit_tree_kind(tags: Dict[str, str], rng: random.Random) -> str:
    """Kind for a tree that is mapped in OSM (palms are recognised by their tags)."""
    species = (tags.get("species", "") + tags.get("genus", "")).lower()
    if "phoenix" in species or "palm" in species or tags.get("leaf_type") == "palm_leaved":
        return rng.choice(("palm_medium", "palm_large"))
    return rng.choice(("shade_medium", "shade_large", "small"))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_tiles(tiles: Dict[Tuple[int, int], TileBuild]) -> int:
    """Write one GLB per tile plus an index file. Returns the number of files."""
    out_dir = config.OUTPUT_DIR / "nature"
    out_dir.mkdir(parents=True, exist_ok=True)
    index: List[dict] = []

    for (tx, tz), tile in sorted(tiles.items()):
        scene = trimesh.Scene()
        for name, buffer in tile.buffers.items():
            if buffer.faces:
                scene.add_geometry(buffer.to_mesh(MATERIALS[name]),
                                   node_name=name, geom_name=name)
        triangles = tile.triangle_count()
        if triangles == 0:
            continue
        file_name = f"nature_{tx}_{tz}.glb"
        try:
            (out_dir / file_name).write_bytes(scene.export(file_type="glb"))
        except (OSError, ValueError) as error:
            print(f"[error] {file_name}: {error}")
            continue
        index.append({"tx": tx, "tz": tz, "file": file_name, "triangles": triangles})

    meta = {
        "tile_size_m": config.TILE_SIZE_M,
        "origin_lat": config.ORIGIN_LAT,
        "origin_lon": config.ORIGIN_LON,
        "tiles": index,
    }
    with open(out_dir / "nature_index.json", "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    return len(index)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    config.ensure_directories()
    try:
        road_elements = load_layer("roads")
        building_elements = load_layer("buildings")
        water_elements = load_layer("water")
        nature_elements = load_layer("nature")
        lamp_elements = load_layer("street_lamps")
    except (FileNotFoundError, RuntimeError) as error:
        print(f"[error] {error}")
        return 1

    print("Reading map data ...")
    roads = collect_roads(road_elements)
    footprints = collect_buildings(building_elements)
    water_shapes = collect_water(water_elements)
    water = unary_union(water_shapes).buffer(0) if water_shapes else Polygon()

    tiles: Dict[Tuple[int, int], TileBuild] = defaultdict(TileBuild)
    stats = Stats()

    # --- grass: all park/grass areas merged, with water and road surfaces cut out
    areas: List[Tuple[Polygon, NatureRule, str]] = []
    for element in nature_elements:
        rule = classify_nature(element.get("tags", {}))
        if rule is None or element.get("type") != "way":
            continue
        try:
            for polygon in buildings_mod.element_polygons(element):
                areas.append((polygon, rule, str(element.get("id", 0))))
        except (KeyError, TypeError, ValueError, GEOSException):
            stats.warnings += 1

    print("Building ground, grass and water ...")
    failures = build_ground(tiles, water)
    if areas:
        grass = unary_union([polygon for polygon, _, _ in areas]).buffer(0)
        road_surface = unary_union([r.corridor for r in roads if not r.spec.pedestrian])
        grass = grass.difference(road_surface)
        if not water.is_empty:
            grass = grass.difference(water)
        failures += add_flat_layer(tiles, grass, "grass", GRASS_Y)
        stats.grass_m2 = grass.area
    if not water.is_empty:
        failures += add_flat_layer(tiles, water, "water", WATER_Y)
        stats.water_m2 = water.area
    stats.warnings += failures

    # --- planting
    world = World(
        tiles=tiles,
        obstacles=Obstacles([r.corridor for r in roads],
                            list(footprints) + list(water_shapes)),
        templates=build_templates(),
        trees=PlacedIndex(),
        lamps=PlacedIndex(),
        stats=stats,
    )

    print("Planting mapped trees and lamps ...")
    for element in nature_elements:
        tags = element.get("tags", {})
        if element.get("type") == "node" and tags.get("natural") == "tree":
            rng = random.Random(f"tree:{element.get('id', 0)}")
            x, z = roads_mod.project(element["lat"], element["lon"])
            if (not world.obstacles.blocked(x, z, TREE_ROAD_CLEARANCE_M, TREE_SOLID_CLEARANCE_M)
                    and world.trees.try_add(x, z, MIN_TREE_DISTANCE_M)):
                plant_tree(world, explicit_tree_kind(tags, rng), x, z, rng)
    for element in lamp_elements:
        try:
            x, z = roads_mod.project(element["lat"], element["lon"])
        except KeyError:
            continue
        rng = random.Random(f"lamp:{element.get('id', 0)}")
        if (not world.obstacles.blocked(x, z, LAMP_ROAD_CLEARANCE_M, LAMP_SOLID_CLEARANCE_M)
                and world.lamps.try_add(x, z, LAMP_MIN_DISTANCE_M)):
            plant_lamp(world, x, z, rng.uniform(0.0, 2.0 * math.pi))

    print("Planting roadside trees and lamps ...")
    for record in roads:
        try:
            plant_road(world, record)
        except (ValueError, KeyError, IndexError, GEOSException) as error:
            stats.warnings += 1
            print(f"[warn] road {record.osm_id} skipped: {error}")

    print("Planting parks ...")
    for polygon, rule, osm_id in areas:
        try:
            plant_area(world, polygon, rule, random.Random(f"area:{osm_id}"))
        except (ValueError, GEOSException) as error:
            stats.warnings += 1
            print(f"[warn] area {osm_id} skipped: {error}")

    files = export_tiles(tiles)
    total_trees = sum(stats.trees.values())
    print(f"Trees planted : {total_trees}  {dict(stats.trees)}")
    print(f"Street lamps  : {stats.lamps}")
    print(f"Grass area    : {stats.grass_m2:,.0f} m2")
    print(f"Water area    : {stats.water_m2:,.0f} m2")
    print(f"Warnings      : {stats.warnings}")
    print(f"Tile files    : {files}  ->  {config.OUTPUT_DIR / 'nature'}")
    print("Nature OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
