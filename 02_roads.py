"""
02_roads.py
Step 2 of the Dubai 3D City Generator: build 3D road meshes.

Reads data/<city>/roads.json (OpenStreetMap, "out geom" format) and builds,
for every 250 m tile, a GLB file containing three meshes:
    - asphalt   : the driving surface
    - sidewalk  : footpaths beside roads, plus pedestrian-only ways
    - markings  : dashed centre lines on two-way roads

World axes (shared by all scripts and Godot):
    +X = east, +Y = up, +Z = south (so north is -Z).
    The origin is the centre of config.BBOX.

Usage:
    python 02_roads.py

Requires:
    pip install numpy trimesh
"""

import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EARTH_RADIUS_M: float = 6_378_137.0

PEDESTRIAN_TYPES = {"footway", "path", "pedestrian", "cycleway", "track", "bridleway"}
SKIPPED_TYPES = {
    "proposed", "construction", "corridor", "steps", "elevator",
    "platform", "raceway", "bus_guideway",
}
TUNNEL_VALUES = {"yes", "building_passage", "culvert"}
NO_SIDEWALK_TYPES = {"motorway", "trunk", "service"}

PEDESTRIAN_WIDTH_M: float = 2.5
LANE_WIDTH_M: float = 3.5
MIN_ROAD_WIDTH_M: float = 3.0
MAX_ROAD_WIDTH_M: float = 40.0
MARKING_MIN_ROAD_WIDTH_M: float = 6.0

# Vertical layout (metres above the future ground plane at y = 0).
ROAD_BASE_Y: float = 0.02
SIDEWALK_RISE_Y: float = 0.13
MARKING_RISE_Y: float = 0.02
LAYER_STEP_Y: float = 0.03        # extra height per OSM "layer" (bridges)

# Centre-line dashes.
MARKING_WIDTH_M: float = 0.15
DASH_LENGTH_M: float = 3.0
DASH_GAP_M: float = 6.0

UV_METRES: float = 4.0            # one texture repeat per 4 m (for later textures)
MAX_MITER_SCALE: float = 2.0      # limits spikes at sharp corners

# Simple PBR colours (0-255). materials.py will replace these with textures later.
MATERIALS: Dict[str, PBRMaterial] = {
    "asphalt": PBRMaterial(name="asphalt", baseColorFactor=[32, 32, 35, 255],
                           roughnessFactor=0.92, metallicFactor=0.0),
    "sidewalk": PBRMaterial(name="sidewalk", baseColorFactor=[150, 148, 142, 255],
                            roughnessFactor=0.85, metallicFactor=0.0),
    "markings": PBRMaterial(name="markings", baseColorFactor=[235, 235, 225, 255],
                            roughnessFactor=0.7, metallicFactor=0.0),
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RoadSpec:
    """Everything needed to build one road, derived from its OSM tags."""
    width: float
    pedestrian: bool
    has_sidewalk: bool
    has_markings: bool
    layer: int


@dataclass
class MeshBuffer:
    """Accumulates triangles for one material inside one tile."""
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    faces: List[Tuple[int, int, int]] = field(default_factory=list)

    def add_quad(self, a: Tuple[float, float, float], b: Tuple[float, float, float],
                 c: Tuple[float, float, float], d: Tuple[float, float, float]) -> None:
        """Add a quad given as a closed loop a-b-c-d (two triangles)."""
        base = len(self.vertices)
        self.vertices.extend((a, b, c, d))
        self.faces.append((base, base + 1, base + 2))
        self.faces.append((base, base + 2, base + 3))

    def to_mesh(self, material: PBRMaterial) -> trimesh.Trimesh:
        """Convert to a trimesh mesh with every face pointing up."""
        vertices = np.asarray(self.vertices, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.int64)

        # Roads are flat, so any face whose normal points down is flipped.
        tri = vertices[faces]
        normal_y = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 1]
        flip = normal_y < 0
        faces[flip] = faces[flip][:, [0, 2, 1]]

        uv = vertices[:, [0, 2]] / UV_METRES
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
        return mesh


@dataclass
class TileMeshes:
    """The three material buffers of a single tile."""
    asphalt: MeshBuffer = field(default_factory=MeshBuffer)
    sidewalk: MeshBuffer = field(default_factory=MeshBuffer)
    markings: MeshBuffer = field(default_factory=MeshBuffer)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def project(lat: float, lon: float) -> Tuple[float, float]:
    """Convert latitude/longitude to local metres (x east, z south)."""
    x = (math.radians(lon - config.ORIGIN_LON) * EARTH_RADIUS_M
         * math.cos(math.radians(config.ORIGIN_LAT)))
    z = -math.radians(lat - config.ORIGIN_LAT) * EARTH_RADIUS_M
    return x, z


def tile_key(x: float, z: float) -> Tuple[int, int]:
    """Return the (tx, tz) index of the tile containing the point."""
    return (math.floor(x / config.TILE_SIZE_M), math.floor(z / config.TILE_SIZE_M))


def clean_polyline(points: List[Tuple[float, float]]) -> np.ndarray:
    """Remove consecutive duplicate points (they break offset maths)."""
    arr = np.asarray(points, dtype=np.float64)
    if len(arr) < 2:
        return arr
    keep = [0]
    for i in range(1, len(arr)):
        if np.linalg.norm(arr[i] - arr[keep[-1]]) > 0.05:
            keep.append(i)
    return arr[keep]


def offset_line(points: np.ndarray, distance: float) -> np.ndarray:
    """
    Offset a polyline sideways by `distance` metres (negative = other side).
    Uses mitred joins so corners stay connected, with a cap on the spike length.
    """
    segment = np.diff(points, axis=0)
    length = np.linalg.norm(segment, axis=1)[:, None]
    direction = segment / length
    seg_normal = np.stack([-direction[:, 1], direction[:, 0]], axis=1)

    vertex_normal = np.empty_like(points)
    scale = np.ones(len(points))
    vertex_normal[0] = seg_normal[0]
    vertex_normal[-1] = seg_normal[-1]

    for i in range(1, len(points) - 1):
        mitre = seg_normal[i - 1] + seg_normal[i]
        norm = np.linalg.norm(mitre)
        if norm < 1e-6:                       # 180 degree turn
            vertex_normal[i] = seg_normal[i]
            continue
        mitre /= norm
        vertex_normal[i] = mitre
        cosine = float(np.dot(mitre, seg_normal[i]))
        scale[i] = min(1.0 / max(cosine, 0.5), MAX_MITER_SCALE)

    return points + vertex_normal * (distance * scale)[:, None]


def dash_segments(points: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield short (start, end) pieces that form a dashed line along the polyline."""
    period = DASH_LENGTH_M + DASH_GAP_M
    travelled = 0.0
    for i in range(len(points) - 1):
        start, end = points[i], points[i + 1]
        seg_len = float(np.linalg.norm(end - start))
        if seg_len < 1e-9:
            continue
        t = 0.0
        while t < seg_len - 1e-9:
            cycle = travelled % period
            if cycle < DASH_LENGTH_M:
                step = min(DASH_LENGTH_M - cycle, seg_len - t)
                a = start + (end - start) * (t / seg_len)
                b = start + (end - start) * ((t + step) / seg_len)
                yield a, b
            else:
                step = min(period - cycle, seg_len - t)
            step = max(step, 1e-6)
            t += step
            travelled += step


def v3(point: np.ndarray, y: float) -> Tuple[float, float, float]:
    """Make a 3D vertex from an (x, z) point and a height."""
    return (float(point[0]), y, float(point[1]))


# ---------------------------------------------------------------------------
# OSM tag handling
# ---------------------------------------------------------------------------
def parse_number(value: Optional[str]) -> Optional[float]:
    """Parse OSM numbers such as '12', '3.5' or '12 m'. Returns None if invalid."""
    if value is None:
        return None
    try:
        return float(str(value).split()[0].replace(",", "."))
    except (ValueError, IndexError):
        return None


def classify(tags: Dict[str, str]) -> Optional[RoadSpec]:
    """Decide how (and whether) to build a road. Returns None to skip it."""
    highway = tags.get("highway", "")
    if not highway or highway in SKIPPED_TYPES:
        return None
    if tags.get("tunnel") in TUNNEL_VALUES or tags.get("area") == "yes":
        return None

    base = highway.removesuffix("_link")
    layer = int(parse_number(tags.get("layer")) or 0)
    if tags.get("bridge") not in (None, "no"):
        layer = max(layer, 1)
    layer = max(0, min(layer, 3))

    if highway in PEDESTRIAN_TYPES:
        return RoadSpec(PEDESTRIAN_WIDTH_M, True, False, False, layer)

    width = parse_number(tags.get("width"))
    if width is None:
        lanes = parse_number(tags.get("lanes"))
        width = lanes * LANE_WIDTH_M if lanes else config.ROAD_WIDTHS_M.get(
            base, config.DEFAULT_ROAD_WIDTH_M)
    width = max(MIN_ROAD_WIDTH_M, min(width, MAX_ROAD_WIDTH_M))

    one_way = tags.get("oneway") in ("yes", "true", "1") or base == "motorway"
    has_sidewalk = base not in NO_SIDEWALK_TYPES and layer == 0
    has_markings = (not one_way) and width >= MARKING_MIN_ROAD_WIDTH_M
    return RoadSpec(width, False, has_sidewalk, has_markings, layer)


# ---------------------------------------------------------------------------
# Mesh building
# ---------------------------------------------------------------------------
def build_road(spec: RoadSpec, points: np.ndarray,
               tiles: Dict[Tuple[int, int], TileMeshes]) -> None:
    """Add the geometry of one road to the tiles it passes through."""
    half = spec.width / 2.0
    base_y = ROAD_BASE_Y + spec.layer * LAYER_STEP_Y
    sidewalk_y = base_y + SIDEWALK_RISE_Y
    marking_y = base_y + MARKING_RISE_Y

    left = offset_line(points, half)
    right = offset_line(points, -half)

    walk_edges: List[Tuple[np.ndarray, np.ndarray]] = []
    if spec.has_sidewalk:
        for side in (1.0, -1.0):
            inner = left if side > 0 else right
            outer = offset_line(points, side * (half + config.SIDEWALK_WIDTH_M))
            walk_edges.append((inner, outer))

    for i in range(len(points) - 1):
        centre = (points[i] + points[i + 1]) / 2.0
        tile = tiles[tile_key(centre[0], centre[1])]

        if spec.pedestrian:
            tile.sidewalk.add_quad(v3(left[i], sidewalk_y), v3(right[i], sidewalk_y),
                                   v3(right[i + 1], sidewalk_y), v3(left[i + 1], sidewalk_y))
            continue

        tile.asphalt.add_quad(v3(left[i], base_y), v3(right[i], base_y),
                              v3(right[i + 1], base_y), v3(left[i + 1], base_y))
        for inner, outer in walk_edges:
            tile.sidewalk.add_quad(v3(inner[i], sidewalk_y), v3(outer[i], sidewalk_y),
                                   v3(outer[i + 1], sidewalk_y), v3(inner[i + 1], sidewalk_y))

    if spec.has_markings:
        add_markings(points, marking_y, tiles)


def add_markings(points: np.ndarray, y: float,
                 tiles: Dict[Tuple[int, int], TileMeshes]) -> None:
    """Add a dashed centre line along the polyline."""
    half = MARKING_WIDTH_M / 2.0
    for start, end in dash_segments(points):
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            continue
        direction /= length
        normal = np.array([-direction[1], direction[0]]) * half
        centre = (start + end) / 2.0
        tile = tiles[tile_key(centre[0], centre[1])]
        tile.markings.add_quad(v3(start + normal, y), v3(start - normal, y),
                               v3(end - normal, y), v3(end + normal, y))


# ---------------------------------------------------------------------------
# Loading and exporting
# ---------------------------------------------------------------------------
def load_roads() -> List[dict]:
    """Read roads.json. Raises a clear error if it is missing or broken."""
    path = config.DATA_DIR / "roads.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 01_download.py first.")
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file).get("elements", [])
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read {path}: {error}") from error


def export_tiles(tiles: Dict[Tuple[int, int], TileMeshes]) -> int:
    """Write one GLB per tile plus an index file. Returns the number of files."""
    out_dir = config.OUTPUT_DIR / "roads"
    out_dir.mkdir(parents=True, exist_ok=True)
    index: List[dict] = []

    for (tx, tz), tile in sorted(tiles.items()):
        scene = trimesh.Scene()
        triangles = 0
        for name, buffer in (("asphalt", tile.asphalt), ("sidewalk", tile.sidewalk),
                             ("markings", tile.markings)):
            if not buffer.faces:
                continue
            mesh = buffer.to_mesh(MATERIALS[name])
            scene.add_geometry(mesh, node_name=name, geom_name=name)
            triangles += len(mesh.faces)
        if triangles == 0:
            continue

        file_name = f"road_{tx}_{tz}.glb"
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
    with open(out_dir / "roads_index.json", "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    return len(index)


def main() -> int:
    config.ensure_directories()
    try:
        elements = load_roads()
    except (FileNotFoundError, RuntimeError) as error:
        print(f"[error] {error}")
        return 1

    tiles: Dict[Tuple[int, int], TileMeshes] = defaultdict(TileMeshes)
    built = skipped = 0

    for element in elements:
        geometry = element.get("geometry")
        spec = classify(element.get("tags", {}))
        if spec is None or not geometry or len(geometry) < 2:
            skipped += 1
            continue

        points = clean_polyline([project(p["lat"], p["lon"]) for p in geometry])
        if len(points) < 2:
            skipped += 1
            continue

        build_road(spec, points, tiles)
        built += 1

    files = export_tiles(tiles)
    print(f"Roads built   : {built}")
    print(f"Roads skipped : {skipped}")
    print(f"Tile files    : {files}  ->  {config.OUTPUT_DIR / 'roads'}")
    print("Roads OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
