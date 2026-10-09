"""
03_buildings.py
Step 3 of the Dubai 3D City Generator: build 3D buildings.

Reads data/<city>/buildings.json (OpenStreetMap, "out geom" format) and
builds, for every tile, a GLB file with one mesh per material:
    glass_blue, glass_dark, concrete_light, concrete_warm,
    sandstone, industrial   -> building walls
    roof                    -> flat roofs plus rooftop units (AC, tanks)

Facade UVs are set so that ONE texture repeat equals ONE window bay
(BAY_WIDTH_M wide) and ONE floor (config.METERS_PER_LEVEL tall). When
materials.py adds window textures later, they will line up automatically.

World axes (shared by all scripts and Godot):
    +X = east, +Y = up, +Z = south. Origin = centre of config.BBOX.
All vertices are written in world coordinates, so tiles need no offset.

Usage:
    python 03_buildings.py

Requires:
    pip install numpy trimesh shapely mapbox-earcut
"""

import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon, box
from shapely.geometry.polygon import orient
from trimesh.visual.material import PBRMaterial

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EARTH_RADIUS_M: float = 6_378_137.0

MAX_HEIGHT_M: float = 900.0              # safety cap (Burj Khalifa is ~828 m)
MIN_FOOTPRINT_AREA_M2: float = 6.0       # smaller footprints are ignored
SIMPLIFY_TOLERANCE_M: float = 0.15       # removes needless footprint points
MIN_WALL_LENGTH_M: float = 0.05

BAY_WIDTH_M: float = 3.0                 # one window bay = one UV repeat in U
ROOF_UV_M: float = 4.0                   # one roof texture repeat per 4 m

ROOF_UNIT_MIN_AREA_M2: float = 120.0
ROOF_UNIT_MAX_HEIGHT_M: float = 60.0
ROOF_UNIT_EDGE_MARGIN_M: float = 1.0

LatLon = Tuple[float, float]
XZ = Tuple[float, float]

# Typical height (m) when OSM has neither "height" nor "building:levels".
TYPE_HEIGHTS: Dict[str, float] = {
    "apartments": 30.0, "residential": 12.0, "house": 8.0, "detached": 8.0,
    "villa": 9.0, "terrace": 8.0, "commercial": 24.0, "office": 40.0,
    "hotel": 40.0, "retail": 9.0, "industrial": 9.0, "warehouse": 8.0,
    "mosque": 14.0, "school": 12.0, "hospital": 18.0, "garage": 3.5,
    "garages": 3.5, "shed": 3.0, "service": 4.0, "roof": 4.0,
}

GLASS_TYPES = {"office", "commercial", "hotel", "skyscraper"}
INDUSTRIAL_TYPES = {"industrial", "warehouse", "garage", "garages", "service",
                    "shed", "hangar", "factory", "roof"}
VILLA_TYPES = {"house", "detached", "villa", "terrace", "bungalow",
               "semidetached_house", "residential"}
TALL_GLASS_HEIGHT_M: float = 40.0

# Simple PBR colours (0-255). materials.py will replace these with textures.
MATERIALS: Dict[str, PBRMaterial] = {
    "glass_blue": PBRMaterial(name="glass_blue", baseColorFactor=[70, 110, 140, 255],
                              roughnessFactor=0.18, metallicFactor=0.35),
    "glass_dark": PBRMaterial(name="glass_dark", baseColorFactor=[40, 52, 64, 255],
                              roughnessFactor=0.15, metallicFactor=0.40),
    "concrete_light": PBRMaterial(name="concrete_light", baseColorFactor=[190, 188, 182, 255],
                                  roughnessFactor=0.90, metallicFactor=0.0),
    "concrete_warm": PBRMaterial(name="concrete_warm", baseColorFactor=[205, 190, 165, 255],
                                 roughnessFactor=0.90, metallicFactor=0.0),
    "sandstone": PBRMaterial(name="sandstone", baseColorFactor=[214, 190, 150, 255],
                             roughnessFactor=0.85, metallicFactor=0.0),
    "industrial": PBRMaterial(name="industrial", baseColorFactor=[150, 152, 155, 255],
                              roughnessFactor=0.80, metallicFactor=0.10),
    "roof": PBRMaterial(name="roof", baseColorFactor=[95, 95, 98, 255],
                        roughnessFactor=0.95, metallicFactor=0.0),
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class MeshBuffer:
    """Accumulates triangles (with UVs) for one material inside one tile."""
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    faces: List[Tuple[int, int, int]] = field(default_factory=list)
    uvs: List[Tuple[float, float]] = field(default_factory=list)

    def add_quad(self, corners: Sequence[Tuple[float, float, float]],
                 uvs: Sequence[Tuple[float, float]]) -> None:
        """Add a quad a-b-c-d as two triangles (a,b,c) and (a,c,d)."""
        base = len(self.vertices)
        self.vertices.extend(corners)
        self.uvs.extend(uvs)
        self.faces.append((base, base + 1, base + 2))
        self.faces.append((base, base + 2, base + 3))

    def add_triangles(self, vertices: np.ndarray, faces: np.ndarray,
                      uvs: np.ndarray) -> None:
        """Add an indexed triangle list."""
        base = len(self.vertices)
        self.vertices.extend(map(tuple, vertices.tolist()))
        self.uvs.extend(map(tuple, uvs.tolist()))
        self.faces.extend((int(a) + base, int(b) + base, int(c) + base)
                          for a, b, c in faces)

    def to_mesh(self, material: PBRMaterial) -> trimesh.Trimesh:
        """Convert to a trimesh mesh carrying UVs and the PBR material."""
        mesh = trimesh.Trimesh(vertices=np.asarray(self.vertices, dtype=np.float64),
                               faces=np.asarray(self.faces, dtype=np.int64),
                               process=False)
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.asarray(self.uvs, dtype=np.float64), material=material)
        return mesh


@dataclass
class TileBuild:
    """All material buffers of a single tile."""
    buffers: Dict[str, MeshBuffer] = field(default_factory=lambda: defaultdict(MeshBuffer))

    def triangle_count(self) -> int:
        return sum(len(buffer.faces) for buffer in self.buffers.values())


@dataclass
class Stats:
    """Counters printed at the end of the run."""
    built: int = 0
    skipped: int = 0
    roof_failed: int = 0
    tallest_m: float = 0.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def project(lat: float, lon: float) -> XZ:
    """Convert latitude/longitude to local metres (x east, z south)."""
    x = (math.radians(lon - config.ORIGIN_LON) * EARTH_RADIUS_M
         * math.cos(math.radians(config.ORIGIN_LAT)))
    z = -math.radians(lat - config.ORIGIN_LAT) * EARTH_RADIUS_M
    return x, z


def tile_key(x: float, z: float) -> Tuple[int, int]:
    """Return the (tx, tz) index of the tile containing the point."""
    return (math.floor(x / config.TILE_SIZE_M), math.floor(z / config.TILE_SIZE_M))


def parse_number(value: Optional[str]) -> Optional[float]:
    """Parse OSM numbers such as '12', '3.5' or '12 m'. Returns None if invalid."""
    if value is None:
        return None
    try:
        return float(str(value).split()[0].replace(",", "."))
    except (ValueError, IndexError):
        return None


def stitch_rings(ways: List[List[LatLon]]) -> List[List[LatLon]]:
    """
    Join OSM way fragments end-to-end into closed rings.
    Fragments that cannot be closed are dropped.
    """
    pending = [list(way) for way in ways if len(way) >= 2]
    rings: List[List[LatLon]] = []
    while pending:
        ring = pending.pop(0)
        while ring[0] != ring[-1]:
            for index, segment in enumerate(pending):
                if segment[0] == ring[-1]:
                    ring.extend(segment[1:])
                elif segment[-1] == ring[-1]:
                    ring.extend(reversed(segment[:-1]))
                else:
                    continue
                pending.pop(index)
                break
            else:
                break                      # no fragment continues this ring
        if ring[0] == ring[-1] and len(ring) >= 4:
            rings.append(ring)
    return rings


def latlon_list(geometry: Optional[List[Optional[dict]]]) -> Optional[List[LatLon]]:
    """Extract (lat, lon) pairs. Returns None if any point is missing."""
    if not geometry:
        return None
    points: List[LatLon] = []
    for node in geometry:
        if not node:
            return None
        points.append((node["lat"], node["lon"]))
    return points


def to_xz(ring: List[LatLon]) -> List[XZ]:
    """Project a lat/lon ring into local metres."""
    return [project(lat, lon) for lat, lon in ring]


def element_polygons(element: dict) -> List[Polygon]:
    """
    Convert an OSM way or multipolygon relation into oriented shapely polygons.
    Orientation: outer rings clockwise, holes counter-clockwise (in the x/z
    plane), which makes every wall face point away from the solid.
    """
    outer_ways: List[List[LatLon]] = []
    inner_ways: List[List[LatLon]] = []

    if element.get("type") == "way":
        points = latlon_list(element.get("geometry"))
        if points:
            outer_ways.append(points)
    elif element.get("type") == "relation":
        for member in element.get("members", []):
            if member.get("type") != "way":
                continue
            points = latlon_list(member.get("geometry"))
            if not points:
                continue
            (inner_ways if member.get("role") == "inner" else outer_ways).append(points)

    outers = [to_xz(ring) for ring in stitch_rings(outer_ways)]
    inners = [to_xz(ring) for ring in stitch_rings(inner_ways)]

    polygons: List[Polygon] = []
    for outer in outers:
        try:
            shell = Polygon(outer)
            holes = []
            for inner in inners:
                hole = Polygon(inner).buffer(0)
                if not hole.is_empty and shell.buffer(0).contains(hole.representative_point()):
                    holes.append(inner)
            polygon = Polygon(outer, holes)
        except ValueError:
            continue
        if not polygon.is_valid:
            polygon = polygon.buffer(0)          # repairs self-intersections
        if polygon.is_empty:
            continue
        parts = list(polygon.geoms) if polygon.geom_type == "MultiPolygon" else [polygon]
        polygons.extend(orient(part, sign=-1.0) for part in parts
                        if part.geom_type == "Polygon")
    return polygons


def triangulate(polygon: Polygon) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Triangulate a polygon (holes supported). Tries earcut through trimesh
    first, then shapely's constrained Delaunay. Returns (vertices_2d, faces).
    """
    try:
        vertices, faces = trimesh.creation.triangulate_polygon(polygon, engine="earcut")
        return np.asarray(vertices, dtype=np.float64), np.array(faces, dtype=np.int64)
    except (ImportError, ValueError, RuntimeError, IndexError):
        pass
    try:
        import shapely
        triangles = shapely.constrained_delaunay_triangles(polygon)
        vertices_list: List[Tuple[float, float]] = []
        faces_list: List[Tuple[int, int, int]] = []
        for triangle in triangles.geoms:
            corners = list(triangle.exterior.coords)[:3]
            base = len(vertices_list)
            vertices_list.extend((float(x), float(z)) for x, z in corners)
            faces_list.append((base, base + 1, base + 2))
        if not faces_list:
            return None
        return (np.asarray(vertices_list, dtype=np.float64),
                np.asarray(faces_list, dtype=np.int64))
    except (ImportError, AttributeError, ValueError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Mesh building
# ---------------------------------------------------------------------------
def add_walls(buffer: MeshBuffer, ring: np.ndarray, y_bottom: float, y_top: float) -> None:
    """
    Add one vertical quad per footprint edge. The ring must be oriented so
    that the wall normal (-dz, 0, dx) points outward (see element_polygons).
    UVs: U = metres along the perimeter / bay width, V = height / floor height.
    """
    count = len(ring)
    v_top = (y_top - y_bottom) / config.METERS_PER_LEVEL
    travelled = 0.0
    for i in range(count):
        a = ring[i]
        b = ring[(i + 1) % count]
        length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if length < MIN_WALL_LENGTH_M:
            continue
        u0 = travelled / BAY_WIDTH_M
        u1 = (travelled + length) / BAY_WIDTH_M
        buffer.add_quad(
            [(a[0], y_bottom, a[1]), (b[0], y_bottom, b[1]),
             (b[0], y_top, b[1]), (a[0], y_top, a[1])],
            [(u0, 0.0), (u1, 0.0), (u1, v_top), (u0, v_top)],
        )
        travelled += length


def add_roof(buffer: MeshBuffer, polygon: Polygon, y: float) -> bool:
    """Add a flat roof (all faces pointing up). Returns False if it failed."""
    result = triangulate(polygon)
    if result is None:
        return False
    vertices_2d, faces = result
    vertices = np.column_stack([vertices_2d[:, 0],
                                np.full(len(vertices_2d), y),
                                vertices_2d[:, 1]])
    corners = vertices[faces]
    normal_y = np.cross(corners[:, 1] - corners[:, 0],
                        corners[:, 2] - corners[:, 0])[:, 1]
    flip = normal_y < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]          # make every face point up
    buffer.add_triangles(vertices, faces, vertices_2d / ROOF_UV_M)
    return True


def add_box(buffer: MeshBuffer, x0: float, z0: float, x1: float, z1: float,
            y: float, height: float) -> None:
    """Add a closed-top box (rooftop unit) standing on y."""
    ring = np.array([(x0, z0), (x0, z1), (x1, z1), (x1, z0)], dtype=np.float64)
    add_walls(buffer, ring, y, y + height)
    top = y + height
    buffer.add_quad(
        [(x0, top, z0), (x0, top, z1), (x1, top, z1), (x1, top, z0)],
        [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)],
    )


def add_roof_units(buffer: MeshBuffer, polygon: Polygon, y: float,
                   rng: random.Random) -> None:
    """Scatter a few AC / tank boxes on a roof, fully inside the footprint."""
    safe_area = polygon.buffer(-ROOF_UNIT_EDGE_MARGIN_M)
    if safe_area.is_empty:
        return
    anchor = polygon.representative_point()
    for _ in range(rng.randint(1, 3)):
        width = rng.uniform(1.5, 3.0)
        depth = rng.uniform(1.2, 2.5)
        height = rng.uniform(0.8, 1.8)
        cx = anchor.x + rng.uniform(-4.0, 4.0)
        cz = anchor.y + rng.uniform(-4.0, 4.0)
        footprint = box(cx - width / 2, cz - depth / 2, cx + width / 2, cz + depth / 2)
        if safe_area.contains(footprint):
            add_box(buffer, cx - width / 2, cz - depth / 2,
                    cx + width / 2, cz + depth / 2, y, height)


# ---------------------------------------------------------------------------
# Building attributes
# ---------------------------------------------------------------------------
def resolve_height(tags: Dict[str, str], rng: random.Random) -> Tuple[float, float]:
    """Return (bottom_y, top_y) in metres from OSM tags, with sensible fallbacks."""
    top = parse_number(tags.get("height"))
    if top is None:
        levels = parse_number(tags.get("building:levels"))
        if levels:
            top = levels * config.METERS_PER_LEVEL
    if top is None:
        base = TYPE_HEIGHTS.get(tags.get("building", ""), config.DEFAULT_BUILDING_HEIGHT_M)
        top = base * rng.uniform(0.8, 1.3)           # avoids a flat, uniform skyline
    top = max(config.MIN_BUILDING_HEIGHT_M, min(top, MAX_HEIGHT_M))

    bottom = parse_number(tags.get("min_height"))
    if bottom is None:
        min_level = parse_number(tags.get("building:min_level"))
        bottom = min_level * config.METERS_PER_LEVEL if min_level else 0.0
    bottom = max(0.0, min(bottom, top - 1.0))
    return bottom, top


def choose_facade(tags: Dict[str, str], top: float, rng: random.Random) -> str:
    """Pick a facade material name from the building type and height."""
    kind = tags.get("building", "yes")
    if kind in INDUSTRIAL_TYPES:
        return "industrial"
    if kind in GLASS_TYPES or top >= TALL_GLASS_HEIGHT_M:
        return rng.choice(("glass_blue", "glass_dark"))
    if kind in VILLA_TYPES:
        return rng.choice(("sandstone", "concrete_warm"))
    return rng.choice(("concrete_light", "concrete_warm", "sandstone"))


def build_building(element: dict, tiles: Dict[Tuple[int, int], TileBuild],
                   stats: Stats) -> None:
    """Build all geometry of one OSM building and add it to its tile."""
    tags = element.get("tags", {})
    rng = random.Random(int(element.get("id", 0)))   # same building = same result
    polygons = element_polygons(element)
    if not polygons:
        stats.skipped += 1
        return

    bottom, top = resolve_height(tags, rng)
    facade = choose_facade(tags, top, rng)
    added = False

    for polygon in polygons:
        polygon = polygon.simplify(SIMPLIFY_TOLERANCE_M, preserve_topology=True)
        if polygon.is_empty or polygon.geom_type != "Polygon":
            continue
        if polygon.area < MIN_FOOTPRINT_AREA_M2:
            continue
        polygon = orient(polygon, sign=-1.0)

        centre = polygon.centroid
        tile = tiles[tile_key(centre.x, centre.y)]

        walls = tile.buffers[facade]
        add_walls(walls, np.asarray(polygon.exterior.coords)[:-1], bottom, top)
        for hole in polygon.interiors:
            add_walls(walls, np.asarray(hole.coords)[:-1], bottom, top)

        roof = tile.buffers["roof"]
        if not add_roof(roof, polygon, top):
            stats.roof_failed += 1
        elif polygon.area >= ROOF_UNIT_MIN_AREA_M2 and top <= ROOF_UNIT_MAX_HEIGHT_M:
            add_roof_units(roof, polygon, top, rng)
        added = True

    if added:
        stats.built += 1
        stats.tallest_m = max(stats.tallest_m, top)
    else:
        stats.skipped += 1


# ---------------------------------------------------------------------------
# Loading and exporting
# ---------------------------------------------------------------------------
def load_buildings() -> List[dict]:
    """Read buildings.json. Raises a clear error if it is missing or broken."""
    path = config.DATA_DIR / "buildings.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 01_download.py first.")
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file).get("elements", [])
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read {path}: {error}") from error


def export_tiles(tiles: Dict[Tuple[int, int], TileBuild]) -> int:
    """Write one GLB per tile plus an index file. Returns the number of files."""
    out_dir = config.OUTPUT_DIR / "buildings"
    out_dir.mkdir(parents=True, exist_ok=True)
    index: List[dict] = []

    for (tx, tz), tile in sorted(tiles.items()):
        scene = trimesh.Scene()
        for name, buffer in tile.buffers.items():
            if not buffer.faces:
                continue
            scene.add_geometry(buffer.to_mesh(MATERIALS[name]),
                               node_name=name, geom_name=name)
        triangles = tile.triangle_count()
        if triangles == 0:
            continue

        file_name = f"building_{tx}_{tz}.glb"
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
    with open(out_dir / "buildings_index.json", "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    return len(index)


def main() -> int:
    config.ensure_directories()
    try:
        elements = load_buildings()
    except (FileNotFoundError, RuntimeError) as error:
        print(f"[error] {error}")
        return 1

    tiles: Dict[Tuple[int, int], TileBuild] = defaultdict(TileBuild)
    stats = Stats()
    for element in elements:
        try:
            build_building(element, tiles, stats)
        except (ValueError, KeyError, IndexError) as error:
            # One broken building must never stop the whole city.
            stats.skipped += 1
            print(f"[warn] building {element.get('id')} skipped: {error}")

    files = export_tiles(tiles)
    print(f"Buildings built   : {stats.built}")
    print(f"Buildings skipped : {stats.skipped}")
    print(f"Roofs failed      : {stats.roof_failed}")
    print(f"Tallest building  : {stats.tallest_m:.0f} m")
    print(f"Tile files        : {files}  ->  {config.OUTPUT_DIR / 'buildings'}")
    print("Buildings OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
