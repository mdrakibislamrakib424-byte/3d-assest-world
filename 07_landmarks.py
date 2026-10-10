"""
07_landmarks.py
Step 7 of the Dubai 3D City Generator: landmark buildings.

Replaces ordinary OpenStreetMap buildings with detailed landmarks such as the
Burj Khalifa. For every landmark in LANDMARKS the script:

  1. Finds the building in the OSM data: first by NAME, then by the fallback
     coordinates (so the landmark always lands on its real spot).
  2. Uses the real OSM footprint (or a procedural Y-shaped footprint if the
     OSM outline is missing or unusable).
  3. Builds a stepped tower: the footprint is shrunk toward its centre tier
     by tier (setbacks with exposed ledges), then a tapering spire is added.
  4. Hides the ordinary buildings that the landmark replaces.

Because ordinary buildings must be hidden, this script rebuilds ALL building
tiles (like 03_buildings.py, but with landmarks) and overwrites
output/<city>/buildings. Always run it AFTER 03_buildings.py, and run
05_tiles.py afterwards to merge the new building tiles.

It also writes output/<city>/landmarks/landmarks_index.json with the world
position and lat/lon of every landmark (useful for minimap and missions).

World axes: +X = east, +Y = up, +Z = south. Origin = centre of config.BBOX.

To add a landmark: append a LandmarkSpec to LANDMARKS. Landmarks that are not
inside the downloaded area are reported and skipped, so this list can already
hold landmarks for areas you add later.

Usage:
    python 07_landmarks.py

Requires:
    pip install numpy trimesh shapely mapbox-earcut
"""

import importlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from shapely import affinity
from shapely.errors import GEOSException
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from trimesh.visual.material import PBRMaterial

import config

# Files that start with a digit cannot be imported with a normal "import".
try:
    roads_mod = importlib.import_module("02_roads")
    buildings_mod = importlib.import_module("03_buildings")
    nature_mod = importlib.import_module("04_nature")
except ImportError as import_error:
    sys.exit(f"[error] cannot load helper scripts: {import_error}")

TileBuild = buildings_mod.TileBuild
XZ = Tuple[float, float]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EARTH_RADIUS_M: float = 6_378_137.0
NAME_TAGS: Tuple[str, ...] = ("name", "name:en", "official_name", "int_name")
SIMPLIFY_TOLERANCE_M: float = 0.3
HIDE_OVERLAP_RATIO: float = 0.30          # hide a neighbour if 30% of it is covered
HIDE_ZONE_BUFFER_M: float = 1.0
SPIRE_SIDES: int = 8
MIN_TIER_HEIGHT_M: float = 0.5

# Procedural Y-shaped footprint (used when the OSM outline is unusable).
Y_WING_LENGTH_M: float = 48.0
Y_WING_WIDTH_M: float = 19.0
Y_CORE_RADIUS_M: float = 13.0

LANDMARK_MATERIALS: Dict[str, PBRMaterial] = {
    "landmark_glass": PBRMaterial(name="landmark_glass", baseColorFactor=[150, 178, 205, 255],
                                  roughnessFactor=0.12, metallicFactor=0.55),
    "landmark_trim": PBRMaterial(name="landmark_trim", baseColorFactor=[165, 170, 176, 255],
                                 roughnessFactor=0.50, metallicFactor=0.40),
    "landmark_metal": PBRMaterial(name="landmark_metal", baseColorFactor=[200, 205, 210, 255],
                                  roughnessFactor=0.25, metallicFactor=0.90),
}
# The exporter of 03_buildings.py looks materials up by name in its own table.
buildings_mod.MATERIALS.update(LANDMARK_MATERIALS)

FACADE_DEFAULT: str = "landmark_glass"
TRIM_MATERIAL: str = "landmark_trim"
SPIRE_MATERIAL: str = "landmark_metal"


# ---------------------------------------------------------------------------
# Landmark definitions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LandmarkSpec:
    """Everything needed to find and build one landmark."""
    key: str
    names: Tuple[str, ...]                          # OSM names (lower case) to search for
    lat: float                                      # fallback position
    lon: float
    tiers: Tuple[Tuple[float, float], ...]          # (top height in m, footprint scale)
    spire: Tuple[Tuple[float, float], ...] = ()     # (height in m, radius in m), bottom to top
    facade: str = FACADE_DEFAULT
    max_anchor_distance_m: float = 80.0             # how far from lat/lon a match may be
    min_area_m2: float = 0.0                        # sanity limits for the OSM footprint
    max_area_m2: float = 1.0e9
    fallback_shape: Optional[str] = None            # "y" = procedural Y footprint
    fallback_rotation_deg: float = 90.0


LANDMARKS: Tuple[LandmarkSpec, ...] = (
    LandmarkSpec(
        key="burj_khalifa",
        names=("burj khalifa", "برج خليفة"),
        lat=25.19720, lon=55.27440,
        tiers=((120.0, 1.00), (200.0, 0.90), (270.0, 0.79), (340.0, 0.68),
               (410.0, 0.57), (480.0, 0.46), (540.0, 0.36), (585.0, 0.26)),
        spire=((585.0, 3.2), (650.0, 2.6), (720.0, 1.8), (790.0, 1.0), (828.0, 0.15)),
        min_area_m2=1500.0, max_area_m2=8000.0,
        fallback_shape="y",
    ),
    LandmarkSpec(
        key="address_downtown",
        names=("the address downtown", "address downtown", "the address downtown dubai"),
        lat=25.19417, lon=55.28083,
        tiers=((290.0, 1.00), (302.0, 0.88)),
        max_anchor_distance_m=120.0,
    ),
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class BuildingRecord:
    """One OSM building: its element, id, names and combined footprint."""
    element: dict
    osm_id: int
    names: FrozenSet[str]
    geometry: BaseGeometry


@dataclass
class LandmarkResult:
    """A landmark that was found and is ready to be built."""
    spec: LandmarkSpec
    source: str                                     # osm-name | osm-coordinates | procedural
    parts: List[Polygon]
    hidden_ids: Set[int] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def unproject(x: float, z: float) -> Tuple[float, float]:
    """Convert local metres back to (latitude, longitude)."""
    lat = config.ORIGIN_LAT - math.degrees(z / EARTH_RADIUS_M)
    lon = config.ORIGIN_LON + math.degrees(
        x / (EARTH_RADIUS_M * math.cos(math.radians(config.ORIGIN_LAT))))
    return lat, lon


def normalize(name: str) -> str:
    """Lower-case and trim a name for comparison."""
    return name.strip().lower()


def y_footprint(cx: float, cz: float, rotation_deg: float) -> Polygon:
    """Three wings at 120 degrees around a hexagonal core (Burj Khalifa plan)."""
    shapes: List[Polygon] = []
    core = [(cx + Y_CORE_RADIUS_M * math.cos(math.radians(60 * i)),
             cz + Y_CORE_RADIUS_M * math.sin(math.radians(60 * i))) for i in range(6)]
    shapes.append(Polygon(core))
    for k in range(3):
        angle = math.radians(rotation_deg + 120.0 * k)
        dx, dz = math.cos(angle), -math.sin(angle)
        sx, sz = -dz, dx
        half = Y_WING_WIDTH_M / 2.0
        shapes.append(Polygon([
            (cx + sx * half, cz + sz * half),
            (cx + dx * Y_WING_LENGTH_M + sx * half, cz + dz * Y_WING_LENGTH_M + sz * half),
            (cx + dx * Y_WING_LENGTH_M - sx * half, cz + dz * Y_WING_LENGTH_M - sz * half),
            (cx - sx * half, cz - sz * half),
        ]))
    return unary_union(shapes).buffer(0)


def polygon_parts(geometry: BaseGeometry) -> List[Polygon]:
    """Split any shapely result into simplified polygons, largest first."""
    parts = nature_mod.flat_parts(geometry)
    cleaned: List[Polygon] = []
    for part in parts:
        simple = part.simplify(SIMPLIFY_TOLERANCE_M, preserve_topology=True)
        cleaned.extend(nature_mod.flat_parts(simple))
    return sorted(cleaned, key=lambda p: p.area, reverse=True)


# ---------------------------------------------------------------------------
# Finding landmarks in the OSM data
# ---------------------------------------------------------------------------
def load_records(elements: List[dict]) -> List[BuildingRecord]:
    """Build a BuildingRecord for every usable OSM building."""
    records: List[BuildingRecord] = []
    for element in elements:
        try:
            polygons = buildings_mod.element_polygons(element)
            if not polygons:
                continue
            tags = element.get("tags", {})
            names = frozenset(normalize(tags[tag]) for tag in NAME_TAGS if tags.get(tag))
            records.append(BuildingRecord(element, int(element.get("id", 0)), names,
                                          unary_union(polygons)))
        except (KeyError, TypeError, ValueError, GEOSException):
            continue
    return records


def resolve_landmark(spec: LandmarkSpec, records: List[BuildingRecord]) -> Optional[LandmarkResult]:
    """Find the footprint(s) of a landmark. Returns None if it is not in the data."""
    wanted = {normalize(name) for name in spec.names}
    matched = [r for r in records if r.names & wanted]
    source = "osm-name"

    anchor_x, anchor_z = roads_mod.project(spec.lat, spec.lon)
    anchor = Point(anchor_x, anchor_z)

    if not matched:
        candidates = [(r.geometry.distance(anchor), r) for r in records]
        candidates = [c for c in candidates if c[0] <= spec.max_anchor_distance_m]
        if candidates:
            matched = [min(candidates, key=lambda c: c[0])[1]]
            source = "osm-coordinates"

    parts: List[Polygon] = []
    if matched:
        try:
            parts = polygon_parts(unary_union([r.geometry for r in matched]))
        except GEOSException:
            parts = []

    # Sanity check: an outline that is far too small or too large (for example
    # a whole podium) is replaced by the procedural shape, if there is one.
    if parts and spec.fallback_shape == "y":
        area = parts[0].area
        if not (spec.min_area_m2 <= area <= spec.max_area_m2):
            centre = parts[0].centroid
            parts = [y_footprint(centre.x, centre.y, spec.fallback_rotation_deg)]
            source = "procedural"
    elif not parts and spec.fallback_shape == "y":
        # Only trust the fallback position if it lies inside the downloaded area.
        south, west, north, east = config.BBOX
        if south <= spec.lat <= north and west <= spec.lon <= east:
            parts = [y_footprint(anchor_x, anchor_z, spec.fallback_rotation_deg)]
            source = "procedural"

    if not parts:
        return None

    result = LandmarkResult(spec=spec, source=source, parts=parts)
    result.hidden_ids.update(r.osm_id for r in matched)

    # Hide ordinary buildings that the landmark covers.
    zone = unary_union(parts).buffer(HIDE_ZONE_BUFFER_M)
    for record in records:
        if record.osm_id in result.hidden_ids:
            continue
        try:
            if record.geometry.intersects(zone):
                overlap = record.geometry.intersection(zone).area
                if overlap / max(record.geometry.area, 1e-6) >= HIDE_OVERLAP_RATIO:
                    result.hidden_ids.add(record.osm_id)
        except GEOSException:
            continue
    return result


# ---------------------------------------------------------------------------
# Building the landmark geometry
# ---------------------------------------------------------------------------
def add_footprint_walls(buffer, polygon: Polygon, y_bottom: float, y_top: float) -> None:
    """Walls for the outer ring and every hole of an oriented polygon."""
    oriented = orient(polygon, sign=-1.0)       # outward-facing walls (see 03_buildings.py)
    buildings_mod.add_walls(buffer, np.asarray(oriented.exterior.coords)[:-1], y_bottom, y_top)
    for hole in oriented.interiors:
        buildings_mod.add_walls(buffer, np.asarray(hole.coords)[:-1], y_bottom, y_top)


def build_tiers(tile: TileBuild, spec: LandmarkSpec, polygon: Polygon) -> float:
    """Stepped tower. Returns the height of the top tier."""
    facade = tile.buffers[spec.facade]
    trim = tile.buffers[TRIM_MATERIAL]
    centre = polygon.centroid
    previous_top = 0.0
    previous: Optional[Polygon] = None

    for top, scale in spec.tiers:
        if top - previous_top < MIN_TIER_HEIGHT_M:
            continue
        tier = affinity.scale(polygon, xfact=scale, yfact=scale, origin=centre)
        add_footprint_walls(facade, tier, previous_top, top)
        if previous is not None:
            # The roof of the lower tier that is not covered by the upper tier.
            for ledge in nature_mod.flat_parts(previous.difference(tier)):
                buildings_mod.add_roof(trim, ledge, previous_top)
        previous, previous_top = tier, top

    if previous is not None:
        buildings_mod.add_roof(trim, previous, previous_top)
    return previous_top


def build_spire(tile: TileBuild, spec: LandmarkSpec, cx: float, cz: float) -> None:
    """Tapering mast on top of the tower."""
    if len(spec.spire) < 2:
        return
    rings = [(height, cx, cz, radius) for height, radius in spec.spire]
    vertices, faces = nature_mod.make_tube(rings, SPIRE_SIDES)
    tile.buffers[SPIRE_MATERIAL].add_triangles(vertices, faces, np.zeros((len(vertices), 2)))


def build_landmark(result: LandmarkResult, tiles: Dict[Tuple[int, int], TileBuild]) -> None:
    """Add all parts of one landmark to the tile that contains them."""
    for polygon in result.parts:
        centre = polygon.centroid
        tile = tiles[roads_mod.tile_key(centre.x, centre.y)]
        build_tiers(tile, result.spec, polygon)
        build_spire(tile, result.spec, centre.x, centre.y)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def write_index(results: List[LandmarkResult]) -> None:
    """Save landmark positions for the game (minimap, missions, markers)."""
    out_dir = config.OUTPUT_DIR / "landmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: List[dict] = []
    for result in results:
        centre = result.parts[0].centroid
        lat, lon = unproject(centre.x, centre.y)
        entries.append({
            "key": result.spec.key,
            "source": result.source,
            "x": round(centre.x, 2), "z": round(centre.y, 2),
            "lat": round(lat, 6), "lon": round(lon, 6),
            "height_m": max(result.spec.tiers[-1][0],
                            result.spec.spire[-1][0] if result.spec.spire else 0.0),
            "tile": list(roads_mod.tile_key(centre.x, centre.y)),
            "hidden_osm_ids": sorted(result.hidden_ids),
        })
    with open(out_dir / "landmarks_index.json", "w", encoding="utf-8") as file:
        json.dump({"landmarks": entries}, file, indent=2)


def main() -> int:
    config.ensure_directories()
    try:
        elements = buildings_mod.load_buildings()
    except (FileNotFoundError, RuntimeError) as error:
        print(f"[error] {error}")
        return 1

    print("Reading buildings ...")
    records = load_records(elements)

    results: List[LandmarkResult] = []
    for spec in LANDMARKS:
        result = resolve_landmark(spec, records)
        if result is None:
            print(f"[skip] {spec.key}: not found inside the downloaded area")
            continue
        results.append(result)
        print(f"[found] {spec.key}: {result.source}, {len(result.parts)} part(s), "
              f"{len(result.hidden_ids)} ordinary building(s) replaced")

    hidden: Set[int] = set()
    for result in results:
        hidden |= result.hidden_ids

    print("Rebuilding building tiles ...")
    tiles: Dict[Tuple[int, int], TileBuild] = defaultdict(TileBuild)
    stats = buildings_mod.Stats()
    for element in elements:
        if int(element.get("id", 0)) in hidden:
            continue
        try:
            buildings_mod.build_building(element, tiles, stats)
        except (ValueError, KeyError, IndexError, GEOSException) as error:
            stats.skipped += 1
            print(f"[warn] building {element.get('id')} skipped: {error}")

    for result in results:
        try:
            build_landmark(result, tiles)
        except (ValueError, IndexError, GEOSException) as error:
            print(f"[error] landmark {result.spec.key} failed: {error}")

    files = buildings_mod.export_tiles(tiles)
    write_index(results)

    print(f"Landmarks built    : {len(results)}")
    print(f"Buildings replaced : {len(hidden)}")
    print(f"Ordinary buildings : {stats.built}")
    print(f"Tile files         : {files}  ->  {config.OUTPUT_DIR / 'buildings'}")
    print("Landmarks OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
