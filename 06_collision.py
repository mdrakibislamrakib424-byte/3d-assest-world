"""
06_collision.py
Step 6 of the Dubai 3D City Generator: build collision tiles for Godot.

Writes one collision-only GLB per tile:
    output/<city>/collision/collision_<tx>_<tz>.glb

Each file holds up to five meshes, named for Godot's import convention. In
Godot 4, a node whose name ends in "-colonly" is turned into a StaticBody3D
with a trimesh collision shape and its visible mesh is discarded, so no
manual collision setup is needed:
    ground-colonly     flat ground at y = 0 (with holes where there is water)
    waterbed-colonly   floor under lakes and canals at y = -2 (cars sink)
    road-colonly       asphalt and sidewalk surfaces (bridges included)
    building-colonly   simplified building walls and roofs
    props-colonly      tree trunks and street-lamp poles

Collision tiles use the same tile grid and WORLD coordinates as the visual
tiles from 05_tiles.py: instantiate them at position (0, 0, 0). The script
also adds "collision_file" and "collision_triangles" to every tile entry of
city_index.json (existing fields are kept).

It reuses the helpers of 02_roads.py, 03_buildings.py, 04_nature.py and
05_tiles.py, so those files must stay in the same folder.

Usage:
    python 06_collision.py

Requires:
    pip install numpy trimesh shapely mapbox-earcut
"""

import importlib
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from shapely.errors import GEOSException
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

import config

# Files that start with a digit cannot be imported with a normal "import".
try:
    roads_mod = importlib.import_module("02_roads")
    buildings_mod = importlib.import_module("03_buildings")
    nature_mod = importlib.import_module("04_nature")
    tiles_mod = importlib.import_module("05_tiles")
except ImportError as import_error:
    sys.exit(f"[error] cannot load helper scripts: {import_error}")

TileBuild = buildings_mod.TileBuild
TileKey = Tuple[int, int]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GROUND_Y: float = 0.0                    # drivable ground (grass level)
WATERBED_Y: float = -2.0                 # floor under water
COLLISION_SIMPLIFY_M: float = 0.4        # building outlines are simplified this much
INCLUDE_PROPS: bool = True               # trunks and lamp poles block cars

# Mesh groups in the order they are written, with their Godot node suffix.
GROUPS: Tuple[str, ...] = ("ground", "waterbed", "road", "building", "props")
GODOT_SUFFIX: str = "-colonly"

# Nature-tile meshes that become "props" collision.
PROP_MESHES = frozenset({"trunk", "lamp_pole"})

ZERO_UV = np.zeros((0, 2))               # placeholder, resized where used


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class Stats:
    """Counters printed at the end of the run."""
    buildings: int = 0
    roads: int = 0
    props_meshes: int = 0
    warnings: int = 0
    triangles: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_bytes: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def add_mesh_data(buffer, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Append an indexed triangle list to a MeshBuffer (UVs are not needed)."""
    buffer.add_triangles(np.asarray(vertices, dtype=np.float64),
                         np.asarray(faces, dtype=np.int64),
                         np.zeros((len(vertices), 2)))


def bbox_region() -> BaseGeometry:
    """A rectangle covering every tile that touches config.BBOX."""
    corners = [roads_mod.project(lat, lon)
               for lat in (config.BBOX[0], config.BBOX[2])
               for lon in (config.BBOX[1], config.BBOX[3])]
    xs, zs = [c[0] for c in corners], [c[1] for c in corners]
    tiles = list(nature_mod.tile_range((min(xs), min(zs), max(xs), max(zs))))
    size = config.TILE_SIZE_M
    return box(min(t[0] for t in tiles) * size, min(t[1] for t in tiles) * size,
               (max(t[0] for t in tiles) + 1) * size, (max(t[1] for t in tiles) + 1) * size)


# ---------------------------------------------------------------------------
# Collision builders
# ---------------------------------------------------------------------------
def build_ground_and_water(tiles: Dict[TileKey, TileBuild], water: BaseGeometry,
                           stats: Stats) -> None:
    """Ground (with water holes) and the floor under the water."""
    region = bbox_region()
    ground = region.difference(water) if not water.is_empty else region
    stats.warnings += nature_mod.add_flat_layer(tiles, ground, "ground", GROUND_Y)
    if not water.is_empty:
        stats.warnings += nature_mod.add_flat_layer(tiles, water, "waterbed", WATERBED_Y)


def build_road_collision(elements: List[dict], tiles: Dict[TileKey, TileBuild],
                         stats: Stats) -> None:
    """
    Road and sidewalk surfaces, built with the same function as 02_roads.py so
    the collision matches the visible road exactly. Lane markings are skipped.
    """
    road_tiles: Dict[TileKey, "roads_mod.TileMeshes"] = defaultdict(roads_mod.TileMeshes)
    for element in elements:
        spec = roads_mod.classify(element.get("tags", {}))
        geometry = element.get("geometry")
        if spec is None or not geometry or len(geometry) < 2:
            continue
        try:
            points = roads_mod.clean_polyline(
                [roads_mod.project(p["lat"], p["lon"]) for p in geometry])
            if len(points) < 2:
                continue
            roads_mod.build_road(spec, points, road_tiles)
            stats.roads += 1
        except (KeyError, TypeError, ValueError, IndexError):
            stats.warnings += 1

    for key, road_tile in road_tiles.items():
        target = tiles[key].buffers["road"]
        for name in ("asphalt", "sidewalk"):
            source = getattr(road_tile, name)
            if not source.faces:
                continue
            mesh = source.to_mesh(roads_mod.MATERIALS[name])     # faces point up
            add_mesh_data(target, mesh.vertices, mesh.faces)


def build_building_collision(elements: List[dict], tiles: Dict[TileKey, TileBuild],
                             stats: Stats) -> None:
    """Simplified walls and roofs. Heights match 03_buildings.py exactly."""
    for element in elements:
        try:
            tags = element.get("tags", {})
            rng = random.Random(int(element.get("id", 0)))        # same seed as step 3
            polygons = buildings_mod.element_polygons(element)
            if not polygons:
                continue
            bottom, top = buildings_mod.resolve_height(tags, rng)
            added = False
            for polygon in polygons:
                polygon = polygon.simplify(COLLISION_SIMPLIFY_M, preserve_topology=True)
                if (polygon.is_empty or polygon.geom_type != "Polygon"
                        or polygon.area < buildings_mod.MIN_FOOTPRINT_AREA_M2):
                    continue
                polygon = orient(polygon, sign=-1.0)
                centre = polygon.centroid
                buffer = tiles[roads_mod.tile_key(centre.x, centre.y)].buffers["building"]
                buildings_mod.add_walls(buffer, np.asarray(polygon.exterior.coords)[:-1],
                                        bottom, top)
                for hole in polygon.interiors:
                    buildings_mod.add_walls(buffer, np.asarray(hole.coords)[:-1], bottom, top)
                if not buildings_mod.add_roof(buffer, polygon, top):
                    stats.warnings += 1
                added = True
            if added:
                stats.buildings += 1
        except (KeyError, TypeError, ValueError, IndexError, GEOSException):
            stats.warnings += 1


def build_prop_collision(tiles: Dict[TileKey, TileBuild], stats: Stats) -> None:
    """Take tree trunks and lamp poles straight from the nature tiles."""
    try:
        nature_files = tiles_mod.load_layer_index("nature")
    except RuntimeError as error:
        print(f"[warn] props skipped: {error}")
        stats.warnings += 1
        return
    for key, path in nature_files.items():
        scene = tiles_mod.read_scene(path)
        if scene is None:
            stats.warnings += 1
            continue
        target = tiles[key].buffers["props"]
        for name, geometry in scene.geometry.items():
            if tiles_mod.mesh_labels(name, geometry) & PROP_MESHES:
                add_mesh_data(target, geometry.vertices, geometry.faces)
                stats.props_meshes += 1


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_tiles(tiles: Dict[TileKey, TileBuild],
                 stats: Stats) -> Dict[TileKey, Tuple[str, int]]:
    """Write one collision GLB per tile. Returns {tile: (file name, triangles)}."""
    out_dir = config.OUTPUT_DIR / "collision"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[TileKey, Tuple[str, int]] = {}

    for (tx, tz), tile in sorted(tiles.items()):
        scene = trimesh.Scene()
        triangles = 0
        for group in GROUPS:
            buffer = tile.buffers.get(group)
            if buffer is None or not buffer.faces:
                continue
            mesh = trimesh.Trimesh(vertices=np.asarray(buffer.vertices, dtype=np.float64),
                                   faces=np.asarray(buffer.faces, dtype=np.int64),
                                   process=False)
            node_name = f"{group}{GODOT_SUFFIX}"
            scene.add_geometry(mesh, node_name=node_name, geom_name=node_name)
            triangles += len(mesh.faces)
            stats.triangles[group] += len(mesh.faces)
        if triangles == 0:
            continue

        file_name = f"collision_{tx}_{tz}.glb"
        try:
            data = scene.export(file_type="glb")
            (out_dir / file_name).write_bytes(data)
        except (OSError, ValueError) as error:
            print(f"[error] {file_name}: {error}")
            stats.warnings += 1
            continue
        stats.total_bytes += len(data)
        written[(tx, tz)] = (file_name, triangles)
    return written


def update_city_index(written: Dict[TileKey, Tuple[str, int]]) -> Optional[int]:
    """Add collision file names to city_index.json. Returns tiles updated, or None on error."""
    path: Path = config.OUTPUT_DIR / "city_index.json"
    if not path.exists():
        print(f"[error] {path} not found. Run 05_tiles.py first.")
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            meta = json.load(file)
        updated = 0
        for record in meta.get("tiles", []):
            entry = written.get((int(record["tx"]), int(record["tz"])))
            if entry is not None:
                record["collision_file"], record["collision_triangles"] = entry
                updated += 1
        meta["folders"] = {"tiles": "tiles", "collision": "collision"}
        with open(path, "w", encoding="utf-8") as file:
            json.dump(meta, file, indent=2)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"[error] cannot update {path}: {error}")
        return None
    return updated


def main() -> int:
    config.ensure_directories()
    try:
        road_elements = nature_mod.load_layer("roads")
        building_elements = nature_mod.load_layer("buildings")
        water_elements = nature_mod.load_layer("water")
    except (FileNotFoundError, RuntimeError) as error:
        print(f"[error] {error}")
        return 1

    stats = Stats()
    tiles: Dict[TileKey, TileBuild] = defaultdict(TileBuild)

    water_shapes = nature_mod.collect_water(water_elements)
    water = unary_union(water_shapes).buffer(0) if water_shapes else Polygon()

    print("Building ground and water floor ...")
    build_ground_and_water(tiles, water, stats)
    print("Building road collision ...")
    build_road_collision(road_elements, tiles, stats)
    print("Building building collision ...")
    build_building_collision(building_elements, tiles, stats)
    if INCLUDE_PROPS:
        print("Collecting tree and lamp collision ...")
        build_prop_collision(tiles, stats)

    written = export_tiles(tiles, stats)
    if not written:
        print("[error] no collision tile was written.")
        return 1
    updated = update_city_index(written)
    if updated is None:
        return 1

    print(f"Roads             : {stats.roads}")
    print(f"Buildings         : {stats.buildings}")
    print(f"Prop meshes       : {stats.props_meshes}")
    print(f"Triangles         : {dict(stats.triangles)}")
    print(f"Warnings          : {stats.warnings}")
    print(f"Collision tiles   : {len(written)}  ({stats.total_bytes / 1048576:.1f} MB)")
    print(f"Index entries     : {updated} updated in city_index.json")
    print("Collision OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
