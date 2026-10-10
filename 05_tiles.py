"""
05_tiles.py
Step 5 of the Dubai 3D City Generator: merge layers into game-ready tiles.

Steps 02-04 write one GLB per tile for each layer (roads, buildings, nature).
This script combines the three layers of every tile into:

    output/<city>/tiles/tile_<tx>_<tz>.glb       full detail (near tiles)
    output/<city>/tiles/tile_<tx>_<tz>_far.glb   light version (distant tiles):
                                                 ground, grass, water, asphalt
                                                 and buildings only - no trees,
                                                 lamps, sidewalks or markings
    output/<city>/city_index.json                master index read by Godot

All vertices are already in WORLD coordinates (origin = centre of
config.BBOX), so Godot must instantiate every tile at position (0, 0, 0).
The tile index (tx, tz) and its bounds in the index file are only used to
decide WHICH tiles to load around the player.

Usage:
    python 05_tiles.py

Requires:
    pip install numpy trimesh
"""

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import trimesh

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TileKey = Tuple[int, int]

LAYERS: Tuple[str, ...] = ("roads", "buildings", "nature")

# Mesh names kept in the far (light) version. None = keep every mesh of the layer.
# A layer that is missing from this dict is left out completely.
FAR_KEEP: Dict[str, Optional[FrozenSet[str]]] = {
    "roads": frozenset({"asphalt"}),
    "buildings": None,
    "nature": frozenset({"ground", "grass", "water"}),
}

MAX_TILE_TRIANGLES_WARN: int = 150_000   # warn when a full tile is heavier than this
BYTES_PER_MB: float = 1024.0 * 1024.0


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class TileRecord:
    """One entry of city_index.json."""
    tx: int
    tz: int
    center_x: float
    center_z: float
    bounds: List[float]                  # [min_x, min_z, max_x, max_z] in metres
    layers: List[str]
    full_file: str
    full_triangles: int
    full_size_kb: float
    far_file: Optional[str] = None
    far_triangles: int = 0
    far_size_kb: float = 0.0


@dataclass
class Summary:
    """Counters printed at the end of the run."""
    tiles: int = 0
    skipped: int = 0
    full_triangles: int = 0
    far_triangles: int = 0
    total_bytes: int = 0
    heaviest_key: Optional[TileKey] = None
    heaviest_triangles: int = 0
    heavy_tiles: List[TileKey] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Reading the layer indexes
# ---------------------------------------------------------------------------
def load_layer_index(layer: str) -> Dict[TileKey, Path]:
    """
    Read <layer>_index.json and return {(tx, tz): path to that tile's GLB}.
    A missing index gives an empty result (with a warning). A tile-size
    mismatch is an error, because tiles would no longer line up.
    """
    folder = config.OUTPUT_DIR / layer
    path = folder / f"{layer}_index.json"
    if not path.exists():
        print(f"[warn] {path} not found - layer '{layer}' will be missing "
              f"(run its script first)")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file:
            meta = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read {path}: {error}") from error

    if abs(float(meta.get("tile_size_m", 0.0)) - config.TILE_SIZE_M) > 1e-6:
        raise RuntimeError(
            f"{path} was built with tile size {meta.get('tile_size_m')} m but "
            f"config.TILE_SIZE_M is {config.TILE_SIZE_M} m. Re-run the "
            f"02/03/04 scripts first.")

    files: Dict[TileKey, Path] = {}
    for entry in meta.get("tiles", []):
        try:
            files[(int(entry["tx"]), int(entry["tz"]))] = folder / entry["file"]
        except (KeyError, TypeError, ValueError):
            print(f"[warn] bad entry in {path}: {entry}")
    return files


def read_scene(path: Path) -> Optional[trimesh.Scene]:
    """Load one GLB as a Scene. Returns None (with a message) if it fails."""
    try:
        loaded = trimesh.load(str(path), force="scene", process=False)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        print(f"[error] cannot read {path.name}: {error}")
        return None
    if isinstance(loaded, trimesh.Scene):
        return loaded
    return trimesh.Scene(loaded)


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------
def mesh_labels(name: str, geometry: trimesh.Trimesh) -> Set[str]:
    """Names that identify a mesh: its own name and its material name."""
    labels = {name}
    material = getattr(geometry.visual, "material", None)
    material_name = getattr(material, "name", None)
    if material_name:
        labels.add(str(material_name))
    return labels


def merge_scenes(parts: Dict[str, trimesh.Scene],
                 keep: Optional[Dict[str, Optional[FrozenSet[str]]]]) -> Tuple[Optional[trimesh.Scene], int]:
    """
    Combine the layer scenes of one tile.
    keep=None keeps everything; otherwise only the layers/meshes listed in keep.
    Returns (scene, triangle_count); scene is None if nothing was kept.
    """
    merged = trimesh.Scene()
    triangles = 0
    for layer in LAYERS:
        scene = parts.get(layer)
        if scene is None:
            continue
        allowed: Optional[FrozenSet[str]] = None
        if keep is not None:
            if layer not in keep:
                continue
            allowed = keep[layer]
        for name, geometry in scene.geometry.items():
            if allowed is not None and not (mesh_labels(name, geometry) & allowed):
                continue
            new_name = f"{layer}_{name}"
            merged.add_geometry(geometry.copy(), node_name=new_name, geom_name=new_name)
            triangles += len(geometry.faces)
    return (merged, triangles) if triangles > 0 else (None, 0)


def write_scene(scene: trimesh.Scene, path: Path) -> Optional[int]:
    """Export a scene as GLB. Returns the file size in bytes, or None on failure."""
    try:
        data = scene.export(file_type="glb")
        path.write_bytes(data)
    except (OSError, ValueError) as error:
        print(f"[error] cannot write {path.name}: {error}")
        return None
    return len(data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_tile(key: TileKey, layer_files: Dict[str, Dict[TileKey, Path]],
               out_dir: Path, summary: Summary) -> Optional[TileRecord]:
    """Build the full and far GLB of one tile. Returns its index record."""
    tx, tz = key
    parts: Dict[str, trimesh.Scene] = {}
    for layer in LAYERS:
        path = layer_files[layer].get(key)
        if path is None:
            continue
        scene = read_scene(path)
        if scene is not None:
            parts[layer] = scene
    if not parts:
        summary.skipped += 1
        return None

    full_scene, full_triangles = merge_scenes(parts, keep=None)
    if full_scene is None:
        summary.skipped += 1
        return None

    full_name = f"tile_{tx}_{tz}.glb"
    full_bytes = write_scene(full_scene, out_dir / full_name)
    if full_bytes is None:
        summary.skipped += 1
        return None

    size = config.TILE_SIZE_M
    record = TileRecord(
        tx=tx, tz=tz,
        center_x=(tx + 0.5) * size, center_z=(tz + 0.5) * size,
        bounds=[tx * size, tz * size, (tx + 1) * size, (tz + 1) * size],
        layers=sorted(parts.keys()),
        full_file=full_name, full_triangles=full_triangles,
        full_size_kb=round(full_bytes / 1024.0, 1),
    )
    summary.full_triangles += full_triangles
    summary.total_bytes += full_bytes
    if full_triangles > summary.heaviest_triangles:
        summary.heaviest_triangles, summary.heaviest_key = full_triangles, key
    if full_triangles > MAX_TILE_TRIANGLES_WARN:
        summary.heavy_tiles.append(key)

    far_scene, far_triangles = merge_scenes(parts, keep=FAR_KEEP)
    if far_scene is not None:
        far_name = f"tile_{tx}_{tz}_far.glb"
        far_bytes = write_scene(far_scene, out_dir / far_name)
        if far_bytes is not None:
            record.far_file = far_name
            record.far_triangles = far_triangles
            record.far_size_kb = round(far_bytes / 1024.0, 1)
            summary.far_triangles += far_triangles
            summary.total_bytes += far_bytes
    return record


def main() -> int:
    config.ensure_directories()
    try:
        layer_files = {layer: load_layer_index(layer) for layer in LAYERS}
    except RuntimeError as error:
        print(f"[error] {error}")
        return 1

    keys = sorted(set().union(*(files.keys() for files in layer_files.values())))
    if not keys:
        print("[error] no tiles found. Run 02_roads.py, 03_buildings.py and "
              "04_nature.py first.")
        return 1

    out_dir = config.OUTPUT_DIR / "tiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Merging {len(keys)} tiles ...")

    summary = Summary()
    records: List[TileRecord] = []
    for number, key in enumerate(keys, start=1):
        record = build_tile(key, layer_files, out_dir, summary)
        if record is not None:
            records.append(record)
            summary.tiles += 1
        if number % 50 == 0:
            print(f"  {number}/{len(keys)} tiles done")

    if not records:
        print("[error] no tile could be built.")
        return 1

    meta = {
        "city": config.CITY_NAME,
        "tile_size_m": config.TILE_SIZE_M,
        "origin_lat": config.ORIGIN_LAT,
        "origin_lon": config.ORIGIN_LON,
        "bbox_south_west_north_east": list(config.BBOX),
        "coordinates": "Vertices are in world metres (x east, y up, z south). "
                       "Instantiate every tile at position (0, 0, 0).",
        "lod_distances_m": {"near": config.LOD_NEAR_M, "far": config.LOD_FAR_M},
        "tile_range": {
            "min_tx": min(r.tx for r in records), "max_tx": max(r.tx for r in records),
            "min_tz": min(r.tz for r in records), "max_tz": max(r.tz for r in records),
        },
        "tiles": [asdict(r) for r in records],
    }
    index_path = config.OUTPUT_DIR / "city_index.json"
    try:
        with open(index_path, "w", encoding="utf-8") as file:
            json.dump(meta, file, indent=2)
    except OSError as error:
        print(f"[error] cannot write {index_path}: {error}")
        return 1

    print(f"Tiles built        : {summary.tiles}  (skipped: {summary.skipped})")
    print(f"Full triangles     : {summary.full_triangles:,}")
    print(f"Far triangles      : {summary.far_triangles:,}")
    print(f"Total size         : {summary.total_bytes / BYTES_PER_MB:.1f} MB")
    if summary.heaviest_key is not None:
        print(f"Heaviest tile      : {summary.heaviest_key} "
              f"with {summary.heaviest_triangles:,} triangles")
    if summary.heavy_tiles:
        print(f"[warn] {len(summary.heavy_tiles)} tile(s) above "
              f"{MAX_TILE_TRIANGLES_WARN:,} triangles. Lower TILE_SIZE_M in "
              f"config.py if Godot runs slowly.")
    print(f"Index              : {index_path}")
    print("Tiles OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
