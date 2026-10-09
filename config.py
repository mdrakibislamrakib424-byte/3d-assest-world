"""
config.py
Dubai 3D City Generator - central configuration.

Every other script imports its settings from here. To generate a different
city, change CITY_NAME and BBOX only.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# City selection
# ---------------------------------------------------------------------------
CITY_NAME: str = "dubai"

# Bounding box in (south, west, north, east) order, degrees (WGS84).
# Start small: Downtown Dubai (Burj Khalifa + Dubai Mall area).
# Later we will widen this to the whole emirate.
BBOX: Tuple[float, float, float, float] = (25.1850, 55.2650, 25.2150, 55.2900)

# Centre point used as the (0, 0, 0) origin of the 3D world.
ORIGIN_LAT: float = (BBOX[0] + BBOX[2]) / 2.0
ORIGIN_LON: float = (BBOX[1] + BBOX[3]) / 2.0


# ---------------------------------------------------------------------------
# Folder layout
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data" / CITY_NAME          # raw OSM downloads
OUTPUT_DIR: Path = BASE_DIR / "output" / CITY_NAME      # generated GLB tiles
TEXTURE_DIR: Path = BASE_DIR / "textures"               # shared textures


# ---------------------------------------------------------------------------
# Tiling (city is split into square tiles so Godot can stream them)
# ---------------------------------------------------------------------------
TILE_SIZE_M: float = 250.0  # tile edge length in metres


# ---------------------------------------------------------------------------
# Building settings
# ---------------------------------------------------------------------------
DEFAULT_BUILDING_HEIGHT_M: float = 12.0   # used when OSM has no height tag
METERS_PER_LEVEL: float = 3.2             # height = levels * this value
MIN_BUILDING_HEIGHT_M: float = 3.0


# ---------------------------------------------------------------------------
# Road settings (width in metres, by OSM highway type)
# ---------------------------------------------------------------------------
ROAD_WIDTHS_M: Dict[str, float] = {
    "motorway": 18.0,
    "trunk": 16.0,
    "primary": 14.0,
    "secondary": 12.0,
    "tertiary": 10.0,
    "residential": 8.0,
    "service": 5.0,
}
DEFAULT_ROAD_WIDTH_M: float = 8.0
SIDEWALK_WIDTH_M: float = 2.0


# ---------------------------------------------------------------------------
# Level of detail (LOD) distances in metres, used later by Godot
# ---------------------------------------------------------------------------
LOD_NEAR_M: float = 150.0
LOD_FAR_M: float = 600.0


# ---------------------------------------------------------------------------
# OpenStreetMap download
# ---------------------------------------------------------------------------
OVERPASS_URL: str = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT_S: int = 180


def ensure_directories() -> None:
    """Create all working folders if they do not exist yet."""
    for folder in (DATA_DIR, OUTPUT_DIR, TEXTURE_DIR):
        folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print(f"City        : {CITY_NAME}")
    print(f"BBOX        : {BBOX}")
    print(f"Origin      : {ORIGIN_LAT:.5f}, {ORIGIN_LON:.5f}")
    print(f"Data folder : {DATA_DIR}")
    print(f"Output      : {OUTPUT_DIR}")
    print("Config OK")
