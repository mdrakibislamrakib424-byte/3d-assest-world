"""
01_download.py
Step 1 of the Dubai 3D City Generator: download raw map data.

Downloads building, road, water and nature layers from OpenStreetMap
(Overpass API) for the BBOX defined in config.py and saves each layer as
a JSON file inside DATA_DIR.

Usage:
    python 01_download.py            # skips layers that already exist
    python 01_download.py --force    # re-downloads everything
"""

import argparse
import json
import sys
import time
from typing import Dict, Optional

import requests

import config

MAX_RETRIES: int = 4
PAUSE_BETWEEN_LAYERS_S: float = 3.0
HEADERS: Dict[str, str] = {"User-Agent": "dubai-city-generator/1.0 (personal game project)"}


def build_queries() -> Dict[str, str]:
    """Return one Overpass QL query per data layer, keyed by layer name."""
    south, west, north, east = config.BBOX
    bbox = f"{south},{west},{north},{east}"
    head = f"[out:json][timeout:{config.OVERPASS_TIMEOUT_S}];"

    return {
        "buildings": (
            f"{head}("
            f'way["building"]({bbox});'
            f'relation["building"]({bbox});'
            f");out geom;"
        ),
        "roads": (
            f"{head}("
            f'way["highway"]["highway"!~"proposed|construction|corridor|steps"]({bbox});'
            f");out geom;"
        ),
        "water": (
            f"{head}("
            f'way["natural"="water"]({bbox});'
            f'relation["natural"="water"]({bbox});'
            f'way["waterway"]({bbox});'
            f'way["natural"="coastline"]({bbox});'
            f");out geom;"
        ),
        "nature": (
            f"{head}("
            f'node["natural"="tree"]({bbox});'
            f'way["leisure"~"park|garden|pitch"]({bbox});'
            f'way["landuse"~"grass|forest|recreation_ground"]({bbox});'
            f");out geom;"
        ),
        "street_lamps": (
            f"{head}"
            f'node["highway"="street_lamp"]({bbox});'
            f"out;"
        ),
    }


def fetch_layer(name: str, query: str) -> Optional[dict]:
    """Run one Overpass query with retries. Returns parsed JSON or None."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  [{name}] request attempt {attempt}/{MAX_RETRIES} ...")
            response = requests.post(
                config.OVERPASS_URL,
                data={"data": query},
                headers=HEADERS,
                timeout=config.OVERPASS_TIMEOUT_S + 30,
            )
            if response.status_code == 200:
                return response.json()
            # 429 = too many requests, 504 = server busy: wait and retry.
            print(f"  [{name}] server replied {response.status_code}")
        except (requests.RequestException, ValueError) as error:
            print(f"  [{name}] error: {error}")
        time.sleep(10 * attempt)  # wait longer after each failure
    return None


def save_layer(name: str, payload: dict) -> int:
    """Save a layer to DATA_DIR/<name>.json. Returns the element count."""
    path = config.DATA_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file)
    return len(payload.get("elements", []))


def main() -> int:
    parser = argparse.ArgumentParser(description="Download OSM data for the city.")
    parser.add_argument("--force", action="store_true", help="re-download existing layers")
    args = parser.parse_args()

    config.ensure_directories()
    print(f"City: {config.CITY_NAME} | BBOX: {config.BBOX}")

    failed = []
    for name, query in build_queries().items():
        target = config.DATA_DIR / f"{name}.json"
        if target.exists() and not args.force:
            print(f"[skip] {name} already downloaded")
            continue

        print(f"[download] {name}")
        payload = fetch_layer(name, query)
        if payload is None:
            print(f"[FAILED] {name}")
            failed.append(name)
        else:
            count = save_layer(name, payload)
            print(f"[ok] {name}: {count} elements saved")
        time.sleep(PAUSE_BETWEEN_LAYERS_S)

    if failed:
        print(f"\nFailed layers: {', '.join(failed)}. Run the script again.")
        return 1
    print("\nDownload complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
