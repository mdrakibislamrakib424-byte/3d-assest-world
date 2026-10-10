"""
build_all.py
Step 9 of the Dubai 3D City Generator: run the whole pipeline with one command.

Runs the generator scripts in the correct order:

    download   01_download.py    OpenStreetMap data
    roads      02_roads.py       road meshes
    buildings  03_buildings.py   building meshes
    nature     04_nature.py      ground, grass, water, trees, lamps
    landmarks  07_landmarks.py   landmarks (rebuilds the building tiles)
    tiles      05_tiles.py       merged near/far tiles + city_index.json
    collision  06_collision.py   collision tiles
    materials  materials.py      textures + Godot materials

Note the order: landmarks must run BEFORE tiles, because landmarks replace
the building tiles that the merge step reads.

Before each step the script checks that its input files exist, and before
the run it checks that the needed Python libraries are installed. The run
stops at the first failing step. A report is written to
output/build_report.json.

Usage:
    python build_all.py                       run everything
    python build_all.py --list                show the steps and exit
    python build_all.py --from tiles          run tiles, collision, materials
    python build_all.py --only roads nature   run just these steps
    python build_all.py --skip download       everything except the download
    python build_all.py --force-download      re-download the map data
    python build_all.py --export-to PATH      also copy the results into a
                                              Godot project folder (PATH/city/)

Requires:
    pip install requests numpy trimesh shapely mapbox-earcut pillow
"""

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config

# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Step:
    """One script of the pipeline and the files it needs before it can run."""
    key: str
    script: str
    title: str
    requires: Tuple[Tuple[str, str], ...] = ()   # ("data" | "output", relative path)


DATA_FILES = ("roads", "buildings", "water", "nature", "street_lamps")

STEPS: Tuple[Step, ...] = (
    Step("download", "01_download.py", "Download OpenStreetMap data"),
    Step("roads", "02_roads.py", "Build road meshes",
         (("data", "roads.json"),)),
    Step("buildings", "03_buildings.py", "Build building meshes",
         (("data", "buildings.json"),)),
    Step("nature", "04_nature.py", "Ground, grass, water, trees, lamps",
         tuple(("data", f"{name}.json") for name in DATA_FILES)),
    Step("landmarks", "07_landmarks.py", "Landmarks (rebuilds building tiles)",
         (("data", "buildings.json"),)),
    Step("tiles", "05_tiles.py", "Merge layers into near/far tiles",
         (("output", "roads/roads_index.json"),
          ("output", "buildings/buildings_index.json"),
          ("output", "nature/nature_index.json"))),
    Step("collision", "06_collision.py", "Build collision tiles",
         (("data", "roads.json"), ("data", "buildings.json"), ("data", "water.json"),
          ("output", "city_index.json"), ("output", "nature/nature_index.json"))),
    Step("materials", "materials.py", "Textures and Godot materials"),
)
STEP_KEYS: List[str] = [step.key for step in STEPS]

# Python module name -> pip package name.
REQUIRED_MODULES: Dict[str, str] = {
    "requests": "requests",
    "numpy": "numpy",
    "trimesh": "trimesh",
    "shapely": "shapely",
    "mapbox_earcut": "mapbox-earcut",
    "PIL": "pillow",
}

REPORT_NAME: str = "build_report.json"


@dataclass
class StepResult:
    """Outcome of one step, written to the build report."""
    key: str
    status: str                    # ok | failed | blocked
    seconds: float
    return_code: Optional[int] = None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def missing_modules() -> List[str]:
    """pip package names of required libraries that are not installed."""
    return [package for module, package in REQUIRED_MODULES.items()
            if importlib.util.find_spec(module) is None]


def resolve_requirement(base: str, relative: str) -> Path:
    """Turn a requirement tuple into a real path."""
    root = config.DATA_DIR if base == "data" else config.OUTPUT_DIR
    return root / relative


def missing_inputs(step: Step) -> List[Path]:
    """Input files of the step that do not exist yet."""
    return [path for path in (resolve_requirement(b, r) for b, r in step.requires)
            if not path.exists()]


def select_steps(only: Optional[List[str]], start: Optional[str],
                 skip: Optional[List[str]]) -> List[Step]:
    """Apply --only, --from and --skip while keeping the canonical order."""
    chosen = list(STEPS)
    if only:
        chosen = [step for step in chosen if step.key in only]
    if start:
        first = STEP_KEYS.index(start)
        chosen = [step for step in chosen if STEP_KEYS.index(step.key) >= first]
    if skip:
        chosen = [step for step in chosen if step.key not in skip]
    return chosen


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
def run_step(step: Step, extra_args: List[str]) -> StepResult:
    """Run one script, streaming its output. Returns the result."""
    script = config.BASE_DIR / step.script
    if not script.exists():
        print(f"[error] {step.script} not found in {config.BASE_DIR}")
        return StepResult(step.key, "failed", 0.0, None)

    started = time.perf_counter()
    try:
        completed = subprocess.run([sys.executable, str(script), *extra_args],
                                   cwd=str(config.BASE_DIR), check=False)
    except OSError as error:
        print(f"[error] cannot start {step.script}: {error}")
        return StepResult(step.key, "failed", time.perf_counter() - started, None)
    seconds = time.perf_counter() - started
    status = "ok" if completed.returncode == 0 else "failed"
    return StepResult(step.key, status, seconds, completed.returncode)


def export_to_godot(project: Path) -> None:
    """Copy tiles, collision and the indexes into <project>/city/<city>/."""
    city_root = project / "city" / config.CITY_NAME
    for folder in ("tiles", "collision"):
        source = config.OUTPUT_DIR / folder
        if source.is_dir():
            shutil.copytree(source, city_root / folder, dirs_exist_ok=True)
    city_root.mkdir(parents=True, exist_ok=True)
    for source in (config.OUTPUT_DIR / "city_index.json",
                   config.OUTPUT_DIR / "landmarks" / "landmarks_index.json"):
        if source.exists():
            shutil.copy2(source, city_root / source.name)
    print(f"Copied tiles, collision and indexes into {city_root}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def city_summary() -> Dict[str, object]:
    """Tile count and triangle totals read from city_index.json (if present)."""
    path = config.OUTPUT_DIR / "city_index.json"
    try:
        with open(path, "r", encoding="utf-8") as file:
            tiles = json.load(file).get("tiles", [])
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "tiles": len(tiles),
        "full_triangles": sum(int(t.get("full_triangles", 0)) for t in tiles),
        "far_triangles": sum(int(t.get("far_triangles", 0)) for t in tiles),
        "collision_triangles": sum(int(t.get("collision_triangles", 0)) for t in tiles),
    }


def write_report(results: List[StepResult], started_at: str, total_seconds: float) -> None:
    """Save a JSON report of the run."""
    report = {
        "started_at": started_at,
        "total_seconds": round(total_seconds, 1),
        "city": config.CITY_NAME,
        "bbox_south_west_north_east": list(config.BBOX),
        "tile_size_m": config.TILE_SIZE_M,
        "steps": [asdict(r) for r in results],
        "city": {"name": config.CITY_NAME, **city_summary()},
    }
    try:
        config.OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
        with open(config.OUTPUT_DIR.parent / REPORT_NAME, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
    except OSError as error:
        print(f"[warn] could not write the build report: {error}")


def print_summary(results: List[StepResult], total_seconds: float) -> None:
    """Print a compact table of the run."""
    print("\n" + "=" * 34)
    print(f"{'Step':<12}{'Status':<10}{'Time':>9}")
    print("-" * 34)
    for result in results:
        print(f"{result.key:<12}{result.status:<10}{result.seconds:>8.1f}s")
    print("-" * 34)
    print(f"{'Total':<22}{total_seconds:>11.1f}s")
    summary = city_summary()
    if summary:
        print(f"Tiles {summary['tiles']} | full {summary['full_triangles']:,} tris "
              f"| far {summary['far_triangles']:,} tris")
    print("=" * 34)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the whole city generator pipeline.")
    parser.add_argument("--list", action="store_true", help="show the steps and exit")
    parser.add_argument("--only", nargs="+", choices=STEP_KEYS, help="run only these steps")
    parser.add_argument("--from", dest="start", choices=STEP_KEYS,
                        help="run this step and every step after it")
    parser.add_argument("--skip", nargs="+", choices=STEP_KEYS, help="skip these steps")
    parser.add_argument("--force-download", action="store_true",
                        help="re-download map data that already exists")
    parser.add_argument("--export-to", type=Path, default=None,
                        help="Godot project folder to copy the results into")
    parser.add_argument("--no-check", action="store_true",
                        help="do not check that the Python libraries are installed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        for number, step in enumerate(STEPS, start=1):
            print(f"{number}. {step.key:<10} {step.script:<18} {step.title}")
        return 0

    selected = select_steps(args.only, args.start, args.skip)
    if not selected:
        print("[error] no steps selected.")
        return 2

    if not args.no_check:
        missing = missing_modules()
        if missing:
            print("[error] missing Python libraries. Run:")
            print(f"        pip install {' '.join(missing)}")
            return 1

    project: Optional[Path] = None
    if args.export_to is not None:
        project = args.export_to.expanduser().resolve()
        if not project.is_dir():
            print(f"[error] --export-to folder does not exist: {project}")
            return 2

    config.ensure_directories()
    print(f"City: {config.CITY_NAME} | tile size: {config.TILE_SIZE_M:g} m")
    print("Steps: " + " -> ".join(step.key for step in selected))

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_started = time.perf_counter()
    results: List[StepResult] = []
    failed = False

    try:
        for number, step in enumerate(selected, start=1):
            print(f"\n=== [{number}/{len(selected)}] {step.key}: {step.title} ===")
            missing_files = missing_inputs(step)
            if missing_files:
                print(f"[blocked] {step.key} needs files that do not exist yet:")
                for path in missing_files:
                    print(f"          {path}")
                print("          Run the earlier steps first (python build_all.py).")
                results.append(StepResult(step.key, "blocked", 0.0, None))
                failed = True
                break

            extra: List[str] = []
            if step.key == "download" and args.force_download:
                extra.append("--force")
            if step.key == "materials" and project is not None:
                extra += ["--export-to", str(project)]

            result = run_step(step, extra)
            results.append(result)
            if result.status != "ok":
                print(f"[failed] {step.key} stopped with code {result.return_code}. "
                      f"Fix the problem, then continue with: "
                      f"python build_all.py --from {step.key}")
                failed = True
                break
    except KeyboardInterrupt:
        print("\n[interrupted] stopped by the user.")
        failed = True

    if not failed and project is not None:
        try:
            export_to_godot(project)
        except OSError as error:
            print(f"[error] export to Godot failed: {error}")
            failed = True

    total = time.perf_counter() - run_started
    write_report(results, started_at, total)
    print_summary(results, total)
    print("Build FAILED" if failed else "Build OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
