"""
materials.py
Step 8 of the Dubai 3D City Generator: textures and Godot materials.

The tile GLBs from steps 02-07 carry only flat colours. This script makes the
city look real by generating, procedurally and reproducibly:

  * PNG textures in config.TEXTURE_DIR (albedo, normal map and, for windows,
    an emission map that lights up at night):
        facades  : glass_blue, glass_dark, landmark_glass, concrete_light,
                   concrete_warm, sandstone, industrial
        surfaces : asphalt, sidewalk, ground (sand), grass, roof, water
  * One Godot 4 StandardMaterial3D (.tres) per material name used by the
    tiles, in <repo>/godot_materials/, plus materials_index.json.

Why .tres files instead of re-embedding textures into every GLB: the GLBs stay
small, and Godot loads each texture once and shares it between all tiles.

UV conventions (set by the earlier steps, handled here via uv1_scale):
  * Facade walls: one UV unit = one window bay (3 m) wide and one floor
    (3.2 m) high. Each facade texture is an atlas of 4 x 4 bays with varied
    windows, so the material scales UV by 0.25 and the pattern repeats every
    4 bays / 4 floors.
  * Roads, roofs, ground, grass, water: one UV unit = 4 m.
  * Trees, fronds and lamps have no UVs and keep their plain colours.

Godot setup (done automatically with --export-to):
    textures   ->  <godot project>/city/textures/
    materials  ->  <godot project>/city/materials/
The next Godot script (CityLoader.gd) swaps the plain GLB materials for these
by material name, using godot_materials/materials_index.json.

Night windows: materials with "night_emission": true have
emission_energy_multiplier = 0 (day). Set it to about 1.5 at night.

Usage:
    python materials.py
    python materials.py --size 512 --export-to /path/to/godot_project

Requires:
    pip install numpy pillow trimesh
"""

import argparse
import importlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:
    sys.exit("[error] Pillow is missing. Run: pip install pillow")

import config

# Files that start with a digit cannot be imported with a normal "import".
try:
    roads_mod = importlib.import_module("02_roads")
    buildings_mod = importlib.import_module("03_buildings")
    nature_mod = importlib.import_module("04_nature")
    landmarks_mod = importlib.import_module("07_landmarks")
except ImportError as import_error:
    sys.exit(f"[error] cannot load helper scripts: {import_error}")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED: int = 20261010
DEFAULT_FACADE_SIZE: int = 512
FACADE_BAYS: int = 4                      # windows across one texture
FACADE_FLOORS: int = 4                    # floors up one texture
GODOT_ROOT: str = "res://city"            # where assets live inside the Godot project
MATERIAL_DIR: Path = config.BASE_DIR / "godot_materials"

WARM_LIGHT: np.ndarray = np.array([255, 214, 150]) / 255.0

Color3 = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def to_uint8(array: np.ndarray) -> np.ndarray:
    """Convert floats in [0, 1] to 8-bit values."""
    return np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8)


def save_png(path: Path, array: np.ndarray) -> None:
    """Write an HxWx3 (or HxW) uint8/float image as PNG."""
    data = array if array.dtype == np.uint8 else to_uint8(array)
    Image.fromarray(data).save(path, optimize=True)


def colored_noise(size: int, rng: np.random.Generator, power: float) -> np.ndarray:
    """
    Tileable noise in [0, 1]. A higher power gives smoother, larger blobs.
    Built in the frequency domain, so it is periodic and tiles without seams.
    """
    spectrum = np.fft.fft2(rng.standard_normal((size, size)))
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    radius = np.sqrt(fx ** 2 + fy ** 2)
    radius[0, 0] = 1.0
    spectrum = spectrum / radius ** power
    spectrum[0, 0] = 0.0
    noise = np.real(np.fft.ifft2(spectrum))
    noise -= noise.min()
    return noise / max(float(noise.max()), 1e-9)


def height_to_normal(height: np.ndarray, strength: float) -> np.ndarray:
    """Turn a height map into an OpenGL-style tangent-space normal map (Godot)."""
    d_col = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5
    d_row = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5
    nx, ny, nz = -d_col * strength, d_row * strength, np.ones_like(height)
    length = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    normal = np.stack([nx / length, ny / length, nz / length], axis=-1)
    return to_uint8(normal * 0.5 + 0.5)


def rgb(color: Color3) -> np.ndarray:
    """0-255 colour to 0-1 floats."""
    return np.asarray(color, dtype=np.float64) / 255.0


# ---------------------------------------------------------------------------
# Facade textures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FacadeStyle:
    """Look of one facade material."""
    wall: Color3                      # wall colour (mullion colour for curtain walls)
    glass: Color3
    curtain_wall: bool = False        # True = whole facade is glass panels
    window_w: float = 0.55            # window size as a fraction of one bay
    window_h: float = 0.55
    frame: Color3 = (205, 205, 200)
    frame_px: int = 3
    spandrel: float = 0.18            # curtain wall: dark band under each floor
    lit_fraction: float = 0.30        # share of windows lit at night
    ribs: int = 0                     # industrial: vertical ribs per bay
    wall_noise: float = 0.06


FACADES: Dict[str, FacadeStyle] = {
    "glass_blue": FacadeStyle(wall=(120, 130, 140), glass=(70, 110, 145),
                              curtain_wall=True, lit_fraction=0.30),
    "glass_dark": FacadeStyle(wall=(70, 76, 84), glass=(38, 50, 64),
                              curtain_wall=True, lit_fraction=0.25),
    "landmark_glass": FacadeStyle(wall=(190, 195, 200), glass=(150, 178, 205),
                                  curtain_wall=True, spandrel=0.12, lit_fraction=0.20),
    "concrete_light": FacadeStyle(wall=(190, 188, 182), glass=(60, 85, 105),
                                  window_w=0.50, window_h=0.55, lit_fraction=0.35),
    "concrete_warm": FacadeStyle(wall=(205, 190, 165), glass=(60, 85, 105),
                                 window_w=0.50, window_h=0.55, lit_fraction=0.35),
    "sandstone": FacadeStyle(wall=(214, 190, 150), glass=(55, 75, 95),
                             window_w=0.40, window_h=0.60, frame=(238, 224, 196),
                             lit_fraction=0.35),
    "industrial": FacadeStyle(wall=(150, 152, 155), glass=(70, 80, 90),
                              window_w=0.60, window_h=0.18, ribs=6,
                              wall_noise=0.08, lit_fraction=0.10),
}


def _vertical_gradient(rows: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, rows)[:, None]


def _draw_window(albedo: np.ndarray, emission: np.ndarray, height: np.ndarray,
                 x0: int, y0: int, w: int, h: int, style: FacadeStyle,
                 tone: float, lit: bool, rng: np.random.Generator) -> None:
    """One recessed window with frame and sill inside a bay cell."""
    fp = style.frame_px
    ww = max(4, int(w * style.window_w))
    wh = max(4, int(h * style.window_h))
    wx = x0 + (w - ww) // 2
    wy = y0 + int(h * 0.20)

    albedo[wy - fp:wy + wh + fp, wx - fp:wx + ww + fp] = rgb(style.frame)
    height[wy - fp:wy + wh + fp, wx - fp:wx + ww + fp] = 0.5

    shade = 1.15 - 0.35 * _vertical_gradient(wh)
    albedo[wy:wy + wh, wx:wx + ww] = rgb(style.glass) * tone * shade[..., None]
    height[wy:wy + wh, wx:wx + ww] = -1.0
    if lit:
        emission[wy:wy + wh, wx:wx + ww] = WARM_LIGHT * rng.uniform(0.55, 1.0)

    sill_y = wy + wh + fp
    albedo[sill_y:sill_y + 2, wx - fp - 2:wx + ww + fp + 2] = rgb(style.frame) * 0.95
    height[sill_y:sill_y + 2, wx - fp - 2:wx + ww + fp + 2] = 0.4


def _draw_curtain_panel(albedo: np.ndarray, emission: np.ndarray, height: np.ndarray,
                        x0: int, y0: int, w: int, h: int, style: FacadeStyle,
                        tone: float, lit: bool, rng: np.random.Generator) -> None:
    """One glass panel of a curtain wall with a dark slab band (spandrel) below."""
    mullion = style.frame_px
    top, bottom = y0 + mullion, y0 + h - mullion
    left, right = x0 + mullion, x0 + w - mullion
    spandrel_rows = int((bottom - top) * style.spandrel)
    vision_rows = (bottom - top) - spandrel_rows
    vision_cols = right - left

    yy = np.linspace(0.0, 1.0, vision_rows)[:, None]
    xx = np.linspace(0.0, 1.0, vision_cols)[None, :]
    shade = 1.12 - 0.30 * yy + 0.10 * np.sin((xx + yy) * np.pi * 1.5)
    albedo[top:top + vision_rows, left:right] = rgb(style.glass) * tone * shade[..., None]
    albedo[top + vision_rows:bottom, left:right] = rgb(style.glass) * 0.45 * tone
    height[top:top + vision_rows, left:right] = -0.6
    height[top + vision_rows:bottom, left:right] = -0.2
    if lit:
        emission[top:top + vision_rows, left:right] = WARM_LIGHT * rng.uniform(0.5, 0.95)


def make_facade(style: FacadeStyle, size: int,
                rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (albedo, emission, height) of a 4x4-bay facade atlas."""
    albedo = np.zeros((size, size, 3))
    emission = np.zeros((size, size, 3))
    height = np.zeros((size, size))

    grain = colored_noise(size, rng, 1.2)
    albedo[:] = rgb(style.wall) * (1.0 + (grain[..., None] - 0.5) * 2.0 * style.wall_noise)

    if style.ribs:
        columns = np.arange(size)
        rib = np.sin(2.0 * np.pi * columns * style.ribs * FACADE_BAYS / size)
        height += 0.6 * rib[None, :]
        albedo *= (1.0 + 0.06 * rib[None, :, None])

    cell_w, cell_h = size // FACADE_BAYS, size // FACADE_FLOORS
    for floor in range(FACADE_FLOORS):
        for bay in range(FACADE_BAYS):
            lit = bool(rng.random() < style.lit_fraction)
            tone = 1.0 + rng.uniform(-0.10, 0.10)
            draw = _draw_curtain_panel if style.curtain_wall else _draw_window
            draw(albedo, emission, height, bay * cell_w, floor * cell_h,
                 cell_w, cell_h, style, tone, lit, rng)
    return albedo, emission, height


# ---------------------------------------------------------------------------
# Surface textures (each tile of the texture covers 4 m, scaled in the material)
# ---------------------------------------------------------------------------
def make_asphalt(size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    grain = colored_noise(size, rng, 0.4)
    blotch = colored_noise(size, rng, 2.0)
    value = 0.16 + 0.07 * grain + 0.04 * blotch
    value[rng.random((size, size)) > 0.995] += 0.18          # bright stone chips
    return value[..., None] * np.array([1.0, 1.0, 1.03]), grain * 0.5


def make_sidewalk(size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    tiles = 8                                                   # 0.5 m paving stones
    step = size // tiles
    base = rgb((150, 148, 142))
    albedo = np.zeros((size, size, 3))
    height = np.zeros((size, size))
    for row in range(tiles):
        for col in range(tiles):
            albedo[row * step:(row + 1) * step, col * step:(col + 1) * step] = \
                base * (1.0 + rng.uniform(-0.06, 0.06))
    joint = 3
    for k in range(tiles):
        albedo[k * step:k * step + joint, :] = base * 0.62
        albedo[:, k * step:k * step + joint] = base * 0.62
        height[k * step:k * step + joint, :] = -1.0
        height[:, k * step:k * step + joint] = -1.0
    grain = colored_noise(size, rng, 0.5)
    return albedo * (0.94 + 0.12 * grain[..., None]), height + grain * 0.3


def make_sand(size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    ripple = colored_noise(size, rng, 2.2)
    grain = colored_noise(size, rng, 0.3)
    tone = 0.94 + 0.12 * ripple + 0.04 * grain
    return rgb((218, 200, 165)) * tone[..., None], ripple * 1.5 + grain * 0.2


def make_grass(size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    patches = colored_noise(size, rng, 1.6)
    blades = colored_noise(size, rng, 0.35)
    tone = 0.72 + 0.45 * patches + 0.15 * blades
    albedo = rgb((76, 120, 52)) * tone[..., None]
    albedo[..., 0] *= 1.0 + 0.15 * patches                      # dry yellowish patches
    return albedo, blades


def make_roof(size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    gravel = colored_noise(size, rng, 0.3)
    blotch = colored_noise(size, rng, 1.8)
    tone = 0.88 + 0.16 * gravel + 0.08 * blotch
    return rgb((95, 95, 98)) * tone[..., None], gravel


def make_water(size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    ripple = colored_noise(size, rng, 2.0)
    albedo = rgb((28, 86, 120)) * (0.9 + 0.2 * ripple[..., None])
    return albedo, ripple * 2.0


# material name -> (generator, normal strength, UV scale applied in the material)
SURFACES = {
    "asphalt": (make_asphalt, 2.0, 1.0),
    "sidewalk": (make_sidewalk, 3.0, 1.0),
    "ground": (make_sand, 2.0, 0.25),
    "grass": (make_grass, 1.5, 0.5),
    "roof": (make_roof, 2.0, 1.0),
    "water": (make_water, 3.0, 0.25),
}
WATER_ALBEDO_FLAT: bool = True            # water keeps its plain colour, only gets ripples


# ---------------------------------------------------------------------------
# Texture generation
# ---------------------------------------------------------------------------
@dataclass
class TextureSet:
    """File names (inside the texture folder) of one material's textures."""
    albedo: Optional[str] = None
    normal: Optional[str] = None
    emission: Optional[str] = None
    uv_scale: float = 1.0


def build_textures(out_dir: Path, facade_size: int) -> Dict[str, TextureSet]:
    """Generate and save every texture. Returns {material name: TextureSet}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    surface_size = max(128, facade_size // 2)
    sets: Dict[str, TextureSet] = {}

    for name, style in FACADES.items():
        albedo, emission, height = make_facade(style, facade_size, rng)
        save_png(out_dir / f"{name}.png", albedo)
        save_png(out_dir / f"{name}_n.png", height_to_normal(height, 2.5))
        save_png(out_dir / f"{name}_e.png", emission)
        sets[name] = TextureSet(f"{name}.png", f"{name}_n.png", f"{name}_e.png",
                                uv_scale=1.0 / FACADE_BAYS)
        print(f"  [texture] {name}")

    for name, (generator, strength, uv_scale) in SURFACES.items():
        albedo, height = generator(surface_size, rng)
        textures = TextureSet(normal=f"{name}_n.png", uv_scale=uv_scale)
        save_png(out_dir / f"{name}_n.png", height_to_normal(height, strength))
        if not (name == "water" and WATER_ALBEDO_FLAT):
            save_png(out_dir / f"{name}.png", albedo)
            textures.albedo = f"{name}.png"
        sets[name] = textures
        print(f"  [texture] {name}")
    return sets


# ---------------------------------------------------------------------------
# Godot materials
# ---------------------------------------------------------------------------
def collect_materials() -> Dict[str, "object"]:
    """All PBR materials used by the tile scripts, keyed by name."""
    merged: Dict[str, object] = {}
    for module in (roads_mod, nature_mod, buildings_mod):
        merged.update(module.MATERIALS)
    merged.update(landmarks_mod.LANDMARK_MATERIALS)
    return merged


def _unit_color(values, default: Tuple[float, ...]) -> Tuple[float, ...]:
    """Normalise a colour that may be stored as 0-255 or 0-1."""
    if values is None:
        return default
    array = np.asarray(values, dtype=np.float64)
    if array.max() > 1.0:
        array = array / 255.0
    return tuple(float(v) for v in array)


def write_material(path: Path, name: str, pbr, textures: Optional[TextureSet],
                   texture_root: str) -> bool:
    """Write one StandardMaterial3D .tres. Returns True if it has night emission."""
    base = _unit_color(getattr(pbr, "baseColorFactor", None), (1.0, 1.0, 1.0, 1.0))
    base = (base + (1.0,))[:4] if len(base) == 3 else base[:4]
    roughness = getattr(pbr, "roughnessFactor", None)
    metallic = getattr(pbr, "metallicFactor", None)
    emissive = _unit_color(getattr(pbr, "emissiveFactor", None), (0.0, 0.0, 0.0))
    has_emissive_factor = max(emissive) > 0.0
    double_sided = bool(getattr(pbr, "doubleSided", False))

    resources: List[Tuple[str, str]] = []              # (id, path)

    def add_resource(file_name: Optional[str]) -> Optional[str]:
        if not file_name:
            return None
        resource_id = str(len(resources) + 1)
        resources.append((resource_id, f"{texture_root}/{file_name}"))
        return resource_id

    albedo_id = add_resource(textures.albedo if textures else None)
    normal_id = add_resource(textures.normal if textures else None)
    emission_id = add_resource(textures.emission if textures else None)
    night = bool(emission_id) or has_emissive_factor

    lines: List[str] = [f'[gd_resource type="StandardMaterial3D" load_steps={len(resources) + 1} format=3]', ""]
    for resource_id, resource_path in resources:
        lines.append(f'[ext_resource type="Texture2D" path="{resource_path}" id="{resource_id}"]')
    lines += ["", "[resource]", f'resource_name = "{name}"']

    colour = (1.0, 1.0, 1.0, base[3]) if albedo_id else base
    lines.append(f"albedo_color = Color({colour[0]:.4f}, {colour[1]:.4f}, {colour[2]:.4f}, {colour[3]:.4f})")
    if albedo_id:
        lines.append(f'albedo_texture = ExtResource("{albedo_id}")')
    lines.append(f"metallic = {0.0 if metallic is None else float(metallic):.3f}")
    lines.append(f"roughness = {1.0 if roughness is None else float(roughness):.3f}")
    if normal_id:
        lines += ["normal_enabled = true", f'normal_texture = ExtResource("{normal_id}")']
    if night:
        lines.append("emission_enabled = true")
        if emission_id:
            lines.append("emission = Color(1, 1, 1, 1)")
            lines.append(f'emission_texture = ExtResource("{emission_id}")')
        else:
            lines.append(f"emission = Color({emissive[0]:.4f}, {emissive[1]:.4f}, {emissive[2]:.4f}, 1)")
        lines.append("emission_energy_multiplier = 0.0")      # day; raise at night
    if textures and textures.uv_scale != 1.0:
        lines.append(f"uv1_scale = Vector3({textures.uv_scale}, {textures.uv_scale}, 1)")
    if double_sided:
        lines.append("cull_mode = 2")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return night


def build_materials(sets: Dict[str, TextureSet]) -> Dict[str, dict]:
    """Write all .tres files and return the index entries."""
    MATERIAL_DIR.mkdir(parents=True, exist_ok=True)
    texture_root = f"{GODOT_ROOT}/textures"
    index: Dict[str, dict] = {}
    for name, pbr in sorted(collect_materials().items()):
        night = write_material(MATERIAL_DIR / f"{name}.tres", name, pbr,
                               sets.get(name), texture_root)
        index[name] = {"resource": f"{GODOT_ROOT}/materials/{name}.tres",
                       "textured": name in sets,
                       "night_emission": night}
        print(f"  [material] {name}{'  (night lights)' if night else ''}")
    return index


def export_to_godot(project: Path) -> None:
    """Copy textures and materials into <project>/city/."""
    targets = {config.TEXTURE_DIR: project / "city" / "textures",
               MATERIAL_DIR: project / "city" / "materials"}
    for source, target in targets.items():
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, dirs_exist_ok=True)
    print(f"Copied textures and materials into {project / 'city'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate textures and Godot materials.")
    parser.add_argument("--size", type=int, default=DEFAULT_FACADE_SIZE,
                        help="facade texture size in pixels (default 512; use 256 for smaller files)")
    parser.add_argument("--export-to", type=Path, default=None,
                        help="Godot project folder to copy the results into")
    args = parser.parse_args()
    if args.size < 128 or args.size % FACADE_BAYS != 0:
        print(f"[error] --size must be at least 128 and divisible by {FACADE_BAYS}")
        return 1

    config.ensure_directories()
    try:
        print("Generating textures ...")
        sets = build_textures(config.TEXTURE_DIR, args.size)
        print("Writing Godot materials ...")
        index = build_materials(sets)
        with open(MATERIAL_DIR / "materials_index.json", "w", encoding="utf-8") as file:
            json.dump({"godot_root": GODOT_ROOT, "materials": index}, file, indent=2)
        if args.export_to is not None:
            export_to_godot(args.export_to)
    except OSError as error:
        print(f"[error] {error}")
        return 1

    total = sum(p.stat().st_size for p in config.TEXTURE_DIR.glob("*.png"))
    print(f"Textures  : {len(list(config.TEXTURE_DIR.glob('*.png')))} files, {total / 1048576:.1f} MB")
    print(f"Materials : {len(index)}  ->  {MATERIAL_DIR}")
    print("Materials OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
