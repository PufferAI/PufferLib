"""Pack Craftax upstream 16x16 PNG assets into a single shared textures.bin.

Consumed by both ocean/craftax (full) and ocean/craftax_classic. All files
live in craftax's asset dir; the classic PNGs that overlap are byte-identical
to the full ones.

Layout: contiguous 16*16*4 RGBA tiles. Order must match the
CRAFTAX_TEX_* / CC_TEX_* enums in the two env headers.

  [0..36]  block textures (37) -- BlockType; first 17 entries also valid for classic
  [37..41] player: down, up, left, right, sleep
  [42..46] items: none(blank), torch, ladder_down, ladder_up, ladder_down_blocked
  [47..49] mobs: zombie, skeleton, cow
  [50..53] arrows: down, up, left, right
"""

from pathlib import Path
from PIL import Image
import numpy as np

ASSETS = Path(__file__).resolve().parents[2] / (
    ".venv/lib/python3.12/site-packages/craftax/craftax/assets"
)
OUT_BIN = Path(__file__).resolve().parents[2] / "resources" / "craftax" / "textures.bin"

TILE = 16

BLOCK_FILES = [
    "debug_tile.png",            # 0 INVALID
    "debug_tile.png",            # 1 OUT_OF_BOUNDS (overwritten solid grey below)
    "grass.png",                 # 2
    "water.png",                 # 3
    "stone.png",                 # 4
    "tree.png",                  # 5
    "wood.png",                  # 6
    "path.png",                  # 7
    "coal.png",                  # 8
    "iron.png",                  # 9
    "diamond.png",               # 10
    "table.png",                 # 11 crafting table
    "furnace.png",               # 12
    "sand.png",                  # 13
    "lava.png",                  # 14
    "plant_on_grass.png",        # 15
    "ripe_plant_on_grass.png",   # 16
    "wall2.png",                 # 17
    "debug_tile.png",            # 18 DARKNESS (overwritten solid black below)
    "wall_moss.png",             # 19
    "stalagmite.png",            # 20
    "sapphire.png",              # 21
    "ruby.png",                  # 22
    "chest.png",                 # 23
    "fountain.png",              # 24
    "fire_grass.png",            # 25
    "ice_grass.png",             # 26
    "gravel.png",                # 27
    "fire_tree.png",             # 28
    "ice_shrub.png",             # 29
    "enchantment_table_fire.png",# 30
    "enchantment_table_ice.png", # 31
    "necromancer.png",           # 32
    "grave.png",                 # 33
    "grave2.png",                # 34
    "grave3.png",                # 35
    "necromancer_vulnerable.png",# 36
]

PLAYER_FILES = [
    "player-down.png",
    "player-up.png",
    "player-left.png",
    "player-right.png",
    "player-sleep.png",
]

ITEM_FILES = [
    None,                        # NONE -> fully transparent
    "torch_on_path.png",
    "ladder_down.png",
    "ladder_up.png",
    "ladder_down_blocked.png",
]

MOB_FILES = [
    "zombie.png",
    "skeleton.png",
    "cow.png",
]

ARROW_FILES = [
    "arrow-down.png",
    "arrow-up.png",
    "arrow-left.png",
    "arrow-right.png",
]


def load_tile(name: str | None) -> np.ndarray:
    if name is None:
        return np.zeros((TILE, TILE, 4), dtype=np.uint8)
    p = ASSETS / name
    img = Image.open(p).convert("RGBA").resize((TILE, TILE), Image.NEAREST)
    return np.asarray(img, dtype=np.uint8)


def main() -> None:
    tiles: list[np.ndarray] = []
    for f in BLOCK_FILES:
        tiles.append(load_tile(f))

    # manual overrides to match upstream renderer
    tiles[1] = np.full((TILE, TILE, 4), 128, dtype=np.uint8)
    tiles[1][..., 3] = 255  # out of bounds
    tiles[18] = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    tiles[18][..., 3] = 255  # darkness

    for f in PLAYER_FILES:
        tiles.append(load_tile(f))

    # torch_in_walls doesn't exist in assets; fall back to torch.png if needed
    for f in ITEM_FILES:
        if f is not None and not (ASSETS / f).exists():
            alt = "torch.png" if "torch" in f else f
            tiles.append(load_tile(alt))
        else:
            tiles.append(load_tile(f))

    for f in MOB_FILES + ARROW_FILES:
        tiles.append(load_tile(f))

    blob = np.stack(tiles, axis=0)  # (N, 16, 16, 4) uint8
    assert blob.dtype == np.uint8
    OUT_BIN.write_bytes(blob.tobytes(order="C"))
    print(f"wrote {OUT_BIN} — {blob.shape[0]} tiles, {OUT_BIN.stat().st_size} bytes")


if __name__ == "__main__":
    main()
