"""Pack Craftax upstream 16x16 PNG assets into resources/craftax/textures.png.

Consumed by ocean/craftax (and craftax_classic). Layout is a 16-column
row-major RGBA atlas of 16x16 tiles. Order must match the TEX_* enums.

  [0..36]  block textures (37) -- BlockType; first 17 entries also valid for classic
  [37..41] player: down, up, left, right, sleep
  [42..46] items: none(blank), torch, ladder_down, ladder_up, ladder_down_blocked
  [47..49] mobs: zombie, skeleton, cow
  [50..53] arrows: down, up, left, right
  [54..61] armour: iron then diamond, each helmet/chest/pants/boots
  [62..65] pickaxes: wood, stone, iron, diamond
  [66..69] swords: wood, stone, iron, diamond
  [70]     bow
  [71..76] potions: red, green, blue, pink, cyan, yellow
  [77..79] HUD-only: sapling, torch_in_inventory, book
  [80..87] melee types: zombie, gnome_warrior, orc_soldier, lizard, knight, troll, pigman, frost_troll
  [88..90] passive types: cow, bat, snail
  [91..98] ranged types: skeleton, gnome_archer, orc_mage, kobold, knight_archer, deep_thing, fire_elemental, ice_elemental
  [99..102] projectiles: dagger, fireball, iceball, slimeball
  [103..104] sword enchant overlays: fire, ice
  [105..106] arrow enchant overlays: fire, ice
  [107..110] armour fire overlays: helmet, chest, pants, boots
  [111..114] armour ice overlays: helmet, chest, pants, boots
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "resources" / "craftax"
OUT_PNG = OUT_DIR / "textures.png"
SHEET_COLS = 16


def find_assets() -> Path:
    env = os.environ.get("CRAFTAX_ASSETS")
    if env:
        p = Path(env)
        if (p / "iron_helmet.png").exists():
            return p
    try:
        import craftax
        pkg = Path(craftax.__file__).resolve().parent
        for cand in (pkg / "craftax" / "assets", pkg / "assets"):
            if (cand / "iron_helmet.png").exists():
                return cand
    except ImportError:
        pass
    candidates = [
        ROOT / ".venv/lib/python3.12/site-packages/craftax/craftax/assets",
        ROOT / ".venv/lib/python3.10/site-packages/craftax/craftax/assets",
        Path.home() / "github/multitask_preplay/.venv/lib/python3.10/site-packages/craftax/craftax/assets",
    ]
    for cand in candidates:
        if (cand / "iron_helmet.png").exists():
            return cand
    raise FileNotFoundError(
        "craftax assets not found (need iron_helmet.png). "
        "Set CRAFTAX_ASSETS or install the craftax package."
    )


ASSETS = find_assets()

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

ARMOUR_FILES = [
    "iron_helmet.png",
    "iron_chestplate.png",
    "iron_pants.png",
    "iron_boots.png",
    "diamond_helmet.png",
    "diamond_chestplate.png",
    "diamond_pants.png",
    "diamond_boots.png",
]

WEAPON_FILES = [
    "wood_pickaxe.png",
    "stone_pickaxe.png",
    "iron_pickaxe.png",
    "diamond_pickaxe.png",
    "wood_sword.png",
    "stone_sword.png",
    "iron_sword.png",
    "diamond_sword.png",
    "bow.png",
]

POTION_FILES = [
    "potion_red.png",
    "potion_green.png",
    "potion_blue.png",
    "potion_pink.png",
    "potion_cyan.png",
    "potion_yellow.png",
]

HUD_ITEM_FILES = [
    "sapling.png",
    "torch_in_inventory.png",
    "book.png",
]

MELEE_TYPE_FILES = [
    "zombie.png",
    "gnome_warrior.png",
    "orc_soldier.png",
    "lizard.png",
    "knight.png",
    "troll.png",
    "pigman.png",
    "frost_troll.png",
]
PASSIVE_TYPE_FILES = [
    "cow.png",
    "bat.png",
    "snail.png",
]
RANGED_TYPE_FILES = [
    "skeleton.png",
    "gnome_archer.png",
    "orc_mage.png",
    "kobold.png",
    "knight_archer.png",
    "deep_thing.png",
    "fire_elemental.png",
    "ice_elemental.png",
]
PROJECTILE_TYPE_FILES = [
    "dagger.png",
    "fireball.png",
    "iceball.png",
    "slimeball.png",
]
ENCHANT_FILES = [
    "sword_fire_enchantment.png",
    "sword_ice_enchantment.png",
    "arrow_fire_enchantment.png",
    "arrow_ice_enchantment.png",
    "helmet_fire_enchantment.png",
    "chestplate_fire_enchantment.png",
    "pants_fire_enchantment.png",
    "boots_fire_enchantment.png",
    "helmet_ice_enchantment.png",
    "chestplate_ice_enchantment.png",
    "pants_ice_enchantment.png",
    "boots_ice_enchantment.png",
]


def load_tile(name: str | None) -> np.ndarray:
    if name is None:
        return np.zeros((TILE, TILE, 4), dtype=np.uint8)
    p = ASSETS / name
    img = Image.open(p).convert("RGBA").resize((TILE, TILE), Image.NEAREST)
    return np.asarray(img, dtype=np.uint8)


def tiles_to_sheet(tiles: list[np.ndarray]) -> Image.Image:
    n = len(tiles)
    rows = (n + SHEET_COLS - 1) // SHEET_COLS
    sheet = Image.new("RGBA", (SHEET_COLS * TILE, rows * TILE), (0, 0, 0, 0))
    for i, tile in enumerate(tiles):
        x = (i % SHEET_COLS) * TILE
        y = (i // SHEET_COLS) * TILE
        sheet.paste(Image.fromarray(tile, mode="RGBA"), (x, y))
    return sheet


def main() -> None:
    print(f"craftax assets: {ASSETS}")
    tiles: list[np.ndarray] = []
    for f in BLOCK_FILES:
        tiles.append(load_tile(f))

    tiles[1] = np.full((TILE, TILE, 4), 128, dtype=np.uint8)
    tiles[1][..., 3] = 255
    tiles[18] = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    tiles[18][..., 3] = 255

    for f in PLAYER_FILES:
        tiles.append(load_tile(f))

    for f in ITEM_FILES:
        if f is not None and not (ASSETS / f).exists():
            alt = "torch.png" if "torch" in f else f
            tiles.append(load_tile(alt))
        else:
            tiles.append(load_tile(f))

    for f in (
        MOB_FILES + ARROW_FILES + ARMOUR_FILES + WEAPON_FILES
        + POTION_FILES + HUD_ITEM_FILES
        + MELEE_TYPE_FILES + PASSIVE_TYPE_FILES + RANGED_TYPE_FILES
        + PROJECTILE_TYPE_FILES + ENCHANT_FILES
    ):
        tiles.append(load_tile(f))

    assert len(tiles) == 115, len(tiles)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet = tiles_to_sheet(tiles)
    sheet.save(OUT_PNG)
    print(f"wrote {OUT_PNG} — {len(tiles)} tiles, {sheet.size[0]}x{sheet.size[1]}")


if __name__ == "__main__":
    main()
