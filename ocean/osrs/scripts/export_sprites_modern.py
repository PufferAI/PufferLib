"""Export sprites from modern OSRS cache (OpenRS2 flat format) to PNG files.

Reads sprite archives from cache index 8 and decodes them using the
SpriteLoader format from the deobfuscated client (trailer-based format
with palette, per-frame offsets, and optional alpha channel).

Exports specific sprite IDs needed for the debug viewer GUI:
  - equipment slot backgrounds (156-165, 170)
  - prayer icons enabled/disabled (115-154, 502-509, 945-951, 1420-1427)
  - tab icons (168, 776, 779, 780, 900, 901)
  - spell icons (325-336, 375-386, 557, 561, 564, 607, 611, 614)
  - combat interface sprites (657)

Usage:
  uv run python scripts/export_sprites_modern.py \
    --cache ../../../.refs/osrs-cache-modern \
    --output data/sprites/gui
"""

import argparse
import io
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

# add parent for modern_cache_reader import
sys.path.insert(0, str(Path(__file__).parent))
from modern_cache_reader import ModernCacheReader

DEFAULT_MODERN_CACHE = Path(__file__).resolve().parents[3] / ".refs" / "osrs-cache-modern"


@dataclass
class SpriteFrame:
    """Single sprite frame decoded from cache archive."""

    group_id: int = 0
    frame: int = 0
    offset_x: int = 0
    offset_y: int = 0
    width: int = 0
    height: int = 0
    max_width: int = 0
    max_height: int = 0
    pixels: list[int] = field(default_factory=list)  # ARGB int array


def decode_sprites(group_id: int, data: bytes) -> list[SpriteFrame]:
    """Decode sprite archive using SpriteLoader format from deob client.

    Ported from SpriteLoader.java (runelite-cache). Format is trailer-based:
      [pixel data for all frames]
      [palette: (palette_len - 1) x 3 bytes RGB]
      [per-frame: offset_x, offset_y, width, height as u16 arrays] x frame_count
      [max_width u16, max_height u16, palette_len_minus1 u8]  (5 bytes)
      [frame_count u16]  (last 2 bytes)
    """
    if len(data) < 2:
        return []

    buf = io.BytesIO(data)

    # trailer: frame_count at very end (SpriteLoader.java line 41)
    buf.seek(len(data) - 2)
    frame_count = struct.unpack(">H", buf.read(2))[0]
    if frame_count == 0:
        return []

    # header block: 5 bytes before per-frame data before frame_count
    # (SpriteLoader.java line 48)
    header_start = len(data) - 7 - frame_count * 8
    buf.seek(header_start)

    max_width = struct.unpack(">H", buf.read(2))[0]
    max_height = struct.unpack(">H", buf.read(2))[0]
    # SpriteLoader.java line 53: paletteLength = readUnsignedByte() + 1
    palette_len = buf.read(1)[0] + 1

    # per-frame dimensions: 4 arrays of frame_count u16 values
    # (SpriteLoader.java lines 64-82)
    offsets_x = [struct.unpack(">H", buf.read(2))[0] for _ in range(frame_count)]
    offsets_y = [struct.unpack(">H", buf.read(2))[0] for _ in range(frame_count)]
    widths = [struct.unpack(">H", buf.read(2))[0] for _ in range(frame_count)]
    heights = [struct.unpack(">H", buf.read(2))[0] for _ in range(frame_count)]

    # palette: (palette_len - 1) RGB entries before the header block
    # (SpriteLoader.java line 85)
    palette_start = header_start - (palette_len - 1) * 3
    buf.seek(palette_start)
    palette = [0] * palette_len  # index 0 = transparent
    for i in range(1, palette_len):
        r = buf.read(1)[0]
        g = buf.read(1)[0]
        b = buf.read(1)[0]
        rgb = (r << 16) | (g << 8) | b
        palette[i] = rgb if rgb != 0 else 1

    # pixel data from start of file (SpriteLoader.java line 98)
    buf.seek(0)
    frames = []
    for fi in range(frame_count):
        w = widths[fi]
        h = heights[fi]
        dimension = w * h

        # per-frame arrays matching Java's byte[] layout
        indices = [0] * dimension
        alphas = [0] * dimension

        flags = buf.read(1)[0]

        # read palette indices (SpriteLoader.java lines 113-131)
        if not (flags & 0x01):
            # horizontal
            for j in range(dimension):
                indices[j] = buf.read(1)[0]
        else:
            # vertical: iterate columns then rows
            for j in range(w):
                for k in range(h):
                    indices[w * k + j] = buf.read(1)[0]

        # read alpha channel if FLAG_ALPHA (SpriteLoader.java lines 134-155)
        if flags & 0x02:
            if not (flags & 0x01):
                for j in range(dimension):
                    alphas[j] = buf.read(1)[0]
            else:
                for j in range(w):
                    for k in range(h):
                        alphas[w * k + j] = buf.read(1)[0]

        # force opaque for all non-zero palette indices
        # (SpriteLoader.java lines 157-166 — runs AFTER alpha read)
        for j in range(dimension):
            if indices[j] != 0:
                alphas[j] = 0xFF

        # build ARGB pixels (SpriteLoader.java lines 168-176)
        pixels = [0] * dimension
        for j in range(dimension):
            idx = indices[j] & 0xFF
            pixels[j] = palette[idx] | (alphas[j] << 24)

        frame = SpriteFrame(
            group_id=group_id,
            frame=fi,
            offset_x=offsets_x[fi],
            offset_y=offsets_y[fi],
            width=w,
            height=h,
            max_width=max_width,
            max_height=max_height,
            pixels=pixels,
        )
        frames.append(frame)

    return frames


def save_sprite_png(sprite: SpriteFrame, path: Path) -> None:
    """Save a sprite frame as RGBA PNG using pure Python (no PIL dependency)."""
    try:
        from PIL import Image
    except ImportError:
        print(f"  pillow not available, skipping {path}", file=sys.stderr)
        return

    img = Image.new("RGBA", (sprite.width, sprite.height))
    rgba_data = []
    for argb in sprite.pixels:
        a = (argb >> 24) & 0xFF
        r = (argb >> 16) & 0xFF
        g = (argb >> 8) & 0xFF
        b = argb & 0xFF
        rgba_data.append((r, g, b, a))
    img.putdata(rgba_data)
    img.save(str(path))


# sprite IDs to export, organized by category
SPRITE_IDS: dict[str, list[int]] = {
    # equipment slot background icons
    "equip": [156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 170],
    # prayer icons (enabled)
    "prayer": [
        115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
        127, 128, 129, 130, 131, 132, 133, 134,  # base prayers
        502, 503, 504, 505,  # hawk eye, mystic lore, eagle eye, mystic might
        945, 946, 947,  # chivalry, piety, preserve
        1420, 1421,  # rigour, augury
    ],
    # prayer icons (disabled/greyed)
    "prayer_off": [
        135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
        147, 148, 149, 150, 151, 152, 153, 154,  # base disabled
        506, 507, 508, 509,  # hawk/mystic disabled
        949, 950, 951,  # chivalry/piety/preserve disabled
        1424, 1425,  # rigour/augury disabled
    ],
    # tab icons (combat, stats, quests, inventory, equipment, prayer, magic)
    "tab": [168, 776, 779, 780, 898, 899, 900, 901],
    # ancient spell icons (enabled + disabled) — full book (smoke/shadow/blood/ice)
    "spell_ancient": [
        325, 326, 327, 328,  # ice rush/burst/blitz/barrage
        329, 330, 331, 332,  # smoke rush/burst/blitz/barrage
        333, 334, 335, 336,  # blood rush/burst/blitz/barrage
        337, 338, 339, 340,  # shadow rush/burst/blitz/barrage
        375, 376, 377, 378,  # ice disabled
        379, 380, 381, 382,  # smoke disabled
        383, 384, 385, 386,  # blood disabled
        387, 388, 389, 390,  # shadow disabled
    ],
    # lunar spell icons
    "spell_lunar": [557, 561, 564, 607, 611, 614],
    # combat interface
    "combat": [657],
    # interface chrome: side panel background, tab stones, equipment slot chrome
    "chrome": [
        1031,  # FIXED_MODE_SIDE_PANEL_BACKGROUND
        1032,  # FIXED_MODE_TABS_ROW_BOTTOM
        1036,  # FIXED_MODE_TABS_ROW_TOP
        1026, 1027, 1028, 1029, 1030,  # TAB_STONE_*_SELECTED corners + middle
        179,   # EQUIPMENT_SLOT_SELECTED
        952, 953,  # SLANTED_TAB, SLANTED_TAB_HOVERED
        1071, 1072,  # MINIMAP_ORB_FRAME, _HOVERED
        1017,  # CHATBOX background
        1018,  # CHATBOX_BUTTONS_BACKGROUND_STONES
        166,   # EQUIPMENT_SLOT_AMMUNITION (we had 156-165, was missing 166)
        171, 172, 173,  # iron rivets (square, vertical, horizontal)
    ],
    # click cross animations (4 yellow move + 4 red attack, 16x16 each)
    "click_cross": [515, 516, 517, 518, 519, 520, 521, 522],
    # overhead prayer headicons (multi-frame: 0=melee, 1=ranged, 2=magic, 3=retribution, 4=smite, 5=redemption)
    "headicons_prayer": [440],
    # hitsplat sprites (each is a single-frame sprite group)
    "hitmarks": [1358, 1359, 1360, 1361, 1362],
}

# human-readable names for specific sprite IDs
SPRITE_NAMES: dict[int, str] = {
    156: "slot_head", 157: "slot_cape", 158: "slot_neck", 159: "slot_weapon",
    160: "slot_ring", 161: "slot_body", 162: "slot_shield", 163: "slot_legs",
    164: "slot_hands", 165: "slot_feet", 170: "slot_tile",
    168: "tab_combat", 776: "tab_quests", 779: "tab_prayer",
    780: "tab_magic", 898: "tab_stats", 899: "tab_quests2",
    900: "tab_inventory", 901: "tab_equipment",
    # prayer/spell icons are loaded BY NUMERIC ID in osrs_gui.h, so keep them
    # as numeric filenames (no names here → fallback to str(sprite_id)).
    657: "special_attack",
    # interface chrome
    1031: "side_panel_bg",
    1032: "tabs_row_bottom",
    1036: "tabs_row_top",
    1026: "tab_stone_tl_sel",
    1027: "tab_stone_tr_sel",
    1028: "tab_stone_bl_sel",
    1029: "tab_stone_br_sel",
    1030: "tab_stone_mid_sel",
    179: "slot_selected",
    952: "slanted_tab",
    953: "slanted_tab_hover",
    1071: "orb_frame",
    1072: "orb_frame_hover",
    1017: "chatbox_bg",
    1018: "chatbox_stones",
    166: "slot_ammo",
    171: "rivets_square",
    172: "rivets_vertical",
    173: "rivets_horizontal",
    515: "cross_yellow_1", 516: "cross_yellow_2",
    517: "cross_yellow_3", 518: "cross_yellow_4",
    519: "cross_red_1", 520: "cross_red_2",
    521: "cross_red_3", 522: "cross_red_4",
    # overhead prayer headicons (group 440, 6 frames)
    440: "headicons_prayer",
    # hitsplats: 0=blue miss, 1=red damage, 2=green poison, 3=disease, 4=venom
    1358: "hitmarks_0", 1359: "hitmarks_1",
    1360: "hitmarks_2", 1361: "hitmarks_3", 1362: "hitmarks_4",
}


def main() -> None:
    """Export GUI sprites from modern OSRS cache."""
    parser = argparse.ArgumentParser(description="Export OSRS GUI sprites")
    parser.add_argument(
        "--cache", default=DEFAULT_MODERN_CACHE,
        help="Path to modern cache directory",
    )
    parser.add_argument(
        "--output", default="data/sprites/gui",
        help="Output directory for PNGs",
    )
    parser.add_argument(
        "--list-all", action="store_true",
        help="List all sprite group IDs in index 8 and exit",
    )
    args = parser.parse_args()

    reader = ModernCacheReader(args.cache)

    if args.list_all:
        manifest = reader.read_index_manifest(8)
        print(f"index 8 has {len(manifest.group_ids)} sprite groups")
        print(f"  range: {min(manifest.group_ids)} - {max(manifest.group_ids)}")
        return

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect all sprite IDs to export
    all_ids: set[int] = set()
    for ids in SPRITE_IDS.values():
        all_ids.update(ids)

    # check which IDs exist in the cache
    manifest = reader.read_index_manifest(8)
    available = set(manifest.group_ids)

    exported = 0
    failed = 0

    for sprite_id in sorted(all_ids):
        if sprite_id not in available:
            print(f"  sprite {sprite_id}: NOT in cache index 8")
            failed += 1
            continue

        data = reader.read_container(8, sprite_id)
        if data is None:
            print(f"  sprite {sprite_id}: failed to read container")
            failed += 1
            continue

        try:
            frames = decode_sprites(sprite_id, data)
        except (IndexError, struct.error, ValueError) as e:
            print(f"  sprite {sprite_id}: decode error: {e}", file=sys.stderr)
            failed += 1
            continue

        if not frames:
            print(f"  sprite {sprite_id}: no frames decoded")
            failed += 1
            continue

        for frame in frames:
            name = SPRITE_NAMES.get(sprite_id, str(sprite_id))
            if len(frames) > 1:
                filename = f"{name}_{frame.frame}.png"
            else:
                filename = f"{name}.png"
            path = out_dir / filename
            save_sprite_png(frame, path)
            exported += 1

        if len(frames) == 1:
            f = frames[0]
            print(f"  sprite {sprite_id} ({SPRITE_NAMES.get(sprite_id, '?')}): "
                  f"{f.width}x{f.height}")
        else:
            print(f"  sprite {sprite_id} ({SPRITE_NAMES.get(sprite_id, '?')}): "
                  f"{len(frames)} frames")

    print(f"\nexported {exported} sprite frames, {failed} failed")
    print(f"output: {out_dir}/")


if __name__ == "__main__":
    main()
