"""Export terrain mesh from modern OpenRS2 OSRS cache to a binary .terrain file.

For each region in the export area:
  1. Parse terrain data: heightmap, underlay IDs, overlay IDs, shapes, settings
  2. Decode floor definitions (underlays/overlays) for tile colors
  3. Compute per-vertex lighting (directional light + shadow map blur)
  4. Blend underlay colors in an 11x11 window (smooth terrain gradients)
  5. Output vertex-colored terrain mesh as binary .terrain file

The terrain binary is loaded by osrs_pvp_terrain.h into raylib meshes.

Usage:
    uv run python scripts/export_terrain.py \
        --modern-cache ../reference/osrs-cache-modern \
        --keys ../reference/osrs-cache-modern/keys.json \
        --regions "35,47 35,48" \
        --output data/zulrah.terrain
"""

import argparse
import io
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from modern_cache_reader import ModernCacheReader

# --- OSRS noise functions (from MapRegion.java) ---

JAGEX_CIRCULAR_ANGLE = 2048
JAGEX_RADIAN = 2.0 * math.pi / JAGEX_CIRCULAR_ANGLE
COS_TABLE = [int(65536.0 * math.cos(i * JAGEX_RADIAN)) for i in range(JAGEX_CIRCULAR_ANGLE)]


def noise(x: int, y: int) -> int:
    """OSRS procedural noise (MapRegion.noise)."""
    n = x + y * 57
    n ^= (n << 13) & 0xFFFFFFFF
    n &= 0xFFFFFFFF
    val = (n * (n * n * 15731 + 789221) + 1376312589) & 0x7FFFFFFF
    return (val >> 19) & 0xFF


def smoothed_noise(x: int, y: int) -> int:
    """Smoothed noise with corner/side/center weighting."""
    corners = noise(x - 1, y - 1) + noise(x + 1, y - 1) + noise(x - 1, y + 1) + noise(x + 1, y + 1)
    sides = noise(x - 1, y) + noise(x + 1, y) + noise(x, y - 1) + noise(x, y + 1)
    center = noise(x, y)
    return center // 4 + sides // 8 + corners // 16


def interpolate(a: int, b: int, x: int, y: int) -> int:
    """Cosine interpolation using Jagex COS table."""
    f = (0x10000 - COS_TABLE[(1024 * x // y) % JAGEX_CIRCULAR_ANGLE]) >> 1
    return (f * b >> 16) + (a * (0x10000 - f) >> 16)


def interpolate_noise(x: int, y: int, frequency: int) -> int:
    """Multi-octave interpolated noise."""
    int_x = x // frequency
    frac_x = x % frequency
    int_y = y // frequency
    frac_y = y % frequency
    v1 = smoothed_noise(int_x, int_y)
    v2 = smoothed_noise(int_x + 1, int_y)
    v3 = smoothed_noise(int_x, int_y + 1)
    v4 = smoothed_noise(int_x + 1, int_y + 1)
    i1 = interpolate(v1, v2, frac_x, frequency)
    i2 = interpolate(v3, v4, frac_x, frequency)
    return interpolate(i1, i2, frac_y, frequency)


def calculate_height(x: int, y: int) -> int:
    """Procedural height for tiles without explicit height (MapRegion.calculate)."""
    n = (
        interpolate_noise(x + 45365, y + 91923, 4)
        - 128
        + ((interpolate_noise(10294 + x, y + 37821, 2) - 128) >> 1)
        + ((interpolate_noise(x, y, 1) - 128) >> 2)
    )
    n = 35 + int(n * 0.3)
    if n < 10:
        n = 10
    elif n > 60:
        n = 60
    return n


# --- floor definition decoder ---


@dataclass
class FloorDef:
    """Floor definition (underlay or overlay)."""

    floor_id: int = 0
    rgb: int = 0
    texture: int = -1
    hide_underlay: bool = True
    secondary_rgb: int = -1
    # computed HSL fields
    hue: int = 0
    saturation: int = 0
    lightness: int = 0
    secondary_hue: int = 0
    secondary_saturation: int = 0
    secondary_lightness: int = 0
    # blend fields (for underlays)
    blend_hue: int = 0
    blend_hue_multiplier: int = 0
    luminance: int = 0
    hsl16: int = 0


def _rgb_to_hsl(rgb: int, flo: FloorDef) -> None:
    """Convert RGB to HSL fields on a FloorDef (MapRegion.rgbToHsl / FloorDefinition.setHsl)."""
    r = ((rgb >> 16) & 0xFF) / 256.0
    g = ((rgb >> 8) & 0xFF) / 256.0
    b = (rgb & 0xFF) / 256.0

    mn = min(r, g, b)
    mx = max(r, g, b)

    h = 0.0
    s = 0.0
    lightness = (mn + mx) / 2.0

    if mn != mx:
        if lightness < 0.5:
            s = (mx - mn) / (mx + mn)
        else:
            s = (mx - mn) / (2.0 - mx - mn)

        if r == mx:
            h = (g - b) / (mx - mn)
        elif g == mx:
            h = 2.0 + (b - r) / (mx - mn)
        elif b == mx:
            h = 4.0 + (r - g) / (mx - mn)

    h /= 6.0

    flo.hue = max(0, min(255, int(h * 256.0)))
    flo.saturation = max(0, min(255, int(s * 256.0)))
    flo.lightness = max(0, min(255, int(lightness * 256.0)))
    flo.luminance = flo.lightness

    if lightness > 0.5:
        flo.blend_hue_multiplier = int((1.0 - lightness) * s * 512.0)
    else:
        flo.blend_hue_multiplier = int(lightness * s * 512.0)

    if flo.blend_hue_multiplier < 1:
        flo.blend_hue_multiplier = 1

    flo.blend_hue = int(h * flo.blend_hue_multiplier)
    flo.hsl16 = _hsl24to16(flo.hue, flo.saturation, flo.luminance)


def _hsl24to16(h: int, s: int, lightness: int) -> int:
    """Convert 24-bit HSL to 16-bit packed HSL (FloorDefinition.hsl24to16)."""
    if lightness > 179:
        s //= 2
    if lightness > 192:
        s //= 2
    if lightness > 217:
        s //= 2
    if lightness > 243:
        s //= 2
    return ((h // 4) << 10) + ((s // 32) << 7) + lightness // 2



# --- modern cache floor definition + texture color decoders ---

# modern cache config index groups for floor definitions
MODERN_UNDERLAY_GROUP = 1
MODERN_OVERLAY_GROUP = 4


def _decode_underlay_entry(fid: int, data: bytes) -> FloorDef:
    """Decode a single underlay floor definition from modern cache opcode stream.

    Underlay opcodes: 1=rgb(3 bytes), 0=terminator.
    """
    flo = FloorDef(floor_id=fid)
    buf = io.BytesIO(data)
    while True:
        op = buf.read(1)
        if not op or op[0] == 0:
            break
        if op[0] == 1:
            flo.rgb = (buf.read(1)[0] << 16) + (buf.read(1)[0] << 8) + buf.read(1)[0]
    if flo.secondary_rgb != -1:
        _rgb_to_hsl(flo.secondary_rgb, flo)
    _rgb_to_hsl(flo.rgb, flo)
    return flo


def _decode_overlay_entry(fid: int, data: bytes) -> FloorDef:
    """Decode a single overlay floor definition from modern cache opcode stream.

    Overlay opcodes (matching RuneLite OverlayDefinition):
      1=rgb(3 bytes), 2=texture(1 byte), 5=hide_underlay=false (no data),
      7=secondary_rgb(3 bytes), 0=terminator.
    """
    flo = FloorDef(floor_id=fid)
    buf = io.BytesIO(data)
    while True:
        op = buf.read(1)
        if not op or op[0] == 0:
            break
        opcode = op[0]
        if opcode == 1:
            flo.rgb = (buf.read(1)[0] << 16) + (buf.read(1)[0] << 8) + buf.read(1)[0]
        elif opcode == 2:
            flo.texture = buf.read(1)[0]
        elif opcode == 5:
            flo.hide_underlay = False
        elif opcode == 7:
            flo.secondary_rgb = (buf.read(1)[0] << 16) + (buf.read(1)[0] << 8) + buf.read(1)[0]
    # post-decode HSL
    if flo.secondary_rgb != -1:
        _rgb_to_hsl(flo.secondary_rgb, flo)
        flo.secondary_hue = flo.hue
        flo.secondary_saturation = flo.saturation
        flo.secondary_lightness = flo.lightness
    _rgb_to_hsl(flo.rgb, flo)
    return flo


def decode_floor_definitions_modern(
    reader: ModernCacheReader,
) -> tuple[dict[int, FloorDef], dict[int, FloorDef]]:
    """Decode underlay and overlay floor definitions from modern cache.

    Modern cache stores floor defs in config index 2:
      group 1 = underlays, group 4 = overlays.
    Each entry is a separate file within the group.
    """
    underlays: dict[int, FloorDef] = {}
    overlays: dict[int, FloorDef] = {}

    # underlays: config index 2, group 1
    underlay_files = reader.read_group(2, MODERN_UNDERLAY_GROUP)
    for fid, data in underlay_files.items():
        underlays[fid] = _decode_underlay_entry(fid, data)

    # overlays: config index 2, group 4
    overlay_files = reader.read_group(2, MODERN_OVERLAY_GROUP)
    for fid, data in overlay_files.items():
        overlays[fid] = _decode_overlay_entry(fid, data)

    return underlays, overlays


def load_texture_average_colors_modern(reader: ModernCacheReader) -> dict[int, int]:
    """Load per-texture average color (HSL16) from modern cache.

    Modern OSRS (rev233+) texture format in index 9, group 0:
      u16 fileId (sprite group in index 8)
      u16 missingColor (HSL16 fallback — used as the texture's average color)
      u8  field1778
      u8  animationDirection
      u8  animationSpeed

    The missingColor field is the precomputed average color of the texture
    sprite, stored as 16-bit packed HSL. This is what the OSRS client uses
    for terrain tile coloring.
    """
    tex_files = reader.read_group(9, 0)
    result: dict[int, int] = {}
    for tex_id, data in tex_files.items():
        if len(data) < 4:
            continue
        missing_color = struct.unpack(">H", data[2:4])[0]
        result[tex_id] = missing_color
    return result


def djb2(name: str) -> int:
    """Compute djb2 hash for modern cache group name lookup."""
    h = 0
    for c in name.lower():
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    if h >= 0x80000000:
        h -= 0x100000000
    return h


def find_map_groups(
    reader: ModernCacheReader,
) -> dict[int, tuple[int | None, int | None]]:
    """Build mapping of mapsquare -> (terrain_group_id, object_group_id).

    Scans the index 5 manifest for groups whose djb2 name hashes match
    m{rx}_{ry} (terrain) patterns.
    """
    manifest = reader.read_index_manifest(5)

    hash_to_gid: dict[int, int] = {}
    for gid in manifest.group_ids:
        nh = manifest.group_name_hashes.get(gid)
        if nh is not None:
            hash_to_gid[nh] = gid

    result: dict[int, tuple[int | None, int | None]] = {}
    for rx in range(256):
        for ry in range(256):
            terrain_hash = djb2(f"m{rx}_{ry}")
            obj_hash = djb2(f"l{rx}_{ry}")

            terrain_gid = hash_to_gid.get(terrain_hash)
            obj_gid = hash_to_gid.get(obj_hash)

            if terrain_gid is not None or obj_gid is not None:
                mapsquare = (rx << 8) | ry
                result[mapsquare] = (terrain_gid, obj_gid)

    return result


# --- terrain parser ---


@dataclass
class RegionTerrain:
    """Parsed terrain data for one 64x64 region."""

    region_x: int = 0
    region_y: int = 0
    # [4][64+1][64+1] - tile corner heights (need +1 for neighbor corners)
    heights: list = field(default_factory=list)
    # [4][64][64]
    underlay_ids: list = field(default_factory=list)
    overlay_ids: list = field(default_factory=list)
    shapes: list = field(default_factory=list)
    rotations: list = field(default_factory=list)
    settings: list = field(default_factory=list)


def parse_terrain_full(
    data: bytes, region_chunk_x: int, region_chunk_y: int
) -> RegionTerrain:
    """Parse terrain binary to extract heights, underlays, overlays, shapes, settings."""
    rt = RegionTerrain()
    rt.heights = [[[0 for _ in range(65)] for _ in range(65)] for _ in range(4)]
    rt.underlay_ids = [[[0 for _ in range(64)] for _ in range(64)] for _ in range(4)]
    rt.overlay_ids = [[[0 for _ in range(64)] for _ in range(64)] for _ in range(4)]
    rt.shapes = [[[0 for _ in range(64)] for _ in range(64)] for _ in range(4)]
    rt.rotations = [[[0 for _ in range(64)] for _ in range(64)] for _ in range(4)]
    rt.settings = [[[0 for _ in range(64)] for _ in range(64)] for _ in range(4)]

    buf = io.BytesIO(data)

    for plane in range(4):
        for x in range(64):
            for y in range(64):
                rt.settings[plane][x][y] = 0
                while True:
                    raw = buf.read(2)
                    if len(raw) < 2:
                        break
                    attr = struct.unpack(">H", raw)[0]

                    if attr == 0:
                        # procedural height
                        if plane == 0:
                            rt.heights[0][x][y] = -calculate_height(
                                0xE3B7B + x + region_chunk_x,
                                0x87CCE + y + region_chunk_y,
                            ) * 8
                        else:
                            rt.heights[plane][x][y] = rt.heights[plane - 1][x][y] - 240
                        break
                    elif attr == 1:
                        height = buf.read(1)[0]
                        if height == 1:
                            height = 0
                        if plane == 0:
                            rt.heights[0][x][y] = -height * 8
                        else:
                            rt.heights[plane][x][y] = rt.heights[plane - 1][x][y] - height * 8
                        break
                    elif attr <= 49:
                        rt.overlay_ids[plane][x][y] = struct.unpack(">h", buf.read(2))[0]
                        rt.shapes[plane][x][y] = (attr - 2) // 4
                        rt.rotations[plane][x][y] = (attr - 2) & 3
                    elif attr <= 81:
                        rt.settings[plane][x][y] = attr - 49
                    else:
                        rt.underlay_ids[plane][x][y] = attr - 81

    return rt


# --- lighting + color blending (mirrors method171) ---


def hsl_encode(hue: int, sat: int, light: int) -> int:
    """Encode HSL to 16-bit (method177 / hsl24to16)."""
    if light > 179:
        sat //= 2
    if light > 192:
        sat //= 2
    if light > 217:
        sat //= 2
    if light > 243:
        sat //= 2
    return ((hue // 4) << 10) + ((sat // 32) << 7) + light // 2


def apply_light(hsl16: int, light: int) -> int:
    """Apply lighting intensity to an HSL16 color (method187)."""
    if hsl16 == -1:
        return 0xBC614E
    packed_lightness = (light * (hsl16 & 0x7F)) // 128
    if packed_lightness < 2:
        packed_lightness = 2
    elif packed_lightness > 126:
        packed_lightness = 126
    return (hsl16 & 0xFF80) + packed_lightness


def hsl16_to_rgb(hsl16: int) -> tuple[int, int, int]:
    """Convert 16-bit packed HSL to RGB.

    16-bit HSL: (hue << 10) | (sat << 7) | lightness
    hue: 6 bits (0-63), sat: 3 bits (0-7), lightness: 7 bits (0-127)
    Same as the Rasterizer3D palette lookup.
    """
    if hsl16 == 0xBC614E or hsl16 < 0:
        return (0, 0, 0)

    h_raw = (hsl16 >> 10) & 0x3F
    s_raw = (hsl16 >> 7) & 0x7
    l_raw = hsl16 & 0x7F

    hue_f = h_raw / 64.0 + 0.0078125
    sat_f = s_raw / 8.0 + 0.0625
    light_f = l_raw / 128.0

    r, g, b = light_f, light_f, light_f

    if sat_f != 0.0:
        if light_f < 0.5:
            q = light_f * (1.0 + sat_f)
        else:
            q = light_f + sat_f - light_f * sat_f
        p = 2.0 * light_f - q

        h_r = hue_f + 1.0 / 3.0
        if h_r > 1.0:
            h_r -= 1.0
        h_g = hue_f
        h_b = hue_f - 1.0 / 3.0
        if h_b < 0.0:
            h_b += 1.0

        r = _hue_channel(p, q, h_r)
        g = _hue_channel(p, q, h_g)
        b = _hue_channel(p, q, h_b)

    ri = max(0, min(255, int(r * 256.0)))
    gi = max(0, min(255, int(g * 256.0)))
    bi = max(0, min(255, int(b * 256.0)))
    return (ri, gi, bi)


def _hue_channel(p: float, q: float, t: float) -> float:
    if 6.0 * t < 1.0:
        return p + (q - p) * 6.0 * t
    if 2.0 * t < 1.0:
        return q
    if 3.0 * t < 2.0:
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0
    return p


# --- terrain mesh builder ---


@dataclass
class TerrainVertex:
    """A single terrain vertex with position and color."""

    x: float = 0.0
    y: float = 0.0  # height (OSRS Y = up)
    z: float = 0.0
    r: int = 0
    g: int = 0
    b: int = 0


def stitch_region_edges(
    regions: dict[tuple[int, int], RegionTerrain],
) -> None:
    """Stitch height values at region boundaries.

    Each region's height array is [65][65] but only indices 0..63 are computed.
    Index 64 (the far edge) defaults to 0, causing visible seams. Fix by copying
    heights from the neighboring region's index 0 to the current region's index 64.

    Also computes edge heights for regions without a neighbor by extrapolating
    from the nearest interior values.
    """
    for (rx, ry), rt in regions.items():
        for plane in range(4):
            # stitch X=64 edge from neighbor (rx+1, ry)
            neighbor_x = regions.get((rx + 1, ry))
            for y in range(65):
                if neighbor_x and y < 65:
                    # use neighbor's x=0 column
                    ny = min(y, 64)
                    rt.heights[plane][64][y] = neighbor_x.heights[plane][0][ny]
                elif y <= 63:
                    # no neighbor: extrapolate from x=63
                    rt.heights[plane][64][y] = rt.heights[plane][63][y]

            # stitch Y=64 edge from neighbor (rx, ry+1)
            neighbor_y = regions.get((rx, ry + 1))
            for x in range(65):
                if neighbor_y and x < 65:
                    nx = min(x, 64)
                    rt.heights[plane][x][64] = neighbor_y.heights[plane][nx][0]
                elif x <= 63:
                    rt.heights[plane][x][64] = rt.heights[plane][x][63]

            # corner (64, 64): prefer diagonal neighbor
            diag = regions.get((rx + 1, ry + 1))
            if diag:
                rt.heights[plane][64][64] = diag.heights[plane][0][0]
            elif neighbor_x:
                rt.heights[plane][64][64] = neighbor_x.heights[plane][0][min(64, 63)]
            elif neighbor_y:
                rt.heights[plane][64][64] = neighbor_y.heights[plane][min(64, 63)][0]
            else:
                rt.heights[plane][64][64] = rt.heights[plane][63][63]


def build_terrain_mesh(
    regions: dict[tuple[int, int], RegionTerrain],
    underlays: dict[int, FloorDef],
    overlays: dict[int, FloorDef],
    target_plane: int = 0,
    tex_colors: dict[int, int] | None = None,
    brightness: float = 1.0,
) -> tuple[list[float], list[int]]:
    """Build a vertex-colored terrain mesh from parsed regions.

    Returns (vertices[N*3], colors[N*4]) as flat lists.
    Each tile = 2 triangles = 6 vertices.
    """
    verts: list[float] = []
    colors: list[int] = []

    # we work region by region. for each tile, emit 2 triangles.
    # world coords: base_x = region_x * 64, base_y = region_y * 64
    # OSRS tile = 128 world units. we'll use 1 tile = 1 unit for simplicity.

    for (rx, ry), rt in sorted(regions.items()):
        base_wx = rx * 64
        base_wy = ry * 64

        z = target_plane

        # compute lighting per vertex (mirrors method171 lighting pass)
        # intensity[65][65] for tile corners
        base_intensity = int(96 * brightness)
        intensity = [[base_intensity] * 65 for _ in range(65)]

        light_x, light_y, light_z = -50, -10, -50
        light_len = int(math.sqrt(light_x * light_x + light_y * light_y + light_z * light_z))
        distribution = (768 * light_len) >> 8

        for ty in range(65):
            for tx in range(65):
                dx = rt.heights[z][min(tx + 1, 64)][ty] - rt.heights[z][max(tx - 1, 0)][ty]
                dy = rt.heights[z][tx][min(ty + 1, 64)] - rt.heights[z][tx][max(ty - 1, 0)]
                length = int(math.sqrt(dx * dx + 256 * 256 + dy * dy))
                if length == 0:
                    length = 1
                nx = (dx << 8) // length
                ny = (256 << 8) // length
                nz = (dy << 8) // length
                intensity[tx][ty] = base_intensity + (light_x * nx + light_y * ny + light_z * nz) // distribution

        # compute blended underlay colors using 11x11 window
        # for simplicity, compute per-tile center color
        for ty in range(64):
            for tx in range(64):
                uid = rt.underlay_ids[z][tx][ty]
                oid = rt.overlay_ids[z][tx][ty] & 0xFFFF

                # default: dark ground color
                tile_rgb = (40, 50, 30)

                if uid > 0 or oid > 0:
                    # get 4 corner heights
                    h_sw = rt.heights[z][tx][ty]
                    h_se = rt.heights[z][min(tx + 1, 64)][ty]
                    h_ne = rt.heights[z][min(tx + 1, 64)][min(ty + 1, 64)]
                    h_nw = rt.heights[z][tx][min(ty + 1, 64)]

                    # underlay color: blend in 11x11 window
                    if uid > 0:
                        blend_h, blend_s, blend_l, blend_m, blend_c = 0, 0, 0, 0, 0
                        for bx in range(max(0, tx - 5), min(64, tx + 6)):
                            for by in range(max(0, ty - 5), min(64, ty + 6)):
                                uid2 = rt.underlay_ids[z][bx][by]
                                if uid2 > 0:
                                    flo = underlays.get(uid2 - 1)
                                    if flo:
                                        blend_h += flo.blend_hue
                                        blend_s += flo.saturation
                                        blend_l += flo.luminance
                                        blend_m += flo.blend_hue_multiplier
                                        blend_c += 1

                        if blend_c > 0 and blend_m > 0:
                            avg_hue = (blend_h * 256) // blend_m
                            avg_sat = blend_s // blend_c
                            avg_lum = blend_l // blend_c
                            underlay_hsl16 = hsl_encode(avg_hue, avg_sat, avg_lum)
                        else:
                            underlay_hsl16 = -1
                    else:
                        underlay_hsl16 = -1

                    # corner lighting
                    l_sw = intensity[tx][ty]
                    l_se = intensity[min(tx + 1, 64)][ty]
                    l_ne = intensity[min(tx + 1, 64)][min(ty + 1, 64)]
                    l_nw = intensity[tx][min(ty + 1, 64)]
                    avg_light = (l_sw + l_se + l_ne + l_nw) // 4

                    if oid > 0:
                        # overlay tile
                        oflo = overlays.get(oid - 1)
                        if oflo and oflo.rgb == 0xFF00FF:
                            continue  # void tile — skip geometry so objects show through
                        elif oflo and oflo.texture >= 0 and tex_colors:
                            # textured overlay: use texture average color (water, dirt, stone)
                            tex_hsl = tex_colors.get(oflo.texture)
                            if tex_hsl is not None:
                                lit = apply_light(tex_hsl, avg_light)
                                tile_rgb = hsl16_to_rgb(lit)
                            elif underlay_hsl16 >= 0:
                                lit = apply_light(underlay_hsl16, avg_light)
                                tile_rgb = hsl16_to_rgb(lit)
                        elif oflo and oflo.texture < 0:
                            overlay_hsl16 = hsl_encode(oflo.hue, oflo.saturation, oflo.lightness)
                            lit = apply_light(overlay_hsl16, avg_light)
                            tile_rgb = hsl16_to_rgb(lit)
                        else:
                            # no overlay def — use underlay
                            if underlay_hsl16 >= 0:
                                lit = apply_light(underlay_hsl16, avg_light)
                                tile_rgb = hsl16_to_rgb(lit)
                    else:
                        # pure underlay tile
                        if underlay_hsl16 >= 0:
                            lit = apply_light(underlay_hsl16, avg_light)
                            tile_rgb = hsl16_to_rgb(lit)

                    # emit 2 triangles (SW, SE, NE) and (SW, NE, NW)
                    # world position: tile (tx, ty) in region (rx, ry)
                    wx = float(base_wx + tx)
                    wz = float(base_wy + ty)

                    # OSRS heights are negative (higher = more negative)
                    # negate so mountains go up, scale by 1/128 to get tile units
                    scale_h = -1.0 / 128.0

                    y_sw = float(h_sw) * scale_h
                    y_se = float(h_se) * scale_h
                    y_ne = float(h_ne) * scale_h
                    y_nw = float(h_nw) * scale_h

                    r, g, b = tile_rgb

                    # negate Z so OSRS north (+Y) maps to -Z (correct for right-handed coords)
                    nz = -wz

                    # triangle 1: SW, SE, NE (with negated Z, gives upward normals)
                    verts.extend([wx, y_sw, nz, wx + 1, y_se, nz, wx + 1, y_ne, nz - 1])
                    colors.extend([r, g, b, 255] * 3)

                    # triangle 2: SW, NE, NW
                    verts.extend([wx, y_sw, nz, wx + 1, y_ne, nz - 1, wx, y_nw, nz - 1])
                    colors.extend([r, g, b, 255] * 3)

    return verts, colors


# --- binary output ---

TERR_MAGIC = 0x54455252  # "TERR"


def build_heightmap(
    regions: dict[tuple[int, int], RegionTerrain],
    target_plane: int = 0,
) -> tuple[int, int, int, int, list[float]]:
    """Build a flat heightmap grid from parsed regions.

    Returns (min_x, min_y, width, height, heights[width*height]).
    Heights are in terrain units (OSRS height / 128.0).
    """
    min_rx = min(rx for rx, _ in regions.keys())
    max_rx = max(rx for rx, _ in regions.keys())
    min_ry = min(ry for _, ry in regions.keys())
    max_ry = max(ry for _, ry in regions.keys())

    min_wx = min_rx * 64
    min_wy = min_ry * 64
    width = (max_rx - min_rx + 1) * 64
    height = (max_ry - min_ry + 1) * 64

    # default to 0 (flat) for uncovered tiles
    hmap = [0.0] * (width * height)
    scale_h = -1.0 / 128.0

    for (rx, ry), rt in regions.items():
        base_x = rx * 64 - min_wx
        base_y = ry * 64 - min_wy
        for tx in range(64):
            for ty in range(64):
                idx = (base_x + tx) + (base_y + ty) * width
                hmap[idx] = float(rt.heights[target_plane][tx][ty]) * scale_h

    return min_wx, min_wy, width, height, hmap


def write_terrain_binary(
    output_path: Path,
    verts: list[float],
    colors: list[int],
    regions: dict[tuple[int, int], RegionTerrain],
    heightmap: tuple[int, int, int, int, list[float]],
) -> None:
    """Write terrain mesh to binary .terrain file.

    Format:
      magic: uint32 "TERR"
      vertex_count: uint32
      region_count: uint32
      min_world_x: int32  (for coordinate offset)
      min_world_y: int32
      vertices: float32[vertex_count * 3]  (x, y, z)
      colors: uint8[vertex_count * 4]  (r, g, b, a)
      heightmap_min_x: int32
      heightmap_min_y: int32
      heightmap_width: uint32
      heightmap_height: uint32
      heightmap: float32[width * height]
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vert_count = len(verts) // 3

    # find min world coords for centering
    min_wx = min(rx * 64 for rx, _ in regions.keys())
    min_wy = min(ry * 64 for _, ry in regions.keys())

    hm_min_x, hm_min_y, hm_w, hm_h, hm_data = heightmap

    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", TERR_MAGIC))
        f.write(struct.pack("<I", vert_count))
        f.write(struct.pack("<I", len(regions)))
        f.write(struct.pack("<i", min_wx))
        f.write(struct.pack("<i", min_wy))

        # vertices
        for v in verts:
            f.write(struct.pack("<f", v))

        # colors
        for c in colors:
            f.write(struct.pack("B", c))

        # heightmap
        f.write(struct.pack("<i", hm_min_x))
        f.write(struct.pack("<i", hm_min_y))
        f.write(struct.pack("<I", hm_w))
        f.write(struct.pack("<I", hm_h))
        for h in hm_data:
            f.write(struct.pack("<f", h))


# --- main ---


def _main_modern(args: argparse.Namespace) -> None:
    """Export terrain from modern OpenRS2 cache."""
    if not args.modern_cache.exists():
        sys.exit(f"modern cache directory not found: {args.modern_cache}")

    print(f"reading modern cache from {args.modern_cache}")
    reader = ModernCacheReader(args.modern_cache)

    print("loading floor definitions from modern cache...")
    underlays, overlays_defs = decode_floor_definitions_modern(reader)
    print(f"  {len(underlays)} underlays, {len(overlays_defs)} overlays")

    print("loading texture average colors from modern cache...")
    tex_colors = load_texture_average_colors_modern(reader)
    print(f"  {len(tex_colors)} texture colors")

    print("scanning index 5 for map groups...")
    map_groups = find_map_groups(reader)
    print(f"  {len(map_groups)} regions found in index 5")

    # determine target regions
    target_coords: set[tuple[int, int]] = set()
    if args.regions:
        for coord in args.regions.split():
            parts = coord.split(",")
            target_coords.add((int(parts[0]), int(parts[1])))
    else:
        sys.exit("--regions is required when using --modern-cache")

    print(f"  exporting {len(target_coords)} specified regions")

    parsed: dict[tuple[int, int], RegionTerrain] = {}
    errors = 0

    for rx, ry in sorted(target_coords):
        ms = (rx << 8) | ry
        if ms not in map_groups:
            print(f"  region ({rx},{ry}): not found in index 5")
            errors += 1
            continue

        terrain_gid, _ = map_groups[ms]
        if terrain_gid is None:
            print(f"  region ({rx},{ry}): no terrain group")
            errors += 1
            continue

        terrain_data = reader.read_container(5, terrain_gid)
        if terrain_data is None:
            print(f"  region ({rx},{ry}): failed to read terrain")
            errors += 1
            continue

        region_chunk_x = rx * 64
        region_chunk_y = ry * 64

        rt = parse_terrain_full(terrain_data, region_chunk_x, region_chunk_y)
        rt.region_x = rx
        rt.region_y = ry
        parsed[(rx, ry)] = rt

    print(f"  parsed {len(parsed)} regions, {errors} errors")
    _build_and_write(args, parsed, underlays, overlays_defs, tex_colors)


def _build_and_write(
    args: argparse.Namespace,
    parsed: dict[tuple[int, int], RegionTerrain],
    underlays: dict[int, FloorDef],
    overlays_defs: dict[int, FloorDef],
    tex_colors: dict[int, int],
) -> None:
    """Build terrain mesh and heightmap, then write to binary output."""
    if not parsed:
        sys.exit("no regions parsed successfully")

    print("stitching region edges...")
    stitch_region_edges(parsed)

    print("building terrain mesh...")
    brightness = getattr(args, 'brightness', 1.0)
    verts, colors = build_terrain_mesh(parsed, underlays, overlays_defs, tex_colors=tex_colors, brightness=brightness)
    vert_count = len(verts) // 3
    tri_count = vert_count // 3
    print(f"  {vert_count} vertices, {tri_count} triangles")

    print("building heightmap...")
    heightmap = build_heightmap(parsed)
    hm_min_x, hm_min_y, hm_w, hm_h, _ = heightmap
    print(f"  {hm_w}x{hm_h} tiles, origin ({hm_min_x}, {hm_min_y})")

    write_terrain_binary(args.output, verts, colors, parsed, heightmap)
    file_size = args.output.stat().st_size
    print(f"\nwrote {file_size:,} bytes to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="export OSRS terrain mesh from modern OpenRS2 cache")
    parser.add_argument(
        "--modern-cache",
        type=Path,
        required=True,
        help="path to modern OpenRS2 cache directory",
    )
    parser.add_argument(
        "--keys",
        type=Path,
        default=None,
        help="path to XTEA keys JSON (for modern cache object data)",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default=None,
        help='space-separated region coordinates as rx,ry pairs (e.g. "35,47 35,48")',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/wilderness.terrain"),
        help="output .terrain binary file",
    )
    parser.add_argument(
        "--brightness",
        type=float,
        default=1.0,
        help="brightness multiplier for terrain lighting (e.g. 1.8 for caves, default: 1.0)",
    )
    args = parser.parse_args()

    _main_modern(args)


if __name__ == "__main__":
    main()
