'''
This script is used to generate scaled and combined sprite sheets for nmmo3
You will need the to put the following folders into the same directory. They
can be purchased from ManaSeed on itch.io

20.04c - Summer Forest 4.2
20.05c - Spring Forest 4.1
20.06a - Autumn Forest 4.1
20.07a - Winter Forest 4.0a
20.01a - Character Base 2.5c
20.01c - Peasant Farmer Pants & Hat 2.1 (comp. v01)
20.01c - Peasant Farmer Pants & Hat 2.2 (optional combat animations)
20.08b - Bow Combat 3.2
21.07b - Sword & Shield Combat 2.3
21.10a - Forester Pointed Hat & Tunic 2.1a (comp. v01)
21.10a - Forester Pointed Hat & Tunic 2.2 (optional, combat animations)
'''

from itertools import product
from PIL import Image
import pyray as ray
import numpy as np
import random
import sys
import os
import cv2


SHEET_SIZE = 2048
N_GENERATE = 10

ELEMENTS = (
    ('neutral', 1, ray.Color(255, 255, 255, 255)),
    ('fire', 5, ray.Color(255, 128, 128, 255)),
    ('water', 9, ray.Color(128, 128, 255, 255)),
    ('earth', 11, ray.Color(128, 255, 128, 255)),
    ('air', 3, ray.Color(255, 255, 128, 255)),
)

BASE = list(range(8))
HAIR = list(range(14))
CLOTHES = list(range(1, 6))
SWORD = list(range(1, 6))
BOW = list(range(1, 6))
QUIVER = list(range(1, 9))

# Hair colors, indices into files
'''
HAIR = {
    ELEM_NEUTRAL: 1,
    ELEM_FIRE: 5,
    ELEM_WATER: 9,
    ELEM_EARTH: 11,
    ELEM_AIR: 3
}
'''


# Character base
character = 'char_a_p1_0bas_humn_v{i:02}.png'
demon = 'char_a_p1_0bas_demn_v{i:02}.png'
goblin = 'char_a_p1_0bas_gbln_v{i:02}.png'
hair_dap = 'char_a_p1_4har_dap1_v{i:02}.png'
hair_bob = 'char_a_p1_4har_bob1_v{i:02}.png'

# Combat animations
sword_character = 'char_a_pONE3_0bas_humn_v{i:02}.png'
sword_weapon = 'char_a_pONE3_6tla_sw01_v{i:02}.png'
sword_hair_bob = 'char_a_pONE3_4har_bob1_v{i:02}.png'
sword_hair_dap = 'char_a_pONE3_4har_dap1_v{i:02}.png'
bow_character = 'char_a_pBOW3_0bas_humn_v{i:02}.png'
bow_hair_dap = 'char_a_pBOW3_4har_dap1_v{i:02}.png'
bow_hair_bob = 'char_a_pBOW3_4har_bob1_v{i:02}.png'
bow_weapon = 'char_a_pBOW3_6tla_bo01_v{i:02}.png'
bow_quiver = 'char_a_pBOW3_7tlb_qv01_v{i:02}.png'
arrow = 'aro_comn_v{i:02}.png'

# Peasant character alternative
peasant_clothes = 'char_a_p1_1out_pfpn_v{i:02}.png'
sword_peasant_clothes = 'char_a_pONE3_1out_pfpn_v{i:02}.png'
bow_peasant_clothes = 'char_a_pBOW3_1out_pfpn_v{i:02}.png'

# Forester character alternative
forester_hat = 'char_a_p1_5hat_pnty_v{i:02}.png'
forester_clothes = 'char_a_p1_1out_fstr_v{i:02}.png'
sword_forester_hat = 'char_a_pONE3_5hat_pnty_v{i:02}.png'
sword_forester_clothes = 'char_a_pONE3_1out_fstr_v{i:02}.png'
bow_forester_hat = 'char_a_pBOW3_5hat_pnty_v{i:02}.png'
bow_forester_clothes = 'char_a_pBOW3_1out_fstr_v{i:02}.png'

sword_mask = np.array((
    (1, 1, 1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 1, 1, 1),
    (1, 0, 1, 1, 1, 1, 1, 1),
    (1, 0, 1, 1, 1, 1, 1, 1),
    (0, 0, 1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 1, 1, 1),
    (0, 0, 1, 1, 0, 0, 0, 0),
    (0, 0, 1, 1, 0, 0, 0, 0),
))

bow_mask = np.array((
    (0, 0, 0, 0, 0, 0, 0, 0),
    (1, 1, 1, 1, 1, 1, 1, 1),
    (1, 0, 0, 0, 0, 0, 0, 0),
    (1, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (1, 1, 1, 1, 1, 1, 1, 1),
    (1, 0, 0, 0, 0, 0, 0, 0),
    (1, 0, 0, 0, 0, 0, 0, 0),
))

quiver_mask = np.array((
    (1, 1, 1, 1, 1, 1, 1, 1),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (1, 1, 1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 1, 1, 1),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (1, 1, 1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 1, 1, 1),
))

def draw_tex(path, f_name, i, x, y, tint=None):
    if tint is None:
        tint = ray.WHITE

    path = os.path.join(path, f_name).format(i=i)
    texture = ray.load_texture(path)
    source_rect = ray.Rectangle(0, 0, texture.width, -texture.height)
    dest_rect = ray.Rectangle(x, y, texture.width, texture.height)
    ray.draw_texture_pro(texture, source_rect, dest_rect, (0, 0), 0, tint)

def draw_masked_tex(path, f_name, i, x, y, mask, tint=None):
    if tint is None:
        tint = ray.WHITE

    path = os.path.join(path, f_name).format(i=i)
    texture = ray.load_texture(path)
    Y, X = mask.shape
    for r, row in enumerate(mask):
        for c, m in enumerate(row):
            if m == 0:
                continue

            src_x = c * 128
            src_y = r * 128
            source_rect = ray.Rectangle(src_x, src_y, 128, -128)

            dst_x = x + src_x
            dst_y = y + (Y-r-1)*128
            dest_rect = ray.Rectangle(dst_x, dst_y, 128, 128)

            ray.draw_texture_pro(texture, source_rect, dest_rect, (0, 0), 0, tint)

def draw_arrow(tex, src_x, src_y, dst_x, dst_y, offset_x, offset_y, rot):
    source_rect = ray.Rectangle(src_x*32, src_y*32, 32, -32)
    dest_rect = ray.Rectangle(dst_x*128 + offset_x, SHEET_SIZE-(dst_y+1)*128+ offset_y, 32, 32)
    ray.draw_texture_pro(tex, source_rect, dest_rect, (0, 0), rot, ray.WHITE)

def draw_sheet(src, hair_i, tint, seed=None):
    if seed is not None:
        random.seed(seed)

    base_i = random.choice(BASE)
    if hair_i is None:
        hair_i = random.choice(HAIR)
    clothes_i = random.choice(CLOTHES)
    sword_i = random.choice(SWORD)
    bow_i = random.choice(BOW)
    quiver_i = random.choice(QUIVER)

    hair_variant = random.randint(0, 1)
    hair = [hair_dap, hair_bob][hair_variant]
    sword_hair = [sword_hair_dap, sword_hair_bob][hair_variant]
    bow_hair = [bow_hair_dap, bow_hair_bob][hair_variant]

    clothes_variant = random.randint(0, 1)
    clothes = [peasant_clothes, forester_clothes][clothes_variant]
    sword_clothes = [sword_peasant_clothes, sword_forester_clothes][clothes_variant]
    bow_clothes = [bow_peasant_clothes, bow_forester_clothes][clothes_variant]

    x = 0
    y = 1024
    draw_tex(src, character, base_i, x, y)
    draw_tex(src, hair, hair_i, x, y)
    draw_tex(src, clothes, clothes_i, x, y)

    x = 0
    y = 0
    draw_masked_tex(src, sword_weapon, sword_i, x, y, sword_mask, tint=tint)
    draw_tex(src, sword_character, base_i, x, y)
    draw_tex(src, sword_hair, hair_i, x, y)
    draw_tex(src, sword_clothes, clothes_i, x, y)
    draw_masked_tex(src, sword_weapon, sword_i, x, y, 1-sword_mask, tint=tint)

    x = 1024
    y = 1024
    draw_masked_tex(src, bow_weapon, bow_i, x, y, bow_mask, tint=tint)
    draw_masked_tex(src, bow_quiver, quiver_i, x, y, quiver_mask, tint=tint)
    draw_tex(src, bow_character, base_i, x, y)
    draw_tex(src, bow_hair, hair_i, x, y)
    draw_tex(src, bow_clothes, clothes_i, x, y)
    draw_masked_tex(src, bow_weapon, bow_i, x, y, 1-bow_mask, tint=tint)
    draw_masked_tex(src, bow_quiver, quiver_i, x, y, 1-quiver_mask, tint=tint)

    arrow_path = os.path.join(src, arrow).format(i=quiver_i)
    arrow_tex = ray.load_texture(arrow_path)

    ### Arrows are manually aligned
    # Left facing arrows
    draw_arrow(arrow_tex, 4, 1, 9, 3, 24, 40, 0)
    draw_arrow(arrow_tex, 4, 1, 10, 3, 24, 40, 0)
    draw_arrow(arrow_tex, 3, 1, 11, 3, 24, 52, 0)
    draw_arrow(arrow_tex, 1, 1, 12, 3, 38, 64, 0)

    # Right facing arrows
    draw_arrow(arrow_tex, 4, 1, 9, 2, 64+42, 48, 120)
    draw_arrow(arrow_tex, 4, 1, 10, 2, 64+42, 48, 120)
    draw_arrow(arrow_tex, 3, 1, 11, 2, 64+32, 82, 180)
    draw_arrow(arrow_tex, 1, 1, 12, 2, 56, 98, 180+80)


def scale_image(image_array, scale_factor):
    if scale_factor < 1:
        # Scale down with exact interpolation
        scaled_image_array = image_array[::int(1/scale_factor), ::int(1/scale_factor)]
    elif scale_factor > 1:
        # Scale up (duplicate pixels)
        scaled_image_array = np.repeat(
            np.repeat(
                image_array, scale_factor, axis=0
            ), scale_factor, axis=1
        )
    else:
        # No scaling
        scaled_image_array = image_array

    return scaled_image_array

def copy_and_scale_files(source_directory, target_directory, scale_factor):
    for root, dirs, files in os.walk(source_directory):
        relative_path = os.path.relpath(root, source_directory)
        target_path = os.path.join(target_directory)
        os.makedirs(target_path, exist_ok=True)

        for file in files:
            src_file_path = os.path.join(root, file)
            target_file_path = os.path.join(target_directory, file)

            path = src_file_path.lower()
            if path.endswith('.ttf'):
                os.copy(src_file_path, target_file_path)
                continue

            if not src_file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image = Image.open(src_file_path)
            image_array = np.array(image)
            scaled_image_array = scale_image(image_array, scale_factor)
            scaled_image = Image.fromarray(scaled_image_array)
            scaled_image.save(target_file_path)

if len(sys.argv) != 4:
    print("Usage: script.py source_directory target_directory scale_factor")
    sys.exit(1)

source_directory = sys.argv[1]
target_directory = sys.argv[2]
scale_factor = float(sys.argv[3])

if not os.path.exists(source_directory):
    print("Source directory does not exist.")
    sys.exit(1)

valid_scales = [0.125, 0.25, 0.5, 1, 2, 4]
if scale_factor not in valid_scales:
    print(f'Scale factor must be one of {valid_scales}')

intermediate_directory = os.path.join(target_directory, 'temp')
if not os.path.exists(intermediate_directory):
    os.makedirs(intermediate_directory)
    copy_and_scale_files(source_directory, intermediate_directory, scale_factor)

ray.init_window(SHEET_SIZE, SHEET_SIZE, "NMMO3")
ray.set_target_fps(60)

output_image = ray.load_render_texture(SHEET_SIZE, SHEET_SIZE)

i = 0
while not ray.window_should_close() and i < N_GENERATE:
    ray.set_window_title(f'Generating sheet {i+1}/{N_GENERATE}')

    for elem in ELEMENTS:
        elem_name, hair_i, tint = elem
        ray.begin_drawing()
        ray.begin_texture_mode(output_image)
        ray.clear_background(ray.BLANK)
        draw_sheet(intermediate_directory, hair_i, tint, seed=i)
        ray.end_texture_mode()

        image = ray.load_image_from_texture(output_image.texture)
        f_path = os.path.join(target_directory, f'{elem_name}_{i}.png')
        ray.export_image(image, f_path)

        ray.clear_background(ray.GRAY)
        ray.draw_texture(output_image.texture, 0, 0, ray.WHITE)
        ray.end_drawing()

    i += 1

coords = (0, 1)
spring = cv2.imread(intermediate_directory + '/spring forest.png')
summer = cv2.imread(intermediate_directory + '/summer forest.png')
autumn = cv2.imread(intermediate_directory + '/autumn forest (bare).png')
winter = cv2.imread(intermediate_directory + '/winter forest (clean).png')

spring = scale_image(spring, 2)
summer = scale_image(summer, 2)
autumn = scale_image(autumn, 2)
winter = scale_image(winter, 2)

SEASONS = [spring, summer, autumn, winter]

spring_sparkle = cv2.imread(intermediate_directory + '/spring water sparkles B.png')
summer_sparkle = cv2.imread(intermediate_directory + '/summer water sparkles B 16x16.png')
autumn_sparkle = cv2.imread(intermediate_directory + '/autumn water sparkles B 16x16.png')
winter_sparkle = cv2.imread(intermediate_directory + '/winter water sparkles B 16x16.png')

spring_sparkle = scale_image(spring_sparkle, 2)
summer_sparkle = scale_image(summer_sparkle, 2)
autumn_sparkle = scale_image(autumn_sparkle, 2)
winter_sparkle = scale_image(winter_sparkle, 2)

SPARKLES = [spring_sparkle, summer_sparkle, autumn_sparkle, winter_sparkle]

GRASS_OFFSET = (0, 0)
DIRT_OFFSET = (5, 0)
STONE_OFFSET = (9, 0)
WATER_OFFSET = (29, 16)

# Not compatible with water
GRASS_1 = (0, 1)
GRASS_2 = (0, 2)
GRASS_3 = (0, 3)
GRASS_4 = (0, 4)
GRASS_5 = (0, 5)

DIRT_1 = (8, 0)
DIRT_2 = (8, 1)
DIRT_3 = (8, 2)
DIRT_4 = (8, 3)
DIRT_5 = (8, 4)

STONE_1 = (12, 0)
STONE_2 = (12, 1)
STONE_3 = (12, 2)
STONE_4 = (12, 3)
STONE_5 = (12, 4)

WATER_1 = (27, 14)
WATER_2 = (28, 13)
WATER_3 = (28, 14)
WATER_4 = (29, 13)
WATER_5 = (29, 14)

GRASS_N = [GRASS_1, GRASS_2, GRASS_3, GRASS_4, GRASS_5]
DIRT_N = [DIRT_1, DIRT_2, DIRT_3, DIRT_4, DIRT_5]
STONE_N = [STONE_1, STONE_2, STONE_3, STONE_4, STONE_5]
WATER_N = [WATER_1, WATER_2, WATER_3, WATER_4, WATER_5]

ALL_MATERIALS = [DIRT_N, STONE_N, WATER_N]
ALL_OFFSETS = [DIRT_OFFSET, STONE_OFFSET, WATER_OFFSET]

# These values are just sentinels
# They will be mapped to GRASS/DIRT/STONE/WATER
FULL = (-1, 0)
EMPTY = (0, -1)

TL_CORNER = (0, 0)
T_FLAT = (1, 0)
TR_CORNER = (2, 0)
L_FLAT = (0, 1)
CENTER = (1, 1)
R_FLAT = (2, 1)
BL_CORNER = (0, 2)
B_FLAT = (1, 2)
BR_CORNER = (2, 2)
TL_DIAG = (0, 3)
TR_DIAG = (1, 3)
BL_DIAG = (0, 4)
BR_DIAG = (1, 4)
TRR_DIAG = (2, 3)
BRR_DIAG = (2, 4)

OFFSETS = [TL_CORNER, T_FLAT, TR_CORNER, L_FLAT, CENTER, R_FLAT, BL_CORNER,
    B_FLAT, BR_CORNER, TL_DIAG, TR_DIAG, BL_DIAG, BR_DIAG, TRR_DIAG, BRR_DIAG]

TILE_SIZE = int(32 * scale_factor)
SHEET_SIZE = 64
SHEET_PX = TILE_SIZE * SHEET_SIZE
merged_sheet = np.zeros((SHEET_PX, SHEET_PX, 3), dtype=np.uint8)

def gen_lerps():
    valid_combinations = []
    for combo in product(range(10), repeat=4):
        if sum(combo) == 9 and any(weight > 0 for weight in combo):
            valid_combinations.append(combo)

    return valid_combinations

def gen_lerps():
    valid_combinations = []
    for total_sum in range(1, 10):  # Loop through all possible sums from 1 to 9
        for combo in product(range(10), repeat=4):
            if sum(combo) == total_sum:
                valid_combinations.append(combo)
    return valid_combinations

def slice(r, c):
    return np.s_[
        r*TILE_SIZE:(r+1)*TILE_SIZE,
        c*TILE_SIZE:(c+1)*TILE_SIZE
    ]

idx = 0
for sheet in SEASONS:
    for offset, material in zip(ALL_OFFSETS, ALL_MATERIALS):
        src_dx, src_dy = offset

        # Write full tile textures. These are irregularly
        # arranged in the source sheet and require manual offsets.
        for src_x, src_y in material:
            dst_r, dst_c = divmod(idx, SHEET_SIZE)
            idx += 1

            src_pos = slice(src_y, src_x)
            tile_tex = sheet[src_pos]

            dst_pos = slice(dst_r, dst_c)
            merged_sheet[dst_pos] = tile_tex

        # Write partial tile textures. These have fixed offsets
        for dx, dy in OFFSETS:
            dst_r, dst_c = divmod(idx, SHEET_SIZE)
            idx += 1

            src_pos = slice(dy+src_dy, dx+src_dx)
            tile_tex = sheet[src_pos]

            dst_pos = slice(dst_r, dst_c)
            merged_sheet[dst_pos] = tile_tex

for x, y in WATER_N:
    # 3 animations
    for anim_y in range(3):
        for season, sparkle in zip(SEASONS, SPARKLES):
            src_pos = slice(y, x)
            tile_tex = season[src_pos]

            # 4 frame animation
            for anim_x in range(4):
                dst_r, dst_c = divmod(idx, SHEET_SIZE)
                idx += 1

                src_pos = slice(anim_y, anim_x)
                sparkle_tex = sparkle[src_pos]

                dst_pos = slice(dst_r, dst_c)
                merged_sheet[dst_pos] = tile_tex
                mask = np.where(sparkle_tex != 0)
                merged_sheet[dst_pos][mask] = sparkle_tex[mask]


for src in range(1, 5):
    tex_src = slice(src, 0)
    tiles = [spring[tex_src], summer[tex_src], autumn[tex_src], winter[tex_src]]
    for combo in gen_lerps():
        tex = np.zeros((TILE_SIZE, TILE_SIZE, 3))
        total_weight = sum(combo)
        for i, weight in enumerate(combo):
            tex += weight/total_weight * tiles[i]

        tex = tex.astype(np.uint8)

        dst_r, dst_c = divmod(idx, SHEET_SIZE)
        idx += 1

        dst_pos = slice(dst_r, dst_c)
        merged_sheet[dst_pos] = tex

    print(idx)

# save image
cv2.imwrite('merged_sheet.png', merged_sheet)
cv2.imshow('merged_sheet', merged_sheet)
cv2.waitKey(0)
