// Native floor-0 Craftax smoothworld generation.
//
// This ports the overworld branch of generate_smoothworld() for the default
// EnvParams. Floors 1..8 and all step logic remain proxy-backed for now.

#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include "noise.h"

#define CRAFTAX_OVERWORLD_SIZE 48
#define CRAFTAX_OVERWORLD_CELLS (CRAFTAX_OVERWORLD_SIZE * CRAFTAX_OVERWORLD_SIZE)

#define CRAFTAX_WG_BLOCK_OUT_OF_BOUNDS 1
#define CRAFTAX_WG_BLOCK_GRASS 2
#define CRAFTAX_WG_BLOCK_WATER 3
#define CRAFTAX_WG_BLOCK_STONE 4
#define CRAFTAX_WG_BLOCK_TREE 5
#define CRAFTAX_WG_BLOCK_PATH 7
#define CRAFTAX_WG_BLOCK_COAL 8
#define CRAFTAX_WG_BLOCK_IRON 9
#define CRAFTAX_WG_BLOCK_DIAMOND 10
#define CRAFTAX_WG_BLOCK_SAND 13
#define CRAFTAX_WG_BLOCK_LAVA 14

#define CRAFTAX_WG_ITEM_NONE 0
#define CRAFTAX_WG_ITEM_LADDER_DOWN 2

typedef struct CraftaxOverworldFloor {
    int32_t map[CRAFTAX_OVERWORLD_SIZE][CRAFTAX_OVERWORLD_SIZE];
    int32_t item_map[CRAFTAX_OVERWORLD_SIZE][CRAFTAX_OVERWORLD_SIZE];
    float light_map[CRAFTAX_OVERWORLD_SIZE][CRAFTAX_OVERWORLD_SIZE];
    int32_t ladder_down[2];
    int32_t ladder_up[2];
} CraftaxOverworldFloor;

static inline float craftax_wg_clampf(float value, float low, float high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static inline size_t craftax_wg_index(int row, int col) {
    return (size_t)row * (size_t)CRAFTAX_OVERWORLD_SIZE + (size_t)col;
}

static inline CraftaxThreefryKey craftax_overworld_rng_from_seed(uint32_t seed) {
    CraftaxThreefryKey key = craftax_prng_key(seed);
    CraftaxThreefryKey carry;
    CraftaxThreefryKey reset_key;
    craftax_threefry_split(key, &carry, &reset_key);

    CraftaxThreefryKey reset_carry;
    CraftaxThreefryKey world_key;
    craftax_threefry_split(reset_key, &reset_carry, &world_key);

    CraftaxThreefryKey world_keys[7];
    craftax_threefry_split_n(world_key, world_keys, 7);
    return world_keys[1];
}

static inline int craftax_choice_bool_flat(
    CraftaxThreefryKey key,
    const bool* valid,
    int count
) {
    int valid_count = 0;
    int last_valid = 0;
    for (int i = 0; i < count; i++) {
        if (valid[i]) {
            valid_count++;
            last_valid = i;
        }
    }
    if (valid_count == 0) {
        return 0;
    }

    float draw = (float)valid_count * (1.0f - craftax_threefry_uniform_f32(key));
    float cumulative = 0.0f;
    for (int i = 0; i < count; i++) {
        if (valid[i]) {
            cumulative += 1.0f;
        }
        if (cumulative >= draw) {
            return i;
        }
    }
    return last_valid;
}

static inline void craftax_generate_overworld_from_rng(
    CraftaxThreefryKey rng,
    CraftaxOverworldFloor* out
) {
    const int size = CRAFTAX_OVERWORLD_SIZE;
    const int player_row = CRAFTAX_OVERWORLD_SIZE / 2;
    const int player_col = CRAFTAX_OVERWORLD_SIZE / 2;
    const size_t cells = CRAFTAX_OVERWORLD_CELLS;

    CraftaxThreefryKey subkey;
    float water[cells];
    float mountain[cells];
    float path_x[cells];
    float tree_noise[cells];

    craftax_threefry_split(rng, &rng, &subkey);
    craftax_generate_fractal_noise_2d(subkey, size, size, 3, 3, 1, 0.5f, 2, NULL, water);

    craftax_threefry_split(rng, &rng, &subkey);
    (void)subkey;

    craftax_threefry_split(rng, &rng, &subkey);
    craftax_generate_fractal_noise_2d(subkey, size, size, 3, 3, 1, 0.5f, 2, NULL, mountain);

    craftax_threefry_split(rng, &rng, &subkey);
    craftax_generate_fractal_noise_2d(subkey, size, size, 6, 24, 1, 0.5f, 2, NULL, path_x);

    craftax_threefry_split(rng, &rng, &subkey);
    (void)subkey;

    craftax_threefry_split(rng, &rng, &subkey);
    CraftaxThreefryKey tree_uniform_key = rng;
    craftax_generate_fractal_noise_2d(subkey, size, size, 12, 12, 1, 0.5f, 2, NULL, tree_noise);

    for (int row = 0; row < size; row++) {
        int dr = row > player_row ? row - player_row : player_row - row;
        for (int col = 0; col < size; col++) {
            int dc = col > player_col ? col - player_col : player_col - col;
            float distance = sqrtf((float)(dr * dr + dc * dc));
            float proximity = craftax_wg_clampf(distance / 5.0f, 0.0f, 1.0f);
            size_t idx = craftax_wg_index(row, col);

            water[idx] = water[idx] + proximity - 1.0f;
            int32_t block = water[idx] > 0.7f
                ? CRAFTAX_WG_BLOCK_WATER
                : CRAFTAX_WG_BLOCK_GRASS;
            bool sand = water[idx] > 0.6f && block != CRAFTAX_WG_BLOCK_WATER;
            if (sand) {
                block = CRAFTAX_WG_BLOCK_SAND;
            }

            mountain[idx] = mountain[idx] + 0.05f + proximity - 1.0f;
            if (mountain[idx] > 0.7f) {
                block = CRAFTAX_WG_BLOCK_STONE;
            }

            bool path = mountain[idx] > 0.7f && path_x[idx] > 0.8f;
            if (path) {
                block = CRAFTAX_WG_BLOCK_PATH;
            }

            float path_y = path_x[craftax_wg_index(col, row)];
            path = mountain[idx] > 0.7f && path_y > 0.8f;
            if (path) {
                block = CRAFTAX_WG_BLOCK_PATH;
            }

            bool cave = mountain[idx] > 0.85f && water[idx] > 0.4f;
            if (cave) {
                block = CRAFTAX_WG_BLOCK_PATH;
            }

            float tree_draw = craftax_threefry_uniform_f32_at(tree_uniform_key, idx);
            bool tree = tree_noise[idx] > 0.5f && tree_draw > 0.8f;
            if (tree && block == CRAFTAX_WG_BLOCK_GRASS) {
                block = CRAFTAX_WG_BLOCK_TREE;
            }

            out->map[row][col] = block;
            out->item_map[row][col] = CRAFTAX_WG_ITEM_NONE;
            out->light_map[row][col] = 1.0f;
        }
    }

    static const int32_t ores[5] = {
        CRAFTAX_WG_BLOCK_COAL,
        CRAFTAX_WG_BLOCK_IRON,
        CRAFTAX_WG_BLOCK_DIAMOND,
        CRAFTAX_WG_BLOCK_OUT_OF_BOUNDS,
        CRAFTAX_WG_BLOCK_OUT_OF_BOUNDS,
    };
    static const float ore_chances[5] = {0.03f, 0.02f, 0.001f, 0.0f, 0.0f};

    CraftaxThreefryKey ore_rng;
    craftax_threefry_split(rng, &rng, &ore_rng);
    for (int ore_index = 0; ore_index < 5; ore_index++) {
        CraftaxThreefryKey ore_key;
        craftax_threefry_split(ore_rng, &ore_rng, &ore_key);
        for (int row = 0; row < size; row++) {
            for (int col = 0; col < size; col++) {
                size_t idx = craftax_wg_index(row, col);
                bool is_ore = out->map[row][col] == CRAFTAX_WG_BLOCK_STONE
                    && craftax_threefry_uniform_f32_at(ore_key, idx) < ore_chances[ore_index];
                if (is_ore) {
                    out->map[row][col] = ores[ore_index];
                }
            }
        }
    }

    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            size_t idx = craftax_wg_index(row, col);
            bool lava = mountain[idx] > 0.85f && tree_noise[idx] > 0.7f;
            if (lava) {
                out->map[row][col] = CRAFTAX_WG_BLOCK_LAVA;
            }
        }
    }

    out->map[player_row][player_col] = CRAFTAX_WG_BLOCK_GRASS;

    craftax_threefry_split(rng, &rng, &subkey);
    (void)subkey;

    bool valid_ladder[cells];
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            valid_ladder[craftax_wg_index(row, col)] =
                out->map[row][col] == CRAFTAX_WG_BLOCK_PATH;
        }
    }

    craftax_threefry_split(rng, &rng, &subkey);
    int ladder_down_index = craftax_choice_bool_flat(subkey, valid_ladder, (int)cells);
    out->ladder_down[0] = ladder_down_index / size;
    out->ladder_down[1] = ladder_down_index % size;
    out->item_map[out->ladder_down[0]][out->ladder_down[1]] = CRAFTAX_WG_ITEM_LADDER_DOWN;

    craftax_threefry_split(rng, &rng, &subkey);
    int ladder_up_index = craftax_choice_bool_flat(subkey, valid_ladder, (int)cells);
    out->ladder_up[0] = ladder_up_index / size;
    out->ladder_up[1] = ladder_up_index % size;
}

static inline void craftax_generate_overworld_from_seed(
    uint32_t seed,
    CraftaxOverworldFloor* out
) {
    craftax_generate_overworld_from_rng(craftax_overworld_rng_from_seed(seed), out);
}
