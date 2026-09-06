// JAX Craftax-Symbolic-v1 lockstep port (Threefry RNG matches the Python env).
// Used by tests/craftax_parity.py. Gameplay uses craftax.h (same rules, simpler RNG).
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

#include "constants.h"
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"
#include <stdio.h>
#include <stdlib.h>

#define ACT_SIZES {ATN_DIM}
#define NUM_ATNS 1
#ifdef PUFFERCPU_EVAL_MAIN
#define PUF_CRAFTAX_NET 1
#endif
#ifdef PUF_CRAFTAX_NET
#include "craftax_net.h"
#endif
#define MY_VEC_INIT
#define MY_VEC_CLOSE
typedef Env Craftax;

// Data structures 
typedef struct {
    int wood;
    int stone;
    int coal;
    int iron;
    int diamond;
    int sapling;
    int pickaxe;
    int sword;
    int bow;
    int arrows;
    int armour[4];
    int torches;
    int ruby;
    int sapphire;
    int potions[NUM_POTIONS];
    int books;
} Inventory;

typedef struct {
    int position[3][2];
    float health[3];
    bool mask[3];
    int attack_cooldown[3];
    int type_id[3];
} Mobs;

typedef struct {
    uint8_t map[NUM_LEVELS][MAP_SIZE][MAP_SIZE];
    uint8_t item_map[NUM_LEVELS][MAP_SIZE][MAP_SIZE];
    uint8_t light_map[NUM_LEVELS][MAP_SIZE][MAP_SIZE];
    uint64_t mob_bits[NUM_LEVELS][MAP_SIZE];
    uint64_t spawn_land[NUM_LEVELS][MAP_SIZE];
    uint64_t spawn_grave[NUM_LEVELS][MAP_SIZE];
    uint64_t spawn_water[NUM_LEVELS][MAP_SIZE];
    int down_ladders[NUM_LEVELS][2];
    int up_ladders[NUM_LEVELS][2];
    int chests_opened[NUM_LEVELS];
    int monsters_killed[NUM_LEVELS];

    int player_position[2];
    int player_level;
    int player_direction;

    // Intrinsics 
    float player_health;
    int player_food;
    int player_drink;
    int player_energy;
    int player_mana;
    int is_sleeping;
    int is_resting;

    // Second order intrinsics
    float player_recover;
    float player_hunger;
    float player_thirst;
    float player_fatigue;
    float player_recover_mana;

    // Attributes
    int player_xp;
    int player_dexterity;
    int player_strength;
    int player_intelligence;

    Inventory inventory;

    Mobs melee_mobs[NUM_LEVELS];
    Mobs passive_mobs[NUM_LEVELS];
    Mobs ranged_mobs[NUM_LEVELS];
    Mobs mob_projectiles[NUM_LEVELS];

    int mob_projectile_dirs[NUM_LEVELS][MAX_MOB_PROJECTILES][2];
    Mobs player_projectiles[NUM_LEVELS];
    int player_projectile_directions[NUM_LEVELS][MAX_PLAYER_PROJECTILES][2];
    int growing_plants_pos[MAX_GROWING_PLANTS][2];
    int growing_plants_age[MAX_GROWING_PLANTS];
    int growing_plants_mask[MAX_GROWING_PLANTS];
    int potion_mapping[NUM_POTIONS];
    int learned_spells[2];
    int sword_enchantment;
    int bow_enchantment;
    int armour_enchantments[4];
    int boss_progress;
    int boss_timestep_to_spawn_this_round;
    float light_level;
    int achievements[NUM_ACHIEVEMENTS];
    uint32_t state_rng[2];
    int timestep;
} State;

struct Log {
    float perf;
    float achievement_rate;
    float score;
    float episode_return;
    float episode_length;
    float floors[NUM_LEVELS];
    float achievements[NUM_ACHIEVEMENTS];
    float n;
};

// Rendering 
typedef struct {
    int cell_size;
    int screen_width;
    int screen_height;
    bool window_ready;
} Client;

// Random number generation
typedef uint64_t Rng;

struct Env {
    Client* client;
    Log log;
    Agent agents[1];
    int num_agents;
    int tag;
    int boundary_reached;
    State state;
    int timestep;
    unsigned int rng;
    uint64_t seed;
    Rng env_rng;
    float episode_return_accum;
    int episode_length_accum;
    int max_floor_accum;
    int achievements[NUM_ACHIEVEMENTS];
    State* levels;
    int num_levels;
    int use_action_mask;
    float predicted_value;
};

static uint32_t rng_rotl32(uint32_t x, uint32_t k) {
    return (uint32_t)((x << k) | (x >> (32u - k)));
}

static void rng_threefry2x32(Rng key, uint32_t count0, uint32_t count1, uint32_t out[2]) {
    static const uint32_t rotations[2][4] = {
        {13u, 15u, 26u, 6u},
        {17u, 29u, 16u, 24u},
    };
    uint32_t k0 = (uint32_t)key;
    uint32_t k1 = (uint32_t)(key >> 32);
    uint32_t ks[3] = {
        k0,
        k1,
        k0 ^ k1 ^ 0x1BD11BDAu,
    };
    uint32_t x0 = count0 + ks[0];
    uint32_t x1 = count1 + ks[1];
    for (uint32_t block = 0; block < 5u; block++) {
        const uint32_t* rs = rotations[block & 1u];
        for (int i = 0; i < 4; i++) {
            x0 += x1;
            x1 = rng_rotl32(x1, rs[i]);
            x1 ^= x0;
        }
        x0 += ks[(block + 1u) % 3u];
        x1 += ks[(block + 2u) % 3u] + block + 1u;
    }
    out[0] = x0;
    out[1] = x1;
}

static Rng rng_counter_key(Rng key, uint32_t count0, uint32_t count1) {
    uint32_t out[2];
    rng_threefry2x32(key, count0, count1, out);
    return (uint64_t)out[0] | ((uint64_t)out[1] << 32);
}

Rng rng_seed(uint32_t seed) {
    return (uint64_t)seed << 32;
}

void rng_split(Rng key, Rng* left, Rng* right) {
    *left = rng_counter_key(key, 0u, 0u);
    *right = rng_counter_key(key, 0u, 1u);
}

void rng_split_n(Rng key, Rng* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = rng_counter_key(key, 0u, (uint32_t)i);
    }
}

Rng rng_key(Rng* rng) {
    Rng draw;
    rng_split(*rng, rng, &draw);
    return draw;
}

uint32_t rng_u32(Rng key, uint64_t i) {
    uint32_t out[2];
    rng_threefry2x32(key, (uint32_t)(i >> 32), (uint32_t)i, out);
    return out[0] ^ out[1];
}

float rng_f32(Rng key, uint64_t i) {
    uint32_t bits = (rng_u32(key, i) >> 9u) | 0x3F800000u;
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v - 1.0f;
}

int randint(Rng key, uint64_t i, int lo, int hi) {
    Rng k1;
    Rng k2;
    rng_split(key, &k1, &k2);
    uint32_t higher_bits = rng_u32(k1, i);
    uint32_t lower_bits = rng_u32(k2, i);
    uint32_t span = (uint32_t)hi > (uint32_t)lo ? (uint32_t)(hi - lo) : 1u;
    uint32_t multiplier = 65536u % span;
    multiplier = (uint32_t)(((uint64_t)multiplier * (uint64_t)multiplier)
        % (uint64_t)span);
    uint32_t random_offset = (uint32_t)(
        (((uint64_t)(higher_bits % span) * (uint64_t)multiplier)
            + (uint64_t)(lower_bits % span))
        % (uint64_t)span
    );
    return lo + (int)random_offset;
}

void store_rng(State* state, Rng rng) {
    state->state_rng[0] = (uint32_t)rng;
    state->state_rng[1] = (uint32_t)(rng >> 32);
}

int choice_valid(Rng key, const bool* valid, int count) {
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
    float draw = valid_count * (1.0f - rng_f32(key, 0));
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

int clampi(int value, int low, int high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

float clampf(float value, float low, float high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

void refresh_spawn_cell(State* state, int level, int row, int col) {
    int block = state->map[level][row][col];
    uint64_t bit = 1ull << col;
    uint64_t* land = &state->spawn_land[level][row];
    uint64_t* grave = &state->spawn_grave[level][row];
    uint64_t* water = &state->spawn_water[level][row];
    *land = (*land & ~bit) | ((block == BLOCK_GRASS || block == BLOCK_PATH
        || block == BLOCK_FIRE_GRASS || block == BLOCK_ICE_GRASS) ? bit : 0);
    *grave = (*grave & ~bit) | ((block == BLOCK_GRAVE || block == BLOCK_GRAVE2
        || block == BLOCK_GRAVE3) ? bit : 0);
    *water = (*water & ~bit) | (block == BLOCK_WATER ? bit : 0);
}

void set_block(State* state, int level, int row, int col, int block) {
    state->map[level][row][col] = block;
    refresh_spawn_cell(state, level, row, col);
}

void generate_fractal(Rng rng, int rows, int cols, int res_rows, int res_cols,
        int octaves, float persistence, int lacunarity, float* out) {
    // Perlin noise for world generation
    int size = rows * cols;
    memset(out, 0, size * sizeof(float));
    int frequency = 1;
    float amplitude = 1.0f;
    for (int octave = 0; octave < octaves; octave++) {
        Rng next_rng;
        Rng noise_key;
        rng_split(rng, &next_rng, &noise_key);
        rng = next_rng;

        Rng unused;
        Rng angle_key;
        rng_split(noise_key, &unused, &angle_key);
        int cell_rows = rows / (frequency * res_rows);
        int cell_cols = cols / (frequency * res_cols);
        int width = frequency * res_cols + 1;

        for (int row = 0; row < rows; row++) {
            int grad_row = row / cell_rows;
            float local_row = (row - grad_row * cell_rows) / (float)cell_rows;
            float interp_row = local_row * local_row * local_row
                * (local_row * (local_row * 6.0f - 15.0f) + 10.0f);
            for (int col = 0; col < cols; col++) {
                int grad_col = col / cell_cols;
                float local_col = (col - grad_col * cell_cols) / (float)cell_cols;
                float interp_col = local_col * local_col * local_col
                    * (local_col * (local_col * 6.0f - 15.0f) + 10.0f);
                float gx[2][2];
                float gy[2][2];
                for (int dr = 0; dr < 2; dr++) {
                    for (int dc = 0; dc < 2; dc++) {
                        uint64_t index = (grad_row + dr) * width + (grad_col + dc);
                        float angle = NOISE_PI2 * rng_f32(angle_key, index);
                        gx[dr][dc] = cosf(angle);
                        gy[dr][dc] = sinf(angle);
                    }
                }
                float n00 = local_row * gx[0][0] + local_col * gy[0][0];
                float n10 = (local_row - 1.0f) * gx[1][0] + local_col * gy[1][0];
                float n01 = local_row * gx[0][1] + (local_col - 1.0f) * gy[0][1];
                float n11 = (local_row - 1.0f) * gx[1][1] + (local_col - 1.0f) * gy[1][1];
                float n0 = n00 * (1.0f - interp_row) + interp_row * n10;
                float n1 = n01 * (1.0f - interp_row) + interp_row * n11;
                out[row * cols + col] += amplitude
                    * NOISE_SQRT2 * ((1.0f - interp_col) * n0 + interp_col * n1);
            }
        }

        frequency *= lacunarity;
        amplitude *= persistence;
    }
    float min_value = out[0];
    float max_value = out[0];
    for (int i = 1; i < size; i++) {
        if (out[i] < min_value) {
            min_value = out[i];
        }
        if (out[i] > max_value) {
            max_value = out[i];
        }
    }
    float scale = max_value - min_value;
    for (int i = 0; i < size; i++) {
        out[i] = (out[i] - min_value) / scale;
    }
}

int cell_index(int row, int col) {
    return row * MAP_SIZE + col;
}

void generate_world_from_key(State* state, Rng rng) {
    memset(state, 0, sizeof(*state));
    Rng smooth_split[7];
    rng_split_n(rng, smooth_split, 7);
    rng = smooth_split[0];

    static const int smooth_floor_order[6] = {0, 2, 5, 6, 7, 8};
    for (int i = 0; i < 6; i++) {
        int level = smooth_floor_order[i];
        Rng level_rng = smooth_split[i + 1];
        const SmoothGenConfig* config = &SMOOTH_LEVEL_CONFIGS[i];
        const int player_row = MAP_SIZE / 2;
        const int player_col = MAP_SIZE / 2;
        float water[MAP_CELLS];
        float mountain[MAP_CELLS];
        float path_x[MAP_CELLS];
        float tree_noise[MAP_CELLS];
        bool lava_map[MAP_SIZE][MAP_SIZE];
        float light_acc[MAP_SIZE][MAP_SIZE];
        Rng subkey;

        rng_split(level_rng, &level_rng, &subkey);
        generate_fractal(subkey, MAP_SIZE, MAP_SIZE, 3, 3, 1, 0.5f, 2, water);
        rng_split(level_rng, &level_rng, &subkey);
        rng_split(level_rng, &level_rng, &subkey);
        generate_fractal(subkey, MAP_SIZE, MAP_SIZE, 3, 3, 1, 0.5f, 2, mountain);
        rng_split(level_rng, &level_rng, &subkey);
        generate_fractal(subkey, MAP_SIZE, MAP_SIZE, 6, 24, 1, 0.5f, 2, path_x);
        rng_split(level_rng, &level_rng, &subkey);
        rng_split(level_rng, &level_rng, &subkey);
        Rng tree_uniform_key = level_rng;
        generate_fractal(subkey, MAP_SIZE, MAP_SIZE, 12, 12, 1, 0.5f, 2, tree_noise);

        for (int row = 0; row < MAP_SIZE; row++) {
            int dr = row > player_row ? row - player_row : player_row - row;
            for (int col = 0; col < MAP_SIZE; col++) {
                int dc = col > player_col ? col - player_col : player_col - col;
                float distance = sqrtf(dr * dr + dc * dc);
                float proximity_water = clampf(distance / config->water_strength,
                    0.0f, config->water_max);
                float proximity_mountain = clampf(distance / config->mountain_strength,
                    0.0f, config->mountain_max);
                int idx = cell_index(row, col);

                water[idx] = water[idx] + proximity_water - 1.0f;
                int block = water[idx] > config->water_threshold
                    ? config->sea_block
                    : config->default_block;
                if (water[idx] > config->sand_threshold && block != config->sea_block) {
                    block = config->coast_block;
                }

                mountain[idx] = mountain[idx] + 0.05f + proximity_mountain - 1.0f;
                if (mountain[idx] > 0.7f) {
                    block = config->mountain_block;
                }
                if (mountain[idx] > 0.7f && path_x[idx] > 0.8f) {
                    block = config->path_block;
                }
                if (mountain[idx] > 0.7f && path_x[cell_index(col, row)] > 0.8f) {
                    block = config->path_block;
                }
                if (mountain[idx] > 0.85f && water[idx] > 0.4f) {
                    block = config->inner_mountain_block;
                }
                if (tree_noise[idx] > config->tree_threshold_perlin
                        && rng_f32(tree_uniform_key, idx) > config->tree_threshold_uniform
                        && block == config->tree_requirement_block) {
                    block = config->tree;
                }

                state->map[level][row][col] = block;
                state->item_map[level][row][col] = ITEM_NONE;
                light_acc[row][col] = config->default_light;
            }
        }

        Rng ore_rng;
        rng_split(level_rng, &level_rng, &ore_rng);
        for (int ore_index = 0; ore_index < 5; ore_index++) {
            Rng ore_key;
            rng_split(ore_rng, &ore_rng, &ore_key);
            for (int row = 0; row < MAP_SIZE; row++) {
                for (int col = 0; col < MAP_SIZE; col++) {
                    int idx = cell_index(row, col);
                    if (state->map[level][row][col] == config->ore_requirement_blocks[ore_index]
                            && rng_f32(ore_key, idx) < config->ore_chances[ore_index]) {
                        state->map[level][row][col] = config->ores[ore_index];
                    }
                }
            }
        }

        for (int row = 0; row < MAP_SIZE; row++) {
            for (int col = 0; col < MAP_SIZE; col++) {
                int idx = cell_index(row, col);
                lava_map[row][col] = mountain[idx] > 0.85f && tree_noise[idx] > 0.7f;
                if (lava_map[row][col]) {
                    state->map[level][row][col] = config->lava;
                }
            }
        }

        rng_split(level_rng, &level_rng, &subkey);
        bool valid_diamond[MAP_CELLS];
        for (int row = 0; row < MAP_SIZE; row++) {
            for (int col = 0; col < MAP_SIZE; col++) {
                valid_diamond[cell_index(row, col)] = state->map[level][row][col] == BLOCK_STONE;
            }
        }
        int diamond_index = choice_valid(subkey, valid_diamond, MAP_CELLS);
        state->map[level][diamond_index / MAP_SIZE][diamond_index % MAP_SIZE] = BLOCK_STONE;
        state->map[level][player_row][player_col] = config->player_spawn;

        bool valid_ladder[MAP_CELLS];
        for (int row = 0; row < MAP_SIZE; row++) {
            for (int col = 0; col < MAP_SIZE; col++) {
                valid_ladder[cell_index(row, col)] =
                    state->map[level][row][col] == config->valid_ladder;
            }
        }

        rng_split(level_rng, &level_rng, &subkey);
        int ladder_down_index = choice_valid(subkey, valid_ladder, MAP_CELLS);
        state->down_ladders[level][0] = ladder_down_index / MAP_SIZE;
        state->down_ladders[level][1] = ladder_down_index % MAP_SIZE;
        if (config->ladder_down) {
            state->item_map[level][state->down_ladders[level][0]][state->down_ladders[level][1]] =
                ITEM_LADDER_DOWN;
        }

        rng_split(level_rng, &level_rng, &subkey);
        int ladder_up_index = choice_valid(subkey, valid_ladder, MAP_CELLS);
        int r = ladder_up_index / MAP_SIZE;
        int c = ladder_up_index % MAP_SIZE;
        state->up_ladders[level][0] = r;
        state->up_ladders[level][1] = c;
        int light_row = r - 4;
        int light_col = c - 4;
        if (light_row < 0) {
            light_row += MAP_SIZE;
        }
        if (light_col < 0) {
            light_col += MAP_SIZE;
        }
        light_row = clampi(light_row, 0, MAP_SIZE - 9);
        light_col = clampi(light_col, 0, MAP_SIZE - 9);
        for (int lr = 0; lr < 9; lr++) {
            for (int lc = 0; lc < 9; lc++) {
                float torch = clampf(1.0f - sqrtf((lr - 4) * (lr - 4)
                    + (lc - 4) * (lc - 4)) / 5.0f, 0.0f, 1.0f);
                float light = torch * (1.0f - config->default_light) + config->default_light;
                light_acc[light_row + lr][light_col + lc] = light;
            }
        }
        if (config->lava == BLOCK_LAVA) {
            static const float kernel[3][3] = {
                {0.2f, 0.7f, 0.2f},
                {0.7f, 1.0f, 0.7f},
                {0.2f, 0.7f, 0.2f},
            };
            for (int row = 0; row < MAP_SIZE; row++) {
                for (int col = 0; col < MAP_SIZE; col++) {
                    float add = 0.0f;
                    for (int kr = 0; kr < 3; kr++) {
                        int src_row = row + kr - 1;
                        if (src_row < 0 || src_row >= MAP_SIZE) {
                            continue;
                        }
                        for (int kc = 0; kc < 3; kc++) {
                            int src_col = col + kc - 1;
                            if (src_col < 0 || src_col >= MAP_SIZE) {
                                continue;
                            }
                            if (lava_map[src_row][src_col]) {
                                add += kernel[kr][kc];
                            }
                        }
                    }
                    light_acc[row][col] = clampf(light_acc[row][col] + add, 0.0f, 1.0f);
                }
            }
        }
        for (int row = 0; row < MAP_SIZE; row++) {
            for (int col = 0; col < MAP_SIZE; col++) {
                float light = clampf(light_acc[row][col], 0.0f, 1.0f);
                state->light_map[level][row][col] = (unsigned char)(light * 255.0f);
            }
        }
        if (config->ladder_up) {
            state->item_map[level][r][c] = ITEM_LADDER_UP;
        }
    }

    Rng dungeon_split[4];
    rng_split_n(rng, dungeon_split, 4);
    rng = dungeon_split[0];
    static const int dungeon_floor_order[3] = {1, 3, 4};
    for (int i = 0; i < 3; i++) {
        int level = dungeon_floor_order[i];
        Rng level_rng = dungeon_split[i + 1];
        const DungeonConfig* config = &DUNGEON_LEVEL_CONFIGS[i];
        const int chunk_size = DUNGEON_CHUNK_SIZE;
        const int world_chunk_height = MAP_SIZE / chunk_size;
        const int num_rooms = DUNGEON_ROOM_COUNT;
        const int min_room_size = DUNGEON_MIN_ROOM_SIZE;
        const int max_room_size = DUNGEON_MAX_ROOM_SIZE;
        const int padded_size = MAP_SIZE + 2 * max_room_size;

        int padded_map[68][68];
        int padded_item[68][68];
        bool room_occupancy[9];
        int room_sizes[8][2];
        int room_positions[8][2];

        for (int row = 0; row < padded_size; row++) {
            for (int col = 0; col < padded_size; col++) {
                bool inner = row >= max_room_size
                    && row < max_room_size + MAP_SIZE
                    && col >= max_room_size
                    && col < max_room_size + MAP_SIZE;
                padded_map[row][col] = inner ? BLOCK_WALL : 0;
                padded_item[row][col] = ITEM_NONE;
            }
        }
        for (int i = 0; i < 9; i++) {
            room_occupancy[i] = true;
        }

        Rng keys3[3];
        rng_split_n(level_rng, keys3, 3);
        level_rng = keys3[0];
        Rng room_size_key = keys3[2];
        for (int room = 0; room < num_rooms; room++) {
            room_sizes[room][0] = randint(room_size_key, room * 2u, min_room_size, max_room_size);
            room_sizes[room][1] = randint(room_size_key, room * 2u + 1u, min_room_size, max_room_size);
        }

        Rng room_rng;
        rng_split(level_rng, &level_rng, &room_rng);
        for (int room_index = 0; room_index < num_rooms; room_index++) {
            Rng choice_key;
            rng_split(room_rng, &room_rng, &choice_key);
            int room_chunk = choice_valid(choice_key, room_occupancy, 9);
            room_occupancy[room_chunk] = false;
            int room_row = (room_chunk % world_chunk_height) * chunk_size + max_room_size;
            int room_col = (room_chunk / world_chunk_height) * chunk_size + max_room_size;
            Rng position_key;
            rng_split(room_rng, &room_rng, &position_key);
            room_row += randint(position_key, 0, 0, chunk_size - min_room_size);
            room_col += randint(position_key, 1, 0, chunk_size - min_room_size);
            room_positions[room_index][0] = room_row;
            room_positions[room_index][1] = room_col;

            for (int row = 0; row < max_room_size; row++) {
                for (int col = 0; col < max_room_size; col++) {
                    if (row < room_sizes[room_index][0] && col < room_sizes[room_index][1]) {
                        padded_map[room_row + row][room_col + col] = BLOCK_PATH;
                    }
                }
            }

            padded_item[room_row][room_col] = ITEM_TORCH;
            padded_item[room_row + room_sizes[room_index][0] - 1][room_col] = ITEM_TORCH;
            padded_item[room_row][room_col + room_sizes[room_index][1] - 1] = ITEM_TORCH;
            padded_item[room_row + room_sizes[room_index][0] - 1][room_col + room_sizes[room_index][1] - 1] = ITEM_TORCH;

            Rng chest_key;
            rng_split(room_rng, &room_rng, &chest_key);
            int chest_row = randint(chest_key, 0, 1, room_sizes[room_index][0] - 1);
            int chest_col = randint(chest_key, 1, 1, room_sizes[room_index][1] - 1);
            padded_map[room_row + chest_row][room_col + chest_col] = BLOCK_CHEST;

            Rng fountain_keys[3];
            rng_split_n(room_rng, fountain_keys, 3);
            room_rng = fountain_keys[0];
            int fountain_row = randint(fountain_keys[1], 0, 1, room_sizes[room_index][0] - 1);
            int fountain_col = randint(fountain_keys[1], 1, 1, room_sizes[room_index][1] - 1);
            if (rng_f32(fountain_keys[2], 0) > 0.5f) {
                padded_map[room_row + fountain_row][room_col + fountain_col] = config->fountain_block;
            }
        }

        Rng path_rng;
        rng_split(level_rng, &level_rng, &path_rng);
        bool included_rooms[8] = {false, false, false, false, false, false, false, true};
        for (int path_index = 0; path_index < num_rooms; path_index++) {
            int source_row = room_positions[path_index][0];
            int source_col = room_positions[path_index][1];
            Rng sink_key;
            rng_split(path_rng, &path_rng, &sink_key);
            int sink_index = choice_valid(sink_key, included_rooms, num_rooms);
            int sink_row = room_positions[sink_index][0];
            int sink_col = room_positions[sink_index][1];

            int horizontal_distance = sink_col - source_col;
            int horizontal_sign = (horizontal_distance > 0) - (horizontal_distance < 0);
            if (horizontal_sign != 0) {
                int abs_distance = horizontal_distance > 0 ? horizontal_distance : -horizontal_distance;
                for (int col = 0; col < padded_size; col++) {
                    int path_index_col = (col - source_col) * horizontal_sign;
                    if (path_index_col >= 0 && path_index_col <= abs_distance
                            && padded_map[source_row][col] == BLOCK_WALL) {
                        padded_map[source_row][col] = BLOCK_PATH;
                    }
                }
            }
            int vertical_distance = sink_row - source_row;
            int vertical_sign = (vertical_distance > 0) - (vertical_distance < 0);
            if (vertical_sign != 0) {
                int abs_distance = vertical_distance > 0 ? vertical_distance : -vertical_distance;
                for (int row = 0; row < padded_size; row++) {
                    int path_index_row = (row - source_row) * vertical_sign;
                    if (path_index_row >= 0 && path_index_row <= abs_distance
                            && padded_map[row][sink_col] == BLOCK_WALL) {
                        padded_map[row][sink_col] = BLOCK_PATH;
                    }
                }
            }

            Rng unused_left;
            Rng next_path_rng;
            rng_split(path_rng, &unused_left, &next_path_rng);
            path_rng = next_path_rng;
            included_rooms[path_index] = true;
        }

        padded_map[room_positions[0][0] + 2][room_positions[0][1] + 2] = config->special_block;

        for (int row = 0; row < MAP_SIZE; row++) {
            for (int col = 0; col < MAP_SIZE; col++) {
                state->map[level][row][col] =
                    padded_map[row + max_room_size][col + max_room_size];
                state->item_map[level][row][col] =
                    padded_item[row + max_room_size][col + max_room_size];
            }
        }

        bool adjacent_path[MAP_SIZE][MAP_SIZE];
        for (int row = 0; row < MAP_SIZE; row++) {
            for (int col = 0; col < MAP_SIZE; col++) {
                bool adjacent = state->map[level][row][col] != BLOCK_WALL;
                adjacent = adjacent || (row > 0 && state->map[level][row - 1][col] != BLOCK_WALL);
                adjacent = adjacent || (row + 1 < MAP_SIZE && state->map[level][row + 1][col] != BLOCK_WALL);
                adjacent = adjacent || (col > 0 && state->map[level][row][col - 1] != BLOCK_WALL);
                adjacent = adjacent || (col + 1 < MAP_SIZE && state->map[level][row][col + 1] != BLOCK_WALL);
                adjacent_path[row][col] = adjacent;
            }
        }

        Rng rare_key;
        rng_split(level_rng, &level_rng, &rare_key);
        for (int row = 0; row < MAP_SIZE; row++) {
            for (int col = 0; col < MAP_SIZE; col++) {
                int idx = cell_index(row, col);
                bool rare = (1.0f - rng_f32(rare_key, idx)) > 0.9f;
                int wall_map = rare ? BLOCK_WALL_MOSS : BLOCK_WALL;
                bool rare_path = rare
                    && state->map[level][row][col] == BLOCK_PATH
                    && state->item_map[level][row][col] == ITEM_NONE;
                int path_map = rare_path ? config->rare_path_replacement_block : state->map[level][row][col];
                bool is_wall_map = state->map[level][row][col] == BLOCK_WALL && adjacent_path[row][col];
                if (!adjacent_path[row][col]) {
                    state->map[level][row][col] = BLOCK_DARKNESS;
                } else if (is_wall_map) {
                    state->map[level][row][col] = wall_map;
                } else {
                    state->map[level][row][col] = path_map;
                }
                state->light_map[level][row][col] = 255;
            }
        }

        bool valid_ladder[MAP_CELLS];
        for (int row = 0; row < MAP_SIZE; row++) {
            for (int col = 0; col < MAP_SIZE; col++) {
                valid_ladder[cell_index(row, col)] = state->map[level][row][col] == BLOCK_PATH;
            }
        }
        Rng ladder_down_key;
        rng_split(level_rng, &level_rng, &ladder_down_key);
        int ladder_down_index = choice_valid(ladder_down_key, valid_ladder, MAP_CELLS);
        int r = ladder_down_index / MAP_SIZE;
        int c = ladder_down_index % MAP_SIZE;
        state->down_ladders[level][0] = r;
        state->down_ladders[level][1] = c;
        state->item_map[level][r][c] = ITEM_LADDER_DOWN;

        Rng ladder_up_key;
        rng_split(level_rng, &level_rng, &ladder_up_key);
        int ladder_up_index = choice_valid(ladder_up_key, valid_ladder, MAP_CELLS);
        r = ladder_up_index / MAP_SIZE;
        c = ladder_up_index % MAP_SIZE;
        state->up_ladders[level][0] = r;
        state->up_ladders[level][1] = c;
        state->item_map[level][r][c] = ITEM_LADDER_UP;
    }

    for (int level = 0; level < NUM_LEVELS; level++) {
        for (int i = 0; i < MAX_MELEE_MOBS; i++) {
            state->melee_mobs[level].health[i] = 1.0f;
            state->passive_mobs[level].health[i] = 1.0f;
            state->mob_projectiles[level].health[i] = 1.0f;
            state->player_projectiles[level].health[i] = 1.0f;
        }
        for (int i = 0; i < MAX_RANGED_MOBS; i++) {
            state->ranged_mobs[level].health[i] = 1.0f;
        }
        for (int projectile = 0; projectile < MAX_MOB_PROJECTILES; projectile++) {
            state->mob_projectile_dirs[level][projectile][0] = 1;
            state->mob_projectile_dirs[level][projectile][1] = 1;
        }
        for (int projectile = 0; projectile < MAX_PLAYER_PROJECTILES; projectile++) {
            state->player_projectile_directions[level][projectile][0] = 1;
            state->player_projectile_directions[level][projectile][1] = 1;
        }
    }

    Rng potion_key;
    rng_split(rng, &rng, &potion_key);
    Rng potion_carry;
    Rng sort_key;
    rng_split(potion_key, &potion_carry, &sort_key);
    uint32_t potion_keys[6];
    for (int i = 0; i < 6; i++) {
        potion_keys[i] = rng_u32(sort_key, i);
        state->potion_mapping[i] = i;
    }
    for (int i = 1; i < 6; i++) {
        uint32_t key_value = potion_keys[i];
        int value = state->potion_mapping[i];
        int j = i - 1;
        while (j >= 0 && potion_keys[j] > key_value) {
            potion_keys[j + 1] = potion_keys[j];
            state->potion_mapping[j + 1] = state->potion_mapping[j];
            j--;
        }
        potion_keys[j + 1] = key_value;
        state->potion_mapping[j + 1] = value;
    }

    Rng state_key;
    rng_split(rng, &rng, &state_key);
    store_rng(state, state_key);

    state->monsters_killed[0] = 10;
    state->player_position[0] = MAP_SIZE / 2;
    state->player_position[1] = MAP_SIZE / 2;
    state->player_level = 0;
    state->player_direction = ACTION_UP;
    state->player_health = 9.0f;
    state->player_food = 9;
    state->player_drink = 9;
    state->player_energy = 9;
    state->player_mana = 9;
    state->player_dexterity = 1;
    state->player_strength = 1;
    state->player_intelligence = 1;
    state->boss_timestep_to_spawn_this_round = BOSS_SPAWN_TURNS;
    float cosine = cosf(3.14159265358979323846f * 0.3f);
    state->light_level = 1.0f - powf(fabsf(cosine), 3.0f);
    memset(state->spawn_land, 0, sizeof(state->spawn_land));
    memset(state->spawn_grave, 0, sizeof(state->spawn_grave));
    memset(state->spawn_water, 0, sizeof(state->spawn_water));
    for (int level = 0; level < NUM_LEVELS; level++) {
        for (int row = 0; row < MAP_SIZE; row++) {
            for (int col = 0; col < MAP_SIZE; col++) {
                refresh_spawn_cell(state, level, row, col);
            }
        }
    }
}

void write_mob_obs(float* obs, const State* state, const Mobs* mobs, int slots,
        int channel) {
    int level = state->player_level;
    int half_r = OBS_ROWS / 2;
    int half_c = OBS_COLS / 2;
    for (int i = 0; i < slots; i++) {
        int local_row = mobs->position[i][0] - state->player_position[0] + half_r;
        int local_col = mobs->position[i][1] - state->player_position[1] + half_c;
        int on_screen = mobs->mask[i]
            && local_row >= 0 && local_row < OBS_ROWS
            && local_col >= 0 && local_col < OBS_COLS;
        if (local_row >= OBS_ROWS || local_row < -OBS_ROWS
                || local_col >= OBS_COLS || local_col < -OBS_COLS) {
            continue;
        }
        if (local_row < 0) {
            local_row += OBS_ROWS;
        }
        if (local_col < 0) {
            local_col += OBS_COLS;
        }
        int dest_row = state->player_position[0] + local_row - half_r;
        int dest_col = state->player_position[1] + local_col - half_c;
        int dest_visible = dest_row >= 0 && dest_row < MAP_SIZE
            && dest_col >= 0 && dest_col < MAP_SIZE
            && state->light_map[level][dest_row][dest_col] > VISIBLE_LIGHT_THRESHOLD;
        float value = 0.0f;
        if (on_screen && dest_visible) {
            value = (float)(mobs->type_id[i] + 1);
        }
        int base = (local_row * OBS_COLS + local_col) * OBS_TILE_CHANNELS;
        obs[base + 3 + channel] = value;
    }
}

int max_health(const State* state) {
    return 8 + state->player_strength;
}

int equipped_armour(const State* state) {
    return state->inventory.armour[0] + state->inventory.armour[1]
        + state->inventory.armour[2] + state->inventory.armour[3];
}

int max_need(const State* state) {
    return 7 + 2 * state->player_dexterity;
}

int max_mana(const State* state) {
    return 6 + 3 * state->player_intelligence;
}

bool fighting_boss(const State* state) {
    return state->player_level == NUM_LEVELS - 1;
}

bool boss_vulnerable(const State* state) {
    if (state->boss_timestep_to_spawn_this_round > 0) {
        return false;
    }
    int level = state->player_level;
    for (int i = 0; i < MAX_MELEE_MOBS; i++) {
        if (state->melee_mobs[level].mask[i]) {
            return false;
        }
    }
    for (int i = 0; i < MAX_RANGED_MOBS; i++) {
        if (state->ranged_mobs[level].mask[i]) {
            return false;
        }
    }
    return true;
}

void action_to_direction(int action, int direction[2]) {
    direction[0] = 0;
    direction[1] = 0;

    if (action == ACTION_LEFT) {
        direction[1] = -1;
    } else if (action == ACTION_RIGHT) {
        direction[1] = 1;
    } else if (action == ACTION_UP) {
        direction[0] = -1;
    } else if (action == ACTION_DOWN) {
        direction[0] = 1;
    }
}

bool is_solid_block(int block) {
    static const unsigned char solid[NUM_BLOCK_TYPES] = {
        [BLOCK_STONE] = 1, [BLOCK_TREE] = 1, [BLOCK_COAL] = 1,
        [BLOCK_IRON] = 1, [BLOCK_DIAMOND] = 1, [BLOCK_CRAFTING_TABLE] = 1,
        [BLOCK_FURNACE] = 1, [BLOCK_PLANT] = 1, [BLOCK_RIPE_PLANT] = 1,
        [BLOCK_WALL] = 1, [BLOCK_WALL_MOSS] = 1, [BLOCK_STALAGMITE] = 1,
        [BLOCK_RUBY] = 1, [BLOCK_SAPPHIRE] = 1, [BLOCK_CHEST] = 1,
        [BLOCK_FOUNTAIN] = 1, [BLOCK_FIRE_TREE] = 1,
        [BLOCK_ENCHANTMENT_TABLE_FIRE] = 1, [BLOCK_ENCHANTMENT_TABLE_ICE] = 1,
        [BLOCK_GRAVE] = 1, [BLOCK_GRAVE2] = 1, [BLOCK_GRAVE3] = 1,
        [BLOCK_NECROMANCER] = 1,
    };
    return (unsigned)block < NUM_BLOCK_TYPES && solid[block];
}

bool mob_at(const State* state, int level, int row, int col) {
    if ((unsigned)row >= MAP_SIZE || (unsigned)col >= MAP_SIZE) {
        return false;
    }
    return (state->mob_bits[level][row] >> col) & 1ull;
}

void set_mob_bit(State* state, int level, int row, int col, bool on) {
    if ((unsigned)row >= MAP_SIZE || (unsigned)col >= MAP_SIZE) {
        return;
    }
    uint64_t bit = 1ull << col;
    if (on) {
        state->mob_bits[level][row] |= bit;
    } else {
        state->mob_bits[level][row] &= ~bit;
    }
}

void move_mob_occupancy(State* state, int level, int old_row, int old_col,
        int new_row, int new_col, bool keep) {
    set_mob_bit(state, level, old_row, old_col, false);
    if (keep) {
        set_mob_bit(state, level, new_row, new_col, true);
    }
}

bool mobs_at(const Mobs* mobs, int slots, int row, int col, int* slot) {
    for (int i = 0; i < slots; i++) {
        if (mobs->mask[i]
                && mobs->position[i][0] == row
                && mobs->position[i][1] == col) {
            *slot = i;
            return true;
        }
    }
    return false;
}

Mobs* mobs_for_class(State* state, int level, int mob_class) {
    if (mob_class == MOB_PASSIVE) {
        return &state->passive_mobs[level];
    }
    if (mob_class == MOB_RANGED) {
        return &state->ranged_mobs[level];
    }
    return &state->melee_mobs[level];
}

bool find_mob_at(const State* state, int level, int row, int col, int* mob_class,
        int* slot) {
    if (mobs_at(&state->melee_mobs[level], MAX_MELEE_MOBS, row, col, slot)) {
        *mob_class = MOB_MELEE;
        return true;
    }
    if (mobs_at(&state->passive_mobs[level], MAX_PASSIVE_MOBS, row, col, slot)) {
        *mob_class = MOB_PASSIVE;
        return true;
    }
    if (mobs_at(&state->ranged_mobs[level], MAX_RANGED_MOBS, row, col, slot)) {
        *mob_class = MOB_RANGED;
        return true;
    }
    return false;
}

bool valid_typed_mob_position(const State* state, int level, int mob_class,
        int type_id, int row, int col, int old_row, int old_col) {
    if (row < 0 || row >= MAP_SIZE || col < 0 || col >= MAP_SIZE) {
        return false;
    }
    if (row == state->player_position[0] && col == state->player_position[1]) {
        return false;
    }
    int block = state->map[level][row][col];
    if (is_solid_block(block)) {
        return false;
    }
    static const bool blocked[NUM_MOB_TYPES][3][3] = {
        {{0,1,1},{0,1,1},{0,1,1}}, {{0,0,0},{0,1,1},{0,1,1}},
        {{0,1,1},{0,1,1},{0,1,1}}, {{0,1,1},{0,0,1},{0,1,1}},
        {{0,1,1},{0,1,1},{0,1,1}}, {{0,1,1},{0,1,1},{1,0,1}},
        {{0,1,1},{0,1,1},{0,0,0}}, {{0,1,1},{0,1,1},{0,0,0}},
    };
    int terrain = block == BLOCK_WATER ? 1 : (block == BLOCK_LAVA ? 2 : 0);
    if (blocked[clampi(type_id, 0, 7)][clampi(mob_class, 0, 2)][terrain]) {
        return false;
    }
    return !mob_at(state, level, row, col) || (row == old_row && col == old_col);
}

typedef struct { float physical, fire, ice; } Damage;

Damage mob_damage_vector(int type, int mob_class) {
    static const float damage[NUM_MOB_TYPES][4][3] = {
        {{0,0,0},{2,0,0},{0,0,0},{2,0,0}}, {{0,0,0},{4,0,0},{0,0,0},{4,0,0}},
        {{0,0,0},{3,0,0},{0,0,0},{0,3,0}}, {{0,0,0},{5,0,0},{0,0,0},{0,0,3}},
        {{0,0,0},{6,0,0},{0,0,0},{5,0,0}}, {{0,0,0},{6,1,1},{0,0,0},{4,3,3}},
        {{0,0,0},{3,5,0},{0,0,0},{3,5,0}}, {{0,0,0},{4,0,5},{0,0,0},{4,0,5}},
    };
    const float* d = damage[clampi(type, 0, 7)][clampi(mob_class, 0, 3)];
    return (Damage){d[0], d[1], d[2]};
}

float damage_to_mob(Damage damage, int type, int mob_class) {
    static const float defense[NUM_MOB_TYPES][4][3] = {
        {{0,0,0},{0,0,0},{0,0,0},{0,0,0}},
        {{0,0,0},{0,0,0},{0,0,0},{0,0,0}},
        {{0,0,0},{0,0,0},{0,0,0},{0,0,0}},
        {{0,0,0},{0,0,0},{0,0,0},{0,0,0}},
        {{0,0,0},{.5f,0,0},{.5f,0,0},{0,0,0}},
        {{0,0,0},{.2f,0,0},{0,0,0},{0,0,0}},
        {{0,0,0},{.9f,1,0},{.9f,1,0},{0,0,0}},
        {{0,0,0},{.9f,0,1},{.9f,0,1},{0,0,0}},
    };
    const float* d = defense[clampi(type, 0, 7)][clampi(mob_class, 0, 3)];
    return damage.physical * (1-d[0]) + damage.fire * (1-d[1]) + damage.ice * (1-d[2]);
}

float damage_to_player(const State* state, Damage damage) {
    float physical_defense = 0, fire_defense = 0, ice_defense = 0;
    for (int i = 0; i < 4; i++) {
        physical_defense += 0.1f * state->inventory.armour[i];
        fire_defense += 0.2f * (state->armour_enchantments[i] == 1);
        ice_defense += 0.2f * (state->armour_enchantments[i] == 2);
    }
    float coeff = fighting_boss(state) ? 1.5f : 1.0f;
    return coeff * (damage.physical * (1 - physical_defense)
        + damage.fire * (1 - fire_defense) + damage.ice * (1 - ice_defense));
}

bool damage_mob_at(State* state, int level, int row, int col, float damage,
        bool can_eat, bool can_get_achievement) {
    int mob_class;
    int slot;
    if (!find_mob_at(state, level, row, col, &mob_class, &slot)) {
        return false;
    }
    Mobs* mobs = mobs_for_class(state, level, mob_class);
    if (!mobs->mask[slot]) {
        return false;
    }

    mobs->health[slot] -= damage;
    if (mobs->health[slot] > 0.0f) {
        return true;
    }

    int type_id = mobs->type_id[slot];
    mobs->mask[slot] = false;
    set_mob_bit(state, level, row, col, false);
    state->monsters_killed[level] += mob_class == MOB_PASSIVE ? 0 : 1;
    if (can_get_achievement) {
        static const int achievements[3][8] = {
            {ACH_EAT_COW, ACH_EAT_BAT, ACH_EAT_SNAIL, 0, 0, 0, 0, 0},
            {ACH_DEFEAT_ZOMBIE, ACH_DEFEAT_GNOME_WARRIOR, ACH_DEFEAT_ORC_SOLIDER,
                ACH_DEFEAT_LIZARD, ACH_DEFEAT_KNIGHT, ACH_DEFEAT_TROLL,
                ACH_DEFEAT_PIGMAN, ACH_DEFEAT_FROST_TROLL},
            {ACH_DEFEAT_SKELETON, ACH_DEFEAT_GNOME_ARCHER, ACH_DEFEAT_ORC_MAGE,
                ACH_DEFEAT_KOBOLD, ACH_DEFEAT_ARCHER, ACH_DEFEAT_DEEP_THING,
                ACH_DEFEAT_FIRE_ELEMENTAL, ACH_DEFEAT_ICE_ELEMENTAL},
        };
        state->achievements[achievements[clampi(mob_class, 0, 2)][clampi(type_id, 0, 7)]] = 1;
    }

    if (mob_class == MOB_PASSIVE && can_eat) {
        state->player_food = clampi(state->player_food + 6, 0, max_need(state));
        state->player_hunger = 0.0f;
    }
    return true;
}

bool spawn_projectile(State* state, bool from_player, int projectile_type,
        int row, int col, int dir_row, int dir_col) {
    int level = state->player_level;
    Mobs* projectiles = from_player ? &state->player_projectiles[level] : &state->mob_projectiles[level];
    int (*directions)[MAX_PLAYER_PROJECTILES][2] =
        from_player ? state->player_projectile_directions : state->mob_projectile_dirs;
    for (int i = 0; i < MAX_PLAYER_PROJECTILES; i++) {
        if (projectiles->mask[i]) {
            continue;
        }
        projectiles->position[i][0] = row;
        projectiles->position[i][1] = col;
        Damage d = mob_damage_vector(projectile_type, MOB_PROJECTILE);
        projectiles->health[i] = d.physical + d.fire + d.ice;
        projectiles->attack_cooldown[i] = 0;
        projectiles->type_id[i] = projectile_type;
        projectiles->mask[i] = true;
        directions[level][i][0] = dir_row;
        directions[level][i][1] = dir_col;
        return true;
    }
    return false;
}

void update_projectile_set(State* state, bool from_player) {
    int level = state->player_level;
    for (int i = 0; i < MAX_PLAYER_PROJECTILES; i++) {
        if (from_player) {
            Mobs* projectiles = &state->player_projectiles[level];
            if (!projectiles->mask[i]) {
                continue;
            }
            int old_row = projectiles->position[i][0];
            int old_col = projectiles->position[i][1];
            int proposed_row = old_row + state->player_projectile_directions[level][i][0];
            int proposed_col = old_col + state->player_projectile_directions[level][i][1];
            int ptype = projectiles->type_id[i];
            Damage vector = mob_damage_vector(ptype, MOB_PROJECTILE);
            bool arrow = ptype == PROJECTILE_ARROW || ptype == PROJECTILE_ARROW2;
            if (arrow && state->bow_enchantment == 1) {
                vector.fire += vector.physical * 0.5f;
            }
            if (arrow && state->bow_enchantment == 2) {
                vector.ice += vector.physical * 0.5f;
            }
            float coeff = 1.0f;
            if (arrow) {
                coeff = 1.0f + 0.2f * (state->player_dexterity - 1);
            } else if (ptype == PROJECTILE_FIREBALL || ptype == PROJECTILE_ICEBALL) {
                coeff = 1.0f + 0.5f * (state->player_intelligence - 1);
            }
            vector.physical *= coeff;
            vector.fire *= coeff;
            vector.ice *= coeff;

            bool hit_old = false;
            int mob_class;
            int mob_slot;
            if (find_mob_at(state, level, old_row, old_col, &mob_class, &mob_slot)) {
                Mobs* target = mobs_for_class(state, level, mob_class);
                hit_old = damage_mob_at(
                    state, level, old_row, old_col,
                    damage_to_mob(vector, target->type_id[mob_slot], mob_class),
                    false, true
                );
            }

            Damage second = vector;
            if (hit_old) {
                second.physical = 0.0f;
                second.fire = 0.0f;
                second.ice = 0.0f;
            }
            bool hit_new = false;
            if (find_mob_at(state, level, proposed_row, proposed_col, &mob_class, &mob_slot)) {
                Mobs* target = mobs_for_class(state, level, mob_class);
                hit_new = damage_mob_at(
                    state, level, proposed_row, proposed_col,
                    damage_to_mob(second, target->type_id[mob_slot], mob_class),
                    false, true
                );
            }

            bool proposed_in_bounds = proposed_row >= 0 && proposed_row < MAP_SIZE
                && proposed_col >= 0 && proposed_col < MAP_SIZE;
            int proposed_block = proposed_in_bounds ? state->map[level][proposed_row][proposed_col] : 0;
            bool in_wall = is_solid_block(proposed_block) && proposed_block != BLOCK_WATER;
            bool keep = proposed_in_bounds && !in_wall && !hit_old && !hit_new;
            projectiles->position[i][0] = proposed_row;
            projectiles->position[i][1] = proposed_col;
            projectiles->mask[i] = keep;
        } else {
            Mobs* projectiles = &state->mob_projectiles[level];
            if (!projectiles->mask[i]) {
                continue;
            }
            int old_row = projectiles->position[i][0];
            int old_col = projectiles->position[i][1];
            int proposed_row = old_row + state->mob_projectile_dirs[level][i][0];
            int proposed_col = old_col + state->mob_projectile_dirs[level][i][1];
            bool proposed_in_player = proposed_row == state->player_position[0]
                && proposed_col == state->player_position[1];
            bool proposed_in_bounds = proposed_row >= 0 && proposed_row < MAP_SIZE
                && proposed_col >= 0 && proposed_col < MAP_SIZE;
            int proposed_block = proposed_in_bounds ? state->map[level][proposed_row][proposed_col] : 0;
            bool in_wall = is_solid_block(proposed_block) && proposed_block != BLOCK_WATER;
            bool in_mob = mob_at(state, level, proposed_row, proposed_col)
                || (state->player_position[0] == proposed_row
                    && state->player_position[1] == proposed_col);
            bool keep_moving = proposed_in_bounds && !in_wall && !in_mob;
            bool hit_player = (
                (old_row == state->player_position[0] && old_col == state->player_position[1])
                || proposed_in_player
            );
            keep_moving = keep_moving && !hit_player;
            bool hit_bench = proposed_block == BLOCK_FURNACE
                || proposed_block == BLOCK_CRAFTING_TABLE;
            int new_block = hit_bench ? BLOCK_PATH : proposed_block;

            projectiles->position[i][0] = proposed_row;
            projectiles->position[i][1] = proposed_col;
            projectiles->mask[i] = keep_moving;
            if (hit_player) {
                state->player_health -= damage_to_player(
                    state, mob_damage_vector(projectiles->type_id[i], MOB_PROJECTILE));
                state->is_sleeping = false;
                state->is_resting = false;
            }
            if ((unsigned)proposed_row < MAP_SIZE && (unsigned)proposed_col < MAP_SIZE) {
                set_block(state, level, proposed_row, proposed_col, new_block);
            }
        }
    }
}

int floor_mob_type(int level, int mob_class) {
    static const int types[NUM_LEVELS][3] = {
        {0, 0, 0}, {2, 2, 2}, {1, 1, 1}, {2, 3, 3}, {2, 4, 4},
        {1, 5, 5}, {1, 6, 6}, {1, 7, 7}, {0, 0, 0},
    };
    return types[clampi(level, 0, NUM_LEVELS - 1)][clampi(mob_class, 0, 2)];
}

int collect_spawn_cells(const State* state, int level, int min_exclusive,
        int max_exclusive, bool boss, bool water_only, int* rows, int* cols) {
    const uint64_t* terrain = boss
        ? state->spawn_grave[level]
        : (water_only ? state->spawn_water[level] : state->spawn_land[level]);

    int pr = state->player_position[0];
    int pc = state->player_position[1];
    int limit = MOB_DESPAWN_DISTANCE - 1;
    int r0 = pr - limit;
    int r1 = pr + limit;
    int c0 = pc - limit;
    int c1 = pc + limit;
    if (r0 < 0) {
        r0 = 0;
    }
    if (r1 > MAP_SIZE - 1) {
        r1 = MAP_SIZE - 1;
    }
    if (c0 < 0) {
        c0 = 0;
    }
    if (c1 > MAP_SIZE - 1) {
        c1 = MAP_SIZE - 1;
    }
    uint64_t col_mask = (~0ull << c0) & ((1ull << (c1 + 1)) - 1);
    int count = 0;
    for (int row = r0; row <= r1; row++) {
        int dr = row - pr;
        int dr2 = dr * dr;
        uint64_t bits = terrain[row] & ~state->mob_bits[level][row] & col_mask;
        while (bits) {
            int col = __builtin_ctzll(bits);
            bits &= bits - 1;
            int dc = col - pc;
            int distance2 = dr2 + dc * dc;
            if (distance2 > min_exclusive && distance2 < max_exclusive) {
                rows[count] = row;
                cols[count] = col;
                count++;
            }
        }
    }
    return count;
}

bool pick_spawn_cell(const int* rows, const int* cols, int count, Rng key,
        int* out_row, int* out_col) {
    if (count <= 0) {
        return false;
    }
    float draw = count * (1.0f - rng_f32(key, 0));
    int chosen = (int)ceilf(draw) - 1;
    if (chosen < 0) {
        chosen = 0;
    }
    if (chosen >= count) {
        chosen = count - 1;
    }
    *out_row = rows[chosen];
    *out_col = cols[chosen];
    return true;
}

void spawn_into_slot(State* state, int level, Mobs* mobs, int slot, int mob_class,
        int type_id, int row, int col) {
    static const float passive_health[NUM_MOB_TYPES] = {3, 4, 6, 8, 0, 0, 0, 0};
    static const float melee_health[NUM_MOB_TYPES] = {5, 7, 9, 11, 12, 20, 20, 24};
    static const float ranged_health[NUM_MOB_TYPES] = {3, 5, 6, 8, 12, 4, 14, 16};
    int idx = clampi(type_id, 0, NUM_MOB_TYPES - 1);
    float health = melee_health[idx];
    if (mob_class == MOB_PASSIVE) {
        health = passive_health[idx];
    } else if (mob_class == MOB_RANGED) {
        health = ranged_health[idx];
    }
    mobs->position[slot][0] = row;
    mobs->position[slot][1] = col;
    mobs->health[slot] = health;
    mobs->mask[slot] = true;
    set_mob_bit(state, level, row, col, true);
}

void count_and_empty(const Mobs* mobs, int slots, int* count, int* empty) {
    int n = 0;
    int first = 0;
    bool found = false;
    for (int i = 0; i < slots; i++) {
        n += mobs->mask[i] ? 1 : 0;
        if (!mobs->mask[i] && !found) {
            first = i;
            found = true;
        }
    }
    *count = n;
    *empty = first;
}

void choose_direction(Rng key, int count, int direction[2]) {
    int choice = randint(key, 0u, 0, count);
    direction[0] = 0;
    direction[1] = 0;
    if (choice == 0) {
        direction[1] = -1;
    } else if (choice == 1) {
        direction[1] = 1;
    } else if (choice == 2) {
        direction[0] = -1;
    } else if (choice == 3) {
        direction[0] = 1;
    }
}

int choose_player_axis(Rng key, int distance_row, int distance_col) {
    int total = distance_row + distance_col;
    if (total == 0) {
        return 1;
    }
    int maximum = distance_row > distance_col ? distance_row : distance_col;
    float weights[2] = {
        distance_row == maximum ? 1.0f / total : 0.0f,
        distance_col == maximum ? 1.0f / total : 0.0f,
    };
    float sum = weights[0] + weights[1];
    float draw = sum * (1.0f - rng_f32(key, 0));
    return (weights[0] >= draw || sum == 0.0f) ? 0 : 1;
}

void move_melee_slot(State* state, int level, int slot, Rng* rng) {
    Mobs* mobs = &state->melee_mobs[level];
    bool alive = mobs->mask[slot];
    int old_row = mobs->position[slot][0];
    int old_col = mobs->position[slot][1];
    int type_id = mobs->type_id[slot];
    int cooldown = mobs->attack_cooldown[slot];

    int random_dir[2];
    choose_direction(rng_key(rng), 4, random_dir);
    int distance_row = abs(state->player_position[0] - old_row);
    int distance_col = abs(state->player_position[1] - old_col);
    int axis = choose_player_axis(rng_key(rng), distance_row, distance_col);
    int player_dir[2] = {0, 0};
    if (axis == 0) {
        int dr = state->player_position[0] - old_row;
        player_dir[0] = (dr > 0) - (dr < 0);
    } else {
        int dc = state->player_position[1] - old_col;
        player_dir[1] = (dc > 0) - (dc < 0);
    }
    int dist = distance_row + distance_col;
    float chase_roll = rng_f32(rng_key(rng), 0);
    bool chase = (dist < 10 || fighting_boss(state)) && chase_roll < 0.75f;
    int proposed_row = chase ? old_row + player_dir[0] : old_row + random_dir[0];
    int proposed_col = chase ? old_col + player_dir[1] : old_col + random_dir[1];
    bool attacking = dist == 1 && cooldown <= 0 && alive;
    if (attacking) {
        proposed_row = old_row;
        proposed_col = old_col;
        Damage damage = mob_damage_vector(type_id, MOB_MELEE);
        float sleep = 1.0f + 2.5f * state->is_sleeping;
        damage.physical *= sleep;
        damage.fire *= sleep;
        damage.ice *= sleep;
        state->player_health -= damage_to_player(state, damage);
        state->achievements[ACH_WAKE_UP] = state->achievements[ACH_WAKE_UP] || state->is_sleeping;
        state->is_sleeping = false;
        state->is_resting = false;
    }
    int new_cooldown = attacking ? 5 : cooldown - 1;
    bool valid = valid_typed_mob_position(state, level, MOB_MELEE, type_id,
        proposed_row, proposed_col, old_row, old_col);
    int new_row = valid ? proposed_row : old_row;
    int new_col = valid ? proposed_col : old_col;
    bool keep = alive && (dist < MOB_DESPAWN_DISTANCE || fighting_boss(state));
    Rng unused;
    rng_split(*rng, &unused, rng);

    move_mob_occupancy(state, level, old_row, old_col, new_row, new_col, keep);
    mobs->position[slot][0] = new_row;
    mobs->position[slot][1] = new_col;
    mobs->attack_cooldown[slot] = new_cooldown;
    mobs->mask[slot] = keep;
}

void move_passive_slot(State* state, int level, int slot, Rng* rng) {
    Mobs* mobs = &state->passive_mobs[level];
    bool alive = mobs->mask[slot];
    int old_row = mobs->position[slot][0];
    int old_col = mobs->position[slot][1];
    int type_id = mobs->type_id[slot];
    int direction[2];
    choose_direction(rng_key(rng), 8, direction);
    int proposed_row = old_row + direction[0];
    int proposed_col = old_col + direction[1];
    bool valid = valid_typed_mob_position(state, level, MOB_PASSIVE, type_id,
        proposed_row, proposed_col, old_row, old_col);
    int new_row = valid ? proposed_row : old_row;
    int new_col = valid ? proposed_col : old_col;
    int dist = abs(state->player_position[0] - old_row) + abs(state->player_position[1] - old_col);
    bool keep = alive && dist < MOB_DESPAWN_DISTANCE;
    move_mob_occupancy(state, level, old_row, old_col, new_row, new_col, keep);
    mobs->position[slot][0] = new_row;
    mobs->position[slot][1] = new_col;
    mobs->mask[slot] = keep;
}

void move_ranged_slot(State* state, int level, int slot, Rng* rng) {
    Mobs* mobs = &state->ranged_mobs[level];
    bool alive = mobs->mask[slot];
    int old_row = mobs->position[slot][0];
    int old_col = mobs->position[slot][1];
    int type_id = mobs->type_id[slot];
    int cooldown = mobs->attack_cooldown[slot];

    int random_dir[2];
    choose_direction(rng_key(rng), 4, random_dir);
    int distance_row = abs(state->player_position[0] - old_row);
    int distance_col = abs(state->player_position[1] - old_col);
    int axis = choose_player_axis(rng_key(rng), distance_row, distance_col);
    int player_dir[2] = {0, 0};
    if (axis == 0) {
        int dr = state->player_position[0] - old_row;
        player_dir[0] = (dr > 0) - (dr < 0);
    } else {
        int dc = state->player_position[1] - old_col;
        player_dir[1] = (dc > 0) - (dc < 0);
    }
    int dist = distance_row + distance_col;
    int proposed_row = dist >= 6 ? old_row + player_dir[0] : old_row + random_dir[0];
    int proposed_col = dist >= 6 ? old_col + player_dir[1] : old_col + random_dir[1];
    if (dist <= 3) {
        proposed_row = old_row - player_dir[0];
        proposed_col = old_col - player_dir[1];
    }
    if (rng_f32(rng_key(rng), 0) <= 0.85f) {
        proposed_row = old_row + random_dir[0];
        proposed_col = old_col + random_dir[1];
    }
    bool valid = valid_typed_mob_position(state, level, MOB_RANGED, type_id,
        proposed_row, proposed_col, old_row, old_col);
    bool attacking = ((dist >= 4 && dist <= 5) || (dist <= 3 && !valid)) && cooldown <= 0 && alive;
    if (attacking) {
        static const int projectile[8] = {
            PROJECTILE_ARROW, PROJECTILE_ARROW, PROJECTILE_FIREBALL, PROJECTILE_DAGGER,
            PROJECTILE_ARROW2, PROJECTILE_SLIMEBALL, PROJECTILE_FIREBALL2, PROJECTILE_ICEBALL2
        };
        spawn_projectile(state, false, projectile[clampi(type_id, 0, 7)],
            old_row, old_col, player_dir[0], player_dir[1]);
        proposed_row = old_row;
        proposed_col = old_col;
    }
    int new_cooldown = attacking ? 4 : cooldown - 1;
    valid = valid_typed_mob_position(state, level, MOB_RANGED, type_id,
        proposed_row, proposed_col, old_row, old_col);
    int new_row = valid ? proposed_row : old_row;
    int new_col = valid ? proposed_col : old_col;
    bool keep = alive && (dist < MOB_DESPAWN_DISTANCE || fighting_boss(state));
    move_mob_occupancy(state, level, old_row, old_col, new_row, new_col, keep);
    mobs->position[slot][0] = new_row;
    mobs->position[slot][1] = new_col;
    mobs->attack_cooldown[slot] = new_cooldown;
    mobs->mask[slot] = keep;
}

int choose_weighted_key(Rng key, const float* weights, int count) {
    float total = 0.0f;
    for (int i = 0; i < count; i++) {
        total += weights[i];
    }
    float draw = total * (1.0f - rng_f32(key, 0));
    float cumulative = 0.0f;
    for (int i = 0; i < count; i++) {
        cumulative += weights[i];
        if (cumulative >= draw) {
            return i;
        }
    }
    return count - 1;
}

void compute_action_mask(Craftax* env) {
    unsigned char* m = env->agents[0].action_mask;
    if (m == NULL) {
        return;
    }
    if (!env->use_action_mask) {
        memset(m, 1, ATN_DIM);
        return;
    }
    const State* s = &env->state;
    const Inventory* inv = &s->inventory;
    memset(m, 0, ATN_DIM);
    m[ACTION_NOOP] = 1;
    if (s->is_sleeping || s->is_resting) {
        return;
    }
    m[ACTION_LEFT] = m[ACTION_RIGHT] = m[ACTION_UP] = m[ACTION_DOWN] = m[ACTION_DO] = 1;
    m[ACTION_SLEEP] = s->player_energy < max_need(s);
    m[ACTION_REST] = s->player_health < max_health(s);
    m[ACTION_PLACE_STONE] = m[ACTION_PLACE_FURNACE] = inv->stone > 0;
    m[ACTION_PLACE_TABLE] = inv->wood >= 2;
    m[ACTION_PLACE_PLANT] = inv->sapling > 0;
    m[ACTION_PLACE_TORCH] = inv->torches > 0;
    m[ACTION_MAKE_WOOD_PICKAXE] = inv->wood > 0 && inv->pickaxe < 1;
    m[ACTION_MAKE_STONE_PICKAXE] = inv->wood > 0 && inv->stone > 0 && inv->pickaxe < 2;
    m[ACTION_MAKE_IRON_PICKAXE] = inv->wood > 0 && inv->stone > 0 && inv->iron > 0
        && inv->coal > 0 && inv->pickaxe < 3;
    m[ACTION_MAKE_DIAMOND_PICKAXE] = inv->wood > 0 && inv->diamond >= 3 && inv->pickaxe < 4;
    m[ACTION_MAKE_WOOD_SWORD] = inv->wood > 0 && inv->sword < 1;
    m[ACTION_MAKE_STONE_SWORD] = inv->wood > 0 && inv->stone > 0 && inv->sword < 2;
    m[ACTION_MAKE_IRON_SWORD] = inv->wood > 0 && inv->stone > 0 && inv->iron > 0
        && inv->coal > 0 && inv->sword < 3;
    m[ACTION_MAKE_DIAMOND_SWORD] = inv->wood > 0 && inv->diamond >= 2 && inv->sword < 4;
    m[ACTION_MAKE_ARROW] = inv->wood > 0 && inv->stone > 0 && inv->arrows < 99;
    m[ACTION_MAKE_TORCH] = inv->wood > 0 && inv->coal > 0 && inv->torches < 99;
    int missing_iron = 0;
    int missing_diamond = 0;
    int armour = 0;
    for (int k = 0; k < 4; k++) {
        missing_iron += inv->armour[k] < 1;
        missing_diamond += inv->armour[k] < 2;
        armour += inv->armour[k];
    }
    m[ACTION_MAKE_IRON_ARMOUR] = missing_iron && inv->iron >= 3 && inv->coal >= 3;
    m[ACTION_MAKE_DIAMOND_ARMOUR] = missing_diamond && inv->diamond >= 3;
    int item = s->item_map[s->player_level][s->player_position[0]][s->player_position[1]];
    m[ACTION_DESCEND] = item == ITEM_LADDER_DOWN
        && s->monsters_killed[s->player_level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL
        && s->player_level < NUM_LEVELS - 1;
    m[ACTION_ASCEND] = item == ITEM_LADDER_UP && s->player_level > 0;
    m[ACTION_SHOOT_ARROW] = inv->bow > 0 && inv->arrows > 0;
    m[ACTION_CAST_FIREBALL] = s->learned_spells[0] && s->player_mana >= 2;
    m[ACTION_CAST_ICEBALL] = s->learned_spells[1] && s->player_mana >= 2;
    for (int k = 0; k < NUM_POTIONS; k++) {
        m[ACTION_DRINK_POTION_RED + k] = inv->potions[k] > 0;
    }
    m[ACTION_READ_BOOK] = inv->books > 0;
    int enchant = s->player_mana >= 9 && (inv->ruby > 0 || inv->sapphire > 0);
    m[ACTION_ENCHANT_SWORD] = enchant && inv->sword > 0;
    m[ACTION_ENCHANT_ARMOUR] = enchant && armour > 0;
    m[ACTION_ENCHANT_BOW] = enchant && inv->bow > 0;
    m[ACTION_LEVEL_UP_DEXTERITY] = s->player_xp > 0 && s->player_dexterity < MAX_ATTRIBUTE;
    m[ACTION_LEVEL_UP_STRENGTH] = s->player_xp > 0 && s->player_strength < MAX_ATTRIBUTE;
    m[ACTION_LEVEL_UP_INTELLIGENCE] = s->player_xp > 0 && s->player_intelligence < MAX_ATTRIBUTE;
}

void compute_observations(Craftax* env) {
    State* state = &env->state;
    float* obs = env->agents[0].observations;
    const int map_obs = OBS_ROWS * OBS_COLS * OBS_TILE_CHANNELS;
    memset(obs, 0, map_obs * sizeof(float));

    int level = state->player_level;
    int row = state->player_position[0];
    int col = state->player_position[1];
    int row_radius = OBS_ROWS / 2;
    int col_radius = OBS_COLS / 2;
    int r0 = clampi(-row, -row_radius, row_radius);
    int r1 = clampi(MAP_SIZE - 1 - row, -row_radius, row_radius);
    int c0 = clampi(-col, -col_radius, col_radius);
    int c1 = clampi(MAP_SIZE - 1 - col, -col_radius, col_radius);

    // Add map information
    for (int r = r0; r <= r1; r++) {
        int obs_row = row + r;
        uint8_t* map_row = state->map[level][obs_row];
        uint8_t* item_row = state->item_map[level][obs_row];
        uint8_t* light_row = state->light_map[level][obs_row];
        float* tile = obs + ((r + row_radius) * OBS_COLS + (c0 + col_radius)) * OBS_TILE_CHANNELS;
        for (int c = c0; c <= c1; c++) {
            int obs_col = col + c;
            if (light_row[obs_col] > VISIBLE_LIGHT_THRESHOLD) {
                tile[0] = map_row[obs_col];
                tile[1] = item_row[obs_col] + 1;
                tile[2] = 1.0f;
            }
            tile += OBS_TILE_CHANNELS;
        }
    }
    // Add mob information
    write_mob_obs(obs, state, &state->melee_mobs[level], MAX_MELEE_MOBS, 0);
    write_mob_obs(obs, state, &state->passive_mobs[level], MAX_PASSIVE_MOBS, 1);
    write_mob_obs(obs, state, &state->ranged_mobs[level], MAX_RANGED_MOBS, 2);
    write_mob_obs(obs, state, &state->mob_projectiles[level], MAX_MOB_PROJECTILES, 3);
    write_mob_obs(obs, state, &state->player_projectiles[level], MAX_PLAYER_PROJECTILES, 4);

    int obs_idx = map_obs;

    // Inventory and player stats
    obs[obs_idx++] = sqrtf(state->inventory.wood) / 10.0f;
    obs[obs_idx++] = sqrtf(state->inventory.stone) / 10.0f;
    obs[obs_idx++] = sqrtf(state->inventory.coal) / 10.0f;
    obs[obs_idx++] = sqrtf(state->inventory.iron) / 10.0f;
    obs[obs_idx++] = sqrtf(state->inventory.diamond) / 10.0f;
    obs[obs_idx++] = sqrtf(state->inventory.sapphire) / 10.0f;
    obs[obs_idx++] = sqrtf(state->inventory.ruby) / 10.0f;
    obs[obs_idx++] = sqrtf(state->inventory.sapling) / 10.0f;
    obs[obs_idx++] = sqrtf(state->inventory.torches) / 10.0f;
    obs[obs_idx++] = sqrtf(state->inventory.arrows) / 10.0f;
    obs[obs_idx++] = state->inventory.books / 2.0f;
    obs[obs_idx++] = state->inventory.pickaxe / 4.0f;
    obs[obs_idx++] = state->inventory.sword / 4.0f;
    obs[obs_idx++] = state->sword_enchantment;
    obs[obs_idx++] = state->bow_enchantment;
    obs[obs_idx++] = state->inventory.bow;
    for (int i = 0; i < NUM_POTIONS; i++) {
        obs[obs_idx++] = sqrtf(state->inventory.potions[i]) / 10.0f;
    }

    obs[obs_idx++] = state->player_health / 10.0f;
    obs[obs_idx++] = state->player_food / 10.0f;
    obs[obs_idx++] = state->player_drink / 10.0f;
    obs[obs_idx++] = state->player_energy / 10.0f;
    obs[obs_idx++] = state->player_mana / 10.0f;
    obs[obs_idx++] = state->player_xp / 10.0f;
    obs[obs_idx++] = state->player_dexterity / 10.0f;
    obs[obs_idx++] = state->player_strength / 10.0f;
    obs[obs_idx++] = state->player_intelligence / 10.0f;

    int direction_index = state->player_direction - ACTION_LEFT;
    for (int i = 0; i < 4; i++) {
        obs[obs_idx++] = i == direction_index ? 1.0f : 0.0f;
    }
    for (int i = 0; i < 4; i++) {
        obs[obs_idx++] = state->inventory.armour[i] / 2.0f;
    }
    for (int i = 0; i < 4; i++) {
        obs[obs_idx++] = state->armour_enchantments[i];
    }

    obs[obs_idx++] = state->light_level;
    obs[obs_idx++] = state->is_sleeping ? 1.0f : 0.0f;
    obs[obs_idx++] = state->is_resting ? 1.0f : 0.0f;
    obs[obs_idx++] = state->learned_spells[0] ? 1.0f : 0.0f;
    obs[obs_idx++] = state->learned_spells[1] ? 1.0f : 0.0f;
    obs[obs_idx++] = state->player_level / 10.0f;
    obs[obs_idx++] = state->monsters_killed[level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL ? 1.0f : 0.0f;
    obs[obs_idx++] = boss_vulnerable(state) ? 1.0f : 0.0f;

    compute_action_mask(env);
}

static int key_to_action(void) {
    static const int map[][2] = {
        {KEY_Q, ACTION_NOOP},
        {KEY_W, ACTION_UP},
        {KEY_UP, ACTION_UP},
        {KEY_D, ACTION_RIGHT},
        {KEY_RIGHT, ACTION_RIGHT},
        {KEY_S, ACTION_DOWN},
        {KEY_DOWN, ACTION_DOWN},
        {KEY_A, ACTION_LEFT},
        {KEY_LEFT, ACTION_LEFT},
        {KEY_SPACE, ACTION_DO},
        {KEY_ONE, ACTION_MAKE_WOOD_PICKAXE},
        {KEY_TWO, ACTION_MAKE_STONE_PICKAXE},
        {KEY_THREE, ACTION_MAKE_IRON_PICKAXE},
        {KEY_FOUR, ACTION_MAKE_DIAMOND_PICKAXE},
        {KEY_FIVE, ACTION_MAKE_WOOD_SWORD},
        {KEY_SIX, ACTION_MAKE_STONE_SWORD},
        {KEY_SEVEN, ACTION_MAKE_IRON_SWORD},
        {KEY_EIGHT, ACTION_MAKE_DIAMOND_SWORD},
        {KEY_T, ACTION_PLACE_TABLE},
        {KEY_TAB, ACTION_SLEEP},
        {KEY_R, ACTION_PLACE_STONE},
        {KEY_F, ACTION_PLACE_FURNACE},
        {KEY_P, ACTION_PLACE_PLANT},
        {KEY_E, ACTION_REST},
        {KEY_COMMA, ACTION_ASCEND},
        {KEY_PERIOD, ACTION_DESCEND},
        {KEY_Y, ACTION_MAKE_IRON_ARMOUR},
        {KEY_U, ACTION_MAKE_DIAMOND_ARMOUR},
        {KEY_I, ACTION_SHOOT_ARROW},
        {KEY_O, ACTION_MAKE_ARROW},
        {KEY_G, ACTION_CAST_FIREBALL},
        {KEY_H, ACTION_CAST_ICEBALL},
        {KEY_J, ACTION_PLACE_TORCH},
        {KEY_Z, ACTION_DRINK_POTION_RED},
        {KEY_X, ACTION_DRINK_POTION_GREEN},
        {KEY_C, ACTION_DRINK_POTION_BLUE},
        {KEY_V, ACTION_DRINK_POTION_PINK},
        {KEY_B, ACTION_DRINK_POTION_CYAN},
        {KEY_N, ACTION_DRINK_POTION_YELLOW},
        {KEY_M, ACTION_READ_BOOK},
        {KEY_K, ACTION_ENCHANT_SWORD},
        {KEY_L, ACTION_ENCHANT_ARMOUR},
        {KEY_LEFT_BRACKET, ACTION_MAKE_TORCH},
        {KEY_RIGHT_BRACKET, ACTION_LEVEL_UP_DEXTERITY},
        {KEY_MINUS, ACTION_LEVEL_UP_STRENGTH},
        {KEY_EQUAL, ACTION_LEVEL_UP_INTELLIGENCE},
        {KEY_SEMICOLON, ACTION_ENCHANT_BOW},
    };
    for (int i = 0; i < (int)(sizeof(map) / sizeof(map[0])); i++) {
        if (IsKeyPressed(map[i][0])) {
            return map[i][1];
        }
    }
    return -1;
}

// Shift + action-panel key. 1 = applied, 0 = policy, -1 = skip tick.
static int got_human_input(Craftax* env) {
    int shift = IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);
    if (!IsWindowReady() || !shift
        || env->state.is_sleeping || env->state.is_resting) {
        return 0;
    }
    int action = key_to_action();
    if (action < 0) {
        return -1;
    }
    env->agents[0].actions[0] = (float)action;
    return 1;
}

void puf_reset(Craftax* env) {
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
    env->episode_return_accum = 0.0f;
    env->episode_length_accum = 0;
    env->max_floor_accum = 0;
    memset(env->achievements, 0, sizeof(env->achievements));

    Rng initial = rng_seed((uint32_t)env->seed);
    if (env->num_levels > 0) {
        Rng discard;
        rng_split(initial, &env->env_rng, &discard);
        int idx = env->seed % env->num_levels;
        env->state = env->levels[idx];
    } else {
        Rng reset_key;
        rng_split(initial, &env->env_rng, &reset_key);
        Rng unused;
        Rng world_key;
        rng_split(reset_key, &unused, &world_key);
        generate_world_from_key(&env->state, world_key);
    }
    compute_observations(env);
    if (env->state.player_level > env->max_floor_accum) {
        env->max_floor_accum = env->state.player_level;
    }
}

void puf_step(Craftax* env) {
    if (got_human_input(env) < 0) {
        return;
    }
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
    int action = env->agents[0].actions[0];

    State* state = &env->state;
    int initial_achievements[NUM_ACHIEVEMENTS];
    memcpy(initial_achievements, state->achievements, sizeof(initial_achievements));

    int initial_armour = equipped_armour(state);
    // Sleep/rest collapsed into one puf_step so credit assignment sees wake/hit/death.
    Rng reset_key = 0;
    bool done = false;
    do {
        Rng step_key;
        rng_split(env->env_rng, &env->env_rng, &step_key);
        Rng step_rng;
        rng_split(step_key, &step_rng, &reset_key);

        if (state->is_sleeping || state->is_resting) {
            action = ACTION_NOOP;
        }

    int level = state->player_level;
    int row = state->player_position[0];
    int col = state->player_position[1];
    bool on_down_ladder = state->item_map[level][row][col] == ITEM_LADDER_DOWN;
    bool on_up_ladder = state->item_map[level][row][col] == ITEM_LADDER_UP;
    bool can_move_down = action == ACTION_DESCEND && on_down_ladder
        && state->monsters_killed[level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL
        && level < NUM_LEVELS - 1;
    bool can_move_up = action == ACTION_ASCEND && on_up_ladder && level > 0;

    if (can_move_down || can_move_up) {
        int new_level = level + (can_move_down ? 1 : -1);
        if (can_move_down) {
            state->player_position[0] = state->up_ladders[new_level][0];
            state->player_position[1] = state->up_ladders[new_level][1];
        } else {
            state->player_position[0] = state->down_ladders[new_level][0];
            state->player_position[1] = state->down_ladders[new_level][1];
        }
        state->player_level = new_level;
        static const int floor_ach[NUM_LEVELS] = {
            -1, ACH_ENTER_DUNGEON, ACH_ENTER_GNOMISH_MINES, ACH_ENTER_SEWERS,
            ACH_ENTER_VAULT, ACH_ENTER_TROLL_MINES, ACH_ENTER_FIRE_REALM,
            ACH_ENTER_ICE_REALM, ACH_ENTER_GRAVEYARD,
        };
        int achievement = floor_ach[new_level];
        if (achievement >= 0 && !state->achievements[achievement]) {
            state->achievements[achievement] = 1;
            state->player_xp += 1;
        }
    }

    // Crafting actions require a crafting table and/or furnace nearby.
    static const int nearby_tiles[8][2] = {
        {0, -1}, {0, 1}, {-1, 0}, {1, 0},
        {-1, -1}, {-1, 1}, {1, -1}, {1, 1},
    };
    level = state->player_level;
    bool at_table = false;
    bool at_furnace = false;
    for (int i = 0; i < 8; i++) {
        int row = state->player_position[0] + nearby_tiles[i][0];
        int col = state->player_position[1] + nearby_tiles[i][1];
        if (row < 0 || row >= MAP_SIZE || col < 0 || col >= MAP_SIZE) {
            continue;
        }
        at_table = at_table || state->map[level][row][col] == BLOCK_CRAFTING_TABLE;
        at_furnace = at_furnace || state->map[level][row][col] == BLOCK_FURNACE;
    }

    Inventory* inv = &state->inventory;

    if (action == ACTION_MAKE_WOOD_PICKAXE && at_table && inv->wood >= 1
            && inv->pickaxe < 1) {
        inv->wood -= 1;
        inv->pickaxe = 1;
    } else if (action == ACTION_MAKE_STONE_PICKAXE && at_table && inv->wood >= 1
            && inv->stone >= 1 && inv->pickaxe < 2) {
        inv->wood -= 1;
        inv->stone -= 1;
        inv->pickaxe = 2;
    } else if (action == ACTION_MAKE_IRON_PICKAXE && at_table && at_furnace
            && inv->wood >= 1 && inv->stone >= 1 && inv->iron >= 1
            && inv->coal >= 1 && inv->pickaxe < 3) {
        inv->wood -= 1;
        inv->stone -= 1;
        inv->iron -= 1;
        inv->coal -= 1;
        inv->pickaxe = 3;
    } else if (action == ACTION_MAKE_DIAMOND_PICKAXE && at_table && inv->wood >= 1
            && inv->diamond >= 3 && inv->pickaxe < 4) {
        inv->wood -= 1;
        inv->diamond -= 3;
        inv->pickaxe = 4;
    } else if (action == ACTION_MAKE_WOOD_SWORD && at_table && inv->wood >= 1
            && inv->sword < 1) {
        inv->wood -= 1;
        inv->sword = 1;
    } else if (action == ACTION_MAKE_STONE_SWORD && at_table && inv->wood >= 1
            && inv->stone >= 1 && inv->sword < 2) {
        inv->wood -= 1;
        inv->stone -= 1;
        inv->sword = 2;
    } else if (action == ACTION_MAKE_IRON_SWORD && at_table && at_furnace
            && inv->wood >= 1 && inv->stone >= 1 && inv->iron >= 1
            && inv->coal >= 1 && inv->sword < 3) {
        inv->wood -= 1;
        inv->stone -= 1;
        inv->iron -= 1;
        inv->coal -= 1;
        inv->sword = 3;
    } else if (action == ACTION_MAKE_DIAMOND_SWORD && at_table && inv->wood >= 1
            && inv->diamond >= 2 && inv->sword < 4) {
        inv->wood -= 1;
        inv->diamond -= 2;
        inv->sword = 4;
    } else if (action == ACTION_MAKE_ARROW && at_table && inv->wood >= 1
            && inv->stone >= 1 && inv->arrows < 99) {
        inv->wood -= 1;
        inv->stone -= 1;
        inv->arrows += 2;
    } else if (action == ACTION_MAKE_TORCH && at_table && inv->wood >= 1
            && inv->coal >= 1 && inv->torches < 99) {
        inv->wood -= 1;
        inv->coal -= 1;
        inv->torches += 4;
    } else if (action == ACTION_MAKE_IRON_ARMOUR && at_table && at_furnace
            && inv->iron >= 3 && inv->coal >= 3) {
        for (int i = 0; i < 4; i++) {
            if (inv->armour[i] < 1) {
                inv->iron -= 3;
                inv->coal -= 3;
                inv->armour[i] = 1;
                state->achievements[ACH_MAKE_IRON_ARMOUR] = 1;
                break;
            }
        }
    } else if (action == ACTION_MAKE_DIAMOND_ARMOUR && at_table
            && inv->diamond >= 3) {
        for (int i = 0; i < 4; i++) {
            if (inv->armour[i] < 2) {
                inv->diamond -= 3;
                inv->armour[i] = 2;
                state->achievements[ACH_MAKE_DIAMOND_ARMOUR] = 1;
                break;
            }
        }
    }

    Rng interact_rng = rng_key(&step_rng);
    int direction[2];
    action_to_direction(state->player_direction, direction);
    row = state->player_position[0] + direction[0];
    col = state->player_position[1] + direction[1];
    bool in_bounds = (unsigned)row < MAP_SIZE && (unsigned)col < MAP_SIZE;
    level = state->player_level;
    inv = &state->inventory;

    if (action == ACTION_DO) {
        bool did_attack = false;
        int attack_class;
        int attack_slot;
        if (find_mob_at(state, level, row, col, &attack_class, &attack_slot)) {
            // Compute damage based on sword level and player stats.
            Mobs* attack_mobs = mobs_for_class(state, level, attack_class);
            static const float base_damage[5] = {1, 2, 3, 5, 8};
            float base = base_damage[clampi(inv->sword, 0, 4)];
            float physical = base * (1.0f + 0.25f * (state->player_strength - 1));
            float magic = base * 0.5f * (1.0f + 0.05f * (state->player_intelligence - 1));
            Damage vector = {
                physical,
                state->sword_enchantment == 1 ? magic : 0,
                state->sword_enchantment == 2 ? magic : 0,
            };
            did_attack = damage_mob_at(
                state, level, row, col,
                damage_to_mob(vector, attack_mobs->type_id[attack_slot], attack_class), true, true);
        }
        Rng sapling_key = rng_key(&interact_rng);
        Rng chest_key = rng_key(&interact_rng);
        if (!did_attack && in_bounds) {
            int block = state->map[level][row][col];

            if (block == BLOCK_TREE || block == BLOCK_FIRE_TREE || block == BLOCK_ICE_SHRUB) {
                int ground = BLOCK_GRASS;
                if (block == BLOCK_FIRE_TREE) {
                    ground = BLOCK_FIRE_GRASS;
                } else if (block == BLOCK_ICE_SHRUB) {
                    ground = BLOCK_ICE_GRASS;
                }
                set_block(state, level, row, col, ground);
                inv->wood += 1;
            } else if (block == BLOCK_STONE && inv->pickaxe >= 1) {
                set_block(state, level, row, col, BLOCK_PATH);
                inv->stone += 1;
            } else if (block == BLOCK_COAL && inv->pickaxe >= 1) {
                set_block(state, level, row, col, BLOCK_PATH);
                inv->coal += 1;
            } else if (block == BLOCK_IRON && inv->pickaxe >= 2) {
                set_block(state, level, row, col, BLOCK_PATH);
                inv->iron += 1;
            } else if (block == BLOCK_DIAMOND && inv->pickaxe >= 3) {
                set_block(state, level, row, col, BLOCK_PATH);
                inv->diamond += 1;
            } else if (block == BLOCK_SAPPHIRE && inv->pickaxe >= 4) {
                set_block(state, level, row, col, BLOCK_PATH);
                inv->sapphire += 1;
            } else if (block == BLOCK_RUBY && inv->pickaxe >= 4) {
                set_block(state, level, row, col, BLOCK_PATH);
                inv->ruby += 1;
            } else if (block == BLOCK_STALAGMITE && inv->pickaxe >= 1) {
                set_block(state, level, row, col, BLOCK_PATH);
                inv->stone += 1;
            } else if (block == BLOCK_CRAFTING_TABLE || block == BLOCK_FURNACE) {
                set_block(state, level, row, col, BLOCK_PATH);
            } else if (block == BLOCK_WATER || block == BLOCK_FOUNTAIN) {
                state->player_drink = clampi(state->player_drink + 1, 0, max_need(state));
                state->player_thirst = 0.0f;
                state->achievements[ACH_COLLECT_DRINK] = 1;
            } else if (block == BLOCK_RIPE_PLANT) {
                set_block(state, level, row, col, BLOCK_PLANT);
                for (int i = 0; i < MAX_GROWING_PLANTS; i++) {
                    if (state->growing_plants_pos[i][0] == row && state->growing_plants_pos[i][1] == col) {
                        state->growing_plants_age[i] = 0;
                        break;
                    }
                }
                state->player_food = clampi(state->player_food + 4, 0, max_need(state));
                state->player_hunger = 0.0f;
                state->achievements[ACH_EAT_PLANT] = 1;
            } else if (block == BLOCK_CHEST) {
                set_block(state, level, row, col, BLOCK_PATH);
                Rng chest_rng = chest_key;
                rng_key(&chest_rng);
                randint(rng_key(&chest_rng), 0u, 1, 6);
                bool torch = rng_f32(rng_key(&chest_rng), 0) < 0.6f;
                int torches = randint(rng_key(&chest_rng), 0u, 4, 8);
                bool ore = rng_f32(rng_key(&chest_rng), 0) < 0.6f;
                float ore_weights[5] = {0.3f, 0.3f, 0.15f, 0.125f, 0.125f};
                int ore_id = choose_weighted_key(rng_key(&chest_rng), ore_weights, 5);
                Rng amount_key = rng_key(&chest_rng);
                int ore_amt[5] = {
                    randint(amount_key, 0u, 1, 4),
                    randint(amount_key, 0u, 1, 3),
                    randint(amount_key, 0u, 1, 2),
                    randint(amount_key, 0u, 1, 2),
                    randint(amount_key, 0u, 1, 2),
                };
                bool potion = rng_f32(rng_key(&chest_rng), 0) < 0.5f;
                int potion_id = randint(rng_key(&chest_rng), 0u, 0, 6);
                int potion_amount = randint(rng_key(&chest_rng), 0u, 1, 3);
                bool arrows = rng_f32(rng_key(&chest_rng), 0) < 0.25f;
                int arrow_amount = randint(rng_key(&chest_rng), 0u, 1, 5);
                bool tool = rng_f32(rng_key(&chest_rng), 0) < 0.2f;
                int tool_id = randint(rng_key(&chest_rng), 0u, 0, 2);
                float tool_weights[4] = {0.4f, 0.3f, 0.2f, 0.1f};
                int pickaxe = choose_weighted_key(rng_key(&chest_rng), tool_weights, 4) + 1;
                int sword = choose_weighted_key(rng_key(&chest_rng), tool_weights, 4) + 1;
                int* ore_inv[5] = {
                    &inv->coal, &inv->iron, &inv->diamond, &inv->sapphire, &inv->ruby,
                };
                inv->torches += torch * torches;
                if (ore) {
                    *ore_inv[ore_id] += ore_amt[ore_id];
                }
                inv->potions[potion_id] += potion * potion_amount;
                inv->arrows += arrows * arrow_amount;
                if (tool && tool_id == 0 && pickaxe > inv->pickaxe) {
                    inv->pickaxe = pickaxe;
                }
                if (tool && tool_id == 1 && sword > inv->sword) {
                    inv->sword = sword;
                }
                if (!state->chests_opened[level]) {
                    if (level == 1) {
                        inv->bow = 1;
                    }
                    if (level == 3 || level == 4) {
                        inv->books += 1;
                    }
                }
                state->achievements[ACH_OPEN_CHEST] = 1;
            } else if (block == BLOCK_NECROMANCER && boss_vulnerable(state) && fighting_boss(state)) {
                state->boss_progress += 1;
                state->boss_timestep_to_spawn_this_round = BOSS_SPAWN_TURNS;
                state->achievements[ACH_DAMAGE_NECROMANCER] = 1;
            }
            if (block == BLOCK_GRASS && rng_f32(sapling_key, 0) < 0.1f) {
                inv->sapling += 1;
            }
            state->chests_opened[level] |= block == BLOCK_CHEST;
        }
    }

    if (in_bounds) {
        int block = state->map[level][row][col];
        bool occupied = is_solid_block(block) || state->item_map[level][row][col] != ITEM_NONE
            || mob_at(state, level, row, col);

        if (action == ACTION_PLACE_TABLE && !occupied && inv->wood >= 2) {
            set_block(state, level, row, col, BLOCK_CRAFTING_TABLE);
            inv->wood -= 2;
            state->achievements[ACH_PLACE_TABLE] = 1;
        } else if (action == ACTION_PLACE_FURNACE && !occupied && inv->stone >= 1) {
            set_block(state, level, row, col, BLOCK_FURNACE);
            inv->stone -= 1;
            state->achievements[ACH_PLACE_FURNACE] = 1;
        } else if (action == ACTION_PLACE_STONE
                && (block == BLOCK_WATER || !occupied) && inv->stone >= 1) {
            set_block(state, level, row, col, BLOCK_STONE);
            inv->stone -= 1;
            state->achievements[ACH_PLACE_STONE] = 1;
        } else if (action == ACTION_PLACE_TORCH
                && (block == BLOCK_GRASS || block == BLOCK_SAND || block == BLOCK_PATH
                    || block == BLOCK_FIRE_GRASS || block == BLOCK_ICE_GRASS)
                && state->item_map[level][row][col] == ITEM_NONE
                && inv->torches >= 1) {
            state->item_map[level][row][col] = ITEM_TORCH;
            for (int dr = -4; dr <= 4; dr++) {
                int light_row = row + dr;
                if ((unsigned)light_row >= MAP_SIZE) {
                    continue;
                }
                for (int dc = -4; dc <= 4; dc++) {
                    int light_col = col + dc;
                    if ((unsigned)light_col >= MAP_SIZE) {
                        continue;
                    }
                    float torch = 1.0f - sqrtf(dr * dr + dc * dc) / 5.0f;
                    if (torch < 0.0f) {
                        torch = 0.0f;
                    }
                    float light = clampf(
                        state->light_map[level][light_row][light_col] / 255.0f + torch,
                        0.0f, 1.0f);
                    state->light_map[level][light_row][light_col] =
                        (unsigned char)(light * 255.0f);
                }
            }
            inv->torches -= 1;
            state->achievements[ACH_PLACE_TORCH] = 1;
        } else if (action == ACTION_PLACE_PLANT && block == BLOCK_GRASS
                && state->item_map[level][row][col] == ITEM_NONE
                && inv->sapling >= 1) {
            set_block(state, level, row, col, BLOCK_PLANT);
            inv->sapling -= 1;
            for (int i = 0; i < MAX_GROWING_PLANTS; i++) {
                if (!state->growing_plants_mask[i]) {
                    state->growing_plants_pos[i][0] = row;
                    state->growing_plants_pos[i][1] = col;
                    state->growing_plants_age[i] = 0;
                    state->growing_plants_mask[i] = 1;
                    break;
                }
            }
            state->achievements[ACH_PLACE_PLANT] = 1;
        }
    }

    int fire_row = direction[0];
    int fire_col = direction[1];
    if (fire_row == 0 && fire_col == 0) {
        fire_row = 1;
    }
    int prow = state->player_position[0];
    int pcol = state->player_position[1];
    if (action == ACTION_SHOOT_ARROW && inv->bow > 0 && inv->arrows > 0) {
        bool fired = spawn_projectile(state, true, PROJECTILE_ARROW2,
            prow, pcol, fire_row, fire_col);
        if (fired) {
            inv->arrows -= 1;
            state->achievements[ACH_FIRE_BOW] = 1;
        }
    } else if (action == ACTION_CAST_FIREBALL && state->learned_spells[0]
            && state->player_mana >= 2) {
        bool cast = spawn_projectile(state, true, PROJECTILE_FIREBALL,
            prow, pcol, fire_row, fire_col);
        if (cast) {
            state->player_mana -= 2;
            state->achievements[ACH_CAST_FIREBALL] = 1;
        }
    } else if (action == ACTION_CAST_ICEBALL && state->learned_spells[1]
            && state->player_mana >= 2) {
        bool cast = spawn_projectile(state, true, PROJECTILE_ICEBALL,
            prow, pcol, fire_row, fire_col);
        if (cast) {
            state->player_mana -= 2;
            state->achievements[ACH_CAST_ICEBALL] = 1;
        }
    }

    int potion = action - ACTION_DRINK_POTION_RED;
    if (potion >= 0 && potion < NUM_POTIONS && inv->potions[potion] > 0) {
        int effect = state->potion_mapping[potion];
        inv->potions[potion] -= 1;
        if (effect == 0) {
            state->player_health += 8.0f;
        } else if (effect == 1) {
            state->player_health -= 3.0f;
        } else if (effect == 2) {
            state->player_mana += 8;
        } else if (effect == 3) {
            state->player_mana -= 3;
        } else if (effect == 4) {
            state->player_energy += 8;
        } else {
            state->player_energy -= 3;
        }
        state->achievements[ACH_DRINK_POTION] = 1;
    }

    Rng book_rng = rng_key(&step_rng);

    bool reading = action == ACTION_READ_BOOK && inv->books > 0;
    Rng unused;
    Rng choice_key;
    rng_split(book_rng, &unused, &choice_key);
    float p0 = state->learned_spells[0] ? 0.0f : 1.0f;
    float p1 = state->learned_spells[1] ? 0.0f : 1.0f;
    int spell = 0;
    if (p0 + p1 != 0.0f) {
        float r = 1.0f - rng_f32(choice_key, 0);
        spell = r <= (p0 / (p0 + p1)) ? 0 : 1;
    }
    if (reading) {
        inv->books -= 1;
        state->learned_spells[spell] = 1;
        state->achievements[spell == 0 ? ACH_LEARN_FIREBALL : ACH_LEARN_ICEBALL] = 1;
    }
    Rng enchant_rng = rng_key(&step_rng);

    int eblock = 0;
    if (in_bounds) {
        eblock = state->map[level][row][col];
    }
    int enchant = eblock == BLOCK_ENCHANTMENT_TABLE_FIRE ? 1 :
        (eblock == BLOCK_ENCHANTMENT_TABLE_ICE ? 2 : 0);
    int gems = enchant == 1 ? inv->ruby : inv->sapphire;
    bool could = state->player_mana >= 9 && enchant != 0 && gems >= 1;
    bool enchanting_sword = could && action == ACTION_ENCHANT_SWORD && inv->sword > 0;
    bool enchanting_bow = could && action == ACTION_ENCHANT_BOW && inv->bow > 0;
    bool enchanting_armour = could && action == ACTION_ENCHANT_ARMOUR
        && equipped_armour(state) > 0;
    Rng armour_key = rng_key(&enchant_rng);
    int unenchanted = 0;
    for (int i = 0; i < 4; i++) {
        unenchanted += state->armour_enchantments[i] == 0;
    }
    float candidates[4];
    for (int i = 0; i < 4; i++) {
        bool opposite = state->armour_enchantments[i] != 0 && state->armour_enchantments[i] != enchant;
        candidates[i] = (state->armour_enchantments[i] == 0 || (unenchanted == 0 && opposite)) ? 1.0f : 0.0f;
    }
    int armour_target = choose_weighted_key(armour_key, candidates, 4);
    if (enchanting_sword) {
        state->sword_enchantment = enchant;
        state->achievements[ACH_ENCHANT_SWORD] = 1;
    }
    if (enchanting_bow) {
        state->bow_enchantment = enchant;
    }
    if (enchanting_armour) {
        state->armour_enchantments[armour_target] = enchant;
        state->achievements[ACH_ENCHANT_ARMOUR] = 1;
    }
    bool enchanting = enchanting_sword || enchanting_bow || enchanting_armour;
    if (enchanting) {
        if (enchant == 1) {
            inv->ruby -= 1;
        } else {
            inv->sapphire -= 1;
        }
        state->player_mana -= 9;
    }
    state->achievements[ACH_DEFEAT_NECROMANCER] |= state->boss_progress >= NUM_LEVELS - 1;
    if (fighting_boss(state)) {
        state->boss_timestep_to_spawn_this_round -= 1;
    }

    if (state->player_xp >= 1) {
        bool leveled = false;
        if (action == ACTION_LEVEL_UP_DEXTERITY && state->player_dexterity < MAX_ATTRIBUTE) {
            state->player_dexterity += 1;
            leveled = true;
        } else if (action == ACTION_LEVEL_UP_STRENGTH && state->player_strength < MAX_ATTRIBUTE) {
            state->player_strength += 1;
            leveled = true;
        } else if (action == ACTION_LEVEL_UP_INTELLIGENCE && state->player_intelligence < MAX_ATTRIBUTE) {
            state->player_intelligence += 1;
            leveled = true;
        }
        if (leveled) {
            state->player_xp -= 1;
        }
    }

    action_to_direction(action, direction);
    int proposed_row = state->player_position[0] + direction[0];
    int proposed_col = state->player_position[1] + direction[1];
    bool valid = (unsigned)proposed_row < MAP_SIZE
        && (unsigned)proposed_col < MAP_SIZE;
    if (valid) {
        int pblock = state->map[level][proposed_row][proposed_col];
        valid = !is_solid_block(pblock) && pblock != BLOCK_WATER && pblock != BLOCK_LAVA
            && !mob_at(state, level, proposed_row, proposed_col);
    }
    if (valid) {
        state->player_position[0] = proposed_row;
        state->player_position[1] = proposed_col;
    }
    if (direction[0] != 0 || direction[1] != 0) {
        state->player_direction = action;
    }

    Rng mobs_rng = rng_key(&step_rng);
    rng_key(&mobs_rng);
    move_melee_slot(state, level, 0, &mobs_rng);
    move_melee_slot(state, level, 1, &mobs_rng);
    move_melee_slot(state, level, 2, &mobs_rng);
    rng_key(&mobs_rng);
    move_passive_slot(state, level, 0, &mobs_rng);
    move_passive_slot(state, level, 1, &mobs_rng);
    move_passive_slot(state, level, 2, &mobs_rng);
    rng_key(&mobs_rng);
    move_ranged_slot(state, level, 0, &mobs_rng);
    move_ranged_slot(state, level, 1, &mobs_rng);
    rng_key(&mobs_rng);
    update_projectile_set(state, false);
    rng_key(&mobs_rng);
    update_projectile_set(state, true);

    Rng spawn_rng = rng_key(&step_rng);
    bool boss = fighting_boss(state);
    int coeff = 1 + (state->monsters_killed[level] < MONSTERS_KILLED_TO_CLEAR_LEVEL ? 2 : 0);
    if (boss) {
        coeff *= (state->boss_timestep_to_spawn_this_round >= 1) ? 1000 : 0;
    }

    static const float chances[NUM_LEVELS][4] = {
        {0.1f, 0.02f, 0.05f, 0.1f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.0f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
    };

    int despawn_radius = MOB_DESPAWN_DISTANCE * MOB_DESPAWN_DISTANCE;
    float night = 1.0f - state->light_level;
    float melee_chance = chances[level][1] + chances[level][3] * night * night;
    int hostile = boss ? state->boss_progress : level;

    int passive_count;
    int passive_slot;
    count_and_empty(&state->passive_mobs[level], MAX_PASSIVE_MOBS,
        &passive_count, &passive_slot);
    Rng passive_prob = rng_key(&spawn_rng);
    Rng passive_pos = rng_key(&spawn_rng);
    int passive_type = floor_mob_type(level, MOB_PASSIVE);
    state->passive_mobs[level].type_id[passive_slot] = passive_type;

    int melee_count;
    int melee_slot;
    count_and_empty(&state->melee_mobs[level], MAX_MELEE_MOBS,
        &melee_count, &melee_slot);
    Rng melee_prob = rng_key(&spawn_rng);
    Rng melee_pos = rng_key(&spawn_rng);
    int melee_type = floor_mob_type(hostile, MOB_MELEE);
    state->melee_mobs[level].type_id[melee_slot] = melee_type;

    int ranged_count;
    int ranged_slot;
    count_and_empty(&state->ranged_mobs[level], MAX_RANGED_MOBS, &ranged_count, &ranged_slot);
    Rng ranged_prob = rng_key(&spawn_rng);
    Rng ranged_pos = rng_key(&spawn_rng);
    int ranged_type = floor_mob_type(hostile, MOB_RANGED);
    state->ranged_mobs[level].type_id[ranged_slot] = ranged_type;

    bool try_passive = !boss && passive_count < MAX_PASSIVE_MOBS
        && rng_f32(passive_prob, 0) < chances[level][0];
    bool try_melee = melee_count < MAX_MELEE_MOBS
        && rng_f32(melee_prob, 0) < melee_chance * coeff;
    bool try_ranged = ranged_count < MAX_RANGED_MOBS
        && rng_f32(ranged_prob, 0) < chances[level][2] * coeff;
    if (try_passive || try_melee || try_ranged) {
        int min_hostile = boss ? -1 : 81;
        int max_hostile = boss ? 37 : despawn_radius;
        int spawn_rows[729];
        int spawn_cols[729];
        int row;
        int col;
        if (try_passive) {
            int n = collect_spawn_cells(
                state, level, 9, despawn_radius,
                false, false, spawn_rows, spawn_cols);
            if (pick_spawn_cell(spawn_rows, spawn_cols, n, passive_pos, &row, &col)) {
                spawn_into_slot(state, level, &state->passive_mobs[level],
                    passive_slot, MOB_PASSIVE, passive_type, row, col);
            }
        }
        if (try_melee) {
            int n = collect_spawn_cells(
                state, level, min_hostile, max_hostile,
                boss, false, spawn_rows, spawn_cols);
            if (pick_spawn_cell(spawn_rows, spawn_cols, n, melee_pos, &row, &col)) {
                spawn_into_slot(state, level, &state->melee_mobs[level],
                    melee_slot, MOB_MELEE, melee_type, row, col);
            }
        }
        if (try_ranged) {
            int n = collect_spawn_cells(
                state, level, min_hostile, max_hostile,
                boss, ranged_type == 5, spawn_rows, spawn_cols);
            if (pick_spawn_cell(spawn_rows, spawn_cols, n, ranged_pos, &row, &col)) {
                spawn_into_slot(state, level, &state->ranged_mobs[level],
                    ranged_slot, MOB_RANGED, ranged_type, row, col);
            }
        }
    }

    for (int plant = 0; plant < MAX_GROWING_PLANTS; plant++) {
        if (!state->growing_plants_mask[plant]) {
            continue;
        }

        state->growing_plants_age[plant] += 1;
        if (state->growing_plants_age[plant] >= 600) {
            set_block(state, 0, state->growing_plants_pos[plant][0],
                state->growing_plants_pos[plant][1], BLOCK_RIPE_PLANT);
        }
    }

    bool start_sleep = action == ACTION_SLEEP && state->player_energy < max_need(state);
    state->is_sleeping = state->is_sleeping || start_sleep;

    bool wake_from_sleep = state->is_sleeping && state->player_energy >= max_need(state);
    state->is_sleeping = state->is_sleeping && !wake_from_sleep;
    state->achievements[ACH_WAKE_UP] = state->achievements[ACH_WAKE_UP] || wake_from_sleep;

    bool start_rest = action == ACTION_REST && state->player_health < max_health(state);
    state->is_resting = state->is_resting || start_rest;

    bool wake_from_rest = state->is_resting && (
        state->player_health >= max_health(state)
        || state->player_food <= 0
        || state->player_drink <= 0
    );
    state->is_resting = state->is_resting && !wake_from_rest;

    bool not_boss = !fighting_boss(state);
    float decay = 1.0f - 0.125f * (state->player_dexterity - 1);

    state->player_hunger += (state->is_sleeping ? 0.5f : 1.0f) * decay;
    if (state->player_hunger > 25.0f) {
        state->player_hunger = 0.0f;
        state->player_food = clampi(state->player_food - (not_boss ? 1 : 0), 0, max_need(state));
    }

    state->player_thirst += (state->is_sleeping ? 0.5f : 1.0f) * decay;
    if (state->player_thirst > 20.0f) {
        state->player_thirst = 0.0f;
        state->player_drink = clampi(state->player_drink - (not_boss ? 1 : 0), 0, max_need(state));
    }

    if (state->is_sleeping) {
        state->player_fatigue = state->player_fatigue - 1.0f;
        if (state->player_fatigue > 0.0f) {
            state->player_fatigue = 0.0f;
        }
    } else {
        state->player_fatigue += decay;
    }
    if (state->player_fatigue > 30.0f) {
        state->player_fatigue = 0.0f;
        state->player_energy = clampi(state->player_energy - (not_boss ? 1 : 0), 0, max_need(state));
    } else if (state->player_fatigue < -10.0f) {
        state->player_fatigue = 0.0f;
        state->player_energy = clampi(state->player_energy + 1, 0, max_need(state));
    }

    bool all_necessities = state->player_food > 0
        && state->player_drink > 0
        && (state->player_energy > 0 || state->is_sleeping);
    state->player_recover += all_necessities
        ? (state->is_sleeping ? 2.0f : 1.0f)
        : (state->is_sleeping ? -0.5f : -1.0f) * (not_boss ? 1.0f : 0.0f);

    if (state->player_recover > 25.0f) {
        state->player_recover = 0.0f;
        state->player_health = clampf(state->player_health + 1.0f, 0.0f, max_health(state));
    } else if (state->player_recover < -15.0f) {
        state->player_recover = 0.0f;
        state->player_health -= 1.0f;
    }

    float mana_gain = state->is_sleeping ? 2.0f : 1.0f;
    float mana_coeff = 1.0f + 0.25f * (state->player_intelligence - 1);
    state->player_recover_mana = (state->player_recover_mana + mana_gain) * mana_coeff;
    if (state->player_recover_mana > 30.0f) {
        state->player_recover_mana = 0.0f;
        state->player_mana = clampi(state->player_mana + 1, 0, max_mana(state));
    }

    int* stacks[] = {
        &state->inventory.wood, &state->inventory.stone, &state->inventory.coal,
        &state->inventory.iron, &state->inventory.diamond, &state->inventory.sapling,
        &state->inventory.pickaxe, &state->inventory.sword, &state->inventory.bow,
        &state->inventory.arrows, &state->inventory.torches, &state->inventory.ruby,
        &state->inventory.sapphire, &state->inventory.books,
    };
    for (int i = 0; i < 14; i++) {
        *stacks[i] = clampi(*stacks[i], 0, 99);
    }
    for (int i = 0; i < 4; i++) {
        state->inventory.armour[i] = clampi(state->inventory.armour[i], 0, 99);
    }
    for (int i = 0; i < NUM_POTIONS; i++) {
        state->inventory.potions[i] = clampi(state->inventory.potions[i], 0, 99);
    }

    state->player_health = clampf(state->player_health, 0.0f, max_health(state));
    state->player_food = clampi(state->player_food, 0, max_need(state));
    state->player_drink = clampi(state->player_drink, 0, max_need(state));
    state->player_energy = clampi(state->player_energy, 0, max_need(state));
    state->player_mana = clampi(state->player_mana, 0, max_mana(state));

    state->achievements[ACH_COLLECT_WOOD] |= state->inventory.wood > 0;
    state->achievements[ACH_COLLECT_STONE] |= state->inventory.stone > 0;
    state->achievements[ACH_COLLECT_COAL] |= state->inventory.coal > 0;
    state->achievements[ACH_COLLECT_IRON] |= state->inventory.iron > 0;
    state->achievements[ACH_COLLECT_DIAMOND] |= state->inventory.diamond > 0;
    state->achievements[ACH_COLLECT_SAPPHIRE] |= state->inventory.sapphire > 0;
    state->achievements[ACH_COLLECT_RUBY] |= state->inventory.ruby > 0;
    state->achievements[ACH_COLLECT_SAPLING] |= state->inventory.sapling > 0;
    state->achievements[ACH_FIND_BOW] |= state->inventory.bow > 0;
    state->achievements[ACH_MAKE_ARROW] |= state->inventory.arrows > 0;
    state->achievements[ACH_MAKE_TORCH] |= state->inventory.torches > 0;
    state->achievements[ACH_MAKE_WOOD_PICKAXE] |= state->inventory.pickaxe >= 1;
    state->achievements[ACH_MAKE_STONE_PICKAXE] |= state->inventory.pickaxe >= 2;
    state->achievements[ACH_MAKE_IRON_PICKAXE] |= state->inventory.pickaxe >= 3;
    state->achievements[ACH_MAKE_DIAMOND_PICKAXE] |= state->inventory.pickaxe >= 4;
    state->achievements[ACH_MAKE_WOOD_SWORD] |= state->inventory.sword >= 1;
    state->achievements[ACH_MAKE_STONE_SWORD] |= state->inventory.sword >= 2;
    state->achievements[ACH_MAKE_IRON_SWORD] |= state->inventory.sword >= 3;
    state->achievements[ACH_MAKE_DIAMOND_SWORD] |= state->inventory.sword >= 4;
    if (env->state.player_level > env->max_floor_accum) {
        env->max_floor_accum = env->state.player_level;
    }

    store_rng(state, rng_key(&step_rng));
    state->timestep += 1;
    float day_progress = fmodf(state->timestep / (float)DAY_LENGTH, 1.0f) + 0.3f;
    state->light_level = 1.0f - powf(fabsf(cosf(3.14159265358979323846f * day_progress)), 3.0f);

    done = state->player_health <= 0.0f || state->timestep >= DEFAULT_MAX_TIMESTEPS;
    } while (!done && (state->is_sleeping || state->is_resting));

    float achievement_reward = 0.0f;
    for (int i = 0; i < NUM_ACHIEVEMENTS; i++) {
        int delta = state->achievements[i] - initial_achievements[i];
        achievement_reward += delta * ACHIEVEMENT_REWARD_MAP[i];
    }
    float reward = achievement_reward + (equipped_armour(state) - initial_armour);
    if (state->player_health <= 0.0f) {
        reward = -1.0f;
    }

    memcpy(env->achievements, env->state.achievements, sizeof(env->achievements));

    env->agents[0].rewards[0] = reward;
    env->agents[0].terminals[0] = done ? 1.0f : 0.0f;
    env->episode_return_accum += achievement_reward;
    env->episode_length_accum += 1;

    if (done) {
        int unlocked = 0;
        float achievement_return = 0.0f;
        for (int i = 0; i < NUM_ACHIEVEMENTS; i++) {
            if (env->achievements[i]) {
                unlocked++;
                achievement_return += ACHIEVEMENT_REWARD_MAP[i];
                env->log.achievements[i] += 1.0f;
            }
        }
        env->log.achievement_rate += unlocked / (float)NUM_ACHIEVEMENTS;
        env->log.perf += achievement_return / max_achievement_return();
        env->log.score += env->episode_return_accum;
        env->log.episode_return += env->episode_return_accum;
        env->log.episode_length += env->episode_length_accum;
        for (int floor = 0; floor <= env->max_floor_accum; floor++) {
            env->log.floors[floor] += 1.0f;
        }
        env->log.n += 1.0f;

        env->episode_return_accum = 0.0f;
        env->episode_length_accum = 0;
        env->max_floor_accum = 0;
        memset(env->achievements, 0, sizeof(env->achievements));
        if (env->num_levels > 0) {
            uint32_t idx = (uint32_t)reset_key % (uint32_t)env->num_levels;
            env->state = env->levels[idx];
        } else {
            Rng done_unused;
            Rng world_key;
            rng_split(reset_key, &done_unused, &world_key);
            generate_world_from_key(&env->state, world_key);
        }
    }

    compute_observations(env);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->agents[0].policy = 0;
    env->agents[0].action_mask = NULL;  // trainer wires mask after puf_init
    env->use_action_mask = 0;
    uint64_t seed_offset = 0;
    for (int i = 0; i < kwargs->size; i++) {
        if (strcmp(kwargs->items[i].key, "seed_offset") == 0) {
            seed_offset = (uint64_t)kwargs->items[i].value;
        } else if (strcmp(kwargs->items[i].key, "action_mask") == 0) {
            env->use_action_mask = kwargs->items[i].value != 0.0;
        }
    }
    env->seed = seed_offset + env->rng;
    memset(&env->state, 0, sizeof(State));
    env->episode_return_accum = 0.0f;
    env->episode_length_accum = 0;
    env->max_floor_accum = 0;
    memset(env->achievements, 0, sizeof(env->achievements));
    memset(&env->log, 0, sizeof(Log));
    env->client = NULL;
}

State* make_craftax_levels(int n) {
    State* levels = (State*)calloc(n, sizeof(State));
    for (int i = 0; i < n; i++) {
        Rng init_key = rng_seed(i);
        Rng discard;
        Rng reset_key;
        rng_split(init_key, &discard, &reset_key);
        Rng unused;
        Rng world_key;
        rng_split(reset_key, &unused, &world_key);
        generate_world_from_key(&levels[i], world_key);
    }
    return levels;
}

Env* my_vec_init(int* num_envs_out, int* env_starts, int* env_counts,
        Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = dict_get(vec_kwargs, "total_agents");
    int num_buffers = dict_get(vec_kwargs, "num_buffers");
    int agents_per_buf = total_agents / num_buffers;
    int num_envs = total_agents;
    int num_levels = 0;
    DictItem* item = dict_find(env_kwargs, "reset_pool_size");
    if (item) {
        num_levels = item->value;
    }
    State* levels = NULL;
    if (num_levels > 0) {
        levels = make_craftax_levels(num_levels);
    }

    Env* envs = (Env*)calloc(num_envs, sizeof(Env));
    int buf = 0;
    int buf_agents = 0;
    env_starts[0] = 0;
    env_counts[0] = 0;
    for (int i = 0; i < num_envs; i++) {
        Env* env = &envs[i];
        env->rng = i;
        env->levels = levels;
        env->num_levels = num_levels;
        puf_init(env, env_kwargs);
        buf_agents += env->num_agents;
        env_counts[buf]++;
        if (buf_agents >= agents_per_buf && buf < num_buffers - 1) {
            buf++;
            env_starts[buf] = i + 1;
            env_counts[buf] = 0;
            buf_agents = 0;
        }
    }
    *num_envs_out = num_envs;
    return envs;
}

void my_vec_close(Env* envs) {
    free(envs[0].levels);
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "achievement_rate", log->achievement_rate);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "floor_0_overworld", log->floors[0]);
    dict_set(out, "floor_1_dungeon", log->floors[1]);
    dict_set(out, "floor_2_gnomish_mines", log->floors[2]);
    dict_set(out, "floor_3_sewers", log->floors[3]);
    dict_set(out, "floor_4_vault", log->floors[4]);
    dict_set(out, "floor_5_troll_mines", log->floors[5]);
    dict_set(out, "floor_6_fire_realm", log->floors[6]);
    dict_set(out, "floor_7_ice_realm", log->floors[7]);
    dict_set(out, "floor_8_graveyard", log->floors[8]);
    dict_set(out, "n", log->n);
}

static Texture2D textures;
static int textures_loaded;

static void draw_tile(int tex_id, int x, int y, int px) {
    Rectangle src = {
        (float)((tex_id % TEX_SHEET_COLS) * TEX_TILE_PX),
        (float)((tex_id / TEX_SHEET_COLS) * TEX_TILE_PX),
        (float)TEX_TILE_PX,
        (float)TEX_TILE_PX,
    };
    Rectangle dst = {(float)x, (float)y, (float)px, (float)px};
    DrawTexturePro(textures, src, dst, (Vector2){0, 0}, 0.0f, WHITE);
}

static int projectile_tex(int ptype, int dr, int dc) {
    if (ptype == PROJECTILE_DAGGER) {
        return TEX_PROJ_DAGGER;
    }
    if (ptype == PROJECTILE_FIREBALL || ptype == PROJECTILE_FIREBALL2) {
        return TEX_PROJ_FIREBALL;
    }
    if (ptype == PROJECTILE_ICEBALL || ptype == PROJECTILE_ICEBALL2) {
        return TEX_PROJ_ICEBALL;
    }
    if (ptype == PROJECTILE_SLIMEBALL) {
        return TEX_PROJ_SLIMEBALL;
    }
    if (dr < 0) {
        return TEX_ARROW_UP;
    }
    if (dr > 0) {
        return TEX_ARROW_DOWN;
    }
    if (dc < 0) {
        return TEX_ARROW_LEFT;
    }
    return TEX_ARROW_RIGHT;
}

static int mob_tex_base[] = {TEX_PASSIVE, TEX_MELEE, TEX_RANGED};

static void draw_agent_obs(Craftax* env, int panel_x, int panel_y,
        int panel_w, int panel_h) {
    State* state = &env->state;
    int pad = 10;
    int px = (panel_w - pad * 2) / OBS_COLS;
    int grid_x = panel_x + (panel_w - OBS_COLS * px) / 2;
    int grid_y = panel_y + 36 + pad;
    int level = clampi(state->player_level, 0, NUM_LEVELS - 1);
    int pr = state->player_position[0];
    int pc = state->player_position[1];
    int rr = OBS_ROWS / 2;
    int rc = OBS_COLS / 2;

    DrawRectangle(panel_x, panel_y, panel_w, panel_h, (Color){8, 10, 14, 255});
    DrawRectangleLines(panel_x, panel_y, panel_w, panel_h, (Color){0, 210, 220, 255});
    DrawText("agent obs", panel_x + pad, panel_y + 8, 18, WHITE);
    DrawText("9x11  light>12", panel_x + pad, panel_y + 28, 12,
        (Color){140, 160, 166, 255});

    for (int vr = 0; vr < OBS_ROWS; vr++) {
        for (int vc = 0; vc < OBS_COLS; vc++) {
            int wr = pr + (vr - rr);
            int wc = pc + (vc - rc);
            int dst_x = grid_x + vc * px;
            int dst_y = grid_y + vr * px;
            int lit = (unsigned)wr < MAP_SIZE && (unsigned)wc < MAP_SIZE
                && state->light_map[level][wr][wc] > VISIBLE_LIGHT_THRESHOLD;
            if (!lit) {
                DrawRectangle(dst_x, dst_y, px, px, BLACK);
                continue;
            }
            int block = state->map[level][wr][wc];
            if (block < 0 || block >= NUM_BLOCK_TYPES) {
                block = BLOCK_INVALID;
            }
            draw_tile(block, dst_x, dst_y, px);
            int item = state->item_map[level][wr][wc];
            if (item > ITEM_NONE) {
                draw_tile(TEX_ITEM_BASE + item, dst_x, dst_y, px);
            }
            int mob_class;
            int slot;
            if (find_mob_at(state, level, wr, wc, &mob_class, &slot)) {
                int type_id = mobs_for_class(state, level, mob_class)->type_id[slot];
                draw_tile(mob_tex_base[mob_class] + type_id, dst_x, dst_y, px);
            }
        }
    }

    for (int from_player = 0; from_player < 2; from_player++) {
        Mobs* projectiles = from_player
            ? &state->player_projectiles[level] : &state->mob_projectiles[level];
        int (*directions)[MAX_PLAYER_PROJECTILES][2] = from_player
            ? state->player_projectile_directions : state->mob_projectile_dirs;
        for (int i = 0; i < MAX_PLAYER_PROJECTILES; i++) {
            if (!projectiles->mask[i]) {
                continue;
            }
            int row = projectiles->position[i][0];
            int col = projectiles->position[i][1];
            int vr = row - pr + rr;
            int vc = col - pc + rc;
            int lit = (unsigned)row < MAP_SIZE && (unsigned)col < MAP_SIZE
                && state->light_map[level][row][col] > VISIBLE_LIGHT_THRESHOLD;
            if (vr < 0 || vr >= OBS_ROWS || vc < 0 || vc >= OBS_COLS || !lit) {
                continue;
            }
            int dr = directions[level][i][0];
            int dc = directions[level][i][1];
            draw_tile(projectile_tex(projectiles->type_id[i], dr, dc),
                grid_x + vc * px, grid_y + vr * px, px);
        }
    }

    int player_tex = TEX_PLAYER_DOWN;
    if (state->is_sleeping) {
        player_tex = TEX_PLAYER_SLEEP;
    } else if (state->player_direction == ACTION_LEFT) {
        player_tex = TEX_PLAYER_LEFT;
    } else if (state->player_direction == ACTION_RIGHT) {
        player_tex = TEX_PLAYER_RIGHT;
    } else if (state->player_direction == ACTION_UP) {
        player_tex = TEX_PLAYER_UP;
    }
    draw_tile(player_tex, grid_x + rc * px, grid_y + rr * px, px);

    float v = env->predicted_value;
    float t = clampf(v, -1.0f, 1.0f);
    unsigned char fade = (unsigned char)(255.0f * (1.0f - fabsf(t)));
    Color vc = WHITE;
    if (t < 0.0f) {
        vc = (Color){255, fade, fade, 255};
    } else if (t > 0.0f) {
        vc = (Color){fade, 255, fade, 255};
    }
    const char* vlabel = TextFormat("V(o, h) = %.2f", v);
    DrawText(vlabel, panel_x + (panel_w - MeasureText(vlabel, 18)) / 2,
        grid_y + OBS_ROWS * px + 10, 18, vc);
}

static void draw_icon_count(int tex_id, int value, int x, int y) {
    draw_tile(tex_id, x, y, 20);
    DrawText(TextFormat("%d", value), x + 23, y + 4, 14, RAYWHITE);
}

static void draw_inv_slot(int tex_id, int overlay, int x, int y) {
    DrawRectangle(x, y, 24, 24, (Color){32, 32, 32, 255});
    DrawRectangleLines(x, y, 24, 24, (Color){80, 80, 80, 255});
    if (tex_id >= 0) {
        draw_tile(tex_id, x, y, 24);
    }
    if (overlay >= 0) {
        draw_tile(overlay, x, y, 24);
    }
}

static const char* action_names[ATN_DIM] = {
    "NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP",
    "PLACE_STONE", "PLACE_TABLE", "PLACE_FURNACE", "PLACE_PLANT",
    "MAKE_WOOD_PICKAXE", "MAKE_STONE_PICKAXE", "MAKE_IRON_PICKAXE",
    "MAKE_WOOD_SWORD", "MAKE_STONE_SWORD", "MAKE_IRON_SWORD",
    "REST", "DESCEND", "ASCEND",
    "MAKE_DIAMOND_PICKAXE", "MAKE_DIAMOND_SWORD",
    "MAKE_IRON_ARMOUR", "MAKE_DIAMOND_ARMOUR",
    "SHOOT_ARROW", "MAKE_ARROW", "CAST_FIREBALL", "CAST_ICEBALL", "PLACE_TORCH",
    "DRINK_POTION_RED", "DRINK_POTION_GREEN", "DRINK_POTION_BLUE",
    "DRINK_POTION_PINK", "DRINK_POTION_CYAN", "DRINK_POTION_YELLOW",
    "READ_BOOK", "ENCHANT_SWORD", "ENCHANT_ARMOUR", "MAKE_TORCH",
    "LEVEL_UP_DEXTERITY", "LEVEL_UP_STRENGTH", "LEVEL_UP_INTELLIGENCE",
    "ENCHANT_BOW",
};

static const char* action_keys[ATN_DIM] = {
    "Q", "A", "D", "W", "S", "Space", "Tab",
    "R", "T", "F", "P",
    "1", "2", "3",
    "5", "6", "7",
    "E", ".", ",",
    "4", "8",
    "Y", "U",
    "I", "O", "G", "H", "J",
    "Z", "X", "C", "V", "B", "N",
    "M", "K", "L", "[",
    "]", "-", "=",
    ";",
};

static const char* ach_names[NUM_ACHIEVEMENTS] = {
    "Collect Wood", "Place Table", "Eat Cow", "Collect Sapling", "Collect Drink",
    "Make Wood Pickaxe", "Make Wood Sword", "Place Plant", "Defeat Zombie",
    "Collect Stone", "Place Stone", "Eat Plant", "Defeat Skeleton",
    "Make Stone Pickaxe", "Make Stone Sword", "Wake Up", "Place Furnace",
    "Collect Coal", "Collect Iron", "Collect Diamond", "Make Iron Pickaxe",
    "Make Iron Sword", "Make Arrow", "Make Torch", "Place Torch",
    "Make Diamond Sword", "Make Iron Armour", "Make Diamond Armour",
    "Enter Gnomish Mines", "Enter Dungeon", "Enter Sewers", "Enter Vault",
    "Enter Troll Mines", "Enter Fire Realm", "Enter Ice Realm", "Enter Graveyard",
    "Defeat Gnome Warrior", "Defeat Gnome Archer", "Defeat Orc Soldier",
    "Defeat Orc Mage", "Defeat Lizard", "Defeat Kobold", "Defeat Troll",
    "Defeat Deep Thing", "Defeat Pigman", "Defeat Fire Elemental",
    "Defeat Frost Troll", "Defeat Ice Elemental", "Damage Necromancer",
    "Defeat Necromancer", "Eat Bat", "Eat Snail", "Find Bow", "Fire Bow",
    "Collect Sapphire", "Learn Fireball", "Cast Fireball", "Learn Iceball",
    "Cast Iceball", "Collect Ruby", "Make Diamond Pickaxe", "Open Chest",
    "Drink Potion", "Enchant Sword", "Enchant Armour", "Defeat Knight",
    "Defeat Archer",
};

void puf_render(Craftax* env) {
    const int view_w = RENDER_COLS * TEX_DRAW_PX;
    const int view_h = RENDER_ROWS * TEX_DRAW_PX;
    const int hud_h = 142;
    const int origin_x = ACH_PANEL_W;
    const int window_w = origin_x + view_w + OBS_PANEL_W + ACTION_PANEL_W;

    if (env->client == NULL) {
        env->client = (Client*)calloc(1, sizeof(Client));
        env->client->cell_size = TEX_DRAW_PX;
        env->client->screen_width = window_w;
        env->client->screen_height = view_h + hud_h;
    }

    Client* client = env->client;
    if (!client->window_ready) {
        InitWindow(client->screen_width, client->screen_height, "Craftax");
        SetTargetFPS(30);
        client->window_ready = true;
    }
    if (!textures_loaded) {
        const char* candidates[] = {
            "resources/craftax/textures.png",
            "../resources/craftax/textures.png",
            "../../resources/craftax/textures.png",
        };
        for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
            if (FileExists(candidates[i])) {
                textures = LoadTexture(candidates[i]);
                break;
            }
        }
        if (textures.id == 0) {
            fprintf(stderr, "craftax textures.png not found in resources/craftax\n");
            exit(1);
        }
        SetTextureFilter(textures, TEXTURE_FILTER_POINT);
        textures_loaded = 1;
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    got_human_input(env);

    int level = clampi(env->state.player_level, 0, NUM_LEVELS - 1);
    int player_row = clampi(env->state.player_position[0], 0, MAP_SIZE - 1);
    int player_col = clampi(env->state.player_position[1], 0, MAP_SIZE - 1);
    int half_r = RENDER_ROWS / 2;
    int half_c = RENDER_COLS / 2;
    int top_row = player_row - half_r;
    int left_col = player_col - half_c;

    BeginDrawing();
    ClearBackground(BLACK);

    for (int vr = 0; vr < RENDER_ROWS; vr++) {
        for (int vc = 0; vc < RENDER_COLS; vc++) {
            int wr = top_row + vr;
            int wc = left_col + vc;
            int dst_x = origin_x + vc * TEX_DRAW_PX;
            int dst_y = vr * TEX_DRAW_PX;

            int in_map = wr >= 0 && wr < MAP_SIZE && wc >= 0 && wc < MAP_SIZE;
            int block = BLOCK_OUT_OF_BOUNDS;
            if (in_map) {
                block = env->state.map[level][wr][wc];
            }
            if (block < 0 || block >= NUM_BLOCK_TYPES) {
                block = BLOCK_INVALID;
            }
            draw_tile(block, dst_x, dst_y, TEX_DRAW_PX);
            if (!in_map) {
                continue;
            }
            int item = env->state.item_map[level][wr][wc];
            if (item > ITEM_NONE) {
                draw_tile(TEX_ITEM_BASE + item, dst_x, dst_y, TEX_DRAW_PX);
            }
            int mob_class;
            int slot;
            if (find_mob_at(&env->state, level, wr, wc, &mob_class, &slot)) {
                int type_id = mobs_for_class(&env->state, level, mob_class)
                    ->type_id[slot];
                draw_tile(mob_tex_base[mob_class] + type_id, dst_x, dst_y, TEX_DRAW_PX);
            }
        }
    }

    for (int from_player = 0; from_player < 2; from_player++) {
        Mobs* projectiles = from_player
            ? &env->state.player_projectiles[level]
            : &env->state.mob_projectiles[level];
        int (*directions)[MAX_PLAYER_PROJECTILES][2] = from_player
            ? env->state.player_projectile_directions
            : env->state.mob_projectile_dirs;
        for (int i = 0; i < MAX_PLAYER_PROJECTILES; i++) {
            if (!projectiles->mask[i]) {
                continue;
            }
            int vr = projectiles->position[i][0] - top_row;
            int vc = projectiles->position[i][1] - left_col;
            if (vr < 0 || vr >= RENDER_ROWS || vc < 0 || vc >= RENDER_COLS) {
                continue;
            }
            int dr = directions[level][i][0];
            int dc = directions[level][i][1];
            draw_tile(projectile_tex(projectiles->type_id[i], dr, dc),
                origin_x + vc * TEX_DRAW_PX, vr * TEX_DRAW_PX, TEX_DRAW_PX);
        }
    }

    int player_tex = TEX_PLAYER_DOWN;
    if (env->state.is_sleeping) {
        player_tex = TEX_PLAYER_SLEEP;
    } else if (env->state.player_direction == ACTION_LEFT) {
        player_tex = TEX_PLAYER_LEFT;
    } else if (env->state.player_direction == ACTION_RIGHT) {
        player_tex = TEX_PLAYER_RIGHT;
    } else if (env->state.player_direction == ACTION_UP) {
        player_tex = TEX_PLAYER_UP;
    }
    draw_tile(player_tex, origin_x + half_c * TEX_DRAW_PX,
        half_r * TEX_DRAW_PX, TEX_DRAW_PX);

    if (env->state.light_level < 1.0f) {
        unsigned char alpha = (unsigned char)((1.0f - env->state.light_level) * 140.0f);
        DrawRectangle(origin_x, 0, view_w, view_h, (Color){0, 0, 40, alpha});
    }

    int floor_bar_h = 16;
    int cell_w = view_w / NUM_LEVELS;
    DrawRectangle(origin_x, 0, view_w, floor_bar_h, (Color){18, 18, 18, 230});
    for (int f = 0; f < NUM_LEVELS; f++) {
        int x = origin_x + f * cell_w;
        int w = (f == NUM_LEVELS - 1) ? (origin_x + view_w - x) : cell_w;
        bool reached = f <= env->max_floor_accum;
        bool here = f == env->state.player_level;
        Color fill = reached ? (Color){255, 210, 40, 255} : (Color){45, 45, 45, 255};
        if (here) {
            fill = (Color){255, 235, 80, 255};
        }
        DrawRectangle(x + 1, 1, w - 2, floor_bar_h - 2, fill);
        if (here) {
            DrawRectangleLines(x + 1, 1, w - 2, floor_bar_h - 2, WHITE);
        }
        DrawText(TextFormat("%d", f), x + 4, 2, 10, reached ? BLACK : (Color){140, 140, 140, 255});
    }

    draw_agent_obs(env, origin_x + view_w, 0, OBS_PANEL_W, view_h + hud_h);

    int hud_y = view_h;
    Inventory* inv = &env->state.inventory;
    DrawRectangle(origin_x, hud_y, view_w, hud_h, (Color){20, 20, 20, 255});

    int health_max = max_health(&env->state);
    float health_frac = clampf(env->state.player_health / health_max, 0.0f, 1.0f);
    int bar_x = origin_x + 4;
    int bar_y = hud_y + 4;
    int bar_w = view_w - 8;
    int bar_h = 18;
    DrawRectangle(bar_x, bar_y, bar_w, bar_h, (Color){115, 25, 25, 255});
    DrawRectangle(bar_x, bar_y, (int)(bar_w * health_frac), bar_h, (Color){35, 190, 75, 255});
    DrawRectangleLines(bar_x, bar_y, bar_w, bar_h, (Color){220, 220, 220, 255});
    DrawText(
        TextFormat("HP %.0f / %d", env->state.player_health, health_max),
        bar_x + 8,
        bar_y + 2,
        14,
        WHITE
    );

    DrawText(
        TextFormat(
            "Food:%d/%d  Drink:%d/%d  Energy:%d/%d  Mana:%d/%d  L:%d  t:%d",
            env->state.player_food,
            max_need(&env->state),
            env->state.player_drink,
            max_need(&env->state),
            env->state.player_energy,
            max_need(&env->state),
            env->state.player_mana,
            max_mana(&env->state),
            env->state.player_level,
            env->state.timestep
        ),
        origin_x + 4,
        hud_y + 26,
        14,
        WHITE
    );
    DrawText(
        TextFormat(
            "XP:%d  DEX:%d  STR:%d  INT:%d  light:%.2f  sleep:%d rest:%d",
            env->state.player_xp,
            env->state.player_dexterity,
            env->state.player_strength,
            env->state.player_intelligence,
            env->state.light_level,
            env->state.is_sleeping,
            env->state.is_resting
        ),
        origin_x + 4,
        hud_y + 44,
        14,
        (Color){200, 200, 200, 255}
    );
    int achievements = 0;
    for (int i = 0; i < NUM_ACHIEVEMENTS; i++) {
        achievements += env->state.achievements[i] ? 1 : 0;
    }
    int inv_y = hud_y + 62;
    int inv_x = origin_x + 4;
    int inv_ids[] = {
        BLOCK_WOOD, BLOCK_STONE, BLOCK_COAL, BLOCK_IRON, BLOCK_DIAMOND,
        TEX_SAPLING, TEX_TORCH_INV, BLOCK_RUBY, BLOCK_SAPPHIRE, TEX_BOOK,
    };
    int inv_counts[] = {
        inv->wood, inv->stone, inv->coal, inv->iron, inv->diamond,
        inv->sapling, inv->torches, inv->ruby, inv->sapphire, inv->books,
    };
    for (int i = 0; i < 10; i++) {
        draw_icon_count(inv_ids[i], inv_counts[i], inv_x + 52 * i, inv_y);
    }
    int armour_x = inv_x + 52 * 10 + 8;
    for (int slot = 0; slot < 4; slot++) {
        int alvl = inv->armour[slot];
        int tex = -1;
        if (alvl > 0) {
            tex = (alvl >= 2 ? TEX_ARMOUR_DIAMOND : TEX_ARMOUR_IRON) + slot;
        }
        int overlay = -1;
        int ench = env->state.armour_enchantments[slot];
        if (ench == 1) {
            overlay = TEX_ARMOUR_ENCHANT_FIRE + slot;
        } else if (ench == 2) {
            overlay = TEX_ARMOUR_ENCHANT_ICE + slot;
        }
        draw_inv_slot(tex, overlay, armour_x + slot * 30, inv_y);
    }
    int weap_y = hud_y + 90;
    int gear[] = {
        inv->pickaxe > 0 ? TEX_PICKAXE_WOOD + inv->pickaxe - 1 : -1,
        inv->sword > 0 ? TEX_SWORD_WOOD + inv->sword - 1 : -1,
        inv->bow > 0 ? TEX_BOW : -1,
        inv->arrows > 0 ? TEX_ARROW_UP : -1,
    };
    int overlays[4] = {-1, -1, -1, -1};
    if (env->state.sword_enchantment == 1) {
        overlays[1] = TEX_SWORD_ENCHANT_FIRE;
    } else if (env->state.sword_enchantment == 2) {
        overlays[1] = TEX_SWORD_ENCHANT_ICE;
    }
    if (inv->arrows > 0) {
        if (env->state.bow_enchantment == 1) {
            overlays[3] = TEX_ARROW_ENCHANT_FIRE;
        } else if (env->state.bow_enchantment == 2) {
            overlays[3] = TEX_ARROW_ENCHANT_ICE;
        }
    }
    for (int i = 0; i < 4; i++) {
        draw_inv_slot(gear[i], overlays[i], armour_x + 30 * i, weap_y);
    }
    DrawText(TextFormat("%d", inv->arrows), armour_x + 117, weap_y + 6, 14, RAYWHITE);
    for (int p = 0; p < NUM_POTIONS; p++) {
        draw_icon_count(TEX_POTION + p, inv->potions[p], inv_x + 52 * p, weap_y);
    }
    draw_inv_slot(env->state.learned_spells[0] ? TEX_PROJ_FIREBALL : -1,
        -1, inv_x + 52 * 6, weap_y);
    draw_inv_slot(env->state.learned_spells[1] ? TEX_PROJ_ICEBALL : -1,
        -1, inv_x + 52 * 7, weap_y);
    int human = IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);
    DrawText(
        TextFormat(
            "ach:%d/%d  ret:%.2f len:%d   %s",
            achievements,
            NUM_ACHIEVEMENTS,
            env->episode_return_accum,
            env->episode_length_accum,
            human ? "HUMAN" : "Hold SHIFT to take control"
        ),
        origin_x + 4,
        hud_y + 118,
        14,
        human ? (Color){255, 210, 40, 255} : (Color){200, 200, 140, 255}
    );

    int panel_x = origin_x + view_w + OBS_PANEL_W;
    int panel_h = view_h + hud_h;
    int taken_action = env->agents[0].actions[0];
    DrawRectangle(panel_x, 0, ACTION_PANEL_W, panel_h, (Color){12, 18, 22, 255});
    DrawRectangleLines(panel_x, 0, ACTION_PANEL_W, panel_h, (Color){55, 70, 76, 255});
    DrawText("Actions", panel_x + 10, 8, 18, RAYWHITE);
    DrawText("key", panel_x + 12, 32, 11, (Color){140, 160, 166, 255});
    DrawText("action", panel_x + 78, 32, 11, (Color){140, 160, 166, 255});
    for (int action = 0; action < ATN_DIM; action++) {
        int y = 48 + action * 15;
        bool selected = action == taken_action;
        bool legal = env->agents[0].action_mask == NULL
            || env->agents[0].action_mask[action];
        if (selected) {
            DrawRectangle(panel_x + 6, y - 2, ACTION_PANEL_W - 12, 15, (Color){0, 210, 220, 255});
        }
        Color text_color = selected ? BLACK
            : (legal ? (Color){220, 230, 230, 255} : (Color){80, 90, 90, 255});
        DrawText(action_keys[action], panel_x + 12, y, 10, text_color);
        DrawText(TextFormat("%02d %s", action, action_names[action]),
            panel_x + 78, y, 10, text_color);
    }

    int ach_h = view_h + hud_h;
    DrawRectangle(0, 0, ACH_PANEL_W, ach_h, WHITE);
    DrawText("Achievements", 8, 6, 16, BLACK);
    int ach_top = 26;
    int ach_row = (ach_h - ach_top) / NUM_ACHIEVEMENTS;
    if (ach_row < 10) {
        ach_row = 10;
    }
    for (int i = 0; i < NUM_ACHIEVEMENTS; i++) {
        int y = ach_top + i * ach_row;
        bool done = env->state.achievements[i] != 0;
        if (done) {
            DrawRectangle(0, y, ACH_PANEL_W, ach_row, (Color){46, 180, 80, 255});
        }
        DrawText(
            ach_names[i],
            6,
            y + (ach_row > 10 ? 1 : 0),
            10,
            done ? WHITE : (Color){50, 50, 50, 255}
        );
    }

    EndDrawing();
    puf_web_vsync();
}

void puf_close(Craftax* env) {
    if (env->client == NULL) {
        return;
    }
    if (env->client->window_ready) {
        CloseWindow();
    }
    free(env->client);
}

