// Standalone native port of Craftax spawn_mobs.
//
// This helper intentionally is not integrated into c_step yet. It mutates a
// full CraftaxState in place so tests can compare the subsystem directly
// against the installed JAX implementation.

#pragma once

#include "step_medium.h"

#define CRAFTAX_SPAWN_MAP_CELLS (CRAFTAX_MAP_SIZE * CRAFTAX_MAP_SIZE)

static inline CraftaxThreefryKey craftax_spawn_next_random_key(
    CraftaxThreefryKey* rng
) {
    CraftaxThreefryKey draw;
    craftax_threefry_split(*rng, rng, &draw);
    return draw;
}

static inline int32_t craftax_spawn_floor_mob_type(
    int32_t floor,
    int32_t mob_class
) {
    static const int32_t mapping[CRAFTAX_NUM_LEVELS][3] = {
        {0, 0, 0},
        {2, 2, 2},
        {1, 1, 1},
        {2, 3, 3},
        {2, 4, 4},
        {1, 5, 5},
        {1, 6, 6},
        {1, 7, 7},
        {0, 0, 0},
    };
    int32_t level = craftax_step_jax_index(floor, CRAFTAX_NUM_LEVELS);
    int32_t class_index = craftax_step_jax_index(mob_class, 3);
    return mapping[level][class_index];
}

static inline float craftax_spawn_floor_spawn_chance(
    int32_t floor,
    int32_t chance_index
) {
    static const float chances[CRAFTAX_NUM_LEVELS][4] = {
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
    int32_t level = craftax_step_jax_index(floor, CRAFTAX_NUM_LEVELS);
    int32_t index = craftax_step_jax_index(chance_index, 4);
    return chances[level][index];
}

static inline float craftax_spawn_mob_type_health(
    int32_t mob_type,
    int32_t mob_class
) {
    static const float health[CRAFTAX_NUM_MOB_TYPES][4] = {
        {3.0f, 5.0f, 3.0f, 0.0f},
        {4.0f, 7.0f, 5.0f, 0.0f},
        {6.0f, 9.0f, 6.0f, 0.0f},
        {8.0f, 11.0f, 8.0f, 0.0f},
        {0.0f, 12.0f, 12.0f, 0.0f},
        {0.0f, 20.0f, 4.0f, 0.0f},
        {0.0f, 20.0f, 14.0f, 0.0f},
        {0.0f, 24.0f, 16.0f, 0.0f},
    };
    int32_t type_index = craftax_step_jax_index(mob_type, CRAFTAX_NUM_MOB_TYPES);
    int32_t class_index = craftax_step_jax_index(mob_class, 4);
    return health[type_index][class_index];
}

static inline bool craftax_spawn_is_all_valid_block(int32_t block) {
    return block == CRAFTAX_BLOCK_GRASS
        || block == CRAFTAX_BLOCK_PATH
        || block == CRAFTAX_BLOCK_FIRE_GRASS
        || block == CRAFTAX_BLOCK_ICE_GRASS;
}

static inline bool craftax_spawn_is_grave_block(int32_t block) {
    return block == CRAFTAX_BLOCK_GRAVE
        || block == CRAFTAX_BLOCK_GRAVE2
        || block == CRAFTAX_BLOCK_GRAVE3;
}

static inline int32_t craftax_spawn_player_distance_squared(
    const CraftaxState* state,
    int32_t row,
    int32_t col
) {
    int32_t dr = row - state->player_position[0];
    int32_t dc = col - state->player_position[1];
    if (dr < 0) {
        dr = -dr;
    }
    if (dc < 0) {
        dc = -dc;
    }
    return dr * dr + dc * dc;
}

static inline int32_t craftax_spawn_count_mobs3(
    const CraftaxMobs3* mobs,
    int32_t level
) {
    int32_t count = 0;
    for (int32_t i = 0; i < 3; i++) {
        count += (int32_t)mobs->mask[level][i];
    }
    return count;
}

static inline int32_t craftax_spawn_count_mobs2(
    const CraftaxMobs2* mobs,
    int32_t level
) {
    int32_t count = 0;
    for (int32_t i = 0; i < 2; i++) {
        count += (int32_t)mobs->mask[level][i];
    }
    return count;
}

static inline int32_t craftax_spawn_first_empty_mobs3(
    const CraftaxMobs3* mobs,
    int32_t level
) {
    for (int32_t i = 0; i < 3; i++) {
        if (!mobs->mask[level][i]) {
            return i;
        }
    }
    return 0;
}

static inline int32_t craftax_spawn_first_empty_mobs2(
    const CraftaxMobs2* mobs,
    int32_t level
) {
    for (int32_t i = 0; i < 2; i++) {
        if (!mobs->mask[level][i]) {
            return i;
        }
    }
    return 0;
}

static inline bool craftax_spawn_update_index(
    int32_t index,
    int32_t size,
    int32_t* mapped_index
) {
    if (index < -size || index >= size) {
        return false;
    }
    *mapped_index = index < 0 ? index + size : index;
    return true;
}

static inline void craftax_spawn_or_mob_map(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col,
    bool mask
) {
    int32_t map_level;
    int32_t map_row;
    int32_t map_col;
    if (!craftax_spawn_update_index(level, CRAFTAX_NUM_LEVELS, &map_level)
        || !craftax_spawn_update_index(row, CRAFTAX_MAP_SIZE, &map_row)
        || !craftax_spawn_update_index(col, CRAFTAX_MAP_SIZE, &map_col)) {
        return;
    }
    state->mob_map[map_level][map_row][map_col] =
        state->mob_map[map_level][map_row][map_col] || mask;
}

static inline int32_t craftax_spawn_fill_passive_map(
    const CraftaxState* state,
    int32_t level,
    bool valid[CRAFTAX_MAP_SIZE][CRAFTAX_MAP_SIZE]
) {
    int32_t count = 0;
    for (int32_t row = 0; row < CRAFTAX_MAP_SIZE; row++) {
        for (int32_t col = 0; col < CRAFTAX_MAP_SIZE; col++) {
            int32_t block = state->map[level][row][col];
            int32_t distance2 = craftax_spawn_player_distance_squared(
                state,
                row,
                col
            );
            bool ok = craftax_spawn_is_all_valid_block(block)
                && distance2 > 9
                && distance2 < (
                    CRAFTAX_MOB_DESPAWN_DISTANCE
                    * CRAFTAX_MOB_DESPAWN_DISTANCE
                )
                && !state->mob_map[level][row][col];
            valid[row][col] = ok;
            count += (int32_t)ok;
        }
    }
    return count;
}

static inline int32_t craftax_spawn_fill_melee_map(
    const CraftaxState* state,
    int32_t level,
    bool fighting_boss,
    bool valid[CRAFTAX_MAP_SIZE][CRAFTAX_MAP_SIZE]
) {
    int32_t count = 0;
    for (int32_t row = 0; row < CRAFTAX_MAP_SIZE; row++) {
        for (int32_t col = 0; col < CRAFTAX_MAP_SIZE; col++) {
            int32_t block = state->map[level][row][col];
            int32_t distance2 = craftax_spawn_player_distance_squared(
                state,
                row,
                col
            );
            bool terrain_ok = fighting_boss
                ? craftax_spawn_is_grave_block(block)
                : craftax_spawn_is_all_valid_block(block);
            bool range_ok = fighting_boss ? distance2 <= 36 : distance2 > 81;
            bool ok = terrain_ok
                && range_ok
                && distance2 < (
                    CRAFTAX_MOB_DESPAWN_DISTANCE
                    * CRAFTAX_MOB_DESPAWN_DISTANCE
                )
                && !state->mob_map[level][row][col];
            valid[row][col] = ok;
            count += (int32_t)ok;
        }
    }
    return count;
}

static inline int32_t craftax_spawn_fill_ranged_map(
    const CraftaxState* state,
    int32_t level,
    int32_t new_ranged_mob_type,
    bool fighting_boss,
    bool valid[CRAFTAX_MAP_SIZE][CRAFTAX_MAP_SIZE]
) {
    int32_t count = 0;
    for (int32_t row = 0; row < CRAFTAX_MAP_SIZE; row++) {
        for (int32_t col = 0; col < CRAFTAX_MAP_SIZE; col++) {
            int32_t block = state->map[level][row][col];
            int32_t distance2 = craftax_spawn_player_distance_squared(
                state,
                row,
                col
            );
            bool terrain_ok = new_ranged_mob_type == 5
                ? block == CRAFTAX_BLOCK_WATER
                : craftax_spawn_is_all_valid_block(block);
            terrain_ok = fighting_boss
                ? craftax_spawn_is_grave_block(block)
                : terrain_ok;
            bool range_ok = fighting_boss ? distance2 <= 36 : distance2 > 81;
            bool ok = terrain_ok
                && range_ok
                && distance2 < (
                    CRAFTAX_MOB_DESPAWN_DISTANCE
                    * CRAFTAX_MOB_DESPAWN_DISTANCE
                )
                && !state->mob_map[level][row][col];
            valid[row][col] = ok;
            count += (int32_t)ok;
        }
    }
    return count;
}

static inline void craftax_spawn_choose_position(
    CraftaxThreefryKey key,
    const bool valid[CRAFTAX_MAP_SIZE][CRAFTAX_MAP_SIZE],
    int32_t position[2]
) {
    int32_t flat_index = craftax_choice_bool_flat(
        key,
        (const bool*)valid,
        CRAFTAX_SPAWN_MAP_CELLS
    );
    position[0] = flat_index / CRAFTAX_MAP_SIZE;
    position[1] = flat_index % CRAFTAX_MAP_SIZE;
}

static inline void craftax_spawn_passive_mob(
    CraftaxState* state,
    CraftaxThreefryKey* rng,
    int32_t level,
    bool fighting_boss
) {
    bool can_spawn = craftax_spawn_count_mobs3(
        &state->passive_mobs,
        level
    ) < CRAFTAX_MAX_PASSIVE_MOBS;

    CraftaxThreefryKey draw_key = craftax_spawn_next_random_key(rng);
    can_spawn = can_spawn
        && craftax_threefry_uniform_f32(draw_key)
            < craftax_spawn_floor_spawn_chance(level, 0);
    can_spawn = can_spawn && !fighting_boss;

    bool valid[CRAFTAX_MAP_SIZE][CRAFTAX_MAP_SIZE];
    int32_t valid_count = craftax_spawn_fill_passive_map(state, level, valid);
    can_spawn = can_spawn && valid_count > 0;

    draw_key = craftax_spawn_next_random_key(rng);
    int32_t candidate_position[2];
    craftax_spawn_choose_position(draw_key, valid, candidate_position);

    int32_t new_type = craftax_spawn_floor_mob_type(
        level,
        CRAFTAX_MOB_PASSIVE
    );
    int32_t new_index = craftax_spawn_first_empty_mobs3(
        &state->passive_mobs,
        level
    );

    int32_t new_position[2] = {
        can_spawn
            ? candidate_position[0]
            : state->passive_mobs.position[level][new_index][0],
        can_spawn
            ? candidate_position[1]
            : state->passive_mobs.position[level][new_index][1],
    };
    float new_health = can_spawn
        ? craftax_spawn_mob_type_health(new_type, CRAFTAX_MOB_PASSIVE)
        : state->passive_mobs.health[level][new_index];
    bool new_mask = can_spawn
        ? true
        : state->passive_mobs.mask[level][new_index];

    state->passive_mobs.position[level][new_index][0] = new_position[0];
    state->passive_mobs.position[level][new_index][1] = new_position[1];
    state->passive_mobs.health[level][new_index] = new_health;
    state->passive_mobs.mask[level][new_index] = new_mask;
    state->passive_mobs.type_id[level][new_index] = new_type;

    craftax_spawn_or_mob_map(
        state,
        level,
        new_position[0],
        new_position[1],
        new_mask
    );
}

static inline void craftax_spawn_melee_mob(
    CraftaxState* state,
    CraftaxThreefryKey* rng,
    int32_t level,
    bool fighting_boss,
    int32_t monster_spawn_coeff
) {
    bool can_spawn = craftax_spawn_count_mobs3(
        &state->melee_mobs,
        level
    ) < CRAFTAX_MAX_MELEE_MOBS;

    int32_t new_type = craftax_spawn_floor_mob_type(level, CRAFTAX_MOB_MELEE);
    int32_t boss_type = craftax_spawn_floor_mob_type(
        state->boss_progress,
        CRAFTAX_MOB_MELEE
    );
    new_type = fighting_boss ? boss_type : new_type;

    CraftaxThreefryKey draw_key = craftax_spawn_next_random_key(rng);
    float night_coeff = 1.0f - state->light_level;
    float spawn_chance = craftax_spawn_floor_spawn_chance(level, 1)
        + craftax_spawn_floor_spawn_chance(level, 3) * night_coeff * night_coeff;
    can_spawn = can_spawn
        && craftax_threefry_uniform_f32(draw_key)
            < spawn_chance * (float)monster_spawn_coeff;

    bool valid[CRAFTAX_MAP_SIZE][CRAFTAX_MAP_SIZE];
    int32_t valid_count = craftax_spawn_fill_melee_map(
        state,
        level,
        fighting_boss,
        valid
    );
    can_spawn = can_spawn && valid_count > 0;

    draw_key = craftax_spawn_next_random_key(rng);
    int32_t candidate_position[2];
    craftax_spawn_choose_position(draw_key, valid, candidate_position);

    int32_t new_index = craftax_spawn_first_empty_mobs3(
        &state->melee_mobs,
        level
    );
    int32_t new_position[2] = {
        can_spawn
            ? candidate_position[0]
            : state->melee_mobs.position[level][new_index][0],
        can_spawn
            ? candidate_position[1]
            : state->melee_mobs.position[level][new_index][1],
    };
    float new_health = can_spawn
        ? craftax_spawn_mob_type_health(new_type, CRAFTAX_MOB_MELEE)
        : state->melee_mobs.health[level][new_index];
    bool new_mask = can_spawn
        ? true
        : state->melee_mobs.mask[level][new_index];

    state->melee_mobs.position[level][new_index][0] = new_position[0];
    state->melee_mobs.position[level][new_index][1] = new_position[1];
    state->melee_mobs.health[level][new_index] = new_health;
    state->melee_mobs.mask[level][new_index] = new_mask;
    state->melee_mobs.type_id[level][new_index] = new_type;

    craftax_spawn_or_mob_map(
        state,
        level,
        new_position[0],
        new_position[1],
        new_mask
    );
}

static inline void craftax_spawn_ranged_mob(
    CraftaxState* state,
    CraftaxThreefryKey* rng,
    int32_t level,
    bool fighting_boss,
    int32_t monster_spawn_coeff
) {
    bool can_spawn = craftax_spawn_count_mobs2(
        &state->ranged_mobs,
        level
    ) < CRAFTAX_MAX_RANGED_MOBS;

    int32_t new_type = craftax_spawn_floor_mob_type(level, CRAFTAX_MOB_RANGED);
    int32_t boss_type = craftax_spawn_floor_mob_type(
        state->boss_progress,
        CRAFTAX_MOB_RANGED
    );
    new_type = fighting_boss ? boss_type : new_type;

    CraftaxThreefryKey draw_key = craftax_spawn_next_random_key(rng);
    can_spawn = can_spawn
        && craftax_threefry_uniform_f32(draw_key)
            < craftax_spawn_floor_spawn_chance(level, 2)
                * (float)monster_spawn_coeff;

    bool valid[CRAFTAX_MAP_SIZE][CRAFTAX_MAP_SIZE];
    int32_t valid_count = craftax_spawn_fill_ranged_map(
        state,
        level,
        new_type,
        fighting_boss,
        valid
    );
    can_spawn = can_spawn && valid_count > 0;

    draw_key = craftax_spawn_next_random_key(rng);
    int32_t candidate_position[2];
    craftax_spawn_choose_position(draw_key, valid, candidate_position);

    int32_t new_index = craftax_spawn_first_empty_mobs2(
        &state->ranged_mobs,
        level
    );
    int32_t new_position[2] = {
        can_spawn
            ? candidate_position[0]
            : state->ranged_mobs.position[level][new_index][0],
        can_spawn
            ? candidate_position[1]
            : state->ranged_mobs.position[level][new_index][1],
    };
    float new_health = can_spawn
        ? craftax_spawn_mob_type_health(new_type, CRAFTAX_MOB_RANGED)
        : state->ranged_mobs.health[level][new_index];
    bool new_mask = can_spawn
        ? true
        : state->ranged_mobs.mask[level][new_index];

    state->ranged_mobs.position[level][new_index][0] = new_position[0];
    state->ranged_mobs.position[level][new_index][1] = new_position[1];
    state->ranged_mobs.health[level][new_index] = new_health;
    state->ranged_mobs.mask[level][new_index] = new_mask;
    state->ranged_mobs.type_id[level][new_index] = new_type;

    craftax_spawn_or_mob_map(
        state,
        level,
        new_position[0],
        new_position[1],
        new_mask
    );
}

static inline void craftax_spawn_mobs_native(
    CraftaxState* state,
    CraftaxThreefryKey rng
) {
    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    bool fighting_boss = craftax_step_is_fighting_boss(state);
    int32_t monster_spawn_coeff =
        1
        + (int32_t)(
            state->monsters_killed[level] < CRAFTAX_MONSTERS_KILLED_TO_CLEAR_LEVEL
        ) * 2;

    bool boss_spawn_wave =
        fighting_boss && state->boss_timesteps_to_spawn_this_round >= 1;
    if (fighting_boss) {
        monster_spawn_coeff *= (int32_t)boss_spawn_wave * 1000;
    }

    craftax_spawn_passive_mob(state, &rng, level, fighting_boss);
    craftax_spawn_melee_mob(
        state,
        &rng,
        level,
        fighting_boss,
        monster_spawn_coeff
    );
    craftax_spawn_ranged_mob(
        state,
        &rng,
        level,
        fighting_boss,
        monster_spawn_coeff
    );
}
