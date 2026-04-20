// Craftax spawn_mobs, optimized for CPU.
//
// Bitwise-equivalent to the prior JAX-transliterated baseline (verified by
// ocean/craftax_exp/parity_vs_baseline.c over 1.28M paired steps), ~6-9x
// faster per step by stripping JAX-isms:
//   - full-grid validity masks -> compact coord list collected in one pass
//   - bounding-box scan (only cells within MOB_DESPAWN_DISTANCE)
//   - early return on mob-cap / probability-roll failure (no dead writes)
//   - merged count + first_empty loops
//
// The prior reference implementation is archived at
// ocean/craftax_exp/step_spawn_mobs_baseline.h.

#pragma once

#include "step_medium.h"

#define CRAFTAX_SPAWN_MAP_CELLS (CRAFTAX_MAP_SIZE * CRAFTAX_MAP_SIZE)
#define CRAFTAX_SPAWN_BBOX_MAX_CELLS 729  // (2*DESPAWN-1)^2 at 14 = 27*27

static inline CraftaxThreefryKey craftax_spawn_next_random_key(
    CraftaxThreefryKey* rng
) {
    CraftaxThreefryKey draw;
    craftax_threefry_split(*rng, rng, &draw);
    return draw;
}

static inline int32_t craftax_spawn_floor_mob_type(
    int32_t floor, int32_t mob_class
) {
    static const int32_t mapping[CRAFTAX_NUM_LEVELS][3] = {
        {0, 0, 0}, {2, 2, 2}, {1, 1, 1}, {2, 3, 3}, {2, 4, 4},
        {1, 5, 5}, {1, 6, 6}, {1, 7, 7}, {0, 0, 0},
    };
    int32_t level = craftax_step_jax_index(floor, CRAFTAX_NUM_LEVELS);
    int32_t class_index = craftax_step_jax_index(mob_class, 3);
    return mapping[level][class_index];
}

static inline float craftax_spawn_floor_spawn_chance(
    int32_t floor, int32_t chance_index
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
    int32_t mob_type, int32_t mob_class
) {
    static const float health[CRAFTAX_NUM_MOB_TYPES][4] = {
        {3.0f, 5.0f, 3.0f, 0.0f}, {4.0f, 7.0f, 5.0f, 0.0f},
        {6.0f, 9.0f, 6.0f, 0.0f}, {8.0f, 11.0f, 8.0f, 0.0f},
        {0.0f, 12.0f, 12.0f, 0.0f}, {0.0f, 20.0f, 4.0f, 0.0f},
        {0.0f, 20.0f, 14.0f, 0.0f}, {0.0f, 24.0f, 16.0f, 0.0f},
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
    const CraftaxState* state, int32_t row, int32_t col
) {
    int32_t dr = row - state->player_position[0];
    int32_t dc = col - state->player_position[1];
    if (dr < 0) dr = -dr;
    if (dc < 0) dc = -dc;
    return dr * dr + dc * dc;
}

static inline int32_t craftax_spawn_count_mobs3(
    const CraftaxMobs3* mobs, int32_t level
) {
    int32_t count = 0;
    for (int32_t i = 0; i < 3; i++) count += (int32_t)mobs->mask[level][i];
    return count;
}

static inline int32_t craftax_spawn_count_mobs2(
    const CraftaxMobs2* mobs, int32_t level
) {
    int32_t count = 0;
    for (int32_t i = 0; i < 2; i++) count += (int32_t)mobs->mask[level][i];
    return count;
}

static inline int32_t craftax_spawn_first_empty_mobs3(
    const CraftaxMobs3* mobs, int32_t level
) {
    for (int32_t i = 0; i < 3; i++) if (!mobs->mask[level][i]) return i;
    return 0;
}

static inline int32_t craftax_spawn_first_empty_mobs2(
    const CraftaxMobs2* mobs, int32_t level
) {
    for (int32_t i = 0; i < 2; i++) if (!mobs->mask[level][i]) return i;
    return 0;
}

static inline void craftax_spawn_mobs3_count_and_empty(
    const CraftaxMobs3* mobs, int32_t level,
    int32_t* count_out, int32_t* first_empty_out
) {
    int32_t count = 0, first_empty = 0;
    bool found = false;
    for (int32_t i = 0; i < 3; i++) {
        bool m = mobs->mask[level][i];
        count += (int32_t)m;
        if (!m && !found) { first_empty = i; found = true; }
    }
    *count_out = count;
    *first_empty_out = first_empty;
}

static inline void craftax_spawn_mobs2_count_and_empty(
    const CraftaxMobs2* mobs, int32_t level,
    int32_t* count_out, int32_t* first_empty_out
) {
    int32_t count = 0, first_empty = 0;
    bool found = false;
    for (int32_t i = 0; i < 2; i++) {
        bool m = mobs->mask[level][i];
        count += (int32_t)m;
        if (!m && !found) { first_empty = i; found = true; }
    }
    *count_out = count;
    *first_empty_out = first_empty;
}

// Baseline algorithm on a bool mask:
//   draw = valid_count * (1.0 - uniform_f32(key));
//   cum = 0;
//   for i: if valid[i] { cum += 1.0; if (cum >= draw) return i; }
// Over a compact list of length valid_count this collapses to a short loop
// using the same FP arithmetic, preserving bitwise-identical choice.
static inline int32_t craftax_spawn_pick_kth(
    int32_t valid_count, CraftaxThreefryKey key
) {
    float draw = (float)valid_count * (1.0f - craftax_threefry_uniform_f32(key));
    float cum = 0.0f;
    for (int32_t k = 0; k < valid_count; k++) {
        cum += 1.0f;
        if (cum >= draw) return k;
    }
    return valid_count - 1;
}

typedef struct { int16_t row, col; } CraftaxSpawnCoord;

static inline bool craftax_spawn_scan_passive(
    const CraftaxState* state, int32_t level, CraftaxThreefryKey pos_key,
    int32_t* out_row, int32_t* out_col
) {
    int32_t pr = state->player_position[0];
    int32_t pc = state->player_position[1];
    int32_t r0 = pr - (CRAFTAX_MOB_DESPAWN_DISTANCE - 1);
    int32_t r1 = pr + (CRAFTAX_MOB_DESPAWN_DISTANCE - 1);
    int32_t c0 = pc - (CRAFTAX_MOB_DESPAWN_DISTANCE - 1);
    int32_t c1 = pc + (CRAFTAX_MOB_DESPAWN_DISTANCE - 1);
    if (r0 < 0) r0 = 0;
    if (c0 < 0) c0 = 0;
    if (r1 > CRAFTAX_MAP_SIZE - 1) r1 = CRAFTAX_MAP_SIZE - 1;
    if (c1 > CRAFTAX_MAP_SIZE - 1) c1 = CRAFTAX_MAP_SIZE - 1;

    const int32_t limit2 = CRAFTAX_MOB_DESPAWN_DISTANCE
                         * CRAFTAX_MOB_DESPAWN_DISTANCE;
    CraftaxSpawnCoord coords[CRAFTAX_SPAWN_BBOX_MAX_CELLS];
    int32_t n = 0;
    for (int32_t row = r0; row <= r1; row++) {
        int32_t dr = row - pr; if (dr < 0) dr = -dr;
        int32_t dr2 = dr * dr;
        const int32_t* map_row = state->map[level][row];
        const bool*    mob_row = state->mob_map[level][row];
        for (int32_t col = c0; col <= c1; col++) {
            int32_t dc = col - pc; if (dc < 0) dc = -dc;
            int32_t distance2 = dr2 + dc * dc;
            if (distance2 <= 9 || distance2 >= limit2) continue;
            if (mob_row[col]) continue;
            int32_t block = map_row[col];
            if (block != CRAFTAX_BLOCK_GRASS && block != CRAFTAX_BLOCK_PATH
                && block != CRAFTAX_BLOCK_FIRE_GRASS
                && block != CRAFTAX_BLOCK_ICE_GRASS) continue;
            coords[n].row = (int16_t)row;
            coords[n].col = (int16_t)col;
            n++;
        }
    }
    if (n == 0) return false;
    int32_t k = craftax_spawn_pick_kth(n, pos_key);
    *out_row = coords[k].row;
    *out_col = coords[k].col;
    return true;
}

static inline bool craftax_spawn_scan_melee(
    const CraftaxState* state, int32_t level, bool fighting_boss,
    CraftaxThreefryKey pos_key, int32_t* out_row, int32_t* out_col
) {
    int32_t pr = state->player_position[0];
    int32_t pc = state->player_position[1];
    int32_t r0 = pr - (CRAFTAX_MOB_DESPAWN_DISTANCE - 1);
    int32_t r1 = pr + (CRAFTAX_MOB_DESPAWN_DISTANCE - 1);
    int32_t c0 = pc - (CRAFTAX_MOB_DESPAWN_DISTANCE - 1);
    int32_t c1 = pc + (CRAFTAX_MOB_DESPAWN_DISTANCE - 1);
    if (r0 < 0) r0 = 0;
    if (c0 < 0) c0 = 0;
    if (r1 > CRAFTAX_MAP_SIZE - 1) r1 = CRAFTAX_MAP_SIZE - 1;
    if (c1 > CRAFTAX_MAP_SIZE - 1) c1 = CRAFTAX_MAP_SIZE - 1;

    const int32_t limit2 = CRAFTAX_MOB_DESPAWN_DISTANCE
                         * CRAFTAX_MOB_DESPAWN_DISTANCE;
    CraftaxSpawnCoord coords[CRAFTAX_SPAWN_BBOX_MAX_CELLS];
    int32_t n = 0;
    for (int32_t row = r0; row <= r1; row++) {
        int32_t dr = row - pr; if (dr < 0) dr = -dr;
        int32_t dr2 = dr * dr;
        const int32_t* map_row = state->map[level][row];
        const bool*    mob_row = state->mob_map[level][row];
        for (int32_t col = c0; col <= c1; col++) {
            int32_t dc = col - pc; if (dc < 0) dc = -dc;
            int32_t distance2 = dr2 + dc * dc;
            if (distance2 >= limit2) continue;
            bool range_ok = fighting_boss ? (distance2 <= 36) : (distance2 > 81);
            if (!range_ok) continue;
            if (mob_row[col]) continue;
            int32_t block = map_row[col];
            bool terrain_ok;
            if (fighting_boss) {
                terrain_ok = (block == CRAFTAX_BLOCK_GRAVE
                           || block == CRAFTAX_BLOCK_GRAVE2
                           || block == CRAFTAX_BLOCK_GRAVE3);
            } else {
                terrain_ok = (block == CRAFTAX_BLOCK_GRASS
                           || block == CRAFTAX_BLOCK_PATH
                           || block == CRAFTAX_BLOCK_FIRE_GRASS
                           || block == CRAFTAX_BLOCK_ICE_GRASS);
            }
            if (!terrain_ok) continue;
            coords[n].row = (int16_t)row;
            coords[n].col = (int16_t)col;
            n++;
        }
    }
    if (n == 0) return false;
    int32_t k = craftax_spawn_pick_kth(n, pos_key);
    *out_row = coords[k].row;
    *out_col = coords[k].col;
    return true;
}

static inline bool craftax_spawn_scan_ranged(
    const CraftaxState* state, int32_t level, int32_t new_type,
    bool fighting_boss, CraftaxThreefryKey pos_key,
    int32_t* out_row, int32_t* out_col
) {
    int32_t pr = state->player_position[0];
    int32_t pc = state->player_position[1];
    int32_t r0 = pr - (CRAFTAX_MOB_DESPAWN_DISTANCE - 1);
    int32_t r1 = pr + (CRAFTAX_MOB_DESPAWN_DISTANCE - 1);
    int32_t c0 = pc - (CRAFTAX_MOB_DESPAWN_DISTANCE - 1);
    int32_t c1 = pc + (CRAFTAX_MOB_DESPAWN_DISTANCE - 1);
    if (r0 < 0) r0 = 0;
    if (c0 < 0) c0 = 0;
    if (r1 > CRAFTAX_MAP_SIZE - 1) r1 = CRAFTAX_MAP_SIZE - 1;
    if (c1 > CRAFTAX_MAP_SIZE - 1) c1 = CRAFTAX_MAP_SIZE - 1;

    const int32_t limit2 = CRAFTAX_MOB_DESPAWN_DISTANCE
                         * CRAFTAX_MOB_DESPAWN_DISTANCE;
    CraftaxSpawnCoord coords[CRAFTAX_SPAWN_BBOX_MAX_CELLS];
    int32_t n = 0;
    bool water_type = (new_type == 5);
    for (int32_t row = r0; row <= r1; row++) {
        int32_t dr = row - pr; if (dr < 0) dr = -dr;
        int32_t dr2 = dr * dr;
        const int32_t* map_row = state->map[level][row];
        const bool*    mob_row = state->mob_map[level][row];
        for (int32_t col = c0; col <= c1; col++) {
            int32_t dc = col - pc; if (dc < 0) dc = -dc;
            int32_t distance2 = dr2 + dc * dc;
            if (distance2 >= limit2) continue;
            bool range_ok = fighting_boss ? (distance2 <= 36) : (distance2 > 81);
            if (!range_ok) continue;
            if (mob_row[col]) continue;
            int32_t block = map_row[col];
            bool terrain_ok;
            if (fighting_boss) {
                terrain_ok = (block == CRAFTAX_BLOCK_GRAVE
                           || block == CRAFTAX_BLOCK_GRAVE2
                           || block == CRAFTAX_BLOCK_GRAVE3);
            } else if (water_type) {
                terrain_ok = (block == CRAFTAX_BLOCK_WATER);
            } else {
                terrain_ok = (block == CRAFTAX_BLOCK_GRASS
                           || block == CRAFTAX_BLOCK_PATH
                           || block == CRAFTAX_BLOCK_FIRE_GRASS
                           || block == CRAFTAX_BLOCK_ICE_GRASS);
            }
            if (!terrain_ok) continue;
            coords[n].row = (int16_t)row;
            coords[n].col = (int16_t)col;
            n++;
        }
    }
    if (n == 0) return false;
    int32_t k = craftax_spawn_pick_kth(n, pos_key);
    *out_row = coords[k].row;
    *out_col = coords[k].col;
    return true;
}

// Both RNG keys are always consumed (preserves baseline RNG sequence).
// Baseline quirk: type_id[level][slot] is written unconditionally, even
// when no mob spawns. We match that for bitwise parity.

static inline void craftax_spawn_passive_mob(
    CraftaxState* state, CraftaxThreefryKey* rng,
    int32_t level, bool fighting_boss
) {
    int32_t count, slot;
    craftax_spawn_mobs3_count_and_empty(&state->passive_mobs, level, &count, &slot);

    CraftaxThreefryKey prob_key = craftax_spawn_next_random_key(rng);
    CraftaxThreefryKey pos_key  = craftax_spawn_next_random_key(rng);

    int32_t type = craftax_spawn_floor_mob_type(level, CRAFTAX_MOB_PASSIVE);
    state->passive_mobs.type_id[level][slot] = type;

    if (fighting_boss) return;
    if (count >= CRAFTAX_MAX_PASSIVE_MOBS) return;
    if (craftax_threefry_uniform_f32(prob_key)
        >= craftax_spawn_floor_spawn_chance(level, 0)) return;

    int32_t row, col;
    if (!craftax_spawn_scan_passive(state, level, pos_key, &row, &col)) return;

    state->passive_mobs.position[level][slot][0] = row;
    state->passive_mobs.position[level][slot][1] = col;
    state->passive_mobs.health[level][slot]      =
        craftax_spawn_mob_type_health(type, CRAFTAX_MOB_PASSIVE);
    state->passive_mobs.mask[level][slot]        = true;
    state->mob_map[level][row][col] = true;
}

static inline void craftax_spawn_melee_mob(
    CraftaxState* state, CraftaxThreefryKey* rng,
    int32_t level, bool fighting_boss, int32_t monster_spawn_coeff
) {
    int32_t count, slot;
    craftax_spawn_mobs3_count_and_empty(&state->melee_mobs, level, &count, &slot);

    int32_t type = fighting_boss
        ? craftax_spawn_floor_mob_type(state->boss_progress, CRAFTAX_MOB_MELEE)
        : craftax_spawn_floor_mob_type(level, CRAFTAX_MOB_MELEE);

    CraftaxThreefryKey prob_key = craftax_spawn_next_random_key(rng);
    float night_coeff = 1.0f - state->light_level;
    float spawn_chance = craftax_spawn_floor_spawn_chance(level, 1)
        + craftax_spawn_floor_spawn_chance(level, 3) * night_coeff * night_coeff;
    CraftaxThreefryKey pos_key = craftax_spawn_next_random_key(rng);

    state->melee_mobs.type_id[level][slot] = type;

    if (count >= CRAFTAX_MAX_MELEE_MOBS) return;
    if (craftax_threefry_uniform_f32(prob_key)
        >= spawn_chance * (float)monster_spawn_coeff) return;

    int32_t row, col;
    if (!craftax_spawn_scan_melee(state, level, fighting_boss, pos_key, &row, &col))
        return;

    state->melee_mobs.position[level][slot][0] = row;
    state->melee_mobs.position[level][slot][1] = col;
    state->melee_mobs.health[level][slot]      =
        craftax_spawn_mob_type_health(type, CRAFTAX_MOB_MELEE);
    state->melee_mobs.mask[level][slot]        = true;
    state->mob_map[level][row][col] = true;
}

static inline void craftax_spawn_ranged_mob(
    CraftaxState* state, CraftaxThreefryKey* rng,
    int32_t level, bool fighting_boss, int32_t monster_spawn_coeff
) {
    int32_t count, slot;
    craftax_spawn_mobs2_count_and_empty(&state->ranged_mobs, level, &count, &slot);

    int32_t type = fighting_boss
        ? craftax_spawn_floor_mob_type(state->boss_progress, CRAFTAX_MOB_RANGED)
        : craftax_spawn_floor_mob_type(level, CRAFTAX_MOB_RANGED);

    CraftaxThreefryKey prob_key = craftax_spawn_next_random_key(rng);
    CraftaxThreefryKey pos_key  = craftax_spawn_next_random_key(rng);

    state->ranged_mobs.type_id[level][slot] = type;

    if (count >= CRAFTAX_MAX_RANGED_MOBS) return;
    if (craftax_threefry_uniform_f32(prob_key)
        >= craftax_spawn_floor_spawn_chance(level, 2) * (float)monster_spawn_coeff)
        return;

    int32_t row, col;
    if (!craftax_spawn_scan_ranged(state, level, type, fighting_boss, pos_key,
                                    &row, &col)) return;

    state->ranged_mobs.position[level][slot][0] = row;
    state->ranged_mobs.position[level][slot][1] = col;
    state->ranged_mobs.health[level][slot]      =
        craftax_spawn_mob_type_health(type, CRAFTAX_MOB_RANGED);
    state->ranged_mobs.mask[level][slot]        = true;
    state->mob_map[level][row][col] = true;
}

static inline void craftax_spawn_mobs_native(
    CraftaxState* state, CraftaxThreefryKey rng
) {
    int32_t level = craftax_step_jax_index(
        state->player_level, CRAFTAX_NUM_LEVELS
    );
    bool fighting_boss = craftax_step_is_fighting_boss(state);
    int32_t monster_spawn_coeff =
        1
        + (int32_t)(state->monsters_killed[level]
                    < CRAFTAX_MONSTERS_KILLED_TO_CLEAR_LEVEL) * 2;

    bool boss_spawn_wave =
        fighting_boss && state->boss_timesteps_to_spawn_this_round >= 1;
    if (fighting_boss) {
        monster_spawn_coeff *= (int32_t)boss_spawn_wave * 1000;
    }

    craftax_spawn_passive_mob(state, &rng, level, fighting_boss);
    craftax_spawn_melee_mob(state, &rng, level, fighting_boss, monster_spawn_coeff);
    craftax_spawn_ranged_mob(state, &rng, level, fighting_boss, monster_spawn_coeff);
}
