// Standalone native port of Craftax update_mobs.
//
// This helper intentionally is not integrated into c_step yet. It mutates a
// full CraftaxState in place so tests can compare the subsystem directly
// against the installed JAX implementation.

#pragma once

#include "step_do_action.h"

#define CRAFTAX_UPDATE_BOSS_FIGHT_EXTRA_DAMAGE 0.5f

static inline CraftaxThreefryKey craftax_update_mobs_next_random_key(
    CraftaxThreefryKey* rng
) {
    CraftaxThreefryKey draw;
    craftax_threefry_split(*rng, rng, &draw);
    return draw;
}

static inline bool craftax_update_mobs_scatter_index(
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

static inline bool craftax_update_mobs_in_bounds(
    int32_t row,
    int32_t col
) {
    return row >= 0
        && row < CRAFTAX_MAP_SIZE
        && col >= 0
        && col < CRAFTAX_MAP_SIZE;
}

static inline int32_t craftax_update_mobs_read_block(
    const CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col
) {
    int32_t map_level = craftax_step_jax_index(level, CRAFTAX_NUM_LEVELS);
    int32_t map_row = craftax_step_jax_index(row, CRAFTAX_MAP_SIZE);
    int32_t map_col = craftax_step_jax_index(col, CRAFTAX_MAP_SIZE);
    return state->map[map_level][map_row][map_col];
}

static inline void craftax_update_mobs_set_block(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col,
    int32_t block
) {
    int32_t map_level;
    int32_t map_row;
    int32_t map_col;
    if (!craftax_update_mobs_scatter_index(
            level,
            CRAFTAX_NUM_LEVELS,
            &map_level
        )
        || !craftax_update_mobs_scatter_index(
            row,
            CRAFTAX_MAP_SIZE,
            &map_row
        )
        || !craftax_update_mobs_scatter_index(
            col,
            CRAFTAX_MAP_SIZE,
            &map_col
        )) {
        return;
    }
    craftax_set_map_block(state, map_level, map_row, map_col, block);
}

static inline bool craftax_update_mobs_read_mob_map(
    const CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col
) {
    int32_t map_level = craftax_step_jax_index(level, CRAFTAX_NUM_LEVELS);
    int32_t map_row = craftax_step_jax_index(row, CRAFTAX_MAP_SIZE);
    int32_t map_col = craftax_step_jax_index(col, CRAFTAX_MAP_SIZE);
    return (state->mob_bits[map_level][map_row] >> map_col) & 1ULL;
}

static inline void craftax_update_mobs_set_mob_map(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col,
    bool value
) {
    int32_t map_level;
    int32_t map_row;
    int32_t map_col;
    if (!craftax_update_mobs_scatter_index(
            level,
            CRAFTAX_NUM_LEVELS,
            &map_level
        )
        || !craftax_update_mobs_scatter_index(
            row,
            CRAFTAX_MAP_SIZE,
            &map_row
        )
        || !craftax_update_mobs_scatter_index(
            col,
            CRAFTAX_MAP_SIZE,
            &map_col
        )) {
        return;
    }
    if (value) {
        state->mob_bits[map_level][map_row] |= (1ULL << map_col);
    } else {
        state->mob_bits[map_level][map_row] &= ~(1ULL << map_col);
    }
}

static inline void craftax_update_mobs_clear_old_map_entry(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col,
    bool old_mask
) {
    bool old_value = craftax_update_mobs_read_mob_map(state, level, row, col);
    craftax_update_mobs_set_mob_map(
        state,
        level,
        row,
        col,
        old_value && !old_mask
    );
}

static inline void craftax_update_mobs_enter_new_map_entry(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col,
    bool new_mask
) {
    bool old_value = craftax_update_mobs_read_mob_map(state, level, row, col);
    craftax_update_mobs_set_mob_map(
        state,
        level,
        row,
        col,
        old_value || new_mask
    );
}

static inline void craftax_update_mobs_damage_vector(
    int32_t type_id,
    int32_t mob_class_index,
    float damage[3]
) {
    static const float damages[CRAFTAX_NUM_MOB_TYPES][4][3] = {
        {
            {0.0f, 0.0f, 0.0f},
            {2.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {2.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {4.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {4.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {3.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 3.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {5.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 3.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {6.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {5.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {6.0f, 1.0f, 1.0f},
            {0.0f, 0.0f, 0.0f},
            {4.0f, 3.0f, 3.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {3.0f, 5.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {3.0f, 5.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {4.0f, 0.0f, 5.0f},
            {0.0f, 0.0f, 0.0f},
            {4.0f, 0.0f, 5.0f},
        },
    };

    int32_t type_index = craftax_step_jax_index(
        type_id,
        CRAFTAX_NUM_MOB_TYPES
    );
    int32_t class_index = craftax_step_jax_index(mob_class_index, 4);
    for (int32_t i = 0; i < 3; i++) {
        damage[i] = damages[type_index][class_index][i];
    }
}

static inline void craftax_update_mobs_collision_map(
    int32_t type_id,
    int32_t mob_class_index,
    bool collision[3]
) {
    static const bool collisions[CRAFTAX_NUM_MOB_TYPES][4][3] = {
        {
            {false, true, true},
            {false, true, true},
            {false, true, true},
            {false, false, false},
        },
        {
            {false, false, false},
            {false, true, true},
            {false, true, true},
            {false, false, false},
        },
        {
            {false, true, true},
            {false, true, true},
            {false, true, true},
            {false, false, false},
        },
        {
            {false, true, true},
            {false, false, true},
            {false, true, true},
            {false, false, false},
        },
        {
            {false, true, true},
            {false, true, true},
            {false, true, true},
            {false, false, false},
        },
        {
            {false, true, true},
            {false, true, true},
            {true, false, true},
            {false, false, false},
        },
        {
            {false, true, true},
            {false, true, true},
            {false, false, false},
            {false, false, false},
        },
        {
            {false, true, true},
            {false, true, true},
            {false, false, false},
            {false, false, false},
        },
    };

    int32_t type_index = craftax_step_jax_index(
        type_id,
        CRAFTAX_NUM_MOB_TYPES
    );
    int32_t class_index = craftax_step_jax_index(mob_class_index, 4);
    for (int32_t i = 0; i < 3; i++) {
        collision[i] = collisions[type_index][class_index][i];
    }
}

static inline int32_t craftax_update_mobs_projectile_type_for_ranged(
    int32_t ranged_type
) {
    static const int32_t mapping[CRAFTAX_NUM_MOB_TYPES] = {
        CRAFTAX_PROJECTILE_ARROW,
        CRAFTAX_PROJECTILE_ARROW,
        CRAFTAX_PROJECTILE_FIREBALL,
        CRAFTAX_PROJECTILE_DAGGER,
        CRAFTAX_PROJECTILE_ARROW2,
        CRAFTAX_PROJECTILE_SLIMEBALL,
        CRAFTAX_PROJECTILE_FIREBALL2,
        CRAFTAX_PROJECTILE_ICEBALL2,
    };
    int32_t type_index = craftax_step_jax_index(
        ranged_type,
        CRAFTAX_NUM_MOB_TYPES
    );
    return mapping[type_index];
}

static inline void craftax_update_mobs_direction_choice(
    CraftaxThreefryKey key,
    int32_t count,
    int32_t direction[2]
) {
    int32_t choice = craftax_medium_randint(key, 0, count);
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

static inline int32_t craftax_update_mobs_abs_i32(int32_t value) {
    return value < 0 ? -value : value;
}

static inline int32_t craftax_update_mobs_sign_i32(int32_t value) {
    if (value < 0) {
        return -1;
    }
    return value > 0 ? 1 : 0;
}

static inline int32_t craftax_update_mobs_player_axis_choice(
    CraftaxThreefryKey key,
    int32_t distance_row,
    int32_t distance_col
) {
    int32_t max_distance = distance_row > distance_col
        ? distance_row
        : distance_col;
    int32_t total_distance = distance_row + distance_col;
    if (total_distance == 0) {
        return 1;
    }

    float weights[2] = {
        (distance_row == max_distance) ? 1.0f / (float)total_distance : 0.0f,
        (distance_col == max_distance) ? 1.0f / (float)total_distance : 0.0f,
    };
    return craftax_medium_choice_weighted(key, weights, 2);
}

static inline bool craftax_update_mobs_valid_position(
    const CraftaxState* state,
    int32_t row,
    int32_t col,
    const bool collision[3]
) {
    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    bool pos_in_bounds = craftax_update_mobs_in_bounds(row, col);
    int32_t block = craftax_update_mobs_read_block(state, level, row, col);
    bool in_solid_block = craftax_step_is_solid_block(block);
    bool in_mob = craftax_step_is_in_mob(state, row, col);
    bool in_lava = block == CRAFTAX_BLOCK_LAVA;
    bool in_water = block == CRAFTAX_BLOCK_WATER;
    bool on_ground_block = !in_solid_block && !in_water && !in_lava;

    bool valid_move = pos_in_bounds && !in_mob && !in_solid_block;
    valid_move = valid_move && (!collision[0] || !on_ground_block);
    valid_move = valid_move && (!collision[1] || !in_water);
    valid_move = valid_move && (!collision[2] || !in_lava);
    return valid_move;
}

static inline int32_t craftax_update_mobs_manhattan_to_player(
    const CraftaxState* state,
    int32_t row,
    int32_t col
) {
    return craftax_update_mobs_abs_i32(row - state->player_position[0])
        + craftax_update_mobs_abs_i32(col - state->player_position[1]);
}

static inline float craftax_update_mobs_damage_done_to_player(
    const CraftaxState* state,
    const float damage_vector[3]
) {
    float defense_vector[3] = {0.0f, 0.0f, 0.0f};
    for (int32_t i = 0; i < 4; i++) {
        defense_vector[0] += (float)state->inventory.armour[i] * 0.1f;
        defense_vector[1] +=
            (float)(int32_t)(state->armour_enchantments[i] == 1) * 0.2f;
        defense_vector[2] +=
            (float)(int32_t)(state->armour_enchantments[i] == 2) * 0.2f;
    }

    float boss_coeff = craftax_step_is_fighting_boss(state)
        ? 1.0f + CRAFTAX_UPDATE_BOSS_FIGHT_EXTRA_DAMAGE
        : 1.0f;
    float damage = 0.0f;
    for (int32_t i = 0; i < 3; i++) {
        damage += (1.0f - defense_vector[i]) * damage_vector[i] * boss_coeff;
    }
    return damage;
}

static inline int32_t craftax_update_mobs_count_mob_projectiles(
    const CraftaxState* state,
    int32_t level
) {
    const bool* mask = state->mob_projectiles.mask[level];
    return (int32_t)mask[0] + (int32_t)mask[1] + (int32_t)mask[2];
}

static inline int32_t craftax_update_mobs_first_empty_mob_projectile(
    const CraftaxState* state,
    int32_t level
) {
    const bool* mask = state->mob_projectiles.mask[level];
    if (!mask[0]) return 0;
    if (!mask[1]) return 1;
    if (!mask[2]) return 2;
    return 0;
}

static inline void craftax_update_mobs_spawn_mob_projectile(
    CraftaxState* state,
    int32_t level,
    bool is_spawning_projectile,
    const int32_t position[2],
    const int32_t direction[2],
    int32_t projectile_type
) {
    if (!is_spawning_projectile) {
        return;
    }

    int32_t index = craftax_update_mobs_first_empty_mob_projectile(
        state,
        level
    );
    state->mob_projectiles.position[level][index][0] = position[0];
    state->mob_projectiles.position[level][index][1] = position[1];
    state->mob_projectiles.mask[level][index] = true;
    state->mob_projectiles.type_id[level][index] = projectile_type;
    state->mob_projectile_directions[level][index][0] = direction[0];
    state->mob_projectile_directions[level][index][1] = direction[1];
}

static inline void craftax_update_mobs_attack_mob_with_damage(
    CraftaxState* state,
    int32_t row,
    int32_t col,
    const float damage_vector[3],
    bool can_eat,
    bool* did_attack_mob,
    bool* did_kill_mob
) {
    bool did_kill_melee_mob = false;
    bool is_attacking_melee_mob = false;
    craftax_do_action_attack_mobs3(
        state,
        &state->melee_mobs,
        row,
        col,
        damage_vector,
        true,
        CRAFTAX_MOB_MELEE,
        &did_kill_melee_mob,
        &is_attacking_melee_mob
    );

    bool did_kill_passive_mob = false;
    bool is_attacking_passive_mob = false;
    craftax_do_action_attack_mobs3(
        state,
        &state->passive_mobs,
        row,
        col,
        damage_vector,
        can_eat,
        CRAFTAX_MOB_PASSIVE,
        &did_kill_passive_mob,
        &is_attacking_passive_mob
    );

    if (did_kill_passive_mob && can_eat) {
        state->player_food = craftax_step_mini32(
            craftax_step_get_max_food(state),
            state->player_food + 6
        );
        state->player_hunger = 0.0f;
    }

    bool did_kill_ranged_mob = false;
    bool is_attacking_ranged_mob = false;
    craftax_do_action_attack_mobs2(
        state,
        &state->ranged_mobs,
        row,
        col,
        damage_vector,
        true,
        CRAFTAX_MOB_RANGED,
        &did_kill_ranged_mob,
        &is_attacking_ranged_mob
    );

    *did_attack_mob = is_attacking_melee_mob
        || is_attacking_passive_mob
        || is_attacking_ranged_mob;
    bool did_kill_monster = did_kill_melee_mob || did_kill_ranged_mob;
    *did_kill_mob = did_kill_monster || did_kill_passive_mob;

    craftax_do_action_update_mob_map(state, row, col, *did_kill_mob);

    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    state->monsters_killed[level] += (int32_t)did_kill_monster;
}

static inline void craftax_update_mobs_player_projectile_damage_vector(
    const CraftaxState* state,
    int32_t level,
    int32_t projectile_index,
    float damage_vector[3]
) {
    int32_t projectile_type =
        state->player_projectiles.type_id[level][projectile_index];
    craftax_update_mobs_damage_vector(
        projectile_type,
        CRAFTAX_MOB_PROJECTILE,
        damage_vector
    );

    float mask = (float)(int32_t)
        state->player_projectiles.mask[level][projectile_index];
    for (int32_t i = 0; i < 3; i++) {
        damage_vector[i] *= mask;
    }

    bool is_arrow = projectile_type == CRAFTAX_PROJECTILE_ARROW
        || projectile_type == CRAFTAX_PROJECTILE_ARROW2;
    if (is_arrow) {
        float arrow_damage_add[3] = {0.0f, 0.0f, 0.0f};
        int32_t enchantment_index;
        if (craftax_update_mobs_scatter_index(
                state->bow_enchantment,
                3,
                &enchantment_index
            )) {
            arrow_damage_add[enchantment_index] = damage_vector[0] / 2.0f;
        }
        arrow_damage_add[0] = 0.0f;
        for (int32_t i = 0; i < 3; i++) {
            damage_vector[i] += arrow_damage_add[i];
        }
    }

    if (is_arrow) {
        float arrow_damage_coeff =
            1.0f + 0.2f * (float)(state->player_dexterity - 1);
        for (int32_t i = 0; i < 3; i++) {
            damage_vector[i] *= arrow_damage_coeff;
        }
    }

    bool is_magic_projectile = projectile_type == CRAFTAX_PROJECTILE_FIREBALL
        || projectile_type == CRAFTAX_PROJECTILE_ICEBALL;
    if (is_magic_projectile) {
        float magic_damage_coeff =
            1.0f + 0.5f * (float)(state->player_intelligence - 1);
        for (int32_t i = 0; i < 3; i++) {
            damage_vector[i] *= magic_damage_coeff;
        }
    }
}

static inline void craftax_update_mobs_move_melee(
    CraftaxState* state,
    CraftaxThreefryKey* rng,
    int32_t index
) {
    int32_t level = state->player_level;
    bool old_mask = state->melee_mobs.mask[level][index];
    // Dead slot early-out: no observable effect on obs/reward/terminal.
    // Skip body and RNG draws for speed. Breaks per-seed replay against
    // JAX; define CRAFTAX_JAX_PARITY at build time to restore the
    // branchless slow path (same pattern in every move_* below).
#ifndef CRAFTAX_JAX_PARITY
    if (!old_mask) return;
#endif
    int32_t old_row = state->melee_mobs.position[level][index][0];
    int32_t old_col = state->melee_mobs.position[level][index][1];
    int32_t old_cooldown = state->melee_mobs.attack_cooldown[level][index];
    int32_t mob_type = state->melee_mobs.type_id[level][index];

    CraftaxThreefryKey draw_key =
        craftax_update_mobs_next_random_key(rng);
    int32_t random_direction[2];
    craftax_update_mobs_direction_choice(draw_key, 4, random_direction);
    int32_t random_row = old_row + random_direction[0];
    int32_t random_col = old_col + random_direction[1];

    int32_t distance_row =
        craftax_update_mobs_abs_i32(state->player_position[0] - old_row);
    int32_t distance_col =
        craftax_update_mobs_abs_i32(state->player_position[1] - old_col);
    draw_key = craftax_update_mobs_next_random_key(rng);
    int32_t player_move_axis = craftax_update_mobs_player_axis_choice(
        draw_key,
        distance_row,
        distance_col
    );
    int32_t player_direction[2] = {0, 0};
    if (player_move_axis == 0) {
        player_direction[0] =
            craftax_update_mobs_sign_i32(state->player_position[0] - old_row);
    } else {
        player_direction[1] =
            craftax_update_mobs_sign_i32(state->player_position[1] - old_col);
    }
    int32_t player_row = old_row + player_direction[0];
    int32_t player_col = old_col + player_direction[1];

    int32_t distance_to_player = distance_row + distance_col;
    bool close_to_player = distance_to_player < 10
        || craftax_step_is_fighting_boss(state);
    draw_key = craftax_update_mobs_next_random_key(rng);
    close_to_player = close_to_player
        && craftax_threefry_uniform_f32(draw_key) < 0.75f;

    int32_t proposed_row = close_to_player ? player_row : random_row;
    int32_t proposed_col = close_to_player ? player_col : random_col;

    bool is_attacking_player = distance_to_player == 1
        && old_cooldown <= 0
        && old_mask;
    if (is_attacking_player) {
        proposed_row = old_row;
        proposed_col = old_col;
    }

    float base_damage[3];
    craftax_update_mobs_damage_vector(
        mob_type,
        CRAFTAX_MOB_MELEE,
        base_damage
    );
    float sleeping_coeff = 1.0f + 2.5f * (float)(int32_t)state->is_sleeping;
    for (int32_t i = 0; i < 3; i++) {
        base_damage[i] *= sleeping_coeff;
    }
    float damage = craftax_update_mobs_damage_done_to_player(
        state,
        base_damage
    );

    int32_t new_cooldown = is_attacking_player ? 5 : old_cooldown - 1;
    bool is_waking_player = state->is_sleeping && is_attacking_player;
    state->player_health -= damage * (float)(int32_t)is_attacking_player;
    state->is_sleeping = state->is_sleeping && !is_attacking_player;
    state->is_resting = state->is_resting && !is_attacking_player;
    state->achievements[CRAFTAX_ACH_WAKE_UP] =
        state->achievements[CRAFTAX_ACH_WAKE_UP] || is_waking_player;

    bool collision[3];
    craftax_update_mobs_collision_map(
        mob_type,
        CRAFTAX_MOB_MELEE,
        collision
    );
    bool valid_move = craftax_update_mobs_valid_position(
        state,
        proposed_row,
        proposed_col,
        collision
    );
    int32_t new_row = valid_move ? proposed_row : old_row;
    int32_t new_col = valid_move ? proposed_col : old_col;

    bool should_not_despawn = distance_to_player < CRAFTAX_MOB_DESPAWN_DISTANCE
        || craftax_step_is_fighting_boss(state);

    CraftaxThreefryKey unused_left;
    CraftaxThreefryKey returned_key;
    craftax_threefry_split(*rng, &unused_left, &returned_key);
    *rng = returned_key;

    craftax_update_mobs_clear_old_map_entry(
        state,
        level,
        old_row,
        old_col,
        old_mask
    );
    bool new_mask = old_mask && should_not_despawn;
    craftax_update_mobs_enter_new_map_entry(
        state,
        level,
        new_row,
        new_col,
        new_mask
    );

    state->melee_mobs.position[level][index][0] = new_row;
    state->melee_mobs.position[level][index][1] = new_col;
    state->melee_mobs.attack_cooldown[level][index] = new_cooldown;
    state->melee_mobs.mask[level][index] = new_mask;
}

static inline void craftax_update_mobs_move_passive(
    CraftaxState* state,
    CraftaxThreefryKey* rng,
    int32_t index
) {
    int32_t level = state->player_level;
    bool old_mask = state->passive_mobs.mask[level][index];
#ifndef CRAFTAX_JAX_PARITY
    if (!old_mask) return;
#endif
    int32_t old_row = state->passive_mobs.position[level][index][0];
    int32_t old_col = state->passive_mobs.position[level][index][1];
    int32_t mob_type = state->passive_mobs.type_id[level][index];

    CraftaxThreefryKey draw_key =
        craftax_update_mobs_next_random_key(rng);
    int32_t direction[2];
    craftax_update_mobs_direction_choice(draw_key, 8, direction);
    int32_t proposed_row = old_row + direction[0];
    int32_t proposed_col = old_col + direction[1];

    bool collision[3];
    craftax_update_mobs_collision_map(
        mob_type,
        CRAFTAX_MOB_PASSIVE,
        collision
    );
    bool valid_move = craftax_update_mobs_valid_position(
        state,
        proposed_row,
        proposed_col,
        collision
    );
    int32_t new_row = valid_move ? proposed_row : old_row;
    int32_t new_col = valid_move ? proposed_col : old_col;

    int32_t distance_to_player = craftax_update_mobs_manhattan_to_player(
        state,
        old_row,
        old_col
    );
    bool should_not_despawn =
        distance_to_player < CRAFTAX_MOB_DESPAWN_DISTANCE;

    craftax_update_mobs_clear_old_map_entry(
        state,
        level,
        old_row,
        old_col,
        old_mask
    );
    bool new_mask = old_mask && should_not_despawn;
    craftax_update_mobs_enter_new_map_entry(
        state,
        level,
        new_row,
        new_col,
        new_mask
    );

    state->passive_mobs.position[level][index][0] = new_row;
    state->passive_mobs.position[level][index][1] = new_col;
    state->passive_mobs.mask[level][index] = new_mask;
}

static inline void craftax_update_mobs_move_ranged(
    CraftaxState* state,
    CraftaxThreefryKey* rng,
    int32_t index
) {
    int32_t level = state->player_level;
    bool old_mask = state->ranged_mobs.mask[level][index];
#ifndef CRAFTAX_JAX_PARITY
    if (!old_mask) return;
#endif
    int32_t old_row = state->ranged_mobs.position[level][index][0];
    int32_t old_col = state->ranged_mobs.position[level][index][1];
    int32_t old_cooldown = state->ranged_mobs.attack_cooldown[level][index];
    int32_t mob_type = state->ranged_mobs.type_id[level][index];

    CraftaxThreefryKey draw_key =
        craftax_update_mobs_next_random_key(rng);
    int32_t random_direction[2];
    craftax_update_mobs_direction_choice(draw_key, 4, random_direction);
    int32_t random_row = old_row + random_direction[0];
    int32_t random_col = old_col + random_direction[1];

    int32_t distance_row =
        craftax_update_mobs_abs_i32(state->player_position[0] - old_row);
    int32_t distance_col =
        craftax_update_mobs_abs_i32(state->player_position[1] - old_col);
    draw_key = craftax_update_mobs_next_random_key(rng);
    int32_t player_move_axis = craftax_update_mobs_player_axis_choice(
        draw_key,
        distance_row,
        distance_col
    );
    int32_t player_direction[2] = {0, 0};
    if (player_move_axis == 0) {
        player_direction[0] =
            craftax_update_mobs_sign_i32(state->player_position[0] - old_row);
    } else {
        player_direction[1] =
            craftax_update_mobs_sign_i32(state->player_position[1] - old_col);
    }
    int32_t towards_row = old_row + player_direction[0];
    int32_t towards_col = old_col + player_direction[1];
    int32_t away_row = old_row - player_direction[0];
    int32_t away_col = old_col - player_direction[1];

    int32_t distance_to_player = distance_row + distance_col;
    bool far_from_player = distance_to_player >= 6;
    bool too_close_to_player = distance_to_player <= 3;
    int32_t proposed_row = far_from_player ? towards_row : random_row;
    int32_t proposed_col = far_from_player ? towards_col : random_col;
    if (too_close_to_player) {
        proposed_row = away_row;
        proposed_col = away_col;
    }

    draw_key = craftax_update_mobs_next_random_key(rng);
    if (!(craftax_threefry_uniform_f32(draw_key) > 0.85f)) {
        proposed_row = random_row;
        proposed_col = random_col;
    }

    bool collision[3];
    craftax_update_mobs_collision_map(
        mob_type,
        CRAFTAX_MOB_RANGED,
        collision
    );

    bool is_attacking_player =
        distance_to_player >= 4 && distance_to_player <= 5;
    bool proposed_valid = craftax_update_mobs_valid_position(
        state,
        proposed_row,
        proposed_col,
        collision
    );
    is_attacking_player = is_attacking_player
        || (too_close_to_player && !proposed_valid);
    is_attacking_player = is_attacking_player
        && old_cooldown <= 0
        && old_mask;

    bool can_spawn_projectile =
        craftax_update_mobs_count_mob_projectiles(state, level)
            < CRAFTAX_MAX_MOB_PROJECTILES;
    bool is_spawning_projectile =
        is_attacking_player && can_spawn_projectile;
    int32_t projectile_position[2] = {old_row, old_col};
    int32_t projectile_type =
        craftax_update_mobs_projectile_type_for_ranged(mob_type);
    craftax_update_mobs_spawn_mob_projectile(
        state,
        level,
        is_spawning_projectile,
        projectile_position,
        player_direction,
        projectile_type
    );

    if (is_attacking_player) {
        proposed_row = old_row;
        proposed_col = old_col;
    }
    int32_t new_cooldown = is_attacking_player ? 4 : old_cooldown - 1;

    bool valid_move = craftax_update_mobs_valid_position(
        state,
        proposed_row,
        proposed_col,
        collision
    );
    int32_t new_row = valid_move ? proposed_row : old_row;
    int32_t new_col = valid_move ? proposed_col : old_col;

    bool should_not_despawn = distance_to_player < CRAFTAX_MOB_DESPAWN_DISTANCE
        || craftax_step_is_fighting_boss(state);

    craftax_update_mobs_clear_old_map_entry(
        state,
        level,
        old_row,
        old_col,
        old_mask
    );
    bool new_mask = old_mask && should_not_despawn;
    craftax_update_mobs_enter_new_map_entry(
        state,
        level,
        new_row,
        new_col,
        new_mask
    );

    state->ranged_mobs.position[level][index][0] = new_row;
    state->ranged_mobs.position[level][index][1] = new_col;
    state->ranged_mobs.attack_cooldown[level][index] = new_cooldown;
    state->ranged_mobs.mask[level][index] = new_mask;
}

static inline void craftax_update_mobs_move_mob_projectile(
    CraftaxState* state,
    int32_t index
) {
    int32_t level = state->player_level;
    bool old_mask = state->mob_projectiles.mask[level][index];
#ifndef CRAFTAX_JAX_PARITY
    if (!old_mask) return;
#endif
    int32_t old_row = state->mob_projectiles.position[level][index][0];
    int32_t old_col = state->mob_projectiles.position[level][index][1];
    int32_t proposed_row =
        old_row + state->mob_projectile_directions[level][index][0];
    int32_t proposed_col =
        old_col + state->mob_projectile_directions[level][index][1];

    bool proposed_in_player =
        proposed_row == state->player_position[0]
        && proposed_col == state->player_position[1];
    bool proposed_in_bounds = craftax_update_mobs_in_bounds(
        proposed_row,
        proposed_col
    );
    int32_t proposed_block = craftax_update_mobs_read_block(
        state,
        level,
        proposed_row,
        proposed_col
    );
    bool in_wall = craftax_step_is_solid_block(proposed_block)
        && proposed_block != CRAFTAX_BLOCK_WATER;
    bool in_mob = craftax_step_is_in_mob(state, proposed_row, proposed_col);
    bool continue_move = proposed_in_bounds && !in_wall && !in_mob;

    bool hit_player0 =
        old_row == state->player_position[0]
        && old_col == state->player_position[1]
        && old_mask;
    bool hit_player1 = proposed_in_player && old_mask;
    bool hit_player = hit_player0 || hit_player1;
    continue_move = continue_move && !hit_player;

    bool new_mask = continue_move && old_mask;

    bool hit_bench_or_furnace = proposed_block == CRAFTAX_BLOCK_FURNACE
        || proposed_block == CRAFTAX_BLOCK_CRAFTING_TABLE;
    bool removing_block = hit_bench_or_furnace && old_mask;
    int32_t new_block = removing_block ? CRAFTAX_BLOCK_PATH : proposed_block;

    int32_t projectile_type =
        state->mob_projectiles.type_id[level][index];
    float damage_vector[3];
    craftax_update_mobs_damage_vector(
        projectile_type,
        CRAFTAX_MOB_PROJECTILE,
        damage_vector
    );
    float damage = craftax_update_mobs_damage_done_to_player(
        state,
        damage_vector
    );

    state->mob_projectiles.position[level][index][0] = proposed_row;
    state->mob_projectiles.position[level][index][1] = proposed_col;
    state->mob_projectiles.mask[level][index] = new_mask;
    state->player_health -= damage * (float)(int32_t)hit_player;
    state->is_sleeping = state->is_sleeping && !hit_player;
    state->is_resting = state->is_resting && !hit_player;
    craftax_update_mobs_set_block(
        state,
        level,
        proposed_row,
        proposed_col,
        new_block
    );
}

static inline void craftax_update_mobs_move_player_projectile(
    CraftaxState* state,
    int32_t index
) {
    int32_t level = state->player_level;
    bool old_mask = state->player_projectiles.mask[level][index];
#ifndef CRAFTAX_JAX_PARITY
    if (!old_mask) return;
#endif
    int32_t old_row = state->player_projectiles.position[level][index][0];
    int32_t old_col = state->player_projectiles.position[level][index][1];
    int32_t proposed_row =
        old_row + state->player_projectile_directions[level][index][0];
    int32_t proposed_col =
        old_col + state->player_projectile_directions[level][index][1];

    float damage_vector[3];
    craftax_update_mobs_player_projectile_damage_vector(
        state,
        level,
        index,
        damage_vector
    );

    bool proposed_in_bounds = craftax_update_mobs_in_bounds(
        proposed_row,
        proposed_col
    );
    int32_t proposed_block = craftax_update_mobs_read_block(
        state,
        level,
        proposed_row,
        proposed_col
    );
    bool in_wall = craftax_step_is_solid_block(proposed_block)
        && proposed_block != CRAFTAX_BLOCK_WATER;

    bool did_attack_mob0 = false;
    bool did_kill_mob0 = false;
    craftax_update_mobs_attack_mob_with_damage(
        state,
        old_row,
        old_col,
        damage_vector,
        false,
        &did_attack_mob0,
        &did_kill_mob0
    );
    (void)did_kill_mob0;

    float second_damage_vector[3];
    for (int32_t i = 0; i < 3; i++) {
        second_damage_vector[i] =
            damage_vector[i] * (float)(int32_t)(!did_attack_mob0);
    }

    bool did_attack_mob1 = false;
    bool did_kill_mob1 = false;
    craftax_update_mobs_attack_mob_with_damage(
        state,
        proposed_row,
        proposed_col,
        second_damage_vector,
        false,
        &did_attack_mob1,
        &did_kill_mob1
    );
    (void)did_kill_mob1;

    bool did_attack_mob = did_attack_mob0 || did_attack_mob1;
    bool continue_move = proposed_in_bounds && !in_wall && !did_attack_mob;
    bool new_mask = continue_move && old_mask;

    state->player_projectiles.position[level][index][0] = proposed_row;
    state->player_projectiles.position[level][index][1] = proposed_col;
    state->player_projectiles.mask[level][index] = new_mask;
}

static inline void craftax_update_mobs_native(
    CraftaxState* state,
    CraftaxThreefryKey rng
) {
    CraftaxThreefryKey unused;

    craftax_threefry_split(rng, &rng, &unused);
    craftax_update_mobs_move_melee(state, &rng, 0);
    craftax_update_mobs_move_melee(state, &rng, 1);
    craftax_update_mobs_move_melee(state, &rng, 2);

    craftax_threefry_split(rng, &rng, &unused);
    craftax_update_mobs_move_passive(state, &rng, 0);
    craftax_update_mobs_move_passive(state, &rng, 1);
    craftax_update_mobs_move_passive(state, &rng, 2);

    craftax_threefry_split(rng, &rng, &unused);
    craftax_update_mobs_move_ranged(state, &rng, 0);
    craftax_update_mobs_move_ranged(state, &rng, 1);

    craftax_threefry_split(rng, &rng, &unused);
    craftax_update_mobs_move_mob_projectile(state, 0);
    craftax_update_mobs_move_mob_projectile(state, 1);
    craftax_update_mobs_move_mob_projectile(state, 2);

    craftax_threefry_split(rng, &rng, &unused);
    craftax_update_mobs_move_player_projectile(state, 0);
    craftax_update_mobs_move_player_projectile(state, 1);
    craftax_update_mobs_move_player_projectile(state, 2);
}
