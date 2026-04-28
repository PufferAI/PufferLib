// Standalone native port of Craftax do_action.
//
// This helper intentionally is not integrated into c_step yet. It mutates a
// full CraftaxState in place so tests can compare the subsystem directly
// against the installed JAX implementation.

#pragma once

#include "step_medium.h"

#define CRAFTAX_DO_ACTION_BOSS_FIGHT_SPAWN_TURNS 7

static inline float craftax_do_action_mob_defense(
    int32_t type_id,
    int32_t mob_class_index,
    int32_t damage_index
) {
    static const float defenses[8][4][3] = {
        {
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {0.5f, 0.0f, 0.0f},
            {0.5f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {0.2f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {0.9f, 1.0f, 0.0f},
            {0.9f, 1.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {0.9f, 0.0f, 1.0f},
            {0.9f, 0.0f, 1.0f},
            {0.0f, 0.0f, 0.0f},
        },
    };

    int32_t type_index = craftax_step_jax_index(type_id, 8);
    int32_t class_index = craftax_step_jax_index(mob_class_index, 4);
    int32_t component = craftax_step_jax_index(damage_index, 3);
    return defenses[type_index][class_index][component];
}

static inline int32_t craftax_do_action_mob_achievement(
    int32_t mob_class_index,
    int32_t type_id
) {
    static const int32_t achievements[3][8] = {
        {
            CRAFTAX_ACH_EAT_COW,
            CRAFTAX_ACH_EAT_BAT,
            CRAFTAX_ACH_EAT_SNAIL,
            0,
            0,
            0,
            0,
            0,
        },
        {
            CRAFTAX_ACH_DEFEAT_ZOMBIE,
            CRAFTAX_ACH_DEFEAT_GNOME_WARRIOR,
            CRAFTAX_ACH_DEFEAT_ORC_SOLIDER,
            CRAFTAX_ACH_DEFEAT_LIZARD,
            CRAFTAX_ACH_DEFEAT_KNIGHT,
            CRAFTAX_ACH_DEFEAT_TROLL,
            CRAFTAX_ACH_DEFEAT_PIGMAN,
            CRAFTAX_ACH_DEFEAT_FROST_TROLL,
        },
        {
            CRAFTAX_ACH_DEFEAT_SKELETON,
            CRAFTAX_ACH_DEFEAT_GNOME_ARCHER,
            CRAFTAX_ACH_DEFEAT_ORC_MAGE,
            CRAFTAX_ACH_DEFEAT_KOBOLD,
            CRAFTAX_ACH_DEFEAT_ARCHER,
            CRAFTAX_ACH_DEFEAT_DEEP_THING,
            CRAFTAX_ACH_DEFEAT_FIRE_ELEMENTAL,
            CRAFTAX_ACH_DEFEAT_ICE_ELEMENTAL,
        },
    };

    int32_t class_index = craftax_step_jax_index(mob_class_index, 3);
    int32_t type_index = craftax_step_jax_index(type_id, 8);
    return achievements[class_index][type_index];
}

static inline void craftax_do_action_player_damage_vector(
    const CraftaxState* state,
    float damage_vector[3]
) {
    static const float physical_damages[5] = {1.0f, 2.0f, 3.0f, 5.0f, 8.0f};

    int32_t sword_index = craftax_step_jax_index(state->inventory.sword, 5);
    float physical_damage = physical_damages[sword_index];
    float fire_damage =
        physical_damage * (float)(state->sword_enchantment == 1) * 0.5f;
    float ice_damage =
        physical_damage * (float)(state->sword_enchantment == 2) * 0.5f;

    physical_damage *= 1.0f + 0.25f * (float)(state->player_strength - 1);
    fire_damage *= 1.0f + 0.05f * (float)(state->player_intelligence - 1);
    ice_damage *= 1.0f + 0.05f * (float)(state->player_intelligence - 1);

    damage_vector[0] = physical_damage;
    damage_vector[1] = fire_damage;
    damage_vector[2] = ice_damage;
}

static inline float craftax_do_action_damage_done(
    const float damage_vector[3],
    int32_t type_id,
    int32_t mob_class_index
) {
    float damage = 0.0f;
    for (int32_t i = 0; i < 3; i++) {
        float defense = craftax_do_action_mob_defense(
            type_id,
            mob_class_index,
            i
        );
        damage += (1.0f - defense) * damage_vector[i];
    }
    return damage;
}

static inline void craftax_do_action_refresh_mobs3_masks(CraftaxMobs3* mobs) {
    for (int32_t level = 0; level < CRAFTAX_NUM_LEVELS; level++) {
        for (int32_t i = 0; i < 3; i++) {
            mobs->mask[level][i] =
                mobs->mask[level][i] && mobs->health[level][i] > 0.0f;
        }
    }
}

static inline void craftax_do_action_refresh_mobs2_masks(CraftaxMobs2* mobs) {
    for (int32_t level = 0; level < CRAFTAX_NUM_LEVELS; level++) {
        for (int32_t i = 0; i < 2; i++) {
            mobs->mask[level][i] =
                mobs->mask[level][i] && mobs->health[level][i] > 0.0f;
        }
    }
}

static inline void craftax_do_action_attack_mobs3(
    CraftaxState* state,
    CraftaxMobs3* mobs,
    int32_t row,
    int32_t col,
    const float damage_vector[3],
    bool can_get_achievement,
    int32_t mob_class_index,
    bool* did_kill_mob,
    bool* is_attacking_mob
) {
    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    bool is_attacking_array[3];
    *is_attacking_mob = false;
    int32_t target_mob_index = 0;

    for (int32_t i = 0; i < 3; i++) {
        bool in_mob = mobs->position[level][i][0] == row
            && mobs->position[level][i][1] == col;
        is_attacking_array[i] = in_mob && mobs->mask[level][i];
        if (is_attacking_array[i] && !*is_attacking_mob) {
            target_mob_index = i;
        }
        *is_attacking_mob = *is_attacking_mob || is_attacking_array[i];
    }

    int32_t target_type_id = mobs->type_id[level][target_mob_index];
    float damage = craftax_do_action_damage_done(
        damage_vector,
        target_type_id,
        mob_class_index
    );
    mobs->health[level][target_mob_index] -=
        damage * (float)(int32_t)(*is_attacking_mob);

    bool old_mask = mobs->mask[level][target_mob_index];
    craftax_do_action_refresh_mobs3_masks(mobs);
    *did_kill_mob = old_mask && !mobs->mask[level][target_mob_index];

    int32_t achievement_for_kill = craftax_do_action_mob_achievement(
        mob_class_index,
        target_type_id
    );
    bool unlock = *did_kill_mob && can_get_achievement;
    state->achievements[achievement_for_kill] =
        state->achievements[achievement_for_kill] || unlock;
}

static inline void craftax_do_action_attack_mobs2(
    CraftaxState* state,
    CraftaxMobs2* mobs,
    int32_t row,
    int32_t col,
    const float damage_vector[3],
    bool can_get_achievement,
    int32_t mob_class_index,
    bool* did_kill_mob,
    bool* is_attacking_mob
) {
    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    bool is_attacking_array[2];
    *is_attacking_mob = false;
    int32_t target_mob_index = 0;

    for (int32_t i = 0; i < 2; i++) {
        bool in_mob = mobs->position[level][i][0] == row
            && mobs->position[level][i][1] == col;
        is_attacking_array[i] = in_mob && mobs->mask[level][i];
        if (is_attacking_array[i] && !*is_attacking_mob) {
            target_mob_index = i;
        }
        *is_attacking_mob = *is_attacking_mob || is_attacking_array[i];
    }

    int32_t target_type_id = mobs->type_id[level][target_mob_index];
    float damage = craftax_do_action_damage_done(
        damage_vector,
        target_type_id,
        mob_class_index
    );
    mobs->health[level][target_mob_index] -=
        damage * (float)(int32_t)(*is_attacking_mob);

    bool old_mask = mobs->mask[level][target_mob_index];
    craftax_do_action_refresh_mobs2_masks(mobs);
    *did_kill_mob = old_mask && !mobs->mask[level][target_mob_index];

    int32_t achievement_for_kill = craftax_do_action_mob_achievement(
        mob_class_index,
        target_type_id
    );
    bool unlock = *did_kill_mob && can_get_achievement;
    state->achievements[achievement_for_kill] =
        state->achievements[achievement_for_kill] || unlock;
}

static inline bool craftax_do_action_update_index(
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

static inline void craftax_do_action_update_mob_map(
    CraftaxState* state,
    int32_t row,
    int32_t col,
    bool did_kill_mob
) {
    int32_t update_row;
    int32_t update_col;
    if (!craftax_do_action_update_index(row, CRAFTAX_MAP_SIZE, &update_row)
        || !craftax_do_action_update_index(col, CRAFTAX_MAP_SIZE, &update_col)) {
        return;
    }

    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    int32_t read_row = craftax_step_jax_index(row, CRAFTAX_MAP_SIZE);
    int32_t read_col = craftax_step_jax_index(col, CRAFTAX_MAP_SIZE);
    bool old_value = (state->mob_bits[level][read_row] >> read_col) & 1ULL;
    bool new_value = old_value && !did_kill_mob;
    if (new_value) {
        state->mob_bits[level][update_row] |= (1ULL << update_col);
    } else {
        state->mob_bits[level][update_row] &= ~(1ULL << update_col);
    }
}

static inline void craftax_do_action_attack_mob(
    CraftaxState* state,
    int32_t row,
    int32_t col,
    bool can_eat,
    bool* did_attack_mob,
    bool* did_kill_mob
) {
    float damage_vector[3];
    craftax_do_action_player_damage_vector(state, damage_vector);

    bool did_kill_melee_mob = false;
    bool is_attacking_melee_mob = false;
    craftax_do_action_attack_mobs3(
        state,
        &state->melee_mobs,
        row,
        col,
        damage_vector,
        true,
        1,
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
        0,
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
        2,
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

static inline bool craftax_do_action_in_bounds(int32_t row, int32_t col) {
    return row >= 0
        && row < CRAFTAX_MAP_SIZE
        && col >= 0
        && col < CRAFTAX_MAP_SIZE;
}

static inline bool craftax_do_action_boss_vulnerable(
    const CraftaxState* state
) {
    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    int32_t melee_count = 0;
    int32_t ranged_count = 0;
    for (int32_t i = 0; i < CRAFTAX_MAX_MELEE_MOBS; i++) {
        melee_count += (int32_t)state->melee_mobs.mask[level][i];
    }
    for (int32_t i = 0; i < CRAFTAX_MAX_RANGED_MOBS; i++) {
        ranged_count += (int32_t)state->ranged_mobs.mask[level][i];
    }
    return melee_count == 0
        && ranged_count == 0
        && state->boss_timesteps_to_spawn_this_round <= 0;
}

static inline void craftax_do_action_update_plants_with_eat(
    CraftaxState* state,
    int32_t row,
    int32_t col
) {
    int32_t plant_index = 0;
    bool found = false;
    for (int32_t i = 0; i < CRAFTAX_MAX_GROWING_PLANTS; i++) {
        bool is_plant = state->growing_plants_positions[i][0] == row
            && state->growing_plants_positions[i][1] == col;
        if (is_plant && !found) {
            plant_index = i;
            found = true;
        }
    }
    state->growing_plants_age[plant_index] = 0;
}

static inline void craftax_do_action_native(
    CraftaxState* state,
    int32_t action,
    CraftaxThreefryKey rng
) {
    if (action != CRAFTAX_ACTION_DO) {
        return;
    }

    int32_t direction[2];
    craftax_step_direction(state->player_direction, direction);
    int32_t target_row = state->player_position[0] + direction[0];
    int32_t target_col = state->player_position[1] + direction[1];

    bool did_attack_mob = false;
    bool did_kill_mob = false;
    craftax_do_action_attack_mob(
        state,
        target_row,
        target_col,
        true,
        &did_attack_mob,
        &did_kill_mob
    );
    (void)did_kill_mob;

    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    int32_t read_row = craftax_step_jax_index(target_row, CRAFTAX_MAP_SIZE);
    int32_t read_col = craftax_step_jax_index(target_col, CRAFTAX_MAP_SIZE);
    int32_t target_block = state->map[level][read_row][read_col];

    CraftaxThreefryKey sapling_key = craftax_medium_next_random_key(&rng);
    CraftaxThreefryKey chest_key = craftax_medium_next_random_key(&rng);

    bool is_opening_chest = target_block == CRAFTAX_BLOCK_CHEST;
    bool is_damaging_boss = target_block == CRAFTAX_BLOCK_NECROMANCER
        && craftax_do_action_boss_vulnerable(state)
        && craftax_step_is_fighting_boss(state);

    bool action_block_in_bounds =
        craftax_do_action_in_bounds(target_row, target_col) && !did_attack_mob;

    if (action_block_in_bounds) {
        bool is_block_tree = target_block == CRAFTAX_BLOCK_TREE;
        bool is_block_fire_tree = target_block == CRAFTAX_BLOCK_FIRE_TREE;
        bool is_block_ice_shrub = target_block == CRAFTAX_BLOCK_ICE_SHRUB;
        bool is_mining_tree =
            is_block_tree || is_block_fire_tree || is_block_ice_shrub;
        if (is_mining_tree) {
            int32_t replacement = is_block_tree
                ? CRAFTAX_BLOCK_GRASS
                : (is_block_fire_tree
                    ? CRAFTAX_BLOCK_FIRE_GRASS
                    : CRAFTAX_BLOCK_ICE_GRASS);
            craftax_set_map_block(state, level, target_row, target_col, replacement);
            state->inventory.wood += 1;
        }

        bool is_mining_stone = target_block == CRAFTAX_BLOCK_STONE
            && state->inventory.pickaxe >= 1;
        if (is_mining_stone) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            state->inventory.stone += 1;
        }

        if (target_block == CRAFTAX_BLOCK_FURNACE) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
        }

        if (target_block == CRAFTAX_BLOCK_CRAFTING_TABLE) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
        }

        bool is_mining_coal = target_block == CRAFTAX_BLOCK_COAL
            && state->inventory.pickaxe >= 1;
        if (is_mining_coal) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            state->inventory.coal += 1;
        }

        bool is_mining_iron = target_block == CRAFTAX_BLOCK_IRON
            && state->inventory.pickaxe >= 2;
        if (is_mining_iron) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            state->inventory.iron += 1;
        }

        bool is_mining_diamond = target_block == CRAFTAX_BLOCK_DIAMOND
            && state->inventory.pickaxe >= 3;
        if (is_mining_diamond) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            state->inventory.diamond += 1;
        }

        bool is_mining_sapphire = target_block == CRAFTAX_BLOCK_SAPPHIRE
            && state->inventory.pickaxe >= 4;
        if (is_mining_sapphire) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            state->inventory.sapphire += 1;
        }

        bool is_mining_ruby = target_block == CRAFTAX_BLOCK_RUBY
            && state->inventory.pickaxe >= 4;
        if (is_mining_ruby) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            state->inventory.ruby += 1;
        }

        bool is_mining_sapling = target_block == CRAFTAX_BLOCK_GRASS
            && craftax_threefry_uniform_f32(sapling_key) < 0.1f;
        state->inventory.sapling += (int32_t)is_mining_sapling;

        bool is_drinking_water = target_block == CRAFTAX_BLOCK_WATER
            || target_block == CRAFTAX_BLOCK_FOUNTAIN;
        if (is_drinking_water) {
            state->player_drink = craftax_step_mini32(
                craftax_step_get_max_drink(state),
                state->player_drink + 1
            );
            state->player_thirst = 0.0f;
            state->achievements[CRAFTAX_ACH_COLLECT_DRINK] = true;
        }

        bool is_eating_plant = target_block == CRAFTAX_BLOCK_RIPE_PLANT;
        if (is_eating_plant) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PLANT);
            state->player_food = craftax_step_mini32(
                craftax_step_get_max_food(state),
                state->player_food + 4
            );
            state->player_hunger = 0.0f;
            state->achievements[CRAFTAX_ACH_EAT_PLANT] = true;
            craftax_do_action_update_plants_with_eat(
                state,
                target_row,
                target_col
            );
        }

        bool is_mining_stalagmite = target_block == CRAFTAX_BLOCK_STALAGMITE
            && state->inventory.pickaxe >= 1;
        if (is_mining_stalagmite) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            state->inventory.stone += 1;
        }

        if (is_opening_chest) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            craftax_add_items_from_chest_native(
                state,
                &state->inventory,
                true,
                chest_key
            );
            state->achievements[CRAFTAX_ACH_OPEN_CHEST] = true;
        }

        if (is_damaging_boss) {
            state->achievements[CRAFTAX_ACH_DAMAGE_NECROMANCER] = true;
        }
    }

    state->chests_opened[level] =
        state->chests_opened[level] || is_opening_chest;

    state->boss_progress += (int32_t)is_damaging_boss;
    if (is_damaging_boss) {
        state->boss_timesteps_to_spawn_this_round =
            CRAFTAX_DO_ACTION_BOSS_FIGHT_SPAWN_TURNS;
    }
}
