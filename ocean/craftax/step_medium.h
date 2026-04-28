// Standalone native ports of medium Craftax step subsystems.
//
// These helpers intentionally are not integrated into c_step yet. They mutate a
// full CraftaxState, or an Inventory plus read-only state context, so tests can
// compare each subsystem directly against the installed JAX implementation.

#pragma once

#include "step_simple.h"

static inline CraftaxThreefryKey craftax_medium_next_random_key(
    CraftaxThreefryKey* rng
) {
    CraftaxThreefryKey draw;
    craftax_threefry_split(*rng, rng, &draw);
    return draw;
}

static inline int32_t craftax_medium_randint(
    CraftaxThreefryKey key,
    int32_t minval,
    int32_t maxval
) {
    return craftax_randint_i32_at(key, 0u, minval, maxval);
}

static inline int32_t craftax_medium_choice_weighted(
    CraftaxThreefryKey key,
    const float* weights,
    int32_t count
) {
    float total = 0.0f;
    for (int32_t i = 0; i < count; i++) {
        total += weights[i];
    }

    float draw = total * (1.0f - craftax_threefry_uniform_f32(key));
    float cumulative = 0.0f;
    for (int32_t i = 0; i < count; i++) {
        cumulative += weights[i];
        if (cumulative >= draw) {
            return i;
        }
    }
    return count - 1;
}

static inline int32_t craftax_medium_projectile_count(const CraftaxState* state) {
    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    int32_t count = 0;
    for (int32_t i = 0; i < CRAFTAX_MAX_PLAYER_PROJECTILES; i++) {
        count += (int32_t)state->player_projectiles.mask[level][i];
    }
    return count;
}

static inline int32_t craftax_medium_first_projectile_slot(
    const CraftaxState* state
) {
    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    for (int32_t i = 0; i < CRAFTAX_MAX_PLAYER_PROJECTILES; i++) {
        if (!state->player_projectiles.mask[level][i]) {
            return i;
        }
    }
    return 0;
}

static inline void craftax_medium_spawn_player_projectile(
    CraftaxState* state,
    bool is_spawning_projectile,
    const int32_t new_projectile_position[2],
    const int32_t direction[2],
    int32_t projectile_type
) {
    if (!is_spawning_projectile) {
        return;
    }

    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    int32_t index = craftax_medium_first_projectile_slot(state);
    state->player_projectiles.position[level][index][0] = new_projectile_position[0];
    state->player_projectiles.position[level][index][1] = new_projectile_position[1];
    state->player_projectiles.mask[level][index] = true;
    state->player_projectiles.type_id[level][index] = projectile_type;
    state->player_projectile_directions[level][index][0] = direction[0];
    state->player_projectile_directions[level][index][1] = direction[1];
}

static inline int32_t craftax_medium_level_achievement(int32_t level) {
    switch (craftax_step_jax_index(level, CRAFTAX_NUM_LEVELS)) {
    case 1:
        return CRAFTAX_ACH_ENTER_DUNGEON;
    case 2:
        return CRAFTAX_ACH_ENTER_GNOMISH_MINES;
    case 3:
        return CRAFTAX_ACH_ENTER_SEWERS;
    case 4:
        return CRAFTAX_ACH_ENTER_VAULT;
    case 5:
        return CRAFTAX_ACH_ENTER_TROLL_MINES;
    case 6:
        return CRAFTAX_ACH_ENTER_FIRE_REALM;
    case 7:
        return CRAFTAX_ACH_ENTER_ICE_REALM;
    case 8:
        return CRAFTAX_ACH_ENTER_GRAVEYARD;
    default:
        return CRAFTAX_ACH_COLLECT_WOOD;
    }
}

static inline void craftax_shoot_projectile_native(
    CraftaxState* state,
    int32_t action
) {
    bool is_shooting_arrow = action == CRAFTAX_ACTION_SHOOT_ARROW
        && state->inventory.bow >= 1
        && state->inventory.arrows >= 1
        && craftax_medium_projectile_count(state) < CRAFTAX_MAX_PLAYER_PROJECTILES;

    int32_t direction[2];
    craftax_step_direction(state->player_direction, direction);
    craftax_medium_spawn_player_projectile(
        state,
        is_shooting_arrow,
        state->player_position,
        direction,
        CRAFTAX_PROJECTILE_ARROW2
    );

    state->achievements[CRAFTAX_ACH_FIRE_BOW] =
        state->achievements[CRAFTAX_ACH_FIRE_BOW] || is_shooting_arrow;
    state->inventory.arrows -= (int32_t)is_shooting_arrow;
}

static inline void craftax_cast_spell_native(
    CraftaxState* state,
    int32_t action
) {
    bool has_projectile_slot =
        craftax_medium_projectile_count(state) < CRAFTAX_MAX_PLAYER_PROJECTILES;
    bool has_mana = state->player_mana >= 2;
    bool is_casting_fireball = action == CRAFTAX_ACTION_CAST_FIREBALL
        && has_mana
        && has_projectile_slot
        && state->learned_spells[0];
    bool is_casting_iceball = action == CRAFTAX_ACTION_CAST_ICEBALL
        && has_mana
        && has_projectile_slot
        && state->learned_spells[1];
    bool is_casting_spell = is_casting_fireball || is_casting_iceball;

    int32_t projectile_type =
        (int32_t)is_casting_fireball * CRAFTAX_PROJECTILE_FIREBALL
        + (int32_t)is_casting_iceball * CRAFTAX_PROJECTILE_ICEBALL;

    int32_t direction[2];
    craftax_step_direction(state->player_direction, direction);
    craftax_medium_spawn_player_projectile(
        state,
        is_casting_spell,
        state->player_position,
        direction,
        projectile_type
    );

    if (is_casting_fireball) {
        state->achievements[CRAFTAX_ACH_CAST_FIREBALL] = true;
    }
    if (is_casting_iceball) {
        state->achievements[CRAFTAX_ACH_CAST_ICEBALL] = true;
    }
    state->player_mana -= (int32_t)is_casting_spell * 2;
}

static inline void craftax_enchant_native(
    CraftaxState* state,
    int32_t action,
    CraftaxThreefryKey rng
) {
    int32_t direction[2];
    craftax_step_direction(state->player_direction, direction);

    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    int32_t target_row = craftax_step_jax_index(
        state->player_position[0] + direction[0],
        CRAFTAX_MAP_SIZE
    );
    int32_t target_col = craftax_step_jax_index(
        state->player_position[1] + direction[1],
        CRAFTAX_MAP_SIZE
    );
    int32_t target_block = state->map[level][target_row][target_col];

    bool is_fire_table = target_block == CRAFTAX_BLOCK_ENCHANTMENT_TABLE_FIRE;
    bool is_ice_table = target_block == CRAFTAX_BLOCK_ENCHANTMENT_TABLE_ICE;
    bool target_block_is_enchantment_table = is_fire_table || is_ice_table;
    int32_t enchantment_type = is_fire_table ? 1 : 2;
    int32_t num_gems = is_fire_table
        ? state->inventory.ruby
        : state->inventory.sapphire;

    bool could_enchant = state->player_mana >= 9
        && target_block_is_enchantment_table
        && num_gems >= 1;
    bool is_enchanting_bow = could_enchant
        && action == CRAFTAX_ACTION_ENCHANT_BOW
        && state->inventory.bow > 0;
    bool is_enchanting_sword = could_enchant
        && action == CRAFTAX_ACTION_ENCHANT_SWORD
        && state->inventory.sword > 0;

    int32_t armour_count = 0;
    for (int32_t i = 0; i < 4; i++) {
        armour_count += state->inventory.armour[i];
    }
    bool is_enchanting_armour = could_enchant
        && action == CRAFTAX_ACTION_ENCHANT_ARMOUR
        && armour_count > 0;

    CraftaxThreefryKey armour_key = craftax_medium_next_random_key(&rng);
    int32_t unenchanted_count = 0;
    for (int32_t i = 0; i < 4; i++) {
        unenchanted_count += (int32_t)(state->armour_enchantments[i] == 0);
    }

    float armour_targets[4];
    for (int32_t i = 0; i < 4; i++) {
        bool unenchanted = state->armour_enchantments[i] == 0;
        bool opposite_enchanted = state->armour_enchantments[i] != 0
            && state->armour_enchantments[i] != enchantment_type;
        armour_targets[i] = (unenchanted || (
            unenchanted_count == 0 && opposite_enchanted
        )) ? 1.0f : 0.0f;
    }
    int32_t armour_target = craftax_medium_choice_weighted(
        armour_key,
        armour_targets,
        4
    );

    bool is_enchanting = is_enchanting_sword
        || is_enchanting_bow
        || is_enchanting_armour;
    if (is_enchanting_sword) {
        state->sword_enchantment = enchantment_type;
        state->achievements[CRAFTAX_ACH_ENCHANT_SWORD] = true;
    }
    if (is_enchanting_bow) {
        state->bow_enchantment = enchantment_type;
    }
    if (is_enchanting_armour) {
        state->armour_enchantments[armour_target] = enchantment_type;
        state->achievements[CRAFTAX_ACH_ENCHANT_ARMOUR] = true;
    }

    state->inventory.sapphire -=
        (int32_t)is_enchanting * (int32_t)(enchantment_type == 2);
    state->inventory.ruby -=
        (int32_t)is_enchanting * (int32_t)(enchantment_type == 1);
    state->player_mana -= (int32_t)is_enchanting * 9;
}

static inline void craftax_change_floor_native(
    CraftaxState* state,
    int32_t action
) {
    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    int32_t player_row = craftax_step_jax_index(
        state->player_position[0],
        CRAFTAX_MAP_SIZE
    );
    int32_t player_col = craftax_step_jax_index(
        state->player_position[1],
        CRAFTAX_MAP_SIZE
    );

    bool on_down_ladder =
        state->item_map[level][player_row][player_col] == CRAFTAX_ITEM_LADDER_DOWN;
    bool is_moving_down = action == CRAFTAX_ACTION_DESCEND
        && on_down_ladder
        && state->monsters_killed[level] >= CRAFTAX_MONSTERS_KILLED_TO_CLEAR_LEVEL
        && state->player_level < CRAFTAX_NUM_LEVELS - 1;

    bool on_up_ladder =
        state->item_map[level][player_row][player_col] == CRAFTAX_ITEM_LADDER_UP;
    bool is_moving_up = action == CRAFTAX_ACTION_ASCEND
        && on_up_ladder
        && state->player_level > 0;

    int32_t delta_floor = (int32_t)is_moving_down - (int32_t)is_moving_up;
    int32_t new_level = state->player_level + delta_floor;
    int32_t achievement = craftax_medium_level_achievement(new_level);
    bool new_floor = new_level != 0 && !state->achievements[achievement];

    if (is_moving_down) {
        int32_t ladder_level = craftax_step_jax_index(
            state->player_level + 1,
            CRAFTAX_NUM_LEVELS
        );
        state->player_position[0] = state->up_ladders[ladder_level][0];
        state->player_position[1] = state->up_ladders[ladder_level][1];
    } else if (is_moving_up) {
        int32_t ladder_level = craftax_step_jax_index(
            state->player_level - 1,
            CRAFTAX_NUM_LEVELS
        );
        state->player_position[0] = state->down_ladders[ladder_level][0];
        state->player_position[1] = state->down_ladders[ladder_level][1];
    }

    state->player_level = new_level;
    state->achievements[achievement] =
        state->achievements[achievement] || new_level != 0;
    state->player_xp += (int32_t)new_floor;
}

static inline void craftax_add_items_from_chest_native(
    const CraftaxState* state,
    CraftaxInventory* inventory,
    bool is_opening_chest,
    CraftaxThreefryKey rng
) {
    CraftaxThreefryKey draw_key;

    draw_key = craftax_medium_next_random_key(&rng);
    bool is_looting_wood = craftax_threefry_uniform_f32(draw_key) < 0.6f;
    draw_key = craftax_medium_next_random_key(&rng);
    int32_t wood_loot_amount =
        craftax_medium_randint(draw_key, 1, 6) * (int32_t)is_looting_wood;
    (void)wood_loot_amount;

    draw_key = craftax_medium_next_random_key(&rng);
    bool is_looting_torch = craftax_threefry_uniform_f32(draw_key) < 0.6f;
    draw_key = craftax_medium_next_random_key(&rng);
    int32_t torch_loot_amount =
        craftax_medium_randint(draw_key, 4, 8) * (int32_t)is_looting_torch;

    draw_key = craftax_medium_next_random_key(&rng);
    bool is_looting_ore = craftax_threefry_uniform_f32(draw_key) < 0.6f;
    draw_key = craftax_medium_next_random_key(&rng);
    float ore_weights[5] = {0.3f, 0.3f, 0.15f, 0.125f, 0.125f};
    int32_t ore_loot_id = craftax_medium_choice_weighted(
        draw_key,
        ore_weights,
        5
    );
    draw_key = craftax_medium_next_random_key(&rng);

    int32_t coal_loot_amount =
        craftax_medium_randint(draw_key, 1, 4)
        * (int32_t)(ore_loot_id == 0)
        * (int32_t)is_looting_ore;
    int32_t iron_loot_amount =
        craftax_medium_randint(draw_key, 1, 3)
        * (int32_t)(ore_loot_id == 1)
        * (int32_t)is_looting_ore;
    int32_t diamond_loot_amount =
        craftax_medium_randint(draw_key, 1, 2)
        * (int32_t)(ore_loot_id == 2)
        * (int32_t)is_looting_ore;
    int32_t sapphire_loot_amount =
        craftax_medium_randint(draw_key, 1, 2)
        * (int32_t)(ore_loot_id == 3)
        * (int32_t)is_looting_ore;
    int32_t ruby_loot_amount =
        craftax_medium_randint(draw_key, 1, 2)
        * (int32_t)(ore_loot_id == 4)
        * (int32_t)is_looting_ore;

    draw_key = craftax_medium_next_random_key(&rng);
    bool is_looting_potion = craftax_threefry_uniform_f32(draw_key) < 0.5f;
    draw_key = craftax_medium_next_random_key(&rng);
    int32_t potion_loot_index = craftax_medium_randint(draw_key, 0, 6);
    draw_key = craftax_medium_next_random_key(&rng);
    int32_t potion_loot_amount = craftax_medium_randint(draw_key, 1, 3);

    draw_key = craftax_medium_next_random_key(&rng);
    bool is_looting_arrows = craftax_threefry_uniform_f32(draw_key) < 0.25f;
    draw_key = craftax_medium_next_random_key(&rng);
    int32_t arrows_loot_amount =
        craftax_medium_randint(draw_key, 1, 5) * (int32_t)is_looting_arrows;

    draw_key = craftax_medium_next_random_key(&rng);
    bool is_looting_tool = craftax_threefry_uniform_f32(draw_key) < 0.2f;
    draw_key = craftax_medium_next_random_key(&rng);
    int32_t tool_id = craftax_medium_randint(draw_key, 0, 2);

    bool is_looting_pickaxe = is_looting_tool
        && tool_id == 0
        && is_opening_chest;
    draw_key = craftax_medium_next_random_key(&rng);
    float tool_weights[4] = {0.4f, 0.3f, 0.2f, 0.1f};
    int32_t pickaxe_loot_level = (
        craftax_medium_choice_weighted(draw_key, tool_weights, 4) + 1
    ) * (int32_t)is_looting_pickaxe;
    pickaxe_loot_level = craftax_step_maxi32(
        pickaxe_loot_level,
        inventory->pickaxe
    );
    int32_t new_pickaxe_level = is_looting_pickaxe
        ? pickaxe_loot_level
        : inventory->pickaxe;

    bool is_looting_sword = is_looting_tool
        && tool_id == 1
        && is_opening_chest;
    draw_key = craftax_medium_next_random_key(&rng);
    int32_t sword_loot_level = (
        craftax_medium_choice_weighted(draw_key, tool_weights, 4) + 1
    ) * (int32_t)is_looting_sword;
    sword_loot_level = craftax_step_maxi32(sword_loot_level, inventory->sword);
    int32_t new_sword_level = is_looting_sword
        ? sword_loot_level
        : inventory->sword;

    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    bool is_looting_bow = is_opening_chest
        && state->player_level == 1
        && !state->chests_opened[level];
    int32_t new_bow_level = is_looting_bow ? 1 : inventory->bow;

    bool is_looting_book = !state->chests_opened[level]
        && (state->player_level == 3 || state->player_level == 4);

    int32_t opening = (int32_t)is_opening_chest;
    inventory->torches += torch_loot_amount * opening;
    inventory->coal += coal_loot_amount * opening;
    inventory->iron += iron_loot_amount * opening;
    inventory->diamond += diamond_loot_amount * opening;
    inventory->sapphire += sapphire_loot_amount * opening;
    inventory->ruby += ruby_loot_amount * opening;
    inventory->arrows += arrows_loot_amount * opening;
    inventory->pickaxe = new_pickaxe_level;
    inventory->sword = new_sword_level;
    inventory->potions[potion_loot_index] +=
        potion_loot_amount * (int32_t)is_looting_potion * opening;
    inventory->bow = new_bow_level;
    inventory->books += (int32_t)is_looting_book * opening;
}
