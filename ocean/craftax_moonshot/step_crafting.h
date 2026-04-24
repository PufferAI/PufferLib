// Standalone native ports of Craftax crafting and placement subsystems.
//
// These helpers intentionally are not integrated into c_step yet. They mutate a
// full CraftaxState in place so tests can compare each subsystem directly
// against the installed JAX implementation.

#pragma once

#include "step_simple.h"

static inline bool craftax_crafting_is_near_block(
    const CraftaxState* state,
    int32_t block_type
) {
    static const int32_t close_blocks[8][2] = {
        {0, -1},
        {0, 1},
        {-1, 0},
        {1, 0},
        {-1, -1},
        {-1, 1},
        {1, -1},
        {1, 1},
    };

    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    for (int32_t i = 0; i < 8; i++) {
        int32_t row = state->player_position[0] + close_blocks[i][0];
        int32_t col = state->player_position[1] + close_blocks[i][1];
        bool in_bounds = row >= 0
            && row < CRAFTAX_MAP_SIZE
            && col >= 0
            && col < CRAFTAX_MAP_SIZE;
        if (in_bounds && state->map[level][row][col] == block_type) {
            return true;
        }
    }
    return false;
}

static inline int32_t craftax_crafting_first_armour_below(
    const CraftaxInventory* inventory,
    int32_t threshold,
    int32_t* count
) {
    int32_t first = 0;
    *count = 0;
    for (int32_t i = 0; i < 4; i++) {
        bool below = inventory->armour[i] < threshold;
        first = (*count == 0 && below) ? i : first;
        *count += (int32_t)below;
    }
    return first;
}

static inline void craftax_do_crafting_native(
    CraftaxState* state,
    int32_t action
) {
    bool is_at_crafting_table = craftax_crafting_is_near_block(
        state,
        CRAFTAX_BLOCK_CRAFTING_TABLE
    );
    bool is_at_furnace = craftax_crafting_is_near_block(
        state,
        CRAFTAX_BLOCK_FURNACE
    );

    CraftaxInventory* inventory = &state->inventory;

    bool can_craft_wood_pickaxe = inventory->wood >= 1;
    bool is_crafting_wood_pickaxe =
        action == CRAFTAX_ACTION_MAKE_WOOD_PICKAXE
        && can_craft_wood_pickaxe
        && is_at_crafting_table
        && inventory->pickaxe < 1;
    inventory->wood -= 1 * (int32_t)is_crafting_wood_pickaxe;
    inventory->pickaxe =
        inventory->pickaxe * (1 - (int32_t)is_crafting_wood_pickaxe)
        + 1 * (int32_t)is_crafting_wood_pickaxe;

    bool can_craft_stone_pickaxe =
        inventory->wood >= 1 && inventory->stone >= 1;
    bool is_crafting_stone_pickaxe =
        action == CRAFTAX_ACTION_MAKE_STONE_PICKAXE
        && can_craft_stone_pickaxe
        && is_at_crafting_table
        && inventory->pickaxe < 2;
    inventory->stone -= 1 * (int32_t)is_crafting_stone_pickaxe;
    inventory->wood -= 1 * (int32_t)is_crafting_stone_pickaxe;
    inventory->pickaxe =
        inventory->pickaxe * (1 - (int32_t)is_crafting_stone_pickaxe)
        + 2 * (int32_t)is_crafting_stone_pickaxe;

    bool can_craft_iron_pickaxe =
        inventory->wood >= 1
        && inventory->stone >= 1
        && inventory->iron >= 1
        && inventory->coal >= 1;
    bool is_crafting_iron_pickaxe =
        action == CRAFTAX_ACTION_MAKE_IRON_PICKAXE
        && can_craft_iron_pickaxe
        && is_at_furnace
        && is_at_crafting_table
        && inventory->pickaxe < 3;
    inventory->iron -= 1 * (int32_t)is_crafting_iron_pickaxe;
    inventory->wood -= 1 * (int32_t)is_crafting_iron_pickaxe;
    inventory->stone -= 1 * (int32_t)is_crafting_iron_pickaxe;
    inventory->coal -= 1 * (int32_t)is_crafting_iron_pickaxe;
    inventory->pickaxe =
        inventory->pickaxe * (1 - (int32_t)is_crafting_iron_pickaxe)
        + 3 * (int32_t)is_crafting_iron_pickaxe;

    bool can_craft_diamond_pickaxe =
        inventory->wood >= 1 && inventory->diamond >= 3;
    bool is_crafting_diamond_pickaxe =
        action == CRAFTAX_ACTION_MAKE_DIAMOND_PICKAXE
        && can_craft_diamond_pickaxe
        && is_at_crafting_table
        && inventory->pickaxe < 4;
    inventory->diamond -= 3 * (int32_t)is_crafting_diamond_pickaxe;
    inventory->wood -= 1 * (int32_t)is_crafting_diamond_pickaxe;
    inventory->pickaxe =
        inventory->pickaxe * (1 - (int32_t)is_crafting_diamond_pickaxe)
        + 4 * (int32_t)is_crafting_diamond_pickaxe;

    bool can_craft_wood_sword = inventory->wood >= 1;
    bool is_crafting_wood_sword =
        action == CRAFTAX_ACTION_MAKE_WOOD_SWORD
        && can_craft_wood_sword
        && is_at_crafting_table
        && inventory->sword < 1;
    inventory->wood -= 1 * (int32_t)is_crafting_wood_sword;
    inventory->sword =
        inventory->sword * (1 - (int32_t)is_crafting_wood_sword)
        + 1 * (int32_t)is_crafting_wood_sword;

    bool can_craft_stone_sword =
        inventory->stone >= 1 && inventory->wood >= 1;
    bool is_crafting_stone_sword =
        action == CRAFTAX_ACTION_MAKE_STONE_SWORD
        && can_craft_stone_sword
        && is_at_crafting_table
        && inventory->sword < 2;
    inventory->wood -= 1 * (int32_t)is_crafting_stone_sword;
    inventory->stone -= 1 * (int32_t)is_crafting_stone_sword;
    inventory->sword =
        inventory->sword * (1 - (int32_t)is_crafting_stone_sword)
        + 2 * (int32_t)is_crafting_stone_sword;

    bool can_craft_iron_sword =
        inventory->iron >= 1
        && inventory->wood >= 1
        && inventory->stone >= 1
        && inventory->coal >= 1;
    bool is_crafting_iron_sword =
        action == CRAFTAX_ACTION_MAKE_IRON_SWORD
        && can_craft_iron_sword
        && is_at_furnace
        && is_at_crafting_table
        && inventory->sword < 3;
    inventory->wood -= 1 * (int32_t)is_crafting_iron_sword;
    inventory->iron -= 1 * (int32_t)is_crafting_iron_sword;
    inventory->stone -= 1 * (int32_t)is_crafting_iron_sword;
    inventory->coal -= 1 * (int32_t)is_crafting_iron_sword;
    inventory->sword =
        inventory->sword * (1 - (int32_t)is_crafting_iron_sword)
        + 3 * (int32_t)is_crafting_iron_sword;

    bool can_craft_diamond_sword =
        inventory->diamond >= 2 && inventory->wood >= 1;
    bool is_crafting_diamond_sword =
        action == CRAFTAX_ACTION_MAKE_DIAMOND_SWORD
        && can_craft_diamond_sword
        && is_at_crafting_table
        && inventory->sword < 4;
    inventory->wood -= 1 * (int32_t)is_crafting_diamond_sword;
    inventory->diamond -= 2 * (int32_t)is_crafting_diamond_sword;
    inventory->sword =
        inventory->sword * (1 - (int32_t)is_crafting_diamond_sword)
        + 4 * (int32_t)is_crafting_diamond_sword;

    int32_t armour_count = 0;
    int32_t iron_armour_index_to_craft =
        craftax_crafting_first_armour_below(inventory, 1, &armour_count);
    bool can_craft_iron_armour =
        armour_count > 0 && inventory->iron >= 3 && inventory->coal >= 3;
    bool is_crafting_iron_armour =
        action == CRAFTAX_ACTION_MAKE_IRON_ARMOUR
        && can_craft_iron_armour
        && is_at_crafting_table
        && is_at_furnace;
    inventory->iron -= 3 * (int32_t)is_crafting_iron_armour;
    inventory->coal -= 3 * (int32_t)is_crafting_iron_armour;
    inventory->armour[iron_armour_index_to_craft] =
        (int32_t)is_crafting_iron_armour * 1
        + (1 - (int32_t)is_crafting_iron_armour)
        * inventory->armour[iron_armour_index_to_craft];
    state->achievements[CRAFTAX_ACH_MAKE_IRON_ARMOUR] =
        state->achievements[CRAFTAX_ACH_MAKE_IRON_ARMOUR]
        || is_crafting_iron_armour;

    int32_t diamond_armour_count = 0;
    int32_t diamond_armour_index_to_craft =
        craftax_crafting_first_armour_below(inventory, 2, &diamond_armour_count);
    bool can_craft_diamond_armour =
        diamond_armour_count > 0 && inventory->diamond >= 3;
    bool is_crafting_diamond_armour =
        action == CRAFTAX_ACTION_MAKE_DIAMOND_ARMOUR
        && can_craft_diamond_armour
        && is_at_crafting_table;
    inventory->diamond -= 3 * (int32_t)is_crafting_diamond_armour;
    inventory->armour[diamond_armour_index_to_craft] =
        (int32_t)is_crafting_diamond_armour * 2
        + (1 - (int32_t)is_crafting_diamond_armour)
        * inventory->armour[diamond_armour_index_to_craft];
    state->achievements[CRAFTAX_ACH_MAKE_DIAMOND_ARMOUR] =
        state->achievements[CRAFTAX_ACH_MAKE_DIAMOND_ARMOUR]
        || is_crafting_diamond_armour;

    bool can_craft_arrow = inventory->stone >= 1 && inventory->wood >= 1;
    bool is_crafting_arrow =
        action == CRAFTAX_ACTION_MAKE_ARROW
        && can_craft_arrow
        && is_at_crafting_table
        && inventory->arrows < 99;
    inventory->wood -= 1 * (int32_t)is_crafting_arrow;
    inventory->stone -= 1 * (int32_t)is_crafting_arrow;
    inventory->arrows += 2 * (int32_t)is_crafting_arrow;

    bool can_craft_torch = inventory->coal >= 1 && inventory->wood >= 1;
    bool is_crafting_torch =
        action == CRAFTAX_ACTION_MAKE_TORCH
        && can_craft_torch
        && is_at_crafting_table
        && inventory->torches < 99;
    inventory->wood -= 1 * (int32_t)is_crafting_torch;
    inventory->coal -= 1 * (int32_t)is_crafting_torch;
    inventory->torches += 4 * (int32_t)is_crafting_torch;
}

static inline bool craftax_crafting_can_place_item(int32_t block) {
    switch (block) {
    case CRAFTAX_BLOCK_GRASS:
    case CRAFTAX_BLOCK_SAND:
    case CRAFTAX_BLOCK_PATH:
    case CRAFTAX_BLOCK_FIRE_GRASS:
    case CRAFTAX_BLOCK_ICE_GRASS:
        return true;
    default:
        return false;
    }
}

static inline float craftax_crafting_torch_light(int32_t row, int32_t col) {
    static const float torch_light_map[9][9] = {
        {0.0f, 0.0f, 0.10557288f, 0.17537886f, 0.19999999f, 0.17537886f, 0.10557288f, 0.0f, 0.0f},
        {0.0f, 0.15147191f, 0.27888972f, 0.36754447f, 0.39999998f, 0.36754447f, 0.27888972f, 0.15147191f, 0.0f},
        {0.10557288f, 0.27888972f, 0.43431455f, 0.55278647f, 0.6f, 0.55278647f, 0.43431455f, 0.27888972f, 0.10557288f},
        {0.17537886f, 0.36754447f, 0.55278647f, 0.71715724f, 0.8f, 0.71715724f, 0.55278647f, 0.36754447f, 0.17537886f},
        {0.19999999f, 0.39999998f, 0.6f, 0.8f, 1.0f, 0.8f, 0.6f, 0.39999998f, 0.19999999f},
        {0.17537886f, 0.36754447f, 0.55278647f, 0.71715724f, 0.8f, 0.71715724f, 0.55278647f, 0.36754447f, 0.17537886f},
        {0.10557288f, 0.27888972f, 0.43431455f, 0.55278647f, 0.6f, 0.55278647f, 0.43431455f, 0.27888972f, 0.10557288f},
        {0.0f, 0.15147191f, 0.27888972f, 0.36754447f, 0.39999998f, 0.36754447f, 0.27888972f, 0.15147191f, 0.0f},
        {0.0f, 0.0f, 0.10557288f, 0.17537886f, 0.19999999f, 0.17537886f, 0.10557288f, 0.0f, 0.0f},
    };
    return torch_light_map[row][col];
}

static inline void craftax_crafting_add_torch_light(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col
) {
    for (int32_t dr = -4; dr <= 4; dr++) {
        int32_t map_row = row + dr;
        if (map_row < 0 || map_row >= CRAFTAX_MAP_SIZE) {
            continue;
        }
        for (int32_t dc = -4; dc <= 4; dc++) {
            int32_t map_col = col + dc;
            if (map_col < 0 || map_col >= CRAFTAX_MAP_SIZE) {
                continue;
            }
            float light = state->light_map[level][map_row][map_col] / 255.0f
                + craftax_crafting_torch_light(dr + 4, dc + 4);
            state->light_map[level][map_row][map_col] =
                (uint8_t)(craftax_step_minf32(craftax_step_maxf32(light, 0.0f), 1.0f) * 255.0f);
        }
    }
}

static inline void craftax_add_new_growing_plant_native(
    CraftaxState* state,
    const int32_t position[2],
    bool is_placing_sapling
) {
    int32_t plant_index = 0;
    int32_t empty_count = 0;
    for (int32_t i = 0; i < CRAFTAX_MAX_GROWING_PLANTS; i++) {
        bool is_empty = !state->growing_plants_mask[i];
        plant_index = (empty_count == 0 && is_empty) ? i : plant_index;
        empty_count += (int32_t)is_empty;
    }

    bool is_adding_plant = empty_count > 0 && is_placing_sapling;
    if (!is_adding_plant) {
        return;
    }

    state->growing_plants_positions[plant_index][0] = position[0];
    state->growing_plants_positions[plant_index][1] = position[1];
    state->growing_plants_age[plant_index] = 0;
    state->growing_plants_mask[plant_index] = true;
}

static inline void craftax_place_block_native(
    CraftaxState* state,
    int32_t action
) {
    int32_t direction[2];
    craftax_step_direction(state->player_direction, direction);

    int32_t row = state->player_position[0] + direction[0];
    int32_t col = state->player_position[1] + direction[1];
    bool in_bounds = row >= 0
        && row < CRAFTAX_MAP_SIZE
        && col >= 0
        && col < CRAFTAX_MAP_SIZE;
    bool in_mob = in_bounds && craftax_step_is_in_mob(state, row, col);
    if (!in_bounds || in_mob) {
        return;
    }

    int32_t level = craftax_step_jax_index(
        state->player_level,
        CRAFTAX_NUM_LEVELS
    );
    int32_t original_block = state->map[level][row][col];
    int32_t original_item = state->item_map[level][row][col];
    bool is_placement_on_solid_block_or_item =
        craftax_step_is_solid_block(original_block)
        || original_item != CRAFTAX_ITEM_NONE;

    CraftaxInventory* inventory = &state->inventory;

    bool is_placing_crafting_table =
        action == CRAFTAX_ACTION_PLACE_TABLE
        && !is_placement_on_solid_block_or_item
        && inventory->wood >= 2;
    if (is_placing_crafting_table) {
        craftax_set_map_block(state, level, row, col, CRAFTAX_BLOCK_CRAFTING_TABLE);
    }
    inventory->wood -= 2 * (int32_t)is_placing_crafting_table;
    state->achievements[CRAFTAX_ACH_PLACE_TABLE] =
        state->achievements[CRAFTAX_ACH_PLACE_TABLE]
        || is_placing_crafting_table;

    bool is_placing_furnace =
        action == CRAFTAX_ACTION_PLACE_FURNACE
        && !is_placement_on_solid_block_or_item
        && inventory->stone > 0;
    if (is_placing_furnace) {
        craftax_set_map_block(state, level, row, col, CRAFTAX_BLOCK_FURNACE);
    }
    inventory->stone -= 1 * (int32_t)is_placing_furnace;
    state->achievements[CRAFTAX_ACH_PLACE_FURNACE] =
        state->achievements[CRAFTAX_ACH_PLACE_FURNACE]
        || is_placing_furnace;

    bool is_placing_on_valid_stone_block =
        original_block == CRAFTAX_BLOCK_WATER
        || !is_placement_on_solid_block_or_item;
    bool is_placing_stone =
        action == CRAFTAX_ACTION_PLACE_STONE
        && is_placing_on_valid_stone_block
        && inventory->stone > 0;
    if (is_placing_stone) {
        craftax_set_map_block(state, level, row, col, CRAFTAX_BLOCK_STONE);
    }
    inventory->stone -= 1 * (int32_t)is_placing_stone;
    state->achievements[CRAFTAX_ACH_PLACE_STONE] =
        state->achievements[CRAFTAX_ACH_PLACE_STONE]
        || is_placing_stone;

    bool is_placing_on_valid_torch_block =
        craftax_crafting_can_place_item(original_block)
        && state->item_map[level][row][col] == CRAFTAX_ITEM_NONE;
    bool is_placing_torch =
        action == CRAFTAX_ACTION_PLACE_TORCH
        && is_placing_on_valid_torch_block
        && inventory->torches > 0;
    if (is_placing_torch) {
        state->item_map[level][row][col] = CRAFTAX_ITEM_TORCH;
        craftax_crafting_add_torch_light(state, level, row, col);
    }
    inventory->torches -= 1 * (int32_t)is_placing_torch;
    state->achievements[CRAFTAX_ACH_PLACE_TORCH] =
        state->achievements[CRAFTAX_ACH_PLACE_TORCH]
        || is_placing_torch;

    bool is_placing_sapling =
        action == CRAFTAX_ACTION_PLACE_PLANT
        && state->map[level][row][col] == CRAFTAX_BLOCK_GRASS
        && inventory->sapling > 0
        && state->item_map[level][row][col] == CRAFTAX_ITEM_NONE;
    if (is_placing_sapling) {
        int32_t position[2] = {row, col};
        craftax_set_map_block(state, level, row, col, CRAFTAX_BLOCK_PLANT);
        craftax_add_new_growing_plant_native(
            state,
            position,
            is_placing_sapling
        );
    }
    inventory->sapling -= 1 * (int32_t)is_placing_sapling;
    state->achievements[CRAFTAX_ACH_PLACE_PLANT] =
        state->achievements[CRAFTAX_ACH_PLACE_PLANT]
        || is_placing_sapling;
}
