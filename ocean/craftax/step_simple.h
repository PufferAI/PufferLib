// Standalone native ports of simple Craftax step subsystems.
//
// These helpers intentionally are not integrated into c_step yet. They mutate a
// full CraftaxState in place so tests can compare each subsystem directly
// against the installed JAX implementation.

#pragma once

#include "craftax.h"

static inline int32_t craftax_step_jax_index(int32_t index, int32_t size) {
    if (index < 0) {
        index += size;
    }
    if (index < 0) {
        return 0;
    }
    if (index >= size) {
        return size - 1;
    }
    return index;
}

static inline int32_t craftax_step_mini32(int32_t a, int32_t b) {
    return a < b ? a : b;
}

static inline int32_t craftax_step_maxi32(int32_t a, int32_t b) {
    return a > b ? a : b;
}

static inline float craftax_step_minf32(float a, float b) {
    if (isnan(a) || isnan(b)) {
        return NAN;
    }
    return a < b ? a : b;
}

static inline float craftax_step_maxf32(float a, float b) {
    if (isnan(a) || isnan(b)) {
        return NAN;
    }
    return a > b ? a : b;
}

static inline int32_t craftax_step_get_max_health(const CraftaxState* state) {
    return 8 + state->player_strength;
}

static inline int32_t craftax_step_get_max_food(const CraftaxState* state) {
    return 7 + 2 * state->player_dexterity;
}

static inline int32_t craftax_step_get_max_drink(const CraftaxState* state) {
    return 7 + 2 * state->player_dexterity;
}

static inline int32_t craftax_step_get_max_energy(const CraftaxState* state) {
    return 7 + 2 * state->player_dexterity;
}

static inline int32_t craftax_step_get_max_mana(const CraftaxState* state) {
    return 6 + 3 * state->player_intelligence;
}

static inline bool craftax_step_is_fighting_boss(const CraftaxState* state) {
    return state->player_level == CRAFTAX_NUM_LEVELS - 1;
}

static inline bool craftax_step_has_beaten_boss(const CraftaxState* state) {
    return state->boss_progress >= CRAFTAX_NUM_LEVELS - 1;
}

static inline void craftax_step_direction(int32_t action, int32_t direction[2]) {
    direction[0] = 0;
    direction[1] = 0;
    int32_t direction_index = craftax_step_jax_index(action, 16);
    if (direction_index == CRAFTAX_ACTION_LEFT) {
        direction[1] = -1;
    } else if (direction_index == CRAFTAX_ACTION_RIGHT) {
        direction[1] = 1;
    } else if (direction_index == CRAFTAX_ACTION_UP) {
        direction[0] = -1;
    } else if (direction_index == CRAFTAX_ACTION_DOWN) {
        direction[0] = 1;
    }
}

static inline bool craftax_step_is_solid_block(int32_t block) {
    switch (block) {
    case CRAFTAX_BLOCK_STONE:
    case CRAFTAX_BLOCK_TREE:
    case CRAFTAX_BLOCK_COAL:
    case CRAFTAX_BLOCK_IRON:
    case CRAFTAX_BLOCK_DIAMOND:
    case CRAFTAX_BLOCK_CRAFTING_TABLE:
    case CRAFTAX_BLOCK_FURNACE:
    case CRAFTAX_BLOCK_PLANT:
    case CRAFTAX_BLOCK_RIPE_PLANT:
    case CRAFTAX_BLOCK_WALL:
    case CRAFTAX_BLOCK_WALL_MOSS:
    case CRAFTAX_BLOCK_STALAGMITE:
    case CRAFTAX_BLOCK_RUBY:
    case CRAFTAX_BLOCK_SAPPHIRE:
    case CRAFTAX_BLOCK_CHEST:
    case CRAFTAX_BLOCK_FOUNTAIN:
    case CRAFTAX_BLOCK_FIRE_TREE:
    case CRAFTAX_BLOCK_ENCHANTMENT_TABLE_FIRE:
    case CRAFTAX_BLOCK_ENCHANTMENT_TABLE_ICE:
    case CRAFTAX_BLOCK_GRAVE:
    case CRAFTAX_BLOCK_GRAVE2:
    case CRAFTAX_BLOCK_GRAVE3:
    case CRAFTAX_BLOCK_NECROMANCER:
        return true;
    default:
        return false;
    }
}

static inline bool craftax_step_is_in_mob(
    const CraftaxState* state,
    int32_t row,
    int32_t col
) {
    int32_t level = craftax_step_jax_index(state->player_level, CRAFTAX_NUM_LEVELS);
    int32_t map_row = craftax_step_jax_index(row, CRAFTAX_MAP_SIZE);
    int32_t map_col = craftax_step_jax_index(col, CRAFTAX_MAP_SIZE);
    bool player_here = state->player_position[0] == row
        && state->player_position[1] == col;
    return ((state->mob_bits[level][map_row] >> map_col) & 1ULL) || player_here;
}

static inline bool craftax_step_valid_land_position(
    const CraftaxState* state,
    int32_t row,
    int32_t col
) {
    bool pos_in_bounds = row >= 0
        && row < CRAFTAX_MAP_SIZE
        && col >= 0
        && col < CRAFTAX_MAP_SIZE;
    int32_t level = craftax_step_jax_index(state->player_level, CRAFTAX_NUM_LEVELS);
    int32_t map_row = craftax_step_jax_index(row, CRAFTAX_MAP_SIZE);
    int32_t map_col = craftax_step_jax_index(col, CRAFTAX_MAP_SIZE);
    int32_t block = state->map[level][map_row][map_col];
    bool in_solid_block = craftax_step_is_solid_block(block);
    bool in_mob = craftax_step_is_in_mob(state, row, col);
    bool in_lava = block == CRAFTAX_BLOCK_LAVA;
    bool in_water = block == CRAFTAX_BLOCK_WATER;

    bool valid_move = pos_in_bounds && !in_mob && !in_solid_block;
    valid_move = valid_move && !in_water;
    valid_move = valid_move && !in_lava;
    return valid_move;
}

static inline void craftax_move_player_native(
    CraftaxState* state,
    int32_t action,
    bool god_mode
) {
    int32_t direction[2];
    craftax_step_direction(action, direction);

    int32_t proposed_row = state->player_position[0] + direction[0];
    int32_t proposed_col = state->player_position[1] + direction[1];
    bool valid_move = craftax_step_valid_land_position(
        state,
        proposed_row,
        proposed_col
    );
    valid_move = valid_move || god_mode;

    state->player_position[0] += (int32_t)valid_move * direction[0];
    state->player_position[1] += (int32_t)valid_move * direction[1];

    bool is_new_direction = direction[0] != 0 || direction[1] != 0;
    state->player_direction = state->player_direction * (1 - (int32_t)is_new_direction)
        + action * (int32_t)is_new_direction;
}

static inline void craftax_update_plants_native(CraftaxState* state) {
    bool finished_growing_plants[CRAFTAX_MAX_GROWING_PLANTS];

    for (int plant = 0; plant < CRAFTAX_MAX_GROWING_PLANTS; plant++) {
        state->growing_plants_age[plant] =
            (state->growing_plants_age[plant] + 1)
            * (int32_t)state->growing_plants_mask[plant];
        finished_growing_plants[plant] = state->growing_plants_age[plant] >= 600;
    }

    for (int plant = 0; plant < CRAFTAX_MAX_GROWING_PLANTS; plant++) {
        int32_t row = craftax_step_jax_index(
            state->growing_plants_positions[plant][0],
            CRAFTAX_MAP_SIZE
        );
        int32_t col = craftax_step_jax_index(
            state->growing_plants_positions[plant][1],
            CRAFTAX_MAP_SIZE
        );
        int32_t new_block = finished_growing_plants[plant]
            ? CRAFTAX_BLOCK_RIPE_PLANT
            : state->map[0][row][col];
        craftax_set_map_block(state, 0, row, col, new_block);
    }
}

static inline void craftax_boss_logic_native(CraftaxState* state) {
    state->achievements[CRAFTAX_ACH_DEFEAT_NECROMANCER] =
        state->achievements[CRAFTAX_ACH_DEFEAT_NECROMANCER]
        || craftax_step_has_beaten_boss(state);
    state->boss_timesteps_to_spawn_this_round -=
        (int32_t)craftax_step_is_fighting_boss(state);
}

static inline void craftax_level_up_attributes_native(
    CraftaxState* state,
    int32_t action,
    int32_t max_attribute
) {
    bool can_level_up = state->player_xp >= 1;
    bool is_levelling_up_dex = can_level_up
        && action == CRAFTAX_ACTION_LEVEL_UP_DEXTERITY
        && state->player_dexterity < max_attribute;
    bool is_levelling_up_str = can_level_up
        && action == CRAFTAX_ACTION_LEVEL_UP_STRENGTH
        && state->player_strength < max_attribute;
    bool is_levelling_up_int = can_level_up
        && action == CRAFTAX_ACTION_LEVEL_UP_INTELLIGENCE
        && state->player_intelligence < max_attribute;
    bool is_levelling_up = is_levelling_up_dex
        || is_levelling_up_str
        || is_levelling_up_int;

    state->player_dexterity += (int32_t)is_levelling_up_dex;
    state->player_strength += (int32_t)is_levelling_up_str;
    state->player_intelligence += (int32_t)is_levelling_up_int;
    state->player_xp -= (int32_t)is_levelling_up;
}

static inline void craftax_clip_inventory_and_intrinsics_native(
    CraftaxState* state,
    bool god_mode
) {
    state->inventory.wood = craftax_step_mini32(state->inventory.wood, 99);
    state->inventory.stone = craftax_step_mini32(state->inventory.stone, 99);
    state->inventory.coal = craftax_step_mini32(state->inventory.coal, 99);
    state->inventory.iron = craftax_step_mini32(state->inventory.iron, 99);
    state->inventory.diamond = craftax_step_mini32(state->inventory.diamond, 99);
    state->inventory.sapling = craftax_step_mini32(state->inventory.sapling, 99);
    state->inventory.pickaxe = craftax_step_mini32(state->inventory.pickaxe, 99);
    state->inventory.sword = craftax_step_mini32(state->inventory.sword, 99);
    state->inventory.bow = craftax_step_mini32(state->inventory.bow, 99);
    state->inventory.arrows = craftax_step_mini32(state->inventory.arrows, 99);
    for (int i = 0; i < 4; i++) {
        state->inventory.armour[i] = craftax_step_mini32(
            state->inventory.armour[i],
            99
        );
    }
    state->inventory.torches = craftax_step_mini32(state->inventory.torches, 99);
    state->inventory.ruby = craftax_step_mini32(state->inventory.ruby, 99);
    state->inventory.sapphire = craftax_step_mini32(state->inventory.sapphire, 99);
    for (int i = 0; i < 6; i++) {
        state->inventory.potions[i] = craftax_step_mini32(
            state->inventory.potions[i],
            99
        );
    }
    state->inventory.books = craftax_step_mini32(state->inventory.books, 99);

    float min_health = god_mode ? 9.0f : 0.0f;
    state->player_health = craftax_step_minf32(
        craftax_step_maxf32(state->player_health, min_health),
        (float)craftax_step_get_max_health(state)
    );
    state->player_food = craftax_step_mini32(
        craftax_step_maxi32(state->player_food, 0),
        craftax_step_get_max_food(state)
    );
    state->player_drink = craftax_step_mini32(
        craftax_step_maxi32(state->player_drink, 0),
        craftax_step_get_max_drink(state)
    );
    state->player_energy = craftax_step_mini32(
        craftax_step_maxi32(state->player_energy, 0),
        craftax_step_get_max_energy(state)
    );
    state->player_mana = craftax_step_mini32(
        craftax_step_maxi32(state->player_mana, 0),
        craftax_step_get_max_mana(state)
    );
}

static inline void craftax_calculate_inventory_achievements_native(
    CraftaxState* state
) {
    state->achievements[CRAFTAX_ACH_COLLECT_WOOD] =
        state->achievements[CRAFTAX_ACH_COLLECT_WOOD] || state->inventory.wood > 0;
    state->achievements[CRAFTAX_ACH_COLLECT_STONE] =
        state->achievements[CRAFTAX_ACH_COLLECT_STONE] || state->inventory.stone > 0;
    state->achievements[CRAFTAX_ACH_COLLECT_COAL] =
        state->achievements[CRAFTAX_ACH_COLLECT_COAL] || state->inventory.coal > 0;
    state->achievements[CRAFTAX_ACH_COLLECT_IRON] =
        state->achievements[CRAFTAX_ACH_COLLECT_IRON] || state->inventory.iron > 0;
    state->achievements[CRAFTAX_ACH_COLLECT_DIAMOND] =
        state->achievements[CRAFTAX_ACH_COLLECT_DIAMOND] || state->inventory.diamond > 0;
    state->achievements[CRAFTAX_ACH_COLLECT_RUBY] =
        state->achievements[CRAFTAX_ACH_COLLECT_RUBY] || state->inventory.ruby > 0;
    state->achievements[CRAFTAX_ACH_COLLECT_SAPPHIRE] =
        state->achievements[CRAFTAX_ACH_COLLECT_SAPPHIRE]
        || state->inventory.sapphire > 0;
    state->achievements[CRAFTAX_ACH_COLLECT_SAPLING] =
        state->achievements[CRAFTAX_ACH_COLLECT_SAPLING]
        || state->inventory.sapling > 0;
    state->achievements[CRAFTAX_ACH_FIND_BOW] =
        state->achievements[CRAFTAX_ACH_FIND_BOW] || state->inventory.bow > 0;
    state->achievements[CRAFTAX_ACH_MAKE_ARROW] =
        state->achievements[CRAFTAX_ACH_MAKE_ARROW] || state->inventory.arrows > 0;
    state->achievements[CRAFTAX_ACH_MAKE_TORCH] =
        state->achievements[CRAFTAX_ACH_MAKE_TORCH] || state->inventory.torches > 0;

    state->achievements[CRAFTAX_ACH_MAKE_WOOD_PICKAXE] =
        state->achievements[CRAFTAX_ACH_MAKE_WOOD_PICKAXE]
        || state->inventory.pickaxe >= 1;
    state->achievements[CRAFTAX_ACH_MAKE_STONE_PICKAXE] =
        state->achievements[CRAFTAX_ACH_MAKE_STONE_PICKAXE]
        || state->inventory.pickaxe >= 2;
    state->achievements[CRAFTAX_ACH_MAKE_IRON_PICKAXE] =
        state->achievements[CRAFTAX_ACH_MAKE_IRON_PICKAXE]
        || state->inventory.pickaxe >= 3;
    state->achievements[CRAFTAX_ACH_MAKE_DIAMOND_PICKAXE] =
        state->achievements[CRAFTAX_ACH_MAKE_DIAMOND_PICKAXE]
        || state->inventory.pickaxe >= 4;

    state->achievements[CRAFTAX_ACH_MAKE_WOOD_SWORD] =
        state->achievements[CRAFTAX_ACH_MAKE_WOOD_SWORD]
        || state->inventory.sword >= 1;
    state->achievements[CRAFTAX_ACH_MAKE_STONE_SWORD] =
        state->achievements[CRAFTAX_ACH_MAKE_STONE_SWORD]
        || state->inventory.sword >= 2;
    state->achievements[CRAFTAX_ACH_MAKE_IRON_SWORD] =
        state->achievements[CRAFTAX_ACH_MAKE_IRON_SWORD]
        || state->inventory.sword >= 3;
    state->achievements[CRAFTAX_ACH_MAKE_DIAMOND_SWORD] =
        state->achievements[CRAFTAX_ACH_MAKE_DIAMOND_SWORD]
        || state->inventory.sword >= 4;
}

static inline void craftax_update_player_intrinsics_native(
    CraftaxState* state,
    int32_t action
) {
    bool is_starting_sleep = action == CRAFTAX_ACTION_SLEEP
        && state->player_energy < craftax_step_get_max_energy(state);
    state->is_sleeping = state->is_sleeping || is_starting_sleep;

    bool is_waking_up = state->player_energy >= craftax_step_get_max_energy(state)
        && state->is_sleeping;
    state->is_sleeping = state->is_sleeping && !is_waking_up;
    state->achievements[CRAFTAX_ACH_WAKE_UP] =
        state->achievements[CRAFTAX_ACH_WAKE_UP] || is_waking_up;

    bool is_starting_rest = action == CRAFTAX_ACTION_REST
        && state->player_health < (float)craftax_step_get_max_health(state);
    state->is_resting = state->is_resting || is_starting_rest;

    is_waking_up = state->is_resting
        && (
            state->player_health >= (float)craftax_step_get_max_health(state)
            || state->player_food <= 0
            || state->player_drink <= 0
        );
    state->is_resting = state->is_resting && !is_waking_up;

    bool not_boss = !craftax_step_is_fighting_boss(state);
    float intrinsic_decay_coeff =
        1.0f - (0.125f * (float)(state->player_dexterity - 1));

    float hunger_add = (state->is_sleeping ? 0.5f : 1.0f) * intrinsic_decay_coeff;
    float new_hunger = state->player_hunger + hunger_add;
    int32_t hungered_food = craftax_step_maxi32(
        state->player_food - (int32_t)not_boss,
        0
    );
    int32_t new_food = new_hunger > 25.0f ? hungered_food : state->player_food;
    new_hunger = new_hunger > 25.0f ? 0.0f : new_hunger;
    state->player_hunger = new_hunger;
    state->player_food = new_food;

    float thirst_add = (state->is_sleeping ? 0.5f : 1.0f) * intrinsic_decay_coeff;
    float new_thirst = state->player_thirst + thirst_add;
    int32_t thirsted_drink = craftax_step_maxi32(
        state->player_drink - (int32_t)not_boss,
        0
    );
    int32_t new_drink = new_thirst > 20.0f ? thirsted_drink : state->player_drink;
    new_thirst = new_thirst > 20.0f ? 0.0f : new_thirst;
    state->player_thirst = new_thirst;
    state->player_drink = new_drink;

    float new_fatigue = state->is_sleeping
        ? craftax_step_minf32(state->player_fatigue - 1.0f, 0.0f)
        : state->player_fatigue + intrinsic_decay_coeff;
    int32_t new_energy = new_fatigue > 30.0f
        ? craftax_step_maxi32(state->player_energy - (int32_t)not_boss, 0)
        : state->player_energy;
    new_fatigue = new_fatigue > 30.0f ? 0.0f : new_fatigue;
    new_energy = new_fatigue < -10.0f
        ? craftax_step_mini32(
            state->player_energy + 1,
            craftax_step_get_max_energy(state)
        )
        : new_energy;
    new_fatigue = new_fatigue < -10.0f ? 0.0f : new_fatigue;
    state->player_fatigue = new_fatigue;
    state->player_energy = new_energy;

    bool all_necessities = state->player_food > 0
        && state->player_drink > 0
        && (state->player_energy > 0 || state->is_sleeping);
    float recover_all = state->is_sleeping ? 2.0f : 1.0f;
    float recover_not_all = (state->is_sleeping ? -0.5f : -1.0f)
        * (float)(int32_t)not_boss;
    float recover_add = all_necessities ? recover_all : recover_not_all;
    float new_recover = state->player_recover + recover_add;

    float recovered_health = craftax_step_minf32(
        state->player_health + 1.0f,
        (float)craftax_step_get_max_health(state)
    );
    float derecovered_health = state->player_health - 1.0f;
    float new_health = new_recover > 25.0f
        ? recovered_health
        : state->player_health;
    new_recover = new_recover > 25.0f ? 0.0f : new_recover;
    new_health = new_recover < -15.0f ? derecovered_health : new_health;
    new_recover = new_recover < -15.0f ? 0.0f : new_recover;
    state->player_recover = new_recover;
    state->player_health = new_health;

    float mana_recover_coeff =
        1.0f + 0.25f * (float)(state->player_intelligence - 1);
    float new_recover_mana = (
        state->is_sleeping
            ? state->player_recover_mana + 2.0f
            : state->player_recover_mana + 1.0f
    ) * mana_recover_coeff;
    int32_t new_mana = new_recover_mana > 30.0f
        ? state->player_mana + 1
        : state->player_mana;
    new_recover_mana = new_recover_mana > 30.0f ? 0.0f : new_recover_mana;
    state->player_recover_mana = new_recover_mana;
    state->player_mana = new_mana;
}

static inline void craftax_drink_potion_native(
    CraftaxState* state,
    int32_t action
) {
    int32_t drinking_potion_index = -1;
    bool is_drinking_potion = false;

    bool is_drinking_red_potion = action == CRAFTAX_ACTION_DRINK_POTION_RED
        && state->inventory.potions[0] > 0;
    drinking_potion_index = (int32_t)is_drinking_red_potion * 0
        + (1 - (int32_t)is_drinking_red_potion) * drinking_potion_index;
    is_drinking_potion = is_drinking_potion || is_drinking_red_potion;

    bool is_drinking_green_potion = action == CRAFTAX_ACTION_DRINK_POTION_GREEN
        && state->inventory.potions[1] > 0;
    drinking_potion_index = (int32_t)is_drinking_green_potion * 1
        + (1 - (int32_t)is_drinking_green_potion) * drinking_potion_index;
    is_drinking_potion = is_drinking_potion || is_drinking_green_potion;

    bool is_drinking_blue_potion = action == CRAFTAX_ACTION_DRINK_POTION_BLUE
        && state->inventory.potions[2] > 0;
    drinking_potion_index = (int32_t)is_drinking_blue_potion * 2
        + (1 - (int32_t)is_drinking_blue_potion) * drinking_potion_index;
    is_drinking_potion = is_drinking_potion || is_drinking_blue_potion;

    bool is_drinking_pink_potion = action == CRAFTAX_ACTION_DRINK_POTION_PINK
        && state->inventory.potions[3] > 0;
    drinking_potion_index = (int32_t)is_drinking_pink_potion * 3
        + (1 - (int32_t)is_drinking_pink_potion) * drinking_potion_index;
    is_drinking_potion = is_drinking_potion || is_drinking_pink_potion;

    bool is_drinking_cyan_potion = action == CRAFTAX_ACTION_DRINK_POTION_CYAN
        && state->inventory.potions[4] > 0;
    drinking_potion_index = (int32_t)is_drinking_cyan_potion * 4
        + (1 - (int32_t)is_drinking_cyan_potion) * drinking_potion_index;
    is_drinking_potion = is_drinking_potion || is_drinking_cyan_potion;

    bool is_drinking_yellow_potion = action == CRAFTAX_ACTION_DRINK_POTION_YELLOW
        && state->inventory.potions[5] > 0;
    drinking_potion_index = (int32_t)is_drinking_yellow_potion * 5
        + (1 - (int32_t)is_drinking_yellow_potion) * drinking_potion_index;
    is_drinking_potion = is_drinking_potion || is_drinking_yellow_potion;

    int32_t potion_index = craftax_step_jax_index(drinking_potion_index, 6);
    int32_t potion_effect_index = state->potion_mapping[potion_index];

    int32_t delta_health = 0;
    delta_health += (int32_t)is_drinking_potion * (int32_t)(potion_effect_index == 0) * 8;
    delta_health += (int32_t)is_drinking_potion * (int32_t)(potion_effect_index == 1) * -3;

    int32_t delta_mana = 0;
    delta_mana += (int32_t)is_drinking_potion * (int32_t)(potion_effect_index == 2) * 8;
    delta_mana += (int32_t)is_drinking_potion * (int32_t)(potion_effect_index == 3) * -3;

    int32_t delta_energy = 0;
    delta_energy += (int32_t)is_drinking_potion * (int32_t)(potion_effect_index == 4) * 8;
    delta_energy += (int32_t)is_drinking_potion * (int32_t)(potion_effect_index == 5) * -3;

    state->achievements[CRAFTAX_ACH_DRINK_POTION] =
        state->achievements[CRAFTAX_ACH_DRINK_POTION] || is_drinking_potion;
    state->inventory.potions[potion_index] =
        state->inventory.potions[potion_index] - (int32_t)is_drinking_potion;
    state->player_health += (float)delta_health;
    state->player_mana += delta_mana;
    state->player_energy += delta_energy;
}

static inline void craftax_read_book_native(
    CraftaxState* state,
    const uint32_t rng_words[2],
    int32_t action
) {
    bool is_reading_book = action == CRAFTAX_ACTION_READ_BOOK
        && state->inventory.books > 0;

    CraftaxThreefryKey rng = {{rng_words[0], rng_words[1]}};
    CraftaxThreefryKey unused;
    CraftaxThreefryKey choice_key;
    craftax_threefry_split(rng, &unused, &choice_key);

    float p0 = state->learned_spells[0] ? 0.0f : 1.0f;
    float p1 = state->learned_spells[1] ? 0.0f : 1.0f;
    float p_sum = p0 + p1;
    int32_t spell_to_learn_index = 0;
    if (p_sum != 0.0f) {
        p0 /= p_sum;
        float r = 1.0f - craftax_threefry_uniform_f32(choice_key);
        spell_to_learn_index = r <= p0 ? 0 : 1;
    }

    int32_t learn_spell_achievement = spell_to_learn_index
        ? CRAFTAX_ACH_LEARN_ICEBALL
        : CRAFTAX_ACH_LEARN_FIREBALL;

    state->achievements[learn_spell_achievement] =
        state->achievements[learn_spell_achievement] || is_reading_book;
    state->inventory.books -= (int32_t)is_reading_book;
    state->learned_spells[spell_to_learn_index] =
        state->learned_spells[spell_to_learn_index] || is_reading_book;
}
