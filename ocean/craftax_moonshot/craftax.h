// Full native Craftax environment for PufferLib Ocean.

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "worldgen.h"
#include "raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// ============================================================
// Optional step profiling (compile with -DCRAFTAX_PROFILE)
// ============================================================
#ifdef CRAFTAX_PROFILE

#define CRAFTAX_NUM_PROFILE_ZONES 18

typedef struct {
    const char* name;
    uint64_t total_ns;
    uint64_t count;
} CraftaxProfileZone;

static CraftaxProfileZone craftax_profile_zones[CRAFTAX_NUM_PROFILE_ZONES] = {
    {"change_floor", 0, 0},
    {"crafting", 0, 0},
    {"do_action", 0, 0},
    {"place+shoot+spell+potion", 0, 0},
    {"read_book", 0, 0},
    {"enchant", 0, 0},
    {"boss+attr+move", 0, 0},
    {"update_mobs", 0, 0},
    {"spawn_mobs", 0, 0},
    {"plants+intrinsics+achieve", 0, 0},
    {"reward+bookkeeping", 0, 0},
    {"encode_obs", 0, 0},
    {"rng_split", 0, 0},
    {"is_game_over", 0, 0},
    {"reset_on_done", 0, 0},
    {"copy_achievements", 0, 0},
    {"reward_bookkeeping", 0, 0},
    {"unprofiled", 0, 0},
};

static inline uint64_t craftax_profile_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline void craftax_profile_record(int zone, uint64_t start) {
    craftax_profile_zones[zone].total_ns += craftax_profile_now() - start;
    craftax_profile_zones[zone].count++;
}

static inline void craftax_profile_report(void) {
    fprintf(stderr, "\n=== Craftax Step Profile ===\n");
    uint64_t total = 0;
    for (int i = 0; i < CRAFTAX_NUM_PROFILE_ZONES; i++) {
        total += craftax_profile_zones[i].total_ns;
    }
    for (int i = 0; i < CRAFTAX_NUM_PROFILE_ZONES; i++) {
        CraftaxProfileZone* z = &craftax_profile_zones[i];
        if (z->count == 0) continue;
        double pct = total > 0 ? (100.0 * (double)z->total_ns / (double)total) : 0.0;
        double avg_us = (double)z->total_ns / (double)z->count / 1000.0;
        fprintf(stderr, "%-28s %8.3f%%  %10.2f us/step  (%lu calls)\n",
                z->name, pct, avg_us, (unsigned long)z->count);
    }
    fprintf(stderr, "%-28s %8.3f%%  %10.2f us/step\n",
            "TOTAL", 100.0, (double)total / (double)craftax_profile_zones[0].count / 1000.0);
}

#define CRAFTAX_PROFILE_START() uint64_t _prof_start = craftax_profile_now(); uint64_t _prof_zone_start;
#define CRAFTAX_PROFILE_ZONE(n) do { _prof_zone_start = craftax_profile_now(); } while(0)
#define CRAFTAX_PROFILE_END(n) craftax_profile_record((n), _prof_zone_start)
#define CRAFTAX_PROFILE_FINAL(n) craftax_profile_record((n), _prof_start)

#else

#define CRAFTAX_PROFILE_START() ((void)0)
#define CRAFTAX_PROFILE_ZONE(n) ((void)0)
#define CRAFTAX_PROFILE_END(n) ((void)0)
#define CRAFTAX_PROFILE_FINAL(n) ((void)0)
#define craftax_profile_report() ((void)0)

#endif // CRAFTAX_PROFILE

// ============================================================
// Constants
// ============================================================
#define CRAFTAX_OBS_ROWS 9
#define CRAFTAX_OBS_COLS 11
#define CRAFTAX_MAP_SIZE 48
#define CRAFTAX_NUM_LEVELS 9

#define CRAFTAX_NUM_BLOCK_TYPES 37
#define CRAFTAX_NUM_ITEM_TYPES 5
#define CRAFTAX_NUM_MOB_CLASSES 5
#define CRAFTAX_NUM_MOB_TYPES 8
#define CRAFTAX_INVENTORY_OBS_SIZE 51
#define CRAFTAX_OBS_SIZE CRAFTAX_WG_OBS_SIZE

#define CRAFTAX_NUM_ACTIONS 43
#define CRAFTAX_NUM_ACHIEVEMENTS 67

#define CRAFTAX_MAX_MELEE_MOBS 3
#define CRAFTAX_MAX_PASSIVE_MOBS 3
#define CRAFTAX_MAX_RANGED_MOBS 2
#define CRAFTAX_MAX_MOB_PROJECTILES 3
#define CRAFTAX_MAX_PLAYER_PROJECTILES 3
#define CRAFTAX_MAX_GROWING_PLANTS 10

#define CRAFTAX_DEFAULT_MAX_TIMESTEPS 100000
#define CRAFTAX_DAY_LENGTH 300
#define CRAFTAX_MAX_ATTRIBUTE 5
#define CRAFTAX_MOB_DESPAWN_DISTANCE 14
#define CRAFTAX_MONSTERS_KILLED_TO_CLEAR_LEVEL 8

// ============================================================
// Enums copied from craftax/craftax/constants.py
// ============================================================
typedef enum CraftaxBlockType {
    CRAFTAX_BLOCK_INVALID = 0,
    CRAFTAX_BLOCK_OUT_OF_BOUNDS = 1,
    CRAFTAX_BLOCK_GRASS = 2,
    CRAFTAX_BLOCK_WATER = 3,
    CRAFTAX_BLOCK_STONE = 4,
    CRAFTAX_BLOCK_TREE = 5,
    CRAFTAX_BLOCK_WOOD = 6,
    CRAFTAX_BLOCK_PATH = 7,
    CRAFTAX_BLOCK_COAL = 8,
    CRAFTAX_BLOCK_IRON = 9,
    CRAFTAX_BLOCK_DIAMOND = 10,
    CRAFTAX_BLOCK_CRAFTING_TABLE = 11,
    CRAFTAX_BLOCK_FURNACE = 12,
    CRAFTAX_BLOCK_SAND = 13,
    CRAFTAX_BLOCK_LAVA = 14,
    CRAFTAX_BLOCK_PLANT = 15,
    CRAFTAX_BLOCK_RIPE_PLANT = 16,
    CRAFTAX_BLOCK_WALL = 17,
    CRAFTAX_BLOCK_DARKNESS = 18,
    CRAFTAX_BLOCK_WALL_MOSS = 19,
    CRAFTAX_BLOCK_STALAGMITE = 20,
    CRAFTAX_BLOCK_SAPPHIRE = 21,
    CRAFTAX_BLOCK_RUBY = 22,
    CRAFTAX_BLOCK_CHEST = 23,
    CRAFTAX_BLOCK_FOUNTAIN = 24,
    CRAFTAX_BLOCK_FIRE_GRASS = 25,
    CRAFTAX_BLOCK_ICE_GRASS = 26,
    CRAFTAX_BLOCK_GRAVEL = 27,
    CRAFTAX_BLOCK_FIRE_TREE = 28,
    CRAFTAX_BLOCK_ICE_SHRUB = 29,
    CRAFTAX_BLOCK_ENCHANTMENT_TABLE_FIRE = 30,
    CRAFTAX_BLOCK_ENCHANTMENT_TABLE_ICE = 31,
    CRAFTAX_BLOCK_NECROMANCER = 32,
    CRAFTAX_BLOCK_GRAVE = 33,
    CRAFTAX_BLOCK_GRAVE2 = 34,
    CRAFTAX_BLOCK_GRAVE3 = 35,
    CRAFTAX_BLOCK_NECROMANCER_VULNERABLE = 36,
} CraftaxBlockType;

typedef enum CraftaxItemType {
    CRAFTAX_ITEM_NONE = 0,
    CRAFTAX_ITEM_TORCH = 1,
    CRAFTAX_ITEM_LADDER_DOWN = 2,
    CRAFTAX_ITEM_LADDER_UP = 3,
    CRAFTAX_ITEM_LADDER_DOWN_BLOCKED = 4,
} CraftaxItemType;

typedef enum CraftaxAction {
    CRAFTAX_ACTION_NOOP = 0,
    CRAFTAX_ACTION_LEFT = 1,
    CRAFTAX_ACTION_RIGHT = 2,
    CRAFTAX_ACTION_UP = 3,
    CRAFTAX_ACTION_DOWN = 4,
    CRAFTAX_ACTION_DO = 5,
    CRAFTAX_ACTION_SLEEP = 6,
    CRAFTAX_ACTION_PLACE_STONE = 7,
    CRAFTAX_ACTION_PLACE_TABLE = 8,
    CRAFTAX_ACTION_PLACE_FURNACE = 9,
    CRAFTAX_ACTION_PLACE_PLANT = 10,
    CRAFTAX_ACTION_MAKE_WOOD_PICKAXE = 11,
    CRAFTAX_ACTION_MAKE_STONE_PICKAXE = 12,
    CRAFTAX_ACTION_MAKE_IRON_PICKAXE = 13,
    CRAFTAX_ACTION_MAKE_WOOD_SWORD = 14,
    CRAFTAX_ACTION_MAKE_STONE_SWORD = 15,
    CRAFTAX_ACTION_MAKE_IRON_SWORD = 16,
    CRAFTAX_ACTION_REST = 17,
    CRAFTAX_ACTION_DESCEND = 18,
    CRAFTAX_ACTION_ASCEND = 19,
    CRAFTAX_ACTION_MAKE_DIAMOND_PICKAXE = 20,
    CRAFTAX_ACTION_MAKE_DIAMOND_SWORD = 21,
    CRAFTAX_ACTION_MAKE_IRON_ARMOUR = 22,
    CRAFTAX_ACTION_MAKE_DIAMOND_ARMOUR = 23,
    CRAFTAX_ACTION_SHOOT_ARROW = 24,
    CRAFTAX_ACTION_MAKE_ARROW = 25,
    CRAFTAX_ACTION_CAST_FIREBALL = 26,
    CRAFTAX_ACTION_CAST_ICEBALL = 27,
    CRAFTAX_ACTION_PLACE_TORCH = 28,
    CRAFTAX_ACTION_DRINK_POTION_RED = 29,
    CRAFTAX_ACTION_DRINK_POTION_GREEN = 30,
    CRAFTAX_ACTION_DRINK_POTION_BLUE = 31,
    CRAFTAX_ACTION_DRINK_POTION_PINK = 32,
    CRAFTAX_ACTION_DRINK_POTION_CYAN = 33,
    CRAFTAX_ACTION_DRINK_POTION_YELLOW = 34,
    CRAFTAX_ACTION_READ_BOOK = 35,
    CRAFTAX_ACTION_ENCHANT_SWORD = 36,
    CRAFTAX_ACTION_ENCHANT_ARMOUR = 37,
    CRAFTAX_ACTION_MAKE_TORCH = 38,
    CRAFTAX_ACTION_LEVEL_UP_DEXTERITY = 39,
    CRAFTAX_ACTION_LEVEL_UP_STRENGTH = 40,
    CRAFTAX_ACTION_LEVEL_UP_INTELLIGENCE = 41,
    CRAFTAX_ACTION_ENCHANT_BOW = 42,
} CraftaxAction;

typedef enum CraftaxMobType {
    CRAFTAX_MOB_PASSIVE = 0,
    CRAFTAX_MOB_MELEE = 1,
    CRAFTAX_MOB_RANGED = 2,
    CRAFTAX_MOB_PROJECTILE = 3,
} CraftaxMobType;

typedef enum CraftaxProjectileType {
    CRAFTAX_PROJECTILE_ARROW = 0,
    CRAFTAX_PROJECTILE_DAGGER = 1,
    CRAFTAX_PROJECTILE_FIREBALL = 2,
    CRAFTAX_PROJECTILE_ICEBALL = 3,
    CRAFTAX_PROJECTILE_ARROW2 = 4,
    CRAFTAX_PROJECTILE_SLIMEBALL = 5,
    CRAFTAX_PROJECTILE_FIREBALL2 = 6,
    CRAFTAX_PROJECTILE_ICEBALL2 = 7,
} CraftaxProjectileType;

typedef enum CraftaxAchievement {
    CRAFTAX_ACH_COLLECT_WOOD = 0,
    CRAFTAX_ACH_PLACE_TABLE = 1,
    CRAFTAX_ACH_EAT_COW = 2,
    CRAFTAX_ACH_COLLECT_SAPLING = 3,
    CRAFTAX_ACH_COLLECT_DRINK = 4,
    CRAFTAX_ACH_MAKE_WOOD_PICKAXE = 5,
    CRAFTAX_ACH_MAKE_WOOD_SWORD = 6,
    CRAFTAX_ACH_PLACE_PLANT = 7,
    CRAFTAX_ACH_DEFEAT_ZOMBIE = 8,
    CRAFTAX_ACH_COLLECT_STONE = 9,
    CRAFTAX_ACH_PLACE_STONE = 10,
    CRAFTAX_ACH_EAT_PLANT = 11,
    CRAFTAX_ACH_DEFEAT_SKELETON = 12,
    CRAFTAX_ACH_MAKE_STONE_PICKAXE = 13,
    CRAFTAX_ACH_MAKE_STONE_SWORD = 14,
    CRAFTAX_ACH_WAKE_UP = 15,
    CRAFTAX_ACH_PLACE_FURNACE = 16,
    CRAFTAX_ACH_COLLECT_COAL = 17,
    CRAFTAX_ACH_COLLECT_IRON = 18,
    CRAFTAX_ACH_COLLECT_DIAMOND = 19,
    CRAFTAX_ACH_MAKE_IRON_PICKAXE = 20,
    CRAFTAX_ACH_MAKE_IRON_SWORD = 21,
    CRAFTAX_ACH_MAKE_ARROW = 22,
    CRAFTAX_ACH_MAKE_TORCH = 23,
    CRAFTAX_ACH_PLACE_TORCH = 24,
    CRAFTAX_ACH_MAKE_DIAMOND_SWORD = 25,
    CRAFTAX_ACH_MAKE_IRON_ARMOUR = 26,
    CRAFTAX_ACH_MAKE_DIAMOND_ARMOUR = 27,
    CRAFTAX_ACH_ENTER_GNOMISH_MINES = 28,
    CRAFTAX_ACH_ENTER_DUNGEON = 29,
    CRAFTAX_ACH_ENTER_SEWERS = 30,
    CRAFTAX_ACH_ENTER_VAULT = 31,
    CRAFTAX_ACH_ENTER_TROLL_MINES = 32,
    CRAFTAX_ACH_ENTER_FIRE_REALM = 33,
    CRAFTAX_ACH_ENTER_ICE_REALM = 34,
    CRAFTAX_ACH_ENTER_GRAVEYARD = 35,
    CRAFTAX_ACH_DEFEAT_GNOME_WARRIOR = 36,
    CRAFTAX_ACH_DEFEAT_GNOME_ARCHER = 37,
    CRAFTAX_ACH_DEFEAT_ORC_SOLIDER = 38,
    CRAFTAX_ACH_DEFEAT_ORC_MAGE = 39,
    CRAFTAX_ACH_DEFEAT_LIZARD = 40,
    CRAFTAX_ACH_DEFEAT_KOBOLD = 41,
    CRAFTAX_ACH_DEFEAT_TROLL = 42,
    CRAFTAX_ACH_DEFEAT_DEEP_THING = 43,
    CRAFTAX_ACH_DEFEAT_PIGMAN = 44,
    CRAFTAX_ACH_DEFEAT_FIRE_ELEMENTAL = 45,
    CRAFTAX_ACH_DEFEAT_FROST_TROLL = 46,
    CRAFTAX_ACH_DEFEAT_ICE_ELEMENTAL = 47,
    CRAFTAX_ACH_DAMAGE_NECROMANCER = 48,
    CRAFTAX_ACH_DEFEAT_NECROMANCER = 49,
    CRAFTAX_ACH_EAT_BAT = 50,
    CRAFTAX_ACH_EAT_SNAIL = 51,
    CRAFTAX_ACH_FIND_BOW = 52,
    CRAFTAX_ACH_FIRE_BOW = 53,
    CRAFTAX_ACH_COLLECT_SAPPHIRE = 54,
    CRAFTAX_ACH_LEARN_FIREBALL = 55,
    CRAFTAX_ACH_CAST_FIREBALL = 56,
    CRAFTAX_ACH_LEARN_ICEBALL = 57,
    CRAFTAX_ACH_CAST_ICEBALL = 58,
    CRAFTAX_ACH_COLLECT_RUBY = 59,
    CRAFTAX_ACH_MAKE_DIAMOND_PICKAXE = 60,
    CRAFTAX_ACH_OPEN_CHEST = 61,
    CRAFTAX_ACH_DRINK_POTION = 62,
    CRAFTAX_ACH_ENCHANT_SWORD = 63,
    CRAFTAX_ACH_ENCHANT_ARMOUR = 64,
    CRAFTAX_ACH_DEFEAT_KNIGHT = 65,
    CRAFTAX_ACH_DEFEAT_ARCHER = 66,
} CraftaxAchievement;

// ============================================================
// State layout declarations matching craftax_state.py field order
// ============================================================
typedef struct CraftaxInventory {
    int32_t wood;
    int32_t stone;
    int32_t coal;
    int32_t iron;
    int32_t diamond;
    int32_t sapling;
    int32_t pickaxe;
    int32_t sword;
    int32_t bow;
    int32_t arrows;
    int32_t armour[4];
    int32_t torches;
    int32_t ruby;
    int32_t sapphire;
    int32_t potions[6];
    int32_t books;
} CraftaxInventory;

typedef struct CraftaxMobs3 {
    int32_t position[CRAFTAX_NUM_LEVELS][3][2];
    float health[CRAFTAX_NUM_LEVELS][3];
    bool mask[CRAFTAX_NUM_LEVELS][3];
    int32_t attack_cooldown[CRAFTAX_NUM_LEVELS][3];
    int32_t type_id[CRAFTAX_NUM_LEVELS][3];
} CraftaxMobs3;

typedef struct CraftaxMobs2 {
    int32_t position[CRAFTAX_NUM_LEVELS][2][2];
    float health[CRAFTAX_NUM_LEVELS][2];
    bool mask[CRAFTAX_NUM_LEVELS][2];
    int32_t attack_cooldown[CRAFTAX_NUM_LEVELS][2];
    int32_t type_id[CRAFTAX_NUM_LEVELS][2];
} CraftaxMobs2;

typedef struct CraftaxState {
    // === Hot data (accessed every step) ===
    int32_t player_position[2];
    int32_t player_level;
    int32_t player_direction;

    float player_health;
    int32_t player_food;
    int32_t player_drink;
    int32_t player_energy;
    int32_t player_mana;
    bool is_sleeping;
    bool is_resting;

    float player_recover;
    float player_hunger;
    float player_thirst;
    float player_fatigue;
    float player_recover_mana;

    int32_t player_xp;
    int32_t player_dexterity;
    int32_t player_strength;
    int32_t player_intelligence;

    CraftaxInventory inventory;

    CraftaxMobs3 melee_mobs;
    CraftaxMobs3 passive_mobs;
    CraftaxMobs2 ranged_mobs;

    CraftaxMobs3 mob_projectiles;
    int32_t mob_projectile_directions[CRAFTAX_NUM_LEVELS][CRAFTAX_MAX_MOB_PROJECTILES][2];
    CraftaxMobs3 player_projectiles;
    int32_t player_projectile_directions[CRAFTAX_NUM_LEVELS][CRAFTAX_MAX_PLAYER_PROJECTILES][2];

    int32_t growing_plants_positions[CRAFTAX_MAX_GROWING_PLANTS][2];
    int32_t growing_plants_age[CRAFTAX_MAX_GROWING_PLANTS];
    bool growing_plants_mask[CRAFTAX_MAX_GROWING_PLANTS];

    int32_t potion_mapping[6];
    bool learned_spells[2];

    int32_t sword_enchantment;
    int32_t bow_enchantment;
    int32_t armour_enchantments[4];

    int32_t boss_progress;
    int32_t boss_timesteps_to_spawn_this_round;

    float light_level;
    bool achievements[CRAFTAX_NUM_ACHIEVEMENTS];
    uint32_t state_rng[2];
    int32_t timestep;
    int32_t fractal_noise_angles[4];

    // === Medium-hot bitmaps, read during mob updates, spawn scans, encode_obs ===
    uint64_t mob_bits[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE];
    uint64_t spawn_all_bits[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE];
    uint64_t spawn_grave_bits[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE];
    uint64_t spawn_water_bits[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE];

    // === Cold data (large maps, scattered access) ===
    uint8_t map[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE][CRAFTAX_MAP_SIZE];
    uint8_t item_map[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE][CRAFTAX_MAP_SIZE];
    uint8_t light_map[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE][CRAFTAX_MAP_SIZE];

    int32_t down_ladders[CRAFTAX_NUM_LEVELS][2];
    int32_t up_ladders[CRAFTAX_NUM_LEVELS][2];
    bool chests_opened[CRAFTAX_NUM_LEVELS];
    int32_t monsters_killed[CRAFTAX_NUM_LEVELS];
} CraftaxState;

typedef char CraftaxStateMatchesWorldState[
    (sizeof(CraftaxState) == sizeof(CraftaxWorldState)) ? 1 : -1
];

static inline uint64_t craftax_spawn_all_bit(uint8_t block) {
    return (uint64_t)(
        block == CRAFTAX_BLOCK_GRASS
        || block == CRAFTAX_BLOCK_PATH
        || block == CRAFTAX_BLOCK_FIRE_GRASS
        || block == CRAFTAX_BLOCK_ICE_GRASS
    );
}

static inline uint64_t craftax_spawn_grave_bit(uint8_t block) {
    return (uint64_t)(
        block == CRAFTAX_BLOCK_GRAVE
        || block == CRAFTAX_BLOCK_GRAVE2
        || block == CRAFTAX_BLOCK_GRAVE3
    );
}

static inline uint64_t craftax_spawn_water_bit(uint8_t block) {
    return (uint64_t)(block == CRAFTAX_BLOCK_WATER);
}

static inline void craftax_refresh_spawn_bits_cell(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col
) {
    uint64_t bit = 1ULL << col;
    uint8_t block = state->map[level][row][col];

    state->spawn_all_bits[level][row] =
        (state->spawn_all_bits[level][row] & ~bit)
        | ((0ULL - craftax_spawn_all_bit(block)) & bit);
    state->spawn_grave_bits[level][row] =
        (state->spawn_grave_bits[level][row] & ~bit)
        | ((0ULL - craftax_spawn_grave_bit(block)) & bit);
    state->spawn_water_bits[level][row] =
        (state->spawn_water_bits[level][row] & ~bit)
        | ((0ULL - craftax_spawn_water_bit(block)) & bit);
}

static inline void craftax_set_map_block(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col,
    int32_t block
) {
    state->map[level][row][col] = (uint8_t)block;
    craftax_refresh_spawn_bits_cell(state, level, row, col);
}

static inline void craftax_refresh_spawn_bits_all(CraftaxState* state) {
    for (int32_t level = 0; level < CRAFTAX_NUM_LEVELS; level++) {
        for (int32_t row = 0; row < CRAFTAX_MAP_SIZE; row++) {
            uint64_t all_bits = 0;
            uint64_t grave_bits = 0;
            uint64_t water_bits = 0;
            for (int32_t col = 0; col < CRAFTAX_MAP_SIZE; col++) {
                uint8_t block = state->map[level][row][col];
                uint64_t bit = 1ULL << col;
                all_bits |= (0ULL - craftax_spawn_all_bit(block)) & bit;
                grave_bits |= (0ULL - craftax_spawn_grave_bit(block)) & bit;
                water_bits |= (0ULL - craftax_spawn_water_bit(block)) & bit;
            }
            state->spawn_all_bits[level][row] = all_bits;
            state->spawn_grave_bits[level][row] = grave_bits;
            state->spawn_water_bits[level][row] = water_bits;
        }
    }
}

#define CRAFTAX_ARENA_PACKET_SIZE 64

typedef struct CraftaxArena {
    CraftaxState* states;
    int num_envs;
    int packet_size;
    int num_packets;
} CraftaxArena;

#ifdef CRAFTAX_ENABLE_ENV_IMPL
static inline void craftax_change_floor_native(CraftaxState* state, int32_t action);
static inline void craftax_do_crafting_native(CraftaxState* state, int32_t action);
static inline void craftax_do_action_native(
    CraftaxState* state,
    int32_t action,
    CraftaxThreefryKey rng
);
static inline void craftax_place_block_native(CraftaxState* state, int32_t action);
static inline void craftax_shoot_projectile_native(
    CraftaxState* state,
    int32_t action
);
static inline void craftax_cast_spell_native(CraftaxState* state, int32_t action);
static inline void craftax_drink_potion_native(CraftaxState* state, int32_t action);
static inline void craftax_read_book_native(
    CraftaxState* state,
    const uint32_t rng_words[2],
    int32_t action
);
static inline void craftax_enchant_native(
    CraftaxState* state,
    int32_t action,
    CraftaxThreefryKey rng
);
static inline void craftax_boss_logic_native(CraftaxState* state);
static inline void craftax_level_up_attributes_native(
    CraftaxState* state,
    int32_t action,
    int32_t max_attribute
);
static inline void craftax_move_player_native(
    CraftaxState* state,
    int32_t action,
    bool god_mode
);
static inline void craftax_update_mobs_native(
    CraftaxState* state,
    CraftaxThreefryKey rng
);
static inline void craftax_spawn_mobs_native(
    CraftaxState* state,
    CraftaxThreefryKey rng
);
static inline void craftax_update_plants_native(CraftaxState* state);
static inline void craftax_update_player_intrinsics_native(
    CraftaxState* state,
    int32_t action
);
static inline void craftax_clip_inventory_and_intrinsics_native(
    CraftaxState* state,
    bool god_mode
);
static inline void craftax_calculate_inventory_achievements_native(
    CraftaxState* state
);
#endif

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float achievements[CRAFTAX_NUM_ACHIEVEMENTS];
    float n;
} Log;

typedef struct Client {
    int unused;
} Client;

typedef struct Craftax {
    Client* client;
    Log log;

    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;

    unsigned int rng;
    uint64_t seed;
    CraftaxThreefryKey rng_key;
    CraftaxArena* arena;
    CraftaxState* state;
    int32_t packet_id;
    int32_t lane_id;
    bool owns_state_storage;

    float achievements[CRAFTAX_NUM_ACHIEVEMENTS];
    float episode_return_accum;
    int32_t episode_length_accum;
} Craftax;

#ifdef CRAFTAX_ENABLE_ENV_IMPL

// ============================================================
// Native reset, observation, reward, and step glue
// ============================================================
static const float CRAFTAX_ACHIEVEMENT_REWARD_MAP[CRAFTAX_NUM_ACHIEVEMENTS] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 5.0f, 5.0f,
    5.0f, 8.0f, 8.0f, 8.0f, 3.0f, 3.0f, 3.0f, 3.0f,
    5.0f, 5.0f, 5.0f, 5.0f, 8.0f, 8.0f, 8.0f, 8.0f,
    8.0f, 8.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 5.0f,
    5.0f, 5.0f, 5.0f, 3.0f, 3.0f, 3.0f, 3.0f, 5.0f,
    5.0f, 5.0f, 5.0f,
};

static inline CraftaxThreefryKey craftax_step_native_next_key(
    CraftaxThreefryKey* rng
) {
    CraftaxThreefryKey subkey;
    craftax_threefry_split(*rng, rng, &subkey);
    return subkey;
}

static inline void craftax_copy_world_state_to_state(
    CraftaxState* dst,
    const CraftaxWorldState* src
) {
    memcpy(dst, src, sizeof(*dst));
}

static inline void craftax_generate_state_from_world_key(
    CraftaxThreefryKey world_key,
    CraftaxState* out
) {
    CraftaxWorldState world_state;
    craftax_generate_world_from_key(world_key, &world_state);
    craftax_copy_world_state_to_state(out, &world_state);
    craftax_refresh_spawn_bits_all(out);
}

static inline void craftax_reset_state_from_reset_key(
    CraftaxState* out,
    CraftaxThreefryKey reset_key
) {
    CraftaxThreefryKey unused;
    CraftaxThreefryKey world_key;
    craftax_threefry_split(reset_key, &unused, &world_key);
    craftax_generate_state_from_world_key(world_key, out);
}

// ============================================================
// Reset pool: pre-generate N worlds once, then memcpy on reset.
// Trades world diversity (<= pool_size unique maps per process) for
// ~500x faster reset. Set pool_size=0 to disable (exact per-seed
// world; required for the parity harness).
// ============================================================
static int g_craftax_reset_pool_size = 0;
static CraftaxState* g_craftax_reset_pool = NULL;
static int g_craftax_reset_pool_ready = 0;

// Called from my_init which runs single-threaded during env creation
// (vecenv.h iterates envs sequentially). First caller populates the
// pool; subsequent callers are no-ops.
static inline void craftax_set_reset_pool_size(int n) {
    if (g_craftax_reset_pool_ready) return;
    g_craftax_reset_pool_size = n;
    if (n > 0) {
        g_craftax_reset_pool = (CraftaxState*)calloc((size_t)n, sizeof(CraftaxState));
        for (int i = 0; i < n; i++) {
            CraftaxThreefryKey init_key = craftax_prng_key((uint32_t)i);
            CraftaxThreefryKey discard, reset_key;
            craftax_threefry_split(init_key, &discard, &reset_key);
            craftax_reset_state_from_reset_key(&g_craftax_reset_pool[i], reset_key);
        }
    }
    g_craftax_reset_pool_ready = 1;
}

static inline void craftax_ensure_state_storage(Craftax* env) {
    if (env->state != NULL) {
        return;
    }

    CraftaxArena* arena = (CraftaxArena*)calloc(1, sizeof(CraftaxArena));
    arena->states = (CraftaxState*)calloc(1, sizeof(CraftaxState));
    arena->num_envs = 1;
    arena->packet_size = 1;
    arena->num_packets = 1;

    env->arena = arena;
    env->state = arena->states;
    env->packet_id = 0;
    env->lane_id = 0;
    env->owns_state_storage = true;
}

static inline void craftax_reset_state_from_seed(Craftax* env) {
    craftax_ensure_state_storage(env);
    CraftaxThreefryKey initial_key = craftax_prng_key((uint32_t)env->seed);
    if (g_craftax_reset_pool_size > 0) {
        CraftaxThreefryKey discard;
        craftax_threefry_split(initial_key, &env->rng_key, &discard);
        int idx = (int)(env->seed % (uint64_t)g_craftax_reset_pool_size);
        memcpy(env->state, &g_craftax_reset_pool[idx], sizeof(CraftaxState));
        return;
    }
    CraftaxThreefryKey reset_key;
    craftax_threefry_split(initial_key, &env->rng_key, &reset_key);
    craftax_reset_state_from_reset_key(env->state, reset_key);
}

// Hot-path reset used by c_step on episode-done. Consults the reset pool
// when enabled, falls through to generate_world otherwise. Pool index is
// derived from the reset_key so different done events pick different
// pooled worlds. The direct craftax_reset_state_from_reset_key stays
// pool-free so the parity harness and any other direct caller get exact
// per-key determinism.
static inline void craftax_reset_state_on_done(
    CraftaxState* out,
    CraftaxThreefryKey reset_key
) {
    if (g_craftax_reset_pool_size > 0) {
        uint32_t idx = reset_key.word[0] % (uint32_t)g_craftax_reset_pool_size;
        memcpy(out, &g_craftax_reset_pool[idx], sizeof(CraftaxState));
        return;
    }
    craftax_reset_state_from_reset_key(out, reset_key);
}

static inline void craftax_encode_native_observation(
    const CraftaxState* state,
    float* obs
) {
    if (obs == NULL) {
        return;
    }
    craftax_encode_reset_observation((const CraftaxWorldState*)(const void*)state, obs);
}

static inline float craftax_calculate_light_level_native(int32_t timestep) {
    float progress = fmodf(
        (float)timestep / (float)CRAFTAX_DAY_LENGTH,
        1.0f
    ) + 0.3f;
    float c = cosf(CRAFTAX_WG_PI * progress);
    return 1.0f - powf(fabsf(c), 3.0f);
}

static inline bool craftax_is_game_over_native(const CraftaxState* state) {
    return state->timestep >= CRAFTAX_DEFAULT_MAX_TIMESTEPS
        || state->player_health <= 0.0f;
}

static inline void craftax_copy_achievements_to_env(
    Craftax* env,
    const CraftaxState* state
) {
    for (int i = 0; i < CRAFTAX_NUM_ACHIEVEMENTS; i++) {
        env->achievements[i] = state->achievements[i] ? 1.0f : 0.0f;
    }
}

static void add_log(Craftax* env) {
    int unlocked = 0;
    for (int i = 0; i < CRAFTAX_NUM_ACHIEVEMENTS; i++) {
        if (env->achievements[i] > 0.5f) {
            unlocked++;
            env->log.achievements[i] += 1.0f;
        }
    }
    env->log.perf += (float)unlocked / (float)CRAFTAX_NUM_ACHIEVEMENTS;
    env->log.score += env->episode_return_accum;
    env->log.episode_return += env->episode_return_accum;
    env->log.episode_length += (float)env->episode_length_accum;
    env->log.n += 1.0f;
}

static float craftax_gameplay_step_native(
    CraftaxState* state,
    int32_t action,
    CraftaxThreefryKey rng
) {
    CRAFTAX_PROFILE_START();
    bool init_achievements[CRAFTAX_NUM_ACHIEVEMENTS];
    memcpy(init_achievements, state->achievements, sizeof(init_achievements));
    float init_health = state->player_health;

    action = state->is_sleeping ? CRAFTAX_ACTION_NOOP : action;
    action = state->is_resting ? CRAFTAX_ACTION_NOOP : action;

    CRAFTAX_PROFILE_ZONE(0);
    craftax_change_floor_native(state, action);
    craftax_do_crafting_native(state, action);
    CRAFTAX_PROFILE_END(0);

    CraftaxThreefryKey subkey = craftax_step_native_next_key(&rng);
    CRAFTAX_PROFILE_ZONE(2);
    craftax_do_action_native(state, action, subkey);
    CRAFTAX_PROFILE_END(2);

    CRAFTAX_PROFILE_ZONE(3);
    craftax_place_block_native(state, action);
    craftax_shoot_projectile_native(state, action);
    craftax_cast_spell_native(state, action);
    craftax_drink_potion_native(state, action);
    CRAFTAX_PROFILE_END(3);

    subkey = craftax_step_native_next_key(&rng);
    CRAFTAX_PROFILE_ZONE(4);
    craftax_read_book_native(state, subkey.word, action);
    CRAFTAX_PROFILE_END(4);

    subkey = craftax_step_native_next_key(&rng);
    CRAFTAX_PROFILE_ZONE(5);
    craftax_enchant_native(state, action, subkey);
    CRAFTAX_PROFILE_END(5);

    CRAFTAX_PROFILE_ZONE(6);
    craftax_boss_logic_native(state);
    craftax_level_up_attributes_native(state, action, CRAFTAX_MAX_ATTRIBUTE);
    craftax_move_player_native(state, action, false);
    CRAFTAX_PROFILE_END(6);

    subkey = craftax_step_native_next_key(&rng);
    CRAFTAX_PROFILE_ZONE(7);
    craftax_update_mobs_native(state, subkey);
    CRAFTAX_PROFILE_END(7);

    subkey = craftax_step_native_next_key(&rng);
    CRAFTAX_PROFILE_ZONE(8);
    craftax_spawn_mobs_native(state, subkey);
    CRAFTAX_PROFILE_END(8);

    CRAFTAX_PROFILE_ZONE(9);
    craftax_update_plants_native(state);
    craftax_update_player_intrinsics_native(state, action);
    craftax_clip_inventory_and_intrinsics_native(state, false);
    craftax_calculate_inventory_achievements_native(state);
    CRAFTAX_PROFILE_END(9);

    CRAFTAX_PROFILE_ZONE(10);
    float reward = 0.0f;
    for (int i = 0; i < CRAFTAX_NUM_ACHIEVEMENTS; i++) {
        int32_t delta = (int32_t)state->achievements[i]
            - (int32_t)init_achievements[i];
        reward += (float)delta * CRAFTAX_ACHIEVEMENT_REWARD_MAP[i];
    }
    reward += (state->player_health - init_health) * 0.1f;

    subkey = craftax_step_native_next_key(&rng);
    state->timestep += 1;
    state->light_level = craftax_calculate_light_level_native(state->timestep);
    state->state_rng[0] = subkey.word[0];
    state->state_rng[1] = subkey.word[1];
    CRAFTAX_PROFILE_END(10);

    return reward;
}

// ============================================================
// Public API expected by vecenv.h
// ============================================================
static void c_init(Craftax* env) {
    env->client = NULL;
    env->num_agents = 1;
    craftax_ensure_state_storage(env);
    env->episode_return_accum = 0.0f;
    env->episode_length_accum = 0;
    memset(env->achievements, 0, sizeof(env->achievements));
    memset(&env->log, 0, sizeof(env->log));
    craftax_wg_init_cell_templates();
    craftax_reset_state_from_seed(env);
}

static void c_reset(Craftax* env) {
    if (env->rewards != NULL) {
        env->rewards[0] = 0.0f;
    }
    if (env->terminals != NULL) {
        env->terminals[0] = 0.0f;
    }
    env->episode_return_accum = 0.0f;
    env->episode_length_accum = 0;
    memset(env->achievements, 0, sizeof(env->achievements));

    craftax_reset_state_from_seed(env);
    craftax_encode_native_observation(env->state, env->observations);
}

#ifdef CRAFTAX_PROFILE
static void c_step_native(Craftax* env) {
    CRAFTAX_PROFILE_START();
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    int action = (int)env->actions[0];
    if (action < 0) {
        action = CRAFTAX_ACTION_NOOP;
    }
    if (action >= CRAFTAX_NUM_ACTIONS) {
        action = CRAFTAX_NUM_ACTIONS - 1;
    }

    CRAFTAX_PROFILE_ZONE(12);
    CraftaxThreefryKey step_key;
    craftax_threefry_split(env->rng_key, &env->rng_key, &step_key);

    CraftaxThreefryKey step_rng;
    CraftaxThreefryKey reset_key;
    craftax_threefry_split(step_key, &step_rng, &reset_key);
    CRAFTAX_PROFILE_END(12);

    float reward = craftax_gameplay_step_native(env->state, action, step_rng);

    CRAFTAX_PROFILE_ZONE(13);
    bool done = craftax_is_game_over_native(env->state);
    CRAFTAX_PROFILE_END(13);

    CRAFTAX_PROFILE_ZONE(15);
    craftax_copy_achievements_to_env(env, env->state);
    CRAFTAX_PROFILE_END(15);

    CRAFTAX_PROFILE_ZONE(16);
    env->rewards[0] = reward;
    env->terminals[0] = done ? 1.0f : 0.0f;
    env->episode_return_accum += reward;
    env->episode_length_accum += 1;
    CRAFTAX_PROFILE_END(16);

    if (done) {
        add_log(env);
        env->episode_return_accum = 0.0f;
        env->episode_length_accum = 0;
        memset(env->achievements, 0, sizeof(env->achievements));
        CRAFTAX_PROFILE_ZONE(14);
        craftax_reset_state_on_done(env->state, reset_key);
        CRAFTAX_PROFILE_END(14);
    }

    CRAFTAX_PROFILE_ZONE(11);
    craftax_encode_native_observation(env->state, env->observations);
    CRAFTAX_PROFILE_END(11);

    // Record unprofiled time
    CRAFTAX_PROFILE_ZONE(17);
    CRAFTAX_PROFILE_END(17);

#ifdef CRAFTAX_PROFILE
    static int profile_step_count = 0;
    profile_step_count++;
    if (profile_step_count >= 100000) {
        craftax_profile_report();
        profile_step_count = 0;
    }

#endif
}

#endif

static void c_step_gameplay(Craftax* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    int action = (int)env->actions[0];
    if (action < 0) action = CRAFTAX_ACTION_NOOP;
    if (action >= CRAFTAX_NUM_ACTIONS) action = CRAFTAX_NUM_ACTIONS - 1;

    CraftaxThreefryKey step_key;
    craftax_threefry_split(env->rng_key, &env->rng_key, &step_key);
    CraftaxThreefryKey step_rng;
    CraftaxThreefryKey reset_key;
    craftax_threefry_split(step_key, &step_rng, &reset_key);

    float reward = craftax_gameplay_step_native(env->state, action, step_rng);
    bool done = craftax_is_game_over_native(env->state);
    craftax_copy_achievements_to_env(env, env->state);

    env->rewards[0] = reward;
    env->terminals[0] = done ? 1.0f : 0.0f;
    env->episode_return_accum += reward;
    env->episode_length_accum += 1;

    if (done) {
        add_log(env);
        env->episode_return_accum = 0.0f;
        env->episode_length_accum = 0;
        memset(env->achievements, 0, sizeof(env->achievements));
        craftax_reset_state_on_done(env->state, reset_key);
    }
}

static void c_step_encode(Craftax* env) {
    craftax_encode_native_observation(env->state, env->observations);
}

static void c_step(Craftax* env) {
    c_step_gameplay(env);
    c_step_encode(env);
}

static void c_close(Craftax* env) {
    if (!env->owns_state_storage || env->arena == NULL) {
        return;
    }
    free(env->arena->states);
    free(env->arena);
    env->arena = NULL;
    env->state = NULL;
    env->owns_state_storage = false;
}

// ------------------------------------------------------------
// Tile-based renderer using upstream Craftax 16x16 PNG assets
// ------------------------------------------------------------
// Packed layout (see ocean/craftax/pack_textures.py):
//   [0..36] block textures (indexed by CraftaxBlockType)
//   [37..41] player: down, up, left, right, sleep
//   [42..46] items: none, torch, ladder_down, ladder_up, ladder_down_blocked

#define CRAFTAX_TEX_TILE_PX 16
#define CRAFTAX_TEX_SCALE 4   // on-screen px = 64
#define CRAFTAX_TEX_DRAW_PX (CRAFTAX_TEX_TILE_PX * CRAFTAX_TEX_SCALE)
#define CRAFTAX_TEX_NUM (37 + 5 + 5 + 3 + 4)

// Render viewport (independent of agent obs window)
#define CRAFTAX_RENDER_ROWS 16
#define CRAFTAX_RENDER_COLS 16

#define CRAFTAX_TEX_PLAYER_DOWN 37
#define CRAFTAX_TEX_PLAYER_UP 38
#define CRAFTAX_TEX_PLAYER_LEFT 39
#define CRAFTAX_TEX_PLAYER_RIGHT 40
#define CRAFTAX_TEX_PLAYER_SLEEP 41
#define CRAFTAX_TEX_ITEM_BASE 42

static Texture2D craftax_textures[CRAFTAX_TEX_NUM];
static bool craftax_textures_loaded = false;

static void craftax_load_textures(void) {
    if (craftax_textures_loaded) return;
    const char* candidates[] = {
        "resources/craftax/textures.bin",
        "../resources/craftax/textures.bin",
        "../../resources/craftax/textures.bin",
    };
    FILE* f = NULL;
    for (size_t i = 0; i < sizeof(candidates)/sizeof(candidates[0]); i++) {
        f = fopen(candidates[i], "rb");
        if (f) break;
    }
    if (!f) {
        fprintf(stderr, "craftax: textures.bin not found in resources/craftax -- run ocean/craftax/pack_textures.py\n");
        exit(1);
    }
    const size_t tile_bytes = CRAFTAX_TEX_TILE_PX * CRAFTAX_TEX_TILE_PX * 4;
    uint8_t* buf = (uint8_t*)malloc(tile_bytes);
    for (int i = 0; i < CRAFTAX_TEX_NUM; i++) {
        if (fread(buf, 1, tile_bytes, f) != tile_bytes) {
            fprintf(stderr, "craftax: short read on textures.bin at tile %d\n", i);
            exit(1);
        }
        Image img = {
            .data = buf,
            .width = CRAFTAX_TEX_TILE_PX,
            .height = CRAFTAX_TEX_TILE_PX,
            .mipmaps = 1,
            .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,
        };
        craftax_textures[i] = LoadTextureFromImage(img);
        SetTextureFilter(craftax_textures[i], TEXTURE_FILTER_POINT);
    }
    free(buf);
    fclose(f);
    craftax_textures_loaded = true;
}

static int craftax_player_tex_id(int32_t direction, bool sleeping) {
    if (sleeping) return CRAFTAX_TEX_PLAYER_SLEEP;
    switch (direction) {
        case 1: return CRAFTAX_TEX_PLAYER_LEFT;
        case 2: return CRAFTAX_TEX_PLAYER_RIGHT;
        case 3: return CRAFTAX_TEX_PLAYER_UP;
        case 4: return CRAFTAX_TEX_PLAYER_DOWN;
        default: return CRAFTAX_TEX_PLAYER_DOWN;
    }
}

static void craftax_draw_tile(int tex_id, int dst_x, int dst_y, float tint_alpha) {
    if (tex_id < 0 || tex_id >= CRAFTAX_TEX_NUM) return;
    Rectangle src = {0, 0, CRAFTAX_TEX_TILE_PX, CRAFTAX_TEX_TILE_PX};
    Rectangle dst = {(float)dst_x, (float)dst_y, CRAFTAX_TEX_DRAW_PX, CRAFTAX_TEX_DRAW_PX};
    Color tint = {255, 255, 255, (unsigned char)(tint_alpha * 255.0f)};
    DrawTexturePro(craftax_textures[tex_id], src, dst, (Vector2){0, 0}, 0.0f, tint);
}

static void c_render(Craftax* env) {
    const int view_w = CRAFTAX_RENDER_COLS * CRAFTAX_TEX_DRAW_PX;
    const int view_h = CRAFTAX_RENDER_ROWS * CRAFTAX_TEX_DRAW_PX;
    const int hud_h = 80;

    if (!IsWindowReady()) {
        InitWindow(view_w, view_h + hud_h, "PufferLib Craftax");
        SetTargetFPS(30);
    }
    if (!craftax_textures_loaded) craftax_load_textures();
    if (IsKeyDown(KEY_ESCAPE)) exit(0);

    CraftaxState* s = env->state;
    int lvl = s->player_level;
    int pr = s->player_position[0];
    int pc = s->player_position[1];
    int half_r = CRAFTAX_RENDER_ROWS / 2;
    int half_c = CRAFTAX_RENDER_COLS / 2;

    BeginDrawing();
    ClearBackground(BLACK);

    for (int vr = 0; vr < CRAFTAX_RENDER_ROWS; vr++) {
        for (int vc = 0; vc < CRAFTAX_RENDER_COLS; vc++) {
            int wr = pr - half_r + vr;
            int wc = pc - half_c + vc;
            int dst_x = vc * CRAFTAX_TEX_DRAW_PX;
            int dst_y = vr * CRAFTAX_TEX_DRAW_PX;

            int blk = CRAFTAX_BLOCK_OUT_OF_BOUNDS;
            if (wr >= 0 && wr < CRAFTAX_MAP_SIZE && wc >= 0 && wc < CRAFTAX_MAP_SIZE) {
                blk = s->map[lvl][wr][wc];
                if (s->light_map[lvl][wr][wc] <= 12) blk = CRAFTAX_BLOCK_DARKNESS;
            }
            if (blk < 0 || blk >= CRAFTAX_NUM_BLOCK_TYPES) blk = 0;
            craftax_draw_tile(blk, dst_x, dst_y, 1.0f);

            // item overlay
            if (wr >= 0 && wr < CRAFTAX_MAP_SIZE && wc >= 0 && wc < CRAFTAX_MAP_SIZE) {
                int it = s->item_map[lvl][wr][wc];
                if (it > 0 && it < 5) {
                    craftax_draw_tile(CRAFTAX_TEX_ITEM_BASE + it, dst_x, dst_y, 1.0f);
                }
            }
        }
    }

    // player in center
    int pid = craftax_player_tex_id(s->player_direction, s->is_sleeping);
    craftax_draw_tile(pid, half_c * CRAFTAX_TEX_DRAW_PX, half_r * CRAFTAX_TEX_DRAW_PX, 1.0f);

    // night dim overlay
    if (s->light_level < 1.0f) {
        unsigned char a = (unsigned char)((1.0f - s->light_level) * 140.0f);
        DrawRectangle(0, 0, view_w, view_h, (Color){0, 0, 40, a});
    }

    // HUD
    int hud_y = view_h;
    DrawRectangle(0, hud_y, view_w, hud_h, (Color){20, 20, 20, 255});
    DrawText(TextFormat("HP:%.0f  F:%d  D:%d  E:%d  M:%d  L:%d  t:%d",
             s->player_health, s->player_food, s->player_drink,
             s->player_energy, s->player_mana, s->player_level, s->timestep),
             4, hud_y + 4, 14, WHITE);
    DrawText(TextFormat("XP:%d  DEX:%d  STR:%d  INT:%d  light:%.2f",
             s->player_xp, s->player_dexterity, s->player_strength,
             s->player_intelligence, s->light_level),
             4, hud_y + 22, 14, (Color){200, 200, 200, 255});
    int ach_count = 0;
    for (int i = 0; i < CRAFTAX_NUM_ACHIEVEMENTS; i++) ach_count += s->achievements[i] ? 1 : 0;
    DrawText(TextFormat("achievements: %d / %d", ach_count, CRAFTAX_NUM_ACHIEVEMENTS),
             4, hud_y + 40, 14, (Color){180, 220, 180, 255});
    DrawText(TextFormat("ret:%.2f len:%d", env->episode_return_accum, env->episode_length_accum),
             4, hud_y + 58, 14, (Color){200, 200, 140, 255});

    EndDrawing();
}

#endif
