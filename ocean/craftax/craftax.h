// Full native Craftax environment for PufferLib Ocean.

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "worldgen.h"

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
#define CRAFTAX_OBS_SIZE 8268

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
    int32_t map[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE][CRAFTAX_MAP_SIZE];
    int32_t item_map[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE][CRAFTAX_MAP_SIZE];
    bool mob_map[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE][CRAFTAX_MAP_SIZE];
    float light_map[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE][CRAFTAX_MAP_SIZE];
    int32_t down_ladders[CRAFTAX_NUM_LEVELS][2];
    int32_t up_ladders[CRAFTAX_NUM_LEVELS][2];
    bool chests_opened[CRAFTAX_NUM_LEVELS];
    int32_t monsters_killed[CRAFTAX_NUM_LEVELS];

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
} CraftaxState;

typedef char CraftaxStateMatchesWorldState[
    (sizeof(CraftaxState) == sizeof(CraftaxWorldState)) ? 1 : -1
];

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
    CraftaxState state;

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

static inline void craftax_reset_state_from_seed(Craftax* env) {
    CraftaxThreefryKey initial_key = craftax_prng_key((uint32_t)env->seed);
    CraftaxThreefryKey reset_key;
    craftax_threefry_split(initial_key, &env->rng_key, &reset_key);
    craftax_reset_state_from_reset_key(&env->state, reset_key);
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
    bool init_achievements[CRAFTAX_NUM_ACHIEVEMENTS];
    memcpy(init_achievements, state->achievements, sizeof(init_achievements));
    float init_health = state->player_health;

    action = state->is_sleeping ? CRAFTAX_ACTION_NOOP : action;
    action = state->is_resting ? CRAFTAX_ACTION_NOOP : action;

    craftax_change_floor_native(state, action);
    craftax_do_crafting_native(state, action);

    CraftaxThreefryKey subkey = craftax_step_native_next_key(&rng);
    craftax_do_action_native(state, action, subkey);

    craftax_place_block_native(state, action);
    craftax_shoot_projectile_native(state, action);
    craftax_cast_spell_native(state, action);
    craftax_drink_potion_native(state, action);

    subkey = craftax_step_native_next_key(&rng);
    craftax_read_book_native(state, subkey.word, action);

    subkey = craftax_step_native_next_key(&rng);
    craftax_enchant_native(state, action, subkey);

    craftax_boss_logic_native(state);
    craftax_level_up_attributes_native(state, action, CRAFTAX_MAX_ATTRIBUTE);
    craftax_move_player_native(state, action, false);

    subkey = craftax_step_native_next_key(&rng);
    craftax_update_mobs_native(state, subkey);

    subkey = craftax_step_native_next_key(&rng);
    craftax_spawn_mobs_native(state, subkey);

    craftax_update_plants_native(state);
    craftax_update_player_intrinsics_native(state, action);
    craftax_clip_inventory_and_intrinsics_native(state, false);
    craftax_calculate_inventory_achievements_native(state);

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

    return reward;
}

// ============================================================
// Public API expected by vecenv.h
// ============================================================
static void c_init(Craftax* env) {
    env->client = NULL;
    env->num_agents = 1;
    env->episode_return_accum = 0.0f;
    env->episode_length_accum = 0;
    memset(env->achievements, 0, sizeof(env->achievements));
    memset(&env->log, 0, sizeof(env->log));
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
    craftax_encode_native_observation(&env->state, env->observations);
}

static void c_step_native(Craftax* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    int action = (int)env->actions[0];
    if (action < 0) {
        action = CRAFTAX_ACTION_NOOP;
    }
    if (action >= CRAFTAX_NUM_ACTIONS) {
        action = CRAFTAX_NUM_ACTIONS - 1;
    }

    CraftaxThreefryKey step_key;
    craftax_threefry_split(env->rng_key, &env->rng_key, &step_key);

    CraftaxThreefryKey step_rng;
    CraftaxThreefryKey reset_key;
    craftax_threefry_split(step_key, &step_rng, &reset_key);

    float reward = craftax_gameplay_step_native(&env->state, action, step_rng);
    bool done = craftax_is_game_over_native(&env->state);

    craftax_copy_achievements_to_env(env, &env->state);

    env->rewards[0] = reward;
    env->terminals[0] = done ? 1.0f : 0.0f;
    env->episode_return_accum += reward;
    env->episode_length_accum += 1;

    if (done) {
        add_log(env);
        env->episode_return_accum = 0.0f;
        env->episode_length_accum = 0;
        memset(env->achievements, 0, sizeof(env->achievements));
        craftax_reset_state_from_reset_key(&env->state, reset_key);
    }

    craftax_encode_native_observation(&env->state, env->observations);
}

static void c_step(Craftax* env) {
    c_step_native(env);
}

static void c_close(Craftax* env) {
    (void)env;
}

static void c_render(Craftax* env) {
    (void)env;
}

#endif
