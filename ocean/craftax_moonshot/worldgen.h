// Native Craftax reset world generation.
//
// This mirrors craftax/craftax/world_gen/world_gen.py for the default
// EnvParams and StaticEnvParams used by Craftax-Symbolic-v1 reset.

#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "noise.h"

#define CRAFTAX_WG_MAP_SIZE 48
#define CRAFTAX_WG_MAP_CELLS (CRAFTAX_WG_MAP_SIZE * CRAFTAX_WG_MAP_SIZE)
#define CRAFTAX_WG_NUM_LEVELS 9
#define CRAFTAX_WG_OBS_ROWS 9
#define CRAFTAX_WG_OBS_COLS 11
#define CRAFTAX_WG_NUM_BLOCK_TYPES 37
#define CRAFTAX_WG_NUM_ITEM_TYPES 5
#define CRAFTAX_WG_NUM_MOB_CLASSES 5
#define CRAFTAX_WG_NUM_MOB_TYPES 8
#define CRAFTAX_WG_INVENTORY_OBS_SIZE 51

// Compact binary observation encoding.
// Each cell uses binary channels instead of one-hot:
//   6 bits: block type (0-63, covers 37 block types)
//   3 bits: item type+1 (0=no item, 1-5=item types)
//   4 bits per mob class: mob type+1 (0=no mob, 1-8=types) x 5 classes
//   1 bit : visibility
// Total: 30 binary channels per cell.
#define CRAFTAX_WG_BINARY_BLOCK_BITS 6
#define CRAFTAX_WG_BINARY_ITEM_BITS 3
#define CRAFTAX_WG_BINARY_MOB_BITS 4
#define CRAFTAX_WG_BINARY_VISIBILITY_BITS 1

#define CRAFTAX_WG_BINARY_CHANNELS_PER_CELL ( \
    CRAFTAX_WG_BINARY_BLOCK_BITS + \
    CRAFTAX_WG_BINARY_ITEM_BITS + \
    CRAFTAX_WG_NUM_MOB_CLASSES * CRAFTAX_WG_BINARY_MOB_BITS + \
    CRAFTAX_WG_BINARY_VISIBILITY_BITS \
)

#define CRAFTAX_WG_BINARY_MAP_OBS_SIZE ( \
    CRAFTAX_WG_OBS_ROWS * CRAFTAX_WG_OBS_COLS * CRAFTAX_WG_BINARY_CHANNELS_PER_CELL \
)
#define CRAFTAX_WG_OBS_WINDOW_CELLS (CRAFTAX_WG_OBS_ROWS * CRAFTAX_WG_OBS_COLS)
#define CRAFTAX_WG_CELL_TEMPLATE_BYTES ( \
    CRAFTAX_WG_BINARY_CHANNELS_PER_CELL * sizeof(float) \
)
#define CRAFTAX_WG_FULL_OBS_SIZE ( \
    CRAFTAX_WG_BINARY_MAP_OBS_SIZE + CRAFTAX_WG_INVENTORY_OBS_SIZE \
)

// Moonshot symbolic observation. Each visible cell stores compact float IDs:
// block, item+1, visible, and one mob type+1 slot for each mob class.
// The 51 scalar channels remain exact floats for oracle-expandability.
#define CRAFTAX_WG_PACKED_CHANNELS_PER_CELL (3 + CRAFTAX_WG_NUM_MOB_CLASSES)
#define CRAFTAX_WG_PACKED_MAP_OBS_SIZE ( \
    CRAFTAX_WG_OBS_ROWS * CRAFTAX_WG_OBS_COLS * CRAFTAX_WG_PACKED_CHANNELS_PER_CELL \
)
#define CRAFTAX_WG_PACKED_OBS_SIZE ( \
    CRAFTAX_WG_PACKED_MAP_OBS_SIZE + CRAFTAX_WG_INVENTORY_OBS_SIZE \
)

// Lookup tables for fast binary bit writing (eliminates loops/branches)
static const float CRAFTAX_WG_BLOCK_LUT[64][6] = {
    {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},{1.0f,0.0f,0.0f,0.0f,0.0f,0.0f},{0.0f,1.0f,0.0f,0.0f,0.0f,0.0f},{1.0f,1.0f,0.0f,0.0f,0.0f,0.0f},
    {0.0f,0.0f,1.0f,0.0f,0.0f,0.0f},{1.0f,0.0f,1.0f,0.0f,0.0f,0.0f},{0.0f,1.0f,1.0f,0.0f,0.0f,0.0f},{1.0f,1.0f,1.0f,0.0f,0.0f,0.0f},
    {0.0f,0.0f,0.0f,1.0f,0.0f,0.0f},{1.0f,0.0f,0.0f,1.0f,0.0f,0.0f},{0.0f,1.0f,0.0f,1.0f,0.0f,0.0f},{1.0f,1.0f,0.0f,1.0f,0.0f,0.0f},
    {0.0f,0.0f,1.0f,1.0f,0.0f,0.0f},{1.0f,0.0f,1.0f,1.0f,0.0f,0.0f},{0.0f,1.0f,1.0f,1.0f,0.0f,0.0f},{1.0f,1.0f,1.0f,1.0f,0.0f,0.0f},
    {0.0f,0.0f,0.0f,0.0f,1.0f,0.0f},{1.0f,0.0f,0.0f,0.0f,1.0f,0.0f},{0.0f,1.0f,0.0f,0.0f,1.0f,0.0f},{1.0f,1.0f,0.0f,0.0f,1.0f,0.0f},
    {0.0f,0.0f,1.0f,0.0f,1.0f,0.0f},{1.0f,0.0f,1.0f,0.0f,1.0f,0.0f},{0.0f,1.0f,1.0f,0.0f,1.0f,0.0f},{1.0f,1.0f,1.0f,0.0f,1.0f,0.0f},
    {0.0f,0.0f,0.0f,1.0f,1.0f,0.0f},{1.0f,0.0f,0.0f,1.0f,1.0f,0.0f},{0.0f,1.0f,0.0f,1.0f,1.0f,0.0f},{1.0f,1.0f,0.0f,1.0f,1.0f,0.0f},
    {0.0f,0.0f,1.0f,1.0f,1.0f,0.0f},{1.0f,0.0f,1.0f,1.0f,1.0f,0.0f},{0.0f,1.0f,1.0f,1.0f,1.0f,0.0f},{1.0f,1.0f,1.0f,1.0f,1.0f,0.0f},
    {0.0f,0.0f,0.0f,0.0f,0.0f,1.0f},{1.0f,0.0f,0.0f,0.0f,0.0f,1.0f},{0.0f,1.0f,0.0f,0.0f,0.0f,1.0f},{1.0f,1.0f,0.0f,0.0f,0.0f,1.0f},
    {0.0f,0.0f,1.0f,0.0f,0.0f,1.0f},{1.0f,0.0f,1.0f,0.0f,0.0f,1.0f},{0.0f,1.0f,1.0f,0.0f,0.0f,1.0f},{1.0f,1.0f,1.0f,0.0f,0.0f,1.0f},
    {0.0f,0.0f,0.0f,1.0f,0.0f,1.0f},{1.0f,0.0f,0.0f,1.0f,0.0f,1.0f},{0.0f,1.0f,0.0f,1.0f,0.0f,1.0f},{1.0f,1.0f,0.0f,1.0f,0.0f,1.0f},
    {0.0f,0.0f,1.0f,1.0f,0.0f,1.0f},{1.0f,0.0f,1.0f,1.0f,0.0f,1.0f},{0.0f,1.0f,1.0f,1.0f,0.0f,1.0f},{1.0f,1.0f,1.0f,1.0f,0.0f,1.0f},
    {0.0f,0.0f,0.0f,0.0f,1.0f,1.0f},{1.0f,0.0f,0.0f,0.0f,1.0f,1.0f},{0.0f,1.0f,0.0f,0.0f,1.0f,1.0f},{1.0f,1.0f,0.0f,0.0f,1.0f,1.0f},
    {0.0f,0.0f,1.0f,0.0f,1.0f,1.0f},{1.0f,0.0f,1.0f,0.0f,1.0f,1.0f},{0.0f,1.0f,1.0f,0.0f,1.0f,1.0f},{1.0f,1.0f,1.0f,0.0f,1.0f,1.0f},
    {0.0f,0.0f,0.0f,1.0f,1.0f,1.0f},{1.0f,0.0f,0.0f,1.0f,1.0f,1.0f},{0.0f,1.0f,0.0f,1.0f,1.0f,1.0f},{1.0f,1.0f,0.0f,1.0f,1.0f,1.0f},
    {0.0f,0.0f,1.0f,1.0f,1.0f,1.0f},{1.0f,0.0f,1.0f,1.0f,1.0f,1.0f},{0.0f,1.0f,1.0f,1.0f,1.0f,1.0f},{1.0f,1.0f,1.0f,1.0f,1.0f,1.0f},
};
static const float CRAFTAX_WG_ITEM_LUT[8][3] = {
    {0.0f,0.0f,0.0f},{1.0f,0.0f,0.0f},{0.0f,1.0f,0.0f},{1.0f,1.0f,0.0f},
    {0.0f,0.0f,1.0f},{1.0f,0.0f,1.0f},{0.0f,1.0f,1.0f},{1.0f,1.0f,1.0f},
};
static const float CRAFTAX_WG_MOB_LUT[16][4] = {
    {0.0f,0.0f,0.0f,0.0f},{1.0f,0.0f,0.0f,0.0f},{0.0f,1.0f,0.0f,0.0f},{1.0f,1.0f,0.0f,0.0f},
    {0.0f,0.0f,1.0f,0.0f},{1.0f,0.0f,1.0f,0.0f},{0.0f,1.0f,1.0f,0.0f},{1.0f,1.0f,1.0f,0.0f},
    {0.0f,0.0f,0.0f,1.0f},{1.0f,0.0f,0.0f,1.0f},{0.0f,1.0f,0.0f,1.0f},{1.0f,1.0f,0.0f,1.0f},
    {0.0f,0.0f,1.0f,1.0f},{1.0f,0.0f,1.0f,1.0f},{0.0f,1.0f,1.0f,1.0f},{1.0f,1.0f,1.0f,1.0f},
};
static float CRAFTAX_WG_VISIBLE_CELL_TEMPLATE_LUT[64][8][CRAFTAX_WG_BINARY_CHANNELS_PER_CELL];
static float CRAFTAX_WG_EMPTY_CELL_TEMPLATE[CRAFTAX_WG_BINARY_CHANNELS_PER_CELL];
static bool CRAFTAX_WG_CELL_TEMPLATE_READY = false;

static inline void craftax_wg_init_cell_templates(void) {
    if (CRAFTAX_WG_CELL_TEMPLATE_READY) {
        return;
    }

    for (int block = 0; block < 64; block++) {
        for (int item = 0; item < 8; item++) {
            float* cell = CRAFTAX_WG_VISIBLE_CELL_TEMPLATE_LUT[block][item];
            memcpy(cell, CRAFTAX_WG_BLOCK_LUT[block], 6 * sizeof(float));
            memcpy(cell + CRAFTAX_WG_BINARY_BLOCK_BITS, CRAFTAX_WG_ITEM_LUT[item], 3 * sizeof(float));
            cell[CRAFTAX_WG_BINARY_CHANNELS_PER_CELL - 1] = 1.0f;
        }
    }

    CRAFTAX_WG_CELL_TEMPLATE_READY = true;
}

#define CRAFTAX_WG_OBS_SIZE CRAFTAX_WG_PACKED_OBS_SIZE
#define CRAFTAX_WG_NUM_ACHIEVEMENTS 67
#define CRAFTAX_WG_MAX_MELEE_MOBS 3
#define CRAFTAX_WG_MAX_PASSIVE_MOBS 3
#define CRAFTAX_WG_MAX_RANGED_MOBS 2
#define CRAFTAX_WG_MAX_MOB_PROJECTILES 3
#define CRAFTAX_WG_MAX_PLAYER_PROJECTILES 3
#define CRAFTAX_WG_MAX_GROWING_PLANTS 10
#define CRAFTAX_WG_MONSTERS_KILLED_TO_CLEAR_LEVEL 8

// Backwards-compatible names used by the phase-1 floor-0 test.
#define CRAFTAX_OVERWORLD_SIZE CRAFTAX_WG_MAP_SIZE
#define CRAFTAX_OVERWORLD_CELLS CRAFTAX_WG_MAP_CELLS

#define CRAFTAX_WG_BLOCK_INVALID 0
#define CRAFTAX_WG_BLOCK_OUT_OF_BOUNDS 1
#define CRAFTAX_WG_BLOCK_GRASS 2
#define CRAFTAX_WG_BLOCK_WATER 3
#define CRAFTAX_WG_BLOCK_STONE 4
#define CRAFTAX_WG_BLOCK_TREE 5
#define CRAFTAX_WG_BLOCK_WOOD 6
#define CRAFTAX_WG_BLOCK_PATH 7
#define CRAFTAX_WG_BLOCK_COAL 8
#define CRAFTAX_WG_BLOCK_IRON 9
#define CRAFTAX_WG_BLOCK_DIAMOND 10
#define CRAFTAX_WG_BLOCK_CRAFTING_TABLE 11
#define CRAFTAX_WG_BLOCK_FURNACE 12
#define CRAFTAX_WG_BLOCK_SAND 13
#define CRAFTAX_WG_BLOCK_LAVA 14
#define CRAFTAX_WG_BLOCK_PLANT 15
#define CRAFTAX_WG_BLOCK_RIPE_PLANT 16
#define CRAFTAX_WG_BLOCK_WALL 17
#define CRAFTAX_WG_BLOCK_DARKNESS 18
#define CRAFTAX_WG_BLOCK_WALL_MOSS 19
#define CRAFTAX_WG_BLOCK_STALAGMITE 20
#define CRAFTAX_WG_BLOCK_SAPPHIRE 21
#define CRAFTAX_WG_BLOCK_RUBY 22
#define CRAFTAX_WG_BLOCK_CHEST 23
#define CRAFTAX_WG_BLOCK_FOUNTAIN 24
#define CRAFTAX_WG_BLOCK_FIRE_GRASS 25
#define CRAFTAX_WG_BLOCK_ICE_GRASS 26
#define CRAFTAX_WG_BLOCK_GRAVEL 27
#define CRAFTAX_WG_BLOCK_FIRE_TREE 28
#define CRAFTAX_WG_BLOCK_ICE_SHRUB 29
#define CRAFTAX_WG_BLOCK_ENCHANTMENT_TABLE_FIRE 30
#define CRAFTAX_WG_BLOCK_ENCHANTMENT_TABLE_ICE 31
#define CRAFTAX_WG_BLOCK_NECROMANCER 32
#define CRAFTAX_WG_BLOCK_GRAVE 33
#define CRAFTAX_WG_BLOCK_GRAVE2 34
#define CRAFTAX_WG_BLOCK_GRAVE3 35
#define CRAFTAX_WG_BLOCK_NECROMANCER_VULNERABLE 36

#define CRAFTAX_WG_ITEM_NONE 0
#define CRAFTAX_WG_ITEM_TORCH 1
#define CRAFTAX_WG_ITEM_LADDER_DOWN 2
#define CRAFTAX_WG_ITEM_LADDER_UP 3
#define CRAFTAX_WG_ITEM_LADDER_DOWN_BLOCKED 4

#define CRAFTAX_WG_ACTION_UP 3
#define CRAFTAX_WG_BOSS_FIGHT_SPAWN_TURNS 7
#define CRAFTAX_WG_PI 3.14159265358979323846f

typedef struct CraftaxOverworldFloor {
    uint8_t map[CRAFTAX_OVERWORLD_SIZE][CRAFTAX_OVERWORLD_SIZE];
    uint8_t item_map[CRAFTAX_OVERWORLD_SIZE][CRAFTAX_OVERWORLD_SIZE];
    uint8_t light_map[CRAFTAX_OVERWORLD_SIZE][CRAFTAX_OVERWORLD_SIZE];
    int32_t ladder_down[2];
    int32_t ladder_up[2];
} CraftaxOverworldFloor;

typedef struct CraftaxWGInventory {
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
} CraftaxWGInventory;

typedef struct CraftaxWGMobs3 {
    int32_t position[CRAFTAX_WG_NUM_LEVELS][3][2];
    float health[CRAFTAX_WG_NUM_LEVELS][3];
    bool mask[CRAFTAX_WG_NUM_LEVELS][3];
    int32_t attack_cooldown[CRAFTAX_WG_NUM_LEVELS][3];
    int32_t type_id[CRAFTAX_WG_NUM_LEVELS][3];
} CraftaxWGMobs3;

typedef struct CraftaxWGMobs2 {
    int32_t position[CRAFTAX_WG_NUM_LEVELS][2][2];
    float health[CRAFTAX_WG_NUM_LEVELS][2];
    bool mask[CRAFTAX_WG_NUM_LEVELS][2];
    int32_t attack_cooldown[CRAFTAX_WG_NUM_LEVELS][2];
    int32_t type_id[CRAFTAX_WG_NUM_LEVELS][2];
} CraftaxWGMobs2;

typedef struct CraftaxWorldState {
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

    CraftaxWGInventory inventory;

    CraftaxWGMobs3 melee_mobs;
    CraftaxWGMobs3 passive_mobs;
    CraftaxWGMobs2 ranged_mobs;

    CraftaxWGMobs3 mob_projectiles;
    int32_t mob_projectile_directions[CRAFTAX_WG_NUM_LEVELS][CRAFTAX_WG_MAX_MOB_PROJECTILES][2];
    CraftaxWGMobs3 player_projectiles;
    int32_t player_projectile_directions[CRAFTAX_WG_NUM_LEVELS][CRAFTAX_WG_MAX_PLAYER_PROJECTILES][2];

    int32_t growing_plants_positions[CRAFTAX_WG_MAX_GROWING_PLANTS][2];
    int32_t growing_plants_age[CRAFTAX_WG_MAX_GROWING_PLANTS];
    bool growing_plants_mask[CRAFTAX_WG_MAX_GROWING_PLANTS];

    int32_t potion_mapping[6];
    bool learned_spells[2];

    int32_t sword_enchantment;
    int32_t bow_enchantment;
    int32_t armour_enchantments[4];

    int32_t boss_progress;
    int32_t boss_timesteps_to_spawn_this_round;

    float light_level;
    bool achievements[CRAFTAX_WG_NUM_ACHIEVEMENTS];
    uint32_t state_rng[2];
    int32_t timestep;
    int32_t fractal_noise_angles[4];

    // === Medium-hot bitmaps ===
    uint64_t mob_bits[CRAFTAX_WG_NUM_LEVELS][CRAFTAX_WG_MAP_SIZE];
    uint64_t spawn_all_bits[CRAFTAX_WG_NUM_LEVELS][CRAFTAX_WG_MAP_SIZE];
    uint64_t spawn_grave_bits[CRAFTAX_WG_NUM_LEVELS][CRAFTAX_WG_MAP_SIZE];
    uint64_t spawn_water_bits[CRAFTAX_WG_NUM_LEVELS][CRAFTAX_WG_MAP_SIZE];

    // === Cold data (large maps) ===
    uint8_t map[CRAFTAX_WG_NUM_LEVELS][CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE];
    uint8_t item_map[CRAFTAX_WG_NUM_LEVELS][CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE];
    uint8_t light_map[CRAFTAX_WG_NUM_LEVELS][CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE];

    int32_t down_ladders[CRAFTAX_WG_NUM_LEVELS][2];
    int32_t up_ladders[CRAFTAX_WG_NUM_LEVELS][2];
    bool chests_opened[CRAFTAX_WG_NUM_LEVELS];
    int32_t monsters_killed[CRAFTAX_WG_NUM_LEVELS];
} CraftaxWorldState;

typedef struct CraftaxSmoothGenConfig {
    int32_t default_block;
    int32_t sea_block;
    int32_t coast_block;
    int32_t mountain_block;
    int32_t path_block;
    int32_t inner_mountain_block;
    int32_t ore_requirement_blocks[5];
    int32_t ores[5];
    float ore_chances[5];
    int32_t tree_requirement_block;
    int32_t tree;
    int32_t lava;
    int32_t player_spawn;
    int32_t valid_ladder;
    bool ladder_up;
    bool ladder_down;
    float player_proximity_map_water_strength;
    float player_proximity_map_water_max;
    float player_proximity_map_mountain_strength;
    float player_proximity_map_mountain_max;
    float default_light;
    float water_threshold;
    float sand_threshold;
    float tree_threshold_uniform;
    float tree_threshold_perlin;
} CraftaxSmoothGenConfig;

typedef struct CraftaxDungeonConfig {
    int32_t special_block;
    int32_t fountain_block;
    int32_t rare_path_replacement_block;
} CraftaxDungeonConfig;

static const CraftaxSmoothGenConfig CRAFTAX_SMOOTHGEN_CONFIGS[6] = {
    {
        CRAFTAX_WG_BLOCK_GRASS,
        CRAFTAX_WG_BLOCK_WATER,
        CRAFTAX_WG_BLOCK_SAND,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_PATH,
        {CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE},
        {CRAFTAX_WG_BLOCK_COAL, CRAFTAX_WG_BLOCK_IRON, CRAFTAX_WG_BLOCK_DIAMOND, CRAFTAX_WG_BLOCK_OUT_OF_BOUNDS, CRAFTAX_WG_BLOCK_OUT_OF_BOUNDS},
        {0.03f, 0.02f, 0.001f, 0.0f, 0.0f},
        CRAFTAX_WG_BLOCK_GRASS,
        CRAFTAX_WG_BLOCK_TREE,
        CRAFTAX_WG_BLOCK_LAVA,
        CRAFTAX_WG_BLOCK_GRASS,
        CRAFTAX_WG_BLOCK_PATH,
        false,
        true,
        5.0f,
        1.0f,
        5.0f,
        1.0f,
        1.0f,
        0.7f,
        0.6f,
        0.8f,
        0.5f,
    },
    {
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_WATER,
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        {CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE},
        {CRAFTAX_WG_BLOCK_COAL, CRAFTAX_WG_BLOCK_IRON, CRAFTAX_WG_BLOCK_DIAMOND, CRAFTAX_WG_BLOCK_SAPPHIRE, CRAFTAX_WG_BLOCK_RUBY},
        {0.04f, 0.02f, 0.005f, 0.0025f, 0.0025f},
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_STALAGMITE,
        CRAFTAX_WG_BLOCK_LAVA,
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_PATH,
        true,
        true,
        5.0f,
        1.0f,
        17.0f,
        1.5f,
        0.0f,
        0.7f,
        0.6f,
        0.8f,
        0.5f,
    },
    {
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_WATER,
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        {CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE},
        {CRAFTAX_WG_BLOCK_COAL, CRAFTAX_WG_BLOCK_IRON, CRAFTAX_WG_BLOCK_DIAMOND, CRAFTAX_WG_BLOCK_SAPPHIRE, CRAFTAX_WG_BLOCK_RUBY},
        {0.04f, 0.03f, 0.01f, 0.01f, 0.01f},
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_STALAGMITE,
        CRAFTAX_WG_BLOCK_LAVA,
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_PATH,
        true,
        true,
        5.0f,
        1.0f,
        17.0f,
        1.5f,
        0.0f,
        0.7f,
        0.6f,
        0.8f,
        0.5f,
    },
    {
        CRAFTAX_WG_BLOCK_FIRE_GRASS,
        CRAFTAX_WG_BLOCK_LAVA,
        CRAFTAX_WG_BLOCK_SAND,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        {CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE},
        {CRAFTAX_WG_BLOCK_COAL, CRAFTAX_WG_BLOCK_IRON, CRAFTAX_WG_BLOCK_DIAMOND, CRAFTAX_WG_BLOCK_SAPPHIRE, CRAFTAX_WG_BLOCK_RUBY},
        {0.05f, 0.0f, 0.0f, 0.0f, 0.025f},
        CRAFTAX_WG_BLOCK_FIRE_GRASS,
        CRAFTAX_WG_BLOCK_FIRE_TREE,
        CRAFTAX_WG_BLOCK_LAVA,
        CRAFTAX_WG_BLOCK_FIRE_GRASS,
        CRAFTAX_WG_BLOCK_FIRE_GRASS,
        true,
        true,
        5.0f,
        1.0f,
        5.0f,
        1.0f,
        1.0f,
        0.5f,
        0.6f,
        0.8f,
        0.5f,
    },
    {
        CRAFTAX_WG_BLOCK_ICE_GRASS,
        CRAFTAX_WG_BLOCK_WATER,
        CRAFTAX_WG_BLOCK_ICE_GRASS,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        {CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE},
        {CRAFTAX_WG_BLOCK_COAL, CRAFTAX_WG_BLOCK_IRON, CRAFTAX_WG_BLOCK_DIAMOND, CRAFTAX_WG_BLOCK_SAPPHIRE, CRAFTAX_WG_BLOCK_RUBY},
        {0.0f, 0.0f, 0.005f, 0.02f, 0.0f},
        CRAFTAX_WG_BLOCK_ICE_GRASS,
        CRAFTAX_WG_BLOCK_ICE_SHRUB,
        CRAFTAX_WG_BLOCK_WATER,
        CRAFTAX_WG_BLOCK_ICE_GRASS,
        CRAFTAX_WG_BLOCK_ICE_GRASS,
        true,
        true,
        5.0f,
        1.0f,
        17.0f,
        1.5f,
        0.0f,
        0.5f,
        0.6f,
        0.4f,
        0.5f,
    },
    {
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_WALL,
        CRAFTAX_WG_BLOCK_WALL,
        CRAFTAX_WG_BLOCK_WALL,
        {CRAFTAX_WG_BLOCK_WALL, CRAFTAX_WG_BLOCK_GRAVE, CRAFTAX_WG_BLOCK_GRAVE, CRAFTAX_WG_BLOCK_WALL, CRAFTAX_WG_BLOCK_WALL},
        {CRAFTAX_WG_BLOCK_WALL_MOSS, CRAFTAX_WG_BLOCK_GRAVE2, CRAFTAX_WG_BLOCK_GRAVE3, CRAFTAX_WG_BLOCK_SAPPHIRE, CRAFTAX_WG_BLOCK_RUBY},
        {0.1f, 0.333f, 0.5f, 0.0f, 0.0f},
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_GRAVE,
        CRAFTAX_WG_BLOCK_WALL,
        CRAFTAX_WG_BLOCK_NECROMANCER,
        CRAFTAX_WG_BLOCK_PATH,
        false,
        false,
        5.0f,
        1.0f,
        10.0f,
        10.0f,
        0.0f,
        0.7f,
        0.6f,
        0.95f,
        -1.0f,
    },
};

static const CraftaxDungeonConfig CRAFTAX_DUNGEON_CONFIGS[3] = {
    {CRAFTAX_WG_BLOCK_PATH, CRAFTAX_WG_BLOCK_FOUNTAIN, CRAFTAX_WG_BLOCK_PATH},
    {CRAFTAX_WG_BLOCK_ENCHANTMENT_TABLE_ICE, CRAFTAX_WG_BLOCK_WATER, CRAFTAX_WG_BLOCK_WATER},
    {CRAFTAX_WG_BLOCK_ENCHANTMENT_TABLE_FIRE, CRAFTAX_WG_BLOCK_FOUNTAIN, CRAFTAX_WG_BLOCK_PATH},
};

static inline float craftax_wg_clampf(float value, float low, float high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static inline int craftax_wg_clampi(int value, int low, int high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static inline size_t craftax_wg_index(int row, int col) {
    return (size_t)row * (size_t)CRAFTAX_WG_MAP_SIZE + (size_t)col;
}

static inline void craftax_threefry_split3(
    CraftaxThreefryKey key,
    CraftaxThreefryKey* first,
    CraftaxThreefryKey* second,
    CraftaxThreefryKey* third
) {
    CraftaxThreefryKey keys[3];
    craftax_threefry_split_n(key, keys, 3);
    *first = keys[0];
    *second = keys[1];
    *third = keys[2];
}

static inline CraftaxThreefryKey craftax_worldgen_key_from_seed(uint32_t seed) {
    CraftaxThreefryKey key = craftax_prng_key(seed);
    CraftaxThreefryKey carry;
    CraftaxThreefryKey reset_key;
    craftax_threefry_split(key, &carry, &reset_key);

    CraftaxThreefryKey reset_carry;
    CraftaxThreefryKey world_key;
    craftax_threefry_split(reset_key, &reset_carry, &world_key);
    return world_key;
}

static inline CraftaxThreefryKey craftax_overworld_rng_from_seed(uint32_t seed) {
    CraftaxThreefryKey world_key = craftax_worldgen_key_from_seed(seed);
    CraftaxThreefryKey world_keys[7];
    craftax_threefry_split_n(world_key, world_keys, 7);
    return world_keys[1];
}

static inline uint32_t craftax_randint_u32_at(
    CraftaxThreefryKey key,
    uint64_t index,
    uint32_t minval,
    uint32_t maxval
) {
    uint32_t span = maxval > minval ? maxval - minval : 1u;
    // Fast path for power-of-2 spans: just mask
    if ((span & (span - 1)) == 0) {
        uint32_t bits = craftax_threefry_uniform_u32_at(key, index);
        return minval + (bits & (span - 1));
    }
    // General path: use top-32 of hash, scale to span
    uint64_t h = craftax_fast_hash64(key, index);
    return minval + (uint32_t)(((h >> 32) * (uint64_t)span) >> 32);
}

static inline int32_t craftax_randint_i32_at(
    CraftaxThreefryKey key,
    uint64_t index,
    int32_t minval,
    int32_t maxval
) {
    return (int32_t)craftax_randint_u32_at(
        key,
        index,
        (uint32_t)minval,
        (uint32_t)maxval
    );
}

static inline int craftax_choice_bool_flat(
    CraftaxThreefryKey key,
    const bool* valid,
    int count
) {
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

    float draw = (float)valid_count * (1.0f - craftax_threefry_uniform_f32(key));
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

static inline float craftax_torch_light_value(int row, int col, float default_light) {
    float dr = (float)(row - 4);
    float dc = (float)(col - 4);
    float distance = sqrtf(dr * dr + dc * dc);
    float torch = craftax_wg_clampf(1.0f - distance / 5.0f, 0.0f, 1.0f);
    return torch * (1.0f - default_light) + default_light;
}

static inline void craftax_apply_ladder_light(
    uint8_t light_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    const int32_t ladder_up[2],
    float default_light
) {
    int start_row = ladder_up[0] - 4;
    int start_col = ladder_up[1] - 4;
    if (start_row < 0) {
        start_row += CRAFTAX_WG_MAP_SIZE;
    }
    if (start_col < 0) {
        start_col += CRAFTAX_WG_MAP_SIZE;
    }
    start_row = craftax_wg_clampi(start_row, 0, CRAFTAX_WG_MAP_SIZE - 9);
    start_col = craftax_wg_clampi(start_col, 0, CRAFTAX_WG_MAP_SIZE - 9);
    for (int row = 0; row < 9; row++) {
        for (int col = 0; col < 9; col++) {
            light_map[start_row + row][start_col + col] =
                (uint8_t)(craftax_torch_light_value(row, col, default_light) * 255.0f);
        }
    }
}

static inline void craftax_add_lava_light(
    uint8_t light_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    const bool lava_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    bool lava_emits_light
) {
    if (!lava_emits_light) {
        return;
    }

    static const float kernel[3][3] = {
        {0.2f, 0.7f, 0.2f},
        {0.7f, 1.0f, 0.7f},
        {0.2f, 0.7f, 0.2f},
    };

    for (int row = 0; row < CRAFTAX_WG_MAP_SIZE; row++) {
        for (int col = 0; col < CRAFTAX_WG_MAP_SIZE; col++) {
            float add = 0.0f;
            for (int kr = 0; kr < 3; kr++) {
                int src_row = row + kr - 1;
                if (src_row < 0 || src_row >= CRAFTAX_WG_MAP_SIZE) {
                    continue;
                }
                for (int kc = 0; kc < 3; kc++) {
                    int src_col = col + kc - 1;
                    if (src_col < 0 || src_col >= CRAFTAX_WG_MAP_SIZE) {
                        continue;
                    }
                    add += lava_map[src_row][src_col] ? kernel[kr][kc] : 0.0f;
                }
            }
            float new_light = craftax_wg_clampf(light_map[row][col] / 255.0f + add, 0.0f, 1.0f);
            light_map[row][col] = (uint8_t)(new_light * 255.0f);
        }
    }
}

static inline int craftax_smooth_config_index_for_floor(int floor_idx) {
    switch (floor_idx) {
        case 0:
            return 0;
        case 2:
            return 1;
        case 5:
            return 2;
        case 6:
            return 3;
        case 7:
            return 4;
        case 8:
            return 5;
        default:
            return -1;
    }
}

static inline int craftax_dungeon_config_index_for_floor(int floor_idx) {
    switch (floor_idx) {
        case 1:
            return 0;
        case 3:
            return 1;
        case 4:
            return 2;
        default:
            return -1;
    }
}

static inline void craftax_generate_smoothworld_config(
    CraftaxThreefryKey rng,
    int config_idx,
    uint8_t map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t item_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t light_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    int32_t ladder_down[2],
    int32_t ladder_up[2]
) {
    const CraftaxSmoothGenConfig* config = &CRAFTAX_SMOOTHGEN_CONFIGS[config_idx];
    const int size = CRAFTAX_WG_MAP_SIZE;
    const int player_row = CRAFTAX_WG_MAP_SIZE / 2;
    const int player_col = CRAFTAX_WG_MAP_SIZE / 2;
    const size_t cells = CRAFTAX_WG_MAP_CELLS;

    CraftaxThreefryKey subkey;
    float water[CRAFTAX_WG_MAP_CELLS];
    float mountain[CRAFTAX_WG_MAP_CELLS];
    float path_x[CRAFTAX_WG_MAP_CELLS];
    float tree_noise[CRAFTAX_WG_MAP_CELLS];
    bool lava_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE];

    craftax_threefry_split(rng, &rng, &subkey);
    craftax_generate_fractal_noise_2d(subkey, size, size, 3, 3, 1, 0.5f, 2, NULL, water);

    craftax_threefry_split(rng, &rng, &subkey);
    (void)subkey;

    craftax_threefry_split(rng, &rng, &subkey);
    craftax_generate_fractal_noise_2d(subkey, size, size, 3, 3, 1, 0.5f, 2, NULL, mountain);

    craftax_threefry_split(rng, &rng, &subkey);
    craftax_generate_fractal_noise_2d(subkey, size, size, 6, 24, 1, 0.5f, 2, NULL, path_x);

    craftax_threefry_split(rng, &rng, &subkey);
    (void)subkey;

    craftax_threefry_split(rng, &rng, &subkey);
    CraftaxThreefryKey tree_uniform_key = rng;
    craftax_generate_fractal_noise_2d(subkey, size, size, 12, 12, 1, 0.5f, 2, NULL, tree_noise);

    for (int row = 0; row < size; row++) {
        int dr = row > player_row ? row - player_row : player_row - row;
        for (int col = 0; col < size; col++) {
            int dc = col > player_col ? col - player_col : player_col - col;
            float distance = sqrtf((float)(dr * dr + dc * dc));
            float proximity_water = craftax_wg_clampf(
                distance / config->player_proximity_map_water_strength,
                0.0f,
                config->player_proximity_map_water_max
            );
            float proximity_mountain = craftax_wg_clampf(
                distance / config->player_proximity_map_mountain_strength,
                0.0f,
                config->player_proximity_map_mountain_max
            );
            size_t idx = craftax_wg_index(row, col);

            water[idx] = water[idx] + proximity_water - 1.0f;
            int32_t block = water[idx] > config->water_threshold
                ? config->sea_block
                : config->default_block;
            bool sand = water[idx] > config->sand_threshold && block != config->sea_block;
            if (sand) {
                block = config->coast_block;
            }

            mountain[idx] = mountain[idx] + 0.05f + proximity_mountain - 1.0f;
            if (mountain[idx] > 0.7f) {
                block = config->mountain_block;
            }

            bool path = mountain[idx] > 0.7f && path_x[idx] > 0.8f;
            if (path) {
                block = config->path_block;
            }

            float path_y = path_x[craftax_wg_index(col, row)];
            path = mountain[idx] > 0.7f && path_y > 0.8f;
            if (path) {
                block = config->path_block;
            }

            bool cave = mountain[idx] > 0.85f && water[idx] > 0.4f;
            if (cave) {
                block = config->inner_mountain_block;
            }

            float tree_draw = craftax_threefry_uniform_f32_at(tree_uniform_key, idx);
            bool tree = tree_noise[idx] > config->tree_threshold_perlin
                && tree_draw > config->tree_threshold_uniform;
            if (tree && block == config->tree_requirement_block) {
                block = config->tree;
            }

            map[row][col] = (uint8_t)block;
            item_map[row][col] = CRAFTAX_WG_ITEM_NONE;
            light_map[row][col] = (uint8_t)(config->default_light * 255.0f);
        }
    }

    CraftaxThreefryKey ore_rng;
    craftax_threefry_split(rng, &rng, &ore_rng);
    for (int ore_index = 0; ore_index < 5; ore_index++) {
        CraftaxThreefryKey ore_key;
        craftax_threefry_split(ore_rng, &ore_rng, &ore_key);
        for (int row = 0; row < size; row++) {
            for (int col = 0; col < size; col++) {
                size_t idx = craftax_wg_index(row, col);
                bool is_ore = map[row][col] == config->ore_requirement_blocks[ore_index]
                    && craftax_threefry_uniform_f32_at(ore_key, idx) < config->ore_chances[ore_index];
                if (is_ore) {
                    map[row][col] = (uint8_t)config->ores[ore_index];
                }
            }
        }
    }

    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            size_t idx = craftax_wg_index(row, col);
            lava_map[row][col] = mountain[idx] > 0.85f && tree_noise[idx] > 0.7f;
            if (lava_map[row][col]) {
                map[row][col] = (uint8_t)config->lava;
            }
        }
    }

    craftax_threefry_split(rng, &rng, &subkey);
    bool valid_diamond[CRAFTAX_WG_MAP_CELLS];
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            valid_diamond[craftax_wg_index(row, col)] = map[row][col] == CRAFTAX_WG_BLOCK_STONE;
        }
    }
    int diamond_index = craftax_choice_bool_flat(subkey, valid_diamond, (int)cells);
    map[diamond_index / size][diamond_index % size] = (uint8_t)CRAFTAX_WG_BLOCK_STONE;

    map[player_row][player_col] = (uint8_t)config->player_spawn;

    bool valid_ladder[CRAFTAX_WG_MAP_CELLS];
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            valid_ladder[craftax_wg_index(row, col)] = map[row][col] == config->valid_ladder;
        }
    }

    craftax_threefry_split(rng, &rng, &subkey);
    int ladder_down_index = craftax_choice_bool_flat(subkey, valid_ladder, (int)cells);
    ladder_down[0] = ladder_down_index / size;
    ladder_down[1] = ladder_down_index % size;
    if (config->ladder_down) {
        item_map[ladder_down[0]][ladder_down[1]] = CRAFTAX_WG_ITEM_LADDER_DOWN;
    }

    craftax_threefry_split(rng, &rng, &subkey);
    int ladder_up_index = craftax_choice_bool_flat(subkey, valid_ladder, (int)cells);
    ladder_up[0] = ladder_up_index / size;
    ladder_up[1] = ladder_up_index % size;

    craftax_apply_ladder_light(light_map, ladder_up, config->default_light);
    craftax_add_lava_light(light_map, lava_map, config->lava == CRAFTAX_WG_BLOCK_LAVA);

    if (config->ladder_up) {
        item_map[ladder_up[0]][ladder_up[1]] = CRAFTAX_WG_ITEM_LADDER_UP;
    }
}

static inline void craftax_generate_smoothworld_floor(
    CraftaxThreefryKey seed_key,
    int floor_idx,
    uint8_t map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t item_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t light_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    int32_t ladder_down[2],
    int32_t ladder_up[2]
) {
    int config_idx = craftax_smooth_config_index_for_floor(floor_idx);
    if (config_idx < 0) {
        memset(map, 0, CRAFTAX_WG_MAP_CELLS * sizeof(uint8_t));
        memset(item_map, 0, CRAFTAX_WG_MAP_CELLS * sizeof(uint8_t));
        memset(light_map, 0, CRAFTAX_WG_MAP_CELLS * sizeof(uint8_t));
        ladder_down[0] = 0;
        ladder_down[1] = 0;
        ladder_up[0] = 0;
        ladder_up[1] = 0;
        return;
    }
    craftax_generate_smoothworld_config(
        seed_key,
        config_idx,
        map,
        item_map,
        light_map,
        ladder_down,
        ladder_up
    );
}

static inline void craftax_generate_dungeon_config(
    CraftaxThreefryKey rng,
    int config_idx,
    uint8_t map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t item_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t light_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    int32_t ladder_down[2],
    int32_t ladder_up[2]
) {
    const CraftaxDungeonConfig* config = &CRAFTAX_DUNGEON_CONFIGS[config_idx];
    const int chunk_size = 16;
    const int world_chunk_height = CRAFTAX_WG_MAP_SIZE / chunk_size;
    const int num_rooms = 8;
    const int min_room_size = 5;
    const int max_room_size = 10;
    const int padded_size = CRAFTAX_WG_MAP_SIZE + 2 * max_room_size;

    uint8_t padded_map[68][68];
    uint8_t padded_item_map[68][68];
    bool room_occupancy_chunks[9];
    int32_t room_sizes[8][2];
    int32_t room_positions[8][2];

    for (int row = 0; row < padded_size; row++) {
        for (int col = 0; col < padded_size; col++) {
            bool inner = row >= max_room_size
                && row < max_room_size + CRAFTAX_WG_MAP_SIZE
                && col >= max_room_size
                && col < max_room_size + CRAFTAX_WG_MAP_SIZE;
            padded_map[row][col] = inner ? CRAFTAX_WG_BLOCK_WALL : 0;
            padded_item_map[row][col] = CRAFTAX_WG_ITEM_NONE;
        }
    }
    for (int i = 0; i < 9; i++) {
        room_occupancy_chunks[i] = true;
    }

    CraftaxThreefryKey room_scan_ignored_key;
    CraftaxThreefryKey room_size_key;
    craftax_threefry_split3(rng, &rng, &room_scan_ignored_key, &room_size_key);
    (void)room_scan_ignored_key;
    for (int room = 0; room < num_rooms; room++) {
        room_sizes[room][0] = craftax_randint_i32_at(room_size_key, (uint64_t)room * 2u, min_room_size, max_room_size);
        room_sizes[room][1] = craftax_randint_i32_at(room_size_key, (uint64_t)room * 2u + 1u, min_room_size, max_room_size);
    }

    CraftaxThreefryKey room_rng;
    craftax_threefry_split(rng, &rng, &room_rng);

    for (int room_index = 0; room_index < num_rooms; room_index++) {
        CraftaxThreefryKey choice_key;
        craftax_threefry_split(room_rng, &room_rng, &choice_key);
        int room_chunk = craftax_choice_bool_flat(choice_key, room_occupancy_chunks, 9);
        room_occupancy_chunks[room_chunk] = false;

        int room_row = (room_chunk % world_chunk_height) * chunk_size + max_room_size;
        int room_col = (room_chunk / world_chunk_height) * chunk_size + max_room_size;
        CraftaxThreefryKey position_key;
        craftax_threefry_split(room_rng, &room_rng, &position_key);
        room_row += craftax_randint_i32_at(position_key, 0, 0, chunk_size - min_room_size);
        room_col += craftax_randint_i32_at(position_key, 1, 0, chunk_size - min_room_size);
        room_positions[room_index][0] = room_row;
        room_positions[room_index][1] = room_col;

        for (int row = 0; row < max_room_size; row++) {
            for (int col = 0; col < max_room_size; col++) {
                if (row < room_sizes[room_index][0] && col < room_sizes[room_index][1]) {
                    padded_map[room_row + row][room_col + col] = CRAFTAX_WG_BLOCK_PATH;
                }
            }
        }

        padded_item_map[room_row][room_col] = CRAFTAX_WG_ITEM_TORCH;
        padded_item_map[room_row + room_sizes[room_index][0] - 1][room_col] = CRAFTAX_WG_ITEM_TORCH;
        padded_item_map[room_row][room_col + room_sizes[room_index][1] - 1] = CRAFTAX_WG_ITEM_TORCH;
        padded_item_map[room_row + room_sizes[room_index][0] - 1][room_col + room_sizes[room_index][1] - 1] = CRAFTAX_WG_ITEM_TORCH;

        CraftaxThreefryKey chest_key;
        craftax_threefry_split(room_rng, &room_rng, &chest_key);
        int chest_row = craftax_randint_i32_at(chest_key, 0, 1, room_sizes[room_index][0] - 1);
        int chest_col = craftax_randint_i32_at(chest_key, 1, 1, room_sizes[room_index][1] - 1);
        padded_map[room_row + chest_row][room_col + chest_col] = CRAFTAX_WG_BLOCK_CHEST;

        CraftaxThreefryKey fountain_key;
        CraftaxThreefryKey fountain_uniform_key;
        craftax_threefry_split3(room_rng, &room_rng, &fountain_key, &fountain_uniform_key);
        int fountain_row = craftax_randint_i32_at(fountain_key, 0, 1, room_sizes[room_index][0] - 1);
        int fountain_col = craftax_randint_i32_at(fountain_key, 1, 1, room_sizes[room_index][1] - 1);
        bool room_has_fountain = craftax_threefry_uniform_f32(fountain_uniform_key) > 0.5f;
        if (room_has_fountain) {
            padded_map[room_row + fountain_row][room_col + fountain_col] = config->fountain_block;
        }
    }

    CraftaxThreefryKey path_rng;
    craftax_threefry_split(rng, &rng, &path_rng);
    bool included_rooms_mask[8] = {false, false, false, false, false, false, false, true};

    for (int path_index = 0; path_index < num_rooms; path_index++) {
        int source_row = room_positions[path_index][0];
        int source_col = room_positions[path_index][1];

        CraftaxThreefryKey sink_key;
        craftax_threefry_split(path_rng, &path_rng, &sink_key);
        int sink_index = craftax_choice_bool_flat(sink_key, included_rooms_mask, num_rooms);
        int sink_row = room_positions[sink_index][0];
        int sink_col = room_positions[sink_index][1];

        int horizontal_distance = sink_col - source_col;
        int horizontal_sign = (horizontal_distance > 0) - (horizontal_distance < 0);
        if (horizontal_sign != 0) {
            int abs_distance = horizontal_distance > 0 ? horizontal_distance : -horizontal_distance;
            for (int col = 0; col < padded_size; col++) {
                int path_index_col = (col - source_col) * horizontal_sign;
                bool horizontal_mask = path_index_col >= 0
                    && path_index_col <= abs_distance
                    && padded_map[source_row][col] == CRAFTAX_WG_BLOCK_WALL;
                if (horizontal_mask) {
                    padded_map[source_row][col] = CRAFTAX_WG_BLOCK_PATH;
                }
            }
        }

        int vertical_distance = sink_row - source_row;
        int vertical_sign = (vertical_distance > 0) - (vertical_distance < 0);
        if (vertical_sign != 0) {
            int abs_distance = vertical_distance > 0 ? vertical_distance : -vertical_distance;
            for (int row = 0; row < padded_size; row++) {
                int path_index_row = (row - source_row) * vertical_sign;
                bool vertical_mask = path_index_row >= 0
                    && path_index_row <= abs_distance
                    && padded_map[row][sink_col] == CRAFTAX_WG_BLOCK_WALL;
                if (vertical_mask) {
                    padded_map[row][sink_col] = CRAFTAX_WG_BLOCK_PATH;
                }
            }
        }

        CraftaxThreefryKey unused_left;
        CraftaxThreefryKey next_path_rng;
        craftax_threefry_split(path_rng, &unused_left, &next_path_rng);
        path_rng = next_path_rng;
        included_rooms_mask[path_index] = true;
    }

    int special_row = room_positions[0][0] + 2;
    int special_col = room_positions[0][1] + 2;
    padded_map[special_row][special_col] = config->special_block;

    for (int row = 0; row < CRAFTAX_WG_MAP_SIZE; row++) {
        for (int col = 0; col < CRAFTAX_WG_MAP_SIZE; col++) {
            map[row][col] = padded_map[row + max_room_size][col + max_room_size];
            item_map[row][col] = padded_item_map[row + max_room_size][col + max_room_size];
        }
    }

    bool adjacent_path[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE];
    for (int row = 0; row < CRAFTAX_WG_MAP_SIZE; row++) {
        for (int col = 0; col < CRAFTAX_WG_MAP_SIZE; col++) {
            bool adjacent = map[row][col] != CRAFTAX_WG_BLOCK_WALL;
            adjacent = adjacent || (row > 0 && map[row - 1][col] != CRAFTAX_WG_BLOCK_WALL);
            adjacent = adjacent || (row + 1 < CRAFTAX_WG_MAP_SIZE && map[row + 1][col] != CRAFTAX_WG_BLOCK_WALL);
            adjacent = adjacent || (col > 0 && map[row][col - 1] != CRAFTAX_WG_BLOCK_WALL);
            adjacent = adjacent || (col + 1 < CRAFTAX_WG_MAP_SIZE && map[row][col + 1] != CRAFTAX_WG_BLOCK_WALL);
            adjacent_path[row][col] = adjacent;
        }
    }

    CraftaxThreefryKey rare_key;
    craftax_threefry_split(rng, &rng, &rare_key);
    for (int row = 0; row < CRAFTAX_WG_MAP_SIZE; row++) {
        for (int col = 0; col < CRAFTAX_WG_MAP_SIZE; col++) {
            size_t idx = craftax_wg_index(row, col);
            bool rare = (1.0f - craftax_threefry_uniform_f32_at(rare_key, idx)) > 0.9f;
            int32_t wall_map = rare ? CRAFTAX_WG_BLOCK_WALL_MOSS : CRAFTAX_WG_BLOCK_WALL;
            bool rare_path = rare && map[row][col] == CRAFTAX_WG_BLOCK_PATH && item_map[row][col] == CRAFTAX_WG_ITEM_NONE;
            int32_t path_map = rare_path ? config->rare_path_replacement_block : map[row][col];
            bool is_wall_map = map[row][col] == CRAFTAX_WG_BLOCK_WALL && adjacent_path[row][col];
            bool is_darkness_map = !adjacent_path[row][col];

            if (is_darkness_map) {
                map[row][col] = CRAFTAX_WG_BLOCK_DARKNESS;
            } else if (is_wall_map) {
                map[row][col] = wall_map;
            } else {
                map[row][col] = path_map;
            }
            light_map[row][col] = 255;
        }
    }

    bool valid_ladder[CRAFTAX_WG_MAP_CELLS];
    for (int row = 0; row < CRAFTAX_WG_MAP_SIZE; row++) {
        for (int col = 0; col < CRAFTAX_WG_MAP_SIZE; col++) {
            valid_ladder[craftax_wg_index(row, col)] = map[row][col] == CRAFTAX_WG_BLOCK_PATH;
        }
    }

    CraftaxThreefryKey ladder_down_key;
    craftax_threefry_split(rng, &rng, &ladder_down_key);
    int ladder_down_index = craftax_choice_bool_flat(ladder_down_key, valid_ladder, CRAFTAX_WG_MAP_CELLS);
    ladder_down[0] = ladder_down_index / CRAFTAX_WG_MAP_SIZE;
    ladder_down[1] = ladder_down_index % CRAFTAX_WG_MAP_SIZE;
    item_map[ladder_down[0]][ladder_down[1]] = CRAFTAX_WG_ITEM_LADDER_DOWN;

    CraftaxThreefryKey ladder_up_key;
    craftax_threefry_split(rng, &rng, &ladder_up_key);
    int ladder_up_index = craftax_choice_bool_flat(ladder_up_key, valid_ladder, CRAFTAX_WG_MAP_CELLS);
    ladder_up[0] = ladder_up_index / CRAFTAX_WG_MAP_SIZE;
    ladder_up[1] = ladder_up_index % CRAFTAX_WG_MAP_SIZE;
    item_map[ladder_up[0]][ladder_up[1]] = CRAFTAX_WG_ITEM_LADDER_UP;
}

static inline void craftax_generate_dungeon_floor(
    CraftaxThreefryKey seed_key,
    int floor_idx,
    uint8_t map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t item_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t light_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    int32_t ladder_down[2],
    int32_t ladder_up[2]
) {
    int config_idx = craftax_dungeon_config_index_for_floor(floor_idx);
    if (config_idx < 0) {
        memset(map, 0, CRAFTAX_WG_MAP_CELLS * sizeof(uint8_t));
        memset(item_map, 0, CRAFTAX_WG_MAP_CELLS * sizeof(uint8_t));
        memset(light_map, 0, CRAFTAX_WG_MAP_CELLS * sizeof(uint8_t));
        ladder_down[0] = 0;
        ladder_down[1] = 0;
        ladder_up[0] = 0;
        ladder_up[1] = 0;
        return;
    }
    craftax_generate_dungeon_config(
        seed_key,
        config_idx,
        map,
        item_map,
        light_map,
        ladder_down,
        ladder_up
    );
}

static inline void craftax_permutation_6(CraftaxThreefryKey key, int32_t out[6]) {
    CraftaxThreefryKey carry;
    CraftaxThreefryKey sort_key;
    craftax_threefry_split(key, &carry, &sort_key);
    (void)carry;

    uint32_t keys[6];
    for (int i = 0; i < 6; i++) {
        keys[i] = craftax_threefry_uniform_u32_at(sort_key, (uint64_t)i);
        out[i] = i;
    }

    for (int i = 1; i < 6; i++) {
        uint32_t key_value = keys[i];
        int32_t value = out[i];
        int j = i - 1;
        while (j >= 0 && keys[j] > key_value) {
            keys[j + 1] = keys[j];
            out[j + 1] = out[j];
            j--;
        }
        keys[j + 1] = key_value;
        out[j + 1] = value;
    }
}

static inline float craftax_calculate_initial_light_level(void) {
    float progress = 0.3f;
    float c = cosf(CRAFTAX_WG_PI * progress);
    return 1.0f - powf(fabsf(c), 3.0f);
}

static inline void craftax_init_empty_mobs3(CraftaxWGMobs3* mobs) {
    for (int level = 0; level < CRAFTAX_WG_NUM_LEVELS; level++) {
        for (int mob = 0; mob < 3; mob++) {
            mobs->health[level][mob] = 1.0f;
        }
    }
}

static inline void craftax_init_empty_mobs2(CraftaxWGMobs2* mobs) {
    for (int level = 0; level < CRAFTAX_WG_NUM_LEVELS; level++) {
        for (int mob = 0; mob < 2; mob++) {
            mobs->health[level][mob] = 1.0f;
        }
    }
}

static inline void craftax_generate_world_from_key(
    CraftaxThreefryKey rng,
    CraftaxWorldState* out
) {
    memset(out, 0, sizeof(*out));

    CraftaxThreefryKey smooth_split[7];
    craftax_threefry_split_n(rng, smooth_split, 7);
    rng = smooth_split[0];

    static const int smooth_floor_order[6] = {0, 2, 5, 6, 7, 8};
    for (int i = 0; i < 6; i++) {
        int level = smooth_floor_order[i];
        craftax_generate_smoothworld_config(
            smooth_split[i + 1],
            i,
            out->map[level],
            out->item_map[level],
            out->light_map[level],
            out->down_ladders[level],
            out->up_ladders[level]
        );
    }

    CraftaxThreefryKey dungeon_split[4];
    craftax_threefry_split_n(rng, dungeon_split, 4);
    rng = dungeon_split[0];

    static const int dungeon_floor_order[3] = {1, 3, 4};
    for (int i = 0; i < 3; i++) {
        int level = dungeon_floor_order[i];
        craftax_generate_dungeon_config(
            dungeon_split[i + 1],
            i,
            out->map[level],
            out->item_map[level],
            out->light_map[level],
            out->down_ladders[level],
            out->up_ladders[level]
        );
    }

    craftax_init_empty_mobs3(&out->melee_mobs);
    craftax_init_empty_mobs3(&out->passive_mobs);
    craftax_init_empty_mobs2(&out->ranged_mobs);
    craftax_init_empty_mobs3(&out->mob_projectiles);
    craftax_init_empty_mobs3(&out->player_projectiles);
    for (int level = 0; level < CRAFTAX_WG_NUM_LEVELS; level++) {
        for (int projectile = 0; projectile < CRAFTAX_WG_MAX_MOB_PROJECTILES; projectile++) {
            out->mob_projectile_directions[level][projectile][0] = 1;
            out->mob_projectile_directions[level][projectile][1] = 1;
        }
        for (int projectile = 0; projectile < CRAFTAX_WG_MAX_PLAYER_PROJECTILES; projectile++) {
            out->player_projectile_directions[level][projectile][0] = 1;
            out->player_projectile_directions[level][projectile][1] = 1;
        }
    }

    CraftaxThreefryKey potion_key;
    craftax_threefry_split(rng, &rng, &potion_key);
    craftax_permutation_6(potion_key, out->potion_mapping);

    CraftaxThreefryKey state_key;
    craftax_threefry_split(rng, &rng, &state_key);
    (void)rng;
    out->state_rng[0] = state_key.word[0];
    out->state_rng[1] = state_key.word[1];

    out->monsters_killed[0] = 10;
    out->player_position[0] = CRAFTAX_WG_MAP_SIZE / 2;
    out->player_position[1] = CRAFTAX_WG_MAP_SIZE / 2;
    out->player_level = 0;
    out->player_direction = CRAFTAX_WG_ACTION_UP;
    out->player_health = 9.0f;
    out->player_food = 9;
    out->player_drink = 9;
    out->player_energy = 9;
    out->player_mana = 9;
    out->player_dexterity = 1;
    out->player_strength = 1;
    out->player_intelligence = 1;
    out->boss_timesteps_to_spawn_this_round = CRAFTAX_WG_BOSS_FIGHT_SPAWN_TURNS;
    out->light_level = craftax_calculate_initial_light_level();
}

static inline void craftax_generate_world_from_seed(
    uint32_t seed,
    CraftaxWorldState* out
) {
    craftax_generate_world_from_key(craftax_worldgen_key_from_seed(seed), out);
}

static inline void craftax_generate_overworld_from_rng(
    CraftaxThreefryKey rng,
    CraftaxOverworldFloor* out
) {
    craftax_generate_smoothworld_config(
        rng,
        0,
        out->map,
        out->item_map,
        out->light_map,
        out->ladder_down,
        out->ladder_up
    );
}

static inline void craftax_generate_overworld_from_seed(
    uint32_t seed,
    CraftaxOverworldFloor* out
) {
    craftax_generate_overworld_from_rng(craftax_overworld_rng_from_seed(seed), out);
}

static inline int craftax_wg_jax_index(int32_t index, int32_t size) {
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

static inline bool craftax_wg_scatter_index(
    int32_t index,
    int32_t size,
    int* mapped_index
) {
    if (index < -size || index >= size) {
        return false;
    }
    *mapped_index = index < 0 ? index + size : index;
    return true;
}

static inline bool craftax_wg_is_boss_vulnerable(
    const CraftaxWorldState* state
) {
    int level = craftax_wg_jax_index(state->player_level, CRAFTAX_WG_NUM_LEVELS);
    bool has_melee = false;
    bool has_ranged = false;
    for (int i = 0; i < CRAFTAX_WG_MAX_MELEE_MOBS; i++) {
        has_melee = has_melee || state->melee_mobs.mask[level][i];
    }
    for (int i = 0; i < CRAFTAX_WG_MAX_RANGED_MOBS; i++) {
        has_ranged = has_ranged || state->ranged_mobs.mask[level][i];
    }
    return !has_melee
        && !has_ranged
        && state->boss_timesteps_to_spawn_this_round <= 0;
}

static inline void craftax_encode_mobs3_observation(
    const CraftaxWorldState* state,
    const CraftaxWGMobs3* mobs,
    int mob_class_index,
    int channels,
    int mob_channels_offset,
    float* obs
) {
    int level = craftax_wg_jax_index(state->player_level, CRAFTAX_WG_NUM_LEVELS);
    for (int i = 0; i < 3; i++) {
        int local_row = mobs->position[level][i][0]
            - state->player_position[0]
            + CRAFTAX_WG_OBS_ROWS / 2;
        int local_col = mobs->position[level][i][1]
            - state->player_position[1]
            + CRAFTAX_WG_OBS_COLS / 2;
        int type_id = mobs->type_id[level][i];
        int scatter_row;
        int scatter_col;
        if (!craftax_wg_scatter_index(
                local_row,
                CRAFTAX_WG_OBS_ROWS,
                &scatter_row
            )
            || !craftax_wg_scatter_index(
                local_col,
                CRAFTAX_WG_OBS_COLS,
                &scatter_col
            )
            || type_id < 0
            || type_id >= CRAFTAX_WG_NUM_MOB_TYPES) {
            continue;
        }

        bool on_screen = local_row >= 0
            && local_row < CRAFTAX_WG_OBS_ROWS
            && local_col >= 0
            && local_col < CRAFTAX_WG_OBS_COLS;
        int world_row = mobs->position[level][i][0];
        int world_col = mobs->position[level][i][1];
        bool in_bounds = world_row >= 0
            && world_row < CRAFTAX_WG_MAP_SIZE
            && world_col >= 0
            && world_col < CRAFTAX_WG_MAP_SIZE;
        bool visible = in_bounds && state->light_map[level][world_row][world_col] > 12;
        int obs_base = (scatter_row * CRAFTAX_WG_OBS_COLS + scatter_col) * channels;
        int channel = mob_channels_offset
            + mob_class_index * CRAFTAX_WG_NUM_MOB_TYPES
            + type_id;
        obs[obs_base + channel] =
            mobs->mask[level][i] && on_screen && visible ? 1.0f : 0.0f;
    }
}

static inline void craftax_encode_mobs2_observation(
    const CraftaxWorldState* state,
    const CraftaxWGMobs2* mobs,
    int mob_class_index,
    int channels,
    int mob_channels_offset,
    float* obs
) {
    int level = craftax_wg_jax_index(state->player_level, CRAFTAX_WG_NUM_LEVELS);
    for (int i = 0; i < 2; i++) {
        int local_row = mobs->position[level][i][0]
            - state->player_position[0]
            + CRAFTAX_WG_OBS_ROWS / 2;
        int local_col = mobs->position[level][i][1]
            - state->player_position[1]
            + CRAFTAX_WG_OBS_COLS / 2;
        int type_id = mobs->type_id[level][i];
        int scatter_row;
        int scatter_col;
        if (!craftax_wg_scatter_index(
                local_row,
                CRAFTAX_WG_OBS_ROWS,
                &scatter_row
            )
            || !craftax_wg_scatter_index(
                local_col,
                CRAFTAX_WG_OBS_COLS,
                &scatter_col
            )
            || type_id < 0
            || type_id >= CRAFTAX_WG_NUM_MOB_TYPES) {
            continue;
        }

        bool on_screen = local_row >= 0
            && local_row < CRAFTAX_WG_OBS_ROWS
            && local_col >= 0
            && local_col < CRAFTAX_WG_OBS_COLS;
        int world_row = mobs->position[level][i][0];
        int world_col = mobs->position[level][i][1];
        bool in_bounds = world_row >= 0
            && world_row < CRAFTAX_WG_MAP_SIZE
            && world_col >= 0
            && world_col < CRAFTAX_WG_MAP_SIZE;
        bool visible = in_bounds && state->light_map[level][world_row][world_col] > 12;
        int obs_base = (scatter_row * CRAFTAX_WG_OBS_COLS + scatter_col) * channels;
        int channel = mob_channels_offset
            + mob_class_index * CRAFTAX_WG_NUM_MOB_TYPES
            + type_id;
        obs[obs_base + channel] =
            mobs->mask[level][i] && on_screen && visible ? 1.0f : 0.0f;
    }
}

static inline void craftax_write_binary_bits(
    float* obs,
    int base,
    int value,
    int num_bits
) {
    if (num_bits == 6) {
        memcpy(obs + base, CRAFTAX_WG_BLOCK_LUT[value], 6 * sizeof(float));
    } else if (num_bits == 3) {
        memcpy(obs + base, CRAFTAX_WG_ITEM_LUT[value], 3 * sizeof(float));
    } else if (num_bits == 4) {
        memcpy(obs + base, CRAFTAX_WG_MOB_LUT[value], 4 * sizeof(float));
    } else {
        for (int i = 0; i < num_bits; i++) {
            obs[base + i] = (value & (1 << i)) ? 1.0f : 0.0f;
        }
    }
}

static inline void craftax_encode_mobs3_binary(
    const CraftaxWorldState* state,
    const CraftaxWGMobs3* mobs,
    int mob_class_index,
    int channels_per_cell,
    int mob_bits_offset,
    float* obs
) {
    int level = craftax_wg_jax_index(state->player_level, CRAFTAX_WG_NUM_LEVELS);
    for (int i = 0; i < 3; i++) {
        int type_id = mobs->type_id[level][i];
        if (type_id < 0 || type_id >= CRAFTAX_WG_NUM_MOB_TYPES
            || !mobs->mask[level][i]) {
            continue;
        }

        int local_row = mobs->position[level][i][0]
            - state->player_position[0]
            + CRAFTAX_WG_OBS_ROWS / 2;
        int local_col = mobs->position[level][i][1]
            - state->player_position[1]
            + CRAFTAX_WG_OBS_COLS / 2;
        if (local_row < 0 || local_row >= CRAFTAX_WG_OBS_ROWS
            || local_col < 0 || local_col >= CRAFTAX_WG_OBS_COLS) {
            continue;
        }

        int world_row = mobs->position[level][i][0];
        int world_col = mobs->position[level][i][1];
        if (world_row < 0 || world_row >= CRAFTAX_WG_MAP_SIZE
            || world_col < 0 || world_col >= CRAFTAX_WG_MAP_SIZE
            || state->light_map[level][world_row][world_col] <= 12) {
            continue;
        }

        int obs_base = (local_row * CRAFTAX_WG_OBS_COLS + local_col)
            * channels_per_cell;
        int class_offset = mob_bits_offset
            + mob_class_index * CRAFTAX_WG_BINARY_MOB_BITS;
        memcpy(obs + obs_base + class_offset,
               CRAFTAX_WG_MOB_LUT[type_id + 1],
               CRAFTAX_WG_BINARY_MOB_BITS * sizeof(float));
    }
}

static inline void craftax_encode_mobs2_binary(
    const CraftaxWorldState* state,
    const CraftaxWGMobs2* mobs,
    int mob_class_index,
    int channels_per_cell,
    int mob_bits_offset,
    float* obs
) {
    int level = craftax_wg_jax_index(state->player_level, CRAFTAX_WG_NUM_LEVELS);
    for (int i = 0; i < 2; i++) {
        int type_id = mobs->type_id[level][i];
        if (type_id < 0 || type_id >= CRAFTAX_WG_NUM_MOB_TYPES
            || !mobs->mask[level][i]) {
            continue;
        }

        int local_row = mobs->position[level][i][0]
            - state->player_position[0]
            + CRAFTAX_WG_OBS_ROWS / 2;
        int local_col = mobs->position[level][i][1]
            - state->player_position[1]
            + CRAFTAX_WG_OBS_COLS / 2;
        if (local_row < 0 || local_row >= CRAFTAX_WG_OBS_ROWS
            || local_col < 0 || local_col >= CRAFTAX_WG_OBS_COLS) {
            continue;
        }

        int world_row = mobs->position[level][i][0];
        int world_col = mobs->position[level][i][1];
        if (world_row < 0 || world_row >= CRAFTAX_WG_MAP_SIZE
            || world_col < 0 || world_col >= CRAFTAX_WG_MAP_SIZE
            || state->light_map[level][world_row][world_col] <= 12) {
            continue;
        }

        int obs_base = (local_row * CRAFTAX_WG_OBS_COLS + local_col)
            * channels_per_cell;
        int class_offset = mob_bits_offset
            + mob_class_index * CRAFTAX_WG_BINARY_MOB_BITS;
        memcpy(obs + obs_base + class_offset,
               CRAFTAX_WG_MOB_LUT[type_id + 1],
               CRAFTAX_WG_BINARY_MOB_BITS * sizeof(float));
    }
}

static inline void craftax_encode_map_base_observation(
    const CraftaxWorldState* state,
    float* obs
) {
    const int channels = CRAFTAX_WG_BINARY_CHANNELS_PER_CELL;
    const int top = state->player_position[0] - CRAFTAX_WG_OBS_ROWS / 2;
    const int left = state->player_position[1] - CRAFTAX_WG_OBS_COLS / 2;
    const int level = state->player_level;
    const float* empty_cell = CRAFTAX_WG_EMPTY_CELL_TEMPLATE;

    for (int row = 0; row < CRAFTAX_WG_OBS_ROWS; row++) {
        int world_row = top + row;
        bool row_in_bounds = world_row >= 0 && world_row < CRAFTAX_WG_MAP_SIZE;
        for (int col = 0; col < CRAFTAX_WG_OBS_COLS; col++) {
            int world_col = left + col;
            int obs_base = (row * CRAFTAX_WG_OBS_COLS + col) * channels;
            const float* cell = empty_cell;

            if (row_in_bounds && world_col >= 0 && world_col < CRAFTAX_WG_MAP_SIZE
                && state->light_map[level][world_row][world_col] > 12) {
                uint8_t block = state->map[level][world_row][world_col];
                uint8_t item = state->item_map[level][world_row][world_col];
                cell = CRAFTAX_WG_VISIBLE_CELL_TEMPLATE_LUT[block][item + 1];
            }

            memcpy(obs + obs_base, cell, CRAFTAX_WG_CELL_TEMPLATE_BYTES);
        }
    }
}

static inline void craftax_encode_packed_map_base_observation(
    const CraftaxWorldState* state,
    float* obs
) {
    const int channels = CRAFTAX_WG_PACKED_CHANNELS_PER_CELL;
    const int top = state->player_position[0] - CRAFTAX_WG_OBS_ROWS / 2;
    const int left = state->player_position[1] - CRAFTAX_WG_OBS_COLS / 2;
    const int level = state->player_level;

    memset(obs, 0, CRAFTAX_WG_PACKED_MAP_OBS_SIZE * sizeof(float));
    for (int row = 0; row < CRAFTAX_WG_OBS_ROWS; row++) {
        int world_row = top + row;
        bool row_in_bounds = world_row >= 0 && world_row < CRAFTAX_WG_MAP_SIZE;
        for (int col = 0; col < CRAFTAX_WG_OBS_COLS; col++) {
            int world_col = left + col;
            int obs_base = (row * CRAFTAX_WG_OBS_COLS + col) * channels;
            if (row_in_bounds && world_col >= 0 && world_col < CRAFTAX_WG_MAP_SIZE
                && state->light_map[level][world_row][world_col] > 12) {
                obs[obs_base + 0] = (float)state->map[level][world_row][world_col];
                obs[obs_base + 1] = (float)state->item_map[level][world_row][world_col] + 1.0f;
                obs[obs_base + 2] = 1.0f;
            }
        }
    }
}

static inline void craftax_clear_mob_channels_observation(float* obs) {
    const int channels = CRAFTAX_WG_BINARY_CHANNELS_PER_CELL;
    const int mob_bits_offset = CRAFTAX_WG_BINARY_BLOCK_BITS + CRAFTAX_WG_BINARY_ITEM_BITS;
    const size_t mob_channel_bytes =
        CRAFTAX_WG_NUM_MOB_CLASSES * CRAFTAX_WG_BINARY_MOB_BITS * sizeof(float);

    for (int cell = 0; cell < CRAFTAX_WG_OBS_WINDOW_CELLS; cell++) {
        memset(obs + cell * channels + mob_bits_offset, 0, mob_channel_bytes);
    }
}

static inline void craftax_encode_mobs3_packed(
    const CraftaxWorldState* state,
    const CraftaxWGMobs3* mobs,
    int mob_class_index,
    float* obs
) {
    const int level = craftax_wg_jax_index(state->player_level, CRAFTAX_WG_NUM_LEVELS);
    const int mob_slot_offset = 3 + mob_class_index;
    for (int i = 0; i < 3; i++) {
        int type_id = mobs->type_id[level][i];
        if (type_id < 0 || type_id >= CRAFTAX_WG_NUM_MOB_TYPES
            || !mobs->mask[level][i]) {
            continue;
        }

        int local_row = mobs->position[level][i][0]
            - state->player_position[0]
            + CRAFTAX_WG_OBS_ROWS / 2;
        int local_col = mobs->position[level][i][1]
            - state->player_position[1]
            + CRAFTAX_WG_OBS_COLS / 2;
        if (local_row < 0 || local_row >= CRAFTAX_WG_OBS_ROWS
            || local_col < 0 || local_col >= CRAFTAX_WG_OBS_COLS) {
            continue;
        }

        int world_row = mobs->position[level][i][0];
        int world_col = mobs->position[level][i][1];
        if (world_row < 0 || world_row >= CRAFTAX_WG_MAP_SIZE
            || world_col < 0 || world_col >= CRAFTAX_WG_MAP_SIZE
            || state->light_map[level][world_row][world_col] <= 12) {
            continue;
        }

        int obs_base = (local_row * CRAFTAX_WG_OBS_COLS + local_col)
            * CRAFTAX_WG_PACKED_CHANNELS_PER_CELL;
        obs[obs_base + mob_slot_offset] = (float)(type_id + 1);
    }
}

static inline void craftax_encode_mobs2_packed(
    const CraftaxWorldState* state,
    const CraftaxWGMobs2* mobs,
    int mob_class_index,
    float* obs
) {
    const int level = craftax_wg_jax_index(state->player_level, CRAFTAX_WG_NUM_LEVELS);
    const int mob_slot_offset = 3 + mob_class_index;
    for (int i = 0; i < 2; i++) {
        int type_id = mobs->type_id[level][i];
        if (type_id < 0 || type_id >= CRAFTAX_WG_NUM_MOB_TYPES
            || !mobs->mask[level][i]) {
            continue;
        }

        int local_row = mobs->position[level][i][0]
            - state->player_position[0]
            + CRAFTAX_WG_OBS_ROWS / 2;
        int local_col = mobs->position[level][i][1]
            - state->player_position[1]
            + CRAFTAX_WG_OBS_COLS / 2;
        if (local_row < 0 || local_row >= CRAFTAX_WG_OBS_ROWS
            || local_col < 0 || local_col >= CRAFTAX_WG_OBS_COLS) {
            continue;
        }

        int world_row = mobs->position[level][i][0];
        int world_col = mobs->position[level][i][1];
        if (world_row < 0 || world_row >= CRAFTAX_WG_MAP_SIZE
            || world_col < 0 || world_col >= CRAFTAX_WG_MAP_SIZE
            || state->light_map[level][world_row][world_col] <= 12) {
            continue;
        }

        int obs_base = (local_row * CRAFTAX_WG_OBS_COLS + local_col)
            * CRAFTAX_WG_PACKED_CHANNELS_PER_CELL;
        obs[obs_base + mob_slot_offset] = (float)(type_id + 1);
    }
}

static inline void craftax_encode_packed_mobs_observation(
    const CraftaxWorldState* state,
    float* obs
) {
    craftax_encode_mobs3_packed(state, &state->melee_mobs, 0, obs);
    craftax_encode_mobs3_packed(state, &state->passive_mobs, 1, obs);
    craftax_encode_mobs2_packed(state, &state->ranged_mobs, 2, obs);
    craftax_encode_mobs3_packed(state, &state->mob_projectiles, 3, obs);
    craftax_encode_mobs3_packed(state, &state->player_projectiles, 4, obs);
}

static inline void craftax_encode_mobs_observation(
    const CraftaxWorldState* state,
    float* obs
) {
    const int channels = CRAFTAX_WG_BINARY_CHANNELS_PER_CELL;
    const int mob_bits_offset = CRAFTAX_WG_BINARY_BLOCK_BITS + CRAFTAX_WG_BINARY_ITEM_BITS;

    craftax_encode_mobs3_binary(
        state,
        &state->melee_mobs,
        0,
        channels,
        mob_bits_offset,
        obs
    );
    craftax_encode_mobs3_binary(
        state,
        &state->passive_mobs,
        1,
        channels,
        mob_bits_offset,
        obs
    );
    craftax_encode_mobs2_binary(
        state,
        &state->ranged_mobs,
        2,
        channels,
        mob_bits_offset,
        obs
    );
    craftax_encode_mobs3_binary(
        state,
        &state->mob_projectiles,
        3,
        channels,
        mob_bits_offset,
        obs
    );
    craftax_encode_mobs3_binary(
        state,
        &state->player_projectiles,
        4,
        channels,
        mob_bits_offset,
        obs
    );
}

static inline void craftax_encode_scalar_observation_tail_at(
    const CraftaxWorldState* state,
    float* obs,
    int index
) {
    const int level = state->player_level;
    obs[index++] = sqrtf((float)state->inventory.wood) / 10.0f;
    obs[index++] = sqrtf((float)state->inventory.stone) / 10.0f;
    obs[index++] = sqrtf((float)state->inventory.coal) / 10.0f;
    obs[index++] = sqrtf((float)state->inventory.iron) / 10.0f;
    obs[index++] = sqrtf((float)state->inventory.diamond) / 10.0f;
    obs[index++] = sqrtf((float)state->inventory.sapphire) / 10.0f;
    obs[index++] = sqrtf((float)state->inventory.ruby) / 10.0f;
    obs[index++] = sqrtf((float)state->inventory.sapling) / 10.0f;
    obs[index++] = sqrtf((float)state->inventory.torches) / 10.0f;
    obs[index++] = sqrtf((float)state->inventory.arrows) / 10.0f;
    obs[index++] = (float)state->inventory.books / 2.0f;
    obs[index++] = (float)state->inventory.pickaxe / 4.0f;
    obs[index++] = (float)state->inventory.sword / 4.0f;
    obs[index++] = (float)state->sword_enchantment;
    obs[index++] = (float)state->bow_enchantment;
    obs[index++] = (float)state->inventory.bow;

    for (int i = 0; i < 6; i++) {
        obs[index++] = sqrtf((float)state->inventory.potions[i]) / 10.0f;
    }

    obs[index++] = state->player_health / 10.0f;
    obs[index++] = (float)state->player_food / 10.0f;
    obs[index++] = (float)state->player_drink / 10.0f;
    obs[index++] = (float)state->player_energy / 10.0f;
    obs[index++] = (float)state->player_mana / 10.0f;
    obs[index++] = (float)state->player_xp / 10.0f;
    obs[index++] = (float)state->player_dexterity / 10.0f;
    obs[index++] = (float)state->player_strength / 10.0f;
    obs[index++] = (float)state->player_intelligence / 10.0f;

    int direction_index = state->player_direction - 1;
    for (int i = 0; i < 4; i++) {
        obs[index++] = i == direction_index ? 1.0f : 0.0f;
    }

    for (int i = 0; i < 4; i++) {
        obs[index++] = (float)state->inventory.armour[i] / 2.0f;
    }
    for (int i = 0; i < 4; i++) {
        obs[index++] = (float)state->armour_enchantments[i];
    }

    obs[index++] = state->light_level;
    obs[index++] = state->is_sleeping ? 1.0f : 0.0f;
    obs[index++] = state->is_resting ? 1.0f : 0.0f;
    obs[index++] = state->learned_spells[0] ? 1.0f : 0.0f;
    obs[index++] = state->learned_spells[1] ? 1.0f : 0.0f;
    obs[index++] = (float)state->player_level / 10.0f;
    obs[index++] = state->monsters_killed[level] >= CRAFTAX_WG_MONSTERS_KILLED_TO_CLEAR_LEVEL ? 1.0f : 0.0f;
    obs[index++] = craftax_wg_is_boss_vulnerable(state) ? 1.0f : 0.0f;
}

static inline void craftax_encode_scalar_observation_tail(
    const CraftaxWorldState* state,
    float* obs
) {
    craftax_encode_scalar_observation_tail_at(state, obs, CRAFTAX_WG_BINARY_MAP_OBS_SIZE);
}

static inline void craftax_encode_reset_observation(
    const CraftaxWorldState* state,
    float* obs
) {
    craftax_encode_packed_map_base_observation(state, obs);
    craftax_encode_packed_mobs_observation(state, obs);
    craftax_encode_scalar_observation_tail_at(state, obs, CRAFTAX_WG_PACKED_MAP_OBS_SIZE);
}
