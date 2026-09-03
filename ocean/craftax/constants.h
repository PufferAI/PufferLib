#pragma once

#include <stdbool.h>

#define OBS_ROWS 9
#define OBS_COLS 11
#define MAP_SIZE 48
#define NUM_LEVELS 9

#define NUM_BLOCK_TYPES 37
#define NUM_ITEM_TYPES 5
#define NUM_MOB_CLASSES 5
#define NUM_MOB_TYPES 8
#define INVENTORY_OBS_SIZE 51
#define OBS_TILE_CHANNELS (3 + NUM_MOB_CLASSES)
#define OBS_SIZE (OBS_ROWS * OBS_COLS * OBS_TILE_CHANNELS + INVENTORY_OBS_SIZE)

#define ATN_DIM 43
#define NUM_ACHIEVEMENTS 67

#define MAX_MELEE_MOBS 3
#define MAX_PASSIVE_MOBS 3
#define MAX_RANGED_MOBS 2
#define MAX_MOB_PROJECTILES 3
#define MAX_PLAYER_PROJECTILES 3
#define MAX_GROWING_PLANTS 10
#define NUM_POTIONS 6

// Environment parameters
#define DEFAULT_MAX_TIMESTEPS 100000
#define DAY_LENGTH 300
#define VISIBLE_LIGHT_THRESHOLD 12
#define MAX_ATTRIBUTE 5
#define MOB_DESPAWN_DISTANCE 14
#define MONSTERS_KILLED_TO_CLEAR_LEVEL 8

#define SURFACE_SPAWN_RADIUS 5
#define BOSS_ARENA_RADIUS 14
#define BOSS_GRAVE_RING_RADIUS 5
#define SURFACE_PLANT_COUNT 18
#define DUNGEON_ROOM_COUNT 8
#define DUNGEON_MIN_ROOM_SIZE 5
#define DUNGEON_MAX_ROOM_SIZE 10
#define DUNGEON_CHUNK_SIZE 16
#define BOSS_GRAVE_COUNT 80
#define BOSS_SPAWN_TURNS 7

#define MAP_CELLS (MAP_SIZE * MAP_SIZE)
#define NOISE_PI2 6.28318530717958647692f
#define NOISE_SQRT2 1.41421356237309504880f

typedef enum {
    LEVEL_SURFACE = 0,
    LEVEL_DUNGEON = 1,
    LEVEL_BOSS = 2,
} LevelKind;

typedef struct {
    int base_block;
    int ore_a;
    float ore_a_rate;
    int ore_b;
    float ore_b_rate;
    int ore_c;
    float ore_c_rate;
} OreConfig;

typedef struct {
    int kind;
    int ground;
    int liquid;
    int shore;
    int wall;
    int path;
    int plant;
    int chest;
    int special;
    bool has_ladder_up;
    bool has_ladder_down;
    OreConfig ores[2];
    int ore_config_count;
} LevelConfig;

typedef struct {
    int default_block;
    int sea_block;
    int coast_block;
    int mountain_block;
    int path_block;
    int inner_mountain_block;
    int ore_requirement_blocks[5];
    int ores[5];
    float ore_chances[5];
    int tree_requirement_block;
    int tree;
    int lava;
    int player_spawn;
    int valid_ladder;
    bool ladder_up;
    bool ladder_down;
    float water_strength;
    float water_max;
    float mountain_strength;
    float mountain_max;
    float default_light;
    float water_threshold;
    float sand_threshold;
    float tree_threshold_uniform;
    float tree_threshold_perlin;
} SmoothGenConfig;

typedef struct {
    int special_block;
    int fountain_block;
    int rare_path_replacement_block;
} DungeonConfig;

typedef enum {
    BLOCK_INVALID = 0,
    BLOCK_OUT_OF_BOUNDS = 1,
    BLOCK_GRASS = 2,
    BLOCK_WATER = 3,
    BLOCK_STONE = 4,
    BLOCK_TREE = 5,
    BLOCK_WOOD = 6,
    BLOCK_PATH = 7,
    BLOCK_COAL = 8,
    BLOCK_IRON = 9,
    BLOCK_DIAMOND = 10,
    BLOCK_CRAFTING_TABLE = 11,
    BLOCK_FURNACE = 12,
    BLOCK_SAND = 13,
    BLOCK_LAVA = 14,
    BLOCK_PLANT = 15,
    BLOCK_RIPE_PLANT = 16,
    BLOCK_WALL = 17,
    BLOCK_DARKNESS = 18,
    BLOCK_WALL_MOSS = 19,
    BLOCK_STALAGMITE = 20,
    BLOCK_SAPPHIRE = 21,
    BLOCK_RUBY = 22,
    BLOCK_CHEST = 23,
    BLOCK_FOUNTAIN = 24,
    BLOCK_FIRE_GRASS = 25,
    BLOCK_ICE_GRASS = 26,
    BLOCK_GRAVEL = 27,
    BLOCK_FIRE_TREE = 28,
    BLOCK_ICE_SHRUB = 29,
    BLOCK_ENCHANTMENT_TABLE_FIRE = 30,
    BLOCK_ENCHANTMENT_TABLE_ICE = 31,
    BLOCK_NECROMANCER = 32,
    BLOCK_GRAVE = 33,
    BLOCK_GRAVE2 = 34,
    BLOCK_GRAVE3 = 35,
    BLOCK_NECROMANCER_VULNERABLE = 36,
} BlockType;

typedef enum {
    ITEM_NONE = 0,
    ITEM_TORCH = 1,
    ITEM_LADDER_DOWN = 2,
    ITEM_LADDER_UP = 3,
    ITEM_LADDER_DOWN_BLOCKED = 4,
} ItemType;

typedef enum {
    MOB_PASSIVE = 0,
    MOB_MELEE = 1,
    MOB_RANGED = 2,
    MOB_PROJECTILE = 3,
} MobType;

typedef enum {
    PROJECTILE_ARROW = 0,
    PROJECTILE_DAGGER = 1,
    PROJECTILE_FIREBALL = 2,
    PROJECTILE_ICEBALL = 3,
    PROJECTILE_ARROW2 = 4,
    PROJECTILE_SLIMEBALL = 5,
    PROJECTILE_FIREBALL2 = 6,
    PROJECTILE_ICEBALL2 = 7,
} ProjectileType;

typedef enum {
    ACH_COLLECT_WOOD = 0,
    ACH_PLACE_TABLE = 1,
    ACH_EAT_COW = 2,
    ACH_COLLECT_SAPLING = 3,
    ACH_COLLECT_DRINK = 4,
    ACH_MAKE_WOOD_PICKAXE = 5,
    ACH_MAKE_WOOD_SWORD = 6,
    ACH_PLACE_PLANT = 7,
    ACH_DEFEAT_ZOMBIE = 8,
    ACH_COLLECT_STONE = 9,
    ACH_PLACE_STONE = 10,
    ACH_EAT_PLANT = 11,
    ACH_DEFEAT_SKELETON = 12,
    ACH_MAKE_STONE_PICKAXE = 13,
    ACH_MAKE_STONE_SWORD = 14,
    ACH_WAKE_UP = 15,
    ACH_PLACE_FURNACE = 16,
    ACH_COLLECT_COAL = 17,
    ACH_COLLECT_IRON = 18,
    ACH_COLLECT_DIAMOND = 19,
    ACH_MAKE_IRON_PICKAXE = 20,
    ACH_MAKE_IRON_SWORD = 21,
    ACH_MAKE_ARROW = 22,
    ACH_MAKE_TORCH = 23,
    ACH_PLACE_TORCH = 24,
    ACH_MAKE_DIAMOND_SWORD = 25,
    ACH_MAKE_IRON_ARMOUR = 26,
    ACH_MAKE_DIAMOND_ARMOUR = 27,
    ACH_ENTER_GNOMISH_MINES = 28,
    ACH_ENTER_DUNGEON = 29,
    ACH_ENTER_SEWERS = 30,
    ACH_ENTER_VAULT = 31,
    ACH_ENTER_TROLL_MINES = 32,
    ACH_ENTER_FIRE_REALM = 33,
    ACH_ENTER_ICE_REALM = 34,
    ACH_ENTER_GRAVEYARD = 35,
    ACH_DEFEAT_GNOME_WARRIOR = 36,
    ACH_DEFEAT_GNOME_ARCHER = 37,
    ACH_DEFEAT_ORC_SOLIDER = 38,
    ACH_DEFEAT_ORC_MAGE = 39,
    ACH_DEFEAT_LIZARD = 40,
    ACH_DEFEAT_KOBOLD = 41,
    ACH_DEFEAT_TROLL = 42,
    ACH_DEFEAT_DEEP_THING = 43,
    ACH_DEFEAT_PIGMAN = 44,
    ACH_DEFEAT_FIRE_ELEMENTAL = 45,
    ACH_DEFEAT_FROST_TROLL = 46,
    ACH_DEFEAT_ICE_ELEMENTAL = 47,
    ACH_DAMAGE_NECROMANCER = 48,
    ACH_DEFEAT_NECROMANCER = 49,
    ACH_EAT_BAT = 50,
    ACH_EAT_SNAIL = 51,
    ACH_FIND_BOW = 52,
    ACH_FIRE_BOW = 53,
    ACH_COLLECT_SAPPHIRE = 54,
    ACH_LEARN_FIREBALL = 55,
    ACH_CAST_FIREBALL = 56,
    ACH_LEARN_ICEBALL = 57,
    ACH_CAST_ICEBALL = 58,
    ACH_COLLECT_RUBY = 59,
    ACH_MAKE_DIAMOND_PICKAXE = 60,
    ACH_OPEN_CHEST = 61,
    ACH_DRINK_POTION = 62,
    ACH_ENCHANT_SWORD = 63,
    ACH_ENCHANT_ARMOUR = 64,
    ACH_DEFEAT_KNIGHT = 65,
    ACH_DEFEAT_ARCHER = 66,
} Achievement;

static const float ACHIEVEMENT_REWARD_MAP[NUM_ACHIEVEMENTS] = {
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

static inline float max_achievement_return(void) {
    float sum = 0.0f;
    for (int i = 0; i < NUM_ACHIEVEMENTS; i++) {
        sum += ACHIEVEMENT_REWARD_MAP[i];
    }
    return sum;
}

typedef enum {
    ACTION_NOOP = 0,
    ACTION_LEFT = 1,
    ACTION_RIGHT = 2,
    ACTION_UP = 3,
    ACTION_DOWN = 4,
    ACTION_DO = 5,
    ACTION_SLEEP = 6,
    ACTION_PLACE_STONE = 7,
    ACTION_PLACE_TABLE = 8,
    ACTION_PLACE_FURNACE = 9,
    ACTION_PLACE_PLANT = 10,
    ACTION_MAKE_WOOD_PICKAXE = 11,
    ACTION_MAKE_STONE_PICKAXE = 12,
    ACTION_MAKE_IRON_PICKAXE = 13,
    ACTION_MAKE_WOOD_SWORD = 14,
    ACTION_MAKE_STONE_SWORD = 15,
    ACTION_MAKE_IRON_SWORD = 16,
    ACTION_REST = 17,
    ACTION_DESCEND = 18,
    ACTION_ASCEND = 19,
    ACTION_MAKE_DIAMOND_PICKAXE = 20,
    ACTION_MAKE_DIAMOND_SWORD = 21,
    ACTION_MAKE_IRON_ARMOUR = 22,
    ACTION_MAKE_DIAMOND_ARMOUR = 23,
    ACTION_SHOOT_ARROW = 24,
    ACTION_MAKE_ARROW = 25,
    ACTION_CAST_FIREBALL = 26,
    ACTION_CAST_ICEBALL = 27,
    ACTION_PLACE_TORCH = 28,
    ACTION_DRINK_POTION_RED = 29,
    ACTION_DRINK_POTION_GREEN = 30,
    ACTION_DRINK_POTION_BLUE = 31,
    ACTION_DRINK_POTION_PINK = 32,
    ACTION_DRINK_POTION_CYAN = 33,
    ACTION_DRINK_POTION_YELLOW = 34,
    ACTION_READ_BOOK = 35,
    ACTION_ENCHANT_SWORD = 36,
    ACTION_ENCHANT_ARMOUR = 37,
    ACTION_MAKE_TORCH = 38,
    ACTION_LEVEL_UP_DEXTERITY = 39,
    ACTION_LEVEL_UP_STRENGTH = 40,
    ACTION_LEVEL_UP_INTELLIGENCE = 41,
    ACTION_ENCHANT_BOW = 42,
} Action;

// {kind, ground, liquid, shore, wall, path, plant, chest, special, has_ladder_up, has_ladder_down, ores, ore_config_count}
static const LevelConfig LEVEL_CONFIGS[NUM_LEVELS] = {
    {
        LEVEL_SURFACE, BLOCK_GRASS, BLOCK_WATER, BLOCK_SAND, BLOCK_STONE,
        BLOCK_PATH, BLOCK_TREE, BLOCK_CHEST, BLOCK_INVALID, false, true,
        {{BLOCK_STONE, BLOCK_COAL, 0.03f, BLOCK_IRON, 0.02f, BLOCK_DIAMOND, 0.001f}},
        1,
    },
    {
        LEVEL_DUNGEON, BLOCK_PATH, BLOCK_WATER, BLOCK_PATH, BLOCK_WALL,
        BLOCK_PATH, BLOCK_STALAGMITE, BLOCK_CHEST, BLOCK_FOUNTAIN, true, true,
        {{BLOCK_WALL, BLOCK_COAL, 0.05f, BLOCK_IRON, 0.025f, BLOCK_DIAMOND, 0.006f}},
        1,
    },
    {
        LEVEL_SURFACE, BLOCK_PATH, BLOCK_WATER, BLOCK_PATH, BLOCK_STONE,
        BLOCK_STONE, BLOCK_STALAGMITE, BLOCK_CHEST, BLOCK_INVALID, true, true,
        {
            {BLOCK_STONE, BLOCK_COAL, 0.04f, BLOCK_IRON, 0.02f, BLOCK_DIAMOND, 0.005f},
            {BLOCK_STONE, BLOCK_SAPPHIRE, 0.0025f, BLOCK_RUBY, 0.0025f, BLOCK_INVALID, 0.0f},
        },
        2,
    },
    {
        LEVEL_DUNGEON, BLOCK_PATH, BLOCK_WATER, BLOCK_PATH, BLOCK_WALL_MOSS,
        BLOCK_PATH, BLOCK_STALAGMITE, BLOCK_CHEST, BLOCK_ENCHANTMENT_TABLE_ICE, true, true,
        {{BLOCK_WALL_MOSS, BLOCK_COAL, 0.05f, BLOCK_IRON, 0.025f, BLOCK_DIAMOND, 0.006f}},
        1,
    },
    {
        LEVEL_DUNGEON, BLOCK_PATH, BLOCK_LAVA, BLOCK_PATH, BLOCK_WALL,
        BLOCK_PATH, BLOCK_STALAGMITE, BLOCK_CHEST, BLOCK_ENCHANTMENT_TABLE_FIRE, true, true,
        {{BLOCK_WALL, BLOCK_COAL, 0.05f, BLOCK_IRON, 0.025f, BLOCK_DIAMOND, 0.006f}},
        1,
    },
    {
        LEVEL_SURFACE, BLOCK_PATH, BLOCK_WATER, BLOCK_PATH, BLOCK_STONE,
        BLOCK_STONE, BLOCK_STALAGMITE, BLOCK_CHEST, BLOCK_INVALID, true, true,
        {
            {BLOCK_STONE, BLOCK_COAL, 0.04f, BLOCK_IRON, 0.03f, BLOCK_DIAMOND, 0.01f},
            {BLOCK_STONE, BLOCK_SAPPHIRE, 0.01f, BLOCK_RUBY, 0.01f, BLOCK_INVALID, 0.0f},
        },
        2,
    },
    {
        LEVEL_SURFACE, BLOCK_FIRE_GRASS, BLOCK_LAVA, BLOCK_SAND, BLOCK_STONE,
        BLOCK_STONE, BLOCK_FIRE_TREE, BLOCK_CHEST, BLOCK_INVALID, true, true,
        {
            {BLOCK_STONE, BLOCK_COAL, 0.05f, BLOCK_IRON, 0.0f, BLOCK_DIAMOND, 0.0f},
            {BLOCK_STONE, BLOCK_SAPPHIRE, 0.0f, BLOCK_RUBY, 0.025f, BLOCK_INVALID, 0.0f},
        },
        2,
    },
    {
        LEVEL_SURFACE, BLOCK_ICE_GRASS, BLOCK_WATER, BLOCK_ICE_GRASS, BLOCK_STONE,
        BLOCK_STONE, BLOCK_ICE_SHRUB, BLOCK_CHEST, BLOCK_INVALID, true, true,
        {
            {BLOCK_STONE, BLOCK_COAL, 0.0f, BLOCK_IRON, 0.0f, BLOCK_DIAMOND, 0.005f},
            {BLOCK_STONE, BLOCK_SAPPHIRE, 0.02f, BLOCK_RUBY, 0.0f, BLOCK_INVALID, 0.0f},
        },
        2,
    },
    {
        LEVEL_BOSS, BLOCK_PATH, BLOCK_PATH, BLOCK_PATH, BLOCK_WALL,
        BLOCK_WALL, BLOCK_GRAVE, BLOCK_CHEST, BLOCK_NECROMANCER, false, false,
        {
            {BLOCK_WALL, BLOCK_WALL_MOSS, 0.1f, BLOCK_GRAVE2, 0.333f, BLOCK_GRAVE3, 0.5f},
            {BLOCK_WALL, BLOCK_SAPPHIRE, 0.0f, BLOCK_RUBY, 0.0f, BLOCK_INVALID, 0.0f},
        },
        2,
    },
};

static const SmoothGenConfig SMOOTH_LEVEL_CONFIGS[6] = {
    {
        BLOCK_GRASS, BLOCK_WATER, BLOCK_SAND, BLOCK_STONE, BLOCK_PATH, BLOCK_PATH,
        {BLOCK_STONE, BLOCK_STONE, BLOCK_STONE, BLOCK_STONE, BLOCK_STONE},
        {BLOCK_COAL, BLOCK_IRON, BLOCK_DIAMOND, BLOCK_OUT_OF_BOUNDS, BLOCK_OUT_OF_BOUNDS},
        {0.03f, 0.02f, 0.001f, 0.0f, 0.0f},
        BLOCK_GRASS, BLOCK_TREE, BLOCK_LAVA, BLOCK_GRASS, BLOCK_PATH,
        false, true, 5.0f, 1.0f, 5.0f, 1.0f, 1.0f, 0.7f, 0.6f, 0.8f, 0.5f,
    },
    {
        BLOCK_PATH, BLOCK_WATER, BLOCK_PATH, BLOCK_STONE, BLOCK_STONE, BLOCK_STONE,
        {BLOCK_STONE, BLOCK_STONE, BLOCK_STONE, BLOCK_STONE, BLOCK_STONE},
        {BLOCK_COAL, BLOCK_IRON, BLOCK_DIAMOND, BLOCK_SAPPHIRE, BLOCK_RUBY},
        {0.04f, 0.02f, 0.005f, 0.0025f, 0.0025f},
        BLOCK_PATH, BLOCK_STALAGMITE, BLOCK_LAVA, BLOCK_PATH, BLOCK_PATH,
        true, true, 5.0f, 1.0f, 17.0f, 1.5f, 0.0f, 0.7f, 0.6f, 0.8f, 0.5f,
    },
    {
        BLOCK_PATH, BLOCK_WATER, BLOCK_PATH, BLOCK_STONE, BLOCK_STONE, BLOCK_STONE,
        {BLOCK_STONE, BLOCK_STONE, BLOCK_STONE, BLOCK_STONE, BLOCK_STONE},
        {BLOCK_COAL, BLOCK_IRON, BLOCK_DIAMOND, BLOCK_SAPPHIRE, BLOCK_RUBY},
        {0.04f, 0.03f, 0.01f, 0.01f, 0.01f},
        BLOCK_PATH, BLOCK_STALAGMITE, BLOCK_LAVA, BLOCK_PATH, BLOCK_PATH,
        true, true, 5.0f, 1.0f, 17.0f, 1.5f, 0.0f, 0.7f, 0.6f, 0.8f, 0.5f,
    },
    {
        BLOCK_FIRE_GRASS, BLOCK_LAVA, BLOCK_SAND, BLOCK_STONE, BLOCK_STONE, BLOCK_STONE,
        {BLOCK_STONE, BLOCK_STONE, BLOCK_STONE, BLOCK_STONE, BLOCK_STONE},
        {BLOCK_COAL, BLOCK_IRON, BLOCK_DIAMOND, BLOCK_SAPPHIRE, BLOCK_RUBY},
        {0.05f, 0.0f, 0.0f, 0.0f, 0.025f},
        BLOCK_FIRE_GRASS, BLOCK_FIRE_TREE, BLOCK_LAVA, BLOCK_FIRE_GRASS, BLOCK_FIRE_GRASS,
        true, true, 5.0f, 1.0f, 5.0f, 1.0f, 1.0f, 0.5f, 0.6f, 0.8f, 0.5f,
    },
    {
        BLOCK_ICE_GRASS, BLOCK_WATER, BLOCK_ICE_GRASS, BLOCK_STONE, BLOCK_STONE, BLOCK_STONE,
        {BLOCK_STONE, BLOCK_STONE, BLOCK_STONE, BLOCK_STONE, BLOCK_STONE},
        {BLOCK_COAL, BLOCK_IRON, BLOCK_DIAMOND, BLOCK_SAPPHIRE, BLOCK_RUBY},
        {0.0f, 0.0f, 0.005f, 0.02f, 0.0f},
        BLOCK_ICE_GRASS, BLOCK_ICE_SHRUB, BLOCK_WATER, BLOCK_ICE_GRASS, BLOCK_ICE_GRASS,
        true, true, 5.0f, 1.0f, 17.0f, 1.5f, 0.0f, 0.5f, 0.6f, 0.4f, 0.5f,
    },
    {
        BLOCK_PATH, BLOCK_PATH, BLOCK_PATH, BLOCK_WALL, BLOCK_WALL, BLOCK_WALL,
        {BLOCK_WALL, BLOCK_GRAVE, BLOCK_GRAVE, BLOCK_WALL, BLOCK_WALL},
        {BLOCK_WALL_MOSS, BLOCK_GRAVE2, BLOCK_GRAVE3, BLOCK_SAPPHIRE, BLOCK_RUBY},
        {0.1f, 0.333f, 0.5f, 0.0f, 0.0f},
        BLOCK_PATH, BLOCK_GRAVE, BLOCK_WALL, BLOCK_NECROMANCER, BLOCK_PATH,
        false, false, 5.0f, 1.0f, 10.0f, 10.0f, 0.0f, 0.7f, 0.6f, 0.95f, -1.0f,
    },
};

static const DungeonConfig DUNGEON_LEVEL_CONFIGS[3] = {
    {BLOCK_PATH, BLOCK_FOUNTAIN, BLOCK_PATH},
    {BLOCK_ENCHANTMENT_TABLE_ICE, BLOCK_WATER, BLOCK_WATER},
    {BLOCK_ENCHANTMENT_TABLE_FIRE, BLOCK_FOUNTAIN, BLOCK_PATH},
};


// Rendering parameters. textures.png is 16x16 RGBA tiles, row-major, 16 columns:
//   [0..36]  blocks  [37..41] player  [42..46] items  [47..49] generic mobs
//   [50..53] arrows  [54..61] armour  [62..70] tools  [71..76] potions
//   [77..79] HUD     [80..87] melee   [88..90] passive [91..98] ranged
//   [99..102] projectiles
//   [103..104] sword enchant  [105..106] arrow enchant
//   [107..110] armour fire overlay  [111..114] armour ice overlay
#define TEX_TILE_PX 16
#define TEX_SHEET_COLS 16
#define TEX_SCALE 3
#define TEX_DRAW_PX (TEX_TILE_PX * TEX_SCALE)
#define TEX_NUM (37 + 5 + 5 + 3 + 4 + 8 + 9 + 6 + 3 + 8 + 3 + 8 + 4 + 12)
#define RENDER_ROWS 14
#define RENDER_COLS 16
#define ACTION_PANEL_W 280
#define ACH_PANEL_W 196
#define OBS_PANEL_W 280
#define TEX_PLAYER_DOWN 37
#define TEX_PLAYER_UP 38
#define TEX_PLAYER_LEFT 39
#define TEX_PLAYER_RIGHT 40
#define TEX_PLAYER_SLEEP 41
#define TEX_ITEM_BASE 42
#define TEX_MOB_ZOMBIE 47
#define TEX_MOB_SKELETON 48
#define TEX_MOB_COW 49
#define TEX_ARROW_DOWN 50
#define TEX_ARROW_UP 51
#define TEX_ARROW_LEFT 52
#define TEX_ARROW_RIGHT 53
#define TEX_ARMOUR_IRON 54
#define TEX_ARMOUR_DIAMOND 58
#define TEX_PICKAXE_WOOD 62
#define TEX_SWORD_WOOD 66
#define TEX_BOW 70
#define TEX_POTION 71
#define TEX_SAPLING 77
#define TEX_TORCH_INV 78
#define TEX_BOOK 79
#define TEX_MELEE 80
#define TEX_PASSIVE 88
#define TEX_RANGED 91
#define TEX_PROJ_DAGGER 99
#define TEX_PROJ_FIREBALL 100
#define TEX_PROJ_ICEBALL 101
#define TEX_PROJ_SLIMEBALL 102
#define TEX_SWORD_ENCHANT_FIRE 103
#define TEX_SWORD_ENCHANT_ICE 104
#define TEX_ARROW_ENCHANT_FIRE 105
#define TEX_ARROW_ENCHANT_ICE 106
#define TEX_ARMOUR_ENCHANT_FIRE 107
#define TEX_ARMOUR_ENCHANT_ICE 111
