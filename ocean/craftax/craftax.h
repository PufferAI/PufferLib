// Full Craftax environment for PufferLib Ocean.
//
// This file intentionally starts as a reference-backed C env: reset/step call
// the installed JAX Craftax-Symbolic-v1 implementation through the Python C
// API and copy the resulting float32 observation/reward/done into PufferLib's
// buffers. The native C state layout and enum constants are declared here so
// the JAX logic can be replaced subsystem-by-subsystem without changing the
// Ocean ABI.

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <dlfcn.h>
#include <sys/types.h>

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
    void* py_proxy;

    float achievements[CRAFTAX_NUM_ACHIEVEMENTS];
    float episode_return_accum;
    int32_t episode_length_accum;
} Craftax;

// ============================================================
// Minimal dynamic Python C API loader
// ============================================================
typedef struct _object PyObject;
typedef int PyGILState_STATE;
typedef ssize_t Py_ssize_t;

typedef struct CraftaxPyApi {
    bool loaded;
    PyGILState_STATE (*PyGILState_Ensure)(void);
    void (*PyGILState_Release)(PyGILState_STATE);
    int (*PyRun_SimpleString)(const char*);
    PyObject* (*PyImport_AddModule)(const char*);
    PyObject* (*PyObject_GetAttrString)(PyObject*, const char*);
    PyObject* (*PyObject_CallFunctionObjArgs)(PyObject*, ...);
    PyObject* (*PyObject_CallMethod)(PyObject*, const char*, const char*, ...);
    PyObject* (*PyLong_FromUnsignedLongLong)(unsigned long long);
    double (*PyFloat_AsDouble)(PyObject*);
    int (*PyObject_IsTrue)(PyObject*);
    Py_ssize_t (*PyTuple_Size)(PyObject*);
    PyObject* (*PyTuple_GetItem)(PyObject*, Py_ssize_t);
    int (*PyBytes_AsStringAndSize)(PyObject*, char**, Py_ssize_t*);
    PyObject* (*PyErr_Occurred)(void);
    void (*PyErr_Print)(void);
    void (*Py_DecRef)(PyObject*);
} CraftaxPyApi;

static CraftaxPyApi craftax_py_api;
static bool craftax_proxy_code_loaded = false;

static void* craftax_py_sym(const char* name) {
    void* sym = dlsym(RTLD_DEFAULT, name);
    if (sym == NULL) {
        fprintf(stderr, "craftax: failed to resolve Python symbol %s\n", name);
        abort();
    }
    return sym;
}

static void craftax_py_load_api(void) {
    if (craftax_py_api.loaded) {
        return;
    }

    craftax_py_api.PyGILState_Ensure = (PyGILState_STATE (*)(void))craftax_py_sym("PyGILState_Ensure");
    craftax_py_api.PyGILState_Release = (void (*)(PyGILState_STATE))craftax_py_sym("PyGILState_Release");
    craftax_py_api.PyRun_SimpleString = (int (*)(const char*))craftax_py_sym("PyRun_SimpleString");
    craftax_py_api.PyImport_AddModule = (PyObject* (*)(const char*))craftax_py_sym("PyImport_AddModule");
    craftax_py_api.PyObject_GetAttrString = (PyObject* (*)(PyObject*, const char*))craftax_py_sym("PyObject_GetAttrString");
    craftax_py_api.PyObject_CallFunctionObjArgs = (PyObject* (*)(PyObject*, ...))craftax_py_sym("PyObject_CallFunctionObjArgs");
    craftax_py_api.PyObject_CallMethod = (PyObject* (*)(PyObject*, const char*, const char*, ...))craftax_py_sym("PyObject_CallMethod");
    craftax_py_api.PyLong_FromUnsignedLongLong = (PyObject* (*)(unsigned long long))craftax_py_sym("PyLong_FromUnsignedLongLong");
    craftax_py_api.PyFloat_AsDouble = (double (*)(PyObject*))craftax_py_sym("PyFloat_AsDouble");
    craftax_py_api.PyObject_IsTrue = (int (*)(PyObject*))craftax_py_sym("PyObject_IsTrue");
    craftax_py_api.PyTuple_Size = (Py_ssize_t (*)(PyObject*))craftax_py_sym("PyTuple_Size");
    craftax_py_api.PyTuple_GetItem = (PyObject* (*)(PyObject*, Py_ssize_t))craftax_py_sym("PyTuple_GetItem");
    craftax_py_api.PyBytes_AsStringAndSize = (int (*)(PyObject*, char**, Py_ssize_t*))craftax_py_sym("PyBytes_AsStringAndSize");
    craftax_py_api.PyErr_Occurred = (PyObject* (*)(void))craftax_py_sym("PyErr_Occurred");
    craftax_py_api.PyErr_Print = (void (*)(void))craftax_py_sym("PyErr_Print");
    craftax_py_api.Py_DecRef = (void (*)(PyObject*))craftax_py_sym("Py_DecRef");
    craftax_py_api.loaded = true;
}

static void craftax_py_print_error(void) {
    if (craftax_py_api.PyErr_Occurred != NULL && craftax_py_api.PyErr_Occurred()) {
        craftax_py_api.PyErr_Print();
    }
}

static void craftax_zero_obs(Craftax* env) {
    if (env->observations != NULL) {
        memset(env->observations, 0, CRAFTAX_OBS_SIZE * sizeof(float));
    }
}

static bool craftax_copy_bytes_to_float_buffer(PyObject* bytes, float* dst, int count) {
    char* data = NULL;
    Py_ssize_t size = 0;
    if (craftax_py_api.PyBytes_AsStringAndSize(bytes, &data, &size) != 0) {
        craftax_py_print_error();
        return false;
    }
    Py_ssize_t expected = (Py_ssize_t)count * (Py_ssize_t)sizeof(float);
    if (size != expected) {
        fprintf(stderr, "craftax: Python helper returned %zd bytes, expected %zd\n",
            (ssize_t)size, (ssize_t)expected);
        return false;
    }
    memcpy(dst, data, (size_t)expected);
    return true;
}

static void craftax_py_define_proxy(void) {
    if (craftax_proxy_code_loaded) {
        return;
    }

    const char* code =
        "import os\n"
        "os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')\n"
        "os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')\n"
        "class _CraftaxOceanProxy:\n"
        "    def __init__(self, seed):\n"
        "        import jax\n"
        "        import numpy as np\n"
        "        from craftax.craftax_env import make_craftax_env_from_name\n"
        "        from craftax.craftax.constants import Achievement\n"
        "        self.jax = jax\n"
        "        self.np = np\n"
        "        self.seed = int(seed)\n"
        "        global _CRAFTAX_OCEAN_ENV\n"
        "        try:\n"
        "            env = _CRAFTAX_OCEAN_ENV\n"
        "        except NameError:\n"
        "            env = None\n"
        "        if env is None:\n"
        "            env = make_craftax_env_from_name('Craftax-Symbolic-v1', auto_reset=True)\n"
        "            _CRAFTAX_OCEAN_ENV = env\n"
        "        self.env = env\n"
        "        self.params = self.env.default_params\n"
        "        max_achievement = max(a.value for a in Achievement) + 1\n"
        "        self.achievement_info_names = [None] * max_achievement\n"
        "        for achievement in Achievement:\n"
        "            self.achievement_info_names[achievement.value] = 'Achievements/' + achievement.name.lower()\n"
        "        self.rng = None\n"
        "        self.state = None\n"
        "        self.obs = None\n"
        "    def _pack_obs(self, obs):\n"
        "        arr = self.np.asarray(obs, dtype=self.np.float32).reshape(-1)\n"
        "        if arr.size != 8268:\n"
        "            raise RuntimeError(f'Craftax obs has {arr.size} floats, expected 8268')\n"
        "        return arr.tobytes()\n"
        "    def _pack_achievements(self, info=None, done=False):\n"
        "        if done and info is not None:\n"
        "            values = [float(info.get(name, 0.0)) / 100.0 for name in self.achievement_info_names]\n"
        "            arr = self.np.asarray(values, dtype=self.np.float32)\n"
        "        else:\n"
        "            arr = self.np.asarray(self.state.achievements, dtype=self.np.float32).reshape(-1)\n"
        "        return arr.tobytes()\n"
        "    def reset(self):\n"
        "        self.rng = self.jax.random.PRNGKey(self.seed)\n"
        "        self.rng, reset_key = self.jax.random.split(self.rng)\n"
        "        self.obs, self.state = self.env.reset(reset_key, self.params)\n"
        "        return self._pack_obs(self.obs)\n"
        "    def step(self, action):\n"
        "        self.rng, step_key = self.jax.random.split(self.rng)\n"
        "        self.obs, self.state, reward, done, info = self.env.step(step_key, self.state, int(action), self.params)\n"
        "        done_bool = bool(done)\n"
        "        return (self._pack_obs(self.obs), float(reward), done_bool, self._pack_achievements(info, done_bool))\n"
        "    def close(self):\n"
        "        try:\n"
        "            self.jax.effects_barrier()\n"
        "        except Exception:\n"
        "            pass\n"
        "        self.state = None\n"
        "        self.obs = None\n"
        "        self.env = None\n"
        "        global _CRAFTAX_OCEAN_ENV\n"
        "        _CRAFTAX_OCEAN_ENV = None\n";

    if (craftax_py_api.PyRun_SimpleString(code) != 0) {
        craftax_py_print_error();
        abort();
    }
    craftax_proxy_code_loaded = true;
}

static bool craftax_ensure_proxy(Craftax* env) {
    if (env->py_proxy != NULL) {
        return true;
    }

    craftax_py_load_api();
    craftax_py_define_proxy();

    PyObject* main_mod = craftax_py_api.PyImport_AddModule("__main__");
    if (main_mod == NULL) {
        craftax_py_print_error();
        return false;
    }

    PyObject* cls = craftax_py_api.PyObject_GetAttrString(main_mod, "_CraftaxOceanProxy");
    if (cls == NULL) {
        craftax_py_print_error();
        return false;
    }

    PyObject* seed = craftax_py_api.PyLong_FromUnsignedLongLong((unsigned long long)env->seed);
    if (seed == NULL) {
        craftax_py_api.Py_DecRef(cls);
        craftax_py_print_error();
        return false;
    }

    env->py_proxy = craftax_py_api.PyObject_CallFunctionObjArgs(cls, seed, NULL);
    craftax_py_api.Py_DecRef(seed);
    craftax_py_api.Py_DecRef(cls);
    if (env->py_proxy == NULL) {
        craftax_py_print_error();
        return false;
    }
    return true;
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

// ============================================================
// Public API expected by vecenv.h
// ============================================================
static void c_init(Craftax* env) {
    env->client = NULL;
    env->num_agents = 1;
    env->py_proxy = NULL;
    env->episode_return_accum = 0.0f;
    env->episode_length_accum = 0;
    memset(env->achievements, 0, sizeof(env->achievements));
    memset(&env->log, 0, sizeof(env->log));
}

static void c_reset(Craftax* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    env->episode_return_accum = 0.0f;
    env->episode_length_accum = 0;
    memset(env->achievements, 0, sizeof(env->achievements));

    craftax_py_load_api();
    PyGILState_STATE gil = craftax_py_api.PyGILState_Ensure();
    if (!craftax_ensure_proxy(env)) {
        craftax_zero_obs(env);
        craftax_py_api.PyGILState_Release(gil);
        return;
    }

    PyObject* obs_bytes = craftax_py_api.PyObject_CallMethod((PyObject*)env->py_proxy, "reset", NULL);
    if (obs_bytes == NULL) {
        craftax_py_print_error();
        craftax_zero_obs(env);
        craftax_py_api.PyGILState_Release(gil);
        return;
    }

    if (!craftax_copy_bytes_to_float_buffer(obs_bytes, env->observations, CRAFTAX_OBS_SIZE)) {
        craftax_zero_obs(env);
    }
    craftax_py_api.Py_DecRef(obs_bytes);
    craftax_py_api.PyGILState_Release(gil);
}

static void c_step(Craftax* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    int action = (int)env->actions[0];
    if (action < 0) {
        action = CRAFTAX_ACTION_NOOP;
    }
    if (action >= CRAFTAX_NUM_ACTIONS) {
        action = CRAFTAX_NUM_ACTIONS - 1;
    }

    craftax_py_load_api();
    PyGILState_STATE gil = craftax_py_api.PyGILState_Ensure();
    if (!craftax_ensure_proxy(env)) {
        craftax_zero_obs(env);
        craftax_py_api.PyGILState_Release(gil);
        return;
    }

    PyObject* result = craftax_py_api.PyObject_CallMethod((PyObject*)env->py_proxy, "step", "i", action);
    if (result == NULL) {
        craftax_py_print_error();
        craftax_zero_obs(env);
        craftax_py_api.PyGILState_Release(gil);
        return;
    }

    bool ok = true;
    if (craftax_py_api.PyTuple_Size(result) != 4) {
        fprintf(stderr, "craftax: Python helper step did not return a 4-tuple\n");
        ok = false;
    }

    float reward = 0.0f;
    int done = 0;
    if (ok) {
        PyObject* obs_bytes = craftax_py_api.PyTuple_GetItem(result, 0);
        PyObject* reward_obj = craftax_py_api.PyTuple_GetItem(result, 1);
        PyObject* done_obj = craftax_py_api.PyTuple_GetItem(result, 2);
        PyObject* ach_bytes = craftax_py_api.PyTuple_GetItem(result, 3);

        ok = craftax_copy_bytes_to_float_buffer(obs_bytes, env->observations, CRAFTAX_OBS_SIZE);
        if (ok) {
            reward = (float)craftax_py_api.PyFloat_AsDouble(reward_obj);
            if (craftax_py_api.PyErr_Occurred()) {
                craftax_py_print_error();
                reward = 0.0f;
                ok = false;
            }
        }
        if (ok) {
            done = craftax_py_api.PyObject_IsTrue(done_obj);
            if (done < 0) {
                craftax_py_print_error();
                done = 0;
                ok = false;
            }
        }
        if (ok) {
            ok = craftax_copy_bytes_to_float_buffer(ach_bytes, env->achievements, CRAFTAX_NUM_ACHIEVEMENTS);
        }
    }

    if (!ok) {
        craftax_zero_obs(env);
        reward = 0.0f;
        done = 1;
    }

    craftax_py_api.Py_DecRef(result);
    craftax_py_api.PyGILState_Release(gil);

    env->rewards[0] = reward;
    env->terminals[0] = done ? 1.0f : 0.0f;
    env->episode_return_accum += reward;
    env->episode_length_accum += 1;

    if (done) {
        add_log(env);
        env->episode_return_accum = 0.0f;
        env->episode_length_accum = 0;
        memset(env->achievements, 0, sizeof(env->achievements));
    }
}

static void c_close(Craftax* env) {
    if (env->py_proxy == NULL) {
        return;
    }

    craftax_py_load_api();
    PyGILState_STATE gil = craftax_py_api.PyGILState_Ensure();
    PyObject* result = craftax_py_api.PyObject_CallMethod((PyObject*)env->py_proxy, "close", NULL);
    if (result == NULL) {
        craftax_py_print_error();
    } else {
        craftax_py_api.Py_DecRef(result);
    }
    craftax_py_api.PyGILState_Release(gil);

    // The reference proxy owns JAX objects with process-level runtime state.
    // DECREFing the wrapper itself during PufferLib shutdown can race XLA
    // cleanup and segfault. The native port will remove this path entirely.
    env->py_proxy = NULL;
}

static void c_render(Craftax* env) {
    (void)env;
}
