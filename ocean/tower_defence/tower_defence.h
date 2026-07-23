#pragma once

#include "raylib.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TD_DEFAULT_MAX_EPISODE_STEPS 25000
#define TD_DEFAULT_BASE_SEED 1
#define TD_DEFAULT_TARGET_ROUND 500
#define TD_DEFAULT_INVALID_ACTION_REWARD (-1.0f)
#define TD_DEFAULT_REWARD_ENEMY_SCALE 0.5f
#define TD_DEFAULT_REWARD_ROUND_CLEAR_SCALE 0.5f
#define TD_DEFAULT_REWARD_TOWER_PLACEMENT_SCALE 0.0f
#define TD_DEFAULT_REWARD_TOWER_INVESTMENT_SCALE 0.0f
#define TD_DEFAULT_REWARD_SELL_PENALTY_SCALE 0.0f
#define TD_DEFAULT_REWARD_TRIGGER_READY_NOOP_PENALTY 0.0f
#define TD_DEFAULT_REWARD_ROUND_ADVANCE_SCALE 25.0f
#define TD_DEFAULT_REWARD_SCORE_ADVANCE_SCALE 0.0f
#define TD_DEFAULT_REWARD_LEAK_PENALTY_SCALE 1.0f
#define TD_DEFAULT_REWARD_COMPLETION_BONUS 50.0f
#define TD_DEFAULT_REWARD_DEFEAT_PENALTY 25.0f
#define TD_DEFAULT_REWARD_TRUNCATION_PENALTY 10.0f
#define TD_DEFAULT_REWARD_CLAMP_ENABLED 0
#define TD_DEFAULT_REWARD_CLAMP_MIN (-1.0f)
#define TD_DEFAULT_REWARD_CLAMP_MAX 1.0f
#define TD_WIDTH 960
#define TD_HEIGHT 540
#define TD_HUD_HEIGHT 110
#define TD_NUM_TOWER_TYPES 3
#define TD_BUILD_GRID_COLS 30
#define TD_BUILD_GRID_ROWS 17
#define TD_NUM_PLACEMENT_SLOTS (TD_BUILD_GRID_COLS * TD_BUILD_GRID_ROWS)
#define TD_BUILD_GRID_ORIGIN_X 16.0f
#define TD_BUILD_GRID_ORIGIN_Y 16.0f
#define TD_BUILD_GRID_STEP_X 32.0f
#define TD_BUILD_GRID_STEP_Y 31.75f
#define TD_BUILD_PATH_CLEARANCE 24.0f
#define TD_NUM_UPGRADE_PATHS 3
#define TD_NUM_SLOT_FEATURES_PER_SLOT 11
#define TD_NUM_SLOT_FEATURES (TD_NUM_PLACEMENT_SLOTS * TD_NUM_SLOT_FEATURES_PER_SLOT)
#define TD_NUM_ENEMY_PROGRESS_BINS 8
#define TD_NUM_ENEMY_PROGRESS_FEATURES_PER_BIN 2
#define TD_NUM_ENEMY_PROGRESS_FEATURES                                                             \
    (TD_NUM_ENEMY_PROGRESS_BINS * TD_NUM_ENEMY_PROGRESS_FEATURES_PER_BIN)
#define TD_NUM_PLACEMENT_ACTIONS (TD_NUM_PLACEMENT_SLOTS * TD_NUM_TOWER_TYPES)
#define TD_NUM_UPGRADE_ACTIONS (TD_NUM_PLACEMENT_SLOTS * TD_NUM_UPGRADE_PATHS)
#define TD_NUM_SELL_ACTIONS TD_NUM_PLACEMENT_SLOTS
#define TD_SCALAR_OBS_SIZE 16
#define TD_NUM_ACTIONS                                                                             \
    (1 + TD_NUM_PLACEMENT_ACTIONS + TD_NUM_UPGRADE_ACTIONS + TD_NUM_SELL_ACTIONS + 1)
#define TD_BASE_OBS_SIZE                                                                           \
    (TD_SCALAR_OBS_SIZE + TD_NUM_SLOT_FEATURES + TD_NUM_ENEMY_PROGRESS_FEATURES)
#define TD_NUM_GLOBAL_OBSERVATION_FEATURES 44
#define TD_GLOBAL_OBSERVATION_OFFSET TD_BASE_OBS_SIZE
#define TD_OBS_SIZE (TD_GLOBAL_OBSERVATION_OFFSET + TD_NUM_GLOBAL_OBSERVATION_FEATURES)
#define TD_MAX_SPAWNS 4
#define TD_MAX_TIER 4
#define TD_REPEAT_UPGRADE_UNLOCK_ROUND 31
#define TD_MAX_ACTIVE_PATHS 2
#define TD_DT 0.25f
#define TD_PATH_LENGTH 1311.0
#define TD_PROGRESS_BIN_EPSILON 1e-8
#define TD_FORTIFIED_HP_MULTIPLIER 1.8f
#define TD_FORTIFIED_REWARD_MULTIPLIER 1.35f
#define TD_RECURSIVE_ENTITY_CAP 950
#define TD_MAX_ENEMIES TD_RECURSIVE_ENTITY_CAP
#define TD_INITIAL_PROJECTILE_CAPACITY 64
#define TD_MAX_SHOT_EVENTS 8
#define TD_MAX_IMPACT_EVENTS 64
#define TD_TOWER_MASK_WORDS ((TD_NUM_PLACEMENT_SLOTS + 63) / 64)

#ifndef TD_RESOURCE_DIR
#define TD_RESOURCE_DIR "resources/tower_defence"
#endif

enum TdActionId {
    TD_ACTION_NOOP = 0,
    TD_ACTION_PLACE_FIRST = 1,
    TD_ACTION_UPGRADE_SLOT_01_TOP = 1 + TD_NUM_PLACEMENT_ACTIONS,
    TD_ACTION_SELL_SLOT_01 = 1 + TD_NUM_PLACEMENT_ACTIONS + TD_NUM_UPGRADE_ACTIONS,
    TD_ACTION_TRIGGER_NEXT_ROUND =
        1 + TD_NUM_PLACEMENT_ACTIONS + TD_NUM_UPGRADE_ACTIONS + TD_NUM_SELL_ACTIONS,
};

enum TdStatusCode {
    TD_STATUS_WARMUP = 0,
    TD_STATUS_SPAWNING = 1,
    TD_STATUS_INTERMISSION = 2,
    TD_STATUS_ACTIVE = 3,
    TD_STATUS_DEFEAT = 4,
    TD_STATUS_COMPLETE = 5,
};

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float invalid_action_rate;
    float n;
} Log;

typedef struct {
    double x;
    double y;
} TdVec2d;

typedef struct TdClient TdClient;

typedef struct {
    int alive;
    int kind;
    int upgrades[TD_NUM_UPGRADE_PATHS];
    float cooldown;
    float invested;
} TdTower;

typedef struct {
    int alive;
    int id;
    int type;
    int camo;
    int fortified;
    int regrow;
    double distance;
    double progress;
    float hp;
    float max_hp;
    float speed;
    float burn_dps;
    float burn_time;
    float slow_mult;
    float slow_time;
    float regrow_cooldown_until;
} TdEnemy;

typedef struct {
    int alive;
    int source_tower;
    int kind;
    int target_enemy;
    int target_enemy_id;
    float x;
    float y;
    float previous_x;
    float previous_y;
    float damage;
    int damage_type;
    float speed;
    float burn_dps;
    float burn_time;
    float slow_mult;
    float slow_time;
} TdProjectile;

typedef struct {
    uint64_t serial;
    float step_time;
    uint64_t fired_towers[TD_TOWER_MASK_WORDS];
} TdShotEvent;

typedef struct {
    uint64_t serial;
    float step_time;
    float from_x;
    float from_y;
    float x;
    float y;
    int kind;
} TdImpactEvent;

typedef struct {
    int type;
    int count;
    int emitted;
    double interval;
    double start_time;
    double next_time;
    int camo;
    int fortified;
    int regrow;
    int use_modifier_chances;
    float camo_chance;
    float fortified_chance;
    float regrow_chance;
    unsigned int modifier_rng_state;
} TdSpawn;

typedef struct {
    int round;
    int type;
    int count;
    double interval;
    double delay;
} TdFixedSpawnSpec;

typedef struct {
    Log log;
    float *observations;
    float *actions;
    float *rewards;
    float *terminals;
    unsigned char *action_mask;
    TdClient *client;
    int num_agents;
    int rng;

    int max_episode_steps;
    int base_seed;
    int target_round;
    unsigned int episode_index;
    int step_count;
    int invalid_action_count;
    int max_tower_count;
    float invalid_action_reward;
    float reward_enemy_scale;
    float reward_round_clear_scale;
    float reward_tower_placement_scale;
    float reward_tower_investment_scale;
    float reward_sell_penalty_scale;
    float reward_trigger_ready_noop_penalty;
    float reward_round_advance_scale;
    float reward_score_advance_scale;
    float reward_leak_penalty_scale;
    float reward_completion_bonus;
    float reward_defeat_penalty;
    float reward_truncation_penalty;
    int reward_clamp_enabled;
    float reward_clamp_min;
    float reward_clamp_max;
    float max_total_invested;
    float episode_return;

    unsigned int rng_state;
    int next_enemy_id;
    int round_pending_advance;
    int round;
    int status_code;
    float time;
    float lives;
    float cash;
    float intermission_remaining;
    float wave_elapsed;
    float score;
    int active_spawns;
    TdSpawn spawns[TD_MAX_SPAWNS];
    TdTower towers[TD_NUM_PLACEMENT_SLOTS];
    TdEnemy enemies[TD_MAX_ENEMIES];
    int enemy_high_water;
    TdProjectile *projectiles;
    int projectile_count;
    int projectile_capacity;
    TdShotEvent shot_events[TD_MAX_SHOT_EVENTS];
    TdImpactEvent impact_events[TD_MAX_IMPACT_EVENTS];
    uint64_t shot_serial;
    uint64_t impact_serial;
} TowerDefence;

static const float TD_PATH_X[] = {0, 384, 384, 672, 672, 960};
static const float TD_PATH_Y[] = {108, 108, 270, 270, 459, 459};
static const double TD_PATH_CUMULATIVE[] = {0, 384, 546, 834, 1023, 1311};
static const float TD_TOWER_COST[] = {100, 260, 420};
static const float TD_TOWER_RANGE[] = {140, 500, 160};
static const float TD_TOWER_FIRE_RATE[] = {0.7f, 1.35f, 1.6f};
static const float TD_TOWER_PROJECTILE_SPEED[] = {260, 520, 180};
static const float TD_TOWER_DAMAGE[] = {1, 2, 3};
static const int TD_TOWER_DAMAGE_TYPE[] = {0, 0, 1};
static const int TD_TOWER_RADIUS[] = {14, 12, 16};
static const int TD_UPGRADE_COST[3][3] = {
    {90, 105, 120},
    {180, 210, 195},
    {240, 200, 225},
};
static const float TD_ENEMY_HP[] = {1, 2, 3, 4, 5, 6, 6, 8, 7, 12};
static const float TD_ENEMY_SPEED[] = {60, 70, 80, 95, 115, 85, 85, 60, 90, 65};
static const float TD_ENEMY_REWARD[] = {1, 1, 1, 1, 1, 1, 1, 2, 2, 4};
static const float TD_ENEMY_LEAK[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 5};
static const float TD_ENEMY_RADIUS[] = {7, 7.5f, 8, 8.5f, 8.5f, 8.5f, 8.5f, 9, 9, 11};
static const int TD_ENEMY_SHARP_IMMUNE[] = {0, 0, 0, 0, 0, 0, 0, 1, 0, 0};
static const int TD_ENEMY_EXPLOSIVE_IMMUNE[] = {0, 0, 0, 0, 0, 1, 0, 0, 1, 0};
static const int TD_ENEMY_BURN_IMMUNE[] = {0, 0, 0, 0, 0, 1, 0, 0, 1, 0};
static const int TD_ENEMY_SLOW_IMMUNE[] = {0, 0, 0, 0, 0, 0, 1, 0, 1, 0};
static const int TD_ENEMY_CHILD_A[] = {-1, 0, 1, 2, 3, 4, 4, 5, 5, 8};
static const int TD_ENEMY_CHILD_B[] = {-1, -1, -1, -1, -1, -1, -1, -1, 6, 8};
static const float TD_ENEMY_RECURSIVE_RBE[] = {1, 3, 6, 10, 15, 21, 21, 29, 49, 110};
static const int TD_ENEMY_RECURSIVE_COUNT[] = {1, 2, 3, 4, 5, 6, 6, 7, 13, 27};
static unsigned char td_site_buildable_cache[TD_NUM_TOWER_TYPES][TD_NUM_PLACEMENT_SLOTS];
static int td_site_buildable_cache_ready;
static int td_render_client_count;

static float td_site_x(int site) {
    return TD_BUILD_GRID_ORIGIN_X + (float)(site % TD_BUILD_GRID_COLS) * TD_BUILD_GRID_STEP_X;
}

static float td_site_y(int site) {
    return TD_BUILD_GRID_ORIGIN_Y + (float)(site / TD_BUILD_GRID_COLS) * TD_BUILD_GRID_STEP_Y;
}

static float td_distance_to_segment(float px, float py, float ax, float ay, float bx, float by) {
    float vx = bx - ax;
    float vy = by - ay;
    float wx = px - ax;
    float wy = py - ay;
    float len2 = vx * vx + vy * vy;
    float t = len2 > 0.0f ? (wx * vx + wy * vy) / len2 : 0.0f;
    t = fminf(1.0f, fmaxf(0.0f, t));
    float dx = px - (ax + t * vx);
    float dy = py - (ay + t * vy);
    return sqrtf(dx * dx + dy * dy);
}

static int td_site_clear_with_clearance(int site, float clearance) {
    if (site < 0 || site >= TD_NUM_PLACEMENT_SLOTS) {
        return 0;
    }
    float x = td_site_x(site);
    float y = td_site_y(site);
    for (int i = 0; i < 5; i++) {
        if (td_distance_to_segment(x, y, TD_PATH_X[i], TD_PATH_Y[i], TD_PATH_X[i + 1],
                                   TD_PATH_Y[i + 1]) < clearance) {
            return 0;
        }
    }
    return 1;
}

static void td_init_static_geometry(void) {
    if (td_site_buildable_cache_ready) {
        return;
    }
    for (int kind = 0; kind < TD_NUM_TOWER_TYPES; kind++) {
        float clearance = TD_BUILD_PATH_CLEARANCE + (float)TD_TOWER_RADIUS[kind];
        for (int site = 0; site < TD_NUM_PLACEMENT_SLOTS; site++) {
            td_site_buildable_cache[kind][site] =
                (unsigned char)td_site_clear_with_clearance(site, clearance);
        }
    }
    td_site_buildable_cache_ready = 1;
}

static int td_site_buildable_for_kind(int site, int kind) {
    if (site < 0 || site >= TD_NUM_PLACEMENT_SLOTS || kind < 0 || kind >= TD_NUM_TOWER_TYPES) {
        return 0;
    }
    return td_site_buildable_cache[kind][site];
}

static int td_site_buildable(int site) {
    for (int kind = 0; kind < TD_NUM_TOWER_TYPES; kind++) {
        if (td_site_buildable_for_kind(site, kind)) {
            return 1;
        }
    }
    return 0;
}

static int td_site_clear_of_towers(const TowerDefence *env, int site, int kind) {
    if (site < 0 || site >= TD_NUM_PLACEMENT_SLOTS || kind < 0 || kind >= TD_NUM_TOWER_TYPES) {
        return 0;
    }
    int row = site / TD_BUILD_GRID_COLS;
    int col = site % TD_BUILD_GRID_COLS;
    float x = td_site_x(site);
    float y = td_site_y(site);
    for (int dr = -1; dr <= 1; dr++) {
        int other_row = row + dr;
        if (other_row < 0 || other_row >= TD_BUILD_GRID_ROWS) {
            continue;
        }
        for (int dc = -1; dc <= 1; dc++) {
            int other_col = col + dc;
            if ((dr == 0 && dc == 0) || other_col < 0 || other_col >= TD_BUILD_GRID_COLS) {
                continue;
            }
            const TdTower *tower = &env->towers[other_row * TD_BUILD_GRID_COLS + other_col];
            if (!tower->alive || tower->kind < 0 || tower->kind >= TD_NUM_TOWER_TYPES) {
                continue;
            }
            float min_dist = (float)(TD_TOWER_RADIUS[kind] + TD_TOWER_RADIUS[tower->kind]);
            float dx = x - td_site_x(other_row * TD_BUILD_GRID_COLS + other_col);
            float dy = y - td_site_y(other_row * TD_BUILD_GRID_COLS + other_col);
            if (dx * dx + dy * dy < min_dist * min_dist) {
                return 0;
            }
        }
    }
    return 1;
}

// clang-format off
static const TdFixedSpawnSpec TD_FIXED_SPAWNS[] = {
    {1, 0, 20, 0.6, 0.0},
    {2, 0, 30, 0.5, 0.0},
    {3, 0, 15, 0.5, 0.0}, {3, 1, 10, 0.55, 1.5},
    {4, 1, 25, 0.5, 0.0},
    {5, 0, 10, 0.5, 0.0}, {5, 1, 18, 0.45, 1.0}, {5, 2, 6, 0.6, 1.0},
    {6, 2, 25, 0.45, 0.0},
    {7, 2, 15, 0.45, 0.0}, {7, 3, 10, 0.45, 1.0},
    {8, 3, 20, 0.4, 0.0},
    {9, 3, 12, 0.4, 0.0}, {9, 4, 10, 0.4, 1.0},
    {10, 5, 8, 0.5, 0.0}, {10, 6, 8, 0.5, 0.5}, {10, 3, 10, 0.35, 1.0},
    {11, 4, 14, 0.35, 0.0}, {11, 5, 10, 0.45, 1.0},
    {12, 4, 12, 0.35, 0.0}, {12, 7, 6, 0.6, 0.75},
    {13, 5, 8, 0.45, 0.0}, {13, 8, 3, 0.6, 0.5}, {13, 4, 8, 0.35, 1.0},
    {14, 7, 7, 0.55, 0.0}, {14, 8, 3, 0.6, 0.75},
    {15, 5, 7, 0.45, 0.0}, {15, 6, 7, 0.45, 0.5}, {15, 7, 4, 0.55, 1.0},
    {16, 4, 14, 0.35, 0.0}, {16, 7, 1, 0.55, 0.75},
    {17, 8, 3, 0.55, 0.0}, {17, 4, 8, 0.35, 0.75},
    {18, 7, 2, 0.55, 0.0}, {18, 8, 3, 0.6, 0.75},
    {19, 5, 8, 0.4, 0.0}, {19, 6, 8, 0.4, 0.5},
    {20, 9, 1, 0.0, 0.0}, {20, 7, 6, 0.5, 1.0},
};
// clang-format on

static float td_clampf(float value, float lo, float hi) {
    if (value < lo) {
        return lo;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}

static float td_squash(float value, float scale) {
    return tanhf(fmaxf(0.0f, value) / fmaxf(0.0001f, scale));
}

static float td_lcg_randf(unsigned int *state) {
    *state = *state * 1664525u + 1013904223u;
    return (float)((double)(*state) / 4294967296.0);
}

static unsigned int td_derive_seed(unsigned int base_seed, unsigned int scope) {
    return (base_seed ^ scope) * 1664525u + 1013904223u;
}

static int td_normalize_action(float raw_action) {
    if (!isfinite(raw_action) || raw_action < 0.0f || raw_action >= (float)TD_NUM_ACTIONS ||
        truncf(raw_action) != raw_action) {
        return -1;
    }
    return (int)raw_action;
}

static int td_tower_count(const TowerDefence *env) {
    int count = 0;
    for (int i = 0; i < TD_NUM_PLACEMENT_SLOTS; i++) {
        count += env->towers[i].alive;
    }
    return count;
}

static float td_total_invested(const TowerDefence *env) {
    float total = 0.0f;
    for (int i = 0; i < TD_NUM_PLACEMENT_SLOTS; i++) {
        if (!env->towers[i].alive) {
            continue;
        }
        total += env->towers[i].invested;
    }
    return total;
}

static int td_enemy_count(const TowerDefence *env) {
    int count = 0;
    for (int i = 0; i < env->enemy_high_water; i++) {
        count += env->enemies[i].alive;
    }
    return count;
}

static void td_compact_projectiles(TowerDefence *env) {
    int write = 0;
    int old_count = env->projectile_count;
    for (int read = 0; read < old_count; read++) {
        if (!env->projectiles[read].alive) {
            continue;
        }
        if (write != read) {
            env->projectiles[write] = env->projectiles[read];
        }
        write += 1;
    }
    if (write < old_count) {
        memset(&env->projectiles[write], 0,
               (size_t)(old_count - write) * sizeof(*env->projectiles));
    }
    env->projectile_count = write;
}

static void td_reserve_projectiles(TowerDefence *env, int capacity) {
    if (capacity <= env->projectile_capacity) {
        return;
    }
    TdProjectile *projectiles =
        (TdProjectile *)realloc(env->projectiles, (size_t)capacity * sizeof(*projectiles));
    if (projectiles == NULL) {
        fprintf(stderr, "Failed to allocate %d tower_defence projectiles\n", capacity);
        exit(1);
    }
    memset(&projectiles[env->projectile_capacity], 0,
           (size_t)(capacity - env->projectile_capacity) * sizeof(*projectiles));
    env->projectiles = projectiles;
    env->projectile_capacity = capacity;
}

static TdVec2d td_path_point_at_distance(double distance) {
    distance = fmin(TD_PATH_LENGTH, fmax(0.0, distance));
    for (int i = 0; i < 5; i++) {
        double dx = TD_PATH_X[i + 1] - TD_PATH_X[i];
        double dy = TD_PATH_Y[i + 1] - TD_PATH_Y[i];
        double seg = TD_PATH_CUMULATIVE[i + 1] - TD_PATH_CUMULATIVE[i];
        if (distance <= TD_PATH_CUMULATIVE[i + 1] || i == 4) {
            double local_distance = distance - TD_PATH_CUMULATIVE[i];
            double t = seg > 0.0 ? local_distance / seg : 0.0;
            return (TdVec2d){TD_PATH_X[i] + dx * t, TD_PATH_Y[i] + dy * t};
        }
    }
    return (TdVec2d){TD_PATH_X[5], TD_PATH_Y[5]};
}

static int td_active_paths(const TdTower *tower) {
    int count = 0;
    for (int p = 0; p < TD_NUM_UPGRADE_PATHS; p++) {
        count += tower->upgrades[p] > 0;
    }
    return count;
}

static float td_upgrade_cost(const TdTower *tower, int path) {
    int next = tower->upgrades[path] + 1;
    return (float)(TD_UPGRADE_COST[tower->kind][path] * next * next);
}

static void td_tower_stats(const TdTower *tower, float *range, float *damage, float *fire_rate,
                           float *projectile_speed, int *damage_type, int *detect_camo,
                           float *burn_dps, float *burn_time, float *slow_mult, float *slow_time) {
    int kind = tower->kind;
    *range = TD_TOWER_RANGE[kind];
    *damage = TD_TOWER_DAMAGE[kind];
    *fire_rate = TD_TOWER_FIRE_RATE[kind];
    *projectile_speed = TD_TOWER_PROJECTILE_SPEED[kind];
    *damage_type = TD_TOWER_DAMAGE_TYPE[kind];
    *detect_camo = 0;
    *burn_dps = kind == 2 ? 1.4f : 0.0f;
    *burn_time = kind == 2 ? 2.2f : 0.0f;
    *slow_mult = kind == 1 ? 0.72f : 1.0f;
    *slow_time = kind == 1 ? 1.2f : 0.0f;

    for (int t = 0; t < tower->upgrades[0]; t++) {
        *damage += kind == 0 ? 1.0f : 2.0f;
        if (kind == 1) {
            *slow_mult = fminf(*slow_mult, 0.65f);
            *slow_time += 0.45f;
        }
        if (kind == 2) {
            *burn_dps += 1.0f;
            *burn_time += 0.75f;
        }
    }
    for (int t = 0; t < tower->upgrades[1]; t++) {
        *range += kind == 0 ? 45.0f : (kind == 1 ? 90.0f : 55.0f);
        if (kind == 0) {
            *projectile_speed += 35.0f;
        }
        if (kind != 0) {
            *detect_camo = 1;
        }
    }
    for (int t = 0; t < tower->upgrades[2]; t++) {
        *fire_rate *= kind == 1 ? 0.82f : 0.8f;
        if (kind == 1) {
            *projectile_speed += 70.0f;
        }
        if (kind == 2) {
            *projectile_speed += 45.0f;
        }
    }
}

static int td_can_upgrade(const TowerDefence *env, int slot, int path) {
    if (slot < 0 || slot >= TD_NUM_PLACEMENT_SLOTS || path < 0 || path >= TD_NUM_UPGRADE_PATHS) {
        return 0;
    }
    const TdTower *tower = &env->towers[slot];
    if (!tower->alive || tower->upgrades[path] >= TD_MAX_TIER) {
        return 0;
    }
    if (tower->upgrades[path] > 0 && env->round < TD_REPEAT_UPGRADE_UNLOCK_ROUND) {
        return 0;
    }
    if (tower->upgrades[path] == 0 && td_active_paths(tower) >= TD_MAX_ACTIVE_PATHS) {
        return 0;
    }
    return env->cash >= td_upgrade_cost(tower, path);
}

static int td_can_sell(const TowerDefence *env, int slot) {
    if (slot < 0 || slot >= TD_NUM_PLACEMENT_SLOTS) {
        return 0;
    }
    return env->towers[slot].alive &&
           (env->status_code == TD_STATUS_WARMUP || env->status_code == TD_STATUS_INTERMISSION);
}

#ifdef TD_TEST
static int td_test_mask_update_count;
#endif

static void td_update_masks(TowerDefence *env) {
#ifdef TD_TEST
    td_test_mask_update_count += 1;
#endif
    memset(env->action_mask, 0, TD_NUM_ACTIONS * sizeof(*env->action_mask));
    env->action_mask[TD_ACTION_NOOP] = 1;
    for (int slot = 0; slot < TD_NUM_PLACEMENT_SLOTS; slot++) {
        int slot_open = !env->towers[slot].alive;
        for (int kind = 0; kind < TD_NUM_TOWER_TYPES; kind++) {
            env->action_mask[TD_ACTION_PLACE_FIRST + slot * TD_NUM_TOWER_TYPES + kind] =
                slot_open && td_site_buildable_for_kind(slot, kind) &&
                td_site_clear_of_towers(env, slot, kind) && env->cash >= TD_TOWER_COST[kind];
        }
        env->action_mask[TD_ACTION_SELL_SLOT_01 + slot] = td_can_sell(env, slot);
        for (int path = 0; path < TD_NUM_UPGRADE_PATHS; path++) {
            env->action_mask[TD_ACTION_UPGRADE_SLOT_01_TOP + slot * TD_NUM_UPGRADE_PATHS + path] =
                td_can_upgrade(env, slot, path);
        }
    }
    env->action_mask[TD_ACTION_TRIGGER_NEXT_ROUND] =
        env->status_code == TD_STATUS_WARMUP || env->status_code == TD_STATUS_INTERMISSION;
}

static void td_init(TowerDefence *env) {
    td_init_static_geometry();
    env->num_agents = 1;
    env->max_episode_steps = TD_DEFAULT_MAX_EPISODE_STEPS;
    env->base_seed = TD_DEFAULT_BASE_SEED;
    env->target_round = TD_DEFAULT_TARGET_ROUND;
    env->invalid_action_reward = TD_DEFAULT_INVALID_ACTION_REWARD;
    env->max_tower_count = 0;
    env->max_total_invested = 0.0f;
    env->reward_enemy_scale = TD_DEFAULT_REWARD_ENEMY_SCALE;
    env->reward_round_clear_scale = TD_DEFAULT_REWARD_ROUND_CLEAR_SCALE;
    env->reward_tower_placement_scale = TD_DEFAULT_REWARD_TOWER_PLACEMENT_SCALE;
    env->reward_tower_investment_scale = TD_DEFAULT_REWARD_TOWER_INVESTMENT_SCALE;
    env->reward_sell_penalty_scale = TD_DEFAULT_REWARD_SELL_PENALTY_SCALE;
    env->reward_trigger_ready_noop_penalty = TD_DEFAULT_REWARD_TRIGGER_READY_NOOP_PENALTY;
    env->reward_round_advance_scale = TD_DEFAULT_REWARD_ROUND_ADVANCE_SCALE;
    env->reward_score_advance_scale = TD_DEFAULT_REWARD_SCORE_ADVANCE_SCALE;
    env->reward_leak_penalty_scale = TD_DEFAULT_REWARD_LEAK_PENALTY_SCALE;
    env->reward_completion_bonus = TD_DEFAULT_REWARD_COMPLETION_BONUS;
    env->reward_defeat_penalty = TD_DEFAULT_REWARD_DEFEAT_PENALTY;
    env->reward_truncation_penalty = TD_DEFAULT_REWARD_TRUNCATION_PENALTY;
    env->reward_clamp_enabled = TD_DEFAULT_REWARD_CLAMP_ENABLED;
    env->reward_clamp_min = TD_DEFAULT_REWARD_CLAMP_MIN;
    env->reward_clamp_max = TD_DEFAULT_REWARD_CLAMP_MAX;
    td_reserve_projectiles(env, TD_INITIAL_PROJECTILE_CAPACITY);
}

static void td_write_observation(TowerDefence *env) {
    memset(env->observations, 0, TD_OBS_SIZE * sizeof(float));
    env->observations[0] = env->time;
    env->observations[1] = (float)env->round;
    if (env->status_code >= 0 && env->status_code <= TD_STATUS_COMPLETE) {
        env->observations[2 + env->status_code] = 1.0f;
    }
    env->observations[8] = env->lives;
    env->observations[9] = env->cash;
    env->observations[10] = env->intermission_remaining;
    env->observations[11] = (float)env->round;
    env->observations[12] = env->wave_elapsed;
    env->observations[13] = (float)td_enemy_count(env);
    env->observations[14] = (float)td_tower_count(env);
    env->observations[15] = (float)env->projectile_count;

    int idx = TD_SCALAR_OBS_SIZE;
    for (int slot = 0; slot < TD_NUM_PLACEMENT_SLOTS; slot++) {
        TdTower *tower = &env->towers[slot];
        env->observations[idx++] = td_site_buildable(slot) ? 1.0f : 0.0f;
        env->observations[idx++] = tower->alive ? 1.0f : 0.0f;
        for (int kind = 0; kind < TD_NUM_TOWER_TYPES; kind++) {
            env->observations[idx++] = tower->alive && tower->kind == kind ? 1.0f : 0.0f;
        }
        env->observations[idx++] =
            tower->alive ? fminf(1.0f, tower->upgrades[0] / (float)TD_MAX_TIER) : 0.0f;
        env->observations[idx++] =
            tower->alive ? fminf(1.0f, tower->upgrades[1] / (float)TD_MAX_TIER) : 0.0f;
        env->observations[idx++] =
            tower->alive ? fminf(1.0f, tower->upgrades[2] / (float)TD_MAX_TIER) : 0.0f;
        env->observations[idx++] = td_site_x(slot) / (float)TD_WIDTH;
        env->observations[idx++] = td_site_y(slot) / (float)TD_HEIGHT;
        env->observations[idx++] = tower->alive ? fminf(1.0f, tower->invested / 3000.0f) : 0.0f;
    }

    float count_bins[TD_NUM_ENEMY_PROGRESS_BINS] = {0};
    float hp_bins[TD_NUM_ENEMY_PROGRESS_BINS] = {0};
    float type_mass[10] = {0};
    float prop_mass[8] = {0};
    float band_mass[4][4] = {{0}};
    for (int i = 0; i < env->enemy_high_water; i++) {
        TdEnemy *enemy = &env->enemies[i];
        if (!enemy->alive) {
            continue;
        }
        double progress = enemy->progress + TD_PROGRESS_BIN_EPSILON;
        int bin =
            (int)fmin(TD_NUM_ENEMY_PROGRESS_BINS - 1, floor(progress * TD_NUM_ENEMY_PROGRESS_BINS));
        int band = (int)fmin(3, floor(progress * 4));
        float hp = fmaxf(0.0f, enemy->hp);
        float threat = fmaxf(1.0f, hp);
        float leak = TD_ENEMY_LEAK[enemy->type];
        count_bins[bin] += 1.0f;
        hp_bins[bin] += fminf(1.0f, hp / fmaxf(1.0f, enemy->max_hp));
        type_mass[enemy->type] += threat;
        prop_mass[0] += enemy->camo ? threat : 0.0f;
        prop_mass[1] += enemy->fortified ? threat : 0.0f;
        prop_mass[2] += enemy->regrow ? threat : 0.0f;
        prop_mass[3] += TD_ENEMY_SHARP_IMMUNE[enemy->type] ? threat : 0.0f;
        prop_mass[4] += TD_ENEMY_EXPLOSIVE_IMMUNE[enemy->type] ? threat : 0.0f;
        prop_mass[5] += TD_ENEMY_BURN_IMMUNE[enemy->type] ? threat : 0.0f;
        prop_mass[6] += TD_ENEMY_SLOW_IMMUNE[enemy->type] ? threat : 0.0f;
        prop_mass[7] += enemy->type == 9 ? threat : 0.0f;
        band_mass[band][0] += 1.0f;
        band_mass[band][1] += hp;
        band_mass[band][2] += leak;
        band_mass[band][3] +=
            (enemy->camo || TD_ENEMY_SHARP_IMMUNE[enemy->type] ||
             TD_ENEMY_EXPLOSIVE_IMMUNE[enemy->type] || TD_ENEMY_BURN_IMMUNE[enemy->type] ||
             TD_ENEMY_SLOW_IMMUNE[enemy->type])
                ? leak
                : 0.0f;
    }
    for (int i = 0; i < TD_NUM_ENEMY_PROGRESS_BINS; i++) {
        env->observations[idx++] = td_squash(count_bins[i], 3.0f);
    }
    for (int i = 0; i < TD_NUM_ENEMY_PROGRESS_BINS; i++) {
        env->observations[idx++] = td_squash(hp_bins[i], 3.0f);
    }

    idx = TD_GLOBAL_OBSERVATION_OFFSET;
    for (int i = 0; i < 10; i++) {
        env->observations[idx++] = td_squash(type_mass[i], 12.0f);
    }
    for (int i = 0; i < 8; i++) {
        env->observations[idx++] = td_squash(prop_mass[i], 12.0f);
    }
    for (int band = 0; band < 4; band++) {
        env->observations[idx++] = td_squash(band_mass[band][0], 4.0f);
        env->observations[idx++] = td_squash(band_mass[band][1], 24.0f);
        env->observations[idx++] = td_squash(band_mass[band][2], 10.0f);
        env->observations[idx++] = td_squash(band_mass[band][3], 10.0f);
    }
    float comp[10] = {0};
    for (int slot = 0; slot < TD_NUM_PLACEMENT_SLOTS; slot++) {
        TdTower *tower = &env->towers[slot];
        if (!tower->alive) {
            continue;
        }
        float range, damage, fire_rate, projectile_speed, burn_dps, burn_time, slow_mult, slow_time;
        int damage_type, detect_camo;
        td_tower_stats(tower, &range, &damage, &fire_rate, &projectile_speed, &damage_type,
                       &detect_camo, &burn_dps, &burn_time, &slow_mult, &slow_time);
        comp[tower->kind] += 1.0f;
        comp[3] += 1.0f;
        comp[damage_type == 0 ? 4 : 5] += damage / fmaxf(0.1f, fire_rate);
        comp[6] += detect_camo ? 1.0f : 0.0f;
        comp[7] += burn_dps;
        comp[8] += (1.0f - slow_mult) * fmaxf(1.0f, slow_time);
        comp[9] += tower->invested;
    }
    env->observations[idx++] = td_squash(comp[0], 6.0f);
    env->observations[idx++] = td_squash(comp[1], 4.0f);
    env->observations[idx++] = td_squash(comp[2], 4.0f);
    env->observations[idx++] = td_squash(comp[3], 12.0f);
    env->observations[idx++] = td_squash(comp[4], 20.0f);
    env->observations[idx++] = td_squash(comp[5], 20.0f);
    env->observations[idx++] = td_squash(comp[6], 4.0f);
    env->observations[idx++] = td_squash(comp[7], 8.0f);
    env->observations[idx++] = td_squash(comp[8], 4.0f);
    env->observations[idx++] = td_squash(comp[9], 8000.0f);

    td_update_masks(env);
}

static int td_spawn_done(const TowerDefence *env) {
    for (int i = 0; i < env->active_spawns; i++) {
        if (env->spawns[i].emitted < env->spawns[i].count) {
            return 0;
        }
    }
    return 1;
}

static float td_linear_hp_multiplier(int round);
static float td_linear_speed_multiplier(int round);

static int td_add_enemy(TowerDefence *env, int type, int camo, int fortified, int regrow,
                        double progress) {
    if (type < 0 || type >= 10) {
        return -1;
    }
    int first_free = -1;
    int last_alive = -1;
    for (int i = 0; i < env->enemy_high_water; i++) {
        if (env->enemies[i].alive) {
            last_alive = i;
        } else if (first_free < 0) {
            first_free = i;
        }
    }
    int idx = last_alive + 1;
    if (idx >= TD_MAX_ENEMIES) {
        idx = first_free;
    }
    if (idx < 0 || idx >= TD_MAX_ENEMIES) {
        return -1;
    }
    if (idx >= env->enemy_high_water) {
        env->enemy_high_water = idx + 1;
    }

    TdEnemy *enemy = &env->enemies[idx];
    memset(enemy, 0, sizeof(*enemy));
    enemy->alive = 1;
    enemy->type = type;
    enemy->camo = camo;
    enemy->fortified = fortified;
    enemy->regrow = regrow;
    enemy->distance = fmax(0.0, progress) * TD_PATH_LENGTH;
    enemy->progress = enemy->distance / TD_PATH_LENGTH;
    enemy->id = env->next_enemy_id++;
    enemy->max_hp = TD_ENEMY_HP[type] * td_linear_hp_multiplier(env->round) *
                    (fortified ? TD_FORTIFIED_HP_MULTIPLIER : 1.0f);
    enemy->hp = enemy->max_hp;
    enemy->speed = TD_ENEMY_SPEED[type] * td_linear_speed_multiplier(env->round);
    enemy->slow_mult = 1.0f;
    return idx;
}

static void td_trim_enemy_storage(TowerDefence *env) {
    while (env->enemy_high_water > 0 && !env->enemies[env->enemy_high_water - 1].alive) {
        env->enemy_high_water -= 1;
    }
}

static void td_add_spawn_at(TowerDefence *env, int type, int count, double interval,
                            double start_time, int camo, int fortified, int regrow) {
    if (env->active_spawns >= TD_MAX_SPAWNS) {
        return;
    }
    TdSpawn *spawn = &env->spawns[env->active_spawns++];
    memset(spawn, 0, sizeof(*spawn));
    spawn->type = type;
    spawn->count = count;
    spawn->interval = interval;
    spawn->start_time = start_time;
    spawn->next_time = start_time;
    spawn->camo = camo;
    spawn->fortified = fortified;
    spawn->regrow = regrow;
    spawn->modifier_rng_state = 1u;
}

static void td_add_spawn_after(TowerDefence *env, double *cursor, int type, int count,
                               double interval, double delay, int camo, int fortified, int regrow) {
    double start = *cursor + delay;
    td_add_spawn_at(env, type, count, interval, start, camo, fortified, regrow);
    if (count > 0) {
        *cursor = start + (count - 1) * fmax(0.06, interval);
    }
}

static int td_prepare_fixed_wave(TowerDefence *env, int round) {
    double cursor = 0.0;
    int added = 0;
    int n = (int)(sizeof(TD_FIXED_SPAWNS) / sizeof(TD_FIXED_SPAWNS[0]));
    for (int i = 0; i < n; i++) {
        const TdFixedSpawnSpec *spec = &TD_FIXED_SPAWNS[i];
        if (spec->round != round) {
            continue;
        }
        td_add_spawn_after(env, &cursor, spec->type, spec->count, spec->interval, spec->delay, 0, 0,
                           0);
        added = 1;
    }
    return added;
}

static float td_type_recursive_rbe(int type);
static int td_type_recursive_entity_count(int type);

static float td_roundf(float value) {
    return floorf(value + 0.5f);
}

static double td_round_places(double value, double scale) {
    return floor(value * scale + 0.5) / scale;
}

static float td_linear_hp_multiplier(int round) {
    float t = fmaxf(0.0f, (float)(round - 80));
    return fminf(2.25f, 1.0f + 0.004f * t + 0.00001f * t * t);
}

static float td_linear_speed_multiplier(int round) {
    float t = fmaxf(0.0f, (float)(round - 80));
    return fminf(1.75f, 1.0f + 0.006f * t);
}

static float td_default_spawn_interval_multiplier(int round) {
    float t = fmaxf(0.0f, (float)(round - 80));
    return powf(0.985f, t);
}

static float td_linear_reward_multiplier(int round) {
    return fminf(2.5f, 1.0f + (td_linear_hp_multiplier(round) - 1.0f) * 0.65f);
}

static float td_linear_modifier_chance(int round, int group, int kind) {
    float generated = fmaxf(0.0f, (float)(round - 20));
    float group_factor = 1.0f + group * 0.08f;
    if (kind == 0) {
        return fminf(0.32f, (0.012f + 0.00055f * generated) * group_factor);
    }
    if (kind == 1) {
        return fminf(0.28f, (0.006f + 0.00045f * generated) * group_factor);
    }
    return fminf(0.25f, (0.008f + 0.0004f * generated) * group_factor);
}

static int td_linear_target_effective_rbe(int round) {
    int generated = round - 20;
    if (generated <= 20) {
        return (int)td_roundf(170.0f + 3.5f * generated);
    }
    if (generated <= 60) {
        return (int)td_roundf(240.0f + (generated - 20));
    }
    return (int)td_roundf(fminf(290.0f, 280.0f + 0.05f * (generated - 60)));
}

static int td_linear_group_count(int round) {
    if (round < 36) {
        return 2;
    }
    if (round < 70) {
        return 3;
    }
    return 4;
}

static const int TD_LINEAR_POOL_21_TYPES[] = {1, 2, 3, 4, 5, 6, 7};
static const int TD_LINEAR_POOL_21_WEIGHTS[] = {2, 4, 4, 3, 2, 2, 1};
static const int TD_LINEAR_POOL_36_TYPES[] = {3, 4, 5, 6, 7, 8};
static const int TD_LINEAR_POOL_36_WEIGHTS[] = {2, 3, 3, 3, 2, 2};
static const int TD_LINEAR_POOL_61_TYPES[] = {5, 6, 7, 8, 9};
static const int TD_LINEAR_POOL_61_WEIGHTS[] = {2, 2, 3, 3, 1};
static const int TD_LINEAR_POOL_101_TYPES[] = {5, 6, 7, 8};
static const int TD_LINEAR_POOL_101_WEIGHTS[] = {2, 2, 3, 3};

static void td_linear_pool(int round, const int **types, const int **weights, int *n) {
    if (round >= 101) {
        *types = TD_LINEAR_POOL_101_TYPES;
        *weights = TD_LINEAR_POOL_101_WEIGHTS;
        *n = 4;
    } else if (round >= 61) {
        *types = TD_LINEAR_POOL_61_TYPES;
        *weights = TD_LINEAR_POOL_61_WEIGHTS;
        *n = 5;
    } else if (round >= 36) {
        *types = TD_LINEAR_POOL_36_TYPES;
        *weights = TD_LINEAR_POOL_36_WEIGHTS;
        *n = 6;
    } else {
        *types = TD_LINEAR_POOL_21_TYPES;
        *weights = TD_LINEAR_POOL_21_WEIGHTS;
        *n = 7;
    }
}

static int td_pick_weighted_type(int round, unsigned int *rng_state) {
    const int *types;
    const int *weights;
    int n;
    td_linear_pool(round, &types, &weights, &n);
    int total = 0;
    for (int i = 0; i < n; i++) {
        total += weights[i];
    }
    float roll = td_lcg_randf(rng_state) * total;
    for (int i = 0; i < n; i++) {
        roll -= weights[i];
        if (roll <= 0.0f) {
            return types[i];
        }
    }
    return types[n - 1];
}

static float td_spawn_effective_rbe(int type, int count, float hp_mult, float fortified_chance) {
    float fortified_mult = 1.0f + fortified_chance * (TD_FORTIFIED_HP_MULTIPLIER - 1.0f);
    return count * td_type_recursive_rbe(type) * hp_mult * fortified_mult;
}

static int td_prepare_linear_wave(TowerDefence *env, int round) {
    unsigned int wave_seed = td_derive_seed(env->rng_state, (unsigned int)round);
    unsigned int rng = wave_seed;
    float hp_mult = td_linear_hp_multiplier(round);
    int target_rbe = td_linear_target_effective_rbe(round);
    int groups = td_linear_group_count(round);
    double effective_interval = round <= 80 ? fmax(0.34, 0.5 - 0.0025 * (round - 20))
                                            : fmax(0.28, 0.35 - 0.0007 * (round - 80));
    double compensated_interval = td_round_places(
        effective_interval / fmax(0.0001f, td_default_spawn_interval_multiplier(round)), 10000.0);
    double interval =
        fmax(0.06, compensated_interval * td_default_spawn_interval_multiplier(round));
    double start_spacing = fmax(1.25, effective_interval * 5.0);
    int max_total = round <= 80 ? 150 : 180;
    int max_group = (int)fmaxf(8.0f, ceilf(max_total / (float)groups));
    float remaining_budget = (float)target_rbe;
    int remaining_entities = TD_RECURSIVE_ENTITY_CAP;
    int added = 0;

    for (int group = 0; group < groups && group < TD_MAX_SPAWNS; group++) {
        int groups_left = groups - group;
        int type = td_pick_weighted_type(round, &rng);
        float camo_chance = td_linear_modifier_chance(round, group, 0);
        float fortified_chance = td_linear_modifier_chance(round, group, 1);
        float regrow_chance = td_linear_modifier_chance(round, group, 2);
        float share =
            group == groups - 1
                ? remaining_budget
                : td_roundf((remaining_budget / groups_left) * (0.9f + td_lcg_randf(&rng) * 0.2f));
        float expected_unit = td_spawn_effective_rbe(type, 1, hp_mult, fortified_chance);
        int entities_per_unit = td_type_recursive_entity_count(type);
        int entity_limit = remaining_entities / (entities_per_unit > 0 ? entities_per_unit : 1);
        if (entity_limit <= 0) {
            break;
        }
        int count = (int)td_roundf(share / fmaxf(0.0001f, expected_unit));
        if (count < 1) {
            count = 1;
        }
        if (count > max_group) {
            count = max_group;
        }
        if (count > entity_limit) {
            count = entity_limit;
        }
        double start = td_round_places(group * start_spacing, 1000.0);
        td_add_spawn_at(env, type, count, interval, start, 0, 0, 0);
        TdSpawn *spawn = &env->spawns[env->active_spawns - 1];
        spawn->use_modifier_chances = 1;
        spawn->camo_chance = camo_chance;
        spawn->fortified_chance = fortified_chance;
        spawn->regrow_chance = regrow_chance;
        remaining_budget =
            fmaxf(0.0f, remaining_budget -
                            td_spawn_effective_rbe(type, count, hp_mult, fortified_chance));
        int consumed_entities = count * td_type_recursive_entity_count(type);
        remaining_entities =
            remaining_entities > consumed_entities ? remaining_entities - consumed_entities : 0;
        added += 1;
    }

    if (added > 0) {
        TdSpawn *final_spawn = &env->spawns[added - 1];
        int final_entities = td_type_recursive_entity_count(final_spawn->type);
        int recursive_entities = 0;
        float effective_wave_rbe = 0.0f;
        for (int i = 0; i < added; i++) {
            TdSpawn *spawn = &env->spawns[i];
            recursive_entities += spawn->count * td_type_recursive_entity_count(spawn->type);
            effective_wave_rbe +=
                td_spawn_effective_rbe(spawn->type, spawn->count, hp_mult, spawn->fortified_chance);
        }
        while (effective_wave_rbe < target_rbe && final_spawn->count < max_group &&
               recursive_entities + final_entities <= TD_RECURSIVE_ENTITY_CAP) {
            final_spawn->count += 1;
            recursive_entities += final_entities;
            effective_wave_rbe += td_spawn_effective_rbe(final_spawn->type, 1, hp_mult,
                                                         final_spawn->fortified_chance);
        }
        unsigned int modifier_seed = td_derive_seed(wave_seed, 1009u + (unsigned int)round);
        for (int i = 0; i < added; i++) {
            env->spawns[i].modifier_rng_state = modifier_seed;
            for (int skip = 0; skip < env->spawns[i].count * 3; skip++) {
                (void)td_lcg_randf(&modifier_seed);
            }
        }
    }
    return added;
}

static void td_prepare_wave(TowerDefence *env) {
    env->active_spawns = 0;
    env->wave_elapsed = 0.0f;
    int r = env->round;
    if (td_prepare_fixed_wave(env, r)) {
        return;
    }
    td_prepare_linear_wave(env, r);
}

static void td_start_round(TowerDefence *env) {
    if (env->status_code != TD_STATUS_WARMUP && env->status_code != TD_STATUS_INTERMISSION) {
        return;
    }
    if (env->round_pending_advance || env->round == 0) {
        env->round += 1;
        env->round_pending_advance = 0;
    }
    env->status_code = TD_STATUS_SPAWNING;
    // Keep the intermission feature at 2 while a wave is spawning; it is not
    // an active countdown outside intermission.
    env->intermission_remaining = 2.0f;
    td_prepare_wave(env);
}

static void td_kill_enemy(TowerDefence *env, int idx, float *reward) {
    TdEnemy *enemy = &env->enemies[idx];
    int child_a = TD_ENEMY_CHILD_A[enemy->type];
    int child_b = TD_ENEMY_CHILD_B[enemy->type];
    double parent_distance = enemy->distance;
    double spacing = fmax(2.0, TD_ENEMY_RADIUS[enemy->type] * 0.35);
    int camo = enemy->camo;
    int fortified = enemy->fortified;
    int regrow = enemy->regrow;
    float enemy_reward =
        floorf(TD_ENEMY_REWARD[enemy->type] * td_linear_reward_multiplier(env->round) *
                   (fortified ? TD_FORTIFIED_REWARD_MULTIPLIER : 1.0f) +
               0.5f);
    *reward += enemy_reward * env->reward_enemy_scale;
    env->cash += enemy_reward;
    enemy->alive = 0;
    if (child_a >= 0) {
        int child = td_add_enemy(env, child_a, camo, fortified, regrow,
                                 fmax(0.0, parent_distance - spacing) / TD_PATH_LENGTH);
        if (child < 0) {
            fprintf(stderr, "Tower defence enemy capacity invariant failed\n");
            exit(1);
        }
    }
    if (child_b >= 0) {
        int child = td_add_enemy(env, child_b, camo, fortified, regrow,
                                 fmax(0.0, parent_distance - 2.0 * spacing) / TD_PATH_LENGTH);
        if (child < 0) {
            fprintf(stderr, "Tower defence enemy capacity invariant failed\n");
            exit(1);
        }
    }
}

static void td_add_projectile(TowerDefence *env, int source_tower, int target_enemy, float damage,
                              int damage_type, float speed, float burn_dps, float burn_time,
                              float slow_mult, float slow_time) {
    if (env->projectile_count == env->projectile_capacity) {
        td_reserve_projectiles(env, env->projectile_capacity * 2);
    }
    const TdEnemy *enemy = &env->enemies[target_enemy];
    TdProjectile *projectile = &env->projectiles[env->projectile_count++];
    memset(projectile, 0, sizeof(*projectile));
    projectile->alive = 1;
    projectile->source_tower = source_tower;
    projectile->kind = env->towers[source_tower].kind;
    projectile->target_enemy = target_enemy;
    projectile->target_enemy_id = enemy->id;
    projectile->x = td_site_x(source_tower);
    projectile->y = td_site_y(source_tower);
    projectile->previous_x = projectile->x;
    projectile->previous_y = projectile->y;
    projectile->damage = damage;
    projectile->damage_type = damage_type;
    projectile->speed = speed;
    projectile->burn_dps = burn_dps;
    projectile->burn_time = burn_time;
    projectile->slow_mult = slow_mult;
    projectile->slow_time = slow_time;
}

static void td_record_shot(TowerDefence *env, int source_tower) {
    TdShotEvent *event = NULL;
    if (env->shot_serial > 0) {
        int latest = (int)((env->shot_serial - 1) % TD_MAX_SHOT_EVENTS);
        if (env->shot_events[latest].serial == env->shot_serial &&
            env->shot_events[latest].step_time == env->time) {
            event = &env->shot_events[latest];
        }
    }
    if (event == NULL) {
        env->shot_serial += 1;
        int index = (int)((env->shot_serial - 1) % TD_MAX_SHOT_EVENTS);
        event = &env->shot_events[index];
        memset(event, 0, sizeof(*event));
        event->serial = env->shot_serial;
        event->step_time = env->time;
    }
    event->fired_towers[source_tower / 64] |= UINT64_C(1) << (source_tower % 64);
}

static void td_record_impact(TowerDefence *env, const TdProjectile *projectile) {
    env->impact_serial += 1;
    int index = (int)((env->impact_serial - 1) % TD_MAX_IMPACT_EVENTS);
    TdImpactEvent *impact = &env->impact_events[index];
    memset(impact, 0, sizeof(*impact));
    impact->serial = env->impact_serial;
    impact->step_time = env->time;
    impact->from_x = projectile->previous_x;
    impact->from_y = projectile->previous_y;
    impact->x = projectile->x;
    impact->y = projectile->y;
    impact->kind = projectile->kind;
}

static void td_apply_tower_damage(TowerDefence *env) {
    TdVec2d enemy_points[TD_MAX_ENEMIES];
    int enemy_points_ready = 0;
    for (int slot = 0; slot < TD_NUM_PLACEMENT_SLOTS; slot++) {
        TdTower *tower = &env->towers[slot];
        if (!tower->alive) {
            continue;
        }
        if (tower->cooldown > 0.0f) {
            tower->cooldown -= TD_DT;
        }
        if (tower->cooldown > 0.0f) {
            continue;
        }
        float cooldown_residual = fmaxf(-TD_DT, tower->cooldown);

        float range, damage, fire_rate, projectile_speed, burn_dps, burn_time, slow_mult, slow_time;
        int damage_type, detect_camo;
        td_tower_stats(tower, &range, &damage, &fire_rate, &projectile_speed, &damage_type,
                       &detect_camo, &burn_dps, &burn_time, &slow_mult, &slow_time);
        int target = -1;
        double best_distance2 = INFINITY;
        double range2 = (double)range * range;
        float tower_x = td_site_x(slot);
        float tower_y = td_site_y(slot);
        if (!enemy_points_ready) {
            for (int i = 0; i < env->enemy_high_water; i++) {
                if (env->enemies[i].alive) {
                    enemy_points[i] = td_path_point_at_distance(env->enemies[i].distance);
                }
            }
            enemy_points_ready = 1;
        }
        for (int i = 0; i < env->enemy_high_water; i++) {
            const TdEnemy *enemy = &env->enemies[i];
            if (!enemy->alive || (enemy->camo && !detect_camo)) {
                continue;
            }
            TdVec2d p = enemy_points[i];
            double dx = p.x - tower_x;
            double dy = p.y - tower_y;
            double distance2 = dx * dx + dy * dy;
            if (distance2 <= range2 && distance2 < best_distance2) {
                target = i;
                best_distance2 = distance2;
            }
        }
        if (target < 0) {
            tower->cooldown = 0.0f;
            continue;
        }
        td_add_projectile(env, slot, target, damage, damage_type, projectile_speed, burn_dps,
                          burn_time, slow_mult, slow_time);
        tower->cooldown = fire_rate + cooldown_residual;
        td_record_shot(env, slot);
    }
}

static void td_update_projectiles(TowerDefence *env, float *reward) {
    for (int i = env->projectile_count - 1; i >= 0; i--) {
        TdProjectile *projectile = &env->projectiles[i];
        if (!projectile->alive) {
            continue;
        }
        projectile->previous_x = projectile->x;
        projectile->previous_y = projectile->y;
        if (projectile->target_enemy < 0 || projectile->target_enemy >= env->enemy_high_water) {
            projectile->alive = 0;
            continue;
        }
        TdEnemy *enemy = &env->enemies[projectile->target_enemy];
        if (!enemy->alive || enemy->id != projectile->target_enemy_id) {
            projectile->alive = 0;
            continue;
        }

        TdVec2d target = td_path_point_at_distance(enemy->distance);
        double dx = target.x - projectile->x;
        double dy = target.y - projectile->y;
        double dist = sqrt(dx * dx + dy * dy);
        double move = projectile->speed * TD_DT;
        int hit = dist <= move;
        if (hit) {
            projectile->x = (float)target.x;
            projectile->y = (float)target.y;
        } else if (dist > 0.0f) {
            projectile->x += (float)((dx / dist) * move);
            projectile->y += (float)((dy / dist) * move);
        }
        if (!hit) {
            continue;
        }
        td_record_impact(env, projectile);

        if (projectile->burn_dps > 0.0f && !TD_ENEMY_BURN_IMMUNE[enemy->type]) {
            float burn_dps = projectile->burn_dps;
            float burn_time = projectile->burn_time;
            if (enemy->type == 9) {
                burn_dps *= 0.85f;
                burn_time *= 0.8f;
            }
            enemy->burn_dps = fmaxf(enemy->burn_dps, burn_dps);
            enemy->burn_time = fmaxf(enemy->burn_time, burn_time);
        }
        if (projectile->slow_mult < 1.0f && !TD_ENEMY_SLOW_IMMUNE[enemy->type]) {
            float slow_mult = projectile->slow_mult;
            float slow_time = projectile->slow_time;
            if (enemy->type == 9) {
                slow_mult = 1.0f - (1.0f - slow_mult) * 0.8f;
                slow_time *= 0.75f;
            }
            enemy->slow_mult = fminf(enemy->slow_mult, slow_mult);
            enemy->slow_time = fmaxf(enemy->slow_time, slow_time);
        }

        float damage = projectile->damage;
        if (projectile->damage_type == 0 && TD_ENEMY_SHARP_IMMUNE[enemy->type]) {
            damage = 0.0f;
        }
        if (projectile->damage_type == 1 && TD_ENEMY_EXPLOSIVE_IMMUNE[enemy->type]) {
            damage = 0.0f;
        }
        float hp_before = enemy->hp;
        enemy->hp -= damage;
        if (enemy->regrow && enemy->hp < hp_before) {
            enemy->regrow_cooldown_until = env->time + 1.25f;
        }
        if (enemy->hp <= 0.0f) {
            td_kill_enemy(env, projectile->target_enemy, reward);
        }
        projectile->alive = 0;
    }
    td_compact_projectiles(env);
}

static float td_type_recursive_rbe(int type) {
    return (type >= 0 && type < 10) ? TD_ENEMY_RECURSIVE_RBE[type] : 1.0f;
}

static int td_type_recursive_entity_count(int type) {
    return (type >= 0 && type < 10) ? TD_ENEMY_RECURSIVE_COUNT[type] : 1;
}

static float td_round_effective_rbe(const TowerDefence *env) {
    if (env->round > 20) {
        return (float)td_linear_target_effective_rbe(env->round);
    }
    float hp_mult = td_linear_hp_multiplier(env->round);
    float total = 0.0f;
    for (int i = 0; i < env->active_spawns; i++) {
        const TdSpawn *spawn = &env->spawns[i];
        float fortified_mult = spawn->fortified ? TD_FORTIFIED_HP_MULTIPLIER : 1.0f;
        total += spawn->count * td_type_recursive_rbe(spawn->type) * hp_mult * fortified_mult;
    }
    return total;
}

static float td_round_clear_bonus(const TowerDefence *env) {
    return floorf(42.0f + 4.0f * env->round + 0.052f * td_round_effective_rbe(env));
}

static void td_advance_world(TowerDefence *env, float *reward) {
    env->time += TD_DT;
    if (env->status_code == TD_STATUS_INTERMISSION) {
        env->intermission_remaining = fmaxf(0.0f, env->intermission_remaining - TD_DT);
        return;
    }
    if (env->status_code != TD_STATUS_SPAWNING && env->status_code != TD_STATUS_ACTIVE) {
        return;
    }

    if (env->status_code == TD_STATUS_SPAWNING) {
        env->wave_elapsed += TD_DT;
        for (;;) {
            int next_spawn = -1;
            double next_time = INFINITY;
            for (int s = 0; s < env->active_spawns; s++) {
                TdSpawn *candidate = &env->spawns[s];
                if (candidate->emitted >= candidate->count) {
                    continue;
                }
                if ((double)env->wave_elapsed < candidate->next_time) {
                    continue;
                }
                if (candidate->next_time < next_time) {
                    next_spawn = s;
                    next_time = candidate->next_time;
                }
            }
            if (next_spawn < 0) {
                break;
            }
            TdSpawn *spawn = &env->spawns[next_spawn];
            int camo = spawn->camo;
            int fortified = spawn->fortified;
            int regrow = spawn->regrow;
            unsigned int modifier_rng_state = spawn->modifier_rng_state;
            if (spawn->use_modifier_chances) {
                camo = camo || td_lcg_randf(&modifier_rng_state) < spawn->camo_chance;
                fortified =
                    fortified || td_lcg_randf(&modifier_rng_state) < spawn->fortified_chance;
                regrow = regrow || td_lcg_randf(&modifier_rng_state) < spawn->regrow_chance;
            }
            if (td_add_enemy(env, spawn->type, camo, fortified, regrow, 0.0f) < 0) {
                break;
            }
            spawn->modifier_rng_state = modifier_rng_state;
            spawn->emitted += 1;
            spawn->next_time = spawn->start_time + spawn->emitted * spawn->interval;
        }
    }
    if (td_spawn_done(env)) {
        env->status_code = TD_STATUS_ACTIVE;
    }

    // Splits created during this pass start moving on the next simulation tick,
    // even when they reuse a lower array slot that this reverse loop has not
    // seen.
    int max_enemy_id_for_update = env->next_enemy_id - 1;
    for (int i = env->enemy_high_water - 1; i >= 0; i--) {
        TdEnemy *enemy = &env->enemies[i];
        if (!enemy->alive || enemy->id > max_enemy_id_for_update) {
            continue;
        }
        if (enemy->burn_time > 0.0f) {
            float burn_seconds = fminf(TD_DT, enemy->burn_time);
            if (burn_seconds > 0.0f) {
                float hp_before = enemy->hp;
                enemy->hp -= enemy->burn_dps * burn_seconds;
                if (enemy->regrow && enemy->hp < hp_before) {
                    enemy->regrow_cooldown_until = env->time + 1.25f;
                }
            }
            enemy->burn_time = fmaxf(0.0f, enemy->burn_time - TD_DT);
            if (enemy->burn_time <= 0.0f) {
                enemy->burn_dps = 0.0f;
            }
        }
        float move_slow_mult = 1.0f;
        if (enemy->slow_time > 0.0f) {
            move_slow_mult = enemy->slow_mult;
        } else {
            enemy->slow_mult = 1.0f;
        }
        if (enemy->regrow && enemy->hp < enemy->max_hp &&
            env->time >= enemy->regrow_cooldown_until) {
            enemy->hp = fminf(enemy->max_hp, enemy->hp + enemy->max_hp * 0.25f * TD_DT);
        }
        if (enemy->hp <= 0.0f) {
            td_kill_enemy(env, i, reward);
            continue;
        }
        enemy->distance += enemy->speed * move_slow_mult * TD_DT;
        enemy->progress = enemy->distance / TD_PATH_LENGTH;
        if (enemy->slow_time > 0.0f) {
            enemy->slow_time = fmaxf(0.0f, enemy->slow_time - TD_DT);
            if (enemy->slow_time <= 0.0f) {
                enemy->slow_mult = 1.0f;
            }
        }
        if (enemy->distance >= TD_PATH_LENGTH) {
            env->lives -= TD_ENEMY_LEAK[enemy->type];
            *reward -= TD_ENEMY_LEAK[enemy->type] * env->reward_leak_penalty_scale;
            enemy->alive = 0;
        }
    }
    td_apply_tower_damage(env);
    td_update_projectiles(env, reward);
    td_trim_enemy_storage(env);

    if (env->lives <= 0.0f) {
        env->status_code = TD_STATUS_DEFEAT;
        return;
    }
    if (td_spawn_done(env) && td_enemy_count(env) == 0) {
        float bonus = td_round_clear_bonus(env);
        env->cash += bonus;
        *reward += bonus * env->reward_round_clear_scale;
        env->score = (float)env->round;
        env->status_code =
            env->round >= env->target_round ? TD_STATUS_COMPLETE : TD_STATUS_INTERMISSION;
        env->intermission_remaining =
            env->status_code == TD_STATUS_INTERMISSION ? fmaxf(0.0f, 2.0f - TD_DT) : 0.0f;
        env->round_pending_advance = env->status_code == TD_STATUS_INTERMISSION;
    }
}

static void td_place(TowerDefence *env, int slot, int kind) {
    TdTower *tower = &env->towers[slot];
    tower->alive = 1;
    tower->kind = kind;
    tower->cooldown = 0.0f;
    tower->invested = TD_TOWER_COST[kind];
    memset(tower->upgrades, 0, sizeof(tower->upgrades));
    env->cash -= TD_TOWER_COST[kind];
}

static void td_upgrade(TowerDefence *env, int slot, int path) {
    TdTower *tower = &env->towers[slot];
    float cost = td_upgrade_cost(tower, path);
    tower->upgrades[path] += 1;
    tower->invested += cost;
    env->cash -= cost;
}

static void td_sell(TowerDefence *env, int slot) {
    TdTower *tower = &env->towers[slot];
    env->cash += floorf(tower->invested * 0.7f);
    memset(tower, 0, sizeof(*tower));
}

static int td_apply_action(TowerDefence *env, int action) {
    if (action < 0 || action >= TD_NUM_ACTIONS || !env->action_mask[action]) {
        return 0;
    }
    if (action >= TD_ACTION_PLACE_FIRST && action < TD_ACTION_UPGRADE_SLOT_01_TOP) {
        int idx = action - TD_ACTION_PLACE_FIRST;
        td_place(env, idx / TD_NUM_TOWER_TYPES, idx % TD_NUM_TOWER_TYPES);
    } else if (action >= TD_ACTION_UPGRADE_SLOT_01_TOP && action < TD_ACTION_SELL_SLOT_01) {
        int idx = action - TD_ACTION_UPGRADE_SLOT_01_TOP;
        td_upgrade(env, idx / TD_NUM_UPGRADE_PATHS, idx % TD_NUM_UPGRADE_PATHS);
    } else if (action >= TD_ACTION_SELL_SLOT_01 && action < TD_ACTION_TRIGGER_NEXT_ROUND) {
        td_sell(env, action - TD_ACTION_SELL_SLOT_01);
    } else if (action == TD_ACTION_TRIGGER_NEXT_ROUND) {
        td_start_round(env);
    }
    return 1;
}

static void td_log_episode(TowerDefence *env) {
    float steps = env->step_count > 0 ? (float)env->step_count : 1.0f;
    env->log.perf += fminf(1.0f, env->score / (float)env->target_round);
    env->log.score += env->score;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->step_count;
    env->log.invalid_action_rate += (float)env->invalid_action_count / steps;
    env->log.n += 1.0f;
}

static void c_reset(TowerDefence *env) {
    env->rng_state =
        (unsigned int)env->base_seed + (unsigned int)env->rng * 100003u + env->episode_index++;
    env->step_count = 0;
    env->invalid_action_count = 0;
    env->max_tower_count = 0;
    env->max_total_invested = 0.0f;
    env->episode_return = 0.0f;
    env->next_enemy_id = 1;
    env->round_pending_advance = 0;
    env->round = 0;
    env->status_code = TD_STATUS_INTERMISSION;
    env->time = 0.0f;
    env->lives = 200.0f;
    env->cash = 10000.0f;
    env->intermission_remaining = 0.0f;
    env->wave_elapsed = 0.0f;
    env->score = 0.0f;
    env->active_spawns = 0;
    env->enemy_high_water = 0;
    env->projectile_count = 0;
    memset(env->shot_events, 0, sizeof(env->shot_events));
    memset(env->impact_events, 0, sizeof(env->impact_events));
    memset(env->spawns, 0, sizeof(env->spawns));
    memset(env->towers, 0, sizeof(env->towers));
    memset(env->enemies, 0, sizeof(env->enemies));
    memset(env->projectiles, 0, (size_t)env->projectile_capacity * sizeof(*env->projectiles));
    td_write_observation(env);
}

static void c_step(TowerDefence *env) {
    int action = td_normalize_action(env->actions[0]);
    float reward = 0.0f;
    int before_round = env->round;
    float before_score = env->score;
    int valid = td_apply_action(env, action);
    if (!valid) {
        env->invalid_action_count += 1;
        reward += env->invalid_action_reward;
    } else {
        if (action >= TD_ACTION_SELL_SLOT_01 && action < TD_ACTION_TRIGGER_NEXT_ROUND) {
            reward -= env->reward_sell_penalty_scale;
        }
        if (action == TD_ACTION_NOOP && env->action_mask[TD_ACTION_TRIGGER_NEXT_ROUND]) {
            reward -= env->reward_trigger_ready_noop_penalty;
        }
        if ((env->reward_tower_placement_scale != 0.0f ||
             env->reward_tower_investment_scale != 0.0f) &&
            action >= TD_ACTION_PLACE_FIRST && action < TD_ACTION_TRIGGER_NEXT_ROUND) {
            int tower_count = td_tower_count(env);
            float total_invested = td_total_invested(env);
            int new_tower_count = tower_count - env->max_tower_count;
            float new_investment = total_invested - env->max_total_invested;
            if (new_tower_count > 0) {
                reward += new_tower_count * env->reward_tower_placement_scale;
                env->max_tower_count = tower_count;
            }
            if (new_investment > 0.0f) {
                reward += new_investment * env->reward_tower_investment_scale;
                env->max_total_invested = total_invested;
            }
        }
    }
    td_advance_world(env, &reward);
    reward += fmaxf(0.0f, (float)(env->round - before_round)) * env->reward_round_advance_scale;
    reward += fmaxf(0.0f, env->score - before_score) * env->reward_score_advance_scale;
    env->step_count += 1;
    int truncated = env->step_count >= env->max_episode_steps;
    if (env->status_code == TD_STATUS_COMPLETE) {
        reward += env->reward_completion_bonus;
    }
    if (env->status_code == TD_STATUS_DEFEAT) {
        reward -= env->reward_defeat_penalty;
    }
    if (truncated && env->status_code != TD_STATUS_COMPLETE &&
        env->status_code != TD_STATUS_DEFEAT) {
        reward -= env->reward_truncation_penalty;
    }
    if (env->reward_clamp_enabled) {
        reward = td_clampf(reward, env->reward_clamp_min, env->reward_clamp_max);
    }
    env->episode_return += reward;
    env->rewards[0] = reward;
    env->terminals[0] = 0.0f;
    int done =
        env->status_code == TD_STATUS_DEFEAT || env->status_code == TD_STATUS_COMPLETE || truncated;
    if (done) {
        env->terminals[0] = 1.0f;
        td_log_episode(env);
        c_reset(env);
        return;
    }
    td_write_observation(env);
}

static Color td_enemy_color(int type) {
    static const Color colors[10] = {
        {239, 68, 68, 255},   {59, 130, 246, 255}, {34, 197, 94, 255},   {250, 204, 21, 255},
        {244, 114, 182, 255}, {17, 24, 39, 255},   {229, 231, 235, 255}, {107, 114, 128, 255},
        {249, 250, 251, 255}, {217, 119, 6, 255},
    };
    return colors[type];
}

struct TdClient {
    int hover_slot;
    int tower_kind;
    int human_control;
    int manual_controls_enabled;
    int policy_loaded;
    unsigned int episode_index;
    uint64_t shot_serial;
    uint64_t impact_serial;
    float observed_sim_time;
    double observed_sim_time_at;
    double fire_until[TD_NUM_PLACEMENT_SLOTS];
    struct {
        float from_x;
        float from_y;
        float x;
        float y;
        int kind;
        double until;
    } impacts[TD_MAX_IMPACT_EVENTS];
    Texture2D background;
    Texture2D towers[TD_NUM_TOWER_TYPES];
    Texture2D enemies[10];
    Texture2D projectiles[TD_NUM_TOWER_TYPES];
};

static Texture2D td_load_texture(const char *file_name) {
    char path[256];
    snprintf(path, sizeof(path), "%s/%s", TD_RESOURCE_DIR, file_name);
    return LoadTexture(path);
}

static int td_texture_loaded(Texture2D texture) {
    return texture.id != 0;
}

static Rectangle td_sheet_source(Texture2D texture, int frame) {
    float cell_w = texture.width * 0.5f;
    float cell_h = texture.height * 0.5f;
    frame = frame < 0 ? 0 : frame;
    frame = frame > 3 ? 3 : frame;
    return (Rectangle){(frame & 1) * cell_w, (frame >> 1) * cell_h, cell_w, cell_h};
}

static int td_tower_frame(const TdClient *client, const TdTower *tower, int slot, double now) {
    if (now < client->fire_until[slot]) {
        return 2;
    }
    return ((int)(now * 4.0) + tower->kind) & 1;
}

static Vector2 td_tower_sprite_center(Vector2 center, int kind, int frame, float size) {
    // The generated attack frames have a shorter transparent baseline than
    // their idle frames. Align their visible bottoms to prevent a fire hop.
    static const float baseline_offsets[TD_NUM_TOWER_TYPES][4] = {
        {0.0f, 0.0f, 57.0f / 512.0f, 58.0f / 512.0f},
        {0.0f, 6.0f / 512.0f, 36.0f / 512.0f, 30.0f / 512.0f},
        {0.0f, 0.0f, 49.0f / 512.0f, 47.0f / 512.0f},
    };
    center.y += size * baseline_offsets[kind][frame];
    return center;
}

static void td_draw_sprite(Texture2D texture, Rectangle source, Vector2 center, float size,
                           float rotation, Color tint) {
    Rectangle dest = {center.x, center.y, size, size};
    Vector2 origin = {size * 0.5f, size * 0.5f};
    DrawTexturePro(texture, source, dest, origin, rotation, tint);
}

static void td_load_render_assets(TowerDefence *env) {
    if (env->client != NULL) {
        return;
    }
    env->client = (TdClient *)calloc(1, sizeof(TdClient));
    if (env->client == NULL) {
        fprintf(stderr, "Failed to allocate tower_defence render client\n");
        exit(1);
    }
    env->client->hover_slot = -1;
    env->client->episode_index = env->episode_index;
    env->client->shot_serial = env->shot_serial;
    env->client->impact_serial = env->impact_serial;
    TdClient *client = env->client;

    client->background = td_load_texture("background.png");
    static const char *tower_files[TD_NUM_TOWER_TYPES] = {
        "tower_dart.png",
        "tower_sniper.png",
        "tower_cannon.png",
    };
    for (int i = 0; i < TD_NUM_TOWER_TYPES; i++) {
        client->towers[i] = td_load_texture(tower_files[i]);
    }

    static const char *enemy_files[10] = {
        "enemy_red.png",   "enemy_blue.png",    "enemy_green.png", "enemy_yellow.png",
        "enemy_pink.png",  "enemy_black.png",   "enemy_white.png", "enemy_lead.png",
        "enemy_zebra.png", "enemy_ceramic.png",
    };
    for (int i = 0; i < 10; i++) {
        client->enemies[i] = td_load_texture(enemy_files[i]);
    }

    static const char *projectile_files[TD_NUM_TOWER_TYPES] = {
        "projectile_dart.png",
        "projectile_sniper.png",
        "projectile_cannon.png",
    };
    for (int i = 0; i < TD_NUM_TOWER_TYPES; i++) {
        client->projectiles[i] = td_load_texture(projectile_files[i]);
    }
    client->observed_sim_time = env->time;
    client->observed_sim_time_at = GetTime();
    td_render_client_count += 1;
}

static void td_unload_texture(Texture2D *texture) {
    if (!td_texture_loaded(*texture)) {
        return;
    }
    UnloadTexture(*texture);
    memset(texture, 0, sizeof(*texture));
}

static void td_unload_render_assets(TowerDefence *env) {
    TdClient *client = env->client;
    if (client == NULL) {
        return;
    }
    td_unload_texture(&client->background);
    for (int i = 0; i < TD_NUM_TOWER_TYPES; i++) {
        td_unload_texture(&client->towers[i]);
    }
    for (int i = 0; i < 10; i++) {
        td_unload_texture(&client->enemies[i]);
    }
    for (int i = 0; i < TD_NUM_TOWER_TYPES; i++) {
        td_unload_texture(&client->projectiles[i]);
    }
    free(client);
    env->client = NULL;
    td_render_client_count -= 1;
    if (td_render_client_count == 0 && IsWindowReady()) {
        CloseWindow();
    }
}

static void td_sync_client_animation(TowerDefence *env, double now) {
    TdClient *client = env->client;
    if (client->episode_index != env->episode_index) {
        memset(client->fire_until, 0, sizeof(client->fire_until));
        memset(client->impacts, 0, sizeof(client->impacts));
        client->episode_index = env->episode_index;
        client->shot_serial = env->shot_serial;
        client->impact_serial = env->impact_serial;
        client->observed_sim_time = env->time;
        client->observed_sim_time_at = now;
        return;
    }
    if (client->observed_sim_time != env->time) {
        client->observed_sim_time = env->time;
        client->observed_sim_time_at = now;
    }
    if (client->shot_serial != env->shot_serial) {
        uint64_t first = client->shot_serial + 1;
        uint64_t oldest = env->shot_serial >= TD_MAX_SHOT_EVENTS
                              ? env->shot_serial - TD_MAX_SHOT_EVENTS + 1
                              : 1;
        if (first < oldest) {
            first = oldest;
        }
        for (uint64_t serial = first; serial <= env->shot_serial; serial++) {
            int index = (int)((serial - 1) % TD_MAX_SHOT_EVENTS);
            const TdShotEvent *event = &env->shot_events[index];
            if (event->serial != serial) {
                continue;
            }
            double fired_at = client->observed_sim_time_at +
                              (double)(event->step_time - client->observed_sim_time);
            double until = fired_at + 0.12;
            if (until <= now) {
                continue;
            }
            for (int slot = 0; slot < TD_NUM_PLACEMENT_SLOTS; slot++) {
                uint64_t bit = UINT64_C(1) << (slot % 64);
                if (event->fired_towers[slot / 64] & bit) {
                    client->fire_until[slot] = fmax(client->fire_until[slot], until);
                }
            }
        }
        client->shot_serial = env->shot_serial;
    }
    if (client->impact_serial == env->impact_serial) {
        return;
    }
    uint64_t first = client->impact_serial + 1;
    uint64_t oldest = env->impact_serial >= TD_MAX_IMPACT_EVENTS
                          ? env->impact_serial - TD_MAX_IMPACT_EVENTS + 1
                          : 1;
    if (first < oldest) {
        first = oldest;
    }
    for (uint64_t serial = first; serial <= env->impact_serial; serial++) {
        int index = (int)((serial - 1) % TD_MAX_IMPACT_EVENTS);
        const TdImpactEvent *event = &env->impact_events[index];
        if (event->serial != serial) {
            continue;
        }
        double started_at = client->observed_sim_time_at +
                            (double)(event->step_time - client->observed_sim_time);
        double until = started_at + 0.14;
        if (until <= now) {
            memset(&client->impacts[index], 0, sizeof(client->impacts[index]));
            continue;
        }
        client->impacts[index].from_x = event->from_x;
        client->impacts[index].from_y = event->from_y;
        client->impacts[index].x = event->x;
        client->impacts[index].y = event->y;
        client->impacts[index].kind = event->kind;
        client->impacts[index].until = until;
    }
    client->impact_serial = env->impact_serial;
}

static float td_render_phase(const TdClient *client, double now) {
    return td_clampf((float)((now - client->observed_sim_time_at) / TD_DT), 0.0f, 1.0f);
}

static double td_render_enemy_distance(const TdEnemy *enemy, float phase) {
    float slow = enemy->slow_time > 0.0f ? enemy->slow_mult : 1.0f;
    return enemy->distance + enemy->speed * slow * TD_DT * phase;
}

static Vector2 td_render_enemy_point(const TdEnemy *enemy, float phase) {
    TdVec2d point = td_path_point_at_distance(td_render_enemy_distance(enemy, phase));
    return (Vector2){(float)point.x, (float)point.y};
}

static Vector2 td_render_projectile_point(const TowerDefence *env, const TdProjectile *projectile,
                                          float phase) {
    Vector2 point = {projectile->x, projectile->y};
    if (projectile->target_enemy < 0 || projectile->target_enemy >= env->enemy_high_water) {
        return point;
    }
    const TdEnemy *enemy = &env->enemies[projectile->target_enemy];
    if (!enemy->alive || enemy->id != projectile->target_enemy_id) {
        return point;
    }
    Vector2 target = td_render_enemy_point(enemy, phase);
    double dx = target.x - point.x;
    double dy = target.y - point.y;
    double distance = sqrt(dx * dx + dy * dy);
    double movement = projectile->speed * TD_DT * phase;
    if (distance <= movement) {
        return target;
    }
    if (distance > 0.0) {
        point.x += (float)(dx / distance * movement);
        point.y += (float)(dy / distance * movement);
    }
    return point;
}

static Vector2 td_render_projectile_tail(const TdProjectile *projectile, float phase) {
    return (Vector2){
        projectile->previous_x + (projectile->x - projectile->previous_x) * phase,
        projectile->previous_y + (projectile->y - projectile->previous_y) * phase,
    };
}

static Color td_projectile_trail_color(int kind, float alpha) {
    static const Color colors[TD_NUM_TOWER_TYPES] = {
        {125, 211, 252, 255},
        {253, 230, 138, 255},
        {251, 146, 60, 255},
    };
    if (kind < 0 || kind >= TD_NUM_TOWER_TYPES) {
        kind = 0;
    }
    return Fade(colors[kind], alpha);
}

static int td_human_control_active(void) {
    return IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);
}

static int td_mouse_hover_slot(void) {
    Vector2 mouse = GetMousePosition();
    if (mouse.x < 0.0f || mouse.x > TD_WIDTH || mouse.y < 0.0f || mouse.y > TD_HEIGHT) {
        return -1;
    }
    float max_distance = fminf(TD_BUILD_GRID_STEP_X, TD_BUILD_GRID_STEP_Y) * 0.5f;
    float best = max_distance * max_distance;
    int hover = -1;
    for (int slot = 0; slot < TD_NUM_PLACEMENT_SLOTS; slot++) {
        float dx = mouse.x - td_site_x(slot);
        float dy = mouse.y - td_site_y(slot);
        float d = dx * dx + dy * dy;
        if (d <= best) {
            best = d;
            hover = slot;
        }
    }
    return hover;
}

static void c_render(TowerDefence *env) {
    if (!IsWindowReady()) {
        InitWindow(TD_WIDTH, TD_HEIGHT + TD_HUD_HEIGHT, "PufferLib Tower Defence");
        SetTargetFPS(60);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    td_load_render_assets(env);
    TdClient *client = env->client;
    double now = GetTime();
    td_sync_client_animation(env, now);
    float render_phase = td_render_phase(client, now);
    client->human_control = client->manual_controls_enabled && td_human_control_active();
    client->hover_slot = td_mouse_hover_slot();

    BeginDrawing();
    if (td_texture_loaded(client->background)) {
        DrawTexturePro(client->background,
                       (Rectangle){0, 0, client->background.width, client->background.height},
                       (Rectangle){0, 0, TD_WIDTH, TD_HEIGHT}, (Vector2){0, 0}, 0.0f, WHITE);
    } else {
        ClearBackground((Color){12, 18, 24, 255});
        for (int y = 0; y < TD_HEIGHT; y += 36) {
            DrawLine(0, y, TD_WIDTH, y, (Color){24, 32, 42, 255});
        }
        for (int i = 0; i < 5; i++) {
            Vector2 a = {TD_PATH_X[i], TD_PATH_Y[i]};
            Vector2 b = {TD_PATH_X[i + 1], TD_PATH_Y[i + 1]};
            DrawLineEx(a, b, 58, (Color){55, 65, 81, 255});
            DrawLineEx(a, b, 42, (Color){17, 24, 39, 255});
        }
    }

    for (int slot = 0; slot < TD_NUM_PLACEMENT_SLOTS; slot++) {
        const TdTower *tower = &env->towers[slot];
        int kind = tower->alive ? tower->kind : client->tower_kind;
        Color color = kind == 0
                          ? (Color){59, 130, 246, 255}
                          : (kind == 1 ? (Color){229, 231, 235, 255} : (Color){156, 163, 175, 255});
        Vector2 p = {td_site_x(slot), td_site_y(slot)};
        if (!tower->alive) {
            if (client->human_control) {
                int action = TD_ACTION_PLACE_FIRST + slot * TD_NUM_TOWER_TYPES + kind;
                int can_place = action >= 0 && action < TD_NUM_ACTIONS && env->action_mask[action];
                int buildable = td_site_buildable_for_kind(slot, kind);
                Color slot_color =
                    can_place ? (Color){34, 197, 94, 145}
                              : (buildable ? (Color){75, 85, 99, 100} : (Color){127, 29, 29, 60});
                if (slot == client->hover_slot) {
                    slot_color = can_place ? (Color){251, 191, 36, 190} : (Color){239, 68, 68, 180};
                }
                if (can_place) {
                    DrawCircleV(p, (float)TD_TOWER_RADIUS[kind] + 3.0f, (Color){34, 197, 94, 22});
                }
                DrawCircleLines((int)p.x, (int)p.y, TD_TOWER_RADIUS[kind], slot_color);
                if (slot == client->hover_slot && !can_place) {
                    DrawLineEx((Vector2){p.x - 7, p.y - 7}, (Vector2){p.x + 7, p.y + 7}, 2,
                               slot_color);
                    DrawLineEx((Vector2){p.x + 7, p.y - 7}, (Vector2){p.x - 7, p.y + 7}, 2,
                               slot_color);
                }
            }
            continue;
        }
        if (slot == client->hover_slot) {
            float range, damage, fire_rate, projectile_speed, burn_dps, burn_time, slow_mult,
                slow_time;
            int damage_type, detect_camo;
            td_tower_stats(tower, &range, &damage, &fire_rate, &projectile_speed, &damage_type,
                           &detect_camo, &burn_dps, &burn_time, &slow_mult, &slow_time);
            DrawCircleLines((int)p.x, (int)p.y, range, (Color){96, 165, 250, 80});
        }
        Texture2D tower_texture = client->towers[kind];
        if (td_texture_loaded(tower_texture)) {
            float size = kind == 1 ? 48.0f : (kind == 2 ? 58.0f : 54.0f);
            int frame = td_tower_frame(client, tower, slot, now);
            td_draw_sprite(tower_texture, td_sheet_source(tower_texture, frame),
                           td_tower_sprite_center(p, kind, frame, size), size, 0.0f, WHITE);
        } else {
            DrawCircleV(p, (float)TD_TOWER_RADIUS[kind] + 3, (Color){15, 23, 42, 255});
            DrawCircleV(p, (float)TD_TOWER_RADIUS[kind], color);
            DrawRectangle((int)p.x - 7, (int)p.y - 3, 14, 6, (Color){251, 146, 60, 255});
        }
    }

    for (int i = 0; i < env->projectile_count; i++) {
        const TdProjectile *projectile = &env->projectiles[i];
        if (!projectile->alive) {
            continue;
        }
        int kind = projectile->kind;
        if (kind < 0 || kind >= TD_NUM_TOWER_TYPES) {
            kind = 0;
        }
        Texture2D projectile_texture = client->projectiles[kind];
        Vector2 p = td_render_projectile_point(env, projectile, render_phase);
        Vector2 tail = td_render_projectile_tail(projectile, render_phase);
        DrawLineEx(tail, p, kind == 2 ? 4.0f : 2.5f,
                   td_projectile_trail_color(kind, 0.55f));
        if (td_texture_loaded(projectile_texture)) {
            int frame = ((int)(now * 12.0) + projectile->target_enemy_id) % 3;
            float size = kind == 2 ? 24.0f : 18.0f;
            td_draw_sprite(projectile_texture, td_sheet_source(projectile_texture, frame), p, size,
                           0.0f, WHITE);
        } else {
            DrawCircleV(p, kind == 2 ? 5.0f : 3.5f,
                        kind == 2 ? (Color){251, 146, 60, 255} : (Color){125, 211, 252, 255});
        }
    }

    for (int i = 0; i < env->enemy_high_water; i++) {
        const TdEnemy *enemy = &env->enemies[i];
        if (!enemy->alive) {
            continue;
        }
        Vector2 p = td_render_enemy_point(enemy, render_phase);
        Color color = td_enemy_color(enemy->type);
        Texture2D enemy_texture = client->enemies[enemy->type];
        if (td_texture_loaded(enemy_texture)) {
            int frame = ((int)(now * 10.0) + enemy->id) & 3;
            float size = TD_ENEMY_RADIUS[enemy->type] * 3.1f + 9.0f;
            td_draw_sprite(enemy_texture, td_sheet_source(enemy_texture, frame), p, size, 0.0f,
                           WHITE);
        } else {
            DrawCircleV(p, enemy->type == 9 ? 11 : 8, color);
        }
        if (enemy->camo) {
            DrawCircleLines((int)p.x, (int)p.y, 13, (Color){34, 211, 238, 255});
        }
        if (enemy->fortified) {
            DrawCircleLines((int)p.x, (int)p.y, 16, (Color){251, 146, 60, 255});
        }
        DrawRectangle((int)p.x - 10, (int)p.y - 17, 20, 3, (Color){31, 41, 55, 255});
        DrawRectangle((int)p.x - 10, (int)p.y - 17,
                      (int)(20 * td_clampf(enemy->hp / fmaxf(1.0f, enemy->max_hp), 0.0f, 1.0f)), 3,
                      (Color){34, 197, 94, 255});
    }

    for (int i = 0; i < TD_MAX_IMPACT_EVENTS; i++) {
        if (now >= client->impacts[i].until) {
            continue;
        }
        int kind = client->impacts[i].kind;
        if (kind < 0 || kind >= TD_NUM_TOWER_TYPES) {
            continue;
        }
        float alpha =
            (float)td_clampf((float)((client->impacts[i].until - now) / 0.14), 0.0f, 1.0f);
        DrawLineEx((Vector2){client->impacts[i].from_x, client->impacts[i].from_y},
                   (Vector2){client->impacts[i].x, client->impacts[i].y},
                   kind == 2 ? 4.0f : 2.5f, td_projectile_trail_color(kind, 0.55f * alpha));
        Texture2D texture = client->projectiles[kind];
        if (!td_texture_loaded(texture)) {
            DrawCircleV((Vector2){client->impacts[i].x, client->impacts[i].y},
                        kind == 2 ? 6.0f : 4.0f, td_projectile_trail_color(kind, alpha));
            continue;
        }
        float size = kind == 2 ? 30.0f : 22.0f;
        td_draw_sprite(texture, td_sheet_source(texture, 3),
                       (Vector2){client->impacts[i].x, client->impacts[i].y}, size, 0.0f,
                       Fade(WHITE, alpha));
    }

    DrawRectangle(0, TD_HEIGHT, TD_WIDTH, TD_HUD_HEIGHT, (Color){3, 7, 18, 255});
    DrawText(TextFormat("Round %d  Lives %.0f  Cash %.0f  Cleared %.0f", env->round, env->lives,
                        env->cash, env->score),
             18, TD_HEIGHT + 10, 20, RAYWHITE);
    static const char *tower_names[TD_NUM_TOWER_TYPES] = {"DART", "SNIPER", "CANNON"};
    const char *status = "EXTERNAL POLICY CONTROL";
    const char *controls = "";
    if (client->manual_controls_enabled) {
        if (client->human_control) {
            status = TextFormat("MANUAL CONTROL | Tower: %s | Keep SHIFT held",
                                tower_names[client->tower_kind]);
            controls = "1 Dart | 2 Sniper | 3 Cannon | Click place | Q/W/E upgrade | "
                       "RMB/X sell | Space wave";
        } else if (client->policy_loaded) {
            status = "LIVE POLICY | Hold SHIFT to play";
            controls = "While holding SHIFT: 1 Dart | 2 Sniper | 3 Cannon";
        } else {
            status = "NO POLICY WEIGHTS | Hold SHIFT to play";
            controls = "While holding SHIFT: 1 Dart | 2 Sniper | 3 Cannon";
        }
    }
    Color status_color = (Color){148, 163, 184, 255};
    if (client->manual_controls_enabled) {
        status_color = client->human_control
                           ? (Color){251, 191, 36, 255}
                           : (client->policy_loaded ? (Color){74, 222, 128, 255}
                                                    : (Color){248, 113, 113, 255});
    }
    DrawText(status, 18, TD_HEIGHT + 42, 20, status_color);
    DrawText(controls, 18, TD_HEIGHT + 74, 20, (Color){226, 232, 240, 255});
    EndDrawing();
}

static void c_close(TowerDefence *env) {
    td_unload_render_assets(env);
    free(env->projectiles);
    env->projectiles = NULL;
    env->projectile_count = 0;
    env->projectile_capacity = 0;
}
