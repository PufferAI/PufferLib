#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"

#define TD_DEFAULT_MAX_EPISODE_STEPS 12000
#define TD_DEFAULT_BASE_SEED 1
#define TD_DEFAULT_INVALID_ACTION_REWARD (-0.25f)
#define TD_WIDTH 960
#define TD_HEIGHT 540
#define TD_NUM_PLACEMENT_SLOTS 20
#define TD_NUM_UPGRADE_PATHS 3
#define TD_NUM_SLOT_FEATURES_PER_SLOT 4
#define TD_NUM_SLOT_FEATURES (TD_NUM_PLACEMENT_SLOTS * TD_NUM_SLOT_FEATURES_PER_SLOT)
#define TD_NUM_ENEMY_PROGRESS_BINS 8
#define TD_NUM_ENEMY_PROGRESS_FEATURES_PER_BIN 2
#define TD_NUM_ENEMY_PROGRESS_FEATURES (TD_NUM_ENEMY_PROGRESS_BINS * TD_NUM_ENEMY_PROGRESS_FEATURES_PER_BIN)
#define TD_NUM_UPGRADE_ACTIONS (TD_NUM_PLACEMENT_SLOTS * TD_NUM_UPGRADE_PATHS)
#define TD_NUM_SELL_ACTIONS TD_NUM_PLACEMENT_SLOTS
#define TD_SCALAR_OBS_SIZE 16
#define TD_NUM_ACTIONS (1 + TD_NUM_PLACEMENT_SLOTS + TD_NUM_UPGRADE_ACTIONS + TD_NUM_SELL_ACTIONS + 1)
#define TD_BASE_OBS_SIZE (TD_SCALAR_OBS_SIZE + TD_NUM_SLOT_FEATURES + TD_NUM_ENEMY_PROGRESS_FEATURES)
#define TD_NUM_OBSERVATION_V2_FEATURES 64
#define TD_OBS_V2_FEATURE_OFFSET TD_BASE_OBS_SIZE
#define TD_OBS_V2_ACTION_MASK_OFFSET (TD_OBS_V2_FEATURE_OFFSET + TD_NUM_OBSERVATION_V2_FEATURES)
#define TD_OBS_V2_SIZE (TD_OBS_V2_ACTION_MASK_OFFSET + TD_NUM_ACTIONS)
#define TD_OBS_ACTION_MASK_OFFSET TD_OBS_V2_ACTION_MASK_OFFSET
#define TD_OBS_SIZE TD_OBS_V2_SIZE
#define TD_FACTOR_NUM_ATNS 3
#define TD_FACTOR_VERB_COUNT 5
#define TD_MAX_ENEMIES 256
#define TD_MAX_SPAWNS 4
#define TD_MAX_TIER 4
#define TD_REPEAT_UPGRADE_UNLOCK_ROUND 31
#define TD_MAX_ACTIVE_PATHS 2
#define TD_DT 0.25f

#ifndef TD_USE_FACTORED_ACTIONS
#define TD_USE_FACTORED_ACTIONS 1
#endif

enum TdActionId {
    TD_ACTION_NOOP = 0,
    TD_ACTION_DART_SLOT_01 = 1,
    TD_ACTION_DART_SLOT_02 = 2,
    TD_ACTION_DART_SLOT_03 = 3,
    TD_ACTION_DART_SLOT_04 = 4,
    TD_ACTION_DART_SLOT_05 = 5,
    TD_ACTION_DART_SLOT_06 = 6,
    TD_ACTION_DART_SLOT_07 = 7,
    TD_ACTION_DART_SLOT_08 = 8,
    TD_ACTION_DART_SLOT_09 = 9,
    TD_ACTION_DART_SLOT_10 = 10,
    TD_ACTION_DART_SLOT_11 = 11,
    TD_ACTION_DART_SLOT_12 = 12,
    TD_ACTION_SNIPER_SLOT_01 = 13,
    TD_ACTION_SNIPER_SLOT_02 = 14,
    TD_ACTION_SNIPER_SLOT_03 = 15,
    TD_ACTION_SNIPER_SLOT_04 = 16,
    TD_ACTION_CANNON_SLOT_01 = 17,
    TD_ACTION_CANNON_SLOT_02 = 18,
    TD_ACTION_CANNON_SLOT_03 = 19,
    TD_ACTION_CANNON_SLOT_04 = 20,
    TD_ACTION_UPGRADE_SLOT_01_TOP = 21,
    TD_ACTION_SELL_SLOT_01 = 81,
    TD_ACTION_TRIGGER_NEXT_ROUND = 101,
    TD_ACTION_LAST = TD_NUM_ACTIONS - 1,
};

enum TdStatusCode {
    TD_STATUS_WARMUP = 0,
    TD_STATUS_SPAWNING = 1,
    TD_STATUS_INTERMISSION = 2,
    TD_STATUS_ACTIVE = 3,
    TD_STATUS_DEFEAT = 4,
    TD_STATUS_COMPLETE = 5,
};

enum TdFactorVerb {
    TD_FACTOR_VERB_NOOP = 0,
    TD_FACTOR_VERB_PLACE = 1,
    TD_FACTOR_VERB_UPGRADE = 2,
    TD_FACTOR_VERB_SELL = 3,
    TD_FACTOR_VERB_TRIGGER = 4,
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
    int kind;
    float x;
    float y;
} TdSlotSpec;

typedef struct {
    int alive;
    int kind;
    int upgrades[TD_NUM_UPGRADE_PATHS];
    float x;
    float y;
    float cooldown;
    float invested;
    float fire_anim;
} TdTower;

typedef struct {
    int alive;
    int type;
    int camo;
    int fortified;
    int regrow;
    float progress;
    float hp;
    float max_hp;
    float speed;
    float burn_dps;
    float burn_time;
    float slow_mult;
    float slow_time;
} TdEnemy;

typedef struct {
    int type;
    int count;
    int emitted;
    float interval;
    float next_time;
    int camo;
    int fortified;
    int regrow;
} TdSpawn;

typedef struct {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    int rng;

    int max_episode_steps;
    int base_seed;
    int episode_index;
    int step_count;
    int invalid_action_count;
    int valid_actions[TD_NUM_ACTIONS];
    float invalid_action_reward;
    float episode_return;
    float latest_score;
    float latest_perf;

    unsigned int rng_state;
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
    int last_shot_tower;
    int last_shot_enemy;
    float last_shot_time;
    int hover_slot;
    int human_control;
} TowerDefence;

static const TdSlotSpec TD_SLOTS[TD_NUM_PLACEMENT_SLOTS] = {
    {0, 20, 70}, {0, 100, 70}, {0, 220, 70}, {0, 300, 70},
    {0, 100, 146}, {0, 340, 146}, {0, 432, 152}, {0, 520, 210},
    {0, 600, 180}, {0, 710, 330}, {0, 620, 421}, {0, 860, 421},
    {1, 420, 30}, {1, 350, 400}, {1, 770, 180}, {1, 920, 520},
    {2, 220, 190}, {2, 370, 350}, {2, 500, 340}, {2, 600, 310},
};

static const float TD_PATH_X[] = {0, 384, 384, 672, 672, 960};
static const float TD_PATH_Y[] = {108, 108, 270, 270, 459, 459};
static const float TD_TOWER_COST[] = {100, 260, 420};
static const float TD_TOWER_RANGE[] = {140, 500, 160};
static const float TD_TOWER_FIRE_RATE[] = {0.7f, 1.35f, 1.6f};
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
static const int TD_ENEMY_SHARP_IMMUNE[] = {0, 0, 0, 0, 0, 0, 0, 1, 0, 0};
static const int TD_ENEMY_EXPLOSIVE_IMMUNE[] = {0, 0, 0, 0, 0, 1, 0, 0, 1, 0};
static const int TD_ENEMY_BURN_IMMUNE[] = {0, 0, 0, 0, 0, 1, 0, 0, 1, 0};
static const int TD_ENEMY_SLOW_IMMUNE[] = {0, 0, 0, 0, 0, 0, 1, 0, 1, 0};
static const int TD_ENEMY_CHILD_A[] = {-1, 0, 1, 2, 3, 4, 4, 5, 5, 8};
static const int TD_ENEMY_CHILD_B[] = {-1, -1, -1, -1, -1, -1, -1, -1, 6, 8};

static void c_reset(TowerDefence* env);

static float td_clampf(float value, float lo, float hi) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

static float td_squash(float value, float scale) {
    return tanhf(fmaxf(0.0f, value) / fmaxf(0.0001f, scale));
}

static unsigned int td_rand(TowerDefence* env) {
    env->rng_state = env->rng_state * 1664525u + 1013904223u;
    return env->rng_state;
}

static float td_randf(TowerDefence* env) {
    return (td_rand(env) >> 8) / 16777216.0f;
}

static int td_normalize_action(float raw_action) {
    if (!isfinite(raw_action)) return TD_ACTION_NOOP;
    int action = (int)lrintf(raw_action);
    return (action < 0 || action >= TD_NUM_ACTIONS) ? TD_ACTION_NOOP : action;
}

static int td_normalize_factor_component(float raw_action, int size) {
    if (!isfinite(raw_action)) return 0;
    int action = (int)lrintf(raw_action);
    return (action < 0 || action >= size) ? 0 : action;
}

static int td_decode_action(const TowerDefence* env) {
#if TD_USE_FACTORED_ACTIONS
    int verb = td_normalize_factor_component(env->actions[0], TD_FACTOR_VERB_COUNT);
    int slot = td_normalize_factor_component(env->actions[1], TD_NUM_PLACEMENT_SLOTS);
    int path = td_normalize_factor_component(env->actions[2], TD_NUM_UPGRADE_PATHS);
    if (verb == TD_FACTOR_VERB_PLACE) return 1 + slot;
    if (verb == TD_FACTOR_VERB_UPGRADE) return 1 + TD_NUM_PLACEMENT_SLOTS + slot * TD_NUM_UPGRADE_PATHS + path;
    if (verb == TD_FACTOR_VERB_SELL) return 1 + TD_NUM_PLACEMENT_SLOTS + TD_NUM_UPGRADE_ACTIONS + slot;
    if (verb == TD_FACTOR_VERB_TRIGGER) return TD_ACTION_TRIGGER_NEXT_ROUND;
    return TD_ACTION_NOOP;
#else
    return td_normalize_action(env->actions[0]);
#endif
}

static int td_tower_count(const TowerDefence* env) {
    int count = 0;
    for (int i = 0; i < TD_NUM_PLACEMENT_SLOTS; i++) count += env->towers[i].alive;
    return count;
}

static int td_enemy_count(const TowerDefence* env) {
    int count = 0;
    for (int i = 0; i < TD_MAX_ENEMIES; i++) count += env->enemies[i].alive;
    return count;
}

static float td_path_length(void) {
    float length = 0.0f;
    for (int i = 0; i < 5; i++) {
        float dx = TD_PATH_X[i + 1] - TD_PATH_X[i];
        float dy = TD_PATH_Y[i + 1] - TD_PATH_Y[i];
        length += sqrtf(dx * dx + dy * dy);
    }
    return length;
}

static Vector2 td_path_point(float progress) {
    float distance = td_clampf(progress, 0.0f, 1.0f) * td_path_length();
    for (int i = 0; i < 5; i++) {
        float dx = TD_PATH_X[i + 1] - TD_PATH_X[i];
        float dy = TD_PATH_Y[i + 1] - TD_PATH_Y[i];
        float seg = sqrtf(dx * dx + dy * dy);
        if (distance <= seg || i == 4) {
            float t = seg > 0 ? distance / seg : 0.0f;
            return (Vector2){TD_PATH_X[i] + dx * t, TD_PATH_Y[i] + dy * t};
        }
        distance -= seg;
    }
    return (Vector2){TD_PATH_X[5], TD_PATH_Y[5]};
}

static int td_active_paths(const TdTower* tower) {
    int count = 0;
    for (int p = 0; p < TD_NUM_UPGRADE_PATHS; p++) count += tower->upgrades[p] > 0;
    return count;
}

static float td_upgrade_cost(const TdTower* tower, int path) {
    int next = tower->upgrades[path] + 1;
    return (float)(TD_UPGRADE_COST[tower->kind][path] * next * next);
}

static void td_tower_stats(const TdTower* tower, float* range, float* damage,
        float* fire_rate, int* damage_type, int* detect_camo, float* burn_dps,
        float* burn_time, float* slow_mult, float* slow_time) {
    int kind = tower->kind;
    *range = TD_TOWER_RANGE[kind];
    *damage = TD_TOWER_DAMAGE[kind];
    *fire_rate = TD_TOWER_FIRE_RATE[kind];
    *damage_type = TD_TOWER_DAMAGE_TYPE[kind];
    *detect_camo = 0;
    *burn_dps = kind == 2 ? 1.4f : 0.0f;
    *burn_time = kind == 2 ? 2.2f : 0.0f;
    *slow_mult = kind == 1 ? 0.72f : 1.0f;
    *slow_time = kind == 1 ? 1.2f : 0.0f;

    for (int t = 0; t < tower->upgrades[0]; t++) {
        *damage += kind == 0 ? 1.0f : (kind == 1 ? 2.0f : 2.0f);
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
        if (kind != 0) *detect_camo = 1;
    }
    for (int t = 0; t < tower->upgrades[2]; t++) {
        *fire_rate *= kind == 1 ? 0.82f : 0.8f;
    }
}

static int td_can_upgrade(const TowerDefence* env, int slot, int path) {
    if (slot < 0 || slot >= TD_NUM_PLACEMENT_SLOTS || path < 0 || path >= TD_NUM_UPGRADE_PATHS) return 0;
    const TdTower* tower = &env->towers[slot];
    if (!tower->alive || tower->upgrades[path] >= TD_MAX_TIER) return 0;
    if (tower->upgrades[path] > 0 && env->round < TD_REPEAT_UPGRADE_UNLOCK_ROUND) return 0;
    if (tower->upgrades[path] == 0 && td_active_paths(tower) >= TD_MAX_ACTIVE_PATHS) return 0;
    return env->cash >= td_upgrade_cost(tower, path);
}

static void td_update_masks(TowerDefence* env) {
    memset(env->valid_actions, 0, sizeof(env->valid_actions));
    env->valid_actions[TD_ACTION_NOOP] = 1;
    for (int slot = 0; slot < TD_NUM_PLACEMENT_SLOTS; slot++) {
        int kind = TD_SLOTS[slot].kind;
        env->valid_actions[1 + slot] = !env->towers[slot].alive && env->cash >= TD_TOWER_COST[kind];
        env->valid_actions[TD_ACTION_SELL_SLOT_01 + slot] = env->towers[slot].alive;
        for (int path = 0; path < TD_NUM_UPGRADE_PATHS; path++) {
            env->valid_actions[TD_ACTION_UPGRADE_SLOT_01_TOP + slot * TD_NUM_UPGRADE_PATHS + path] =
                td_can_upgrade(env, slot, path);
        }
    }
    env->valid_actions[TD_ACTION_TRIGGER_NEXT_ROUND] =
        env->status_code == TD_STATUS_WARMUP || env->status_code == TD_STATUS_INTERMISSION;
}

static void td_init(TowerDefence* env) {
    env->num_agents = 1;
    env->max_episode_steps = TD_DEFAULT_MAX_EPISODE_STEPS;
    env->base_seed = TD_DEFAULT_BASE_SEED;
    env->invalid_action_reward = TD_DEFAULT_INVALID_ACTION_REWARD;
    env->hover_slot = -1;
}

static void allocate(TowerDefence* env) {
    td_init(env);
    env->observations = (float*)calloc(TD_OBS_SIZE, sizeof(float));
    env->actions = (float*)calloc(TD_USE_FACTORED_ACTIONS ? TD_FACTOR_NUM_ATNS : 1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (float*)calloc(1, sizeof(float));
}

static void free_allocated(TowerDefence* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
}

static void td_write_observation(TowerDefence* env) {
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
    env->observations[15] = env->time - env->last_shot_time < 0.35f ? 1.0f : 0.0f;

    int idx = TD_SCALAR_OBS_SIZE;
    for (int slot = 0; slot < TD_NUM_PLACEMENT_SLOTS; slot++) {
        TdTower* tower = &env->towers[slot];
        env->observations[idx++] = tower->alive ? 1.0f : 0.0f;
        env->observations[idx++] = tower->alive ? fminf(1.0f, tower->upgrades[0] / (float)TD_MAX_TIER) : 0.0f;
        env->observations[idx++] = tower->alive ? fminf(1.0f, tower->upgrades[1] / (float)TD_MAX_TIER) : 0.0f;
        env->observations[idx++] = tower->alive ? fminf(1.0f, tower->upgrades[2] / (float)TD_MAX_TIER) : 0.0f;
    }

    float count_bins[TD_NUM_ENEMY_PROGRESS_BINS] = {0};
    float hp_bins[TD_NUM_ENEMY_PROGRESS_BINS] = {0};
    float type_mass[10] = {0};
    float prop_mass[8] = {0};
    float band_mass[4][4] = {{0}};
    for (int i = 0; i < TD_MAX_ENEMIES; i++) {
        TdEnemy* enemy = &env->enemies[i];
        if (!enemy->alive) continue;
        int bin = (int)fminf(TD_NUM_ENEMY_PROGRESS_BINS - 1, floorf(enemy->progress * TD_NUM_ENEMY_PROGRESS_BINS));
        int band = (int)fminf(3, floorf(enemy->progress * 4));
        float hp = fmaxf(0.0f, enemy->hp);
        float leak = TD_ENEMY_LEAK[enemy->type];
        count_bins[bin] += 1.0f;
        hp_bins[bin] += fminf(1.0f, hp / fmaxf(1.0f, enemy->max_hp));
        type_mass[enemy->type] += fmaxf(1.0f, hp);
        prop_mass[0] += enemy->camo ? hp : 0.0f;
        prop_mass[1] += enemy->fortified ? hp : 0.0f;
        prop_mass[2] += enemy->regrow ? hp : 0.0f;
        prop_mass[3] += TD_ENEMY_SHARP_IMMUNE[enemy->type] ? hp : 0.0f;
        prop_mass[4] += TD_ENEMY_EXPLOSIVE_IMMUNE[enemy->type] ? hp : 0.0f;
        prop_mass[5] += TD_ENEMY_BURN_IMMUNE[enemy->type] ? hp : 0.0f;
        prop_mass[6] += TD_ENEMY_SLOW_IMMUNE[enemy->type] ? hp : 0.0f;
        prop_mass[7] += enemy->type == 9 ? hp : 0.0f;
        band_mass[band][0] += 1.0f;
        band_mass[band][1] += hp;
        band_mass[band][2] += leak;
        band_mass[band][3] += (enemy->camo || TD_ENEMY_SHARP_IMMUNE[enemy->type] ||
            TD_ENEMY_EXPLOSIVE_IMMUNE[enemy->type]) ? leak : 0.0f;
    }
    for (int i = 0; i < TD_NUM_ENEMY_PROGRESS_BINS; i++) env->observations[idx++] = td_squash(count_bins[i], 3.0f);
    for (int i = 0; i < TD_NUM_ENEMY_PROGRESS_BINS; i++) env->observations[idx++] = td_squash(hp_bins[i], 3.0f);

    idx = TD_OBS_V2_FEATURE_OFFSET;
    for (int i = 0; i < 10; i++) env->observations[idx++] = td_squash(type_mass[i], 12.0f);
    for (int i = 0; i < 8; i++) env->observations[idx++] = td_squash(prop_mass[i], 12.0f);
    for (int band = 0; band < 4; band++) {
        env->observations[idx++] = td_squash(band_mass[band][0], 4.0f);
        env->observations[idx++] = td_squash(band_mass[band][1], 24.0f);
        env->observations[idx++] = td_squash(band_mass[band][2], 10.0f);
        env->observations[idx++] = td_squash(band_mass[band][3], 10.0f);
    }
    float comp[10] = {0};
    for (int slot = 0; slot < TD_NUM_PLACEMENT_SLOTS; slot++) {
        TdTower* tower = &env->towers[slot];
        env->observations[idx++] = tower->alive ? fminf(1.0f, tower->invested / 3000.0f) : 0.0f;
        if (!tower->alive) continue;
        float range, damage, fire_rate, burn_dps, burn_time, slow_mult, slow_time;
        int damage_type, detect_camo;
        td_tower_stats(tower, &range, &damage, &fire_rate, &damage_type, &detect_camo,
            &burn_dps, &burn_time, &slow_mult, &slow_time);
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
    for (int action = 0; action < TD_NUM_ACTIONS; action++) {
        env->observations[TD_OBS_ACTION_MASK_OFFSET + action] = (float)env->valid_actions[action];
    }
}

static int td_spawn_done(const TowerDefence* env) {
    for (int i = 0; i < env->active_spawns; i++) {
        if (env->spawns[i].emitted < env->spawns[i].count) return 0;
    }
    return 1;
}

static int td_add_enemy(TowerDefence* env, int type, int camo, int fortified, int regrow, float progress) {
    for (int i = 0; i < TD_MAX_ENEMIES; i++) {
        TdEnemy* enemy = &env->enemies[i];
        if (enemy->alive) continue;
        float round_scale = env->round > 20 ? 1.0f + 0.018f * (env->round - 20) : 1.0f;
        enemy->alive = 1;
        enemy->type = type;
        enemy->camo = camo;
        enemy->fortified = fortified;
        enemy->regrow = regrow;
        enemy->progress = progress;
        enemy->max_hp = TD_ENEMY_HP[type] * round_scale * (fortified ? 1.8f : 1.0f);
        enemy->hp = enemy->max_hp;
        enemy->speed = TD_ENEMY_SPEED[type] * (1.0f + 0.002f * fmaxf(0.0f, env->round - 20));
        enemy->burn_dps = 0.0f;
        enemy->burn_time = 0.0f;
        enemy->slow_mult = 1.0f;
        enemy->slow_time = 0.0f;
        return i;
    }
    return -1;
}

static void td_add_spawn(TowerDefence* env, int type, int count, float interval,
        int camo, int fortified, int regrow) {
    if (env->active_spawns >= TD_MAX_SPAWNS) return;
    TdSpawn* spawn = &env->spawns[env->active_spawns++];
    spawn->type = type;
    spawn->count = count;
    spawn->emitted = 0;
    spawn->interval = interval;
    spawn->next_time = env->wave_elapsed + 0.01f * env->active_spawns;
    spawn->camo = camo;
    spawn->fortified = fortified;
    spawn->regrow = regrow;
}

static void td_prepare_wave(TowerDefence* env) {
    env->active_spawns = 0;
    env->wave_elapsed = 0.0f;
    int r = env->round;
    if (r == 1) td_add_spawn(env, 0, 20, 0.6f, 0, 0, 0);
    else if (r == 2) td_add_spawn(env, 0, 30, 0.5f, 0, 0, 0);
    else if (r <= 5) {
        td_add_spawn(env, 0, 12 + r * 2, 0.5f, 0, 0, 0);
        td_add_spawn(env, r >= 3 ? 1 : 0, 6 + r, 0.55f, 0, 0, 0);
    } else if (r <= 10) {
        td_add_spawn(env, 2 + (r > 7), 12 + r, 0.45f, 0, 0, 0);
        td_add_spawn(env, r >= 10 ? 5 : 4, 6 + r / 2, 0.42f, 0, 0, 0);
    } else if (r <= 20) {
        td_add_spawn(env, (r % 3) + 4, 8 + r / 2, 0.42f, 0, 0, 0);
        td_add_spawn(env, r >= 12 ? 7 : 5, 3 + r / 3, 0.55f, 0, 0, 0);
        if (r >= 13) td_add_spawn(env, 8, 2 + r / 5, 0.58f, 0, 0, 0);
        if (r == 20) td_add_spawn(env, 9, 1, 0.01f, 0, 0, 0);
    } else {
        int base = r < 36 ? 1 : (r < 61 ? 3 : 5);
        int groups = r < 36 ? 2 : (r < 70 ? 3 : 4);
        int count = r <= 80 ? (int)fminf(150.0f / groups, 12.0f + r * 0.45f)
                            : (int)fminf(180.0f / groups, 36.0f + r * 0.04f);
        for (int g = 0; g < groups; g++) {
            int type = (base + g + (td_rand(env) % 3)) % 10;
            if (r > 100 && type == 9) type = 8;
            int camo = r > 32 && td_randf(env) < fminf(0.32f, 0.03f + r * 0.0015f);
            int fortified = r > 45 && td_randf(env) < fminf(0.28f, 0.02f + r * 0.0012f);
            int regrow = r > 28 && td_randf(env) < fminf(0.25f, 0.02f + r * 0.001f);
            td_add_spawn(env, type, count, fmaxf(0.28f, 0.52f - r * 0.002f), camo, fortified, regrow);
        }
    }
}

static void td_start_round(TowerDefence* env) {
    if (env->status_code != TD_STATUS_WARMUP && env->status_code != TD_STATUS_INTERMISSION) return;
    env->status_code = TD_STATUS_SPAWNING;
    env->intermission_remaining = 0.0f;
    td_prepare_wave(env);
}

static void td_kill_enemy(TowerDefence* env, int idx, float* reward) {
    TdEnemy* enemy = &env->enemies[idx];
    int child_a = TD_ENEMY_CHILD_A[enemy->type];
    int child_b = TD_ENEMY_CHILD_B[enemy->type];
    float progress = enemy->progress;
    int camo = enemy->camo;
    int regrow = enemy->regrow;
    *reward += TD_ENEMY_REWARD[enemy->type] * 0.01f;
    env->cash += TD_ENEMY_REWARD[enemy->type];
    enemy->alive = 0;
    if (child_a >= 0) td_add_enemy(env, child_a, camo, 0, regrow, progress);
    if (child_b >= 0) td_add_enemy(env, child_b, camo, 0, regrow, progress);
}

static void td_apply_tower_damage(TowerDefence* env, float* reward) {
    for (int slot = 0; slot < TD_NUM_PLACEMENT_SLOTS; slot++) {
        TdTower* tower = &env->towers[slot];
        if (!tower->alive) continue;
        tower->cooldown -= TD_DT;
        tower->fire_anim = fmaxf(0.0f, tower->fire_anim - TD_DT);
        if (tower->cooldown > 0.0f) continue;

        float range, damage, fire_rate, burn_dps, burn_time, slow_mult, slow_time;
        int damage_type, detect_camo;
        td_tower_stats(tower, &range, &damage, &fire_rate, &damage_type, &detect_camo,
            &burn_dps, &burn_time, &slow_mult, &slow_time);
        int target = -1;
        float best_progress = -1.0f;
        for (int i = 0; i < TD_MAX_ENEMIES; i++) {
            TdEnemy* enemy = &env->enemies[i];
            if (!enemy->alive) continue;
            if (enemy->camo && !detect_camo) continue;
            if (damage_type == 0 && TD_ENEMY_SHARP_IMMUNE[enemy->type]) continue;
            if (damage_type == 1 && TD_ENEMY_EXPLOSIVE_IMMUNE[enemy->type]) continue;
            Vector2 p = td_path_point(enemy->progress);
            float dx = p.x - tower->x;
            float dy = p.y - tower->y;
            if (sqrtf(dx * dx + dy * dy) <= range && enemy->progress > best_progress) {
                target = i;
                best_progress = enemy->progress;
            }
        }
        if (target < 0) continue;
        TdEnemy* enemy = &env->enemies[target];
        enemy->hp -= damage;
        if (burn_dps > 0.0f && !TD_ENEMY_BURN_IMMUNE[enemy->type]) {
            enemy->burn_dps = fmaxf(enemy->burn_dps, burn_dps);
            enemy->burn_time = fmaxf(enemy->burn_time, burn_time);
        }
        if (slow_mult < 1.0f && !TD_ENEMY_SLOW_IMMUNE[enemy->type]) {
            enemy->slow_mult = fminf(enemy->slow_mult, slow_mult);
            enemy->slow_time = fmaxf(enemy->slow_time, slow_time);
        }
        tower->cooldown = fire_rate;
        tower->fire_anim = 0.22f;
        env->last_shot_tower = slot;
        env->last_shot_enemy = target;
        env->last_shot_time = env->time;
        if (enemy->hp <= 0.0f) td_kill_enemy(env, target, reward);
    }
}

static float td_round_clear_bonus(const TowerDefence* env) {
    float rbe = 170.0f + fminf(120.0f, fmaxf(0.0f, env->round - 20) * 3.5f);
    return floorf(42.0f + 4.0f * env->round + 0.052f * rbe);
}

static void td_advance_world(TowerDefence* env, float* reward) {
    env->time += TD_DT;
    if (env->status_code == TD_STATUS_INTERMISSION) {
        env->intermission_remaining = fmaxf(0.0f, env->intermission_remaining - TD_DT);
        return;
    }
    if (env->status_code != TD_STATUS_SPAWNING && env->status_code != TD_STATUS_ACTIVE) return;

    env->wave_elapsed += TD_DT;
    for (int s = 0; s < env->active_spawns; s++) {
        TdSpawn* spawn = &env->spawns[s];
        while (spawn->emitted < spawn->count && env->wave_elapsed >= spawn->next_time) {
            if (td_add_enemy(env, spawn->type, spawn->camo, spawn->fortified, spawn->regrow, 0.0f) < 0) {
                break;
            }
            spawn->emitted += 1;
            spawn->next_time += spawn->interval;
        }
    }
    if (td_spawn_done(env)) env->status_code = TD_STATUS_ACTIVE;

    for (int i = 0; i < TD_MAX_ENEMIES; i++) {
        TdEnemy* enemy = &env->enemies[i];
        if (!enemy->alive) continue;
        if (enemy->burn_time > 0.0f) {
            enemy->hp -= enemy->burn_dps * TD_DT * (enemy->type == 9 ? 0.85f : 1.0f);
            enemy->burn_time = fmaxf(0.0f, enemy->burn_time - TD_DT);
        }
        if (enemy->slow_time > 0.0f) {
            enemy->slow_time = fmaxf(0.0f, enemy->slow_time - TD_DT);
        } else {
            enemy->slow_mult = 1.0f;
        }
        if (enemy->regrow && enemy->hp < enemy->max_hp) {
            enemy->hp = fminf(enemy->max_hp, enemy->hp + 0.25f * TD_DT);
        }
        if (enemy->hp <= 0.0f) {
            td_kill_enemy(env, i, reward);
            continue;
        }
        enemy->progress += (enemy->speed * enemy->slow_mult * TD_DT) / td_path_length();
        if (enemy->progress >= 1.0f) {
            env->lives -= TD_ENEMY_LEAK[enemy->type];
            *reward -= TD_ENEMY_LEAK[enemy->type] * 0.05f;
            enemy->alive = 0;
        }
    }
    td_apply_tower_damage(env, reward);

    if (env->lives <= 0.0f) {
        env->status_code = TD_STATUS_DEFEAT;
        return;
    }
    if (td_spawn_done(env) && td_enemy_count(env) == 0) {
        float bonus = td_round_clear_bonus(env);
        env->cash += bonus;
        *reward += 1.0f;
        env->score = (float)env->round;
        env->round += 1;
        env->status_code = env->round > 200 ? TD_STATUS_COMPLETE : TD_STATUS_INTERMISSION;
        env->intermission_remaining = 2.0f;
    }
}

static void td_place(TowerDefence* env, int slot) {
    int kind = TD_SLOTS[slot].kind;
    TdTower* tower = &env->towers[slot];
    tower->alive = 1;
    tower->kind = kind;
    tower->x = TD_SLOTS[slot].x;
    tower->y = TD_SLOTS[slot].y;
    tower->cooldown = 0.0f;
    tower->invested = TD_TOWER_COST[kind];
    tower->fire_anim = 0.0f;
    memset(tower->upgrades, 0, sizeof(tower->upgrades));
    env->cash -= TD_TOWER_COST[kind];
}

static void td_upgrade(TowerDefence* env, int slot, int path) {
    TdTower* tower = &env->towers[slot];
    float cost = td_upgrade_cost(tower, path);
    tower->upgrades[path] += 1;
    tower->invested += cost;
    env->cash -= cost;
}

static void td_sell(TowerDefence* env, int slot) {
    TdTower* tower = &env->towers[slot];
    env->cash += floorf(tower->invested * 0.7f);
    memset(tower, 0, sizeof(*tower));
}

static int td_apply_action(TowerDefence* env, int action) {
    td_update_masks(env);
    if (action < 0 || action >= TD_NUM_ACTIONS || !env->valid_actions[action]) return 0;
    if (action >= 1 && action <= TD_NUM_PLACEMENT_SLOTS) td_place(env, action - 1);
    else if (action >= TD_ACTION_UPGRADE_SLOT_01_TOP && action < TD_ACTION_SELL_SLOT_01) {
        int idx = action - TD_ACTION_UPGRADE_SLOT_01_TOP;
        td_upgrade(env, idx / TD_NUM_UPGRADE_PATHS, idx % TD_NUM_UPGRADE_PATHS);
    } else if (action >= TD_ACTION_SELL_SLOT_01 && action < TD_ACTION_TRIGGER_NEXT_ROUND) {
        td_sell(env, action - TD_ACTION_SELL_SLOT_01);
    } else if (action == TD_ACTION_TRIGGER_NEXT_ROUND) td_start_round(env);
    return 1;
}

static void td_log_episode(TowerDefence* env) {
    float steps = env->step_count > 0 ? (float)env->step_count : 1.0f;
    env->log.perf += fminf(1.0f, env->score / 200.0f);
    env->log.score += env->score;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->step_count;
    env->log.invalid_action_rate += (float)env->invalid_action_count / steps;
    env->log.n += 1.0f;
}

static void c_reset(TowerDefence* env) {
    env->rng_state = (unsigned int)(env->base_seed + env->rng * 100003 + env->episode_index++);
    env->step_count = 0;
    env->invalid_action_count = 0;
    env->episode_return = 0.0f;
    env->latest_score = 0.0f;
    env->latest_perf = 0.0f;
    env->round = 1;
    env->status_code = TD_STATUS_WARMUP;
    env->time = 0.0f;
    env->lives = 200.0f;
    env->cash = 10000.0f;
    env->intermission_remaining = 0.0f;
    env->wave_elapsed = 0.0f;
    env->score = 0.0f;
    env->active_spawns = 0;
    env->last_shot_tower = -1;
    env->last_shot_enemy = -1;
    env->last_shot_time = -999.0f;
    memset(env->spawns, 0, sizeof(env->spawns));
    memset(env->towers, 0, sizeof(env->towers));
    memset(env->enemies, 0, sizeof(env->enemies));
    td_write_observation(env);
}

static void c_step(TowerDefence* env) {
    int action = td_decode_action(env);
    float reward = 0.0f;
    int valid = td_apply_action(env, action);
    if (!valid) {
        env->invalid_action_count += 1;
        reward += env->invalid_action_reward;
    }
    td_advance_world(env, &reward);
    env->step_count += 1;
    env->episode_return += reward;
    env->latest_score = env->score;
    env->latest_perf = fminf(1.0f, env->score / 200.0f);
    env->rewards[0] = reward;
    env->terminals[0] = 0.0f;
    int done = env->status_code == TD_STATUS_DEFEAT ||
        env->status_code == TD_STATUS_COMPLETE ||
        env->step_count >= env->max_episode_steps;
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
        {239, 68, 68, 255}, {59, 130, 246, 255}, {34, 197, 94, 255},
        {250, 204, 21, 255}, {244, 114, 182, 255}, {17, 24, 39, 255},
        {229, 231, 235, 255}, {107, 114, 128, 255}, {249, 250, 251, 255},
        {217, 119, 6, 255},
    };
    return colors[type];
}

static void c_render(TowerDefence* env) {
    if (!IsWindowReady()) {
        InitWindow(TD_WIDTH, TD_HEIGHT + 80, "PufferLib Tower Defence");
        SetTargetFPS(60);
    }
    if (IsKeyDown(KEY_ESCAPE)) exit(0);

    BeginDrawing();
    ClearBackground((Color){12, 18, 24, 255});
    for (int y = 0; y < TD_HEIGHT; y += 36) {
        DrawLine(0, y, TD_WIDTH, y, (Color){24, 32, 42, 255});
    }
    for (int i = 0; i < 5; i++) {
        Vector2 a = {TD_PATH_X[i], TD_PATH_Y[i]};
        Vector2 b = {TD_PATH_X[i + 1], TD_PATH_Y[i + 1]};
        DrawLineEx(a, b, 58, (Color){55, 65, 81, 255});
        DrawLineEx(a, b, 42, (Color){17, 24, 39, 255});
        DrawLineEx(a, b, 6, (Color){251, 146, 60, 255});
    }

    env->human_control = IsKeyDown(KEY_LEFT_SHIFT);
    env->hover_slot = -1;
    Vector2 mouse = GetMousePosition();
    float best = 999999.0f;
    for (int slot = 0; slot < TD_NUM_PLACEMENT_SLOTS; slot++) {
        float dx = mouse.x - TD_SLOTS[slot].x;
        float dy = mouse.y - TD_SLOTS[slot].y;
        float d = dx * dx + dy * dy;
        if (d < best) {
            best = d;
            env->hover_slot = slot;
        }
    }

    for (int slot = 0; slot < TD_NUM_PLACEMENT_SLOTS; slot++) {
        TdTower* tower = &env->towers[slot];
        int kind = tower->alive ? tower->kind : TD_SLOTS[slot].kind;
        Color color = kind == 0 ? (Color){59, 130, 246, 255}
            : (kind == 1 ? (Color){229, 231, 235, 255} : (Color){156, 163, 175, 255});
        Vector2 p = {TD_SLOTS[slot].x, TD_SLOTS[slot].y};
        if (!tower->alive) {
            DrawCircleLines((int)p.x, (int)p.y, TD_TOWER_RADIUS[kind], (Color){75, 85, 99, 160});
            continue;
        }
        if (slot == env->hover_slot || tower->fire_anim > 0.0f) {
            float range, damage, fire_rate, burn_dps, burn_time, slow_mult, slow_time;
            int damage_type, detect_camo;
            td_tower_stats(tower, &range, &damage, &fire_rate, &damage_type, &detect_camo,
                &burn_dps, &burn_time, &slow_mult, &slow_time);
            DrawCircleLines((int)p.x, (int)p.y, range, (Color){96, 165, 250, 80});
        }
        DrawCircleV(p, (float)TD_TOWER_RADIUS[kind] + 3, (Color){15, 23, 42, 255});
        DrawCircleV(p, (float)TD_TOWER_RADIUS[kind], color);
        DrawRectangle((int)p.x - 7, (int)p.y - 3, 14, 6, (Color){251, 146, 60, 255});
    }

    if (env->time - env->last_shot_time < 0.18f &&
            env->last_shot_tower >= 0 && env->last_shot_enemy >= 0) {
        TdTower* tower = &env->towers[env->last_shot_tower];
        TdEnemy* enemy = &env->enemies[env->last_shot_enemy];
        if (tower->alive && enemy->alive) {
            DrawLineEx((Vector2){tower->x, tower->y}, td_path_point(enemy->progress),
                3, (Color){251, 191, 36, 220});
        }
    }

    for (int i = 0; i < TD_MAX_ENEMIES; i++) {
        TdEnemy* enemy = &env->enemies[i];
        if (!enemy->alive) continue;
        Vector2 p = td_path_point(enemy->progress);
        Color color = td_enemy_color(enemy->type);
        DrawCircleV(p, enemy->type == 9 ? 11 : 8, color);
        if (enemy->camo) DrawCircleLines((int)p.x, (int)p.y, 13, (Color){34, 211, 238, 255});
        if (enemy->fortified) DrawCircleLines((int)p.x, (int)p.y, 16, (Color){251, 146, 60, 255});
        DrawRectangle((int)p.x - 10, (int)p.y - 17, 20, 3, (Color){31, 41, 55, 255});
        DrawRectangle((int)p.x - 10, (int)p.y - 17,
            (int)(20 * td_clampf(enemy->hp / fmaxf(1.0f, enemy->max_hp), 0.0f, 1.0f)),
            3, (Color){34, 197, 94, 255});
    }

    DrawRectangle(0, TD_HEIGHT, TD_WIDTH, 80, (Color){3, 7, 18, 255});
    DrawText(TextFormat("Round %d  Lives %.0f  Cash %.0f  Score %.0f", env->round, env->lives, env->cash, env->score),
        18, TD_HEIGHT + 12, 22, RAYWHITE);
    DrawText(env->human_control ? "SHIFT takeover: LMB place, Q/W/E upgrade, RMB/X sell, Space trigger"
                                : "Policy playback",
        18, TD_HEIGHT + 44, 18, env->human_control ? (Color){251, 191, 36, 255} : (Color){148, 163, 184, 255});
    EndDrawing();
}

static void c_close(TowerDefence* env) {
    (void)env;
    if (IsWindowReady()) CloseWindow();
}
