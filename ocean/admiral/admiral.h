#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"

#define NUM_ACTIONS 3
#if PUF_BACKEND != PUF_GPU
#define MY_VEC_INIT
#endif

// TODO add predictable wind patterns the policy can learn.
// TODO add a spatial wind field.

#define ENVS_PER_LEVEL 32
#define FAVORABLE_WIND_LEVELS 14
#define FLAWLESS_LEVELS 7
#define FULL_KILL_LEVELS 7
#define MASTERY_ENVS_BIN 100
#define MASTERY_WINS 90

#define DT 0.5f                     // seconds per tick
#define COOLDOWN_PER_TICK 0.5f
#define CANNON_MAX_DAMAGE 0.1f
#define CANNON_RANGE 400.0f         // meters
#define K_SAIL 4.0f                 // more or less air density * normal-force coefficient (kg/m^3)
#define GLOBAL_OBS_FEATURES 3
#define MASS 3000000.0f             // 3M kg
#define MAX_HEALTH 1.0f
#define MAX_RUDDER 0.523598776f     // radians
#define RUDDER_RATE 0.523598776f    // radians per second
#define SAIL_RATE 0.785398163f      // radians per second
#define MAX_TURN_RATE 0.049649123f  // radians per second
#define MAX_SAIL_ANGLE 0.785398163f // radians
#define MS 5.66f                    // m/s   MAX_SPEED
#define STATIONARY_SPEED_THRESHOLD (0.01f * MS) // m/s
#define N_TEAMS 2                   // Number of teams
#define H_PI 1.570796327f           // half pi
#define Q_PI 0.785398163f           // quarter pi
#define HULL_DRAG_COEFF 7056.0f     // N / (m/s)^2
#define DAMAGE_DRAG_SCALE 12.0f     // unitless
#define RUDDER_DRAG_RATE 0.02f      // 1/s at full rudder and max speed
#define SAIL_AREA 6000.0f           // square meters
#define SHIP_LENGTH 57              // meters
#define SHIP_OBS_FEATURES 11
#define SHIP_SPACING 200.0f         // meters
#define SHIP_SPACING_SPAN 50.0f     // meters
#define RELATIVE_POSITION_FEATURES 2
#define SAIL_OBS_FEATURES 5
#define SHIP_WIDTH 16               // meters
#define SHIPS_PER_TEAM 5
#define RENDER_SHOT_TICKS 6
#define NUM_SHIPS (N_TEAMS * SHIPS_PER_TEAM)
#define TWO_PI 6.283185307f
#define D2R 0.017453292519943295f // Degrees to Radians

#define BASE_OBS_SIZE (GLOBAL_OBS_FEATURES + NUM_SHIPS * SHIP_OBS_FEATURES)
#define RELATIVE_OBS_SIZE (SHIPS_PER_TEAM * SHIPS_PER_TEAM * RELATIVE_POSITION_FEATURES)
#define SAIL_OBS_START (BASE_OBS_SIZE + RELATIVE_OBS_SIZE)
#define SAIL_OBS_SIZE (SHIPS_PER_TEAM * SAIL_OBS_FEATURES)
#define OBS_SIZE (SAIL_OBS_START + SAIL_OBS_SIZE)
#define SHIP_ACT_SIZES 5, 5, 3
#define NUM_ATNS (NUM_ACTIONS * SHIPS_PER_TEAM)
#define ACT_SIZES { \
    SHIP_ACT_SIZES, \
    SHIP_ACT_SIZES, \
    SHIP_ACT_SIZES, \
    SHIP_ACT_SIZES, \
    SHIP_ACT_SIZES \
}

typedef struct {
    int level;
    const char* description;
    float separation;           // meters
    float bearing;              // degrees
    float bearing_jitter;       // degrees
    float enemy_heading;        // relative to advantaged ship heading
    float enemy_heading_jitter; // degrees
    float adv_speed;            // Multiple of MS (max speed)
    float adv_speed_jitter;     // Multiple of MS (max speed)
    float enemy_speed;          // Multiple of MS (max speed)
    float enemy_speed_jitter;   // Multiple of MS (max speed)
    float wind_heading;         // direction wind moves toward, relative to advantaged ship heading
    float wind_heading_jitter;  // degrees
    float wind_speed;           // m/s
    float damage;               // multiple
    float enemy_health;         // fraction of MAX_HEALTH
    int ticks;
} CurriculumConfig;

static const float RUDDER_VALUES[5] = {-1.0f, -0.25f, 0, 0.25f, 1.0f};

static const float SAIL_ANGLE_VALUES[5] = {-1.0f, -0.25f, 0, 0.25f, 1.0f};

static const float FIRE_VALUES[3] = {-1.0f, 0, 1.0f};

static const CurriculumConfig CURRICULUM[] = {
    //  sep(m)  brg    jitter nme-hdg jitter Aspd  Ajit  Bspd  Bjit  wind  jitter windspd damage  health ticks
    {1, "Cardinal broadside",
        150.0f, 90.0f,  5.0f, 225.0f, 15.0f, 0.5f, 0.4f, 0.0f, 0.2f, 75.0f, 15.0f, 10.0f,  3.0f, 0.50f,    10},
    {2, "10 degree turn-to-acquire",
        150.0f, 80.0f, 10.0f, 225.0f, 20.0f, 0.5f, 0.3f, 0.1f, 0.2f, 75.0f, 15.0f, 10.0f,  2.5f, 0.75f,    50},
    {3, "Zero-speed sail-to-fire, 3 m",
        164.0f, 79.0f,  0.0f,   3.0f,  1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 90.0f, 15.0f, 15.0f,  4.0f, 0.02f,    15},
    {4, "Zero-speed sail-to-fire, 20 m",
        275.0f, 80.0f,  0.5f,   0.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 90.0f,  0.0f, 10.0f,  3.0f, 0.02f,    50},
    {5, "20 degree turn-to-acquire",
        100.0f, 70.0f, 10.0f, 225.0f, 15.0f, 0.5f, 0.0f, 0.1f, 0.0f, 60.0f, 30.0f, 10.0f,  2.0f, 0.50f,    50},
    {6, "Close parallel acquisition",
        100.0f, 60.0f, 10.0f,   0.0f, 20.0f, 0.5f, 0.0f, 0.0f, 0.0f, 60.0f, 30.0f, 10.0f,  2.5f, 0.10f,    60},
    {7, "Close moving pursuit",
        150.0f, 60.0f, 10.0f,   0.0f, 20.0f, 1.0f, 0.0f, 0.2f, 0.0f, 60.0f, 30.0f, 10.0f,  2.5f, 0.10f,    70},
    {8, "In-range coupled-wind pursuit, 10 percent health target",
        300.0f, 60.0f, 15.0f,   0.0f,  5.0f, 1.0f, 0.0f, 0.1f, 0.0f, 60.0f, 15.0f, 10.0f,  1.0f, 0.10f,   140},
    {9, "Close broadside, 50 percent health target",
        100.0f, 90.0f,  5.0f, 225.0f, 15.0f, 0.6f, 0.0f, 0.3f, 0.0f, 75.0f, 15.0f, 10.0f,  2.0f, 0.50f,    20},
    {10, "Cannon-boundary coupled-wind pursuit, 10 percent health target",
        400.0f, 60.0f, 30.0f,   0.0f, 30.0f, 1.0f, 0.2f, 0.2f, 0.1f, 60.0f, 30.0f, 15.0f,  1.5f, 0.10f,   240},
    {11, "Close broadside, 75 percent health target",
        100.0f, 90.0f,  5.0f, 225.0f, 15.0f, 0.6f, 0.0f, 0.3f, 0.0f, 75.0f, 15.0f, 10.0f,  2.0f, 0.75f,    30},
    {12, "Outside-range coupled-wind pursuit, 2 percent health target",
        450.0f, 90.0f, 30.0f,   0.0f, 30.0f, 0.5f, 0.3f, 0.1f, 0.1f, 60.0f, 20.0f, 20.0f,  2.0f, 0.02f,   240},
    {13, "Coupled-wind reciprocal intercept, 10 percent health target",
        500.0f, 60.0f, 30.0f, 180.0f, 10.0f, 1.0f, 0.0f, 0.1f, 0.0f, 60.0f, 10.0f, 15.0f,  2.5f, 0.10f,   280},
    {14, "Coupled-wind pursuit, 25 percent health target",
        500.0f, 60.0f, 30.0f,   0.0f, 10.0f, 1.0f, 0.0f, 0.1f, 0.0f, 60.0f, 10.0f, 15.0f,  2.5f, 0.25f,   360},
    {15, "Independent-wind reciprocal intercept, 10 percent health target",
        500.0f, 60.0f, 30.0f, 180.0f, 10.0f, 1.0f, 0.0f, 0.1f, 0.0f, 60.0f, 10.0f, 15.0f,  2.5f, 0.10f,   280},
    {16, "Independent-wind close broadside, 15 degree reply",
        100.0f, 90.0f,  5.0f, 195.0f,  0.0f, 0.5f, 0.0f, 0.1f, 0.0f, 90.0f,  0.0f, 10.0f,  2.5f, 1.00f,   130},
    {17, "Independent-wind close reciprocal broadside, slow enemy",
        100.0f, 90.0f,  5.0f, 180.0f,  0.0f, 0.5f, 0.0f, 0.1f, 0.0f, 90.0f,  0.0f, 10.0f,  2.5f, 1.00f,   140},
    {18, "Independent-wind pursuit, 25 percent health target",
        500.0f, 60.0f, 30.0f,   0.0f, 10.0f, 1.0f, 0.0f, 0.1f, 0.0f, 60.0f, 10.0f, 15.0f,  2.5f, 0.25f,   360},
    {19, "Independent-wind variable-speed reciprocal intercept",
        500.0f, 60.0f, 30.0f, 180.0f, 10.0f, 1.0f, 0.3f, 0.2f, 0.1f, 60.0f, 10.0f, 10.0f,  2.5f, 0.25f,   360},
    {20, "Symmetric variable-speed reciprocal intercept",
        350.0f, 60.0f, 30.0f, 180.0f, 10.0f, 1.0f, 0.3f, 1.0f, 0.3f, 60.0f, 10.0f, 10.0f,  1.0f, 1.00f,   300},
    //  sep(m)  brg    jitter nme-hdg jitter Aspd  Ajit  Bspd  Bjit  wind  jitter windspd damage  health ticks
};

#define MAX_LEVEL (sizeof CURRICULUM / sizeof *CURRICULUM)

static inline int curriculum_level_for_env(int env_id, int current_level, int num_levels) {
    if (current_level == num_levels) return current_level;
    if (env_id < MASTERY_ENVS_BIN) return current_level;
    int level = (env_id - MASTERY_ENVS_BIN) / ENVS_PER_LEVEL + 1;
    if (level >= current_level) level++;
    return level <= num_levels ? level : current_level;
}

static inline int curriculum_perf_slot(int env_id, int level, int num_levels) {
    if (env_id < ENVS_PER_LEVEL) return (level - 1) * ENVS_PER_LEVEL + env_id;
    if (env_id < MASTERY_ENVS_BIN) return -1;
    int level_env_offset = env_id - MASTERY_ENVS_BIN;
    if (level_env_offset >= (num_levels - 1) * ENVS_PER_LEVEL) return -1;
    return (level - 1) * ENVS_PER_LEVEL + level_env_offset % ENVS_PER_LEVEL;
}

static inline float curriculum_perf(const unsigned char* level_wins, int mastered_level) {
    float accum = 0.0f;
    float max_perf = 0.0f;
    float mastered = mastered_level / (float)MAX_LEVEL;
    float mastered2 = mastered * mastered;
    for (int level = 1; level <= MAX_LEVEL; level++) {
        int level_start = (level - 1) * ENVS_PER_LEVEL;
        int wins = 0;
        for (int slot = 0; slot < ENVS_PER_LEVEL; slot++) {
            wins += level_wins[level_start + slot];
        }
        float win_rate = wins / (float)ENVS_PER_LEVEL;
        float level_weight = level / (float)MAX_LEVEL;
        accum += mastered2 * win_rate * win_rate * level_weight * level_weight;
        max_perf += level_weight * level_weight;
    }
    return accum / max_perf;
}

static inline bool cannon_ray_hits_ship(
    float ray_x, float ray_y,
    float ray_dx, float ray_dy,
    float ship_x, float ship_y,
    float ship_heading,
    float* hit_distance
) {
    float half_length = SHIP_LENGTH * 0.5f;
    float centerline_dx = cos(ship_heading) * half_length;
    float centerline_dy = sin(ship_heading) * half_length;
    float ax = ship_x - centerline_dx;
    float ay = ship_y - centerline_dy;
    float bx = ship_x + centerline_dx;
    float by = ship_y + centerline_dy;

    float segment_dx = bx - ax;
    float segment_dy = by - ay;
    float denominator = ray_dx * segment_dy - ray_dy * segment_dx;
    if (denominator == 0.0f) return false;

    float offset_x = ax - ray_x;
    float offset_y = ay - ray_y;
    float inverse_denominator = 1.0f / denominator;
    float ray_distance = (offset_x * segment_dy - offset_y * segment_dx) * inverse_denominator;
    float segment_fraction = (offset_x * ray_dy - offset_y * ray_dx) * inverse_denominator;

    if (ray_distance < 0.0f || ray_distance > CANNON_RANGE) return false;
    if (segment_fraction < 0.0f || segment_fraction > 1.0f) return false;

    *hit_distance = ray_distance;
    return true;
}

typedef struct Log Log;
struct Log {
    float kills;
    float adv_kills;
    float episode_return;
    float episode_length;
    float score;            // damage dealt this episode
    float policy_0_score;
    float policy_1_score;
    float draw_rate;
    float curr_level;
    float curr_win_rate;
    float curr_mastered_level;
    float n;
};

typedef struct Ship Ship;
struct Ship {
    float x;
    float y;
    float cooldown_left;
    float cooldown_right;
    float heading;
    float health;
    float health_old;
    float rudder;
    float sail_angle;
    float speed;
    float vx;
    float vy;
    float render_shot_heading;
    int render_shot_ticks;
    int team_idx;
};

typedef struct Client Client;
struct Client {};

#if PUF_BACKEND != PUF_GPU
typedef struct {
    unsigned char wins[MASTERY_ENVS_BIN];
    unsigned char level_wins[MAX_LEVEL * ENVS_PER_LEVEL];
    int num_wins;
    int mastered_level;
    int total_games;
    Env* envs;
    int num_envs;
    int stepped;
} Curriculum;

static Curriculum curriculum;

typedef Env Admiral;

struct Env {
    Client* client;
    Agent agents[N_TEAMS];

    int env_id;
    int num_agents;
    int tick;
    int max_ticks;
    int width;
    int height;
    int curr_level;
    int curr_adv_team;
    int next_adv_team;
    int curr_fire_side;
    int curr_side_episodes;
    int curr_wind_sign;
    float spawn_bearing;
    float spawn_rotation;
    float spawn_spacing;
    float damage_mult;
    float wind_vx;
    float wind_vy;
    int team_kills[N_TEAMS];
    Ship* ships;
    Log log;
    Log* logs;

    float reward_damage_mult;
    float reward_kill;
    float penalty_hit_ally;
    float penalty_used_volley;
    float penalty_stationary;

    // Selfplay-pool tagging. tag = 0 means pure selfplay (both slots = primary
    // policy). tag > 0 means historical: slot 0 = primary, slot 1 = frozen
    // historical opponent. boundary_reached is set when a historical game ends.
    int tag;
    int boundary_reached;

    unsigned int rng;
};

static inline void rotate_spawn(Admiral* env, float center_x, float center_y, float rotation) {
    float rotation_cos = cosf(rotation);
    float rotation_sin = sinf(rotation);
    float wind_x = env->wind_vx;
    float wind_y = env->wind_vy;
    env->wind_vx = wind_x * rotation_cos - wind_y * rotation_sin;
    env->wind_vy = wind_x * rotation_sin + wind_y * rotation_cos;

    for (int idx = 0; idx < NUM_SHIPS; idx++) {
        Ship* ship = &env->ships[idx];
        float x = ship->x - center_x;
        float y = ship->y - center_y;
        float vx = ship->vx;
        float vy = ship->vy;
        ship->x = center_x + x * rotation_cos - y * rotation_sin;
        ship->y = center_y + x * rotation_sin + y * rotation_cos;
        ship->vx = vx * rotation_cos - vy * rotation_sin;
        ship->vy = vx * rotation_sin + vy * rotation_cos;
        ship->heading = remainderf(ship->heading + rotation, TWO_PI);
        if (ship->heading < 0.0f) ship->heading += TWO_PI;
    }
}

void init(Admiral* env){
    int spawn_variant = env->rng % (4 * N_TEAMS);
    int pair_variant = spawn_variant / N_TEAMS;
    env->next_adv_team = spawn_variant % N_TEAMS;
    env->curr_fire_side = pair_variant < 2 ? -1 : 1;
    env->curr_side_episodes = 0;
    env->curr_wind_sign = pair_variant % 2 == 0 ? 1 : -1;
    env->spawn_bearing = 0.0f;
    env->spawn_rotation = 0.0f;
    env->num_agents = N_TEAMS;
    env->ships = (Ship*)calloc(NUM_SHIPS, sizeof(Ship));
    env->logs = (Log*)calloc(env->num_agents, sizeof(Log));
}

static inline float admiral_get_float(Dict* kwargs, const char* key, float default_value) {
    for (int i = 0; i < kwargs->size; i++) {
        if (strcmp(kwargs->items[i].key, key) == 0) {
            return (float)kwargs->items[i].value;
        }
    }
    return default_value;
}

void puf_init(Env* env, Dict* kwargs) {
    env->width = dict_get(kwargs, "width");
    env->height = dict_get(kwargs, "height");
    env->curr_level = (int)admiral_get_float(kwargs, "curriculum_level", 1);
    env->reward_damage_mult = admiral_get_float(kwargs, "reward_damage_mult", 0.0f);
    env->reward_kill = admiral_get_float(kwargs, "reward_kill", 0.0f);
    env->penalty_hit_ally = admiral_get_float(kwargs, "penalty_hit_ally", 0.0f);
    env->penalty_used_volley = admiral_get_float(kwargs, "penalty_used_volley", 0.0f);
    env->penalty_stationary = admiral_get_float(kwargs, "penalty_stationary", 0.0f);
    env->agents[0].policy = 0;
    env->agents[1].policy = 1;
    init(env);
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", curriculum_perf(curriculum.level_wins, curriculum.mastered_level));
    dict_set(out, "kills", log->kills);
    dict_set(out, "adv_kills", log->adv_kills);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "policy_0_score", log->policy_0_score);
    dict_set(out, "policy_1_score", log->policy_1_score);
    dict_set(out, "draw_rate", log->draw_rate);
    dict_set(out, "curr_level", log->curr_level);
    dict_set(out, "curr_win_rate", log->curr_win_rate);
    dict_set(out, "curr_mastered_level", log->curr_mastered_level);
    dict_set(out, "total_games", curriculum.total_games);
    dict_set(out, "n", log->n);
}

void puf_close(Admiral* env) {
    free(env->ships);
    free(env->logs);
}

Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents");
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers");
    assert(num_buffers == 1 && "Admiral curriculum requires one buffer");
    int num_envs = total_agents / N_TEAMS;
    int curriculum_level = (int)admiral_get_float(env_kwargs, "curriculum_level", 1);
    int num_levels = MAX_LEVEL;
    if (dict_get(vec_kwargs, "hist_policy_percent") == 1.0f) curriculum_level = num_levels;

    curriculum = (Curriculum){0};
    curriculum.mastered_level = curriculum_level - 1;

    Env* envs = (Env*)calloc(num_envs, sizeof(Env));
    buffer_env_starts[0] = 0;
    buffer_env_counts[0] = num_envs;
    for (int i = 0; i < num_envs; i++) {
        Env* env = &envs[i];
        env->env_id = i;
        env->rng = i;
        puf_init(env, env_kwargs);
        env->curr_level = curriculum_level_for_env(i, curriculum_level, num_levels);
    }

    curriculum.envs = envs;
    curriculum.num_envs = num_envs;
    *num_envs_out = num_envs;
    return envs;
}

void add_log(Admiral* env) {
    for (int i = 0; i < N_TEAMS; i++) {
        env->log.kills           += env->team_kills[i];
        env->log.episode_return  += env->logs[i].episode_return;
        env->log.episode_length  += env->logs[i].episode_length;
        env->log.score           += env->logs[i].score;
        env->log.n               += 1.0f;
    }
}

static inline void record_curriculum_result(Admiral* env, bool mastery_win) {
    int num_levels = MAX_LEVEL;
    int perf_slot = curriculum_perf_slot(env->env_id, env->curr_level, num_levels);
    if (perf_slot >= 0) curriculum.level_wins[perf_slot] = mastery_win;

    curriculum.total_games += 1;
    int env_id = env->env_id;
    if (env_id < MASTERY_ENVS_BIN && env->curr_level > curriculum.mastered_level) {
        curriculum.num_wins = curriculum.num_wins - curriculum.wins[env_id] + mastery_win;
        curriculum.wins[env_id] = mastery_win;
        if (curriculum.num_wins >= MASTERY_WINS) {
            curriculum.mastered_level = env->curr_level;
            curriculum.num_wins = 0;
            memset(curriculum.wins, 0, sizeof curriculum.wins);
        }
    }

    env->log.curr_mastered_level += curriculum.mastered_level * env->num_agents;
    if (curriculum.mastered_level == num_levels) {
        env->curr_level = num_levels;
    } else {
        env->curr_level = curriculum_level_for_env(
            env_id, curriculum.mastered_level + 1, num_levels);
    }
}

static inline void add_agent_reward(Admiral* env, int agent_idx, float reward) {
    *env->agents[agent_idx].rewards += reward;
    env->logs[agent_idx].episode_return += reward;
}

static inline int team_kill_outcome(Admiral* env) {
    if (env->team_kills[0] > env->team_kills[1]) return +1;
    if (env->team_kills[1] > env->team_kills[0]) return -1;
    return 0;
}

void move(Admiral* env, Ship* ship) {
    float dx = cos(ship->heading);
    float dy = sin(ship->heading);

    float rel_vx = env->wind_vx - ship->vx;
    float rel_vy = env->wind_vy - ship->vy;
    float rel_speed = sqrtf(rel_vx * rel_vx + rel_vy * rel_vy);

    float sail_world_angle = ship->sail_angle + ship->heading;
    float sail_world_nx = cos(sail_world_angle);
    float sail_world_ny = sin(sail_world_angle);
    float normal_speed = rel_vx * sail_world_nx + rel_vy * sail_world_ny;
    float forward_projection = sail_world_nx * dx + sail_world_ny * dy;
    float F_forward = 0.5f * K_SAIL * SAIL_AREA * rel_speed * normal_speed * forward_projection * ship->health;

    float rudder_fraction = ship->rudder / MAX_RUDDER;
    float speed = ship->speed;
    float damage = 1.0f - ship->health;
    float drag = HULL_DRAG_COEFF
        * (1.0f + DAMAGE_DRAG_SCALE * damage * damage) * speed * speed
        + (MASS * RUDDER_DRAG_RATE * rudder_fraction * rudder_fraction * speed * speed) / MS;
    float acceleration = (F_forward - drag) / MASS;
    ship->speed += acceleration * DT;
    ship->speed = fmaxf(0.0f, fminf(ship->speed, MS));
    if (ship->health > 0.0f && ship->speed < STATIONARY_SPEED_THRESHOLD) {
        add_agent_reward(env, ship->team_idx, env->penalty_stationary);
    }
    ship->vx = ship->speed * dx;
    ship->vy = ship->speed * dy;

    ship->x += ship->speed * dx * DT;
    ship->y += ship->speed * dy * DT;
}

void turn(float* heading, float radians, float d_angle) {
    if (radians > d_angle) {
        radians = d_angle;
    } else if (radians < -d_angle) {
        radians = -d_angle;
    }

    *heading += radians;
    if (*heading >= TWO_PI) {
        *heading -= TWO_PI;
    } else if (*heading < 0.0f) {
        *heading += TWO_PI;
    }
}

void fire(Admiral* env, Ship* ship, int ship_idx, int fire_side) {
    add_agent_reward(env, ship->team_idx, env->penalty_used_volley);
    if (fire_side < 0) {
        if (ship->cooldown_left < 0.9999f) return;
        ship->cooldown_left = 0.0f;
    } else {
        if (ship->cooldown_right < 0.9999f) return;
        ship->cooldown_right = 0.0f;
    }

    float shot_heading = ship->heading + H_PI * fire_side;
    float ray_dx = cos(shot_heading);
    float ray_dy = sin(shot_heading);
    ship->render_shot_heading = shot_heading;
    ship->render_shot_ticks = RENDER_SHOT_TICKS;

    Ship* hit_ship = NULL;
    float hit_distance = CANNON_RANGE;

    for (int target_idx = 0; target_idx < NUM_SHIPS; target_idx++) {
        if (target_idx == ship_idx) continue;

        Ship* target = &env->ships[target_idx];
        if (target->health <= 0.0f) continue;

        float target_distance;
        if (!cannon_ray_hits_ship(
            ship->x, ship->y, ray_dx, ray_dy, target->x, target->y, target->heading, &target_distance
        )) continue;

        if (target_distance >= hit_distance) continue;
        hit_ship = target;
        hit_distance = target_distance;
    }

    if (hit_ship == NULL) return;
    float range_factor = 1.0f - hit_distance / CANNON_RANGE;
    float damage_mult = ship->team_idx == env->curr_adv_team ? env->damage_mult : 1.0f;
    float damage = fminf(CANNON_MAX_DAMAGE * damage_mult * range_factor, hit_ship->health);
    bool hit_enemy = hit_ship->team_idx != ship->team_idx;
    if (hit_enemy) {
        env->logs[ship->team_idx].score += damage;
        add_agent_reward(env, ship->team_idx, env->reward_damage_mult * damage);
    } else {
        add_agent_reward(env, ship->team_idx, env->penalty_hit_ally * damage);
    }

    if (hit_enemy && damage >= hit_ship->health) {
        add_agent_reward(env, ship->team_idx, env->reward_kill);
        env->team_kills[ship->team_idx]++;
    }
    hit_ship->health -= damage;
}

void compute_observations(Admiral* env) {
    float inverse_diagonal = 1.0f / hypotf(env->width, env->height);
    for (int team = 0; team < env->num_agents; team++) {
        obs_t* obs = (obs_t*)env->agents[team].observations;
        int opponent = 1 - team;
        float flip = team == 0 ? 1.0f : -1.0f;
        int idx = 0;

        obs[idx++] = (float)env->tick / (float)env->max_ticks;
        obs[idx++] = flip * env->wind_vx * 0.1f;
        obs[idx++] = flip * env->wind_vy * 0.1f;

        for (int side = 0; side < env->num_agents; side++) {
            int owner = side == 0 ? team : opponent;

            for (int ship_num = 0; ship_num < SHIPS_PER_TEAM; ship_num++) {
                int ship_idx = owner * SHIPS_PER_TEAM + ship_num;
                Ship* ship = &env->ships[ship_idx];

                // Team 1's observation is rotated 180 degrees
                obs[idx++] = flip * (2.0f * ship->x / (float)env->width - 1.0f);
                obs[idx++] = flip * (2.0f * ship->y / (float)env->height - 1.0f);
                obs[idx++] = flip * ship->vx / MS;
                obs[idx++] = flip * ship->vy / MS;
                obs[idx++] = ship->health;
                obs[idx++] = ship->cooldown_left;
                obs[idx++] = ship->cooldown_right;
                obs[idx++] = ship->rudder / MAX_RUDDER;
                obs[idx++] = ship->sail_angle / MAX_SAIL_ANGLE;
                obs[idx++] = flip * cosf(ship->heading);
                obs[idx++] = flip * sinf(ship->heading);
            }
        }

        for (int own_num = 0; own_num < SHIPS_PER_TEAM; own_num++) {
            Ship* own = &env->ships[team * SHIPS_PER_TEAM + own_num];
            float heading_cos = cosf(own->heading);
            float heading_sin = sinf(own->heading);
            for (int enemy_num = 0; enemy_num < SHIPS_PER_TEAM; enemy_num++) {
                Ship* enemy = &env->ships[opponent * SHIPS_PER_TEAM + enemy_num];
                float dx = enemy->x - own->x;
                float dy = enemy->y - own->y;
                obs[idx++] = (heading_cos * dx + heading_sin * dy) * inverse_diagonal;
                obs[idx++] = (-heading_sin * dx + heading_cos * dy) * inverse_diagonal;
            }
        }

        for (int ship_num = 0; ship_num < SHIPS_PER_TEAM; ship_num++) {
            Ship* ship = &env->ships[team * SHIPS_PER_TEAM + ship_num];
            float heading_cos = cosf(ship->heading);
            float heading_sin = sinf(ship->heading);
            float apparent_x = env->wind_vx - ship->vx;
            float apparent_y = env->wind_vy - ship->vy;
            float apparent_forward = heading_cos * apparent_x + heading_sin * apparent_y;
            float apparent_lateral = -heading_sin * apparent_x + heading_cos * apparent_y;
            float apparent_speed = hypotf(apparent_forward, apparent_lateral);
            float sail_cos = cosf(ship->sail_angle);
            float sail_sin = sinf(ship->sail_angle);
            float incidence = 0.0f;
            if (apparent_speed > 0.0f) {
                incidence = (apparent_forward * sail_cos + apparent_lateral * sail_sin) / apparent_speed;
                incidence = fminf(1.0f, fmaxf(-1.0f, incidence));
            }

            obs[idx++] = apparent_forward * 0.1f;
            obs[idx++] = apparent_lateral * 0.1f;
            obs[idx++] = ship->speed / MS;
            obs[idx++] = incidence * sail_cos * ship->health;
            obs[idx++] = asinf(incidence) / H_PI;
        }
    }
}

static inline float add_jitter(Admiral* env, float center, float span) {
    if (span == 0.0f) return center;
    float sample = (float)rand_r(&env->rng) / ((float)RAND_MAX + 1.0f);
    return center + span * (2.0f * sample - 1.0f);
}

static inline void spawn_curriculum(Admiral* env, int level) {
    const CurriculumConfig* config = &CURRICULUM[level - 1];
    env->damage_mult = config->damage;
    env->max_ticks = config->ticks;
    float center_x = 0.5f * env->width;
    float center_y = 0.5f * env->height;
    env->curr_adv_team = env->next_adv_team;
    int target_team = 1 - env->curr_adv_team;
    float shooter_heading = env->curr_adv_team == 0 ? 0.0f : PI;
    int rotation_level = level <= 3 ? 2 : (level < 6 ? level - 1 : 5);
    if (env->curr_side_episodes % N_TEAMS == 0) {
        env->spawn_bearing = D2R * add_jitter(env, config->bearing, config->bearing_jitter);
        env->spawn_spacing = add_jitter(env, SHIP_SPACING, SHIP_SPACING_SPAN);
        int rotation_idx = (env->env_id / (4 * N_TEAMS) + env->curr_side_episodes / N_TEAMS) % 4;
        float rotation_span = rotation_level < 3 ? 0.0f : Q_PI * (rotation_level - 2) / 3.0f;
        env->spawn_rotation = rotation_idx * H_PI + add_jitter(env, 0.0f, rotation_span);
    }
    float enemy_heading = D2R * add_jitter(env, config->enemy_heading, config->enemy_heading_jitter);
    float wind_heading = D2R * add_jitter(env, config->wind_heading, config->wind_heading_jitter);
    float enemy_direction = shooter_heading + env->curr_fire_side * env->spawn_bearing;
    float separation_x = config->separation * cosf(enemy_direction);
    float separation_y = config->separation * sinf(enemy_direction);
    int wind_side = level <= FAVORABLE_WIND_LEVELS ? env->curr_fire_side : env->curr_wind_sign;
    float wind_direction = shooter_heading + wind_side * wind_heading;
    env->wind_vx = config->wind_speed * cosf(wind_direction);
    env->wind_vy = config->wind_speed * sinf(wind_direction);
    float adv_speed = add_jitter(env, config->adv_speed, config->adv_speed_jitter);
    float enemy_speed = add_jitter(env, config->enemy_speed, config->enemy_speed_jitter);
    if (adv_speed < 0.0f) adv_speed = 0.0f;
    if (enemy_speed < 0.0f) enemy_speed = 0.0f;

    for (int idx = 0; idx < NUM_SHIPS; idx++) {
        env->ships[idx] = (Ship){
            .cooldown_left = 1.0f,
            .cooldown_right = 1.0f,
            .team_idx = idx / SHIPS_PER_TEAM,
        };
    }

    for (int ship_num = 0; ship_num < SHIPS_PER_TEAM; ship_num++) {
        float lane_offset = (ship_num - 0.5f * (SHIPS_PER_TEAM - 1)) * env->spawn_spacing;
        Ship* shooter = &env->ships[env->curr_adv_team * SHIPS_PER_TEAM + ship_num];
        Ship* target = &env->ships[target_team * SHIPS_PER_TEAM + ship_num];

        shooter->x = center_x - 0.5f * separation_x + lane_offset * cosf(shooter_heading);
        shooter->y = center_y - 0.5f * separation_y + lane_offset * sinf(shooter_heading);
        shooter->heading = shooter_heading;
        target->x = shooter->x + separation_x;
        target->y = shooter->y + separation_y;
        target->heading = fmodf(shooter_heading + env->curr_fire_side * enemy_heading, TWO_PI);
        if (target->heading < 0.0f) target->heading += TWO_PI;

        for (int team = 0; team < N_TEAMS; team++) {
            Ship* ship = &env->ships[team * SHIPS_PER_TEAM + ship_num];
            float health = team == env->curr_adv_team ? MAX_HEALTH : config->enemy_health;
            ship->health = health;
            ship->health_old = health;
            ship->speed = MS * (team == env->curr_adv_team ? adv_speed : enemy_speed);
            ship->vx = ship->speed * cosf(ship->heading);
            ship->vy = ship->speed * sinf(ship->heading);
        }
    }

    rotate_spawn(env, center_x, center_y, env->spawn_rotation);

    env->next_adv_team = target_team;
    env->curr_side_episodes++;
    if (env->curr_side_episodes % N_TEAMS == 0) {
        env->curr_wind_sign = -env->curr_wind_sign;
    }
    if (env->curr_side_episodes == 2 * N_TEAMS) {
        env->curr_side_episodes = 0;
        env->curr_fire_side = -env->curr_fire_side;
    }
}

void puf_reset(Admiral* env) {
    env->tick = 0;
    for (int team = 0; team < N_TEAMS; team++) {
        env->logs[team] = (Log){0};
        env->team_kills[team] = 0;
    }

    spawn_curriculum(env, env->curr_level);
    compute_observations(env);
}

// outcome: +1 slot-0 won, -1 slot-0 lost, 0 draw
// Historical accounting only applies when env->tag > 0.
static inline void end_episode(Admiral* env, int outcome) {
    if (outcome == 0) {
        add_agent_reward(env, 0, -1.0f);
        add_agent_reward(env, 1, -1.0f);
    } else {
        float reward = env->curr_level == MAX_LEVEL ? outcome : (env->team_kills[0] - env->team_kills[1]) / (float)SHIPS_PER_TEAM;
        add_agent_reward(env, 0, reward);
        add_agent_reward(env, 1, -reward);
    }

    float s0_score = outcome > 0 ? 1.0f : 0.0f;
    float s1_score = outcome < 0 ? 1.0f : 0.0f;
    env->log.policy_0_score += s0_score * env->num_agents;
    env->log.policy_1_score += s1_score * env->num_agents;
    if (outcome == 0) env->log.draw_rate += env->num_agents;
    int adv_team = env->curr_adv_team;
    bool won = (adv_team == 0 && outcome > 0) || (adv_team == 1 && outcome < 0);
    bool mastery_win = won;
    if (env->curr_level <= FULL_KILL_LEVELS) {
        mastery_win = won && env->team_kills[adv_team] == SHIPS_PER_TEAM;
    }
    if (env->curr_level <= FLAWLESS_LEVELS) {
        mastery_win = mastery_win && env->team_kills[1 - adv_team] == 0;
    }

    env->log.adv_kills += env->team_kills[adv_team] * env->num_agents;
    env->log.curr_level += env->curr_level * env->num_agents;
    env->log.curr_win_rate += won * env->num_agents;
    record_curriculum_result(env, mastery_win);
    if (env->tag > 0) {
        env->boundary_reached = 1;
    }
    for (int a = 0; a < env->num_agents; a++) {
        *env->agents[a].terminals = 1.0f;
    }
    add_log(env);
    puf_reset(env);
}

static void admiral_human_controls(Admiral* env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_RIGHT_SHIFT)) return;
    float* actions = env->agents[0].actions;
    actions[0] = 2.0f;
    actions[1] = 2.0f;
    actions[2] = 1.0f;
    if (IsKeyDown(KEY_W)) actions[1] = 4.0f; // Increase sail angle
    if (IsKeyDown(KEY_S)) actions[1] = 0.0f; // Decrease sail angle
    if (IsKeyDown(KEY_A)) actions[0] = 0.0f; // Rudder left
    if (IsKeyDown(KEY_D)) actions[0] = 4.0f; // Rudder right
    if (IsKeyDown(KEY_Q)) actions[2] = 0.0f; // Fire left broadside
    if (IsKeyDown(KEY_E)) actions[2] = 2.0f; // Fire right broadside
}

bool step(Admiral* env) {
    admiral_human_controls(env);
    for (int a = 0; a < N_TEAMS; a++) {
        *env->agents[a].rewards = 0.0f;
        *env->agents[a].terminals = 0.0f;
    }

    env->tick += 1;
    if (env->tick > env->max_ticks) {
        return true;
    }

    for (int i = 0; i < env->num_agents; i++) {
        float* team_atn = env->agents[i].actions;
        env->logs[i].episode_length += 1.0f;
        for (int j = 0; j < SHIPS_PER_TEAM; j++) {
            int ship_idx = i * SHIPS_PER_TEAM + j;
            Ship* ship = &env->ships[ship_idx];
            if (ship->health_old <= 0.0f) continue;
            float* atn = team_atn + j * NUM_ACTIONS;

            if (ship->render_shot_ticks > 0) {
                ship->render_shot_ticks--;
            }
            if (ship->cooldown_left < 1.0f) {
                ship->cooldown_left = fminf(1.0f, ship->cooldown_left + COOLDOWN_PER_TICK);
            }
            if (ship->cooldown_right < 1.0f) {
                ship->cooldown_right = fminf(1.0f, ship->cooldown_right + COOLDOWN_PER_TICK);
            }

            float rudder_atn = RUDDER_VALUES[(int)atn[0]];
            ship->rudder += rudder_atn * RUDDER_RATE * DT;
            ship->rudder = fmaxf(-MAX_RUDDER, fminf(ship->rudder, MAX_RUDDER));

            float sail_angle_atn = SAIL_ANGLE_VALUES[(int)atn[1]];
            ship->sail_angle += sail_angle_atn * SAIL_RATE * DT;
            ship->sail_angle = fmaxf(-MAX_SAIL_ANGLE, fminf(ship->sail_angle, MAX_SAIL_ANGLE));

            float speed_fraction = ship->speed / MS;
            float rudder_fraction = ship->rudder / MAX_RUDDER;
            float body_turn_radians = MAX_TURN_RATE * speed_fraction * rudder_fraction * DT;
            turn(&ship->heading, body_turn_radians, MAX_TURN_RATE * DT);

            float fire_atn = FIRE_VALUES[(int)atn[2]];
            if (fire_atn != 0.0f) {
                fire(env, ship, ship_idx, fire_atn);
            }

            move(env, ship);

            float px = ship->x;
            float py = ship->y;
            ship->x = fmaxf(0.0f, fminf(ship->x, (float)env->width));
            ship->y = fmaxf(0.0f, fminf(ship->y, (float)env->height));
            bool hit_wall = (ship->x != px) || (ship->y != py);

            if(!hit_wall) continue;
            float wall_dmg = 0.25f;
            ship->health = fmaxf(0.0f, ship->health - wall_dmg);
            ship->speed = 0.0f;
            ship->vx = 0.0f;
            ship->vy = 0.0f;
        }
    }

    if (env->team_kills[0] == SHIPS_PER_TEAM || env->team_kills[1] == SHIPS_PER_TEAM) {
        return true;
    }

    for (int i = 0; i < NUM_SHIPS; i++) {
        env->ships[i].health_old = env->ships[i].health;
    }

    compute_observations(env);
    return false;
}

void puf_step(Admiral* env) {
    *env->agents[0].terminals = step(env);
    if (__atomic_add_fetch(&curriculum.stepped, 1, __ATOMIC_ACQ_REL) != curriculum.num_envs) return;

    // The last arrival finishes terminals in order, before the trainer's loop barrier releases.
    for (int i = 0; i < curriculum.num_envs; i++) {
        Admiral* finished = &curriculum.envs[i];
        if (*finished->agents[0].terminals) end_episode(finished, team_kill_outcome(finished));
    }
    curriculum.stepped = 0;
}

Client* make_client(Admiral* env) {
    InitWindow(env->width / 2, env->height / 2, "PufferLib Ray Admiral");
    SetTargetFPS(80);
    Client* client = (Client*)calloc(1, sizeof(Client));
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void puf_render(Admiral* env) {
    typedef struct {
        Vector2 muzzle;
        Vector2 side;
        int smoke_frames;
    } RenderShot;
    enum { SHIP_TRAIL_LENGTH = 400 };
    typedef struct {
        Vector2 pos[SHIP_TRAIL_LENGTH];
        int index;
        int count;
    } RenderTrail;
    static RenderShot shots[NUM_SHIPS];
    static RenderTrail trails[NUM_SHIPS];
    static int last_episode;

    if (env->client == NULL) {
        memset(shots, 0, sizeof(shots));
        memset(trails, 0, sizeof(trails));
        last_episode = env->curr_side_episodes;
        env->client = make_client(env);
    }
    SetTargetFPS(IsKeyDown(KEY_RIGHT_SHIFT) ? 20 : 80);
    BeginDrawing();
    ClearBackground((Color){6, 6, 120, 255});

    float ship_length = SHIP_LENGTH * 0.5f;
    float ship_width = SHIP_WIDTH * 0.5f;
    float smoke_scale = 5.0f;
    Color trail_color = (Color){0, 180, 255, 255};
    float bow_length = ship_width;
    float body_length = ship_length - bow_length;
    const float mast_offsets[3] = {-body_length * 0.25f, 0.0f, body_length * 0.25f};
    const float sail_half_lengths[3] = {0.75f * ship_width, 1.1f * ship_width, ship_width};

    if (env->curr_side_episodes != last_episode) {
        memset(trails, 0, sizeof(trails));
        last_episode = env->curr_side_episodes;
    }

    for (int team = 0; team < N_TEAMS; team++) {
        for (int i = 0; i < SHIPS_PER_TEAM; i++) {
            int ship_idx = i + team * SHIPS_PER_TEAM;
            Ship ship = env->ships[ship_idx];
            Vector2 pos = {ship.x * 0.5f, ship.y * 0.5f};

            RenderTrail* trail = &trails[ship_idx];
            if (ship.health > 0.0f) {
                trail->pos[trail->index] = pos;
                trail->index = (trail->index + 1) % SHIP_TRAIL_LENGTH;
                if (trail->count < SHIP_TRAIL_LENGTH) trail->count++;
            }

            float trail_alpha = ship.health > 0.0f ? 0.7f : 0.35f;
            for (int j = 0; j < trail->count - 1; j++) {
                int idx0 = (trail->index - j - 1 + SHIP_TRAIL_LENGTH) % SHIP_TRAIL_LENGTH;
                int idx1 = (trail->index - j - 2 + SHIP_TRAIL_LENGTH) % SHIP_TRAIL_LENGTH;
                float alpha = trail_alpha * (trail->count - j) / trail->count;
                DrawLineEx(trail->pos[idx0], trail->pos[idx1], 2.0f, Fade(trail_color, alpha));
            }

            RenderShot* shot = &shots[ship_idx];
            if (ship.health <= 0.0f) {
                shot->smoke_frames = 0;
            } else if (ship.render_shot_ticks > 0 && shot->smoke_frames == 0) {
                Vector2 shot_direction = {cosf(ship.render_shot_heading), sinf(ship.render_shot_heading)};
                shot->side = (Vector2){-shot_direction.y, shot_direction.x};
                shot->muzzle = (Vector2){
                    pos.x + shot_direction.x * (ship_width * 0.5f + 3.0f * smoke_scale),
                    pos.y + shot_direction.y * (ship_width * 0.5f + 3.0f * smoke_scale)
                };
                Vector2 shot_end = {
                    pos.x + shot_direction.x * CANNON_RANGE * 0.5f,
                    pos.y + shot_direction.y * CANNON_RANGE * 0.5f
                };
                DrawLineEx(shot->muzzle, shot_end, 1.5f, Fade(RAYWHITE, 0.5f));
                shot->smoke_frames = 32;
            }

            if (shot->smoke_frames > 0) {
                float alpha = shot->smoke_frames / 32.0f;
                float age = 4.0f * (1.0f - alpha);
                Color smoke = Fade(GRAY, 0.6f * alpha);
                DrawCircleV((Vector2){
                    shot->muzzle.x + smoke_scale * age * shot->side.y,
                    shot->muzzle.y - smoke_scale * age * shot->side.x},
                    smoke_scale * (3.0f + 0.5f * age), smoke);
                DrawCircleV((Vector2){
                    shot->muzzle.x + 2.5f * shot->side.x,
                    shot->muzzle.y + 2.5f * shot->side.y},
                    smoke_scale * (2.5f + 0.25f * age), smoke);
                DrawCircleV((Vector2){
                    shot->muzzle.x - 2.5f * shot->side.x,
                    shot->muzzle.y - 2.5f * shot->side.y},
                    smoke_scale * (2.5f + 0.25f * age), smoke);
                shot->smoke_frames--;
            }

            float ship_alpha = ship.health > 0.0f ? 1.0f : 0.2f;
            if (ship_idx == 0 && IsKeyDown(KEY_RIGHT_SHIFT)) {
                DrawCircleLinesV(pos, ship_length * 0.5f + 8.0f, YELLOW);
            }
            Color ship_color = Fade(team ? GREEN : RED, ship_alpha);
            Vector2 forward = {cosf(ship.heading), sinf(ship.heading)};
            Vector2 side = {-forward.y, forward.x};
            Vector2 body_center = {
                pos.x - 0.5f * bow_length * forward.x,
                pos.y - 0.5f * bow_length * forward.y
            };
            DrawRectanglePro((Rectangle){
                body_center.x, body_center.y, body_length, ship_width},
                (Vector2){body_length * 0.5f, ship_width * 0.5f},
                ship.heading / D2R, ship_color);

            Vector2 bow_base = {
                pos.x + (ship_length * 0.5f - bow_length) * forward.x,
                pos.y + (ship_length * 0.5f - bow_length) * forward.y
            };
            Vector2 bow_tip = {
                pos.x + ship_length * 0.5f * forward.x,
                pos.y + ship_length * 0.5f * forward.y
            };
            Vector2 bow_right = {
                bow_base.x - ship_width * 0.5f * side.x,
                bow_base.y - ship_width * 0.5f * side.y
            };
            Vector2 bow_left = {
                bow_base.x + ship_width * 0.5f * side.x,
                bow_base.y + ship_width * 0.5f * side.y
            };
            DrawTriangle(bow_tip, bow_right, bow_left, ship_color);

            float sail_world_angle = ship.heading + ship.sail_angle;
            Vector2 sail_direction = {-sinf(sail_world_angle), cosf(sail_world_angle)};
            for (int sail = 0; sail < 3; sail++) {
                Vector2 mast = {
                    pos.x + mast_offsets[sail] * forward.x,
                    pos.y + mast_offsets[sail] * forward.y
                };
                Vector2 sail_start = {
                    mast.x - sail_half_lengths[sail] * sail_direction.x,
                    mast.y - sail_half_lengths[sail] * sail_direction.y
                };
                Vector2 sail_end = {
                    mast.x + sail_half_lengths[sail] * sail_direction.x,
                    mast.y + sail_half_lengths[sail] * sail_direction.y
                };
                DrawLineEx(sail_start, sail_end, 2.0f, Fade(RAYWHITE, ship_alpha));
            }

            if (ship.health > 0.0f) {
                float bar_width = SHIP_LENGTH * 0.5f;
                Rectangle health_bar = {
                    pos.x - bar_width * 0.5f,
                    pos.y - ship_length * 0.5f - 6.0f,
                    bar_width, 4.0f
                };
                DrawRectangleRec(health_bar, RED);
                health_bar.width *= ship.health / MAX_HEALTH;
                DrawRectangleRec(health_bar, GREEN);
            }
        }
    }

    const char* tick_text = TextFormat("%i", env->tick);
    DrawText(tick_text, 10, 10, 10, WHITE);
    const char* adv_text = "adv =";
    DrawText(adv_text, 10, 25, 20, WHITE);
    DrawRectangle(10 + MeasureText(adv_text, 20) + 6, 29, 28, 12, env->curr_adv_team == 0 ? RED : GREEN);
    const char* stage_text = TextFormat("Stage %i", env->curr_level);
    DrawText(stage_text, GetScreenWidth() - MeasureText(stage_text, 20) - 10, 10, 20, WHITE);
    Vector2 wind_tip = {GetScreenWidth() / 2 + 2.5f * env->wind_vx, 55.0f + 2.5f * env->wind_vy};
    DrawLineEx((Vector2){GetScreenWidth() / 2 - 2.5f * env->wind_vx, 55.0f - 2.5f * env->wind_vy}, wind_tip, 4.0f, SKYBLUE);
    float wind_angle = atan2f(env->wind_vy, env->wind_vx);
    Vector2 arrow_base = {wind_tip.x - 10.0f * cosf(wind_angle), wind_tip.y - 10.0f * sinf(wind_angle)};
    Vector2 arrow_left = {arrow_base.x - 6.0f * sinf(wind_angle), arrow_base.y + 6.0f * cosf(wind_angle)};
    Vector2 arrow_right = {arrow_base.x + 6.0f * sinf(wind_angle), arrow_base.y - 6.0f * cosf(wind_angle)};
    DrawTriangle(wind_tip, arrow_right, arrow_left, SKYBLUE);

    EndDrawing();
}
#endif  // PUF_BACKEND != PUF_GPU
