#ifndef PUFFER_ADMIRAL_GPU_CU
#define PUFFER_ADMIRAL_GPU_CU

#define PUF_BACKEND PUF_GPU
#include <assert.h>
#include <cuda_runtime.h>
typedef float obs_t;
#include "admiral.h"

#define ADMIRAL_THREADS_PER_ENV 8

typedef struct {
    float separation;
    float bearing;
    float bearing_jitter;
    float enemy_heading;
    float enemy_heading_jitter;
    float adv_speed;
    float adv_speed_jitter;
    float enemy_speed;
    float enemy_speed_jitter;
    float wind_heading;
    float wind_heading_jitter;
    float wind_speed;
    float damage;
    float enemy_health;
    int ticks;
} GpuCurriculumConfig;

typedef struct {
    int width;
    int height;
    int curriculum_level;
    int num_levels;
    float reward_damage;
    float reward_kill;
    float penalty_hit_ally;
    float penalty_used_volley;
    float penalty_stationary;
      GpuCurriculumConfig curriculum[MAX_LEVEL];
} GpuAdmiralConfig;

typedef struct {
      unsigned char wins[MASTERY_ENVS_BIN];
      unsigned char level_wins[MAX_LEVEL * ENVS_PER_LEVEL];
    int num_wins;
    int mastered_level;
    int total_games;
} GpuCurriculum;

struct Env {
    Log log;
    Agent agents[N_TEAMS];
    int num_agents;
    int tag;
    int boundary_reached;
    Log logs[N_TEAMS];
    Ship ships[NUM_SHIPS];
    unsigned int rng;
    int env_id;
    int tick;
    int max_ticks;
    int curr_level;
    int curr_adv_team;
    int next_adv_team;
    int curr_fire_side;
    int curr_side_episodes;
    int curr_wind_sign;
    int team_kills[N_TEAMS];
    float spawn_bearing;
    float spawn_rotation;
    float spawn_spacing;
    float damage_mult;
    float wind_vx;
    float wind_vy;
    unsigned char episode_ended;
    unsigned char mastery_win;
};

static __constant__ GpuAdmiralConfig d_admiral;
static __device__ GpuCurriculum d_curriculum;
static int gpu_render_width;
static int gpu_render_height;
static bool gpu_render_initialized;
static struct {
    int total_agents;
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    cudaStream_t stream;
} g_gpu;
static __constant__ float d_rudder_values[5] = {
    -1.0f, -0.25f, 0.0f, 0.25f, 1.0f
};
static __constant__ float d_sail_angle_values[5] = {
    -1.0f, -0.25f, 0.0f, 0.25f, 1.0f
};
static __constant__ float d_fire_values[3] = {-1.0f, 0.0f, 1.0f};

static __device__ int gpu_curriculum_level_for_env(
        int env_id, int current_level, int num_levels) {
      if (env_id < MASTERY_ENVS_BIN) return current_level;
      int level = (env_id - MASTERY_ENVS_BIN) / ENVS_PER_LEVEL + 1;
    if (level >= current_level) level += 1;
    return level <= num_levels ? level : current_level;
}

static __device__ int gpu_curriculum_perf_slot(
        int env_id, int level, int num_levels) {
      if (env_id < ENVS_PER_LEVEL) {
        return (level - 1) * ENVS_PER_LEVEL + env_id;
    }
      if (env_id < MASTERY_ENVS_BIN) return -1;
    int level_env_offset = env_id - MASTERY_ENVS_BIN;
    if (level_env_offset >= (num_levels - 1) * ENVS_PER_LEVEL) return -1;
    return (level - 1) * ENVS_PER_LEVEL + level_env_offset % ENVS_PER_LEVEL;
}

static __device__ __forceinline__ unsigned int gpu_rand_r(unsigned int* seed) {
    unsigned int next = *seed;
    unsigned int result;

    next = next * 1103515245u + 12345u;
    result = (next / 65536u) % 2048u;
    next = next * 1103515245u + 12345u;
    result = (result << 10) ^ ((next / 65536u) % 1024u);
    next = next * 1103515245u + 12345u;
    result = (result << 10) ^ ((next / 65536u) % 1024u);

    *seed = next;
    return result;
}

static __device__ __forceinline__ float gpu_sample(
        Env* env, float center, float span) {
    if (span == 0.0f) return center;
    float sample = (float)gpu_rand_r(&env->rng) / 2147483648.0f;
    return center + span * (2.0f * sample - 1.0f);
}

static __device__ void gpu_rotate_spawn(
        Env* env, float center_x, float center_y, float rotation) {
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

static __device__ void gpu_spawn(Env* env) {
    const GpuCurriculumConfig* config = &d_admiral.curriculum[env->curr_level - 1];
    env->damage_mult = config->damage;
    env->max_ticks = config->ticks;
    float center_x = 0.5f * d_admiral.width;
    float center_y = 0.5f * d_admiral.height;
    env->curr_adv_team = env->next_adv_team;
    int target_team = 1 - env->curr_adv_team;
    float shooter_heading = env->curr_adv_team == 0 ? 0.0f : PI;
    int rotation_level = env->curr_level <= 3
        ? 2 : (env->curr_level < 6 ? env->curr_level - 1 : 5);

    if (env->curr_side_episodes % N_TEAMS == 0) {
        env->spawn_bearing = D2R * gpu_sample(env, config->bearing, config->bearing_jitter);
        env->spawn_spacing = gpu_sample(env, SHIP_SPACING, SHIP_SPACING_SPAN);
        int rotation_idx = (env->env_id / (4 * N_TEAMS) + env->curr_side_episodes / N_TEAMS) % 4;
        float rotation_span = rotation_level < 3 ? 0.0f
            : Q_PI * (rotation_level - 2) / 3.0f;
        env->spawn_rotation = rotation_idx * H_PI
            + gpu_sample(env, 0.0f, rotation_span);
    }

    float enemy_heading = D2R * gpu_sample(
        env, config->enemy_heading, config->enemy_heading_jitter);
    float wind_heading = D2R * gpu_sample(
        env, config->wind_heading, config->wind_heading_jitter);
    float enemy_direction = shooter_heading + env->curr_fire_side * env->spawn_bearing;
    float separation_x = config->separation * cosf(enemy_direction);
    float separation_y = config->separation * sinf(enemy_direction);
    int wind_side = env->curr_level <= FAVORABLE_WIND_LEVELS ? env->curr_fire_side : env->curr_wind_sign;
    float wind_direction = shooter_heading + wind_side * wind_heading;
    env->wind_vx = config->wind_speed * cosf(wind_direction);
    env->wind_vy = config->wind_speed * sinf(wind_direction);
    float adv_speed = gpu_sample(env, config->adv_speed, config->adv_speed_jitter);
    float enemy_speed = gpu_sample(env, config->enemy_speed, config->enemy_speed_jitter);
    if (adv_speed < 0.0f) adv_speed = 0.0f;
    if (enemy_speed < 0.0f) enemy_speed = 0.0f;

    for (int idx = 0; idx < NUM_SHIPS; idx++) {
        Ship ship = {};
        ship.cooldown_left = 1.0f;
        ship.cooldown_right = 1.0f;
        ship.team_idx = idx / SHIPS_PER_TEAM;
        env->ships[idx] = ship;
    }

    for (int ship_num = 0; ship_num < SHIPS_PER_TEAM; ship_num++) {
        float lane_offset =
            (ship_num - 0.5f * (SHIPS_PER_TEAM - 1)) * env->spawn_spacing;
        Ship* shooter = &env->ships[
            env->curr_adv_team * SHIPS_PER_TEAM + ship_num];
        Ship* target = &env->ships[target_team * SHIPS_PER_TEAM + ship_num];

        shooter->x = center_x - 0.5f * separation_x
            + lane_offset * cosf(shooter_heading);
        shooter->y = center_y - 0.5f * separation_y
            + lane_offset * sinf(shooter_heading);
        shooter->heading = shooter_heading;
        target->x = shooter->x + separation_x;
        target->y = shooter->y + separation_y;
        target->heading = fmodf(
            shooter_heading + env->curr_fire_side * enemy_heading, TWO_PI);
        if (target->heading < 0.0f) target->heading += TWO_PI;

        for (int team = 0; team < N_TEAMS; team++) {
            Ship* ship = &env->ships[team * SHIPS_PER_TEAM + ship_num];
            float health = team == env->curr_adv_team
                ? MAX_HEALTH : config->enemy_health;
            ship->health = health;
            ship->health_old = health;
            ship->speed = MS * (team == env->curr_adv_team ? adv_speed : enemy_speed);
            ship->vx = ship->speed * cosf(ship->heading);
            ship->vy = ship->speed * sinf(ship->heading);
        }
    }

    gpu_rotate_spawn(env, center_x, center_y, env->spawn_rotation);

    env->next_adv_team = target_team;
    env->curr_side_episodes += 1;
    if (env->curr_side_episodes % N_TEAMS == 0) {
        env->curr_wind_sign = -env->curr_wind_sign;
    }
    if (env->curr_side_episodes == 2 * N_TEAMS) {
        env->curr_side_episodes = 0;
        env->curr_fire_side = -env->curr_fire_side;
    }
}

static __device__ void gpu_reset(Env* env) {
    env->tick = 0;
    env->episode_ended = 0;
    env->mastery_win = 0;
    for (int team = 0; team < N_TEAMS; team++) {
        env->logs[team] = {};
        env->team_kills[team] = 0;
    }
    gpu_spawn(env);
}

static __device__ void gpu_observe(
        Env* env, obs_t* observations, int game, int lane) {
    float inverse_diagonal =
        1.0f / hypotf((float)d_admiral.width, (float)d_admiral.height);

    for (int team = 0; team < N_TEAMS; team++) {
        obs_t* obs = observations + (long)(game * N_TEAMS + team) * OBS_SIZE;
        int opponent = 1 - team;
        float flip = team == 0 ? 1.0f : -1.0f;

        if (lane == 0) obs[0] = (float)env->tick / (float)env->max_ticks;
        if (lane == 1) obs[1] = flip * env->wind_vx * 0.1f;
        if (lane == 2) obs[2] = flip * env->wind_vy * 0.1f;

        for (int slot = lane; slot < NUM_SHIPS; slot += ADMIRAL_THREADS_PER_ENV) {
            int side = slot / SHIPS_PER_TEAM;
            int ship_num = slot % SHIPS_PER_TEAM;
            int owner = side == 0 ? team : opponent;
            Ship* ship = &env->ships[owner * SHIPS_PER_TEAM + ship_num];
            int idx = GLOBAL_OBS_FEATURES + slot * SHIP_OBS_FEATURES;

            obs[idx++] = flip * (2.0f * ship->x / d_admiral.width - 1.0f);
            obs[idx++] = flip * (2.0f * ship->y / d_admiral.height - 1.0f);
            obs[idx++] = flip * ship->vx / MS;
            obs[idx++] = flip * ship->vy / MS;
            obs[idx++] = ship->health;
            obs[idx++] = ship->cooldown_left;
            obs[idx++] = ship->cooldown_right;
            obs[idx++] = ship->rudder / MAX_RUDDER;
            obs[idx++] = ship->sail_angle / MAX_SAIL_ANGLE;
            obs[idx++] = flip * cosf(ship->heading);
            obs[idx] = flip * sinf(ship->heading);
        }

        for (int pair = lane;
                pair < SHIPS_PER_TEAM * SHIPS_PER_TEAM;
                pair += ADMIRAL_THREADS_PER_ENV) {
            int idx = BASE_OBS_SIZE + pair * RELATIVE_POSITION_FEATURES;
            int own_num = pair / SHIPS_PER_TEAM;
            int enemy_num = pair % SHIPS_PER_TEAM;
            Ship* own = &env->ships[team * SHIPS_PER_TEAM + own_num];
            Ship* enemy = &env->ships[opponent * SHIPS_PER_TEAM + enemy_num];
            float heading_cos = cosf(own->heading);
            float heading_sin = sinf(own->heading);
            float dx = enemy->x - own->x;
            float dy = enemy->y - own->y;
            obs[idx] = (heading_cos * dx + heading_sin * dy) * inverse_diagonal;
            obs[idx + 1] =
                (-heading_sin * dx + heading_cos * dy) * inverse_diagonal;
        }

        for (int ship_num = lane;
                ship_num < SHIPS_PER_TEAM;
                ship_num += ADMIRAL_THREADS_PER_ENV) {
            int idx = SAIL_OBS_START + ship_num * SAIL_OBS_FEATURES;
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
                incidence = (apparent_forward * sail_cos + apparent_lateral * sail_sin)
                    / apparent_speed;
                incidence = fminf(1.0f, fmaxf(-1.0f, incidence));
            }

            obs[idx++] = apparent_forward * 0.1f;
            obs[idx++] = apparent_lateral * 0.1f;
            obs[idx++] = ship->speed / MS;
            obs[idx++] = incidence * sail_cos * ship->health;
            obs[idx] = asinf(incidence) / H_PI;
        }
    }
}

static __device__ __forceinline__ void gpu_add_reward(
        Env* env, float* rewards, int team, float reward) {
    rewards[team] += reward;
    env->logs[team].episode_return += reward;
}

static __device__ __forceinline__ int gpu_kill_outcome(Env* env) {
    if (env->team_kills[0] > env->team_kills[1]) return 1;
    if (env->team_kills[1] > env->team_kills[0]) return -1;
    return 0;
}

static __device__ __forceinline__ void gpu_turn(
        float* heading, float radians, float d_angle) {
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

static __device__ void gpu_move(Env* env, float* rewards, Ship* ship) {
    float dx = cosf(ship->heading);
    float dy = sinf(ship->heading);
    float rel_vx = env->wind_vx - ship->vx;
    float rel_vy = env->wind_vy - ship->vy;
    float rel_speed = sqrtf(rel_vx * rel_vx + rel_vy * rel_vy);
    float sail_world_angle = ship->sail_angle + ship->heading;
    float sail_world_nx = cosf(sail_world_angle);
    float sail_world_ny = sinf(sail_world_angle);
    float normal_speed = rel_vx * sail_world_nx + rel_vy * sail_world_ny;
    float forward_projection = sail_world_nx * dx + sail_world_ny * dy;
    float forward_force = 0.5f * K_SAIL * SAIL_AREA * rel_speed
        * normal_speed * forward_projection * ship->health;

    float rudder_fraction = ship->rudder / MAX_RUDDER;
    float speed = ship->speed;
    float damage = 1.0f - ship->health;
    float drag = HULL_DRAG_COEFF
        * (1.0f + DAMAGE_DRAG_SCALE * damage * damage) * speed * speed
        + MASS * RUDDER_DRAG_RATE * rudder_fraction * rudder_fraction
            * speed * speed / MS;
    float acceleration = (forward_force - drag) / MASS;
    ship->speed += acceleration * DT;
    ship->speed = fmaxf(0.0f, fminf(ship->speed, MS));
    if (ship->health > 0.0f && ship->speed < STATIONARY_SPEED_THRESHOLD) {
        gpu_add_reward(env, rewards, ship->team_idx, d_admiral.penalty_stationary);
    }
    ship->vx = ship->speed * dx;
    ship->vy = ship->speed * dy;
    ship->x += ship->speed * dx * DT;
    ship->y += ship->speed * dy * DT;
}

static __device__ bool gpu_ray_hits_ship(
        float ray_x, float ray_y, float ray_dx, float ray_dy,
        Ship* ship, float* hit_distance) {
    float half_length = SHIP_LENGTH * 0.5f;
    float centerline_dx = cosf(ship->heading) * half_length;
    float centerline_dy = sinf(ship->heading) * half_length;
    float ax = ship->x - centerline_dx;
    float ay = ship->y - centerline_dy;
    float bx = ship->x + centerline_dx;
    float by = ship->y + centerline_dy;
    float segment_dx = bx - ax;
    float segment_dy = by - ay;
    float denominator = ray_dx * segment_dy - ray_dy * segment_dx;
    if (denominator == 0.0f) return false;

    float offset_x = ax - ray_x;
    float offset_y = ay - ray_y;
    float inverse_denominator = 1.0f / denominator;
    float ray_distance =
        (offset_x * segment_dy - offset_y * segment_dx) * inverse_denominator;
    float segment_fraction =
        (offset_x * ray_dy - offset_y * ray_dx) * inverse_denominator;
    if (ray_distance < 0.0f || ray_distance > CANNON_RANGE) return false;
    if (segment_fraction < 0.0f || segment_fraction > 1.0f) return false;

    *hit_distance = ray_distance;
    return true;
}

static __device__ void gpu_fire(
        Env* env, float* rewards, Ship* ship, int ship_idx, int fire_side) {
    gpu_add_reward(env, rewards, ship->team_idx, d_admiral.penalty_used_volley);
    if (fire_side < 0) {
        if (ship->cooldown_left < 0.9999f) return;
        ship->cooldown_left = 0.0f;
    } else {
        if (ship->cooldown_right < 0.9999f) return;
        ship->cooldown_right = 0.0f;
    }

    float shot_heading = ship->heading + H_PI * fire_side;
    float ray_dx = cosf(shot_heading);
    float ray_dy = sinf(shot_heading);
    ship->render_shot_heading = shot_heading;
    ship->render_shot_ticks = RENDER_SHOT_TICKS;
    int hit_idx = -1;
    float hit_distance = CANNON_RANGE;

    for (int target_idx = 0; target_idx < NUM_SHIPS; target_idx++) {
        if (target_idx == ship_idx) continue;
        Ship* target = &env->ships[target_idx];
        if (target->health <= 0.0f) continue;

        float target_distance;
        if (!gpu_ray_hits_ship(
                ship->x, ship->y, ray_dx, ray_dy,
                target, &target_distance)) continue;
        if (target_distance >= hit_distance) continue;
        hit_idx = target_idx;
        hit_distance = target_distance;
    }

    if (hit_idx < 0) return;
    Ship* hit_ship = &env->ships[hit_idx];
    float range_factor = 1.0f - hit_distance / CANNON_RANGE;
    float damage_mult = ship->team_idx == env->curr_adv_team ? env->damage_mult : 1.0f;
    float damage = fminf(
        CANNON_MAX_DAMAGE * damage_mult * range_factor, hit_ship->health);
    bool hit_enemy = hit_ship->team_idx != ship->team_idx;
    if (hit_enemy) {
        env->logs[ship->team_idx].score += damage;
        gpu_add_reward(
            env, rewards, ship->team_idx, d_admiral.reward_damage * damage);
    } else {
        gpu_add_reward(
            env, rewards, ship->team_idx, d_admiral.penalty_hit_ally * damage);
    }

    if (hit_enemy && damage >= hit_ship->health) {
        gpu_add_reward(env, rewards, ship->team_idx, d_admiral.reward_kill);
        env->team_kills[ship->team_idx] += 1;
    }
    hit_ship->health -= damage;
}

static __device__ void gpu_add_log(Env* env) {
    for (int team = 0; team < N_TEAMS; team++) {
        env->log.kills += env->team_kills[team];
        env->log.episode_return += env->logs[team].episode_return;
        env->log.episode_length += env->logs[team].episode_length;
        env->log.score += env->logs[team].score;
        env->log.n += 1.0f;
    }
}

static __device__ void gpu_end_episode(
        Env* env, float* rewards, float* terminals, int outcome) {
    if (outcome == 0) {
        gpu_add_reward(env, rewards, 0, -1.0f);
        gpu_add_reward(env, rewards, 1, -1.0f);
    } else {
        float reward = env->curr_level == MAX_LEVEL ? outcome : (env->team_kills[0] - env->team_kills[1]) / (float)SHIPS_PER_TEAM;
        gpu_add_reward(env, rewards, 0, reward);
        gpu_add_reward(env, rewards, 1, -reward);
    }

    float policy_0_score = outcome > 0 ? 1.0f : 0.0f;
    float policy_1_score = outcome < 0 ? 1.0f : 0.0f;
    env->log.policy_0_score += policy_0_score * N_TEAMS;
    env->log.policy_1_score += policy_1_score * N_TEAMS;
    if (outcome == 0) env->log.draw_rate += N_TEAMS;

    int adv_team = env->curr_adv_team;
    bool won = (adv_team == 0 && outcome > 0)
        || (adv_team == 1 && outcome < 0);
    bool mastery_win = won;
    if (env->curr_level <= FULL_KILL_LEVELS) {
        mastery_win = won && env->team_kills[adv_team] == SHIPS_PER_TEAM;
    }
    if (env->curr_level <= FLAWLESS_LEVELS) {
        mastery_win = mastery_win && env->team_kills[1 - adv_team] == 0;
    }

    env->log.adv_kills += env->team_kills[adv_team] * N_TEAMS;
    env->log.curr_level += env->curr_level * N_TEAMS;
    env->log.curr_win_rate += won * N_TEAMS;
    env->mastery_win = mastery_win;
    env->episode_ended = 1;

    for (int team = 0; team < N_TEAMS; team++) {
        terminals[team] = 1.0f;
    }
    gpu_add_log(env);
}

static __device__ void gpu_step(
        Env* env, const float* actions, float* rewards, float* terminals) {
    for (int team = 0; team < N_TEAMS; team++) {
        rewards[team] = 0.0f;
        terminals[team] = 0.0f;
    }

    env->tick += 1;
    if (env->tick > env->max_ticks) {
        gpu_end_episode(env, rewards, terminals, gpu_kill_outcome(env));
        return;
    }

    for (int team = 0; team < N_TEAMS; team++) {
        const float* team_actions = actions + team * NUM_ATNS;
        env->logs[team].episode_length += 1.0f;
        for (int ship_num = 0; ship_num < SHIPS_PER_TEAM; ship_num++) {
            int ship_idx = team * SHIPS_PER_TEAM + ship_num;
            Ship* ship = &env->ships[ship_idx];
            if (ship->health_old <= 0.0f) continue;
            const float* ship_actions = team_actions + ship_num * NUM_ACTIONS;

            if (ship->render_shot_ticks > 0) ship->render_shot_ticks -= 1;
            if (ship->cooldown_left < 1.0f) {
                ship->cooldown_left =
                    fminf(1.0f, ship->cooldown_left + COOLDOWN_PER_TICK);
            }
            if (ship->cooldown_right < 1.0f) {
                ship->cooldown_right =
                    fminf(1.0f, ship->cooldown_right + COOLDOWN_PER_TICK);
            }

            float rudder_action = d_rudder_values[(int)ship_actions[0]];
            ship->rudder += rudder_action * RUDDER_RATE * DT;
            ship->rudder = fmaxf(-MAX_RUDDER, fminf(ship->rudder, MAX_RUDDER));

            float sail_action = d_sail_angle_values[(int)ship_actions[1]];
            ship->sail_angle += sail_action * SAIL_RATE * DT;
            ship->sail_angle = fmaxf(
                -MAX_SAIL_ANGLE, fminf(ship->sail_angle, MAX_SAIL_ANGLE));

            float speed_fraction = ship->speed / MS;
            float rudder_fraction = ship->rudder / MAX_RUDDER;
            float body_turn =
                MAX_TURN_RATE * speed_fraction * rudder_fraction * DT;
            gpu_turn(&ship->heading, body_turn, MAX_TURN_RATE * DT);

            float fire_action = d_fire_values[(int)ship_actions[2]];
            if (fire_action != 0.0f) {
                gpu_fire(env, rewards, ship, ship_idx, (int)fire_action);
            }

            gpu_move(env, rewards, ship);
            float old_x = ship->x;
            float old_y = ship->y;
            ship->x = fmaxf(0.0f, fminf(ship->x, (float)d_admiral.width));
            ship->y = fmaxf(0.0f, fminf(ship->y, (float)d_admiral.height));
            bool hit_wall = ship->x != old_x || ship->y != old_y;
            if (!hit_wall) continue;

            ship->health = fmaxf(0.0f, ship->health - 0.25f);
            ship->speed = 0.0f;
            ship->vx = 0.0f;
            ship->vy = 0.0f;
        }
    }

    if (env->team_kills[0] == SHIPS_PER_TEAM
            || env->team_kills[1] == SHIPS_PER_TEAM) {
        gpu_end_episode(env, rewards, terminals, gpu_kill_outcome(env));
        return;
    }

    for (int i = 0; i < NUM_SHIPS; i++) {
        env->ships[i].health_old = env->ships[i].health;
    }
}

__global__ void gpu_admiral_step_kernel(
        Env* envs, const float* actions, float* rewards, float* terminals,
        int num_games) {
    int game = blockIdx.x * blockDim.x + threadIdx.x;
    if (game >= num_games) return;
    gpu_step(
        &envs[game],
        actions + (long)game * N_TEAMS * NUM_ATNS,
        rewards + game * N_TEAMS,
        terminals + game * N_TEAMS);
}

__global__ void gpu_admiral_curriculum_kernel(
        Env* envs, int num_games) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    for (int game = 0; game < num_games; game++) {
        Env* env = &envs[game];
        if (!env->episode_ended) continue;

        int perf_slot = gpu_curriculum_perf_slot(
            env->env_id, env->curr_level, d_admiral.num_levels);
        if (perf_slot >= 0) {
            d_curriculum.level_wins[perf_slot] = env->mastery_win;
        }

        d_curriculum.total_games += 1;
        int env_id = env->env_id;
      if (env_id < MASTERY_ENVS_BIN
                && env->curr_level > d_curriculum.mastered_level) {
            int won = env->mastery_win;
            d_curriculum.num_wins =
                d_curriculum.num_wins - d_curriculum.wins[env_id] + won;
            d_curriculum.wins[env_id] = won;
              if (d_curriculum.num_wins >= MASTERY_WINS) {
                d_curriculum.mastered_level = env->curr_level;
                d_curriculum.num_wins = 0;
                for (int i = 0; i < MASTERY_ENVS_BIN; i++) {
                    d_curriculum.wins[i] = 0;
                }
            }
        }

        env->log.curr_mastered_level +=
            d_curriculum.mastered_level * N_TEAMS;
        if (d_curriculum.mastered_level == d_admiral.num_levels) {
            env->curr_level =
                d_curriculum.total_games % d_admiral.num_levels + 1;
        } else {
            env->curr_level = gpu_curriculum_level_for_env(
                env_id, d_curriculum.mastered_level + 1,
                d_admiral.num_levels);
        }
    }
}

__global__ void gpu_admiral_observe_kernel(
        Env* envs, obs_t* observations, float* rewards, float* terminals,
        int num_games, bool reset_all) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int game = thread / ADMIRAL_THREADS_PER_ENV;
    int lane = thread % ADMIRAL_THREADS_PER_ENV;
    bool valid = game < num_games;

    if (valid && lane == 0 && (reset_all || envs[game].episode_ended)) {
        if (reset_all) {
            rewards[game * N_TEAMS] = 0.0f;
            rewards[game * N_TEAMS + 1] = 0.0f;
            terminals[game * N_TEAMS] = 0.0f;
            terminals[game * N_TEAMS + 1] = 0.0f;
        }
        gpu_reset(&envs[game]);
    }
    __syncwarp();

    if (valid) gpu_observe(&envs[game], observations, game, lane);
}

static float gpu_admiral_get_float(
        Dict* kwargs, const char* key, float default_value) {
    for (int i = 0; i < kwargs->size; i++) {
        if (strcmp(kwargs->items[i].key, key) == 0) {
            return (float)kwargs->items[i].value;
        }
    }
    return default_value;
}

static GpuAdmiralConfig gpu_admiral_config(Dict* kwargs) {
    GpuAdmiralConfig config = {};
    config.width = dict_get(kwargs, "width");
    config.height = dict_get(kwargs, "height");
    config.curriculum_level =
        (int)gpu_admiral_get_float(kwargs, "curriculum_level", 1);
    config.reward_damage = gpu_admiral_get_float(kwargs, "reward_damage_mult", 0.0f);
    config.reward_kill = gpu_admiral_get_float(kwargs, "reward_kill", 0.0f);
    config.penalty_hit_ally =
        gpu_admiral_get_float(kwargs, "penalty_hit_ally", 0.0f);
    config.penalty_used_volley =
        gpu_admiral_get_float(kwargs, "penalty_used_volley", 0.0f);
    config.penalty_stationary =
        gpu_admiral_get_float(kwargs, "penalty_stationary", 0.0f);
    config.num_levels = MAX_LEVEL;

    for (int i = 0; i < config.num_levels; i++) {
        const CurriculumConfig* source = &CURRICULUM[i];
        GpuCurriculumConfig* destination = &config.curriculum[i];
        destination->separation = source->separation;
        destination->bearing = source->bearing;
        destination->bearing_jitter = source->bearing_jitter;
        destination->enemy_heading = source->enemy_heading;
        destination->enemy_heading_jitter = source->enemy_heading_jitter;
        destination->adv_speed = source->adv_speed;
        destination->adv_speed_jitter = source->adv_speed_jitter;
        destination->enemy_speed = source->enemy_speed;
        destination->enemy_speed_jitter = source->enemy_speed_jitter;
        destination->wind_heading = source->wind_heading;
        destination->wind_heading_jitter = source->wind_heading_jitter;
        destination->wind_speed = source->wind_speed;
        destination->damage = source->damage;
        destination->enemy_health = source->enemy_health;
        destination->ticks = source->ticks;
    }
    return config;
}

void puf_log(Log* log, Dict* out) {
    GpuCurriculum curriculum;
    cudaMemcpyFromSymbol(&curriculum, d_curriculum, sizeof(curriculum));
    dict_set(out, "perf",
        curriculum_perf(curriculum.level_wins, curriculum.mastered_level));
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

Env* puf_vec_create(int total_agents, Dict* env_kwargs,
        obs_t* observations, float* actions, float* rewards, float* terminals) {
    assert(total_agents >= N_TEAMS && total_agents % N_TEAMS == 0);
    GpuAdmiralConfig config = gpu_admiral_config(env_kwargs);
    gpu_render_width = config.width;
    gpu_render_height = config.height;
    cudaMemcpyToSymbol(d_admiral, &config, sizeof(config));

    GpuCurriculum curriculum = {};
    curriculum.mastered_level = config.curriculum_level - 1;
    cudaMemcpyToSymbol(d_curriculum, &curriculum, sizeof(curriculum));

    Env* host_envs = (Env*)calloc((size_t)total_agents, sizeof(Env));
    int num_games = total_agents / N_TEAMS;
    for (int game = 0; game < num_games; game++) {
        Env* env = &host_envs[game];
        env->env_id = game;
        env->rng = game;
        env->num_agents = N_TEAMS;
        env->curr_level = curriculum_level_for_env(
            game, config.curriculum_level, config.num_levels);
        int spawn_variant = env->rng % (4 * N_TEAMS);
        int pair_variant = spawn_variant / N_TEAMS;
        env->next_adv_team = spawn_variant % N_TEAMS;
        env->curr_fire_side = pair_variant < 2 ? -1 : 1;
        env->curr_wind_sign = pair_variant % 2 == 0 ? 1 : -1;
    }

    Env* envs = NULL;
    cudaMalloc((void**)&envs, (size_t)total_agents * sizeof(Env));
    cudaMemcpy(envs, host_envs,
        (size_t)total_agents * sizeof(Env), cudaMemcpyHostToDevice);
    free(host_envs);
    g_gpu.total_agents = total_agents;
    g_gpu.observations = observations;
    g_gpu.actions = actions;
    g_gpu.rewards = rewards;
    g_gpu.terminals = terminals;
    g_gpu.stream = 0;
    return envs;
}

void puf_bind_stream(cudaStream_t stream) {
    g_gpu.stream = stream;
}

void puf_init(Env*, Dict*) {
}

void puf_reset(Env* envs) {
    int num_games = g_gpu.total_agents / N_TEAMS;
    int threads = num_games * ADMIRAL_THREADS_PER_ENV;
    gpu_admiral_observe_kernel<<<grid_size(threads), BLOCK_SIZE, 0, g_gpu.stream>>>(
        envs, g_gpu.observations, g_gpu.rewards, g_gpu.terminals, num_games, true);
}

void puf_step(Env* envs) {
    int num_games = g_gpu.total_agents / N_TEAMS;
    gpu_admiral_step_kernel<<<grid_size(num_games), BLOCK_SIZE, 0, g_gpu.stream>>>(
        envs, g_gpu.actions, g_gpu.rewards, g_gpu.terminals, num_games);
    gpu_admiral_curriculum_kernel<<<1, 1, 0, g_gpu.stream>>>(envs, num_games);
    int threads = num_games * ADMIRAL_THREADS_PER_ENV;
    gpu_admiral_observe_kernel<<<grid_size(threads), BLOCK_SIZE, 0, g_gpu.stream>>>(
        envs, g_gpu.observations, g_gpu.rewards, g_gpu.terminals, num_games, false);
}

void puf_close(Env* envs) {
    if (gpu_render_initialized) {
        CloseWindow();
        gpu_render_initialized = false;
    }
    cudaFree(envs);
    g_gpu = {};
}

void puf_render(Env* env) {
    if (!env || g_gpu.total_agents < N_TEAMS) return;
    if (g_gpu.stream) cudaStreamSynchronize(g_gpu.stream);
    Env snapshot = {};
    cudaMemcpy(&snapshot, env, sizeof(snapshot), cudaMemcpyDeviceToHost);

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

    if (!gpu_render_initialized) {
        memset(shots, 0, sizeof(shots));
        memset(trails, 0, sizeof(trails));
        last_episode = snapshot.curr_side_episodes;
        InitWindow(gpu_render_width / 2, gpu_render_height / 2,
            "PufferLib Ray Admiral");
        SetTargetFPS(80);
        gpu_render_initialized = true;
    }

    BeginDrawing();
    ClearBackground((Color){6, 6, 120, 255});

    float ship_length = SHIP_LENGTH / 2.0f;
    float ship_width = SHIP_WIDTH / 2.0f;
    float smoke_scale = 5.0f;
    Color trail_color = (Color){0, 180, 255, 255};
    float bow_length = ship_width;
    float body_length = ship_length - bow_length;
    const float mast_offsets[3] = {
        -body_length / 4.0f, 0.0f, body_length / 4.0f
    };
    const float sail_half_lengths[3] = {
        0.75f * ship_width, 1.1f * ship_width, ship_width
    };

    if (snapshot.curr_side_episodes != last_episode) {
        memset(trails, 0, sizeof(trails));
        last_episode = snapshot.curr_side_episodes;
    }

    for (int team = 0; team < N_TEAMS; team++) {
        for (int i = 0; i < SHIPS_PER_TEAM; i++) {
            int ship_idx = i + team * SHIPS_PER_TEAM;
            Ship ship = snapshot.ships[ship_idx];
            Vector2 pos = {ship.x / 2.0f, ship.y / 2.0f};

            RenderTrail* trail = &trails[ship_idx];
            if (ship.health > 0.0f) {
                trail->pos[trail->index] = pos;
                trail->index = (trail->index + 1) % SHIP_TRAIL_LENGTH;
                if (trail->count < SHIP_TRAIL_LENGTH) trail->count++;
            }

            float trail_alpha = ship.health > 0.0f ? 0.7f : 0.35f;
            for (int j = 0; j < trail->count - 1; j++) {
                int idx0 = (trail->index - j - 1 + SHIP_TRAIL_LENGTH)
                    % SHIP_TRAIL_LENGTH;
                int idx1 = (trail->index - j - 2 + SHIP_TRAIL_LENGTH)
                    % SHIP_TRAIL_LENGTH;
                float alpha = trail_alpha * (trail->count - j)
                    / trail->count;
                DrawLineEx(trail->pos[idx0], trail->pos[idx1], 2.0f,
                    Fade(trail_color, alpha));
            }

            RenderShot* shot = &shots[ship_idx];
            if (ship.health <= 0.0f) {
                shot->smoke_frames = 0;
            } else if (ship.render_shot_ticks > 0
                    && shot->smoke_frames == 0) {
                Vector2 shot_direction = {
                    cosf(ship.render_shot_heading),
                    sinf(ship.render_shot_heading)
                };
                shot->side = (Vector2){
                    -shot_direction.y, shot_direction.x
                };
                shot->muzzle = (Vector2){
                    pos.x + shot_direction.x
                        * (ship_width / 2.0f + 3.0f * smoke_scale),
                    pos.y + shot_direction.y
                        * (ship_width / 2.0f + 3.0f * smoke_scale)
                };
                Vector2 shot_end = {
                    pos.x + shot_direction.x * CANNON_RANGE / 2.0f,
                    pos.y + shot_direction.y * CANNON_RANGE / 2.0f
                };
                DrawLineEx(shot->muzzle, shot_end, 1.5f,
                    Fade(RAYWHITE, 0.5f));
                shot->smoke_frames = 32;
            }

            if (shot->smoke_frames > 0) {
                float alpha = shot->smoke_frames / 32.0f;
                float age = 4.0f * (1.0f - alpha);
                Color smoke = Fade(GRAY, 0.6f * alpha);
                DrawCircleV((Vector2){
                    shot->muzzle.x + smoke_scale * age * shot->side.y,
                    shot->muzzle.y - smoke_scale * age * shot->side.x
                }, smoke_scale * (3.0f + 0.5f * age), smoke);
                DrawCircleV((Vector2){
                    shot->muzzle.x + 2.5f * shot->side.x,
                    shot->muzzle.y + 2.5f * shot->side.y
                }, smoke_scale * (2.5f + 0.25f * age), smoke);
                DrawCircleV((Vector2){
                    shot->muzzle.x - 2.5f * shot->side.x,
                    shot->muzzle.y - 2.5f * shot->side.y
                }, smoke_scale * (2.5f + 0.25f * age), smoke);
                shot->smoke_frames--;
            }

            float ship_alpha = ship.health > 0.0f ? 1.0f : 0.2f;
            Color ship_color = Fade(team ? GREEN : RED, ship_alpha);
            Vector2 forward = {cosf(ship.heading), sinf(ship.heading)};
            Vector2 side = {-forward.y, forward.x};
            Vector2 body_center = {
                pos.x - 0.5f * bow_length * forward.x,
                pos.y - 0.5f * bow_length * forward.y
            };
            DrawRectanglePro((Rectangle){
                body_center.x, body_center.y, body_length, ship_width
            }, (Vector2){body_length / 2.0f, ship_width / 2.0f},
                ship.heading / D2R, ship_color);

            Vector2 bow_base = {
                pos.x + (ship_length / 2.0f - bow_length) * forward.x,
                pos.y + (ship_length / 2.0f - bow_length) * forward.y
            };
            Vector2 bow_tip = {
                pos.x + ship_length / 2.0f * forward.x,
                pos.y + ship_length / 2.0f * forward.y
            };
            Vector2 bow_right = {
                bow_base.x - ship_width / 2.0f * side.x,
                bow_base.y - ship_width / 2.0f * side.y
            };
            Vector2 bow_left = {
                bow_base.x + ship_width / 2.0f * side.x,
                bow_base.y + ship_width / 2.0f * side.y
            };
            DrawTriangle(bow_tip, bow_right, bow_left, ship_color);

            float sail_world_angle = ship.heading + ship.sail_angle;
            Vector2 sail_direction = {
                -sinf(sail_world_angle), cosf(sail_world_angle)
            };
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
                DrawLineEx(sail_start, sail_end, 2.0f,
                    Fade(RAYWHITE, ship_alpha));
            }

            if (ship.health > 0.0f) {
                float bar_width = SHIP_LENGTH / 2.0f;
                Rectangle health_bar = {
                    pos.x - bar_width / 2.0f,
                    pos.y - ship_length / 2.0f - 6.0f,
                    bar_width, 4.0f
                };
                DrawRectangleRec(health_bar, RED);
                health_bar.width *= ship.health / MAX_HEALTH;
                DrawRectangleRec(health_bar, GREEN);
            }
        }
    }

    DrawText(TextFormat("%i", snapshot.tick), 10, 10, 10, WHITE);
    const char* adv_text = "adv =";
    DrawText(adv_text, 10, 25, 20, WHITE);
    DrawRectangle(10 + MeasureText(adv_text, 20) + 6, 29, 28, 12,
        snapshot.curr_adv_team == 0 ? RED : GREEN);
    const char* stage_text = TextFormat("Stage %i", snapshot.curr_level);
    DrawText(stage_text, GetScreenWidth() - MeasureText(stage_text, 20) - 10,
        10, 20, WHITE);

    Vector2 wind_tip = {
        GetScreenWidth() / 2 + 2.5f * snapshot.wind_vx,
        55.0f + 2.5f * snapshot.wind_vy
    };
    DrawLineEx((Vector2){
        GetScreenWidth() / 2 - 2.5f * snapshot.wind_vx,
        55.0f - 2.5f * snapshot.wind_vy
    }, wind_tip, 4.0f, SKYBLUE);
    float wind_angle = atan2f(snapshot.wind_vy, snapshot.wind_vx);
    Vector2 arrow_base = {
        wind_tip.x - 10.0f * cosf(wind_angle),
        wind_tip.y - 10.0f * sinf(wind_angle)
    };
    Vector2 arrow_left = {
        arrow_base.x - 6.0f * sinf(wind_angle),
        arrow_base.y + 6.0f * cosf(wind_angle)
    };
    Vector2 arrow_right = {
        arrow_base.x + 6.0f * sinf(wind_angle),
        arrow_base.y - 6.0f * cosf(wind_angle)
    };
    DrawTriangle(wind_tip, arrow_right, arrow_left, SKYBLUE);

    EndDrawing();
}

#endif  // PUFFER_ADMIRAL_GPU_CU
