#pragma once

#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>

#ifndef BAT_HEADLESS
#include "raylib.h"
#endif

#define BAT_OBS_SIZE 70
#define BAT_NUM_ACTIONS 6
#define BAT_MOVE_ACTIONS 5
#define BAT_TURN_ACTIONS 3
#define BAT_CHIRP_FREQ_BINS 8
#define BAT_CHIRP_DURATION_BINS 4
#define BAT_CHIRP_EMIT_ACTIONS 2

#define BAT_RANGE_BINS 16
#define BAT_DOPPLER_BINS 16
#define BAT_LEFT_RANGE_OFFSET 0
#define BAT_LEFT_DOPPLER_OFFSET 16
#define BAT_RIGHT_RANGE_OFFSET 32
#define BAT_RIGHT_DOPPLER_OFFSET 48
#define BAT_CHIRP_AGE_OBS 64
#define BAT_CHIRP_START_OBS 65
#define BAT_CHIRP_END_OBS 66
#define BAT_CHIRP_DURATION_OBS 67
#define BAT_FORWARD_SPEED_OBS 68
#define BAT_TURN_RATE_OBS 69

#define BAT_NOOP 0
#define BAT_THRUST_FORWARD 1
#define BAT_BRAKE 2
#define BAT_STRAFE_LEFT 3
#define BAT_STRAFE_RIGHT 4

#define BAT_TURN_NONE 0
#define BAT_TURN_LEFT 1
#define BAT_TURN_RIGHT 2

#define BAT_MAX_OBSTACLES 16
#define BAT_TICK_RATE (1.0f/60.0f)
#define BAT_PI 3.14159265358979323846f
#define BAT_CHIRP_HISTORY 4
#define BAT_CHIRP_RINGS 5

typedef struct BatColor {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
} BatColor;

typedef struct ChirpEvent {
    float x;
    float y;
    float start_freq;
    float end_freq;
    float duration;
    int birth_tick;
    int active;
} ChirpEvent;

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float success;
    float collision;
    float timeout;
    float bug_distance_start;
    float bug_distance_final;
    float bug_distance_delta;
    float chirps_emitted;
    float mean_chirp_duration;
    float mean_chirp_bandwidth;
    float mean_echo_energy_left;
    float mean_echo_energy_right;
    float n;
} Log;

typedef struct Client {
    int width;
    int height;
} Client;

typedef struct Bat {
    Client* client;
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;

    int frameskip;
    int width;
    int height;
    int tick;
    int max_steps;
    int num_obstacles;

    float bat_x;
    float bat_y;
    float bat_vx;
    float bat_vy;
    float bat_heading;
    float bat_turn_velocity;
    float bat_radius;
    float bat_max_speed;
    float bat_accel;
    float bat_turn_rate;

    float bug_x;
    float bug_y;
    float bug_vx;
    float bug_vy;
    float bug_radius;
    float bug_speed;

    float* obstacle_x;
    float* obstacle_y;
    float* obstacle_w;
    float* obstacle_h;

    int range_bins_per_ear;
    int doppler_bins_per_ear;
    float max_echo_range;
    float sound_speed;
    float reflector_spacing;
    int max_chirp_age_ticks;
    int chirp_cooldown_ticks;
    int chirp_age_ticks;
    int last_chirp_tick;
    float last_chirp_start_freq;
    float last_chirp_end_freq;
    float last_chirp_duration;
    ChirpEvent chirps[BAT_CHIRP_HISTORY];
    int chirp_head;
    int chirps_emitted_episode;
    float chirp_duration_sum;
    float chirp_bandwidth_sum;
    float echo_energy_left_sum;
    float echo_energy_right_sum;

    float chirp_cost;
    float step_cost;
    float progress_reward_scale;
    float collision_penalty;
    float prev_bug_dist;
    float start_bug_dist;
    float episode_return;

    unsigned int rng;
} Bat;

static inline unsigned int bat_rand(Bat* env) {
    env->rng = env->rng * 1664525u + 1013904223u;
    return env->rng;
}

static inline float bat_randf(Bat* env) {
    return (bat_rand(env) >> 8) * (1.0f / 16777216.0f);
}

static inline float bat_clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline int bat_action_index(float v, int n) {
    int idx = (int)v;
    if (idx < 0) return 0;
    if (idx >= n) return n - 1;
    return idx;
}

static inline float bat_chirp_duration_seconds(float duration_norm) {
    return 0.04f + 0.18f * bat_clampf(duration_norm, 0.0f, 1.0f);
}

static inline float bat_chirp_ring_radius(float age_seconds, float slice,
        float duration_seconds, float sound_speed) {
    float ring_age = age_seconds - slice * duration_seconds;
    if (ring_age < 0.0f) return 0.0f;
    return sound_speed * ring_age;
}

static inline float bat_echo_time_seconds(float distance, float sound_speed) {
    if (sound_speed <= 0.0f) return 0.0f;
    return 2.0f * distance / sound_speed;
}

static inline bool bat_echo_is_arriving(float echo_time, float chirp_age,
        float window) {
    return fabsf(chirp_age - echo_time) <= window;
}

static inline BatColor bat_freq_color(float freq_norm, float alpha_norm) {
    float f = bat_clampf(freq_norm, 0.0f, 1.0f);
    float mid = 1.0f - fabsf(2.0f * f - 1.0f);
    BatColor color = {
        .r = (unsigned char)(255.0f * (1.0f - f) + 45.0f * f),
        .g = (unsigned char)(45.0f + 180.0f * mid),
        .b = (unsigned char)(45.0f * (1.0f - f) + 255.0f * f),
        .a = (unsigned char)(255.0f * bat_clampf(alpha_norm, 0.0f, 1.0f)),
    };
    return color;
}

static inline float bat_norm_bin(int idx, int count) {
    if (count <= 1) return 0.0f;
    return idx / (float)(count - 1);
}

static inline float bat_len(float x, float y) {
    return sqrtf(x*x + y*y);
}

static inline float bat_dist(float ax, float ay, float bx, float by) {
    return bat_len(bx - ax, by - ay);
}

static inline void bat_norm_vec(float x, float y, float* ox, float* oy) {
    float l = bat_len(x, y);
    if (l <= 0.000001f) {
        *ox = 1.0f;
        *oy = 0.0f;
        return;
    }
    *ox = x / l;
    *oy = y / l;
}

static inline bool bat_circle_rect_collision(float cx, float cy, float r,
        float rx, float ry, float rw, float rh) {
    float px = bat_clampf(cx, rx, rx + rw);
    float py = bat_clampf(cy, ry, ry + rh);
    return bat_dist(cx, cy, px, py) <= r;
}

static inline void generate_obstacles(Bat* env) {
    for (int i = 0; i < env->num_obstacles; i++) {
        float w = 7.0f + 2.0f * (float)(i % 3);
        float h = 7.0f + 2.0f * (float)((i + 1) % 3);
        float lane = (i + 1.0f) / (env->num_obstacles + 1.0f);
        float jitter = (bat_randf(env) - 0.5f) * 6.0f;
        env->obstacle_w[i] = w;
        env->obstacle_h[i] = h;
        env->obstacle_x[i] = bat_clampf(env->width * lane - w * 0.5f + jitter,
            env->bat_radius + 2.0f, env->width - w - env->bat_radius - 2.0f);
        env->obstacle_y[i] = bat_clampf(env->height * (0.35f + 0.3f * (i % 2)) - h * 0.5f - jitter,
            env->bat_radius + 2.0f, env->height - h - env->bat_radius - 2.0f);
    }
}

void init(Bat* env) {
    env->tick = 0;
    if (env->num_agents <= 0) env->num_agents = 1;
    if (env->frameskip <= 0) env->frameskip = 1;
    if (env->width <= 0) env->width = 64;
    if (env->height <= 0) env->height = 64;
    if (env->max_steps <= 0) env->max_steps = 512;
    if (env->bat_radius <= 0.0f) env->bat_radius = 2.0f;
    if (env->bug_radius <= 0.0f) env->bug_radius = 1.5f;
    if (env->bat_max_speed <= 0.0f) env->bat_max_speed = 12.0f;
    if (env->bat_accel <= 0.0f) env->bat_accel = 30.0f;
    if (env->bat_turn_rate <= 0.0f) env->bat_turn_rate = BAT_PI;
    if (env->bug_speed <= 0.0f) env->bug_speed = 4.0f;
    if (env->range_bins_per_ear <= 0) env->range_bins_per_ear = BAT_RANGE_BINS;
    if (env->doppler_bins_per_ear <= 0) env->doppler_bins_per_ear = BAT_DOPPLER_BINS;
    if (env->max_echo_range <= 0.0f) env->max_echo_range = 80.0f;
    if (env->sound_speed <= 0.0f) env->sound_speed = 100.0f;
    if (env->reflector_spacing <= 0.0f) env->reflector_spacing = 8.0f;
    if (env->max_chirp_age_ticks <= 0) env->max_chirp_age_ticks = 30;
    if (env->chirp_cooldown_ticks <= 0) env->chirp_cooldown_ticks = 12;
    if (env->step_cost <= 0.0f) env->step_cost = 0.001f;
    if (env->progress_reward_scale <= 0.0f) env->progress_reward_scale = 0.05f;
    if (env->collision_penalty <= 0.0f) env->collision_penalty = 1.0f;
    if (env->chirp_cost <= 0.0f) env->chirp_cost = 0.0005f;
    if (env->rng == 0) env->rng = 1;

    if (env->num_obstacles < 0) env->num_obstacles = 0;
    if (env->num_obstacles > BAT_MAX_OBSTACLES) env->num_obstacles = BAT_MAX_OBSTACLES;
    env->obstacle_x = (float*)calloc(env->num_obstacles, sizeof(float));
    env->obstacle_y = (float*)calloc(env->num_obstacles, sizeof(float));
    env->obstacle_w = (float*)calloc(env->num_obstacles, sizeof(float));
    env->obstacle_h = (float*)calloc(env->num_obstacles, sizeof(float));
}

void allocate(Bat* env) {
    init(env);
    env->observations = (float*)calloc(BAT_OBS_SIZE, sizeof(float));
    env->actions = (float*)calloc(BAT_NUM_ACTIONS, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (float*)calloc(1, sizeof(float));
}

void c_close(Bat* env) {
    free(env->obstacle_x);
    free(env->obstacle_y);
    free(env->obstacle_w);
    free(env->obstacle_h);
}

void free_allocated(Bat* env) {
    free(env->actions);
    free(env->observations);
    free(env->terminals);
    free(env->rewards);
    c_close(env);
}

static inline void add_log(Bat* env, float success, float collision, float timeout) {
    float final_dist = bat_dist(env->bat_x, env->bat_y, env->bug_x, env->bug_y);
    env->log.perf += success;
    env->log.score += env->episode_return;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->tick;
    env->log.success += success;
    env->log.collision += collision;
    env->log.timeout += timeout;
    env->log.bug_distance_start += env->start_bug_dist;
    env->log.bug_distance_final += final_dist;
    env->log.bug_distance_delta += env->start_bug_dist - final_dist;
    env->log.chirps_emitted += env->chirps_emitted_episode;
    if (env->chirps_emitted_episode > 0) {
        env->log.mean_chirp_duration += env->chirp_duration_sum / env->chirps_emitted_episode;
        env->log.mean_chirp_bandwidth += env->chirp_bandwidth_sum / env->chirps_emitted_episode;
    }
    env->log.mean_echo_energy_left += env->echo_energy_left_sum / fmaxf(1.0f, (float)(env->tick + 1));
    env->log.mean_echo_energy_right += env->echo_energy_right_sum / fmaxf(1.0f, (float)(env->tick + 1));
    env->log.n += 1.0f;
}

static inline void bat_add_echo(Bat* env, float rx, float ry, float rvx, float rvy,
        float strength) {
    float fx = cosf(env->bat_heading);
    float fy = sinf(env->bat_heading);
    float lx = -sinf(env->bat_heading);
    float ly = cosf(env->bat_heading);
    float ear_sep = env->bat_radius * 0.75f;
    float left_ear_x = env->bat_x - lx * ear_sep * 0.5f;
    float left_ear_y = env->bat_y - ly * ear_sep * 0.5f;
    float right_ear_x = env->bat_x + lx * ear_sep * 0.5f;
    float right_ear_y = env->bat_y + ly * ear_sep * 0.5f;

    float ux, uy;
    bat_norm_vec(rx - env->bat_x, ry - env->bat_y, &ux, &uy);
    float forward = ux * fx + uy * fy;
    if (forward < -0.35f) return;

    float left_dir_x = -lx;
    float left_dir_y = -ly;
    float right_dir_x = lx;
    float right_dir_y = ly;
    float left_gain = bat_clampf(0.75f + 0.25f * (ux * left_dir_x + uy * left_dir_y), 0.1f, 1.0f);
    float right_gain = bat_clampf(0.75f + 0.25f * (ux * right_dir_x + uy * right_dir_y), 0.1f, 1.0f);

    float left_path = bat_dist(env->bat_x, env->bat_y, rx, ry) + bat_dist(rx, ry, left_ear_x, left_ear_y);
    float right_path = bat_dist(env->bat_x, env->bat_y, rx, ry) + bat_dist(rx, ry, right_ear_x, right_ear_y);
    float path = 0.5f * (left_path + right_path);
    if (path > env->max_echo_range) return;

    int bin = (int)((path / env->max_echo_range) * env->range_bins_per_ear);
    if (bin < 0) bin = 0;
    if (bin >= env->range_bins_per_ear) bin = env->range_bins_per_ear - 1;

    float attenuation = strength / (1.0f + 0.02f * path * path);
    float rel_vx = rvx - env->bat_vx;
    float rel_vy = rvy - env->bat_vy;
    float distance_rate = rel_vx * ux + rel_vy * uy;
    float doppler = bat_clampf(-distance_rate / (env->bat_max_speed + env->bug_speed + 0.0001f), -1.0f, 1.0f);

    float left_energy = attenuation * left_gain;
    float right_energy = attenuation * right_gain;
    env->observations[BAT_LEFT_RANGE_OFFSET + bin] += left_energy;
    env->observations[BAT_RIGHT_RANGE_OFFSET + bin] += right_energy;
    env->observations[BAT_LEFT_DOPPLER_OFFSET + bin] += left_energy * doppler;
    env->observations[BAT_RIGHT_DOPPLER_OFFSET + bin] += right_energy * doppler;
}

static inline void bat_add_segment_reflectors(Bat* env, float x1, float y1,
        float x2, float y2, float strength) {
    float len = bat_dist(x1, y1, x2, y2);
    int count = (int)(len / env->reflector_spacing) + 1;
    if (count < 1) count = 1;
    for (int i = 0; i <= count; i++) {
        float t = count == 0 ? 0.0f : i / (float)count;
        float x = x1 + (x2 - x1) * t;
        float y = y1 + (y2 - y1) * t;
        bat_add_echo(env, x, y, 0.0f, 0.0f, strength);
    }
}

static inline void bat_add_obstacle_echoes(Bat* env, int i) {
    float x = env->obstacle_x[i];
    float y = env->obstacle_y[i];
    float w = env->obstacle_w[i];
    float h = env->obstacle_h[i];
    bat_add_segment_reflectors(env, x, y, x + w, y, 0.55f);
    bat_add_segment_reflectors(env, x, y + h, x + w, y + h, 0.55f);
    bat_add_segment_reflectors(env, x, y, x, y + h, 0.55f);
    bat_add_segment_reflectors(env, x + w, y, x + w, y + h, 0.55f);
}

void compute_observations(Bat* env) {
    memset(env->observations, 0, BAT_OBS_SIZE * sizeof(float));

    bat_add_echo(env, env->bug_x, env->bug_y, env->bug_vx, env->bug_vy, 8.0f);
    bat_add_segment_reflectors(env, 0.0f, 0.0f, (float)env->width, 0.0f, 0.12f);
    bat_add_segment_reflectors(env, 0.0f, (float)env->height, (float)env->width, (float)env->height, 0.12f);
    bat_add_segment_reflectors(env, 0.0f, 0.0f, 0.0f, (float)env->height, 0.12f);
    bat_add_segment_reflectors(env, (float)env->width, 0.0f, (float)env->width, (float)env->height, 0.12f);
    for (int i = 0; i < env->num_obstacles; i++) {
        bat_add_obstacle_echoes(env, i);
    }

    float left_energy = 0.0f;
    float right_energy = 0.0f;
    for (int i = 0; i < BAT_RANGE_BINS; i++) {
        env->observations[BAT_LEFT_RANGE_OFFSET + i] = bat_clampf(env->observations[BAT_LEFT_RANGE_OFFSET + i], 0.0f, 1.0f);
        env->observations[BAT_RIGHT_RANGE_OFFSET + i] = bat_clampf(env->observations[BAT_RIGHT_RANGE_OFFSET + i], 0.0f, 1.0f);
        env->observations[BAT_LEFT_DOPPLER_OFFSET + i] = bat_clampf(env->observations[BAT_LEFT_DOPPLER_OFFSET + i], -1.0f, 1.0f);
        env->observations[BAT_RIGHT_DOPPLER_OFFSET + i] = bat_clampf(env->observations[BAT_RIGHT_DOPPLER_OFFSET + i], -1.0f, 1.0f);
        left_energy += env->observations[BAT_LEFT_RANGE_OFFSET + i];
        right_energy += env->observations[BAT_RIGHT_RANGE_OFFSET + i];
    }
    env->echo_energy_left_sum += left_energy;
    env->echo_energy_right_sum += right_energy;

    env->observations[BAT_CHIRP_AGE_OBS] = bat_clampf(env->chirp_age_ticks / (float)env->max_chirp_age_ticks, 0.0f, 1.0f);
    env->observations[BAT_CHIRP_START_OBS] = env->last_chirp_start_freq;
    env->observations[BAT_CHIRP_END_OBS] = env->last_chirp_end_freq;
    env->observations[BAT_CHIRP_DURATION_OBS] = env->last_chirp_duration;
    float fwd_speed = env->bat_vx * cosf(env->bat_heading) + env->bat_vy * sinf(env->bat_heading);
    env->observations[BAT_FORWARD_SPEED_OBS] = bat_clampf(fwd_speed / env->bat_max_speed, -1.0f, 1.0f);
    env->observations[BAT_TURN_RATE_OBS] = bat_clampf(env->bat_turn_velocity / env->bat_turn_rate, -1.0f, 1.0f);
}

static inline void bat_reset_episode(Bat* env) {
    env->tick = 0;
    env->bat_x = env->width * 0.25f;
    env->bat_y = env->height * 0.5f;
    env->bat_vx = 0.0f;
    env->bat_vy = 0.0f;
    env->bat_heading = 0.0f;
    env->bat_turn_velocity = 0.0f;
    generate_obstacles(env);
    env->bug_x = env->width * 0.75f;
    env->bug_y = env->height * (0.35f + 0.30f * bat_randf(env));
    env->bug_vx = -env->bug_speed;
    env->bug_vy = (bat_randf(env) - 0.5f) * env->bug_speed * 0.5f;
    env->last_chirp_start_freq = 0.0f;
    env->last_chirp_end_freq = 1.0f;
    env->last_chirp_duration = 0.33333334f;
    env->chirp_age_ticks = 0;
    env->last_chirp_tick = -env->chirp_cooldown_ticks;
    memset(env->chirps, 0, sizeof(env->chirps));
    env->chirp_head = 0;
    env->chirps_emitted_episode = 0;
    env->chirp_duration_sum = 0.0f;
    env->chirp_bandwidth_sum = 0.0f;
    env->echo_energy_left_sum = 0.0f;
    env->echo_energy_right_sum = 0.0f;
    env->episode_return = 0.0f;
    env->start_bug_dist = bat_dist(env->bat_x, env->bat_y, env->bug_x, env->bug_y);
    env->prev_bug_dist = env->start_bug_dist;
    compute_observations(env);
}

void c_reset(Bat* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    bat_reset_episode(env);
}

static inline bool bat_hits_obstacle(Bat* env) {
    for (int i = 0; i < env->num_obstacles; i++) {
        if (bat_circle_rect_collision(env->bat_x, env->bat_y, env->bat_radius,
                env->obstacle_x[i], env->obstacle_y[i], env->obstacle_w[i], env->obstacle_h[i])) {
            return true;
        }
    }
    return false;
}

static inline bool bat_hits_wall(Bat* env) {
    return env->bat_x - env->bat_radius < 0.0f ||
        env->bat_x + env->bat_radius > env->width ||
        env->bat_y - env->bat_radius < 0.0f ||
        env->bat_y + env->bat_radius > env->height;
}

static inline void bat_update_bug(Bat* env, float dt) {
    env->bug_x += env->bug_vx * dt;
    env->bug_y += env->bug_vy * dt;
    if (env->bug_x - env->bug_radius < 0.0f) {
        env->bug_x = env->bug_radius;
        env->bug_vx = fabsf(env->bug_vx);
    }
    if (env->bug_x + env->bug_radius > env->width) {
        env->bug_x = env->width - env->bug_radius;
        env->bug_vx = -fabsf(env->bug_vx);
    }
    if (env->bug_y - env->bug_radius < 0.0f) {
        env->bug_y = env->bug_radius;
        env->bug_vy = fabsf(env->bug_vy);
    }
    if (env->bug_y + env->bug_radius > env->height) {
        env->bug_y = env->height - env->bug_radius;
        env->bug_vy = -fabsf(env->bug_vy);
    }
}

static inline void bat_update_motion(Bat* env, float dt) {
    int move = bat_action_index(env->actions[0], BAT_MOVE_ACTIONS);
    int turn = bat_action_index(env->actions[1], BAT_TURN_ACTIONS);
    float fx = cosf(env->bat_heading);
    float fy = sinf(env->bat_heading);
    float rx = -sinf(env->bat_heading);
    float ry = cosf(env->bat_heading);
    float ax = 0.0f;
    float ay = 0.0f;

    if (move == BAT_THRUST_FORWARD) {
        ax += fx * env->bat_accel;
        ay += fy * env->bat_accel;
    } else if (move == BAT_BRAKE) {
        ax -= fx * env->bat_accel;
        ay -= fy * env->bat_accel;
    } else if (move == BAT_STRAFE_LEFT) {
        ax -= rx * env->bat_accel;
        ay -= ry * env->bat_accel;
    } else if (move == BAT_STRAFE_RIGHT) {
        ax += rx * env->bat_accel;
        ay += ry * env->bat_accel;
    }

    env->bat_turn_velocity = 0.0f;
    if (turn == BAT_TURN_LEFT) env->bat_turn_velocity = -env->bat_turn_rate;
    if (turn == BAT_TURN_RIGHT) env->bat_turn_velocity = env->bat_turn_rate;
    env->bat_heading += env->bat_turn_velocity * dt;
    if (env->bat_heading > BAT_PI) env->bat_heading -= 2.0f * BAT_PI;
    if (env->bat_heading < -BAT_PI) env->bat_heading += 2.0f * BAT_PI;

    env->bat_vx += ax * dt;
    env->bat_vy += ay * dt;
    float speed = bat_len(env->bat_vx, env->bat_vy);
    if (speed > env->bat_max_speed) {
        env->bat_vx = env->bat_vx / speed * env->bat_max_speed;
        env->bat_vy = env->bat_vy / speed * env->bat_max_speed;
    }
    env->bat_x += env->bat_vx * dt;
    env->bat_y += env->bat_vy * dt;
}

static inline bool bat_try_emit_chirp(Bat* env) {
    int start_idx = bat_action_index(env->actions[2], BAT_CHIRP_FREQ_BINS);
    int end_idx = bat_action_index(env->actions[3], BAT_CHIRP_FREQ_BINS);
    int duration_idx = bat_action_index(env->actions[4], BAT_CHIRP_DURATION_BINS);

    if (env->tick - env->last_chirp_tick < env->chirp_cooldown_ticks) {
        return false;
    }

    env->last_chirp_start_freq = bat_norm_bin(start_idx, BAT_CHIRP_FREQ_BINS);
    env->last_chirp_end_freq = bat_norm_bin(end_idx, BAT_CHIRP_FREQ_BINS);
    env->last_chirp_duration = bat_norm_bin(duration_idx, BAT_CHIRP_DURATION_BINS);
    env->chirp_age_ticks = 0;
    env->last_chirp_tick = env->tick;
    env->chirps_emitted_episode += 1;
    env->chirp_duration_sum += env->last_chirp_duration;
    env->chirp_bandwidth_sum += fabsf(env->last_chirp_end_freq - env->last_chirp_start_freq);
    ChirpEvent* chirp = &env->chirps[env->chirp_head];
    chirp->x = env->bat_x;
    chirp->y = env->bat_y;
    chirp->start_freq = env->last_chirp_start_freq;
    chirp->end_freq = env->last_chirp_end_freq;
    chirp->duration = bat_chirp_duration_seconds(env->last_chirp_duration);
    chirp->birth_tick = env->tick;
    chirp->active = 1;
    env->chirp_head = (env->chirp_head + 1) % BAT_CHIRP_HISTORY;
    return true;
}

static inline bool bat_update_chirp(Bat* env) {
    int emit = bat_action_index(env->actions[5], BAT_CHIRP_EMIT_ACTIONS);
    if (emit) {
        return bat_try_emit_chirp(env);
    } else if (env->chirp_age_ticks < env->max_chirp_age_ticks) {
        env->chirp_age_ticks += 1;
    }
    return false;
}

static inline bool bat_caught_bug(Bat* env) {
    return bat_dist(env->bat_x, env->bat_y, env->bug_x, env->bug_y) <= env->bat_radius + env->bug_radius;
}

void c_step(Bat* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    bool accepted_chirp = bat_update_chirp(env);
    if (bat_caught_bug(env)) {
        env->rewards[0] = 1.0f;
        env->terminals[0] = 1.0f;
        env->episode_return += env->rewards[0];
        add_log(env, 1.0f, 0.0f, 0.0f);
        bat_reset_episode(env);
        return;
    }

    for (int i = 0; i < env->frameskip; i++) {
        bat_update_motion(env, BAT_TICK_RATE);
        bat_update_bug(env, BAT_TICK_RATE);
        if (bat_hits_wall(env) || bat_hits_obstacle(env)) {
            env->rewards[0] = -env->collision_penalty;
            env->terminals[0] = 1.0f;
            env->episode_return += env->rewards[0];
            add_log(env, 0.0f, 1.0f, 0.0f);
            bat_reset_episode(env);
            return;
        }
        if (bat_caught_bug(env)) {
            env->rewards[0] = 1.0f;
            env->terminals[0] = 1.0f;
            env->episode_return += env->rewards[0];
            add_log(env, 1.0f, 0.0f, 0.0f);
            bat_reset_episode(env);
            return;
        }
    }

    env->tick += 1;
    float bug_dist = bat_dist(env->bat_x, env->bat_y, env->bug_x, env->bug_y);
    float progress = env->prev_bug_dist - bug_dist;
    env->rewards[0] += env->progress_reward_scale * progress;
    env->rewards[0] -= env->step_cost;
    if (accepted_chirp) {
        env->rewards[0] -= env->chirp_cost;
    }
    env->prev_bug_dist = bug_dist;

    if (env->tick >= env->max_steps) {
        env->terminals[0] = 1.0f;
        env->episode_return += env->rewards[0];
        add_log(env, 0.0f, 0.0f, 1.0f);
        bat_reset_episode(env);
        return;
    }

    env->episode_return += env->rewards[0];
    compute_observations(env);
}

#ifndef BAT_HEADLESS
static inline Color bat_ray_color(BatColor c) {
    return (Color){c.r, c.g, c.b, c.a};
}

static inline void bat_draw_chirp_rings(Bat* env, float sx, float sy) {
    float scale = fminf(sx, sy);
    for (int i = 0; i < BAT_CHIRP_HISTORY; i++) {
        ChirpEvent* chirp = &env->chirps[i];
        if (!chirp->active) continue;

        float age_seconds = (env->tick - chirp->birth_tick) * BAT_TICK_RATE;
        float max_age = env->max_echo_range / env->sound_speed + chirp->duration;
        if (age_seconds < 0.0f || age_seconds > max_age) {
            chirp->active = 0;
            continue;
        }

        for (int ring = 0; ring < BAT_CHIRP_RINGS; ring++) {
            float slice = ring / (float)(BAT_CHIRP_RINGS - 1);
            float freq = chirp->start_freq + slice * (chirp->end_freq - chirp->start_freq);
            float radius = bat_chirp_ring_radius(age_seconds, slice, chirp->duration, env->sound_speed);
            if (radius <= 0.0f || radius > env->max_echo_range) continue;

            float fade = 1.0f - radius / env->max_echo_range;
            float alpha = 0.18f + 0.42f * bat_clampf(fade, 0.0f, 1.0f);
            DrawCircleLines(
                (int)(chirp->x * sx),
                (int)(chirp->y * sy),
                radius * scale,
                bat_ray_color(bat_freq_color(freq, alpha)));
        }
    }
}

static inline Color bat_doppler_ray_color(float doppler, float alpha) {
    BatColor c;
    if (doppler > 0.05f) {
        c = bat_freq_color(1.0f, alpha);
    } else if (doppler < -0.05f) {
        c = bat_freq_color(0.0f, alpha);
    } else {
        c = (BatColor){210, 210, 220, (unsigned char)(255.0f * bat_clampf(alpha, 0.0f, 1.0f))};
    }
    return bat_ray_color(c);
}

static inline void bat_draw_echo_flash(Bat* env, ChirpEvent* chirp,
        float rx, float ry, float rvx, float rvy, float strength,
        float sx, float sy) {
    float age_seconds = (env->tick - chirp->birth_tick) * BAT_TICK_RATE;
    float distance = bat_dist(chirp->x, chirp->y, rx, ry);
    float echo_time = bat_echo_time_seconds(distance, env->sound_speed);
    if (!bat_echo_is_arriving(echo_time, age_seconds, 0.025f)) return;

    float ux, uy;
    bat_norm_vec(rx - chirp->x, ry - chirp->y, &ux, &uy);
    float rel_vx = rvx - env->bat_vx;
    float rel_vy = rvy - env->bat_vy;
    float distance_rate = rel_vx * ux + rel_vy * uy;
    float doppler = bat_clampf(-distance_rate / (env->bat_max_speed + env->bug_speed + 0.0001f), -1.0f, 1.0f);
    float amp = strength / (1.0f + 0.02f * distance * distance);
    float alpha = bat_clampf(0.20f + amp * 2.0f, 0.20f, 0.90f);
    Color color = bat_doppler_ray_color(doppler, alpha);

    DrawLine((int)(chirp->x * sx), (int)(chirp->y * sy),
        (int)(rx * sx), (int)(ry * sy), color);
    DrawCircleLines((int)(rx * sx), (int)(ry * sy),
        fmaxf(3.0f, 8.0f * alpha), color);
}

static inline void bat_draw_segment_echoes(Bat* env, ChirpEvent* chirp,
        float x1, float y1, float x2, float y2, float strength,
        float sx, float sy) {
    float len = bat_dist(x1, y1, x2, y2);
    int count = (int)(len / env->reflector_spacing) + 1;
    if (count < 1) count = 1;
    for (int i = 0; i <= count; i++) {
        float t = i / (float)count;
        float x = x1 + (x2 - x1) * t;
        float y = y1 + (y2 - y1) * t;
        bat_draw_echo_flash(env, chirp, x, y, 0.0f, 0.0f, strength, sx, sy);
    }
}

static inline void bat_draw_obstacle_echoes(Bat* env, ChirpEvent* chirp,
        int i, float sx, float sy) {
    float x = env->obstacle_x[i];
    float y = env->obstacle_y[i];
    float w = env->obstacle_w[i];
    float h = env->obstacle_h[i];
    bat_draw_segment_echoes(env, chirp, x, y, x + w, y, 0.55f, sx, sy);
    bat_draw_segment_echoes(env, chirp, x, y + h, x + w, y + h, 0.55f, sx, sy);
    bat_draw_segment_echoes(env, chirp, x, y, x, y + h, 0.55f, sx, sy);
    bat_draw_segment_echoes(env, chirp, x + w, y, x + w, y + h, 0.55f, sx, sy);
}

static inline void bat_draw_echo_reflections(Bat* env, float sx, float sy) {
    for (int i = 0; i < BAT_CHIRP_HISTORY; i++) {
        ChirpEvent* chirp = &env->chirps[i];
        if (!chirp->active) continue;
        bat_draw_echo_flash(env, chirp, env->bug_x, env->bug_y,
            env->bug_vx, env->bug_vy, 4.0f, sx, sy);
        bat_draw_segment_echoes(env, chirp, 0.0f, 0.0f, (float)env->width, 0.0f, 0.18f, sx, sy);
        bat_draw_segment_echoes(env, chirp, 0.0f, (float)env->height, (float)env->width, (float)env->height, 0.18f, sx, sy);
        bat_draw_segment_echoes(env, chirp, 0.0f, 0.0f, 0.0f, (float)env->height, 0.18f, sx, sy);
        bat_draw_segment_echoes(env, chirp, (float)env->width, 0.0f, (float)env->width, (float)env->height, 0.18f, sx, sy);
        for (int j = 0; j < env->num_obstacles; j++) {
            bat_draw_obstacle_echoes(env, chirp, j, sx, sy);
        }
    }
}

Client* make_client(Bat* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width * 10;
    client->height = env->height * 10;
    InitWindow(client->width, client->height, "Bat");
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void c_render(Bat* env) {
    if (IsKeyPressed(KEY_ESCAPE)) {
        exit(0);
    }
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    float sx = env->client->width / (float)env->width;
    float sy = env->client->height / (float)env->height;
    BeginDrawing();
    ClearBackground((Color){18, 20, 24, 255});
    bat_draw_chirp_rings(env, sx, sy);
    bat_draw_echo_reflections(env, sx, sy);
    DrawRectangleLines(0, 0, env->client->width, env->client->height, GRAY);
    for (int i = 0; i < env->num_obstacles; i++) {
        DrawRectangle(
            (int)(env->obstacle_x[i] * sx),
            (int)(env->obstacle_y[i] * sy),
            (int)(env->obstacle_w[i] * sx),
            (int)(env->obstacle_h[i] * sy),
            (Color){92, 92, 96, 255});
    }
    DrawCircle((int)(env->bug_x * sx), (int)(env->bug_y * sy),
        env->bug_radius * sx, GREEN);
    DrawCircle((int)(env->bat_x * sx), (int)(env->bat_y * sy),
        env->bat_radius * sx, BLUE);
    float hx = env->bat_x + cosf(env->bat_heading) * env->bat_radius * 2.0f;
    float hy = env->bat_y + sinf(env->bat_heading) * env->bat_radius * 2.0f;
    DrawLine((int)(env->bat_x * sx), (int)(env->bat_y * sy), (int)(hx * sx), (int)(hy * sy), WHITE);
    int cooldown = env->chirp_cooldown_ticks - (env->tick - env->last_chirp_tick);
    if (cooldown < 0) cooldown = 0;
    DrawText(TextFormat("reward %.3f tick %d chirps %d cooldown %d ESC exits", env->rewards[0], env->tick,
        env->chirps_emitted_episode, cooldown), 10, 10, 20, RAYWHITE);
    EndDrawing();
}
#else
Client* make_client(Bat* env) {
    (void)env;
    return NULL;
}

void close_client(Client* client) {
    (void)client;
}

void c_render(Bat* env) {
    (void)env;
}
#endif
