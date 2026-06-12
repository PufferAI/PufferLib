#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>

#ifndef BAT_HEADLESS
#include "raylib.h"
#endif

#define OBS_SIZE 41
#define NUM_AGENTS 1
#define NUM_ACTIONS 6
#define MOVE_ACTIONS 3
#define TURN_ACTIONS 3
#define CHIRP_FREQ_BINS 8
#define CHIRP_DURATION_BINS 4
#define CHIRP_EMIT_ACTIONS 2

#define FREQ_BINS 16
#define LEFT_FREQ_OFFSET 0
#define RIGHT_FREQ_OFFSET 16
#define CHIRP_AGE_OBS 32
#define CHIRP_COOLDOWN_OBS 33
#define CHIRP_START_OBS 34
#define CHIRP_END_OBS 35
#define CHIRP_DURATION_OBS 36
#define CHIRPS_USED_OBS 37
#define FORWARD_SPEED_OBS 38
#define TURN_RATE_OBS 39

#define NOOP 0
#define THRUST_FORWARD 1
#define BRAKE 2

#define TURN_NONE 0
#define TURN_LEFT 1
#define TURN_RIGHT 2

#define MAX_OBSTACLES 16
#define MAX_STEPS 512
#define TICK_RATE (1.0f/60.0f)
#define ARENA_WIDTH 64
#define ARENA_HEIGHT 64
#define AGENT_RADIUS 2.0f
#define BUG_RADIUS 1.5f
#define BUG_SPEED 4.0f
#define BUG_MANEUVER_START_LEVEL 7
#define BUG_MANEUVER_STRENGTH 0.4f
#define BUG_MANEUVER_FREQUENCY 0.4f
#define INBOUND_BUG_SPEED_MULTIPLIER 1.75f
#define INBOUND_HEADING_NOISE_DEGREES 18.0f
#define REFLECTOR_SPACING 8.0f
#define MAX_ECHO_RANGE 128.0f
#define BUG_ECHO_MIN_DISPLACEMENT 1.0f
#define CURRICULUM_START_OBSTACLES 0
#define CURRICULUM_MAX_OBSTACLES 3
#define CURRICULUM_BUG_DISTANCE_STEP 2.0f
#define CURRICULUM_MAX_BUG_DISTANCE 40.0f
#define CURRICULUM_INBOUND_START_LEVEL 8
#define CURRICULUM_INBOUND_MAX_BUG_DISTANCE 56.0f
#define CURRICULUM_INBOUND_BUG_DISTANCE_STEP 4.0f
#define PI_F 3.14159265358979323846f
#define TWO_PI (2.0f * PI_F)
#define CHIRP_HISTORY 4
#define CHIRP_RINGS 5
#define MAX_CHIRP_SLICES 16
#define ECHO_QUEUE_TICKS 256
#define CORNER_REFLECTORS 1
#define AUDIO_VOICES 8
#define AUDIO_SAMPLE_RATE 48000
#define AUDIO_MIN_HZ 600.0f
#define AUDIO_MAX_HZ 3600.0f
#define AUDIO_VOLUME 0.22f
#define RECORD_MAX_VOICES 16
#define CHIRP_PERF_FLOOR 0.05f
#define CHIRP_COST 0.0f
#define MAX_CHIRP_AGE_TICKS 30
#define MAX_CHIRPS_PER_EPISODE 15

#define ECHO_STATIC 0
#define ECHO_BUG 1

typedef struct ChirpEvent {
    float x;
    float y;
    float source_x[MAX_CHIRP_SLICES];
    float source_y[MAX_CHIRP_SLICES];
    float start_freq;
    float end_freq;
    float duration;
    int birth_tick;
    int slice_count;
    int slices_scheduled;
    int active;
} ChirpEvent;

typedef struct EchoBucket {
    float energy[2][FREQ_BINS];
    float bug_energy;
    float bug_path;
    int tick;
} EchoBucket;

typedef struct BatRecordVoice {
    int active;
    int start_sample;
    float start_freq;
    float end_freq;
    float duration;
} BatRecordVoice;

typedef struct Log {
    float perf;
    float base_perf;
    float score;
    float episode_return;
    float episode_length;
    float collision;
    float timeout;
    float curriculum_level;
    float curriculum_difficulty;
    float curriculum_perf;
    float curriculum_distance_difficulty;
    float curriculum_obstacle_difficulty;
    float curriculum_motion_difficulty;
    float num_obstacles;
    float chirps_emitted;
    float chirp_perf;
    float chirp_overlap_fraction;
    float n;
} Log;

typedef struct Client {
    int width;
    int height;
#ifndef BAT_HEADLESS
    int audio_ready;
    int last_audio_chirp_serial;
    int audio_voice_cursor;
    Sound chirp_sounds[AUDIO_VOICES];
    int chirp_sound_loaded[AUDIO_VOICES];
    int recording_initialized;
    int recording_finalized;
    int record_frame;
    int record_max_frames;
    int record_fps;
    int record_audio;
    int record_last_audio_chirp_serial;
    int record_audio_sample_cursor;
    int record_audio_data_bytes;
    int record_voice_cursor;
    FILE* record_wav;
    char record_frame_dir[256];
    char record_wav_path[256];
    char record_mp4_path[256];
    BatRecordVoice record_voices[RECORD_MAX_VOICES];
#endif
} Client;

typedef enum ChirpStatus {
    CHIRP_STATUS_OVER_BUDGET = -2,
    CHIRP_STATUS_COOLDOWN = -1,
    CHIRP_STATUS_NONE = 0,
    CHIRP_STATUS_EMITTED = 1,
} ChirpStatus;

typedef struct Bat {
    Client* client;
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;

    int tick;
    int render_target_fps;
    int record_video;
    int record_video_fps;
    int record_video_seconds;
    int record_video_audio;
    int num_obstacles;
    int curriculum_level;
    int curriculum_initial_level;
    int curriculum_obstacle_step;
    int curriculum_successes_per_level;
    int curriculum_successes_at_level;
    float curriculum_start_bug_distance;

    float x;
    float y;
    float vx;
    float vy;
    float heading;
    float turn_velocity;
    float ear_separation_scale;
    float ear_rear_gain;
    float ear_front_gain;
    float ear_side_gain;
    float max_speed;
    float min_speed;
    float accel;
    float turn_rate;

    float bug_x;
    float bug_y;
    float bug_vx;
    float bug_vy;
    int bug_inbound;
    int bug_maneuver_mode;
    float bug_base_heading;
    float bug_maneuver_phase;
    float bug_maneuver_rate;
    float bug_maneuver_sign;

    float* obstacle_x;
    float* obstacle_y;
    float* obstacle_w;
    float* obstacle_h;

    float sound_speed;
    float reflector_strength;
    int chirp_cooldown_ticks;
    int chirp_budget;
    int chirp_age_ticks;
    int last_chirp_tick;
    float last_chirp_start_freq;
    float last_chirp_end_freq;
    float last_chirp_duration;
    ChirpEvent chirps[CHIRP_HISTORY];
    int chirp_head;
    EchoBucket echo_queue[ECHO_QUEUE_TICKS];
    int chirps_emitted;
    int audio_chirp_serial;
    int chirps_overlapped;

    float chirp_efficiency_reward;
    float valid_chirp_reward;
    float early_chirp_penalty;
    float chirp_overlap_penalty;
    float step_cost;
    float progress_reward_scale;
    float bug_echo_reward_scale;
    float bug_echo_farther_penalty_scale;
    float bug_wing_sideband_gain;
    float tick_bug_echo_energy;
    float tick_bug_echo_path;
    float last_bug_echo_path;
    float last_bug_echo_expected_tick;
    float last_bug_echo_x;
    float last_bug_echo_y;
    float collision_penalty;
    float prev_bug_dist;
    float start_bug_dist;
    float episode_return;

    unsigned int rng;
} Bat;

static inline unsigned int rng_next(Bat* env) {
    env->rng = env->rng * 1664525u + 1013904223u;
    return env->rng;
}

static inline float randf(Bat* env) {
    return (rng_next(env) >> 8) * (1.0f / 16777216.0f);
}

static inline float bat_clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline int action_index(float v, int n) {
    int idx = (int)v;
    return idx;
}

static inline float chirp_duration_seconds(float duration_norm) {
    return 0.04f + 0.18f * duration_norm;
}

#include "bat_audio.h"

static inline float chirp_ring_radius(float age_seconds, float slice,
        float duration_seconds, float sound_speed) {
    float ring_age = age_seconds - slice * duration_seconds;
    if (ring_age < 0.0f) return 0.0f;
    return sound_speed * ring_age;
}

static inline float chirp_slice_ticks(ChirpEvent* chirp, int slice_idx) {
    return ((slice_idx + 0.5f) / (float)chirp->slice_count) *
        chirp->duration / TICK_RATE;
}

static inline void chirp_source_for_slice(ChirpEvent* chirp, int slice_idx,
        float* source_x, float* source_y) {
    int scheduled = chirp->slices_scheduled;
    if (slice_idx >= 0 && slice_idx < scheduled &&
            slice_idx < MAX_CHIRP_SLICES) {
        *source_x = chirp->source_x[slice_idx];
        *source_y = chirp->source_y[slice_idx];
        return;
    }
    *source_x = chirp->x;
    *source_y = chirp->y;
}

static inline void chirp_source_for_fraction(ChirpEvent* chirp, float slice,
        float* source_x, float* source_y) {
    int slices = chirp->slice_count;
    int slice_idx = (int)floorf(slice * (float)slices);
    if (slice_idx >= slices) slice_idx = slices - 1;
    chirp_source_for_slice(chirp, slice_idx, source_x, source_y);
}

static inline float echo_time_seconds(float distance, float sound_speed) {
    return 2.0f * distance / sound_speed;
}

static inline float chirp_age_norm_denominator(Bat* env) {
    float travel_ticks = MAX_ECHO_RANGE / env->sound_speed / TICK_RATE;
    float chirp_ticks = chirp_duration_seconds(1.0f) / TICK_RATE;
    return 1.25f * (travel_ticks + chirp_ticks);
}

static inline float norm_bin(int idx, int count) {
    return idx / (float)(count - 1);
}

static inline float dist(float ax, float ay, float bx, float by) {
    float dx = bx - ax;
    float dy = by - ay;
    return sqrtf(dx*dx + dy*dy);
}

static inline void norm_vec(float x, float y, float* ox, float* oy) {
    float l = sqrtf(x*x + y*y);
    if (l <= 0.000001f) {
        *ox = 1.0f;
        *oy = 0.0f;
        return;
    }
    *ox = x / l;
    *oy = y / l;
}

static inline bool circle_rect_collision(float cx, float cy, float r,
        float rx, float ry, float rw, float rh) {
    float px = bat_clampf(cx, rx, rx + rw);
    float py = bat_clampf(cy, ry, ry + rh);
    return dist(cx, cy, px, py) <= r;
}

static inline bool rects_overlap(float ax, float ay, float aw, float ah,
        float bx, float by, float bw, float bh, float margin) {
    return ax - margin < bx + bw &&
        ax + aw + margin > bx &&
        ay - margin < by + bh &&
        ay + ah + margin > by;
}

static inline void sample_in_quadrant(Bat* env, int quadrant, float radius,
        float* x, float* y) {
    int east = quadrant & 1;
    int south = (quadrant >> 1) & 1;
    float margin = fmaxf(6.0f, radius + 3.0f);
    float half_w = ARENA_WIDTH * 0.5f;
    float half_h = ARENA_HEIGHT * 0.5f;
    float min_x = (east ? half_w : 0.0f) + margin;
    float max_x = (east ? (float)ARENA_WIDTH : half_w) - margin;
    float min_y = (south ? half_h : 0.0f) + margin;
    float max_y = (south ? (float)ARENA_HEIGHT : half_h) - margin;
    *x = min_x + randf(env) * (max_x - min_x);
    *y = min_y + randf(env) * (max_y - min_y);
}

static inline void sample_spawns(Bat* env) {
    int agent_quadrant = (int)(randf(env) * 4.0f);
    int bug_quadrant = agent_quadrant ^ 3;
    float min_sep = fminf(ARENA_WIDTH, ARENA_HEIGHT) * 0.31f;

    for (int attempt = 0; attempt < 64; attempt++) {
        sample_in_quadrant(env, agent_quadrant, AGENT_RADIUS, &env->x, &env->y);
        sample_in_quadrant(env, bug_quadrant, BUG_RADIUS, &env->bug_x, &env->bug_y);
        if (dist(env->x, env->y, env->bug_x, env->bug_y) >= min_sep) {
            return;
        }
    }

    float qx[4] = {0.25f, 0.75f, 0.25f, 0.75f};
    float qy[4] = {0.25f, 0.25f, 0.75f, 0.75f};
    env->x = ARENA_WIDTH * qx[agent_quadrant];
    env->y = ARENA_HEIGHT * qy[agent_quadrant];
    env->bug_x = ARENA_WIDTH * qx[bug_quadrant];
    env->bug_y = ARENA_HEIGHT * qy[bug_quadrant];
}

static inline int curriculum_obstacles(Bat* env) {
    int step = env->curriculum_obstacle_step;
    int count = CURRICULUM_START_OBSTACLES;
    if (env->curriculum_level > 0) {
        count = CURRICULUM_START_OBSTACLES + 1 + (env->curriculum_level - 1) / step;
    }
    if (count > CURRICULUM_MAX_OBSTACLES) count = CURRICULUM_MAX_OBSTACLES;
    if (count > MAX_OBSTACLES) count = MAX_OBSTACLES;
    return count;
}

static inline float curriculum_bug_distance(Bat* env) {
    float distance = env->curriculum_start_bug_distance
        + CURRICULUM_BUG_DISTANCE_STEP * env->curriculum_level;
    return bat_clampf(distance, env->curriculum_start_bug_distance,
        CURRICULUM_MAX_BUG_DISTANCE);
}

static inline bool curriculum_inbound_enabled(Bat* env) {
    return env->curriculum_level >= CURRICULUM_INBOUND_START_LEVEL;
}

static inline float curriculum_inbound_bug_distance(Bat* env) {
    float base = CURRICULUM_MAX_BUG_DISTANCE;
    int extra_levels = env->curriculum_level - CURRICULUM_INBOUND_START_LEVEL + 1;
    float distance = base + CURRICULUM_INBOUND_BUG_DISTANCE_STEP * extra_levels;
    return bat_clampf(distance, base, CURRICULUM_INBOUND_MAX_BUG_DISTANCE);
}

static inline float curriculum_spawn_distance(Bat* env) {
    if (curriculum_inbound_enabled(env)) {
        return curriculum_inbound_bug_distance(env);
    }
    return curriculum_bug_distance(env);
}

static inline float curriculum_bug_speed(Bat* env) {
    float speed = BUG_SPEED;
    if (curriculum_inbound_enabled(env)) {
        speed *= INBOUND_BUG_SPEED_MULTIPLIER;
    }
    return speed;
}

static inline float curriculum_bug_maneuver_strength(Bat* env) {
    if (env->curriculum_level < BUG_MANEUVER_START_LEVEL) return 0.0f;
    int extra_levels = env->curriculum_level - BUG_MANEUVER_START_LEVEL;
    float ramp = extra_levels <= 0 ? 0.25f : 0.75f + 0.25f * (extra_levels - 1);
    return BUG_MANEUVER_STRENGTH * bat_clampf(ramp, 0.0f, 1.0f);
}

static inline float curriculum_bug_maneuver_frequency(Bat* env) {
    if (env->curriculum_level < BUG_MANEUVER_START_LEVEL) {
        return BUG_MANEUVER_FREQUENCY;
    }
    int extra_levels = env->curriculum_level - BUG_MANEUVER_START_LEVEL;
    float multiplier = 1.0f + 0.50f * extra_levels;
    return BUG_MANEUVER_FREQUENCY * bat_clampf(multiplier, 1.0f, 2.5f);
}

static inline float chirps_used_ratio(Bat* env) {
    return bat_clampf(env->chirps_emitted / (float)env->chirp_budget, 0.0f, 1.0f);
}

static inline float chirp_efficiency(Bat* env) {
    return 0.5f + 0.5f * (1.0f - chirps_used_ratio(env));
}

static inline float chirp_perf(Bat* env) {
    float reference_chirps = fmaxf(1.0f, (float)MAX_CHIRPS_PER_EPISODE);
    float raw = 1.0f - env->chirps_emitted / reference_chirps;
    return bat_clampf(raw, CHIRP_PERF_FLOOR, 1.0f);
}

static inline float norm_range(float value, float lo, float hi) {
    float span = hi - lo;
    return bat_clampf((value - lo) / span, 0.0f, 1.0f);
}

static inline float curriculum_distance_difficulty(Bat* env) {
    float max_distance = fmaxf(CURRICULUM_MAX_BUG_DISTANCE,
        CURRICULUM_INBOUND_MAX_BUG_DISTANCE);
    return norm_range(env->start_bug_dist,
        env->curriculum_start_bug_distance, max_distance);
}

static inline float curriculum_obstacle_difficulty(Bat* env) {
    return norm_range((float)env->num_obstacles,
        (float)CURRICULUM_START_OBSTACLES, (float)CURRICULUM_MAX_OBSTACLES);
}

static inline float curriculum_motion_difficulty(Bat* env) {
    if (env->curriculum_level < BUG_MANEUVER_START_LEVEL) return 0.0f;
    float span = (float)(CURRICULUM_INBOUND_START_LEVEL + 4 - BUG_MANEUVER_START_LEVEL);
    return bat_clampf((env->curriculum_level - BUG_MANEUVER_START_LEVEL + 1) / span,
        0.0f, 1.0f);
}

static inline float curriculum_difficulty(Bat* env) {
    float distance = curriculum_distance_difficulty(env);
    float obstacles = curriculum_obstacle_difficulty(env);
    float active_weight = 0.0f;
    float weighted = 0.0f;
    if (CURRICULUM_MAX_BUG_DISTANCE > env->curriculum_start_bug_distance) {
        weighted += 0.5f * distance;
        active_weight += 0.5f;
    }
    if (CURRICULUM_MAX_OBSTACLES > CURRICULUM_START_OBSTACLES) {
        weighted += 0.5f * obstacles;
        active_weight += 0.5f;
    }
    float motion = curriculum_motion_difficulty(env);
    if (BUG_MANEUVER_STRENGTH > 0.0f) {
        weighted += 0.5f * motion;
        active_weight += 0.5f;
    }
    return bat_clampf(weighted / active_weight, 0.0f, 1.0f);
}

static inline float success_reward(Bat* env) {
    return env->chirp_efficiency_reward * chirp_efficiency(env);
}

static inline void sample_spawns_at_distance(Bat* env, float target_distance) {
    float margin = fmaxf(6.0f, fmaxf(AGENT_RADIUS, BUG_RADIUS) + 3.0f);
    for (int attempt = 0; attempt < 96; attempt++) {
        float angle = randf(env) * TWO_PI - PI_F;
        float dx = cosf(angle) * target_distance;
        float dy = sinf(angle) * target_distance;
        float min_bat_x = fmaxf(margin, margin - dx);
        float max_bat_x = fminf(ARENA_WIDTH - margin, ARENA_WIDTH - margin - dx);
        float min_bat_y = fmaxf(margin, margin - dy);
        float max_bat_y = fminf(ARENA_HEIGHT - margin, ARENA_HEIGHT - margin - dy);
        if (max_bat_x < min_bat_x || max_bat_y < min_bat_y) continue;

        env->x = min_bat_x + randf(env) * (max_bat_x - min_bat_x);
        env->y = min_bat_y + randf(env) * (max_bat_y - min_bat_y);
        env->bug_x = env->x + dx;
        env->bug_y = env->y + dy;
        return;
    }

    sample_spawns(env);
}

static inline void set_bug_velocity(Bat* env, float heading, float speed) {
    env->bug_base_heading = heading;
    env->bug_vx = cosf(heading) * speed;
    env->bug_vy = sinf(heading) * speed;
}

static inline void reset_bug_motion(Bat* env) {
    env->bug_inbound = curriculum_inbound_enabled(env) ? 1 : 0;
    float strength = curriculum_bug_maneuver_strength(env);
    env->bug_maneuver_mode = strength > 0.000001f ? 1 + (int)(rng_next(env) % 3u) : 0;
    env->bug_maneuver_phase = randf(env) * TWO_PI;
    env->bug_maneuver_rate = TWO_PI * curriculum_bug_maneuver_frequency(env) *
        (0.75f + 0.50f * randf(env));
    env->bug_maneuver_sign = (rng_next(env) & 1u) ? -1.0f : 1.0f;

    float speed = curriculum_bug_speed(env);
    if (env->bug_inbound) {
        float tx, ty;
        norm_vec(env->x - env->bug_x, env->y - env->bug_y, &tx, &ty);
        float noise = INBOUND_HEADING_NOISE_DEGREES * (PI_F / 180.0f);
        float heading = atan2f(ty, tx) + (2.0f * randf(env) - 1.0f) * noise;
        set_bug_velocity(env, heading, speed);
    } else {
        float heading = randf(env) * TWO_PI - PI_F;
        set_bug_velocity(env, heading, speed);
    }
}

static inline void apply_curriculum(Bat* env) {
    env->num_obstacles = curriculum_obstacles(env);
}

static inline void advance_curriculum(Bat* env) {
    env->curriculum_successes_at_level += 1;
    if (env->curriculum_successes_at_level >= env->curriculum_successes_per_level) {
        env->curriculum_level += 1;
        env->curriculum_successes_at_level = 0;
    }
}

static inline bool obstacle_clear(Bat* env, int idx, float x, float y,
        float w, float h) {
    if (circle_rect_collision(env->x, env->y, AGENT_RADIUS + 2.0f, x, y, w, h)) {
        return false;
    }
    if (circle_rect_collision(env->bug_x, env->bug_y, BUG_RADIUS + 2.0f, x, y, w, h)) {
        return false;
    }
    for (int j = 0; j < idx; j++) {
        if (rects_overlap(x, y, w, h,
                env->obstacle_x[j], env->obstacle_y[j], env->obstacle_w[j], env->obstacle_h[j], 3.0f)) {
            return false;
        }
    }
    return true;
}

static inline void generate_obstacles(Bat* env) {
    for (int i = 0; i < env->num_obstacles; i++) {
        bool placed = false;
        for (int attempt = 0; attempt < 96; attempt++) {
            float w = 3.0f + 5.0f * randf(env);
            float h = 3.0f + 5.0f * randf(env);
            float margin = 4.0f;
            float x = margin + randf(env) * (ARENA_WIDTH - w - 2.0f * margin);
            float y = margin + randf(env) * (ARENA_HEIGHT - h - 2.0f * margin);
            if (obstacle_clear(env, i, x, y, w, h)) {
                env->obstacle_x[i] = x;
                env->obstacle_y[i] = y;
                env->obstacle_w[i] = w;
                env->obstacle_h[i] = h;
                placed = true;
                break;
            }
        }
        if (!placed) {
            float w = 6.0f;
            float h = 6.0f;
            float x = ARENA_WIDTH * (0.30f + 0.20f * (i % 2)) - w * 0.5f;
            float y = ARENA_HEIGHT * (0.30f + 0.20f * ((i + 1) % 2)) - h * 0.5f;
            env->obstacle_x[i] = x;
            env->obstacle_y[i] = y;
            env->obstacle_w[i] = w;
            env->obstacle_h[i] = h;
        }
    }
}

void init(Bat* env) {
    env->tick = 0;
    env->obstacle_x = (float*)calloc(MAX_OBSTACLES, sizeof(float));
    env->obstacle_y = (float*)calloc(MAX_OBSTACLES, sizeof(float));
    env->obstacle_w = (float*)calloc(MAX_OBSTACLES, sizeof(float));
    env->obstacle_h = (float*)calloc(MAX_OBSTACLES, sizeof(float));
}

void allocate(Bat* env) {
    init(env);
    env->observations = (float*)calloc(OBS_SIZE, sizeof(float));
    env->actions = (float*)calloc(NUM_ACTIONS, sizeof(float));
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
    float curriculum_difficulty_value = curriculum_difficulty(env);
    float distance_difficulty = curriculum_distance_difficulty(env);
    float obstacle_difficulty = curriculum_obstacle_difficulty(env);
    float motion_difficulty = curriculum_motion_difficulty(env);
    float chirp_perf_value = chirp_perf(env);
    env->log.perf += success * curriculum_difficulty_value * chirp_perf_value;
    env->log.base_perf += success;
    env->log.score += env->episode_return;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->tick;
    env->log.collision += collision;
    env->log.timeout += timeout;
    env->log.curriculum_level += env->curriculum_level;
    env->log.curriculum_difficulty += curriculum_difficulty_value;
    env->log.curriculum_perf += success * curriculum_difficulty_value;
    env->log.curriculum_distance_difficulty += distance_difficulty;
    env->log.curriculum_obstacle_difficulty += obstacle_difficulty;
    env->log.curriculum_motion_difficulty += motion_difficulty;
    env->log.num_obstacles += env->num_obstacles;
    env->log.chirps_emitted += env->chirps_emitted;
    env->log.chirp_perf += chirp_perf_value;
    float chirps = fmaxf(1.0f, (float)env->chirps_emitted);
    env->log.chirp_overlap_fraction += env->chirps_overlapped / chirps;
    env->log.n += 1.0f;
}

static inline int freq_bin_index(Bat* env, float freq_norm) {
    (void)env;
    int bins = FREQ_BINS;
    int bin = (int)(freq_norm * bins);
    if (bin >= bins) bin = bins - 1;
    return bin;
}

static inline void clear_echo_bucket(EchoBucket* bucket) {
    memset(bucket, 0, sizeof(*bucket));
    bucket->bug_path = -1.0f;
    bucket->tick = -1;
}

static inline void clear_echo_queue(Bat* env) {
    for (int i = 0; i < ECHO_QUEUE_TICKS; i++) {
        clear_echo_bucket(&env->echo_queue[i]);
    }
}

static inline void add_echo_event(Bat* env, int ear, float receive_tick,
        float freq, float intensity, float path, int source) {
    if (receive_tick <= env->tick) return;
    if (intensity <= 0.000001f) return;
    int arrival_tick = (int)ceilf(receive_tick);
    int delay = arrival_tick - env->tick;
    if (delay <= 0 || delay >= ECHO_QUEUE_TICKS) return;
    int slot = arrival_tick % ECHO_QUEUE_TICKS;
    EchoBucket* bucket = &env->echo_queue[slot];
    if (bucket->tick != arrival_tick) {
        clear_echo_bucket(bucket);
        bucket->tick = arrival_tick;
    }

    int ear_idx = ear == 0 ? 0 : 1;
    int bin = freq_bin_index(env, freq);
    bucket->energy[ear_idx][bin] += intensity;
    if (source == ECHO_BUG) {
        float sideband = intensity * env->bug_wing_sideband_gain;
        if (sideband > 0.000001f) {
            if (bin > 0) bucket->energy[ear_idx][bin - 1] += sideband;
            if (bin + 1 < FREQ_BINS) bucket->energy[ear_idx][bin + 1] += sideband;
        }
        bucket->bug_energy += intensity;
        if (bucket->bug_path < 0.0f || path < bucket->bug_path) {
            bucket->bug_path = path;
        }
    }
}

static inline void ear_positions(Bat* env, float* left_x, float* left_y,
        float* right_x, float* right_y) {
    float lx = -sinf(env->heading);
    float ly = cosf(env->heading);
    float ear_sep = AGENT_RADIUS * env->ear_separation_scale;
    *left_x = env->x - lx * ear_sep * 0.5f;
    *left_y = env->y - ly * ear_sep * 0.5f;
    *right_x = env->x + lx * ear_sep * 0.5f;
    *right_y = env->y + ly * ear_sep * 0.5f;
}

static inline float expected_bug_echo_tick(Bat* env, ChirpEvent* chirp) {
    float fx = cosf(env->heading);
    float fy = sinf(env->heading);
    float source_x, source_y;
    chirp_source_for_slice(chirp, 0, &source_x, &source_y);
    float ux, uy;
    norm_vec(env->bug_x - source_x, env->bug_y - source_y, &ux, &uy);
    float forward = ux * fx + uy * fy;
    if (forward < -0.35f) return -1.0f;

    float left_ear_x, left_ear_y, right_ear_x, right_ear_y;
    ear_positions(env, &left_ear_x, &left_ear_y, &right_ear_x, &right_ear_y);
    float source_path = dist(source_x, source_y, env->bug_x, env->bug_y);
    float left_path = source_path + dist(env->bug_x, env->bug_y, left_ear_x, left_ear_y);
    float right_path = source_path + dist(env->bug_x, env->bug_y, right_ear_x, right_ear_y);
    float best_path = -1.0f;
    if (left_path <= MAX_ECHO_RANGE) best_path = left_path;
    if (right_path <= MAX_ECHO_RANGE && (best_path < 0.0f || right_path < best_path)) {
        best_path = right_path;
    }
    if (best_path < 0.0f) return -1.0f;

    float first_slice_ticks = chirp_slice_ticks(chirp, 0);
    return chirp->birth_tick + first_slice_ticks + best_path / env->sound_speed / TICK_RATE;
}

static inline void schedule_echo(Bat* env, ChirpEvent* chirp,
        float slice_ticks, float freq, float rx, float ry, float rvx, float rvy,
        float strength, int source) {
    float fx = cosf(env->heading);
    float fy = sinf(env->heading);
    float lx = -sinf(env->heading);
    float ly = cosf(env->heading);
    float left_ear_x, left_ear_y, right_ear_x, right_ear_y;
    ear_positions(env, &left_ear_x, &left_ear_y, &right_ear_x, &right_ear_y);

    float ux, uy;
    norm_vec(rx - chirp->x, ry - chirp->y, &ux, &uy);
    float forward = ux * fx + uy * fy;
    if (forward < -0.35f) return;

    float left_dir_x = -lx;
    float left_dir_y = -ly;
    float right_dir_x = lx;
    float right_dir_y = ly;
    float front_gain = bat_clampf(forward, 0.0f, 1.0f);
    float left_side_gain = bat_clampf(ux * left_dir_x + uy * left_dir_y, 0.0f, 1.0f);
    float right_side_gain = bat_clampf(ux * right_dir_x + uy * right_dir_y, 0.0f, 1.0f);
    front_gain *= front_gain;
    left_side_gain *= left_side_gain;
    right_side_gain *= right_side_gain;
    float left_gain = env->ear_rear_gain + env->ear_front_gain * front_gain +
        env->ear_side_gain * left_side_gain;
    float right_gain = env->ear_rear_gain + env->ear_front_gain * front_gain +
        env->ear_side_gain * right_side_gain;

    float source_path = dist(chirp->x, chirp->y, rx, ry);
    float left_path = source_path + dist(rx, ry, left_ear_x, left_ear_y);
    float right_path = source_path + dist(rx, ry, right_ear_x, right_ear_y);
    if (left_path > MAX_ECHO_RANGE && right_path > MAX_ECHO_RANGE) return;

    float rel_vx = rvx - env->vx;
    float rel_vy = rvy - env->vy;
    float distance_rate = rel_vx * ux + rel_vy * uy;
    float doppler = bat_clampf(-distance_rate / (env->max_speed + BUG_SPEED), -1.0f, 1.0f);
    float shifted_freq = bat_clampf(freq + 0.20f * doppler, 0.0f, 1.0f);

    if (left_path <= MAX_ECHO_RANGE) {
        float attenuation = strength / (1.0f + 0.02f * left_path * left_path);
        float receive_tick = chirp->birth_tick + slice_ticks + left_path / env->sound_speed / TICK_RATE;
        add_echo_event(env, 0, receive_tick, shifted_freq, attenuation * left_gain, left_path, source);
    }
    if (right_path <= MAX_ECHO_RANGE) {
        float attenuation = strength / (1.0f + 0.02f * right_path * right_path);
        float receive_tick = chirp->birth_tick + slice_ticks + right_path / env->sound_speed / TICK_RATE;
        add_echo_event(env, 1, receive_tick, shifted_freq, attenuation * right_gain, right_path, source);
    }
}

static inline void schedule_segment_reflectors(Bat* env, ChirpEvent* chirp,
        float slice_ticks, float freq, float x1, float y1, float x2, float y2,
        float strength) {
    float len = dist(x1, y1, x2, y2);
    int count = (int)(len / REFLECTOR_SPACING) + 1;
    if (count < 1) count = 1;
    for (int i = 0; i <= count; i++) {
        float t = count == 0 ? 0.0f : i / (float)count;
        float x = x1 + (x2 - x1) * t;
        float y = y1 + (y2 - y1) * t;
        schedule_echo(env, chirp, slice_ticks, freq, x, y, 0.0f, 0.0f, strength, ECHO_STATIC);
    }
}

static inline void schedule_corner_reflector_echoes(Bat* env, ChirpEvent* chirp,
        float slice_ticks, float freq) {
#if CORNER_REFLECTORS
    float w = (float)ARENA_WIDTH;
    float h = (float)ARENA_HEIGHT;
    float strength = env->reflector_strength;
    schedule_echo(env, chirp, slice_ticks, freq, 0.0f, 0.0f,
        0.0f, 0.0f, strength, ECHO_STATIC);
    schedule_echo(env, chirp, slice_ticks, freq, w, 0.0f,
        0.0f, 0.0f, strength, ECHO_STATIC);
    schedule_echo(env, chirp, slice_ticks, freq, 0.0f, h,
        0.0f, 0.0f, strength, ECHO_STATIC);
    schedule_echo(env, chirp, slice_ticks, freq, w, h,
        0.0f, 0.0f, strength, ECHO_STATIC);
    schedule_echo(env, chirp, slice_ticks, freq, 0.5f * w, 0.0f,
        0.0f, 0.0f, strength, ECHO_STATIC);
    schedule_echo(env, chirp, slice_ticks, freq, 0.5f * w, h,
        0.0f, 0.0f, strength, ECHO_STATIC);
    schedule_echo(env, chirp, slice_ticks, freq, 0.0f, 0.5f * h,
        0.0f, 0.0f, strength, ECHO_STATIC);
    schedule_echo(env, chirp, slice_ticks, freq, w, 0.5f * h,
        0.0f, 0.0f, strength, ECHO_STATIC);
#else
    (void)env;
    (void)chirp;
    (void)slice_ticks;
    (void)freq;
#endif
}

static inline void schedule_obstacle_echoes(Bat* env, ChirpEvent* chirp,
        float slice_ticks, float freq, int i) {
    float x = env->obstacle_x[i];
    float y = env->obstacle_y[i];
    float w = env->obstacle_w[i];
    float h = env->obstacle_h[i];
    schedule_segment_reflectors(env, chirp, slice_ticks, freq, x, y, x + w, y, 0.55f);
    schedule_segment_reflectors(env, chirp, slice_ticks, freq, x, y + h, x + w, y + h, 0.55f);
    schedule_segment_reflectors(env, chirp, slice_ticks, freq, x, y, x, y + h, 0.55f);
    schedule_segment_reflectors(env, chirp, slice_ticks, freq, x + w, y, x + w, y + h, 0.55f);
}

static inline void schedule_chirp_slice_echoes(Bat* env, ChirpEvent* chirp,
        int slice_idx) {
    int slices = chirp->slice_count;
    if (slice_idx >= slices || slice_idx >= MAX_CHIRP_SLICES) {
        return;
    }

    float t = (slice_idx + 0.5f) / (float)slices;
    float slice_ticks = chirp_slice_ticks(chirp, slice_idx);
    float freq = chirp->start_freq + t * (chirp->end_freq - chirp->start_freq);

    chirp->source_x[slice_idx] = env->x;
    chirp->source_y[slice_idx] = env->y;
    ChirpEvent slice_chirp = *chirp;
    slice_chirp.x = chirp->source_x[slice_idx];
    slice_chirp.y = chirp->source_y[slice_idx];

    schedule_echo(env, &slice_chirp, slice_ticks, freq,
        env->bug_x, env->bug_y, env->bug_vx, env->bug_vy, 8.0f, ECHO_BUG);
    schedule_segment_reflectors(env, &slice_chirp, slice_ticks, freq,
        0.0f, 0.0f, (float)ARENA_WIDTH, 0.0f, 0.12f);
    schedule_segment_reflectors(env, &slice_chirp, slice_ticks, freq,
        0.0f, (float)ARENA_HEIGHT, (float)ARENA_WIDTH, (float)ARENA_HEIGHT, 0.12f);
    schedule_segment_reflectors(env, &slice_chirp, slice_ticks, freq,
        0.0f, 0.0f, 0.0f, (float)ARENA_HEIGHT, 0.12f);
    schedule_segment_reflectors(env, &slice_chirp, slice_ticks, freq,
        (float)ARENA_WIDTH, 0.0f, (float)ARENA_WIDTH, (float)ARENA_HEIGHT, 0.12f);
    schedule_corner_reflector_echoes(env, &slice_chirp, slice_ticks, freq);
    for (int j = 0; j < env->num_obstacles; j++) {
        schedule_obstacle_echoes(env, &slice_chirp, slice_ticks, freq, j);
    }
}

static inline void schedule_due_chirp_slices(Bat* env) {
    for (int i = 0; i < CHIRP_HISTORY; i++) {
        ChirpEvent* chirp = &env->chirps[i];
        if (!chirp->active) continue;
        int slices = chirp->slice_count;

        float age_ticks = (float)(env->tick - chirp->birth_tick);
        while (chirp->slices_scheduled < slices) {
            int slice_idx = chirp->slices_scheduled;
            float slice_ticks = chirp_slice_ticks(chirp, slice_idx);
            if (slice_ticks >= age_ticks + 1.0f) break;
            schedule_chirp_slice_echoes(env, chirp, slice_idx);
            chirp->slices_scheduled += 1;
        }
    }
}

static inline void process_echo_events(Bat* env) {
    int slot = env->tick % ECHO_QUEUE_TICKS;
    EchoBucket* bucket = &env->echo_queue[slot];
    if (bucket->tick != env->tick) return;

    for (int i = 0; i < FREQ_BINS; i++) {
        int left_idx = LEFT_FREQ_OFFSET + i;
        int right_idx = RIGHT_FREQ_OFFSET + i;
        env->observations[left_idx] = bat_clampf(
            env->observations[left_idx] + bucket->energy[0][i], 0.0f, 1.0f);
        env->observations[right_idx] = bat_clampf(
            env->observations[right_idx] + bucket->energy[1][i], 0.0f, 1.0f);
    }
    if (bucket->bug_energy > 0.0f) {
        env->tick_bug_echo_energy += bucket->bug_energy;
        if (env->tick_bug_echo_path < 0.0f || bucket->bug_path < env->tick_bug_echo_path) {
            env->tick_bug_echo_path = bucket->bug_path;
        }
    }
    clear_echo_bucket(bucket);
}

void compute_observations(Bat* env) {
    memset(env->observations, 0, OBS_SIZE * sizeof(float));
    env->tick_bug_echo_energy = 0.0f;
    env->tick_bug_echo_path = -1.0f;

    process_echo_events(env);

    for (int i = 0; i < FREQ_BINS; i++) {
        env->observations[LEFT_FREQ_OFFSET + i] = bat_clampf(env->observations[LEFT_FREQ_OFFSET + i], 0.0f, 1.0f);
        env->observations[RIGHT_FREQ_OFFSET + i] = bat_clampf(env->observations[RIGHT_FREQ_OFFSET + i], 0.0f, 1.0f);
    }

    float chirp_age_denom = chirp_age_norm_denominator(env);
    int chirp_age = env->tick - env->last_chirp_tick;
    if (env->last_chirp_tick < 0) chirp_age = (int)ceilf(chirp_age_denom);
    env->chirp_age_ticks = chirp_age;
    int cooldown = env->chirp_cooldown_ticks - (env->tick - env->last_chirp_tick);
    env->observations[CHIRP_AGE_OBS] = bat_clampf(chirp_age / chirp_age_denom, 0.0f, 1.0f);
    env->observations[CHIRP_COOLDOWN_OBS] = bat_clampf(cooldown / (float)env->chirp_cooldown_ticks, 0.0f, 1.0f);
    env->observations[CHIRP_START_OBS] = env->last_chirp_start_freq;
    env->observations[CHIRP_END_OBS] = env->last_chirp_end_freq;
    env->observations[CHIRP_DURATION_OBS] = env->last_chirp_duration;
    env->observations[CHIRPS_USED_OBS] = chirps_used_ratio(env);
    float fwd_speed = env->vx * cosf(env->heading) + env->vy * sinf(env->heading);
    env->observations[FORWARD_SPEED_OBS] = bat_clampf(fwd_speed / env->max_speed, 0.0f, 1.0f);
    env->observations[TURN_RATE_OBS] = bat_clampf(env->turn_velocity / env->turn_rate, -1.0f, 1.0f);
    float timer_norm = env->tick / (float)MAX_STEPS;
    env->observations[40] = bat_clampf(timer_norm, 0.0f, 1.0f);
}

static inline void reset_episode(Bat* env) {
    env->tick = 0;
    env->turn_velocity = 0.0f;
    env->heading = randf(env) * TWO_PI - PI_F;
    float initial_speed = env->min_speed;
    env->vx = cosf(env->heading) * initial_speed;
    env->vy = sinf(env->heading) * initial_speed;
    if (env->curriculum_level < env->curriculum_initial_level) {
        env->curriculum_level = env->curriculum_initial_level;
    }
    apply_curriculum(env);
    sample_spawns_at_distance(env, curriculum_spawn_distance(env));
    generate_obstacles(env);
    reset_bug_motion(env);
    env->last_chirp_start_freq = 0.0f;
    env->last_chirp_end_freq = 1.0f;
    env->last_chirp_duration = 0.33333334f;
    env->chirp_age_ticks = 0;
    env->last_chirp_tick = -env->chirp_cooldown_ticks;
    memset(env->chirps, 0, sizeof(env->chirps));
    env->chirp_head = 0;
    clear_echo_queue(env);
    env->chirp_budget = MAX_CHIRPS_PER_EPISODE;
    env->tick_bug_echo_energy = 0.0f;
    env->tick_bug_echo_path = -1.0f;
    env->last_bug_echo_path = -1.0f;
    env->last_bug_echo_expected_tick = -1.0f;
    env->chirps_emitted = 0;
    env->chirps_overlapped = 0;
    env->episode_return = 0.0f;
    env->start_bug_dist = dist(env->x, env->y, env->bug_x, env->bug_y);
    env->prev_bug_dist = env->start_bug_dist;
    env->last_bug_echo_x = env->x;
    env->last_bug_echo_y = env->y;
    compute_observations(env);
}

void c_reset(Bat* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    reset_episode(env);
}

static inline bool hits_obstacle(Bat* env) {
    for (int i = 0; i < env->num_obstacles; i++) {
        if (circle_rect_collision(env->x, env->y, AGENT_RADIUS,
                env->obstacle_x[i], env->obstacle_y[i], env->obstacle_w[i], env->obstacle_h[i])) {
            return true;
        }
    }
    return false;
}

static inline bool hits_wall(Bat* env) {
    return env->x - AGENT_RADIUS < 0.0f ||
        env->x + AGENT_RADIUS > ARENA_WIDTH ||
        env->y - AGENT_RADIUS < 0.0f ||
        env->y + AGENT_RADIUS > ARENA_HEIGHT;
}

static inline void update_bug(Bat* env, float dt) {
    float speed = curriculum_bug_speed(env);
    float strength = curriculum_bug_maneuver_strength(env);
    if (env->bug_maneuver_mode > 0) {
        env->bug_maneuver_phase += env->bug_maneuver_rate * dt;
        if (env->bug_maneuver_phase > TWO_PI) {
            env->bug_maneuver_phase -= TWO_PI;
        }
    }

    if (env->bug_inbound) {
        float tx, ty;
        norm_vec(env->x - env->bug_x, env->y - env->bug_y, &tx, &ty);
        float px = -ty;
        float py = tx;
        float lateral = 0.0f;
        if (env->bug_maneuver_mode > 0) {
            lateral = strength * sinf(env->bug_maneuver_phase);
            if (env->bug_maneuver_mode == 2) {
                lateral += 0.5f * strength * env->bug_maneuver_sign;
            } else if (env->bug_maneuver_mode == 3) {
                lateral += 0.35f * strength * cosf(0.5f * env->bug_maneuver_phase);
            }
        }
        lateral = bat_clampf(lateral, -0.50f, 0.50f);
        float forward = sqrtf(fmaxf(0.0f, 1.0f - lateral * lateral));
        env->bug_vx = (tx * forward + px * lateral) * speed;
        env->bug_vy = (ty * forward + py * lateral) * speed;
    } else if (env->bug_maneuver_mode > 0) {
        float heading = env->bug_base_heading;
        if (env->bug_maneuver_mode == 1) {
            heading += strength * sinf(env->bug_maneuver_phase);
        } else if (env->bug_maneuver_mode == 2) {
            env->bug_base_heading += env->bug_maneuver_sign * strength * dt;
            heading = env->bug_base_heading;
        } else {
            heading += strength * sinf(env->bug_maneuver_phase)
                + 0.35f * strength * cosf(0.5f * env->bug_maneuver_phase);
        }
        env->bug_vx = cosf(heading) * speed;
        env->bug_vy = sinf(heading) * speed;
    }

    env->bug_x += env->bug_vx * dt;
    env->bug_y += env->bug_vy * dt;
    bool bounced = false;
    if (env->bug_x - BUG_RADIUS < 0.0f) {
        env->bug_x = BUG_RADIUS;
        env->bug_vx = fabsf(env->bug_vx);
        bounced = true;
    }
    if (env->bug_x + BUG_RADIUS > ARENA_WIDTH) {
        env->bug_x = ARENA_WIDTH - BUG_RADIUS;
        env->bug_vx = -fabsf(env->bug_vx);
        bounced = true;
    }
    if (env->bug_y - BUG_RADIUS < 0.0f) {
        env->bug_y = BUG_RADIUS;
        env->bug_vy = fabsf(env->bug_vy);
        bounced = true;
    }
    if (env->bug_y + BUG_RADIUS > ARENA_HEIGHT) {
        env->bug_y = ARENA_HEIGHT - BUG_RADIUS;
        env->bug_vy = -fabsf(env->bug_vy);
        bounced = true;
    }
    if (bounced) {
        env->bug_base_heading = atan2f(env->bug_vy, env->bug_vx);
        if (env->bug_inbound) {
            float tx, ty;
            norm_vec(env->x - env->bug_x, env->y - env->bug_y, &tx, &ty);
            env->bug_vx = tx * speed;
            env->bug_vy = ty * speed;
            env->bug_base_heading = atan2f(env->bug_vy, env->bug_vx);
        }
    }
}

static inline void update_motion(Bat* env, float dt) {
    int move = action_index(env->actions[0], MOVE_ACTIONS);
    int turn = action_index(env->actions[1], TURN_ACTIONS);
    float fx = cosf(env->heading);
    float fy = sinf(env->heading);
    float speed = env->vx * fx + env->vy * fy;
    float min_speed = env->min_speed;
    if (speed < min_speed) speed = min_speed;

    if (move == THRUST_FORWARD) speed += env->accel * dt;
    if (move == BRAKE) speed -= env->accel * dt;
    speed = bat_clampf(speed, min_speed, env->max_speed);

    float turn_command = 0.0f;
    if (turn == TURN_LEFT) turn_command = -1.0f;
    if (turn == TURN_RIGHT) turn_command = 1.0f;
    float speed_ratio = env->max_speed > 0.0f ? speed / env->max_speed : 0.0f;
    env->turn_velocity = turn_command * env->turn_rate * bat_clampf(speed_ratio, 0.0f, 1.0f);
    env->heading += env->turn_velocity * dt;
    if (env->heading > PI_F) env->heading -= TWO_PI;
    if (env->heading < -PI_F) env->heading += TWO_PI;

    float heading_fx = cosf(env->heading);
    float heading_fy = sinf(env->heading);
    env->vx = heading_fx * speed;
    env->vy = heading_fy * speed;
    env->x += env->vx * dt;
    env->y += env->vy * dt;
}

static inline bool try_emit_chirp(Bat* env) {
    if (env->tick - env->last_chirp_tick < env->chirp_cooldown_ticks) {
        return false;
    }

    int start_idx = action_index(env->actions[2], CHIRP_FREQ_BINS);
    int end_idx = action_index(env->actions[3], CHIRP_FREQ_BINS);
    int duration_idx = action_index(env->actions[4], CHIRP_DURATION_BINS);

    env->last_chirp_start_freq = norm_bin(start_idx, CHIRP_FREQ_BINS);
    env->last_chirp_end_freq = norm_bin(end_idx, CHIRP_FREQ_BINS);
    env->last_chirp_duration = norm_bin(duration_idx, CHIRP_DURATION_BINS);
    env->chirp_age_ticks = 0;
    env->last_chirp_tick = env->tick;
    env->chirps_emitted += 1;
    ChirpEvent* chirp = &env->chirps[env->chirp_head];
    chirp->x = env->x;
    chirp->y = env->y;
    chirp->start_freq = env->last_chirp_start_freq;
    chirp->end_freq = env->last_chirp_end_freq;
    chirp->duration = chirp_duration_seconds(env->last_chirp_duration);
    chirp->birth_tick = env->tick;
    chirp->slice_count = (int)ceilf(chirp->duration / TICK_RATE);
    chirp->slices_scheduled = 0;
    for (int i = 0; i < MAX_CHIRP_SLICES; i++) {
        chirp->source_x[i] = chirp->x;
        chirp->source_y[i] = chirp->y;
    }
    chirp->active = 1;
    env->chirp_head = (env->chirp_head + 1) % CHIRP_HISTORY;
    env->audio_chirp_serial += 1;
    env->last_bug_echo_expected_tick = expected_bug_echo_tick(env, chirp);
    return true;
}

static inline float next_chirp_overlap_fraction(Bat* env) {
    if (env->last_bug_echo_expected_tick <= (float)env->tick) return 0.0f;
    float wait_ticks = env->last_bug_echo_expected_tick - (float)env->last_chirp_tick;
    float remaining_ticks = env->last_bug_echo_expected_tick - (float)env->tick;
    return bat_clampf(remaining_ticks / wait_ticks, 0.0f, 1.0f);
}

static inline ChirpStatus update_chirp(Bat* env) {
    int emit = action_index(env->actions[5], CHIRP_EMIT_ACTIONS);
    if (emit) {
        if (env->chirps_emitted >= env->chirp_budget) {
            return CHIRP_STATUS_OVER_BUDGET;
        }
        return try_emit_chirp(env) ? CHIRP_STATUS_EMITTED : CHIRP_STATUS_COOLDOWN;
    } else if (env->chirp_age_ticks < MAX_CHIRP_AGE_TICKS) {
        env->chirp_age_ticks += 1;
    }
    return CHIRP_STATUS_NONE;
}

static inline bool caught_bug(Bat* env) {
    return dist(env->x, env->y, env->bug_x, env->bug_y) <= AGENT_RADIUS + BUG_RADIUS;
}

void c_step(Bat* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    float success = 0.0f;
    float collision = 0.0f;
    float timeout = 0.0f;

    float chirp_overlap_fraction = next_chirp_overlap_fraction(env);
    ChirpStatus chirp_status = update_chirp(env);
    if (chirp_status == CHIRP_STATUS_OVER_BUDGET) {
        env->tick += 1;
        env->rewards[0] = -1.0f;
        collision = 1.0f;
    } else {
        schedule_due_chirp_slices(env);

        update_motion(env, TICK_RATE);
        update_bug(env, TICK_RATE);
        env->tick += 1;
        if (hits_wall(env) || hits_obstacle(env)) {
            env->rewards[0] = -env->collision_penalty;
            collision = 1.0f;
        } else if (caught_bug(env)) {
            env->rewards[0] = success_reward(env);
            success = 1.0f;
        } else {
            float bug_dist = dist(env->x, env->y, env->bug_x, env->bug_y);
            float progress = env->prev_bug_dist - bug_dist;
            env->rewards[0] += env->progress_reward_scale * progress;
            env->rewards[0] -= env->step_cost; // TODO: Fold this only when we are ready to break training determinism.
            if (chirp_status == CHIRP_STATUS_EMITTED) {
                env->rewards[0] += env->valid_chirp_reward; // TODO: Remove this; chirps should only pay when bug echoes improve.
                env->rewards[0] -= CHIRP_COST;
                if (chirp_overlap_fraction > 0.0f) {
                    env->rewards[0] -= env->chirp_overlap_penalty * chirp_overlap_fraction;
                    env->chirps_overlapped += 1;
                }
            } else if (chirp_status == CHIRP_STATUS_COOLDOWN) {
                env->rewards[0] -= env->early_chirp_penalty;
            }
            env->prev_bug_dist = bug_dist;

            if (env->tick >= MAX_STEPS) {
                env->rewards[0] = -1.0f;
                timeout = 1.0f;
            }
        }
    }

    if (success || collision || timeout) {
        env->terminals[0] = 1.0f;
        env->episode_return += env->rewards[0];
        if (success) {
            advance_curriculum(env);
        }
        add_log(env, success, collision, timeout);
        reset_episode(env);
        return;
    }

    compute_observations(env);
    if (env->tick_bug_echo_path > 0.0f) {
        if (env->last_bug_echo_path > 0.0f) {
            float echo_displacement = dist(env->last_bug_echo_x, env->last_bug_echo_y,
                env->x, env->y);
            if (echo_displacement >= BUG_ECHO_MIN_DISPLACEMENT) {
                float echo_progress = (env->last_bug_echo_path - env->tick_bug_echo_path)
                    / MAX_ECHO_RANGE;
                if (echo_progress > 0.0f) {
                    env->rewards[0] += env->bug_echo_reward_scale * echo_progress;
                } else if (echo_progress < 0.0f) {
                    env->rewards[0] += env->bug_echo_reward_scale
                        * env->bug_echo_farther_penalty_scale * echo_progress;
                }
            }
        }
        env->last_bug_echo_path = env->tick_bug_echo_path;
        env->last_bug_echo_x = env->x;
        env->last_bug_echo_y = env->y;
    }
    env->episode_return += env->rewards[0];
}

#ifndef BAT_HEADLESS
static inline Color freq_color(float freq_norm, float alpha_norm) {
    float f = freq_norm;
    float mid = 1.0f - fabsf(2.0f * f - 1.0f);
    return (Color){
        (unsigned char)(255.0f * (1.0f - f) + 45.0f * f),
        (unsigned char)(45.0f + 180.0f * mid),
        (unsigned char)(45.0f * (1.0f - f) + 255.0f * f),
        (unsigned char)(255.0f * alpha_norm),
    };
}

static inline void draw_chirp_rings(Bat* env, float sx, float sy) {
    float scale = fminf(sx, sy);
    for (int i = 0; i < CHIRP_HISTORY; i++) {
        ChirpEvent* chirp = &env->chirps[i];
        if (!chirp->active) continue;

        float age_seconds = (env->tick - chirp->birth_tick) * TICK_RATE;
        float max_age = MAX_ECHO_RANGE / env->sound_speed + chirp->duration;
        if (age_seconds < 0.0f || age_seconds > max_age) {
            chirp->active = 0;
            continue;
        }

        for (int ring = 0; ring < CHIRP_RINGS; ring++) {
            float slice = ring / (float)(CHIRP_RINGS - 1);
            float freq = chirp->start_freq + slice * (chirp->end_freq - chirp->start_freq);
            float radius = chirp_ring_radius(age_seconds, slice, chirp->duration, env->sound_speed);
            if (radius <= 0.0f || radius > MAX_ECHO_RANGE) continue;

            float fade = 1.0f - radius / MAX_ECHO_RANGE;
            float alpha = 0.18f + 0.42f * bat_clampf(fade, 0.0f, 1.0f);
            float source_x, source_y;
            chirp_source_for_fraction(chirp, slice, &source_x, &source_y);
            DrawCircleLines(
                (int)(source_x * sx),
                (int)(source_y * sy),
                radius * scale,
                freq_color(freq, alpha));
        }
    }
}

static inline Color doppler_ray_color(float doppler, float alpha) {
    if (doppler > 0.05f) {
        return freq_color(1.0f, alpha);
    } else if (doppler < -0.05f) {
        return freq_color(0.0f, alpha);
    }
    return (Color){210, 210, 220,
        (unsigned char)(255.0f * bat_clampf(alpha, 0.0f, 1.0f))};
}

static inline void draw_echo_flash(Bat* env, ChirpEvent* chirp,
        float rx, float ry, float rvx, float rvy, float strength,
        float sx, float sy) {
    float age_seconds = (env->tick - chirp->birth_tick) * TICK_RATE;
    float distance = dist(chirp->x, chirp->y, rx, ry);
    float echo_time = echo_time_seconds(distance, env->sound_speed);
    bool echo_arriving_now = fabsf(age_seconds - echo_time) <= 0.025f;
    if (!echo_arriving_now) return;

    float ux, uy;
    norm_vec(rx - chirp->x, ry - chirp->y, &ux, &uy);
    float rel_vx = rvx - env->vx;
    float rel_vy = rvy - env->vy;
    float distance_rate = rel_vx * ux + rel_vy * uy;
    float doppler = bat_clampf(-distance_rate / (env->max_speed + BUG_SPEED), -1.0f, 1.0f);
    float amp = strength / (1.0f + 0.02f * distance * distance);
    float alpha = bat_clampf(0.20f + amp * 2.0f, 0.20f, 0.90f);
    Color color = doppler_ray_color(doppler, alpha);

    DrawLine((int)(chirp->x * sx), (int)(chirp->y * sy),
        (int)(rx * sx), (int)(ry * sy), color);
    DrawCircleLines((int)(rx * sx), (int)(ry * sy),
        fmaxf(3.0f, 8.0f * alpha), color);
}

static inline void draw_segment_echoes(Bat* env, ChirpEvent* chirp,
        float x1, float y1, float x2, float y2, float strength,
        float sx, float sy) {
    float len = dist(x1, y1, x2, y2);
    int count = (int)(len / REFLECTOR_SPACING) + 1;
    if (count < 1) count = 1;
    for (int i = 0; i <= count; i++) {
        float t = i / (float)count;
        float x = x1 + (x2 - x1) * t;
        float y = y1 + (y2 - y1) * t;
        draw_echo_flash(env, chirp, x, y, 0.0f, 0.0f, strength, sx, sy);
    }
}

static inline void draw_obstacle_echoes(Bat* env, ChirpEvent* chirp,
        int i, float sx, float sy) {
    float x = env->obstacle_x[i];
    float y = env->obstacle_y[i];
    float w = env->obstacle_w[i];
    float h = env->obstacle_h[i];
    draw_segment_echoes(env, chirp, x, y, x + w, y, 0.55f, sx, sy);
    draw_segment_echoes(env, chirp, x, y + h, x + w, y + h, 0.55f, sx, sy);
    draw_segment_echoes(env, chirp, x, y, x, y + h, 0.55f, sx, sy);
    draw_segment_echoes(env, chirp, x + w, y, x + w, y + h, 0.55f, sx, sy);
}

static inline void draw_corner_reflector_echoes(Bat* env, ChirpEvent* chirp,
        float sx, float sy) {
#if CORNER_REFLECTORS
    float w = (float)ARENA_WIDTH;
    float h = (float)ARENA_HEIGHT;
    float strength = env->reflector_strength;
    draw_echo_flash(env, chirp, 0.0f, 0.0f, 0.0f, 0.0f, strength, sx, sy);
    draw_echo_flash(env, chirp, w, 0.0f, 0.0f, 0.0f, strength, sx, sy);
    draw_echo_flash(env, chirp, 0.0f, h, 0.0f, 0.0f, strength, sx, sy);
    draw_echo_flash(env, chirp, w, h, 0.0f, 0.0f, strength, sx, sy);
    draw_echo_flash(env, chirp, 0.5f * w, 0.0f, 0.0f, 0.0f, strength, sx, sy);
    draw_echo_flash(env, chirp, 0.5f * w, h, 0.0f, 0.0f, strength, sx, sy);
    draw_echo_flash(env, chirp, 0.0f, 0.5f * h, 0.0f, 0.0f, strength, sx, sy);
    draw_echo_flash(env, chirp, w, 0.5f * h, 0.0f, 0.0f, strength, sx, sy);
#else
    (void)env;
    (void)chirp;
    (void)sx;
    (void)sy;
#endif
}

static inline void draw_corner_reflector_markers(Bat* env) {
#if CORNER_REFLECTORS
    const int size = 8;
    const Color fill = (Color){128, 128, 132, 255};
    const Color outline = (Color){202, 202, 208, 255};
    int max_x = env->client->width - size;
    int max_y = env->client->height - size;
    int mid_x = env->client->width / 2 - size / 2;
    int mid_y = env->client->height / 2 - size / 2;
    DrawRectangle(0, 0, size, size, fill);
    DrawRectangleLines(0, 0, size, size, outline);
    DrawRectangle(max_x, 0, size, size, fill);
    DrawRectangleLines(max_x, 0, size, size, outline);
    DrawRectangle(0, max_y, size, size, fill);
    DrawRectangleLines(0, max_y, size, size, outline);
    DrawRectangle(max_x, max_y, size, size, fill);
    DrawRectangleLines(max_x, max_y, size, size, outline);
    DrawRectangle(mid_x, 0, size, size, fill);
    DrawRectangleLines(mid_x, 0, size, size, outline);
    DrawRectangle(mid_x, max_y, size, size, fill);
    DrawRectangleLines(mid_x, max_y, size, size, outline);
    DrawRectangle(0, mid_y, size, size, fill);
    DrawRectangleLines(0, mid_y, size, size, outline);
    DrawRectangle(max_x, mid_y, size, size, fill);
    DrawRectangleLines(max_x, mid_y, size, size, outline);
#else
    (void)env;
#endif
}

static inline void draw_echo_reflections(Bat* env, float sx, float sy) {
    for (int i = 0; i < CHIRP_HISTORY; i++) {
        ChirpEvent* chirp = &env->chirps[i];
        if (!chirp->active) continue;
        draw_echo_flash(env, chirp, env->bug_x, env->bug_y,
            env->bug_vx, env->bug_vy, 4.0f, sx, sy);
        draw_segment_echoes(env, chirp, 0.0f, 0.0f, (float)ARENA_WIDTH, 0.0f, 0.18f, sx, sy);
        draw_segment_echoes(env, chirp, 0.0f, (float)ARENA_HEIGHT, (float)ARENA_WIDTH, (float)ARENA_HEIGHT, 0.18f, sx, sy);
        draw_segment_echoes(env, chirp, 0.0f, 0.0f, 0.0f, (float)ARENA_HEIGHT, 0.18f, sx, sy);
        draw_segment_echoes(env, chirp, (float)ARENA_WIDTH, 0.0f, (float)ARENA_WIDTH, (float)ARENA_HEIGHT, 0.18f, sx, sy);
        draw_corner_reflector_echoes(env, chirp, sx, sy);
        for (int j = 0; j < env->num_obstacles; j++) {
            draw_obstacle_echoes(env, chirp, j, sx, sy);
        }
    }
}

#include "bat_record.h"

Client* make_client(Bat* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = ARENA_WIDTH * 10;
    client->height = ARENA_HEIGHT * 10;
    InitWindow(client->width, client->height, "Bat");
    int target_fps = env->render_target_fps;
    if (target_fps > 0) {
        SetTargetFPS(target_fps);
    }
    InitAudioDevice();
    client->audio_ready = IsAudioDeviceReady();
    record_init(env, client);
    return client;
}

void close_client(Client* client) {
    record_finalize(client);
    if (client->audio_ready) {
        for (int i = 0; i < AUDIO_VOICES; i++) {
            unload_chirp_sound(client, i);
        }
        CloseAudioDevice();
    }
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
    play_chirp_audio(env);
    float sx = env->client->width / (float)ARENA_WIDTH;
    float sy = env->client->height / (float)ARENA_HEIGHT;
    BeginDrawing();
    ClearBackground((Color){18, 20, 24, 255});
    draw_chirp_rings(env, sx, sy);
    draw_echo_reflections(env, sx, sy);
    DrawRectangleLines(0, 0, env->client->width, env->client->height, GRAY);
    for (int i = 0; i < env->num_obstacles; i++) {
        DrawRectangle(
            (int)(env->obstacle_x[i] * sx),
            (int)(env->obstacle_y[i] * sy),
            (int)(env->obstacle_w[i] * sx),
            (int)(env->obstacle_h[i] * sy),
            (Color){92, 92, 96, 255});
    }
    draw_corner_reflector_markers(env);
    DrawCircle((int)(env->bug_x * sx), (int)(env->bug_y * sy),
        BUG_RADIUS * sx, GREEN);
    DrawCircle((int)(env->x * sx), (int)(env->y * sy),
        AGENT_RADIUS * sx, BLUE);
    float hx = env->x + cosf(env->heading) * AGENT_RADIUS * 2.0f;
    float hy = env->y + sinf(env->heading) * AGENT_RADIUS * 2.0f;
    DrawLine((int)(env->x * sx), (int)(env->y * sy), (int)(hx * sx), (int)(hy * sy), WHITE);
    int cooldown = env->chirp_cooldown_ticks - (env->tick - env->last_chirp_tick);
    DrawText(TextFormat("reward %.3f tick %d chirps %d cooldown %d ESC exits", env->rewards[0], env->tick,
        env->chirps_emitted, cooldown), 10, 10, 20, RAYWHITE);
    EndDrawing();
    record_capture_frame(env);
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
