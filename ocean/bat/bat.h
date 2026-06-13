#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

#ifndef BAT_HEADLESS
#include "raylib.h"
#endif

#define NUM_AGENTS 1
#define NUM_ACTIONS 6
#define ACTION_MOVE 0
#define ACTION_TURN 1
#define ACTION_CHIRP_FREQ_START 2
#define ACTION_CHIRP_FREQ_END 3
#define ACTION_CHIRP_DURATION 4
#define ACTION_CHIRP_EMIT 5
#define MOVE_ACTIONS 3
#define TURN_ACTIONS 3
#define CHIRP_FREQ_BINS 8
#define CHIRP_DURATION_BINS 4
#define CHIRP_EMIT_ACTIONS 2

#define FREQ_BINS 16
#define LEFT_FREQ_OFFSET 0
#define RIGHT_FREQ_OFFSET FREQ_BINS
#define CHIRP_AGE_OBS (RIGHT_FREQ_OFFSET + FREQ_BINS)
#define CHIRP_COOLDOWN_OBS (CHIRP_AGE_OBS + 1)
#define CHIRP_START_OBS (CHIRP_COOLDOWN_OBS + 1)
#define CHIRP_END_OBS (CHIRP_START_OBS + 1)
#define CHIRP_DURATION_OBS (CHIRP_END_OBS + 1)
#define CHIRPS_USED_OBS (CHIRP_DURATION_OBS + 1)
#define FORWARD_SPEED_OBS (CHIRPS_USED_OBS + 1)
#define TURN_RATE_OBS (FORWARD_SPEED_OBS + 1)
#define TIMER_OBS (TURN_RATE_OBS + 1)
#define OBS_SIZE (TIMER_OBS + 1)

#define NOOP 0
#define THRUST_FORWARD 1
#define BRAKE 2

#define TURN_NONE 0
#define TURN_LEFT 1
#define TURN_RIGHT 2

#define MAX_STEPS 512
#define TICK_RATE (1.0f/60.0f)
#define ARENA_WIDTH 64
#define ARENA_HEIGHT 64
#define AGENT_RADIUS 2.0f
#define BUG_RADIUS 1.5f
#define SPAWN_MARGIN 6.0f
#define BUG_SPEED 4.0f
#define BUG_MANEUVER_START_LEVEL 7
#define BUG_MANEUVER_STRENGTH 0.4f
#define BUG_MANEUVER_FREQUENCY 0.4f
#define INBOUND_BUG_SPEED_MULTIPLIER 1.75f
#define INBOUND_HEADING_NOISE_DEGREES 18.0f
#define REFLECTOR_SPACING 8.0f
#define MAX_ECHO_RANGE 128.0f
#define ECHO_MIN_FORWARD -0.35f
#define BUG_ECHO_MIN_DISPLACEMENT 1.0f
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
#define AUDIO_VOICES 8
#define AUDIO_SAMPLE_RATE 48000
#define AUDIO_MIN_HZ 600.0f
#define AUDIO_MAX_HZ 3600.0f
#define AUDIO_VOLUME 0.22f
#define AUDIO_ENVELOPE_FADE 0.08f
#define RECORD_MAX_VOICES 16
#define FREQ_HISTORY_TICKS 96
#define FREQ_PANEL_WIDTH 384
#define FREQ_WATERFALL_WIDTH 192
#define FREQ_PANEL_MARGIN 8
#define CHIRP_PERF_FLOOR 0.05f
#define CHIRP_MIN_DURATION_SECONDS 0.04f
#define CHIRP_DURATION_RANGE_SECONDS 0.18f
#define MAX_CHIRPS_PER_EPISODE 15

#define ECHO_STATIC 0
#define ECHO_BUG 1
#define ARENA_REFLECTORS 8

static const float ARENA_REFLECTOR_X[ARENA_REFLECTORS] = {0.0f, 1.0f, 0.0f, 1.0f, 0.5f, 0.5f, 0.0f, 1.0f};
static const float ARENA_REFLECTOR_Y[ARENA_REFLECTORS] = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.5f, 0.5f};

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
    float closest_bug_echo_path;
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
    float num_obstacles;
    float chirps_emitted;
    float chirp_perf;
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
    float freq_history[FREQ_HISTORY_TICKS][2][FREQ_BINS];
    int freq_history_head;
    int freq_history_last_tick;
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

    float obstacle_x[CURRICULUM_MAX_OBSTACLES];
    float obstacle_y[CURRICULUM_MAX_OBSTACLES];
    float obstacle_w[CURRICULUM_MAX_OBSTACLES];
    float obstacle_h[CURRICULUM_MAX_OBSTACLES];

    float sound_speed;
    float reflector_strength;
    int chirp_cooldown_ticks;
    int last_chirp_tick;
    float last_chirp_start_freq;
    float last_chirp_end_freq;
    float last_chirp_duration;
    ChirpEvent chirps[CHIRP_HISTORY];
    int chirp_head;
    EchoBucket echo_queue[ECHO_QUEUE_TICKS];
    int chirps_emitted;
    int audio_chirp_serial;

    float chirp_efficiency_reward;
    float valid_chirp_reward;
    float early_chirp_penalty;
    float chirp_overlap_penalty;
    float step_cost;
    float progress_reward_scale;
    float bug_echo_reward_scale;
    float bug_echo_farther_penalty_scale;
    float bug_wing_sideband_gain;
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

static inline float chirp_duration_seconds(float duration_norm) {
    return CHIRP_MIN_DURATION_SECONDS + CHIRP_DURATION_RANGE_SECONDS * duration_norm;
}

#include "bat_audio.h"

static inline float chirp_slice_ticks(ChirpEvent* chirp, int slice_idx) {
    return ((slice_idx + 0.5f) / (float)chirp->slice_count) *
        chirp->duration / TICK_RATE;
}

static inline void chirp_source_for_slice(ChirpEvent* chirp, int slice_idx,
        float* source_x, float* source_y) {
    if (slice_idx < chirp->slices_scheduled) {
        *source_x = chirp->source_x[slice_idx];
        *source_y = chirp->source_y[slice_idx];
        return;
    }
    *source_x = chirp->x;
    *source_y = chirp->y;
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

static inline void sample_in_quadrant(Bat* env, int quadrant, float* x, float* y) {
    int east = quadrant & 1;
    int south = (quadrant >> 1) & 1;
    float half_w = ARENA_WIDTH * 0.5f;
    float half_h = ARENA_HEIGHT * 0.5f;
    float min_x = (east ? half_w : 0.0f) + SPAWN_MARGIN;
    float max_x = (east ? (float)ARENA_WIDTH : half_w) - SPAWN_MARGIN;
    float min_y = (south ? half_h : 0.0f) + SPAWN_MARGIN;
    float max_y = (south ? (float)ARENA_HEIGHT : half_h) - SPAWN_MARGIN;
    *x = min_x + randf(env) * (max_x - min_x);
    *y = min_y + randf(env) * (max_y - min_y);
}

static inline int curriculum_obstacles(Bat* env) {
    int count = env->curriculum_level > 0
        ? 1 + (env->curriculum_level - 1) / env->curriculum_obstacle_step : 0;
    return count > CURRICULUM_MAX_OBSTACLES ? CURRICULUM_MAX_OBSTACLES : count;
}

static inline float curriculum_bug_distance(Bat* env) {
    return bat_clampf(env->curriculum_start_bug_distance
        + CURRICULUM_BUG_DISTANCE_STEP * env->curriculum_level,
        env->curriculum_start_bug_distance,
        CURRICULUM_MAX_BUG_DISTANCE);
}

static inline float curriculum_inbound_bug_distance(Bat* env) {
    return bat_clampf(CURRICULUM_MAX_BUG_DISTANCE
        + CURRICULUM_INBOUND_BUG_DISTANCE_STEP
            * (env->curriculum_level - CURRICULUM_INBOUND_START_LEVEL + 1),
        CURRICULUM_MAX_BUG_DISTANCE, CURRICULUM_INBOUND_MAX_BUG_DISTANCE);
}

static inline float curriculum_bug_maneuver_strength(Bat* env) {
    if (env->curriculum_level < BUG_MANEUVER_START_LEVEL) return 0.0f;
    int extra_levels = env->curriculum_level - BUG_MANEUVER_START_LEVEL;
    float ramp = extra_levels <= 0 ? 0.25f : 0.75f + 0.25f * (extra_levels - 1);
    return BUG_MANEUVER_STRENGTH * bat_clampf(ramp, 0.0f, 1.0f);
}

// TODO: When we are ready to break determinism, simplify bug maneuvering to one
// always-active sine wave with curriculum-ramped amplitude, then remove the mode
// and sign branches below.
static inline float curriculum_bug_maneuver_frequency(Bat* env) {
    if (env->curriculum_level < BUG_MANEUVER_START_LEVEL) {
        return BUG_MANEUVER_FREQUENCY;
    }
    return BUG_MANEUVER_FREQUENCY * bat_clampf(
        1.0f + 0.50f * (env->curriculum_level - BUG_MANEUVER_START_LEVEL),
        1.0f, 2.5f);
}

static inline float chirps_used_ratio(Bat* env) {
    return bat_clampf(env->chirps_emitted / (float)MAX_CHIRPS_PER_EPISODE, 0.0f, 1.0f);
}

// TODO: Revisit this when we are ready to break reward determinism. The ratio is
// still an observation, but this reward bonus may be removable before merge.
static inline float chirp_efficiency(Bat* env) {
    return 0.5f + 0.5f * (1.0f - chirps_used_ratio(env));
}

static inline float chirp_perf(Bat* env) {
    return bat_clampf(1.0f - env->chirps_emitted / (float)MAX_CHIRPS_PER_EPISODE,
        CHIRP_PERF_FLOOR, 1.0f);
}

// TODO: Revisit whether these curriculum difficulty diagnostics are worth logging;
// they add a lot of code and may be removable before merge.
static inline float curriculum_distance_difficulty(Bat* env) {
    return bat_clampf((env->start_bug_dist - env->curriculum_start_bug_distance)
        / (CURRICULUM_INBOUND_MAX_BUG_DISTANCE - env->curriculum_start_bug_distance),
        0.0f, 1.0f);
}

static inline float curriculum_obstacle_difficulty(Bat* env) {
    return bat_clampf(env->num_obstacles / (float)CURRICULUM_MAX_OBSTACLES, 0.0f, 1.0f);
}

static inline float curriculum_motion_difficulty(Bat* env) {
    if (env->curriculum_level < BUG_MANEUVER_START_LEVEL) return 0.0f;
    return bat_clampf((env->curriculum_level - BUG_MANEUVER_START_LEVEL + 1)
        / (float)(CURRICULUM_INBOUND_START_LEVEL + 4 - BUG_MANEUVER_START_LEVEL),
        0.0f, 1.0f);
}

static inline float curriculum_difficulty(Bat* env) {
    return bat_clampf((curriculum_distance_difficulty(env)
        + curriculum_obstacle_difficulty(env)
        + curriculum_motion_difficulty(env)) / 3.0f, 0.0f, 1.0f);
}

static inline void sample_spawns_at_distance(Bat* env, float target_distance) {
    for (int attempt = 0; attempt < 96; attempt++) {
        float angle = randf(env) * TWO_PI - PI_F;
        float dx = cosf(angle) * target_distance;
        float dy = sinf(angle) * target_distance;
        float min_bat_x = fmaxf(SPAWN_MARGIN, SPAWN_MARGIN - dx);
        float max_bat_x = fminf(ARENA_WIDTH - SPAWN_MARGIN, ARENA_WIDTH - SPAWN_MARGIN - dx);
        float min_bat_y = fmaxf(SPAWN_MARGIN, SPAWN_MARGIN - dy);
        float max_bat_y = fminf(ARENA_HEIGHT - SPAWN_MARGIN, ARENA_HEIGHT - SPAWN_MARGIN - dy);
        if (max_bat_x < min_bat_x || max_bat_y < min_bat_y) continue;

        env->x = min_bat_x + randf(env) * (max_bat_x - min_bat_x);
        env->y = min_bat_y + randf(env) * (max_bat_y - min_bat_y);
        env->bug_x = env->x + dx;
        env->bug_y = env->y + dy;
        return;
    }

    int agent_quadrant = (int)(randf(env) * 4.0f);
    int bug_quadrant = agent_quadrant ^ 3;
    float min_sep = fminf(ARENA_WIDTH, ARENA_HEIGHT) * 0.31f;
    for (int attempt = 0; attempt < 64; attempt++) {
        sample_in_quadrant(env, agent_quadrant, &env->x, &env->y);
        sample_in_quadrant(env, bug_quadrant, &env->bug_x, &env->bug_y);
        if (dist(env->x, env->y, env->bug_x, env->bug_y) >= min_sep) {
            return;
        }
    }

    env->x = ARENA_WIDTH * ((agent_quadrant & 1) ? 0.75f : 0.25f);
    env->y = ARENA_HEIGHT * ((agent_quadrant & 2) ? 0.75f : 0.25f);
    env->bug_x = ARENA_WIDTH * ((bug_quadrant & 1) ? 0.75f : 0.25f);
    env->bug_y = ARENA_HEIGHT * ((bug_quadrant & 2) ? 0.75f : 0.25f);
}

static inline void reset_bug_motion(Bat* env) {
    float strength = curriculum_bug_maneuver_strength(env);
    env->bug_maneuver_mode = strength > 0.000001f ? 1 + (int)(rng_next(env) % 3u) : 0;
    env->bug_maneuver_phase = randf(env) * TWO_PI;
    env->bug_maneuver_rate = TWO_PI * curriculum_bug_maneuver_frequency(env) *
        (0.75f + 0.50f * randf(env));
    env->bug_maneuver_sign = (rng_next(env) & 1u) ? -1.0f : 1.0f;

    float speed = env->bug_inbound ? BUG_SPEED * INBOUND_BUG_SPEED_MULTIPLIER : BUG_SPEED;
    float heading;
    if (env->bug_inbound) {
        float tx, ty;
        norm_vec(env->x - env->bug_x, env->y - env->bug_y, &tx, &ty);
        float noise = INBOUND_HEADING_NOISE_DEGREES * (PI_F / 180.0f);
        heading = atan2f(ty, tx) + (2.0f * randf(env) - 1.0f) * noise;
    } else {
        heading = randf(env) * TWO_PI - PI_F;
    }
    env->bug_base_heading = heading;
    env->bug_vx = cosf(heading) * speed;
    env->bug_vy = sinf(heading) * speed;
}

static inline void advance_curriculum(Bat* env) {
    env->curriculum_successes_at_level += 1;
    if (env->curriculum_successes_at_level >= env->curriculum_successes_per_level) {
        env->curriculum_level += 1;
        env->curriculum_successes_at_level = 0;
    }
}

// TODO: Revisit this when we are ready to break reset determinism. If overlapping
// random obstacles are acceptable, remove rects_overlap(), obstacle_clear(), and
// the attempt loop/fallback placement in generate_obstacles().
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
    env->curriculum_level = env->curriculum_initial_level;
    env->curriculum_successes_at_level = 0;
}

void allocate(Bat* env) {
    init(env);
    env->observations = (float*)calloc(OBS_SIZE, sizeof(float));
    env->actions = (float*)calloc(NUM_ACTIONS, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (float*)calloc(1, sizeof(float));
}

void c_close(Bat* env) {
    (void)env;
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
    env->log.num_obstacles += env->num_obstacles;
    env->log.chirps_emitted += env->chirps_emitted;
    env->log.chirp_perf += chirp_perf_value;
    env->log.n += 1.0f;
}

static inline void clear_echo_bucket(EchoBucket* bucket) {
    memset(bucket, 0, sizeof(*bucket));
    bucket->closest_bug_echo_path = -1.0f;
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
    if (arrival_tick - env->tick >= ECHO_QUEUE_TICKS) return;
    int slot = arrival_tick % ECHO_QUEUE_TICKS;
    EchoBucket* bucket = &env->echo_queue[slot];
    if (bucket->tick != arrival_tick) {
        clear_echo_bucket(bucket);
        bucket->tick = arrival_tick;
    }

    int bin = (int)(freq * FREQ_BINS);
    if (bin >= FREQ_BINS) bin = FREQ_BINS - 1;
    bucket->energy[ear][bin] += intensity;
    if (source == ECHO_BUG) {
        float sideband = intensity * env->bug_wing_sideband_gain;
        if (sideband > 0.000001f) {
            if (bin > 0) bucket->energy[ear][bin - 1] += sideband;
            if (bin + 1 < FREQ_BINS) bucket->energy[ear][bin + 1] += sideband;
        }
        if (bucket->closest_bug_echo_path < 0.0f || path < bucket->closest_bug_echo_path) {
            bucket->closest_bug_echo_path = path;
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

static inline void schedule_ear_echo(Bat* env, int birth_tick, int ear,
        float slice_ticks, float freq, float strength, float path,
        float gain, int source) {
    if (path > MAX_ECHO_RANGE) return;
    float attenuation = strength / (1.0f + 0.02f * path * path);
    float receive_tick = birth_tick + slice_ticks + path / env->sound_speed / TICK_RATE;
    add_echo_event(env, ear, receive_tick, freq, attenuation * gain, path, source);
}

static inline float expected_bug_echo_tick(Bat* env, ChirpEvent* chirp) {
    float fx = cosf(env->heading);
    float fy = sinf(env->heading);
    float ux, uy;
    norm_vec(env->bug_x - chirp->x, env->bug_y - chirp->y, &ux, &uy);
    float forward = ux * fx + uy * fy;
    if (forward < ECHO_MIN_FORWARD) return -1.0f;

    float left_ear_x, left_ear_y, right_ear_x, right_ear_y;
    ear_positions(env, &left_ear_x, &left_ear_y, &right_ear_x, &right_ear_y);
    float source_path = dist(chirp->x, chirp->y, env->bug_x, env->bug_y);
    float left_path = source_path + dist(env->bug_x, env->bug_y, left_ear_x, left_ear_y);
    float right_path = source_path + dist(env->bug_x, env->bug_y, right_ear_x, right_ear_y);
    float best_path = fminf(left_path, right_path);
    if (best_path > MAX_ECHO_RANGE) return -1.0f;

    return chirp->birth_tick + chirp_slice_ticks(chirp, 0)
        + best_path / env->sound_speed / TICK_RATE;
}

static inline void schedule_echo(Bat* env, ChirpEvent* chirp,
        float slice_ticks, float freq, float rx, float ry, float rvx, float rvy,
        float strength, int source) {
    float fx = cosf(env->heading);
    float fy = sinf(env->heading);
    float lateral_x = -fy;
    float lateral_y = fx;
    float left_ear_x, left_ear_y, right_ear_x, right_ear_y;
    ear_positions(env, &left_ear_x, &left_ear_y, &right_ear_x, &right_ear_y);

    float ux, uy;
    norm_vec(rx - chirp->x, ry - chirp->y, &ux, &uy);
    float forward = ux * fx + uy * fy;
    if (forward < ECHO_MIN_FORWARD) return;

    float front_gain = bat_clampf(forward, 0.0f, 1.0f);
    float left_side_gain = bat_clampf(ux * -lateral_x + uy * -lateral_y, 0.0f, 1.0f);
    float right_side_gain = bat_clampf(ux * lateral_x + uy * lateral_y, 0.0f, 1.0f);
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

    schedule_ear_echo(env, chirp->birth_tick, 0,
        slice_ticks, shifted_freq, strength, left_path, left_gain, source);
    schedule_ear_echo(env, chirp->birth_tick, 1,
        slice_ticks, shifted_freq, strength, right_path, right_gain, source);
}

static inline void schedule_segment_reflectors(Bat* env, ChirpEvent* chirp,
        float slice_ticks, float freq, float x1, float y1, float x2, float y2,
        float strength) {
    float len = dist(x1, y1, x2, y2);
    int count = (int)(len / REFLECTOR_SPACING) + 1;
    for (int i = 0; i <= count; i++) {
        float t = i / (float)count;
        float x = x1 + (x2 - x1) * t;
        float y = y1 + (y2 - y1) * t;
        schedule_echo(env, chirp, slice_ticks, freq, x, y, 0.0f, 0.0f, strength, ECHO_STATIC);
    }
}

static inline void schedule_corner_reflector_echoes(Bat* env, ChirpEvent* chirp,
        float slice_ticks, float freq) {
    float w = (float)ARENA_WIDTH;
    float h = (float)ARENA_HEIGHT;
    for (int i = 0; i < ARENA_REFLECTORS; i++) {
        schedule_echo(env, chirp, slice_ticks, freq,
            ARENA_REFLECTOR_X[i] * w, ARENA_REFLECTOR_Y[i] * h,
            0.0f, 0.0f, env->reflector_strength, ECHO_STATIC);
    }
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
    if (slice_idx >= chirp->slice_count) {
        return;
    }

    float t = (slice_idx + 0.5f) / (float)chirp->slice_count;
    float slice_ticks = chirp_slice_ticks(chirp, slice_idx);
    float freq = chirp->start_freq + t * (chirp->end_freq - chirp->start_freq);

    ChirpEvent slice_chirp = {
        .x = env->x,
        .y = env->y,
        .birth_tick = chirp->birth_tick,
    };
    chirp->source_x[slice_idx] = slice_chirp.x;
    chirp->source_y[slice_idx] = slice_chirp.y;

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

        float age_ticks = (float)(env->tick - chirp->birth_tick);
        while (chirp->slices_scheduled < chirp->slice_count) {
            int slice_idx = chirp->slices_scheduled;
            float slice_ticks = chirp_slice_ticks(chirp, slice_idx);
            if (slice_ticks >= age_ticks + 1.0f) break;
            schedule_chirp_slice_echoes(env, chirp, slice_idx);
            chirp->slices_scheduled += 1;
        }
    }
}

void compute_observations(Bat* env) {
    memset(env->observations, 0, OBS_SIZE * sizeof(float));
    env->tick_bug_echo_path = -1.0f;

    int slot = env->tick % ECHO_QUEUE_TICKS;
    EchoBucket* bucket = &env->echo_queue[slot];
    if (bucket->tick == env->tick) {
        for (int i = 0; i < FREQ_BINS; i++) {
            env->observations[LEFT_FREQ_OFFSET + i] = bat_clampf(bucket->energy[0][i], 0.0f, 1.0f);
            env->observations[RIGHT_FREQ_OFFSET + i] = bat_clampf(bucket->energy[1][i], 0.0f, 1.0f);
        }
        if (bucket->closest_bug_echo_path >= 0.0f) {
            env->tick_bug_echo_path = bucket->closest_bug_echo_path;
        }
        clear_echo_bucket(bucket);
    }

    float chirp_age_denom = chirp_age_norm_denominator(env);
    int chirp_age = env->tick - env->last_chirp_tick;
    if (env->last_chirp_tick < 0) chirp_age = (int)ceilf(chirp_age_denom);
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
    env->observations[TIMER_OBS] = bat_clampf(env->tick / (float)MAX_STEPS, 0.0f, 1.0f);
}

static inline void reset_episode(Bat* env) {
    env->tick = 0;
    env->turn_velocity = 0.0f;
    env->heading = randf(env) * TWO_PI - PI_F;
    env->vx = cosf(env->heading) * env->min_speed;
    env->vy = sinf(env->heading) * env->min_speed;
    if (env->curriculum_level < env->curriculum_initial_level) {
        env->curriculum_level = env->curriculum_initial_level;
    }
    env->num_obstacles = curriculum_obstacles(env);
    env->bug_inbound = env->curriculum_level >= CURRICULUM_INBOUND_START_LEVEL;
    sample_spawns_at_distance(env, env->bug_inbound
        ? curriculum_inbound_bug_distance(env)
        : curriculum_bug_distance(env));
    generate_obstacles(env);
    reset_bug_motion(env);
    // TODO: Revisit these first-observation defaults when we are ready to break determinism.
    env->last_chirp_start_freq = 0.0f;
    env->last_chirp_end_freq = 1.0f;
    env->last_chirp_duration = 0.33333334f;
    env->last_chirp_tick = -env->chirp_cooldown_ticks;
    memset(env->chirps, 0, sizeof(env->chirps));
    env->chirp_head = 0;
    clear_echo_queue(env);
    env->tick_bug_echo_path = -1.0f;
    env->last_bug_echo_path = -1.0f;
    env->last_bug_echo_expected_tick = -1.0f;
    env->chirps_emitted = 0;
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
    float speed = env->bug_inbound ? BUG_SPEED * INBOUND_BUG_SPEED_MULTIPLIER : BUG_SPEED;
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
        if (env->bug_inbound) {
            float tx, ty;
            norm_vec(env->x - env->bug_x, env->y - env->bug_y, &tx, &ty);
            env->bug_vx = tx * speed;
            env->bug_vy = ty * speed;
        }
        env->bug_base_heading = atan2f(env->bug_vy, env->bug_vx);
    }
}

static inline void update_motion(Bat* env, float dt) {
    int move = (int)env->actions[ACTION_MOVE];
    int turn = (int)env->actions[ACTION_TURN];
    float fx = cosf(env->heading);
    float fy = sinf(env->heading);
    float speed = env->vx * fx + env->vy * fy;
    if (speed < env->min_speed) speed = env->min_speed;

    if (move == THRUST_FORWARD) speed += env->accel * dt;
    if (move == BRAKE) speed -= env->accel * dt;
    speed = bat_clampf(speed, env->min_speed, env->max_speed);

    float turn_command = 0.0f;
    if (turn == TURN_LEFT) turn_command = -1.0f;
    if (turn == TURN_RIGHT) turn_command = 1.0f;
    float speed_ratio = speed / env->max_speed;
    env->turn_velocity = turn_command * env->turn_rate * bat_clampf(speed_ratio, 0.0f, 1.0f);
    env->heading += env->turn_velocity * dt;
    if (env->heading > PI_F) env->heading -= TWO_PI;
    if (env->heading < -PI_F) env->heading += TWO_PI;

    env->vx = cosf(env->heading) * speed;
    env->vy = sinf(env->heading) * speed;
    env->x += env->vx * dt;
    env->y += env->vy * dt;
}

static inline bool try_emit_chirp(Bat* env) {
    if (env->tick - env->last_chirp_tick < env->chirp_cooldown_ticks) {
        return false;
    }

    int start_idx = (int)env->actions[ACTION_CHIRP_FREQ_START];
    int end_idx = (int)env->actions[ACTION_CHIRP_FREQ_END];
    int duration_idx = (int)env->actions[ACTION_CHIRP_DURATION];

    env->last_chirp_start_freq = norm_bin(start_idx, CHIRP_FREQ_BINS);
    env->last_chirp_end_freq = norm_bin(end_idx, CHIRP_FREQ_BINS);
    env->last_chirp_duration = norm_bin(duration_idx, CHIRP_DURATION_BINS);
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
    int emit = (int)env->actions[ACTION_CHIRP_EMIT];
    if (emit) {
        if (env->chirps_emitted >= MAX_CHIRPS_PER_EPISODE) {
            return CHIRP_STATUS_OVER_BUDGET;
        }
        return try_emit_chirp(env) ? CHIRP_STATUS_EMITTED : CHIRP_STATUS_COOLDOWN;
    }

    return CHIRP_STATUS_NONE;
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
        } else if (dist(env->x, env->y, env->bug_x, env->bug_y) <= AGENT_RADIUS + BUG_RADIUS) {
            env->rewards[0] = env->chirp_efficiency_reward * chirp_efficiency(env);
            success = 1.0f;
        } else {
            float bug_dist = dist(env->x, env->y, env->bug_x, env->bug_y);
            env->rewards[0] += env->progress_reward_scale * (env->prev_bug_dist - bug_dist);
            env->rewards[0] -= env->step_cost; // TODO: Fold this only when we are ready to break training determinism.
            if (chirp_status == CHIRP_STATUS_EMITTED) {
                env->rewards[0] += env->valid_chirp_reward; // TODO: Remove this; chirps should only pay when bug echoes improve.
                if (chirp_overlap_fraction > 0.0f) {
                    env->rewards[0] -= env->chirp_overlap_penalty * chirp_overlap_fraction;
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
        if (env->last_bug_echo_path > 0.0f && dist(env->last_bug_echo_x, env->last_bug_echo_y,
                env->x, env->y) >= BUG_ECHO_MIN_DISPLACEMENT) {
            float echo_progress = (env->last_bug_echo_path - env->tick_bug_echo_path)
                / MAX_ECHO_RANGE;
            if (echo_progress > 0.0f) {
                env->rewards[0] += env->bug_echo_reward_scale * echo_progress;
            } else if (echo_progress < 0.0f) {
                env->rewards[0] += env->bug_echo_reward_scale
                    * env->bug_echo_farther_penalty_scale * echo_progress;
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
    float mid = 1.0f - fabsf(2.0f * freq_norm - 1.0f);
    return (Color){
        (unsigned char)(255.0f * (1.0f - freq_norm) + 45.0f * freq_norm),
        (unsigned char)(45.0f + 180.0f * mid),
        (unsigned char)(45.0f * (1.0f - freq_norm) + 255.0f * freq_norm),
        (unsigned char)(255.0f * alpha_norm),
    };
}

static inline void draw_chirp_rings(Bat* env, float sx, float sy) {
    for (int i = 0; i < CHIRP_HISTORY; i++) {
        ChirpEvent* chirp = &env->chirps[i];
        if (!chirp->active) continue;

        float age_seconds = (env->tick - chirp->birth_tick) * TICK_RATE;
        if (age_seconds > MAX_ECHO_RANGE / env->sound_speed + chirp->duration) {
            chirp->active = 0;
            continue;
        }

        for (int ring = 0; ring < CHIRP_RINGS; ring++) {
            float slice = ring / (float)(CHIRP_RINGS - 1);
            float freq = chirp->start_freq + slice * (chirp->end_freq - chirp->start_freq);
            float ring_age = age_seconds - slice * chirp->duration;
            if (ring_age <= 0.0f) continue;
            float radius = env->sound_speed * ring_age;
            if (radius > MAX_ECHO_RANGE) continue;

            float alpha = 0.18f + 0.42f * bat_clampf(
                1.0f - radius / MAX_ECHO_RANGE, 0.0f, 1.0f);
            float source_x, source_y;
            int slice_idx = (int)floorf(slice * (float)chirp->slice_count);
            if (slice_idx >= chirp->slice_count) slice_idx = chirp->slice_count - 1;
            chirp_source_for_slice(chirp, slice_idx, &source_x, &source_y);
            DrawCircleLines(
                (int)(source_x * sx),
                (int)(source_y * sy),
                radius * fminf(sx, sy),
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

static inline void clear_freq_history(Client* client) {
    memset(client->freq_history, 0, sizeof(client->freq_history));
    client->freq_history_head = 0;
    client->freq_history_last_tick = -1;
}

static inline void capture_freq_history(Bat* env) {
    Client* client = env->client;
    if (env->tick < client->freq_history_last_tick) {
        clear_freq_history(client);
    }
    if (env->tick == client->freq_history_last_tick) return;

    float (*sample)[FREQ_BINS] = client->freq_history[client->freq_history_head];
    for (int i = 0; i < FREQ_BINS; i++) {
        sample[0][i] = env->observations[LEFT_FREQ_OFFSET + i];
        sample[1][i] = env->observations[RIGHT_FREQ_OFFSET + i];
    }

    client->freq_history_head = (client->freq_history_head + 1) % FREQ_HISTORY_TICKS;
    client->freq_history_last_tick = env->tick;
}

static inline Color freq_history_color(int bin, float energy) {
    float e = sqrtf(bat_clampf(energy, 0.0f, 1.0f));
    if (e <= 0.001f) return (Color){42, 46, 56, 255};

    Color base = freq_color(bin / (float)(FREQ_BINS - 1), 1.0f);
    float brightness = 0.25f + 0.75f * e;
    return (Color){
        (unsigned char)(36.0f + 219.0f * (base.r / 255.0f) * brightness),
        (unsigned char)(36.0f + 219.0f * (base.g / 255.0f) * brightness),
        (unsigned char)(36.0f + 219.0f * (base.b / 255.0f) * brightness),
        255,
    };
}

static inline void draw_freq_history_band(Client* client,
        int ear, int x, int y, int width, int height) {
    float col_width = width / (float)FREQ_HISTORY_TICKS;
    float row_height = height / (float)FREQ_BINS;
    for (int t = 0; t < FREQ_HISTORY_TICKS; t++) {
        int history_idx = (client->freq_history_head + FREQ_HISTORY_TICKS - 1 - t)
            % FREQ_HISTORY_TICKS;
        int x0 = x + (int)(t * col_width);
        int x1 = x + (int)((t + 1) * col_width);
        if (x1 <= x0) x1 = x0 + 1;

        for (int row = 0; row < FREQ_BINS; row++) {
            int bin = FREQ_BINS - 1 - row;
            int y0 = y + (int)(row * row_height);
            int y1 = y + (int)((row + 1) * row_height);
            if (y1 <= y0) y1 = y0 + 1;
            DrawRectangle(x0, y0, x1 - x0, y1 - y0,
                freq_history_color(bin, client->freq_history[history_idx][ear][bin]));
        }
    }
}

typedef struct ObsBar {
    const char* label;
    int obs_idx;
    Color color;
    bool signed_value;
} ObsBar;

static inline void draw_obs_bar(int x, int y, int width,
        const ObsBar* bar, const float* observations) {
    const int label_width = 68;
    const int bar_height = 12;
    int bar_x = x + label_width;
    int bar_width = width - label_width;

    DrawText(bar->label, x, y - 1, 10, (Color){226, 230, 238, 255});
    DrawRectangle(bar_x, y, bar_width, bar_height, (Color){48, 52, 62, 255});

    if (bar->signed_value) {
        int center = bar_x + bar_width / 2;
        float value = bat_clampf(observations[bar->obs_idx], -1.0f, 1.0f);
        int fill = (int)(fabsf(value) * bar_width * 0.5f);
        if (value >= 0.0f) {
            DrawRectangle(center, y, fill, bar_height, bar->color);
        } else {
            DrawRectangle(center - fill, y, fill, bar_height, bar->color);
        }
        DrawLine(center, y, center, y + bar_height, (Color){196, 200, 210, 255});
    } else {
        float value = bat_clampf(observations[bar->obs_idx], 0.0f, 1.0f);
        DrawRectangle(bar_x, y, (int)(value * bar_width), bar_height, bar->color);
    }

    DrawRectangleLines(bar_x, y, bar_width, bar_height, (Color){118, 126, 142, 255});
}

static inline void draw_arrow_line(int x0, int y0, int x1, int y1, Color color) {
    DrawLine(x0, y0, x1, y1, color);
    float angle = atan2f((float)(y1 - y0), (float)(x1 - x0));
    const float head = 7.0f;
    DrawLine(x1, y1,
        (int)(x1 - cosf(angle - 0.45f) * head),
        (int)(y1 - sinf(angle - 0.45f) * head), color);
    DrawLine(x1, y1,
        (int)(x1 - cosf(angle + 0.45f) * head),
        (int)(y1 - sinf(angle + 0.45f) * head), color);
}

static inline void draw_observation_bars(Bat* env, int x, int y, int width) {
    static const ObsBar chirp_bars[] = {
        {"age", CHIRP_AGE_OBS, {112, 196, 255, 255}, false},
        {"cooldown", CHIRP_COOLDOWN_OBS, {255, 206, 96, 255}, false},
        {"start", CHIRP_START_OBS, {255, 112, 160, 255}, false},
        {"end", CHIRP_END_OBS, {126, 224, 255, 255}, false},
        {"duration", CHIRP_DURATION_OBS, {190, 154, 255, 255}, false},
        {"used", CHIRPS_USED_OBS, {255, 150, 96, 255}, false},
    };
    static const ObsBar action_bars[] = {
        {"speed", FORWARD_SPEED_OBS, {120, 226, 142, 255}, false},
        {"turn", TURN_RATE_OBS, {255, 112, 112, 255}, true},
    };
    static const ObsBar timer_bar = {"timer", TIMER_OBS, {88, 164, 255, 255}, false};

    const int row_step = 18;
    const Color header = (Color){246, 248, 255, 255};
    int chirp_count = (int)(sizeof(chirp_bars) / sizeof(chirp_bars[0]));
    int action_count = (int)(sizeof(action_bars) / sizeof(action_bars[0]));

    DrawText("Chirp", x, y, 12, header);
    y += 18;
    for (int i = 0; i < chirp_count; i++) {
        draw_obs_bar(x, y + i * row_step, width, &chirp_bars[i], env->observations);
    }
    y += chirp_count * row_step + 14;

    DrawText("Actions", x, y, 12, header);
    y += 18;
    for (int i = 0; i < action_count; i++) {
        draw_obs_bar(x, y + i * row_step, width, &action_bars[i], env->observations);
    }
    y += action_count * row_step + 14;

    DrawText("Episode", x, y, 12, header);
    y += 18;
    draw_obs_bar(x, y, width, &timer_bar, env->observations);
}

static inline void draw_freq_history_panel(Bat* env, int x, int y, int width, int height) {
    capture_freq_history(env);

    DrawRectangle(x, y, width, height, (Color){32, 36, 46, 255});
    int band_width = FREQ_WATERFALL_WIDTH - 2 * FREQ_PANEL_MARGIN;
    int band_height = (height - 3 * FREQ_PANEL_MARGIN) / 2;
    int left_y = y + FREQ_PANEL_MARGIN;
    int right_y = left_y + band_height + FREQ_PANEL_MARGIN;
    int obs_x = x + FREQ_WATERFALL_WIDTH + FREQ_PANEL_MARGIN;
    int obs_width = width - FREQ_WATERFALL_WIDTH - 2 * FREQ_PANEL_MARGIN;

    draw_freq_history_band(env->client, 0, x + FREQ_PANEL_MARGIN, left_y,
        band_width, band_height);
    draw_freq_history_band(env->client, 1, x + FREQ_PANEL_MARGIN, right_y,
        band_width, band_height);
    draw_observation_bars(env, obs_x, y + FREQ_PANEL_MARGIN, obs_width);
    Color reflection_color = (Color){255, 96, 96, 255};
    int reflection_text_x = obs_x + 40;
    int reflection_text_y = (left_y + right_y + band_height) / 2 - 6;
    int reflection_source_x = reflection_text_x - 8;
    int reflection_source_y = reflection_text_y + 8;
    int reflection_target_x = x + FREQ_PANEL_MARGIN + band_width - 4;
    DrawText("Reflections L/R", reflection_text_x, reflection_text_y, 12, reflection_color);
    draw_arrow_line(reflection_source_x, reflection_source_y,
        reflection_target_x, left_y + band_height / 2, reflection_color);
    draw_arrow_line(reflection_source_x, reflection_source_y + 10,
        reflection_target_x, right_y + band_height / 2, reflection_color);

    DrawRectangleLines(x, y, width, height, (Color){124, 132, 148, 255});
    DrawRectangleLines(x + FREQ_PANEL_MARGIN, left_y, band_width, band_height,
        (Color){102, 110, 126, 255});
    DrawRectangleLines(x + FREQ_PANEL_MARGIN, right_y, band_width, band_height,
        (Color){102, 110, 126, 255});
    DrawLine(x + FREQ_WATERFALL_WIDTH, y, x + FREQ_WATERFALL_WIDTH, y + height,
        (Color){86, 94, 110, 255});
}

static inline void draw_echo_flash(Bat* env, ChirpEvent* chirp,
        float rx, float ry, float rvx, float rvy, float strength,
        float sx, float sy) {
    float age_seconds = (env->tick - chirp->birth_tick) * TICK_RATE;
    float distance = dist(chirp->x, chirp->y, rx, ry);
    float echo_time = 2.0f * distance / env->sound_speed;
    if (fabsf(age_seconds - echo_time) > 0.025f) return;

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
    float w = (float)ARENA_WIDTH;
    float h = (float)ARENA_HEIGHT;
    for (int i = 0; i < ARENA_REFLECTORS; i++) {
        draw_echo_flash(env, chirp, ARENA_REFLECTOR_X[i] * w,
            ARENA_REFLECTOR_Y[i] * h, 0.0f, 0.0f, env->reflector_strength, sx, sy);
    }
}

static inline void draw_corner_reflector_markers(int width, int height) {
    const int size = 8;
    const Color fill = (Color){128, 128, 132, 255};
    const Color outline = (Color){202, 202, 208, 255};
    int max_x = width - size;
    int max_y = height - size;
    for (int i = 0; i < ARENA_REFLECTORS; i++) {
        int x = (int)(ARENA_REFLECTOR_X[i] * max_x);
        int y = (int)(ARENA_REFLECTOR_Y[i] * max_y);
        DrawRectangle(x, y, size, size, fill);
        DrawRectangleLines(x, y, size, size, outline);
    }
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
    client->width = ARENA_WIDTH * 10 + FREQ_PANEL_WIDTH;
    client->height = ARENA_HEIGHT * 10;
    clear_freq_history(client);
    InitWindow(client->width, client->height, "Bat");
    SetTargetFPS(env->render_target_fps);
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
    int arena_width = env->client->width - FREQ_PANEL_WIDTH;
    int arena_height = env->client->height;
    float sx = arena_width / (float)ARENA_WIDTH;
    float sy = env->client->height / (float)ARENA_HEIGHT;
    BeginDrawing();
    ClearBackground((Color){18, 20, 24, 255});
    draw_chirp_rings(env, sx, sy);
    draw_echo_reflections(env, sx, sy);
    DrawRectangleLines(0, 0, arena_width, arena_height, GRAY);
    for (int i = 0; i < env->num_obstacles; i++) {
        DrawRectangle(
            (int)(env->obstacle_x[i] * sx),
            (int)(env->obstacle_y[i] * sy),
            (int)(env->obstacle_w[i] * sx),
            (int)(env->obstacle_h[i] * sy),
            (Color){92, 92, 96, 255});
    }
    draw_corner_reflector_markers(arena_width, arena_height);
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
    draw_freq_history_panel(env, arena_width, 0, FREQ_PANEL_WIDTH, arena_height);
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
