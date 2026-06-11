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

#define BAT_OBS_SIZE 41
#define BAT_NUM_ACTIONS 6
#define BAT_MOVE_ACTIONS 3
#define BAT_TURN_ACTIONS 3
#define BAT_CHIRP_FREQ_BINS 8
#define BAT_CHIRP_DURATION_BINS 4
#define BAT_CHIRP_EMIT_ACTIONS 2

#define BAT_FREQ_BINS 16
#define BAT_LEFT_FREQ_OFFSET 0
#define BAT_RIGHT_FREQ_OFFSET 16
#define BAT_CHIRP_AGE_OBS 32
#define BAT_CHIRP_COOLDOWN_OBS 33
#define BAT_CHIRP_START_OBS 34
#define BAT_CHIRP_END_OBS 35
#define BAT_CHIRP_DURATION_OBS 36
#define BAT_CHIRPS_USED_OBS 37
#define BAT_FORWARD_SPEED_OBS 38
#define BAT_TURN_RATE_OBS 39

#define BAT_NOOP 0
#define BAT_THRUST_FORWARD 1
#define BAT_BRAKE 2

#define BAT_TURN_NONE 0
#define BAT_TURN_LEFT 1
#define BAT_TURN_RIGHT 2

#define BAT_MAX_OBSTACLES 16
#define BAT_TICK_RATE (1.0f/60.0f)
#define BAT_DEFAULT_MAX_STEPS 512
#define BAT_DEFAULT_MAX_STEPS_INV (1.0f / (float)BAT_DEFAULT_MAX_STEPS)
#define BAT_PI 3.14159265358979323846f
#define BAT_TWO_PI (2.0f * BAT_PI)
#define BAT_CHIRP_HISTORY 4
#define BAT_CHIRP_RINGS 5
#define BAT_MAX_CHIRP_SLICES 16
#define BAT_ECHO_QUEUE_TICKS 256
#define BAT_AUDIO_VOICES 8
#define BAT_AUDIO_SAMPLE_RATE 48000
#define BAT_AUDIO_MIN_HZ 600.0f
#define BAT_AUDIO_MAX_HZ 3600.0f
#define BAT_AUDIO_VOLUME 0.22f
#define BAT_RECORD_MAX_VOICES 16
#define BAT_CHIRP_PERF_REFERENCE_CHIRPS 15.0f
#define BAT_CHIRP_PERF_FLOOR 0.05f

#define BAT_ECHO_STATIC 0
#define BAT_ECHO_BUG 1

typedef struct ChirpEvent {
    float x;
    float y;
    float source_x[BAT_MAX_CHIRP_SLICES];
    float source_y[BAT_MAX_CHIRP_SLICES];
    float start_freq;
    float end_freq;
    float duration;
    int birth_tick;
    int slice_count;
    int slices_scheduled;
    int active;
} ChirpEvent;

typedef struct EchoBucket {
    float energy[2][BAT_FREQ_BINS];
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
    float curriculum_chirp_budget_difficulty;
    float curriculum_motion_difficulty;
    float num_obstacles;
    float chirps_emitted;
    float chirp_budget;
    float chirps_used_ratio;
    float chirp_efficiency;
    float chirp_perf;
    float chirp_overlap_fraction;
    float far_chirp_fraction;
    float near_chirp_fraction;
    float far_chirp_rate;
    float near_chirp_rate;
    float chirp_tempo_ratio;
    float first_chirp_tick_norm;
    float mean_chirp_tick_norm;
    float mean_chirp_duration;
    float mean_chirp_bandwidth;
    float n;
} Log;

typedef struct Client {
    int width;
    int height;
#ifndef BAT_HEADLESS
    int audio_ready;
    int last_audio_chirp_serial;
    int audio_voice_cursor;
    Sound chirp_sounds[BAT_AUDIO_VOICES];
    int chirp_sound_loaded[BAT_AUDIO_VOICES];
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
    BatRecordVoice record_voices[BAT_RECORD_MAX_VOICES];
#endif
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
    int render_target_fps;
    int record_video;
    int record_video_fps;
    int record_video_seconds;
    int record_video_audio;
    int num_obstacles;
    int curriculum_enabled;
    int curriculum_level;
    int curriculum_initial_level;
    int curriculum_start_obstacles;
    int curriculum_max_obstacles;
    int curriculum_obstacle_step;
    int curriculum_successes_per_level;
    int curriculum_successes_at_level;
    float curriculum_start_bug_distance;
    float curriculum_max_bug_distance;
    float curriculum_bug_distance_step;
    int curriculum_inbound_start_level;
    float curriculum_inbound_max_bug_distance;
    float curriculum_inbound_bug_distance_step;
    float inbound_bug_speed_multiplier;
    float inbound_heading_noise_degrees;
    int bug_maneuver_start_level;
    float bug_maneuver_strength;
    float bug_maneuver_frequency;

    float bat_x;
    float bat_y;
    float bat_vx;
    float bat_vy;
    float bat_heading;
    float bat_turn_velocity;
    float bat_radius;
    float ear_separation_scale;
    float ear_rear_gain;
    float ear_front_gain;
    float ear_side_gain;
    float bat_max_speed;
    float bat_min_speed;
    float bat_accel;
    float bat_turn_rate;

    float bug_x;
    float bug_y;
    float bug_vx;
    float bug_vy;
    float bug_radius;
    float bug_speed;
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

    int freq_bins_per_ear;
    float max_echo_range;
    float sound_speed;
    float reflector_spacing;
    int corner_reflectors;
    float reflector_strength;
    int max_chirp_age_ticks;
    int chirp_cooldown_ticks;
    int max_chirps_per_episode;
    int min_chirps_per_episode;
    int chirp_budget_decay_levels;
    int chirp_budget;
    int chirp_age_ticks;
    int last_chirp_tick;
    float last_chirp_start_freq;
    float last_chirp_end_freq;
    float last_chirp_duration;
    ChirpEvent chirps[BAT_CHIRP_HISTORY];
    int chirp_head;
    EchoBucket echo_queue[BAT_ECHO_QUEUE_TICKS];
    int chirps_emitted_episode;
    int audio_chirp_serial;
    int chirps_overlapped;
    float chirp_duration_sum;
    float chirp_bandwidth_sum;
    float chirps_far;
    float chirps_mid;
    float chirps_near;
    float ticks_far;
    float ticks_mid;
    float ticks_near;
    float first_chirp_tick;
    float chirp_tick_sum;

    float chirp_cost;
    float chirp_efficiency_reward;
    float valid_chirp_reward;
    float early_chirp_penalty;
    float chirp_overlap_penalty;
    float step_cost;
    float progress_reward_scale;
    float bug_echo_reward_scale;
    float bug_echo_farther_penalty_scale;
    float bug_echo_min_displacement;
    float bug_wing_sideband_gain;
    float tick_bug_echo_energy;
    float tick_bug_echo_path;
    float last_bug_echo_path;
    float last_bug_echo_expected_tick;
    float last_bug_echo_bat_x;
    float last_bug_echo_bat_y;
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
    return idx;
}

static inline float bat_chirp_duration_seconds(float duration_norm) {
    return 0.04f + 0.18f * duration_norm;
}

#include "bat_audio.h"

static inline float bat_chirp_ring_radius(float age_seconds, float slice,
        float duration_seconds, float sound_speed) {
    float ring_age = age_seconds - slice * duration_seconds;
    if (ring_age < 0.0f) return 0.0f;
    return sound_speed * ring_age;
}

static inline float bat_chirp_slice_ticks(ChirpEvent* chirp, int slice_idx) {
    return ((slice_idx + 0.5f) / (float)chirp->slice_count) *
        chirp->duration / BAT_TICK_RATE;
}

static inline void bat_chirp_source_for_slice(ChirpEvent* chirp, int slice_idx,
        float* source_x, float* source_y) {
    int scheduled = chirp->slices_scheduled;
    if (slice_idx >= 0 && slice_idx < scheduled &&
            slice_idx < BAT_MAX_CHIRP_SLICES) {
        *source_x = chirp->source_x[slice_idx];
        *source_y = chirp->source_y[slice_idx];
        return;
    }
    *source_x = chirp->x;
    *source_y = chirp->y;
}

static inline void bat_chirp_source_for_fraction(ChirpEvent* chirp, float slice,
        float* source_x, float* source_y) {
    int slices = chirp->slice_count;
    int slice_idx = (int)floorf(slice * (float)slices);
    if (slice_idx >= slices) slice_idx = slices - 1;
    bat_chirp_source_for_slice(chirp, slice_idx, source_x, source_y);
}

static inline float bat_echo_time_seconds(float distance, float sound_speed) {
    return 2.0f * distance / sound_speed;
}

static inline float bat_chirp_age_norm_denominator(Bat* env) {
    float travel_ticks = env->max_echo_range / env->sound_speed / BAT_TICK_RATE;
    float chirp_ticks = bat_chirp_duration_seconds(1.0f) / BAT_TICK_RATE;
    return 1.25f * (travel_ticks + chirp_ticks);
}

static inline float bat_norm_bin(int idx, int count) {
    return idx / (float)(count - 1);
}

static inline float bat_dist(float ax, float ay, float bx, float by) {
    float dx = bx - ax;
    float dy = by - ay;
    return sqrtf(dx*dx + dy*dy);
}

static inline void bat_norm_vec(float x, float y, float* ox, float* oy) {
    float l = sqrtf(x*x + y*y);
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

static inline bool bat_rects_overlap(float ax, float ay, float aw, float ah,
        float bx, float by, float bw, float bh, float margin) {
    return ax - margin < bx + bw &&
        ax + aw + margin > bx &&
        ay - margin < by + bh &&
        ay + ah + margin > by;
}

static inline void bat_sample_in_quadrant(Bat* env, int quadrant, float radius,
        float* x, float* y) {
    int east = quadrant & 1;
    int south = (quadrant >> 1) & 1;
    float margin = fmaxf(6.0f, radius + 3.0f);
    float half_w = env->width * 0.5f;
    float half_h = env->height * 0.5f;
    float min_x = (east ? half_w : 0.0f) + margin;
    float max_x = (east ? (float)env->width : half_w) - margin;
    float min_y = (south ? half_h : 0.0f) + margin;
    float max_y = (south ? (float)env->height : half_h) - margin;
    *x = min_x + bat_randf(env) * (max_x - min_x);
    *y = min_y + bat_randf(env) * (max_y - min_y);
}

static inline void bat_sample_spawns(Bat* env) {
    int bat_quadrant = (int)(bat_randf(env) * 4.0f);
    int bug_quadrant = bat_quadrant ^ 3;
    float min_sep = fminf(env->width, env->height) * 0.31f;

    for (int attempt = 0; attempt < 64; attempt++) {
        bat_sample_in_quadrant(env, bat_quadrant, env->bat_radius, &env->bat_x, &env->bat_y);
        bat_sample_in_quadrant(env, bug_quadrant, env->bug_radius, &env->bug_x, &env->bug_y);
        if (bat_dist(env->bat_x, env->bat_y, env->bug_x, env->bug_y) >= min_sep) {
            return;
        }
    }

    float qx[4] = {0.25f, 0.75f, 0.25f, 0.75f};
    float qy[4] = {0.25f, 0.25f, 0.75f, 0.75f};
    env->bat_x = env->width * qx[bat_quadrant];
    env->bat_y = env->height * qy[bat_quadrant];
    env->bug_x = env->width * qx[bug_quadrant];
    env->bug_y = env->height * qy[bug_quadrant];
}

static inline int bat_curriculum_obstacles(Bat* env) {
    if (!env->curriculum_enabled) return env->num_obstacles;
    int step = env->curriculum_obstacle_step;
    int count = env->curriculum_start_obstacles;
    if (env->curriculum_level > 0) {
        count = env->curriculum_start_obstacles + 1 + (env->curriculum_level - 1) / step;
    }
    if (count > env->curriculum_max_obstacles) count = env->curriculum_max_obstacles;
    if (count > BAT_MAX_OBSTACLES) count = BAT_MAX_OBSTACLES;
    return count;
}

static inline float bat_curriculum_bug_distance(Bat* env) {
    float distance = env->curriculum_start_bug_distance
        + env->curriculum_bug_distance_step * env->curriculum_level;
    return bat_clampf(distance, env->curriculum_start_bug_distance,
        env->curriculum_max_bug_distance);
}

static inline bool bat_curriculum_inbound_enabled(Bat* env) {
    if (!env->curriculum_enabled) return false;
    return env->curriculum_level >= env->curriculum_inbound_start_level;
}

static inline float bat_curriculum_inbound_bug_distance(Bat* env) {
    float base = env->curriculum_max_bug_distance;
    int extra_levels = env->curriculum_level - env->curriculum_inbound_start_level + 1;
    float distance = base + env->curriculum_inbound_bug_distance_step * extra_levels;
    return bat_clampf(distance, base, env->curriculum_inbound_max_bug_distance);
}

static inline float bat_curriculum_spawn_distance(Bat* env) {
    if (bat_curriculum_inbound_enabled(env)) {
        return bat_curriculum_inbound_bug_distance(env);
    }
    return bat_curriculum_bug_distance(env);
}

static inline float bat_curriculum_bug_speed(Bat* env) {
    float speed = env->bug_speed;
    if (bat_curriculum_inbound_enabled(env)) {
        speed *= env->inbound_bug_speed_multiplier;
    }
    return speed;
}

static inline float bat_curriculum_bug_maneuver_strength(Bat* env) {
    if (!env->curriculum_enabled) return 0.0f;
    if (env->curriculum_level < env->bug_maneuver_start_level) return 0.0f;
    int extra_levels = env->curriculum_level - env->bug_maneuver_start_level;
    float ramp = extra_levels <= 0 ? 0.25f : 0.75f + 0.25f * (extra_levels - 1);
    return env->bug_maneuver_strength * bat_clampf(ramp, 0.0f, 1.0f);
}

static inline float bat_curriculum_bug_maneuver_frequency(Bat* env) {
    if (!env->curriculum_enabled) return env->bug_maneuver_frequency;
    if (env->curriculum_level < env->bug_maneuver_start_level) {
        return env->bug_maneuver_frequency;
    }
    int extra_levels = env->curriculum_level - env->bug_maneuver_start_level;
    float multiplier = 1.0f + 0.50f * extra_levels;
    return env->bug_maneuver_frequency * bat_clampf(multiplier, 1.0f, 2.5f);
}

static inline float bat_chirps_used_ratio(Bat* env) {
    return bat_clampf(env->chirps_emitted_episode / (float)env->chirp_budget, 0.0f, 1.0f);
}

static inline float bat_chirp_efficiency(Bat* env) {
    return 0.5f + 0.5f * (1.0f - bat_chirps_used_ratio(env));
}

static inline float bat_chirp_perf(Bat* env) {
    float raw = 1.0f - env->chirps_emitted_episode / BAT_CHIRP_PERF_REFERENCE_CHIRPS;
    return bat_clampf(raw, BAT_CHIRP_PERF_FLOOR, 1.0f);
}

static inline float bat_norm_range(float value, float lo, float hi) {
    float span = hi - lo;
    return bat_clampf((value - lo) / span, 0.0f, 1.0f);
}

static inline float bat_curriculum_distance_difficulty(Bat* env) {
    float max_distance = fmaxf(env->curriculum_max_bug_distance,
        env->curriculum_inbound_max_bug_distance);
    return bat_norm_range(env->start_bug_dist,
        env->curriculum_start_bug_distance, max_distance);
}

static inline float bat_curriculum_obstacle_difficulty(Bat* env) {
    return bat_norm_range((float)env->num_obstacles,
        (float)env->curriculum_start_obstacles, (float)env->curriculum_max_obstacles);
}

static inline float bat_curriculum_motion_difficulty(Bat* env) {
    if (!env->curriculum_enabled) return 0.0f;
    if (env->curriculum_level < env->bug_maneuver_start_level) return 0.0f;
    float span = (float)(env->curriculum_inbound_start_level + 4 - env->bug_maneuver_start_level);
    return bat_clampf((env->curriculum_level - env->bug_maneuver_start_level + 1) / span,
        0.0f, 1.0f);
}

static inline float bat_curriculum_difficulty(Bat* env) {
    float distance = bat_curriculum_distance_difficulty(env);
    float obstacles = bat_curriculum_obstacle_difficulty(env);
    float active_weight = 0.0f;
    float weighted = 0.0f;
    if (env->curriculum_max_bug_distance > env->curriculum_start_bug_distance) {
        weighted += 0.5f * distance;
        active_weight += 0.5f;
    }
    if (env->curriculum_max_obstacles > env->curriculum_start_obstacles) {
        weighted += 0.5f * obstacles;
        active_weight += 0.5f;
    }
    float motion = bat_curriculum_motion_difficulty(env);
    if (env->bug_maneuver_strength > 0.0f) {
        weighted += 0.5f * motion;
        active_weight += 0.5f;
    }
    return bat_clampf(weighted / active_weight, 0.0f, 1.0f);
}

static inline float bat_success_reward(Bat* env) {
    return env->chirp_efficiency_reward * bat_chirp_efficiency(env);
}

static inline float bat_current_distance_ratio(Bat* env) {
    float dist = bat_dist(env->bat_x, env->bat_y, env->bug_x, env->bug_y);
    return dist / env->start_bug_dist;
}

static inline void bat_accumulate_distance_region(float ratio, float amount,
        float* far, float* mid, float* near) {
    if (ratio > 0.66f) {
        *far += amount;
    } else if (ratio < 0.33f) {
        *near += amount;
    } else {
        *mid += amount;
    }
}

static inline void bat_record_distance_tick(Bat* env) {
    bat_accumulate_distance_region(bat_current_distance_ratio(env), 1.0f,
        &env->ticks_far, &env->ticks_mid, &env->ticks_near);
}

static inline void bat_record_chirp_timing(Bat* env) {
    if (env->first_chirp_tick < 0.0f) {
        env->first_chirp_tick = (float)env->tick;
    }
    env->chirp_tick_sum += (float)env->tick;
    bat_accumulate_distance_region(bat_current_distance_ratio(env), 1.0f,
        &env->chirps_far, &env->chirps_mid, &env->chirps_near);
}

static inline void bat_sample_spawns_at_distance(Bat* env, float target_distance) {
    float margin = fmaxf(6.0f, fmaxf(env->bat_radius, env->bug_radius) + 3.0f);
    for (int attempt = 0; attempt < 96; attempt++) {
        float angle = bat_randf(env) * BAT_TWO_PI - BAT_PI;
        float dx = cosf(angle) * target_distance;
        float dy = sinf(angle) * target_distance;
        float min_bat_x = fmaxf(margin, margin - dx);
        float max_bat_x = fminf(env->width - margin, env->width - margin - dx);
        float min_bat_y = fmaxf(margin, margin - dy);
        float max_bat_y = fminf(env->height - margin, env->height - margin - dy);
        if (max_bat_x < min_bat_x || max_bat_y < min_bat_y) continue;

        env->bat_x = min_bat_x + bat_randf(env) * (max_bat_x - min_bat_x);
        env->bat_y = min_bat_y + bat_randf(env) * (max_bat_y - min_bat_y);
        env->bug_x = env->bat_x + dx;
        env->bug_y = env->bat_y + dy;
        return;
    }

    bat_sample_spawns(env);
}

static inline void bat_set_bug_velocity(Bat* env, float heading, float speed) {
    env->bug_base_heading = heading;
    env->bug_vx = cosf(heading) * speed;
    env->bug_vy = sinf(heading) * speed;
}

static inline void bat_reset_bug_motion(Bat* env) {
    env->bug_inbound = bat_curriculum_inbound_enabled(env) ? 1 : 0;
    float strength = bat_curriculum_bug_maneuver_strength(env);
    env->bug_maneuver_mode = strength > 0.000001f ? 1 + (int)(bat_rand(env) % 3u) : 0;
    env->bug_maneuver_phase = bat_randf(env) * BAT_TWO_PI;
    env->bug_maneuver_rate = BAT_TWO_PI * bat_curriculum_bug_maneuver_frequency(env) *
        (0.75f + 0.50f * bat_randf(env));
    env->bug_maneuver_sign = (bat_rand(env) & 1u) ? -1.0f : 1.0f;

    float speed = bat_curriculum_bug_speed(env);
    if (env->bug_inbound) {
        float tx, ty;
        bat_norm_vec(env->bat_x - env->bug_x, env->bat_y - env->bug_y, &tx, &ty);
        float noise = env->inbound_heading_noise_degrees * (BAT_PI / 180.0f);
        float heading = atan2f(ty, tx) + (2.0f * bat_randf(env) - 1.0f) * noise;
        bat_set_bug_velocity(env, heading, speed);
    } else {
        float heading = bat_randf(env) * BAT_TWO_PI - BAT_PI;
        bat_set_bug_velocity(env, heading, speed);
    }
}

static inline void bat_apply_curriculum(Bat* env) {
    if (env->curriculum_enabled) {
        env->num_obstacles = bat_curriculum_obstacles(env);
    }
}

static inline void bat_advance_curriculum(Bat* env) {
    if (env->curriculum_enabled) {
        env->curriculum_successes_at_level += 1;
        if (env->curriculum_successes_at_level >= env->curriculum_successes_per_level) {
            env->curriculum_level += 1;
            env->curriculum_successes_at_level = 0;
        }
    }
}

static inline bool bat_obstacle_clear(Bat* env, int idx, float x, float y,
        float w, float h) {
    if (bat_circle_rect_collision(env->bat_x, env->bat_y, env->bat_radius + 2.0f, x, y, w, h)) {
        return false;
    }
    if (bat_circle_rect_collision(env->bug_x, env->bug_y, env->bug_radius + 2.0f, x, y, w, h)) {
        return false;
    }
    for (int j = 0; j < idx; j++) {
        if (bat_rects_overlap(x, y, w, h,
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
            float w = 3.0f + 5.0f * bat_randf(env);
            float h = 3.0f + 5.0f * bat_randf(env);
            float margin = 4.0f;
            float x = margin + bat_randf(env) * (env->width - w - 2.0f * margin);
            float y = margin + bat_randf(env) * (env->height - h - 2.0f * margin);
            if (bat_obstacle_clear(env, i, x, y, w, h)) {
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
            float x = env->width * (0.30f + 0.20f * (i % 2)) - w * 0.5f;
            float y = env->height * (0.30f + 0.20f * ((i + 1) % 2)) - h * 0.5f;
            env->obstacle_x[i] = x;
            env->obstacle_y[i] = y;
            env->obstacle_w[i] = w;
            env->obstacle_h[i] = h;
        }
    }
}

void init(Bat* env) {
    env->tick = 0;
    env->obstacle_x = (float*)calloc(BAT_MAX_OBSTACLES, sizeof(float));
    env->obstacle_y = (float*)calloc(BAT_MAX_OBSTACLES, sizeof(float));
    env->obstacle_w = (float*)calloc(BAT_MAX_OBSTACLES, sizeof(float));
    env->obstacle_h = (float*)calloc(BAT_MAX_OBSTACLES, sizeof(float));
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
    float curriculum_difficulty = bat_curriculum_difficulty(env);
    float distance_difficulty = bat_curriculum_distance_difficulty(env);
    float obstacle_difficulty = bat_curriculum_obstacle_difficulty(env);
    float motion_difficulty = bat_curriculum_motion_difficulty(env);
    float chirp_efficiency = bat_chirp_efficiency(env);
    float chirp_perf = bat_chirp_perf(env);
    env->log.perf += success * curriculum_difficulty * chirp_perf;
    env->log.base_perf += success;
    env->log.score += env->episode_return;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->tick;
    env->log.collision += collision;
    env->log.timeout += timeout;
    env->log.curriculum_level += env->curriculum_level;
    env->log.curriculum_difficulty += curriculum_difficulty;
    env->log.curriculum_perf += success * curriculum_difficulty;
    env->log.curriculum_distance_difficulty += distance_difficulty;
    env->log.curriculum_obstacle_difficulty += obstacle_difficulty;
    env->log.curriculum_chirp_budget_difficulty += 0.0f;
    env->log.curriculum_motion_difficulty += motion_difficulty;
    env->log.num_obstacles += env->num_obstacles;
    env->log.chirps_emitted += env->chirps_emitted_episode;
    env->log.chirp_budget += env->chirp_budget;
    env->log.chirps_used_ratio += bat_chirps_used_ratio(env);
    env->log.chirp_efficiency += chirp_efficiency;
    env->log.chirp_perf += chirp_perf;
    float chirps = fmaxf(1.0f, (float)env->chirps_emitted_episode);
    env->log.chirp_overlap_fraction += env->chirps_overlapped / chirps;
    env->log.far_chirp_fraction += env->chirps_far / chirps;
    env->log.near_chirp_fraction += env->chirps_near / chirps;
    float far_rate = env->chirps_far / fmaxf(1.0f, env->ticks_far);
    float near_rate = env->chirps_near / fmaxf(1.0f, env->ticks_near);
    env->log.far_chirp_rate += far_rate;
    env->log.near_chirp_rate += near_rate;
    float tempo_ratio = 0.0f;
    if (far_rate > 0.000001f) {
        tempo_ratio = near_rate / far_rate;
    } else if (near_rate > 0.000001f) {
        tempo_ratio = 10.0f;
    }
    env->log.chirp_tempo_ratio += bat_clampf(tempo_ratio, 0.0f, 10.0f);
    env->log.first_chirp_tick_norm += env->first_chirp_tick >= 0.0f
        ? bat_clampf(env->first_chirp_tick / (float)env->max_steps, 0.0f, 1.0f)
        : 1.0f;
    env->log.mean_chirp_tick_norm += env->chirps_emitted_episode > 0
        ? bat_clampf((env->chirp_tick_sum / chirps) / (float)env->max_steps, 0.0f, 1.0f)
        : 1.0f;
    if (env->chirps_emitted_episode > 0) {
        env->log.mean_chirp_duration += env->chirp_duration_sum / env->chirps_emitted_episode;
        env->log.mean_chirp_bandwidth += env->chirp_bandwidth_sum / env->chirps_emitted_episode;
    }
    env->log.n += 1.0f;
}

static inline int bat_freq_bin_index(Bat* env, float freq_norm) {
    int bins = env->freq_bins_per_ear;
    int bin = (int)(freq_norm * bins);
    if (bin >= bins) bin = bins - 1;
    return bin;
}

static inline void bat_clear_echo_bucket(EchoBucket* bucket) {
    memset(bucket, 0, sizeof(*bucket));
    bucket->bug_path = -1.0f;
    bucket->tick = -1;
}

static inline void bat_clear_echo_queue(Bat* env) {
    for (int i = 0; i < BAT_ECHO_QUEUE_TICKS; i++) {
        bat_clear_echo_bucket(&env->echo_queue[i]);
    }
}

static inline void bat_add_echo_event(Bat* env, int ear, float receive_tick,
        float freq, float intensity, float path, int source) {
    if (receive_tick <= env->tick) return;
    if (intensity <= 0.000001f) return;
    int arrival_tick = (int)ceilf(receive_tick);
    int delay = arrival_tick - env->tick;
    if (delay <= 0 || delay >= BAT_ECHO_QUEUE_TICKS) return;
    int slot = arrival_tick % BAT_ECHO_QUEUE_TICKS;
    EchoBucket* bucket = &env->echo_queue[slot];
    if (bucket->tick != arrival_tick) {
        bat_clear_echo_bucket(bucket);
        bucket->tick = arrival_tick;
    }

    int ear_idx = ear == 0 ? 0 : 1;
    int bin = bat_freq_bin_index(env, freq);
    bucket->energy[ear_idx][bin] += intensity;
    if (source == BAT_ECHO_BUG) {
        float sideband = intensity * env->bug_wing_sideband_gain;
        int bins = env->freq_bins_per_ear;
        if (sideband > 0.000001f) {
            if (bin > 0) bucket->energy[ear_idx][bin - 1] += sideband;
            if (bin + 1 < bins) bucket->energy[ear_idx][bin + 1] += sideband;
        }
        bucket->bug_energy += intensity;
        if (bucket->bug_path < 0.0f || path < bucket->bug_path) {
            bucket->bug_path = path;
        }
    }
}

static inline void bat_ear_positions(Bat* env, float* left_x, float* left_y,
        float* right_x, float* right_y) {
    float lx = -sinf(env->bat_heading);
    float ly = cosf(env->bat_heading);
    float ear_sep = env->bat_radius * env->ear_separation_scale;
    *left_x = env->bat_x - lx * ear_sep * 0.5f;
    *left_y = env->bat_y - ly * ear_sep * 0.5f;
    *right_x = env->bat_x + lx * ear_sep * 0.5f;
    *right_y = env->bat_y + ly * ear_sep * 0.5f;
}

static inline float bat_expected_bug_echo_tick(Bat* env, ChirpEvent* chirp) {
    float fx = cosf(env->bat_heading);
    float fy = sinf(env->bat_heading);
    float source_x, source_y;
    bat_chirp_source_for_slice(chirp, 0, &source_x, &source_y);
    float ux, uy;
    bat_norm_vec(env->bug_x - source_x, env->bug_y - source_y, &ux, &uy);
    float forward = ux * fx + uy * fy;
    if (forward < -0.35f) return -1.0f;

    float left_ear_x, left_ear_y, right_ear_x, right_ear_y;
    bat_ear_positions(env, &left_ear_x, &left_ear_y, &right_ear_x, &right_ear_y);
    float source_path = bat_dist(source_x, source_y, env->bug_x, env->bug_y);
    float left_path = source_path + bat_dist(env->bug_x, env->bug_y, left_ear_x, left_ear_y);
    float right_path = source_path + bat_dist(env->bug_x, env->bug_y, right_ear_x, right_ear_y);
    float best_path = -1.0f;
    if (left_path <= env->max_echo_range) best_path = left_path;
    if (right_path <= env->max_echo_range && (best_path < 0.0f || right_path < best_path)) {
        best_path = right_path;
    }
    if (best_path < 0.0f) return -1.0f;

    float first_slice_ticks = bat_chirp_slice_ticks(chirp, 0);
    return chirp->birth_tick + first_slice_ticks + best_path / env->sound_speed / BAT_TICK_RATE;
}

static inline void bat_schedule_echo(Bat* env, ChirpEvent* chirp,
        float slice_ticks, float freq, float rx, float ry, float rvx, float rvy,
        float strength, int source) {
    float fx = cosf(env->bat_heading);
    float fy = sinf(env->bat_heading);
    float lx = -sinf(env->bat_heading);
    float ly = cosf(env->bat_heading);
    float left_ear_x, left_ear_y, right_ear_x, right_ear_y;
    bat_ear_positions(env, &left_ear_x, &left_ear_y, &right_ear_x, &right_ear_y);

    float ux, uy;
    bat_norm_vec(rx - chirp->x, ry - chirp->y, &ux, &uy);
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

    float source_path = bat_dist(chirp->x, chirp->y, rx, ry);
    float left_path = source_path + bat_dist(rx, ry, left_ear_x, left_ear_y);
    float right_path = source_path + bat_dist(rx, ry, right_ear_x, right_ear_y);
    if (left_path > env->max_echo_range && right_path > env->max_echo_range) return;

    float rel_vx = rvx - env->bat_vx;
    float rel_vy = rvy - env->bat_vy;
    float distance_rate = rel_vx * ux + rel_vy * uy;
    float doppler = bat_clampf(-distance_rate / (env->bat_max_speed + env->bug_speed), -1.0f, 1.0f);
    float shifted_freq = bat_clampf(freq + 0.20f * doppler, 0.0f, 1.0f);

    if (left_path <= env->max_echo_range) {
        float attenuation = strength / (1.0f + 0.02f * left_path * left_path);
        float receive_tick = chirp->birth_tick + slice_ticks + left_path / env->sound_speed / BAT_TICK_RATE;
        bat_add_echo_event(env, 0, receive_tick, shifted_freq, attenuation * left_gain, left_path, source);
    }
    if (right_path <= env->max_echo_range) {
        float attenuation = strength / (1.0f + 0.02f * right_path * right_path);
        float receive_tick = chirp->birth_tick + slice_ticks + right_path / env->sound_speed / BAT_TICK_RATE;
        bat_add_echo_event(env, 1, receive_tick, shifted_freq, attenuation * right_gain, right_path, source);
    }
}

static inline void bat_schedule_segment_reflectors(Bat* env, ChirpEvent* chirp,
        float slice_ticks, float freq, float x1, float y1, float x2, float y2,
        float strength) {
    float len = bat_dist(x1, y1, x2, y2);
    int count = (int)(len / env->reflector_spacing) + 1;
    if (count < 1) count = 1;
    for (int i = 0; i <= count; i++) {
        float t = count == 0 ? 0.0f : i / (float)count;
        float x = x1 + (x2 - x1) * t;
        float y = y1 + (y2 - y1) * t;
        bat_schedule_echo(env, chirp, slice_ticks, freq, x, y, 0.0f, 0.0f, strength, BAT_ECHO_STATIC);
    }
}

static inline void bat_schedule_corner_reflector_echoes(Bat* env, ChirpEvent* chirp,
        float slice_ticks, float freq) {
    if (!env->corner_reflectors) return;
    float w = (float)env->width;
    float h = (float)env->height;
    float strength = env->reflector_strength;
    bat_schedule_echo(env, chirp, slice_ticks, freq, 0.0f, 0.0f,
        0.0f, 0.0f, strength, BAT_ECHO_STATIC);
    bat_schedule_echo(env, chirp, slice_ticks, freq, w, 0.0f,
        0.0f, 0.0f, strength, BAT_ECHO_STATIC);
    bat_schedule_echo(env, chirp, slice_ticks, freq, 0.0f, h,
        0.0f, 0.0f, strength, BAT_ECHO_STATIC);
    bat_schedule_echo(env, chirp, slice_ticks, freq, w, h,
        0.0f, 0.0f, strength, BAT_ECHO_STATIC);
    bat_schedule_echo(env, chirp, slice_ticks, freq, 0.5f * w, 0.0f,
        0.0f, 0.0f, strength, BAT_ECHO_STATIC);
    bat_schedule_echo(env, chirp, slice_ticks, freq, 0.5f * w, h,
        0.0f, 0.0f, strength, BAT_ECHO_STATIC);
    bat_schedule_echo(env, chirp, slice_ticks, freq, 0.0f, 0.5f * h,
        0.0f, 0.0f, strength, BAT_ECHO_STATIC);
    bat_schedule_echo(env, chirp, slice_ticks, freq, w, 0.5f * h,
        0.0f, 0.0f, strength, BAT_ECHO_STATIC);
}

static inline void bat_schedule_obstacle_echoes(Bat* env, ChirpEvent* chirp,
        float slice_ticks, float freq, int i) {
    float x = env->obstacle_x[i];
    float y = env->obstacle_y[i];
    float w = env->obstacle_w[i];
    float h = env->obstacle_h[i];
    bat_schedule_segment_reflectors(env, chirp, slice_ticks, freq, x, y, x + w, y, 0.55f);
    bat_schedule_segment_reflectors(env, chirp, slice_ticks, freq, x, y + h, x + w, y + h, 0.55f);
    bat_schedule_segment_reflectors(env, chirp, slice_ticks, freq, x, y, x, y + h, 0.55f);
    bat_schedule_segment_reflectors(env, chirp, slice_ticks, freq, x + w, y, x + w, y + h, 0.55f);
}

static inline void bat_schedule_chirp_slice_echoes(Bat* env, ChirpEvent* chirp,
        int slice_idx) {
    int slices = chirp->slice_count;
    if (slice_idx >= slices || slice_idx >= BAT_MAX_CHIRP_SLICES) {
        return;
    }

    float t = (slice_idx + 0.5f) / (float)slices;
    float slice_ticks = bat_chirp_slice_ticks(chirp, slice_idx);
    float freq = chirp->start_freq + t * (chirp->end_freq - chirp->start_freq);

    chirp->source_x[slice_idx] = env->bat_x;
    chirp->source_y[slice_idx] = env->bat_y;
    ChirpEvent slice_chirp = *chirp;
    slice_chirp.x = chirp->source_x[slice_idx];
    slice_chirp.y = chirp->source_y[slice_idx];

    bat_schedule_echo(env, &slice_chirp, slice_ticks, freq,
        env->bug_x, env->bug_y, env->bug_vx, env->bug_vy, 8.0f, BAT_ECHO_BUG);
    bat_schedule_segment_reflectors(env, &slice_chirp, slice_ticks, freq,
        0.0f, 0.0f, (float)env->width, 0.0f, 0.12f);
    bat_schedule_segment_reflectors(env, &slice_chirp, slice_ticks, freq,
        0.0f, (float)env->height, (float)env->width, (float)env->height, 0.12f);
    bat_schedule_segment_reflectors(env, &slice_chirp, slice_ticks, freq,
        0.0f, 0.0f, 0.0f, (float)env->height, 0.12f);
    bat_schedule_segment_reflectors(env, &slice_chirp, slice_ticks, freq,
        (float)env->width, 0.0f, (float)env->width, (float)env->height, 0.12f);
    bat_schedule_corner_reflector_echoes(env, &slice_chirp, slice_ticks, freq);
    for (int j = 0; j < env->num_obstacles; j++) {
        bat_schedule_obstacle_echoes(env, &slice_chirp, slice_ticks, freq, j);
    }
}

static inline void bat_schedule_due_chirp_slices(Bat* env) {
    for (int i = 0; i < BAT_CHIRP_HISTORY; i++) {
        ChirpEvent* chirp = &env->chirps[i];
        if (!chirp->active) continue;
        int slices = chirp->slice_count;

        float age_ticks = (float)(env->tick - chirp->birth_tick);
        while (chirp->slices_scheduled < slices) {
            int slice_idx = chirp->slices_scheduled;
            float slice_ticks = bat_chirp_slice_ticks(chirp, slice_idx);
            if (slice_ticks >= age_ticks + 1.0f) break;
            bat_schedule_chirp_slice_echoes(env, chirp, slice_idx);
            chirp->slices_scheduled += 1;
        }
    }
}

static inline void bat_process_echo_events(Bat* env) {
    int slot = env->tick % BAT_ECHO_QUEUE_TICKS;
    EchoBucket* bucket = &env->echo_queue[slot];
    if (bucket->tick != env->tick) return;

    for (int i = 0; i < BAT_FREQ_BINS; i++) {
        int left_idx = BAT_LEFT_FREQ_OFFSET + i;
        int right_idx = BAT_RIGHT_FREQ_OFFSET + i;
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
    bat_clear_echo_bucket(bucket);
}

void compute_observations(Bat* env) {
    memset(env->observations, 0, BAT_OBS_SIZE * sizeof(float));
    env->tick_bug_echo_energy = 0.0f;
    env->tick_bug_echo_path = -1.0f;

    bat_process_echo_events(env);

    for (int i = 0; i < BAT_FREQ_BINS; i++) {
        env->observations[BAT_LEFT_FREQ_OFFSET + i] = bat_clampf(env->observations[BAT_LEFT_FREQ_OFFSET + i], 0.0f, 1.0f);
        env->observations[BAT_RIGHT_FREQ_OFFSET + i] = bat_clampf(env->observations[BAT_RIGHT_FREQ_OFFSET + i], 0.0f, 1.0f);
    }

    float chirp_age_denom = bat_chirp_age_norm_denominator(env);
    int chirp_age = env->tick - env->last_chirp_tick;
    if (env->last_chirp_tick < 0) chirp_age = (int)ceilf(chirp_age_denom);
    env->chirp_age_ticks = chirp_age;
    int cooldown = env->chirp_cooldown_ticks - (env->tick - env->last_chirp_tick);
    env->observations[BAT_CHIRP_AGE_OBS] = bat_clampf(chirp_age / chirp_age_denom, 0.0f, 1.0f);
    env->observations[BAT_CHIRP_COOLDOWN_OBS] = bat_clampf(cooldown / (float)env->chirp_cooldown_ticks, 0.0f, 1.0f);
    env->observations[BAT_CHIRP_START_OBS] = env->last_chirp_start_freq;
    env->observations[BAT_CHIRP_END_OBS] = env->last_chirp_end_freq;
    env->observations[BAT_CHIRP_DURATION_OBS] = env->last_chirp_duration;
    env->observations[BAT_CHIRPS_USED_OBS] = bat_chirps_used_ratio(env);
    float fwd_speed = env->bat_vx * cosf(env->bat_heading) + env->bat_vy * sinf(env->bat_heading);
    env->observations[BAT_FORWARD_SPEED_OBS] = bat_clampf(fwd_speed / env->bat_max_speed, 0.0f, 1.0f);
    env->observations[BAT_TURN_RATE_OBS] = bat_clampf(env->bat_turn_velocity / env->bat_turn_rate, -1.0f, 1.0f);
    float timer_norm = env->max_steps == BAT_DEFAULT_MAX_STEPS
        ? env->tick * BAT_DEFAULT_MAX_STEPS_INV
        : env->tick / (float)env->max_steps;
    env->observations[40] = bat_clampf(timer_norm, 0.0f, 1.0f);
}

static inline void bat_reset_episode(Bat* env) {
    env->tick = 0;
    env->bat_turn_velocity = 0.0f;
    env->bat_heading = bat_randf(env) * BAT_TWO_PI - BAT_PI;
    float initial_speed = env->bat_min_speed;
    env->bat_vx = cosf(env->bat_heading) * initial_speed;
    env->bat_vy = sinf(env->bat_heading) * initial_speed;
    if (env->curriculum_enabled && env->curriculum_level < env->curriculum_initial_level) {
        env->curriculum_level = env->curriculum_initial_level;
    }
    bat_apply_curriculum(env);
    if (env->curriculum_enabled) {
        bat_sample_spawns_at_distance(env, bat_curriculum_spawn_distance(env));
    } else {
        bat_sample_spawns(env);
    }
    generate_obstacles(env);
    bat_reset_bug_motion(env);
    env->last_chirp_start_freq = 0.0f;
    env->last_chirp_end_freq = 1.0f;
    env->last_chirp_duration = 0.33333334f;
    env->chirp_age_ticks = 0;
    env->last_chirp_tick = -env->chirp_cooldown_ticks;
    memset(env->chirps, 0, sizeof(env->chirps));
    env->chirp_head = 0;
    bat_clear_echo_queue(env);
    env->chirp_budget = env->max_chirps_per_episode;
    env->tick_bug_echo_energy = 0.0f;
    env->tick_bug_echo_path = -1.0f;
    env->last_bug_echo_path = -1.0f;
    env->last_bug_echo_expected_tick = -1.0f;
    env->chirps_emitted_episode = 0;
    env->chirps_overlapped = 0;
    env->chirp_duration_sum = 0.0f;
    env->chirp_bandwidth_sum = 0.0f;
    env->chirps_far = 0.0f;
    env->chirps_mid = 0.0f;
    env->chirps_near = 0.0f;
    env->ticks_far = 0.0f;
    env->ticks_mid = 0.0f;
    env->ticks_near = 0.0f;
    env->first_chirp_tick = -1.0f;
    env->chirp_tick_sum = 0.0f;
    env->episode_return = 0.0f;
    env->start_bug_dist = bat_dist(env->bat_x, env->bat_y, env->bug_x, env->bug_y);
    env->prev_bug_dist = env->start_bug_dist;
    env->last_bug_echo_bat_x = env->bat_x;
    env->last_bug_echo_bat_y = env->bat_y;
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
    float speed = bat_curriculum_bug_speed(env);
    float strength = bat_curriculum_bug_maneuver_strength(env);
    if (env->bug_maneuver_mode > 0) {
        env->bug_maneuver_phase += env->bug_maneuver_rate * dt;
        if (env->bug_maneuver_phase > BAT_TWO_PI) {
            env->bug_maneuver_phase -= BAT_TWO_PI;
        }
    }

    if (env->bug_inbound) {
        float tx, ty;
        bat_norm_vec(env->bat_x - env->bug_x, env->bat_y - env->bug_y, &tx, &ty);
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
    if (env->bug_x - env->bug_radius < 0.0f) {
        env->bug_x = env->bug_radius;
        env->bug_vx = fabsf(env->bug_vx);
        bounced = true;
    }
    if (env->bug_x + env->bug_radius > env->width) {
        env->bug_x = env->width - env->bug_radius;
        env->bug_vx = -fabsf(env->bug_vx);
        bounced = true;
    }
    if (env->bug_y - env->bug_radius < 0.0f) {
        env->bug_y = env->bug_radius;
        env->bug_vy = fabsf(env->bug_vy);
        bounced = true;
    }
    if (env->bug_y + env->bug_radius > env->height) {
        env->bug_y = env->height - env->bug_radius;
        env->bug_vy = -fabsf(env->bug_vy);
        bounced = true;
    }
    if (bounced) {
        env->bug_base_heading = atan2f(env->bug_vy, env->bug_vx);
        if (env->bug_inbound) {
            float tx, ty;
            bat_norm_vec(env->bat_x - env->bug_x, env->bat_y - env->bug_y, &tx, &ty);
            env->bug_vx = tx * speed;
            env->bug_vy = ty * speed;
            env->bug_base_heading = atan2f(env->bug_vy, env->bug_vx);
        }
    }
}

static inline void bat_update_motion(Bat* env, float dt) {
    int move = bat_action_index(env->actions[0], BAT_MOVE_ACTIONS);
    int turn = bat_action_index(env->actions[1], BAT_TURN_ACTIONS);
    float fx = cosf(env->bat_heading);
    float fy = sinf(env->bat_heading);
    float speed = env->bat_vx * fx + env->bat_vy * fy;
    float min_speed = env->bat_min_speed;
    if (speed < min_speed) speed = min_speed;

    if (move == BAT_THRUST_FORWARD) speed += env->bat_accel * dt;
    if (move == BAT_BRAKE) speed -= env->bat_accel * dt;
    speed = bat_clampf(speed, min_speed, env->bat_max_speed);

    float turn_command = 0.0f;
    if (turn == BAT_TURN_LEFT) turn_command = -1.0f;
    if (turn == BAT_TURN_RIGHT) turn_command = 1.0f;
    float speed_ratio = env->bat_max_speed > 0.0f ? speed / env->bat_max_speed : 0.0f;
    env->bat_turn_velocity = turn_command * env->bat_turn_rate * bat_clampf(speed_ratio, 0.0f, 1.0f);
    env->bat_heading += env->bat_turn_velocity * dt;
    if (env->bat_heading > BAT_PI) env->bat_heading -= BAT_TWO_PI;
    if (env->bat_heading < -BAT_PI) env->bat_heading += BAT_TWO_PI;

    float heading_fx = cosf(env->bat_heading);
    float heading_fy = sinf(env->bat_heading);
    env->bat_vx = heading_fx * speed;
    env->bat_vy = heading_fy * speed;
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

    if (env->chirps_emitted_episode >= env->chirp_budget) {
        return false;
    }

    env->last_chirp_start_freq = bat_norm_bin(start_idx, BAT_CHIRP_FREQ_BINS);
    env->last_chirp_end_freq = bat_norm_bin(end_idx, BAT_CHIRP_FREQ_BINS);
    env->last_chirp_duration = bat_norm_bin(duration_idx, BAT_CHIRP_DURATION_BINS);
    env->chirp_age_ticks = 0;
    env->last_chirp_tick = env->tick;
    bat_record_chirp_timing(env);
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
    chirp->slice_count = (int)ceilf(chirp->duration / BAT_TICK_RATE);
    chirp->slices_scheduled = 0;
    for (int i = 0; i < BAT_MAX_CHIRP_SLICES; i++) {
        chirp->source_x[i] = chirp->x;
        chirp->source_y[i] = chirp->y;
    }
    chirp->active = 1;
    env->chirp_head = (env->chirp_head + 1) % BAT_CHIRP_HISTORY;
    env->audio_chirp_serial += 1;
    env->last_bug_echo_expected_tick = bat_expected_bug_echo_tick(env, chirp);
    return true;
}

static inline float bat_next_chirp_overlap_fraction(Bat* env) {
    if (env->last_bug_echo_expected_tick <= (float)env->tick) return 0.0f;
    float wait_ticks = env->last_bug_echo_expected_tick - (float)env->last_chirp_tick;
    float remaining_ticks = env->last_bug_echo_expected_tick - (float)env->tick;
    return bat_clampf(remaining_ticks / wait_ticks, 0.0f, 1.0f);
}

static inline int bat_update_chirp(Bat* env) {
    int emit = bat_action_index(env->actions[5], BAT_CHIRP_EMIT_ACTIONS);
    if (emit) {
        if (env->chirps_emitted_episode >= env->chirp_budget) {
            return -2;
        }
        return bat_try_emit_chirp(env) ? 1 : -1;
    } else if (env->chirp_age_ticks < env->max_chirp_age_ticks) {
        env->chirp_age_ticks += 1;
    }
    return 0;
}

static inline bool bat_caught_bug(Bat* env) {
    return bat_dist(env->bat_x, env->bat_y, env->bug_x, env->bug_y) <= env->bat_radius + env->bug_radius;
}

void c_step(Bat* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    float chirp_overlap_fraction = bat_next_chirp_overlap_fraction(env);
    int chirp_status = bat_update_chirp(env);
    if (chirp_status == -2) {
        env->rewards[0] = -1.0f;
        env->terminals[0] = 1.0f;
        env->episode_return += env->rewards[0];
        add_log(env, 0.0f, 1.0f, 0.0f);
        bat_reset_episode(env);
        return;
    }
    if (bat_caught_bug(env)) {
        env->rewards[0] = bat_success_reward(env);
        env->terminals[0] = 1.0f;
        env->episode_return += env->rewards[0];
        bat_advance_curriculum(env);
        add_log(env, 1.0f, 0.0f, 0.0f);
        bat_reset_episode(env);
        return;
    }
    bat_schedule_due_chirp_slices(env);

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
            env->rewards[0] = bat_success_reward(env);
            env->terminals[0] = 1.0f;
            env->episode_return += env->rewards[0];
            bat_advance_curriculum(env);
            add_log(env, 1.0f, 0.0f, 0.0f);
            bat_reset_episode(env);
            return;
        }
    }

    env->tick += 1;
    bat_record_distance_tick(env);
    float bug_dist = bat_dist(env->bat_x, env->bat_y, env->bug_x, env->bug_y);
    float progress = env->prev_bug_dist - bug_dist;
    env->rewards[0] += env->progress_reward_scale * progress;
    env->rewards[0] -= env->step_cost;
    if (chirp_status > 0) {
        env->rewards[0] += env->valid_chirp_reward;
        if (chirp_overlap_fraction > 0.0f) {
            env->rewards[0] -= env->chirp_overlap_penalty * chirp_overlap_fraction;
            env->chirps_overlapped += 1;
        }
    } else if (chirp_status < 0) {
        env->rewards[0] -= env->early_chirp_penalty;
    }
    env->prev_bug_dist = bug_dist;

    if (env->tick >= env->max_steps) {
        env->rewards[0] = -1.0f;
        env->terminals[0] = 1.0f;
        env->episode_return += env->rewards[0];
        add_log(env, 0.0f, 0.0f, 1.0f);
        bat_reset_episode(env);
        return;
    }

    compute_observations(env);
    if (env->tick_bug_echo_path > 0.0f) {
        if (env->last_bug_echo_path > 0.0f) {
            float bat_echo_displacement = bat_dist(env->last_bug_echo_bat_x, env->last_bug_echo_bat_y,
                env->bat_x, env->bat_y);
            if (bat_echo_displacement >= env->bug_echo_min_displacement) {
                float echo_progress = (env->last_bug_echo_path - env->tick_bug_echo_path)
                    / env->max_echo_range;
                if (echo_progress > 0.0f) {
                    env->rewards[0] += env->bug_echo_reward_scale * echo_progress;
                } else if (echo_progress < 0.0f) {
                    env->rewards[0] += env->bug_echo_reward_scale
                        * env->bug_echo_farther_penalty_scale * echo_progress;
                }
            }
        }
        env->last_bug_echo_path = env->tick_bug_echo_path;
        env->last_bug_echo_bat_x = env->bat_x;
        env->last_bug_echo_bat_y = env->bat_y;
    }
    env->episode_return += env->rewards[0];
}

#ifndef BAT_HEADLESS
static inline Color bat_freq_color(float freq_norm, float alpha_norm) {
    float f = freq_norm;
    float mid = 1.0f - fabsf(2.0f * f - 1.0f);
    return (Color){
        (unsigned char)(255.0f * (1.0f - f) + 45.0f * f),
        (unsigned char)(45.0f + 180.0f * mid),
        (unsigned char)(45.0f * (1.0f - f) + 255.0f * f),
        (unsigned char)(255.0f * alpha_norm),
    };
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
            float source_x, source_y;
            bat_chirp_source_for_fraction(chirp, slice, &source_x, &source_y);
            DrawCircleLines(
                (int)(source_x * sx),
                (int)(source_y * sy),
                radius * scale,
                bat_freq_color(freq, alpha));
        }
    }
}

static inline Color bat_doppler_ray_color(float doppler, float alpha) {
    if (doppler > 0.05f) {
        return bat_freq_color(1.0f, alpha);
    } else if (doppler < -0.05f) {
        return bat_freq_color(0.0f, alpha);
    }
    return (Color){210, 210, 220,
        (unsigned char)(255.0f * bat_clampf(alpha, 0.0f, 1.0f))};
}

static inline void bat_draw_echo_flash(Bat* env, ChirpEvent* chirp,
        float rx, float ry, float rvx, float rvy, float strength,
        float sx, float sy) {
    float age_seconds = (env->tick - chirp->birth_tick) * BAT_TICK_RATE;
    float distance = bat_dist(chirp->x, chirp->y, rx, ry);
    float echo_time = bat_echo_time_seconds(distance, env->sound_speed);
    bool echo_arriving_now = fabsf(age_seconds - echo_time) <= 0.025f;
    if (!echo_arriving_now) return;

    float ux, uy;
    bat_norm_vec(rx - chirp->x, ry - chirp->y, &ux, &uy);
    float rel_vx = rvx - env->bat_vx;
    float rel_vy = rvy - env->bat_vy;
    float distance_rate = rel_vx * ux + rel_vy * uy;
    float doppler = bat_clampf(-distance_rate / (env->bat_max_speed + env->bug_speed), -1.0f, 1.0f);
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

static inline void bat_draw_corner_reflector_echoes(Bat* env, ChirpEvent* chirp,
        float sx, float sy) {
    if (!env->corner_reflectors) return;
    float w = (float)env->width;
    float h = (float)env->height;
    float strength = env->reflector_strength;
    bat_draw_echo_flash(env, chirp, 0.0f, 0.0f, 0.0f, 0.0f, strength, sx, sy);
    bat_draw_echo_flash(env, chirp, w, 0.0f, 0.0f, 0.0f, strength, sx, sy);
    bat_draw_echo_flash(env, chirp, 0.0f, h, 0.0f, 0.0f, strength, sx, sy);
    bat_draw_echo_flash(env, chirp, w, h, 0.0f, 0.0f, strength, sx, sy);
    bat_draw_echo_flash(env, chirp, 0.5f * w, 0.0f, 0.0f, 0.0f, strength, sx, sy);
    bat_draw_echo_flash(env, chirp, 0.5f * w, h, 0.0f, 0.0f, strength, sx, sy);
    bat_draw_echo_flash(env, chirp, 0.0f, 0.5f * h, 0.0f, 0.0f, strength, sx, sy);
    bat_draw_echo_flash(env, chirp, w, 0.5f * h, 0.0f, 0.0f, strength, sx, sy);
}

static inline void bat_draw_corner_reflector_markers(Bat* env) {
    if (!env->corner_reflectors) return;
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
        bat_draw_corner_reflector_echoes(env, chirp, sx, sy);
        for (int j = 0; j < env->num_obstacles; j++) {
            bat_draw_obstacle_echoes(env, chirp, j, sx, sy);
        }
    }
}

#include "bat_record.h"

Client* make_client(Bat* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width * 10;
    client->height = env->height * 10;
    InitWindow(client->width, client->height, "Bat");
    int target_fps = env->render_target_fps;
    if (target_fps > 0) {
        SetTargetFPS(target_fps);
    }
    InitAudioDevice();
    client->audio_ready = IsAudioDeviceReady();
    bat_record_init(env, client);
    return client;
}

void close_client(Client* client) {
    bat_record_finalize(client);
    if (client->audio_ready) {
        for (int i = 0; i < BAT_AUDIO_VOICES; i++) {
            bat_unload_chirp_sound(client, i);
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
    bat_play_chirp_audio(env);
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
    bat_draw_corner_reflector_markers(env);
    DrawCircle((int)(env->bug_x * sx), (int)(env->bug_y * sy),
        env->bug_radius * sx, GREEN);
    DrawCircle((int)(env->bat_x * sx), (int)(env->bat_y * sy),
        env->bat_radius * sx, BLUE);
    float hx = env->bat_x + cosf(env->bat_heading) * env->bat_radius * 2.0f;
    float hy = env->bat_y + sinf(env->bat_heading) * env->bat_radius * 2.0f;
    DrawLine((int)(env->bat_x * sx), (int)(env->bat_y * sy), (int)(hx * sx), (int)(hy * sy), WHITE);
    int cooldown = env->chirp_cooldown_ticks - (env->tick - env->last_chirp_tick);
    DrawText(TextFormat("reward %.3f tick %d chirps %d cooldown %d ESC exits", env->rewards[0], env->tick,
        env->chirps_emitted_episode, cooldown), 10, 10, 20, RAYWHITE);
    EndDrawing();
    bat_record_capture_frame(env);
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
