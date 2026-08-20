// CUDA Breakout — standalone env source for --cu builds.
// Completely separate from breakout.h (CPU). No trainer macros
// (precision_t / from_float / BLOCK_SIZE / grid_size). Obs is always bf16;
// pufferl casts to train precision when they differ.
#ifndef PUFFER_BREAKOUT_GPU_CU
#define PUFFER_BREAKOUT_GPU_CU

// Device Env* batch backend (see pufferenv.h PUF_BACKEND).
#define PUF_BACKEND PUF_GPU

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// Fixed env obs dtype (not tied to train precision build flags).
typedef __nv_bfloat16 obs_t;
#include "pufferenv.h"

#define ACT_SIZES {3}
#define NUM_ATNS 1
#define OBS_SIZE 118
#define HALF_PADDLE_WIDTH 31
#define Y_OFFSET 50
#define TICK_RATE (1.0f / 60.0f)
#define LEFT 1
#define RIGHT 2
#define BRICK_INDEX_NO_COLLISION -4
#define BRICK_INDEX_SIDEWALL_COLLISION -3
#define BRICK_INDEX_BACKWALL_COLLISION -2
#define BRICK_INDEX_PADDLE_COLLISION -1

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}

#define BREAKOUT_GPU_MAX_BRICKS 128
#define BREAKOUT_GPU_BRICK_WORDS ((BREAKOUT_GPU_MAX_BRICKS + 31) / 32)
#define BREAKOUT_GPU_PI 3.14159265358979323846f
#define BREAKOUT_THREADS_PER_ENV 8  // coop lanes for coalesced obs write
#define BREAKOUT_BLOCK 256
static int breakout_grid(int n) {
    return (n + BREAKOUT_BLOCK - 1) / BREAKOUT_BLOCK;
}

// Shared config (constant mem).
typedef struct GpuBreakoutConfig {
    int width;
    int height;
    int brick_width;
    int brick_height;
    int brick_rows;
    int brick_cols;
    int num_bricks;
    int ball_width;
    int ball_height;
    float paddle_height;
    float paddle_speed;
    float initial_paddle_width;
    float initial_ball_speed;
    float max_ball_speed;
    int max_score;
    int half_max_score;
    int frameskip;
    int continuous;
} GpuBreakoutConfig;

// Per-env state. Log first for parallel log reduce. Trainer-facing fields
// (agents, num_agents, tag, boundary_reached) match CPU Env contract; GPU
// kernels ignore them. Selfplay not wired for GPU yet.
typedef struct Env {
    Log log;
    Agent agents[1];
    int num_agents;
    int tag;
    int boundary_reached;
    float paddle_x;
    float paddle_y;
    float ball_x;
    float ball_y;
    float ball_vx;
    float ball_vy;
    float paddle_width;
    float ball_speed;
    int score;
    int balls_fired;
    int hits;
    int num_balls;
    int tick;
    unsigned int rng;
    // bit set = destroyed (was float 1.0); clear = alive (was float 0.0)
    unsigned int brick_mask[BREAKOUT_GPU_BRICK_WORDS];
} Env;

typedef struct GpuBreakoutCollisionInfo {
    float t;
    float overlap;
    float x;
    float y;
    float vx;
    float vy;
    int brick_index;
} GpuBreakoutCollisionInfo;

__constant__ GpuBreakoutConfig d_bcfg;

// ---- brick bitset helpers ----
__device__ __host__ static inline int gpu_brick_alive(const unsigned int* mask, int i) {
    return (mask[i >> 5] & (1u << (i & 31))) == 0u;
}

__device__ __host__ static inline void gpu_brick_destroy(unsigned int* mask, int i) {
    mask[i >> 5] |= (1u << (i & 31));
}

__device__ __host__ static inline void gpu_brick_clear_all(unsigned int* mask) {
    #pragma unroll
    for (int w = 0; w < BREAKOUT_GPU_BRICK_WORDS; w++) {
        mask[w] = 0u;
    }
}

__host__ static int gpu_breakout_compute_half_max_score(int brick_rows, int brick_cols) {
    int half = 0;
    for (int row = 0; row < brick_rows; row++) {
        for (int col = 0; col < brick_cols; col++) {
            int idx = row * brick_cols + col;
            half += 7 - 3 * (idx / brick_cols / 2);
        }
    }
    return half;
}

// Cheap xorshift32 — only used for a coin flip on ball launch.
__device__ static inline unsigned int gpu_breakout_xorshift(unsigned int* seed) {
    unsigned int x = *seed;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *seed = x ? x : 0xA341316Cu; // never stick at 0
    return *seed;
}

__device__ static inline void gpu_breakout_add_log(Env* env) {
    env->log.episode_length += env->tick;
    env->log.episode_return += env->score;
    env->log.score += env->score;
    env->log.perf += env->score / (float)d_bcfg.max_score;
    env->log.n += 1.0f;
}

// Store obs feature: float compute → bf16 obs.
__device__ __forceinline__ void gpu_breakout_store_obs(obs_t* dst, float v) {
    *dst = __float2bfloat16(v);
}

// Single-thread obs write (reset path / fallback). Prefer coop variant in step.
__device__ static inline void gpu_breakout_compute_observations(const Env* env, obs_t* obs) {
    const GpuBreakoutConfig* c = &d_bcfg;
    gpu_breakout_store_obs(&obs[0], env->paddle_x / (float)c->width);
    gpu_breakout_store_obs(&obs[1], env->paddle_y / (float)c->height);
    gpu_breakout_store_obs(&obs[2], env->ball_x / (float)c->width);
    gpu_breakout_store_obs(&obs[3], env->ball_y / (float)c->height);
    gpu_breakout_store_obs(&obs[4], env->ball_vx / 512.0f);
    gpu_breakout_store_obs(&obs[5], env->ball_vy / 512.0f);
    gpu_breakout_store_obs(&obs[6], env->balls_fired / 5.0f);
    gpu_breakout_store_obs(&obs[7], env->score / 864.0f);
    gpu_breakout_store_obs(&obs[8], env->num_balls / 5.0f);
    gpu_breakout_store_obs(&obs[9], env->paddle_width / (2.0f * HALF_PADDLE_WIDTH));

    const unsigned int* mask = env->brick_mask;
    int n = c->num_bricks;
    for (int i = 0; i < n; i++) {
        gpu_breakout_store_obs(&obs[10 + i], gpu_brick_alive(mask, i) ? 0.0f : 1.0f);
    }
}

// Multi-lane coalesced obs write. `lane` in [0, nlanes). Adjacent lanes store
// adjacent elements so a warp's stores hit contiguous 128B segments.
__device__ static inline void gpu_breakout_compute_observations_coop(
        const Env* env, obs_t* obs, int lane, int nlanes) {
    const GpuBreakoutConfig* c = &d_bcfg;
    // 10 scalar features
    if (lane < 10) {
        float v;
        switch (lane) {
            case 0: v = env->paddle_x / (float)c->width; break;
            case 1: v = env->paddle_y / (float)c->height; break;
            case 2: v = env->ball_x / (float)c->width; break;
            case 3: v = env->ball_y / (float)c->height; break;
            case 4: v = env->ball_vx / 512.0f; break;
            case 5: v = env->ball_vy / 512.0f; break;
            case 6: v = env->balls_fired / 5.0f; break;
            case 7: v = env->score / 864.0f; break;
            case 8: v = env->num_balls / 5.0f; break;
            default: v = env->paddle_width / (2.0f * HALF_PADDLE_WIDTH); break;
        }
        gpu_breakout_store_obs(&obs[lane], v);
    }
    const unsigned int* mask = env->brick_mask;
    int n = c->num_bricks;
    for (int i = lane; i < n; i += nlanes) {
        gpu_breakout_store_obs(&obs[10 + i], gpu_brick_alive(mask, i) ? 0.0f : 1.0f);
    }
}

__device__ static inline bool gpu_breakout_calc_vline_collision(float xw, float yw, float hw,
        float x, float y, float vx, float vy, float h, GpuBreakoutCollisionInfo* col) {
    float t_new = (xw - x) / vx;
    float topmost = fminf(yw + hw, y + h + vy * t_new);
    float botmost = fmaxf(yw, y + vy * t_new);
    float overlap_new = topmost - botmost;

    if (overlap_new > 0.0f && t_new > 0.0f && t_new <= 1.0f &&
            (t_new < col->t || (t_new == col->t && overlap_new > col->overlap))) {
        col->t = t_new;
        col->overlap = overlap_new;
        col->x = xw;
        col->y = y + vy * t_new;
        col->vx = -vx;
        col->vy = vy;
        return true;
    }
    return false;
}

__device__ static inline bool gpu_breakout_calc_hline_collision(float xw, float yw, float ww,
        float x, float y, float vx, float vy, float w, GpuBreakoutCollisionInfo* col) {
    float t_new = (yw - y) / vy;
    float rightmost = fminf(xw + ww, x + w + vx * t_new);
    float leftmost = fmaxf(xw, x + vx * t_new);
    float overlap_new = rightmost - leftmost;

    if (overlap_new > 0.0f && t_new > 0.0f && t_new <= 1.0f &&
            (t_new < col->t || (t_new == col->t && overlap_new > col->overlap))) {
        col->t = t_new;
        col->overlap = overlap_new;
        col->x = x + vx * t_new;
        col->y = yw;
        col->vx = vx;
        col->vy = -vy;
        return true;
    }
    return false;
}

__device__ static inline void gpu_breakout_calc_brick_collision(
        float brick_x, float brick_y, float bw, float bh,
        float ball_x, float ball_y, float ball_vx, float ball_vy,
        float ball_w, float ball_h, int idx,
        GpuBreakoutCollisionInfo* collision_info) {
    bool collision = false;

    if (ball_vx > 0.0f) {
        if (gpu_breakout_calc_vline_collision(brick_x, brick_y, bh,
                ball_x + ball_w, ball_y, ball_vx, ball_vy, ball_h, collision_info)) {
            collision = true;
            collision_info->x -= ball_w;
        }
    } else if (ball_vx < 0.0f) {
        if (gpu_breakout_calc_vline_collision(brick_x + bw, brick_y, bh,
                ball_x, ball_y, ball_vx, ball_vy, ball_h, collision_info)) {
            collision = true;
        }
    }

    if (ball_vy > 0.0f) {
        if (gpu_breakout_calc_hline_collision(brick_x, brick_y, bw,
                ball_x, ball_y + ball_h, ball_vx, ball_vy, ball_w, collision_info)) {
            collision = true;
            collision_info->y -= ball_h;
        }
    } else if (ball_vy < 0.0f) {
        if (gpu_breakout_calc_hline_collision(brick_x, brick_y + bh, bw,
                ball_x, ball_y, ball_vx, ball_vy, ball_w, collision_info)) {
            collision = true;
        }
    }
    if (collision) {
        collision_info->brick_index = idx;
    }
}

__device__ static inline void gpu_breakout_calc_all_brick_collisions(Env* env,
        GpuBreakoutCollisionInfo* collision_info) {
    const GpuBreakoutConfig* c = &d_bcfg;
    float ball_x = env->ball_x;
    float ball_vx = env->ball_vx;
    float ball_y = env->ball_y;
    float ball_vy = env->ball_vy;
    float ball_w = (float)c->ball_width;
    float ball_h = (float)c->ball_height;
    float ball_x_dst = ball_x + ball_vx;
    float ball_y_dst = ball_y + ball_vy;
    int bw = c->brick_width;
    int bh = c->brick_height;
    int rows = c->brick_rows;
    int cols = c->brick_cols;

    // Early-out: ball fully below the brick field and moving down can't hit bricks.
    float brick_bottom = (float)(Y_OFFSET + rows * bh);
    float ball_top = fminf(ball_y, ball_y_dst);
    if (ball_top >= brick_bottom) {
        return;
    }

    int row_from = (int)((fminf(ball_y, ball_y_dst) - Y_OFFSET) / (float)bh);
    if (row_from < 0) row_from = 0;
    if (row_from >= rows) return;

    int column_from = (int)(fminf(ball_x, ball_x_dst) / (float)bw);
    if (column_from < 0) column_from = 0;

    float ball_x_end = ball_x + ball_w;
    float ball_x_dst_end = ball_x_dst + ball_w;
    int column_to = (int)(fmaxf(ball_x_end, ball_x_dst_end) / (float)bw);
    if (column_to >= cols) column_to = cols - 1;

    float ball_y_end = ball_y + ball_h;
    float ball_y_dst_end = ball_y_dst + ball_h;
    int row_to = (int)((fmaxf(ball_y_end, ball_y_dst_end) - Y_OFFSET) / (float)bh);
    if (row_to >= rows) row_to = rows - 1;

    const unsigned int* mask = env->brick_mask;
    float bwf = (float)bw;
    float bhf = (float)bh;

    for (int row = row_from; row <= row_to; row++) {
        float brick_y = (float)(row * bh + Y_OFFSET);
        int row_base = row * cols;
        for (int column = column_from; column <= column_to; column++) {
            int brick_index = row_base + column;
            if (gpu_brick_alive(mask, brick_index)) {
                float brick_x = (float)(column * bw);
                gpu_breakout_calc_brick_collision(
                    brick_x, brick_y, bwf, bhf,
                    ball_x, ball_y, ball_vx, ball_vy, ball_w, ball_h,
                    brick_index, collision_info);
            }
        }
    }
}

__device__ static inline bool gpu_breakout_calc_paddle_ball_collisions(Env* env,
        GpuBreakoutCollisionInfo* collision_info) {
    const GpuBreakoutConfig* c = &d_bcfg;
    float base_angle = BREAKOUT_GPU_PI / 4.0f;
    float ball_h = (float)c->ball_height;
    float ball_w = (float)c->ball_width;

    if (env->ball_y + ball_h + env->ball_vy < env->paddle_y) {
        return false;
    }

    if (!gpu_breakout_calc_hline_collision(env->paddle_x, env->paddle_y, env->paddle_width,
            env->ball_x, env->ball_y + ball_h, env->ball_vx, env->ball_vy,
            ball_w, collision_info) || collision_info->t > 1.0f) {
        return false;
    }

    collision_info->y -= ball_h;
    collision_info->brick_index = BRICK_INDEX_PADDLE_COLLISION;

    float relative_intersection =
        ((env->ball_x + ball_w * 0.5f) - env->paddle_x) / env->paddle_width;
    float angle = -base_angle + relative_intersection * 2.0f * base_angle;
    float speed_tick = env->ball_speed * TICK_RATE;
    env->ball_vx = sinf(angle) * speed_tick;
    env->ball_vy = -cosf(angle) * speed_tick;
    env->hits += 1;
    if (env->hits % 4 == 0 && env->ball_speed < c->max_ball_speed) {
        env->ball_speed += 64;
    }
    if (env->score == c->half_max_score) {
        gpu_brick_clear_all(env->brick_mask);
    }
    return true;
}

__device__ static inline void gpu_breakout_calc_all_wall_collisions(Env* env,
        GpuBreakoutCollisionInfo* collision_info) {
    const GpuBreakoutConfig* c = &d_bcfg;
    float ball_x = env->ball_x;
    float ball_y = env->ball_y;
    float ball_vx = env->ball_vx;
    float ball_vy = env->ball_vy;
    float ball_w = (float)c->ball_width;
    float ball_h = (float)c->ball_height;
    float width = (float)c->width;
    float height = (float)c->height;

    if (ball_vx < 0.0f) {
        if (gpu_breakout_calc_vline_collision(0.0f, 0.0f, height,
                ball_x, ball_y, ball_vx, ball_vy, ball_h, collision_info)) {
            collision_info->brick_index = BRICK_INDEX_SIDEWALL_COLLISION;
        }
    } else if (ball_vx > 0.0f) {
        if (gpu_breakout_calc_vline_collision(width, 0.0f, height,
                ball_x + ball_w, ball_y, ball_vx, ball_vy, ball_h, collision_info)) {
            collision_info->x -= ball_w;
            collision_info->brick_index = BRICK_INDEX_SIDEWALL_COLLISION;
        }
    }
    if (ball_vy < 0.0f) {
        if (gpu_breakout_calc_hline_collision(0.0f, 0.0f, width,
                ball_x, ball_y, ball_vx, ball_vy, ball_w, collision_info)) {
            collision_info->brick_index = BRICK_INDEX_BACKWALL_COLLISION;
        }
    }
}

__device__ static inline void gpu_breakout_check_wall_bounds(Env* env) {
    float offset = d_bcfg.max_ball_speed * 1.1f * TICK_RATE;
    float width = (float)d_bcfg.width;
    if (env->ball_x < 0.0f) {
        env->ball_x += offset;
    }
    if (env->ball_x > width) {
        env->ball_x -= offset;
    }
    if (env->ball_y < 0.0f) {
        env->ball_y += offset;
    }
}

__device__ static inline void gpu_breakout_destroy_brick(Env* env, int brick_idx,
        float* reward) {
    const GpuBreakoutConfig* c = &d_bcfg;
    float gained_points = 7 - 3 * ((brick_idx / c->brick_cols) / 2);

    env->score += (int)gained_points;
    gpu_brick_destroy(env->brick_mask, brick_idx);
    *reward += gained_points;

    if (brick_idx / c->brick_cols < 3) {
        env->ball_speed = c->max_ball_speed;
    }
}

__device__ static inline bool gpu_breakout_handle_collisions(Env* env, float* reward) {
    GpuBreakoutCollisionInfo collision_info = {
        .t = 2.0f,
        .overlap = -1.0f,
        .x = 0.0f,
        .y = 0.0f,
        .vx = 0.0f,
        .vy = 0.0f,
        .brick_index = BRICK_INDEX_NO_COLLISION,
    };

    gpu_breakout_check_wall_bounds(env);

    gpu_breakout_calc_all_brick_collisions(env, &collision_info);
    gpu_breakout_calc_all_wall_collisions(env, &collision_info);
    gpu_breakout_calc_paddle_ball_collisions(env, &collision_info);
    if (collision_info.brick_index != BRICK_INDEX_PADDLE_COLLISION && collision_info.t <= 1.0f) {
        env->ball_x = collision_info.x;
        env->ball_y = collision_info.y;
        env->ball_vx = collision_info.vx;
        env->ball_vy = collision_info.vy;
        if (collision_info.brick_index >= 0) {
            gpu_breakout_destroy_brick(env, collision_info.brick_index, reward);
        }
        if (collision_info.brick_index == BRICK_INDEX_BACKWALL_COLLISION) {
            env->paddle_width = HALF_PADDLE_WIDTH;
        }
    }
    return collision_info.brick_index != BRICK_INDEX_NO_COLLISION;
}

__device__ static inline void gpu_breakout_reset_round(Env* env) {
    const GpuBreakoutConfig* c = &d_bcfg;
    env->balls_fired = 0;
    env->hits = 0;
    env->ball_speed = c->initial_ball_speed;
    env->paddle_width = c->initial_paddle_width;

    env->paddle_x = c->width / 2.0f - env->paddle_width / 2.0f;
    env->paddle_y = c->height - c->paddle_height - 10;

    env->ball_x = env->paddle_x + (env->paddle_width / 2.0f - c->ball_width / 2.0f);
    env->ball_y = c->height / 2.0f - 30;

    env->ball_vx = 0.0f;
    env->ball_vy = 0.0f;
}

__device__ static inline void gpu_breakout_reset_state(Env* env) {
    env->score = 0;
    env->num_balls = 5;
    gpu_brick_clear_all(env->brick_mask);
    gpu_breakout_reset_round(env);
    env->tick = 0;
}

__device__ static inline void gpu_breakout_step_frame(Env* env, float action,
        float* reward, float* terminal) {
    const GpuBreakoutConfig* c = &d_bcfg;
    float act = 0.0f;
    if (env->balls_fired == 0) {
        env->balls_fired = 1;
        float direction = BREAKOUT_GPU_PI / 3.25f;
        float speed_tick = env->ball_speed * TICK_RATE;

        env->ball_vy = cosf(direction) * speed_tick;
        env->ball_vx = sinf(direction) * speed_tick;
        if ((gpu_breakout_xorshift(&env->rng) & 1u) == 0u) {
            env->ball_vx = -env->ball_vx;
        }
    } else if (action == LEFT) {
        act = -1.0f;
    } else if (action == RIGHT) {
        act = 1.0f;
    }
    if (c->continuous) {
        act = action;
    }
    env->paddle_x = fminf((float)c->width - env->paddle_width,
        fmaxf(0.0f, env->paddle_x + act * c->paddle_speed * TICK_RATE));

    if (!gpu_breakout_handle_collisions(env, reward)) {
        env->ball_x += env->ball_vx;
        env->ball_y += env->ball_vy;
    }

    if (env->ball_y >= env->paddle_y + c->paddle_height) {
        env->num_balls -= 1;
        gpu_breakout_reset_round(env);
    }
    if (env->num_balls < 0 || env->score == c->max_score) {
        *terminal = 1.0f;
        gpu_breakout_add_log(env);
        gpu_breakout_reset_state(env);
    }
}

__global__ void gpu_breakout_reset_kernel(Env* envs, obs_t* observations,
        float* rewards, float* terminals, int num_envs) {
    // BREAKOUT_THREADS_PER_ENV threads per env (coalesced obs).
    // No early-return before __syncwarp — partial warps would deadlock.
    int thr = blockIdx.x * blockDim.x + threadIdx.x;
    int rel = thr / BREAKOUT_THREADS_PER_ENV;
    int lane = thr - rel * BREAKOUT_THREADS_PER_ENV;
    int active = rel < num_envs;

    if (active && lane == 0) {
        gpu_breakout_reset_state(&envs[rel]);
        rewards[rel] = 0.0f;
        terminals[rel] = 0.0f;
    }
    __syncwarp();
    if (active) {
        gpu_breakout_compute_observations_coop(
            &envs[rel], observations + (long)rel * OBS_SIZE, lane, BREAKOUT_THREADS_PER_ENV);
    }
}

// Lane 0: in-place physics for frameskip frames (no full-struct local copy / stack).
// All lanes: coalesced AoS obs write (1-thread AoS was ~80% of kernel time on 5090).
__global__ void gpu_breakout_step_kernel(Env* __restrict__ envs,
        const float* __restrict__ actions, obs_t* __restrict__ observations,
        float* __restrict__ rewards, float* __restrict__ terminals, int start, int count) {
    // No early-return before __syncwarp — partial warps would deadlock.
    int thr = blockIdx.x * blockDim.x + threadIdx.x;
    int rel = thr / BREAKOUT_THREADS_PER_ENV;
    int lane = thr - rel * BREAKOUT_THREADS_PER_ENV;
    int active = rel < count;
    int idx = start + rel;

    if (active && lane == 0) {
        Env* env = &envs[idx];
        float reward = 0.0f;
        float terminal = 0.0f;
        float action = actions[(long)idx * NUM_ATNS];
        int frameskip = d_bcfg.frameskip;
        for (int i = 0; i < frameskip; i++) {
            env->tick += 1;
            gpu_breakout_step_frame(env, action, &reward, &terminal);
        }
        rewards[idx] = reward;
        terminals[idx] = terminal;
    }
    // Reconv + visibility: other lanes must see lane-0's stores before reading state.
    __syncwarp();
    if (active) {
        gpu_breakout_compute_observations_coop(
            &envs[idx], observations + (long)idx * OBS_SIZE, lane, BREAKOUT_THREADS_PER_ENV);
    }
}


static void gpu_breakout_host_fill_config(GpuBreakoutConfig* cfg, Dict* kwargs) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->frameskip = dict_get(kwargs, "frameskip");
    cfg->width = dict_get(kwargs, "width");
    cfg->height = dict_get(kwargs, "height");
    cfg->initial_paddle_width = dict_get(kwargs, "paddle_width");
    cfg->paddle_height = dict_get(kwargs, "paddle_height");
    cfg->ball_width = dict_get(kwargs, "ball_width");
    cfg->ball_height = dict_get(kwargs, "ball_height");
    cfg->brick_width = dict_get(kwargs, "brick_width");
    cfg->brick_height = dict_get(kwargs, "brick_height");
    cfg->brick_rows = dict_get(kwargs, "brick_rows");
    cfg->brick_cols = dict_get(kwargs, "brick_cols");
    cfg->initial_ball_speed = dict_get(kwargs, "initial_ball_speed");
    cfg->max_ball_speed = dict_get(kwargs, "max_ball_speed");
    cfg->paddle_speed = dict_get(kwargs, "paddle_speed");
    cfg->continuous = dict_get(kwargs, "continuous");
    cfg->num_bricks = cfg->brick_rows * cfg->brick_cols;
    if (cfg->num_bricks <= 0 || cfg->num_bricks > BREAKOUT_GPU_MAX_BRICKS ||
            cfg->num_bricks > OBS_SIZE - 10) {
        fprintf(stderr, "Breakout GPU env supports 1..%d bricks and OBS_SIZE-10 slots; got %d\n",
            BREAKOUT_GPU_MAX_BRICKS, cfg->num_bricks);
        exit(1);
    }
    cfg->half_max_score = gpu_breakout_compute_half_max_score(cfg->brick_rows, cfg->brick_cols);
    cfg->max_score = 2 * cfg->half_max_score;
}

// Bound EnvBuf + device Env array. Same puf_reset/step/close(Env*) sig as CPU:
// Env* is the device base; step/reset always run the full batch (env handles the loop).
typedef struct BreakoutClient {
    int width;
    int height;
    Texture2D ball;
} BreakoutClient;

static struct {
    Env* envs;
    int n;
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    cudaStream_t stream;
    GpuBreakoutConfig cfg;
    BreakoutClient* client;
} g_gpu;

static Color BREAKOUT_BRICK_COLORS[6] = {
    RED, ORANGE, YELLOW, GREEN, SKYBLUE, BLUE,
};

// Trainer create: returns device Env base (also stored in g_gpu).
Env* puf_vec_create(int n, Dict* env_kwargs,
        obs_t* observations, float* actions, float* rewards, float* terminals) {
    gpu_breakout_host_fill_config(&g_gpu.cfg, env_kwargs);
    cudaMemcpyToSymbol(d_bcfg, &g_gpu.cfg, sizeof(GpuBreakoutConfig));

    Env* host_envs = (Env*)calloc(n, sizeof(Env));
    for (int i = 0; i < n; i++) {
        host_envs[i].rng = (unsigned int)(i + 1);
        host_envs[i].num_balls = -1;
        host_envs[i].num_agents = 1;
        host_envs[i].tag = 0;
        host_envs[i].boundary_reached = 0;
        host_envs[i].agents[0].policy = 0;
        host_envs[i].agents[0].observations = NULL;
        host_envs[i].agents[0].actions = NULL;
        host_envs[i].agents[0].rewards = NULL;
        host_envs[i].agents[0].terminals = NULL;
        host_envs[i].agents[0].action_mask = NULL;
    }
    Env* envs = NULL;
    cudaMalloc((void**)&envs, n * sizeof(Env));
    cudaMemcpy(envs, host_envs, n * sizeof(Env), cudaMemcpyHostToDevice);
    free(host_envs);

    g_gpu.envs = envs;
    g_gpu.n = n;
    g_gpu.observations = observations;
    g_gpu.actions = actions;
    g_gpu.rewards = rewards;
    g_gpu.terminals = terminals;
    g_gpu.stream = 0;
    g_gpu.client = NULL;
    return envs;
}

void puf_bind_stream(cudaStream_t stream) {
    g_gpu.stream = stream;
}

// Same signatures as CPU. GPU path uses batch base; puf_init is unused (create fills).
void puf_init(Env* env, Dict* kwargs) {
}

void puf_reset(Env* env) {
    int threads = g_gpu.n * BREAKOUT_THREADS_PER_ENV;
    gpu_breakout_reset_kernel<<<breakout_grid(threads), BREAKOUT_BLOCK>>>(
        g_gpu.envs, g_gpu.observations, g_gpu.rewards, g_gpu.terminals, g_gpu.n);
}

void puf_step(Env* env) {
    int threads = g_gpu.n * BREAKOUT_THREADS_PER_ENV;
    gpu_breakout_step_kernel<<<breakout_grid(threads), BREAKOUT_BLOCK, 0, g_gpu.stream>>>(
        g_gpu.envs, g_gpu.actions, g_gpu.observations, g_gpu.rewards, g_gpu.terminals,
        0, g_gpu.n);
}

void puf_close(Env* env) {
    if (g_gpu.client) {
        UnloadTexture(g_gpu.client->ball);
        CloseWindow();
        free(g_gpu.client);
        g_gpu.client = NULL;
    }
    cudaFree(g_gpu.envs);
    g_gpu.envs = NULL;
}

// D2H env 0 and draw (same layout as breakout.h). env is device batch base.
void puf_render(Env* env) {
    if (!g_gpu.envs || g_gpu.n < 1) {
        return;
    }
    if (g_gpu.stream) {
        cudaStreamSynchronize(g_gpu.stream);
    }
    Env h;
    cudaMemcpy(&h, g_gpu.envs, sizeof(Env), cudaMemcpyDeviceToHost);
    GpuBreakoutConfig* c = &g_gpu.cfg;

    if (!g_gpu.client) {
        BreakoutClient* client = (BreakoutClient*)calloc(1, sizeof(BreakoutClient));
        client->width = c->width;
        client->height = c->height;
        InitWindow(c->width, c->height, "PufferLib Breakout (GPU)");
        SetTargetFPS(60 / (c->frameskip > 0 ? c->frameskip : 1));
        client->ball = LoadTexture("resources/shared/puffers_128.png");
        g_gpu.client = client;
    }
    BreakoutClient* client = g_gpu.client;

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_TAB)) {
        ToggleFullscreen();
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    DrawRectangle(
        (int)h.paddle_x, (int)h.paddle_y,
        (int)h.paddle_width, (int)c->paddle_height,
        (Color){0, 255, 255, 255});

    DrawTexturePro(
        client->ball,
        (Rectangle){
            (h.ball_vx > 0) ? 0.0f : 128.0f,
            0, 128, 128,
        },
        (Rectangle){
            h.ball_x,
            h.ball_y,
            (float)c->ball_width,
            (float)c->ball_height,
        },
        (Vector2){0, 0},
        0,
        WHITE);

    for (int row = 0; row < c->brick_rows; row++) {
        for (int col = 0; col < c->brick_cols; col++) {
            int brick_idx = row * c->brick_cols + col;
            if (!gpu_brick_alive(h.brick_mask, brick_idx)) {
                continue;
            }
            int x = col * c->brick_width;
            int y = row * c->brick_height + Y_OFFSET;
            Color brick_color = BREAKOUT_BRICK_COLORS[row % 6];
            DrawRectangle(x, y, c->brick_width, c->brick_height, brick_color);
        }
    }

    DrawText(TextFormat("Score: %i", h.score), 10, 10, 20, WHITE);
    DrawText(TextFormat("Balls: %i", h.num_balls), client->width - 80, 10, 20, WHITE);
    EndDrawing();
    puf_web_vsync();
}

#endif
