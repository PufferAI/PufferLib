// Vibe coded by OpenAI Codex.
// GPU Affine Lock environment. This is intentionally standalone from
// affine_lock.h: --gpu builds include this file instead of the CPU source.
#ifndef PUFFER_AFFINE_LOCK_GPU_CU
#define PUFFER_AFFINE_LOCK_GPU_CU

#define PUF_BACKEND PUF_GPU

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Environment observations are fixed bf16. All bit observations are +/-1
// (exact in bf16); only the timer is rounded. Keeping them bf16 halves the
// rollout bandwidth compared with the CPU float representation.
typedef __nv_bfloat16 obs_t;
#include "pufferenv.h"
#include "affine_lock_visible_targets.h"

#define BITS 16
#define TIMER_INDEX (2 * BITS)
#define OBS_SIZE (TIMER_INDEX + 1)
#define NUM_ATNS 1
#define NUM_ACTIONS 8
#define MAX_SOLUTION_DEPTH 16
#define CURRICULUM_DEPTH_COUNT 6
#define STEP_REWARD (-0.01f)
#ifndef VISIBLE_TARGET_TABLE_PATH
#define VISIBLE_TARGET_TABLE_PATH "ocean/affine_lock/generated/affine_lock_8action_visible_targets.bin"
#endif
#define ACT_SIZES {NUM_ACTIONS}
#define PUF_STEPS_PER_SEC 2

#define PERF_WEIGHTING_LINEAR 0
#define PERF_WEIGHTING_QUADRATIC 1

#ifndef AFFINE_LOCK_GPU_SHARED_OBS
#define AFFINE_LOCK_GPU_SHARED_OBS 1
#endif
#ifndef AFFINE_LOCK_GPU_SHARED_BLOCK
#define AFFINE_LOCK_GPU_SHARED_BLOCK 128
#endif
#define AFFINE_LOCK_GPU_DEPTH_LUT_SIZE (MAX_SOLUTION_DEPTH + 1)

static_assert(AFFINE_LOCK_GPU_SHARED_BLOCK >= 32 &&
    AFFINE_LOCK_GPU_SHARED_BLOCK <= 256 &&
    AFFINE_LOCK_GPU_SHARED_BLOCK % 32 == 0,
    "AFFINE_LOCK_GPU_SHARED_BLOCK must contain whole warps");

#if !AFFINE_LOCK_GPU_SHARED_OBS
#ifndef AFFINE_LOCK_GPU_LANES
#define AFFINE_LOCK_GPU_LANES 4
#endif
#define AFFINE_LOCK_GPU_BLOCK 256
static_assert(AFFINE_LOCK_GPU_LANES == 4 || AFFINE_LOCK_GPU_LANES == 8 ||
    AFFINE_LOCK_GPU_LANES == 16 || AFFINE_LOCK_GPU_LANES == 32,
    "AFFINE_LOCK_GPU_LANES must be a power-of-two subwarp");
static_assert(AFFINE_LOCK_GPU_BLOCK % AFFINE_LOCK_GPU_LANES == 0,
    "block size must contain whole environments");
#endif

struct Log {
    float perf;
    float score;
    float solve_rate;
    float max_depth_solve;
    float episode_return;
    float episode_length;
    float solve_steps;
    float timeout_rate;
    float solve_efficiency;
    float target_distance;
    float solved_target_distance;
    float d6_rate;
    float d6_solve_rate;
    float d8_rate;
    float d8_solve_rate;
    float d16_rate;
    float d16_solve_rate;
    float n;
};

static_assert(sizeof(Log) == 18 * sizeof(float),
    "trainer log reduction requires a packed float-only Log");

// The trainer only reads Env::log for a GPU backend. Runtime state is kept in
// a separate compact array so log scans do not pull state into cache and state
// updates do not stride over the relatively large log payload.
struct Env {
    Log log;
    Agent agents[1];
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng;
};

// Exactly 32 bytes: four adjacent environment records fit in one 128-byte
// transaction. The default shared-observation kernel reads one record per env.
typedef struct GpuAffineLockState {
    uint32_t rng;
    uint16_t state;
    uint16_t target;
    int step_count;
    int max_steps;
    int scramble_depth;
    int curriculum_depth;
    int target_distance;
    float episode_return;
} GpuAffineLockState;

static_assert(sizeof(GpuAffineLockState) == 32,
    "GpuAffineLockState layout is performance-sensitive");

typedef struct GpuAffineLockConfig {
    int start_depth;
    int max_depth;
    int step_grace;
    int perf_weighting;
    uint32_t depth_first[AFFINE_LOCK_GPU_DEPTH_LUT_SIZE];
    uint32_t depth_counts[AFFINE_LOCK_GPU_DEPTH_LUT_SIZE];
} GpuAffineLockConfig;

__constant__ GpuAffineLockConfig d_affine_lock_config;

static struct {
    Env* envs;
    GpuAffineLockState* states;
    uint32_t* target_pairs;
    int n;
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    cudaStream_t stream;
    GpuAffineLockConfig config;
} g_gpu;

static void gpu_affine_lock_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Affine Lock CUDA: %s failed: %s\n",
            operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

#if !AFFINE_LOCK_GPU_SHARED_OBS
static int gpu_affine_lock_grid(int threads) {
    return (threads + AFFINE_LOCK_GPU_BLOCK - 1) / AFFINE_LOCK_GPU_BLOCK;
}
#endif

__device__ __forceinline__ uint32_t gpu_affine_lock_random_mixed_u32(
        GpuAffineLockState* env) {
    env->rng = env->rng * 1664525u + 1013904223u;
    uint32_t x = env->rng;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ int gpu_affine_lock_random_bounded(
        GpuAffineLockState* env, int bound) {
    uint32_t ubound = (uint32_t)bound;
    uint32_t limit = UINT32_MAX - UINT32_MAX % ubound;
    uint32_t value = gpu_affine_lock_random_mixed_u32(env);
    while (value >= limit) {
        value = gpu_affine_lock_random_mixed_u32(env);
    }
    return (int)(value % ubound);
}

__device__ __forceinline__ void gpu_affine_lock_reset_state(
        GpuAffineLockState* env, const uint32_t* target_pairs) {
    env->scramble_depth = env->curriculum_depth;
    env->step_count = 0;
    env->episode_return = 0.0f;
    int depth = env->scramble_depth;
    uint32_t count = d_affine_lock_config.depth_counts[depth];
    int choice = gpu_affine_lock_random_bounded(env, (int)count);
    uint32_t record_index = d_affine_lock_config.depth_first[depth]
        + (uint32_t)choice;
    uint32_t pair = target_pairs[record_index];
    env->state = (uint16_t)(pair & 0xffffu);
    env->target = (uint16_t)(pair >> 16);
    env->target_distance = env->scramble_depth;
    env->max_steps = env->target_distance + d_affine_lock_config.step_grace;
}

__device__ __forceinline__ uint16_t gpu_affine_lock_apply_action(
        uint16_t state, int action) {
    uint32_t next = state;
    switch (action) {
        case 0:
            next = (state >> 1) | ((state & 1u) << 15);
            break;
        case 1:
            next = ((state << 1) & 0xffffu) | ((state >> 15) & 1u);
            break;
        case 2:
            next = state ^ 0xfe00u;
            break;
        case 3:
            next = ((state & 0x5555u) << 1) | ((state & 0xaaaau) >> 1);
            break;
        case 4:
            next = ((state & 0x3333u) << 2) | ((state & 0xccccu) >> 2);
            break;
        case 5:
            next = ((state & 0x0f0fu) << 4) | ((state & 0xf0f0u) >> 4);
            break;
        case 6:
            next = ((state & 0x5555u) << 1) | ((state & 0xaaaau) >> 1);
            next = ((next & 0x3333u) << 2) | ((next & 0xccccu) >> 2);
            break;
        case 7:
            next = ((state & 0x5555u) << 1) | ((state & 0xaaaau) >> 1);
            next = ((next & 0x3333u) << 2) | ((next & 0xccccu) >> 2);
            next = ((next & 0x0f0fu) << 4) | ((next & 0xf0f0u) >> 4);
            break;
    }
    return (uint16_t)(next & 0xffffu);
}

__device__ __forceinline__ int gpu_affine_lock_next_curriculum_depth(
        int current_depth) {
    constexpr int curriculum_depths[CURRICULUM_DEPTH_COUNT] = {2, 4, 5, 6, 8, 16};
#pragma unroll
    for (int i = 0; i < CURRICULUM_DEPTH_COUNT; i++) {
        int depth = curriculum_depths[i];
        if (depth > current_depth) {
            return depth < d_affine_lock_config.max_depth
                ? depth : d_affine_lock_config.max_depth;
        }
    }
    return d_affine_lock_config.max_depth;
}

__device__ __forceinline__ void gpu_affine_lock_add_log(
        Env* trainer_env, const GpuAffineLockState* env, int solved) {
    int log_depth = env->target_distance;
    int at_max_depth = log_depth == d_affine_lock_config.max_depth;
    float ratio = log_depth / (float)d_affine_lock_config.max_depth;
    float solve_credit = 0.0f;
    if (solved) {
        solve_credit = d_affine_lock_config.perf_weighting == PERF_WEIGHTING_QUADRATIC
            ? ratio * ratio : ratio;
    }
    Log* log = &trainer_env->log;
    log->perf += solve_credit;
    log->score += solve_credit;
    log->solve_rate += solved;
    log->max_depth_solve += solved && at_max_depth;
    log->episode_return += env->episode_return;
    log->episode_length += env->step_count;
    log->solve_steps += solved ? env->step_count : 0;
    log->timeout_rate += !solved;
    log->solve_efficiency += solved
        ? env->step_count / (float)log_depth : 0.0f;
    log->target_distance += env->target_distance;
    log->solved_target_distance += solved ? env->target_distance : 0;
    log->d6_rate += log_depth == 6;
    log->d6_solve_rate += solved && log_depth == 6;
    log->d8_rate += log_depth == 8;
    log->d8_solve_rate += solved && log_depth == 8;
    log->d16_rate += log_depth == 16;
    log->d16_solve_rate += solved && log_depth == 16;
    log->n += 1;
}

__device__ __forceinline__ uint32_t gpu_affine_lock_step_one(
        Env* trainer_env, GpuAffineLockState* env,
        const uint32_t* target_pairs, float action,
        float* reward_out, float* terminal_out, float* timer_out) {
    float reward = STEP_REWARD;
    float terminal = 0.0f;
    int solved = 0;
    env->step_count += 1;
    int invalid = !isfinite(action) || action < 0.0f || action > NUM_ACTIONS - 1;
    if (invalid) {
        reward = -1.0f;
        terminal = 1.0f;
    } else {
        env->state = gpu_affine_lock_apply_action(env->state, (int)action);
        if (env->state == env->target) {
            reward = 1.0f;
            terminal = 1.0f;
            solved = 1;
        } else if (env->step_count >= env->max_steps) {
            reward = -1.0f;
            terminal = 1.0f;
        }
    }
    env->episode_return += reward;
    if (terminal != 0.0f) {
        gpu_affine_lock_add_log(trainer_env, env, solved);
        env->curriculum_depth = solved
            ? gpu_affine_lock_next_curriculum_depth(env->scramble_depth)
            : d_affine_lock_config.start_depth;
        gpu_affine_lock_reset_state(env, target_pairs);
    }
    *reward_out = reward;
    *terminal_out = terminal;
    *timer_out = env->step_count / (float)env->max_steps;
    return (uint32_t)env->state | ((uint32_t)env->target << 16);
}

#if !AFFINE_LOCK_GPU_SHARED_OBS
__device__ __forceinline__ void gpu_affine_lock_write_observations(
        obs_t* observations, uint32_t packed_bits, float timer, int lane) {
#pragma unroll
    for (int bit = lane; bit < 2 * BITS; bit += AFFINE_LOCK_GPU_LANES) {
        observations[bit] = __float2bfloat16(
            (packed_bits & (1u << bit)) ? 1.0f : -1.0f);
    }
    if (lane == 0) {
        observations[TIMER_INDEX] = __float2bfloat16(timer);
    }
}

__global__ __launch_bounds__(AFFINE_LOCK_GPU_BLOCK)
void gpu_affine_lock_reset_kernel(Env* envs, GpuAffineLockState* states,
        const uint32_t* target_pairs, obs_t* observations,
        float* rewards, float* terminals, int num_envs) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int relative_env = thread / AFFINE_LOCK_GPU_LANES;
    int lane = thread & (AFFINE_LOCK_GPU_LANES - 1);
    int active = relative_env < num_envs;
    uint32_t packed_bits = 0;
    float timer = 0.0f;
    if (active && lane == 0) {
        GpuAffineLockState* env = &states[relative_env];
        gpu_affine_lock_reset_state(env, target_pairs);
        rewards[relative_env] = 0.0f;
        terminals[relative_env] = 0.0f;
        packed_bits = (uint32_t)env->state | ((uint32_t)env->target << 16);
    }
    int leader = (threadIdx.x & 31) & ~(AFFINE_LOCK_GPU_LANES - 1);
    packed_bits = __shfl_sync(0xffffffffu, packed_bits, leader);
    timer = __shfl_sync(0xffffffffu, timer, leader);
    if (active) {
        gpu_affine_lock_write_observations(
            observations + (size_t)relative_env * OBS_SIZE,
            packed_bits, timer, lane);
    }
    (void)envs;
}

__global__ __launch_bounds__(AFFINE_LOCK_GPU_BLOCK)
void gpu_affine_lock_step_kernel(Env* envs, GpuAffineLockState* states,
        const uint32_t* target_pairs, const float* actions,
        obs_t* observations, float* rewards, float* terminals,
        int num_envs) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int relative_env = thread / AFFINE_LOCK_GPU_LANES;
    int lane = thread & (AFFINE_LOCK_GPU_LANES - 1);
    int active = relative_env < num_envs;
    uint32_t packed_bits = 0;
    float timer = 0.0f;
    if (active && lane == 0) {
        packed_bits = gpu_affine_lock_step_one(
            &envs[relative_env], &states[relative_env], target_pairs,
            actions[(size_t)relative_env * NUM_ATNS],
            &rewards[relative_env], &terminals[relative_env], &timer);
    }
    int leader = (threadIdx.x & 31) & ~(AFFINE_LOCK_GPU_LANES - 1);
    packed_bits = __shfl_sync(0xffffffffu, packed_bits, leader);
    timer = __shfl_sync(0xffffffffu, timer, leader);
    if (active) {
        gpu_affine_lock_write_observations(
            observations + (size_t)relative_env * OBS_SIZE,
            packed_bits, timer, lane);
    }
}
#endif

#if AFFINE_LOCK_GPU_SHARED_OBS
// One simulation thread per environment writes
// a conflict-free 33-float shared-memory row, then the whole block converts and
// stores a linear bf16 tile with fully coalesced global writes.
__global__ __launch_bounds__(AFFINE_LOCK_GPU_SHARED_BLOCK)
void gpu_affine_lock_shared_reset_kernel(Env* envs,
        GpuAffineLockState* states, const uint32_t* target_pairs,
        obs_t* observations, float* rewards, float* terminals, int num_envs) {
    __shared__ float observation_tile[AFFINE_LOCK_GPU_SHARED_BLOCK * OBS_SIZE];
    int block_start = blockIdx.x * AFFINE_LOCK_GPU_SHARED_BLOCK;
    int relative_env = block_start + threadIdx.x;
    int active_count = num_envs - block_start;
    if (active_count > AFFINE_LOCK_GPU_SHARED_BLOCK) {
        active_count = AFFINE_LOCK_GPU_SHARED_BLOCK;
    }
    if (active_count < 0) {
        active_count = 0;
    }
    if (threadIdx.x < active_count) {
        GpuAffineLockState* env = &states[relative_env];
        gpu_affine_lock_reset_state(env, target_pairs);
        rewards[relative_env] = 0.0f;
        terminals[relative_env] = 0.0f;
        uint32_t bits = (uint32_t)env->state | ((uint32_t)env->target << 16);
        float* row = observation_tile + threadIdx.x * OBS_SIZE;
#pragma unroll
        for (int bit = 0; bit < 2 * BITS; bit++) {
            row[bit] = (bits & (1u << bit)) ? 1.0f : -1.0f;
        }
        row[TIMER_INDEX] = 0.0f;
    }
    __syncthreads();
    int tile_values = active_count * OBS_SIZE;
    for (int value = threadIdx.x; value < tile_values; value += blockDim.x) {
        observations[(size_t)block_start * OBS_SIZE + value] =
            __float2bfloat16(observation_tile[value]);
    }
    (void)envs;
}

__global__ __launch_bounds__(AFFINE_LOCK_GPU_SHARED_BLOCK)
void gpu_affine_lock_shared_step_kernel(Env* envs,
        GpuAffineLockState* states, const uint32_t* target_pairs,
        const float* actions, obs_t* observations,
        float* rewards, float* terminals, int num_envs) {
    __shared__ float observation_tile[AFFINE_LOCK_GPU_SHARED_BLOCK * OBS_SIZE];
    int block_start = blockIdx.x * AFFINE_LOCK_GPU_SHARED_BLOCK;
    int relative_env = block_start + threadIdx.x;
    int active_count = num_envs - block_start;
    if (active_count > AFFINE_LOCK_GPU_SHARED_BLOCK) {
        active_count = AFFINE_LOCK_GPU_SHARED_BLOCK;
    }
    if (active_count < 0) {
        active_count = 0;
    }
    if (threadIdx.x < active_count) {
        float timer = 0.0f;
        uint32_t bits = gpu_affine_lock_step_one(
            &envs[relative_env], &states[relative_env], target_pairs,
            actions[(size_t)relative_env * NUM_ATNS],
            &rewards[relative_env], &terminals[relative_env], &timer);
        float* row = observation_tile + threadIdx.x * OBS_SIZE;
#pragma unroll
        for (int bit = 0; bit < 2 * BITS; bit++) {
            row[bit] = (bits & (1u << bit)) ? 1.0f : -1.0f;
        }
        row[TIMER_INDEX] = timer;
    }
    __syncthreads();
    int tile_values = active_count * OBS_SIZE;
    for (int value = threadIdx.x; value < tile_values; value += blockDim.x) {
        observations[(size_t)block_start * OBS_SIZE + value] =
            __float2bfloat16(observation_tile[value]);
    }
}
#endif

void puf_log(Log* log, Dict* out) {
    float nsolve = log->solve_rate;
    float solved_min_win_moves = nsolve
        ? log->solved_target_distance / nsolve : 0;
    float conditional_solve_steps = nsolve ? log->solve_steps / nsolve : 0;
    float conditional_solve_efficiency = nsolve
        ? log->solve_efficiency / nsolve : 0;

    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "solve_rate", log->solve_rate);
    dict_set(out, "max_depth_solve", log->max_depth_solve);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "timeout_rate", log->timeout_rate);
    dict_set(out, "min_win_moves", log->target_distance);
    dict_set(out, "solved_min_win_moves", solved_min_win_moves);
    dict_set(out, "conditional_solve_steps", conditional_solve_steps);
    dict_set(out, "conditional_solve_efficiency", conditional_solve_efficiency);
    dict_set(out, "d6_solve_rate", log->d6_rate
        ? log->d6_solve_rate / log->d6_rate : 0);
    dict_set(out, "d8_solve_rate", log->d8_rate
        ? log->d8_solve_rate / log->d8_rate : 0);
    dict_set(out, "d16_solve_rate", log->d16_rate
        ? log->d16_solve_rate / log->d16_rate : 0);
    dict_set(out, "n", log->n);
}

static int gpu_affine_lock_host_has_depth(
        const GpuAffineLockConfig* config, int depth) {
    return depth >= 0 && depth < AFFINE_LOCK_GPU_DEPTH_LUT_SIZE
        && config->depth_counts[depth] != 0;
}

static int gpu_affine_lock_host_next_depth(int current_depth, int max_depth) {
    static const int curriculum_depths[CURRICULUM_DEPTH_COUNT] = {2, 4, 5, 6, 8, 16};
    for (int i = 0; i < CURRICULUM_DEPTH_COUNT; i++) {
        int depth = curriculum_depths[i];
        if (depth > current_depth) {
            return depth < max_depth ? depth : max_depth;
        }
    }
    return max_depth;
}

static void gpu_affine_lock_validate_curriculum(
        const GpuAffineLockConfig* config) {
    if (config->start_depth <= 0 ||
            config->max_depth < config->start_depth ||
            config->max_depth > MAX_SOLUTION_DEPTH) {
        std::fprintf(stderr,
            "Affine Lock CUDA: invalid curriculum range start=%d max=%d\n",
            config->start_depth, config->max_depth);
        std::exit(1);
    }
    int depth = config->start_depth;
    for (int i = 0; i <= CURRICULUM_DEPTH_COUNT; i++) {
        if (!gpu_affine_lock_host_has_depth(config, depth)) {
            std::fprintf(stderr,
                "Affine Lock CUDA: target table has no depth %d section\n", depth);
            std::exit(1);
        }
        if (depth + config->step_grace <= 0) {
            std::fprintf(stderr,
                "Affine Lock CUDA: depth %d with step_grace=%d has no valid steps\n",
                depth, config->step_grace);
            std::exit(1);
        }
        if (depth == config->max_depth) {
            return;
        }
        int next = gpu_affine_lock_host_next_depth(depth, config->max_depth);
        if (next == depth) {
            break;
        }
        depth = next;
    }
    std::fprintf(stderr, "Affine Lock CUDA: curriculum does not reach max depth %d\n",
        config->max_depth);
    std::exit(1);
}

Env* puf_vec_create(int n, Dict* env_kwargs,
        obs_t* observations, float* actions,
        float* rewards, float* terminals) {
    if (n <= 0) {
        std::fprintf(stderr, "Affine Lock CUDA: vector size must be positive\n");
        std::exit(1);
    }
    if (g_gpu.envs != nullptr) {
        std::fprintf(stderr, "Affine Lock CUDA: vector already exists\n");
        std::exit(1);
    }

    VisibleTargetTable table = {};
    if (visible_targets_load(VISIBLE_TARGET_TABLE_PATH,
            VISIBLE_TARGET_8ACTION_V1_HASH, &table) != 0) {
        std::fprintf(stderr,
            "Affine Lock CUDA: failed to load visible target table %s\n",
            VISIBLE_TARGET_TABLE_PATH);
        std::exit(1);
    }
    if (table.num_actions != NUM_ACTIONS) {
        std::fprintf(stderr,
            "Affine Lock CUDA: target table has %u actions, expected %d\n",
            table.num_actions, NUM_ACTIONS);
        visible_targets_free(&table);
        std::exit(1);
    }

    GpuAffineLockConfig config = {};
    config.start_depth = (int)dict_get(env_kwargs, "start_depth");
    config.max_depth = (int)dict_get(env_kwargs, "max_depth");
    config.step_grace = (int)dict_get(env_kwargs, "step_grace");
    config.perf_weighting = (int)dict_get(env_kwargs, "perf_weighting");
    for (uint32_t i = 0; i < table.depth_count; i++) {
        const VisibleTargetDepth* depth = &table.depths[i];
        if (depth->depth >= AFFINE_LOCK_GPU_DEPTH_LUT_SIZE ||
                depth->stored_count == 0 ||
                config.depth_counts[depth->depth] != 0) {
            std::fprintf(stderr,
                "Affine Lock CUDA: invalid target-table depth section %u\n",
                depth->depth);
            visible_targets_free(&table);
            std::exit(1);
        }
        config.depth_first[depth->depth] = depth->first_record;
        config.depth_counts[depth->depth] = depth->stored_count;
        for (uint32_t record_offset = 0;
                record_offset < depth->stored_count; record_offset++) {
            const VisibleTargetRecord* record =
                &table.records[depth->first_record + record_offset];
            if (record->depth != depth->depth) {
                std::fprintf(stderr,
                    "Affine Lock CUDA: record depth %u does not match section %u\n",
                    (unsigned int)record->depth, depth->depth);
                visible_targets_free(&table);
                std::exit(1);
            }
        }
    }
    gpu_affine_lock_validate_curriculum(&config);

    uint32_t* host_pairs = (uint32_t*)std::malloc(
        (size_t)table.record_count * sizeof(uint32_t));
    if (host_pairs == nullptr) {
        std::perror("malloc");
        visible_targets_free(&table);
        std::exit(1);
    }
    for (uint32_t i = 0; i < table.record_count; i++) {
        host_pairs[i] = (uint32_t)table.records[i].start
            | ((uint32_t)table.records[i].target << 16);
    }

    GpuAffineLockState* host_states = (GpuAffineLockState*)std::calloc(
        (size_t)n, sizeof(GpuAffineLockState));
    if (host_states == nullptr) {
        std::perror("calloc");
        std::free(host_pairs);
        visible_targets_free(&table);
        std::exit(1);
    }
    unsigned int running_seed = (unsigned int)dict_get(env_kwargs, "seed");
    for (int i = 0; i < n; i++) {
        host_states[i].rng = (uint32_t)rand_r(&running_seed);
        host_states[i].curriculum_depth = config.start_depth;
    }

    Env* device_envs = nullptr;
    GpuAffineLockState* device_states = nullptr;
    uint32_t* device_pairs = nullptr;
    gpu_affine_lock_check(cudaMalloc((void**)&device_envs,
        (size_t)n * sizeof(Env)), "cudaMalloc envs");
    gpu_affine_lock_check(cudaMalloc((void**)&device_states,
        (size_t)n * sizeof(GpuAffineLockState)), "cudaMalloc states");
    gpu_affine_lock_check(cudaMalloc((void**)&device_pairs,
        (size_t)table.record_count * sizeof(uint32_t)), "cudaMalloc target pairs");
    gpu_affine_lock_check(cudaMemset(device_envs, 0,
        (size_t)n * sizeof(Env)), "clear env logs");
    gpu_affine_lock_check(cudaMemcpy(device_states, host_states,
        (size_t)n * sizeof(GpuAffineLockState), cudaMemcpyHostToDevice), "copy states");
    gpu_affine_lock_check(cudaMemcpy(device_pairs, host_pairs,
        (size_t)table.record_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
        "copy target pairs");
    gpu_affine_lock_check(cudaMemcpyToSymbol(d_affine_lock_config,
        &config, sizeof(config)), "copy config");

    std::free(host_pairs);
    std::free(host_states);
    visible_targets_free(&table);

    g_gpu.envs = device_envs;
    g_gpu.states = device_states;
    g_gpu.target_pairs = device_pairs;
    g_gpu.n = n;
    g_gpu.observations = observations;
    g_gpu.actions = actions;
    g_gpu.rewards = rewards;
    g_gpu.terminals = terminals;
    g_gpu.stream = nullptr;
    g_gpu.config = config;
    return device_envs;
}

void puf_bind_stream(cudaStream_t stream) {
    g_gpu.stream = stream;
}

// GPU creation is vector-only; puf_init exists to satisfy the common API.
void puf_init(Env* env, Dict* kwargs) {
    (void)env;
    (void)kwargs;
}

void puf_reset(Env* env) {
    (void)env;
#if AFFINE_LOCK_GPU_SHARED_OBS
    int blocks = (g_gpu.n + AFFINE_LOCK_GPU_SHARED_BLOCK - 1)
        / AFFINE_LOCK_GPU_SHARED_BLOCK;
    gpu_affine_lock_shared_reset_kernel<<<
        blocks, AFFINE_LOCK_GPU_SHARED_BLOCK, 0, g_gpu.stream>>>(
        g_gpu.envs, g_gpu.states, g_gpu.target_pairs,
        g_gpu.observations, g_gpu.rewards, g_gpu.terminals, g_gpu.n);
#else
    int threads = g_gpu.n * AFFINE_LOCK_GPU_LANES;
    gpu_affine_lock_reset_kernel<<<
        gpu_affine_lock_grid(threads), AFFINE_LOCK_GPU_BLOCK, 0, g_gpu.stream>>>(
        g_gpu.envs, g_gpu.states, g_gpu.target_pairs,
        g_gpu.observations, g_gpu.rewards, g_gpu.terminals, g_gpu.n);
#endif
    gpu_affine_lock_check(cudaPeekAtLastError(), "launch reset kernel");
}

void puf_step(Env* env) {
    (void)env;
#if AFFINE_LOCK_GPU_SHARED_OBS
    int blocks = (g_gpu.n + AFFINE_LOCK_GPU_SHARED_BLOCK - 1)
        / AFFINE_LOCK_GPU_SHARED_BLOCK;
    gpu_affine_lock_shared_step_kernel<<<
        blocks, AFFINE_LOCK_GPU_SHARED_BLOCK, 0, g_gpu.stream>>>(
        g_gpu.envs, g_gpu.states, g_gpu.target_pairs,
        g_gpu.actions, g_gpu.observations,
        g_gpu.rewards, g_gpu.terminals, g_gpu.n);
#else
    int threads = g_gpu.n * AFFINE_LOCK_GPU_LANES;
    gpu_affine_lock_step_kernel<<<
        gpu_affine_lock_grid(threads), AFFINE_LOCK_GPU_BLOCK, 0, g_gpu.stream>>>(
        g_gpu.envs, g_gpu.states, g_gpu.target_pairs,
        g_gpu.actions, g_gpu.observations,
        g_gpu.rewards, g_gpu.terminals, g_gpu.n);
#endif
    gpu_affine_lock_check(cudaPeekAtLastError(), "launch step kernel");
}

void puf_close(Env* env) {
    (void)env;
    if (IsWindowReady()) {
        CloseWindow();
    }
    if (g_gpu.envs != nullptr) {
        gpu_affine_lock_check(cudaFree(g_gpu.envs), "cudaFree envs");
    }
    if (g_gpu.states != nullptr) {
        gpu_affine_lock_check(cudaFree(g_gpu.states), "cudaFree states");
    }
    if (g_gpu.target_pairs != nullptr) {
        gpu_affine_lock_check(cudaFree(g_gpu.target_pairs), "cudaFree target pairs");
    }
    g_gpu = {};
}

void puf_render(Env* env) {
    (void)env;
    if (g_gpu.envs == nullptr || g_gpu.n < 1) {
        return;
    }
    if (IsWindowReady() && (WindowShouldClose() || IsKeyPressed(KEY_ESCAPE))) {
        puf_close(g_gpu.envs);
        std::exit(0);
    }
    if (!IsWindowReady()) {
        InitWindow(780, 360, "PufferLib AffineLock CUDA");
        SetTargetFPS(30);
    }
    if (g_gpu.stream != nullptr) {
        gpu_affine_lock_check(cudaStreamSynchronize(g_gpu.stream),
            "synchronize render stream");
    }
    GpuAffineLockState state;
    float reward = 0.0f;
    float terminal = 0.0f;
    gpu_affine_lock_check(cudaMemcpy(&state, g_gpu.states, sizeof(state),
        cudaMemcpyDeviceToHost), "copy render state");
    gpu_affine_lock_check(cudaMemcpy(&reward, g_gpu.rewards, sizeof(reward),
        cudaMemcpyDeviceToHost), "copy render reward");
    gpu_affine_lock_check(cudaMemcpy(&terminal, g_gpu.terminals, sizeof(terminal),
        cudaMemcpyDeviceToHost), "copy render terminal");

    uint32_t mismatches = (state.state ^ state.target) & 0xffffu;
    const char* status = terminal == 0.0f
        ? "running" : (reward > 0.0f ? "solved" : "failed");
    Color status_color = terminal == 0.0f
        ? (Color){190, 198, 206, 255}
        : (reward > 0.0f
            ? (Color){80, 210, 140, 255}
            : (Color){238, 88, 88, 255});

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    DrawText("Affine Lock CUDA", 30, 24, 28, RAYWHITE);
    DrawText(TextFormat("depth %d/%d  step %d/%d  last reward %.2f",
        state.scramble_depth, g_gpu.config.max_depth,
        state.step_count, state.max_steps, reward),
        30, 62, 20, (Color){180, 190, 200, 255});
    DrawText(TextFormat("status %s  mismatches 0x%04x",
        status, mismatches), 30, 90, 20, status_color);

    const char* row_label[2] = {"current", "target"};
    uint32_t row_value[2] = {state.state, state.target};
    int row_y[2] = {138, 220};
    for (int row = 0; row < 2; row++) {
        DrawText(row_label[row], 30, row_y[row] + 9, 20, RAYWHITE);
        for (int bit = 0; bit < BITS; bit++) {
            int x = 145 + bit * 34;
            int on = (row_value[row] >> bit) & 1u;
            int mismatch = ((state.state ^ state.target) >> bit) & 1u;
            Color fill = on
                ? (Color){80, 210, 140, 255}
                : (Color){38, 48, 58, 255};
            Color border = mismatch
                ? (Color){238, 88, 88, 255}
                : (Color){182, 196, 205, 255};
            DrawRectangle(x, row_y[row], 24, 34, fill);
            DrawRectangleLinesEx((Rectangle){(float)x, (float)row_y[row], 24, 34},
                mismatch ? 3 : 1, border);
            DrawText(TextFormat("%d", bit), x + 5, row_y[row] + 40, 10,
                (Color){128, 140, 150, 255});
        }
    }
    DrawText("GPU-resident environment", 30, 310, 16,
        (Color){160, 170, 178, 255});
    EndDrawing();
    puf_web_vsync();
}

#endif
