#include "affine_lock.h"

#define OBS_SIZE AFFINE_LOCK_OBS_SIZE
#define NUM_ATNS AFFINE_LOCK_NUM_ATNS
#define ACT_SIZES {AFFINE_LOCK_NUM_ACTIONS}
#define OBS_TENSOR_T FloatTensor

#define MY_VEC_INIT
#define MY_VEC_CLOSE
#define Env AffineLock
#include "vecenv.h"

static uint32_t affine_lock_mix_seed(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

static unsigned int affine_lock_env_seed(int base_seed, int env_id) {
    uint32_t value = 0x811c9dc5u;
    value = (value ^ (uint32_t)base_seed) * 0x01000193u;
    value = (value ^ (uint32_t)env_id) * 0x01000193u;
    return affine_lock_mix_seed(value);
}

Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int agents_per_buffer = total_agents / num_buffers;
    int base_seed = (int)dict_get(env_kwargs, "seed")->value;

    int start_depth = (int)dict_get(env_kwargs, "start_depth")->value;
    int max_depth = (int)dict_get(env_kwargs, "max_depth")->value;
    int depth_multiplier = (int)dict_get(env_kwargs, "depth_multiplier")->value;
    int step_grace = (int)dict_get(env_kwargs, "step_grace")->value;
    int initialization_mode =
        (int)dict_get(env_kwargs, "initialization_mode")->value;

    AffineLockShared* shared =
        (AffineLockShared*)calloc(1, sizeof(AffineLockShared));
    if (shared == NULL || affine_lock_init_shared(
            shared, start_depth, max_depth, depth_multiplier, step_grace) != 0) {
        fprintf(stderr, "affine_lock: failed to initialize shared state\n");
        free(shared);
        abort();
    }
    if (affine_lock_configure_initialization(shared, initialization_mode) != 0) {
        affine_lock_free_shared(shared);
        free(shared);
        abort();
    }

    Env* envs = (Env*)calloc((size_t)total_agents, sizeof(Env));
    if (envs == NULL) {
        fprintf(stderr, "affine_lock: failed to allocate envs\n");
        affine_lock_free_shared(shared);
        free(shared);
        abort();
    }

    int buf = 0;
    int buf_agents = 0;
    buffer_env_starts[0] = 0;
    buffer_env_counts[0] = 0;

    for (int i = 0; i < total_agents; i++) {
        Env* env = &envs[i];
        affine_lock_init_env(env, shared, affine_lock_env_seed(base_seed, i));

        buf_agents += env->num_agents;
        buffer_env_counts[buf]++;
        if (buf_agents >= agents_per_buffer && buf < num_buffers - 1) {
            buf++;
            buffer_env_starts[buf] = i + 1;
            buffer_env_counts[buf] = 0;
            buf_agents = 0;
        }
    }

    *num_envs_out = total_agents;
    return envs;
}

void my_vec_close(Env* envs) {
    if (envs == NULL || envs[0].shared == NULL) {
        return;
    }
    AffineLockShared* shared = envs[0].shared;
    affine_lock_free_shared(shared);
    free(shared);
}

void my_init(Env* env, Dict* kwargs) {
    (void)env;
    (void)kwargs;
}

static float conditional_rate(float numerator, float denominator) {
    return denominator > 0.0f ? numerator / denominator : 0.0f;
}

void my_log(Log* log, Dict* out) {
    float conditional_solve_steps =
        log->solve_rate > 0.0f ? log->solve_steps / log->solve_rate : 0.0f;
    float conditional_solve_efficiency =
        log->solve_rate > 0.0f ?
            log->solve_efficiency / log->solve_rate : 0.0f;
    float solved_min_win_moves =
        log->solve_rate > 0.0f ?
            log->solved_target_distance / log->solve_rate : 0.0f;
    float depth_2_solve_rate =
        conditional_rate(log->depth_2_solve_rate, log->depth_2_rate);
    float depth_4_solve_rate =
        conditional_rate(log->depth_4_solve_rate, log->depth_4_rate);
    float depth_8_solve_rate =
        conditional_rate(log->depth_8_solve_rate, log->depth_8_rate);
    float depth_16_solve_rate =
        conditional_rate(log->depth_16_solve_rate, log->depth_16_rate);

    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "solve_rate", log->solve_rate);
    dict_set(out, "max_depth_solve", log->max_depth_solve);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "timeout_rate", log->timeout_rate);
    dict_set(out, "invalid_rate", log->invalid_rate);
    dict_set(out, "min_win_moves", log->target_distance);
    dict_set(out, "solved_min_win_moves", solved_min_win_moves);
    dict_set(out, "conditional_solve_steps", conditional_solve_steps);
    dict_set(out, "conditional_solve_efficiency", conditional_solve_efficiency);
    dict_set(out, "depth_2_solve_rate", depth_2_solve_rate);
    dict_set(out, "depth_4_solve_rate", depth_4_solve_rate);
    dict_set(out, "depth_8_solve_rate", depth_8_solve_rate);
    dict_set(out, "depth_16_solve_rate", depth_16_solve_rate);
}
