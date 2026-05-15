#ifndef PUFFERLIB_OCEAN_CLIFFORD_CLIFFORD_H
#define PUFFERLIB_OCEAN_CLIFFORD_CLIFFORD_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define GATE_H 0
#define GATE_S 1
#define GATE_V 2
#define GATE_HS 3
#define GATE_HV 4
#define GATE_CZ 5

#define REWARD_GATE_COST 0
#define REWARD_HAMMING_LEFT 1

typedef struct {
    int gate_kind;
    int q0;
    int q1;
} CliffordAction;

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float episode_cz_sum;
    float episode_cz_max;
    float success_rate;
    float n;
    float success_count;
    float success_step_sum;
    float success_step_sq_sum;
    float success_cz_sum;
    float success_cz_max;
    float success_step_min;
    float success_step_max;
} CliffordLog;

typedef struct {
    uint64_t state;
    double next_gaussian;
    int has_gaussian;
} XorShift64;

typedef struct {
    uint64_t* cols;
    unsigned char* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    unsigned char* truncations;
    float episode_return;
    int episode_length;
    int episode_cz_count;
    int steps;
    int episode_max_steps;
    XorShift64 rng;
} CliffordEnv;

typedef struct {
    CliffordEnv* envs;
    int num_envs;
    int n_qubits;
    int dim;
    int obs_size;
    int difficulty;
    float difficulty_fraction;
    int max_steps;
    int num_actions;
    CliffordAction* actions;
    int reset_pool_enabled;
    uint64_t* reset_pool_cols;
    int reset_pool_size;
    int reset_pool_walk_steps;
    int reset_tail_steps;
    int reset_refresh_stride;
    int reset_refresh_credit;
    XorShift64 pool_rng;
    uint64_t identity_cols[64];
    float single_qubit_cost;
    float goal_bonus;
    int reward_mode;
    float hamming_left_scale;
    CliffordLog log;
} CliffordVecEnv;

static uint64_t BYTE_TO_BYTES64[256];
static uint64_t ROWMASK_ACCUM[8][256];
static int OBS_TABLES_READY = 0;

static inline uint64_t splitmix64_next(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline void rng_seed(XorShift64* rng, uint64_t seed) {
    if (seed == 0) {
        seed = 0x123456789ABCDEFULL;
    }
    rng->state = seed;
    rng->next_gaussian = 0.0;
    rng->has_gaussian = 0;
}

static inline uint64_t rng_next_u64(XorShift64* rng) {
    uint64_t x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return x * 2685821657736338717ULL;
}

static inline int rng_below(XorShift64* rng, int upper) {
    if (upper <= 1) {
        return 0;
    }
    return (int)(rng_next_u64(rng) % (uint64_t)upper);
}

static inline int popcount_u64(uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll((unsigned long long)value);
#else
    int count = 0;
    while (value != 0ULL) {
        value &= value - 1ULL;
        count += 1;
    }
    return count;
#endif
}

static inline float rng_float01(XorShift64* rng) {
    return (float)((rng_next_u64(rng) >> 40) * (1.0 / 16777216.0));
}

static inline void set_difficulty_level(CliffordVecEnv* vec, double difficulty_level) {
    if (difficulty_level < 0.0) {
        difficulty_level = 0.0;
    }
    double floor_level = floor(difficulty_level + 1e-12);
    vec->difficulty = (int)floor_level;
    vec->difficulty_fraction = (float)(difficulty_level - floor_level);
    if (vec->difficulty_fraction <= 1e-6f) {
        vec->difficulty_fraction = 0.0f;
    } else if (vec->difficulty_fraction >= 1.0f - 1e-6f) {
        vec->difficulty += 1;
        vec->difficulty_fraction = 0.0f;
    }
}

static inline int effective_reset_difficulty(const CliffordVecEnv* vec) {
    return vec->difficulty + (vec->difficulty_fraction > 0.0f ? 1 : 0);
}

static inline int sample_reset_difficulty(const CliffordVecEnv* vec, XorShift64* rng) {
    if (vec->difficulty_fraction <= 0.0f) {
        return vec->difficulty;
    }
    return vec->difficulty + (rng_float01(rng) < vec->difficulty_fraction ? 1 : 0);
}

static inline void init_observation_tables(void) {
    if (OBS_TABLES_READY) {
        return;
    }
    for (int value = 0; value < 256; ++value) {
        uint64_t expanded = 0ULL;
        for (int bit = 0; bit < 8; ++bit) {
            expanded |= (uint64_t)((value >> bit) & 1U) << (bit * 8);
        }
        BYTE_TO_BYTES64[value] = expanded;
    }
    for (int col = 0; col < 8; ++col) {
        const uint64_t row_bit = (uint64_t)(1U << col);
        for (int value = 0; value < 256; ++value) {
            uint64_t packed = 0ULL;
            for (int row = 0; row < 8; ++row) {
                if ((value >> row) & 1U) {
                    packed |= row_bit << (row * 8);
                }
            }
            ROWMASK_ACCUM[col][value] = packed;
        }
    }
    OBS_TABLES_READY = 1;
}

static inline void copy_identity_cols(const CliffordVecEnv* vec, CliffordEnv* env) {
    memcpy(env->cols, vec->identity_cols, (size_t)vec->dim * sizeof(uint64_t));
}

static inline void copy_cols(const CliffordVecEnv* vec, uint64_t* dst, const uint64_t* src) {
    memcpy(dst, src, (size_t)vec->dim * sizeof(uint64_t));
}

static inline void reset_episode_state(const CliffordVecEnv* vec, CliffordEnv* env) {
    env->steps = 0;
    env->episode_max_steps = vec->max_steps;
    env->episode_return = 0.0f;
    env->episode_length = 0;
    env->episode_cz_count = 0;
}

static inline int is_identity(const CliffordVecEnv* vec, const CliffordEnv* env) {
    for (int col = 0; col < vec->dim; ++col) {
        if (env->cols[col] != vec->identity_cols[col]) {
            return 0;
        }
    }
    return 1;
}

static inline int sample_action(const CliffordVecEnv* vec, XorShift64* rng) {
    if (vec->num_actions <= 0) {
        return -1;
    }
    return rng_below(rng, vec->num_actions);
}

static inline void apply_action(const CliffordVecEnv* vec, CliffordEnv* env, int action_idx) {
    const CliffordAction* action = &vec->actions[action_idx];
    const int n = vec->n_qubits;
    const int q = action->q0;
    const uint64_t x = env->cols[q];
    const uint64_t z = env->cols[n + q];
    if (action->gate_kind == GATE_H) {
        env->cols[q] = z;
        env->cols[n + q] = x;
    } else if (action->gate_kind == GATE_S) {
        env->cols[q] = x;
        env->cols[n + q] = z ^ x;
    } else if (action->gate_kind == GATE_V) {
        env->cols[q] = x ^ z;
        env->cols[n + q] = z;
    } else if (action->gate_kind == GATE_HS) {
        env->cols[q] = z;
        env->cols[n + q] = x ^ z;
    } else if (action->gate_kind == GATE_HV) {
        env->cols[q] = x ^ z;
        env->cols[n + q] = x;
    } else {
        env->cols[n + action->q0] ^= env->cols[action->q1];
        env->cols[n + action->q1] ^= env->cols[action->q0];
    }
}

static inline int compute_reset_tail_steps(int difficulty) {
    if (difficulty < 8) {
        return 0;
    }
    int tail = difficulty / 4;
    if (tail < 8) {
        tail = 8;
    }
    if (tail > 32) {
        tail = 32;
    }
    if (tail >= difficulty) {
        tail = difficulty / 2;
    }
    if (tail < 1) {
        tail = 1;
    }
    return tail;
}

static inline int floor_power_of_two(int value) {
    int power = 1;
    while ((power << 1) > 0 && (power << 1) <= value) {
        power <<= 1;
    }
    return power;
}

static inline int compute_reset_pool_size(const CliffordVecEnv* vec) {
    if (!vec->reset_pool_enabled) {
        return 0;
    }
    if (vec->difficulty_fraction > 0.0f) {
        return 0;
    }
    if (vec->difficulty < 8) {
        return 0;
    }
    const size_t bytes_per_entry = (size_t)vec->dim * sizeof(uint64_t);
    const size_t target_bytes = 16u << 20;
    int desired = vec->num_envs * 8;
    if (desired < 2048) {
        desired = 2048;
    }
    if (desired > 32768) {
        desired = 32768;
    }
    if (bytes_per_entry > 0) {
        const size_t cap_by_memory = target_bytes / bytes_per_entry;
        if (cap_by_memory > 0 && (size_t)desired > cap_by_memory) {
            desired = (int)cap_by_memory;
        }
    }
    if (desired < 256) {
        desired = 256;
    }
    return floor_power_of_two(desired);
}

static inline void generate_random_walk_cols(
    const CliffordVecEnv* vec,
    XorShift64* rng,
    int walk_steps,
    uint64_t* out_cols
) {
    CliffordEnv env = {
        .cols = out_cols,
    };
    if (walk_steps <= 0) {
        copy_identity_cols(vec, &env);
        return;
    }
    do {
        copy_identity_cols(vec, &env);
        for (int step = 0; step < walk_steps; ++step) {
            const int action_idx = sample_action(vec, rng);
            if (action_idx < 0) {
                break;
            }
            apply_action(vec, &env, action_idx);
        }
    } while (is_identity(vec, &env));
}

static inline void rebuild_reset_pool(CliffordVecEnv* vec, uint64_t seed) {
    free(vec->reset_pool_cols);
    vec->reset_pool_cols = NULL;
    const int pool_difficulty = effective_reset_difficulty(vec);
    if (!vec->reset_pool_enabled) {
        vec->reset_pool_size = 0;
        vec->reset_pool_walk_steps = 0;
        vec->reset_tail_steps = 0;
        vec->reset_refresh_stride = 0;
        vec->reset_refresh_credit = 0;
        rng_seed(&vec->pool_rng, seed ^ 0x9E3779B97F4A7C15ULL ^ (uint64_t)pool_difficulty);
        return;
    }
    vec->reset_pool_size = compute_reset_pool_size(vec);
    vec->reset_tail_steps = compute_reset_tail_steps(pool_difficulty);
    vec->reset_pool_walk_steps = pool_difficulty - vec->reset_tail_steps;
    vec->reset_refresh_stride = 32;
    vec->reset_refresh_credit = 0;
    rng_seed(&vec->pool_rng, seed ^ 0x9E3779B97F4A7C15ULL ^ (uint64_t)pool_difficulty);
    if (vec->reset_pool_size <= 0 || vec->reset_pool_walk_steps <= 0) {
        vec->reset_pool_size = 0;
        vec->reset_pool_walk_steps = 0;
        vec->reset_tail_steps = 0;
        return;
    }
    vec->reset_pool_cols = (uint64_t*)calloc(
        (size_t)vec->reset_pool_size * (size_t)vec->dim,
        sizeof(uint64_t)
    );
    if (vec->reset_pool_cols == NULL) {
        vec->reset_pool_size = 0;
        vec->reset_pool_walk_steps = 0;
        vec->reset_tail_steps = 0;
        return;
    }
    for (int slot = 0; slot < vec->reset_pool_size; ++slot) {
        generate_random_walk_cols(
            vec,
            &vec->pool_rng,
            vec->reset_pool_walk_steps,
            vec->reset_pool_cols + ((size_t)slot * (size_t)vec->dim)
        );
    }
}

static inline void maybe_refresh_reset_pool(CliffordVecEnv* vec) {
    if (vec->reset_pool_cols == NULL || vec->reset_pool_size <= 0 || vec->reset_pool_walk_steps <= 0) {
        return;
    }
    vec->reset_refresh_credit += 1;
    if (vec->reset_refresh_credit < vec->reset_refresh_stride) {
        return;
    }
    vec->reset_refresh_credit -= vec->reset_refresh_stride;
    const int slot = (int)(rng_next_u64(&vec->pool_rng) & (uint64_t)(vec->reset_pool_size - 1));
    generate_random_walk_cols(
        vec,
        &vec->pool_rng,
        vec->reset_pool_walk_steps,
        vec->reset_pool_cols + ((size_t)slot * (size_t)vec->dim)
    );
}

static inline void write_observation(const CliffordVecEnv* vec, CliffordEnv* env) {
    const int dim = vec->dim;
    int row_block = 0;
    for (; row_block + 8 <= dim; row_block += 8) {
        int col_block = 0;
        for (; col_block + 8 <= dim; col_block += 8) {
            uint64_t row_masks = 0ULL;
            for (int col = 0; col < 8; ++col) {
                const uint8_t column_chunk = (uint8_t)((env->cols[col_block + col] >> row_block) & 0xFFU);
                row_masks |= ROWMASK_ACCUM[col][column_chunk];
            }
            for (int row = 0; row < 8; ++row) {
                const uint8_t row_mask = (uint8_t)((row_masks >> (row * 8)) & 0xFFU);
                const uint64_t expanded = BYTE_TO_BYTES64[row_mask];
                memcpy(env->observations + ((row_block + row) * dim) + col_block, &expanded, 8);
            }
        }
        for (int row = 0; row < 8; ++row) {
            const int dst_row = row_block + row;
            for (int col = col_block; col < dim; ++col) {
                env->observations[dst_row * dim + col] = (unsigned char)((env->cols[col] >> dst_row) & 1ULL);
            }
        }
    }
    for (int row = row_block; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            env->observations[row * dim + col] = (unsigned char)((env->cols[col] >> row) & 1ULL);
        }
    }
}

static inline void reset_single(CliffordVecEnv* vec, CliffordEnv* env) {
    reset_episode_state(vec, env);
    const int reset_difficulty = sample_reset_difficulty(vec, &env->rng);
    if (reset_difficulty <= 0) {
        copy_identity_cols(vec, env);
        write_observation(vec, env);
        return;
    }

    if (vec->reset_pool_cols != NULL && vec->reset_pool_size > 0) {
        do {
            maybe_refresh_reset_pool(vec);
            const int slot = (int)(rng_next_u64(&env->rng) & (uint64_t)(vec->reset_pool_size - 1));
            copy_cols(
                vec,
                env->cols,
                vec->reset_pool_cols + ((size_t)slot * (size_t)vec->dim)
            );
            for (int step = 0; step < vec->reset_tail_steps; ++step) {
                const int action_idx = sample_action(vec, &env->rng);
                if (action_idx < 0) {
                    break;
                }
                apply_action(vec, env, action_idx);
            }
        } while (is_identity(vec, env));
        write_observation(vec, env);
        return;
    }

    do {
        copy_identity_cols(vec, env);
        for (int step = 0; step < reset_difficulty; ++step) {
            const int action_idx = sample_action(vec, &env->rng);
            if (action_idx < 0) {
                break;
            }
            apply_action(vec, env, action_idx);
        }
    } while (is_identity(vec, env));

    write_observation(vec, env);
}

static inline int set_matrix_from_dense(CliffordVecEnv* vec, CliffordEnv* env, const unsigned char* data, int rows, int cols) {
    if (rows != vec->dim || cols != vec->dim) {
        return 0;
    }
    for (int col = 0; col < vec->dim; ++col) {
        uint64_t packed = 0ULL;
        for (int row = 0; row < vec->dim; ++row) {
            packed |= ((uint64_t)(data[row * cols + col] & 1U)) << row;
        }
        env->cols[col] = packed;
    }
    reset_episode_state(vec, env);
    write_observation(vec, env);
    return 1;
}

static inline int identity_hamming_distance(const CliffordVecEnv* vec, const CliffordEnv* env) {
    int distance = 0;
    for (int col = 0; col < vec->dim; ++col) {
        distance += popcount_u64(env->cols[col] ^ vec->identity_cols[col]);
    }
    return distance;
}

static inline float normalized_identity_hamming_distance(const CliffordVecEnv* vec, const CliffordEnv* env) {
    return (float)identity_hamming_distance(vec, env) / (float)vec->obs_size;
}

static inline void add_log(CliffordVecEnv* vec, CliffordEnv* env, int success) {
    vec->log.perf += success ? 1.0f : 0.0f;
    vec->log.score += env->episode_return;
    vec->log.episode_return += env->episode_return;
    vec->log.episode_length += (float)env->episode_length;
    vec->log.episode_cz_sum += (float)env->episode_cz_count;
    if ((float)env->episode_cz_count > vec->log.episode_cz_max) {
        vec->log.episode_cz_max = (float)env->episode_cz_count;
    }
    vec->log.success_rate += success ? 1.0f : 0.0f;
    vec->log.n += 1.0f;
    if (success) {
        vec->log.success_count += 1.0f;
        vec->log.success_step_sum += (float)env->episode_length;
        vec->log.success_step_sq_sum += (float)(env->episode_length * env->episode_length);
        vec->log.success_cz_sum += (float)env->episode_cz_count;
        if ((float)env->episode_cz_count > vec->log.success_cz_max) {
            vec->log.success_cz_max = (float)env->episode_cz_count;
        }
        if (vec->log.success_count <= 1.0f) {
            vec->log.success_step_min = (float)env->episode_length;
            vec->log.success_step_max = (float)env->episode_length;
        } else {
            if ((float)env->episode_length < vec->log.success_step_min) {
                vec->log.success_step_min = (float)env->episode_length;
            }
            if ((float)env->episode_length > vec->log.success_step_max) {
                vec->log.success_step_max = (float)env->episode_length;
            }
        }
    }
}

static inline float gate_cost_reward(const CliffordVecEnv* vec, int gate_kind) {
    return gate_kind == GATE_CZ ? -1.0f : -vec->single_qubit_cost;
}

static inline float step_single(CliffordVecEnv* vec, CliffordEnv* env) {
    int action_idx = env->actions[0] % vec->num_actions;
    if (action_idx < 0) {
        action_idx += vec->num_actions;
    }
    const CliffordAction* action = &vec->actions[action_idx];
    const int gate_kind = action->gate_kind;
    apply_action(vec, env, action_idx);
    env->steps += 1;
    env->episode_length += 1;
    if (gate_kind == GATE_CZ) {
        env->episode_cz_count += 1;
    }

    int terminated = is_identity(vec, env);
    int truncated = (!terminated && env->steps >= env->episode_max_steps);
    float reward = gate_cost_reward(vec, gate_kind);
    if (vec->reward_mode == REWARD_HAMMING_LEFT) {
        reward -= vec->hamming_left_scale * normalized_identity_hamming_distance(vec, env);
    }
    if (terminated) {
        reward += vec->goal_bonus;
    }
    env->episode_return += reward;
    env->rewards[0] = reward;
    env->terminals[0] = (unsigned char)terminated;
    env->truncations[0] = (unsigned char)truncated;

    if (terminated || truncated) {
        add_log(vec, env, terminated);
        reset_single(vec, env);
    } else {
        write_observation(vec, env);
    }
    return reward;
}

#endif
