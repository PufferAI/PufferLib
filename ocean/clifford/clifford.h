#ifndef PUFFERLIB_OCEAN_CLIFFORD_CLIFFORD_H
#define PUFFERLIB_OCEAN_CLIFFORD_CLIFFORD_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef CLIFFORD_N_QUBITS
#define CLIFFORD_N_QUBITS 6
#endif
#if CLIFFORD_N_QUBITS < 1 || CLIFFORD_N_QUBITS > 32
#error "CLIFFORD_N_QUBITS must be in [1, 32]; tableau rows are stored in uint64_t"
#endif
#define CLIFFORD_DIM (2 * CLIFFORD_N_QUBITS)
#define CLIFFORD_OBS_SIZE (CLIFFORD_DIM * CLIFFORD_DIM)
#ifndef CLIFFORD_USE_SHORTCUT_GATES
#define CLIFFORD_USE_SHORTCUT_GATES 1
#endif
#define CLIFFORD_SINGLE_QUBIT_ACTIONS (CLIFFORD_USE_SHORTCUT_GATES ? 5 : 2)
#define CLIFFORD_NUM_ACTIONS (CLIFFORD_SINGLE_QUBIT_ACTIONS * CLIFFORD_N_QUBITS + (CLIFFORD_N_QUBITS * (CLIFFORD_N_QUBITS - 1)) / 2)

#define GATE_H 0
#define GATE_S 1
#define GATE_V 2
#define GATE_HS 3
#define GATE_HV 4
#define GATE_CZ 5

typedef struct {
    int gate_kind;
    int q0;
    int q1;
} CliffordAction;

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float episode_cz_sum;
    float success_rate;
    float difficulty;
    float max_steps;
    float n;
    float success_count;
    float success_step_sum;
    float success_step_sq_sum;
    float success_cz_sum;
} Log;

typedef struct {
    uint64_t state;
} XorShift64;

typedef struct CliffordVecEnv CliffordVecEnv;

typedef struct {
    Log log;
    uint64_t* cols;
    CliffordVecEnv* vec;
    unsigned char* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    float episode_return;
    int episode_length;
    int episode_cz_count;
    int steps;
    int episode_max_steps;
    XorShift64 rng;
} CliffordEnv;

struct CliffordVecEnv {
    CliffordEnv* envs;
    int num_envs;
    int n_qubits;
    int dim;
    int difficulty;
    float difficulty_fraction;
    int max_steps;
    int num_actions;
    CliffordAction* actions;
    uint64_t identity_cols[64];
    float single_qubit_cost;
    float cz_cost;
    float goal_bonus;
    float failure_penalty;
};

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

static inline void build_clifford_actions(CliffordVecEnv* vec) {
    vec->num_actions = CLIFFORD_NUM_ACTIONS;
    vec->actions = (CliffordAction*)calloc((size_t)vec->num_actions, sizeof(CliffordAction));
    if (vec->actions == NULL) {
        return;
    }

    int idx = 0;
    const int max_single_gate = CLIFFORD_USE_SHORTCUT_GATES ? GATE_HV : GATE_S;
    for (int gate = GATE_H; gate <= max_single_gate; ++gate) {
        for (int qubit = 0; qubit < vec->n_qubits; ++qubit) {
            vec->actions[idx++] = (CliffordAction){.gate_kind = gate, .q0 = qubit, .q1 = -1};
        }
    }
    for (int src = 0; src < vec->n_qubits; ++src) {
        for (int dst = src + 1; dst < vec->n_qubits; ++dst) {
            vec->actions[idx++] = (CliffordAction){.gate_kind = GATE_CZ, .q0 = src, .q1 = dst};
        }
    }
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

static inline void write_observation(const CliffordVecEnv* vec, CliffordEnv* env) {
    const int dim = vec->dim;
    for (int row = 0; row < dim; ++row) {
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

static inline void add_log(CliffordVecEnv* vec, CliffordEnv* env, int success) {
    Log* log = &env->log;
    float difficulty_level = (float)vec->difficulty + vec->difficulty_fraction;
    log->perf += success ? 1.0f : 0.0f;
    log->score += env->episode_return;
    log->episode_return += env->episode_return;
    log->episode_length += (float)env->episode_length;
    log->episode_cz_sum += (float)env->episode_cz_count;
    log->success_rate += success ? 1.0f : 0.0f;
    log->difficulty += difficulty_level;
    log->max_steps += (float)vec->max_steps;
    log->n += 1.0f;
    if (success) {
        log->success_count += 1.0f;
        log->success_step_sum += (float)env->episode_length;
        log->success_step_sq_sum += (float)(env->episode_length * env->episode_length);
        log->success_cz_sum += (float)env->episode_cz_count;
    }
}

static inline float gate_cost_reward(const CliffordVecEnv* vec, int gate_kind) {
    return gate_kind == GATE_CZ ? -vec->cz_cost : -vec->single_qubit_cost;
}

static inline float step_single(CliffordVecEnv* vec, CliffordEnv* env) {
    int action_idx = ((int)env->actions[0]) % vec->num_actions;
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
    if (terminated) {
        reward += vec->goal_bonus;
    } else if (truncated) {
        reward += vec->failure_penalty;
    }
    env->episode_return += reward;
    env->rewards[0] = reward;
    env->terminals[0] = (float)(terminated || truncated);

    if (terminated || truncated) {
        add_log(vec, env, terminated);
        reset_single(vec, env);
    } else {
        write_observation(vec, env);
    }
    return reward;
}

static inline void c_reset(CliffordEnv* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    reset_single(env->vec, env);
}

static inline void c_step(CliffordEnv* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    step_single(env->vec, env);
}

static inline void c_render(CliffordEnv* env) {
    (void)env;
}

static inline void c_close(CliffordEnv* env) {
    (void)env;
}

#endif
