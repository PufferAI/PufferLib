#include "clifford.h"

#define OBS_SIZE CLIFFORD_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {CLIFFORD_NUM_ACTIONS}
#define OBS_TENSOR_T ByteTensor

#define Env CliffordEnv
#define MY_VEC_INIT
#define MY_VEC_CLOSE
#include "vecenv.h"

static double dict_get_or(Dict* dict, const char* key, double fallback) {
    DictItem* item = dict_get_unsafe(dict, key);
    return item == NULL ? fallback : item->value;
}

Env* my_vec_init(
        int* num_envs_out,
        int* buffer_env_starts,
        int* buffer_env_counts,
        Dict* vec_kwargs,
        Dict* env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    assert(total_agents > 0);
    assert(num_buffers > 0);
    assert(total_agents % num_buffers == 0);

    int requested_n_qubits = (int)dict_get_or(env_kwargs, "n_qubits", CLIFFORD_N_QUBITS);
    if (requested_n_qubits != CLIFFORD_N_QUBITS) {
        fprintf(stderr,
            "clifford is compiled for n_qubits=%d, got n_qubits=%d\n",
            CLIFFORD_N_QUBITS, requested_n_qubits);
    }
    assert(requested_n_qubits == CLIFFORD_N_QUBITS);
    int requested_shortcuts = (int)dict_get_or(env_kwargs, "use_shortcut_gates", CLIFFORD_USE_SHORTCUT_GATES);
    if (requested_shortcuts != CLIFFORD_USE_SHORTCUT_GATES) {
        fprintf(stderr,
            "Clifford env was compiled with CLIFFORD_USE_SHORTCUT_GATES=%d but got use_shortcut_gates=%d\n",
            CLIFFORD_USE_SHORTCUT_GATES, requested_shortcuts);
    }
    assert(requested_shortcuts == CLIFFORD_USE_SHORTCUT_GATES);

    CliffordVecEnv* shared = (CliffordVecEnv*)calloc(1, sizeof(CliffordVecEnv));
    assert(shared != NULL);
    shared->num_envs = total_agents;
    shared->n_qubits = CLIFFORD_N_QUBITS;
    shared->dim = CLIFFORD_DIM;
    set_difficulty_level(shared, dict_get_or(env_kwargs, "difficulty", 10.0));
    shared->max_steps = (int)dict_get_or(env_kwargs, "max_steps", 200.0);
    shared->single_qubit_cost = (float)dict_get_or(env_kwargs, "single_qubit_cost", 0.001);
    shared->cz_cost = (float)dict_get_or(env_kwargs, "cz_cost", 0.1);
    shared->goal_bonus = (float)dict_get_or(env_kwargs, "goal_bonus", 0.0);
    shared->failure_penalty = (float)dict_get_or(env_kwargs, "failure_penalty", -1.0);
    assert(shared->max_steps > 0);
    assert(shared->single_qubit_cost >= 0.0f);
    assert(shared->cz_cost >= 0.0f);
    assert(shared->goal_bonus >= 0.0f);
    assert(shared->failure_penalty <= 0.0f);

    build_clifford_actions(shared);
    assert(shared->actions != NULL);
    for (int col = 0; col < shared->dim; ++col) {
        shared->identity_cols[col] = 1ULL << col;
    }

    uint64_t seed = (uint64_t)(uint32_t)dict_get_or(env_kwargs, "seed", 0.0);

    Env* envs = (Env*)calloc((size_t)total_agents, sizeof(Env));
    assert(envs != NULL);
    shared->envs = envs;

    int agents_per_buffer = total_agents / num_buffers;
    for (int buf = 0; buf < num_buffers; ++buf) {
        buffer_env_starts[buf] = buf * agents_per_buffer;
        buffer_env_counts[buf] = agents_per_buffer;
    }

    uint64_t seed_state = seed;
    for (int env_idx = 0; env_idx < total_agents; ++env_idx) {
        Env* env = &envs[env_idx];
        env->vec = shared;
        env->num_agents = 1;
        env->cols = (uint64_t*)calloc((size_t)shared->dim, sizeof(uint64_t));
        assert(env->cols != NULL);
        rng_seed(&env->rng, splitmix64_next(&seed_state) ^ (uint64_t)(env_idx + 1));
    }

    *num_envs_out = total_agents;
    return envs;
}

void my_vec_close(Env* envs) {
    if (envs == NULL) {
        return;
    }
    CliffordVecEnv* shared = envs[0].vec;
    if (shared == NULL) {
        return;
    }
    for (int env_idx = 0; env_idx < shared->num_envs; ++env_idx) {
        free(envs[env_idx].cols);
        envs[env_idx].cols = NULL;
    }
    free(shared->actions);
    free(shared);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "mean_cz", log->episode_cz_sum);
    dict_set(out, "success_rate", log->success_rate);
    dict_set(out, "difficulty", log->difficulty);
    dict_set(out, "max_steps", log->max_steps);

    float success_rate = log->success_count;
    float mean_success_steps = success_rate > 0.0f ? log->success_step_sum / success_rate : 0.0f;
    float mean_success_cz = success_rate > 0.0f ? log->success_cz_sum / success_rate : 0.0f;
    float success_step_second = success_rate > 0.0f ? log->success_step_sq_sum / success_rate : 0.0f;
    float success_step_var = success_step_second - mean_success_steps * mean_success_steps;
    if (success_step_var < 0.0f) {
        success_step_var = 0.0f;
    }
    dict_set(out, "success_step_mean", mean_success_steps);
    dict_set(out, "success_step_std", sqrtf(success_step_var));
    dict_set(out, "mean_success_cz", mean_success_cz);
}
