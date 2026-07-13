#include "chain_reaction.h"

#define OBS_SIZE CHAINENV_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {CHAINENV_MAX_ACTIONS}
#define OBS_TENSOR_T FloatTensor
#define MY_ACTION_MASK CHAINENV_MAX_ACTIONS
#define MY_VEC_INIT
#define MY_USES_PERM
#define MY_USES_TAGS

#define Env ChainEnv
#include "vecenv.h"

void my_setup_perm(StaticVec* vec, Env* env, int slot_base) {
    size_t obs_elem_size = obs_element_size();
    for (int slot = 0; slot < env->num_agents; slot++) {
        int phys = vec->agent_perm != NULL
            ? vec->agent_perm[slot_base + slot]
            : slot_base + slot;
        env->obs_ptr[slot] = (float*)((char*)vec->observations
            + (size_t)phys * OBS_SIZE * obs_elem_size);
        env->action_ptr[slot] = vec->actions + (size_t)phys * NUM_ATNS;
        env->reward_ptr[slot] = vec->rewards + phys;
        env->terminal_ptr[slot] = vec->terminals + phys;
        env->action_mask_ptr[slot] = vec->action_mask + (size_t)phys * MY_ACTION_MASK;
    }
}

Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int agents_per_buffer = total_agents / num_buffers;
    int agents_per_env = (int)dict_get(env_kwargs, "num_agents")->value;
    if (total_agents % agents_per_env != 0) {
        fprintf(stderr,
            "chain_reaction requires total_agents (%d) divisible by num_agents (%d)\n",
            total_agents, agents_per_env);
        exit(1);
    }

    int num_envs = total_agents / agents_per_env;
    Env* envs = (Env*)calloc(num_envs, sizeof(Env));
    for (int i = 0; i < num_envs; i++) {
        envs[i].rng = i;
        my_init(&envs[i], env_kwargs);
    }

    int buf = 0;
    int buf_agents = 0;
    buffer_env_starts[0] = 0;
    buffer_env_counts[0] = 0;
    for (int i = 0; i < num_envs; i++) {
        buf_agents += agents_per_env;
        buffer_env_counts[buf]++;
        if (buf_agents >= agents_per_buffer && buf < num_buffers - 1) {
            buf++;
            buffer_env_starts[buf] = i + 1;
            buffer_env_counts[buf] = 0;
            buf_agents = 0;
        }
    }

    *num_envs_out = num_envs;
    return envs;
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = (int)dict_get(kwargs, "num_agents")->value;
    env->rows = (int)dict_get(kwargs, "rows")->value;
    env->cols = (int)dict_get(kwargs, "cols")->value;
    env->max_steps = (int)dict_get(kwargs, "max_steps")->value;
    env->opponent_policy = (int)dict_get(kwargs, "opponent_policy")->value;
    env->win_reward = (float)dict_get(kwargs, "win_reward")->value;
    env->loss_reward = (float)dict_get(kwargs, "loss_reward")->value;
    env->invalid_move_reward = (float)dict_get(kwargs, "invalid_move_reward")->value;

    if (env->rows < 2) env->rows = 2;
    if (env->cols < 2) env->cols = 2;
    if (env->rows > CHAINENV_MAX_ROWS) env->rows = CHAINENV_MAX_ROWS;
    if (env->cols > CHAINENV_MAX_COLS) env->cols = CHAINENV_MAX_COLS;
    if (env->max_steps < 1) env->max_steps = 1;

    init_chainenv(env);
}

void my_log(Log* log, Dict* out) {
    static const char* hist_score_keys[CHAINENV_MAX_BANKS] = {
        "hist_score_bank_0", "hist_score_bank_1", "hist_score_bank_2", "hist_score_bank_3",
        "hist_score_bank_4", "hist_score_bank_5", "hist_score_bank_6", "hist_score_bank_7",
    };
    static const char* hist_n_keys[CHAINENV_MAX_BANKS] = {
        "hist_n_bank_0", "hist_n_bank_1", "hist_n_bank_2", "hist_n_bank_3",
        "hist_n_bank_4", "hist_n_bank_5", "hist_n_bank_6", "hist_n_bank_7",
    };
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "chain_bursts", log->chain_bursts);
    dict_set(out, "invalid_rate", log->invalid_rate);
    dict_set(out, "hist_score", log->hist_score);
    dict_set(out, "hist_n", log->hist_n);
    for (int bank = 0; bank < CHAINENV_MAX_BANKS; bank++) {
        dict_set(out, hist_score_keys[bank], log->hist_score_bank[bank]);
        dict_set(out, hist_n_keys[bank], log->hist_n_bank[bank]);
    }
    dict_set(out, "slot_0_score", log->slot_0_score);
    dict_set(out, "slot_1_score", log->slot_1_score);
    dict_set(out, "draw_rate", log->draw_rate);
}
