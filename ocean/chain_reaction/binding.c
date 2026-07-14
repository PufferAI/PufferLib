#include <unistd.h>

#include "chain_reaction.h"

#define NUM_ATNS 1
#define ACT_SIZES {MAX_ACTIONS}
#define OBS_TENSOR_T FloatTensor
#define MY_ACTION_MASK MAX_ACTIONS
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
    int total_agents = dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = dict_get(vec_kwargs, "num_buffers")->value;
    int agents_per_env = dict_get(env_kwargs, "num_agents")->value;
    if (agents_per_env < 1 || agents_per_env > MAX_SLOTS) {
        fprintf(stderr, "chain_reaction num_agents must be 1 or 2, got %d\n", agents_per_env);
        exit(1);
    }
    if (total_agents < 1 || num_buffers < 1 || total_agents % num_buffers != 0) {
        fprintf(stderr,
            "chain_reaction requires positive total_agents (%d) divisible by num_buffers (%d)\n",
            total_agents, num_buffers);
        exit(1);
    }

    int agents_per_buffer = total_agents / num_buffers;
    if (agents_per_buffer % agents_per_env != 0) {
        fprintf(stderr,
            "chain_reaction requires %d agents per buffer divisible by num_agents (%d)\n",
            agents_per_buffer, agents_per_env);
        exit(1);
    }

    int num_envs = total_agents / agents_per_env;
    int envs_per_buffer = agents_per_buffer / agents_per_env;
    Env* envs = calloc(num_envs, sizeof(Env));
    unsigned int process_rng = (unsigned int)getpid();
    for (int i = 0; i < num_envs; i++) {
        envs[i].rng = rand_r(&process_rng);
        my_init(&envs[i], env_kwargs);
    }
    for (int buffer = 0; buffer < num_buffers; buffer++) {
        buffer_env_starts[buffer] = buffer * envs_per_buffer;
        buffer_env_counts[buffer] = envs_per_buffer;
    }

    *num_envs_out = num_envs;
    return envs;
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = dict_get(kwargs, "num_agents")->value;
    env->rows = dict_get(kwargs, "rows")->value;
    env->cols = dict_get(kwargs, "cols")->value;
    env->max_steps = dict_get(kwargs, "max_steps")->value;
    env->opponent_policy = dict_get(kwargs, "opponent_policy")->value;
    env->win_reward = dict_get(kwargs, "win_reward")->value;
    env->loss_reward = dict_get(kwargs, "loss_reward")->value;
    env->invalid_move_reward = dict_get(kwargs, "invalid_move_reward")->value;

    if (env->rows < 2) {
        env->rows = 2;
    }
    if (env->cols < 2) {
        env->cols = 2;
    }
    if (env->rows > MAX_ROWS) {
        env->rows = MAX_ROWS;
    }
    if (env->cols > MAX_COLS) {
        env->cols = MAX_COLS;
    }
    if (env->max_steps < 1) {
        env->max_steps = 1;
    }

    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "chain_bursts", log->chain_bursts);
    dict_set(out, "invalid_rate", log->invalid_rate);
    dict_set(out, "hist_score", log->hist_score);
    dict_set(out, "hist_n", log->hist_n);
    dict_set(out, "hist_score_bank_0", log->hist_score_bank[0]);
    dict_set(out, "hist_score_bank_1", log->hist_score_bank[1]);
    dict_set(out, "hist_score_bank_2", log->hist_score_bank[2]);
    dict_set(out, "hist_score_bank_3", log->hist_score_bank[3]);
    dict_set(out, "hist_score_bank_4", log->hist_score_bank[4]);
    dict_set(out, "hist_score_bank_5", log->hist_score_bank[5]);
    dict_set(out, "hist_score_bank_6", log->hist_score_bank[6]);
    dict_set(out, "hist_score_bank_7", log->hist_score_bank[7]);
    dict_set(out, "hist_n_bank_0", log->hist_n_bank[0]);
    dict_set(out, "hist_n_bank_1", log->hist_n_bank[1]);
    dict_set(out, "hist_n_bank_2", log->hist_n_bank[2]);
    dict_set(out, "hist_n_bank_3", log->hist_n_bank[3]);
    dict_set(out, "hist_n_bank_4", log->hist_n_bank[4]);
    dict_set(out, "hist_n_bank_5", log->hist_n_bank[5]);
    dict_set(out, "hist_n_bank_6", log->hist_n_bank[6]);
    dict_set(out, "hist_n_bank_7", log->hist_n_bank[7]);
    dict_set(out, "slot_0_score", log->slot_0_score);
    dict_set(out, "slot_1_score", log->slot_1_score);
    dict_set(out, "draw_rate", log->draw_rate);
}
