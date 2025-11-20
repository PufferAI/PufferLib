#include "g2048.h"

#define Env Game
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->can_go_over_65536 = unpack(kwargs, "can_go_over_65536");
    env->reward_scaler = unpack(kwargs, "reward_scaler");
    env->endgame_env_prob = unpack(kwargs, "endgame_env_prob");
    env->scaffolding_ratio = unpack(kwargs, "scaffolding_ratio");
    env->use_heuristic_rewards = unpack(kwargs, "use_heuristic_rewards");
    env->snake_reward_weight = unpack(kwargs, "snake_reward_weight");
    env->use_sparse_reward = unpack(kwargs, "use_sparse_reward");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "merge_score", log->merge_score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "lifetime_max_tile", log->lifetime_max_tile);
    assign_to_dict(dict, "reached_32768", log->reached_32768);
    assign_to_dict(dict, "reached_65536", log->reached_65536);
    assign_to_dict(dict, "monotonicity_reward", log->monotonicity_reward);
    assign_to_dict(dict, "snake_state", log->snake_state);
    assign_to_dict(dict, "snake_reward", log->snake_reward);
    return 0;
}