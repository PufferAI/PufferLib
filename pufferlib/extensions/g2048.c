#include "../ocean/g2048/g2048.h"

#define OBS_SIZE 289
#define ACT_SIZE 1
#define OBS_TYPE UNSIGNED_CHAR
#define ACT_TYPE INT


#define Env Game
#include "env_binding.h"

void my_init(Env* env, Dict* kwargs) {
    env->can_go_over_65536 = dict_get(kwargs, "can_go_over_65536")->int_value;
    env->reward_scaler = dict_get(kwargs, "reward_scaler")->float_value;
    env->endgame_env_prob = dict_get(kwargs, "endgame_env_prob")->float_value;
    env->scaffolding_ratio = dict_get(kwargs, "scaffolding_ratio")->float_value;
    env->use_heuristic_rewards = dict_get(kwargs, "use_heuristic_rewards")->int_value;
    env->snake_reward_weight = dict_get(kwargs, "snake_reward_weight")->float_value;
    env->use_sparse_reward = dict_get(kwargs, "use_sparse_reward")->int_value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set_float(out, "perf", log->perf);
    dict_set_float(out, "score", log->score);
    dict_set_float(out, "merge_score", log->merge_score);
    dict_set_float(out, "episode_return", log->episode_return);
    dict_set_float(out, "episode_length", log->episode_length);
    dict_set_float(out, "lifetime_max_tile", log->lifetime_max_tile);
    dict_set_float(out, "reached_32768", log->reached_32768);
    dict_set_float(out, "reached_65536", log->reached_65536);
    dict_set_float(out, "monotonicity_reward", log->monotonicity_reward);
    dict_set_float(out, "snake_state", log->snake_state);
    dict_set_float(out, "snake_reward", log->snake_reward);
}
