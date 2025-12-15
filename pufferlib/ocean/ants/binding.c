#include "ants.h"

#define Env AntsEnv
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->num_ants = unpack(kwargs, "num_ants");
    env->reward_food = unpack(kwargs, "reward_food");
    env->reward_delivery = unpack(kwargs, "reward_delivery");
    env->reward_demo_match = unpack(kwargs, "reward_demo_match");
    env->reward_demo_mismatch = unpack(kwargs, "reward_demo_mismatch");
    env->reward_progress = unpack(kwargs, "reward_progress");
    env->reward_time_penalty = unpack(kwargs, "reward_time_penalty");
    env->reward_wrong_direction = unpack(kwargs, "reward_wrong_direction");
    env->reward_efficiency_bonus = unpack(kwargs, "reward_efficiency_bonus");
    env->cell_size = unpack(kwargs, "cell_size");

    init_ants_env(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "n", log->n);
    assign_to_dict(dict, "reward", log->reward);
    return 0;
}