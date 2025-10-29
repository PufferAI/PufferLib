#include "predprey.h"

#define Env PredPrey
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->num_agents = unpack(kwargs, "num_agents");
    env->vision = unpack(kwargs, "vision");
    env->reward_food = unpack(kwargs, "reward_food");
    env->food_base_spawn_rate = unpack(kwargs, "food_base_spawn_rate");
    init_cenv(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "steals", log->steals);
    assign_to_dict(dict, "collects", log->collects);
    return 0;
}