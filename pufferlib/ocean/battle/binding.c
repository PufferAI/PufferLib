#include "battle.h"

#define Env Battle
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->size_x = unpack(kwargs, "size_x");
    env->size_y = unpack(kwargs, "size_y");
    env->size_z = unpack(kwargs, "size_z");
    env->num_agents = unpack(kwargs, "num_agents");
    env->num_armies = unpack(kwargs, "num_armies");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "collision_rate", log->collision_rate);
    assign_to_dict(dict, "oob_rate", log->oob_rate);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
