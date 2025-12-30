#include "dinosaur.h"

#define Env Dinosaur
#include "../env_binding.h"

// Python -> C
static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->speed_init = unpack(kwargs, "speed_init");
    env->speed_max = unpack(kwargs, "speed_max");
    env->spawn_rate_min = unpack(kwargs, "spawn_rate_min");
    env->spawn_rate_max = unpack(kwargs, "spawn_rate_max");
    env->rate_increment_rate = unpack(kwargs, "rate_increment_rate");
    env->max_obstacles = unpack(kwargs, "max_obstacles");
    init(env);
    return 0;
}

// C -> Python
static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
