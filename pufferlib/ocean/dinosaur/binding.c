#include "dinosaur.h"

#define Env Dinosaur
#include "../env_binding.h"

// Python -> C
static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->speed_init = unpack(kwargs, "speed_init");
    env->speed_max = unpack(kwargs, "speed_max");
    env->obstacle_spawn_rate_init = unpack(kwargs, "obstacle_spawn_rate_init");
    env->obstacle_spawn_rate_min = unpack(kwargs, "obstacle_spawn_rate_min");
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
