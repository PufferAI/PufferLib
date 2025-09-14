#include "onestateworld.h"

#define Env World
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->mean_left = unpack(kwargs, "mean_left");
    env->mean_right = unpack(kwargs, "mean_right");
    env->var_right = unpack(kwargs, "var_right");
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
