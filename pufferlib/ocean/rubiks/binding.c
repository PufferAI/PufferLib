#include "rubiks.h"

#define Env Cube
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->N = (int) unpack(kwargs, "N");
    env->shuffles = (int) unpack(kwargs, "shuffles");
    env->size = (int) unpack(kwargs, "size");
    env->max_episode_steps = (int) unpack(kwargs, "max_episode_steps");
    env->anim_time = (float) unpack(kwargs, "anim_time");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
