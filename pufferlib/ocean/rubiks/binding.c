#include "cube.h"

#define Env Cube
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->N = unpack(kwargs, "N");
    env->shuffles = unpack(kwargs, "shuffles;");
    env->obs_type = unpack(kwargs, "obs_type;");
    env->size = unpack(kwargs, "size;");
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
