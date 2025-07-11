#include "froggy.h"
#define Env Froggy 
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->episode_length = unpack(kwargs, "episode_length");
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "lives_remaining", log->lives_remaining);
    assign_to_dict(dict, "crossings", log->crossings);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "episode_return", log->episode_return);
    // assign_to_dict(dict, "score", log->score);
    // assign_to_dict(dict, "n", log->n);
    return 0;
}


