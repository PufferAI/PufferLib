#include "lightsout.h"

#define Env LightsOut
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->grid_size = unpack(kwargs, "grid_size");
    env->cell_size = unpack(kwargs, "cell_size");
    env->max_steps = unpack(kwargs, "max_steps");
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    return 0;
}
