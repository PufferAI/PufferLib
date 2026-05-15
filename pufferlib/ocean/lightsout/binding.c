#include "lightsout.h"

#define Env LightsOut
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->grid_size = unpack(kwargs, "grid_size");
    env->cell_size = unpack(kwargs, "cell_size");
    env->max_steps = unpack(kwargs, "max_steps");
    env->ema = 0.5f;
    env->score_ema = 0.0f;
    env->scramble_prob = 0.15f;
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "scramble_p", log->scramble_p);
    return 0;
}
