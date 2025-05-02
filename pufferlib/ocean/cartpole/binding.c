#include "cartpole.h"
#define Env Cartpole
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {   
    env->continuous = unpack(kwargs, "continuous");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "x_threshold_termination", log->x_threshold_termination);
    assign_to_dict(dict, "pole_angle_termination", log->pole_angle_termination);
    assign_to_dict(dict, "max_steps_termination", log->max_steps_termination);
    assign_to_dict(dict, "n", log->n);
    return 0;
}
