#include "boids.h"

#define Env Boids
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->num_boids = unpack(kwargs, "num_boids");
    env->report_interval = unpack(kwargs, "report_interval");
    env->margin_turn_factor = unpack(kwargs, "margin_turn_factor");
    env->centering_factor = unpack(kwargs, "centering_factor");
    env->avoid_factor = unpack(kwargs, "avoid_factor");
    env->matching_factor = unpack(kwargs, "matching_factor");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "n", log->n);
    return 0;
}
