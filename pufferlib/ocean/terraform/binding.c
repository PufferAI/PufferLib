#include "terraform.h"

#define Env Terraform
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->size = unpack(kwargs, "size");
    env->num_agents = unpack(kwargs, "num_agents");
    env->reward_scale = unpack(kwargs, "reward_scale");
    env->reset_frequency = unpack(kwargs, "reset_frequency");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "quadrant_progress", log->quadrant_progress);
    return 0;
}
