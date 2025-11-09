#include "artymulti.h"

#define Env ArtyMulti
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->frameskip = unpack(kwargs, "frameskip");
    env->render = unpack(kwargs, "render");
    env->max_reward_dist = unpack(kwargs, "max_reward_dist");
    env->dist_fade = unpack(kwargs, "dist_fade");
    env->turn_penalty = unpack(kwargs, "turn_penalty");
    env->turn_penalty_delay = unpack(kwargs, "turn_penalty_delay");
    env->turn_penalty_ramp = unpack(kwargs, "turn_penalty_ramp");
    env->miss_penalty = unpack(kwargs, "miss_penalty");
    env->rng = unpack(kwargs, "rng");
    env->debug = unpack(kwargs, "debug");
    env->max_dist0 = unpack(kwargs, "max_dist0");
    env->out_bounds_penalty = unpack(kwargs, "out_bounds_penalty");
    env->i = unpack(kwargs, "i");

    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "dist", log->dist);
    assign_to_dict(dict, "max_reward_distn", log->max_reward_distn);
    assign_to_dict(dict, "turn_penaltyn", log->turn_penaltyn);
    assign_to_dict(dict, "acc1000", log->acc1000);
    //assign_to_dict(dict, "sigman", log->sigman);
    return 0;
}
