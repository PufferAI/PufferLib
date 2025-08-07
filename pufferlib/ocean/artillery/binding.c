#include "artillery.h"

#define Env Artillery
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->frameskip = unpack(kwargs, "frameskip");
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->render = unpack(kwargs, "render");
    env->continuous = unpack(kwargs, "continuous");
    env->target_min_x = unpack(kwargs, "target_min_x");
    env->target_max_x = unpack(kwargs, "target_max_x");
    env->target_min_y = unpack(kwargs, "target_min_y");
    env->target_max_y = unpack(kwargs, "target_max_y");
    env->min_aim_angle = unpack(kwargs, "min_aim_angle");
    env->max_aim_angle = unpack(kwargs, "max_aim_angle");
    env->max_reward = unpack(kwargs, "max_reward");
    env->max_reward_dist = unpack(kwargs, "max_reward_dist");
    env->dist_fade = unpack(kwargs, "dist_fade");
    env->turn_penalty = unpack(kwargs, "turn_penalty");
    env->turn_penalty_delay = unpack(kwargs, "turn_penalty_delay");
    env->turn_penalty_ramp = unpack(kwargs, "turn_penalty_ramp");
    env->miss_penalty = unpack(kwargs, "miss_penalty");
    env->ftmp1 = unpack(kwargs, "ftmp1");
    env->ftmp2 = unpack(kwargs, "ftmp2");
    env->ftmp3 = unpack(kwargs, "ftmp3");
    env->ftmp4 = unpack(kwargs, "ftmp4");
    env->render_many = unpack(kwargs, "render_many");
    env->rng = unpack(kwargs, "rng");
    env->method = unpack(kwargs, "method");
    env->debug = unpack(kwargs, "debug");
    env->same_runs = unpack(kwargs, "same_runs");
    env->vm = unpack(kwargs, "vm");
    env->max_dist0 = unpack(kwargs, "max_dist0");
    env->out_bounds_penalty = unpack(kwargs, "out_bounds_penalty");
    env->i = unpack(kwargs, "i");

    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
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
