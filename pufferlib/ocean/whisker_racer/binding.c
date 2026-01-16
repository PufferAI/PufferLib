#include "whisker_racer.h"

#define Env WhiskerRacer
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->frameskip = unpack(kwargs, "frameskip");
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->llw_ang = unpack(kwargs, "llw_ang");
    env->flw_ang = unpack(kwargs, "flw_ang");
    env->frw_ang = unpack(kwargs, "frw_ang");
    env->rrw_ang = unpack(kwargs, "rrw_ang");
    env->max_whisker_length = unpack(kwargs, "max_whisker_length");
    env->turn_pi_frac = unpack(kwargs, "turn_pi_frac");
    env->maxv = unpack(kwargs, "maxv");
    env->render = unpack(kwargs, "render");
    env->continuous = unpack(kwargs, "continuous");
    env->reward_yellow = unpack(kwargs, "reward_yellow");
    env->reward_green = unpack(kwargs, "reward_green");
    env->gamma = unpack(kwargs, "gamma");
    env->track_width = unpack(kwargs, "track_width");
    env->num_radial_sectors = unpack(kwargs, "num_radial_sectors");
    env->num_points = unpack(kwargs, "num_points");
    env->bezier_resolution = unpack(kwargs, "bezier_resolution");
    env->turn_pi_frac = unpack(kwargs, "turn_pi_frac");
    env->w_ang = unpack(kwargs, "w_ang");
    env->corner_thresh = unpack(kwargs, "corner_thresh");
    env->ftmp1 = unpack(kwargs, "ftmp1");
    env->ftmp2 = unpack(kwargs, "ftmp2");
    env->ftmp3 = unpack(kwargs, "ftmp3");
    env->ftmp4 = unpack(kwargs, "ftmp4");
    env->mode7 = unpack(kwargs, "mode7");
    env->render_many = unpack(kwargs, "render_many");
    env->rng = unpack(kwargs, "rng");
    env->method = unpack(kwargs, "method");
    env->i = unpack(kwargs, "i");

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
