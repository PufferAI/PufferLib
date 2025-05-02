#include "breakout.h"

#define Env Breakout
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->frameskip = unpack(kwargs, "frameskip");
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->paddle_width = unpack(kwargs, "paddle_width");
    env->paddle_height = unpack(kwargs, "paddle_height");
    env->ball_width = unpack(kwargs, "ball_width");
    env->ball_height = unpack(kwargs, "ball_height");
    env->brick_width = unpack(kwargs, "brick_width");
    env->brick_height = unpack(kwargs, "brick_height");
    env->brick_rows = unpack(kwargs, "brick_rows");
    env->brick_cols = unpack(kwargs, "brick_cols");
    env->continuous = unpack(kwargs, "continuous");
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
