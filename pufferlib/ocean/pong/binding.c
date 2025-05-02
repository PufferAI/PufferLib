#include "pong.h"

#define Env Pong
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->paddle_width = unpack(kwargs, "paddle_width");
    env->paddle_height = unpack(kwargs, "paddle_height");
    env->ball_width = unpack(kwargs, "ball_width");
    env->ball_height = unpack(kwargs, "ball_height");
    env->paddle_speed = unpack(kwargs, "paddle_speed");
    env->ball_initial_speed_x = unpack(kwargs, "ball_initial_speed_x");
    env->ball_initial_speed_y = unpack(kwargs, "ball_initial_speed_y");
    env->ball_max_speed_y = unpack(kwargs, "ball_max_speed_y");
    env->ball_speed_y_increment = unpack(kwargs, "ball_speed_y_increment");
    env->max_score = unpack(kwargs, "max_score");
    env->frameskip = unpack(kwargs, "frameskip");
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
