#include "../ocean/breakout/breakout.h"
#define OBS_SIZE 118
#define ACT_SIZE 1
#define OBS_TYPE FLOAT
#define ACT_TYPE DOUBLE

#define Env Breakout
#include "env_binding.h"

void my_init(Env* env, Dict* kwargs) {
    env->frameskip = dict_get(kwargs, "frameskip")->int_value;
    env->width = dict_get(kwargs, "width")->int_value;
    env->height = dict_get(kwargs, "height")->int_value;
    env->initial_paddle_width = dict_get(kwargs, "paddle_width")->int_value;
    env->paddle_height = dict_get(kwargs, "paddle_height")->int_value;
    env->ball_width = dict_get(kwargs, "ball_width")->int_value;
    env->ball_height = dict_get(kwargs, "ball_height")->int_value;
    env->brick_width = dict_get(kwargs, "brick_width")->int_value;
    env->brick_height = dict_get(kwargs, "brick_height")->int_value;
    env->brick_rows = dict_get(kwargs, "brick_rows")->int_value;
    env->brick_cols = dict_get(kwargs, "brick_cols")->int_value;
    env->initial_ball_speed = dict_get(kwargs, "initial_ball_speed")->int_value;
    env->max_ball_speed = dict_get(kwargs, "max_ball_speed")->int_value;
    env->paddle_speed = dict_get(kwargs, "paddle_speed")->int_value;
    env->continuous = dict_get(kwargs, "continuous")->int_value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set_float(out, "perf", log->perf);
    dict_set_float(out, "score", log->score);
    dict_set_float(out, "episode_return", log->episode_return);
    dict_set_float(out, "episode_length", log->episode_length);
}
