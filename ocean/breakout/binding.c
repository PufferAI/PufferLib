#include "breakout.h"
#define OBS_SIZE 118
#define NUM_ATNS 1
#define ACT_SIZES {3}
#define OBS_TENSOR_T FloatTensor

#define Env Breakout
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->frameskip = dict_get(kwargs, "frameskip")->value;
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->initial_paddle_width = dict_get(kwargs, "paddle_width")->value;
    env->paddle_height = dict_get(kwargs, "paddle_height")->value;
    env->ball_width = dict_get(kwargs, "ball_width")->value;
    env->ball_height = dict_get(kwargs, "ball_height")->value;
    env->brick_width = dict_get(kwargs, "brick_width")->value;
    env->brick_height = dict_get(kwargs, "brick_height")->value;
    env->brick_rows = dict_get(kwargs, "brick_rows")->value;
    env->brick_cols = dict_get(kwargs, "brick_cols")->value;
    env->initial_ball_speed = dict_get(kwargs, "initial_ball_speed")->value;
    env->max_ball_speed = dict_get(kwargs, "max_ball_speed")->value;
    env->paddle_speed = dict_get(kwargs, "paddle_speed")->value;
    env->continuous = dict_get(kwargs, "continuous")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
