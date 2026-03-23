#include "pong.h"
#define OBS_SIZE 8
#define NUM_ATNS 1
#define ACT_SIZES {3}
#define OBS_TENSOR_T FloatTensor
#define ACT_TYPE DOUBLE

#define Env Pong
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->paddle_width = dict_get(kwargs, "paddle_width")->value;
    env->paddle_height = dict_get(kwargs, "paddle_height")->value;
    env->ball_width = dict_get(kwargs, "ball_width")->value;
    env->ball_height = dict_get(kwargs, "ball_height")->value;
    env->paddle_speed = dict_get(kwargs, "paddle_speed")->value;
    env->ball_initial_speed_x = dict_get(kwargs, "ball_initial_speed_x")->value;
    env->ball_initial_speed_y = dict_get(kwargs, "ball_initial_speed_y")->value;
    env->ball_max_speed_y = dict_get(kwargs, "ball_max_speed_y")->value;
    env->ball_speed_y_increment = dict_get(kwargs, "ball_speed_y_increment")->value;
    env->max_score = dict_get(kwargs, "max_score")->value;
    env->frameskip = dict_get(kwargs, "frameskip")->value;
    env->continuous = dict_get(kwargs, "continuous")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
