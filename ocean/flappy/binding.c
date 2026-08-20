#include "flappy.h"

#define OBS_SIZE FLAPPY_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {2}
#define OBS_TENSOR_T FloatTensor

#define Env Flappy
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->max_steps = dict_get(kwargs, "max_steps")->value;
    env->gravity = dict_get(kwargs, "gravity")->value;
    env->flap_velocity = dict_get(kwargs, "flap_velocity")->value;
    env->pipe_speed = dict_get(kwargs, "pipe_speed")->value;
    env->pipe_gap = dict_get(kwargs, "pipe_gap")->value;
    env->pipe_width = dict_get(kwargs, "pipe_width")->value;
    env->pipe_spacing = dict_get(kwargs, "pipe_spacing")->value;
    env->first_pipe_x = dict_get(kwargs, "first_pipe_x")->value;
    env->bird_x = dict_get(kwargs, "bird_x")->value;
    env->bird_radius = dict_get(kwargs, "bird_radius")->value;
    env->alive_reward = dict_get(kwargs, "alive_reward")->value;
    env->pass_reward = dict_get(kwargs, "pass_reward")->value;
    env->crash_reward = dict_get(kwargs, "crash_reward")->value;
    env->center_reward = dict_get(kwargs, "center_reward")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}

