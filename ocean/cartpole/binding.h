#include "cartpole.h"
#define OBS_SIZE 4
#define NUM_ATNS 1
#define ACT_SIZES {2}
#define OBS_TYPE FLOAT
#define ACT_TYPE DOUBLE

#define Env Cartpole
#include "env_binding.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->cart_mass = dict_get(kwargs, "cart_mass")->value;
    env->pole_mass = dict_get(kwargs, "pole_mass")->value;
    env->pole_length = dict_get(kwargs, "pole_length")->value;
    env->gravity = dict_get(kwargs, "gravity")->value;
    env->force_mag = dict_get(kwargs, "force_mag")->value;
    env->tau = dict_get(kwargs, "dt")->value;
    env->continuous = dict_get(kwargs, "continuous")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "perf", log->perf);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "x_threshold_termination", log->x_threshold_termination);
    dict_set(out, "pole_angle_termination", log->pole_angle_termination);
    dict_set(out, "max_steps_termination", log->max_steps_termination);
    dict_set(out, "n", log->n);
}
