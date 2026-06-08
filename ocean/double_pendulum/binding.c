#include "double_pendulum.h"

#define OBS_SIZE DP_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {DP_ACTIONS}
#define OBS_TENSOR_T FloatTensor

#define Env DoublePendulum
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->cart_mass = dict_get(kwargs, "cart_mass")->value;
    env->link1_mass = dict_get(kwargs, "link1_mass")->value;
    env->link2_mass = dict_get(kwargs, "link2_mass")->value;
    env->link1_length = dict_get(kwargs, "link1_length")->value;
    env->link2_length = dict_get(kwargs, "link2_length")->value;
    env->gravity = dict_get(kwargs, "gravity")->value;
    env->force_mag = dict_get(kwargs, "force_mag")->value;
    env->dt = dict_get(kwargs, "dt")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "perf", log->perf);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "x_threshold_termination", log->x_threshold_termination);
    dict_set(out, "max_steps_termination", log->max_steps_termination);
    dict_set(out, "hold_time", log->hold_time);
    dict_set(out, "n", log->n);
}
