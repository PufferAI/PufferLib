#include "dino.h"

#define OBS_SIZE (5 + 3 * 9)
#define NUM_ATNS 1
#define ACT_SIZES {3}
#define OBS_TENSOR_T FloatTensor

#define Env Dinosaur
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->speed_init = dict_get(kwargs, "speed_init")->value;
    env->speed_max = dict_get(kwargs, "speed_max")->value;
    env->spawn_rate_min = dict_get(kwargs, "spawn_rate_min")->value;
    env->spawn_rate_max = dict_get(kwargs, "spawn_rate_max")->value;
    env->rate_increment_rate = dict_get(kwargs, "rate_increment_rate")->value;
    c_init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}