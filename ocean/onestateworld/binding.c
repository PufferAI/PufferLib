#include "onestateworld.h"

#define OBS_SIZE 1
#define NUM_ATNS 1
#define ACT_SIZES {2}
#define OBS_TENSOR_T ByteTensor

#define Env World
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->mean_left = dict_get(kwargs, "mean_left")->value;
    env->mean_right = dict_get(kwargs, "mean_right")->value;
    env->var_right = dict_get(kwargs, "var_right")->value;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
