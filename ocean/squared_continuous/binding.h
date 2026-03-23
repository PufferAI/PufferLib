#include "squared_continuous.h"
#define OBS_SIZE 121
#define NUM_ATNS 2
#define ACT_SIZES {1, 1}  // Continuous: 2 dimensions, each size 1
#define OBS_TYPE UNSIGNED_CHAR
#define ACT_TYPE DOUBLE

#define Env Squared
#include "env_binding.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->size = dict_get(kwargs, "size")->value;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
