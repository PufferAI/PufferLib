#include "tmaze.h"

#define OBS_SIZE 4
#define NUM_ATNS 1
#define ACT_SIZES {3}
#define OBS_TENSOR_T ByteTensor

#define Env TMaze
#include "vecenv.h"

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
