#include "onlyfish.h"

#define OBS_SIZE 21
#define NUM_ATNS 2
#define ACT_SIZES {9, 5}
#define OBS_TENSOR_T FloatTensor

#define Env OnlyFish
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = dict_get(kwargs, "num_agents")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
