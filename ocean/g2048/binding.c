#include "g2048.h"
#define OBS_SIZE 16
#define NUM_ATNS 1
#define ACT_SIZES {4}
#define OBS_TENSOR_T ByteTensor
#define ACT_TYPE DOUBLE

#define Env Game
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->scaffolding_ratio = dict_get(kwargs, "scaffolding_ratio")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "merge_score", log->merge_score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "lifetime_max_tile", log->lifetime_max_tile);
    dict_set(out, "reached_16384", log->reached_16384);
    dict_set(out, "reached_32768", log->reached_32768);
    dict_set(out, "reached_65536", log->reached_65536);
    dict_set(out, "reached_131072", log->reached_131072);
}
