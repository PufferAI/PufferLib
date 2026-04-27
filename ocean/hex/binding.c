#include "hex.h"
#define OBS_SIZE 2*TOTAL_CELLS
#define NUM_ATNS 1
#define ACT_SIZES {TOTAL_CELLS}
#define OBS_TENSOR_T FloatTensor

#define Env Hex
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->random_opponent=dict_get(kwargs, "random_opponent")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}
