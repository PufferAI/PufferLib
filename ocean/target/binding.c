#include "target.h"
#define OBS_SIZE 28  // num_goals*2 + num_agents*2 + 4 = 4*2 + 8*2 + 4
#define NUM_ATNS 2
#define ACT_SIZES {9, 5}
#define OBS_TENSOR_T FloatTensor

#define Env Target
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->width = 952;
    env->height = 592;
    env->num_agents = 8;
    env->num_goals = 4;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
