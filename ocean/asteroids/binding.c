#include "asteroids.h"

#define OBS_SIZE (4 + 5*MAX_ASTEROIDS)
#define NUM_ATNS 1
#define ACT_SIZES {4}
#define OBS_TENSOR_T FloatTensor

#define Env Asteroids
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->size = dict_get(kwargs, "size")->value;
    env->frameskip = dict_get(kwargs, "frameskip")->value;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
