#include "slimevolley.h"
#define OBS_SIZE 12
#define NUM_ATNS 3
#define ACT_SIZES {2, 2, 2}
#define OBS_TYPE FLOAT
#define ACT_TYPE DOUBLE

#define Env SlimeVolley
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
