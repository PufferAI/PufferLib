#include "convert.h"
#define OBS_SIZE 28
#define NUM_ATNS 2
#define ACT_SIZES {9, 5}
#define OBS_TYPE FLOAT
#define ACT_TYPE DOUBLE

#define Env Convert
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = dict_get(kwargs, "num_agents")->value;
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->num_factories = dict_get(kwargs, "num_factories")->value;
    env->num_resources = dict_get(kwargs, "num_resources")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
