#include "click.h"
#define NUM_ATNS 3
#define ACT_SIZES {5, 5, 2}
#define OBS_TENSOR_T FloatTensor

#define Env ClickEnv
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1; // Single agent environment
    env->width = 800;
    env->height = 600;
    env->target_spawn_duration = dict_get(kwargs, "target_spawn_duration")->value;
    env->episode_length = dict_get(kwargs, "episode_length")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);  
}                                                                       
