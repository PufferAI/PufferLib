#include "terraform.h"
#define OBS_SIZE 319
#define NUM_ATNS 3
#define ACT_SIZES {5, 5, 3}
#define OBS_TYPE FLOAT
#define ACT_TYPE DOUBLE

#define Env Terraform
#include "env_binding.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = dict_get(kwargs, "num_agents")->value;
    env->size = dict_get(kwargs, "size")->value;
    env->reset_frequency = dict_get(kwargs, "reset_frequency")->value;
    env->reward_scale = dict_get(kwargs, "reward_scale")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "quadrant_progress", log->quadrant_progress);
}
