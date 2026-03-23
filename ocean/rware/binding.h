#include "rware.h"
#define OBS_SIZE 27
#define NUM_ATNS 1
#define ACT_SIZES {5}
#define OBS_TYPE FLOAT
#define ACT_TYPE DOUBLE

#define Env CRware
#include "env_binding.h"

void my_init(Env* env, Dict* kwargs) {
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->map_choice = dict_get(kwargs, "map_choice")->value;
    env->num_agents = dict_get(kwargs, "num_agents")->value;
    env->num_requested_shelves = dict_get(kwargs, "num_requested_shelves")->value;
    env->grid_square_size = dict_get(kwargs, "grid_square_size")->value;
    env->human_agent_idx = dict_get(kwargs, "human_agent_idx")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
