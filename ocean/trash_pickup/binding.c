#include "trash_pickup.h"
#define OBS_SIZE 605
#define NUM_ATNS 1
#define ACT_SIZES {4}
#define OBS_TYPE CHAR
#define ACT_TYPE DOUBLE

#define Env CTrashPickupEnv
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = dict_get(kwargs, "num_agents")->value;
    env->grid_size = dict_get(kwargs, "grid_size")->value;
    env->num_trash = dict_get(kwargs, "num_trash")->value;
    env->num_bins = dict_get(kwargs, "num_bins")->value;
    env->max_steps = dict_get(kwargs, "max_steps")->value;
    env->agent_sight_range = dict_get(kwargs, "agent_sight_range")->value;
    initialize_env(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "trash_collected", log->trash_collected);
}
