#include "pacman.h"
#define OBS_SIZE 291
#define NUM_ATNS 1
#define ACT_SIZES {4}
#define OBS_TYPE FLOAT
#define ACT_TYPE DOUBLE

#define Env PacmanEnv
#include "env_binding.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->randomize_starting_position = dict_get(kwargs, "randomize_starting_position")->value;
    env->min_start_timeout = dict_get(kwargs, "min_start_timeout")->value;
    env->max_start_timeout = dict_get(kwargs, "max_start_timeout")->value;
    env->frightened_time = dict_get(kwargs, "frightened_time")->value;
    env->max_mode_changes = dict_get(kwargs, "max_mode_changes")->value;
    env->scatter_mode_length = dict_get(kwargs, "scatter_mode_length")->value;
    env->chase_mode_length = dict_get(kwargs, "chase_mode_length")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
