#include "freeway.h"
#define OBS_SIZE 34
#define NUM_ATNS 1
#define ACT_SIZES {3}
#define OBS_TYPE FLOAT
#define ACT_TYPE DOUBLE

#define Env Freeway
#include "env_binding.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->frameskip = dict_get(kwargs, "frameskip")->value;
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->player_width = dict_get(kwargs, "player_width")->value;
    env->player_height = dict_get(kwargs, "player_height")->value;
    env->car_width = dict_get(kwargs, "car_width")->value;
    env->car_height = dict_get(kwargs, "car_height")->value;
    env->lane_size = dict_get(kwargs, "lane_size")->value;
    env->difficulty = dict_get(kwargs, "difficulty")->value;
    env->level = dict_get(kwargs, "level")->value;
    env->enable_human_player = dict_get(kwargs, "enable_human_player")->value;
    env->env_randomization = dict_get(kwargs, "env_randomization")->value;
    env->use_dense_rewards = dict_get(kwargs, "use_dense_rewards")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "up_action_frac", log->up_action_frac);
    dict_set(out, "hits", log->hits);
}
