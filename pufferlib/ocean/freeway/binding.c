#include "freeway.h"

#define Env Freeway
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->frameskip = unpack(kwargs, "frameskip");
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->player_width = unpack(kwargs, "player_width");
    env->player_height = unpack(kwargs, "player_height");
    env->car_width = unpack(kwargs, "car_width");
    env->car_height = unpack(kwargs, "car_height");
    env->lane_size = unpack(kwargs, "lane_size");
    env->level = unpack(kwargs, "level");
    env->difficulty = unpack(kwargs, "difficulty");
    env->use_dense_rewards = unpack(kwargs, "use_dense_rewards");
    env->env_randomization = unpack(kwargs, "env_randomization");
    env->enable_human_player = unpack(kwargs, "enable_human_player");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "up_action_frac", log->up_action_frac);
    assign_to_dict(dict, "hits", log->hits);
    return 0;
}
