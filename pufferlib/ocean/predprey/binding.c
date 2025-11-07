#include "predprey.h"

#define Env PredPrey
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->num_agents = unpack(kwargs, "num_agents");
    env->vision = unpack(kwargs, "vision");
    env->reward_death_scale = unpack(kwargs, "reward_death_scale");
    env->reward_eat = unpack(kwargs, "reward_eat");
    env->reward_collect = unpack(kwargs, "reward_collect");
    env->timestep_reward = unpack(kwargs, "timestep_reward");
    env->reward_steal = unpack(kwargs, "reward_steal");
    env->hp_reward_scale = unpack(kwargs, "hp_reward_scale");
    env->held_food_reward_scale = unpack(kwargs, "held_food_reward_scale");
    env->food_base_spawn_rate = unpack(kwargs, "food_base_spawn_rate");
    init_cenv(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "steals", log->steals);
    assign_to_dict(dict, "collects", log->collects);
    return 0;
}