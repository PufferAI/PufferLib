#include "overcooked.h"

#define Env Overcooked
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->num_agents = unpack(kwargs, "num_agents");
    env->max_steps = unpack(kwargs, "max_steps");
    env->grid_size = unpack(kwargs, "grid_size");
    env->observation_size = unpack(kwargs, "observation_size");
    env->reward_dish_served = unpack(kwargs, "reward_dish_served");
    env->reward_step_penalty = unpack(kwargs, "reward_step_penalty");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "dishes_served", log->dishes_served);
    assign_to_dict(dict, "cooperation_score", log->cooperation_score);
    // User-defined stats
    assign_to_dict(dict, "correct_dishes", log->correct_dishes);
    assign_to_dict(dict, "wrong_dishes", log->wrong_dishes);
    assign_to_dict(dict, "ingredients_picked", log->ingredients_picked);
    assign_to_dict(dict, "pots_started", log->pots_started);
    assign_to_dict(dict, "items_dropped", log->items_dropped);
    assign_to_dict(dict, "agent_collisions", log->agent_collisions);
    assign_to_dict(dict, "cooking_time_efficiency", log->cooking_time_efficiency);
    return 0;
}
