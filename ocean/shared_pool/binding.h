#include "shared_pool.h"

#define Env CCpr
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->num_agents = unpack(kwargs, "num_agents");
    env->vision = unpack(kwargs, "vision");
    env->reward_food = unpack(kwargs, "reward_food");
    env->interactive_food_reward = unpack(kwargs, "interactive_food_reward");
    env->reward_move = unpack(kwargs, "reward_move");
    env->food_base_spawn_rate = unpack(kwargs, "food_base_spawn_rate");
    init_ccpr(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "moves", log->moves);
    assign_to_dict(dict, "food_nb", log->food_nb);
    assign_to_dict(dict, "alive_steps", log->alive_steps);
    return 0;
}
