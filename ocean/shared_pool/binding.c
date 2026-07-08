#include "shared_pool.h"

// (2*vision + 1)^2 with vision = 3 (must match config)
#define OBS_SIZE 49
#define NUM_ATNS 1
#define ACT_SIZES {5}
#define OBS_TENSOR_T ByteTensor

#define Env CCpr
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->num_agents = dict_get(kwargs, "num_agents")->value;
    env->vision = dict_get(kwargs, "vision")->value;
    env->reward_food = dict_get(kwargs, "reward_food")->value;
    env->interactive_food_reward = dict_get(kwargs, "interactive_food_reward")->value;
    env->reward_move = dict_get(kwargs, "reward_move")->value;
    env->food_base_spawn_rate = dict_get(kwargs, "food_base_spawn_rate")->value;
    init_ccpr(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "moves", log->moves);
    dict_set(out, "food_nb", log->food_nb);
    dict_set(out, "alive_steps", log->alive_steps);
}
