#include "overcooked.h"

#define OBS_SIZE 43
#define NUM_ATNS 1
#define ACT_SIZES {6}
#define OBS_TENSOR_T FloatTensor

#define Env Overcooked
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->layout_id = (LayoutType)dict_get(kwargs, "layout")->value;
    env->num_agents = (int)dict_get(kwargs, "num_agents")->value;
    env->grid_size = (int)dict_get(kwargs, "grid_size")->value;
    env->observation_size = OBS_SIZE;

    env->rewards_config.dish_served_whole_team = dict_get(kwargs, "reward_dish_served_whole_team")->value;
    env->rewards_config.dish_served_agent = dict_get(kwargs, "reward_dish_served_agent")->value;
    env->rewards_config.pot_started = dict_get(kwargs, "reward_pot_started")->value;
    env->rewards_config.ingredient_added = dict_get(kwargs, "reward_ingredient_added")->value;
    env->rewards_config.ingredient_picked = dict_get(kwargs, "reward_ingredient_picked")->value;
    env->rewards_config.plate_picked = dict_get(kwargs, "reward_plate_picked")->value;
    env->rewards_config.soup_plated = dict_get(kwargs, "reward_soup_plated")->value;
    env->rewards_config.wrong_dish_served = dict_get(kwargs, "reward_wrong_dish_served")->value;
    env->rewards_config.step_penalty = dict_get(kwargs, "reward_step_penalty")->value;

    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "dishes_served", log->dishes_served);
    dict_set(out, "correct_dishes", log->correct_dishes);
    dict_set(out, "wrong_dishes", log->wrong_dishes);
    dict_set(out, "ingredients_picked", log->ingredients_picked);
    dict_set(out, "pots_started", log->pots_started);
    dict_set(out, "items_dropped", log->items_dropped);
    dict_set(out, "agent_collisions", log->agent_collisions);
}
