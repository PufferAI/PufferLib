#include "snake.h"
#define OBS_SIZE 121
#define NUM_ATNS 1
#define ACT_SIZES {4}
#define OBS_TYPE CHAR
#define ACT_TYPE DOUBLE

#define Env CSnake
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->num_agents = dict_get(kwargs, "num_agents")->value;
    env->vision = dict_get(kwargs, "vision")->value;
    env->leave_corpse_on_death = dict_get(kwargs, "leave_corpse_on_death")->value;
    env->food = dict_get(kwargs, "num_food")->value;
    env->reward_food = dict_get(kwargs, "reward_food")->value;
    env->reward_corpse = dict_get(kwargs, "reward_corpse")->value;
    env->reward_death = dict_get(kwargs, "reward_death")->value;
    env->max_snake_length = dict_get(kwargs, "max_snake_length")->value;
    env->cell_size = dict_get(kwargs, "cell_size")->value;
    init_csnake(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}
