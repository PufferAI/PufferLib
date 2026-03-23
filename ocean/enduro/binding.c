#include "enduro.h"
#define OBS_SIZE 68
#define NUM_ATNS 1
#define ACT_SIZES {9}
#define OBS_TYPE FLOAT
#define ACT_TYPE DOUBLE

#define Env Enduro
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->car_width = dict_get(kwargs, "car_width")->value;
    env->car_height = dict_get(kwargs, "car_height")->value;
    env->max_enemies = dict_get(kwargs, "max_enemies")->value;
    env->continuous = dict_get(kwargs, "continuous")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "reward", log->reward);
    dict_set(out, "step_rew_car_passed_no_crash", log->step_rew_car_passed_no_crash);
    dict_set(out, "crashed_penalty", log->crashed_penalty);
    dict_set(out, "passed_cars", log->passed_cars);
    dict_set(out, "passed_by_enemy", log->passed_by_enemy);
    dict_set(out, "cars_to_pass", log->cars_to_pass);
    dict_set(out, "days_completed", log->days_completed);
    dict_set(out, "days_failed", log->days_failed);
    dict_set(out, "collisions_player_vs_car", log->collisions_player_vs_car);
    dict_set(out, "collisions_player_vs_road", log->collisions_player_vs_road);
}
