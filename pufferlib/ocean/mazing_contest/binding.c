#include "mazing_contest.h"

#define Env MazingContest
#include "../env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
    env->build_time_limit = unpack(kwargs, "build_time_limit");
    env->max_moves = unpack(kwargs, "max_moves");
    env->max_rounds = unpack(kwargs, "max_rounds");
    env->min_gold = unpack(kwargs, "min_gold");
    env->max_gold = unpack(kwargs, "max_gold");
    env->min_lumber = unpack(kwargs, "min_lumber");
    env->max_lumber = unpack(kwargs, "max_lumber");
    allocate(env);
    return 0;
}

static int my_log(PyObject *dict, Log *log) {
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "best_time", log->best_time);
    assign_to_dict(dict, "path_length_rewards", log->path_length_rewards);
    assign_to_dict(dict, "wall_touch_rewards", log->wall_touch_rewards);
    assign_to_dict(dict, "thunderclap_rewards", log->thunderclap_rewards);
    assign_to_dict(dict, "n", log->n);
    return 0;
}