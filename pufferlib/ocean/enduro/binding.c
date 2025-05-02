#include "enduro.h"
#include <Python.h>
#define Env Enduro
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->car_width = unpack(kwargs, "car_width");
    env->car_height = unpack(kwargs, "car_height");
    env->max_enemies = unpack(kwargs, "max_enemies");
    env->continuous = unpack(kwargs, "continuous");

    PyObject* seed_val = PyDict_GetItemString(kwargs, "seed");
    if (seed_val) {
        env->seed = unpack(kwargs, "seed");
        // Initialize the RNG state with the seed
        env->rng_state = env->seed;
    }
    
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "reward", log->reward);
    assign_to_dict(dict, "step_rew_car_passed_no_crash", log->step_rew_car_passed_no_crash);
    assign_to_dict(dict, "crashed_penalty", log->crashed_penalty);
    assign_to_dict(dict, "passed_cars", log->passed_cars);
    assign_to_dict(dict, "passed_by_enemy", log->passed_by_enemy);
    assign_to_dict(dict, "cars_to_pass", log->cars_to_pass);
    assign_to_dict(dict, "days_completed", log->days_completed);
    assign_to_dict(dict, "days_failed", log->days_failed);
    assign_to_dict(dict, "collisions_player_vs_car", log->collisions_player_vs_car);
    assign_to_dict(dict, "collisions_player_vs_road", log->collisions_player_vs_road);
    return 0;
}
