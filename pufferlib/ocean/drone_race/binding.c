#include "drone_race.h"

#define Env DroneRace
#include "../env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
    env->max_rings = unpack(kwargs, "max_rings");
    env->max_moves = unpack(kwargs, "max_moves");
    init(env);
    return 0;
}

static int my_log(PyObject *dict, Log *log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "collision_rate", log->collision_rate);
    assign_to_dict(dict, "oob", log->oob);
    assign_to_dict(dict, "timeout", log->timeout);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "n", log->n);
    return 0;
}
