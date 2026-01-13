#include "dogfight.h"

#define Env Dogfight
#include "../env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
    env->max_steps = unpack(kwargs, "max_steps");
    init(env);
    return 0;
}

static int my_log(PyObject *dict, Log *log) {
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "kills", log->kills);
    assign_to_dict(dict, "deaths", log->deaths);
    assign_to_dict(dict, "shots_fired", log->shots_fired);
    assign_to_dict(dict, "shots_hit", log->shots_hit);
    assign_to_dict(dict, "n", log->n);
    return 0;
}
