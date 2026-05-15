#include "light_thief.h"
#define Env LightThief
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    c_reset(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
