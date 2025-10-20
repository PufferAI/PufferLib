#include "coin_finder.h"

#define Env CoinFinder
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    // For now everything is a constant
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    // Export your log fields to Python
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}