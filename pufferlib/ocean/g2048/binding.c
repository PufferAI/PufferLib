#include "g2048.h"

#define Env Game
#include "../env_binding.h"

// g2048.h does not have a 'size' field, so my_init can just return 0
static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    // No custom initialization needed for 2048
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "merge_score", log->merge_score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}