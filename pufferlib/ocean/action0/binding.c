#include "action0.h"

#define Env Action0
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->horizon = (int)unpack(kwargs, "horizon");
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    return 0;
}
