#include "tactical.h"
#define Env Tactical
#include "../env_binding.h"

// no init args needed
static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    return 0;
}

// no logging implemented atm
static int my_log(PyObject* dict, Log* log) {
    return 0;
}
