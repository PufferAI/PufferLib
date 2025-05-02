#include "grid.h"

#define Env Grid 
#define MY_SHARED
#include "../env_binding.h"

static PyObject* my_shared(PyObject* self, PyObject* args, PyObject* kwargs) {
    int num_maps = unpack(kwargs, "num_maps");
    int max_size = unpack(kwargs, "max_size");
    int size = unpack(kwargs, "size");
    State* levels = calloc(num_maps, sizeof(State));

    if (max_size <= 5) {
        PyErr_SetString(PyExc_ValueError, "max_size must be >5");
        return NULL;
    }

    // Temporary env used to gen maps
    Grid env;
    env.max_size = max_size;
    init_grid(&env);

    for (int i = 0; i < num_maps; i++) {
        int sz = size;
        if (size == -1) {
            sz = 5 + (rand() % (max_size-5));
        }

        if (sz % 2 == 0) {
            sz -= 1;
        }

        float difficulty = (float)rand()/(float)(RAND_MAX);
        create_maze_level(&env, sz, sz, difficulty, i);
        init_state(&levels[i], max_size, 1);
        get_state(&env, &levels[i]);
    }

    return PyLong_FromVoidPtr(levels);
}

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->max_size = unpack(kwargs, "max_size");
    env->num_maps = unpack(kwargs, "num_maps");
    init_grid(env);

    PyObject* handle_obj = PyDict_GetItemString(kwargs, "state");
    if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "state handle must be an integer");
        return 1;
    }

    State* levels = (State*)PyLong_AsVoidPtr(handle_obj);
    if (!levels) {
        PyErr_SetString(PyExc_ValueError, "Invalid state handle");
        return 1;
    }

    env->levels = levels;
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
