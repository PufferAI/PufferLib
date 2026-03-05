#define BOXOBAN_MAPS_IMPLEMENTATION //enables mmap
#include "boxoban.h"
#define Env Boxoban
#include "../env_binding.h"

//Map stuff
static int update_map_path(PyObject* kwargs) {
    PyObject* map_path_obj = PyDict_GetItemString(kwargs, "map_path");
    if (map_path_obj == NULL || !PyUnicode_Check(map_path_obj)) {
        PyErr_SetString(PyExc_TypeError, "Boxoban requires a string 'map_path' kwarg");
        return -1;
    }

    const char* new_path = PyUnicode_AsUTF8(map_path_obj);
    if (new_path == NULL) {
        return -1;
    }

    if (boxoban_set_map_path(new_path) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to set Boxoban map path");
        return -1;
    }

    return 0;
}


static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    if (update_map_path(kwargs) != 0) {
        return -1;
    }
    env->size = (int)unpack(kwargs, "size");
    env->max_steps = (int)unpack(kwargs, "max_steps");
    env->int_r_coeff = (float)unpack(kwargs, "int_r_coeff");
    env->target_loss_pen_coeff = (float)unpack(kwargs, "target_loss_pen_coeff");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "targets_hit", log->on_targets);
    return 0;
}
