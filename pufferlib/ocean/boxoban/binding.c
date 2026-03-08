#define BOXOBAN_MAPS_IMPLEMENTATION //enables mmap
#include "boxoban.h"
#define Env Boxoban
#include "../env_binding.h"

static int parse_difficulty_id(PyObject* kwargs, int* out_difficulty_id) {
    int difficulty_id = 0;
    PyObject* difficulty_obj = PyDict_GetItemString(kwargs, "difficulty");
    if (difficulty_obj != NULL) {
        if (PyLong_Check(difficulty_obj)) {
            long parsed_id = PyLong_AsLong(difficulty_obj);
            if (boxoban_difficulty_name_from_id((int)parsed_id) == NULL) {
                PyErr_Format(
                    PyExc_ValueError,
                    "Boxoban 'difficulty' int must be in [0, 4], got %ld (0=basic, 1=easy, 2=medium, 3=hard, 4=unfiltered)",
                    parsed_id
                );
                return -1;
            }
            difficulty_id = (int)parsed_id;
        } else if (PyUnicode_Check(difficulty_obj)) {
            const char* difficulty_name = PyUnicode_AsUTF8(difficulty_obj);
            if (difficulty_name == NULL) {
                return -1;
            }
            difficulty_id = boxoban_difficulty_id_from_name(difficulty_name);
            if (difficulty_id < 0) {
                PyErr_Format(
                    PyExc_ValueError,
                    "Boxoban 'difficulty' string must be one of: basic, easy, medium, hard, unfiltered (got '%s')",
                    difficulty_name
                );
                return -1;
            }
        } else {
            PyErr_SetString(
                PyExc_TypeError,
                "Boxoban 'difficulty' must be an int (0..4) or string (basic/easy/medium/hard/unfiltered)"
            );
            return -1;
        }
    }
    *out_difficulty_id = difficulty_id;
    return 0;
}

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    if (parse_difficulty_id(kwargs, &env->difficulty_id) != 0) {
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
