#include <Python.h>
#include "nonogram.h"

// Forward declare custom method
static PyObject* vec_get_solutions(PyObject* self, PyObject* args);

#define Env Nonogram
#define MY_METHODS {"vec_get_solutions", vec_get_solutions, METH_VARARGS, "Get solutions from all environments"}

#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->size = unpack(kwargs, "size");
    env->max_steps = 4 * env->size * env->size;
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}

// Custom method to get solutions from all environments
static PyObject* vec_get_solutions(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_get_solutions requires 2 arguments");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    PyObject* solutions_obj = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(solutions_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "solutions must be a NumPy array");
        return NULL;
    }
    PyArrayObject* solutions = (PyArrayObject*)solutions_obj;
    if (!PyArray_ISCONTIGUOUS(solutions)) {
        PyErr_SetString(PyExc_ValueError, "solutions must be contiguous");
        return NULL;
    }

    // Copy solutions from each environment
    unsigned char* sol_ptr = PyArray_DATA(solutions);
    for (int i = 0; i < vec->num_envs; i++) {
        Nonogram* env = vec->envs[i];
        int grid_size = env->size * env->size;
        memcpy(sol_ptr + i * grid_size, env->solution, grid_size);
    }

    Py_RETURN_NONE;
}
