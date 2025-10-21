#include "nonogram.h"
#include <Python.h>

// Forward declare custom methods
static PyObject *vec_get_solutions(PyObject *self, PyObject *args);
static PyObject *vec_get_size(PyObject *self, PyObject *args);

#define Env Nonogram
#define MY_METHODS                                                             \
  {"vec_get_solutions", vec_get_solutions, METH_VARARGS,                       \
   "Get solutions from all environments"},                                     \
      {"vec_get_size", vec_get_size, METH_VARARGS, "Get current board size"}

#include "../env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
  env->min_size = unpack(kwargs, "min_size");
  env->max_size = unpack(kwargs, "max_size");
  env->easy_learn = unpack(kwargs, "easy_learn");
  env->size = env->max_size;
  env->max_steps = 4 * env->max_size * env->max_size;
  return 0;
}

static int my_log(PyObject *dict, Log *log) {
  assign_to_dict(dict, "score", log->score);
  assign_to_dict(dict, "episode_return", log->episode_return);
  assign_to_dict(dict, "episode_length", log->episode_length);
  assign_to_dict(dict, "solved", log->solved);
  return 0;
}

// Custom method to get solutions from all environments
static PyObject *vec_get_solutions(PyObject *self, PyObject *args) {
  if (PyTuple_Size(args) != 2) {
    PyErr_SetString(PyExc_TypeError, "vec_get_solutions requires 2 arguments");
    return NULL;
  }

  VecEnv *vec = unpack_vecenv(args);
  if (!vec) {
    return NULL;
  }

  PyObject *solutions_obj = PyTuple_GetItem(args, 1);
  if (!PyObject_TypeCheck(solutions_obj, &PyArray_Type)) {
    PyErr_SetString(PyExc_TypeError, "solutions must be a NumPy array");
    return NULL;
  }
  PyArrayObject *solutions = (PyArrayObject *)solutions_obj;
  if (!PyArray_ISCONTIGUOUS(solutions)) {
    PyErr_SetString(PyExc_ValueError, "solutions must be contiguous");
    return NULL;
  }

  // Copy solutions from each environment (always use max_size for buffer)
  unsigned char *sol_ptr = PyArray_DATA(solutions);
  int max_grid_size = MAX_SIZE * MAX_SIZE;
  for (int i = 0; i < vec->num_envs; i++) {
    Nonogram *env = vec->envs[i];
    memcpy(sol_ptr + i * max_grid_size, env->solution, max_grid_size);
  }

  Py_RETURN_NONE;
}

// Get current board size from first environment
static PyObject *vec_get_size(PyObject *self, PyObject *args) {
  VecEnv *vec = unpack_vecenv(args);
  if (!vec) {
    return NULL;
  }

  Nonogram *env = vec->envs[0];
  return PyLong_FromLong(env->size);
}
