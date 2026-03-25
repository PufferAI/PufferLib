#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>

#include "clifford.h"

static int require_contiguous_array(PyObject* obj, int typenum, int ndim, const char* name, PyArrayObject** out) {
    if (!PyObject_TypeCheck(obj, &PyArray_Type)) {
        PyErr_Format(PyExc_TypeError, "%s must be a NumPy array", name);
        return 0;
    }
    PyArrayObject* arr = (PyArrayObject*)obj;
    if (!PyArray_ISCONTIGUOUS(arr)) {
        PyErr_Format(PyExc_ValueError, "%s must be contiguous", name);
        return 0;
    }
    if (PyArray_TYPE(arr) != typenum) {
        PyErr_Format(PyExc_TypeError, "%s has unexpected dtype", name);
        return 0;
    }
    if (ndim >= 0 && PyArray_NDIM(arr) != ndim) {
        PyErr_Format(PyExc_ValueError, "%s must have ndim=%d", name, ndim);
        return 0;
    }
    *out = arr;
    return 1;
}

static PyObject* require_kwarg(PyObject* kwargs, const char* name) {
    PyObject* obj = PyDict_GetItemString(kwargs, name);
    if (obj == NULL) {
        PyErr_Format(PyExc_TypeError, "missing required keyword argument '%s'", name);
    }
    return obj;
}

static CliffordVecEnv* unpack_handle(PyObject* args) {
    PyObject* handle_obj = PyTuple_GetItem(args, 0);
    if (!PyLong_Check(handle_obj)) {
        PyErr_SetString(PyExc_TypeError, "handle must be an integer");
        return NULL;
    }
    CliffordVecEnv* vec = (CliffordVecEnv*)PyLong_AsVoidPtr(handle_obj);
    if (vec == NULL) {
        PyErr_SetString(PyExc_ValueError, "invalid native Clifford handle");
        return NULL;
    }
    return vec;
}

static PyObject* py_vec_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;
    if (PyTuple_Size(args) != 7) {
        PyErr_SetString(PyExc_TypeError, "vec_init requires 7 positional arguments");
        return NULL;
    }

    PyArrayObject *obs_arr = NULL, *act_arr = NULL, *rew_arr = NULL, *term_arr = NULL, *trunc_arr = NULL;
    if (!require_contiguous_array(PyTuple_GetItem(args, 0), NPY_UINT8, 2, "observations", &obs_arr)) return NULL;
    if (!require_contiguous_array(PyTuple_GetItem(args, 1), NPY_INT32, 1, "actions", &act_arr)) return NULL;
    if (!require_contiguous_array(PyTuple_GetItem(args, 2), NPY_FLOAT32, 1, "rewards", &rew_arr)) return NULL;
    if (!require_contiguous_array(PyTuple_GetItem(args, 3), NPY_BOOL, 1, "terminals", &term_arr)) return NULL;
    if (!require_contiguous_array(PyTuple_GetItem(args, 4), NPY_BOOL, 1, "truncations", &trunc_arr)) return NULL;

    int num_envs = (int)PyLong_AsLong(PyTuple_GetItem(args, 5));
    int seed = (int)PyLong_AsLong(PyTuple_GetItem(args, 6));
    if (PyErr_Occurred()) {
        return NULL;
    }
    if (num_envs <= 0) {
        PyErr_SetString(PyExc_ValueError, "num_envs must be positive");
        return NULL;
    }

    PyArrayObject *gate_kinds = NULL, *action_q0 = NULL, *action_q1 = NULL;
    if (!require_contiguous_array(require_kwarg(kwargs, "gate_kinds"), NPY_INT32, 1, "gate_kinds", &gate_kinds)) return NULL;
    if (!require_contiguous_array(require_kwarg(kwargs, "action_q0"), NPY_INT32, 1, "action_q0", &action_q0)) return NULL;
    if (!require_contiguous_array(require_kwarg(kwargs, "action_q1"), NPY_INT32, 1, "action_q1", &action_q1)) return NULL;

    int num_actions = (int)PyArray_DIM(gate_kinds, 0);
    if (num_actions <= 0 || PyArray_DIM(action_q0, 0) != num_actions || PyArray_DIM(action_q1, 0) != num_actions) {
        PyErr_SetString(PyExc_ValueError, "action arrays must have the same positive length");
        return NULL;
    }

    int n_qubits = (int)PyLong_AsLong(require_kwarg(kwargs, "n_qubits"));
    double difficulty = PyFloat_AsDouble(require_kwarg(kwargs, "difficulty"));
    int max_steps = (int)PyLong_AsLong(require_kwarg(kwargs, "max_steps"));
    float single_qubit_cost = (float)PyFloat_AsDouble(require_kwarg(kwargs, "single_qubit_cost"));
    float goal_bonus = (float)PyFloat_AsDouble(require_kwarg(kwargs, "goal_bonus"));
    int reward_mode = (int)PyLong_AsLong(require_kwarg(kwargs, "reward_mode"));
    float hamming_left_scale = (float)PyFloat_AsDouble(require_kwarg(kwargs, "hamming_left_scale"));
    PyObject* reset_pool_enabled_obj = PyDict_GetItemString(kwargs, "use_reset_pool");
    int reset_pool_enabled = reset_pool_enabled_obj == NULL ? 1 : (int)PyObject_IsTrue(reset_pool_enabled_obj);
    if (PyErr_Occurred()) {
        return NULL;
    }
    if (n_qubits <= 0 || 2 * n_qubits > 64) {
        PyErr_SetString(PyExc_ValueError, "native Clifford env requires 0 < n_qubits and 2*n_qubits <= 64");
        return NULL;
    }
    if (max_steps <= 0) {
        PyErr_SetString(PyExc_ValueError, "max_steps must be positive");
        return NULL;
    }
    if (reward_mode != REWARD_GATE_COST && reward_mode != REWARD_HAMMING_LEFT) {
        PyErr_SetString(
            PyExc_ValueError,
            "reward_mode must be gate_cost or hamming_left"
        );
        return NULL;
    }
    if (hamming_left_scale < 0.0f) {
        PyErr_SetString(PyExc_ValueError, "hamming_left_scale must be non-negative");
        return NULL;
    }

    const int dim = 2 * n_qubits;
    const int obs_size = dim * dim;
    if (PyArray_DIM(obs_arr, 0) != num_envs || PyArray_DIM(obs_arr, 1) != obs_size) {
        PyErr_SetString(PyExc_ValueError, "observations must have shape (num_envs, (2*n_qubits)^2)");
        return NULL;
    }
    if (PyArray_DIM(act_arr, 0) != num_envs || PyArray_DIM(rew_arr, 0) != num_envs ||
            PyArray_DIM(term_arr, 0) != num_envs || PyArray_DIM(trunc_arr, 0) != num_envs) {
        PyErr_SetString(PyExc_ValueError, "action and buffer arrays must have length num_envs");
        return NULL;
    }

    CliffordVecEnv* vec = (CliffordVecEnv*)calloc(1, sizeof(CliffordVecEnv));
    if (vec == NULL) {
        return PyErr_NoMemory();
    }
    vec->num_envs = num_envs;
    vec->n_qubits = n_qubits;
    vec->dim = dim;
    vec->obs_size = obs_size;
    set_difficulty_level(vec, difficulty);
    vec->max_steps = max_steps;
    vec->num_actions = num_actions;
    vec->reset_pool_enabled = reset_pool_enabled;
    vec->single_qubit_cost = single_qubit_cost;
    vec->goal_bonus = goal_bonus;
    vec->reward_mode = reward_mode;
    vec->hamming_left_scale = hamming_left_scale;
    init_observation_tables();

    vec->actions = (CliffordAction*)calloc((size_t)num_actions, sizeof(CliffordAction));
    vec->envs = (CliffordEnv*)calloc((size_t)num_envs, sizeof(CliffordEnv));
    if (vec->actions == NULL || vec->envs == NULL) {
        free(vec->actions);
        free(vec->envs);
        free(vec);
        return PyErr_NoMemory();
    }

    int* gate_kind_ptr = (int*)PyArray_DATA(gate_kinds);
    int* q0_ptr = (int*)PyArray_DATA(action_q0);
    int* q1_ptr = (int*)PyArray_DATA(action_q1);
    for (int i = 0; i < num_actions; ++i) {
        vec->actions[i].gate_kind = gate_kind_ptr[i];
        vec->actions[i].q0 = q0_ptr[i];
        vec->actions[i].q1 = q1_ptr[i];
    }
    for (int col = 0; col < vec->dim; ++col) {
        vec->identity_cols[col] = 1ULL << col;
    }
    rebuild_reset_pool(vec, (uint64_t)(unsigned int)seed);

    unsigned char* obs_ptr = (unsigned char*)PyArray_DATA(obs_arr);
    int* act_ptr = (int*)PyArray_DATA(act_arr);
    float* rew_ptr = (float*)PyArray_DATA(rew_arr);
    unsigned char* term_ptr = (unsigned char*)PyArray_DATA(term_arr);
    unsigned char* trunc_ptr = (unsigned char*)PyArray_DATA(trunc_arr);
    uint64_t seed_state = (uint64_t)(unsigned int)seed;
    for (int env_idx = 0; env_idx < num_envs; ++env_idx) {
        CliffordEnv* env = &vec->envs[env_idx];
        env->cols = (uint64_t*)calloc((size_t)vec->dim, sizeof(uint64_t));
        if (env->cols == NULL) {
            for (int j = 0; j < env_idx; ++j) {
                free(vec->envs[j].cols);
            }
            free(vec->reset_pool_cols);
            free(vec->actions);
            free(vec->envs);
            free(vec);
            return PyErr_NoMemory();
        }
        env->observations = obs_ptr + (env_idx * vec->obs_size);
        env->actions = act_ptr + env_idx;
        env->rewards = rew_ptr + env_idx;
        env->terminals = term_ptr + env_idx;
        env->truncations = trunc_ptr + env_idx;
        rng_seed(&env->rng, splitmix64_next(&seed_state) ^ (uint64_t)(env_idx + 1));
        reset_single(vec, env);
    }
    return PyLong_FromVoidPtr(vec);
}

static PyObject* py_vec_reset(PyObject* self, PyObject* args) {
    (void)self;
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_reset requires handle and seed");
        return NULL;
    }
    CliffordVecEnv* vec = unpack_handle(args);
    if (vec == NULL) {
        return NULL;
    }
    int seed = (int)PyLong_AsLong(PyTuple_GetItem(args, 1));
    if (PyErr_Occurred()) {
        return NULL;
    }
    if (seed >= 0) {
        rebuild_reset_pool(vec, (uint64_t)(unsigned int)seed);
    }
    uint64_t seed_state = (uint64_t)(unsigned int)seed;
    for (int env_idx = 0; env_idx < vec->num_envs; ++env_idx) {
        if (seed >= 0) {
            rng_seed(&vec->envs[env_idx].rng, splitmix64_next(&seed_state) ^ (uint64_t)(env_idx + 1));
        }
        vec->envs[env_idx].terminals[0] = 0;
        vec->envs[env_idx].truncations[0] = 0;
        vec->envs[env_idx].rewards[0] = 0.0f;
        reset_single(vec, &vec->envs[env_idx]);
    }
    Py_RETURN_NONE;
}

static PyObject* py_vec_step(PyObject* self, PyObject* args) {
    (void)self;
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "vec_step requires handle");
        return NULL;
    }
    CliffordVecEnv* vec = unpack_handle(args);
    if (vec == NULL) {
        return NULL;
    }
    for (int env_idx = 0; env_idx < vec->num_envs; ++env_idx) {
        CliffordEnv* env = &vec->envs[env_idx];
        env->terminals[0] = 0;
        env->truncations[0] = 0;
        env->rewards[0] = 0.0f;
        step_single(vec, env);
    }
    Py_RETURN_NONE;
}

static PyObject* py_vec_close(PyObject* self, PyObject* args) {
    (void)self;
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "vec_close requires handle");
        return NULL;
    }
    CliffordVecEnv* vec = unpack_handle(args);
    if (vec == NULL) {
        return NULL;
    }
    for (int env_idx = 0; env_idx < vec->num_envs; ++env_idx) {
        free(vec->envs[env_idx].cols);
    }
    free(vec->reset_pool_cols);
    free(vec->actions);
    free(vec->envs);
    free(vec);
    Py_RETURN_NONE;
}

static PyObject* py_vec_set_difficulty(PyObject* self, PyObject* args) {
    (void)self;
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_set_difficulty requires handle and difficulty");
        return NULL;
    }
    CliffordVecEnv* vec = unpack_handle(args);
    if (vec == NULL) {
        return NULL;
    }
    double difficulty = PyFloat_AsDouble(PyTuple_GetItem(args, 1));
    if (PyErr_Occurred()) {
        return NULL;
    }
    set_difficulty_level(vec, difficulty);
    rebuild_reset_pool(vec, rng_next_u64(&vec->pool_rng));
    Py_RETURN_NONE;
}

static PyObject* py_vec_set_max_steps(PyObject* self, PyObject* args) {
    (void)self;
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_set_max_steps requires handle and max_steps");
        return NULL;
    }
    CliffordVecEnv* vec = unpack_handle(args);
    if (vec == NULL) {
        return NULL;
    }
    vec->max_steps = (int)PyLong_AsLong(PyTuple_GetItem(args, 1));
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* py_vec_set_matrix(PyObject* self, PyObject* args) {
    (void)self;
    if (PyTuple_Size(args) != 3) {
        PyErr_SetString(PyExc_TypeError, "vec_set_matrix requires handle, env_index, matrix");
        return NULL;
    }
    CliffordVecEnv* vec = unpack_handle(args);
    if (vec == NULL) {
        return NULL;
    }
    int env_index = (int)PyLong_AsLong(PyTuple_GetItem(args, 1));
    if (PyErr_Occurred()) {
        return NULL;
    }
    if (env_index < 0 || env_index >= vec->num_envs) {
        PyErr_SetString(PyExc_IndexError, "env_index out of range");
        return NULL;
    }
    PyArrayObject* matrix = NULL;
    if (!require_contiguous_array(PyTuple_GetItem(args, 2), NPY_UINT8, 2, "matrix", &matrix)) {
        return NULL;
    }
    if (!set_matrix_from_dense(
            vec,
            &vec->envs[env_index],
            (unsigned char*)PyArray_DATA(matrix),
            (int)PyArray_DIM(matrix, 0),
            (int)PyArray_DIM(matrix, 1))) {
        PyErr_SetString(PyExc_ValueError, "matrix shape does not match native Clifford env");
        return NULL;
    }
    vec->envs[env_index].rewards[0] = 0.0f;
    vec->envs[env_index].terminals[0] = 0;
    vec->envs[env_index].truncations[0] = 0;
    Py_RETURN_NONE;
}

static PyObject* py_vec_log(PyObject* self, PyObject* args) {
    (void)self;
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "vec_log requires handle");
        return NULL;
    }
    CliffordVecEnv* vec = unpack_handle(args);
    if (vec == NULL) {
        return NULL;
    }
    PyObject* out = PyDict_New();
    if (vec->log.n <= 0.0f) {
        return out;
    }
    const float inv_n = 1.0f / vec->log.n;
    PyDict_SetItemString(out, "perf", PyFloat_FromDouble(vec->log.perf * inv_n));
    PyDict_SetItemString(out, "score", PyFloat_FromDouble(vec->log.score * inv_n));
    PyDict_SetItemString(out, "episode_return", PyFloat_FromDouble(vec->log.episode_return * inv_n));
    PyDict_SetItemString(out, "episode_length", PyFloat_FromDouble(vec->log.episode_length * inv_n));
    PyDict_SetItemString(out, "mean_cz", PyFloat_FromDouble(vec->log.episode_cz_sum * inv_n));
    PyDict_SetItemString(out, "success_rate", PyFloat_FromDouble(vec->log.success_rate * inv_n));
    PyDict_SetItemString(out, "n", PyFloat_FromDouble(vec->log.n));
    PyDict_SetItemString(out, "episode_count", PyFloat_FromDouble(vec->log.n));
    PyDict_SetItemString(out, "episode_return_sum", PyFloat_FromDouble(vec->log.episode_return));
    PyDict_SetItemString(out, "episode_length_sum", PyFloat_FromDouble(vec->log.episode_length));
    PyDict_SetItemString(out, "episode_cz_sum", PyFloat_FromDouble(vec->log.episode_cz_sum));
    PyDict_SetItemString(out, "episode_cz_max", PyFloat_FromDouble(vec->log.episode_cz_max));
    PyDict_SetItemString(out, "success_count", PyFloat_FromDouble(vec->log.success_count));
    PyDict_SetItemString(out, "success_step_sum", PyFloat_FromDouble(vec->log.success_step_sum));
    PyDict_SetItemString(out, "success_step_sq_sum", PyFloat_FromDouble(vec->log.success_step_sq_sum));
    PyDict_SetItemString(
        out,
        "mean_success_cz",
        PyFloat_FromDouble(vec->log.success_count > 0.0f ? (vec->log.success_cz_sum / vec->log.success_count) : 0.0)
    );
    PyDict_SetItemString(out, "success_cz_sum", PyFloat_FromDouble(vec->log.success_cz_sum));
    PyDict_SetItemString(out, "success_cz_max", PyFloat_FromDouble(vec->log.success_cz_max));
    PyDict_SetItemString(out, "success_step_min", PyFloat_FromDouble(vec->log.success_step_min));
    PyDict_SetItemString(out, "success_step_max", PyFloat_FromDouble(vec->log.success_step_max));
    memset(&vec->log, 0, sizeof(CliffordLog));
    return out;
}

static PyMethodDef MODULE_METHODS[] = {
    {"vec_init", (PyCFunction)py_vec_init, METH_VARARGS | METH_KEYWORDS, "Create a native Clifford vec env"},
    {"vec_reset", py_vec_reset, METH_VARARGS, "Reset the native Clifford vec env"},
    {"vec_step", py_vec_step, METH_VARARGS, "Step the native Clifford vec env"},
    {"vec_close", py_vec_close, METH_VARARGS, "Close the native Clifford vec env"},
    {"vec_log", py_vec_log, METH_VARARGS, "Aggregate native Clifford logs"},
    {"vec_set_difficulty", py_vec_set_difficulty, METH_VARARGS, "Update native Clifford difficulty"},
    {"vec_set_max_steps", py_vec_set_max_steps, METH_VARARGS, "Update native Clifford max_steps"},
    {"vec_set_matrix", py_vec_set_matrix, METH_VARARGS, "Set the matrix for one native Clifford env"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef MODULE_DEF = {
    PyModuleDef_HEAD_INIT,
    "binding",
    "Native Clifford environment binding",
    -1,
    MODULE_METHODS,
};

PyMODINIT_FUNC PyInit_binding(void) {
    import_array();
    return PyModule_Create(&MODULE_DEF);
}
