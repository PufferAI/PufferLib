#include <Python.h>
#include <numpy/arrayobject.h>

#include <pthread.h>
#include <stdatomic.h>

// Forward declarations for env-specific functions supplied by user
static int my_log(PyObject* dict, Log* log);
static int my_init(Env* env, PyObject* args, PyObject* kwargs);

static PyObject* my_shared(PyObject* self, PyObject* args, PyObject* kwargs);
#ifndef MY_SHARED
static PyObject* my_shared(PyObject* self, PyObject* args, PyObject* kwargs) {
    return NULL;
}
#endif

static PyObject* my_get(PyObject* dict, Env* env);
#ifndef MY_GET
static PyObject* my_get(PyObject* dict, Env* env) {
    return NULL;
}
#endif

static int my_put(Env* env, PyObject* args, PyObject* kwargs);
#ifndef MY_PUT
static int my_put(Env* env, PyObject* args, PyObject* kwargs) {
    return 0;
}
#endif

#ifndef MY_METHODS
#define MY_METHODS {NULL, NULL, 0, NULL}
#endif

static Env* unpack_env(PyObject* args) {
    PyObject* handle_obj = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "env_handle must be an integer");
        return NULL;
    }

    Env* env = (Env*)PyLong_AsVoidPtr(handle_obj);
    if (!env) {
        PyErr_SetString(PyExc_ValueError, "Invalid env handle");
        return NULL;
    }

    return env;
}

// Python function to initialize the environment
static PyObject* env_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    if (PyTuple_Size(args) != 6) {
        PyErr_SetString(PyExc_TypeError, "Environment requires 5 arguments");
        return NULL;
    }

    Env* env = (Env*)calloc(1, sizeof(Env));
    if (!env) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate environment");
        return NULL;
    }

    PyObject* obs = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(obs, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Observations must be a NumPy array");
        return NULL;
    }
    PyArrayObject* observations = (PyArrayObject*)obs;
    if (!PyArray_ISCONTIGUOUS(observations)) {
        PyErr_SetString(PyExc_ValueError, "Observations must be contiguous");
        return NULL;
    }
    env->observations = PyArray_DATA(observations);

    PyObject* act = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(act, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Actions must be a NumPy array");
        return NULL;
    }
    PyArrayObject* actions = (PyArrayObject*)act;
    if (!PyArray_ISCONTIGUOUS(actions)) {
        PyErr_SetString(PyExc_ValueError, "Actions must be contiguous");
        return NULL;
    }
    env->actions = PyArray_DATA(actions);
    if (PyArray_ITEMSIZE(actions) == sizeof(double)) {
        PyErr_SetString(PyExc_ValueError, "Action tensor passed as float64 (pass np.float32 buffer)");
        return NULL;
    }

    PyObject* rew = PyTuple_GetItem(args, 2);
    if (!PyObject_TypeCheck(rew, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Rewards must be a NumPy array");
        return NULL;
    }
    PyArrayObject* rewards = (PyArrayObject*)rew;
    if (!PyArray_ISCONTIGUOUS(rewards)) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(rewards) != 1) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be 1D");
        return NULL;
    }
    env->rewards = PyArray_DATA(rewards);

    PyObject* term = PyTuple_GetItem(args, 3);
    if (!PyObject_TypeCheck(term, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Terminals must be a NumPy array");
        return NULL;
    }
    PyArrayObject* terminals = (PyArrayObject*)term;
    if (!PyArray_ISCONTIGUOUS(terminals)) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(terminals) != 1) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be 1D");
        return NULL;
    }
    env->terminals = PyArray_DATA(terminals);

    PyObject* trunc = PyTuple_GetItem(args, 4);
    if (!PyObject_TypeCheck(trunc, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Truncations must be a NumPy array");
        return NULL;
    }
    PyArrayObject* truncations = (PyArrayObject*)trunc;
    if (!PyArray_ISCONTIGUOUS(truncations)) {
        PyErr_SetString(PyExc_ValueError, "Truncations must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(truncations) != 1) {
        PyErr_SetString(PyExc_ValueError, "Truncations must be 1D");
        return NULL;
    }
    // env->truncations = PyArray_DATA(truncations);
    
    
    PyObject* seed_arg = PyTuple_GetItem(args, 5);
    if (!PyObject_TypeCheck(seed_arg, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "seed must be an integer");
        return NULL;
    }
    int seed = PyLong_AsLong(seed_arg);
 
    // Assumes each process has the same number of environments
    srand(seed);

    // If kwargs is NULL, create a new dictionary
    if (kwargs == NULL) {
        kwargs = PyDict_New();
    } else {
        Py_INCREF(kwargs);  // We need to increment the reference since we'll be modifying it
    }

    // Add the seed to kwargs
    PyObject* py_seed = PyLong_FromLong(seed);
    if (PyDict_SetItemString(kwargs, "seed", py_seed) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to set seed in kwargs");
        Py_DECREF(py_seed);
        Py_DECREF(kwargs);
        return NULL;
    }
    Py_DECREF(py_seed);

    PyObject* empty_args = PyTuple_New(0);
    my_init(env, empty_args, kwargs);
    Py_DECREF(kwargs);
    if (PyErr_Occurred()) {
        return NULL;
    }

    return PyLong_FromVoidPtr(env);
}

// Python function to reset the environment
static PyObject* env_reset(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "env_reset requires 2 arguments");
        return NULL;
    }

    Env* env = unpack_env(args);
    if (!env){
        return NULL;
    }
    c_reset(env);
    Py_RETURN_NONE;
}

// Python function to step the environment
static PyObject* env_step(PyObject* self, PyObject* args) {
    int num_args = PyTuple_Size(args);
    if (num_args != 1) {
        PyErr_SetString(PyExc_TypeError, "env_step requires 1 argument");
        return NULL;
    }

    Env* env = unpack_env(args);
    if (!env){
        return NULL;
    }
    c_step(env);
    Py_RETURN_NONE;
}

// Python function to step the environment
static PyObject* env_render(PyObject* self, PyObject* args) {
    Env* env = unpack_env(args);
    if (!env){
        return NULL;
    }
    c_render(env);
    Py_RETURN_NONE;
}

// Python function to close the environment
static PyObject* env_close(PyObject* self, PyObject* args) {
    Env* env = unpack_env(args);
    if (!env){
        return NULL;
    }
    c_close(env);
    free(env);
    Py_RETURN_NONE;
}

static PyObject* env_get(PyObject* self, PyObject* args) {
    Env* env = unpack_env(args);
    if (!env){
        return NULL;
    }
    PyObject* dict = PyDict_New();
    my_get(dict, env);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return dict;
}

static PyObject* env_put(PyObject* self, PyObject* args, PyObject* kwargs) {
    int num_args = PyTuple_Size(args);
    if (num_args != 1) {
        PyErr_SetString(PyExc_TypeError, "env_put requires 1 positional argument");
        return NULL;
    }

    Env* env = unpack_env(args);
    if (!env){
        return NULL;
    }

    PyObject* empty_args = PyTuple_New(0);
    my_put(env, empty_args, kwargs);
    if (PyErr_Occurred()) {
        return NULL;
    }

    Py_RETURN_NONE;
}


typedef struct
{
    atomic_int work_index;
    atomic_int num_running_threads;
    volatile int num_threads;
    pthread_cond_t wake_cnd;
    pthread_t* threads;
} ThreadData;

typedef struct {
    Env** envs;
    int num_envs;
    ThreadData* thread_data;
} VecEnv;

static int global_num_threads = 0;

// Main worker thread; initializes itself and runs a tight loop running through c_step (after waiting for work signal).
static void* c_threadstep(void* arg)
{
    VecEnv* vec_env = (VecEnv*)arg;

    pthread_mutex_t mtx;
    pthread_mutex_init(&mtx, NULL);
    pthread_cond_t* wake = &vec_env->thread_data->wake_cnd;

    atomic_int* work_index = &vec_env->thread_data->work_index;
    atomic_int* num_running_threads = &vec_env->thread_data->num_running_threads;
    volatile int* num_threads = &vec_env->thread_data->num_threads;
    int index;
    atomic_fetch_add(num_running_threads, 1);
    while (1)
    {
        // Wait for work
        pthread_mutex_lock(&mtx);
        pthread_cond_wait(wake, &mtx);
        pthread_mutex_unlock(&mtx);

        if (*num_threads <= 0) { break; } // Exit thread gracefully.

        // Got work to do now.
        atomic_fetch_add(num_running_threads, 1);
        do
        {
            // This is important: Go do a bunch of work in our thread, without context switches or locks
            // or any new allocs. This is the main speedup and core to ensuring the threads do as little work
            // as part of their main loop as possible. We can afford to this as the load balancing naturally happens
            // with mutually exclusive index values spread across threads.
            index = atomic_fetch_sub(work_index, 1);
            if (index >= 0) { c_step(vec_env->envs[index]); }
        }
        while (index > 0);
        atomic_fetch_sub(num_running_threads, 1);
    }
    pthread_mutex_destroy(&mtx);
    return NULL;
}

// Waits for and exits all threads (if needed).
static void c_vecclose(VecEnv* vec_env)
{
    if (global_num_threads <= 2 || vec_env->num_envs <= 2 || !vec_env->thread_data || vec_env->thread_data->num_threads == 0) { return; }
    if (vec_env->thread_data->threads)
    {
        int num_threads = vec_env->thread_data->num_threads;
        atomic_store(&vec_env->thread_data->work_index, -1);
        vec_env->thread_data->num_threads = 0; // Signal to threads to exit
        pthread_cond_broadcast(&vec_env->thread_data->wake_cnd);
        // Wait for them to exit.
        while (atomic_load(&vec_env->thread_data->num_running_threads) > 0) {}

        for (int i = 0; i < num_threads; ++i)
        {
            pthread_join(vec_env->thread_data->threads[i], NULL);
        }
        pthread_cond_destroy(&vec_env->thread_data->wake_cnd);
        free(vec_env->thread_data->threads);
        vec_env->thread_data->threads = NULL;
    }
    free(vec_env->thread_data);
}

// Inits multi-threading if enabled via vec_enable_mt.
static int c_vecinit(VecEnv* vec_env)
{
    // If we have only a couple envs, it's not worth parallelizing. Also, don't penalize the user as they
    // may want to change the .ini dynamically without having to worry about this.
    if (global_num_threads <= 2 || vec_env->num_envs <= 2)
    {
        global_num_threads = 0;
        return 1;
    }
    // NOTE: On failure, we may have sem-initialized state - but it's okay because we will quit the entire program at that point.  
    vec_env->thread_data = (ThreadData*)calloc(1, sizeof(ThreadData));
    vec_env->thread_data->num_threads = global_num_threads;
    vec_env->thread_data->threads = (pthread_t*)calloc(vec_env->thread_data->num_threads, sizeof(pthread_t));
    if (!vec_env->thread_data->threads) { return 0; }
    if (pthread_cond_init(&vec_env->thread_data->wake_cnd, NULL) != 0) { return 0; }
    atomic_store(&vec_env->thread_data->num_running_threads, 0);
    atomic_store(&vec_env->thread_data->work_index, -1);

    for (int i = 0; i < vec_env->thread_data->num_threads; ++i)
    {
        if (pthread_create(&vec_env->thread_data->threads[i], NULL, c_threadstep, vec_env) != 0) { return 0; }
    }

    // Wait for all threads to initialize (okay to busy wait here).
    while (atomic_load(&vec_env->thread_data->num_running_threads) < vec_env->thread_data->num_threads) {}
    atomic_store_explicit(&vec_env->thread_data->num_running_threads, 0, memory_order_relaxed);
    return 1;
}

// Signals worker threads to step across all environments. This is called from the main thread.
// NOTE: Also uses the main thread to avoid having a signal/wait object.
static int c_vecstep(VecEnv* vec_env)
{
    if (vec_env->thread_data->num_threads == 0 || atomic_load(&vec_env->thread_data->work_index) >= 0) { return 0; }

    // Produce work for the worker threads.
    atomic_int* work_index = &vec_env->thread_data->work_index;
    atomic_store_explicit(work_index, vec_env->num_envs - 1, memory_order_relaxed);

    // Signal to other threads that there is new work to be done.
    pthread_cond_broadcast(&vec_env->thread_data->wake_cnd);

    // Why waste a (main) thread? (Also no need for a lock/condition variable etc).
    int index;
    do
    {
        index = atomic_fetch_sub(work_index, 1);
        if (index >= 0) { c_step(vec_env->envs[index]); }
    }
    while (index > 0);

    // Wait for all threads to finish fully.
    // TODO(perumaal): I think this is a bad idea though - we should never spin CPU cycles busy waiting. 
    //      This is a simple initial solution and assumes SIMD-like work happening in the worker threads
    //      which significantly reduces the chance of busy waiting here.
    while (atomic_load(&vec_env->thread_data->num_running_threads) > 0) {}

    return 1;
}

static VecEnv* unpack_vecenv(PyObject* args) {
    PyObject* handle_obj = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "env_handle must be an integer");
        return NULL;
    }

    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(handle_obj);
    if (!vec) {
        PyErr_SetString(PyExc_ValueError, "Missing or invalid vec env handle");
        return NULL;
    }

    if (vec->num_envs <= 0) {
        PyErr_SetString(PyExc_ValueError, "Missing or invalid vec env handle");
        return NULL;
    }

    return vec;
}

static PyObject* vec_enable_mt(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "vec_enable_mt requires 1 arguments");
        return NULL;
    }

    PyObject* num_threads_arg = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(num_threads_arg, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "num_threads_arg must be an integer");
        return NULL;
    }
    global_num_threads = PyLong_AsLong(num_threads_arg);
    Py_RETURN_NONE;
}



static PyObject* vec_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    if (PyTuple_Size(args) != 7) {
        PyErr_SetString(PyExc_TypeError, "vec_init requires 6 arguments");
        return NULL;
    }

    VecEnv* vec = (VecEnv*)calloc(1, sizeof(VecEnv));
    if (!vec) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate vec env");
        return NULL;
    }
    PyObject* num_envs_arg = PyTuple_GetItem(args, 5);
    if (!PyObject_TypeCheck(num_envs_arg, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "num_envs must be an integer");
        return NULL;
    }
    int num_envs = PyLong_AsLong(num_envs_arg);
    if (num_envs <= 0) {
        PyErr_SetString(PyExc_TypeError, "num_envs must be greater than 0");
        return NULL;
    }
    vec->num_envs = num_envs;
    vec->envs = (Env**)calloc(num_envs, sizeof(Env*));
    if (!vec->envs) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate vec env");
        return NULL;
    }

    PyObject* seed_obj = PyTuple_GetItem(args, 6);
    if (!PyObject_TypeCheck(seed_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "seed must be an integer");
        return NULL;
    }
    int seed = PyLong_AsLong(seed_obj);

    PyObject* obs = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(obs, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Observations must be a NumPy array");
        return NULL;
    }
    PyArrayObject* observations = (PyArrayObject*)obs;
    if (!PyArray_ISCONTIGUOUS(observations)) {
        PyErr_SetString(PyExc_ValueError, "Observations must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(observations) < 2) {
        PyErr_SetString(PyExc_ValueError, "Batched Observations must be at least 2D");
        return NULL;
    }

    PyObject* act = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(act, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Actions must be a NumPy array");
        return NULL;
    }
    PyArrayObject* actions = (PyArrayObject*)act;
    if (!PyArray_ISCONTIGUOUS(actions)) {
        PyErr_SetString(PyExc_ValueError, "Actions must be contiguous");
        return NULL;
    }
    if (PyArray_ITEMSIZE(actions) == sizeof(double)) {
        PyErr_SetString(PyExc_ValueError, "Action tensor passed as float64 (pass np.float32 buffer)");
        return NULL;
    }

    PyObject* rew = PyTuple_GetItem(args, 2);
    if (!PyObject_TypeCheck(rew, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Rewards must be a NumPy array");
        return NULL;
    }
    PyArrayObject* rewards = (PyArrayObject*)rew;
    if (!PyArray_ISCONTIGUOUS(rewards)) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(rewards) != 1) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be 1D");
        return NULL;
    }

    PyObject* term = PyTuple_GetItem(args, 3);
    if (!PyObject_TypeCheck(term, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Terminals must be a NumPy array");
        return NULL;
    }
    PyArrayObject* terminals = (PyArrayObject*)term;
    if (!PyArray_ISCONTIGUOUS(terminals)) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(terminals) != 1) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be 1D");
        return NULL;
    }

    PyObject* trunc = PyTuple_GetItem(args, 4);
    if (!PyObject_TypeCheck(trunc, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Truncations must be a NumPy array");
        return NULL;
    }
    PyArrayObject* truncations = (PyArrayObject*)trunc;
    if (!PyArray_ISCONTIGUOUS(truncations)) {
        PyErr_SetString(PyExc_ValueError, "Truncations must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(truncations) != 1) {
        PyErr_SetString(PyExc_ValueError, "Truncations must be 1D");
        return NULL;
    }

    // If kwargs is NULL, create a new dictionary
    if (kwargs == NULL) {
        kwargs = PyDict_New();
    } else {
        Py_INCREF(kwargs);  // We need to increment the reference since we'll be modifying it
    }
    for (int i = 0; i < num_envs; i++) {
        Env* env = (Env*)calloc(1, sizeof(Env));
        if (!env) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate environment");
            Py_DECREF(kwargs);
            return NULL;
        }
        vec->envs[i] = env;
        
        // // Make sure the log is initialized to 0
        memset(&env->log, 0, sizeof(Log));
        
        env->observations = (void*)((char*)PyArray_DATA(observations) + i*PyArray_STRIDE(observations, 0));
        env->actions = (void*)((char*)PyArray_DATA(actions) + i*PyArray_STRIDE(actions, 0));
        env->rewards = (void*)((char*)PyArray_DATA(rewards) + i*PyArray_STRIDE(rewards, 0));
        env->terminals = (void*)((char*)PyArray_DATA(terminals) + i*PyArray_STRIDE(terminals, 0));
        // env->truncations = (void*)((char*)PyArray_DATA(truncations) + i*PyArray_STRIDE(truncations, 0));

        // Assumes each process has the same number of environments
        int env_seed = i + seed*vec->num_envs;
        srand(env_seed);
 
        // Add the seed to kwargs for this environment
        PyObject* py_seed = PyLong_FromLong(env_seed);
        if (PyDict_SetItemString(kwargs, "seed", py_seed) < 0) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to set seed in kwargs");
            Py_DECREF(py_seed);
            Py_DECREF(kwargs);
            return NULL;
        }
        Py_DECREF(py_seed);

        PyObject* empty_args = PyTuple_New(0);
        my_init(env, empty_args, kwargs);
        if (PyErr_Occurred()) {
            return NULL;
        }
    }
    if (!c_vecinit(vec)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize vec env threads");
        return NULL;
    }

    Py_DECREF(kwargs);
    return PyLong_FromVoidPtr(vec);
}


// Python function to vectorize an array of enviroments and return a strong pointer 
// to an internal structure (VecEnv) for use later.
static PyObject* vectorize(PyObject* self, PyObject* args) {
    int num_envs = PyTuple_Size(args);
    if (num_envs == 0) {
        PyErr_SetString(PyExc_TypeError, "make_vec requires at least 1 env id");
        return NULL;
    }

    VecEnv* vec = (VecEnv*)calloc(1, sizeof(VecEnv));
    if (!vec) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate vec env");
        return NULL;
    }

    vec->envs = (Env**)calloc(num_envs, sizeof(Env*));
    if (!vec->envs) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate vec env");
        return NULL;
    }

    vec->num_envs = num_envs;
    for (int i = 0; i < num_envs; i++) {
        PyObject* handle_obj = PyTuple_GetItem(args, i);
        if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
            PyErr_SetString(PyExc_TypeError, "Env ids must be integers. Pass them as separate args with *env_ids, not as a list.");
            return NULL;
        }
        vec->envs[i] = (Env*)PyLong_AsVoidPtr(handle_obj);
    }
    if (!c_vecinit(vec)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize vec env threads");
        return NULL;
    }
    return PyLong_FromVoidPtr(vec);
}

static PyObject* vec_reset(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_reset requires 2 arguments");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    PyObject* seed_arg = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(seed_arg, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "seed must be an integer");
        return NULL;
    }
    int seed = PyLong_AsLong(seed_arg);

    // TODO(perumaal): Should this be multi-thread aware as well? (see vec_step below).
    // Main issue is that srand is not thread-safe. But do we care?
    for (int i = 0; i < vec->num_envs; i++) {
        // Assumes each process has the same number of environments
        srand(i + seed*vec->num_envs);
        c_reset(vec->envs[i]);
    }
    Py_RETURN_NONE;
}

static PyObject* vec_step(PyObject* self, PyObject* arg) {
    int num_args = PyTuple_Size(arg);
    if (num_args != 1) {
        PyErr_SetString(PyExc_TypeError, "vec_step requires 1 argument");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(arg);
    if (!vec) {
        return NULL;
    }
    if (global_num_threads > 2) {
        c_vecstep(vec);
    }
    else {
        for (int i = 0; i < vec->num_envs; i++) {
            c_step(vec->envs[i]);
        }
    }
    Py_RETURN_NONE;
}

static PyObject* vec_render(PyObject* self, PyObject* args) {
    int num_args = PyTuple_Size(args);
    if (num_args != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_render requires 2 arguments");
        return NULL;
    }

    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
    if (!vec) {
        PyErr_SetString(PyExc_ValueError, "Invalid vec_env handle");
        return NULL;
    }

    PyObject* env_id_arg = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(env_id_arg, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "env_id must be an integer");
        return NULL;
    }
    int env_id = PyLong_AsLong(env_id_arg);
 
    c_render(vec->envs[env_id]);
    Py_RETURN_NONE;
}

static int assign_to_dict(PyObject* dict, char* key, float value) {
    PyObject* v = PyFloat_FromDouble(value);
    if (v == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert log value");
        return 1;
    }
    if(PyDict_SetItemString(dict, key, v) < 0) {
        PyErr_SetString(PyExc_TypeError, "Failed to set log value");
        return 1;
    }
    Py_DECREF(v);
    return 0;
}

static PyObject* vec_log(PyObject* self, PyObject* args) {
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    // Iterates over logs one float at a time. Will break
    // horribly if Log has non-float data.
    Log aggregate = {0};
    int num_keys = sizeof(Log) / sizeof(float);
    for (int i = 0; i < vec->num_envs; i++) {
        Env* env = vec->envs[i];
        for (int j = 0; j < num_keys; j++) {
            ((float*)&aggregate)[j] += ((float*)&env->log)[j];
            ((float*)&env->log)[j] = 0.0f;
        }
    }

    PyObject* dict = PyDict_New();
    if (aggregate.n == 0.0f) {
        return dict;
    }

    // Average
    float n = aggregate.n;
    for (int i = 0; i < num_keys; i++) {
        ((float*)&aggregate)[i] /= n;
    }

    // User populates dict
    my_log(dict, &aggregate);
    assign_to_dict(dict, "n", n);

    return dict;
}

static PyObject* vec_close(PyObject* self, PyObject* args) {
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    c_vecclose(vec);
    for (int i = 0; i < vec->num_envs; i++) {
        c_close(vec->envs[i]);
        free(vec->envs[i]);
    }
    free(vec->envs);
    free(vec);
    Py_RETURN_NONE;
}

static double unpack(PyObject* kwargs, char* key) {
    PyObject* val = PyDict_GetItemString(kwargs, key);
    if (val == NULL) {
        char error_msg[100];
        snprintf(error_msg, sizeof(error_msg), "Missing required keyword argument '%s'", key);
        PyErr_SetString(PyExc_TypeError, error_msg);
        return 1;
    }
    if (PyLong_Check(val)) {
        long out = PyLong_AsLong(val);
        if (out > INT_MAX || out < INT_MIN) {
            char error_msg[100];
            snprintf(error_msg, sizeof(error_msg), "Value %ld of integer argument %s is out of range", out, key);
            PyErr_SetString(PyExc_TypeError, error_msg);
            return 1;
        }
        // Cast on return. Safe because double can represent all 32-bit ints exactly
        return out;
    }
    if (PyFloat_Check(val)) {
        return PyFloat_AsDouble(val);
    }
    char error_msg[100];
    snprintf(error_msg, sizeof(error_msg), "Failed to unpack keyword %s as int", key);
    PyErr_SetString(PyExc_TypeError, error_msg);
    return 1;
}

// Method table
static PyMethodDef methods[] = {
    {"env_init", (PyCFunction)env_init, METH_VARARGS | METH_KEYWORDS, "Init environment with observation, action, reward, terminal, truncation arrays"},
    {"env_reset", env_reset, METH_VARARGS, "Reset the environment"},
    {"env_step", env_step, METH_VARARGS, "Step the environment"},
    {"env_render", env_render, METH_VARARGS, "Render the environment"},
    {"env_close", env_close, METH_VARARGS, "Close the environment"},
    {"env_get", env_get, METH_VARARGS, "Get the environment state"},
    {"env_put", (PyCFunction)env_put, METH_VARARGS | METH_KEYWORDS, "Put stuff into env"},
    {"vec_enable_mt", vec_enable_mt, METH_VARARGS, "Sets up multi-threading with provided number of threads"},
    {"vectorize", vectorize, METH_VARARGS, "Make a vector of environment handles"},
    {"vec_init", (PyCFunction)vec_init, METH_VARARGS | METH_KEYWORDS, "Initialize a vector of environments"},
    {"vec_reset", vec_reset, METH_VARARGS, "Reset the vector of environments"},
    {"vec_step", vec_step, METH_VARARGS, "Step the vector of environments"},
    {"vec_log", vec_log, METH_VARARGS, "Log the vector of environments"},
    {"vec_render", vec_render, METH_VARARGS, "Render the vector of environments"},
    {"vec_close", vec_close, METH_VARARGS, "Close the vector of environments"},
    {"shared", (PyCFunction)my_shared, METH_VARARGS | METH_KEYWORDS, "Shared state"},
    MY_METHODS,
    {NULL, NULL, 0, NULL}
};

// Module definition
static PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "binding",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_binding(void) {
    import_array();
    return PyModule_Create(&module);
}
