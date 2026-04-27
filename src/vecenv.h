// vecenv.h - Static env binding: types + implementation
// Types/declarations always available (for pufferlib.cu).
// Implementations compiled only when OBS_SIZE is defined (by binding.c).

#pragma once

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "tensor.h"

// Dict types
typedef struct {
    const char* key;
    double value;
    void* ptr;
} DictItem;

typedef struct {
    DictItem* items;
    int size;
    int capacity;
} Dict;

static inline Dict* create_dict(int capacity) {
    Dict* dict = (Dict*)calloc(1, sizeof(Dict));
    dict->capacity = capacity;
    dict->items = (DictItem*)calloc(capacity, sizeof(DictItem));
    return dict;
}

static inline DictItem* dict_get_unsafe(Dict* dict, const char* key) {
    for (int i = 0; i < dict->size; i++) {
        if (strcmp(dict->items[i].key, key) == 0) {
            return &dict->items[i];
        }
    }
    return NULL;
}

static inline DictItem* dict_get(Dict* dict, const char* key) {
    DictItem* item = dict_get_unsafe(dict, key);
    if (item == NULL) printf("dict_get failed to find key: %s\n", key);
    assert(item != NULL);
    return item;
}

static inline void dict_set(Dict* dict, const char* key, double value) {
    assert(dict->size < dict->capacity);
    DictItem* item = dict_get_unsafe(dict, key);
    if (item != NULL) {
        item->value = value;
        return;
    }
    dict->items[dict->size].key = key;
    dict->items[dict->size].value = value;
    dict->size++;
}

// Forward declare CUDA stream type
typedef struct CUstream_st* cudaStream_t;

// Threading state
typedef struct StaticThreading StaticThreading;

// Generic VecEnv - envs is void* to be type-agnostic
typedef struct StaticVec {
    void* envs;
    int size;
    int total_agents;
    int buffers;
    int agents_per_buffer;
    int* buffer_env_starts;
    int* buffer_env_counts;
    void* observations;
    float* actions;
    float* rewards;
    float* terminals;
    void* gpu_observations;
    float* gpu_actions;
    float* gpu_rewards;
    float* gpu_terminals;
    cudaStream_t* streams;
    StaticThreading* threading;
    int obs_size;
    int num_atns;
    int gpu;
} StaticVec;

// Callback types
typedef void (*net_callback_fn)(void* ctx, int buf, int t);
typedef void (*thread_init_fn)(void* ctx, int buf);
typedef void (*step_fn)(void* env);

enum EvalProfileIdx {
    EVAL_GPU = 0,   // forward + D2H (everything before env step)
    EVAL_ENV_STEP,  // OMP c_step (pure CPU)
    NUM_EVAL_PROF,
};

// Functions implemented by env's static library
StaticVec* create_static_vec(int total_agents, int num_buffers, int gpu, Dict* vec_kwargs, Dict* env_kwargs);
void static_vec_reset(StaticVec* vec);
void static_vec_close(StaticVec* vec);
void static_vec_log(StaticVec* vec, Dict* out);
void static_vec_eval_log(StaticVec* vec, Dict* out);
void create_static_threads(StaticVec* vec, int num_threads, int horizon,
    void* ctx, net_callback_fn net_callback, thread_init_fn thread_init);
void static_vec_omp_step(StaticVec* vec);
void static_vec_seq_step(StaticVec* vec);
void static_vec_render(StaticVec* vec, int env_id);
void static_vec_read_profile(StaticVec* vec, float out[NUM_EVAL_PROF]);

// Env info
int get_obs_size(void);
int get_num_atns(void);
int* get_act_sizes(void);
int get_num_act_sizes(void);
const char* get_obs_dtype(void);
size_t get_obs_elem_size(void);

// Synchronous single-step
void static_vec_step(StaticVec* vec);
void gpu_vec_step(StaticVec* vec);
void cpu_vec_step(StaticVec* vec);

// Optional shared state functions
void* my_shared(void* env, Dict* kwargs);
void my_shared_close(void* env);
void* my_get(void* env, Dict* out);
int my_put(void* env, Dict* kwargs);

#ifdef __cplusplus
}
#endif

// ============================================================================
// Implementation — only compiled when OBS_SIZE is defined (i.e. from binding.c)
// ============================================================================

#ifdef OBS_SIZE

static inline size_t obs_element_size(void) {
    OBS_TENSOR_T t;
    return sizeof(*t.data);
}

// Usually near the top, after any #includes
#define _STRINGIFY(x)   #x
#define  STRINGIFY(x)  _STRINGIFY(x)
const char dtype_symbol[] = STRINGIFY(OBS_TENSOR_T);

#include <omp.h>
#include <stdatomic.h>
#include <pthread.h>
#include <stdbool.h>
#include <time.h>

// Forward declare CUDA types and functions to avoid conflicts with raylib's float3
typedef int cudaError_t;
typedef int cudaMemcpyKind;
#define cudaSuccess 0
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2
#define cudaHostAllocPortable 1
#define cudaStreamNonBlocking 1

extern cudaError_t cudaHostAlloc(void**, size_t, unsigned int);
extern cudaError_t cudaMalloc(void**, size_t);
extern cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind);
extern cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
extern cudaError_t cudaMemset(void*, int, size_t);
extern cudaError_t cudaFree(void*);
extern cudaError_t cudaFreeHost(void*);
extern cudaError_t cudaSetDevice(int);
extern cudaError_t cudaDeviceSynchronize(void);
extern cudaError_t cudaStreamSynchronize(cudaStream_t);
extern cudaError_t cudaStreamCreateWithFlags(cudaStream_t*, unsigned int);
extern cudaError_t cudaStreamQuery(cudaStream_t);
extern const char* cudaGetErrorString(cudaError_t);

#define OMP_WAITING 5
#define OMP_RUNNING 6

// Forward declare env-provided functions (defined in binding.c after this include)
void my_init(Env* env, Dict* kwargs);
void my_log(Log* log, Dict* out);


struct StaticThreading {
    atomic_int* buffer_states;
    atomic_int shutdown;
    int num_threads;
    int num_buffers;
    pthread_t* threads;
    float* accum;  // [num_buffers * NUM_EVAL_PROF] per-buffer timing in ms
};

typedef struct StaticOMPArg {
    StaticVec* vec;
    int buf;
    int horizon;
    void* ctx;
    net_callback_fn net_callback;
    thread_init_fn thread_init;
} StaticOMPArg;

// OMP thread manager
static void* static_omp_threadmanager(void* arg) {
    StaticOMPArg* worker_arg = (StaticOMPArg*)arg;
    StaticVec* vec = worker_arg->vec;
    StaticThreading* threading = vec->threading;
    int buf = worker_arg->buf;
    int horizon = worker_arg->horizon;
    void* ctx = worker_arg->ctx;
    net_callback_fn net_callback = worker_arg->net_callback;
    thread_init_fn thread_init = worker_arg->thread_init;

    if (thread_init != NULL) {
        thread_init(ctx, buf);
    }

    int agents_per_buffer = vec->agents_per_buffer;
    int agent_start = buf * agents_per_buffer;
    int env_start = vec->buffer_env_starts[buf];
    int env_count = vec->buffer_env_counts[buf];
    atomic_int* buffer_states = threading->buffer_states;
    int num_workers = threading->num_threads / vec->buffers;
    if (num_workers < 1) num_workers = 1;

    Env* envs = (Env*)vec->envs;

    printf("Num workers: %d\n", num_workers);
    while (true) {
        while (atomic_load(&buffer_states[buf]) != OMP_RUNNING) {
            if (atomic_load(&threading->shutdown)) {
                return NULL;
            }
        }
        cudaStream_t stream = vec->streams[buf];

        float* my_accum = &threading->accum[buf * NUM_EVAL_PROF];
        struct timespec t0, t1;

        for (int t = 0; t < horizon; t++) {
            clock_gettime(CLOCK_MONOTONIC, &t0);
            net_callback(ctx, buf, t);

            cudaMemcpyAsync(
                &vec->actions[agent_start * NUM_ATNS],
                &vec->gpu_actions[agent_start * NUM_ATNS],
                agents_per_buffer * NUM_ATNS * sizeof(float),
                cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            my_accum[EVAL_GPU] += (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_nsec - t0.tv_nsec) / 1e6f;

            memset(&vec->rewards[agent_start], 0, agents_per_buffer * sizeof(float));
            memset(&vec->terminals[agent_start], 0, agents_per_buffer * sizeof(float));
            clock_gettime(CLOCK_MONOTONIC, &t0);
            #pragma omp parallel for schedule(static) num_threads(num_workers)
            for (int i = env_start; i < env_start + env_count; i++) {
                c_step(&envs[i]);
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            my_accum[EVAL_ENV_STEP] += (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_nsec - t0.tv_nsec) / 1e6f;

            cudaMemcpyAsync(
                (char*)vec->gpu_observations + agent_start * OBS_SIZE * obs_element_size(),
                (char*)vec->observations + agent_start * OBS_SIZE * obs_element_size(),
                agents_per_buffer * OBS_SIZE * obs_element_size(),
                cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(
                &vec->gpu_rewards[agent_start],
                &vec->rewards[agent_start],
                agents_per_buffer * sizeof(float),
                cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(
                &vec->gpu_terminals[agent_start],
                &vec->terminals[agent_start],
                agents_per_buffer * sizeof(float),
                cudaMemcpyHostToDevice, stream);
        }
        cudaStreamSynchronize(stream);
        atomic_store(&buffer_states[buf], OMP_WAITING);
    }
}

void static_vec_omp_step(StaticVec* vec) {
    StaticThreading* threading = vec->threading;
    for (int buf = 0; buf < vec->buffers; buf++) {
        atomic_store(&threading->buffer_states[buf], OMP_RUNNING);
    }
    for (int buf = 0; buf < vec->buffers; buf++) {
        while (atomic_load(&threading->buffer_states[buf]) != OMP_WAITING) {}
    }
}

void static_vec_seq_step(StaticVec* vec) {
    StaticThreading* threading = vec->threading;
    for (int buf = 0; buf < vec->buffers; buf++) {
        atomic_store(&threading->buffer_states[buf], OMP_RUNNING);
        while (atomic_load(&threading->buffer_states[buf]) != OMP_WAITING) {}
    }
}

// Optional: Initialize all envs at once (for shared state, variable agents per env, etc.)
// Default implementation creates envs until total_agents is reached
#ifdef MY_VEC_INIT
Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs);
#else
Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {

    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int agents_per_buffer = total_agents / num_buffers;

    // Allocate max possible envs (1 agent per env worst case)
    Env* envs = (Env*)calloc(total_agents, sizeof(Env));

    int num_envs = 0;
    int agents_created = 0;
    while (agents_created < total_agents) {
        srand(num_envs);
        envs[num_envs].rng = num_envs;
        my_init(&envs[num_envs], env_kwargs);
        agents_created += envs[num_envs].num_agents;
        num_envs++;
    }

    // Shrink to actual size needed
    envs = (Env*)realloc(envs, num_envs * sizeof(Env));

    // Fill buffer info by iterating through envs
    int buf = 0;
    int buf_agents = 0;
    buffer_env_starts[0] = 0;
    buffer_env_counts[0] = 0;
    for (int i = 0; i < num_envs; i++) {
        buf_agents += envs[i].num_agents;
        buffer_env_counts[buf]++;
        if (buf_agents >= agents_per_buffer && buf < num_buffers - 1) {
            buf++;
            buffer_env_starts[buf] = i + 1;
            buffer_env_counts[buf] = 0;
            buf_agents = 0;
        }
    }

    *num_envs_out = num_envs;
    return envs;
}
#endif

#ifdef MY_VEC_CLOSE
void my_vec_close(Env* envs);
#else
void my_vec_close(Env* envs) {
    return;
}
#endif

StaticVec* create_static_vec(int total_agents, int num_buffers, int gpu, Dict* vec_kwargs, Dict* env_kwargs) {
    StaticVec* vec = (StaticVec*)calloc(1, sizeof(StaticVec));
    vec->total_agents = total_agents;
    vec->buffers = num_buffers;
    vec->agents_per_buffer = total_agents / num_buffers;
    vec->obs_size = OBS_SIZE;
    vec->num_atns = NUM_ATNS;
    vec->gpu = gpu;

    vec->buffer_env_starts = (int*)calloc(num_buffers, sizeof(int));
    vec->buffer_env_counts = (int*)calloc(num_buffers, sizeof(int));

    // Let my_vec_init allocate and initialize envs, fill buffer info
    int num_envs = 0;
    vec->envs = my_vec_init(&num_envs, vec->buffer_env_starts, vec->buffer_env_counts,
                            vec_kwargs, env_kwargs);
    vec->size = num_envs;

    size_t obs_elem_size = obs_element_size();
    if (gpu) {
        cudaHostAlloc((void**)&vec->observations, total_agents * OBS_SIZE * obs_elem_size, cudaHostAllocPortable);
        cudaHostAlloc((void**)&vec->actions, total_agents * NUM_ATNS * sizeof(float), cudaHostAllocPortable);
        cudaHostAlloc((void**)&vec->rewards, total_agents * sizeof(float), cudaHostAllocPortable);
        cudaHostAlloc((void**)&vec->terminals, total_agents * sizeof(float), cudaHostAllocPortable);

        cudaMalloc((void**)&vec->gpu_observations, total_agents * OBS_SIZE * obs_elem_size);
        cudaMalloc((void**)&vec->gpu_actions, total_agents * NUM_ATNS * sizeof(float));
        cudaMalloc((void**)&vec->gpu_rewards, total_agents * sizeof(float));
        cudaMalloc((void**)&vec->gpu_terminals, total_agents * sizeof(float));

        cudaMemset(vec->gpu_observations, 0, total_agents * OBS_SIZE * obs_elem_size);
        cudaMemset(vec->gpu_actions, 0, total_agents * NUM_ATNS * sizeof(float));
        cudaMemset(vec->gpu_rewards, 0, total_agents * sizeof(float));
        cudaMemset(vec->gpu_terminals, 0, total_agents * sizeof(float));
    } else {
        vec->observations = calloc(total_agents * OBS_SIZE, obs_elem_size);
        vec->actions = (float*)calloc(total_agents * NUM_ATNS, sizeof(float));
        vec->rewards = (float*)calloc(total_agents, sizeof(float));
        vec->terminals = (float*)calloc(total_agents, sizeof(float));
        // CPU mode: gpu pointers alias the same buffers (no copy needed)
        vec->gpu_observations = vec->observations;
        vec->gpu_actions = vec->actions;
        vec->gpu_rewards = vec->rewards;
        vec->gpu_terminals = vec->terminals;
    }

    // Streams allocated here, created in create_static_threads
    vec->streams = (cudaStream_t*)calloc(num_buffers, sizeof(cudaStream_t));

    // Assign pointers to envs based on buffer layout
    Env* envs = (Env*)vec->envs;
    for (int buf = 0; buf < num_buffers; buf++) {
        int buf_start = buf * vec->agents_per_buffer;
        int buf_agent = 0;
        int env_start = vec->buffer_env_starts[buf];
        int env_count = vec->buffer_env_counts[buf];

        for (int e = 0; e < env_count; e++) {
            Env* env = &envs[env_start + e];
            int slot = buf_start + buf_agent;
            env->observations = (void*)((char*)vec->observations + slot * OBS_SIZE * obs_elem_size);
            env->actions = vec->actions + slot * NUM_ATNS;
            env->rewards = vec->rewards + slot;
            env->terminals = vec->terminals + slot;
            buf_agent += env->num_agents;
        }
    }

    return vec;
}

void static_vec_reset(StaticVec* vec) {
    Env* envs = (Env*)vec->envs;
    for (int i = 0; i < vec->size; i++) {
        c_reset(&envs[i]);
    }
    if (vec->gpu) {
        cudaMemcpy(vec->gpu_observations, vec->observations,
            vec->total_agents * OBS_SIZE * obs_element_size(), cudaMemcpyHostToDevice);
        cudaMemset(vec->gpu_rewards,   0, vec->total_agents * sizeof(float));
        cudaMemset(vec->gpu_terminals, 0, vec->total_agents * sizeof(float));
        cudaDeviceSynchronize();
    } else {
        memset(vec->rewards, 0, vec->total_agents * sizeof(float));
        memset(vec->terminals, 0, vec->total_agents * sizeof(float));
    }
}

void create_static_threads(StaticVec* vec, int num_threads, int horizon,
        void* ctx, net_callback_fn net_callback, thread_init_fn thread_init) {
    vec->threading = (StaticThreading*)calloc(1, sizeof(StaticThreading));
    vec->threading->num_threads = num_threads;
    vec->threading->num_buffers = vec->buffers;
    vec->threading->buffer_states = (atomic_int*)calloc(vec->buffers, sizeof(atomic_int));
    vec->threading->threads = (pthread_t*)calloc(vec->buffers, sizeof(pthread_t));
    vec->threading->accum = (float*)calloc(vec->buffers * NUM_EVAL_PROF, sizeof(float));

    // Streams are now created by pufferlib.cu (PyTorch-managed streams)
    // Do NOT create streams here - they've already been set up

    StaticOMPArg* args = (StaticOMPArg*)calloc(vec->buffers, sizeof(StaticOMPArg));
    for (int i = 0; i < vec->buffers; i++) {
        args[i].vec = vec;
        args[i].buf = i;
        args[i].horizon = horizon;
        args[i].ctx = ctx;
        args[i].net_callback = net_callback;
        args[i].thread_init = thread_init;
        pthread_create(&vec->threading->threads[i], NULL, static_omp_threadmanager, &args[i]);
    }
}

void static_vec_close(StaticVec* vec) {
    Env* envs = (Env*)vec->envs;

    if (vec->threading != NULL) {
        atomic_store(&vec->threading->shutdown, 1);
        for (int i = 0; i < vec->buffers; i++) {
            pthread_join(vec->threading->threads[i], NULL);
        }
    }

    for (int i = 0; i < vec->size; i++) {
        Env* env = &envs[i];
        c_close(env);
    }

    my_vec_close(envs);
    free(vec->envs);
    if (vec->threading != NULL) {
        free(vec->threading->buffer_states);
        free(vec->threading->threads);
        free(vec->threading->accum);
        free(vec->threading);
    }
    free(vec->buffer_env_starts);
    free(vec->buffer_env_counts);

    if (vec->gpu) {
        cudaDeviceSynchronize();
        cudaFree(vec->gpu_observations);
        cudaFree(vec->gpu_actions);
        cudaFree(vec->gpu_rewards);
        cudaFree(vec->gpu_terminals);
        cudaFreeHost(vec->observations);
        cudaFreeHost(vec->actions);
        cudaFreeHost(vec->rewards);
        cudaFreeHost(vec->terminals);
    } else {
        free(vec->observations);
        free(vec->actions);
        free(vec->rewards);
        free(vec->terminals);
    }

    free(vec->streams);
    free(vec);
}

static inline float static_vec_aggregate_logs(StaticVec* vec, Log* out) {
    Env* envs = (Env*)vec->envs;
    memset(out, 0, sizeof(Log));
    int num_keys = sizeof(Log) / sizeof(float);
    for (int i = 0; i < vec->size; i++) {
        Env* env = &envs[i];
        if (env->log.n == 0) {
            continue;
        }
        for (int j = 0; j < num_keys; j++) {
            ((float*)out)[j] += ((float*)&env->log)[j];
        }
    }
    float n = out->n;
    if (n == 0.0f) {
        return 0;
    }
    for (int i = 0; i < num_keys; i++) {
        ((float*)out)[i] /= n;
    }
    return n;
}

void static_vec_log(StaticVec* vec, Dict* out) {
    Env* envs = (Env*)vec->envs;
    Log aggregate;
    float n = static_vec_aggregate_logs(vec, &aggregate);
    if (n == 0) {
        return;
    }
    for (int i = 0; i < vec->size; i++) {
        memset(&envs[i].log, 0, sizeof(Log));
    }
    my_log(&aggregate, out);
    dict_set(out, "n", n);
}

void static_vec_eval_log(StaticVec* vec, Dict* out) {
    Log aggregate;
    float n = static_vec_aggregate_logs(vec, &aggregate);
    if (n == 0) {
        return;
    }
    my_log(&aggregate, out);
    dict_set(out, "n", n);
}

void static_vec_read_profile(StaticVec* vec, float out[NUM_EVAL_PROF]) {
    StaticThreading* threading = vec->threading;
    memset(out, 0, NUM_EVAL_PROF * sizeof(float));
    for (int buf = 0; buf < threading->num_buffers; buf++) {
        float* src = &threading->accum[buf * NUM_EVAL_PROF];
        for (int i = 0; i < NUM_EVAL_PROF; i++) {
            out[i] += src[i];
        }
        memset(src, 0, NUM_EVAL_PROF * sizeof(float));
    }
    // Average across buffers (they run in parallel)
    for (int i = 0; i < NUM_EVAL_PROF; i++) {
        out[i] /= threading->num_buffers;
    }
}

void static_vec_render(StaticVec* vec, int env_id) {
    Env* envs = (Env*)vec->envs;
    c_render(&envs[env_id]);
}

int get_obs_size(void) { return OBS_SIZE; }
int get_num_atns(void) { return NUM_ATNS; }
static int _act_sizes[] = ACT_SIZES;
int* get_act_sizes(void) { return _act_sizes; }
int get_num_act_sizes(void) { return (int)(sizeof(_act_sizes) / sizeof(_act_sizes[0])); }
const char* get_obs_dtype(void) { return dtype_symbol; }
size_t get_obs_elem_size(void) { return obs_element_size(); }

static inline void _static_vec_env_step(StaticVec* vec) {
    memset(vec->rewards, 0, vec->total_agents * sizeof(float));
    memset(vec->terminals, 0, vec->total_agents * sizeof(float));
    Env* envs = (Env*)vec->envs;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < vec->size; i++) {
        c_step(&envs[i]);
    }
}

void gpu_vec_step(StaticVec* vec) {
    assert(vec->buffers == 1);
    cudaMemcpy(vec->actions, vec->gpu_actions,
        (size_t)vec->total_agents * NUM_ATNS * sizeof(float),
        cudaMemcpyDeviceToHost);
    _static_vec_env_step(vec);
    cudaMemcpy(vec->gpu_observations, vec->observations,
        (size_t)vec->total_agents * OBS_SIZE * obs_element_size(),
        cudaMemcpyHostToDevice);
    cudaMemcpy(vec->gpu_rewards, vec->rewards,
        vec->total_agents * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vec->gpu_terminals, vec->terminals,
        vec->total_agents * sizeof(float), cudaMemcpyHostToDevice);
}

void cpu_vec_step(StaticVec* vec) {
    assert(vec->buffers == 1);
    _static_vec_env_step(vec);
}

void static_vec_step(StaticVec* vec) {
    if (vec->gpu) gpu_vec_step(vec);
    else cpu_vec_step(vec);
}

// Optional shared state functions - default implementations
#ifndef MY_SHARED
void* my_shared(void* env, Dict* kwargs) {
    return NULL;
}
#endif

#ifndef MY_SHARED_CLOSE
void my_shared_close(void* env) {}
#endif

#ifndef MY_GET
void* my_get(void* env, Dict* out) {
    return NULL;
}
#endif

#ifndef MY_PUT
int my_put(void* env, Dict* kwargs) {
    return 0;
}
#endif

#endif // OBS_SIZE
