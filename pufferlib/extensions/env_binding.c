// static_envbinding.c - Template for sttic env binding
// Include this AFTER defining: Env, OBS_SIZE, NUM_ATNS, my_init, my_log, c_step, c_reset

#include <omp.h>
#include <stdatomic.h>
#include <pthread.h>
#include <stdbool.h>

#include "env_binding.h"
#include "binding.h"

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

// Helper to get observation element size based on OBS_TYPE
static inline size_t obs_element_size(void) {
    switch (OBS_TYPE) {
        case FLOAT: return sizeof(float);
        case INT: return sizeof(int);
        case UNSIGNED_CHAR: return sizeof(unsigned char);
        case DOUBLE: return sizeof(double);
        case CHAR: return sizeof(char);
        default: return sizeof(float);
    }
}

struct StaticThreading {
    atomic_int* buffer_states;
    atomic_int shutdown;
    int num_threads;
    int num_buffers;
    pthread_t* threads;
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

        for (int t = 0; t < horizon; t++) {
            net_callback(ctx, buf, t);

            cudaMemcpyAsync(
                &vec->actions[agent_start * NUM_ATNS],
                &vec->gpu_actions[agent_start * NUM_ATNS],
                agents_per_buffer * NUM_ATNS * sizeof(double),
                cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            if (num_workers > 1) {
                #pragma omp parallel for schedule(static) num_threads(num_workers)
                for (int i = env_start; i < env_start + env_count; i++) {
                    c_step(&envs[i]);
                }
            } else {
                for (int i = env_start; i < env_start + env_count; i++) {
                    c_step(&envs[i]);
                }
            }

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
#ifndef MY_VEC_INIT
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

StaticVec* create_static_vec(int total_agents, int num_buffers, Dict* vec_kwargs, Dict* env_kwargs) {
    StaticVec* vec = (StaticVec*)calloc(1, sizeof(StaticVec));
    vec->total_agents = total_agents;
    vec->buffers = num_buffers;
    vec->agents_per_buffer = total_agents / num_buffers;
    vec->obs_size = OBS_SIZE;
    vec->num_atns = NUM_ATNS;

    vec->buffer_env_starts = (int*)calloc(num_buffers, sizeof(int));
    vec->buffer_env_counts = (int*)calloc(num_buffers, sizeof(int));

    // Let my_vec_init allocate and initialize envs, fill buffer info
    int num_envs = 0;
    vec->envs = my_vec_init(&num_envs, vec->buffer_env_starts, vec->buffer_env_counts,
                            vec_kwargs, env_kwargs);
    vec->size = num_envs;

    size_t obs_elem_size = obs_element_size();
    cudaHostAlloc((void**)&vec->observations, total_agents * OBS_SIZE * obs_elem_size, cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->actions, total_agents * NUM_ATNS * sizeof(double), cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->rewards, total_agents * sizeof(float), cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->terminals, total_agents * sizeof(float), cudaHostAllocPortable);

    cudaMalloc((void**)&vec->gpu_observations, total_agents * OBS_SIZE * obs_elem_size);
    cudaMalloc((void**)&vec->gpu_actions, total_agents * NUM_ATNS * sizeof(double));
    cudaMalloc((void**)&vec->gpu_rewards, total_agents * sizeof(float));
    cudaMalloc((void**)&vec->gpu_terminals, total_agents * sizeof(float));

    cudaMemset(vec->gpu_observations, 0, total_agents * OBS_SIZE * obs_elem_size);
    cudaMemset(vec->gpu_actions, 0, total_agents * NUM_ATNS * sizeof(double));
    cudaMemset(vec->gpu_rewards, 0, total_agents * sizeof(float));
    cudaMemset(vec->gpu_terminals, 0, total_agents * sizeof(float));

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
    cudaMemcpy(vec->gpu_observations, vec->observations,
        vec->total_agents * OBS_SIZE * obs_element_size(), cudaMemcpyHostToDevice);
    cudaMemcpy(vec->gpu_rewards, vec->rewards,
        vec->total_agents * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vec->gpu_terminals, vec->terminals,
        vec->total_agents * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void create_static_threads(StaticVec* vec, int num_threads, int horizon,
        void* ctx, net_callback_fn net_callback, thread_init_fn thread_init) {
    vec->threading = (StaticThreading*)calloc(1, sizeof(StaticThreading));
    vec->threading->num_threads = num_threads;
    vec->threading->num_buffers = vec->buffers;
    vec->threading->buffer_states = (atomic_int*)calloc(vec->buffers, sizeof(atomic_int));
    vec->threading->threads = (pthread_t*)calloc(vec->buffers, sizeof(pthread_t));

    // Streams are now created by pufferlib.cpp (PyTorch-managed streams)
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

    // Ask threads to stop. todo: robustify
    atomic_store(&vec->threading->shutdown, 1);
    for (int i = 0; i < vec->buffers; i++) {
        pthread_join(vec->threading->threads[i], NULL);
    }

    for (int i = 0; i < vec->size; i++) {
        Env* env = &envs[i];
        c_close(env);
    }

    free(vec->envs);
    free(vec->threading->buffer_states);
    free(vec->threading->threads);
    free(vec->threading);
    free(vec->buffer_env_starts);
    free(vec->buffer_env_counts);

    cudaDeviceSynchronize();
    size_t obs_bytes = vec->total_agents * OBS_SIZE * obs_element_size();
    size_t act_bytes = vec->total_agents * NUM_ATNS * sizeof(double);
    size_t rew_bytes = vec->total_agents * sizeof(float);
    size_t term_bytes = vec->total_agents * sizeof(float);
    cudaFree(vec->gpu_observations);
    cudaFree(vec->gpu_actions);
    cudaFree(vec->gpu_rewards);
    cudaFree(vec->gpu_terminals);
    cudaFreeHost(vec->observations);
    cudaFreeHost(vec->actions);
    cudaFreeHost(vec->rewards);
    cudaFreeHost(vec->terminals);

    free(vec->streams);
    free(vec);
}

void static_vec_log(StaticVec* vec, Dict* out) {
    Env* envs = (Env*)vec->envs;
    Log aggregate = {0};
    int num_keys = sizeof(Log) / sizeof(float);
    for (int i = 0; i < vec->size; i++) {
        Env* env = &envs[i];
        if (env->log.n == 0) {
            continue;
        }
        for (int j = 0; j < num_keys; j++) {
            ((float*)&aggregate)[j] += ((float*)&env->log)[j];
        }
        memset(&env->log, 0, sizeof(Log));
    }
    float n = aggregate.n;
    if (n == 0.0f) {
        return;
    }
    for (int i = 0; i < num_keys; i++) {
        ((float*)&aggregate)[i] /= n;
    }
    my_log(&aggregate, out);
    dict_set(out, "n", n);
}

int get_obs_size(void) { return OBS_SIZE; }
int get_obs_type(void) { return OBS_TYPE; }
int get_num_atns(void) { return NUM_ATNS; }
static int _act_sizes[] = ACT_SIZES;
int* get_act_sizes(void) { return _act_sizes; }

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
