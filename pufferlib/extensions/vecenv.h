#ifndef PUFFERLIB_VECENV_H
#define PUFFERLIB_VECENV_H

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#ifndef __cplusplus
#include <stdatomic.h>
#endif

// Forward declare CUDA stream type
typedef struct CUstream_st* cudaStream_t;

#define FLOAT 1
#define INT 2
#define UNSIGNED_CHAR 3
#define DOUBLE 4
#define CHAR 5

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

typedef struct Env Env;
typedef struct Threading Threading;

typedef struct {
    Env* envs;
    int size;
    int total_agents;
    int agents_per_buffer;
    int* buffer_env_starts;
    int* buffer_env_counts;
    float* mask;
    float* gpu_mask;
    void* observations;
    double* actions;
    float* rewards;
    float* terminals;
    void* gpu_observations;
    double* gpu_actions;
    float* gpu_rewards;
    float* gpu_terminals;
    Threading* threading;
    cudaStream_t* streams;
    int buffers;
} VecEnv;

// Single struct containing all exported functions and data
typedef struct {
    // Functions
    VecEnv* (*create_environments)(int buffers, bool use_gpu, int test_idx, Dict* vec_kwargs, Dict* env_kwargs);
    void (*create_threads)(VecEnv* vec, int threads, int block_size, bool use_omp, void* ctx,
                           void (*net_callback)(void* ctx, int buf, int t),
                           void (*thread_init)(void* ctx, int buf),
                           int horizon);
    Env* (*env_init)(float* observations, double* actions, float* rewards, float* terminals, int seed, Dict* kwargs);
    void (*vec_reset)(VecEnv* vec);
    void (*vec_step)(VecEnv* vec, int buffer, cudaStream_t stream);
    void (*vec_send)(VecEnv* vec, int buffer, cudaStream_t stream);
    void (*vec_recv)(VecEnv* vec, int buffer, cudaStream_t stream);
    void (*vec_omp_step)(VecEnv* vec);
    void (*env_close)(Env* env);
    void (*vec_close)(VecEnv* vec);
    void (*vec_log)(VecEnv* vec, Dict* out);
    void (*vec_render)(VecEnv* vec, int env_idx);
    // Data
    int obs_n;
    int num_atns;
    int* act_sizes;
    int obs_type;
    int act_type;
} EnvExports;

// Dict helper functions
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

static inline void dict_set_ptr(Dict* dict, const char* key, void* ptr) {
    assert(dict->size < dict->capacity);
    DictItem* item = dict_get_unsafe(dict, key);
    if (item != NULL) {
        item->ptr = ptr;
        return;
    }
    dict->items[dict->size].key = key;
    dict->items[dict->size].ptr = ptr;
    dict->size++;
}

static inline void dict_set_int(Dict* dict, const char* key, int value) {
    dict_set(dict, key, (double)value);
}

// Forward declarations for env-specific functions
typedef struct Log Log;
void my_log(Log* log, Dict* out);
void* my_shared(Env* env, Dict* kwargs);
void my_shared_close(Env* env);
void* my_get(Env* env, Dict* out);
int my_put(Env* env, Dict* kwargs);

typedef struct Log Log;
void my_log(Log* log, Dict* out);

// Sharp bit (puffers have spikes)
// Define function types to be exported to the shared library
// You don't need these, but you have to do some really gross
// casts after loading the library without them.
typedef VecEnv* (*create_environments_fn)(int num_envs, int buffers, bool use_gpu, int test_idx, Dict* kwargs);
typedef Env* (*env_init_fn)(float* observations, double* actions, float* rewards,
        float* terminals, int seed, Dict* kwargs);
typedef void (*create_threads_fn)(VecEnv* vec, int threads, int block_size);
typedef void (*vec_reset_fn)(VecEnv* vec);
typedef void (*vec_step_fn)(VecEnv* vec);
typedef void (*vec_recv_fn)(VecEnv* vec, int buffer);
typedef void (*vec_send_fn)(VecEnv* vec, int buffer);
typedef void (*env_close_fn)(Env* env);
typedef void (*vec_close_fn)(VecEnv* vec);
typedef void (*vec_render_fn)(VecEnv* vec, int env_idx);
typedef void (*vec_log_fn)(VecEnv* vec, Dict* out);

typedef void (*c_reset_fn)(Env* env);
typedef void (*c_step_fn)(Env* env);
typedef void (*c_close_fn)(Env* env);
typedef void (*c_render_fn)(Env* env);

#endif // PUFFERLIB_VECENV_H
