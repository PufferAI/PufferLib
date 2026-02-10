// static_envbinding.h - Shared types for static env linking
// Included by both env .c files and pufferlib.cpp

#pragma once

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

// Type constants
#define FLOAT 1
#define INT 2
#define UNSIGNED_CHAR 3
#define DOUBLE 4
#define CHAR 5

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
    double* actions;
    float* rewards;
    float* terminals;
    void* gpu_observations;
    double* gpu_actions;
    float* gpu_rewards;
    float* gpu_terminals;
    cudaStream_t* streams;
    StaticThreading* threading;
    int obs_size;
    int num_atns;
} StaticVec;

// Callback types
typedef void (*net_callback_fn)(void* ctx, int buf, int t);
typedef void (*thread_init_fn)(void* ctx, int buf);
typedef void (*step_fn)(void* env);

// Functions implemented by env's static library
StaticVec* create_static_vec(int total_agents, int num_buffers, Dict* vec_kwargs, Dict* env_kwargs);
void static_vec_reset(StaticVec* vec);
void static_vec_close(StaticVec* vec);
void static_vec_log(StaticVec* vec, Dict* out);
void create_static_threads(StaticVec* vec, int num_threads, int horizon,
    void* ctx, net_callback_fn net_callback, thread_init_fn thread_init);
void static_vec_omp_step(StaticVec* vec);
void static_vec_seq_step(StaticVec* vec);

// Env info
int get_obs_size(void);
int get_obs_type(void);
int get_num_atns(void);
int* get_act_sizes(void);

// Optional shared state functions
void* my_shared(void* env, Dict* kwargs);
void my_shared_close(void* env);
void* my_get(void* env, Dict* out);
int my_put(void* env, Dict* kwargs);

#ifdef __cplusplus
}
#endif
