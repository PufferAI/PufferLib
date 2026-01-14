#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <cuda_runtime.h>

#define FLOAT 1
#define INT 2
#define UNSIGNED_CHAR 3
#define DOUBLE 4

typedef struct {
    const char* key;
    void* void_value;
    int int_value;
    float float_value;
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
    float* observations;
    double* actions;
    float* rewards;
    float* terminals;
    float* gpu_observations;
    double* gpu_actions;
    float* gpu_rewards;
    float* gpu_terminals;
    Threading* threading;
    cudaStream_t* streams;
    int buffers;
} VecEnv;

Dict* create_dict(int capacity) {
    Dict* dict = (Dict*)calloc(1, sizeof(Dict));
    dict->capacity = capacity;
    dict->items = (DictItem*)calloc(capacity, sizeof(DictItem));
    return dict;
}

DictItem* dict_get_unsafe(Dict* dict, const char* key) {
    for (int i = 0; i < dict->size; i++) {
        if (strcmp(dict->items[i].key, key) == 0) {
            return &dict->items[i];
        }
    }
    return NULL;
}

DictItem* dict_get(Dict* dict, const char* key) {
    DictItem* item = dict_get_unsafe(dict, key);
    assert(item != NULL && "dict_get failed to find key");
    return item;
}

void dict_set_int(Dict* dict, const char* key, int value) {
    assert(dict->size < dict->capacity);
    DictItem* item = dict_get_unsafe(dict, key);

    if (item != NULL) {
        item->int_value = value;
        return;
    }

    dict->items[dict->size].key = key;
    dict->items[dict->size].int_value = value;
    dict->size++;
}

void dict_set_float(Dict* dict, const char* key, float value) {
    assert(dict->size < dict->capacity);
    DictItem* item = dict_get_unsafe(dict, key);

    if (item != NULL) {
        item->float_value = value;
        return;
    }

    dict->items[dict->size].key = key;
    dict->items[dict->size].float_value = value;
    dict->size++;
}

void dict_set_void(Dict* dict, const char* key, void* value) {
    assert(dict->size < dict->capacity);
    DictItem* item = dict_get_unsafe(dict, key);

    if (item != NULL) {
        item->void_value = value;
        return;
    }

    dict->items[dict->size].key = key;
    dict->items[dict->size].void_value = value;
    dict->size++;
}

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
