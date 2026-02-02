#ifndef PUFFERLIB_VECENV_H
#define PUFFERLIB_VECENV_H

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

void dict_set(Dict* dict, const char* key, double value) {
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

void dict_set_ptr(Dict* dict, const char* key, void* ptr) {
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

void* my_shared(Env* env, Dict* kwargs);
void my_shared_close(Env* env);
void* my_get(Env* env, Dict* out);
int my_put(Env* env, Dict* kwargs);

typedef struct Log Log;
void my_log(Log* log, Dict* out);

// Extern function declarations for environment interface
#ifdef __cplusplus
extern "C" {
#endif

typedef struct  {
    int obs_size;
    int act_size;
    int obs_type;
    int act_type;
} PufferEnvParams;


extern VecEnv* create_environments(int num_envs, int buffers, bool use_gpu, int test_idx, Dict* kwargs);
extern Env* env_init(float* observations, double* actions, float* rewards,
        float* terminals, int seed, Dict* kwargs);
extern void create_threads(VecEnv* vec, int threads, int block_size);
extern void vec_reset(VecEnv* vec);
extern void vec_step(VecEnv* vec, int buffer);
extern void vec_recv(VecEnv* vec, int buffer);
extern void vec_send(VecEnv* vec, int buffer);
extern void env_close(Env* env);
extern void vec_close(VecEnv* vec);
extern void vec_render(VecEnv* vec, int env_idx);
extern void vec_log(VecEnv* vec, Dict* out);
extern void update_env_params(PufferEnvParams* params);

#ifdef __cplusplus
}
#endif

#endif // PUFFERLIB_VECENV_H
