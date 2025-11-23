#include "lock_key.h"

#define Env LockKey 
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->size = unpack(kwargs, "size");
    env->num_keys = unpack(kwargs, "num_keys");
    env->obs_dist = unpack(kwargs, "obs_dist");

    int tiles = env->size * env->size;
    env->state = (unsigned char*)calloc(tiles, sizeof(unsigned char));
    if (!env->state) return -1;

    return 0;
}

static int my_close(Env* env) {
    if (env->state) {
        free(env->state);
        env->state = NULL;
    }
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
