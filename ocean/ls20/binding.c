#include "ls20.h"

#define OBS_SIZE LS20_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {LS20_ACTION_COUNT}
#define OBS_TENSOR_T ByteTensor
#define MY_ACTION_MASK LS20_ACTION_COUNT

#define Env Ls20
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    DictItem* fps = dict_get_unsafe(kwargs, "fps");
    env->fps = fps == NULL ? LS20_DEFAULT_FPS : (int)fps->value;
    DictItem* reset_enabled = dict_get_unsafe(kwargs, "reset_enabled");
    env->reset_enabled = reset_enabled != NULL && reset_enabled->value != 0.0;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
