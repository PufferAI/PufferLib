#include "tripletriad.h"
#define OBS_SIZE 114
#define NUM_ATNS 1
#define ACT_SIZES {14}
#define OBS_TYPE FLOAT
#define ACT_TYPE DOUBLE

#define Env CTripleTriad
#include "env_binding.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->card_width = dict_get(kwargs, "card_width")->value;
    env->card_height = dict_get(kwargs, "card_height")->value;
    init_ctripletriad(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}
