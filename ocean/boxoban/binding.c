#define BOXOBAN_MAPS_IMPLEMENTATION //enables mmap
#include "boxoban.h"
#define OBS_SIZE 400
#define NUM_ATNS 1
#define ACT_SIZES {5}
#define OBS_TENSOR_T ByteTensor


#define Env Boxoban
#include "vecenv.h"


void my_init(Env* env, Dict* kwargs) {
    env->difficulty_id = (int)dict_get(kwargs, "difficulty")->value;
    env->size = 10;
    env->num_agents = 1;
    env->max_steps = (int)dict_get(kwargs, "max_steps")->value;
    env->int_r_coeff = (float)dict_get(kwargs, "int_r_coeff")->value;
    env->target_loss_pen_coeff = (float)dict_get(kwargs, "target_loss_pen_coeff")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "targets_hit", log->on_targets);
}
