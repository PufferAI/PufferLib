#include "drmario.h"

#define OBS_SIZE 133
#define NUM_ATNS 1       
#define ACT_SIZES {7}     
#define OBS_TENSOR_T FloatTensor

#define Env DrMario
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents=1;
    env->n_rows = dict_get(kwargs, "n_rows")->value;
    env->n_cols = dict_get(kwargs, "n_cols")->value;
    env->n_init_viruses = dict_get(kwargs, "n_init_viruses")->value;
        c_init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "viruses_cleared", log->viruses_cleared);
}
