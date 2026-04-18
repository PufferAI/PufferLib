#include "flappy_puffer.h"
#include "flappy_puffer.c"
#define Log             FPLog

#define OBS_SIZE        FP_OBS_SIZE   
#define NUM_ATNS        1            
#define ACT_SIZES       {2}           
#define OBS_TENSOR_T    FloatTensor   

#define Env             FlappyPuffer
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents  = 1;  

    void* seed_ptr = dict_get(kwargs, "seed");
    env->rng = (seed_ptr != NULL)
        ? (unsigned int)(uintptr_t)seed_ptr
        : (unsigned int)time(NULL);
    allocate_flappy(env);
}

void my_log(FPLog* log, Dict* out) {
    dict_set(out, "perf",           log->perf);
    dict_set(out, "score",          log->score);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n_episodes",     log->n);
}