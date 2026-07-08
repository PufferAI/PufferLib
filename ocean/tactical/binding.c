#include "tactical.h"

#define OBS_SIZE 10
#define NUM_ATNS 1
#define ACT_SIZES {4}
#define OBS_TENSOR_T ByteTensor

#define Env Tactical
#include "vecenv.h"

// no init args needed
void my_init(Env* env, Dict* kwargs) {
    (void)kwargs;
    env->num_agents = 1;
}

// no logging implemented atm
void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
}
