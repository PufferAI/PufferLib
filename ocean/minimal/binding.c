// Include your .h first
#include "minimal.h"

// Required metadata
#define OBS_SIZE (2 + 4*(AGENTS+TARGETS))
#define NUM_ATNS 2
#define ACT_SIZES {9, 5}
#define OBS_TENSOR_T FloatTensor

// You can macro your struct and function names here
#define Env Env

// Include the vecenv.h here
#include "vecenv.h"

// Include any custom init logic here
void my_init(Env* env, Dict* kwargs) {
    env->num_agents = AGENTS;
}

// Specify log fields to export during training
void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
}
