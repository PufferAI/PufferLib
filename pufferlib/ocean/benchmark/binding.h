#include "benchmark.h"
#define OBS_SIZE 512 // TODO: Current API forces you to edit this per obs size
#define NUM_ATNS 1
#define ACT_SIZES {2}
#define OBS_TYPE UNSIGNED_CHAR
#define ACT_TYPE DOUBLE

#define Env Benchmark
#include "env_binding.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->compute = dict_get(kwargs, "compute")->value;
    env->bandwidth = dict_get(kwargs, "bandwidth")->value;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
}
