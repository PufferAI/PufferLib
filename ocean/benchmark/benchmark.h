#include <string.h>
#include <math.h>

typedef struct {
    float perf;
    float score;
    float n;
} Log;

typedef struct {
    Log log;
    unsigned char* observations;
    double* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    int bandwidth;
    int compute;
} Benchmark;

void c_reset(Benchmark* env) {}

void c_step(Benchmark* env) {
    float result = 0;
    for (int i=0; i<env->compute; i++) {
        result = sinf(result + 0.1f);
    }

    //memset(env->observations, result, env->bandwidth);
}

void c_render(Benchmark* env) { }

void c_close(Benchmark* env) { }
