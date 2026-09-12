// Env-only Craftax SPS. From the repo root:
//
//   clang -O2 -mavx2 -mfma -fopenmp -std=c11 -D_POSIX_C_SOURCE=200809L -DPLATFORM_DESKTOP \
//     -I. -Isrc -Ivendor -Iocean/craftax -Iraylib-5.5_linux_amd64/include \
//     scripts/bench_craftax_sps.c raylib-5.5_linux_amd64/lib/libraylib.a \
//     -lm -lpthread -lGL -ldl -o bench_craftax_sps
//
//   ./bench_craftax_sps [n_envs=8192] [n_threads=1]

#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "ocean/craftax/craftax.h"
#define BENCH_OBS OBS_SIZE
#define BENCH_ACTIONS ATN_DIM
#define BENCH_ENV_NAME "craftax"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint32_t xorshift(uint32_t* s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

static void step_all(Craftax* envs, uint32_t* rngs, int n_envs, int n_threads) {
    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int i = 0; i < n_envs; i++) {
        envs[i].agents[0].actions[0] =
            (float)(xorshift(&rngs[i]) % (uint32_t)BENCH_ACTIONS);
        puf_step(&envs[i]);
    }
}

int main(int argc, char** argv) {
    int n_envs = argc > 1 ? atoi(argv[1]) : 8192;
    int n_threads = argc > 2 ? atoi(argv[2]) : 1;
    const int warmup = 32;
    const int measure = 128;
    const int pool = 32;

    if (n_envs <= 0 || n_threads <= 0) {
        fprintf(stderr, "usage: %s [n_envs] [n_threads]\n", argv[0]);
        return 1;
    }

    omp_set_dynamic(0);
    omp_set_num_threads(n_threads);

    Dict kwargs = {0};
    dict_set(&kwargs, "reset_pool_size", (double)pool);

    Craftax* envs = calloc((size_t)n_envs, sizeof(Craftax));
    uint32_t* rngs = calloc((size_t)n_envs, sizeof(uint32_t));
    if (!envs || !rngs) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    for (int i = 0; i < n_envs; i++) {
        envs[i].num_agents = 1;
        envs[i].rng = (unsigned int)i;
        envs[i].seed = (uint64_t)i;
        envs[i].agents[0].observations = calloc((size_t)BENCH_OBS, sizeof(obs_t));
        envs[i].agents[0].actions = calloc(1, sizeof(float));
        envs[i].agents[0].rewards = calloc(1, sizeof(float));
        envs[i].agents[0].terminals = calloc(1, sizeof(float));
        rngs[i] = 0x9E3779B9u ^ (uint32_t)(i + 1);
        puf_init(&envs[i], &kwargs);
    }

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int i = 0; i < n_envs; i++) {
        puf_reset(&envs[i]);
    }

    for (int step = 0; step < warmup; step++) {
        step_all(envs, rngs, n_envs, n_threads);
    }

    double t0 = now_s();
    for (int step = 0; step < measure; step++) {
        step_all(envs, rngs, n_envs, n_threads);
    }
    double elapsed = now_s() - t0;
    double agent_steps = (double)n_envs * (double)measure;
    printf("%s  envs=%d  threads=%d  sps=%.0f\n",
        BENCH_ENV_NAME, n_envs, n_threads, agent_steps / elapsed);
    return 0;
}
