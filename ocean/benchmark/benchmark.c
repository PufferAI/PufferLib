#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "benchmark.h"

int main(int argc, char** argv) {
    int num_envs = argc > 1 ? atoi(argv[1]) : 1;
    int threads = argc > 2 ? atoi(argv[2]) : 1;
    int compute = argc > 3 ? atoi(argv[3]) : 0;
    int bandwidth = argc > 4 ? atoi(argv[4]) : 1000;
    int timeout = argc > 5 ? atoi(argv[5]) : 3;

    if (argc == 1) {
        printf("Usage: %s [num_envs] [threads] [compute] [bandwidth] [timeout]\n", argv[0]);
        printf("  num_envs: number of parallel environments (default: 1)\n");
        printf("  threads: OpenMP threads (default: 1)\n");
        printf("  compute: sinf() iterations per step (default: 0)\n");
        printf("  bandwidth: bytes written per step (default: 1000)\n");
        printf("  timeout: test duration in seconds (default: 3)\n\n");
    }

    if (threads > 0) {
        omp_set_num_threads(threads);
    }

    printf("Benchmark: envs=%d, threads=%d, compute=%d, bandwidth=%d\n",
           num_envs, threads, compute, bandwidth);

    unsigned char* observations = (unsigned char*)calloc(num_envs*bandwidth, sizeof(unsigned char));
    double* actions = (double*)calloc(num_envs, sizeof(double));
    float* rewards = (float*)calloc(num_envs, sizeof(float));
    float* terminals = (float*)calloc(num_envs, sizeof(float));

    // Allocate env array with pointers into contiguous buffers
    Benchmark* envs = (Benchmark*)calloc(num_envs, sizeof(Benchmark));
    for (int i = 0; i < num_envs; i++) {
        envs[i].bandwidth = bandwidth;
        envs[i].compute = compute;
        envs[i].observations = observations + i * bandwidth;
        envs[i].actions = actions + i;
        envs[i].rewards = rewards + i;
        envs[i].terminals = terminals + i;
        c_reset(&envs[i]);
    }

    // Warmup
    for (int w = 0; w < 10; w++) {
        if (threads > 0) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < num_envs; i++) {
                c_step(&envs[i]);
            }
        } else {
            for (int i = 0; i < num_envs; i++) {
                c_step(&envs[i]);
            }
        }
    }

    int start = time(NULL);
    long num_steps = 0;
    while (time(NULL) - start < timeout) {
        if (threads > 0) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < num_envs; i++) {
                c_step(&envs[i]);
            }
        } else {
            for (int i = 0; i < num_envs; i++) {
                c_step(&envs[i]);
            }
        }
        num_steps += num_envs;
    }

    int elapsed = time(NULL) - start;
    double sps = (double)num_steps / elapsed;
    double bytes_per_sec = sps * bandwidth;

    // Checksum to prevent optimization
    unsigned char checksum = 0;
    for (int i = 0; i < num_envs; i++) {
        checksum ^= envs[i].observations[0];
    }

    printf("Checksum: %d\n", checksum);
    printf("  steps=%ld, elapsed=%ds\n", num_steps, elapsed);
    printf("  throughput: %.2f M steps/s\n", sps / 1e6);
    printf("  bandwidth: %.2f GB/s\n", bytes_per_sec / 1e9);

    // Cleanup
    free(observations);
    free(actions);
    free(rewards);
    free(terminals);
    free(envs);

    return 0;
}
