// profile_common.h
// Shared header for profile_kernels.cu and profile_torch_lib.cu
// Contains constants, helpers, raw Args structs, and function declarations.
// NO torch dependencies.

#pragma once

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <chrono>
#include <cuda_bf16.h>
#include "pufferlib/src/models.cu"

// Compile-time precision: default bf16, pass -DPRECISION_FLOAT for float32
#ifdef PRECISION_FLOAT
typedef float precision_t;
#else
typedef __nv_bfloat16 precision_t;
#endif

// ============================================================================
// Constants
// ============================================================================

const int WARMUP_ITERS = 100;
const int TIMING_ITERS = 1000;

const int BUF = 2;
const int BR = 4096;   // Rollout batch (no T dim)
const int BT = 512;    // Train batch (with T dim)
const int T = 64;
const int H = 128;
const int A = 4;
const int INPUT_SIZE = 96;

typedef void (*kernel_fn)(void*);

// ============================================================================
// Helpers
// ============================================================================

inline void print_timing(const char* name, float ms, int N) {
    printf("  %-28s %8.1f us  %8.2f M elem/s\n", name, ms * 1000, N / ms / 1e3);
}

inline void print_timing_pct(const char* name, float ms, int N, float total_ms) {
    float pct = (total_ms > 0) ? (ms / total_ms * 100.0f) : 0.0f;
    printf("  %-28s %8.1f us  %5.1f%%\n", name, ms * 1000, pct);
}

inline void warmup_gpu() {
    float* dummy;
    cudaMalloc(&dummy, 64 * 1024 * 1024);
    for (int i = 0; i < 100; i++) {
        cudaMemset(dummy, 0, 64 * 1024 * 1024);
    }
    cudaDeviceSynchronize();
    cudaFree(dummy);
}

inline float rand1() {
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

// Safe host-to-device copy: converts float[] -> precision_t[] before cudaMemcpy.
// Prevents buffer overflow when precision_t is bf16 (2 bytes) but source is float (4 bytes).
inline void float_to_device(precision_t* dst, const float* src, int count) {
    precision_t* tmp = (precision_t*)malloc(count * sizeof(precision_t));
    for (int i = 0; i < count; ++i) tmp[i] = (precision_t)src[i];
    cudaMemcpy(dst, tmp, count * sizeof(precision_t), cudaMemcpyHostToDevice);
    free(tmp);
}

// ============================================================================
// Profiling harness (no torch dependency — uses cudaEvent only)
// ============================================================================

inline float profile_kernel(kernel_fn fn, void* args, const char* name = nullptr) {
    for (int i = 0; i < WARMUP_ITERS; ++i) fn(args);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaProfilerStart();
    if (name) nvtxRangePushA(name);
    cudaEventRecord(start);

    for (int i = 0; i < TIMING_ITERS; ++i) {
        fn(args);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    if (name) nvtxRangePop();
    cudaProfilerStop();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();
    return ms / TIMING_ITERS;
}

// ============================================================================
// Raw Args structs (device pointers + dimensions, no torch)
// ============================================================================

typedef struct {
    precision_t* state;       // (B, H)
    precision_t* combined;    // (B, 3*H) = [hidden, gate, proj]
    precision_t* out;         // (B, H)
    precision_t* next_state;  // (B, H)
    int B;
    int H;
} MingruGateArgs;

typedef struct {
    precision_t* x;
    precision_t* out;
    double* s_buf;
    precision_t* grad_x;
    precision_t* grad_out;
    int B;
    int T;
    int H;
    int N;
} LogcumsumexpArgs;

typedef struct {
    precision_t* combined;       // (B, T, 3*H)
    precision_t* state;          // (B, 1, H)
    precision_t* out;            // (B, T, H)
    precision_t* next_state;     // (B, 1, H)
    float* a_star;               // (B, T+1, H)
    float* s_vals;               // (B, T+1, H)
    float* log_values_buf;       // (B, T+1, H)
    precision_t* grad_combined;  // (B, T, 3*H)
    precision_t* grad_state;     // (B, 1, H)
    precision_t* grad_out;       // (B, T, H)
    precision_t* grad_next_state;// (B, 1, H)
    int B;
    int T;
    int H;
    int N;
} FusedScanArgs;

typedef struct {
    precision_t* x;              // (B, N, D_in)
    float* W;              // (D_out, D_in) - always float32
    float* b;              // (D_out) - always float32
    precision_t* out;            // (B, D_out)
    int* argmax_indices;   // (B, D_out)
    float* grad_x;         // (B, N, D_in) - always float32 for atomicAdd
    float* grad_W;         // (D_out, D_in)
    float* grad_b;         // (D_out)
    precision_t* grad_out;       // (B, D_out)
    int B;
    int N;
    int D_in;
    int D_out;
} FCMaxArgs;

typedef struct {
    precision_t* logits;
    precision_t* values_pred;
    double* actions;          // float64 for both continuous and discrete
    precision_t* old_logprobs;
    float* advantages;        // always fp32 for precision
    precision_t* prio;
    precision_t* values;
    precision_t* returns;
    float* adv_mean;
    float* adv_var;           // variance, kernel does sqrt
    float* loss;
    float* losses_acc;         // (LOSS_N,) accumulator for loss components
    double* saved_for_backward;
    precision_t* ratio_out;
    precision_t* newvalue_out;
    float* grad_logits;
    float* grad_values_pred;
    float* grad_loss;
    int* act_sizes;           // (num_atns,) action head sizes
    int num_atns;
    float clip_coef;
    float vf_clip_coef;
    float vf_coef;
    float ent_coef;
    int N;
    int T;
    int A;
    int logits_stride_n;
    int logits_stride_t;
    int logits_stride_a;
    int values_stride_n;
    int values_stride_t;
} PPOLossArgs;

typedef struct {
    precision_t* logits;        // (B, A)
    precision_t* value;         // (B, 1)
    double* actions;      // (B, 1) - float64 for discrete/continuous compatibility
    precision_t* logprobs;      // (B,)
    precision_t* value_out;     // (B,)
    int64_t* offset;      // RNG offset (on device for CUDA graph support)
    int* act_sizes;       // (1,) - single action head
    uint64_t seed;
    int B;
    int A;
} SampleLogitsArgs;

// ============================================================================
// Raw kernel function declarations (implemented in profile_kernels.cu)
// ============================================================================

MingruGateArgs* create_mingrugateargs(int batch, int hidden);
void free_mingrugateargs(MingruGateArgs* args);
void run_mingrugate_forward(MingruGateArgs* args);

LogcumsumexpArgs* create_logcumsumexpargs(int batch, int seq, int hidden);
void free_logcumsumexpargs(LogcumsumexpArgs* args);
void run_logcumsumexp_forward(LogcumsumexpArgs* args);
void run_logcumsumexp_backward(LogcumsumexpArgs* args);

FusedScanArgs* create_fusedscanargs(int batch, int seq, int hidden);
void free_fusedscanargs(FusedScanArgs* args);
void run_fusedscan_forward(FusedScanArgs* args);
void run_fusedscan_backward(FusedScanArgs* args);

FCMaxArgs* create_fcmaxargs(int batch, int num_points, int d_in, int d_out);
void free_fcmaxargs(FCMaxArgs* args);
void run_fcmax_forward(FCMaxArgs* args);
void run_fcmax_backward(FCMaxArgs* args);

PPOLossArgs* create_ppolossargs(int batch, int seq, int actions);
void free_ppolossargs(PPOLossArgs* args);
void run_ppoloss_forward(PPOLossArgs* args);
void run_ppoloss_backward(PPOLossArgs* args);

SampleLogitsArgs* create_samplelogitsargs(int batch, int num_actions);
void free_samplelogitsargs(SampleLogitsArgs* args);
void run_samplelogits_forward(SampleLogitsArgs* args);

// ============================================================================
// C-linkage API for torch library (only when USE_TORCH is defined)
// ============================================================================

#ifdef USE_TORCH
extern "C" {
    // Per-kernel: correctness test + torch/cpp/graph benchmarks
    void torch_bench_mingrugate(MingruGateArgs* args);
    void torch_bench_logcumsumexp(LogcumsumexpArgs* args);
    void torch_bench_fusedscan(FusedScanArgs* args);
    void torch_bench_fcmax(FCMaxArgs* args);
    void torch_bench_ppoloss(PPOLossArgs* args);
    void torch_bench_samplelogits(SampleLogitsArgs* args);

    // Composite profiles (all torch-internal)
    void torch_profile_forwardcall(int batch, int input_size, int hidden, int act_n, int num_layers);
    void torch_profile_trainforward(int N, int T, int input, int hidden, int act_n, int layers);
    void torch_profile_trainstep(int N, int T, int input, int hidden, int act_n, int layers);
    void torch_profile_rolloutcopy(int S, int T, int mb_segs, int input, int A, int layers, int H);
}
#endif
