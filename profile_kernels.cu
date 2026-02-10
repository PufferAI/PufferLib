// profile_kernels.cu
// Minimal standalone profiler for CUDA kernels
//
// Without torch: nvcc -O3 -arch=sm_80 -DPRECISION_FLOAT profile_kernels.cu -o profile_kernels -I.
// With torch:    Build with cmake/pytorch and -DUSE_TORCH
// With env:      ./scripts/build_profile_kernels.sh <env_name>
//
// Run: ./profile_kernels <profile>
//   kernels  - Individual kernel profiling
//   envspeed - Environment step throughput (requires -DUSE_STATIC_ENV)

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <chrono>


#ifdef USE_TORCH
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAGraph.h>
#include "pufferlib/extensions/cuda/kernels.cu"
#include "pufferlib/extensions/modules.cu"

// models.cpp provides Policy, MinGRU, DefaultEncoder, DefaultDecoder,
// reference impls, PRECISION_DTYPE, cuda_f32/f64/i32/i64, Tensor typedef
namespace pufferlib {
#include "pufferlib/extensions/models.cpp"
}
using namespace pufferlib;
#endif

#ifndef USE_TORCH
#include "pufferlib/extensions/cuda/kernels.cu"
#endif

const int WARMUP_ITERS = 100;
const int TIMING_ITERS = 1000;
const float TIMEOUT_SEC = 3.0f;

const int BUF = 2;
const int BR = 4096;  // Rollout batch (no T dim)
const int BT = 512;   // Train batch (with T dim)
const int T = 64;
const int H = 128;
const int A = 4;
const int INPUT_SIZE = 96;

typedef void (*kernel_fn)(void*);

void print_timing(const char* name, float ms, int N) {
    printf("  %-18s %6.1f us  %6.2f M elem/s\n", name, ms * 1000, N / ms / 1e3);
}

// Wall-clock time for timeout checks (only checked every BATCH_SIZE iters)
float get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9f;
}

void warmup_gpu() {
    // Warm up GPU clocks with some busy work
    float* dummy;
    cudaMalloc(&dummy, 64 * 1024 * 1024);  // 64MB
    for (int i = 0; i < 100; i++) {
        cudaMemset(dummy, 0, 64 * 1024 * 1024);
    }
    cudaDeviceSynchronize();
    cudaFree(dummy);
}

const int BATCH_SIZE = 100;  // Check timeout every BATCH_SIZE iterations

float profile_kernel(kernel_fn fn, void* args, const char* name = nullptr) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup with timeout
    float warmup_start = get_time_sec();
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        fn(args);
        if (i % BATCH_SIZE == 0) {
            cudaDeviceSynchronize();
            if (get_time_sec() - warmup_start > TIMEOUT_SEC) break;
        }
    }
    cudaDeviceSynchronize();

    // Timed runs with timeout - check wall clock every BATCH_SIZE iters
    cudaProfilerStart();
    if (name) nvtxRangePushA(name);
    cudaEventRecord(start);

    float timing_start = get_time_sec();
    long iters = 0;
    float elapsed = 0;
    while (elapsed < TIMEOUT_SEC) {
        for (int i = 0; i < BATCH_SIZE; ++i) {
            fn(args);
        }
        iters += BATCH_SIZE;
        cudaDeviceSynchronize();
        elapsed = get_time_sec() - timing_start;
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
#ifdef USE_TORCH
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif
    return ms / iters;
}

#ifdef USE_TORCH
float profile_graph(kernel_fn fn, void* args, const char* name = nullptr) {
    cudaDeviceSynchronize();

    at::cuda::CUDAGraph cuda_graph;
    at::cuda::CUDAStream current_stream = at::cuda::getCurrentCUDAStream();

    // Warmup with timeout
    at::cuda::CUDAStream warmup_stream = at::cuda::getStreamFromPool();
    at::cuda::setCurrentCUDAStream(warmup_stream);
    float warmup_start = get_time_sec();
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        fn(args);
        if (i % BATCH_SIZE == 0) {
            warmup_stream.synchronize();
            if (get_time_sec() - warmup_start > TIMEOUT_SEC) break;
        }
    }
    warmup_stream.synchronize();

    // Capture graph
    at::cuda::CUDAStream cap_stream = at::cuda::getStreamFromPool();
    at::cuda::setCurrentCUDAStream(cap_stream);
    cuda_graph.capture_begin();
    fn(args);
    cuda_graph.capture_end();
    cap_stream.synchronize();

    cudaDeviceSynchronize();
    at::cuda::setCurrentCUDAStream(current_stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Timed runs with timeout
    cudaProfilerStart();
    if (name) nvtxRangePushA(name);
    cudaEventRecord(start);

    float timing_start = get_time_sec();
    long iters = 0;
    float elapsed = 0;
    while (elapsed < TIMEOUT_SEC) {
        for (int i = 0; i < BATCH_SIZE; ++i) {
            cuda_graph.replay();
        }
        iters += BATCH_SIZE;
        cudaDeviceSynchronize();
        elapsed = get_time_sec() - timing_start;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    if (name) nvtxRangePop();
    cudaProfilerStop();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / iters;
}
#endif

float rand1() {
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

// ============================================================================
// mingru_gate inference kernel profiling
// ============================================================================

typedef struct {
    precision_t* state;       // (B, H)
    precision_t* combined;    // (B, 3*H) = [hidden, gate, proj]
    precision_t* out;         // (B, H)
    precision_t* next_state;  // (B, H)
    int B;
    int H;
} MingruGateArgs;

MingruGateArgs* create_mingrugateargs(int batch, int hidden) {
    MingruGateArgs* args = (MingruGateArgs*)calloc(1, sizeof(MingruGateArgs));
    args->B = batch;
    args->H = hidden;

    int N_state = batch * hidden;
    int N_combined = batch * 3 * hidden;

    cudaMalloc(&args->state, N_state * sizeof(precision_t));
    cudaMalloc(&args->combined, N_combined * sizeof(precision_t));
    cudaMalloc(&args->out, N_state * sizeof(precision_t));
    cudaMalloc(&args->next_state, N_state * sizeof(precision_t));

    float* state_buf = (float*)malloc(N_state * sizeof(float));
    float* combined_buf = (float*)malloc(N_combined * sizeof(float));

    // Initialize state with positive values
    for (int i = 0; i < N_state; ++i) {
        state_buf[i] = fabsf(rand1()) + 0.1f;
    }
    // Initialize combined = [hidden, gate, proj]
    for (int b = 0; b < batch; ++b) {
        int base = b * 3 * hidden;
        for (int h = 0; h < hidden; ++h) {
            combined_buf[base + h] = rand1() * 5.0f;              // hidden
            combined_buf[base + hidden + h] = rand1() * 5.0f;     // gate
            combined_buf[base + 2 * hidden + h] = rand1() * 2.0f; // proj
        }
    }

    // Copy as float, kernel uses precision_t (may be bf16 or float)
    // For non-torch path with PRECISION_FLOAT, precision_t == float
    cudaMemcpy(args->state, state_buf, N_state * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->combined, combined_buf, N_combined * sizeof(float), cudaMemcpyHostToDevice);

    free(state_buf);
    free(combined_buf);
    return args;
}

void free_mingrugateargs(MingruGateArgs* args) {
    cudaFree(args->state);
    cudaFree(args->combined);
    cudaFree(args->out);
    cudaFree(args->next_state);
    free(args);
}

void run_mingrugate_forward(MingruGateArgs* args) {
    mingru_gate_inference_kernel<<<grid_size(args->B * args->H), BLOCK_SIZE>>>(
        args->out, args->next_state, args->combined, args->state,
        args->H, args->B);
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor state;     // (B, H)
    torch::Tensor combined;  // (B, 3*H)
    int B;
    int H;
} MingruGateArgsTorch;

MingruGateArgsTorch* create_mingrugateargs_torch(MingruGateArgs* raw) {
    MingruGateArgsTorch* args = new MingruGateArgsTorch();
    args->B = raw->B;
    args->H = raw->H;

    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    args->state = torch::from_blob(raw->state, {raw->B, raw->H}, opts);
    args->combined = torch::from_blob(raw->combined, {raw->B, 3 * raw->H}, opts);

    return args;
}

void run_mingrugate_forward_torch(MingruGateArgsTorch* args) {
    torch::NoGradGuard no_grad;
    mingru_gate(args->state, args->combined);
}

void run_mingrugate_forward_cpp(MingruGateArgsTorch* args) {
    torch::NoGradGuard no_grad;
    mingru_gate_cpp(args->state, args->combined);
}

void test_mingrugate_correct(MingruGateArgsTorch* args) {
    // Run CUDA kernel via torch wrapper
    auto kernel_outputs = mingru_gate(args->state, args->combined);
    auto kernel_out = kernel_outputs[0];
    auto kernel_next_state = kernel_outputs[1];

    // Run cpp reference
    auto cpp_outputs = mingru_gate_cpp(args->state, args->combined);
    auto cpp_out = cpp_outputs[0];
    auto cpp_next_state = cpp_outputs[1];

    // Numerical comparison
    float rtol = 1e-3f, atol = 1e-4f;
    bool out_match = torch::allclose(kernel_out.to(torch::kFloat32), cpp_out.to(torch::kFloat32), rtol, atol);
    float out_max_diff = (kernel_out.to(torch::kFloat32) - cpp_out.to(torch::kFloat32)).abs().max().item<float>();
    bool next_state_match = torch::allclose(kernel_next_state.to(torch::kFloat32), cpp_next_state.to(torch::kFloat32), rtol, atol);
    float next_state_max_diff = (kernel_next_state.to(torch::kFloat32) - cpp_next_state.to(torch::kFloat32)).abs().max().item<float>();

    printf("  correctness: out=%s(%.2e) next_state=%s(%.2e)\n",
           out_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", out_max_diff,
           next_state_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", next_state_max_diff);
}

#endif

void profile_mingrugate(int batch, int hidden) {
    printf("mingru_gate (B=%d, H=%d, combined=%dx%d)\n", batch, hidden, batch, 3*hidden);

    MingruGateArgs* args = create_mingrugateargs(batch, hidden);

    float fwd_ms = profile_kernel((kernel_fn)run_mingrugate_forward, args);
    print_timing("\tforward", fwd_ms, batch);

#ifdef USE_TORCH
    MingruGateArgsTorch* args_torch = create_mingrugateargs_torch(args);

    test_mingrugate_correct(args_torch);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_mingrugate_forward_torch, args_torch);
    print_timing("\tforward (torch)", fwd_torch_ms, batch);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_mingrugate_forward_cpp, args_torch);
    print_timing("\tforward (cpp)", fwd_cpp_ms, batch);

    float fwd_graph_ms = profile_graph((kernel_fn)run_mingrugate_forward_cpp, args_torch);
    print_timing("\tforward (graph)", fwd_graph_ms, batch);

    delete args_torch;
#endif
    printf("\n");

    free_mingrugateargs(args);
}

// ============================================================================
// logcumsumexp kernel profiling
// ============================================================================

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

LogcumsumexpArgs* create_logcumsumexpargs(int batch, int seq, int hidden) {
    LogcumsumexpArgs* args = (LogcumsumexpArgs*)calloc(1, sizeof(LogcumsumexpArgs));
    args->B = batch;
    args->T = seq;
    args->H = hidden;
    args->N = batch*seq * hidden;

    cudaMalloc(&args->x, args->N * sizeof(precision_t));
    cudaMalloc(&args->out, args->N * sizeof(precision_t));
    cudaMalloc(&args->s_buf, args->N * sizeof(double));
    cudaMalloc(&args->grad_x, args->N * sizeof(precision_t));
    cudaMalloc(&args->grad_out, args->N * sizeof(precision_t));

    float* buf = (float*)malloc(args->N * sizeof(float) * 2);
    float* x_buf = buf;
    float* grad_out_buf = buf + args->N;
    for (int i = 0; i < args->N; ++i) {
        x_buf[i] = rand1();
        grad_out_buf[i] = rand1();
    }

    cudaMemcpy(args->x, x_buf, args->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->grad_out, grad_out_buf, args->N * sizeof(float), cudaMemcpyHostToDevice);

    free(buf);
    return args;
}

void free_logcumsumexpargs(LogcumsumexpArgs* args) {
    cudaFree(args->x);
    cudaFree(args->out);
    cudaFree(args->s_buf);
    cudaFree(args->grad_x);
    cudaFree(args->grad_out);
    free(args);
}

void run_logcumsumexp_forward(LogcumsumexpArgs* args) {
    logcumsumexp_forward_kernel<<<grid_size(args->B * args->H), BLOCK_SIZE>>>(
        args->out, args->s_buf, args->x, args->T, args->H, args->B);
}

void run_logcumsumexp_backward(LogcumsumexpArgs* args) {
    logcumsumexp_backward_kernel<<<grid_size(args->B * args->H), BLOCK_SIZE>>>(
        args->grad_x, args->grad_out, args->x, args->s_buf, args->T, args->H, args->B);
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor x;
    torch::Tensor out;
    torch::Tensor grad_out;
    int N;
} LogcumsumexpArgsTorch;

LogcumsumexpArgsTorch* create_logcumsumexpargs_torch(LogcumsumexpArgs* raw) {
    LogcumsumexpArgsTorch* args = new LogcumsumexpArgsTorch();
    args->N = raw->N;

    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    args->x = torch::from_blob(raw->x, {raw->B, raw->T, raw->H}, opts).clone().to(torch::kFloat32).requires_grad_(true);
    args->grad_out = torch::from_blob(raw->grad_out, {raw->B, raw->T, raw->H}, opts).clone().to(torch::kFloat32);

    return args;
}

void run_logcumsumexp_forward_torch(LogcumsumexpArgsTorch* args) {
    torch::NoGradGuard no_grad;
    logcumsumexp_cuda(args->x);
}

void run_logcumsumexp_backward_torch(LogcumsumexpArgsTorch* args) {
    args->x.mutable_grad() = torch::Tensor();
    args->out.backward(args->grad_out, /*retain_graph=*/true);
}

void run_logcumsumexp_forward_cpp(LogcumsumexpArgsTorch* args) {
    torch::NoGradGuard no_grad;
    logcumsumexp_cpp(args->x);
}

void test_logcumsumexp_correct(LogcumsumexpArgsTorch* args) {
    // Kernel forward
    auto x_k = args->x.detach().clone().requires_grad_(true);
    auto out_k = logcumsumexp_cuda(x_k);

    // Cpp reference forward
    auto x_c = args->x.detach().clone().requires_grad_(true);
    auto out_c = logcumsumexp_cpp(x_c);

    float rtol = 1e-3f, atol = 1e-4f;
    float out_max_diff = (out_k - out_c).abs().max().item<float>();
    bool out_match = torch::allclose(out_k, out_c, rtol, atol);
    printf("  forward correctness: %s(%.2e)\n",
           out_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", out_max_diff);

    // Backward
    auto grad = args->grad_out.detach().clone();
    out_k.backward(grad, /*retain_graph=*/false);
    out_c.backward(grad, /*retain_graph=*/false);

    float grad_max_diff = (x_k.grad() - x_c.grad()).abs().max().item<float>();
    bool grad_match = torch::allclose(x_k.grad(), x_c.grad(), rtol, atol);
    printf("  backward correctness: %s(%.2e)\n",
           grad_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", grad_max_diff);
}

#endif

void profile_logcumsumexp(int batch, int seq, int hidden) {
    LogcumsumexpArgs* args = create_logcumsumexpargs(batch, seq, hidden);

    printf("logcumsumexp (N=%d, %dx%dx%d)\n", args->N, batch, seq, hidden);

    float fwd_ms = profile_kernel((kernel_fn)run_logcumsumexp_forward, args);
    print_timing("\tforward", fwd_ms, batch*seq);

    float bwd_ms = profile_kernel((kernel_fn)run_logcumsumexp_backward, args);
    print_timing("\tbackward", bwd_ms, batch*seq);

#ifdef USE_TORCH
    LogcumsumexpArgsTorch* args_torch = create_logcumsumexpargs_torch(args);

    test_logcumsumexp_correct(args_torch);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_logcumsumexp_forward_torch, args_torch);
    print_timing("\tforward (torch)", fwd_torch_ms, batch*seq);

    args_torch->out = logcumsumexp_cuda(args_torch->x);

    float bwd_torch_ms = profile_kernel((kernel_fn)run_logcumsumexp_backward_torch, args_torch);
    print_timing("\tbackward (torch)", bwd_torch_ms, batch*seq);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_logcumsumexp_forward_cpp, args_torch);
    print_timing("\tforward (cpp)", fwd_cpp_ms, batch*seq);

    args_torch->out = logcumsumexp_cpp(args_torch->x);

    float bwd_cpp_ms = profile_kernel((kernel_fn)run_logcumsumexp_backward_torch, args_torch);
    print_timing("\tbackward (cpp)", bwd_cpp_ms, batch*seq);

    float fwd_graph_ms = profile_graph((kernel_fn)run_logcumsumexp_forward_cpp, args_torch);
    print_timing("\tforward (graph)", fwd_graph_ms, batch*seq);

    delete args_torch;
#endif
    printf("\n");

    free_logcumsumexpargs(args);
}

// ============================================================================
// fused_scan (checkpointed) kernel profiling
// ============================================================================

typedef struct {
    precision_t* combined;       // (B, T, 3*H) = [hidden, gate, proj]
    precision_t* state;          // (B, 1, H)
    precision_t* out;            // (B, T, H)
    precision_t* next_state;     // (B, 1, H)
    float* a_star;         // (B, T+1, H)
    float* s_vals;         // (B, T+1, H)
    float* log_values_buf; // (B, T+1, H)
    precision_t* grad_combined;  // (B, T, 3*H)
    precision_t* grad_state;     // (B, 1, H)
    precision_t* grad_out;       // (B, T, H)
    precision_t* grad_next_state;// (B, 1, H)
    int B;
    int T;
    int H;
    int N;
} FusedScanArgs;

FusedScanArgs* create_fusedscanargs(int batch, int seq, int hidden) {
    FusedScanArgs* args = (FusedScanArgs*)calloc(1, sizeof(FusedScanArgs));
    args->B = batch;
    args->T = seq;
    args->H = hidden;
    args->N = batch * seq * hidden;

    int N_combined = batch * seq * 3 * hidden;
    int N_state = batch * hidden;
    int N_buf = batch * (seq + 1) * hidden;

    cudaMalloc(&args->combined, N_combined * sizeof(precision_t));
    cudaMalloc(&args->state, N_state * sizeof(precision_t));
    cudaMalloc(&args->out, args->N * sizeof(precision_t));
    cudaMalloc(&args->next_state, N_state * sizeof(precision_t));
    cudaMalloc(&args->a_star, N_buf * sizeof(float));
    cudaMalloc(&args->s_vals, N_buf * sizeof(float));
    cudaMalloc(&args->log_values_buf, N_buf * sizeof(float));
    cudaMalloc(&args->grad_combined, N_combined * sizeof(precision_t));
    cudaMalloc(&args->grad_state, N_state * sizeof(precision_t));
    cudaMalloc(&args->grad_out, args->N * sizeof(precision_t));
    cudaMalloc(&args->grad_next_state, N_state * sizeof(precision_t));

    // Allocate and initialize host buffers
    float* combined_buf = (float*)malloc(N_combined * sizeof(float));
    float* state_buf = (float*)malloc(N_state * sizeof(float));
    float* grad_out_buf = (float*)malloc(args->N * sizeof(float));
    float* grad_next_state_buf = (float*)malloc(N_state * sizeof(float));

    // Initialize combined = [hidden, gate, proj] with reasonable values
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < seq; ++t) {
            for (int h = 0; h < hidden; ++h) {
                int base = b * seq * 3 * hidden + t * 3 * hidden;
                combined_buf[base + h] = rand1() * 5.0f;             // hidden
                combined_buf[base + hidden + h] = rand1() * 5.0f;    // gate
                combined_buf[base + 2 * hidden + h] = rand1() * 2.0f; // proj
            }
        }
    }

    // Initialize state with positive values (will be log'd)
    for (int i = 0; i < N_state; ++i) {
        state_buf[i] = fabsf(rand1()) + 0.1f;
    }
    // Initialize gradients
    for (int i = 0; i < args->N; ++i) {
        grad_out_buf[i] = rand1();
    }
    for (int i = 0; i < N_state; ++i) {
        grad_next_state_buf[i] = rand1();
    }

    cudaMemcpy(args->combined, combined_buf, N_combined * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->state, state_buf, N_state * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->grad_out, grad_out_buf, args->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->grad_next_state, grad_next_state_buf, N_state * sizeof(float), cudaMemcpyHostToDevice);

    free(combined_buf);
    free(state_buf);
    free(grad_out_buf);
    free(grad_next_state_buf);
    return args;
}

void free_fusedscanargs(FusedScanArgs* args) {
    cudaFree(args->combined);
    cudaFree(args->state);
    cudaFree(args->out);
    cudaFree(args->next_state);
    cudaFree(args->a_star);
    cudaFree(args->s_vals);
    cudaFree(args->log_values_buf);
    cudaFree(args->grad_combined);
    cudaFree(args->grad_state);
    cudaFree(args->grad_out);
    cudaFree(args->grad_next_state);
    free(args);
}

void run_fusedscan_forward(FusedScanArgs* args) {
    fused_scan_forward_kernel_checkpointed<<<grid_size(args->B * args->H), BLOCK_SIZE>>>(
        args->out, args->next_state,
        args->a_star, args->s_vals, args->log_values_buf,
        args->combined, args->state,
        args->T, args->H, args->B);
}

void run_fusedscan_backward(FusedScanArgs* args) {
    fused_scan_backward_kernel_checkpointed<<<grid_size(args->B * args->H), BLOCK_SIZE>>>(
        args->grad_combined, args->grad_state,
        args->grad_out, args->grad_next_state,
        args->combined, args->state,
        args->a_star, args->s_vals, args->log_values_buf,
        args->T, args->H, args->B);
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor combined;    // (B, T, 3*H)
    torch::Tensor state;       // (B, 1, H)
    torch::Tensor out;         // (B, T, H)
    torch::Tensor next_state;  // (B, 1, H)
    torch::Tensor grad_out;    // (B, T, H)
    torch::Tensor grad_next_state; // (B, 1, H)
    int B;
    int T;
    int H;
} FusedScanArgsTorch;

FusedScanArgsTorch* create_fusedscanargs_torch(FusedScanArgs* raw) {
    FusedScanArgsTorch* args = new FusedScanArgsTorch();
    args->B = raw->B;
    args->T = raw->T;
    args->H = raw->H;

    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    args->combined = torch::from_blob(raw->combined, {raw->B, raw->T, 3 * raw->H}, opts).clone().to(torch::kFloat32).requires_grad_(true);
    args->state = torch::from_blob(raw->state, {raw->B, 1, raw->H}, opts).clone().to(torch::kFloat32).requires_grad_(true);
    args->grad_out = torch::from_blob(raw->grad_out, {raw->B, raw->T, raw->H}, opts).clone().to(torch::kFloat32);
    args->grad_next_state = torch::from_blob(raw->grad_next_state, {raw->B, 1, raw->H}, opts).clone().to(torch::kFloat32);

    return args;
}

void run_fusedscan_forward_torch(FusedScanArgsTorch* args) {
    torch::NoGradGuard no_grad;
    fused_scan_checkpointed(args->combined, args->state);
}

void run_fusedscan_backward_torch(FusedScanArgsTorch* args) {
    args->combined.mutable_grad() = torch::Tensor();
    args->state.mutable_grad() = torch::Tensor();
    torch::autograd::backward(
        {args->out, args->next_state},
        {args->grad_out, args->grad_next_state},
        /*retain_graph=*/true);
}

void run_fusedscan_forward_cpp(FusedScanArgsTorch* args) {
    torch::NoGradGuard no_grad;
    fused_scan_cpp(args->combined, args->state);
}

void test_fusedscan_correct(FusedScanArgsTorch* args) {
    // Run checkpointed kernel forward via torch wrapper
    auto combined_ref = args->combined.detach().clone().requires_grad_(true);
    auto state_ref = args->state.detach().clone().requires_grad_(true);
    combined_ref.retain_grad();
    state_ref.retain_grad();
    auto ref_outputs = fused_scan_checkpointed(combined_ref, state_ref);
    auto ref_out = ref_outputs[0];
    auto ref_next_state = ref_outputs[1];

    // Run cpp reference forward
    auto cpp_outputs = fused_scan_cpp(args->combined.detach(), args->state.detach());
    auto cpp_out = cpp_outputs[0];
    auto cpp_next_state = cpp_outputs[1];

    // Numerical comparison
    float rtol = 1e-3f, atol = 1e-4f;
    bool out_match = torch::allclose(ref_out, cpp_out, rtol, atol);
    float out_max_diff = (ref_out - cpp_out).abs().max().item<float>();
    bool next_state_match = torch::allclose(ref_next_state, cpp_next_state, rtol, atol);
    float next_state_max_diff = (ref_next_state - cpp_next_state).abs().max().item<float>();

    printf("  forward correctness: out=%s(%.2e) next_state=%s(%.2e)\n",
           out_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", out_max_diff,
           next_state_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", next_state_max_diff);

    // Test backward pass
    torch::autograd::backward({ref_out, ref_next_state}, {args->grad_out, args->grad_next_state});
    auto grad_combined_ref = combined_ref.grad().clone();
    auto grad_state_ref = state_ref.grad().clone();

    // Run cpp backward
    auto combined_cpp = args->combined.detach().clone().requires_grad_(true);
    auto state_cpp = args->state.detach().clone().requires_grad_(true);
    auto cpp_out2 = fused_scan_cpp(combined_cpp, state_cpp);
    torch::autograd::backward({cpp_out2[0], cpp_out2[1]}, {args->grad_out, args->grad_next_state});

    bool grad_combined_match = torch::allclose(grad_combined_ref, combined_cpp.grad(), rtol, atol);
    float grad_combined_max_diff = (grad_combined_ref - combined_cpp.grad()).abs().max().item<float>();
    bool grad_state_match = torch::allclose(grad_state_ref, state_cpp.grad(), rtol, atol);
    float grad_state_max_diff = (grad_state_ref - state_cpp.grad()).abs().max().item<float>();

    printf("  backward correctness: grad_combined=%s(%.2e) grad_state=%s(%.2e)\n",
           grad_combined_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", grad_combined_max_diff,
           grad_state_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", grad_state_max_diff);
}

#endif

void profile_fusedscan(int batch, int seq, int hidden) {
    FusedScanArgs* args = create_fusedscanargs(batch, seq, hidden);

    printf("fused_scan (N=%d, %dx%dx%d, combined=%dx%dx%d)\n",
           args->N, batch, seq, hidden, batch, seq, 3*hidden);

    float fwd_ms = profile_kernel((kernel_fn)run_fusedscan_forward, args);
    print_timing("\tforward", fwd_ms, batch*seq);

    float bwd_ms = profile_kernel((kernel_fn)run_fusedscan_backward, args);
    print_timing("\tbackward", bwd_ms, batch*seq);

#ifdef USE_TORCH
    FusedScanArgsTorch* args_torch = create_fusedscanargs_torch(args);

    test_fusedscan_correct(args_torch);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_fusedscan_forward_torch, args_torch);
    print_timing("\tforward (torch)", fwd_torch_ms, batch*seq);

    auto scan_out = fused_scan_checkpointed(args_torch->combined, args_torch->state);
    args_torch->out = scan_out[0];
    args_torch->next_state = scan_out[1];

    float bwd_torch_ms = profile_kernel((kernel_fn)run_fusedscan_backward_torch, args_torch);
    print_timing("\tbackward (torch)", bwd_torch_ms, batch*seq);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_fusedscan_forward_cpp, args_torch);
    print_timing("\tforward (cpp)", fwd_cpp_ms, batch*seq);

    auto scan_out_cpp = fused_scan_cpp(args_torch->combined, args_torch->state);
    args_torch->out = scan_out_cpp[0];
    args_torch->next_state = scan_out_cpp[1];

    float bwd_cpp_ms = profile_kernel((kernel_fn)run_fusedscan_backward_torch, args_torch);
    print_timing("\tbackward (cpp)", bwd_cpp_ms, batch*seq);

    float fwd_graph_ms = profile_graph((kernel_fn)run_fusedscan_forward_cpp, args_torch);
    print_timing("\tforward (graph)", fwd_graph_ms, batch*seq);

    delete args_torch;
#endif
    printf("\n");

    free_fusedscanargs(args);
}

// =============================================================================
// FCMax: Simple FC -> Max kernel (no intermediate ReLU layer)
// Input: x (B, N, D_in), W (D_out, D_in), b (D_out)
// Output: (B, D_out) = max_over_N(x @ W.T + b)
// =============================================================================

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

FCMaxArgs* create_fcmaxargs(int batch, int num_points, int d_in, int d_out) {
    FCMaxArgs* args = (FCMaxArgs*)calloc(1, sizeof(FCMaxArgs));
    args->B = batch;
    args->N = num_points;
    args->D_in = d_in;
    args->D_out = d_out;

    int N_x = batch * num_points * d_in;
    int N_W = d_out * d_in;
    int N_out = batch * d_out;

    cudaMalloc(&args->x, N_x * sizeof(precision_t));
    cudaMalloc(&args->W, N_W * sizeof(float));
    cudaMalloc(&args->b, d_out * sizeof(float));
    cudaMalloc(&args->out, N_out * sizeof(precision_t));
    cudaMalloc(&args->argmax_indices, N_out * sizeof(int));
    cudaMalloc(&args->grad_x, N_x * sizeof(float));
    cudaMalloc(&args->grad_W, N_W * sizeof(float));
    cudaMalloc(&args->grad_b, d_out * sizeof(float));
    cudaMalloc(&args->grad_out, N_out * sizeof(precision_t));

    float* x_buf = (float*)malloc(N_x * sizeof(float));
    float* W_buf = (float*)malloc(N_W * sizeof(float));
    float* b_buf = (float*)malloc(d_out * sizeof(float));
    float* grad_out_buf = (float*)malloc(N_out * sizeof(float));

    for (int i = 0; i < N_x; ++i) x_buf[i] = rand1();
    for (int i = 0; i < N_W; ++i) W_buf[i] = rand1() * 0.1f;
    for (int i = 0; i < d_out; ++i) b_buf[i] = 0.0f;
    for (int i = 0; i < N_out; ++i) grad_out_buf[i] = rand1();

    cudaMemcpy(args->x, x_buf, N_x * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->W, W_buf, N_W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->b, b_buf, d_out * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->grad_out, grad_out_buf, N_out * sizeof(float), cudaMemcpyHostToDevice);

    free(x_buf);
    free(W_buf);
    free(b_buf);
    free(grad_out_buf);
    return args;
}

void free_fcmaxargs(FCMaxArgs* args) {
    cudaFree(args->x);
    cudaFree(args->W);
    cudaFree(args->b);
    cudaFree(args->out);
    cudaFree(args->argmax_indices);
    cudaFree(args->grad_x);
    cudaFree(args->grad_W);
    cudaFree(args->grad_b);
    cudaFree(args->grad_out);
    free(args);
}

void run_fcmax_forward(FCMaxArgs* args) {
    fc_max_forward_kernel<<<grid_size(args->B * args->D_out), BLOCK_SIZE>>>(
        args->out, args->argmax_indices,
        args->x, args->W, args->b,
        args->B, args->N, args->D_in, args->D_out);
}

void run_fcmax_backward(FCMaxArgs* args) {
    cudaMemset(args->grad_x, 0, args->B * args->N * args->D_in * sizeof(float));
    cudaMemset(args->grad_W, 0, args->D_out * args->D_in * sizeof(float));
    cudaMemset(args->grad_b, 0, args->D_out * sizeof(float));

    fc_max_backward_kernel<<<grid_size(args->B * args->D_out), BLOCK_SIZE>>>(
        args->grad_x, args->grad_W, args->grad_b,
        args->grad_out, args->x, args->W,
        args->argmax_indices,
        args->B, args->N, args->D_in, args->D_out);
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor x;       // (B, N, D_in)
    torch::Tensor W;       // (D_out, D_in)
    torch::Tensor b;       // (D_out)
    torch::Tensor out;     // (B, D_out)
    torch::Tensor grad_out;// (B, D_out)
    int B, N, D_in, D_out;
} FCMaxArgsTorch;

FCMaxArgsTorch* create_fcmaxargs_torch(FCMaxArgs* raw) {
    FCMaxArgsTorch* args = new FCMaxArgsTorch();
    args->B = raw->B;
    args->N = raw->N;
    args->D_in = raw->D_in;
    args->D_out = raw->D_out;

    auto opts = cuda_f32;
    args->x = torch::from_blob(raw->x, {raw->B, raw->N, raw->D_in}, opts).clone().requires_grad_(true);
    args->W = torch::from_blob(raw->W, {raw->D_out, raw->D_in}, opts).clone().requires_grad_(true);
    args->b = torch::from_blob(raw->b, {raw->D_out}, opts).clone().requires_grad_(true);
    args->grad_out = torch::from_blob(raw->grad_out, {raw->B, raw->D_out}, opts).clone();

    return args;
}

void run_fcmax_forward_torch(FCMaxArgsTorch* args) {
    torch::NoGradGuard no_grad;
    fc_max(args->x, args->W, args->b);
}

void run_fcmax_backward_torch(FCMaxArgsTorch* args) {
    // Recompute forward each time since backward frees the graph
    auto out = fc_max(args->x, args->W, args->b);
    if (args->x.grad().defined()) args->x.grad().zero_();
    if (args->W.grad().defined()) args->W.grad().zero_();
    if (args->b.grad().defined()) args->b.grad().zero_();
    out.backward(args->grad_out);
}

void run_fcmax_forward_cpp(FCMaxArgsTorch* args) {
    torch::NoGradGuard no_grad;
    fc_max_cpp(args->x, args->W, args->b);
}

void test_fcmax_correct(FCMaxArgsTorch* args) {
    auto x_fused = args->x.detach().clone().requires_grad_(true);
    auto W_fused = args->W.detach().clone().requires_grad_(true);
    auto b_fused = args->b.detach().clone().requires_grad_(true);
    auto fused_out = fc_max(x_fused, W_fused, b_fused);

    auto x_ref = args->x.detach().clone().requires_grad_(true);
    auto W_ref = args->W.detach().clone().requires_grad_(true);
    auto b_ref = args->b.detach().clone().requires_grad_(true);
    auto ref_out = fc_max_cpp(x_ref, W_ref, b_ref);

    float rtol = 1e-3f, atol = 1e-4f;
    bool out_match = torch::allclose(fused_out, ref_out, rtol, atol);
    float out_max_diff = (fused_out - ref_out).abs().max().item<float>();

    printf("  forward correctness: out=%s(%.2e)\n",
           out_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", out_max_diff);

    // Backward
    fused_out.backward(args->grad_out);
    ref_out.backward(args->grad_out);

    bool grad_x_match = torch::allclose(x_fused.grad(), x_ref.grad(), rtol, atol);
    float grad_x_max_diff = (x_fused.grad() - x_ref.grad()).abs().max().item<float>();
    bool grad_W_match = torch::allclose(W_fused.grad(), W_ref.grad(), rtol, atol);
    float grad_W_max_diff = (W_fused.grad() - W_ref.grad()).abs().max().item<float>();

    printf("  backward correctness: grad_x=%s(%.2e) grad_W=%s(%.2e)\n",
           grad_x_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", grad_x_max_diff,
           grad_W_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", grad_W_max_diff);
}

#endif

void profile_fcmax(int batch, int num_points, int d_in, int d_out) {
    FCMaxArgs* args = create_fcmaxargs(batch, num_points, d_in, d_out);

    printf("fc_max (B=%d, N=%d, D_in=%d, D_out=%d)\n", batch, num_points, d_in, d_out);

    float fwd_ms = profile_kernel((kernel_fn)run_fcmax_forward, args);
    print_timing("\tforward", fwd_ms, batch);

    float bwd_ms = profile_kernel((kernel_fn)run_fcmax_backward, args);
    print_timing("\tbackward", bwd_ms, batch);

#ifdef USE_TORCH
    FCMaxArgsTorch* args_torch = create_fcmaxargs_torch(args);

    test_fcmax_correct(args_torch);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_fcmax_forward_torch, args_torch);
    print_timing("\tforward (torch)", fwd_torch_ms, batch);

    args_torch->out = fc_max(args_torch->x, args_torch->W, args_torch->b);

    float bwd_torch_ms = profile_kernel((kernel_fn)run_fcmax_backward_torch, args_torch);
    print_timing("\tbackward (torch)", bwd_torch_ms, batch);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_fcmax_forward_cpp, args_torch);
    print_timing("\tforward (cpp)", fwd_cpp_ms, batch);

    float fwd_graph_ms = profile_graph((kernel_fn)run_fcmax_forward_cpp, args_torch);
    print_timing("\tforward (graph)", fwd_graph_ms, batch);

    delete args_torch;
#endif
    printf("\n");

    free_fcmaxargs(args);
}

// ============================================================================
// PPO loss (optimized kernel) profiling
// ============================================================================

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

PPOLossArgs* create_ppolossargs(int batch, int seq, int actions) {
    PPOLossArgs* args = (PPOLossArgs*)calloc(1, sizeof(PPOLossArgs));
    args->N = batch;
    args->T = seq;
    args->A = actions;
    args->num_atns = 1;  // single action head for profiling

    int NT = batch*seq;
    int NTA = batch*seq * actions;

    cudaMalloc(&args->logits, NTA * sizeof(precision_t));
    cudaMalloc(&args->values_pred, NT * sizeof(precision_t));
    cudaMalloc(&args->actions, NT * sizeof(double));
    cudaMalloc(&args->old_logprobs, NT * sizeof(precision_t));
    cudaMalloc(&args->advantages, NT * sizeof(float));
    cudaMalloc(&args->prio, batch * sizeof(precision_t));
    cudaMalloc(&args->values, NT * sizeof(precision_t));
    cudaMalloc(&args->returns, NT * sizeof(precision_t));
    cudaMalloc(&args->adv_mean, sizeof(float));
    cudaMalloc(&args->adv_var, sizeof(float));
    cudaMalloc(&args->loss, sizeof(float));
    cudaMalloc(&args->saved_for_backward, NT * 5 * sizeof(double));
    cudaMalloc(&args->ratio_out, NT * sizeof(precision_t));
    cudaMalloc(&args->newvalue_out, NT * sizeof(precision_t));
    cudaMalloc(&args->grad_logits, NTA * sizeof(float));
    cudaMalloc(&args->grad_values_pred, NT * sizeof(float));
    cudaMalloc(&args->grad_loss, sizeof(float));
    cudaMalloc(&args->act_sizes, sizeof(int));

    // Set act_sizes to single head of size A
    cudaMemcpy(args->act_sizes, &actions, sizeof(int), cudaMemcpyHostToDevice);

    float* buf = (float*)malloc((NTA + NT * 5 + batch) * sizeof(float));
    float* logits_buf = buf;
    float* values_pred_buf = buf + NTA;
    float* old_logprobs_buf = buf + NTA + NT;
    float* advantages_buf = buf + NTA + NT * 2;
    float* values_buf = buf + NTA + NT * 3;
    float* returns_buf = buf + NTA + NT * 4;
    float* prio_buf = buf + NTA + NT * 5;

    double* actions_buf = (double*)malloc(NT * sizeof(double));

    float adv_sum = 0.0f, adv_sq_sum = 0.0f;
    for (int i = 0; i < NT; ++i) {
        advantages_buf[i] = rand1();
        adv_sum += advantages_buf[i];
        adv_sq_sum += advantages_buf[i] * advantages_buf[i];
    }
    float adv_mean = adv_sum / NT;
    float adv_var = adv_sq_sum / NT - adv_mean * adv_mean;

    for (int i = 0; i < NTA; ++i) {
        logits_buf[i] = rand1() * 2.0f;
    }
    for (int i = 0; i < NT; ++i) {
        values_pred_buf[i] = rand1();
        actions_buf[i] = (double)(rand() % actions);
        old_logprobs_buf[i] = rand1() * 2.0f;
        values_buf[i] = rand1();
        returns_buf[i] = rand1();
    }
    for (int i = 0; i < batch; ++i) {
        prio_buf[i] = (float)rand() / RAND_MAX;
    }

    cudaMemcpy(args->logits, logits_buf, NTA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->values_pred, values_pred_buf, NT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->actions, actions_buf, NT * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(args->old_logprobs, old_logprobs_buf, NT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->advantages, advantages_buf, NT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->prio, prio_buf, batch * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->values, values_buf, NT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->returns, returns_buf, NT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->adv_mean, &adv_mean, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->adv_var, &adv_var, sizeof(float), cudaMemcpyHostToDevice);

    float grad_loss_val = 1.0f;
    cudaMemcpy(args->grad_loss, &grad_loss_val, sizeof(float), cudaMemcpyHostToDevice);

    args->clip_coef = 0.1f;
    args->vf_clip_coef = 0.1f;
    args->vf_coef = 0.5f;
    args->ent_coef = 0.01f;

    args->logits_stride_n = seq * actions;  // T * A
    args->logits_stride_t = actions;        // A
    args->logits_stride_a = 1;
    args->values_stride_n = seq;            // T
    args->values_stride_t = 1;

    free(buf);
    free(actions_buf);
    return args;
}

void free_ppolossargs(PPOLossArgs* args) {
    cudaFree(args->logits);
    cudaFree(args->values_pred);
    cudaFree(args->actions);
    cudaFree(args->old_logprobs);
    cudaFree(args->advantages);
    cudaFree(args->prio);
    cudaFree(args->values);
    cudaFree(args->returns);
    cudaFree(args->adv_mean);
    cudaFree(args->adv_var);
    cudaFree(args->loss);
    cudaFree(args->saved_for_backward);
    cudaFree(args->ratio_out);
    cudaFree(args->newvalue_out);
    cudaFree(args->grad_logits);
    cudaFree(args->grad_values_pred);
    cudaFree(args->grad_loss);
    cudaFree(args->act_sizes);
    free(args);
}

void run_ppoloss_forward(PPOLossArgs* args) {
    int total = args->N * args->T;
    int ppo_grid = (total + PPO_THREADS - 1) / PPO_THREADS;
    cudaMemset(args->loss, 0, sizeof(float));
    ppo_loss_forward_kernel_optimized<<<ppo_grid, PPO_THREADS>>>(
        args->loss, args->saved_for_backward,
        args->ratio_out, args->newvalue_out,
        args->logits,
        nullptr,  // logstd (nullptr for discrete)
        args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, args->adv_var,
        args->act_sizes, args->num_atns,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        args->T, args->A, args->N,
        args->logits_stride_n, args->logits_stride_t, args->logits_stride_a,
        args->values_stride_n, args->values_stride_t,
        false);  // is_continuous
}

void run_ppoloss_backward(PPOLossArgs* args) {
    int total = args->N * args->T;
    int ppo_grid = (total + PPO_THREADS - 1) / PPO_THREADS;
    ppo_loss_backward_kernel_optimized<<<ppo_grid, PPO_THREADS>>>(
        args->grad_logits,
        nullptr,  // grad_logstd (nullptr for discrete)
        args->grad_values_pred, args->grad_loss,
        args->logits,
        nullptr,  // logstd (nullptr for discrete)
        args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, args->adv_var,
        args->act_sizes, args->num_atns,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        args->T, args->A, args->N,
        args->logits_stride_n, args->logits_stride_t, args->logits_stride_a,
        args->values_stride_n, args->values_stride_t,
        false);  // is_continuous
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor logits;
    torch::Tensor values_pred;
    torch::Tensor actions;
    torch::Tensor old_logprobs;
    torch::Tensor advantages;
    torch::Tensor prio;
    torch::Tensor values;
    torch::Tensor returns;
    torch::Tensor adv_mean;
    torch::Tensor adv_var;  // variance, kernel computes sqrt
    torch::Tensor ratio_out;
    torch::Tensor newvalue_out;
    torch::Tensor act_sizes;
    torch::Tensor loss;
    float clip_coef;
    float vf_clip_coef;
    float vf_coef;
    float ent_coef;
    int N;
    int T;
    int A;
} PPOLossArgsTorch;

PPOLossArgsTorch* create_ppolossargs_torch(PPOLossArgs* raw) {
    PPOLossArgsTorch* args = new PPOLossArgsTorch();
    args->N = raw->N;
    args->T = raw->T;
    args->A = raw->A;
    args->clip_coef = raw->clip_coef;
    args->vf_clip_coef = raw->vf_clip_coef;
    args->vf_coef = raw->vf_coef;
    args->ent_coef = raw->ent_coef;

    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);

    args->logits = torch::from_blob(raw->logits, {raw->N, raw->T, raw->A}, opts).clone().requires_grad_(true);
    args->values_pred = torch::from_blob(raw->values_pred, {raw->N, raw->T, 1}, opts).clone().requires_grad_(true);
    args->actions = torch::from_blob(raw->actions, {raw->N, raw->T, 1}, cuda_f64).clone();
    args->old_logprobs = torch::from_blob(raw->old_logprobs, {raw->N, raw->T}, opts).clone();
    args->advantages = torch::from_blob(raw->advantages, {raw->N, raw->T}, cuda_f32).clone();
    args->prio = torch::from_blob(raw->prio, {raw->N, 1}, opts).clone();
    args->values = torch::from_blob(raw->values, {raw->N, raw->T}, opts).clone();
    args->returns = torch::from_blob(raw->returns, {raw->N, raw->T}, opts).clone();
    args->adv_mean = torch::from_blob(raw->adv_mean, {1}, cuda_f32).clone();
    args->adv_var = torch::from_blob(raw->adv_var, {1}, cuda_f32).clone();
    args->ratio_out = torch::zeros({raw->N, raw->T}, opts);
    args->newvalue_out = torch::zeros({raw->N, raw->T}, opts);
    args->act_sizes = torch::tensor({raw->A}, cuda_i32);

    return args;
}

void run_ppoloss_forward_torch(PPOLossArgsTorch* args) {
    torch::NoGradGuard no_grad;
    auto logstd = torch::empty({0}, args->logits.options());  // empty for discrete
    fused_ppo_loss_optimized(
        args->logits, logstd, args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, args->adv_var,
        args->ratio_out, args->newvalue_out, args->act_sizes,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef);
}

void run_ppoloss_backward_torch(PPOLossArgsTorch* args) {
    args->logits.mutable_grad() = torch::Tensor();
    args->values_pred.mutable_grad() = torch::Tensor();
    args->loss.backward({}, /*retain_graph=*/true);
}

void test_ppoloss_correct(PPOLossArgsTorch* args) {
    int N = args->N;
    int T = args->T;
    int A = args->A;
    int minibatch_size = N * T;

    // Kernel path via compute_train_loss
    auto logits_k = args->logits.detach().clone().requires_grad_(true);
    auto values_pred_k = args->values_pred.detach().clone().requires_grad_(true);
    auto ratio_out_k = torch::zeros({N, T}, logits_k.options());
    auto newvalue_out_k = torch::zeros({N, T}, logits_k.options());
    Logits logits_struct_k = {.mean = logits_k};
    auto loss_k = compute_train_loss(
        logits_struct_k, values_pred_k,
        args->actions, args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, ratio_out_k, newvalue_out_k,
        args->act_sizes, torch::tensor({(int64_t)A}, torch::dtype(torch::kInt64)),
        minibatch_size, T,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        /*is_continuous=*/false, /*kernels=*/true);

    // Cpp reference path via compute_train_loss
    auto logits_c = args->logits.detach().clone().requires_grad_(true);
    auto values_pred_c = args->values_pred.detach().clone().requires_grad_(true);
    auto ratio_out_c = torch::zeros({N, T}, logits_c.options());
    auto newvalue_out_c = torch::zeros({N, T}, logits_c.options());
    Logits logits_struct_c = {.mean = logits_c};
    auto loss_c = compute_train_loss(
        logits_struct_c, values_pred_c,
        args->actions, args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, ratio_out_c, newvalue_out_c,
        args->act_sizes, torch::tensor({(int64_t)A}, torch::dtype(torch::kInt64)),
        minibatch_size, T,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        /*is_continuous=*/false, /*kernels=*/false);

    // Forward comparison
    float rtol = 1e-2f, atol = 1e-3f;
    float loss_diff = (loss_k - loss_c).abs().item<float>();
    bool loss_match = loss_diff < atol;
    printf("  forward correctness: loss=%s(%.2e)\n",
           loss_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", loss_diff);

    float ratio_max_diff = (ratio_out_k - ratio_out_c).abs().max().item<float>();
    bool ratio_match = torch::allclose(ratio_out_k, ratio_out_c, rtol, atol);
    printf("  ratio correctness: %s(%.2e)\n",
           ratio_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", ratio_max_diff);

    // Backward comparison
    loss_k.backward();
    loss_c.backward();

    auto grad_logits_k = logits_k.grad();
    auto grad_logits_c = logits_c.grad();
    float grad_logits_diff = (grad_logits_k - grad_logits_c).abs().max().item<float>();
    bool grad_logits_match = torch::allclose(grad_logits_k, grad_logits_c, rtol, atol);
    printf("  backward correctness: grad_logits=%s(%.2e)\n",
           grad_logits_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", grad_logits_diff);

    auto grad_values_k = values_pred_k.grad();
    auto grad_values_c = values_pred_c.grad();
    float grad_values_diff = (grad_values_k - grad_values_c).abs().max().item<float>();
    bool grad_values_match = torch::allclose(grad_values_k, grad_values_c, rtol, atol);
    printf("  backward correctness: grad_values=%s(%.2e)\n",
           grad_values_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", grad_values_diff);
}

#endif

void profile_ppoloss(int batch, int seq, int actions) {
    PPOLossArgs* args = create_ppolossargs(batch, seq, actions);

    int NT = batch*seq;
    printf("ppo_loss (NT=%d, %dx%d, A=%d)\n", NT, batch, seq, actions);

    float fwd_ms = profile_kernel((kernel_fn)run_ppoloss_forward, args);
    print_timing("\tforward", fwd_ms, NT);

    float bwd_ms = profile_kernel((kernel_fn)run_ppoloss_backward, args);
    print_timing("\tbackward", bwd_ms, NT);

#ifdef USE_TORCH
    PPOLossArgsTorch* args_torch = create_ppolossargs_torch(args);

    test_ppoloss_correct(args_torch);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_ppoloss_forward_torch, args_torch);
    print_timing("\tforward (torch)", fwd_torch_ms, NT);

    auto logstd_empty = torch::empty({0}, args_torch->logits.options());
    args_torch->loss = fused_ppo_loss_optimized(
        args_torch->logits, logstd_empty, args_torch->values_pred, args_torch->actions,
        args_torch->old_logprobs, args_torch->advantages, args_torch->prio,
        args_torch->values, args_torch->returns, args_torch->adv_mean, args_torch->adv_var,
        args_torch->ratio_out, args_torch->newvalue_out, args_torch->act_sizes,
        args_torch->clip_coef, args_torch->vf_clip_coef, args_torch->vf_coef, args_torch->ent_coef)[0];

    float bwd_torch_ms = profile_kernel((kernel_fn)run_ppoloss_backward_torch, args_torch);
    print_timing("\tbackward (torch)", bwd_torch_ms, NT);

    float fwd_graph_ms = profile_graph((kernel_fn)run_ppoloss_forward_torch, args_torch);
    print_timing("\tforward (graph)", fwd_graph_ms, NT);

    delete args_torch;
#endif
    printf("\n");

    free_ppolossargs(args);
}

// ============================================================================
// sample_logits profiling
// ============================================================================

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

SampleLogitsArgs* create_samplelogitsargs(int batch, int num_actions) {
    SampleLogitsArgs* args = (SampleLogitsArgs*)calloc(1, sizeof(SampleLogitsArgs));
    args->B = batch;
    args->A = num_actions;
    args->seed = 42;

    int N_logits = batch * num_actions;
    int N_batch = batch;

    cudaMalloc(&args->logits, N_logits * sizeof(precision_t));
    cudaMalloc(&args->value, N_batch * sizeof(precision_t));
    cudaMalloc(&args->actions, N_batch * sizeof(double));
    cudaMalloc(&args->logprobs, N_batch * sizeof(precision_t));
    cudaMalloc(&args->value_out, N_batch * sizeof(precision_t));
    cudaMalloc(&args->offset, sizeof(int64_t));
    cudaMalloc(&args->act_sizes, sizeof(int));
    cudaMemset(args->offset, 0, sizeof(int64_t));
    cudaMemcpy(args->act_sizes, &num_actions, sizeof(int), cudaMemcpyHostToDevice);

    float* logits_buf = (float*)malloc(N_logits * sizeof(float));
    float* value_buf = (float*)malloc(N_batch * sizeof(float));

    for (int i = 0; i < N_logits; ++i) {
        logits_buf[i] = rand1() * 5.0f;
    }
    for (int i = 0; i < N_batch; ++i) {
        value_buf[i] = rand1();
    }

    cudaMemcpy(args->logits, logits_buf, N_logits * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->value, value_buf, N_batch * sizeof(float), cudaMemcpyHostToDevice);

    free(logits_buf);
    free(value_buf);
    return args;
}

void free_samplelogitsargs(SampleLogitsArgs* args) {
    cudaFree(args->logits);
    cudaFree(args->value);
    cudaFree(args->actions);
    cudaFree(args->logprobs);
    cudaFree(args->value_out);
    cudaFree(args->offset);
    cudaFree(args->act_sizes);
    free(args);
}

void run_samplelogits_forward(SampleLogitsArgs* args) {
    sample_logits_kernel<<<grid_size(args->B), BLOCK_SIZE>>>(
        args->actions, args->logprobs, args->value_out,
        args->logits,
        nullptr,  // logstd (nullptr for discrete)
        args->value,
        args->act_sizes, args->seed, args->offset,
        1,  // num_atns
        args->B,
        args->A,  // logits_stride
        0,        // logstd_stride (unused for discrete)
        1,        // value_stride
        false);   // is_continuous
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor logits;       // (B, A)
    torch::Tensor value;        // (B, 1) - input
    torch::Tensor actions;      // (B, 1) float64 - output
    torch::Tensor logprobs;     // (B,) - output
    torch::Tensor value_out;    // (B,) - output
    torch::Tensor offset;       // (1,) int64 - RNG offset tensor
    torch::Tensor act_sizes;    // (1,) int32 - action sizes
    torch::Tensor act_sizes_cpu;// (1,) int64 - CPU version for cpp path
    uint64_t seed;
    int B;
    int A;
} SampleLogitsArgsTorch;

SampleLogitsArgsTorch* create_samplelogitsargs_torch(SampleLogitsArgs* raw) {
    SampleLogitsArgsTorch* args = new SampleLogitsArgsTorch();
    args->B = raw->B;
    args->A = raw->A;
    args->seed = 42;

    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    args->logits = torch::from_blob(raw->logits, {raw->B, raw->A}, opts);
    args->value = torch::from_blob(raw->value, {raw->B, 1}, opts);
    args->actions = torch::empty({raw->B, 1}, opts.dtype(torch::kFloat64));
    args->logprobs = torch::empty({raw->B}, opts);
    args->value_out = torch::empty({raw->B}, opts);
    args->offset = torch::zeros({1}, opts.dtype(torch::kInt64));
    args->act_sizes = torch::tensor({raw->A}, cuda_i32);
    args->act_sizes_cpu = torch::tensor({(int64_t)raw->A}, torch::dtype(torch::kInt64));

    return args;
}

void run_samplelogits_forward_torch(SampleLogitsArgsTorch* args) {
    torch::NoGradGuard no_grad;
    auto logstd = torch::Tensor();  // empty/undefined for discrete
    sample_logits(args->logits, logstd, args->value, args->actions, args->logprobs,
        args->value_out, args->act_sizes, args->seed, args->offset);
}

void run_samplelogits_forward_cpp(SampleLogitsArgsTorch* args) {
    torch::NoGradGuard no_grad;
    sample_discrete_cpp(args->logits, args->act_sizes_cpu, 1);
}

void test_samplelogits_correct(SampleLogitsArgsTorch* args) {
    torch::NoGradGuard no_grad;

    // Run kernel to get actions + logprobs
    auto logstd = torch::Tensor();
    sample_logits(args->logits, logstd, args->value, args->actions, args->logprobs,
        args->value_out, args->act_sizes, args->seed, args->offset);

    // Verify logprobs are consistent with sampled actions:
    // recompute logprobs from logits + actions using torch ops
    auto log_probs = torch::log_softmax(args->logits.to(torch::kFloat32), 1);
    auto actions_i64 = args->actions.to(torch::kInt64);
    auto expected_logprobs = log_probs.gather(1, actions_i64).squeeze(1);
    auto actual_logprobs = args->logprobs.to(torch::kFloat32);

    float rtol = 1e-2f, atol = 1e-3f;
    float logprob_max_diff = (actual_logprobs - expected_logprobs).abs().max().item<float>();
    bool logprob_match = torch::allclose(actual_logprobs, expected_logprobs, rtol, atol);
    printf("  logprob consistency: %s(%.2e)\n",
           logprob_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", logprob_max_diff);

    // Verify value passthrough
    auto expected_values = args->value.squeeze(1).to(torch::kFloat32);
    auto actual_values = args->value_out.to(torch::kFloat32);
    float value_max_diff = (actual_values - expected_values).abs().max().item<float>();
    bool value_match = torch::allclose(actual_values, expected_values, rtol, atol);
    printf("  value passthrough: %s(%.2e)\n",
           value_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", value_max_diff);

    // Verify actions are valid indices
    auto actions_flat = args->actions.to(torch::kInt64).flatten();
    bool valid_actions = (actions_flat.ge(0) & actions_flat.lt(args->A)).all().item<bool>();
    printf("  action validity: %s\n",
           valid_actions ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m");
}

#endif

void profile_samplelogits(int batch, int num_actions) {
    SampleLogitsArgs* args = create_samplelogitsargs(batch, num_actions);

    printf("sample_logits (B=%d, A=%d)\n", batch, num_actions);

    float fwd_ms = profile_kernel((kernel_fn)run_samplelogits_forward, args);
    print_timing("\tforward", fwd_ms, batch);

#ifdef USE_TORCH
    SampleLogitsArgsTorch* args_torch = create_samplelogitsargs_torch(args);

    test_samplelogits_correct(args_torch);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_samplelogits_forward_torch, args_torch);
    print_timing("\tforward (torch)", fwd_torch_ms, batch);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_samplelogits_forward_cpp, args_torch);
    print_timing("\tforward (cpp)", fwd_cpp_ms, batch);

    float fwd_graph_ms = profile_graph((kernel_fn)run_samplelogits_forward_torch, args_torch);
    print_timing("\tforward (graph)", fwd_graph_ms, batch);

    delete args_torch;
#endif
    printf("\n");

    free_samplelogitsargs(args);
}

// ============================================================================
// forward_call profiling (inference forward pass)
// ============================================================================

#ifdef USE_TORCH

typedef struct {
    std::shared_ptr<Policy> policy;
    Tensor obs;
    Tensor state;
    Tensor actions;
    Tensor logprobs;
    Tensor values;
    Tensor rng_offset;
    Tensor act_sizes;
    Tensor act_sizes_cpu;
    uint64_t seed;
    bool use_kernels;
    int batch;
} ForwardCallArgs;

ForwardCallArgs* create_forwardcallargs(int batch, int input_size, int hidden_size,
                                        int act_n, int num_layers, bool use_kernels) {
    int num_action_heads = 1;  // Using discrete for profiling

    ForwardCallArgs* args = new ForwardCallArgs();
    args->use_kernels = use_kernels;
    args->seed = 42;
    args->batch = batch;

    // Create policy: Policy(encoder, decoder, rnn, input_size, num_atns, hidden_size)
    auto enc = std::make_shared<DefaultEncoder>(input_size, hidden_size);
    auto dec = std::make_shared<DefaultDecoder>(hidden_size, act_n);
    auto rnn = std::make_shared<MinGRU>(hidden_size, num_layers, use_kernels);
    args->policy = std::make_shared<Policy>(enc, dec, rnn, input_size, act_n, hidden_size);
    args->policy->to(torch::kCUDA);

    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    args->obs = torch::randn({batch, input_size}, opts);
    args->state = args->policy->initial_state(batch, torch::kCUDA);
    args->actions = torch::zeros({batch, num_action_heads}, cuda_f64);
    args->logprobs = torch::zeros({batch}, opts);
    args->values = torch::zeros({batch}, opts);
    args->rng_offset = torch::zeros({1}, cuda_i64);
    args->act_sizes = torch::tensor({act_n}, cuda_i32);
    args->act_sizes_cpu = torch::tensor({(int64_t)act_n}, torch::dtype(torch::kInt64));

    return args;
}

void free_forwardcallargs(ForwardCallArgs* args) {
    delete args;
}

void run_forward_call(ForwardCallArgs* args) {
    torch::NoGradGuard no_grad;

    // Run policy forward - returns tuple<Logits, Tensor, Tensor>
    auto [logits_out, value, state_out] = args->policy->forward(args->obs, args->state);

    // Sample actions using shared sample_actions from models.cpp
    sample_actions(logits_out, value, args->actions, args->logprobs, args->values,
        args->act_sizes, args->act_sizes_cpu,
        /*is_continuous=*/false, args->use_kernels, args->seed, args->rng_offset);

    // Update state
    args->state.copy_(state_out, false);
}

#endif

void profile_forwardcall(int batch, int input_size, int hidden_size, int num_atns, int num_layers) {
#ifdef USE_TORCH
    printf("forward_call (B=%d, in=%d, H=%d, A=%d, layers=%d)\n",
           batch, input_size, hidden_size, num_atns, num_layers);

    // Profile with kernels
    ForwardCallArgs* args_kernel = create_forwardcallargs(batch, input_size, hidden_size, num_atns, num_layers, true);
    float fwd_kernel_ms = profile_kernel((kernel_fn)run_forward_call, args_kernel, "forward_call_kernel");
    print_timing("\tkernel path", fwd_kernel_ms, batch);

    float fwd_kernel_graph_ms = profile_graph((kernel_fn)run_forward_call, args_kernel, "forward_call_kernel_graph");
    print_timing("\tkernel (graph)", fwd_kernel_graph_ms, batch);

    free_forwardcallargs(args_kernel);

    // Profile without kernels (torch ops path)
    ForwardCallArgs* args_torch = create_forwardcallargs(batch, input_size, hidden_size, num_atns, num_layers, false);
    float fwd_torch_ms = profile_kernel((kernel_fn)run_forward_call, args_torch, "forward_call_torch");
    print_timing("\ttorch path", fwd_torch_ms, batch);

    float fwd_torch_graph_ms = profile_graph((kernel_fn)run_forward_call, args_torch, "forward_call_torch_graph");
    print_timing("\ttorch (graph)", fwd_torch_graph_ms, batch);

    free_forwardcallargs(args_torch);

    printf("\n");
#else
    printf("forward_call: requires USE_TORCH\n\n");
#endif
}

// ============================================================================
// Environment speed test - uses static linking (compile with specific env)
// ============================================================================

#ifdef USE_STATIC_ENV

#include "pufferlib/extensions/env_binding.h"
#include "pufferlib/extensions/ini.h"

#ifndef ENV_NAME
#error "ENV_NAME must be defined at compile time (e.g. -DENV_NAME=breakout)"
#endif
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// Empty callback for OMP test (no-op, just testing env stepping speed)
static void empty_net_callback(void* ctx, int buf, int t) {
    (void)ctx; (void)buf; (void)t;
}

static void empty_thread_init(void* ctx, int buf) {
    (void)ctx; (void)buf;
}

typedef struct {
    StaticVec* vec;
    int num_envs;
    int num_buffers;
    int num_threads;
    int horizon;
    int obs_size;
    int num_atns;
} EnvSpeedArgs;

// INI parser handler - load [env] section into Dict
static int ini_handler_env(void* user, const char* section,
                           const char* name, const char* value) {
    Dict* env_kwargs = (Dict*)user;
    if (strcmp(section, "env") == 0) {
        dict_set(env_kwargs, strdup(name), atof(value));
    }
    return 1;
}

// INI parser handler - load [vec] section for defaults
typedef struct { int total_agents; int num_buffers; } VecDefaults;
static int ini_handler_vec(void* user, const char* section,
                           const char* name, const char* value) {
    VecDefaults* defaults = (VecDefaults*)user;
    if (strcmp(section, "vec") == 0) {
        if (strcmp(name, "total_agents") == 0) defaults->total_agents = atoi(value);
        else if (strcmp(name, "num_buffers") == 0) defaults->num_buffers = atoi(value);
    }
    return 1;
}

EnvSpeedArgs* create_envspeedargs(int total_agents, int num_buffers, int num_threads, int horizon) {
    // Load config from .ini file
    char ini_path[512];
    snprintf(ini_path, sizeof(ini_path), "pufferlib/config/ocean/%s.ini", TOSTRING(ENV_NAME));

    VecDefaults defaults = {0};
    if (ini_parse(ini_path, ini_handler_vec, &defaults) < 0) {
        fprintf(stderr, "Warning: Could not load config %s\n", ini_path);
    }

    // Use INI defaults if CLI args not provided (0 means use default)
    if (total_agents == 0) total_agents = defaults.total_agents > 0 ? defaults.total_agents : 8192;
    if (num_buffers == 0) num_buffers = defaults.num_buffers > 0 ? defaults.num_buffers : 2;

    // Load env_kwargs from [env] section
    Dict* env_kwargs = create_dict(64);
    if (ini_parse(ini_path, ini_handler_env, env_kwargs) < 0) {
        fprintf(stderr, "Warning: Could not load [env] config from %s\n", ini_path);
    }

    // Create vec_kwargs
    Dict* vec_kwargs = create_dict(8);
    dict_set(vec_kwargs, "total_agents", (double)total_agents);
    dict_set(vec_kwargs, "num_buffers", (double)num_buffers);

    // Create environments using static binding
    StaticVec* vec = create_static_vec(total_agents, num_buffers, vec_kwargs, env_kwargs);
    if (!vec) {
        fprintf(stderr, "Failed to create environments\n");
        return nullptr;
    }
    for (int i = 0; i < num_buffers; i++) {
        cudaStreamCreateWithFlags(&vec->streams[i], cudaStreamNonBlocking);
    }

    int num_envs = vec->size;
    printf("Created %d envs (%s) for %d total_agents\n", num_envs, TOSTRING(ENV_NAME), total_agents);

    // Create threads for OMP stepping
    create_static_threads(vec, num_threads, horizon, nullptr, empty_net_callback, empty_thread_init);

    // Reset
    static_vec_reset(vec);
    cudaDeviceSynchronize();

    EnvSpeedArgs* args = (EnvSpeedArgs*)calloc(1, sizeof(EnvSpeedArgs));
    args->vec = vec;
    args->num_envs = num_envs;
    args->num_buffers = num_buffers;
    args->num_threads = num_threads;
    args->horizon = horizon;
    args->obs_size = get_obs_size();
    args->num_atns = get_num_atns();

    return args;
}

void free_envspeedargs(EnvSpeedArgs* args) {
    // Note: no static_vec_close yet, just free the args
    free(args);
}

// Run full rollout using OMP threading
void run_env_rollout(EnvSpeedArgs* args) {
    static_vec_omp_step(args->vec);
}

float profile_env_rollout(EnvSpeedArgs* args, const char* name) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    float start_time = get_time_sec();
    for (int i = 0; i < 100; ++i) {
        run_env_rollout(args);
        cudaDeviceSynchronize();
        auto elapsed = get_time_sec() - start_time;
        if (elapsed > TIMEOUT_SEC) break;
    }

    start_time = get_time_sec();
    cudaProfilerStart();
    if (name) nvtxRangePushA(name);
    cudaEventRecord(start);
    float completed = 0;
    for (int i = 0; i < 1000; ++i) {
        run_env_rollout(args);
        completed += 1;
        float elapsed = get_time_sec() - start_time;
        if (elapsed > TIMEOUT_SEC) break;
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    if (name) nvtxRangePop();
    cudaProfilerStop();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / completed;  // per rollout
}

void profile_envspeed(int total_agents, int num_buffers, int num_threads, int horizon) {
    printf("env_speed_static (total_agents=%d, buffers=%d, threads=%d, horizon=%d)\n",
           total_agents, num_buffers, num_threads, horizon);

    EnvSpeedArgs* args = create_envspeedargs(total_agents, num_buffers, num_threads, horizon);
    if (!args) {
        printf("\tFailed to create env - skipping\n\n");
        return;
    }

    printf("\tnum_envs=%d, obs_size=%d, num_atns=%d\n", args->num_envs, args->obs_size, args->num_atns);

    // Profile full rollout (horizon steps per OMP call)
    float rollout_ms = profile_env_rollout(args, "env_rollout");
    int total_steps = total_agents * horizon;
    printf("\trollout time: %.2f ms (%d steps)\n", rollout_ms, total_steps);

    // Compute throughput
    float sps = total_steps / rollout_ms * 1000.0f;
    printf("\tthroughput: %.2f M steps/s\n", sps / 1e6);

    free_envspeedargs(args);
    printf("\n");
}

#endif  // USE_STATIC_ENV

void print_usage(const char* prog) {
    printf("Usage: %s <profile> [options]\n", prog);
    printf("  kernels        - Individual kernel profiling (no nsys needed)\n");
#ifdef USE_TORCH
    printf("  forwardcall    - Inference forward pass\n");
#endif
#ifdef USE_STATIC_ENV
    printf("  envspeed       - Environment step throughput (static linked)\n");
    printf("    --buffers N  - Number of buffers (default: %d)\n", BUF);
    printf("    --threads N  - Number of threads (default: 16)\n");
    printf("    --horizon N  - Horizon length (default: %d)\n", T);
#endif
    printf("  all            - Run all profiles\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* profile = argv[1];

    // Parse optional CLI args for envspeed
    int buffers = BUF;
    int threads = 16;
    int horizon = T;
    int total_agents = BR * buffers;
    for (int i = 2; i < argc - 1; i++) {
        if (strcmp(argv[i], "--buffers") == 0) buffers = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0) threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--horizon") == 0) horizon = atoi(argv[++i]);
        else if (strcmp(argv[i], "--total-agents") == 0) total_agents = atoi(argv[++i]);
    }

    warmup_gpu();

    // Using typical breakout settings: INPUT_SIZE=96, H=128, A=4

    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "all") == 0) {
        profile_mingrugate(BR, H);
        profile_logcumsumexp(BT, T, H);
        profile_fusedscan(BT, T, H);
        profile_samplelogits(BR, A);
        profile_ppoloss(BT, T, A);

        // FCMax: simple FC -> Max (no intermediate layer)
        // Drive encoder dimensions: partner (B, 63, 7) -> 128, road (B, 200, 13) -> 128
        profile_fcmax(BR, 63, 7, 128);    // partner encoder
        profile_fcmax(BR, 200, 13, 128);  // road encoder
    }

#ifdef USE_TORCH
    if (strcmp(profile, "forwardcall") == 0 || strcmp(profile, "all") == 0) {
        profile_forwardcall(BR, INPUT_SIZE, H, A, 1);
    }
#endif

#ifdef USE_STATIC_ENV
    if (strcmp(profile, "envspeed") == 0 || strcmp(profile, "all") == 0) {
        profile_envspeed(total_agents, buffers, threads, horizon);
    }
#endif

    return 0;
}
