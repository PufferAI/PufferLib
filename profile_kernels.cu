// profile_kernels.cu
// Minimal standalone profiler for CUDA kernels
//
// Without torch: nvcc -O3 -arch=sm_80 profile_kernels.cu -o profile_kernels -I.
// With torch:    Build with cmake/pytorch and -DUSE_TORCH
//
// Run: ./profile_kernels

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef USE_TORCH
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGraph.h>
#include "pufferlib/extensions/cuda/modules.cu"
#else
#include "pufferlib/extensions/cuda/kernels.cu"
#endif

const int WARMUP_ITERS = 1000;
const int TIMING_ITERS = 10000;

const int BR = 4096;  // Rollout batch (no T dim)
const int BT = 512;   // Train batch (with T dim)
const int T = 64;
const int H = 128;
const int A = 4;

typedef void (*kernel_fn)(void*);

void print_timing(const char* name, float ms, int N) {
    printf("  %-18s %6.1f us  %6.2f M elem/s\n", name, ms * 1000, N / ms / 1e3);
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

float profile_kernel(kernel_fn fn, void* args) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        fn(args);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(start);
    for (int i = 0; i < TIMING_ITERS; ++i) {
        fn(args);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();
    c10::cuda::CUDACachingAllocator::emptyCache();
    return ms / TIMING_ITERS;
}

#ifdef USE_TORCH
float profile_graph(kernel_fn fn, void* args) {
    cudaDeviceSynchronize();

    at::cuda::CUDAGraph cuda_graph;
    at::cuda::CUDAStream current_stream = at::cuda::getCurrentCUDAStream();

    at::cuda::CUDAStream warmup_stream = at::cuda::getStreamFromPool();
    at::cuda::setCurrentCUDAStream(warmup_stream);
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        fn(args);
    }
    warmup_stream.synchronize();

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

    cudaEventRecord(start);
    for (int i = 0; i < TIMING_ITERS; ++i) {
        cuda_graph.replay();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / TIMING_ITERS;
}
#endif

float rand1() {
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

// Fused mingru_gate for inference: takes combined (B, 1, 3*H) = [hidden, gate, proj]
// Outputs: out = sigmoid(proj) * mingru_out, next_state = mingru_out (for recurrence)
typedef struct {
    float* state;       // (B, 1, H) - input state
    float* combined;    // (B, 1, 3*H) = [hidden, gate, proj]
    float* out;         // (B, 1, H) - sigmoid(proj) * mingru_out
    float* next_state;  // (B, 1, H) - raw mingru_out
    int B;
    int H;
} MingruGateArgs;

MingruGateArgs* create_mingrugateargs(int batch, int hidden) {
    MingruGateArgs* args = (MingruGateArgs*)calloc(1, sizeof(MingruGateArgs));
    args->B = batch;
    args->H = hidden;

    int N_state = batch * hidden;
    int N_combined = batch * 3 * hidden;

    cudaMalloc(&args->state, N_state * sizeof(float));
    cudaMalloc(&args->combined, N_combined * sizeof(float));
    cudaMalloc(&args->out, N_state * sizeof(float));
    cudaMalloc(&args->next_state, N_state * sizeof(float));

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
    launch_mingru_gate_inference<float>(
        args->out, args->next_state, args->combined, args->state,
        args->H, args->B, 0);
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor state;     // (B, 1, H)
    torch::Tensor combined;  // (B, 1, 3*H)
    int B;
    int H;
} MingruGateArgsTorch;

MingruGateArgsTorch* create_mingrugateargs_torch(MingruGateArgs* raw) {
    MingruGateArgsTorch* args = new MingruGateArgsTorch();
    args->B = raw->B;
    args->H = raw->H;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    args->state = torch::from_blob(raw->state, {raw->B, 1, raw->H}, opts);
    args->combined = torch::from_blob(raw->combined, {raw->B, 1, 3 * raw->H}, opts);

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

#endif

void profile_mingrugate(int batch, int hidden) {
    MingruGateArgs* args = create_mingrugateargs(batch, hidden);

    printf("mingru_gate (B=%d, H=%d, combined=%dx%d)\n", batch, hidden, batch, 3*hidden);

    float fwd_ms = profile_kernel((kernel_fn)run_mingrugate_forward, args);
    print_timing("\tforward", fwd_ms, batch);

#ifdef USE_TORCH
    MingruGateArgsTorch* args_torch = create_mingrugateargs_torch(args);

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

typedef struct {
    float* gate;
    float* hidden;
    float* log_coeffs;
    float* log_values;
    float* grad_log_coeffs;
    float* grad_log_values;
    float* grad_gate;
    float* grad_hidden;
    int N;
} LogCoeffsAndValuesArgs;

LogCoeffsAndValuesArgs* create_logcoeffsandvaluesargs(int batch, int seq, int hidden) {
    LogCoeffsAndValuesArgs* args = (LogCoeffsAndValuesArgs*)calloc(1, sizeof(LogCoeffsAndValuesArgs));
    args->N = batch*seq * hidden;

    cudaMalloc(&args->gate, args->N * sizeof(float));
    cudaMalloc(&args->hidden, args->N * sizeof(float));
    cudaMalloc(&args->log_coeffs, args->N * sizeof(float));
    cudaMalloc(&args->log_values, args->N * sizeof(float));
    cudaMalloc(&args->grad_gate, args->N * sizeof(float));
    cudaMalloc(&args->grad_hidden, args->N * sizeof(float));
    cudaMalloc(&args->grad_log_coeffs, args->N * sizeof(float));
    cudaMalloc(&args->grad_log_values, args->N * sizeof(float));

    float* gate_buf = (float*)malloc(args->N * sizeof(float));
    float* hidden_buf = (float*)malloc(args->N * sizeof(float));
    float* grad_log_coeffs_buf = (float*)malloc(args->N * sizeof(float));
    float* grad_log_values_buf = (float*)malloc(args->N * sizeof(float));
    for (int i = 0; i < args->N; ++i) {
        gate_buf[i] = rand1() * 5.0f;
        hidden_buf[i] = rand1() * 5.0f;
        grad_log_coeffs_buf[i] = rand1();
        grad_log_values_buf[i] = rand1();
    }

    cudaMemcpy(args->gate, gate_buf, args->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->hidden, hidden_buf, args->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->grad_log_coeffs, grad_log_coeffs_buf, args->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->grad_log_values, grad_log_values_buf, args->N * sizeof(float), cudaMemcpyHostToDevice);

    free(gate_buf);
    free(hidden_buf);
    free(grad_log_coeffs_buf);
    free(grad_log_values_buf);

    return args;
}

void free_logcoeffsandvaluesargs(LogCoeffsAndValuesArgs* args) {
    cudaFree(args->gate);
    cudaFree(args->hidden);
    cudaFree(args->log_coeffs);
    cudaFree(args->log_values);
    cudaFree(args->grad_gate);
    cudaFree(args->grad_hidden);
    cudaFree(args->grad_log_coeffs);
    cudaFree(args->grad_log_values);
    free(args);
}

void run_logcoeffsandvalues_forward(LogCoeffsAndValuesArgs* args) {
    launch_log_coeffs_and_values<float>(
        args->log_coeffs, args->log_values, args->gate, args->hidden, args->N, 0);
}

void run_logcoeffsandvalues_backward(LogCoeffsAndValuesArgs* args) {
    launch_log_coeffs_and_values_backward<float>(
        args->grad_gate, args->grad_hidden, args->grad_log_coeffs,
        args->grad_log_values, args->gate, args->hidden, args->N, 0);
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor gate;
    torch::Tensor hidden;
    torch::Tensor grad_log_coeffs;
    torch::Tensor grad_log_values;
    torch::Tensor out_log_coeffs;
    torch::Tensor out_log_values;
    int N;
} LogCoeffsAndValuesArgsTorch;

LogCoeffsAndValuesArgsTorch* create_logcoeffsandvaluesargs_torch(LogCoeffsAndValuesArgs* raw) {
    LogCoeffsAndValuesArgsTorch* args = new LogCoeffsAndValuesArgsTorch();
    args->N = raw->N;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    args->gate = torch::from_blob(raw->gate, {raw->N}, opts).requires_grad_(true);
    args->hidden = torch::from_blob(raw->hidden, {raw->N}, opts).requires_grad_(true);
    args->grad_log_coeffs = torch::from_blob(raw->grad_log_coeffs, {raw->N}, opts);
    args->grad_log_values = torch::from_blob(raw->grad_log_values, {raw->N}, opts);

    return args;
}

void run_logcoeffsandvalues_forward_torch(LogCoeffsAndValuesArgsTorch* args) {
    torch::NoGradGuard no_grad;
    log_coeffs_and_values(args->gate, args->hidden);
}

void run_logcoeffsandvalues_backward_torch(LogCoeffsAndValuesArgsTorch* args) {
    args->gate.mutable_grad() = torch::Tensor();
    args->hidden.mutable_grad() = torch::Tensor();
    torch::autograd::backward(
        {args->out_log_coeffs, args->out_log_values},
        {args->grad_log_coeffs, args->grad_log_values},
        /*retain_graph=*/true);
}

void run_logcoeffsandvalues_forward_cpp(LogCoeffsAndValuesArgsTorch* args) {
    torch::NoGradGuard no_grad;
    log_coeffs_and_values_cpp(args->gate, args->hidden);
}

#endif

void profile_logcoeffsandvalues(int batch, int seq, int hidden) {
    LogCoeffsAndValuesArgs* args = create_logcoeffsandvaluesargs(batch, seq, hidden);

    printf("log_coeffs_and_values (N=%d, %dx%dx%d)\n", args->N, batch, seq, hidden);

    float fwd_ms = profile_kernel((kernel_fn)run_logcoeffsandvalues_forward, args);
    print_timing("\tforward", fwd_ms, batch*seq);

    float bwd_ms = profile_kernel((kernel_fn)run_logcoeffsandvalues_backward, args);
    print_timing("\tbackward", bwd_ms, batch*seq);

#ifdef USE_TORCH
    LogCoeffsAndValuesArgsTorch* args_torch = create_logcoeffsandvaluesargs_torch(args);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_logcoeffsandvalues_forward_torch, args_torch);
    print_timing("\tforward (torch)", fwd_torch_ms, batch*seq);

    auto kernel_outputs = log_coeffs_and_values(args_torch->gate, args_torch->hidden);
    args_torch->out_log_coeffs = kernel_outputs[0];
    args_torch->out_log_values = kernel_outputs[1];

    float bwd_torch_ms = profile_kernel((kernel_fn)run_logcoeffsandvalues_backward_torch, args_torch);
    print_timing("\tbackward (torch)", bwd_torch_ms, batch*seq);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_logcoeffsandvalues_forward_cpp, args_torch);
    print_timing("\tforward (cpp)", fwd_cpp_ms, batch*seq);

    auto cpp_outputs = log_coeffs_and_values_cpp(args_torch->gate, args_torch->hidden);
    args_torch->out_log_coeffs = cpp_outputs[0];
    args_torch->out_log_values = cpp_outputs[1];

    float bwd_cpp_ms = profile_kernel((kernel_fn)run_logcoeffsandvalues_backward_torch, args_torch);
    print_timing("\tbackward (cpp)", bwd_cpp_ms, batch*seq);

    float fwd_graph_ms = profile_graph((kernel_fn)run_logcoeffsandvalues_forward_cpp, args_torch);
    print_timing("\tforward (graph)", fwd_graph_ms, batch*seq);

    delete args_torch;
#endif
    printf("\n");

    free_logcoeffsandvaluesargs(args);
}

typedef struct {
    float* x;
    float* out;
    double* s_buf;
    float* grad_x;
    float* grad_out;
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

    cudaMalloc(&args->x, args->N * sizeof(float));
    cudaMalloc(&args->out, args->N * sizeof(float));
    cudaMalloc(&args->s_buf, args->N * sizeof(double));
    cudaMalloc(&args->grad_x, args->N * sizeof(float));
    cudaMalloc(&args->grad_out, args->N * sizeof(float));

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
    launch_logcumsumexp_forward<float>(
        args->out, args->s_buf, args->x, args->T, args->H, args->B, 0);
}

void run_logcumsumexp_backward(LogcumsumexpArgs* args) {
    launch_logcumsumexp_backward<float>(
        args->grad_x, args->grad_out, args->x, args->s_buf, args->T, args->H, args->B, 0);
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

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    args->x = torch::from_blob(raw->x, {raw->B, raw->T, raw->H}, opts).requires_grad_(true);
    args->grad_out = torch::from_blob(raw->grad_out, {raw->B, raw->T, raw->H}, opts);

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

// New fused_scan takes combined (B, T, 3*H) = [hidden, gate, proj] and state (B, 1, H)
// Outputs: out (B, T, H) = sigmoid(proj) * scan_result, next_state (B, 1, H)
typedef struct {
    float* combined;       // (B, T, 3*H) = [hidden, gate, proj]
    float* state;          // (B, 1, H)
    float* out;            // (B, T, H)
    float* next_state;     // (B, 1, H)
    float* a_star;         // (B, T+1, H)
    float* s_vals;         // (B, T+1, H)
    float* log_values_buf; // (B, T+1, H)
    float* grad_combined;  // (B, T, 3*H)
    float* grad_state;     // (B, 1, H)
    float* grad_out;       // (B, T, H)
    float* grad_next_state;// (B, 1, H)
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

    cudaMalloc(&args->combined, N_combined * sizeof(float));
    cudaMalloc(&args->state, N_state * sizeof(float));
    cudaMalloc(&args->out, args->N * sizeof(float));
    cudaMalloc(&args->next_state, N_state * sizeof(float));
    cudaMalloc(&args->a_star, N_buf * sizeof(float));
    cudaMalloc(&args->s_vals, N_buf * sizeof(float));
    cudaMalloc(&args->log_values_buf, N_buf * sizeof(float));
    cudaMalloc(&args->grad_combined, N_combined * sizeof(float));
    cudaMalloc(&args->grad_state, N_state * sizeof(float));
    cudaMalloc(&args->grad_out, args->N * sizeof(float));
    cudaMalloc(&args->grad_next_state, N_state * sizeof(float));

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
    launch_fused_scan_forward<float>(
        args->out, args->next_state,
        args->a_star, args->s_vals, args->log_values_buf,
        args->combined, args->state,
        args->T, args->H, args->B, 0);
}

void run_fusedscan_backward(FusedScanArgs* args) {
    launch_fused_scan_backward<float>(
        args->grad_combined, args->grad_state,
        args->grad_out, args->grad_next_state,
        args->combined, args->state,
        args->a_star, args->s_vals, args->log_values_buf,
        args->T, args->H, args->B, 0);
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

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    args->combined = torch::from_blob(raw->combined, {raw->B, raw->T, 3 * raw->H}, opts).requires_grad_(true);
    args->state = torch::from_blob(raw->state, {raw->B, 1, raw->H}, opts).requires_grad_(true);
    args->grad_out = torch::from_blob(raw->grad_out, {raw->B, raw->T, raw->H}, opts);
    args->grad_next_state = torch::from_blob(raw->grad_next_state, {raw->B, 1, raw->H}, opts);

    return args;
}

void run_fusedscan_forward_torch(FusedScanArgsTorch* args) {
    torch::NoGradGuard no_grad;
    fused_scan(args->combined, args->state);
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

    float fwd_torch_ms = profile_kernel((kernel_fn)run_fusedscan_forward_torch, args_torch);
    print_timing("\tforward (torch)", fwd_torch_ms, batch*seq);

    auto scan_out = fused_scan(args_torch->combined, args_torch->state);
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

typedef struct {
    float* logits;
    float* values_pred;
    int64_t* actions;
    float* old_logprobs;
    float* advantages;
    float* prio;
    float* values;
    float* returns;
    float* adv_mean;
    float* adv_std;
    float* loss;
    double* saved_for_backward;
    float* grad_logits;
    float* grad_values_pred;
    float* grad_loss;
    float clip_coef;
    float vf_clip_coef;
    float vf_coef;
    float ent_coef;
    int N;
    int T;
    int A;
} PPOLossArgs;

PPOLossArgs* create_ppolossargs(int batch, int seq, int actions) {
    PPOLossArgs* args = (PPOLossArgs*)calloc(1, sizeof(PPOLossArgs));
    args->N = batch;
    args->T = seq;
    args->A = actions;

    int NT = batch*seq;
    int NTA = batch*seq * actions;

    cudaMalloc(&args->logits, NTA * sizeof(float));
    cudaMalloc(&args->values_pred, NT * sizeof(float));
    cudaMalloc(&args->actions, NT * sizeof(int64_t));
    cudaMalloc(&args->old_logprobs, NT * sizeof(float));
    cudaMalloc(&args->advantages, NT * sizeof(float));
    cudaMalloc(&args->prio, batch * sizeof(float));
    cudaMalloc(&args->values, NT * sizeof(float));
    cudaMalloc(&args->returns, NT * sizeof(float));
    cudaMalloc(&args->adv_mean, sizeof(float));
    cudaMalloc(&args->adv_std, sizeof(float));
    cudaMalloc(&args->loss, sizeof(float));
    cudaMalloc(&args->saved_for_backward, NT * 5 * sizeof(double));
    cudaMalloc(&args->grad_logits, NTA * sizeof(float));
    cudaMalloc(&args->grad_values_pred, NT * sizeof(float));
    cudaMalloc(&args->grad_loss, sizeof(float));

    float* buf = (float*)malloc((NTA + NT * 5 + batch) * sizeof(float));
    float* logits_buf = buf;
    float* values_pred_buf = buf + NTA;
    float* old_logprobs_buf = buf + NTA + NT;
    float* advantages_buf = buf + NTA + NT * 2;
    float* values_buf = buf + NTA + NT * 3;
    float* returns_buf = buf + NTA + NT * 4;
    float* prio_buf = buf + NTA + NT * 5;

    int64_t* actions_buf = (int64_t*)malloc(NT * sizeof(int64_t));

    float adv_sum = 0.0f, adv_sq_sum = 0.0f;
    for (int i = 0; i < NT; ++i) {
        advantages_buf[i] = rand1();
        adv_sum += advantages_buf[i];
        adv_sq_sum += advantages_buf[i] * advantages_buf[i];
    }
    float adv_mean = adv_sum / NT;
    float adv_std = sqrtf(adv_sq_sum / NT - adv_mean * adv_mean);

    for (int i = 0; i < NTA; ++i) {
        logits_buf[i] = rand1() * 2.0f;
    }
    for (int i = 0; i < NT; ++i) {
        values_pred_buf[i] = rand1();
        actions_buf[i] = rand() % actions;
        old_logprobs_buf[i] = rand1() * 2.0f;
        values_buf[i] = rand1();
        returns_buf[i] = rand1();
    }
    for (int i = 0; i < batch; ++i) {
        prio_buf[i] = (float)rand() / RAND_MAX;
    }

    cudaMemcpy(args->logits, logits_buf, NTA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->values_pred, values_pred_buf, NT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->actions, actions_buf, NT * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(args->old_logprobs, old_logprobs_buf, NT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->advantages, advantages_buf, NT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->prio, prio_buf, batch * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->values, values_buf, NT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->returns, returns_buf, NT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->adv_mean, &adv_mean, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->adv_std, &adv_std, sizeof(float), cudaMemcpyHostToDevice);

    float grad_loss_val = 1.0f;
    cudaMemcpy(args->grad_loss, &grad_loss_val, sizeof(float), cudaMemcpyHostToDevice);

    args->clip_coef = 0.1f;
    args->vf_clip_coef = 0.1f;
    args->vf_coef = 0.5f;
    args->ent_coef = 0.01f;

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
    cudaFree(args->adv_std);
    cudaFree(args->loss);
    cudaFree(args->saved_for_backward);
    cudaFree(args->grad_logits);
    cudaFree(args->grad_values_pred);
    cudaFree(args->grad_loss);
    free(args);
}

void run_ppoloss_forward(PPOLossArgs* args) {
    launch_ppo_loss_forward<float>(
        args->loss, args->saved_for_backward,
        args->logits, args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, args->adv_std,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        args->T, args->A, args->N, 0);
}

void run_ppoloss_backward(PPOLossArgs* args) {
    launch_ppo_loss_backward<float>(
        args->grad_logits, args->grad_values_pred, args->grad_loss,
        args->logits, args->actions, args->old_logprobs,
        args->advantages, args->prio, args->values, args->returns,
        args->saved_for_backward, args->adv_mean, args->adv_std,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        args->T, args->A, args->N, 0);
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
    torch::Tensor adv_std;
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

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto opts_int = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

    args->logits = torch::from_blob(raw->logits, {raw->N, raw->T, raw->A}, opts).requires_grad_(true);
    args->values_pred = torch::from_blob(raw->values_pred, {raw->N, raw->T}, opts).requires_grad_(true);
    args->actions = torch::from_blob(raw->actions, {raw->N, raw->T}, opts_int);
    args->old_logprobs = torch::from_blob(raw->old_logprobs, {raw->N, raw->T}, opts);
    args->advantages = torch::from_blob(raw->advantages, {raw->N, raw->T}, opts);
    args->prio = torch::from_blob(raw->prio, {raw->N}, opts);
    args->values = torch::from_blob(raw->values, {raw->N, raw->T}, opts);
    args->returns = torch::from_blob(raw->returns, {raw->N, raw->T}, opts);
    args->adv_mean = torch::from_blob(raw->adv_mean, {1}, opts);
    args->adv_std = torch::from_blob(raw->adv_std, {1}, opts);

    return args;
}

void run_ppoloss_forward_torch(PPOLossArgsTorch* args) {
    torch::NoGradGuard no_grad;
    fused_ppo_loss(
        args->logits, args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, args->adv_std,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef);
}

void run_ppoloss_backward_torch(PPOLossArgsTorch* args) {
    args->logits.mutable_grad() = torch::Tensor();
    args->values_pred.mutable_grad() = torch::Tensor();
    args->loss.backward({}, /*retain_graph=*/true);
}

void run_ppoloss_forward_cpp(PPOLossArgsTorch* args) {
    torch::NoGradGuard no_grad;
    fused_ppo_loss_cpp(
        args->logits, args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, args->adv_std,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef);
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

    float fwd_torch_ms = profile_kernel((kernel_fn)run_ppoloss_forward_torch, args_torch);
    print_timing("\tforward (torch)", fwd_torch_ms, NT);

    args_torch->loss = fused_ppo_loss(
        args_torch->logits, args_torch->values_pred, args_torch->actions,
        args_torch->old_logprobs, args_torch->advantages, args_torch->prio,
        args_torch->values, args_torch->returns, args_torch->adv_mean, args_torch->adv_std,
        args_torch->clip_coef, args_torch->vf_clip_coef, args_torch->vf_coef, args_torch->ent_coef)[0];

    float bwd_torch_ms = profile_kernel((kernel_fn)run_ppoloss_backward_torch, args_torch);
    print_timing("\tbackward (torch)", bwd_torch_ms, NT);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_ppoloss_forward_cpp, args_torch);
    print_timing("\tforward (cpp)", fwd_cpp_ms, NT);

    args_torch->loss = fused_ppo_loss_cpp(
        args_torch->logits, args_torch->values_pred, args_torch->actions,
        args_torch->old_logprobs, args_torch->advantages, args_torch->prio,
        args_torch->values, args_torch->returns, args_torch->adv_mean, args_torch->adv_std,
        args_torch->clip_coef, args_torch->vf_clip_coef, args_torch->vf_coef, args_torch->ent_coef);

    float bwd_cpp_ms = profile_kernel((kernel_fn)run_ppoloss_backward_torch, args_torch);
    print_timing("\tbackward (cpp)", bwd_cpp_ms, NT);

    float fwd_graph_ms = profile_graph((kernel_fn)run_ppoloss_forward_cpp, args_torch);
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
    float* logits;        // (B, A)
    float* value;         // (B, 1)
    double* actions;      // (B,) - float64 for discrete/continuous compatibility
    float* logprobs;      // (B,)
    float* value_out;     // (B,)
    int64_t* offset;      // RNG offset (on device for CUDA graph support)
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

    cudaMalloc(&args->logits, N_logits * sizeof(float));
    cudaMalloc(&args->value, N_batch * sizeof(float));  // (B, 1) flattened
    cudaMalloc(&args->actions, N_batch * sizeof(double));
    cudaMalloc(&args->logprobs, N_batch * sizeof(float));
    cudaMalloc(&args->value_out, N_batch * sizeof(float));
    cudaMalloc(&args->offset, sizeof(int64_t));
    cudaMemset(args->offset, 0, sizeof(int64_t));  // Initialize offset to 0

    float* logits_buf = (float*)malloc(N_logits * sizeof(float));
    float* value_buf = (float*)malloc(N_batch * sizeof(float));

    // Initialize logits and value with random values
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
    free(args);
}

void run_samplelogits_forward(SampleLogitsArgs* args) {
    launch_sample_logits<float>(
        args->actions, args->logprobs, args->value_out,
        args->logits, args->value,
        args->seed, args->offset,
        args->A, args->B,
        args->A,  // logits_stride = A (contiguous)
        1,        // value_stride = 1 (contiguous, 1D)
        0);
    // Note: not incrementing offset here since this is just for profiling
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor logits;       // (B, A)
    torch::Tensor value;        // (B, 1) - input
    torch::Tensor actions;      // (B,) float64 - output
    torch::Tensor logprobs;     // (B,) - output
    torch::Tensor value_out;    // (B,) - output
    torch::Tensor offset;       // (1,) int64 - RNG offset tensor
    uint64_t seed;
    int B;
    int A;
} SampleLogitsArgsTorch;

SampleLogitsArgsTorch* create_samplelogitsargs_torch(SampleLogitsArgs* raw) {
    SampleLogitsArgsTorch* args = new SampleLogitsArgsTorch();
    args->B = raw->B;
    args->A = raw->A;
    args->seed = 42;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    args->logits = torch::from_blob(raw->logits, {raw->B, raw->A}, opts);
    args->value = torch::from_blob(raw->value, {raw->B, 1}, opts);
    args->actions = torch::empty({raw->B}, opts.dtype(torch::kFloat64));
    args->logprobs = torch::empty({raw->B}, opts);
    args->value_out = torch::empty({raw->B}, opts);
    args->offset = torch::zeros({1}, opts.dtype(torch::kInt64));

    return args;
}

void run_samplelogits_forward_torch(SampleLogitsArgsTorch* args) {
    torch::NoGradGuard no_grad;
    sample_logits(args->logits, args->value, args->actions, args->logprobs, args->value_out, args->seed, args->offset);
    args->offset.add_(1);  // Increment with CUDA op
}

void run_samplelogits_forward_cpp(SampleLogitsArgsTorch* args) {
    torch::NoGradGuard no_grad;
    sample_logits_cpp(args->logits);
}

#endif

void profile_samplelogits(int batch, int num_actions) {
    SampleLogitsArgs* args = create_samplelogitsargs(batch, num_actions);

    printf("sample_logits (B=%d, A=%d)\n", batch, num_actions);

    float fwd_ms = profile_kernel((kernel_fn)run_samplelogits_forward, args);
    print_timing("\tforward", fwd_ms, batch);

#ifdef USE_TORCH
    SampleLogitsArgsTorch* args_torch = create_samplelogitsargs_torch(args);

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

int main(int argc, char** argv) {
    warmup_gpu();
    //profile_mingrugate(BR, H);
    //profile_logcoeffsandvalues(BT, T, H);
    //profile_logcumsumexp(BT, T, H);
    //profile_fusedscan(BT, T, H);
    profile_samplelogits(BR, A);
    //profile_ppoloss(BT, T, A);
    return 0;
}
