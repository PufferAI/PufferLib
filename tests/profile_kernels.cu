#include <string>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <chrono>

#include "pufferl.cu"
#include "ini.h"

const int WARMUP_ITERS = 100;
const int TIMING_ITERS = 1000;

const int BUF = 2;
const int BR = 4096;   // Rollout batch (no T dim)
const int BT = 512;    // Train batch (with T dim)
const int T_ = 64;     // T_ to avoid collision with PrefixScan::T
const int H_ = 128;
const int A_ = 4;
const int INPUT_SIZE = 96;

#ifndef ENV_NAME
#error "ENV_NAME must be defined at compile time (e.g. -DENV_NAME=breakout)"
#endif
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

typedef void (*kernel_fn)(void*);

void print_usage(const char* prog) {
    printf("Usage: %s <profile>\n", prog);
    printf("\nProfiles:\n");
    printf("  kernels        - All individual kernel microbenchmarks\n");
    printf("  mingrugate     - MinGRU gate kernel only\n");
    printf("  logcoeffsvals  - log_coeffs_and_values fwd+bwd\n");
    printf("  fusedscan      - Fused scan (checkpointed) kernel only\n");
    printf("  samplelogits   - Sample logits kernel only\n");
    printf("  ppoloss        - PPO loss fused fwd+bwd kernel\n");
    printf("  im2col         - im2col + col2im (nmmo3 conv sizes, B=1024)\n");
    printf("  minimalenc     - Minimal entity encoder core ops vs cuBLAS\n");
    printf("    --batch N    - Encoder batch size (default: %d)\n", BR);
    printf("    --hidden N   - Encoder hidden size (default: %d)\n", H_);
    printf("  envspeed       - Environment step throughput\n");
    printf("    --buffers N  - Number of buffers (default: %d)\n", BUF);
    printf("    --threads N  - Number of threads (default: 16)\n");
    printf("    --horizon N  - Horizon length (default: %d)\n", T_);
    printf("  all            - Run all available profiles\n");
}

inline void print_timing(const char* name, float ms, int N) {
    printf("  %-28s %8.1f us  %8.2f M elem/s\n", name, ms * 1000, N / ms / 1e3);
}

inline void print_encoder_timing(const char* name, float ms, int B, double flops, double bytes) {
    double samples_per_s_m = B / ms / 1e3;
    double gflops = flops / ms / 1e6;
    double gbps = bytes / ms / 1e6;
    printf("  %-30s %8.1f us  %8.2f M samples/s  %8.1f GF/s  %8.1f GB/s\n",
        name, ms * 1000, samples_per_s_m, gflops, gbps);
}

inline void warmup_gpu() {
    float* dummy;
    cudaMalloc(&dummy, 64 * 1024 * 1024);
    for (int i = 0; i < 100; i++) cudaMemset(dummy, 0, 64 * 1024 * 1024);
    cudaDeviceSynchronize();
    cudaFree(dummy);
}

inline float rand1() {
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

inline void float_to_device(precision_t* dst, const float* src, int count) {
    precision_t* tmp = (precision_t*)malloc(count * sizeof(precision_t));
    for (int i = 0; i < count; ++i) tmp[i] = (precision_t)src[i];
    cudaMemcpy(dst, tmp, count * sizeof(precision_t), cudaMemcpyHostToDevice);
    free(tmp);
}

inline float profile_kernel(kernel_fn fn, void* args) {
    for (int i = 0; i < WARMUP_ITERS; ++i) fn(args);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaProfilerStart();
    cudaEventRecord(start);
    for (int i = 0; i < TIMING_ITERS; ++i) fn(args);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaProfilerStop();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceSynchronize();
    return ms / TIMING_ITERS;
}

struct MingruGateProfile {
    PrecisionTensor state, combined, x_in, out, next_state;
    Allocator alloc;
    int B, H;
};

MingruGateProfile* create_mingrugate(int B, int H) {
    auto* p = (MingruGateProfile*)calloc(1, sizeof(MingruGateProfile));
    p->B = B; p->H = H;
    p->state     = {.shape = {B, H}};
    p->combined  = {.shape = {B, 3*H}};
    p->x_in      = {.shape = {B, H}};
    p->out       = {.shape = {B, H}};
    p->next_state = {.shape = {B, H}};
    p->alloc = {};
    alloc_register(&p->alloc, &p->state);
    alloc_register(&p->alloc, &p->combined);
    alloc_register(&p->alloc, &p->x_in);
    alloc_register(&p->alloc, &p->out);
    alloc_register(&p->alloc, &p->next_state);
    alloc_create(&p->alloc);

    int N = B * H;
    float* buf = (float*)malloc((N + 3*N + N) * sizeof(float));
    for (int i = 0; i < N; ++i) buf[i] = fabsf(rand1()) + 0.1f;
    float_to_device(p->state.data, buf, N);
    for (int i = 0; i < 3*N; ++i) buf[i] = rand1() * 5.0f;
    float_to_device(p->combined.data, buf, 3*N);
    for (int i = 0; i < N; ++i) buf[i] = rand1();
    float_to_device(p->x_in.data, buf, N);
    free(buf);
    return p;
}

void run_mingrugate(MingruGateProfile* p) {
    mingru_gate<<<grid_size(p->B * p->H), BLOCK_SIZE>>>(
        p->out.data, p->next_state.data, p->combined.data,
        p->state.data, p->x_in.data, p->H, p->B);
}

void profile_mingrugate(int B, int H) {
    printf("mingru_gate (B=%d, H=%d)\n", B, H);
    auto* p = create_mingrugate(B, H);
    float ms = profile_kernel((kernel_fn)run_mingrugate, p);
    print_timing("forward", ms, B);
    printf("\n");
    alloc_free(&p->alloc);
    free(p);
}

__global__ void log_coeffs_and_values_fwd_kernel(
        float* log_coeff_out, float* log_value_out,
        const float* gate, const float* hidden, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    log_coeffs_and_values_fwd(gate[idx], hidden[idx],
        &log_coeff_out[idx], &log_value_out[idx]);
}

__global__ void log_coeffs_and_values_bwd_kernel(
        float* grad_gate_out, float* grad_hidden_out,
        const float* grad_log_coeffs, const float* grad_log_values,
        const float* gate, const float* hidden, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    log_coeffs_and_values_bwd(grad_log_coeffs[idx], grad_log_values[idx],
        gate[idx], hidden[idx], &grad_gate_out[idx], &grad_hidden_out[idx]);
}

struct LogCoeffsProfile {
    FloatTensor gate, hidden, log_coeff, log_value;
    FloatTensor grad_log_coeffs, grad_log_values, grad_gate, grad_hidden;
    Allocator alloc;
    int N;
};

LogCoeffsProfile* create_logcoeffs(int N) {
    auto* p = (LogCoeffsProfile*)calloc(1, sizeof(LogCoeffsProfile));
    p->N = N;
    p->gate = {.shape = {N}};
    p->hidden = {.shape = {N}};
    p->log_coeff = {.shape = {N}};
    p->log_value = {.shape = {N}};
    p->grad_log_coeffs = {.shape = {N}};
    p->grad_log_values = {.shape = {N}};
    p->grad_gate = {.shape = {N}};
    p->grad_hidden = {.shape = {N}};
    p->alloc = {};
    alloc_register(&p->alloc, &p->gate);
    alloc_register(&p->alloc, &p->hidden);
    alloc_register(&p->alloc, &p->log_coeff);
    alloc_register(&p->alloc, &p->log_value);
    alloc_register(&p->alloc, &p->grad_log_coeffs);
    alloc_register(&p->alloc, &p->grad_log_values);
    alloc_register(&p->alloc, &p->grad_gate);
    alloc_register(&p->alloc, &p->grad_hidden);
    alloc_create(&p->alloc);

    float* buf = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) buf[i] = rand1() * 5.0f;
    cudaMemcpy(p->gate.data, buf, N * sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < N; ++i) buf[i] = rand1() * 5.0f;
    cudaMemcpy(p->hidden.data, buf, N * sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < N; ++i) buf[i] = rand1();
    cudaMemcpy(p->grad_log_coeffs.data, buf, N * sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < N; ++i) buf[i] = rand1();
    cudaMemcpy(p->grad_log_values.data, buf, N * sizeof(float), cudaMemcpyHostToDevice);
    free(buf);
    return p;
}

void run_logcoeffs_fwd(LogCoeffsProfile* p) {
    log_coeffs_and_values_fwd_kernel<<<grid_size(p->N), BLOCK_SIZE>>>(
        p->log_coeff.data, p->log_value.data,
        p->gate.data, p->hidden.data, p->N);
}

void run_logcoeffs_bwd(LogCoeffsProfile* p) {
    log_coeffs_and_values_bwd_kernel<<<grid_size(p->N), BLOCK_SIZE>>>(
        p->grad_gate.data, p->grad_hidden.data,
        p->grad_log_coeffs.data, p->grad_log_values.data,
        p->gate.data, p->hidden.data, p->N);
}

void profile_logcoeffs(int B, int T, int H) {
    int N = B * T * H;
    printf("log_coeffs_and_values (N=%d, %dx%dx%d)\n", N, B, T, H);
    auto* p = create_logcoeffs(N);
    float fwd = profile_kernel((kernel_fn)run_logcoeffs_fwd, p);
    print_timing("forward", fwd, N);
    float bwd = profile_kernel((kernel_fn)run_logcoeffs_bwd, p);
    print_timing("backward", bwd, N);
    printf("\n");
    alloc_free(&p->alloc);
    free(p);
}

struct FusedScanProfile {
    PrefixScan scan;
    PrecisionTensor grad_out;
    Allocator alloc;
    int B, T, H;
};

FusedScanProfile* create_fusedscan(int B, int T, int H) {
    auto* p = (FusedScanProfile*)calloc(1, sizeof(FusedScanProfile));
    p->B = B; p->T = T; p->H = H;

    PrefixScan& s = p->scan;
    s.B = B; s.T = T; s.H = H;

    // Allocator needs PrecisionTensor/FloatTensor, but PrefixScan uses raw ptrs
    // for combined/state/input. Allocate those via tensors then assign.
    PrecisionTensor combined_t = {.shape = {B, T, 3*H}};
    PrecisionTensor state_t    = {.shape = {B, H}};
    PrecisionTensor input_t    = {.shape = {B, T, H}};

    s.out            = {.shape = {B, T, H}};
    s.next_state     = {.shape = {B, H}};
    s.a_star         = {.shape = {B, T+1, H}};
    s.s_vals         = {.shape = {B, T+1, H}};
    s.log_values_buf = {.shape = {B, T+1, H}};
    s.grad_combined  = {.shape = {B, T, 3*H}};
    s.grad_state     = {.shape = {B, H}};
    s.grad_input     = {.shape = {B, T, H}};

    p->grad_out        = {.shape = {B, T, H}};

    p->alloc = {};
    alloc_register(&p->alloc, &combined_t);
    alloc_register(&p->alloc, &state_t);
    alloc_register(&p->alloc, &input_t);
    alloc_register(&p->alloc, &s.out);
    alloc_register(&p->alloc, &s.next_state);
    alloc_register(&p->alloc, &s.a_star);
    alloc_register(&p->alloc, &s.s_vals);
    alloc_register(&p->alloc, &s.log_values_buf);
    alloc_register(&p->alloc, &s.grad_combined);
    alloc_register(&p->alloc, &s.grad_state);
    alloc_register(&p->alloc, &s.grad_input);
    alloc_register(&p->alloc, &p->grad_out);
    alloc_create(&p->alloc);

    s.combined_ptr = combined_t.data;
    s.state_ptr    = state_t.data;
    s.input_ptr    = input_t.data;

    int N_combined = B * T * 3 * H;
    int N_state = B * H;
    int N_out = B * T * H;
    float* buf = (float*)malloc(N_combined * sizeof(float));
    for (int i = 0; i < N_combined; ++i) buf[i] = rand1() * 5.0f;
    float_to_device(s.combined_ptr, buf, N_combined);
    for (int i = 0; i < N_state; ++i) buf[i] = fabsf(rand1()) + 0.1f;
    float_to_device(s.state_ptr, buf, N_state);
    for (int i = 0; i < N_out; ++i) buf[i] = rand1();
    float_to_device(s.input_ptr, buf, N_out);
    float_to_device(p->grad_out.data, buf, N_out);
    for (int i = 0; i < N_state; ++i) buf[i] = rand1();
    free(buf);
    return p;
}

void run_fusedscan_fwd(FusedScanProfile* p) {
    mingru_scan_forward<<<grid_size(p->B * p->H), BLOCK_SIZE>>>(p->scan);
}

void run_fusedscan_bwd(FusedScanProfile* p) {
    // fbr: Training truncates the recurrent state boundary, so the production
    // scan backward accepts only the timestep output gradient.
    mingru_scan_backward<<<grid_size(p->B * p->H), BLOCK_SIZE>>>(
        p->scan, p->grad_out.data);
}

void profile_fusedscan(int B, int T, int H) {
    printf("fused_scan (N=%d, %dx%dx%d)\n", B*T*H, B, T, H);
    auto* p = create_fusedscan(B, T, H);
    float fwd = profile_kernel((kernel_fn)run_fusedscan_fwd, p);
    print_timing("forward", fwd, B*T);
    float bwd = profile_kernel((kernel_fn)run_fusedscan_bwd, p);
    print_timing("backward", bwd, B*T);
    printf("\n");
    alloc_free(&p->alloc);
    free(p);
}

struct PPOProfile {
    PPOKernelArgs ka;
    PPOGraphArgs ga;
    FloatTensor loss, losses_acc, ppo_partials;
    FloatTensor grad_logits_t, grad_values_t, adv_mean_t, adv_var_t;
    PrecisionTensor logits_t, actions_t, old_logprobs_t, advantages_t, prio_t, values_t, returns_t;
    PrecisionTensor ratio_t, newvalue_t;
    IntTensor act_sizes_t;
    Allocator alloc;
    int N, T, A, ppo_grid;
};

PPOProfile* create_ppoloss(int N, int T, int A) {
    auto* p = (PPOProfile*)calloc(1, sizeof(PPOProfile));
    p->N = N; p->T = T; p->A = A;

    int NT = N * T;
    int fused_cols = A + 1;
    int ppo_grid = (NT + PPO_THREADS - 1) / PPO_THREADS;
    p->ppo_grid = ppo_grid;

    p->logits_t       = {.shape = {N, T, fused_cols}};
    p->actions_t      = {.shape = {NT}};
    p->old_logprobs_t = {.shape = {NT}};
    p->advantages_t   = {.shape = {NT}};
    p->prio_t         = {.shape = {N}};
    p->values_t       = {.shape = {NT}};
    p->returns_t      = {.shape = {NT}};
    p->ratio_t        = {.shape = {NT}};
    p->newvalue_t     = {.shape = {NT}};
    p->grad_logits_t  = {.shape = {N, T, A}};
    p->grad_values_t  = {.shape = {NT}};
    p->adv_mean_t     = {.shape = {1}};
    p->adv_var_t      = {.shape = {1}};
    p->loss           = {.shape = {1}};
    p->losses_acc     = {.shape = {LOSS_N + 1}};
    p->ppo_partials   = {.shape = {ppo_grid, LOSS_N + 1}};
    p->act_sizes_t    = {.shape = {1}};

    p->alloc = {};
    alloc_register(&p->alloc, &p->logits_t);
    alloc_register(&p->alloc, &p->actions_t);
    alloc_register(&p->alloc, &p->old_logprobs_t);
    alloc_register(&p->alloc, &p->advantages_t);
    alloc_register(&p->alloc, &p->prio_t);
    alloc_register(&p->alloc, &p->values_t);
    alloc_register(&p->alloc, &p->returns_t);
    alloc_register(&p->alloc, &p->ratio_t);
    alloc_register(&p->alloc, &p->newvalue_t);
    alloc_register(&p->alloc, &p->grad_logits_t);
    alloc_register(&p->alloc, &p->grad_values_t);
    alloc_register(&p->alloc, &p->adv_mean_t);
    alloc_register(&p->alloc, &p->adv_var_t);
    alloc_register(&p->alloc, &p->loss);
    alloc_register(&p->alloc, &p->losses_acc);
    alloc_register(&p->alloc, &p->ppo_partials);
    alloc_register(&p->alloc, &p->act_sizes_t);
    alloc_create(&p->alloc);

    cudaMemcpy(p->act_sizes_t.data, &A, sizeof(int), cudaMemcpyHostToDevice);

    // Fill with random data
    float* buf = (float*)malloc(NT * fused_cols * sizeof(float));

    // Advantages (precision_t) + compute mean/var
    float adv_sum = 0, adv_sq = 0;
    for (int i = 0; i < NT; ++i) {
        float a = rand1();
        buf[i] = a;
        adv_sum += a;
        adv_sq += a * a;
    }
    float adv_mean = adv_sum / NT;
    float adv_var = adv_sq / NT - adv_mean * adv_mean;
    float_to_device(p->advantages_t.data, buf, NT);
    cudaMemcpy(p->adv_mean_t.data, &adv_mean, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(p->adv_var_t.data, &adv_var, sizeof(float), cudaMemcpyHostToDevice);

    // Fill logits (fused: A logit cols + 1 value col per row)
    for (int i = 0; i < NT * fused_cols; ++i) buf[i] = rand1() * 2.0f;
    float_to_device(p->logits_t.data, buf, NT * fused_cols);
    // actions
    for (int i = 0; i < NT; ++i) buf[i] = (float)(rand() % A);
    float_to_device(p->actions_t.data, buf, NT);
    // old_logprobs
    for (int i = 0; i < NT; ++i) buf[i] = rand1() * 2.0f;
    float_to_device(p->old_logprobs_t.data, buf, NT);
    // values + returns
    for (int i = 0; i < NT; ++i) buf[i] = rand1();
    float_to_device(p->values_t.data, buf, NT);
    for (int i = 0; i < NT; ++i) buf[i] = rand1();
    float_to_device(p->returns_t.data, buf, NT);
    // prio
    for (int i = 0; i < N; ++i) buf[i] = (float)rand() / RAND_MAX;
    float_to_device(p->prio_t.data, buf, N);
    free(buf);

    // Wire up kernel args
    p->ka = {
        .grad_logits = p->grad_logits_t.data,
        .grad_logstd = nullptr,
        .grad_values_pred = p->grad_values_t.data,
        .logits = p->logits_t.data,
        .logstd = nullptr,
        .values_pred = p->logits_t.data + A,  // value is last col in fused layout
        .adv_mean = p->adv_mean_t.data,
        .adv_var = p->adv_var_t.data,
        .act_sizes = p->act_sizes_t.data,
        .num_atns = 1,
        .clip_coef = 0.1f, .vf_clip_coef = 0.1f, .vf_coef = 0.5f, .ent_coef = 0.01f,
        .T_seq = T, .A_total = A, .N = N,
        .logits_stride_n = T * fused_cols, .logits_stride_t = fused_cols, .logits_stride_a = 1,
        .values_stride_n = T * fused_cols, .values_stride_t = fused_cols,
        .is_continuous = false,
    };
    p->ga = {
        .out_ratio = p->ratio_t.data,
        .out_newvalue = p->newvalue_t.data,
        .actions = p->actions_t.data,
        .old_logprobs = p->old_logprobs_t.data,
        .advantages = p->advantages_t.data,
        .prio = p->prio_t.data,
        .values = p->values_t.data,
        .returns = p->returns_t.data,
    };

    return p;
}

void run_ppoloss(PPOProfile* p) {
    cudaMemset(p->loss.data, 0, sizeof(float));
    ppo_loss_compute<<<p->ppo_grid, PPO_THREADS>>>(
        p->ppo_partials.data, p->ka, p->ga);
    ppo_loss_reduce<<<1, LOSS_N + 1>>>(
        p->loss.data, p->losses_acc.data, p->ppo_partials.data, p->ppo_grid);
}

void profile_ppoloss(int N, int T, int A) {
    int NT = N * T;
    printf("ppo_loss_fwd_bwd (NT=%d, %dx%d, A=%d)\n", NT, N, T, A);
    auto* p = create_ppoloss(N, T, A);
    float ms = profile_kernel((kernel_fn)run_ppoloss, p);
    print_timing("fwd+bwd", ms, NT);
    printf("\n");
    alloc_free(&p->alloc);
    free(p);
}

struct SampleLogitsProfile {
    PrecisionTensor dec_out, logstd;
    IntTensor act_sizes;
    PrecisionTensor actions_t, logprobs_t, value_out_t;
    curandStatePhilox4_32_10_t* rng_states;
    Allocator alloc;
    int B, A;
};

SampleLogitsProfile* create_samplelogits(int B, int A) {
    auto* p = (SampleLogitsProfile*)calloc(1, sizeof(SampleLogitsProfile));
    p->B = B; p->A = A;

    int fused_cols = A + 1;
    p->dec_out     = {.shape = {B, fused_cols}};
    p->logstd      = {.shape = {0}};  // empty for discrete
    p->act_sizes   = {.shape = {1}};
    p->actions_t   = {.shape = {B}};
    p->logprobs_t  = {.shape = {B}};
    p->value_out_t = {.shape = {B}};

    p->alloc = {};
    alloc_register(&p->alloc, &p->dec_out);
    alloc_register(&p->alloc, &p->act_sizes);
    alloc_register(&p->alloc, &p->actions_t);
    alloc_register(&p->alloc, &p->logprobs_t);
    alloc_register(&p->alloc, &p->value_out_t);
    alloc_create(&p->alloc);

    cudaMemcpy(p->act_sizes.data, &A, sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&p->rng_states, B * sizeof(curandStatePhilox4_32_10_t));
    rng_init<<<grid_size(B), BLOCK_SIZE>>>(p->rng_states, 42, B);
    cudaDeviceSynchronize();

    float* buf = (float*)malloc(B * fused_cols * sizeof(float));
    for (int i = 0; i < B * fused_cols; ++i) buf[i] = rand1() * 5.0f;
    float_to_device(p->dec_out.data, buf, B * fused_cols);
    free(buf);
    return p;
}

void run_samplelogits(SampleLogitsProfile* p) {
    sample_logits<<<grid_size(p->B), BLOCK_SIZE>>>(
        p->dec_out, p->logstd, p->act_sizes,
        p->actions_t.data, p->logprobs_t.data, p->value_out_t.data,
        p->rng_states, nullptr, 0);
}

void profile_samplelogits(int B, int A) {
    printf("sample_logits (B=%d, A=%d)\n", B, A);
    auto* p = create_samplelogits(B, A);
    float ms = profile_kernel((kernel_fn)run_samplelogits, p);
    print_timing("forward", ms, B);
    printf("\n");
    cudaFree(p->rng_states);
    alloc_free(&p->alloc);
    free(p);
}

struct Im2ColProfile {
    PrecisionTensor input, col, grad_input;
    Allocator alloc;
    int B, IC, IH, IW, K, S, OH, OW;
};

Im2ColProfile* create_im2col(int B, int IC, int IH, int IW, int K, int S, int OH, int OW) {
    auto* p = (Im2ColProfile*)calloc(1, sizeof(Im2ColProfile));
    p->B = B; p->IC = IC; p->IH = IH; p->IW = IW;
    p->K = K; p->S = S; p->OH = OH; p->OW = OW;
    int in_size  = B * IC * IH * IW;
    int col_size = B * OH * OW * IC * K * K;
    p->input      = {.shape = {in_size}};
    p->col        = {.shape = {col_size}};
    p->grad_input = {.shape = {in_size}};
    p->alloc = {};
    alloc_register(&p->alloc, &p->input);
    alloc_register(&p->alloc, &p->col);
    alloc_register(&p->alloc, &p->grad_input);
    alloc_create(&p->alloc);
    float* buf = (float*)malloc(std::max(in_size, col_size) * sizeof(float));
    for (int i = 0; i < in_size; ++i) buf[i] = rand1();
    float_to_device(p->input.data, buf, in_size);
    for (int i = 0; i < col_size; ++i) buf[i] = rand1();
    float_to_device(p->col.data, buf, col_size);
    free(buf);
    return p;
}

void run_im2col(Im2ColProfile* p) {
    int total = p->B * p->OH * p->OW * p->IC * p->K * p->K;
    im2col_kernel<<<grid_size(total), BLOCK_SIZE>>>(
        p->input.data, p->col.data,
        p->B, p->IC, p->IH, p->IW, p->K, p->S, p->OH, p->OW);
}

void run_col2im(Im2ColProfile* p) {
    int total = p->B * p->IC * p->IH * p->IW;
    col2im_kernel<<<grid_size(total), BLOCK_SIZE>>>(
        p->col.data, p->grad_input.data,
        p->B, p->IC, p->IH, p->IW, p->K, p->S, p->OH, p->OW);
}

void profile_im2col(int B, int IC, int IH, int IW, int K, int S, int OH, int OW) {
    int total = B * OH * OW * IC * K * K;
    printf("im2col/col2im (B=%d, IC=%d, %dx%d, K=%d, S=%d -> %dx%d)\n",
           B, IC, IH, IW, K, S, OH, OW);
    auto* p = create_im2col(B, IC, IH, IW, K, S, OH, OW);
    float fwd = profile_kernel((kernel_fn)run_im2col, p);
    print_timing("im2col", fwd, total);
    float bwd = profile_kernel((kernel_fn)run_col2im, p);
    print_timing("col2im", bwd, total);
    printf("\n");
    alloc_free(&p->alloc);
    free(p);
}

__global__ void me_relu_kernel(
        precision_t* __restrict__ data,
        int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;

    float val = to_float(data[idx]);
    data[idx] = from_float(fmaxf(0.0f, val));
}

__global__ void me_point_max_kernel(
        precision_t* __restrict__ output,
        int* __restrict__ argmax,
        const precision_t* __restrict__ point_logits,
        int B, int hidden) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * hidden;
    if (idx >= total) return;

    int h = idx % hidden;
    int b = idx / hidden;
    float max_val = -FLT_MAX;
    int best_point = 0;
    for (int point_idx = 0; point_idx < ME_NUM_POINTS; ++point_idx) {
        float val = to_float(point_logits[((int64_t)b * ME_NUM_POINTS + point_idx) * hidden + h]);
        if (val > max_val) {
            max_val = val;
            best_point = point_idx;
        }
    }

    output[idx] = from_float(max_val);
    if (argmax) argmax[idx] = best_point;
}

static constexpr int ME_PROFILE_WGRAD_CHUNKS = 256;

__device__ __forceinline__ float profile_me_obs_value(
        const precision_t* __restrict__ obs, int obs_size, int b, int point_idx, int d) {
    int obs_idx = (d < ME_SELF_DIM)
        ? d
        : ME_SELF_DIM + point_idx * ME_POINT_DIM + (d - ME_SELF_DIM);
    return to_float(obs[(int64_t)b * obs_size + obs_idx]);
}

__device__ __forceinline__ float profile_me_block_reduce_sum(float sum) {
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float sdata[32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) sdata[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = (lane < (blockDim.x + 31) / 32) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    return sum;
}

__global__ void me_projection_relu_kernel(
        precision_t* __restrict__ entity_hidden,
        const precision_t* __restrict__ obs,
        const precision_t* __restrict__ weight,
        int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * ME_NUM_POINTS * ME_ENTITY_HIDDEN;
    if (idx >= total) return;

    int h = idx % ME_ENTITY_HIDDEN;
    int point_idx = (idx / ME_ENTITY_HIDDEN) % ME_NUM_POINTS;
    int b = idx / (ME_NUM_POINTS * ME_ENTITY_HIDDEN);
    const precision_t* row = weight + h * ME_ENTITY_IN;

    float sum = 0.0f;
    for (int d = 0; d < ME_ENTITY_IN; ++d) {
        sum += to_float(row[d]) * profile_me_obs_value(obs, obs_size, b, point_idx, d);
    }
    entity_hidden[idx] = from_float(fmaxf(0.0f, sum));
}

__global__ void me_grad_entity_scan_kernel(
        precision_t* __restrict__ grad_entity,
        const precision_t* __restrict__ grad_out,
        const precision_t* __restrict__ output_w,
        const precision_t* __restrict__ entity_hidden,
        const int* __restrict__ argmax,
        int B, int hidden) {
    int b = blockIdx.x;
    if (b >= B) return;

    __shared__ int arg_s[BLOCK_SIZE];
    __shared__ float grad_s[BLOCK_SIZE];

    int out_idx = threadIdx.x;
    int point_idx = out_idx / ME_ENTITY_HIDDEN;
    int k = out_idx - point_idx * ME_ENTITY_HIDDEN;
    float acc = 0.0f;

    for (int base = 0; base < hidden; base += blockDim.x) {
        int h = base + threadIdx.x;
        if (h < hidden) {
            arg_s[threadIdx.x] = argmax[(int64_t)b * hidden + h];
            grad_s[threadIdx.x] = to_float(grad_out[(int64_t)b * hidden + h]);
        }
        __syncthreads();

        int tile = hidden - base;
        if (tile > blockDim.x) tile = blockDim.x;
        if (out_idx < ME_NUM_POINTS * ME_ENTITY_HIDDEN) {
            for (int j = 0; j < tile; ++j) {
                if (arg_s[j] == point_idx) {
                    acc += grad_s[j] * to_float(output_w[(int64_t)(base + j) * ME_ENTITY_HIDDEN + k]);
                }
            }
        }
        __syncthreads();
    }

    if (out_idx < ME_NUM_POINTS * ME_ENTITY_HIDDEN) {
        int64_t offset = ((int64_t)b * ME_NUM_POINTS + point_idx) * ME_ENTITY_HIDDEN + k;
        float g = to_float(entity_hidden[offset]) > 0.0f ? acc : 0.0f;
        grad_entity[offset] = from_float(g);
    }
}

__global__ void me_input_wgrad_partial_kernel(
        float* __restrict__ partials,
        const precision_t* __restrict__ grad_entity,
        const precision_t* __restrict__ point_input,
        int B) {
    int h = blockIdx.x;
    int chunk = blockIdx.y;
    if (h >= ME_ENTITY_HIDDEN || chunk >= ME_PROFILE_WGRAD_CHUNKS) return;

    float sum[ME_ENTITY_IN];
#pragma unroll
    for (int d = 0; d < ME_ENTITY_IN; ++d) sum[d] = 0.0f;

    int rows = B * ME_NUM_POINTS;
    int chunk_size = (rows + ME_PROFILE_WGRAD_CHUNKS - 1) / ME_PROFILE_WGRAD_CHUNKS;
    int start = chunk * chunk_size;
    int end = start + chunk_size;
    if (end > rows) end = rows;

    for (int row = start + threadIdx.x; row < end; row += blockDim.x) {
        float g = to_float(grad_entity[(int64_t)row * ME_ENTITY_HIDDEN + h]);
        const precision_t* point = point_input + (int64_t)row * ME_ENTITY_IN;
#pragma unroll
        for (int d = 0; d < ME_ENTITY_IN; ++d) {
            sum[d] += g * to_float(point[d]);
        }
    }

    __shared__ float warp_sums[ME_ENTITY_IN * 32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;
#pragma unroll
    for (int d = 0; d < ME_ENTITY_IN; ++d) {
        float s = sum[d];
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_down_sync(0xffffffff, s, offset);
        if (lane == 0) warp_sums[d * 32 + warp] = s;
    }
    __syncthreads();

    if (warp == 0) {
#pragma unroll
        for (int d = 0; d < ME_ENTITY_IN; ++d) {
            float s = lane < num_warps ? warp_sums[d * 32 + lane] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1)
                s += __shfl_down_sync(0xffffffff, s, offset);
            if (lane == 0) partials[((h * ME_PROFILE_WGRAD_CHUNKS) + chunk) * ME_ENTITY_IN + d] = s;
        }
    }
}

__global__ void me_input_wgrad_reduce_kernel(
        precision_t* __restrict__ wgrad,
        const float* __restrict__ partials) {
    int h = blockIdx.x;
    int d = blockIdx.y;
    if (h >= ME_ENTITY_HIDDEN || d >= ME_ENTITY_IN) return;

    float sum = 0.0f;
    for (int chunk = threadIdx.x; chunk < ME_PROFILE_WGRAD_CHUNKS; chunk += blockDim.x) {
        sum += partials[((h * ME_PROFILE_WGRAD_CHUNKS) + chunk) * ME_ENTITY_IN + d];
    }
    sum = profile_me_block_reduce_sum(sum);
    if (threadIdx.x == 0) wgrad[h * ME_ENTITY_IN + d] = from_float(sum);
}

struct EncoderWork {
    double flops;
    double bytes;
};

inline int ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

EncoderWork work_flat_encoder(int B, int H) {
    double p = sizeof(precision_t);
    double flops = 2.0 * B * ME_OBS_SIZE * H;
    double bytes = p * (B * ME_OBS_SIZE + H * ME_OBS_SIZE + B * H);
    return {flops, bytes};
}

EncoderWork work_flat_encoder_bwd(int B, int H) {
    double p = sizeof(precision_t);
    double flops = 2.0 * B * H * ME_OBS_SIZE;
    double bytes = p * ((double)B * H + (double)B * ME_OBS_SIZE + (double)H * ME_OBS_SIZE);
    return {flops, bytes};
}

EncoderWork work_custom_projection(int B) {
    double p = sizeof(precision_t);
    double outputs = (double)B * ME_NUM_POINTS * ME_ENTITY_HIDDEN;
    double flops = outputs * (2.0 * ME_ENTITY_IN + 1.0);
    double bytes = p * outputs * (2.0 * ME_ENTITY_IN + 1.0);
    return {flops, bytes};
}

EncoderWork work_custom_linear_max(int B, int H, bool write_argmax) {
    double p = sizeof(precision_t);
    double hidden_tiles = ceil_div_int(H, ME_HIDDEN_TILE);
    double batch_tiles = ceil_div_int(B, ME_BATCH_TILE);
    double flops = 2.0 * B * H * ME_NUM_POINTS * ME_ENTITY_HIDDEN
        + (double)B * H * (ME_NUM_POINTS - 1)
        + hidden_tiles * B * ME_NUM_POINTS * ME_ENTITY_HIDDEN;
    double point_reads = hidden_tiles * B * ME_NUM_POINTS * ME_ENTITY_HIDDEN;
    double weight_reads = batch_tiles * H * ME_ENTITY_HIDDEN;
    double output_writes = (double)B * H;
    double bytes = p * (point_reads + weight_reads + output_writes);
    if (write_argmax) bytes += (double)B * H * sizeof(int);
    return {flops, bytes};
}

EncoderWork work_materialize_points(int B) {
    double p = sizeof(precision_t);
    double elems = (double)B * ME_NUM_POINTS * ME_ENTITY_IN;
    return {0.0, 2.0 * p * elems};
}

EncoderWork work_cublas_projection(int B) {
    double p = sizeof(precision_t);
    double rows = (double)B * ME_NUM_POINTS;
    double gemm_flops = 2.0 * rows * ME_ENTITY_IN * ME_ENTITY_HIDDEN;
    double bias_relu_flops = rows * ME_ENTITY_HIDDEN;
    double gemm_bytes = p * (rows * ME_ENTITY_IN
        + ME_ENTITY_HIDDEN * ME_ENTITY_IN
        + rows * ME_ENTITY_HIDDEN);
    double relu_bytes = p * rows * ME_ENTITY_HIDDEN * 2.0;
    return {gemm_flops + bias_relu_flops, gemm_bytes + relu_bytes};
}

EncoderWork work_cublas_projection_preact(int B) {
    double p = sizeof(precision_t);
    double rows = (double)B * ME_NUM_POINTS;
    double gemm_flops = 2.0 * rows * ME_ENTITY_IN * ME_ENTITY_HIDDEN;
    double gemm_bytes = p * (rows * ME_ENTITY_IN
        + ME_ENTITY_HIDDEN * ME_ENTITY_IN
        + rows * ME_ENTITY_HIDDEN);
    return {gemm_flops, gemm_bytes};
}

EncoderWork work_cublas_output_gemm(int B, int H) {
    double p = sizeof(precision_t);
    double rows = (double)B * ME_NUM_POINTS;
    double flops = 2.0 * rows * ME_ENTITY_HIDDEN * H;
    double bytes = p * (rows * ME_ENTITY_HIDDEN
        + H * ME_ENTITY_HIDDEN
        + rows * H);
    return {flops, bytes};
}

EncoderWork work_cublas_point_max(int B, int H, bool write_argmax) {
    double p = sizeof(precision_t);
    double flops = (double)B * H * (ME_NUM_POINTS - 1);
    double bytes = p * ((double)B * ME_NUM_POINTS * H + (double)B * H);
    if (write_argmax) bytes += (double)B * H * sizeof(int);
    return {flops, bytes};
}

EncoderWork work_custom_output_wgrad(int B, int H) {
    double p = sizeof(precision_t);
    double reductions = (double)B * H * ME_ENTITY_HIDDEN;
    double flops = 2.0 * reductions;
    double bytes = (double)B * H * (sizeof(int) + p)
        + reductions * p
        + p * H * ME_ENTITY_HIDDEN;
    return {flops, bytes};
}

EncoderWork work_custom_grad_entity(int B, int H) {
    double p = sizeof(precision_t);
    double probes = (double)B * ME_NUM_POINTS * ME_ENTITY_HIDDEN * H;
    double contribs = (double)B * H * ME_ENTITY_HIDDEN;
    double relu = (double)B * ME_NUM_POINTS * ME_ENTITY_HIDDEN;
    double flops = probes + 2.0 * contribs + relu;
    double bytes = (double)B * H * (sizeof(int) + p)
        + contribs * p
        + relu * 2.0 * p;
    return {flops, bytes};
}

EncoderWork work_atomic_grad_entity(int B, int H) {
    double p = sizeof(precision_t);
    double contribs = (double)B * H * ME_ENTITY_HIDDEN;
    double relu = (double)B * ME_NUM_POINTS * ME_ENTITY_HIDDEN;
    double flops = 2.0 * contribs + relu;
    double bytes = (double)B * H * (sizeof(int) + p)
        + contribs * p
        + relu * 2.0 * p;
    return {flops, bytes};
}

EncoderWork work_custom_input_wgrad(int B) {
    double p = sizeof(precision_t);
    double reductions = (double)B * ME_NUM_POINTS * ME_ENTITY_HIDDEN * ME_ENTITY_IN;
    double flops = 2.0 * reductions;
    double partials = (double)ME_ENTITY_HIDDEN * ME_PROFILE_WGRAD_CHUNKS * ME_ENTITY_IN;
    double bytes = (double)B * ME_NUM_POINTS * ME_ENTITY_HIDDEN * p
        + reductions * p
        + partials * sizeof(float) * 2.0
        + p * ME_ENTITY_HIDDEN * ME_ENTITY_IN;
    return {flops, bytes};
}

EncoderWork work_puf_input_wgrad(int B) {
    double p = sizeof(precision_t);
    double rows = (double)B * ME_NUM_POINTS;
    double flops = 2.0 * rows * ME_ENTITY_HIDDEN * ME_ENTITY_IN;
    double bytes = p * (rows * ME_ENTITY_HIDDEN
        + rows * ME_ENTITY_IN
        + ME_ENTITY_HIDDEN * ME_ENTITY_IN);
    return {flops, bytes};
}

inline EncoderWork add_work(EncoderWork a, EncoderWork b) {
    return {a.flops + b.flops, a.bytes + b.bytes};
}

struct MinimalEncoderProfile {
    PrecisionTensor obs;
    PrecisionTensor flat_w, flat_out, flat_wgrad, grad_out;
    MinimalEntityEncoderWeights me_w;
    MinimalEntityEncoderActivations me_a;
    FloatTensor input_wgrad_partials;
    PrecisionTensor point_input, cublas_entity_hidden, point_logits, cublas_out;
    Allocator alloc;
    int B, H;
};

MinimalEncoderProfile* create_minimalenc(int B, int H) {
    auto* p = (MinimalEncoderProfile*)calloc(1, sizeof(MinimalEncoderProfile));
    p->B = B; p->H = H;

    p->obs = {.shape = {B, ME_OBS_SIZE}};
    p->flat_w = {.shape = {H, ME_OBS_SIZE}};
    p->flat_out = {.shape = {B, H}};
    p->flat_wgrad = {.shape = {H, ME_OBS_SIZE}};
    p->grad_out = {.shape = {B, H}};

    p->me_w.input_w = {.shape = {ME_ENTITY_HIDDEN, ME_ENTITY_IN}};
    p->me_w.output_w = {.shape = {H, ME_ENTITY_HIDDEN}};
    p->me_w.obs_size = ME_OBS_SIZE;
    p->me_w.hidden = H;

    p->me_a.point_input = {.shape = {B, ME_NUM_POINTS, ME_ENTITY_IN}};
    p->me_a.entity_hidden = {.shape = {B, ME_NUM_POINTS, ME_ENTITY_HIDDEN}};
    p->me_a.out = {.shape = {B, H}};
    p->me_a.argmax = {.shape = {B, H}};
    p->me_a.grad_entity = {.shape = {B, ME_NUM_POINTS, ME_ENTITY_HIDDEN}};
    p->me_a.input_wgrad = {.shape = {ME_ENTITY_HIDDEN, ME_ENTITY_IN}};
    p->me_a.output_wgrad = {.shape = {H, ME_ENTITY_HIDDEN}};
    p->input_wgrad_partials = {.shape = {ME_ENTITY_HIDDEN, ME_PROFILE_WGRAD_CHUNKS, ME_ENTITY_IN}};

    p->point_input = {.shape = {B * ME_NUM_POINTS, ME_ENTITY_IN}};
    p->cublas_entity_hidden = {.shape = {B * ME_NUM_POINTS, ME_ENTITY_HIDDEN}};
    p->point_logits = {.shape = {B * ME_NUM_POINTS, H}};
    p->cublas_out = {.shape = {B, H}};

    p->alloc = {};
    alloc_register(&p->alloc, &p->obs);
    alloc_register(&p->alloc, &p->flat_w);
    alloc_register(&p->alloc, &p->flat_out);
    alloc_register(&p->alloc, &p->flat_wgrad);
    alloc_register(&p->alloc, &p->grad_out);
    alloc_register(&p->alloc, &p->me_w.input_w);
    alloc_register(&p->alloc, &p->me_w.output_w);
    alloc_register(&p->alloc, &p->me_a.point_input);
    alloc_register(&p->alloc, &p->me_a.entity_hidden);
    alloc_register(&p->alloc, &p->me_a.out);
    alloc_register(&p->alloc, &p->me_a.argmax);
    alloc_register(&p->alloc, &p->me_a.grad_entity);
    alloc_register(&p->alloc, &p->me_a.input_wgrad);
    alloc_register(&p->alloc, &p->me_a.output_wgrad);
    alloc_register(&p->alloc, &p->input_wgrad_partials);
    alloc_register(&p->alloc, &p->point_input);
    alloc_register(&p->alloc, &p->cublas_entity_hidden);
    alloc_register(&p->alloc, &p->point_logits);
    alloc_register(&p->alloc, &p->cublas_out);
    alloc_create(&p->alloc);

    int64_t max_count = std::max<int64_t>({
        numel(p->obs.shape),
        numel(p->flat_w.shape),
        numel(p->grad_out.shape),
        numel(p->me_w.input_w.shape),
        numel(p->me_w.output_w.shape),
    });
    float* buf = (float*)malloc(max_count * sizeof(float));

    for (int64_t i = 0; i < numel(p->obs.shape); ++i) buf[i] = rand1();
    float_to_device(p->obs.data, buf, numel(p->obs.shape));
    for (int64_t i = 0; i < numel(p->flat_w.shape); ++i) buf[i] = rand1() * 0.1f;
    float_to_device(p->flat_w.data, buf, numel(p->flat_w.shape));
    for (int64_t i = 0; i < numel(p->grad_out.shape); ++i) buf[i] = rand1() * 0.1f;
    float_to_device(p->grad_out.data, buf, numel(p->grad_out.shape));
    for (int64_t i = 0; i < numel(p->me_w.input_w.shape); ++i) buf[i] = rand1() * 0.1f;
    float_to_device(p->me_w.input_w.data, buf, numel(p->me_w.input_w.shape));
    for (int64_t i = 0; i < numel(p->me_w.output_w.shape); ++i) buf[i] = rand1() * 0.1f;
    float_to_device(p->me_w.output_w.data, buf, numel(p->me_w.output_w.shape));
    free(buf);

    return p;
}

void run_flat_encoder(MinimalEncoderProfile* p) {
    cublasGemmExDense(CUBLAS_OP_N, CUBLAS_OP_T,
        p->B, p->H, ME_OBS_SIZE,
        p->obs.data, p->flat_w.data, p->flat_out.data, 0);
}

void run_flat_encoder_bwd(MinimalEncoderProfile* p) {
    puf_mm_tn(&p->grad_out, &p->obs, &p->flat_wgrad, 0);
}

void run_flat_encoder_fwd_bwd(MinimalEncoderProfile* p) {
    run_flat_encoder(p);
    run_flat_encoder_bwd(p);
}

void run_me_projection_kernel(MinimalEncoderProfile* p) {
    me_projection_relu_kernel<<<grid_size(p->B * ME_NUM_POINTS * ME_ENTITY_HIDDEN), BLOCK_SIZE>>>(
        p->me_a.entity_hidden.data, p->obs.data, p->me_w.input_w.data,
        p->B, ME_OBS_SIZE);
}

void run_me_projection(MinimalEncoderProfile* p) {
    me_materialize_points_kernel<<<grid_size(p->B * ME_NUM_POINTS * ME_ENTITY_IN), BLOCK_SIZE>>>(
        p->me_a.point_input.data, p->obs.data, p->B, ME_OBS_SIZE);
    PrecisionTensor point_input = {
        .data = p->me_a.point_input.data,
        .shape = {p->B, ME_NUM_POINTS, ME_ENTITY_IN},
    };
    PrecisionTensor entity_hidden = {
        .data = p->me_a.entity_hidden.data,
        .shape = {p->B, ME_NUM_POINTS, ME_ENTITY_HIDDEN},
    };
    puf_mm(&point_input, &p->me_w.input_w, &entity_hidden, 0);
}

void run_me_linearmax(MinimalEncoderProfile* p) {
    dim3 block(ME_HIDDEN_TILE, ME_BATCH_TILE);
    dim3 grid(ceil_div_int(p->B, ME_BATCH_TILE), ceil_div_int(p->H, ME_HIDDEN_TILE));
    size_t shared_bytes = (
        (size_t)ME_BATCH_TILE * ME_NUM_POINTS * ME_ENTITY_HIDDEN +
        (size_t)ME_HIDDEN_TILE * ME_ENTITY_HIDDEN) * sizeof(precision_t);
    me_linear_max_kernel<<<grid, block, shared_bytes>>>(
        p->me_a.out.data, p->me_a.argmax.data, p->me_a.entity_hidden.data,
        p->me_w.output_w.data, p->B, p->H);
}

void run_me_full(MinimalEncoderProfile* p) {
    run_me_projection(p);
    run_me_linearmax(p);
}

void run_me_projection_kernel_full(MinimalEncoderProfile* p) {
    run_me_projection_kernel(p);
    run_me_linearmax(p);
}

void run_me_output_wgrad(MinimalEncoderProfile* p) {
    me_output_wgrad_kernel<<<p->H, 256>>>(
        p->me_a.output_wgrad.data, p->grad_out.data,
        p->me_a.entity_hidden.data, p->me_a.argmax.data, p->B, p->H);
}

void run_me_grad_entity(MinimalEncoderProfile* p) {
    me_grad_entity_kernel<<<p->B, BLOCK_SIZE>>>(
        p->me_a.grad_entity.data, p->grad_out.data, p->me_w.output_w.data,
        p->me_a.entity_hidden.data, p->me_a.argmax.data, p->B, p->H);
}

void run_me_grad_entity_scan(MinimalEncoderProfile* p) {
    me_grad_entity_scan_kernel<<<p->B, BLOCK_SIZE>>>(
        p->me_a.grad_entity.data, p->grad_out.data, p->me_w.output_w.data,
        p->me_a.entity_hidden.data, p->me_a.argmax.data, p->B, p->H);
}

void run_me_input_wgrad_kernel(MinimalEncoderProfile* p) {
    dim3 input_wpartial_grid(ME_ENTITY_HIDDEN, ME_PROFILE_WGRAD_CHUNKS);
    me_input_wgrad_partial_kernel<<<input_wpartial_grid, 256>>>(
        p->input_wgrad_partials.data, p->me_a.grad_entity.data,
        p->me_a.point_input.data, p->B);
    dim3 input_wreduce_grid(ME_ENTITY_HIDDEN, ME_ENTITY_IN);
    me_input_wgrad_reduce_kernel<<<input_wreduce_grid, 256>>>(
        p->me_a.input_wgrad.data, p->input_wgrad_partials.data);
}

void run_me_input_wgrad(MinimalEncoderProfile* p) {
    puf_mm_tn(&p->me_a.grad_entity, &p->me_a.point_input, &p->me_a.input_wgrad, 0);
}

void run_me_backward_full(MinimalEncoderProfile* p) {
    run_me_output_wgrad(p);
    run_me_grad_entity(p);
    run_me_input_wgrad(p);
}

void run_me_fwd_bwd_full(MinimalEncoderProfile* p) {
    run_me_full(p);
    run_me_backward_full(p);
}

void run_materialize_points(MinimalEncoderProfile* p) {
    me_materialize_points_kernel<<<grid_size(p->B * ME_NUM_POINTS * ME_ENTITY_IN), BLOCK_SIZE>>>(
        p->point_input.data, p->obs.data, p->B, ME_OBS_SIZE);
}

void run_cublas_projection(MinimalEncoderProfile* p) {
    cublasGemmExDense(CUBLAS_OP_N, CUBLAS_OP_T,
        p->B * ME_NUM_POINTS, ME_ENTITY_HIDDEN, ME_ENTITY_IN,
        p->point_input.data, p->me_w.input_w.data,
        p->cublas_entity_hidden.data, 0);
    me_relu_kernel<<<grid_size(p->B * ME_NUM_POINTS * ME_ENTITY_HIDDEN), BLOCK_SIZE>>>(
        p->cublas_entity_hidden.data, p->B * ME_NUM_POINTS, ME_ENTITY_HIDDEN);
}

void run_cublas_output_gemm(MinimalEncoderProfile* p) {
    cublasGemmExDense(CUBLAS_OP_N, CUBLAS_OP_T,
        p->B * ME_NUM_POINTS, p->H, ME_ENTITY_HIDDEN,
        p->cublas_entity_hidden.data, p->me_w.output_w.data,
        p->point_logits.data, 0);
}

void run_cublas_point_max(MinimalEncoderProfile* p) {
    me_point_max_kernel<<<grid_size(p->B * p->H), BLOCK_SIZE>>>(
        p->cublas_out.data, p->me_a.argmax.data, p->point_logits.data,
        p->B, p->H);
}

void run_cublas_output_max(MinimalEncoderProfile* p) {
    run_cublas_output_gemm(p);
    run_cublas_point_max(p);
}

void run_cublas_entity_full(MinimalEncoderProfile* p) {
    run_materialize_points(p);
    run_cublas_projection(p);
    run_cublas_output_max(p);
}

void profile_minimalenc(int B, int H) {
    printf("minimal_entity_encoder (B=%d, H=%d, obs=%d, points=%d, point_in=%d, entity_hidden=%d, precision=%s)\n",
        B, H, ME_OBS_SIZE, ME_NUM_POINTS, ME_ENTITY_IN, ME_ENTITY_HIDDEN,
        USE_BF16 ? "bf16" : "float32");
    printf("  %-30s %8s  %19s  %12s  %12s\n",
        "op", "time", "throughput", "math rate", "byte rate");

    auto* p = create_minimalenc(B, H);

    run_me_projection_kernel(p);
    run_me_projection(p);
    run_materialize_points(p);
    run_cublas_projection(p);
    run_cublas_output_gemm(p);
    cudaDeviceSynchronize();

    EncoderWork flat = work_flat_encoder(B, H);
    float ms = profile_kernel((kernel_fn)run_flat_encoder, p);
    print_encoder_timing("flat cuBLAS encoder", ms, B, flat.flops, flat.bytes);

    EncoderWork custom_proj = work_custom_projection(B);
    ms = profile_kernel((kernel_fn)run_me_projection_kernel, p);
    print_encoder_timing("custom projection+ReLU", ms, B, custom_proj.flops, custom_proj.bytes);

    EncoderWork materialize = work_materialize_points(B);
    EncoderWork puf_proj = add_work(materialize, work_cublas_projection_preact(B));
    ms = profile_kernel((kernel_fn)run_me_projection, p);
    print_encoder_timing("puf_mm materialize+proj", ms, B, puf_proj.flops, puf_proj.bytes);

    EncoderWork custom_lm = work_custom_linear_max(B, H, true);
    ms = profile_kernel((kernel_fn)run_me_linearmax, p);
    print_encoder_timing("custom fused ReLU+linear+max", ms, B, custom_lm.flops, custom_lm.bytes);

    EncoderWork custom_full = add_work(puf_proj, custom_lm);
    ms = profile_kernel((kernel_fn)run_me_full, p);
    print_encoder_timing("custom full forward", ms, B, custom_full.flops, custom_full.bytes);

    EncoderWork old_custom_full = add_work(custom_proj, custom_lm);
    ms = profile_kernel((kernel_fn)run_me_projection_kernel_full, p);
    print_encoder_timing("old custom-kernel full fwd", ms, B, old_custom_full.flops, old_custom_full.bytes);

    ms = profile_kernel((kernel_fn)run_materialize_points, p);
    print_encoder_timing("cuBLAS materialize points", ms, B, materialize.flops, materialize.bytes);

    EncoderWork cublas_proj = work_cublas_projection(B);
    ms = profile_kernel((kernel_fn)run_cublas_projection, p);
    print_encoder_timing("cuBLAS projection+ReLU", ms, B, cublas_proj.flops, cublas_proj.bytes);

    EncoderWork cublas_out = work_cublas_output_gemm(B, H);
    ms = profile_kernel((kernel_fn)run_cublas_output_gemm, p);
    print_encoder_timing("cuBLAS output GEMM", ms, B, cublas_out.flops, cublas_out.bytes);

    EncoderWork cublas_max = work_cublas_point_max(B, H, true);
    ms = profile_kernel((kernel_fn)run_cublas_point_max, p);
    print_encoder_timing("cuBLAS point max", ms, B, cublas_max.flops, cublas_max.bytes);

    EncoderWork cublas_output_max = add_work(cublas_out, cublas_max);
    ms = profile_kernel((kernel_fn)run_cublas_output_max, p);
    print_encoder_timing("cuBLAS output GEMM+max", ms, B, cublas_output_max.flops, cublas_output_max.bytes);

    EncoderWork cublas_full = add_work(add_work(materialize, cublas_proj), cublas_output_max);
    ms = profile_kernel((kernel_fn)run_cublas_entity_full, p);
    print_encoder_timing("cuBLAS entity full", ms, B, cublas_full.flops, cublas_full.bytes);

    printf("\n  Workload estimates include scalar comparisons and argmax writes.\n");
    printf("  The custom linear+max estimate uses tiled global traffic: entity_hidden is reread per hidden tile,\n");
    printf("  output_w is reread per batch tile, and the point-logit tensor is never materialized.\n");
    printf("  The cuBLAS entity path materializes point_logits with %0.2f MiB of write+read traffic before max.\n\n",
        (2.0 * sizeof(precision_t) * (double)B * ME_NUM_POINTS * H) / (1024.0 * 1024.0));

    printf("  backward\n");

    EncoderWork flat_bwd = work_flat_encoder_bwd(B, H);
    ms = profile_kernel((kernel_fn)run_flat_encoder_bwd, p);
    print_encoder_timing("flat cuBLAS backward", ms, B, flat_bwd.flops, flat_bwd.bytes);

    EncoderWork flat_fwd_bwd = add_work(flat, flat_bwd);
    ms = profile_kernel((kernel_fn)run_flat_encoder_fwd_bwd, p);
    print_encoder_timing("flat fwd+bwd", ms, B, flat_fwd_bwd.flops, flat_fwd_bwd.bytes);

    run_me_full(p);
    cudaDeviceSynchronize();

    EncoderWork output_wgrad = work_custom_output_wgrad(B, H);
    ms = profile_kernel((kernel_fn)run_me_output_wgrad, p);
    print_encoder_timing("custom output weight grad", ms, B, output_wgrad.flops, output_wgrad.bytes);

    EncoderWork grad_entity_scan = work_custom_grad_entity(B, H);
    ms = profile_kernel((kernel_fn)run_me_grad_entity_scan, p);
    print_encoder_timing("custom grad entity scan", ms, B,
        grad_entity_scan.flops, grad_entity_scan.bytes);

    EncoderWork grad_entity = work_atomic_grad_entity(B, H);
    ms = profile_kernel((kernel_fn)run_me_grad_entity, p);
    print_encoder_timing("shared-atomic grad entity", ms, B, grad_entity.flops, grad_entity.bytes);

    EncoderWork custom_input_wgrad = work_custom_input_wgrad(B);
    ms = profile_kernel((kernel_fn)run_me_input_wgrad_kernel, p);
    print_encoder_timing("custom input weight grad", ms, B,
        custom_input_wgrad.flops, custom_input_wgrad.bytes);

    EncoderWork input_wgrad = work_puf_input_wgrad(B);
    ms = profile_kernel((kernel_fn)run_me_input_wgrad, p);
    print_encoder_timing("puf_mm input weight grad", ms, B, input_wgrad.flops, input_wgrad.bytes);

    EncoderWork custom_bwd = add_work(add_work(output_wgrad, grad_entity), input_wgrad);
    ms = profile_kernel((kernel_fn)run_me_backward_full, p);
    print_encoder_timing("custom full backward", ms, B, custom_bwd.flops, custom_bwd.bytes);

    EncoderWork custom_fwd_bwd = add_work(custom_full, custom_bwd);
    ms = profile_kernel((kernel_fn)run_me_fwd_bwd_full, p);
    print_encoder_timing("custom fwd+bwd", ms, B, custom_fwd_bwd.flops, custom_fwd_bwd.bytes);

    printf("\n  Backward estimates model the kernels as written. The scan grad_entity estimate counts\n");
    printf("  one argmax probe per (batch, point, entity_hidden, hidden) tuple. The shared-atomic\n");
    printf("  grad_entity estimate counts one routed contribution per (batch, hidden, entity_hidden).\n\n");

    alloc_free(&p->alloc);
    free(p);
}

#ifdef PUFFERLIB_BUILD_MAIN
static void empty_net_callback(void* ctx, int buf, int t) {
    (void)ctx; (void)buf; (void)t;
}
static void empty_thread_init(void* ctx, int buf) {
    (void)ctx; (void)buf;
}

typedef struct {
    StaticVec* vec;
    int num_envs, num_buffers, num_threads, horizon, obs_size, num_atns;
} EnvSpeedArgs;

static int ini_handler_env(void* user, const char* section,
                           const char* name, const char* value) {
    Dict* env_kwargs = (Dict*)user;
    if (strcmp(section, "env") == 0) dict_set(env_kwargs, strdup(name), atof(value));
    return 1;
}

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

EnvSpeedArgs* create_envspeed(int total_agents, int num_buffers, int num_threads, int horizon) {
    char ini_path[512];
    snprintf(ini_path, sizeof(ini_path), "config/%s.ini", TOSTRING(ENV_NAME));

    VecDefaults defaults = {0};
    ini_parse(ini_path, ini_handler_vec, &defaults);
    if (total_agents == 0) total_agents = defaults.total_agents > 0 ? defaults.total_agents : 8192;
    if (num_buffers == 0) num_buffers = defaults.num_buffers > 0 ? defaults.num_buffers : 2;

    Dict* env_kwargs = create_dict(64);
    ini_parse(ini_path, ini_handler_env, env_kwargs);
    Dict* vec_kwargs = create_dict(8);
    dict_set(vec_kwargs, "total_agents", (double)total_agents);
    dict_set(vec_kwargs, "num_buffers", (double)num_buffers);

    StaticVec* vec = create_static_vec(total_agents, num_buffers, 1, vec_kwargs, env_kwargs);
    if (!vec) { fprintf(stderr, "Failed to create environments\n"); return nullptr; }
    for (int i = 0; i < num_buffers; i++)
        cudaStreamCreateWithFlags(&vec->streams[i], cudaStreamNonBlocking);

    printf("Created %d envs (%s) for %d total_agents\n", vec->size, TOSTRING(ENV_NAME), total_agents);
    create_static_threads(vec, num_threads, horizon, nullptr, empty_net_callback, empty_thread_init);
    static_vec_reset(vec);
    cudaDeviceSynchronize();

    EnvSpeedArgs* args = (EnvSpeedArgs*)calloc(1, sizeof(EnvSpeedArgs));
    args->vec = vec;
    args->num_envs = vec->size;
    args->num_buffers = num_buffers;
    args->num_threads = num_threads;
    args->horizon = horizon;
    args->obs_size = get_obs_size();
    args->num_atns = get_num_atns();
    return args;
}

void profile_envspeed(int total_agents, int num_buffers, int num_threads, int horizon) {
    printf("env_speed_static (total_agents=%d, buffers=%d, threads=%d, horizon=%d)\n",
           total_agents, num_buffers, num_threads, horizon);
    EnvSpeedArgs* args = create_envspeed(total_agents, num_buffers, num_threads, horizon);
    if (!args) { printf("  Failed to create env - skipping\n\n"); return; }
    printf("  num_envs=%d, obs_size=%d, num_atns=%d\n", args->num_envs, args->obs_size, args->num_atns);

    // Warmup
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 10; ++i) {
        static_vec_omp_step(args->vec);
        cudaDeviceSynchronize();
        float elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count();
        if (elapsed > 3.0f) break;
    }

    // Timed
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    t0 = std::chrono::steady_clock::now();
    cudaEventRecord(start);
    float completed = 0;
    for (int i = 0; i < 1000; ++i) {
        static_vec_omp_step(args->vec);
        completed += 1;
        float elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count();
        if (elapsed > 3.0f) break;
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float rollout_ms = ms / completed;
    int total_steps = total_agents * horizon;
    printf("  rollout time: %.2f ms (%d steps)\n", rollout_ms, total_steps);
    printf("  throughput: %.2f M steps/s\n", total_steps / rollout_ms / 1e3);
    free(args);
    printf("\n");
}
#else
void profile_envspeed(int total_agents, int num_buffers, int num_threads, int horizon) {
    (void)total_agents; (void)num_buffers; (void)num_threads; (void)horizon;
    printf("env_speed_static is unavailable in this profile build; rebuild with PUFFERLIB_BUILD_MAIN.\n\n");
}
#endif

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    const char* profile = argv[1];
    int buffers = BUF, threads = 16, horizon = T_;
    int total_agents = BR * buffers;
    int encoder_batch = BR, encoder_hidden = H_;
    for (int i = 2; i < argc - 1; i++) {
        if (strcmp(argv[i], "--buffers") == 0) buffers = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0) threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--horizon") == 0) horizon = atoi(argv[++i]);
        else if (strcmp(argv[i], "--total-agents") == 0) total_agents = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch") == 0) encoder_batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hidden") == 0) encoder_hidden = atoi(argv[++i]);
    }

    warmup_gpu();
    bool run_all = strcmp(profile, "all") == 0;

    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "mingrugate") == 0 || run_all)
        profile_mingrugate(BR, H_);
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "logcoeffsvals") == 0 || run_all)
        profile_logcoeffs(BT, T_, H_);
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "fusedscan") == 0 || run_all)
        profile_fusedscan(BT, T_, H_);
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "samplelogits") == 0 || run_all)
        profile_samplelogits(BR, A_);
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "ppoloss") == 0 || run_all)
        profile_ppoloss(BT, T_, A_);
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "im2col") == 0 || run_all) {
        profile_im2col(1024, N3_C1_IC, N3_MAP_H, N3_MAP_W, N3_C1_K, N3_C1_S, N3_C1_OH, N3_C1_OW);
        profile_im2col(1024, N3_C2_IC, N3_C1_OH, N3_C1_OW, N3_C2_K, N3_C2_S, N3_C2_OH, N3_C2_OW);
    }
    if (strcmp(profile, "minimalenc") == 0 || run_all)
        profile_minimalenc(encoder_batch, encoder_hidden);

    if (strcmp(profile, "envspeed") == 0 || run_all)
        profile_envspeed(total_agents, buffers, threads, horizon);

    if (!run_all
        && strcmp(profile, "kernels") != 0
        && strcmp(profile, "mingrugate") != 0
        && strcmp(profile, "logcoeffsvals") != 0
        && strcmp(profile, "fusedscan") != 0
        && strcmp(profile, "samplelogits") != 0
        && strcmp(profile, "ppoloss") != 0
        && strcmp(profile, "im2col") != 0
        && strcmp(profile, "minimalenc") != 0
        && strcmp(profile, "envspeed") != 0
    ) {
        printf("Unknown profile: %s\n\n", profile);
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
