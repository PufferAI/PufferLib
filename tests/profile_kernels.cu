#include <string>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <type_traits>

#include "pufferlib.cu"
#include "ini.h"

static_assert(std::is_same_v<std::remove_reference_t<decltype(RolloutBuf{}.actions)>, FloatTensor>);
static_assert(std::is_same_v<std::remove_reference_t<decltype(TrainGraph{}.mb_actions)>, FloatTensor>);
static_assert(std::is_same_v<decltype(PPOGraphArgs{}.actions), const float*>);
static_assert(std::is_same_v<std::remove_reference_t<decltype(RolloutBuf{}.observations)>, PrecisionTensor>);
static_assert(std::is_same_v<std::remove_reference_t<decltype(RolloutBuf{}.values)>, PrecisionTensor>);
static_assert(std::is_same_v<std::remove_reference_t<decltype(TrainGraph{}.mb_obs)>, PrecisionTensor>);
static_assert(std::is_same_v<std::remove_reference_t<decltype(TrainGraph{}.mb_logprobs)>, PrecisionTensor>);

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
    printf("  actiontransport - Verify exact discrete and rounded continuous action transport\n");
    printf("  ppoloss        - PPO loss fused fwd+bwd kernel\n");
    printf("  im2col         - im2col + col2im (nmmo3 conv sizes, B=1024)\n");
    printf("  envspeed       - Environment step throughput\n");
    printf("    --buffers N  - Number of buffers (default: %d)\n", BUF);
    printf("    --threads N  - Number of threads (default: 16)\n");
    printf("    --horizon N  - Horizon length (default: %d)\n", T_);
    printf("  all            - Run all available profiles\n");
}

inline void print_timing(const char* name, float ms, int N) {
    printf("  %-28s %8.1f us  %8.2f M elem/s\n", name, ms * 1000, N / ms / 1e3);
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
    PrecisionTensor grad_out, grad_next_state;
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
    p->grad_next_state = {.shape = {B, H}};

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
    alloc_register(&p->alloc, &p->grad_next_state);
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
    float_to_device(p->grad_next_state.data, buf, N_state);
    free(buf);
    return p;
}

void run_fusedscan_fwd(FusedScanProfile* p) {
    mingru_scan_forward<<<grid_size(p->B * p->H), BLOCK_SIZE>>>(p->scan);
}

void run_fusedscan_bwd(FusedScanProfile* p) {
    mingru_scan_backward<<<grid_size(p->B * p->H), BLOCK_SIZE>>>(
        p->scan, p->grad_out.data, p->grad_next_state.data);
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
    FloatTensor grad_logits_t, grad_values_t, adv_mean_t, adv_var_t, ent_coef_t;
    FloatTensor actions_t;
    PrecisionTensor logits_t, old_logprobs_t, advantages_t, prio_t, values_t, returns_t;
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
    p->ent_coef_t     = {.shape = {1}};
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
    alloc_register(&p->alloc, &p->ent_coef_t);
    alloc_register(&p->alloc, &p->loss);
    alloc_register(&p->alloc, &p->losses_acc);
    alloc_register(&p->alloc, &p->ppo_partials);
    alloc_register(&p->alloc, &p->act_sizes_t);
    alloc_create(&p->alloc);

    cudaMemcpy(p->act_sizes_t.data, &A, sizeof(int), cudaMemcpyHostToDevice);

    float ent_coef_val = 0.01f;
    cudaMemcpy(p->ent_coef_t.data, &ent_coef_val, sizeof(float), cudaMemcpyHostToDevice);

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
    cudaMemcpy(p->actions_t.data, buf, NT * sizeof(float), cudaMemcpyHostToDevice);
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
        .clip_coef = 0.1f, .vf_clip_coef = 0.1f, .vf_coef = 0.5f, .ent_coef = p->ent_coef_t.data,
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
    FloatTensor actions_t, env_actions_t;
    PrecisionTensor logprobs_t, value_out_t;
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
    p->env_actions_t = {.shape = {B}};
    p->logprobs_t  = {.shape = {B}};
    p->value_out_t = {.shape = {B}};

    p->alloc = {};
    alloc_register(&p->alloc, &p->dec_out);
    alloc_register(&p->alloc, &p->act_sizes);
    alloc_register(&p->alloc, &p->actions_t);
    alloc_register(&p->alloc, &p->env_actions_t);
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
        p->actions_t.data, p->env_actions_t.data,
        p->logprobs_t.data, p->value_out_t.data,
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

static float round_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void require_equal(const char* path, const std::vector<float>& actual,
        const std::vector<float>& expected) {
    for (size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            throw std::runtime_error(std::string(path) + " changed action "
                + std::to_string(expected[i]) + " to " + std::to_string(actual[i]));
        }
    }
}

static std::vector<float> precision_to_host(const precision_t* device, int count) {
    std::vector<precision_t> stored(count);
    std::vector<float> result(count);
    cudaMemcpy(stored.data(), device, count * sizeof(precision_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < count; ++i) result[i] = to_float(stored[i]);
    return result;
}

void test_action_transport() {
    constexpr int T = 2;
    constexpr int B = 4;
    constexpr int H = 2;
    constexpr int MB = 2;
    constexpr int A0 = 3572;
    constexpr int A1 = 7;
    constexpr int A = A0 + A1;
    constexpr int S = T * B;
    const std::vector<float> expected = {
        257, 1, 511, 2, 2049, 3, 3571, 4,
        511, 5, 2049, 6, 3571, 0, 257, 1,
    };

    PrecisionTensor dec_out = {.shape = {S, A + 1}};
    PrecisionTensor action_mask = {.shape = {S, A}};
    PrecisionTensor logprobs = {.shape = {S}};
    PrecisionTensor values = {.shape = {S}};
    FloatTensor rollout_actions = {.shape = {S, H}};
    FloatTensor env_actions = {.shape = {S, H}};
    IntTensor act_sizes = {.shape = {H}};

    PrecisionTensor advantages = {.shape = {B, T}};
    FloatTensor priorities = {.shape = {MB}};
    IntTensor indices = {.shape = {MB}};

    Allocator alloc = {};
    RolloutBuf transposed;
    TrainGraph minibatch;
    register_rollout_buffers(transposed, &alloc, B, T, 1, H, 0);
    register_train_buffers(minibatch, &alloc, MB, T, 1, 1, H, 1, 0);
    alloc_register(&alloc, &dec_out);
    alloc_register(&alloc, &action_mask);
    alloc_register(&alloc, &logprobs);
    alloc_register(&alloc, &values);
    alloc_register(&alloc, &rollout_actions);
    alloc_register(&alloc, &env_actions);
    alloc_register(&alloc, &act_sizes);
    alloc_register(&alloc, &advantages);
    alloc_register(&alloc, &priorities);
    alloc_register(&alloc, &indices);
    if (alloc_create(&alloc) != cudaSuccess) {
        throw std::runtime_error("action transport test allocation failed");
    }

    std::vector<float> logits(S * (A + 1), 0.0f);
    std::vector<float> mask(S * A, 0.0f);
    for (int i = 0; i < S; ++i) {
        mask[i * A + (int)expected[i * H]] = 1.0f;
        mask[i * A + A0 + (int)expected[i * H + 1]] = 1.0f;
    }
    float_to_device(dec_out.data, logits.data(), logits.size());
    float_to_device(action_mask.data, mask.data(), mask.size());
    cudaMemset(transposed.observations.data, 0, B * T * sizeof(precision_t));
    cudaMemset(transposed.logprobs.data, 0, B * T * sizeof(precision_t));
    cudaMemset(transposed.values.data, 0, B * T * sizeof(precision_t));
    cudaMemset(advantages.data, 0, B * T * sizeof(precision_t));
    const float host_priorities[MB] = {1.0f, 1.0f};
    const int action_sizes[H] = {A0, A1};
    const int host_indices[MB] = {2, 0};
    cudaMemcpy(priorities.data, host_priorities, sizeof(host_priorities), cudaMemcpyHostToDevice);
    cudaMemcpy(act_sizes.data, action_sizes, sizeof(action_sizes), cudaMemcpyHostToDevice);
    cudaMemcpy(indices.data, host_indices, sizeof(host_indices), cudaMemcpyHostToDevice);

    curandStatePhilox4_32_10_t* rng_states = nullptr;
    cudaMalloc(&rng_states, S * sizeof(*rng_states));
    rng_init<<<grid_size(S), BLOCK_SIZE>>>(rng_states, 42, S);
    PrecisionTensor no_logstd = {};
    sample_logits<<<grid_size(S), BLOCK_SIZE>>>(
        dec_out, no_logstd, act_sizes, rollout_actions.data, env_actions.data,
        logprobs.data, values.data, rng_states, action_mask.data, A);
    transpose_102<<<grid_size(S * H), BLOCK_SIZE>>>(
        transposed.actions.data, rollout_actions.data, T, B, H);
    select_copy<<<dim3(MB, 5), SELECT_COPY_THREADS>>>(
        transposed, minibatch, indices.data, advantages.data, priorities.data);

    std::vector<float> expected_transposed(S * H);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < H; ++h) {
                expected_transposed[(b * T + t) * H + h] = expected[(t * B + b) * H + h];
            }
        }
    }
    std::vector<float> expected_minibatch(MB * T * H);
    for (int mb = 0; mb < MB; ++mb) {
        std::copy_n(expected_transposed.begin() + host_indices[mb] * T * H, T * H,
                    expected_minibatch.begin() + mb * T * H);
    }

    std::vector<float> rollout(S * H), env(S * H), train(S * H), ppo(MB * T * H);
    cudaMemcpy(rollout.data(), rollout_actions.data, rollout.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(env.data(), env_actions.data, env.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(train.data(), transposed.actions.data, train.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(ppo.data(), minibatch.mb_actions.data, ppo.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    require_equal("rollout", rollout, expected);
    require_equal("environment", env, expected);
    require_equal("transpose", train, expected_transposed);
    require_equal("PPO minibatch", ppo, expected_minibatch);

    // Synthetic primary + two frozen-bank slices. This isolates the captured
    // sampler's pointer offsets; it is not a full WeightBank integration test.
    constexpr int BANKS = 3;
    const int bank_starts[BANKS + 1] = {0, 2, 5, S};
    cudaStream_t graph_stream;
    cudaGraph_t action_graph;
    cudaGraphExec_t action_graph_exec;
    if (cudaStreamCreateWithFlags(&graph_stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeGlobal) != cudaSuccess) {
        throw std::runtime_error("synthetic bank graph capture setup failed");
    }
    for (int bank = 0; bank < BANKS; ++bank) {
        int start = bank_starts[bank];
        int count = bank_starts[bank + 1] - start;
        PrecisionTensor bank_out = {
            .data = dec_out.data + (long)start * (A + 1), .shape = {count, A + 1}};
        sample_logits<<<grid_size(count), BLOCK_SIZE, 0, graph_stream>>>(
            bank_out, no_logstd, act_sizes,
            rollout_actions.data + (long)start * H,
            env_actions.data + (long)start * H,
            logprobs.data + start, values.data + start, rng_states + start,
            action_mask.data + (long)start * A, A);
    }
    if (cudaStreamEndCapture(graph_stream, &action_graph) != cudaSuccess ||
        cudaGraphInstantiate(&action_graph_exec, action_graph, 0) != cudaSuccess) {
        throw std::runtime_error("synthetic bank graph creation failed");
    }
    for (int replay = 0; replay < 2; ++replay) {
        cudaMemsetAsync(rollout_actions.data, 0xff, S * H * sizeof(float), graph_stream);
        cudaMemsetAsync(env_actions.data, 0xff, S * H * sizeof(float), graph_stream);
        if (cudaGraphLaunch(action_graph_exec, graph_stream) != cudaSuccess ||
            cudaStreamSynchronize(graph_stream) != cudaSuccess) {
            throw std::runtime_error("synthetic bank graph replay failed");
        }
        cudaMemcpy(rollout.data(), rollout_actions.data, rollout.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(env.data(), env_actions.data, env.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
        require_equal("synthetic bank graph rollout", rollout, expected);
        require_equal("synthetic bank graph environment", env, expected);
    }
    cudaGraphExecDestroy(action_graph_exec);
    cudaGraphDestroy(action_graph);
    cudaStreamDestroy(graph_stream);

    PPOProfile* discrete_ppo = create_ppoloss(1, 1, A0);
    std::vector<float> ppo_logits(A0 + 1, 0.0f);
    float_to_device(discrete_ppo->logits_t.data, ppo_logits.data(), ppo_logits.size());
    const float one = 1.0f;
    const float zero = 0.0f;
    const float old_logprob = -logf((float)A0);
    float_to_device(discrete_ppo->old_logprobs_t.data, &old_logprob, 1);
    float_to_device(discrete_ppo->advantages_t.data, &one, 1);
    float_to_device(discrete_ppo->prio_t.data, &one, 1);
    float_to_device(discrete_ppo->values_t.data, &zero, 1);
    float_to_device(discrete_ppo->returns_t.data, &zero, 1);
    cudaMemcpy(discrete_ppo->adv_mean_t.data, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(discrete_ppo->adv_var_t.data, &one, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(discrete_ppo->ent_coef_t.data, &zero, sizeof(float), cudaMemcpyHostToDevice);
    discrete_ppo->ga.actions = minibatch.mb_actions.data;
    run_ppoloss(discrete_ppo);
    std::vector<float> discrete_grad(A0);
    cudaMemcpy(discrete_grad.data(), discrete_ppo->grad_logits_t.data,
               A0 * sizeof(float), cudaMemcpyDeviceToHost);
    int selected_action = (int)expected_minibatch[0];
    if (!(discrete_grad[selected_action] < 0.0f &&
          discrete_grad[selected_action - 1] > 0.0f)) {
        throw std::runtime_error("PPO did not consume the selected exact large action");
    }
    alloc_free(&discrete_ppo->alloc);
    free(discrete_ppo);

    constexpr int CB = 32;
    constexpr int CH = 2;
    PrecisionTensor continuous_out = {.shape = {CB, CH + 1}};
    PrecisionTensor logstd = {.shape = {CH}};
    PrecisionTensor continuous_logprobs = {.shape = {CB}};
    PrecisionTensor continuous_values = {.shape = {CB}};
    FloatTensor continuous_rollout = {.shape = {CB, CH}};
    FloatTensor continuous_env = {.shape = {CB, CH}};
    IntTensor continuous_size = {.shape = {CH}};
    Allocator continuous_alloc = {};
    alloc_register(&continuous_alloc, &continuous_out);
    alloc_register(&continuous_alloc, &logstd);
    alloc_register(&continuous_alloc, &continuous_logprobs);
    alloc_register(&continuous_alloc, &continuous_values);
    alloc_register(&continuous_alloc, &continuous_rollout);
    alloc_register(&continuous_alloc, &continuous_env);
    alloc_register(&continuous_alloc, &continuous_size);
    if (alloc_create(&continuous_alloc) != cudaSuccess) {
        throw std::runtime_error("continuous action transport test allocation failed");
    }

    std::vector<float> continuous_logits(CB * (CH + 1));
    for (int i = 0; i < CB; ++i) {
        continuous_logits[i * (CH + 1)] = 0.1234567f;
        continuous_logits[i * (CH + 1) + 1] = -0.2345678f;
        continuous_logits[i * (CH + 1) + 2] = 0.0f;
    }
    const float continuous_logstd[CH] = {-1.25f, -0.7f};
    const int continuous_sizes[CH] = {1, 1};
    float_to_device(continuous_out.data, continuous_logits.data(), continuous_logits.size());
    float_to_device(logstd.data, continuous_logstd, CH);
    cudaMemcpy(continuous_size.data, continuous_sizes, sizeof(continuous_sizes),
               cudaMemcpyHostToDevice);
    cudaFree(rng_states);
    cudaMalloc(&rng_states, CB * sizeof(*rng_states));
    rng_init<<<grid_size(CB), BLOCK_SIZE>>>(rng_states, 73, CB);
    sample_logits<<<grid_size(CB), BLOCK_SIZE>>>(
        continuous_out, logstd, continuous_size,
        continuous_rollout.data, continuous_env.data,
        continuous_logprobs.data, continuous_values.data,
        rng_states, nullptr, 0);

    std::vector<float> continuous_rollout_host(CB * CH), continuous_env_host(CB * CH);
    cudaMemcpy(continuous_rollout_host.data(), continuous_rollout.data,
        CB * CH * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(continuous_env_host.data(), continuous_env.data,
        CB * CH * sizeof(float), cudaMemcpyDeviceToHost);
    require_equal("continuous environment", continuous_env_host, continuous_rollout_host);
    bool retained_non_bf16 = false;
    if (USE_BF16) {
        for (float action : continuous_rollout_host) {
            if (action != round_to_bf16(action)) {
                throw std::runtime_error("continuous action bypassed BF16 rounding");
            }
        }
    } else {
        for (float action : continuous_rollout_host) {
            retained_non_bf16 |= action != round_to_bf16(action);
        }
        if (!retained_non_bf16) {
            throw std::runtime_error("FP32 continuous probe did not retain extra precision");
        }
    }

    std::vector<float> continuous_logprobs_host =
        precision_to_host(continuous_logprobs.data, CB);
    std::vector<float> recomputed_logprobs(CB);
    const float means[CH] = {
        USE_BF16 ? round_to_bf16(0.1234567f) : 0.1234567f,
        USE_BF16 ? round_to_bf16(-0.2345678f) : -0.2345678f,
    };
    const float logstds[CH] = {
        USE_BF16 ? round_to_bf16(continuous_logstd[0]) : continuous_logstd[0],
        USE_BF16 ? round_to_bf16(continuous_logstd[1]) : continuous_logstd[1],
    };
    for (int i = 0; i < CB; ++i) {
        float expected_logprob = 0.0f;
        for (int h = 0; h < CH; ++h) {
            float normalized = (continuous_rollout_host[i * CH + h] - means[h]) / expf(logstds[h]);
            expected_logprob +=
                -0.5f * normalized * normalized - 0.9189385332046727f - logstds[h];
        }
        recomputed_logprobs[i] = expected_logprob;
        float tolerance = USE_BF16 ? 0.02f : 2e-5f;
        if (fabsf(continuous_logprobs_host[i] - expected_logprob) > tolerance) {
            throw std::runtime_error("continuous transported action/logprob mismatch");
        }
    }

    PrecisionTensor ppo_advantages = {.shape = {1}};
    PrecisionTensor ppo_priorities = {.shape = {1}};
    PrecisionTensor ppo_values = {.shape = {1}};
    PrecisionTensor ppo_returns = {.shape = {1}};
    PrecisionTensor ppo_ratio = {.shape = {1}};
    PrecisionTensor ppo_newvalue = {.shape = {1}};
    FloatTensor ppo_grad_logits = {.shape = {CH}};
    FloatTensor ppo_grad_logstd = {.shape = {CH}};
    FloatTensor ppo_grad_values = {.shape = {1}};
    FloatTensor ppo_adv_mean = {.shape = {1}};
    FloatTensor ppo_adv_var = {.shape = {1}};
    FloatTensor ppo_ent_coef = {.shape = {1}};
    FloatTensor ppo_partials = {.shape = {LOSS_N + 1}};
    Allocator ppo_alloc = {};
    alloc_register(&ppo_alloc, &ppo_advantages);
    alloc_register(&ppo_alloc, &ppo_priorities);
    alloc_register(&ppo_alloc, &ppo_values);
    alloc_register(&ppo_alloc, &ppo_returns);
    alloc_register(&ppo_alloc, &ppo_ratio);
    alloc_register(&ppo_alloc, &ppo_newvalue);
    alloc_register(&ppo_alloc, &ppo_grad_logits);
    alloc_register(&ppo_alloc, &ppo_grad_logstd);
    alloc_register(&ppo_alloc, &ppo_grad_values);
    alloc_register(&ppo_alloc, &ppo_adv_mean);
    alloc_register(&ppo_alloc, &ppo_adv_var);
    alloc_register(&ppo_alloc, &ppo_ent_coef);
    alloc_register(&ppo_alloc, &ppo_partials);
    if (alloc_create(&ppo_alloc) != cudaSuccess) {
        throw std::runtime_error("continuous PPO probe allocation failed");
    }
    float_to_device(ppo_advantages.data, &one, 1);
    float_to_device(ppo_priorities.data, &one, 1);
    float_to_device(ppo_values.data, &zero, 1);
    float_to_device(ppo_returns.data, &zero, 1);
    cudaMemcpy(ppo_adv_mean.data, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ppo_adv_var.data, &one, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ppo_ent_coef.data, &zero, sizeof(float), cudaMemcpyHostToDevice);
    PPOKernelArgs ka = {
        .grad_logits = ppo_grad_logits.data,
        .grad_logstd = ppo_grad_logstd.data,
        .grad_values_pred = ppo_grad_values.data,
        .logits = continuous_out.data,
        .logstd = logstd.data,
        .values_pred = continuous_out.data + CH,
        .adv_mean = ppo_adv_mean.data,
        .adv_var = ppo_adv_var.data,
        .act_sizes = continuous_size.data,
        .num_atns = CH,
        .clip_coef = 0.1f,
        .vf_clip_coef = 0.1f,
        .vf_coef = 0.5f,
        .ent_coef = ppo_ent_coef.data,
        .T_seq = 1,
        .A_total = CH,
        .N = 1,
        .logits_stride_n = CH + 1,
        .logits_stride_t = CH + 1,
        .logits_stride_a = 1,
        .values_stride_n = CH + 1,
        .values_stride_t = CH + 1,
        .is_continuous = true,
    };
    PPOGraphArgs ga = {
        .out_ratio = ppo_ratio.data,
        .out_newvalue = ppo_newvalue.data,
        .actions = continuous_rollout.data,
        .old_logprobs = continuous_logprobs.data,
        .advantages = ppo_advantages.data,
        .prio = ppo_priorities.data,
        .values = ppo_values.data,
        .returns = ppo_returns.data,
    };
    ppo_loss_compute<<<1, PPO_THREADS>>>(ppo_partials.data, ka, ga);
    std::vector<float> continuous_grad_logits(CH), continuous_grad_logstd(CH);
    cudaMemcpy(continuous_grad_logits.data(), ppo_grad_logits.data, CH * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(continuous_grad_logstd.data(), ppo_grad_logstd.data, CH * sizeof(float),
               cudaMemcpyDeviceToHost);
    float expected_ratio = expf(recomputed_logprobs[0] - continuous_logprobs_host[0]);
    for (int h = 0; h < CH; ++h) {
        float action = continuous_rollout_host[h];
        float variance = expf(2.0f * logstds[h]);
        float diff = action - means[h];
        float expected_mean_grad = -expected_ratio * diff / variance;
        float expected_std_grad = -expected_ratio * (diff * diff / variance - 1.0f);
        if (!isfinite(continuous_grad_logits[h]) || !isfinite(continuous_grad_logstd[h]) ||
            fabsf(continuous_grad_logits[h] - expected_mean_grad) > 2e-4f ||
            fabsf(continuous_grad_logstd[h] - expected_std_grad) > 2e-4f) {
            throw std::runtime_error("continuous PPO did not consume transported float actions");
        }
    }

    alloc_free(&ppo_alloc);
    cudaFree(rng_states);
    alloc_free(&continuous_alloc);
    alloc_free(&alloc);
    printf("action transport: passed\n");
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

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    const char* profile = argv[1];
    int buffers = BUF, threads = 16, horizon = T_;
    int total_agents = BR * buffers;
    for (int i = 2; i < argc - 1; i++) {
        if (strcmp(argv[i], "--buffers") == 0) buffers = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0) threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--horizon") == 0) horizon = atoi(argv[++i]);
        else if (strcmp(argv[i], "--total-agents") == 0) total_agents = atoi(argv[++i]);
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
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "actiontransport") == 0 || run_all)
        test_action_transport();
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "ppoloss") == 0 || run_all)
        profile_ppoloss(BT, T_, A_);
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "im2col") == 0 || run_all) {
        profile_im2col(1024, N3_C1_IC, N3_MAP_H, N3_MAP_W, N3_C1_K, N3_C1_S, N3_C1_OH, N3_C1_OW);
        profile_im2col(1024, N3_C2_IC, N3_C1_OH, N3_C1_OW, N3_C2_K, N3_C2_S, N3_C2_OH, N3_C2_OW);
    }

    if (strcmp(profile, "envspeed") == 0 || run_all)
        profile_envspeed(total_agents, buffers, threads, horizon);

    if (!run_all
        && strcmp(profile, "kernels") != 0
        && strcmp(profile, "mingrugate") != 0
        && strcmp(profile, "logcoeffsvals") != 0
        && strcmp(profile, "fusedscan") != 0
        && strcmp(profile, "samplelogits") != 0
        && strcmp(profile, "actiontransport") != 0
        && strcmp(profile, "ppoloss") != 0
        && strcmp(profile, "im2col") != 0
        && strcmp(profile, "envspeed") != 0
    ) {
        printf("Unknown profile: %s\n\n", profile);
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
