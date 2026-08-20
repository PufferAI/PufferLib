#include <string>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <cstdlib>

#include "pufferlib.cu"
#include "ini.h"

const int WARMUP_ITERS = 100;
const int TIMING_ITERS = 1000;
constexpr int kFusedscanBlockSweepBlockSize = 256;

const int BUF = 2;
const int BR = 4096;   // Rollout batch (no T dim)
const int BT = 512;    // Train batch (with T dim)
const int T_ = 64;     // T_ to avoid collision with PrefixScan::T
const int H_ = 128;
const int A_ = 4;

#ifndef ENV_NAME
#error "ENV_NAME must be defined at compile time (e.g. -DENV_NAME=breakout)"
#endif
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

typedef void (*kernel_fn)(void*);
void print_selected_sweep_sizes();

void print_usage(const char* prog) {
    printf("Usage: %s <profile>\n", prog);
    printf("\nProfiles:\n");
    printf("  kernels                - All individual kernel microbenchmarks\n");
    printf("  mingrugate             - MinGRU gate kernel only\n");
    printf("  logcoeffsvals          - log_coeffs_and_values fwd+bwd\n");
    printf("  fusedscan              - Fused scan (checkpointed) kernel only\n");
    printf("  fusedscan_correctness  - Correctness-only check: log vec variants vs log_scalar\n");
    printf("  fusedscan_sweep        - Fixed-block (256) sweep for log scalar/vec32/vec64/vec128 kernels\n");
    printf("  fusedscan_selector_bench - Baseline selector vs depth2 selector benchmark\n");
    printf("  samplelogits           - Sample logits kernel only\n");
    printf("  ppoloss                - PPO loss fused fwd+bwd kernel\n");
    printf("  im2col                 - im2col + col2im (nmmo3 conv sizes, B=1024)\n");
    printf("  envspeed               - Environment step throughput\n");
    printf("    --ckpt-intervals CSV  - Requested checkpoint intervals for fusedscan sweeps (e.g. 1,4,8)\n");
    printf("    --b-sizes CSV         - Sweep B sizes for fusedscan *_sweep profiles (e.g. 64,128,256)\n");
    printf("    --t-sizes CSV         - Sweep T sizes for fusedscan *_sweep profiles (e.g. 64,128,256)\n");
    printf("    --h-sizes CSV         - Sweep H sizes for fusedscan *_sweep profiles (e.g. 128,256,512)\n");
    printf("    --buffers N           - Number of buffers (default: %d)\n", BUF);
    printf("    --threads N           - Number of threads (default: 16)\n");
    printf("    --horizon N           - Horizon length (default: %d)\n", T_);
    printf("  all                    - Run all available profiles\n");
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

__device__ __forceinline__ uint32_t scan_hash_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__global__ void fill_precision_pseudorand_signed_kernel(
        precision_t* __restrict__ dst, int n, uint32_t seed, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    uint32_t h = scan_hash_u32((uint32_t)idx ^ seed);
    float u = (float)(h & 0x00ffffffU) * (1.0f / 16777215.0f); // [0, 1]
    float centered = 2.0f * u - 1.0f;                           // [-1, 1]
    dst[idx] = from_float(centered * scale);
}

__global__ void fill_precision_pseudorand_positive_kernel(
        precision_t* __restrict__ dst, int n, uint32_t seed, float base, float span) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    uint32_t h = scan_hash_u32((uint32_t)idx ^ seed);
    float u = (float)(h & 0x00ffffffU) * (1.0f / 16777215.0f); // [0, 1]
    dst[idx] = from_float(base + span * u);
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

struct LogCoeffsProfile {
    FloatTensor gate, hidden, log_coeff, log_value;
    FloatTensor grad_log_coeffs, grad_log_values, grad_gate, grad_hidden;
    Allocator alloc;
    int N;
};

LogCoeffsProfile* create_logcoeffs(int N, const char* tag = "logcoeffs") {
    auto* p = (LogCoeffsProfile*)calloc(1, sizeof(LogCoeffsProfile));
    if (!p) {
        printf("%s N=%d SKIP: host allocation failed\n", tag, N);
        return nullptr;
    }
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
    cudaError_t alloc_err = alloc_create(&p->alloc);
    if (alloc_err != cudaSuccess) {
        double gib = (double)p->alloc.total_bytes / (1024.0 * 1024.0 * 1024.0);
        printf("%s N=%d SKIP: alloc_create %.2f GiB failed (%s)\n",
            tag, N, gib, cudaGetErrorString(alloc_err));
        cudaGetLastError(); // clear sticky runtime error so subsequent cases can proceed
        alloc_free(&p->alloc);
        free(p);
        return nullptr;
    }

    float* buf = (float*)malloc(N * sizeof(float));
    if (!buf) {
        printf("%s N=%d SKIP: host input buffer allocation failed\n", tag, N);
        alloc_free(&p->alloc);
        free(p);
        return nullptr;
    }
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
    auto* p = create_logcoeffs(N, "logcoeffsvals");
    if (!p) {
        printf("\n");
        return;
    }
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

static const int kSweepBValsDefault[] = {64, 128, 256, 512};
static const int kSweepTValsDefault[] = {64, 128, 256, 512};
static const int kSweepHValsDefault[] = {128, 256, 512, 1024};
static constexpr int kSweepDefaultBCount = sizeof(kSweepBValsDefault) / sizeof(kSweepBValsDefault[0]);
static constexpr int kSweepDefaultTCount = sizeof(kSweepTValsDefault) / sizeof(kSweepTValsDefault[0]);
static constexpr int kSweepDefaultHCount = sizeof(kSweepHValsDefault) / sizeof(kSweepHValsDefault[0]);
static constexpr int kMaxSweepSizes = 64;
static int gSelectedBVals[kMaxSweepSizes] = {64, 128, 256, 512};
static int gSelectedTVals[kMaxSweepSizes] = {64, 128, 256, 512};
static int gSelectedHVals[kMaxSweepSizes] = {128, 256, 512, 1024};
static int gSelectedBCount = kSweepDefaultBCount;
static int gSelectedTCount = kSweepDefaultTCount;
static int gSelectedHCount = kSweepDefaultHCount;
FusedScanProfile* create_fusedscan(int B, int T, int H, const char* tag = "fused_scan") {
    auto* p = (FusedScanProfile*)calloc(1, sizeof(FusedScanProfile));
    if (!p) {
        printf("%s %dx%dx%d SKIP: host allocation failed\n", tag, B, T, H);
        return nullptr;
    }
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
    cudaError_t alloc_err = alloc_create(&p->alloc);
    if (alloc_err != cudaSuccess) {
        double gib = (double)p->alloc.total_bytes / (1024.0 * 1024.0 * 1024.0);
        printf("%s %dx%dx%d SKIP: alloc_create %.2f GiB failed (%s)\n",
            tag, B, T, H, gib, cudaGetErrorString(alloc_err));
        cudaGetLastError(); // clear sticky runtime error so subsequent cases can proceed
        alloc_free(&p->alloc);
        free(p);
        return nullptr;
    }

    s.combined_ptr = combined_t.data;
    s.state_ptr    = state_t.data;
    s.input_ptr    = input_t.data;

    // Use non-trivial deterministic values for correctness checks.
    // Keep state strictly positive to avoid log(0) in log-space kernels.
    int N_combined = B * T * 3 * H;
    int N_state = B * H;
    int N_input = B * T * H;
    int N_grad_out = N_input;
    int N_grad_next_state = N_state;
    uint32_t seed_base = 0x9e3779b9U
        ^ ((uint32_t)B * 73856093U)
        ^ ((uint32_t)T * 19349663U)
        ^ ((uint32_t)H * 83492791U);

    fill_precision_pseudorand_signed_kernel<<<grid_size(N_combined), BLOCK_SIZE>>>(
        s.combined_ptr, N_combined, seed_base ^ 0x11111111U, 5.0f);
    cudaError_t init_err = cudaGetLastError();
    if (init_err != cudaSuccess) {
        printf("%s %dx%dx%d SKIP: combined init launch failed (%s)\n",
            tag, B, T, H, cudaGetErrorString(init_err));
        cudaGetLastError();
        alloc_free(&p->alloc);
        free(p);
        return nullptr;
    }

    fill_precision_pseudorand_positive_kernel<<<grid_size(N_state), BLOCK_SIZE>>>(
        s.state_ptr, N_state, seed_base ^ 0x22222222U, 0.25f, 1.25f);
    init_err = cudaGetLastError();
    if (init_err != cudaSuccess) {
        printf("%s %dx%dx%d SKIP: state init launch failed (%s)\n",
            tag, B, T, H, cudaGetErrorString(init_err));
        cudaGetLastError();
        alloc_free(&p->alloc);
        free(p);
        return nullptr;
    }

    fill_precision_pseudorand_signed_kernel<<<grid_size(N_input), BLOCK_SIZE>>>(
        s.input_ptr, N_input, seed_base ^ 0x33333333U, 1.0f);
    init_err = cudaGetLastError();
    if (init_err != cudaSuccess) {
        printf("%s %dx%dx%d SKIP: input init launch failed (%s)\n",
            tag, B, T, H, cudaGetErrorString(init_err));
        cudaGetLastError();
        alloc_free(&p->alloc);
        free(p);
        return nullptr;
    }

    fill_precision_pseudorand_signed_kernel<<<grid_size(N_grad_out), BLOCK_SIZE>>>(
        p->grad_out.data, N_grad_out, seed_base ^ 0x44444444U, 1.0f);
    init_err = cudaGetLastError();
    if (init_err != cudaSuccess) {
        printf("%s %dx%dx%d SKIP: grad_out init launch failed (%s)\n",
            tag, B, T, H, cudaGetErrorString(init_err));
        cudaGetLastError();
        alloc_free(&p->alloc);
        free(p);
        return nullptr;
    }

    fill_precision_pseudorand_signed_kernel<<<grid_size(N_grad_next_state), BLOCK_SIZE>>>(
        p->grad_next_state.data, N_grad_next_state, seed_base ^ 0x55555555U, 1.0f);
    init_err = cudaGetLastError();
    if (init_err != cudaSuccess) {
        printf("%s %dx%dx%d SKIP: grad_next_state init launch failed (%s)\n",
            tag, B, T, H, cudaGetErrorString(init_err));
        cudaGetLastError();
        alloc_free(&p->alloc);
        free(p);
        return nullptr;
    }

    // Ensure all init kernels are complete before first use.
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        printf("%s %dx%dx%d SKIP: init sync failed (%s)\n",
            tag, B, T, H, cudaGetErrorString(sync_err));
        cudaGetLastError();
        alloc_free(&p->alloc);
        free(p);
        return nullptr;
    }

    return p;
}

void run_fusedscan_fwd(FusedScanProfile* p) {
    mingru_scan_forward<<<grid_size(p->B * p->H), BLOCK_SIZE>>>(p->scan);
}

void run_fusedscan_bwd(FusedScanProfile* p) {
    mingru_scan_backward<<<grid_size(p->B * p->H), BLOCK_SIZE>>>(
        p->scan, p->grad_out.data, p->grad_next_state.data);
}

template<int CKPT_INTERVAL>
void run_fusedscan_fwd_ckpt_tuned(FusedScanProfile* p);
template<int CKPT_INTERVAL>
void run_fusedscan_bwd_ckpt_tuned(FusedScanProfile* p);

template<int CKPT_INTERVAL>
void run_fusedscan_fwd_vec32_ckpt_tuned(FusedScanProfile* p) {
#ifdef PRECISION_FLOAT
    run_fusedscan_fwd_ckpt_tuned<CKPT_INTERVAL>(p);
#else
    if ((p->H & 1) == 0) {
        int H2 = p->H >> 1;
        mingru_scan_forward_ckpt_tuned_vec32<CKPT_INTERVAL><<<grid_size(p->B * H2), BLOCK_SIZE>>>(p->scan);
    } else {
        run_fusedscan_fwd_ckpt_tuned<CKPT_INTERVAL>(p);
    }
#endif
}

template<int CKPT_INTERVAL>
void run_fusedscan_bwd_vec32_ckpt_tuned(FusedScanProfile* p) {
#ifdef PRECISION_FLOAT
    run_fusedscan_bwd_ckpt_tuned<CKPT_INTERVAL>(p);
#else
    if ((p->H & 1) == 0) {
        int H2 = p->H >> 1;
        mingru_scan_backward_ckpt_tuned_vec32<CKPT_INTERVAL><<<grid_size(p->B * H2), BLOCK_SIZE>>>(
            p->scan, p->grad_out.data, p->grad_next_state.data);
    } else {
        run_fusedscan_bwd_ckpt_tuned<CKPT_INTERVAL>(p);
    }
#endif
}

template<int CKPT_INTERVAL>
void run_fusedscan_fwd_vec64_ckpt_tuned(FusedScanProfile* p) {
    if ((p->H % MINGRU_SCAN_VEC64_WIDTH) == 0) {
        int HW = p->H / MINGRU_SCAN_VEC64_WIDTH;
        mingru_scan_forward_ckpt_tuned_vec64<CKPT_INTERVAL><<<grid_size(p->B * HW), BLOCK_SIZE>>>(p->scan);
    } else {
        run_fusedscan_fwd_ckpt_tuned<CKPT_INTERVAL>(p);
    }
}

template<int CKPT_INTERVAL>
void run_fusedscan_bwd_vec64_ckpt_tuned(FusedScanProfile* p) {
    if ((p->H % MINGRU_SCAN_VEC64_WIDTH) == 0) {
        int HW = p->H / MINGRU_SCAN_VEC64_WIDTH;
        mingru_scan_backward_ckpt_tuned_vec64<CKPT_INTERVAL><<<grid_size(p->B * HW), BLOCK_SIZE>>>(
            p->scan, p->grad_out.data, p->grad_next_state.data);
    } else {
        run_fusedscan_bwd_ckpt_tuned<CKPT_INTERVAL>(p);
    }
}

template<int CKPT_INTERVAL>
void run_fusedscan_fwd_vec128_ckpt_tuned(FusedScanProfile* p) {
    if ((p->H % MINGRU_SCAN_VEC128_WIDTH) == 0) {
        int HW = p->H / MINGRU_SCAN_VEC128_WIDTH;
        mingru_scan_forward_ckpt_tuned_vec128<CKPT_INTERVAL><<<grid_size(p->B * HW), BLOCK_SIZE>>>(p->scan);
    } else {
        run_fusedscan_fwd_ckpt_tuned<CKPT_INTERVAL>(p);
    }
}

template<int CKPT_INTERVAL>
void run_fusedscan_bwd_vec128_ckpt_tuned(FusedScanProfile* p) {
    if ((p->H % MINGRU_SCAN_VEC128_WIDTH) == 0) {
        int HW = p->H / MINGRU_SCAN_VEC128_WIDTH;
        mingru_scan_backward_ckpt_tuned_vec128<CKPT_INTERVAL><<<grid_size(p->B * HW), BLOCK_SIZE>>>(
            p->scan, p->grad_out.data, p->grad_next_state.data);
    } else {
        run_fusedscan_bwd_ckpt_tuned<CKPT_INTERVAL>(p);
    }
}

void run_fusedscan_fwd_vec32_ckpt4(FusedScanProfile* p) {
    run_fusedscan_fwd_vec32_ckpt_tuned<4>(p);
}

void run_fusedscan_bwd_vec32_ckpt4(FusedScanProfile* p) {
    run_fusedscan_bwd_vec32_ckpt_tuned<4>(p);
}

void run_fusedscan_fwd_vec64_ckpt4(FusedScanProfile* p) {
    run_fusedscan_fwd_vec64_ckpt_tuned<4>(p);
}

void run_fusedscan_bwd_vec64_ckpt4(FusedScanProfile* p) {
    run_fusedscan_bwd_vec64_ckpt_tuned<4>(p);
}

void run_fusedscan_fwd_vec128_ckpt4(FusedScanProfile* p) {
    run_fusedscan_fwd_vec128_ckpt_tuned<4>(p);
}

void run_fusedscan_bwd_vec128_ckpt4(FusedScanProfile* p) {
    run_fusedscan_bwd_vec128_ckpt_tuned<4>(p);
}

template<int CKPT_INTERVAL>
void run_fusedscan_fwd_ckpt_tuned(FusedScanProfile* p) {
    mingru_scan_forward_ckpt_tuned<CKPT_INTERVAL><<<grid_size(p->B * p->H), BLOCK_SIZE>>>(p->scan);
}

template<int CKPT_INTERVAL>
void run_fusedscan_bwd_ckpt_tuned(FusedScanProfile* p) {
    mingru_scan_backward_ckpt_tuned<CKPT_INTERVAL><<<grid_size(p->B * p->H), BLOCK_SIZE>>>(
        p->scan, p->grad_out.data, p->grad_next_state.data);
}

template <typename KernelT>
void print_kernel_attrs(const char* name, KernelT kernel) {
    cudaFuncAttributes attrs = {};
    cudaError_t err = cudaFuncGetAttributes(&attrs, kernel);
    if (err != cudaSuccess) {
        printf("  %-34s unavailable (%s)\n", name, cudaGetErrorString(err));
        cudaGetLastError(); // clear sticky error so profiling can continue
        return;
    }
    printf("  %-34s regs=%3d local=%5zuB smem=%5zuB maxT=%4d bin=%2d ptx=%2d\n",
        name,
        attrs.numRegs,
        (size_t)attrs.localSizeBytes,
        (size_t)attrs.sharedSizeBytes,
        attrs.maxThreadsPerBlock,
        attrs.binaryVersion,
        attrs.ptxVersion);
}

void print_fusedscan_kernel_diagnostics_once() {
    static bool printed = false;
    if (printed) {
        return;
    }
    printed = true;

    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop = {};
    cudaError_t prop_err = cudaGetDeviceProperties(&prop, dev);

    printf("fused_scan diagnostics\n");
    if (prop_err == cudaSuccess) {
        printf("  device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    } else {
        printf("  device: <unavailable> (%s)\n", cudaGetErrorString(prop_err));
        cudaGetLastError();
    }
#ifdef PRECISION_FLOAT
    printf("  precision: float32 (PRECISION_FLOAT)\n");
#else
    printf("  precision: bfloat16 (default)\n");
#endif
    printf("  kernel attributes:\n");
    print_kernel_attrs("log_ckpt4_fwd", mingru_scan_forward);
    print_kernel_attrs("log_ckpt4_bwd", mingru_scan_backward);
#ifdef PRECISION_FLOAT
    print_kernel_attrs("log_vec32_ckpt4_fwd", mingru_scan_forward_ckpt_tuned<4>);
    print_kernel_attrs("log_vec32_ckpt4_bwd", mingru_scan_backward_ckpt_tuned<4>);
#else
    print_kernel_attrs("log_vec32_ckpt4_fwd", mingru_scan_forward_ckpt_tuned_vec32<4>);
    print_kernel_attrs("log_vec32_ckpt4_bwd", mingru_scan_backward_ckpt_tuned_vec32<4>);
#endif
    print_kernel_attrs("log_vec64_ckpt4_fwd", mingru_scan_forward_ckpt_tuned_vec64<4>);
    print_kernel_attrs("log_vec64_ckpt4_bwd", mingru_scan_backward_ckpt_tuned_vec64<4>);
    print_kernel_attrs("log_vec128_ckpt4_fwd", mingru_scan_forward_ckpt_tuned_vec128<4>);
    print_kernel_attrs("log_vec128_ckpt4_bwd", mingru_scan_backward_ckpt_tuned_vec128<4>);
    printf("\n");
}

void copy_fusedscan_inputs(FusedScanProfile* dst, FusedScanProfile* src) {
    int N_combined = src->B * src->T * 3 * src->H;
    int N_state = src->B * src->H;
    int N_out = src->B * src->T * src->H;
    cudaMemcpy(dst->scan.combined_ptr, src->scan.combined_ptr,
        N_combined * sizeof(precision_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dst->scan.state_ptr, src->scan.state_ptr,
        N_state * sizeof(precision_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dst->scan.input_ptr, src->scan.input_ptr,
        N_out * sizeof(precision_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dst->grad_out.data, src->grad_out.data,
        N_out * sizeof(precision_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dst->grad_next_state.data, src->grad_next_state.data,
        N_state * sizeof(precision_t), cudaMemcpyDeviceToDevice);
}

struct CompareStats {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    float mean_abs = 0.0f;
    float mean_rel = 0.0f;
    long long mismatch_count = 0;
    long long total_count = 0;
};

CompareStats compare_precision_tensors(
        const PrecisionTensor& a, const PrecisionTensor& b,
        float abs_tol, float rel_tol) {
    long n = numel(a.shape);
    precision_t* ah = (precision_t*)malloc(n * sizeof(precision_t));
    precision_t* bh = (precision_t*)malloc(n * sizeof(precision_t));
    cudaMemcpy(ah, a.data, n * sizeof(precision_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(bh, b.data, n * sizeof(precision_t), cudaMemcpyDeviceToHost);

    CompareStats stats;
    stats.total_count = (long long)n;
    double sum_abs = 0.0;
    double sum_rel = 0.0;
    for (long i = 0; i < n; i++) {
        float av = to_float(ah[i]);
        float bv = to_float(bh[i]);
        float abs_err = fabsf(av - bv);
        float denom = fmaxf(fabsf(av), fabsf(bv));
        float rel_err = abs_err / fmaxf(denom, 1e-6f);
        sum_abs += (double)abs_err;
        sum_rel += (double)rel_err;
        stats.max_abs = fmaxf(stats.max_abs, abs_err);
        stats.max_rel = fmaxf(stats.max_rel, rel_err);
        if (abs_err > abs_tol && rel_err > rel_tol) {
            stats.mismatch_count++;
        }
    }
    if (n > 0) {
        stats.mean_abs = (float)(sum_abs / (double)n);
        stats.mean_rel = (float)(sum_rel / (double)n);
    }

    free(ah);
    free(bh);
    return stats;
}

bool print_compare_result(const char* name, CompareStats stats, float abs_tol, float rel_tol) {
    bool ok = stats.max_abs <= abs_tol || stats.max_rel <= rel_tol;
    printf("    %-16s %s  max_abs=%9.4g  max_rel=%9.4g  avg_abs=%9.4g  avg_rel=%9.4g  mismatches=%lld/%lld\n",
        name, ok ? "PASS" : "FAIL",
        stats.max_abs, stats.max_rel, stats.mean_abs, stats.mean_rel,
        stats.mismatch_count, stats.total_count);
    return ok;
}

void profile_fusedscan(int B, int T, int H) {
    printf("fused_scan (N=%d, %dx%dx%d)\n", B*T*H, B, T, H);
    auto* p = create_fusedscan(B, T, H, "fused_scan");
    if (!p) {
        printf("\n");
        return;
    }
    float fwd = profile_kernel((kernel_fn)run_fusedscan_fwd, p);
    print_timing("forward", fwd, B*T);
    float bwd = profile_kernel((kernel_fn)run_fusedscan_bwd, p);
    print_timing("backward", bwd, B*T);
    printf("\n");
    alloc_free(&p->alloc);
    free(p);
}

typedef void (*fusedscan_run_fn)(FusedScanProfile*);

struct FusedScanVariant {
    const char* name;
    fusedscan_run_fn fwd;
    fusedscan_run_fn bwd;
};

static const FusedScanVariant kLogBaseVariant = {
    "base_log_ckpt4",
    run_fusedscan_fwd,
    run_fusedscan_bwd,
};

#ifdef PRECISION_FLOAT
static const FusedScanVariant kLogVectorVariants[] = {
    {"log_vec32", run_fusedscan_fwd_vec32_ckpt4, run_fusedscan_bwd_vec32_ckpt4},
    {"log_vec64", run_fusedscan_fwd_vec64_ckpt4, run_fusedscan_bwd_vec64_ckpt4},
    {"log_vec128", run_fusedscan_fwd_vec128_ckpt4, run_fusedscan_bwd_vec128_ckpt4},
};
#else
static const FusedScanVariant kLogVectorVariants[] = {
    {"log_vec32", run_fusedscan_fwd_vec32_ckpt4, run_fusedscan_bwd_vec32_ckpt4},
    {"log_vec64", run_fusedscan_fwd_vec64_ckpt4, run_fusedscan_bwd_vec64_ckpt4},
    {"log_vec128", run_fusedscan_fwd_vec128_ckpt4, run_fusedscan_bwd_vec128_ckpt4},
};
#endif
static const int kLogVectorVariantCount =
    sizeof(kLogVectorVariants) / sizeof(kLogVectorVariants[0]);

bool check_variant_correctness_case(int B, int T, int H, const FusedScanVariant& variant) {
    printf("  correctness-log %-20s %dx%dx%d\n", variant.name, B, T, H);
    auto* ref = create_fusedscan(B, T, H, "fused_scan_log_base_ref");
    if (!ref) {
        return true;
    }
    auto* test = create_fusedscan(B, T, H, "fused_scan_log_variant");
    if (!test) {
        alloc_free(&ref->alloc);
        free(ref);
        return true;
    }

    copy_fusedscan_inputs(test, ref);
    kLogBaseVariant.fwd(ref);
    variant.fwd(test);
    cudaDeviceSynchronize();

    kLogBaseVariant.bwd(ref);
    variant.bwd(test);
    cudaDeviceSynchronize();

#ifdef PRECISION_FLOAT
    float abs_tol = 2e-4f, rel_tol = 2e-4f;
#else
    float abs_tol = 5e-2f, rel_tol = 5e-2f;
#endif
    bool ok = true;
    ok &= print_compare_result("out",
        compare_precision_tensors(ref->scan.out, test->scan.out, abs_tol, rel_tol), abs_tol, rel_tol);
    ok &= print_compare_result("next_state",
        compare_precision_tensors(ref->scan.next_state, test->scan.next_state, abs_tol, rel_tol), abs_tol, rel_tol);
    ok &= print_compare_result("grad_combined",
        compare_precision_tensors(ref->scan.grad_combined, test->scan.grad_combined, abs_tol, rel_tol), abs_tol, rel_tol);
    ok &= print_compare_result("grad_input",
        compare_precision_tensors(ref->scan.grad_input, test->scan.grad_input, abs_tol, rel_tol), abs_tol, rel_tol);
    ok &= print_compare_result("grad_state",
        compare_precision_tensors(ref->scan.grad_state, test->scan.grad_state, abs_tol, rel_tol), abs_tol, rel_tol);

    alloc_free(&ref->alloc);
    alloc_free(&test->alloc);
    free(ref);
    free(test);
    return ok;
}

bool profile_fusedscan_correctness_suite() {
    printf("fused_scan_correctness suite (log variants vs log_scalar, correctness-only)\n");
    print_fusedscan_kernel_diagnostics_once();
    print_selected_sweep_sizes();
    bool ok = true;
    int nb = gSelectedBCount;
    int nt = gSelectedTCount;
    int nh = gSelectedHCount;
    for (int ib = 0; ib < nb; ib++) {
        for (int it = 0; it < nt; it++) {
            for (int ih = 0; ih < nh; ih++) {
                int B = gSelectedBVals[ib];
                int T = gSelectedTVals[it];
                int H = gSelectedHVals[ih];
                printf("  case %dx%dx%d\n", B, T, H);
                for (int variant_i = 0; variant_i < kLogVectorVariantCount; variant_i++) {
                    ok &= check_variant_correctness_case(B, T, H, kLogVectorVariants[variant_i]);
                }
            }
        }
    }
    printf("fused_scan_correctness suite: %s\n\n", ok ? "PASS" : "FAIL");
    return ok;
}

static const int kCheckpointIntervalsAll[] = {1, 2, 4, 8, 16, 32};

static constexpr int kCheckpointIntervalOptions =
    sizeof(kCheckpointIntervalsAll) / sizeof(kCheckpointIntervalsAll[0]);
static int gSelectedCheckpointVals[kMaxSweepSizes] = {1, 2, 4, 8, 16, 32};
static int gSelectedCheckpointCount = kCheckpointIntervalOptions;

inline void copy_selected_values(int* dst_vals, const int* src_vals, int count) {
    for (int idx = 0; idx < count; idx++) {
        dst_vals[idx] = src_vals[idx];
    }
}

inline void print_selected_values(const char* label, const int* values, int count) {
    printf("  %s:", label);
    for (int idx = 0; idx < count; idx++) {
        printf(" %d", values[idx]);
    }
    printf("\n");
}

void reset_b_size_selection() {
    gSelectedBCount = kSweepDefaultBCount;
    copy_selected_values(gSelectedBVals, kSweepBValsDefault, kSweepDefaultBCount);
}

void reset_t_size_selection() {
    gSelectedTCount = kSweepDefaultTCount;
    copy_selected_values(gSelectedTVals, kSweepTValsDefault, kSweepDefaultTCount);
}

void reset_h_size_selection() {
    gSelectedHCount = kSweepDefaultHCount;
    copy_selected_values(gSelectedHVals, kSweepHValsDefault, kSweepDefaultHCount);
}

void reset_sweep_size_selection() {
    reset_b_size_selection();
    reset_t_size_selection();
    reset_h_size_selection();
}

void reset_checkpoint_interval_selection() {
    gSelectedCheckpointCount = kCheckpointIntervalOptions;
    copy_selected_values(gSelectedCheckpointVals, kCheckpointIntervalsAll, kCheckpointIntervalOptions);
}

void print_selected_checkpoint_intervals() {
    print_selected_values("checkpoint intervals", gSelectedCheckpointVals, gSelectedCheckpointCount);
}

void print_selected_sweep_sizes() {
    print_selected_values("B sizes", gSelectedBVals, gSelectedBCount);
    print_selected_values("T sizes", gSelectedTVals, gSelectedTCount);
    print_selected_values("H sizes", gSelectedHVals, gSelectedHCount);
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("O0")))
#endif
int parse_positive_int_csv(const char* csv, int* out_vals, int out_cap, const char* what) {
    if (!csv || csv[0] == '\0') {
        return 0;
    }

    int out_count = 0;
    const char* p = csv;
    while (*p != '\0') {
        while (*p == ' ' || *p == '\t' || *p == ',') {
            p++;
        }
        if (*p == '\0') {
            break;
        }

        char* parse_end = nullptr;
        long val = strtol(p, &parse_end, 10);
        if (parse_end == p) {
            printf("warning: invalid %s token near '%.16s' (ignored)\n", what, p);
            while (*p != '\0' && *p != ',') {
                p++;
            }
            continue;
        }

        const char* tail = parse_end;
        while (*tail == ' ' || *tail == '\t') {
            tail++;
        }
        if (*tail != '\0' && *tail != ',') {
            printf("warning: invalid %s token near '%.16s' (ignored)\n", what, p);
            p = tail;
            while (*p != '\0' && *p != ',') {
                p++;
            }
            continue;
        }

        if (val <= 0) {
            printf("warning: invalid %s value %ld (ignored)\n", what, val);
        } else {
            bool exists = false;
            for (int out_i = 0; out_i < out_count; out_i++) {
                if (out_vals[out_i] == (int)val) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                if (out_count < out_cap) {
                    out_vals[out_count++] = (int)val;
                } else {
                    printf("warning: too many %s values; max=%d (remaining ignored)\n", what, out_cap);
                    break;
                }
            }
        }

        p = tail;
        if (*p == ',') {
            p++;
        }
    }

    return out_count;
}

typedef void (*reset_selection_fn)();

bool set_selection_from_csv(const char* csv, int* selected_vals, int* selected_count,
        const char* what, reset_selection_fn reset_selection) {
    if (!csv || csv[0] == '\0') {
        reset_selection();
        return true;
    }

    int parsed[kMaxSweepSizes];
    int parsed_count = parse_positive_int_csv(csv, parsed, kMaxSweepSizes, what);
    if (parsed_count == 0) {
        printf("warning: no valid %s parsed from '%s'; using defaults\n", what, csv);
        reset_selection();
        return false;
    }

    *selected_count = parsed_count;
    copy_selected_values(selected_vals, parsed, parsed_count);
    return true;
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("O0")))
#endif
bool set_checkpoint_intervals_from_csv(const char* csv) {
    return set_selection_from_csv(csv, gSelectedCheckpointVals, &gSelectedCheckpointCount,
        "checkpoint intervals", reset_checkpoint_interval_selection);
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("O0")))
#endif
bool set_b_sizes_from_csv(const char* csv) {
    return set_selection_from_csv(csv, gSelectedBVals, &gSelectedBCount,
        "B sizes", reset_b_size_selection);
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("O0")))
#endif
bool set_t_sizes_from_csv(const char* csv) {
    return set_selection_from_csv(csv, gSelectedTVals, &gSelectedTCount,
        "T sizes", reset_t_size_selection);
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("O0")))
#endif
bool set_h_sizes_from_csv(const char* csv) {
    return set_selection_from_csv(csv, gSelectedHVals, &gSelectedHCount,
        "H sizes", reset_h_size_selection);
}

inline int grid_size_for_block(int N, int block_size) {
    return (N + block_size - 1) / block_size;
}

struct FusedScanBlockLaunch {
    FusedScanProfile* p;
    int block_size;
};

template<int CKPT_INTERVAL>
void run_fusedscan_fwd_ckpt_tuned_block(FusedScanBlockLaunch* args) {
    FusedScanProfile* p = args->p;
    int bs = args->block_size;
    mingru_scan_forward_ckpt_tuned<CKPT_INTERVAL><<<grid_size_for_block(p->B * p->H, bs), bs>>>(p->scan);
}

template<int CKPT_INTERVAL>
void run_fusedscan_bwd_ckpt_tuned_block(FusedScanBlockLaunch* args) {
    FusedScanProfile* p = args->p;
    int bs = args->block_size;
    mingru_scan_backward_ckpt_tuned<CKPT_INTERVAL><<<grid_size_for_block(p->B * p->H, bs), bs>>>(
        p->scan, p->grad_out.data, p->grad_next_state.data);
}

template<int CKPT_INTERVAL>
void run_fusedscan_fwd_vec32_ckpt_tuned_block(FusedScanBlockLaunch* args) {
#ifdef PRECISION_FLOAT
    run_fusedscan_fwd_ckpt_tuned_block<CKPT_INTERVAL>(args);
#else
    FusedScanProfile* p = args->p;
    int bs = args->block_size;
    if ((p->H & 1) == 0) {
        int H2 = p->H >> 1;
        mingru_scan_forward_ckpt_tuned_vec32<CKPT_INTERVAL><<<grid_size_for_block(p->B * H2, bs), bs>>>(p->scan);
    } else {
        run_fusedscan_fwd_ckpt_tuned_block<CKPT_INTERVAL>(args);
    }
#endif
}

template<int CKPT_INTERVAL>
void run_fusedscan_bwd_vec32_ckpt_tuned_block(FusedScanBlockLaunch* args) {
#ifdef PRECISION_FLOAT
    run_fusedscan_bwd_ckpt_tuned_block<CKPT_INTERVAL>(args);
#else
    FusedScanProfile* p = args->p;
    int bs = args->block_size;
    if ((p->H & 1) == 0) {
        int H2 = p->H >> 1;
        mingru_scan_backward_ckpt_tuned_vec32<CKPT_INTERVAL><<<grid_size_for_block(p->B * H2, bs), bs>>>(
            p->scan, p->grad_out.data, p->grad_next_state.data);
    } else {
        run_fusedscan_bwd_ckpt_tuned_block<CKPT_INTERVAL>(args);
    }
#endif
}

template<int CKPT_INTERVAL>
void run_fusedscan_fwd_vec64_ckpt_tuned_block(FusedScanBlockLaunch* args) {
    FusedScanProfile* p = args->p;
    int bs = args->block_size;
    if ((p->H % MINGRU_SCAN_VEC64_WIDTH) == 0) {
        int HW = p->H / MINGRU_SCAN_VEC64_WIDTH;
        mingru_scan_forward_ckpt_tuned_vec64<CKPT_INTERVAL><<<grid_size_for_block(p->B * HW, bs), bs>>>(p->scan);
    } else {
        run_fusedscan_fwd_ckpt_tuned_block<CKPT_INTERVAL>(args);
    }
}

template<int CKPT_INTERVAL>
void run_fusedscan_bwd_vec64_ckpt_tuned_block(FusedScanBlockLaunch* args) {
    FusedScanProfile* p = args->p;
    int bs = args->block_size;
    if ((p->H % MINGRU_SCAN_VEC64_WIDTH) == 0) {
        int HW = p->H / MINGRU_SCAN_VEC64_WIDTH;
        mingru_scan_backward_ckpt_tuned_vec64<CKPT_INTERVAL><<<grid_size_for_block(p->B * HW, bs), bs>>>(
            p->scan, p->grad_out.data, p->grad_next_state.data);
    } else {
        run_fusedscan_bwd_ckpt_tuned_block<CKPT_INTERVAL>(args);
    }
}

template<int CKPT_INTERVAL>
void run_fusedscan_fwd_vec128_ckpt_tuned_block(FusedScanBlockLaunch* args) {
    FusedScanProfile* p = args->p;
    int bs = args->block_size;
    if ((p->H % MINGRU_SCAN_VEC128_WIDTH) == 0) {
        int HW = p->H / MINGRU_SCAN_VEC128_WIDTH;
        mingru_scan_forward_ckpt_tuned_vec128<CKPT_INTERVAL><<<grid_size_for_block(p->B * HW, bs), bs>>>(p->scan);
    } else {
        run_fusedscan_fwd_ckpt_tuned_block<CKPT_INTERVAL>(args);
    }
}

template<int CKPT_INTERVAL>
void run_fusedscan_bwd_vec128_ckpt_tuned_block(FusedScanBlockLaunch* args) {
    FusedScanProfile* p = args->p;
    int bs = args->block_size;
    if ((p->H % MINGRU_SCAN_VEC128_WIDTH) == 0) {
        int HW = p->H / MINGRU_SCAN_VEC128_WIDTH;
        mingru_scan_backward_ckpt_tuned_vec128<CKPT_INTERVAL><<<grid_size_for_block(p->B * HW, bs), bs>>>(
            p->scan, p->grad_out.data, p->grad_next_state.data);
    } else {
        run_fusedscan_bwd_ckpt_tuned_block<CKPT_INTERVAL>(args);
    }
}

template<int CKPT_INTERVAL>
float profile_fusedscan_ckpt_tuned_speed_block(FusedScanProfile* p, int N, int block_size,
        float* fwd_out = nullptr, float* bwd_out = nullptr) {
    (void)N;
    FusedScanBlockLaunch args = {p, block_size};
    float fwd = profile_kernel((kernel_fn)run_fusedscan_fwd_ckpt_tuned_block<CKPT_INTERVAL>, &args);
    float bwd = profile_kernel((kernel_fn)run_fusedscan_bwd_ckpt_tuned_block<CKPT_INTERVAL>, &args);
    if (fwd_out) {
        *fwd_out = fwd;
    }
    if (bwd_out) {
        *bwd_out = bwd;
    }
    return fwd + bwd;
}

template<int CKPT_INTERVAL>
float profile_fusedscan_vec32_ckpt_speed_block(FusedScanProfile* p, int N, int block_size,
        float* fwd_out = nullptr, float* bwd_out = nullptr) {
    (void)N;
    FusedScanBlockLaunch args = {p, block_size};
    float fwd = profile_kernel((kernel_fn)run_fusedscan_fwd_vec32_ckpt_tuned_block<CKPT_INTERVAL>, &args);
    float bwd = profile_kernel((kernel_fn)run_fusedscan_bwd_vec32_ckpt_tuned_block<CKPT_INTERVAL>, &args);
    if (fwd_out) {
        *fwd_out = fwd;
    }
    if (bwd_out) {
        *bwd_out = bwd;
    }
    return fwd + bwd;
}

template<int CKPT_INTERVAL>
float profile_fusedscan_vec64_ckpt_speed_block(FusedScanProfile* p, int N, int block_size,
        float* fwd_out = nullptr, float* bwd_out = nullptr) {
    (void)N;
    FusedScanBlockLaunch args = {p, block_size};
    float fwd = profile_kernel((kernel_fn)run_fusedscan_fwd_vec64_ckpt_tuned_block<CKPT_INTERVAL>, &args);
    float bwd = profile_kernel((kernel_fn)run_fusedscan_bwd_vec64_ckpt_tuned_block<CKPT_INTERVAL>, &args);
    if (fwd_out) {
        *fwd_out = fwd;
    }
    if (bwd_out) {
        *bwd_out = bwd;
    }
    return fwd + bwd;
}

template<int CKPT_INTERVAL>
float profile_fusedscan_vec128_ckpt_speed_block(FusedScanProfile* p, int N, int block_size,
        float* fwd_out = nullptr, float* bwd_out = nullptr) {
    (void)N;
    FusedScanBlockLaunch args = {p, block_size};
    float fwd = profile_kernel((kernel_fn)run_fusedscan_fwd_vec128_ckpt_tuned_block<CKPT_INTERVAL>, &args);
    float bwd = profile_kernel((kernel_fn)run_fusedscan_bwd_vec128_ckpt_tuned_block<CKPT_INTERVAL>, &args);
    if (fwd_out) {
        *fwd_out = fwd;
    }
    if (bwd_out) {
        *bwd_out = bwd;
    }
    return fwd + bwd;
}

inline bool is_supported_checkpoint_interval(int ckpt_interval) {
    switch (ckpt_interval) {
        case 1:
        case 2:
        case 4:
        case 8:
        case 16:
        case 32:
            return true;
        default:
            return false;
    }
}

float dispatch_scalar_ckpt_speed(int ckpt_interval, FusedScanProfile* p, int N, int block_size,
        float* fwd_out = nullptr, float* bwd_out = nullptr) {
    switch (ckpt_interval) {
        case 1: return profile_fusedscan_ckpt_tuned_speed_block<1>(p, N, block_size, fwd_out, bwd_out);
        case 2: return profile_fusedscan_ckpt_tuned_speed_block<2>(p, N, block_size, fwd_out, bwd_out);
        case 4: return profile_fusedscan_ckpt_tuned_speed_block<4>(p, N, block_size, fwd_out, bwd_out);
        case 8: return profile_fusedscan_ckpt_tuned_speed_block<8>(p, N, block_size, fwd_out, bwd_out);
        case 16: return profile_fusedscan_ckpt_tuned_speed_block<16>(p, N, block_size, fwd_out, bwd_out);
        case 32: return profile_fusedscan_ckpt_tuned_speed_block<32>(p, N, block_size, fwd_out, bwd_out);
        default: return -1.0f;
    }
}

float dispatch_vec32_ckpt_speed(int ckpt_interval, FusedScanProfile* p, int N, int block_size,
        float* fwd_out = nullptr, float* bwd_out = nullptr) {
    switch (ckpt_interval) {
        case 1: return profile_fusedscan_vec32_ckpt_speed_block<1>(p, N, block_size, fwd_out, bwd_out);
        case 2: return profile_fusedscan_vec32_ckpt_speed_block<2>(p, N, block_size, fwd_out, bwd_out);
        case 4: return profile_fusedscan_vec32_ckpt_speed_block<4>(p, N, block_size, fwd_out, bwd_out);
        case 8: return profile_fusedscan_vec32_ckpt_speed_block<8>(p, N, block_size, fwd_out, bwd_out);
        case 16: return profile_fusedscan_vec32_ckpt_speed_block<16>(p, N, block_size, fwd_out, bwd_out);
        case 32: return profile_fusedscan_vec32_ckpt_speed_block<32>(p, N, block_size, fwd_out, bwd_out);
        default: return -1.0f;
    }
}

float dispatch_vec64_ckpt_speed(int ckpt_interval, FusedScanProfile* p, int N, int block_size,
        float* fwd_out = nullptr, float* bwd_out = nullptr) {
    switch (ckpt_interval) {
        case 1: return profile_fusedscan_vec64_ckpt_speed_block<1>(p, N, block_size, fwd_out, bwd_out);
        case 2: return profile_fusedscan_vec64_ckpt_speed_block<2>(p, N, block_size, fwd_out, bwd_out);
        case 4: return profile_fusedscan_vec64_ckpt_speed_block<4>(p, N, block_size, fwd_out, bwd_out);
        case 8: return profile_fusedscan_vec64_ckpt_speed_block<8>(p, N, block_size, fwd_out, bwd_out);
        case 16: return profile_fusedscan_vec64_ckpt_speed_block<16>(p, N, block_size, fwd_out, bwd_out);
        case 32: return profile_fusedscan_vec64_ckpt_speed_block<32>(p, N, block_size, fwd_out, bwd_out);
        default: return -1.0f;
    }
}

float dispatch_vec128_ckpt_speed(int ckpt_interval, FusedScanProfile* p, int N, int block_size,
        float* fwd_out = nullptr, float* bwd_out = nullptr) {
    switch (ckpt_interval) {
        case 1: return profile_fusedscan_vec128_ckpt_speed_block<1>(p, N, block_size, fwd_out, bwd_out);
        case 2: return profile_fusedscan_vec128_ckpt_speed_block<2>(p, N, block_size, fwd_out, bwd_out);
        case 4: return profile_fusedscan_vec128_ckpt_speed_block<4>(p, N, block_size, fwd_out, bwd_out);
        case 8: return profile_fusedscan_vec128_ckpt_speed_block<8>(p, N, block_size, fwd_out, bwd_out);
        case 16: return profile_fusedscan_vec128_ckpt_speed_block<16>(p, N, block_size, fwd_out, bwd_out);
        case 32: return profile_fusedscan_vec128_ckpt_speed_block<32>(p, N, block_size, fwd_out, bwd_out);
        default: return -1.0f;
    }
}

template <typename KernelT>
int kernel_max_threads_per_block(KernelT kernel) {
    cudaFuncAttributes attrs = {};
    cudaError_t err = cudaFuncGetAttributes(&attrs, kernel);
    if (err != cudaSuccess) {
        cudaGetLastError(); // clear sticky runtime error so profiling can continue
        return 0;
    }
    return attrs.maxThreadsPerBlock;
}

template<int CKPT_INTERVAL>
int max_block_scalar_ckpt(const FusedScanProfile* /*p*/) {
    int fwd = kernel_max_threads_per_block(mingru_scan_forward_ckpt_tuned<CKPT_INTERVAL>);
    int bwd = kernel_max_threads_per_block(mingru_scan_backward_ckpt_tuned<CKPT_INTERVAL>);
    if (fwd <= 0 || bwd <= 0) {
        return 0;
    }
    return std::min(fwd, bwd);
}

template<int CKPT_INTERVAL>
int max_block_vec32_ckpt(const FusedScanProfile* p) {
#ifdef PRECISION_FLOAT
    return max_block_scalar_ckpt<CKPT_INTERVAL>(p);
#else
    if ((p->H & 1) == 0) {
        int fwd = kernel_max_threads_per_block(mingru_scan_forward_ckpt_tuned_vec32<CKPT_INTERVAL>);
        int bwd = kernel_max_threads_per_block(mingru_scan_backward_ckpt_tuned_vec32<CKPT_INTERVAL>);
        if (fwd <= 0 || bwd <= 0) {
            return 0;
        }
        return std::min(fwd, bwd);
    }
    return max_block_scalar_ckpt<CKPT_INTERVAL>(p);
#endif
}

template<int CKPT_INTERVAL>
int max_block_vec64_ckpt(const FusedScanProfile* p) {
    if ((p->H % MINGRU_SCAN_VEC64_WIDTH) == 0) {
        int fwd = kernel_max_threads_per_block(mingru_scan_forward_ckpt_tuned_vec64<CKPT_INTERVAL>);
        int bwd = kernel_max_threads_per_block(mingru_scan_backward_ckpt_tuned_vec64<CKPT_INTERVAL>);
        if (fwd <= 0 || bwd <= 0) {
            return 0;
        }
        return std::min(fwd, bwd);
    }
    return max_block_scalar_ckpt<CKPT_INTERVAL>(p);
}

template<int CKPT_INTERVAL>
int max_block_vec128_ckpt(const FusedScanProfile* p) {
    if ((p->H % MINGRU_SCAN_VEC128_WIDTH) == 0) {
        int fwd = kernel_max_threads_per_block(mingru_scan_forward_ckpt_tuned_vec128<CKPT_INTERVAL>);
        int bwd = kernel_max_threads_per_block(mingru_scan_backward_ckpt_tuned_vec128<CKPT_INTERVAL>);
        if (fwd <= 0 || bwd <= 0) {
            return 0;
        }
        return std::min(fwd, bwd);
    }
    return max_block_scalar_ckpt<CKPT_INTERVAL>(p);
}

int dispatch_scalar_ckpt_max_block(int ckpt_interval, const FusedScanProfile* p) {
    switch (ckpt_interval) {
        case 1: return max_block_scalar_ckpt<1>(p);
        case 2: return max_block_scalar_ckpt<2>(p);
        case 4: return max_block_scalar_ckpt<4>(p);
        case 8: return max_block_scalar_ckpt<8>(p);
        case 16: return max_block_scalar_ckpt<16>(p);
        case 32: return max_block_scalar_ckpt<32>(p);
        default: return 0;
    }
}

int dispatch_vec32_ckpt_max_block(int ckpt_interval, const FusedScanProfile* p) {
    switch (ckpt_interval) {
        case 1: return max_block_vec32_ckpt<1>(p);
        case 2: return max_block_vec32_ckpt<2>(p);
        case 4: return max_block_vec32_ckpt<4>(p);
        case 8: return max_block_vec32_ckpt<8>(p);
        case 16: return max_block_vec32_ckpt<16>(p);
        case 32: return max_block_vec32_ckpt<32>(p);
        default: return 0;
    }
}

int dispatch_vec64_ckpt_max_block(int ckpt_interval, const FusedScanProfile* p) {
    switch (ckpt_interval) {
        case 1: return max_block_vec64_ckpt<1>(p);
        case 2: return max_block_vec64_ckpt<2>(p);
        case 4: return max_block_vec64_ckpt<4>(p);
        case 8: return max_block_vec64_ckpt<8>(p);
        case 16: return max_block_vec64_ckpt<16>(p);
        case 32: return max_block_vec64_ckpt<32>(p);
        default: return 0;
    }
}

int dispatch_vec128_ckpt_max_block(int ckpt_interval, const FusedScanProfile* p) {
    switch (ckpt_interval) {
        case 1: return max_block_vec128_ckpt<1>(p);
        case 2: return max_block_vec128_ckpt<2>(p);
        case 4: return max_block_vec128_ckpt<4>(p);
        case 8: return max_block_vec128_ckpt<8>(p);
        case 16: return max_block_vec128_ckpt<16>(p);
        case 32: return max_block_vec128_ckpt<32>(p);
        default: return 0;
    }
}

template<int CKPT_INTERVAL>
float profile_fusedscan_fwd_variant_speed_block(
        FusedScanProfile* p, int block_size, MingruScanVariant variant) {
    FusedScanBlockLaunch args = {p, block_size};
    switch (variant) {
        case MingruScanVariant::kVec128:
            return profile_kernel((kernel_fn)run_fusedscan_fwd_vec128_ckpt_tuned_block<CKPT_INTERVAL>, &args);
        case MingruScanVariant::kVec64:
            return profile_kernel((kernel_fn)run_fusedscan_fwd_vec64_ckpt_tuned_block<CKPT_INTERVAL>, &args);
        case MingruScanVariant::kVec32:
            return profile_kernel((kernel_fn)run_fusedscan_fwd_vec32_ckpt_tuned_block<CKPT_INTERVAL>, &args);
        case MingruScanVariant::kScalar:
        default:
            return profile_kernel((kernel_fn)run_fusedscan_fwd_ckpt_tuned_block<CKPT_INTERVAL>, &args);
    }
}

template<int CKPT_INTERVAL>
float profile_fusedscan_bwd_variant_speed_block(
        FusedScanProfile* p, int block_size, MingruScanVariant variant) {
    FusedScanBlockLaunch args = {p, block_size};
    switch (variant) {
        case MingruScanVariant::kVec128:
            return profile_kernel((kernel_fn)run_fusedscan_bwd_vec128_ckpt_tuned_block<CKPT_INTERVAL>, &args);
        case MingruScanVariant::kVec64:
            return profile_kernel((kernel_fn)run_fusedscan_bwd_vec64_ckpt_tuned_block<CKPT_INTERVAL>, &args);
        case MingruScanVariant::kVec32:
            return profile_kernel((kernel_fn)run_fusedscan_bwd_vec32_ckpt_tuned_block<CKPT_INTERVAL>, &args);
        case MingruScanVariant::kScalar:
        default:
            return profile_kernel((kernel_fn)run_fusedscan_bwd_ckpt_tuned_block<CKPT_INTERVAL>, &args);
    }
}

float dispatch_variant_ckpt_fwd_speed(MingruScanVariant variant, int ckpt_interval,
        FusedScanProfile* p, int block_size) {
    switch (ckpt_interval) {
        case 1: return profile_fusedscan_fwd_variant_speed_block<1>(p, block_size, variant);
        case 2: return profile_fusedscan_fwd_variant_speed_block<2>(p, block_size, variant);
        case 4: return profile_fusedscan_fwd_variant_speed_block<4>(p, block_size, variant);
        case 8: return profile_fusedscan_fwd_variant_speed_block<8>(p, block_size, variant);
        case 16: return profile_fusedscan_fwd_variant_speed_block<16>(p, block_size, variant);
        case 32: return profile_fusedscan_fwd_variant_speed_block<32>(p, block_size, variant);
        default: return -1.0f;
    }
}

float dispatch_variant_ckpt_bwd_speed(MingruScanVariant variant, int ckpt_interval,
        FusedScanProfile* p, int block_size) {
    switch (ckpt_interval) {
        case 1: return profile_fusedscan_bwd_variant_speed_block<1>(p, block_size, variant);
        case 2: return profile_fusedscan_bwd_variant_speed_block<2>(p, block_size, variant);
        case 4: return profile_fusedscan_bwd_variant_speed_block<4>(p, block_size, variant);
        case 8: return profile_fusedscan_bwd_variant_speed_block<8>(p, block_size, variant);
        case 16: return profile_fusedscan_bwd_variant_speed_block<16>(p, block_size, variant);
        case 32: return profile_fusedscan_bwd_variant_speed_block<32>(p, block_size, variant);
        default: return -1.0f;
    }
}

int dispatch_variant_ckpt_max_block(MingruScanVariant variant, int ckpt_interval, const FusedScanProfile* p) {
    switch (variant) {
        case MingruScanVariant::kVec128:
            return dispatch_vec128_ckpt_max_block(ckpt_interval, p);
        case MingruScanVariant::kVec64:
            return dispatch_vec64_ckpt_max_block(ckpt_interval, p);
        case MingruScanVariant::kVec32:
            return dispatch_vec32_ckpt_max_block(ckpt_interval, p);
        case MingruScanVariant::kScalar:
        default:
            return dispatch_scalar_ckpt_max_block(ckpt_interval, p);
    }
}

bool profile_fusedscan_selector_policy(FusedScanProfile* p, int H, int block_size,
        MingruScanKernelSelection selection, MingruScanKernelSelection* resolved_out,
        float* fwd_ms_out, float* bwd_ms_out, float* total_ms_out) {
    selection.fwd_variant = mingru_scan_resolve_variant(selection.fwd_variant, H);
    selection.bwd_variant = mingru_scan_resolve_variant(selection.bwd_variant, H);
    if (resolved_out) {
        *resolved_out = selection;
    }
    if (!is_supported_checkpoint_interval(selection.ckpt_interval)) {
        return false;
    }
    int max_fwd_block = dispatch_variant_ckpt_max_block(selection.fwd_variant, selection.ckpt_interval, p);
    int max_bwd_block = dispatch_variant_ckpt_max_block(selection.bwd_variant, selection.ckpt_interval, p);
    if (max_fwd_block <= 0 || max_bwd_block <= 0) {
        return false;
    }
    if (block_size > max_fwd_block || block_size > max_bwd_block) {
        return false;
    }

    float fwd_ms = dispatch_variant_ckpt_fwd_speed(
        selection.fwd_variant, selection.ckpt_interval, p, block_size);
    if (fwd_ms < 0.0f) {
        return false;
    }

    float bwd_ms = dispatch_variant_ckpt_bwd_speed(
        selection.bwd_variant, selection.ckpt_interval, p, block_size);
    if (bwd_ms < 0.0f) {
        return false;
    }

    if (fwd_ms_out) {
        *fwd_ms_out = fwd_ms;
    }
    if (bwd_ms_out) {
        *bwd_ms_out = bwd_ms;
    }
    if (total_ms_out) {
        *total_ms_out = fwd_ms + bwd_ms;
    }
    return true;
}

bool profile_fusedscan_selector_bench_case(int B, int T, int H, float* baseline_total_ms_out,
        float* depth2_total_ms_out) {
    printf("speed fused_scan selector bench (B=%d, T=%d, H=%d, N=%d)\n",
        B, T, H, B * T * H);
    auto* p = create_fusedscan(B, T, H, "fused_scan_selector_bench");
    if (!p) {
        printf("\n");
        return false;
    }

    const int block = kFusedscanBlockSweepBlockSize;
    MingruScanKernelSelection baseline = mingru_scan_select_baseline_policy(B, T, H);
    MingruScanKernelSelection depth2 = mingru_scan_select_depth2_policy(B, T, H);
    MingruScanKernelSelection baseline_resolved = baseline;
    MingruScanKernelSelection depth2_resolved = depth2;
    float baseline_fwd = 0.0f;
    float baseline_bwd = 0.0f;
    float baseline_total = 0.0f;
    float depth2_fwd = 0.0f;
    float depth2_bwd = 0.0f;
    float depth2_total = 0.0f;

    bool baseline_ok = profile_fusedscan_selector_policy(
        p, H, block, baseline, &baseline_resolved, &baseline_fwd, &baseline_bwd, &baseline_total);
    bool depth2_ok = profile_fusedscan_selector_policy(
        p, H, block, depth2, &depth2_resolved, &depth2_fwd, &depth2_bwd, &depth2_total);

    printf("  %-12s %6s %-12s %-12s %10s %10s %10s %12s\n",
        "policy", "ckpt", "fwd", "bwd", "fwd_us", "bwd_us", "total_us", "speedup_x");
    if (baseline_ok) {
        printf("  %-12s %6d %-12s %-12s %10.1f %10.1f %10.1f %12s\n",
            "baseline", baseline_resolved.ckpt_interval,
            mingru_scan_variant_name(baseline_resolved.fwd_variant),
            mingru_scan_variant_name(baseline_resolved.bwd_variant),
            baseline_fwd * 1000.0f, baseline_bwd * 1000.0f, baseline_total * 1000.0f, "-");
    } else {
        printf("  %-12s %6s %-12s %-12s %10s %10s %10s %12s\n",
            "baseline", "-", "-", "-", "-", "-", "-", "-");
    }
    if (depth2_ok && baseline_ok && depth2_total > 0.0f) {
        printf("  %-12s %6d %-12s %-12s %10.1f %10.1f %10.1f %12.3f\n",
            "depth2", depth2_resolved.ckpt_interval,
            mingru_scan_variant_name(depth2_resolved.fwd_variant),
            mingru_scan_variant_name(depth2_resolved.bwd_variant),
            depth2_fwd * 1000.0f, depth2_bwd * 1000.0f, depth2_total * 1000.0f,
            baseline_total / depth2_total);
        printf("  %-12s %6s %-12s %-12s %10s %10s %10.1f %12s\n",
            "delta", "-", "-", "-", "-", "-",
            (depth2_total - baseline_total) * 1000.0f, "depth2-baseline");
    } else if (depth2_ok) {
        printf("  %-12s %6d %-12s %-12s %10.1f %10.1f %10.1f %12s\n",
            "depth2", depth2_resolved.ckpt_interval,
            mingru_scan_variant_name(depth2_resolved.fwd_variant),
            mingru_scan_variant_name(depth2_resolved.bwd_variant),
            depth2_fwd * 1000.0f, depth2_bwd * 1000.0f, depth2_total * 1000.0f, "-");
    } else {
        printf("  %-12s %6s %-12s %-12s %10s %10s %10s %12s\n",
            "depth2", "-", "-", "-", "-", "-", "-", "-");
    }
    printf("\n");

    if (baseline_total_ms_out) {
        *baseline_total_ms_out = baseline_ok ? baseline_total : 0.0f;
    }
    if (depth2_total_ms_out) {
        *depth2_total_ms_out = depth2_ok ? depth2_total : 0.0f;
    }
    bool ok = baseline_ok && depth2_ok;
    alloc_free(&p->alloc);
    free(p);
    return ok;
}

bool profile_fusedscan_selector_bench() {
    printf("fused_scan selector benchmark (baseline vs depth2 policy)\n");
    print_fusedscan_kernel_diagnostics_once();
    printf("  launch block size: %d\n", kFusedscanBlockSweepBlockSize);
    print_selected_sweep_sizes();

    int wins = 0;
    int losses = 0;
    int ties = 0;
    int measured = 0;
    double baseline_sum_ms = 0.0;
    double depth2_sum_ms = 0.0;
    double sum_log_speedup = 0.0;
    const double tie_eps = 1e-4;

    int nb = gSelectedBCount;
    int nt = gSelectedTCount;
    int nh = gSelectedHCount;
    for (int ib = 0; ib < nb; ib++) {
        for (int it = 0; it < nt; it++) {
            for (int ih = 0; ih < nh; ih++) {
                float baseline_total_ms = 0.0f;
                float depth2_total_ms = 0.0f;
                bool ok = profile_fusedscan_selector_bench_case(
                    gSelectedBVals[ib], gSelectedTVals[it], gSelectedHVals[ih],
                    &baseline_total_ms, &depth2_total_ms);
                if (!ok || baseline_total_ms <= 0.0f || depth2_total_ms <= 0.0f) {
                    continue;
                }

                measured++;
                baseline_sum_ms += baseline_total_ms;
                depth2_sum_ms += depth2_total_ms;
                if (depth2_total_ms < baseline_total_ms * (1.0 - tie_eps)) {
                    wins++;
                } else if (depth2_total_ms > baseline_total_ms * (1.0 + tie_eps)) {
                    losses++;
                } else {
                    ties++;
                }
                sum_log_speedup += log((double)baseline_total_ms / (double)depth2_total_ms);
            }
        }
    }

    printf("selector benchmark summary\n");
    printf("  measured cases: %d\n", measured);
    printf("  depth2 wins/losses/ties: %d/%d/%d\n", wins, losses, ties);
    if (measured > 0 && depth2_sum_ms > 0.0) {
        double overall_speedup = baseline_sum_ms / depth2_sum_ms;
        double geomean_speedup = exp(sum_log_speedup / measured);
        printf("  total baseline_us: %.1f\n", baseline_sum_ms * 1000.0);
        printf("  total depth2_us:   %.1f\n", depth2_sum_ms * 1000.0);
        printf("  overall speedup:   %.4fx\n", overall_speedup);
        printf("  geomean speedup:   %.4fx\n", geomean_speedup);
    }
    printf("\n");
    return measured > 0;
}

void profile_fusedscan_sweep_speed_case(int B, int T, int H) {
    printf("speed fused_scan sweep (B=%d, T=%d, H=%d, N=%d)\n",
        B, T, H, B*T*H);
    int N = B * T;
    auto* p = create_fusedscan(B, T, H, "fused_scan_sweep");
    if (!p) {
        printf("\n");
        return;
    }

    const int intervals = gSelectedCheckpointCount;
    float overall_best_total = std::numeric_limits<float>::infinity();
    float overall_best_fwd = 0.0f;
    float overall_best_bwd = 0.0f;
    int overall_best_ckpt = -1;
    const char* overall_best_variant = "n/a";
    float overall_best_combo_total = std::numeric_limits<float>::infinity();
    float overall_best_combo_fwd = 0.0f;
    float overall_best_combo_bwd = 0.0f;
    int overall_best_combo_ckpt = -1;
    const char* overall_best_combo_fwd_variant = "n/a";
    const char* overall_best_combo_bwd_variant = "n/a";
    bool overall_any_combo = false;
    bool overall_any_legal = false;
    const int block = kFusedscanBlockSweepBlockSize;

    for (int i = 0; i < intervals; i++) {
        int ckpt_interval = gSelectedCheckpointVals[i];

        float best_total = std::numeric_limits<float>::infinity();
        float best_fwd = 0.0f;
        float best_bwd = 0.0f;
        const char* best_variant = "n/a";
        float best_combo_fwd = std::numeric_limits<float>::infinity();
        float best_combo_bwd = std::numeric_limits<float>::infinity();
        const char* best_combo_fwd_variant = "n/a";
        const char* best_combo_bwd_variant = "n/a";
        bool any_legal = false;
        printf("  --------------------------------------------------------------------------------------------------------\n");
        printf("  ckpt=%d results:\n", ckpt_interval);
        printf("    %-24s %10s %10s %10s\n",
            "variant", "fwd_us", "bwd_us", "total_us");

        if (!is_supported_checkpoint_interval(ckpt_interval)) {
            printf("    %-24s %10s %10s %10s\n", "log_scalar", "-", "-", "-");
            printf("    %-24s %10s %10s %10s\n", "log_vec32", "-", "-", "-");
            printf("    %-24s %10s %10s %10s\n", "log_vec64", "-", "-", "-");
            printf("    %-24s %10s %10s %10s\n", "log_vec128", "-", "-", "-");
            printf("    %-24s %10s %10s %10s\n", "winner: (none)", "-", "-", "-");
            printf("    %-24s %10s %10s %10s\n", "best_combo: (none)", "-", "-", "-");
            continue;
        }

        const char* variant_names[] = {"log_scalar", "log_vec32", "log_vec64", "log_vec128"};
        float (*variant_speed_fns[])(int, FusedScanProfile*, int, int, float*, float*) = {
            dispatch_scalar_ckpt_speed,
            dispatch_vec32_ckpt_speed,
            dispatch_vec64_ckpt_speed,
            dispatch_vec128_ckpt_speed,
        };
        int (*variant_max_block_fns[])(int, const FusedScanProfile*) = {
            dispatch_scalar_ckpt_max_block,
            dispatch_vec32_ckpt_max_block,
            dispatch_vec64_ckpt_max_block,
            dispatch_vec128_ckpt_max_block,
        };
        int variant_max_blocks[] = {
            variant_max_block_fns[0](ckpt_interval, p),
            variant_max_block_fns[1](ckpt_interval, p),
            variant_max_block_fns[2](ckpt_interval, p),
            variant_max_block_fns[3](ckpt_interval, p),
        };
        static constexpr int kVariantCount = sizeof(variant_names) / sizeof(variant_names[0]);

        for (int variant_i = 0; variant_i < kVariantCount; variant_i++) {
            if (variant_max_blocks[variant_i] <= 0 || block > variant_max_blocks[variant_i]) {
                printf("    %-24s %10s %10s %10s\n", variant_names[variant_i], "-", "-", "-");
                continue;
            }

            float fwd = 0.0f;
            float bwd = 0.0f;
            float total = variant_speed_fns[variant_i](ckpt_interval, p, N, block, &fwd, &bwd);
            printf("    %-24s %10.1f %10.1f %10.1f\n",
                variant_names[variant_i], fwd * 1000.0f, bwd * 1000.0f, total * 1000.0f);
            any_legal = true;

            if (fwd < best_combo_fwd) {
                best_combo_fwd = fwd;
                best_combo_fwd_variant = variant_names[variant_i];
            }
            if (bwd < best_combo_bwd) {
                best_combo_bwd = bwd;
                best_combo_bwd_variant = variant_names[variant_i];
            }
            if (total < best_total) {
                best_total = total;
                best_fwd = fwd;
                best_bwd = bwd;
                best_variant = variant_names[variant_i];
            }
            if (total < overall_best_total) {
                overall_best_total = total;
                overall_best_fwd = fwd;
                overall_best_bwd = bwd;
                overall_best_ckpt = ckpt_interval;
                overall_best_variant = variant_names[variant_i];
                overall_any_legal = true;
            }
        }

        if (any_legal) {
            char winner_label[64];
            snprintf(winner_label, sizeof(winner_label), "winner: %s", best_variant);
            printf("    %-24s %10.1f %10.1f %10.1f\n",
                winner_label, best_fwd * 1000.0f, best_bwd * 1000.0f, best_total * 1000.0f);
            float best_combo_total = best_combo_fwd + best_combo_bwd;
            printf("    %-24s %10.1f %10.1f %10.1f\n",
                "best_combo", best_combo_fwd * 1000.0f, best_combo_bwd * 1000.0f, best_combo_total * 1000.0f);
            printf("      (fwd from %s, bwd from %s)\n",
                best_combo_fwd_variant, best_combo_bwd_variant);
            if (best_combo_total < overall_best_combo_total) {
                overall_best_combo_total = best_combo_total;
                overall_best_combo_fwd = best_combo_fwd;
                overall_best_combo_bwd = best_combo_bwd;
                overall_best_combo_ckpt = ckpt_interval;
                overall_best_combo_fwd_variant = best_combo_fwd_variant;
                overall_best_combo_bwd_variant = best_combo_bwd_variant;
                overall_any_combo = true;
            }
        } else {
            printf("    %-24s %10s %10s %10s\n", "winner: (none)", "-", "-", "-");
            printf("    %-24s %10s %10s %10s\n", "best_combo: (none)", "-", "-", "-");
        }
    }

    printf("  --------------------------------------------------------------------------------------------------------\n");
    if (overall_any_legal) {
        printf("  overall winner:\n");
        printf("    %6s %-24s %10s %10s %10s\n",
            "ckpt", "variant", "fwd_us", "bwd_us", "total_us");
        printf("    %6d %-24s %10.1f %10.1f %10.1f\n",
            overall_best_ckpt, overall_best_variant,
            overall_best_fwd * 1000.0f, overall_best_bwd * 1000.0f, overall_best_total * 1000.0f);
    } else {
        printf("  overall winner:\n");
        printf("    %6s %-24s %10s %10s %10s\n",
            "ckpt", "variant", "fwd_us", "bwd_us", "total_us");
        printf("    %6s %-24s %10s %10s %10s\n",
            "-", "(none)", "-", "-", "-");
    }

    if (overall_any_combo) {
        printf("  overall best_combo (single ckpt):\n");
        printf("    %6s %-24s %10s %10s %10s\n",
            "ckpt", "combo", "fwd_us", "bwd_us", "total_us");
        printf("    %6d %-24s %10.1f %10.1f %10.1f\n",
            overall_best_combo_ckpt, "best_combo",
            overall_best_combo_fwd * 1000.0f,
            overall_best_combo_bwd * 1000.0f,
            overall_best_combo_total * 1000.0f);
        printf("      (fwd from %s, bwd from %s)\n",
            overall_best_combo_fwd_variant, overall_best_combo_bwd_variant);
    } else {
        printf("  overall best_combo (single ckpt):\n");
        printf("    %6s %-24s %10s %10s %10s\n",
            "ckpt", "combo", "fwd_us", "bwd_us", "total_us");
        printf("    %6s %-24s %10s %10s %10s\n",
            "-", "(none)", "-", "-", "-");
    }

    printf("\n");
    alloc_free(&p->alloc);
    free(p);
}

bool profile_fusedscan_sweep() {
    printf("fused_scan sweep speed (no correctness checks)\n");
    print_fusedscan_kernel_diagnostics_once();
    print_selected_checkpoint_intervals();
    printf("  launch block size: %d\n", kFusedscanBlockSweepBlockSize);
    print_selected_sweep_sizes();
    int nb = gSelectedBCount;
    int nt = gSelectedTCount;
    int nh = gSelectedHCount;
    for (int ib = 0; ib < nb; ib++) {
        for (int it = 0; it < nt; it++) {
            for (int ih = 0; ih < nh; ih++) {
                profile_fusedscan_sweep_speed_case(gSelectedBVals[ib], gSelectedTVals[it], gSelectedHVals[ih]);
            }
        }
    }
    return true;
}

struct PPOProfile {
    PPOKernelArgs ka;
    PPOGraphArgs ga;
    FloatTensor loss, losses_acc, ppo_partials;
    FloatTensor grad_logits_t, grad_values_t, adv_mean_t, adv_var_t, ent_coef_t;
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

inline bool profile_is(const char* profile, const char* name) {
    return strcmp(profile, name) == 0;
}

inline bool should_run_kernel_profile(const char* profile, bool run_all, const char* name) {
    return run_all || profile_is(profile, "kernels") || profile_is(profile, name);
}

bool is_known_profile_name(const char* profile) {
    static const char* kKnownProfiles[] = {
        "all",
        "kernels",
        "mingrugate",
        "logcoeffsvals",
        "fusedscan",
        "fusedscan_correctness",
        "fusedscan_sweep",
        "fusedscan_selector_bench",
        "samplelogits",
        "ppoloss",
        "im2col",
        "envspeed",
    };
    static constexpr int kKnownProfileCount = sizeof(kKnownProfiles) / sizeof(kKnownProfiles[0]);
    for (int profile_i = 0; profile_i < kKnownProfileCount; profile_i++) {
        if (profile_is(profile, kKnownProfiles[profile_i])) {
            return true;
        }
    }
    return false;
}

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    const char* profile = argv[1];
    const char* ckpt_intervals_csv = nullptr;
    const char* b_sizes_csv = nullptr;
    const char* t_sizes_csv = nullptr;
    const char* h_sizes_csv = nullptr;
    int buffers = BUF, threads = 16, horizon = T_;
    int total_agents = BR * buffers;
    for (int arg_i = 2; arg_i < argc - 1; arg_i++) {
        if (strcmp(argv[arg_i], "--buffers") == 0) buffers = atoi(argv[++arg_i]);
        else if (strcmp(argv[arg_i], "--threads") == 0) threads = atoi(argv[++arg_i]);
        else if (strcmp(argv[arg_i], "--horizon") == 0) horizon = atoi(argv[++arg_i]);
        else if (strcmp(argv[arg_i], "--total-agents") == 0) total_agents = atoi(argv[++arg_i]);
        else if (strcmp(argv[arg_i], "--ckpt-intervals") == 0) ckpt_intervals_csv = argv[++arg_i];
        else if (strcmp(argv[arg_i], "--b-sizes") == 0) b_sizes_csv = argv[++arg_i];
        else if (strcmp(argv[arg_i], "--t-sizes") == 0) t_sizes_csv = argv[++arg_i];
        else if (strcmp(argv[arg_i], "--h-sizes") == 0) h_sizes_csv = argv[++arg_i];
    }
    reset_checkpoint_interval_selection();
    reset_sweep_size_selection();
    if (ckpt_intervals_csv) {
        set_checkpoint_intervals_from_csv(ckpt_intervals_csv);
    }
    if (b_sizes_csv) {
        set_b_sizes_from_csv(b_sizes_csv);
    }
    if (t_sizes_csv) {
        set_t_sizes_from_csv(t_sizes_csv);
    }
    if (h_sizes_csv) {
        set_h_sizes_from_csv(h_sizes_csv);
    }

    warmup_gpu();
    bool run_all = profile_is(profile, "all");
    bool ok = true;

    if (should_run_kernel_profile(profile, run_all, "mingrugate"))
        profile_mingrugate(BR, H_);
    if (should_run_kernel_profile(profile, run_all, "logcoeffsvals"))
        profile_logcoeffs(BT, T_, H_);
    if (should_run_kernel_profile(profile, run_all, "fusedscan"))
        profile_fusedscan(BT, T_, H_);
    if (profile_is(profile, "fusedscan_correctness"))
        ok &= profile_fusedscan_correctness_suite();
    if (profile_is(profile, "fusedscan_sweep"))
        ok &= profile_fusedscan_sweep();
    if (profile_is(profile, "fusedscan_selector_bench"))
        ok &= profile_fusedscan_selector_bench();
    if (should_run_kernel_profile(profile, run_all, "samplelogits"))
        profile_samplelogits(BR, A_);
    if (should_run_kernel_profile(profile, run_all, "ppoloss"))
        profile_ppoloss(BT, T_, A_);
    if (should_run_kernel_profile(profile, run_all, "im2col")) {
        profile_im2col(1024, N3_C1_IC, N3_MAP_H, N3_MAP_W, N3_C1_K, N3_C1_S, N3_C1_OH, N3_C1_OW);
        profile_im2col(1024, N3_C2_IC, N3_C1_OH, N3_C1_OW, N3_C2_K, N3_C2_S, N3_C2_OH, N3_C2_OW);
    }

    if (profile_is(profile, "envspeed") || run_all)
        profile_envspeed(total_agents, buffers, threads, horizon);

    if (!is_known_profile_name(profile)) {
        printf("Unknown profile: %s\n\n", profile);
        print_usage(argv[0]);
        return 1;
    }

    return ok ? 0 : 1;
}
