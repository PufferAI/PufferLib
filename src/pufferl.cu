#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include <cublas_v2.h>
#include <curand.h>
#include <cassert>
#include <cmath>
#include <stdlib.h>
#include <cuda_bf16.h>

#ifndef PUFFERLIB_KERNELS_ONLY
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <nvml.h>
#include <nccl.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <omp.h>
#include <pthread.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "ini.h"
#ifdef PUFFERLIB_BUILD_MAIN
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/ioctl.h>
#include <sys/file.h>
#include <sys/types.h>
#include <sys/wait.h>
#endif
#endif

#define PUF_MAX_DIMS 8

typedef struct {
    float* data;
    int64_t shape[PUF_MAX_DIMS];
} FloatTensor;

typedef struct {
    unsigned char* data;
    int64_t shape[PUF_MAX_DIMS];
} ByteTensor;

typedef struct {
    long* data;
    int64_t shape[PUF_MAX_DIMS];
} LongTensor;

typedef struct {
    int* data;
    int64_t shape[PUF_MAX_DIMS];
} IntTensor;

#ifdef PRECISION_FLOAT
typedef float precision_t;
constexpr bool USE_BF16 = false;
static constexpr cudaDataType_t CUBLAS_PRECISION = CUDA_R_32F;
static constexpr cublasComputeType_t CUBLAS_COMPUTE_PRECISION = CUBLAS_COMPUTE_32F;
#define NCCL_PRECISION ncclFloat
#define to_float(x) (x)
#define from_float(x) (x)
#else
typedef __nv_bfloat16 precision_t;
constexpr bool USE_BF16 = true;
static constexpr cudaDataType_t CUBLAS_PRECISION = CUDA_R_16BF;
static constexpr cublasComputeType_t CUBLAS_COMPUTE_PRECISION = CUBLAS_COMPUTE_32F;
#define NCCL_PRECISION ncclBfloat16
#define to_float(x) __bfloat162float(x)
#define from_float(x) __float2bfloat16(x)
#endif

typedef struct {
    precision_t* data;
    int64_t shape[PUF_MAX_DIMS];
} PrecisionTensor;

__host__ __device__ inline int ndim(const int64_t* shape) {
    int n = 0; while (n < PUF_MAX_DIMS && shape[n] != 0) n++; return n;
}

__host__ __device__ inline int64_t numel(const int64_t* shape) {
    int64_t n = 1; for (int i = 0; i < PUF_MAX_DIMS && shape[i] != 0; i++) n *= shape[i]; return n;
}

inline int64_t batch_size(const int64_t* shape) {
    int n = ndim(shape);
    int64_t b = 1;
    for (int i = 0; i < n - 2; i++) b *= shape[i];
    return b;
}

#define BLOCK_SIZE 256
inline int grid_size(int N) {
    return (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

// Dense row-major GEMM: C(M,N) = alpha * op_a(A) @ op_b(B) + beta * C
// Strides derived from M, N, K assuming tightly packed row-major storage.
static const size_t CUBLAS_WS_BYTES = 32 * 1024 * 1024;

static cublasHandle_t cublas_get_handle() {
    static thread_local cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
        void* ws = nullptr;
        cudaMalloc(&ws, CUBLAS_WS_BYTES);
        cublasSetWorkspace(handle, ws, CUBLAS_WS_BYTES);
    }
    return handle;
}

static inline void cublasGemmExDense(
        cublasOperation_t op_a, cublasOperation_t op_b,
        int M, int N, int K, void* A, void* B, void* C,
        cudaStream_t stream, float alpha = 1.0f, float beta = 0.0f) {
    int lda = (op_a == CUBLAS_OP_N) ? K : M;
    int ldb = (op_b == CUBLAS_OP_N) ? N : K;

    cublasHandle_t handle = cublas_get_handle();
    cublasSetStream(handle, stream);
    cublasGemmEx(handle, op_b, op_a, N, M, K, &alpha,
        B, CUBLAS_PRECISION, ldb, A, CUBLAS_PRECISION, lda, &beta,
        C, CUBLAS_PRECISION, N, CUBLAS_COMPUTE_PRECISION, CUBLAS_GEMM_DEFAULT);
}

// out(...,N) = a(...,K) @ b(N,K)^T  — leading dims folded into M
void puf_mm(PrecisionTensor* a, PrecisionTensor* b, PrecisionTensor* out, cudaStream_t stream) {
    int M = batch_size(a->shape) * a->shape[ndim(a->shape)-2];
    int K = a->shape[ndim(a->shape)-1];
    int N = b->shape[ndim(b->shape)-2];
    cublasGemmExDense(CUBLAS_OP_N, CUBLAS_OP_T, M, N, K,
        a->data, b->data, out->data, stream);
}

// out(M,N) = a(...,M)^T @ b(...,N)  — leading dims folded into K
void puf_mm_tn(PrecisionTensor* a, PrecisionTensor* b, PrecisionTensor* out, cudaStream_t stream) {
    int M = a->shape[ndim(a->shape)-1];
    int K = batch_size(a->shape) * a->shape[ndim(a->shape)-2];
    int N = b->shape[ndim(b->shape)-1];
    cublasGemmExDense(CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
        a->data, b->data, out->data, stream);
}

// out(...,N) = a(...,K) @ b(K,N)  — leading dims folded into M
void puf_mm_nn(PrecisionTensor* a, PrecisionTensor* b, PrecisionTensor* out,
        cudaStream_t stream, float alpha = 1.0f, float beta = 0.0f) {
    int M = batch_size(a->shape) * a->shape[ndim(a->shape)-2];
    int K = a->shape[ndim(a->shape)-1];
    int N = b->shape[ndim(b->shape)-1];
    cublasGemmExDense(CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
        a->data, b->data, out->data, stream, alpha, beta);
}

__global__ void cast(precision_t* __restrict__ dst,
        const float* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(src[idx]);
    }
}

#ifndef PRECISION_FLOAT
__global__ void cast(float* __restrict__ dst,
        const precision_t* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = to_float(src[idx]);
    }
}
#endif

__global__ void cast(precision_t* __restrict__ dst,
        const unsigned char* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float((float)src[idx]);
    }
}

__global__ void cast(unsigned char* __restrict__ dst,
        const precision_t* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = to_float(src[idx]);
    }
}

void puf_copy(PrecisionTensor* dst, const PrecisionTensor* src, cudaStream_t stream) {
    assert(numel(dst->shape) == numel(src->shape) && "puf_copy: size mismatch");
    cudaMemcpyAsync(dst->data, src->data, numel(dst->shape) * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream);
}

__global__ void uniform_scale_kernel(float* data, float bound, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = data[idx] * 2.0f * bound - bound;
}

// Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))
void puf_kaiming_init(PrecisionTensor* dst, float gain, ulong seed, cudaStream_t stream) {
    assert(ndim(dst->shape) == 2);
    long rows = dst->shape[0], cols = dst->shape[1];
    assert(rows > 0 && cols > 0);
    long n = rows * cols;
    float bound = gain / std::sqrt((float)cols);
    float* buf;
    cudaMalloc(&buf, n * sizeof(float));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, buf, n);
    curandDestroyGenerator(gen);
    uniform_scale_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(buf, bound, n);
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(dst->data, buf, n);
    cudaFree(buf);
}

struct AllocEntry {
    void** data_ptr;    // address of the tensor's data field
    int64_t* shape;     // pointer to the tensor's shape array
    int elem_size;      // sizeof element type
};

struct Allocator {
    AllocEntry* regs = nullptr;
    int num_regs = 0;
    void* mem = nullptr;
    long total_elems = 0;
    long total_bytes = 0;
};

static void alloc_register_impl(Allocator* alloc, void** data_ptr, int64_t* shape, int elem_size) {
    alloc->regs = (AllocEntry*)realloc(alloc->regs, (alloc->num_regs + 1) * sizeof(AllocEntry));
    alloc->regs[alloc->num_regs++] = {data_ptr, shape, elem_size};
    int64_t n = numel(shape);
    alloc->total_elems += n;
    alloc->total_bytes = (alloc->total_bytes + 15) & ~15;
    alloc->total_bytes += n * elem_size;
}
void alloc_register(Allocator* a, PrecisionTensor* t) {
    alloc_register_impl(a, (void**)&t->data, t->shape, sizeof(precision_t));
}
void alloc_register(Allocator* a, FloatTensor* t) {
    alloc_register_impl(a, (void**)&t->data, t->shape, sizeof(float));
}
void alloc_register(Allocator* a, LongTensor* t) {
    alloc_register_impl(a, (void**)&t->data, t->shape, sizeof(long));
}
void alloc_register(Allocator* a, IntTensor* t) {
    alloc_register_impl(a, (void**)&t->data, t->shape, sizeof(int));
}

cudaError_t alloc_create(Allocator* alloc) {
    if (alloc->total_bytes == 0) return cudaSuccess;
    cudaError_t err = cudaMalloc(&alloc->mem, alloc->total_bytes);
    if (err != cudaSuccess) return err;
    cudaMemset(alloc->mem, 0, alloc->total_bytes);
    long offset = 0;
    for (int i = 0; i < alloc->num_regs; i++) {
        offset = (offset + 15) & ~15;
        *alloc->regs[i].data_ptr = (char*)alloc->mem + offset;
        offset += numel(alloc->regs[i].shape) * alloc->regs[i].elem_size;
    }
    return cudaSuccess;
}

void alloc_free(Allocator* alloc) {
    if (alloc->mem) { cudaFree(alloc->mem); alloc->mem = nullptr; }
    if (alloc->regs) { free(alloc->regs); alloc->regs = nullptr; }
    alloc->num_regs = 0;
    alloc->total_elems = 0;
    alloc->total_bytes = 0;
}

__device__ __forceinline__ void copy_bytes(
        const char* __restrict__ src, char* __restrict__ dst,
        int src_row, int dst_row, int row_bytes) {
    const char* s = src + (int64_t)src_row * row_bytes;
    char* d = dst + (int64_t)dst_row * row_bytes;
    for (int i = threadIdx.x; i < row_bytes; i += blockDim.x) {
        d[i] = s[i];
    }
}

// Transpose dims 0,1: [A, B, C] -> [B, A, C]. For 2D, pass C=1.
__global__ void transpose_102(precision_t* __restrict__ dst,
        const precision_t* __restrict__ src, int A, int B, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = A * B * C;
    if (idx >= total) {
        return;
    }
    int a = idx / (B * C);
    int rem = idx % (B * C);
    int b = rem / C;
    int c = rem % C;
    dst[b * A * C + a * C + c] = src[idx];
}

__global__ void fill_precision_kernel(precision_t* __restrict__ dst, precision_t val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = val;
    }
}

__global__ void clamp_precision_kernel(precision_t* __restrict__ dst, float lo, float hi, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = to_float(dst[idx]);
        dst[idx] = from_float(fminf(fmaxf(v, lo), hi));
    }
}

#ifndef PRECISION_FLOAT
inline void cast_dispatch(precision_t* dst, const precision_t* src, int n, cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, n * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream);
}
#endif

inline void cast_dispatch(precision_t* dst, const float* src, int n, cudaStream_t stream) {
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(dst, src, n);
}

inline void cast_dispatch(precision_t* dst, const unsigned char* src, int n, cudaStream_t stream) {
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(dst, src, n);
}

void puf_zero(PrecisionTensor* dst, cudaStream_t stream) {
    cudaMemsetAsync(dst->data, 0, numel(dst->shape) * sizeof(precision_t), stream);
}

void puf_zero(FloatTensor* dst, cudaStream_t stream) {
    cudaMemsetAsync(dst->data, 0, numel(dst->shape) * sizeof(float), stream);
}

inline PrecisionTensor* puf_squeeze(PrecisionTensor* t, int dim) {
    int n = ndim(t->shape);
    t->shape[dim] *= t->shape[dim + 1];
    for (int i = dim + 1; i < n - 1; i++) {
        t->shape[i] = t->shape[i + 1];
    }
    t->shape[n - 1] = 0;
    return t;
}

inline FloatTensor* puf_squeeze(FloatTensor* t, int dim) {
    int n = ndim(t->shape);
    t->shape[dim] *= t->shape[dim + 1];
    for (int i = dim + 1; i < n - 1; i++) {
        t->shape[i] = t->shape[i + 1];
    }
    t->shape[n - 1] = 0;
    return t;
}

inline PrecisionTensor* puf_unsqueeze(PrecisionTensor* t, int dim, int64_t d0, int64_t d1) {
    int n = ndim(t->shape);
    assert(n + 1 <= PUF_MAX_DIMS);
    assert(t->shape[dim] == d0 * d1);
    for (int i = n; i > dim; i--) {
        t->shape[i] = t->shape[i - 1];
    }
    t->shape[dim] = d0;
    t->shape[dim + 1] = d1;
    return t;
}

#ifndef PUFFERLIB_KERNELS_ONLY

#define TRAIN_RESULT_MAX_POINTS 64

typedef struct {
    Ini ini;
    Dict env;
} Config;

typedef struct {
    int horizon;
    int total_agents;
    int num_buffers;
    int num_atns;
    int hidden_size;
    int num_layers;
    float lr;
    float min_lr_ratio;
    bool anneal_lr;
    float beta1;
    float beta2;
    float eps;
    int minibatch_size;
    float replay_ratio;
    long total_timesteps;
    float max_grad_norm;
    float clip_coef;
    float vf_clip_coef;
    float vf_coef;
    float ent_coef;
    float min_ent_coef_ratio;
    bool anneal_ent_coef;
    float gamma;
    float gae_lambda;
    float vtrace_rho_clip;
    float vtrace_c_clip;
    float prio_alpha;
    float prio_beta0;
    bool reset_state;
    int cudagraphs;
    bool profile;
    int rank;
    int world_size;
    int gpu_id;
    int num_threads;
    int seed;
} HypersT;

static void puf_config_assert(int ok, const char* fmt, ...) {
    if (ok) {
        return;
    }

    va_list args;
    fprintf(stderr, "config error: ");
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    exit(1);
}

static int puf_config_error(char* out, size_t out_size, const char* fmt, ...) {
    if (out && out_size > 0) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(out, out_size, fmt, args);
        va_end(args);
    }
    return 0;
}

static double puf_config_get(Config* cfg, const char* section, const char* key) {
    Dict* dict = puf_ini_section(&cfg->ini, section, 0);
    DictItem* item = dict_find(dict, key);
    puf_config_assert(item != NULL, "missing key [%s] %s", section, key);
    return item->value;
}

static inline int puf_config_int(Config* cfg, const char* section, const char* key) {
    return (int)puf_config_get(cfg, section, key);
}

static inline long puf_config_long(Config* cfg, const char* section, const char* key) {
    return (long)puf_config_get(cfg, section, key);
}

static inline float puf_config_float(Config* cfg, const char* section, const char* key) {
    return (float)puf_config_get(cfg, section, key);
}

static const char* puf_config_str(Config* cfg, const char* section, const char* key) {
    Dict* dict = puf_ini_section(&cfg->ini, section, 0);
    DictItem* item = dict_find(dict, key);
    puf_config_assert(item != NULL && item->str != NULL,
        "missing string [%s] %s", section, key);
    return item->str;
}

static void puf_config_put(Config* cfg, const char* full_key, const char* raw) {
    const char* split = strrchr(full_key, '.');
    puf_config_assert(split != NULL, "expected section.key, got %s", full_key);

    char section[128];
    char key[PUF_DICT_MAX_KEY];
    snprintf(section, sizeof(section), "%.*s", (int)(split - full_key), full_key);
    snprintf(key, sizeof(key), "%s", split + 1);

    Dict* dict = puf_ini_section(&cfg->ini, section, 0);
    puf_config_assert(dict_find(dict, key) != NULL, "missing key [%s] %s", section, key);
    puf_ini_set(dict, key, raw);
    if (strcmp(section, "env") == 0) {
        puf_config_assert(dict_find(&cfg->env, key) != NULL, "missing env key %s", key);
        puf_ini_set(&cfg->env, key, raw);
    }
}

static inline HypersT puf_config_to_hypers(Config* cfg, int rank,
        int world_size, int gpu_id) {
    HypersT h = {0};
    h.total_agents = puf_config_int(cfg, "vec", "total_agents");
    h.num_buffers = puf_config_int(cfg, "vec", "num_buffers");
    h.num_threads = puf_config_int(cfg, "vec", "num_threads");
    h.horizon = puf_config_int(cfg, "train", "horizon");
    h.hidden_size = puf_config_int(cfg, "policy", "hidden_size");
    h.num_layers = puf_config_int(cfg, "policy", "num_layers");
    h.lr = puf_config_float(cfg, "train", "learning_rate");
    h.min_lr_ratio = puf_config_float(cfg, "train", "min_lr_ratio");
    h.anneal_lr = puf_config_int(cfg, "train", "anneal_lr");
    h.beta1 = puf_config_float(cfg, "train", "beta1");
    h.beta2 = puf_config_float(cfg, "train", "beta2");
    h.eps = puf_config_float(cfg, "train", "eps");
    h.minibatch_size = puf_config_int(cfg, "train", "minibatch_size");
    h.replay_ratio = puf_config_float(cfg, "train", "replay_ratio");
    h.total_timesteps = puf_config_long(cfg, "train", "total_timesteps");
    h.max_grad_norm = puf_config_float(cfg, "train", "max_grad_norm");
    h.clip_coef = puf_config_float(cfg, "train", "clip_coef");
    h.vf_clip_coef = puf_config_float(cfg, "train", "vf_clip_coef");
    h.vf_coef = puf_config_float(cfg, "train", "vf_coef");
    h.ent_coef = puf_config_float(cfg, "train", "ent_coef");
    h.min_ent_coef_ratio = puf_config_float(cfg, "train", "min_ent_coef_ratio");
    h.anneal_ent_coef = puf_config_int(cfg, "train", "anneal_ent_coef");
    h.gamma = puf_config_float(cfg, "train", "gamma");
    h.gae_lambda = puf_config_float(cfg, "train", "gae_lambda");
    h.vtrace_rho_clip = puf_config_float(cfg, "train", "vtrace_rho_clip");
    h.vtrace_c_clip = puf_config_float(cfg, "train", "vtrace_c_clip");
    h.prio_alpha = puf_config_float(cfg, "train", "prio_alpha");
    h.prio_beta0 = puf_config_float(cfg, "train", "prio_beta0");
    h.reset_state = puf_config_int(cfg, "base", "reset_state");
    h.cudagraphs = puf_config_int(cfg, "base", "cudagraphs");
    h.profile = puf_config_int(cfg, "base", "profile");
    h.rank = rank;
    h.world_size = world_size;
    h.gpu_id = gpu_id;
    h.seed = puf_config_int(cfg, "base", "seed");
    return h;
}

static void puf_config_load_env(Config* cfg, const char* env_name,
        int argc, char** argv) {
    puf_ini_load_env(&cfg->ini, env_name, argc, argv);
    dict_clear(&cfg->env);
    dict_copy(&cfg->env, puf_ini_section(&cfg->ini, "env", 0));
}

static inline void puf_config_copy(Config* dst, Config* src) {
    memset(dst, 0, sizeof(*dst));
    if (src->ini.num_sections) {
        dst->ini.sections = (Dict*)calloc((size_t)src->ini.num_sections, sizeof(Dict));
        if (!dst->ini.sections) {
            perror("calloc");
            exit(1);
        }
        dst->ini.num_sections = src->ini.num_sections;
    }
    for (int i = 0; i < src->ini.num_sections; i++) {
        dict_copy(&dst->ini.sections[i], &src->ini.sections[i]);
    }
    dict_copy(&dst->env, &src->env);
}

static void puf_config_free(Config* cfg) {
    puf_ini_free(&cfg->ini);
    dict_clear(&cfg->env);
    memset(cfg, 0, sizeof(*cfg));
}

#ifdef PUFFERLIB_BUILD_MAIN
#include "protein.cu"

typedef struct {
    char section[64];
    char key[64];
} SweepParam;

static float puf_config_sweep_num(Dict* dict, const char* key) {
    const char* raw = dict_get_str(dict, key);
    double value = 0;
    puf_config_assert(puf_ini_parse_val(raw, &value),
        "invalid numeric field [%s] %s = %s", dict->name, key, raw);
    return (float)value;
}

static SpaceType puf_config_sweep_space_type(Dict* dict, int* is_integer) {
    const char* dist = dict_get_str(dict, "distribution");
    *is_integer = 0;
    if (strcmp(dist, "uniform") == 0) {
        return SPACE_LINEAR;
    }
    if (strcmp(dist, "int_uniform") == 0) {
        *is_integer = 1;
        return SPACE_LINEAR;
    }
    if (strcmp(dist, "uniform_pow2") == 0) {
        *is_integer = 1;
        return SPACE_POW2;
    }
    if (strcmp(dist, "log_normal") == 0) {
        return SPACE_LOG;
    }
    if (strcmp(dist, "logit_normal") == 0) {
        return SPACE_LOGIT;
    }

    puf_config_assert(0, "invalid sweep distribution [%s] %s", dict->name, dist);
    return SPACE_LINEAR;
}

static int puf_config_train_valid(Config* cfg, char* err, size_t err_size) {
    int minibatch_size = puf_config_int(cfg, "train", "minibatch_size");
    int horizon = puf_config_int(cfg, "train", "horizon");
    int total_agents = puf_config_int(cfg, "vec", "total_agents");
    int train_gpus = puf_config_int(cfg, "train", "gpus");

    if (train_gpus < 1) {
        return puf_config_error(err, err_size, "train.gpus must be >= 1");
    }
    if (minibatch_size % horizon != 0) {
        return puf_config_error(err, err_size,
            "train.minibatch_size must be divisible by train.horizon");
    }
    if (minibatch_size > (long)horizon * total_agents) {
        return puf_config_error(err, err_size,
            "train.minibatch_size > train.horizon * vec.total_agents");
    }
    return 1;
}

static void puf_config_validate(Config* cfg) {
    char err[256];
    puf_config_assert(puf_config_train_valid(cfg, err, sizeof(err)), "%s", err);
    int train_gpus = puf_config_int(cfg, "train", "gpus");

    int league = puf_config_int(cfg, "sweep", "league");
    const char* metric = puf_config_str(cfg, "sweep", "metric");
    puf_config_assert(league || strcmp(metric, "score") == 0,
        "native sweep currently scores env/score, got env/%s", metric);

    const char* metric_dist = puf_config_str(cfg, "sweep", "metric_distribution");
    puf_config_assert(strcmp(metric_dist, "linear") == 0 ||
            strcmp(metric_dist, "logit") == 0,
        "sweep.metric_distribution must be linear or logit");

    const char* goal = puf_config_str(cfg, "sweep", "goal");
    puf_config_assert(strcmp(goal, "maximize") == 0 ||
            strcmp(goal, "minimize") == 0,
        "sweep.goal must be maximize or minimize");

    int max_runs = puf_config_int(cfg, "sweep", "max_runs");
    int downsample = puf_config_int(cfg, "sweep", "downsample");
    int sweep_gpus = puf_config_int(cfg, "sweep", "gpus");
    puf_config_assert(max_runs >= 1, "sweep.max_runs must be >= 1");
    puf_config_assert(downsample >= 1 && downsample <= TRAIN_RESULT_MAX_POINTS,
        "sweep.downsample must be in [1, %d]", TRAIN_RESULT_MAX_POINTS);
    puf_config_assert(sweep_gpus >= 0, "sweep.gpus must be >= 0");
    puf_config_assert(sweep_gpus == 0 || sweep_gpus >= train_gpus + league,
        "sweep.gpus must be >= train.gpus%s",
        league ? " + 1 for league sweeps" : "");
    puf_config_assert(puf_config_float(cfg, "sweep", "max_suggestion_cost") > 0,
        "sweep.max_suggestion_cost must be > 0");

    float q = puf_config_float(cfg, "sweep", "early_stop_quantile");
    puf_config_assert(q > 0 && q < 1, "sweep.early_stop_quantile must be in (0, 1)");
    puf_config_assert(!league ||
            strcmp(puf_config_str(cfg, "base", "env_name"), "robocode") == 0,
        "league sweep currently requires robocode");

    for (int i = 0; i < cfg->ini.num_sections; i++) {
        Dict* dict = &cfg->ini.sections[i];
        if (strncmp(dict->name, "sweep.", 6) != 0) {
            continue;
        }

        const char* sweep_key = dict->name + 6;
        const char* dot = strrchr(sweep_key, '.');
        puf_config_assert(dot && dot != sweep_key && dot[1],
            "expected section [sweep.<section>.<key>]");

        int is_integer = 0;
        puf_config_sweep_space_type(dict, &is_integer);

        float min_v = puf_config_sweep_num(dict, "min");
        float max_v = puf_config_sweep_num(dict, "max");
        puf_config_assert(max_v > min_v, "[%s] max must be greater than min", dict->name);

        const char* scale = dict_get_str(dict, "scale");
        if (strcmp(scale, "time") == 0) {
            puf_config_assert(min_v > 0 && max_v > 0,
                "[%s] scale=time requires positive min/max", dict->name);
        } else if (strcmp(scale, "auto") != 0) {
            puf_config_sweep_num(dict, "scale");
        }
    }
}

static float puf_config_sweep_scale(Dict* dict, float min_v, float max_v) {
    const char* raw = dict_get_str(dict, "scale");
    if (strcmp(raw, "auto") == 0) {
        return 0.5f;
    }
    if (strcmp(raw, "time") == 0) {
        return 1.0f / (log2f(max_v) - log2f(min_v));
    }
    return puf_config_sweep_num(dict, "scale");
}

static SweepSpace* puf_config_sweep_space(Config* cfg, SweepParam** params_out) {
    SweepParam* params = (SweepParam*)calloc((size_t)cfg->ini.num_sections,
        sizeof(SweepParam));
    int direction = strcmp(puf_config_str(cfg, "sweep", "goal"), "minimize") == 0 ?
        -1 : 1;
    SweepSpace* space = sweep_space_create(cfg->ini.num_sections, -1, direction);
    int n = 0;

    for (int i = 0; i < cfg->ini.num_sections; i++) {
        Dict* dict = &cfg->ini.sections[i];
        if (strncmp(dict->name, "sweep.", 6) != 0) {
            continue;
        }

        const char* sweep_key = dict->name + 6;
        const char* dot = strrchr(sweep_key, '.');
        int section_len = (int)(dot - sweep_key);
        snprintf(params[n].section, sizeof(params[n].section), "%.*s",
            section_len, sweep_key);
        snprintf(params[n].key, sizeof(params[n].key), "%s", dot + 1);

        int is_integer = 0;
        SpaceType type = puf_config_sweep_space_type(dict, &is_integer);
        float min_v = puf_config_sweep_num(dict, "min");
        float max_v = puf_config_sweep_num(dict, "max");
        float scale = puf_config_sweep_scale(dict, min_v, max_v);
        space_init(&space->spaces[n], type, min_v, max_v, scale, is_integer);

        if (strcmp(params[n].section, "train") == 0 &&
                strcmp(params[n].key, "total_timesteps") == 0) {
            space->cost_idx = n;
        }
        n++;
    }

    space->num = n;
    *params_out = params;
    return space;
}

static void puf_config_sweep_apply(Config* cfg, SweepParam* params,
        SweepSpace* space, const float* sample) {
    for (int i = 0; i < space->num; i++) {
        float val = space_unnormalize(&space->spaces[i], sample[i]);
        char buf[64];
        snprintf(buf, sizeof(buf), "%.9g", val);
        char key[256];
        snprintf(key, sizeof(key), "%s.%s", params[i].section, params[i].key);
        puf_config_put(cfg, key, buf);
    }
}
#endif

// Algo needs tensor, allocator, and GEMM helpers; envs can override encoders via ocean.cu.
#include "algo.cu"

#include ENV_HEADER

typedef int atomic_int;

struct PuffeRL;
void pufferl_forward(struct PuffeRL* pufferl, int buf, int t, cudaStream_t stream);
#ifdef PUFFER_CUDA_ENV
void pufferl_rollout_buffer(struct PuffeRL* pufferl, int buf, cudaStream_t stream);
#endif

typedef struct ObsTensor {
    obs_t* data;
    int64_t shape[8];
} ObsTensor;

static inline int atomic_load(const atomic_int* ptr) {
    return __atomic_load_n(ptr, __ATOMIC_SEQ_CST);
}

static inline void atomic_store(atomic_int* ptr, int value) {
    __atomic_store_n(ptr, value, __ATOMIC_SEQ_CST);
}

enum VecProfileIdx {
    VEC_GPU = 0,
    VEC_ENV_STEP,
    NUM_VEC_PROF,
};

#define OMP_WAITING 5
#define OMP_RUNNING 6

typedef struct VecWorker VecWorker;

typedef struct VecEnv {
    Env* envs;
    int size;
    int total_agents;
    int buffers;
    int agents_per_buffer;
    int* buffer_env_starts;
    int* buffer_env_counts;
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    obs_t* gpu_observations;
    float* gpu_actions;
    float* gpu_rewards;
    float* gpu_terminals;
    unsigned char* gpu_action_mask;
    cudaStream_t* streams;
    atomic_int* buffer_states;
    atomic_int shutdown;
    pthread_t* threads;
    VecWorker* workers;
    float* accum;
    int num_workers;
    int action_mask_size;
    int num_banks;
    int* bank_layout;
#ifdef PUFFER_CUDA_ENV
    // fbr: Pointer-free environment state remains resident on the device.
    void* gpu_env_state;
#endif
} VecEnv;

#ifdef PUFFER_CUDA_ENV
// fbr: CUDA environment callbacks enqueue device-only work on the rollout stream.
static void puf_cuda_env_init(VecEnv* vec, cudaStream_t stream);
static void puf_cuda_env_step(VecEnv* vec, int env_start, int env_count,
    int agent_start, int agent_count, const precision_t* actions,
    float* observations, float* rewards, float* terminals,
    cudaStream_t stream);
static void puf_cuda_env_step_direct(VecEnv* vec, int env_start, int env_count,
    int agent_start, int agent_count, const precision_t* actions,
    precision_t* observations, precision_t* rewards, precision_t* terminals,
    cudaStream_t stream);
static void puf_cuda_env_sync_logs(VecEnv* vec, int clear, cudaStream_t stream);
static void puf_cuda_env_close(VecEnv* vec);
#ifndef PUFFER_CUDA_ENV_IMPL
#error "PUFFER_CUDA_ENV requires PUFFER_CUDA_ENV_IMPL to name its adapter implementation"
#endif
#include PUFFER_CUDA_ENV_IMPL
#endif

struct VecWorker {
    VecEnv* vec;
    int buf;
    int horizon;
    struct PuffeRL* pufferl;
};

static void* vec_thread_main(void* arg) {
    VecWorker* worker_arg = (VecWorker*)arg;
    VecEnv* vec = worker_arg->vec;
    int buf = worker_arg->buf;
#ifndef PUFFER_CUDA_ENV
    int horizon = worker_arg->horizon;
#endif
    struct PuffeRL* pufferl = worker_arg->pufferl;

#ifndef PUFFER_CUDA_ENV
    int agents_per_buffer = vec->agents_per_buffer;
    int agent_start = buf * agents_per_buffer;
    int env_start = vec->buffer_env_starts[buf];
    int env_count = vec->buffer_env_counts[buf];
    Env* envs = vec->envs;
#endif

    while (true) {
        while (atomic_load(&vec->buffer_states[buf]) != OMP_RUNNING) {
            if (atomic_load(&vec->shutdown)) {
                return NULL;
            }
        }
        cudaStream_t stream = vec->streams[buf];

        float* my_accum = &vec->accum[buf * NUM_VEC_PROF];
        struct timespec t0, t1;

#ifdef PUFFER_CUDA_ENV
        clock_gettime(CLOCK_MONOTONIC, &t0);
        pufferl_rollout_buffer(pufferl, buf, stream);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        my_accum[VEC_GPU] += (t1.tv_sec - t0.tv_sec) * 1000.0f
            + (t1.tv_nsec - t0.tv_nsec) / 1e6f;
#else
        for (int t = 0; t < horizon; t++) {
            clock_gettime(CLOCK_MONOTONIC, &t0);
            pufferl_forward(pufferl, buf, t, stream);

            cudaMemcpyAsync(
                &vec->actions[agent_start * NUM_ATNS],
                &vec->gpu_actions[agent_start * NUM_ATNS],
                agents_per_buffer * NUM_ATNS * sizeof(float),
                cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            my_accum[VEC_GPU] += (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_nsec - t0.tv_nsec) / 1e6f;

            memset(&vec->rewards[agent_start], 0, agents_per_buffer * sizeof(float));
            memset(&vec->terminals[agent_start], 0, agents_per_buffer * sizeof(float));
            clock_gettime(CLOCK_MONOTONIC, &t0);
            #pragma omp parallel for schedule(static) num_threads(vec->num_workers)
            for (int i = env_start; i < env_start + env_count; i++) {
                puf_step(&envs[i]);
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            my_accum[VEC_ENV_STEP] += (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_nsec - t0.tv_nsec) / 1e6f;

            cudaMemcpyAsync(
                vec->gpu_observations + (size_t)agent_start * OBS_SIZE,
                vec->observations + (size_t)agent_start * OBS_SIZE,
                (size_t)agents_per_buffer * OBS_SIZE * sizeof(obs_t),
                cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(
                &vec->gpu_rewards[agent_start],
                &vec->rewards[agent_start],
                agents_per_buffer * sizeof(float),
                cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(
                &vec->gpu_terminals[agent_start],
                &vec->terminals[agent_start],
                agents_per_buffer * sizeof(float),
                cudaMemcpyHostToDevice, stream);
            if (vec->action_mask_size > 0) {
                cudaMemcpyAsync(
                    vec->gpu_action_mask + agent_start * vec->action_mask_size,
                    vec->action_mask     + agent_start * vec->action_mask_size,
                    (size_t)agents_per_buffer * vec->action_mask_size * sizeof(unsigned char),
                    cudaMemcpyHostToDevice, stream);
            }
        }
#endif
        cudaStreamSynchronize(stream);
        atomic_store(&vec->buffer_states[buf], OMP_WAITING);
    }
}

void vec_step(VecEnv* vec) {
    for (int buf = 0; buf < vec->buffers; buf++) {
        atomic_store(&vec->buffer_states[buf], OMP_RUNNING);
    }
    for (int buf = 0; buf < vec->buffers; buf++) {
        while (atomic_load(&vec->buffer_states[buf]) != OMP_WAITING) {}
    }
}

Env* vec_init_envs(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {

    int total_agents = (int)dict_get(vec_kwargs, "total_agents");
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers");
    int agents_per_buffer = total_agents / num_buffers;

    Env* envs = (Env*)calloc(total_agents, sizeof(Env));

    int num_envs = 0;
    int agents_created = 0;
    while (agents_created < total_agents) {
        envs[num_envs].rng = num_envs;
        puf_init(&envs[num_envs], env_kwargs);
        agents_created += envs[num_envs].num_agents;
        num_envs++;
    }

    envs = (Env*)realloc(envs, num_envs * sizeof(Env));

    int buf = 0;
    int buf_agents = 0;
    buffer_env_starts[0] = 0;
    buffer_env_counts[0] = 0;
    for (int i = 0; i < num_envs; i++) {
        buf_agents += envs[i].num_agents;
        buffer_env_counts[buf]++;
        if (buf_agents >= agents_per_buffer && buf < num_buffers - 1) {
            buf++;
            buffer_env_starts[buf] = i + 1;
            buffer_env_counts[buf] = 0;
            buf_agents = 0;
        }
    }

    *num_envs_out = num_envs;
    return envs;
}

VecEnv* vec_create(Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents");
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers");
    VecEnv* vec = (VecEnv*)calloc(1, sizeof(VecEnv));
    vec->total_agents = total_agents;
    vec->buffers = num_buffers;
    vec->agents_per_buffer = total_agents / num_buffers;
    vec->num_workers = (int)dict_get(vec_kwargs, "num_threads") / num_buffers;
    if (vec->num_workers < 1) {
        vec->num_workers = 1;
    }
    int frozen_banks = (int)dict_get(vec_kwargs, "num_frozen_banks");
    vec->num_banks = frozen_banks + 1;
    vec->bank_layout = (int*)calloc(vec->num_banks + 1, sizeof(int));

    vec->buffer_env_starts = (int*)calloc(num_buffers, sizeof(int));
    vec->buffer_env_counts = (int*)calloc(num_buffers, sizeof(int));

    int num_envs = 0;
    vec->envs = vec_init_envs(&num_envs, vec->buffer_env_starts, vec->buffer_env_counts,
                            vec_kwargs, env_kwargs);
    vec->size = num_envs;

    obs_t* observations = NULL;
    obs_t* gpu_observations = NULL;
    cudaHostAlloc((void**)&observations,
        (size_t)total_agents * OBS_SIZE * sizeof(obs_t), cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->actions, total_agents * NUM_ATNS * sizeof(float), cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->rewards, total_agents * sizeof(float), cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->terminals, total_agents * sizeof(float), cudaHostAllocPortable);

    cudaMalloc((void**)&gpu_observations,
        (size_t)total_agents * OBS_SIZE * sizeof(obs_t));
    cudaMalloc((void**)&vec->gpu_actions, total_agents * NUM_ATNS * sizeof(float));
    cudaMalloc((void**)&vec->gpu_rewards, total_agents * sizeof(float));
    cudaMalloc((void**)&vec->gpu_terminals, total_agents * sizeof(float));

    vec->observations = observations;
    vec->gpu_observations = gpu_observations;

    cudaMemset(vec->gpu_observations, 0,
        (size_t)total_agents * OBS_SIZE * sizeof(obs_t));
    cudaMemset(vec->gpu_actions, 0, total_agents * NUM_ATNS * sizeof(float));
    cudaMemset(vec->gpu_rewards, 0, total_agents * sizeof(float));
    cudaMemset(vec->gpu_terminals, 0, total_agents * sizeof(float));

    vec->action_mask_size = (int)dict_get(vec_kwargs, "action_mask_size");
    if (vec->action_mask_size > 0) {
        size_t mask_bytes = (size_t)total_agents * vec->action_mask_size * sizeof(unsigned char);
        cudaHostAlloc((void**)&vec->action_mask, mask_bytes, cudaHostAllocPortable);
        cudaMalloc((void**)&vec->gpu_action_mask, mask_bytes);
        cudaMemset(vec->gpu_action_mask, 0, mask_bytes);
    }
    vec->streams = (cudaStream_t*)calloc(num_buffers, sizeof(cudaStream_t));

    Env* envs = vec->envs;
    for (int buf = 0; buf < num_buffers; buf++) {
        int buf_start = buf * vec->agents_per_buffer;
        int env_start = vec->buffer_env_starts[buf];
        int env_count = vec->buffer_env_counts[buf];

        float frozen_pct = (float)dict_get(vec_kwargs, "frozen_bank_pct");
        int frozen_envs = (int)(frozen_pct * env_count);
        int frozen_start = env_count - frozen_envs;
        if (frozen_banks == 0 || frozen_pct <= 0.0f) {
            frozen_start = env_count;
        }

        int* counts = (int*)calloc(vec->num_banks, sizeof(int));
        for (int e = 0; e < env_count; e++) {
            Env* env = &envs[env_start + e];
            for (int s = 0; s < env->num_agents; s++) {
                int policy = e < frozen_start ? 0 : env->agents[s].policy;
                if (policy < 0 || policy >= vec->num_banks) {
                    fprintf(stderr, "Agent policy %d outside bank range [0, %d)\n",
                        policy, vec->num_banks);
                    exit(1);
                }
                counts[policy]++;
            }
        }

        int offset = 0;
        for (int b = 0; b < vec->num_banks; b++) {
            if (buf == 0) {
                vec->bank_layout[b] = offset;
            } else if (vec->bank_layout[b] != offset) {
                fprintf(stderr, "Bank layout must match across buffers\n");
                exit(1);
            }
            offset += counts[b];
        }
        if (offset != vec->agents_per_buffer) {
            fprintf(stderr, "Buffer has %d agents, expected %d\n",
                offset, vec->agents_per_buffer);
            exit(1);
        }
        if (buf == 0) {
            vec->bank_layout[vec->num_banks] = offset;
        } else if (vec->bank_layout[vec->num_banks] != offset) {
            fprintf(stderr, "Bank layout must match across buffers\n");
            exit(1);
        }

        int* cursors = (int*)calloc(vec->num_banks, sizeof(int));
        for (int b = 0; b < vec->num_banks; b++) {
            cursors[b] = buf_start + vec->bank_layout[b];
        }
        for (int e = 0; e < env_count; e++) {
            Env* env = &envs[env_start + e];
            int tag = 0;
            for (int s = 0; s < env->num_agents; s++) {
                int policy = e < frozen_start ? 0 : env->agents[s].policy;
                if (policy > tag) {
                    tag = policy;
                }
                int phys = cursors[policy];
                env->agents[s].observations = vec->observations + (size_t)phys * OBS_SIZE;
                env->agents[s].actions = vec->actions + (size_t)phys * NUM_ATNS;
                env->agents[s].rewards = vec->rewards + phys;
                env->agents[s].terminals = vec->terminals + phys;
                if (vec->action_mask_size > 0) {
                    env->agents[s].action_mask =
                        vec->action_mask + (size_t)phys * vec->action_mask_size;
                } else {
                    env->agents[s].action_mask = NULL;
                }
                cursors[policy]++;
            }
            env->tag = tag;
            env->boundary_reached = 0;
        }
        free(cursors);
        free(counts);
    }

    return vec;
}

void vec_reset(VecEnv* vec) {
    Env* envs = vec->envs;
    #pragma omp parallel for schedule(static) num_threads(vec->num_workers)
    for (int i = 0; i < vec->size; i++) {
        puf_reset(&envs[i]);
    }
    cudaMemcpy(vec->gpu_observations, vec->observations,
        (size_t)vec->total_agents * OBS_SIZE * sizeof(obs_t), cudaMemcpyHostToDevice);
    cudaMemset(vec->gpu_rewards,   0, vec->total_agents * sizeof(float));
    cudaMemset(vec->gpu_terminals, 0, vec->total_agents * sizeof(float));
    if (vec->action_mask_size > 0) {
        cudaMemcpy(vec->gpu_action_mask, vec->action_mask,
            (size_t)vec->total_agents * vec->action_mask_size * sizeof(unsigned char),
            cudaMemcpyHostToDevice);
    }
#ifdef PUFFER_CUDA_ENV
    puf_cuda_env_init(vec, 0);
#endif
    cudaDeviceSynchronize();
}

void vec_create_threads(VecEnv* vec, int num_threads, int horizon, struct PuffeRL* pufferl) {
    vec->buffer_states = (atomic_int*)calloc(vec->buffers, sizeof(atomic_int));
    vec->threads = (pthread_t*)calloc(vec->buffers, sizeof(pthread_t));
    vec->workers = (VecWorker*)calloc(vec->buffers, sizeof(VecWorker));
    vec->accum = (float*)calloc(vec->buffers * NUM_VEC_PROF, sizeof(float));
    for (int i = 0; i < vec->buffers; i++) {
        vec->workers[i].vec = vec;
        vec->workers[i].buf = i;
        vec->workers[i].horizon = horizon;
        vec->workers[i].pufferl = pufferl;
        pthread_create(&vec->threads[i], NULL, vec_thread_main, &vec->workers[i]);
    }
}

void vec_close(VecEnv* vec) {
    Env* envs = vec->envs;

    if (vec->threads != NULL) {
        atomic_store(&vec->shutdown, 1);
        for (int i = 0; i < vec->buffers; i++) {
            pthread_join(vec->threads[i], NULL);
        }
    }

    for (int i = 0; i < vec->size; i++) {
        Env* env = &envs[i];
        puf_close(env);
    }

    free(vec->envs);
    free(vec->buffer_states);
    free(vec->threads);
    free(vec->workers);
    free(vec->accum);
    free(vec->buffer_env_starts);
    free(vec->buffer_env_counts);
    free(vec->bank_layout);

    cudaDeviceSynchronize();
#ifdef PUFFER_CUDA_ENV
    puf_cuda_env_close(vec);
#endif
    cudaFree(vec->gpu_observations);
    cudaFree(vec->gpu_actions);
    cudaFree(vec->gpu_rewards);
    cudaFree(vec->gpu_terminals);
    cudaFreeHost(vec->observations);
    cudaFreeHost(vec->actions);
    cudaFreeHost(vec->rewards);
    cudaFreeHost(vec->terminals);
    if (vec->action_mask_size > 0) {
        cudaFree(vec->gpu_action_mask);
        cudaFreeHost(vec->action_mask);
    }

    free(vec->streams);
    free(vec);
}

void vec_log(VecEnv* vec, Dict* out, int clear) {
#ifdef PUFFER_CUDA_ENV
    puf_cuda_env_sync_logs(vec, clear, 0);
    cudaStreamSynchronize(0);
#endif
    Env* envs = vec->envs;
    Log aggregate;
    memset(&aggregate, 0, sizeof(Log));
    int num_keys = sizeof(Log) / sizeof(float);
    for (int i = 0; i < vec->size; i++) {
        Env* env = &envs[i];
        if (env->log.n == 0) {
            continue;
        }
        for (int j = 0; j < num_keys; j++) {
            ((float*)&aggregate)[j] += ((float*)&env->log)[j];
        }
    }

    float n = aggregate.n;
    if (n == 0) {
        return;
    }
    for (int i = 0; i < num_keys; i++) {
        ((float*)&aggregate)[i] /= n;
    }
    if (clear) {
        for (int i = 0; i < vec->size; i++) {
            memset(&envs[i].log, 0, sizeof(Log));
        }
    }
    puf_log(&aggregate, out);
    dict_set(out, "n", n);
}


#define SELECT_COPY_THREADS 256

#define _PUFFER_STRINGIFY(x) #x
#define PUFFER_STRINGIFY(x) _PUFFER_STRINGIFY(x)

static double wall_clock() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

enum ProfileIdx {
    PROF_ROLLOUT = 0,
    PROF_EVAL_GPU,
    PROF_EVAL_ENV,
    PROF_TRAIN_MISC,
    PROF_TRAIN_FORWARD,
    NUM_PROF,
};

static const char* PROF_NAMES[NUM_PROF] = {
    "rollout",
    "eval_gpu",
    "eval_env",
    "train_misc",
    "train_forward",
};

#define NUM_TRAIN_EVENTS 5
typedef struct {
    cudaEvent_t events[NUM_TRAIN_EVENTS];
    float accum[NUM_PROF];
} ProfileT;

// Data collected by parallel environment workers. Each worker handles
// a constant subset of agents
struct RolloutBuf {
    PrecisionTensor observations;  // (horizon, agents, input_size)
    PrecisionTensor actions;       // (horizon, agents, num_atns)
    PrecisionTensor values;        // (horizon, agents)
    PrecisionTensor logprobs;      // ...
    PrecisionTensor rewards;
    PrecisionTensor terminals;
    PrecisionTensor ratio;
    PrecisionTensor importance;
    PrecisionTensor action_mask;   // (horizon, agents, mask_size); .data=nullptr when env opts out
};

// Buffers are initialized as raw structs with only shape information. alloc_register
// stores the shape and data pointer. Memory is only allocated after all buffers are registered.
void register_rollout_buffers(RolloutBuf& bufs, Allocator* alloc, int T, int B, int input_size,
        int num_atns, int mask_size) {
    bufs = (RolloutBuf){
        .observations = {.shape = {T, B, input_size}},
        .actions      = {.shape = {T, B, num_atns}},
        .values       = {.shape = {T, B}},
        .logprobs     = {.shape = {T, B}},
        .rewards      = {.shape = {T, B}},
        .terminals    = {.shape = {T, B}},
        .ratio        = {.shape = {T, B}},
        .importance   = {.shape = {T, B}},
        .action_mask  = {},
    };
    alloc_register(alloc, &bufs.observations);
    alloc_register(alloc, &bufs.actions);
    alloc_register(alloc, &bufs.values);
    alloc_register(alloc, &bufs.logprobs);
    alloc_register(alloc, &bufs.rewards);
    alloc_register(alloc, &bufs.terminals);
    alloc_register(alloc, &bufs.ratio);
    alloc_register(alloc, &bufs.importance);
    if (mask_size > 0) {
        bufs.action_mask = {.shape = {T, B, mask_size}};
        alloc_register(alloc, &bufs.action_mask);
    }
}


// Slice: select dim0 index t, then narrow dim0 from start for count.
// 3D (T, B, F) -> (count, F); 2D (T, B) -> (count,)
inline PrecisionTensor puf_slice(PrecisionTensor& p, int t, int start, int count) {
    if (ndim(p.shape) == 3) {
        long B = p.shape[1], F = p.shape[2];
        return {.data = p.data + (t*B + start)*F, .shape = {count, F}};
    } else {
        long B = p.shape[1];
        return {.data = p.data + (t*B + start), .shape = {count}};
    }
}

struct EnvBuf {
    ObsTensor obs;         // (total_agents, obs_size)
    FloatTensor actions;   // (total_agents, num_atns)
    FloatTensor rewards;   // (total_agents,)
    FloatTensor terminals; // (total_agents,)
    ByteTensor action_mask; // (total_agents, mask_size); .data=nullptr when env opts out
};

static int puf_act_sizes[] = ACT_SIZES;

VecEnv* create_environments(Dict* vec_kwargs, Dict* env_kwargs, EnvBuf& env) {
    VecEnv* vec = vec_create(vec_kwargs, env_kwargs);
    int total_agents = vec->total_agents;
    env.obs = { .data = vec->gpu_observations, .shape = {total_agents, OBS_SIZE} };
    env.actions = { .data = (float*)vec->gpu_actions, .shape = {total_agents, NUM_ATNS} };
    env.rewards = { .data = (float*)vec->gpu_rewards, .shape = {total_agents} };
    env.terminals = { .data = (float*)vec->gpu_terminals, .shape = {total_agents} };
    if (vec->action_mask_size > 0) {
        env.action_mask = { .data = vec->gpu_action_mask,
                            .shape = {total_agents, vec->action_mask_size} };
    } else {
        env.action_mask = { .data = nullptr, .shape = {0} };
    }
    return vec;
}

// A frozen weight bank: same shape as the primary, but its own params buffer
// (and per-buffer rollout states/activations). Used for match (eval) and league
// (frozen historical opponents). Not trained; updated only via load.
typedef struct {
    Policy policy;  // Bank-owned Policy; lets banks have different arch than primary.
    PolicyWeights weights;
    Allocator params_alloc;
    Allocator acts_alloc;
    PrecisionTensor param_puf;
    FloatTensor master_weights;
    PrecisionTensor* buffer_states;         // [num_buffers]
    PolicyActivations* buffer_activations;  // [num_buffers]
    int slice_size;  // # agents per buffer this bank owns; sets activation/state batch dim
    int hidden_size;
    int num_layers;
} WeightBank;

typedef struct PuffeRL {
    Policy policy;
    PolicyWeights weights;       // current precision_t weights (structured)
    PolicyActivations train_activations;
    Allocator params_alloc;
    Allocator grads_alloc;
    Allocator activations_alloc;
    VecEnv* vec;
    Muon muon;
    ncclComm_t nccl_comm;  // NCCL communicator for multi-GPU
    HypersT hypers;
    bool is_continuous;  // True if all action dimensions are continuous (size==1)
    PrecisionTensor* buffer_states;  // Per-buffer states for contiguous access
    PolicyActivations* buffer_activations;  // Per-buffer inference activations
    RolloutBuf rollouts;
    RolloutBuf train_rollouts;  // Pre-allocated transposed copy for train_impl
    EnvBuf env;
    TrainGraph train_buf;
    PrecisionTensor advantages_puf;  // Pre-allocated for train_impl (B, T)
    // fbr: CUDA environments capture one entire horizon per buffer. CPU environments
    // retain one graph per timestep because host stepping separates inference
    // calls and cannot be captured into a device-only horizon.
    cudaGraphExec_t* fused_rollout_cudagraphs;
    cudaGraphExec_t train_cudagraph;
    cudaStream_t* streams;  // per-buffer raw CUDA streams
    cudaStream_t default_stream;  // main-thread stream (captured once at init)
    IntTensor act_sizes_puf;    // CUDA int32 tensor of action head sizes
    FloatTensor losses_puf;     // (NUM_LOSSES,) f32 accumulator
    PPOBuffersPuf ppo_bufs_puf; // Pre-allocated buffers for ppo_loss_fwd_bwd
    PrioBuffers prio_bufs;      // Pre-allocated buffers for prio_replay
    FloatTensor master_weights;  // fp32 master weights (flat); same buffer as param_puf in fp32 mode
    PrecisionTensor param_puf;
    PrecisionTensor grad_puf;
    LongTensor rng_offset_puf;   // (num_buffers+1,) int64 CUDA device counters
    ProfileT profile;
    nvmlDevice_t nvml_device;
    long epoch;
    long global_step;
    double start_time;
    double last_log_time;
    long last_log_step;
    int train_warmup;
    bool rollout_captured;
    bool train_captured;
    ulong seed;
    curandStatePhilox4_32_10_t** rng_states;  // per-buffer persistent RNG states [num_buffers]
    // Optional frozen weight banks for match / league.
    WeightBank* frozen_banks;  // [num_frozen_banks]
    int num_frozen_banks;
    char env_name[64];  // Kept for post-init bank adds.
    // Per-buffer-relative bank layout: bank_layout[b] = first agent within each
    // buffer chunk owned by bank b. Length num_banks+1; ends at agents_per_buffer.
    // Same shape applied to every buffer (each buffer hosts every bank), so each
    // worker thread only writes inside its own physical chunk.
    // Bank 0 = primary (learner). NULL = no layout set (primary owns full chunk).
    int* bank_layout;
} PuffeRL;

static void mkdir_p(const char* path) {
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            if (mkdir(tmp, 0777) != 0 && errno != EEXIST) {
                fprintf(stderr, "failed to create directory %s: %s\n", tmp, strerror(errno));
                exit(1);
            }
            *p = '/';
        }
    }
    if (mkdir(tmp, 0777) != 0 && errno != EEXIST) {
        fprintf(stderr, "failed to create directory %s: %s\n", tmp, strerror(errno));
        exit(1);
    }
}

static int puf_has_suffix(const char* s, const char* suffix) {
    size_t n = strlen(s);
    size_t m = strlen(suffix);
    return n >= m && strcmp(s + n - m, suffix) == 0;
}

static void puf_find_latest_checkpoint(const char* dir,
        char* out, size_t out_size, time_t* best_time) {
    DIR* dp = opendir(dir);
    if (!dp) {
        return;
    }

    struct dirent* ent = NULL;
    while ((ent = readdir(dp))) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
            continue;
        }

        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name);

        struct stat st;
        if (stat(path, &st) != 0) {
            continue;
        }

        if (S_ISDIR(st.st_mode)) {
            puf_find_latest_checkpoint(path, out, out_size, best_time);
        } else if (S_ISREG(st.st_mode) && puf_has_suffix(path, ".bin") &&
                st.st_ctime >= *best_time) {
            *best_time = st.st_ctime;
            snprintf(out, out_size, "%s", path);
        }
    }

    closedir(dp);
}

static const char* puf_checkpoint_path_key(Config* cfg, const char* key,
        char* out, size_t out_size) {
    const char* load_path = puf_config_str(cfg, "base", key);
    if (!load_path || strcmp(load_path, "None") == 0) {
        return NULL;
    }

    if (strcmp(load_path, "latest") != 0) {
        return load_path;
    }

    char root[2048];
    snprintf(root, sizeof(root), "%s/%s",
        puf_config_str(cfg, "base", "checkpoint_dir"),
        puf_config_str(cfg, "base", "env_name"));

    out[0] = 0;
    time_t best_time = 0;
    puf_find_latest_checkpoint(root, out, out_size, &best_time);
    if (!out[0]) {
        fprintf(stderr, "no .bin checkpoints found in %s\n", root);
        exit(1);
    }
    return out;
}

static void puf_save_weights(PuffeRL* p, const char* path) {
    int64_t nbytes = numel(p->master_weights.shape) * sizeof(float);
    char* buf = (char*)malloc(nbytes);
    cudaMemcpy(buf, p->master_weights.data, nbytes, cudaMemcpyDeviceToHost);
    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path, getpid());
    FILE* fp = fopen(tmp, "wb");
    if (!fp) {
        fprintf(stderr, "failed to open %s for writing\n", tmp);
        free(buf);
        exit(1);
    }
    if (fwrite(buf, 1, nbytes, fp) != (size_t)nbytes) {
        fprintf(stderr, "failed to write weights to %s\n", tmp);
        fclose(fp);
        free(buf);
        exit(1);
    }
    fclose(fp);
    free(buf);
    if (rename(tmp, path) != 0) {
        fprintf(stderr, "failed to publish weights to %s\n", path);
        exit(1);
    }
}

static void puf_load_weights_into(FloatTensor dst, PrecisionTensor params,
        cudaStream_t stream, const char* path) {
    int64_t nbytes = numel(dst.shape) * sizeof(float);
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "failed to open %s for reading\n", path);
        exit(1);
    }
    char* buf = (char*)malloc(nbytes);
    size_t nread = fread(buf, 1, nbytes, fp);
    fclose(fp);
    if ((int64_t)nread != nbytes) {
        fprintf(stderr, "failed to read weights from %s\n", path);
        free(buf);
        exit(1);
    }
    cudaMemcpy(dst.data, buf, nbytes, cudaMemcpyHostToDevice);
    free(buf);
    if (USE_BF16) {
        int n = numel(params.shape);
        cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(params.data, dst.data, n);
    }
}

static void puf_load_primary_if_configured(PuffeRL* p, Config* cfg) {
    char resolved_path[4096];
    const char* load_path = puf_checkpoint_path_key(cfg,
        "load_model_path", resolved_path, sizeof(resolved_path));
    if (load_path) {
        puf_load_weights_into(p->master_weights, p->param_puf, p->default_stream, load_path);
        printf("Loaded weights from %s\n", load_path);
    }
}

inline void profile_begin(const char* tag, bool enable) {
    if (enable) {
        nvtxRangePushA(tag);
    }
}

inline void profile_end(bool enable) {
    if (enable) {
        nvtxRangePop();
    }
}

__global__ void rng_init(curandStatePhilox4_32_10_t* states, uint64_t seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__device__ __forceinline__ float masked_logit(const precision_t* logits,
        int logits_base, int logits_offset, int offset,
        const precision_t* mask, int mask_base) {
    float l = safe_logit(logits, logits_base, logits_offset, offset);
    if (mask != nullptr) {
        float m = to_float(mask[mask_base + logits_offset + offset]);
        if (m == 0.0f) l = -1e4f;
    }
    return l;
}

// Expects action logits and values to be in the same contiguous buffer. See default decoder
__global__ void sample_logits(
        PrecisionTensor dec_out,              // (B, logits_dim + 1 for values)
        PrecisionTensor logstd_puf,           // (1, od) - continuous actions only
        IntTensor act_sizes_puf,              // (num_atns,) action head sizes
        precision_t* __restrict__ actions,    // (B, num_atns)
        precision_t* __restrict__ logprobs,   // (B,)
        precision_t* __restrict__ value_out,  // (B,)
        curandStatePhilox4_32_10_t* __restrict__ rng_states,
        const precision_t* __restrict__ action_mask, // (B, A_total) or nullptr
        int mask_stride) {                    // 0 when action_mask is nullptr
    int B = dec_out.shape[0];
    int fused_cols = dec_out.shape[1];
    int num_atns = numel(act_sizes_puf.shape);
    const int* act_sizes = act_sizes_puf.data;
    const precision_t* logits = dec_out.data;
    int logits_stride = fused_cols;
    int value_stride = fused_cols;
    bool is_continuous = logstd_puf.data != nullptr && numel(logstd_puf.shape) > 0;
    const precision_t* logstd = logstd_puf.data;
    int logstd_stride = is_continuous ? 0 : 0;  // 1D broadcast: stride 0
    const precision_t* value = logits + (fused_cols - 1);  // last column

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) {
        return;
    }

    // Load persistent RNG state (advanced in-place each call)
    curandStatePhilox4_32_10_t state = rng_states[idx];

    int logits_base = idx * logits_stride;
    float total_log_prob = 0.0f;

    if (is_continuous) {
        // Continuous action sampling from Normal(mean, exp(logstd))
        constexpr float LOG_2PI = 1.8378770664093453f;  // log(2*pi)
        int logstd_base = idx * logstd_stride;  // separate stride for logstd (may be 0 for broadcast)

        for (int h = 0; h < num_atns; ++h) {
            float mean = safe_continuous_mean(logits, logits_base + h);
            float log_std = safe_continuous_logstd(logstd, logstd_base + h);
            float std = expf(log_std);

            // Sample from N(0,1) and transform: action = mean + std * noise
            float noise = curand_normal(&state);
            float action = finite_or_clamp(mean + std * noise, -1.0e6f, 1.0e6f);

            precision_t stored_action_p = from_float(action);
            float stored_action = to_float(stored_action_p);
            // Log probability: -0.5 * ((action - mean) / std)^2 - 0.5 * log(2*pi) - log(std)
            float normalized = (stored_action - mean) / std;
            float log_prob = -0.5f * normalized * normalized - 0.5f * LOG_2PI - log_std;

            actions[idx * num_atns + h] = stored_action_p;
            total_log_prob += log_prob;
        }
    } else {
        // Discrete action sampling (original multinomial logic)
        int logits_offset = 0;  // offset within row for current action head
        int mask_base = (action_mask != nullptr) ? idx * mask_stride : 0;

        for (int h = 0; h < num_atns; ++h) {
            int A = act_sizes[h];  // size of this action head

            // Step 1: Find max and sum for numerical stability (with nan_to_num)
            float max_val = -INFINITY;
            float sum_exp = 0.0f;
            for (int a = 0; a < A; ++a) {
                float l = masked_logit(logits, logits_base, logits_offset, a, action_mask, mask_base);
                if (l > max_val) {
                    sum_exp *= expf(max_val - l);
                    max_val = l;
                }
                sum_exp += expf(l - max_val);
            }
            float logsumexp = max_val + logf(sum_exp);

            // Step 3: Generate random value for this action head
            float rand_val = curand_uniform(&state);

            // Step 4: Multinomial sampling using inverse CDF
            float cumsum = 0.0f;
            int sampled_action = -1;  // sentinel: no action chosen yet

            for (int a = 0; a < A; ++a) {
                float l = masked_logit(logits, logits_base, logits_offset, a, action_mask, mask_base);
                float prob = expf(l - logsumexp);
                cumsum += prob;
                if (rand_val < cumsum) {
                    sampled_action = a;
                    break;
                }
            }

            // Float rounding can leave cumsum < 1.0; fall back to the last legal action.
            if (sampled_action < 0) {
                sampled_action = A - 1;
                if (action_mask != nullptr) {
                    for (int a = A - 1; a >= 0; --a) {
                        if (to_float(action_mask[mask_base + logits_offset + a]) != 0.0f) {
                            sampled_action = a;
                            break;
                        }
                    }
                }
            }

            // Step 5: Gather log probability of sampled action
            float sampled_logit = masked_logit(logits, logits_base, logits_offset, sampled_action, action_mask, mask_base);
            float log_prob = sampled_logit - logsumexp;

            // Write action for this head
            actions[idx * num_atns + h] = from_float(sampled_action);
            total_log_prob += log_prob;

            // Advance to next action head
            logits_offset += A;
        }
    }

    // Write summed log probability (log of joint probability)
    logprobs[idx] = from_float(total_log_prob);

    // Copy value (fused to avoid separate elementwise kernel for strided->contiguous copy)
    value_out[idx] = value[idx * value_stride];

    // Save RNG state back for next call
    rng_states[idx] = state;
}

void pufferl_forward(PuffeRL* pufferl, int buf, int t, cudaStream_t stream) {
    HypersT& hypers = pufferl->hypers;
#ifndef PUFFER_CUDA_ENV
    int graph = t * hypers.num_buffers + buf;
    profile_begin("fused_rollout", hypers.profile);

    if (pufferl->rollout_captured) {
        assert(cudaGraphLaunch(pufferl->fused_rollout_cudagraphs[graph], stream) == cudaSuccess
                && "cudaGraphLaunch failed");
        profile_end(hypers.profile);
        return;
    }

    bool capturing = pufferl->epoch == hypers.cudagraphs;
    if (capturing) {
        assert(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess
                && "cudaStreamBeginCapture failed");
    }
#endif

    RolloutBuf& rollouts = pufferl->rollouts;
    EnvBuf& env = pufferl->env;
    int block_size = pufferl->vec->total_agents / hypers.num_buffers;
    int start = buf * block_size;
    // Copy observations, rewards, terminals from GPU env buffers to rollout buffer
    ObsTensor& obs_env = env.obs;
    int n = block_size * obs_env.shape[1];
    PrecisionTensor obs_dst = puf_slice(rollouts.observations, t, start, block_size);
#ifdef PUFFER_CUDA_ENV
    // fbr: Later timesteps are written directly by the previous CUDA environment step.
    if (t == 0) {
#endif
    cast_dispatch(obs_dst.data, obs_env.data + (long)start*obs_env.shape[1], n, stream);

    PrecisionTensor rew_dst = puf_slice(rollouts.rewards, t, start, block_size);
    n = block_size;
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
        rew_dst.data, env.rewards.data + start, n);

    PrecisionTensor term_dst = puf_slice(rollouts.terminals, t, start, block_size);
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
        term_dst.data, env.terminals.data + start, n);
#ifdef PUFFER_CUDA_ENV
    }
#endif

    // Copy action mask from env into rollout buffer (if env opted in)
    PrecisionTensor mask_slice = {};
    int mask_stride = 0;
    if (rollouts.action_mask.data != nullptr) {
        int mask_size = rollouts.action_mask.shape[2];
        mask_stride = mask_size;
        mask_slice = puf_slice(rollouts.action_mask, t, start, block_size);
        int mask_n = block_size * mask_size;
        cast<<<grid_size(mask_n), BLOCK_SIZE, 0, stream>>>(
            mask_slice.data,
            env.action_mask.data + (long)start * mask_size,
            mask_n);
    }

    // Per-bank policy forward + sampling. Each bank owns a contiguous sub-range
    // [bank_layout[b], bank_layout[b+1]) within every buffer's chunk; layout is
    // per-buffer-relative so each worker writes only inside its own chunk.
    // Cudagraph capture absorbs the extra kernel launches.
    int num_banks = 1 + pufferl->num_frozen_banks;
#ifndef PUFFER_CUDA_ENV
    long act_cols = env.actions.shape[1];
#endif
    for (int b = 0; b < num_banks; b++) {
        int bank_off = pufferl->bank_layout ? pufferl->bank_layout[b] : 0;
        int bank_end = pufferl->bank_layout ? pufferl->bank_layout[b + 1] : block_size;
        int bank_size = bank_end - bank_off;
        if (bank_size == 0) continue;

        Policy* p_bank;
        PolicyWeights* w_bank;
        PolicyActivations* a_bank;
        PrecisionTensor* s_bank;
        if (b == 0) {
            p_bank = &pufferl->policy;
            w_bank = &pufferl->weights;
            a_bank = &pufferl->buffer_activations[buf];
            s_bank = &pufferl->buffer_states[buf];
        } else {
            WeightBank* fb = &pufferl->frozen_banks[b - 1];
            p_bank = &fb->policy;
            w_bank = &fb->weights;
            a_bank = &fb->buffer_activations[buf];
            s_bank = &fb->buffer_states[buf];
        }

        int sub_start = start + bank_off;
        PrecisionTensor obs_b   = puf_slice(rollouts.observations, t, sub_start, bank_size);
        PrecisionTensor act_b   = puf_slice(rollouts.actions,      t, sub_start, bank_size);
        PrecisionTensor lp_b    = puf_slice(rollouts.logprobs,     t, sub_start, bank_size);
        PrecisionTensor val_b   = puf_slice(rollouts.values,       t, sub_start, bank_size);
        PrecisionTensor mask_b  = {};
        int mask_stride_b = 0;
        if (rollouts.action_mask.data != nullptr) {
            mask_b = puf_slice(rollouts.action_mask, t, sub_start, bank_size);
            mask_stride_b = mask_stride;
        }

        PrecisionTensor dec_puf = policy_forward(p_bank, *w_bank, *a_bank, obs_b, *s_bank, stream);

        PrecisionTensor p_logstd = {};
        DecoderWeights* dw = (DecoderWeights*)w_bank->decoder;
        if (dw->continuous) {
            p_logstd = dw->logstd;
        }

        // Offset RNG by bank_off so banks don't collide on per-buffer rng slots.
        sample_logits<<<grid_size(bank_size), BLOCK_SIZE, 0, stream>>>(
            dec_puf, p_logstd, pufferl->act_sizes_puf,
            act_b.data, lp_b.data, val_b.data,
            pufferl->rng_states[buf] + bank_off,
            mask_b.data, mask_stride_b);

#ifndef PUFFER_CUDA_ENV
        cast<<<grid_size(numel(act_b.shape)), BLOCK_SIZE, 0, stream>>>(
            env.actions.data + (long)sub_start * act_cols,
            act_b.data, numel(act_b.shape));
#endif
    }

#ifdef PUFFER_CUDA_ENV
    // fbr: Pass sampled precision_t actions straight into the device environment.
    int env_start = pufferl->vec->buffer_env_starts[buf];
    int env_count = pufferl->vec->buffer_env_counts[buf];
    PrecisionTensor action_slice = puf_slice(
        rollouts.actions, t, start, block_size);
    const precision_t* action_src = action_slice.data;
    if (t + 1 < hypers.horizon) {
        PrecisionTensor next_obs = puf_slice(
            rollouts.observations, t + 1, start, block_size);
        PrecisionTensor next_rew = puf_slice(
            rollouts.rewards, t + 1, start, block_size);
        PrecisionTensor next_term = puf_slice(
            rollouts.terminals, t + 1, start, block_size);
        puf_cuda_env_step_direct(pufferl->vec, env_start, env_count,
            start, block_size, action_src,
            next_obs.data, next_rew.data, next_term.data, stream);
    } else {
        // fbr: Preserve the final state in the float buffers used by timestep zero.
        puf_cuda_env_step(pufferl->vec, env_start, env_count,
            start, block_size, action_src,
            obs_env.data + (long)start * obs_env.shape[1],
            env.rewards.data + start, env.terminals.data + start, stream);
    }
#endif

#ifndef PUFFER_CUDA_ENV
    if (capturing) {
        cudaGraph_t _graph;
        assert(cudaStreamEndCapture(stream, &_graph) == cudaSuccess
                && "cudaStreamEndCapture failed");
        assert(cudaGraphInstantiate(&pufferl->fused_rollout_cudagraphs[graph], _graph, 0) == cudaSuccess
                && "cudaGraphInstantiate failed");
        assert(cudaGraphDestroy(_graph) == cudaSuccess && "cudaGraphDestroy failed");
        cudaDeviceSynchronize();
    }
    profile_end(hypers.profile);
#endif
}

#ifdef PUFFER_CUDA_ENV
// fbr: Launch one captured 64-step rollout graph per CUDA environment buffer.
void pufferl_rollout_buffer(PuffeRL* pufferl, int buf, cudaStream_t stream) {
    HypersT& hypers = pufferl->hypers;
    profile_begin("fused_rollout_horizon", hypers.profile);
    if (pufferl->rollout_captured) {
        assert(cudaGraphLaunch(pufferl->fused_rollout_cudagraphs[buf], stream)
                == cudaSuccess && "full-horizon cudaGraphLaunch failed");
    } else {
        for (int t = 0; t < hypers.horizon; ++t) {
            pufferl_forward(pufferl, buf, t, stream);
        }
    }
    profile_end(hypers.profile);
}
#endif

// Zero advantages on frozen-bank rows so prio_replay never samples them. Frozen
// rollout rows hold actions/logprobs from the frozen policy; training the
// primary's PPO on them produces garbage ratios and poisoned gradients.
__global__ void zero_frozen_advantages_kernel(precision_t* advantages,
        int agents_per_buffer, int primary_per_buffer, int total_rows, int horizon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_rows * horizon;
    if (idx >= total) {
        return;
    }
    int row = idx / horizon;
    int rel = row % agents_per_buffer;
    if (rel >= primary_per_buffer) {
        advantages[idx] = from_float(0.0f);
    }
}

__device__ __forceinline__ void copy_values_adv_returns(
        const precision_t* __restrict__ src_values, precision_t* __restrict__ dst_values,
        const precision_t* __restrict__ src_advantages, precision_t* __restrict__ dst_advantages,
        precision_t* __restrict__ dst_returns,
        int src_row, int dst_row, int horizon) {
    int srh = (int64_t)src_row * horizon;
    int drh = (int64_t)dst_row * horizon;
    const precision_t* s_values = src_values + srh;
    const precision_t* s_adv = src_advantages + srh;
    precision_t* d_values = dst_values + drh;
    precision_t* d_adv = dst_advantages + drh;
    precision_t* d_returns = dst_returns + drh;
    for (int i = threadIdx.x; i < horizon; i += blockDim.x) {
        precision_t val = s_values[i];
        precision_t adv = s_adv[i];
        d_values[i] = val;
        d_adv[i] = adv;
        d_returns[i] = from_float(to_float(val) + to_float(adv));
    }
}

__global__ void select_copy(RolloutBuf rollouts, TrainGraph graph,
        const int* __restrict__ idx, const precision_t* __restrict__ advantages,
        const float* __restrict__ mb_prio) {
    int mb = blockIdx.x;
    int ch = blockIdx.y;
    int src_row = idx[mb];

    int obs_row_bytes = (numel(rollouts.observations.shape)
        / rollouts.observations.shape[0]) * sizeof(precision_t);
    int act_row_bytes = (numel(rollouts.actions.shape)
        / rollouts.actions.shape[0]) * sizeof(precision_t);
    int lp_row_bytes = (numel(rollouts.logprobs.shape)
        / rollouts.logprobs.shape[0]) * sizeof(precision_t);
    int horizon = rollouts.values.shape[1];

    switch (ch) {
    case 0:
        copy_bytes((const char*)rollouts.observations.data,
            (char*)graph.mb_obs.data, src_row, mb, obs_row_bytes);
        break;
    case 1:
        copy_bytes((const char*)rollouts.actions.data,
            (char*)graph.mb_actions.data, src_row, mb, act_row_bytes);
        break;
    case 2:
        copy_bytes((const char*)rollouts.logprobs.data,
            (char*)graph.mb_logprobs.data, src_row, mb, lp_row_bytes);
        break;
    case 3:
        copy_values_adv_returns(rollouts.values.data, graph.mb_values.data,
            advantages, graph.mb_advantages.data,
            graph.mb_returns.data, src_row, mb, horizon);
        break;
    case 4:
        if (threadIdx.x == 0) {
            graph.mb_prio.data[mb] = from_float(mb_prio[mb]);
        }
        break;
    case 5:
        if (graph.mb_action_mask.data != nullptr) {
            int mask_row_bytes = (numel(rollouts.action_mask.shape)
                / rollouts.action_mask.shape[0]) * sizeof(precision_t);
            copy_bytes((const char*)rollouts.action_mask.data,
                (char*)graph.mb_action_mask.data, src_row, mb, mask_row_bytes);
        }
        break;
    }
}

// Minor copy bandwidth optimizations
__global__ void index_copy(char* __restrict__ dst, const int* __restrict__ idx,
        const char* __restrict__ src, int num_idx, int row_bytes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_idx) {
        int dst_row = idx[i];
        memcpy(dst + (int64_t)dst_row * row_bytes, src + (int64_t)i * row_bytes, row_bytes);
    }
}

inline float cosine_annealing(float lr_base, float lr_min, long t, long T) {
    if (T == 0) return lr_base;
    float ratio = (double )t / (double) T;
    ratio = std::max(0.0f, std::min(1.0f, ratio));
    return lr_min + 0.5f*(lr_base - lr_min)*(1.0f + std::cos(M_PI * ratio));
}

void train_impl(PuffeRL& pufferl) {
    HypersT& hypers = pufferl.hypers;

    cudaEventRecord(pufferl.profile.events[0]);  // pre-loop start
    cudaStream_t train_stream = pufferl.default_stream;

    // Transpose from rollout layout (T, B, ...) to train layout (B, T, ...)
    RolloutBuf& src = pufferl.rollouts;
    RolloutBuf& rollouts = pufferl.train_rollouts;
    PrecisionTensor& advantages_puf = pufferl.advantages_puf;

    int T = src.observations.shape[0], B = src.observations.shape[1];
    int obs_size = (ndim(src.observations.shape) >= 3) ? src.observations.shape[2] : 1;
    int num_atns = (ndim(src.actions.shape) >= 3) ? src.actions.shape[2] : 1;

    transpose_102<<<grid_size(T*B*obs_size), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.observations.data, src.observations.data, T, B, obs_size);
    transpose_102<<<grid_size(T*B*num_atns), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.actions.data, src.actions.data, T, B, num_atns);
    transpose_102<<<grid_size(T*B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.logprobs.data, src.logprobs.data, T, B, 1);
    transpose_102<<<grid_size(T*B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.rewards.data, src.rewards.data, T, B, 1);
    transpose_102<<<grid_size(T*B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.terminals.data, src.terminals.data, T, B, 1);
    transpose_102<<<grid_size(T*B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.ratio.data, src.ratio.data, T, B, 1);
    transpose_102<<<grid_size(T*B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.values.data, src.values.data, T, B, 1);
    if (src.action_mask.data != nullptr) {
        int mask_size = src.action_mask.shape[2];
        transpose_102<<<grid_size(T*B*mask_size), BLOCK_SIZE, 0, train_stream>>>(
            rollouts.action_mask.data, src.action_mask.data, T, B, mask_size);
    }

    // We hard-clamp rewards to -1, 1. Our envs are mostly designed to respect this range
    clamp_precision_kernel<<<grid_size(numel(rollouts.rewards.shape)), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.rewards.data, -1.0f, 1.0f, numel(rollouts.rewards.shape));

    // Set importance weights to 1.0
    fill_precision_kernel<<<grid_size(numel(rollouts.ratio.shape)), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.ratio.data, from_float(1.0f), numel(rollouts.ratio.shape));

    int minibatch_size = hypers.minibatch_size;
    int batch_size = hypers.total_agents * hypers.horizon;
    int minibatch_segments = minibatch_size / hypers.horizon;
    float prio_beta0 = hypers.prio_beta0;
    float prio_alpha = hypers.prio_alpha;
    bool anneal_lr = hypers.anneal_lr;
    int current_epoch = pufferl.epoch;

    Muon* muon = &pufferl.muon;
    int total_epochs = hypers.total_timesteps / batch_size;
    if (anneal_lr) {
        float lr_min = hypers.min_lr_ratio * hypers.lr;
        float lr = cosine_annealing(hypers.lr, lr_min, current_epoch, total_epochs);
        cudaMemcpy(muon->lr_ptr, &lr, sizeof(float), cudaMemcpyHostToDevice);
    }

    // Annealed entropy coefficient — same cosine shape as lr. With PG signal
    // alive, the entropy bonus that kept early-training exploratory becomes
    // load-bearing dead weight late in training; cosine-decay frees the policy
    // to commit harder on what it has already learned.
    float current_ent_coef = hypers.ent_coef;
    if (hypers.anneal_ent_coef) {
        float ent_min = hypers.min_ent_coef_ratio * hypers.ent_coef;
        current_ent_coef = cosine_annealing(hypers.ent_coef, ent_min,
                                            current_epoch, total_epochs);
    }

    // Annealed priority exponent
    float anneal_beta = prio_beta0 + (1.0f - prio_beta0) * prio_alpha * (float)current_epoch/(float)total_epochs;
    TrainGraph& graph = pufferl.train_buf;
    cudaEventRecord(pufferl.profile.events[1]);  // pre-loop end

    int total_minibatches = hypers.replay_ratio * batch_size / hypers.minibatch_size;
    for (int mb = 0; mb < total_minibatches; ++mb) {
        cudaEventRecord(pufferl.profile.events[2]);  // start of misc (overwritten each iter)
        puf_zero(&advantages_puf, train_stream);

        profile_begin("compute_advantage", hypers.profile);
        puff_advantage_cuda(rollouts.values, rollouts.rewards, rollouts.terminals,
            rollouts.ratio, advantages_puf, hypers.gamma, hypers.gae_lambda,
            hypers.vtrace_rho_clip, hypers.vtrace_c_clip, train_stream);
        if (pufferl.num_frozen_banks > 0 && pufferl.bank_layout != NULL) {
            int apb = hypers.total_agents / hypers.num_buffers;
            int rows = advantages_puf.shape[0];
            int horizon = advantages_puf.shape[1];
            int total = rows * horizon;
            zero_frozen_advantages_kernel<<<grid_size(total), BLOCK_SIZE, 0, train_stream>>>(
                advantages_puf.data, apb, pufferl.bank_layout[1], rows, horizon);
        }
        profile_end(hypers.profile);

        profile_begin("compute_prio", hypers.profile);
        // Use the training RNG offset slot (last slot, index num_buffers)
        long* train_rng_offset = pufferl.rng_offset_puf.data + hypers.num_buffers;
        prio_replay_cuda(advantages_puf, prio_alpha, minibatch_segments,
            hypers.total_agents, anneal_beta,
            pufferl.prio_bufs, pufferl.seed, train_rng_offset, train_stream);
        profile_end(hypers.profile);

        profile_begin("train_select_and_copy", hypers.profile);
        if (hypers.reset_state) puf_zero(&graph.mb_state, train_stream);
        {
            RolloutBuf sel_src = rollouts;
            sel_src.values = rollouts.values;
            int mb_segs = pufferl.prio_bufs.idx.shape[0];
            int channels = (graph.mb_action_mask.data != nullptr) ? 6 : 5;
            select_copy<<<dim3(mb_segs, channels), SELECT_COPY_THREADS, 0, train_stream>>>(
                sel_src, graph, pufferl.prio_bufs.idx.data,
                advantages_puf.data, pufferl.prio_bufs.mb_prio.data);
        }
        profile_end(hypers.profile);

        cudaEventRecord(pufferl.profile.events[3]);  // end misc / start forward
        profile_begin("train_forward_backward", hypers.profile);
        if (pufferl.train_captured) {
            cudaGraphLaunch(pufferl.train_cudagraph, train_stream);
        } else {
            bool capturing = pufferl.train_warmup == hypers.cudagraphs;
            if (capturing) {
                assert(cudaStreamBeginCapture(train_stream, cudaStreamCaptureModeGlobal) == cudaSuccess
                        && "cudaStreamBeginCapture failed");
            }

            cudaStream_t stream = train_stream;
            PrecisionTensor obs_puf = graph.mb_obs;
            PrecisionTensor state_puf = graph.mb_state;
            PrecisionTensor dec_puf = policy_forward_train(&pufferl.policy, pufferl.weights, pufferl.train_activations, obs_puf, state_puf, stream);
            DecoderWeights* dw_train = (DecoderWeights*)pufferl.weights.decoder;
            PrecisionTensor p_logstd;
            if (dw_train->continuous) {
                p_logstd = dw_train->logstd;
            }

            ppo_loss_fwd_bwd(dec_puf, p_logstd, graph,
                pufferl.act_sizes_puf, pufferl.losses_puf,
                hypers.clip_coef, hypers.vf_clip_coef, hypers.vf_coef, current_ent_coef,
                pufferl.ppo_bufs_puf, pufferl.is_continuous, stream);

            FloatTensor grad_logits_puf = pufferl.ppo_bufs_puf.grad_logits;
            FloatTensor grad_logstd_puf = pufferl.is_continuous ? pufferl.ppo_bufs_puf.grad_logstd : FloatTensor();
            FloatTensor grad_values_puf = pufferl.ppo_bufs_puf.grad_values;
            policy_backward(&pufferl.policy, pufferl.weights, pufferl.train_activations,
                grad_logits_puf, grad_logstd_puf, grad_values_puf, stream);

            if (pufferl.nccl_comm != nullptr && hypers.world_size > 1) {
                ncclAllReduce(pufferl.grad_puf.data, pufferl.grad_puf.data,
                    numel(pufferl.grad_puf.shape), NCCL_PRECISION, ncclAvg,
                    pufferl.nccl_comm, stream);
            }
            muon_step(&pufferl.muon, pufferl.master_weights, pufferl.grad_puf, hypers.max_grad_norm, stream);
            if (USE_BF16) {
                int n = numel(pufferl.param_puf.shape);
                cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
                    pufferl.param_puf.data, pufferl.master_weights.data, n);
            }
            if (capturing) {
                cudaGraph_t _graph;
                assert(cudaStreamEndCapture(train_stream, &_graph) == cudaSuccess
                        && "cudaStreamEndCapture failed");
                assert(cudaGraphInstantiate(&pufferl.train_cudagraph, _graph, 0) == cudaSuccess
                        && "cudaGraphInstantiate failed");
                assert(cudaGraphDestroy(_graph) == cudaSuccess && "cudaGraphDestroy failed");
                cudaDeviceSynchronize();
                pufferl.train_captured = true;
            }
            pufferl.train_warmup++;
        }
        profile_end(hypers.profile);

        // This version is consistent with PufferLib 3.0. One of the major algorithmic
        // questions remaining is how and when to update value and advantage estimates.
        {
            int num_idx = numel(pufferl.prio_bufs.idx.shape);
            int row_bytes = (numel(graph.mb_ratio.shape) / graph.mb_ratio.shape[0]) * sizeof(precision_t);
            index_copy<<<grid_size(num_idx), BLOCK_SIZE, 0, train_stream>>>(
                (char*)rollouts.ratio.data, pufferl.prio_bufs.idx.data,
                (const char*)graph.mb_ratio.data, num_idx, row_bytes);
        }
        {
            int num_idx = numel(pufferl.prio_bufs.idx.shape);
            int row_bytes = graph.mb_newvalue.shape[1] * sizeof(precision_t);
            index_copy<<<grid_size(num_idx), BLOCK_SIZE, 0, train_stream>>>(
                (char*)rollouts.values.data, pufferl.prio_bufs.idx.data,
                (const char*)graph.mb_newvalue.data, num_idx, row_bytes);
        }
        cudaEventRecord(pufferl.profile.events[4]);  // end forward
    }
    pufferl.epoch += 1;

    cudaStreamSynchronize(pufferl.default_stream);

    if (total_minibatches > 0) {
        float ms;
        // Pre-loop setup (transpose, advantage, allocs)
        cudaEventElapsedTime(&ms, pufferl.profile.events[0], pufferl.profile.events[1]);
        pufferl.profile.accum[PROF_TRAIN_MISC] += ms;
        // In-loop misc (last iteration, representative) scaled by count
        cudaEventElapsedTime(&ms, pufferl.profile.events[2], pufferl.profile.events[3]);
        pufferl.profile.accum[PROF_TRAIN_MISC] += ms * total_minibatches;
        // In-loop forward (last iteration, representative) scaled by count
        cudaEventElapsedTime(&ms, pufferl.profile.events[3], pufferl.profile.events[4]);
        pufferl.profile.accum[PROF_TRAIN_FORWARD] += ms * total_minibatches;
    }

}

// Allocate a fresh frozen WeightBank with its own Policy (may differ in
// hidden_size/num_layers from primary). slice_size = how many agents per buffer
// this bank will own. Weights are uninitialized — caller must load before use.
static void weight_bank_create_for_pufferl(WeightBank* bank, PuffeRL* pufferl,
        int slice_size, int hidden_size, int num_layers) {
    int num_buffers = pufferl->hypers.num_buffers;

    // Rebuild arch-varying Policy from env metadata already on pufferl.
    int input_size = pufferl->env.obs.shape[1];
    int num_action_heads = pufferl->env.actions.shape[1];
    int act_n = 0;
    for (int i = 0; i < num_action_heads; i++) act_n += puf_act_sizes[i];
    int decoder_output_size = pufferl->is_continuous ? num_action_heads : act_n;
    bank->policy = build_policy(pufferl->env_name, input_size, hidden_size,
        num_layers, decoder_output_size, act_n, pufferl->is_continuous, pufferl->hypers.horizon);
    bank->hidden_size = hidden_size;
    bank->num_layers = num_layers;

    Allocator* params = &bank->params_alloc;
    Allocator* acts = &bank->acts_alloc;

    bank->slice_size = slice_size;
    bank->weights = policy_weights_create(&bank->policy, params);
    bank->buffer_activations = (PolicyActivations*)calloc(num_buffers, sizeof(PolicyActivations));
    bank->buffer_states = (PrecisionTensor*)calloc(num_buffers, sizeof(PrecisionTensor));
    for (int i = 0; i < num_buffers; i++) {
        bank->buffer_activations[i] = policy_reg_rollout(&bank->policy, bank->weights, acts, slice_size);
        bank->buffer_states[i] = {.shape = {num_layers, slice_size, hidden_size}};
        alloc_register(acts, &bank->buffer_states[i]);
    }

    alloc_create(params);
    alloc_create(acts);

    bank->param_puf = {.data = (precision_t*)params->mem, .shape = {params->total_elems}};
    if (USE_BF16) {
        bank->master_weights = {.shape = {params->total_elems}};
        cudaMalloc(&bank->master_weights.data, params->total_elems * sizeof(float));
    } else {
        bank->master_weights = {.data = (float*)bank->param_puf.data, .shape = {params->total_elems}};
    }
}

// Mirror of weight_bank_create_for_pufferl. Frees the bank's weights, per-buffer
// activations, allocators, and master_weights (BF16 only). Does not free the
// WeightBank struct itself — caller owns that.
static void weight_bank_destroy(WeightBank* bank, PuffeRL* pufferl) {
    int num_buffers = pufferl->hypers.num_buffers;
    policy_weights_free(&bank->policy, &bank->weights);
    if (bank->buffer_activations != NULL) {
        for (int i = 0; i < num_buffers; i++) {
            policy_activations_free(&bank->policy, bank->buffer_activations[i]);
        }
        free(bank->buffer_activations);
    }
    free(bank->buffer_states);
    alloc_free(&bank->params_alloc);
    alloc_free(&bank->acts_alloc);
    if (USE_BF16 && bank->master_weights.data != NULL) {
        cudaFree(bank->master_weights.data);
    }
}

// Append a fresh frozen bank with the given per-buffer slice size. Must be
// called before cudagraph capture.
int pufferl_add_frozen_bank(PuffeRL* pufferl, int slice_size,
        int hidden_size, int num_layers) {
    int idx = pufferl->num_frozen_banks;
    pufferl->frozen_banks = (WeightBank*)realloc(
        pufferl->frozen_banks, (idx + 1) * sizeof(WeightBank));
    memset(&pufferl->frozen_banks[idx], 0, sizeof(WeightBank));
    weight_bank_create_for_pufferl(&pufferl->frozen_banks[idx], pufferl,
        slice_size, hidden_size, num_layers);
    pufferl->num_frozen_banks++;
    return idx;
}

// Load a frozen bank's weights from a file (same format as save_weights — flat fp32).
// Safe to call between rollouts (in-place cudaMemcpy; cudagraphs hold the pointer,
// not a copy of the data).
void pufferl_load_frozen_bank(PuffeRL* pufferl, int bank_idx, const char* path) {
    if (bank_idx < 0 || bank_idx >= pufferl->num_frozen_banks) {
        fprintf(stderr, "pufferl_load_frozen_bank: bank_idx %d out of range\n", bank_idx);
        exit(1);
    }
    WeightBank* bank = &pufferl->frozen_banks[bank_idx];
    puf_load_weights_into(bank->master_weights, bank->param_puf,
        pufferl->default_stream, path);
    cudaDeviceSynchronize();
}

static bool create_allocator_or_report(const char* name, Allocator* alloc) {
    cudaError_t err = alloc_create(alloc);
    if (err == cudaSuccess) {
        return true;
    }

    fprintf(stderr, "create_pufferl: alloc_create(%s) failed for %ld bytes: %s\n",
        name, alloc->total_bytes, cudaGetErrorString(err));
    return false;
}

PuffeRL* create_pufferl_impl(HypersT& hypers, Dict* vec_kwargs,
        Dict* env_kwargs, ncclUniqueId* nccl_id) {
    PuffeRL* pufferl = new PuffeRL();
    pufferl->hypers = hypers;
    pufferl->nccl_comm = nullptr;
    pufferl->default_stream = 0;
    snprintf(pufferl->env_name, sizeof(pufferl->env_name), "%s", PUFFER_STRINGIFY(ENV_NAME));

    cudaSetDevice(hypers.gpu_id);

    // Multi-GPU: initialize NCCL
    if (hypers.world_size > 1) {
        ncclCommInitRank(&pufferl->nccl_comm, hypers.world_size, *nccl_id, hypers.rank);
        printf("Rank %d/%d: NCCL initialized\n", hypers.rank, hypers.world_size);
    }

    ulong seed = hypers.seed + hypers.rank;
    pufferl->seed = seed;

    // Load environment first to get input_size and action info from env
    // Create environments and set up action sizes
    VecEnv* vec = create_environments(vec_kwargs, env_kwargs, pufferl->env);
    pufferl->vec = vec;
    pufferl->bank_layout = vec->bank_layout;

    // Sanity check action space
    int num_action_heads = pufferl->env.actions.shape[1];
    int act_n = 0;
    int num_continuous = 0;
    int num_discrete = 0;
    for (int i = 0; i < num_action_heads; i++) {
        int val = puf_act_sizes[i];
        if (val == 1) {
            num_continuous++;
        } else {
            num_discrete++;
        }
        act_n += val;
    }
    assert((num_continuous == 0 || num_discrete == 0) &&
        "Mixed continuous/discrete action spaces not supported");
    pufferl->is_continuous = (num_continuous > 0);
    if (pufferl->is_continuous) {
        printf("Detected continuous action space with %d dimensions\n", num_action_heads);
    } else {
        printf("Detected discrete action space with %d heads\n", num_action_heads);
    }

    // Create profiling events
    for (int i = 0; i < NUM_TRAIN_EVENTS; i++) {
        cudaEventCreate(&pufferl->profile.events[i]);
    }
    memset(pufferl->profile.accum, 0, sizeof(pufferl->profile.accum));
    nvmlInit();
    nvmlDeviceGetHandleByIndex(hypers.gpu_id, &pufferl->nvml_device);

    // Create policy
    int input_size = pufferl->env.obs.shape[1];
    int hidden_size = hypers.hidden_size;
    int num_layers = hypers.num_layers;
    bool is_continuous = pufferl->is_continuous;
    int decoder_output_size = is_continuous ? num_action_heads : act_n;
    int minibatch_segments = hypers.minibatch_size / hypers.horizon;
    int inf_batch = vec->total_agents / hypers.num_buffers;
    int B_TT = minibatch_segments * hypers.horizon;
    int horizon = hypers.horizon;
    int total_agents = vec->total_agents;
    int batch = total_agents / hypers.num_buffers;
    int num_buffers = hypers.num_buffers;

    pufferl->policy = build_policy(pufferl->env_name, input_size, hidden_size,
        num_layers, decoder_output_size, act_n, is_continuous, hypers.horizon);

    // Create and allocate params
    Allocator* params = &pufferl->params_alloc;
    Allocator* acts = &pufferl->activations_alloc;
    Allocator* grads = &pufferl->grads_alloc;

    // Buffers for weights, grads, and activations
    pufferl->weights = policy_weights_create(&pufferl->policy, params);
    pufferl->train_activations = policy_reg_train(&pufferl->policy, pufferl->weights, acts, grads, B_TT);
    pufferl->buffer_activations = (PolicyActivations*)calloc(num_buffers, sizeof(PolicyActivations));
    pufferl->buffer_states = (PrecisionTensor*)calloc(num_buffers, sizeof(PrecisionTensor));
    for (int i = 0; i < num_buffers; i++) {
        pufferl->buffer_activations[i] = policy_reg_rollout(
            &pufferl->policy, pufferl->weights, acts, inf_batch);
        pufferl->buffer_states[i] = {
            .shape = {num_layers, batch, hidden_size},
        };
        alloc_register(acts, &pufferl->buffer_states[i]);
    }
    int mask_size = pufferl->vec->action_mask_size;
    register_rollout_buffers(pufferl->rollouts,
        acts, horizon, total_agents, input_size, num_action_heads, mask_size);
    register_train_buffers(pufferl->train_buf,
        acts, minibatch_segments, horizon, input_size,
        hidden_size, num_action_heads, num_layers, mask_size);
    register_rollout_buffers(pufferl->train_rollouts,
        acts, total_agents, horizon, input_size, num_action_heads, mask_size);
    register_ppo_buffers(pufferl->ppo_bufs_puf,
        acts, minibatch_segments, hypers.horizon, decoder_output_size, is_continuous);
    register_prio_buffers(pufferl->prio_bufs,
        acts, hypers.total_agents, minibatch_segments);

    // Extra cuda buffers just reuse activ allocator
    pufferl->rng_offset_puf = {.shape = {num_buffers + 1}};
    alloc_register(acts, &pufferl->rng_offset_puf);

    pufferl->act_sizes_puf  = {.shape = {num_action_heads}};
    alloc_register(acts, &pufferl->act_sizes_puf);

    pufferl->losses_puf = {.shape = {NUM_LOSSES}};
    alloc_register(acts, &pufferl->losses_puf);

    pufferl->advantages_puf = {.shape = {total_agents, horizon}};
    alloc_register(acts, &pufferl->advantages_puf);

    muon_init(&pufferl->muon, params, hypers.lr, hypers.beta1, hypers.eps, 0.0, acts);

    // All buffers allocated here
    if (!create_allocator_or_report("params", params)) {
        return nullptr;
    }
    if (!create_allocator_or_report("grads", grads)) {
        return nullptr;
    }
    if (!create_allocator_or_report("acts", acts)) {
        return nullptr;
    }

    pufferl->grad_puf = {.data = (precision_t*)grads->mem, .shape = {grads->total_elems}};
    pufferl->param_puf = {.data = (precision_t*)params->mem, .shape = {params->total_elems}};

    ulong init_seed = hypers.seed;
    policy_init_weights(&pufferl->policy, pufferl->weights, &init_seed, pufferl->default_stream);
    pufferl->master_weights = {.data = (float*)pufferl->param_puf.data, .shape = {params->total_elems}};
    if (USE_BF16) {
        pufferl->master_weights = {.shape = {params->total_elems}};
        cudaMalloc(&pufferl->master_weights.data, params->total_elems * sizeof(float));
        int n = numel(pufferl->param_puf.shape);
        cast<<<grid_size(n), BLOCK_SIZE, 0, pufferl->default_stream>>>(
            pufferl->master_weights.data, pufferl->param_puf.data, n);
    }

    // Per-buffer persistent RNG states
    int agents_per_buf = total_agents / num_buffers;
    pufferl->rng_states = (curandStatePhilox4_32_10_t**)calloc(num_buffers, sizeof(curandStatePhilox4_32_10_t*));
    for (int i = 0; i < num_buffers; i++) {
        cudaMalloc(&pufferl->rng_states[i], agents_per_buf * sizeof(curandStatePhilox4_32_10_t));
        rng_init<<<grid_size(agents_per_buf), BLOCK_SIZE>>>(
            pufferl->rng_states[i], pufferl->seed + i, agents_per_buf);
    }

    // Post-create initialization
    cudaMemcpy(pufferl->act_sizes_puf.data, puf_act_sizes, num_action_heads * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(pufferl->losses_puf.data, 0, NUM_LOSSES * sizeof(float));
    float one = 1.0f;
    cudaMemcpy(pufferl->ppo_bufs_puf.grad_loss.data, &one, sizeof(float), cudaMemcpyHostToDevice);
    muon_post_create(&pufferl->muon);

    // Set up frozen banks from the layout computed by VecEnv.
    // Must happen before cudagraph capture so graph launch counts are fixed.
    int num_frozen = vec->num_banks - 1;
    int frozen_hidden = hidden_size;
    int frozen_layers = num_layers;
    for (int i = 0; i < vec_kwargs->size; i++) {
        if (strcmp(vec_kwargs->items[i].key, "frozen_bank_hidden_size") == 0) {
            int value = (int)vec_kwargs->items[i].value;
            if (value > 0) {
                frozen_hidden = value;
            }
        }
        if (strcmp(vec_kwargs->items[i].key, "frozen_bank_num_layers") == 0) {
            int value = (int)vec_kwargs->items[i].value;
            if (value > 0) {
                frozen_layers = value;
            }
        }
    }
    if (num_frozen > 0) {
        for (int b = 0; b < num_frozen; b++) {
            int frozen_size = vec->bank_layout[b + 2] - vec->bank_layout[b + 1];
            if (frozen_size <= 0) {
                fprintf(stderr, "create_pufferl: frozen bank %d has no agents\n", b);
                return nullptr;
            }
            pufferl_add_frozen_bank(pufferl, frozen_size, frozen_hidden, frozen_layers);
        }
    }

    // Cudagraph rolluts and entire training step
    if (hypers.cudagraphs >= 0) {
#ifdef PUFFER_CUDA_ENV
        // fbr: Capture runs before the normal post-create reset, so seed the
        // device state before recording the full-horizon graphs.
        vec_reset(vec);
        pufferl->fused_rollout_cudagraphs =
            (cudaGraphExec_t*)calloc(num_buffers, sizeof(cudaGraphExec_t));
#else
        pufferl->fused_rollout_cudagraphs =
            (cudaGraphExec_t*)calloc(horizon*num_buffers, sizeof(cudaGraphExec_t));
#endif
        pufferl->train_warmup = 0;

        // Snapshot weights + optimizer state before init-time capture
        long wb_bytes = numel(pufferl->master_weights.shape) * sizeof(float);
        void* saved_weights;
        cudaMalloc(&saved_weights, wb_bytes);
        cudaMemcpy(saved_weights, pufferl->master_weights.data, wb_bytes, cudaMemcpyDeviceToDevice);
        void* saved_momentum;
        cudaMalloc(&saved_momentum, wb_bytes);
        cudaMemcpy(saved_momentum, pufferl->muon.mb_puf.data, wb_bytes, cudaMemcpyDeviceToDevice);

        // Create per-buffer streams before capture so graphs are
        // captured and replayed on the same streams.
        pufferl->streams = (cudaStream_t*)calloc(num_buffers, sizeof(cudaStream_t));
        for (int i = 0; i < num_buffers; i++) {
            cudaStreamCreate(&pufferl->streams[i]);
            vec->streams[i] = pufferl->streams[i];
        }

        cudaStream_t saved_default = pufferl->default_stream;
        cudaStream_t warmup_stream;
        cudaStreamCreate(&warmup_stream);
        pufferl->default_stream = warmup_stream;

#ifdef PUFFER_CUDA_ENV
        // fbr: Warm every library/kernel path before capture. Then record all
        // horizon timesteps into one graph per buffer. Timestep-specific rollout
        // pointers become fixed graph-node arguments, while stream ordering
        // carries recurrent state and environment outputs through the horizon.
        for (pufferl->epoch = 0; pufferl->epoch < hypers.cudagraphs; pufferl->epoch++) {
            for (int buf = 0; buf < num_buffers; ++buf) {
                pufferl_rollout_buffer(pufferl, buf, pufferl->streams[buf]);
            }
            cudaDeviceSynchronize();
        }
        for (int buf = 0; buf < num_buffers; ++buf) {
            cudaStream_t stream = pufferl->streams[buf];
            assert(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)
                    == cudaSuccess && "full-horizon cudaStreamBeginCapture failed");
            for (int t = 0; t < horizon; ++t) {
                pufferl_forward(pufferl, buf, t, stream);
            }
            cudaGraph_t graph;
            assert(cudaStreamEndCapture(stream, &graph) == cudaSuccess
                    && "full-horizon cudaStreamEndCapture failed");
            assert(cudaGraphInstantiate(&pufferl->fused_rollout_cudagraphs[buf],
                    graph, 0) == cudaSuccess
                    && "full-horizon cudaGraphInstantiate failed");
            assert(cudaGraphDestroy(graph) == cudaSuccess
                    && "full-horizon cudaGraphDestroy failed");
        }
        cudaDeviceSynchronize();
#else
        for (pufferl->epoch = 0; pufferl->epoch <= hypers.cudagraphs; pufferl->epoch++) {
            for (int i = 0; i < num_buffers * horizon; ++i) {
                int buf = i % num_buffers;
                pufferl_forward(pufferl, buf, i / num_buffers, pufferl->streams[buf]);
                cudaDeviceSynchronize();
            }
        }
#endif
        pufferl->rollout_captured = true;

        for (int i = 0; i <= hypers.cudagraphs; i++) {
            train_impl(*pufferl);
        }

        cudaStreamSynchronize(warmup_stream);
        cudaDeviceSynchronize();
        pufferl->default_stream = saved_default;
        cudaStreamDestroy(warmup_stream);

        // Restore weights + optimizer state corrupted by warmup/capture
        cudaMemcpy(pufferl->master_weights.data, saved_weights, wb_bytes, cudaMemcpyDeviceToDevice);
        cudaFree(saved_weights);
        cudaMemcpy(pufferl->muon.mb_puf.data, saved_momentum, wb_bytes, cudaMemcpyDeviceToDevice);
        cudaFree(saved_momentum);
        if (USE_BF16) {
            int n = numel(pufferl->param_puf.shape);
            cast<<<grid_size(n), BLOCK_SIZE, 0, pufferl->default_stream>>>(
                pufferl->param_puf.data, pufferl->master_weights.data, n);
        }

        // Re-init RNG states corrupted by warmup
        for (int i = 0; i < num_buffers; i++) {
            rng_init<<<grid_size(agents_per_buf), BLOCK_SIZE>>>(
                pufferl->rng_states[i], pufferl->seed + i, agents_per_buf);
        }
        cudaDeviceSynchronize();

        pufferl->epoch = 0;
        pufferl->global_step = 0;
    }

    // Create per-buffer streams if not already created by cudagraph path
    if (!pufferl->streams) {
        pufferl->streams = (cudaStream_t*)calloc(num_buffers, sizeof(cudaStream_t));
        for (int i = 0; i < num_buffers; i++) {
            cudaStreamCreate(&pufferl->streams[i]);
            vec->streams[i] = pufferl->streams[i];
        }
    }

    vec_create_threads(vec, hypers.num_threads, horizon, pufferl);
    vec_reset(vec);

    if (hypers.profile) {
        cudaDeviceSynchronize();
        cudaProfilerStart();
    }

    double now = wall_clock();
    pufferl->start_time = now;
    pufferl->last_log_time = now;
    pufferl->last_log_step = 0;

    return pufferl;
}

void close_impl(PuffeRL& pufferl) {
    cudaDeviceSynchronize();
    if (pufferl.hypers.profile) {
        cudaProfilerStop();
    }

    if (pufferl.train_cudagraph) {
        cudaGraphExecDestroy(pufferl.train_cudagraph);
    }
    if (pufferl.fused_rollout_cudagraphs) {
#ifdef PUFFER_CUDA_ENV
        int rollout_graph_count = pufferl.hypers.num_buffers;
#else
        int rollout_graph_count =
            pufferl.hypers.horizon * pufferl.hypers.num_buffers;
#endif
        for (int i = 0; i < rollout_graph_count; i++) {
            if (pufferl.fused_rollout_cudagraphs[i]) {
                cudaGraphExecDestroy(pufferl.fused_rollout_cudagraphs[i]);
            }
        }
    }

    policy_weights_free(&pufferl.policy, &pufferl.weights);
    policy_activations_free(&pufferl.policy, pufferl.train_activations);
    for (int buf = 0; buf < pufferl.hypers.num_buffers; buf++) {
        policy_activations_free(&pufferl.policy, pufferl.buffer_activations[buf]);
    }

    for (int i = 0; i < pufferl.hypers.num_buffers; i++) {
        cudaFree(pufferl.rng_states[i]);
    }
    free(pufferl.rng_states);

    if (USE_BF16) {
        cudaFree(pufferl.master_weights.data);
    }

    alloc_free(&pufferl.params_alloc);
    alloc_free(&pufferl.grads_alloc);
    alloc_free(&pufferl.activations_alloc);

    for (int i = 0; i < pufferl.hypers.num_buffers; i++) {
        cudaStreamDestroy(pufferl.streams[i]);
    }
    for (int i = 0; i < NUM_TRAIN_EVENTS; i++) {
        cudaEventDestroy(pufferl.profile.events[i]);
    }
    nvmlShutdown();

    vec_close(pufferl.vec);

    free(pufferl.buffer_states);
    free(pufferl.buffer_activations);
    free(pufferl.fused_rollout_cudagraphs);
    free(pufferl.streams);

    for (int b = 0; b < pufferl.num_frozen_banks; b++) {
        weight_bank_destroy(&pufferl.frozen_banks[b], &pufferl);
    }
    free(pufferl.frozen_banks);

    if (pufferl.nccl_comm != nullptr) {
        ncclCommDestroy(pufferl.nccl_comm);
    }
}

#ifdef PUFFERLIB_BUILD_MAIN

static double puf_log_get_or(Dict* dict, const char* key, double fallback) {
    for (int i = 0; i < dict->size; i++) {
        if (strcmp(dict->items[i].key, key) == 0) {
            return dict->items[i].value;
        }
    }
    return fallback;
}

static int puf_dashboard_tty = 0;
static int puf_dashboard_last_rows = 0;
static int puf_dashboard_last_cols = 0;

#define PUF_DASH_WIDTH 80
#define PUF_DASH_BASE_ROWS 14
#define PUF_DASH_MAX_USER_ROWS 15

static void puf_term_size(int* rows, int* cols) {
    *rows = 1000;
    *cols = PUF_DASH_WIDTH;

    if (!puf_dashboard_tty) {
        return;
    }

    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
        if (ws.ws_row > 0) {
            *rows = ws.ws_row;
        }
        if (ws.ws_col > 0) {
            *cols = ws.ws_col;
        }
    }
}

static void puf_dashboard_begin(int rows, int cols) {
    if (!puf_dashboard_tty) {
        return;
    }

    printf("\033[?2026h");
    if (rows != puf_dashboard_last_rows || cols != puf_dashboard_last_cols) {
        printf("\033[H\033[J");
    } else {
        printf("\033[H");
    }

    puf_dashboard_last_rows = rows;
    puf_dashboard_last_cols = cols;
}

static void puf_dashboard_end(void) {
    if (puf_dashboard_tty) {
        printf("\033[J\033[?2026l");
    }
    fflush(stdout);
}

static const char* puf_cyan(void) {
    return puf_dashboard_tty ? "\033[36m" : "";
}

static const char* puf_bcyan(void) {
    return puf_dashboard_tty ? "\033[96m" : "";
}

static const char* puf_white(void) {
    return puf_dashboard_tty ? "\033[37m" : "";
}

static const char* puf_bwhite(void) {
    return puf_dashboard_tty ? "\033[97m" : "";
}

static const char* puf_ansi_reset(void) {
    return puf_dashboard_tty ? "\033[0m" : "";
}

static void puf_dashboard_eol(void) {
    if (puf_dashboard_tty) {
        printf("\033[K");
    }
    putchar('\n');
}

static void puf_abbrev(char* out, size_t out_len, double val) {
    const char* suffix[] = {"", "K", "M", "B", "T"};
    int i = 0;
    while (val >= 1000.0 && i < 4) {
        val /= 1000.0;
        i++;
    }
    snprintf(out, out_len, "%.1f%s", val, suffix[i]);
}

static void puf_duration(char* out, size_t out_len, double seconds) {
    if (seconds < 0) {
        seconds = 0;
    }
    if (seconds < 1.0) {
        snprintf(out, out_len, "%.0fms", seconds * 1000.0);
        return;
    }

    long s = (long)seconds;
    snprintf(out, out_len, "%ldd %ldh %ldm %lds",
        s / 86400, (s / 3600) % 24, (s / 60) % 60, s % 60);
}

static void puf_perf_value(char* time_out, size_t time_len, char* pct_out, size_t pct_len,
        double part, double total) {
    int pct = total > 0 ? (int)(100.0 * part / total) : 0;
    puf_duration(time_out, time_len, part);
    snprintf(pct_out, pct_len, "%d%%", pct);
}

static void puf_strip_prefix(char* out, size_t out_len, const char* key, const char* prefix) {
    size_t n = strlen(prefix);
    if (strncmp(key, prefix, n) == 0) {
        snprintf(out, out_len, "%s", key + n);
    } else {
        snprintf(out, out_len, "%s", key);
    }
}

static int puf_loss_value(Dict* log, const char* key, char* out, size_t out_len) {
    for (int i = 0; i < log->size; i++) {
        if (strcmp(log->items[i].key, key) == 0) {
            snprintf(out, out_len, "%.3f", log->items[i].value);
            return 1;
        }
    }
    if (out_len > 0) {
        out[0] = 0;
    }
    return 0;
}

static void puf_panel_header(const char* eval_t, const char* eval_pct) {
    printf("%s│", puf_bcyan());
    printf("%s %-9.9s %13.13s%s    %s%-12.12s%s %s%6.6s %4.4s%s    %s%-10.10s %7.7s%s    ",
        puf_cyan(), "Summary", "Value", puf_ansi_reset(),
        puf_bcyan(), "Evaluate", puf_ansi_reset(), puf_bwhite(), eval_t, eval_pct, puf_ansi_reset(),
        puf_cyan(), "Losses", "Value", puf_ansi_reset());
    printf("%s│%s", puf_bcyan(), puf_ansi_reset());
    puf_dashboard_eol();
}

static void puf_panel_row(const char* s_name, const char* s_val,
        const char* p_name, const char* p_time, const char* p_pct,
        const char* l_name, const char* l_val, int emph_perf) {
    const char* perf_color = emph_perf ? puf_bcyan() : puf_bwhite();
    printf("%s│", puf_bcyan());
    printf("%s %s%-9.9s%s %s%13.13s%s    %s%-12.12s%s %s%6.6s %4.4s%s    %s%-10.10s %7.7s%s    ",
        puf_ansi_reset(),
        puf_white(), s_name, puf_ansi_reset(), puf_bwhite(), s_val, puf_ansi_reset(),
        perf_color, p_name, puf_ansi_reset(), puf_bwhite(), p_time, p_pct, puf_ansi_reset(),
        puf_bwhite(), l_name, l_val, puf_ansi_reset());
    printf("%s│%s", puf_bcyan(), puf_ansi_reset());
    puf_dashboard_eol();
}

static void puf_user_header(void) {
    printf("%s│", puf_bcyan());
    printf("%s %-23.23s %9.9s%s   %s%-23.23s %9.9s%s        ",
        puf_cyan(), "User Stats", "Value", puf_ansi_reset(),
        puf_cyan(), "User Stats", "Value", puf_ansi_reset());
    printf("%s│%s", puf_bcyan(), puf_ansi_reset());
    puf_dashboard_eol();
}

static void puf_user_row(const char* left_key, double left_val,
        const char* right_key, double right_val, int has_right) {
    char left_s[32];
    char right_s[32];
    snprintf(left_s, sizeof(left_s), "%.3f", left_val);
    snprintf(right_s, sizeof(right_s), "%.3f", right_val);

    printf("%s│", puf_bcyan());
    if (has_right) {
        printf("%s %s%-23.23s %9.9s%s   %s%-23.23s %9.9s%s        ",
            puf_ansi_reset(),
            puf_bwhite(), left_key, left_s, puf_ansi_reset(),
            puf_bwhite(), right_key, right_s, puf_ansi_reset());
    } else {
        printf("%s %s%-23.23s %9.9s%s   %-23.23s %9.9s        ",
            puf_ansi_reset(),
            puf_bwhite(), left_key, left_s, puf_ansi_reset(), "", "");
    }
    printf("%s│%s", puf_bcyan(), puf_ansi_reset());
    puf_dashboard_eol();
}

static void puf_dashboard_blank(void) {
    printf("%s│%*s│%s", puf_bcyan(), PUF_DASH_WIDTH - 2, "", puf_ansi_reset());
    puf_dashboard_eol();
}

static void puf_dashboard_rule(const char* left, const char* right) {
    printf("%s%s", puf_bcyan(), left);
    for (int i = 0; i < PUF_DASH_WIDTH - 2; i++) {
        printf("─");
    }
    printf("%s%s", right, puf_ansi_reset());
    puf_dashboard_eol();
}

static void puf_dashboard_print(Config* cfg, PuffeRL* p, Dict* log, int epoch) {
    puf_dashboard_tty = isatty(STDOUT_FILENO);
    int term_rows = 1000;
    int term_cols = PUF_DASH_WIDTH;
    puf_term_size(&term_rows, &term_cols);

    const char* env_name = puf_config_str(cfg, "base", "env_name");
    double steps = puf_log_get_or(log, "agent_steps", (double)p->global_step);
    double sps = puf_log_get_or(log, "SPS", 0);
    double target_steps = puf_config_get(cfg, "train", "total_timesteps");
    double remaining_sec = sps > 0 ? (target_steps - steps) / sps : 0;
    double rollout = puf_log_get_or(log, "perf/rollout", 0);
    double train_time = puf_log_get_or(log, "perf/train", 0);
    double perf_total = rollout + train_time;

    char params[32];
    char steps_s[32];
    char sps_s[32];
    char uptime[64];
    char remaining[64];
    puf_abbrev(params, sizeof(params), (double)numel(p->master_weights.shape));
    puf_abbrev(steps_s, sizeof(steps_s), steps);
    puf_abbrev(sps_s, sizeof(sps_s), sps);
    puf_duration(uptime, sizeof(uptime), puf_log_get_or(log, "uptime", 0));
    puf_duration(remaining, sizeof(remaining), remaining_sec);

    char epoch_s[32];
    snprintf(epoch_s, sizeof(epoch_s), "%d", epoch);

    puf_dashboard_begin(term_rows, term_cols);
    if (puf_dashboard_tty && (term_cols < PUF_DASH_WIDTH || term_rows <= PUF_DASH_BASE_ROWS)) {
        char compact[512];
        snprintf(compact, sizeof(compact),
            "PufferLib 4.0  env=%s  steps=%s  SPS=%s  score=%.3f  epoch=%s  to_go=%s",
            env_name, steps_s, sps_s, puf_log_get_or(log, "env/score", 0), epoch_s, remaining);
        int max_cols = term_cols > 1 ? term_cols - 1 : term_cols;
        printf("%.*s", max_cols, compact);
        if (term_rows > 1) {
            puf_dashboard_eol();
        } else if (puf_dashboard_tty) {
            printf("\033[K");
        }
        puf_dashboard_end();
        return;
    }

    puf_dashboard_rule("╭", "╮");
    printf("%s│", puf_bcyan());
    printf("%s %sPufferLib %s4.0%s        %s🐡%s        %sGPU:%s %2.0f%%    %sVRAM:%s %.1f/%.0fG    %sRAM:%s %.1fG        ",
        puf_ansi_reset(),
        puf_bcyan(), puf_bwhite(), puf_ansi_reset(),
        puf_bcyan(), puf_ansi_reset(),
        puf_cyan(), puf_bwhite(),
        puf_log_get_or(log, "util/gpu_percent", 0),
        puf_cyan(), puf_bwhite(),
        puf_log_get_or(log, "util/vram_used_gb", 0),
        puf_log_get_or(log, "util/vram_total_gb", 0),
        puf_cyan(), puf_bwhite(),
        puf_log_get_or(log, "util/cpu_mem_gb", 0));
    printf("%s│%s", puf_bcyan(), puf_ansi_reset());
    puf_dashboard_eol();
    puf_dashboard_blank();

    char eval_t[64];
    char eval_pct[16];
    char gpu_t[64];
    char gpu_pct[16];
    char env_t[64];
    char env_pct[16];
    char train_t[64];
    char train_pct[16];
    char misc_t[64];
    char misc_pct[16];
    char forward_t[64];
    char forward_pct[16];
    char loss_policy[32];
    char loss_value[32];
    char loss_entropy[32];
    char loss_total[32];
    char loss_old_kl[32];
    char loss_kl[32];
    char loss_clipfrac[32];
    puf_perf_value(eval_t, sizeof(eval_t), eval_pct, sizeof(eval_pct), rollout, perf_total);
    puf_perf_value(gpu_t, sizeof(gpu_t), gpu_pct, sizeof(gpu_pct),
        puf_log_get_or(log, "perf/eval_gpu", 0), perf_total);
    puf_perf_value(env_t, sizeof(env_t), env_pct, sizeof(env_pct),
        puf_log_get_or(log, "perf/eval_env", 0), perf_total);
    puf_perf_value(train_t, sizeof(train_t), train_pct, sizeof(train_pct), train_time, perf_total);
    puf_perf_value(misc_t, sizeof(misc_t), misc_pct, sizeof(misc_pct),
        puf_log_get_or(log, "perf/train_misc", 0), perf_total);
    puf_perf_value(forward_t, sizeof(forward_t), forward_pct, sizeof(forward_pct),
        puf_log_get_or(log, "perf/train_forward", 0), perf_total);
    puf_loss_value(log, "loss/policy", loss_policy, sizeof(loss_policy));
    puf_loss_value(log, "loss/value", loss_value, sizeof(loss_value));
    puf_loss_value(log, "loss/entropy", loss_entropy, sizeof(loss_entropy));
    puf_loss_value(log, "loss/total", loss_total, sizeof(loss_total));
    puf_loss_value(log, "loss/old_kl", loss_old_kl, sizeof(loss_old_kl));
    puf_loss_value(log, "loss/kl", loss_kl, sizeof(loss_kl));
    puf_loss_value(log, "loss/clipfrac", loss_clipfrac, sizeof(loss_clipfrac));

    puf_panel_header(eval_t, eval_pct);
    puf_panel_row("Env", env_name, "  GPU", gpu_t, gpu_pct, "policy", loss_policy, 0);
    puf_panel_row("Params", params, "  Env", env_t, env_pct, "value", loss_value, 0);
    puf_panel_row("Steps", steps_s, "Train", train_t, train_pct, "entropy", loss_entropy, 1);
    puf_panel_row("SPS", sps_s, "  Misc", misc_t, misc_pct, "total", loss_total, 0);
    puf_panel_row("Epoch", epoch_s, "  Forward", forward_t, forward_pct, "old_kl", loss_old_kl, 0);
    puf_panel_row("Uptime", uptime, "", "", "", "kl", loss_kl, 0);
    puf_panel_row("To go", remaining, "", "", "", "clipfrac", loss_clipfrac, 0);
    puf_dashboard_blank();

    puf_user_header();
    int user_rows = PUF_DASH_MAX_USER_ROWS;
    if (puf_dashboard_tty) {
        user_rows = term_rows - PUF_DASH_BASE_ROWS - 1;
        if (user_rows < 0) {
            user_rows = 0;
        }
        if (user_rows > PUF_DASH_MAX_USER_ROWS) {
            user_rows = PUF_DASH_MAX_USER_ROWS;
        }
    }

    char pending_key[128];
    double pending_val = 0;
    int pending = 0;
    int n = 0;
    int max_user_items = 2 * user_rows;
    for (int i = 0; i < log->size && n < max_user_items; i++) {
        const char* key = log->items[i].key;
        if (strncmp(key, "env/", 4) != 0 || strcmp(key, "env/n") == 0) {
            continue;
        }

        char short_key[128];
        puf_strip_prefix(short_key, sizeof(short_key), key, "env/");
        if (!pending) {
            snprintf(pending_key, sizeof(pending_key), "%s", short_key);
            pending_val = log->items[i].value;
            pending = 1;
        } else {
            puf_user_row(pending_key, pending_val, short_key, log->items[i].value, 1);
            pending = 0;
        }
        n++;
    }
    if (pending) {
        puf_user_row(pending_key, pending_val, "", 0, 0);
    }
    puf_dashboard_rule("╰", "╯");
    puf_dashboard_end();
}

static void log_util(PuffeRL* p, Dict* out) {
    nvmlUtilization_t util;
    nvmlDeviceGetUtilizationRates(p->nvml_device, &util);
    dict_set(out, "util/gpu_percent", (double)util.gpu);

    size_t cuda_free;
    size_t cuda_total;
    cudaMemGetInfo(&cuda_free, &cuda_total);
    dict_set(out, "util/vram_used_gb",
        (double)(cuda_total - cuda_free) / (1024.0 * 1024.0 * 1024.0));
    dict_set(out, "util/vram_total_gb",
        (double)cuda_total / (1024.0 * 1024.0 * 1024.0));

    long rss_kb = 0;
    FILE* status = fopen("/proc/self/status", "r");
    if (status) {
        char line[256];
        while (fgets(line, sizeof(line), status)) {
            if (sscanf(line, "VmRSS: %ld", &rss_kb) == 1) {
                break;
            }
        }
        fclose(status);
    }
    dict_set(out, "util/cpu_mem_gb", (double)rss_kb / (1024.0 * 1024.0));
}

static void puf_log_env(Dict* out, Dict* env_out) {
    for (int i = 0; i < env_out->size; i++) {
        char key[256];
        snprintf(key, sizeof(key), "env/%s", env_out->items[i].key);
        dict_set(out, key, env_out->items[i].value);
    }
}

static void trainer_log(PuffeRL* p, Dict* out) {
    long global_step = p->global_step;
    double now = wall_clock();
    double dt = now - p->last_log_time;
    long sps = dt > 0 ? (long)((global_step - p->last_log_step) / dt) : 0;
    p->last_log_time = now;
    p->last_log_step = global_step;

    dict_set(out, "SPS", (double)sps * p->hypers.world_size);
    dict_set(out, "agent_steps", (double)global_step * p->hypers.world_size);
    dict_set(out, "uptime", now - p->start_time);
    dict_set(out, "epoch", (double)p->epoch);

    Dict env_out = {0};
    vec_log(p->vec, &env_out, 1);
    puf_log_env(out, &env_out);

    float losses_host[NUM_LOSSES];
    cudaMemcpy(losses_host, p->losses_puf.data, sizeof(losses_host), cudaMemcpyDeviceToHost);
    float loss_n = losses_host[LOSS_N];
    if (loss_n > 0) {
        float inv_n = 1.0f / loss_n;
        dict_set(out, "loss/policy", losses_host[LOSS_PG] * inv_n);
        dict_set(out, "loss/value", losses_host[LOSS_VF] * inv_n);
        dict_set(out, "loss/entropy", losses_host[LOSS_ENT] * inv_n);
        dict_set(out, "loss/total", losses_host[LOSS_TOTAL] * inv_n);
        dict_set(out, "loss/old_kl", losses_host[LOSS_OLD_APPROX_KL] * inv_n);
        dict_set(out, "loss/kl", losses_host[LOSS_APPROX_KL] * inv_n);
        dict_set(out, "loss/clipfrac", losses_host[LOSS_CLIPFRAC] * inv_n);
    }
    cudaMemset(p->losses_puf.data, 0, numel(p->losses_puf.shape) * sizeof(float));

    log_util(p, out);

    float train_total = 0;
    for (int i = 0; i < NUM_PROF; i++) {
        float sec = p->profile.accum[i] / 1000.0f;
        char key[256];
        snprintf(key, sizeof(key), "perf/%s", PROF_NAMES[i]);
        dict_set(out, key, sec);
        if (i >= PROF_TRAIN_MISC) {
            train_total += sec;
        }
    }
    dict_set(out, "perf/train", train_total);
    memset(p->profile.accum, 0, sizeof(p->profile.accum));
}

static void trainer_eval_log(PuffeRL* p, Dict* out) {
    double now = wall_clock();
    p->last_log_time = now;
    p->last_log_step = p->global_step;
    log_util(p, out);

    Dict env_out = {0};
    vec_log(p->vec, &env_out, 0);
    puf_log_env(out, &env_out);
}

typedef struct {
    Dict* items;
    int size;
    int capacity;
} PufLogHistory;

static DictItem* puf_log_find(Dict* dict, const char* key) {
    for (int i = 0; i < dict->size; i++) {
        if (strcmp(dict->items[i].key, key) == 0) {
            return &dict->items[i];
        }
    }
    return NULL;
}

static void puf_log_update(Dict* dst, Dict* src) {
    for (int i = 0; i < src->size; i++) {
        DictItem* item = &src->items[i];
        if (item->str) {
            dict_set_str(dst, item->key, item->str);
            puf_log_find(dst, item->key)->value = item->value;
        } else {
            dict_set(dst, item->key, item->value);
        }
    }
}

static void puf_log_history_add(PufLogHistory* history, Dict* log) {
    if (history->size == history->capacity) {
        history->capacity = history->capacity ? 2 * history->capacity : 64;
        history->items = (Dict*)realloc(history->items, (size_t)history->capacity * sizeof(Dict));
        if (!history->items) {
            perror("realloc");
            exit(1);
        }
    }

    dict_copy(&history->items[history->size], log);
    history->size++;
}

static void puf_log_history_free(PufLogHistory* history) {
    for (int i = 0; i < history->size; i++) {
        dict_clear(&history->items[i]);
    }
    free(history->items);
    memset(history, 0, sizeof(*history));
}

static void puf_log_collect_keys(PufLogHistory* history, Dict* keys) {
    for (int i = 0; i < history->size; i++) {
        Dict* log = &history->items[i];
        for (int j = 0; j < log->size; j++) {
            if (!puf_log_find(keys, log->items[j].key)) {
                dict_set(keys, log->items[j].key, 0);
            }
        }
    }
}

static double puf_log_reduce(double* vals, int n, double fallback) {
    if (n == 0) {
        return fallback;
    }

    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += vals[i];
    }
    return sum / n;
}

static void puf_log_write_metric(FILE* fp, const char* key, double* values, int n) {
    fprintf(fp, "%s = ", key);
    for (int i = 0; i < n; i++) {
        if (i > 0) {
            fputc(',', fp);
        }
        fprintf(fp, "%.17g", values[i]);
    }
    fputc('\n', fp);
}

static void puf_log_write(const char* path, Config* cfg, PufLogHistory* history) {
    if (history->size == 0) {
        fprintf(stderr, "cannot write empty log history\n");
        exit(1);
    }

    FILE* fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "failed to write log %s\n", path);
        exit(1);
    }

    fprintf(fp, "# PufferLib log v1\n");
    puf_ini_write(fp, &cfg->ini);
    fprintf(fp, "\n[metrics]\n");

    int downsample = (int)puf_config_get(cfg, "sweep", "downsample");
    Dict keys = {0};
    puf_log_collect_keys(history, &keys);
    int points = downsample <= 1 ? 1 : downsample;
    double* out = (double*)calloc((size_t)points, sizeof(double));
    double* bin = (double*)calloc((size_t)history->size, sizeof(double));
    double final_steps = dict_get(&history->items[history->size - 1], "agent_steps");

    for (int k = 0; k < keys.size; k++) {
        const char* key = keys.items[k].key;
        if (strncmp(key, "loss/", 5) == 0) {
            continue;
        }

        double first_value = 0;
        for (int i = 0; i < history->size; i++) {
            DictItem* item = puf_log_find(&history->items[i], key);
            if (item) {
                first_value = item->value;
                break;
            }
        }

        if (points == 1) {
            DictItem* item = puf_log_find(&history->items[history->size - 1], key);
            out[0] = item ? item->value : first_value;
            puf_log_write_metric(fp, key, out, points);
            continue;
        }

        int out_idx = 0;
        int bin_n = 0;
        double fallback = first_value;
        double next_bin = final_steps / (points - 1);
        for (int i = 0; i < history->size; i++) {
            Dict* log = &history->items[i];
            DictItem* item = puf_log_find(log, key);
            if (item) {
                bin[bin_n++] = item->value;
            }

            double steps = dict_get(log, "agent_steps");
            if (steps < next_bin || out_idx >= points - 1) {
                continue;
            }

            double reduced = puf_log_reduce(bin, bin_n, fallback);
            out[out_idx++] = reduced;
            fallback = reduced;
            bin_n = 0;
            next_bin += final_steps / (points - 1);
        }

        DictItem* final_item = puf_log_find(&history->items[history->size - 1], key);
        out[points - 1] = final_item ? final_item->value : puf_log_reduce(bin, bin_n, fallback);
        while (out_idx < points - 1) {
            out[out_idx++] = fallback;
        }
        puf_log_write_metric(fp, key, out, points);
    }

    free(bin);
    free(out);
    fclose(fp);
}

typedef struct {
    int rank;
    int world_size;
    int gpu_id;
    int artifact_owner;
    ncclUniqueId* nccl_id;
} TrainContext;

static PuffeRL* create_trainer(Config* cfg, TrainContext* ctx) {
    HypersT hypers = puf_config_to_hypers(cfg,
        ctx->rank, ctx->world_size, ctx->gpu_id);
    Dict vec = {0};
    dict_copy(&vec, puf_ini_section(&cfg->ini, "vec", 0));
    PuffeRL* pufferl = create_pufferl_impl(hypers, &vec, &cfg->env, ctx->nccl_id);
    dict_clear(&vec);
    if (!pufferl) {
        fprintf(stderr, "create_pufferl_impl failed\n");
        exit(1);
    }
    return pufferl;
}

static void rollouts(PuffeRL* p) {
    if (p->hypers.reset_state) {
        for (int i = 0; i < p->hypers.num_buffers; i++) {
            puf_zero(&p->buffer_states[i], p->default_stream);
        }
        for (int b = 0; b < p->num_frozen_banks; b++) {
            for (int i = 0; i < p->hypers.num_buffers; i++) {
                puf_zero(&p->frozen_banks[b].buffer_states[i], p->default_stream);
            }
        }
    }

    double t0 = wall_clock();
    vec_step(p->vec);
    float sec = (float)(wall_clock() - t0);
    p->profile.accum[PROF_ROLLOUT] += sec * 1000.0f;

    float eval_prof[NUM_VEC_PROF] = {0};
    for (int buf = 0; buf < p->vec->buffers; buf++) {
        float* src = &p->vec->accum[buf * NUM_VEC_PROF];
        for (int i = 0; i < NUM_VEC_PROF; i++) {
            eval_prof[i] += src[i];
        }
        memset(src, 0, NUM_VEC_PROF * sizeof(float));
    }
    p->profile.accum[PROF_EVAL_GPU] += eval_prof[VEC_GPU] / p->vec->buffers;
    p->profile.accum[PROF_EVAL_ENV] += eval_prof[VEC_ENV_STEP] / p->vec->buffers;
    p->global_step += p->hypers.horizon * p->hypers.total_agents;
}

static void close_trainer(PuffeRL* p) {
    close_impl(*p);
    delete p;
}

typedef struct {
    float score;
    float draw;
    int games;
} EvalResult;

typedef struct {
    float score;
    float cost;
    float steps;
    int points;
    char checkpoint_path[4096];
    float scores[TRAIN_RESULT_MAX_POINTS];
    float costs[TRAIN_RESULT_MAX_POINTS];
    float step_points[TRAIN_RESULT_MAX_POINTS];
} TrainResult;

#define EVAL_RENDER 0
#define EVAL_SCORE 1
#define EVAL_MATCH 2

static EvalResult run_eval(Config* cfg, TrainContext* ctx, int mode, int verbose);

#define SELFPLAY_MAX_BANKS 8
#define SELFPLAY_MAX_POOL 1024
#define SELFPLAY_PATH_MAX 4096

typedef struct {
    char pending_path[SELFPLAY_PATH_MAX];
    long opp_started_step;
    int num_envs;
} SelfplayBank;

typedef struct {
    int num_banks;
    int max_size;
    long opp_timeout_steps;
    unsigned int rng;
    char pool[SELFPLAY_MAX_POOL][SELFPLAY_PATH_MAX];
    int pool_size;
    SelfplayBank banks[SELFPLAY_MAX_BANKS];
} Selfplay;

static void selfplay_add_pool(Selfplay* sp, const char* path) {
    for (int i = 0; i < sp->pool_size; i++) {
        if (strcmp(sp->pool[i], path) == 0) {
            return;
        }
    }
    if (sp->pool_size >= SELFPLAY_MAX_POOL) {
        fprintf(stderr, "selfplay pool exceeds SELFPLAY_MAX_POOL\n");
        exit(1);
    }
    snprintf(sp->pool[sp->pool_size++], sizeof(sp->pool[0]), "%s", path);
}

static void selfplay_evict(Selfplay* sp) {
    if (sp->pool_size <= sp->max_size) {
        return;
    }

    int start = sp->pool_size - sp->max_size;
    memmove(sp->pool, sp->pool + start, (size_t)sp->max_size * sizeof(sp->pool[0]));
    sp->pool_size = sp->max_size;
}

static void selfplay_add_checkpoint(Selfplay* sp, const char* path) {
    while (access(path, R_OK) != 0) {
        usleep(50000);
    }
    selfplay_add_pool(sp, path);
    selfplay_evict(sp);
}

static const char* selfplay_sample(Selfplay* sp) {
    if (sp->pool_size == 0) {
        fprintf(stderr, "selfplay opponent pool is empty\n");
        exit(1);
    }
    int idx = (int)(rand_r(&sp->rng) % (unsigned int)sp->pool_size);
    return sp->pool[idx];
}

static int selfplay_count_aligned(PuffeRL* p, int tag) {
    Env* envs = (Env*)p->vec->envs;
    int count = 0;
    for (int i = 0; i < p->vec->size; i++) {
        if (envs[i].tag == tag && envs[i].boundary_reached) {
            count++;
        }
    }
    return count;
}

static void selfplay_clear_aligned(PuffeRL* p, int tag) {
    Env* envs = (Env*)p->vec->envs;
    for (int i = 0; i < p->vec->size; i++) {
        if (envs[i].tag == tag) {
            envs[i].boundary_reached = 0;
        }
    }
}

static void selfplay_init(Selfplay* sp, Config* cfg, PuffeRL* p,
        const char* initial_checkpoint) {
    memset(sp, 0, sizeof(*sp));
    sp->num_banks = p->num_frozen_banks;
    if (sp->num_banks <= 0 || sp->num_banks > SELFPLAY_MAX_BANKS) {
        fprintf(stderr, "selfplay requires 1..%d frozen banks\n", SELFPLAY_MAX_BANKS);
        exit(1);
    }
    sp->max_size = (int)puf_config_get(cfg, "selfplay", "max_size");
    sp->opp_timeout_steps = (long)puf_config_get(cfg, "selfplay", "opp_timeout_steps");
    sp->rng = (unsigned int)puf_config_get(cfg, "selfplay", "seed") + (unsigned int)p->hypers.rank;
    long current_step = p->global_step * p->hypers.world_size;

    Env* envs = (Env*)p->vec->envs;
    for (int i = 0; i < p->vec->size; i++) {
        int tag = envs[i].tag;
        if (tag > 0 && tag <= sp->num_banks) {
            sp->banks[tag - 1].num_envs++;
        }
    }

    selfplay_add_checkpoint(sp, initial_checkpoint);
    for (int b = 0; b < sp->num_banks; b++) {
        const char* path = selfplay_sample(sp);
        pufferl_load_frozen_bank(p, b, path);
        sp->banks[b].opp_started_step = current_step;
    }
}

static void selfplay_step(Selfplay* sp, PuffeRL* p, Dict* log) {
    long current_step = p->global_step * p->hypers.world_size;
    for (int b = 0; b < sp->num_banks; b++) {
        SelfplayBank* bank = &sp->banks[b];
        int timed_out = sp->opp_timeout_steps > 0 &&
            current_step - bank->opp_started_step >= sp->opp_timeout_steps;
        int tag = b + 1;
        if (bank->pending_path[0]) {
            if (selfplay_count_aligned(p, tag) >= bank->num_envs) {
                pufferl_load_frozen_bank(p, b, bank->pending_path);
                selfplay_clear_aligned(p, tag);
                bank->pending_path[0] = 0;
                bank->opp_started_step = current_step;
            }
        } else if (timed_out) {
            const char* path = selfplay_sample(sp);
            snprintf(bank->pending_path, sizeof(bank->pending_path), "%s", path);
            selfplay_clear_aligned(p, tag);
        }
    }
    dict_set(log, "pool/size", sp->pool_size);
    dict_set(log, "pool/num_banks", sp->num_banks);
}

#define LEAGUE_ID_MAX 128
#define LEAGUE_PATH_MAX 4096

typedef struct {
    char id[LEAGUE_ID_MAX];
    char path[LEAGUE_PATH_MAX];
    float elo;
} LeaguePlayer;

typedef struct {
    char a[LEAGUE_ID_MAX];
    char b[LEAGUE_ID_MAX];
    int games;
    float score;
    float draw;
} LeagueMatch;

typedef struct {
    LeaguePlayer* players;
    LeagueMatch* matches;
    int num_players;
    int num_matches;
} LeagueState;

static int league_lock(const char* path) {
    char lock_path[LEAGUE_PATH_MAX];
    snprintf(lock_path, sizeof(lock_path), "%s.lock", path);
    int fd = open(lock_path, O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        perror("open league lock");
        exit(1);
    }
    if (flock(fd, LOCK_EX) != 0) {
        perror("flock");
        exit(1);
    }
    return fd;
}

static void league_unlock(int fd) {
    flock(fd, LOCK_UN);
    close(fd);
}

static int league_player_index(LeagueState* st, const char* id) {
    for (int i = 0; i < st->num_players; i++) {
        if (strcmp(st->players[i].id, id) == 0) {
            return i;
        }
    }
    return -1;
}

static void league_free(LeagueState* st) {
    free(st->players);
    free(st->matches);
}

static void league_load_unlocked(const char* path, LeagueState* st) {
    memset(st, 0, sizeof(*st));
    FILE* fp = fopen(path, "r");
    if (!fp) {
        return;
    }

    char type[32];
    while (fscanf(fp, "%31s", type) == 1) {
        if (strcmp(type, "PLAYER") == 0) {
            st->players = (LeaguePlayer*)realloc(st->players,
                (size_t)(st->num_players + 1) * sizeof(*st->players));
            LeaguePlayer* p = &st->players[st->num_players++];
            int n = fscanf(fp, "%127s %4095s", p->id, p->path);
            (void)n;
        } else if (strcmp(type, "MATCH") == 0) {
            st->matches = (LeagueMatch*)realloc(st->matches,
                (size_t)(st->num_matches + 1) * sizeof(*st->matches));
            LeagueMatch* m = &st->matches[st->num_matches++];
            int n = fscanf(fp, "%127s %127s %d %f %f",
                m->a, m->b, &m->games, &m->score, &m->draw);
            (void)n;
        } else {
            char line[4096];
            if (!fgets(line, sizeof(line), fp)) {
                break;
            }
        }
    }
    fclose(fp);
}

static void league_write_unlocked(const char* path, LeagueState* st) {
    char tmp[LEAGUE_PATH_MAX];
    snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path, getpid());
    FILE* fp = fopen(tmp, "w");
    if (!fp) {
        fprintf(stderr, "failed to write league state %s\n", tmp);
        exit(1);
    }
    fprintf(fp, "# PufferLib native league v1\n");
    for (int i = 0; i < st->num_players; i++) {
        LeaguePlayer* p = &st->players[i];
        fprintf(fp, "PLAYER %s %s\n", p->id, p->path);
    }
    for (int i = 0; i < st->num_matches; i++) {
        LeagueMatch* m = &st->matches[i];
        fprintf(fp, "MATCH %s %s %d %.9g %.9g\n",
            m->a, m->b, m->games, m->score, m->draw);
    }
    fclose(fp);
    if (rename(tmp, path) != 0) {
        fprintf(stderr, "failed to publish league state %s\n", path);
        exit(1);
    }
}

static void league_recompute(LeagueState* st) {
    for (int i = 0; i < st->num_players; i++) {
        st->players[i].elo = 0;
    }
    for (int iter = 0; iter < 100; iter++) {
        for (int i = 0; i < st->num_matches; i++) {
            LeagueMatch* m = &st->matches[i];
            int ai = league_player_index(st, m->a);
            int bi = league_player_index(st, m->b);
            if (ai < 0 || bi < 0 || ai == bi || m->games <= 0) {
                continue;
            }
            float ea = 1.0f / (1.0f + powf(10.0f,
                (st->players[bi].elo - st->players[ai].elo) / 400.0f));
            float delta = 0.02f * (float)m->games * (m->score - ea);
            st->players[ai].elo += delta;
            st->players[bi].elo -= delta;
        }
    }
}

static float league_register_player(const char* path, const char* id,
        const char* checkpoint) {
    int lock = league_lock(path);
    LeagueState st;
    league_load_unlocked(path, &st);
    int idx = league_player_index(&st, id);
    if (idx < 0) {
        st.players = (LeaguePlayer*)realloc(st.players,
            (size_t)(st.num_players + 1) * sizeof(*st.players));
        idx = st.num_players++;
    }
    LeaguePlayer* p = &st.players[idx];
    snprintf(p->id, sizeof(p->id), "%s", id);
    snprintf(p->path, sizeof(p->path), "%s", checkpoint);
    league_recompute(&st);
    float elo = st.players[idx].elo;
    league_write_unlocked(path, &st);
    league_free(&st);
    league_unlock(lock);
    return elo;
}

static void league_record_match(const char* path, const char* a, const char* b,
        int games, float score, float draw) {
    int lock = league_lock(path);
    LeagueState st;
    league_load_unlocked(path, &st);
    st.matches = (LeagueMatch*)realloc(st.matches,
        (size_t)(st.num_matches + 1) * sizeof(*st.matches));
    LeagueMatch* m = &st.matches[st.num_matches++];
    snprintf(m->a, sizeof(m->a), "%s", a);
    snprintf(m->b, sizeof(m->b), "%s", b);
    m->games = games;
    m->score = score;
    m->draw = draw;
    league_write_unlocked(path, &st);
    league_free(&st);
    league_unlock(lock);
}

static int league_choose_pair(const char* path, LeaguePlayer* a, LeaguePlayer* b,
        unsigned int* rng) {
    int lock = league_lock(path);
    LeagueState st;
    league_load_unlocked(path, &st);
    int n = st.num_players;
    if (n < 2) {
        league_free(&st);
        league_unlock(lock);
        return 0;
    }
    int ai = (int)(rand_r(rng) % (unsigned int)n);
    int bi = ai;
    for (int tries = 0; tries < 32 && bi == ai; tries++) {
        bi = (int)(rand_r(rng) % (unsigned int)n);
    }
    if (bi == ai) {
        bi = (ai + 1) % n;
    }
    *a = st.players[ai];
    *b = st.players[bi];
    league_free(&st);
    league_unlock(lock);
    return 1;
}

static void run_league_match_worker(Config* cfg, TrainContext* ctx) {
    const char* state_path = puf_config_str(cfg, "sweep", "league_state_path");
    long games = puf_config_long(cfg, "base", "num_games");
    if (!games) {
        games = puf_config_long(cfg, "sweep", "league_match_games");
    }
    unsigned int rng = (unsigned int)puf_config_int(cfg, "base", "seed") + 1009U;

    for (;;) {
        LeaguePlayer a;
        LeaguePlayer b;
        if (!league_choose_pair(state_path, &a, &b, &rng)) {
            usleep(500000);
            continue;
        }

        char buf[64];
        puf_config_put(cfg, "base.load_model_path", a.path);
        puf_config_put(cfg, "base.load_enemy_model_path", b.path);
        snprintf(buf, sizeof(buf), "%ld", games);
        puf_config_put(cfg, "base.num_games", buf);

        EvalResult result = run_eval(cfg, ctx, EVAL_MATCH, 0);
        league_record_match(state_path, a.id, b.id,
            result.games, result.score, result.draw);
        printf("league_match %s vs %s games=%d score=%.4f draw=%.4f\n",
            a.id, b.id, result.games, result.score, result.draw);
    }
}

extern char** environ;

typedef struct {
    int run;
    int random;
    int gp_obs;
    int pareto;
    int fd;
    pid_t pid;
    char run_id[128];
    float* sample;
    TrainResult result;
} SweepJob;

static int native_num_gpus(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count < 1) {
        fprintf(stderr, "sweep error: no CUDA devices available\n");
        exit(1);
    }
    return count;
}

static int sweep_read_result(int fd, TrainResult* out) {
    char* dst = (char*)out;
    size_t need = sizeof(*out);
    while (need > 0) {
        ssize_t n = read(fd, dst, need);
        if (n <= 0) {
            return 0;
        }
        dst += n;
        need -= (size_t)n;
    }
    return 1;
}

static char* sweep_arg_kv(const char* full_key, DictItem* item) {
    char val[128];
    const char* src = item->str;
    if (!src) {
        snprintf(val, sizeof(val), "%.17g", item->value);
        src = val;
    }

    size_t n = strlen(full_key) + strlen(src) + 2;
    char* out = (char*)malloc(n);
    if (!out) {
        perror("malloc");
        exit(1);
    }
    snprintf(out, n, "%s=%s", full_key, src);
    return out;
}

static void sweep_free_argv(char** argv, int argc) {
    for (int i = 3; i < argc; i++) {
        free(argv[i]);
    }
    free(argv);
}

static int sweep_config_count(Config* cfg) {
    int count = 0;
    for (int s = 0; s < cfg->ini.num_sections; s++) {
        count += cfg->ini.sections[s].size;
    }
    return count;
}

static int sweep_fill_args(Config* cfg, char** argv, int idx) {
    char full_key[PUF_DICT_MAX_KEY * 2];
    for (int s = 0; s < cfg->ini.num_sections; s++) {
        Dict* dict = &cfg->ini.sections[s];
        for (int i = 0; i < dict->size; i++) {
            snprintf(full_key, sizeof(full_key), "%s.%s",
                dict->name, dict->items[i].key);
            argv[idx++] = sweep_arg_kv(full_key, &dict->items[i]);
        }
    }
    return idx;
}

static int sweep_sample_valid(Config* cfg, SweepParam* params,
        SweepSpace* space, const float* sample, char* err, size_t err_size) {
    Config trial = {0};
    puf_config_copy(&trial, cfg);
    puf_config_sweep_apply(&trial, params, space, sample);
    int ok = puf_config_train_valid(&trial, err, err_size);
    puf_config_free(&trial);
    return ok;
}

static SweepJob sweep_start_job(Config* cfg, const char* exe_path,
        SweepParam* params, SweepSpace* space, const float* sample,
        ProteinSweepInfo info, int run, int gpu_offset) {
    SweepJob job = {0};
    job.run = run;
    job.random = info.is_random;
    job.gp_obs = info.n_gp_obs;
    job.pareto = info.n_pareto;
    job.sample = (float*)calloc((size_t)space->num, sizeof(float));
    memcpy(job.sample, sample, (size_t)space->num * sizeof(float));

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        perror("pipe");
        exit(1);
    }

    Config trial = {0};
    puf_config_copy(&trial, cfg);
    puf_config_sweep_apply(&trial, params, space, sample);
    char offset[32];

    char run_id[64];
    snprintf(run_id, sizeof(run_id), "sweep_%ld_%04d",
        (long)(1000.0 * wall_clock()), run);
    snprintf(job.run_id, sizeof(job.run_id), "%s", run_id);
    puf_config_put(&trial, "base.run_id", run_id);

    snprintf(offset, sizeof(offset), "%d", gpu_offset);
    puf_config_put(&trial, "base.gpu_offset", offset);

    char result_fd[32];
    snprintf(result_fd, sizeof(result_fd), "%d", pipefd[1]);
    puf_config_put(&trial, "base.result_fd", result_fd);
    puf_config_validate(&trial);

    int argc = sweep_config_count(&trial) + 4;
    char** argv = (char**)calloc((size_t)argc, sizeof(char*));
    if (!argv) {
        perror("calloc");
        exit(1);
    }
    argv[0] = (char*)exe_path;
    argv[1] = (char*)"train";
    argv[2] = (char*)puf_config_str(&trial, "base", "env_name");
    sweep_fill_args(&trial, argv, 3);
    argv[argc - 1] = NULL;

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addclose(&actions, pipefd[0]);
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO,
        "/dev/null", O_WRONLY, 0);
    int err = posix_spawnp(&job.pid, exe_path, &actions, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    sweep_free_argv(argv, argc - 1);
    puf_config_free(&trial);
    if (err != 0) {
        fprintf(stderr, "posix_spawn failed: %s\n", strerror(err));
        exit(1);
    }

    close(pipefd[1]);
    job.fd = pipefd[0];
    return job;
}

static void sweep_wait_job(ProteinSweep* protein, SweepJob* job,
        int league, const char* league_state_path) {
    int ok = sweep_read_result(job->fd, &job->result);
    close(job->fd);

    int status = 0;
    if (waitpid(job->pid, &status, 0) < 0) {
        perror("waitpid");
        exit(1);
    }
    if (!ok || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "sweep worker run=%d failed\n", job->run);
        exit(1);
    }

    if (league) {
        if (!job->result.checkpoint_path[0]) {
            fprintf(stderr, "league trial run=%d did not produce a checkpoint\n", job->run);
            exit(1);
        }
        float elo = league_register_player(league_state_path, job->run_id,
            job->result.checkpoint_path);
        protein_sweep_observe(protein, job->sample, elo, job->result.cost, 0);
        job->result.score = elo;
    } else {
        int points = job->result.points > 0 ? job->result.points : 1;
        for (int i = 0; i < points; i++) {
            protein_sweep_observe(protein, job->sample,
                job->result.scores[i], job->result.costs[i], 0);
        }
    }
    printf("sweep run=%d score=%.4f cost=%.2f steps=%.0f random=%d gp_obs=%d pareto=%d\n",
        job->run, job->result.score, job->result.cost, job->result.steps,
        job->random, job->gp_obs, job->pareto);
    free(job->sample);
}

static void sweep_state_path(Config* cfg, char* out, size_t out_size) {
    const char* configured = puf_config_str(cfg, "sweep", "league_state_path");
    if (configured && configured[0]) {
        snprintf(out, out_size, "%s", configured);
        return;
    }

    char dir[2048];
    snprintf(dir, sizeof(dir), "%s/%s",
        puf_config_str(cfg, "base", "log_dir"),
        puf_config_str(cfg, "base", "env_name"));
    mkdir_p(dir);
    snprintf(out, out_size, "%s/%ld_league.txt",
        dir, (long)(1000.0 * wall_clock()));
    puf_config_put(cfg, "sweep.league_state_path", out);
}

static pid_t sweep_start_match_worker(Config* cfg, const char* exe_path,
        const char* state_path, int gpu_id) {
    Config worker = {0};
    puf_config_copy(&worker, cfg);
    puf_config_put(&worker, "sweep.league_state_path", state_path);
    puf_config_put(&worker, "selfplay.enabled", "0");

    char offset[32];
    snprintf(offset, sizeof(offset), "%d", gpu_id);
    puf_config_put(&worker, "base.gpu_offset", offset);

    int argc = sweep_config_count(&worker) + 4;
    char** argv = (char**)calloc((size_t)argc, sizeof(char*));
    argv[0] = (char*)exe_path;
    argv[1] = (char*)"league_match_worker";
    argv[2] = (char*)puf_config_str(&worker, "base", "env_name");
    sweep_fill_args(&worker, argv, 3);
    argv[argc - 1] = NULL;

    pid_t pid = 0;
    int err = posix_spawnp(&pid, exe_path, NULL, NULL, argv, environ);
    sweep_free_argv(argv, argc - 1);
    puf_config_free(&worker);
    if (err != 0) {
        fprintf(stderr, "posix_spawn match worker failed: %s\n", strerror(err));
        exit(1);
    }
    return pid;
}

static void run_sweep(Config* cfg, const char* exe_path) {
    int league = (int)puf_config_get(cfg, "sweep", "league");
    SweepParam* params = NULL;
    SweepSpace* space = puf_config_sweep_space(cfg, &params);

    int max_runs = (int)puf_config_get(cfg, "sweep", "max_runs");
    int downsample = (int)puf_config_get(cfg, "sweep", "downsample");
    int prune_pareto = (int)puf_config_get(cfg, "sweep", "prune_pareto");
    int use_logit = strcmp(puf_config_str(cfg, "sweep", "metric_distribution"),
        "logit") == 0;
    float max_cost = (float)puf_config_get(cfg, "sweep", "max_suggestion_cost");
    float early_stop_quantile = (float)puf_config_get(cfg, "sweep", "early_stop_quantile");
    int success_cap = max_runs * downsample * 2;
    if (success_cap < 8192) {
        success_cap = 8192;
    }

    int total_gpus = native_num_gpus();
    int sweep_gpus = (int)puf_config_get(cfg, "sweep", "gpus");
    int train_gpus = (int)puf_config_get(cfg, "train", "gpus");
    if (sweep_gpus == 0) {
        sweep_gpus = total_gpus;
    }
    if (sweep_gpus > total_gpus) {
        fprintf(stderr, "sweep error: sweep.gpus=%d but only %d CUDA devices are visible\n",
            sweep_gpus, total_gpus);
        exit(1);
    }
    int use_gpu = (int)puf_config_get(cfg, "sweep", "use_gpu");
    if (use_gpu) {
        cudaSetDevice(sweep_gpus - 1);
    }

    char league_state_path[LEAGUE_PATH_MAX] = {0};
    pid_t match_pid = 0;
    int train_gpu_count = sweep_gpus;
    if (league) {
        train_gpu_count = sweep_gpus - 1;
        sweep_state_path(cfg, league_state_path, sizeof(league_state_path));
        match_pid = sweep_start_match_worker(cfg, exe_path,
            league_state_path, sweep_gpus - 1);
    }

    int parallel = train_gpu_count / train_gpus;
    if (parallel < 1) {
        parallel = 1;
    }

    ProteinSweep* protein = protein_sweep_create(space,
        10, 256, 50, 0.001f, 50, 750, 4096,
        downsample == 1, prune_pareto, use_logit,
        1.0f, max_cost, 0.1f, -0.8f, early_stop_quantile,
        success_cap, 1024, 5, 73ULL);

    float* sample = (float*)calloc((size_t)space->num, sizeof(float));
    SweepJob* jobs = (SweepJob*)calloc((size_t)parallel, sizeof(SweepJob));
    int invalid_samples = 0;
    for (int run = 0; run < max_runs;) {
        int batch = max_runs - run;
        if (batch > parallel) {
            batch = parallel;
        }

        for (int i = 0; i < batch; i++) {
            ProteinSweepInfo info = {0};
            char err[256];
            for (;;) {
                info = protein_sweep_suggest(protein, sample, NAN);
                if (sweep_sample_valid(cfg, params, space, sample,
                        err, sizeof(err))) {
                    break;
                }

                protein_sweep_observe(protein, sample, NAN, max_cost, 1);
                invalid_samples++;
                if (invalid_samples <= 5 || invalid_samples % 100 == 0) {
                    fprintf(stderr, "sweep skipped invalid sample: %s\n", err);
                }
                if (invalid_samples > 10000) {
                    fprintf(stderr, "sweep error: too many invalid samples\n");
                    exit(1);
                }
            }
            jobs[i] = sweep_start_job(cfg, exe_path, params, space, sample,
                info, run + i, i * train_gpus);
        }
        for (int i = 0; i < batch; i++) {
            sweep_wait_job(protein, &jobs[i], league, league_state_path);
        }
        run += batch;
    }

    if (match_pid > 0) {
        kill(match_pid, SIGTERM);
        waitpid(match_pid, NULL, 0);
    }
    free(jobs);
    free(sample);
    free(params);
    protein_sweep_destroy(protein);
    sweep_space_destroy(space);
}

static float log_value(Dict* log, const char* key, float fallback) {
    for (int i = 0; i < log->size; i++) {
        if (strcmp(log->items[i].key, key) == 0) {
            return (float)log->items[i].value;
        }
    }
    return fallback;
}

static void train_result_fill(TrainResult* result, PufLogHistory* history,
        Dict* last_log, Config* cfg, const char* target_key) {
    result->score = (float)puf_log_get_or(last_log, target_key, 0);
    result->cost = (float)puf_log_get_or(last_log, "uptime", 0);
    result->steps = (float)puf_log_get_or(last_log, "agent_steps", 0);

    int points = puf_config_int(cfg, "sweep", "downsample");
    if (points < 1) {
        points = 1;
    }
    if (points > TRAIN_RESULT_MAX_POINTS) {
        points = TRAIN_RESULT_MAX_POINTS;
    }
    result->points = points;

    if (history->size == 0 || points == 1) {
        result->scores[0] = result->score;
        result->costs[0] = result->cost;
        result->step_points[0] = result->steps;
        return;
    }

    float final_steps = log_value(&history->items[history->size - 1],
        "agent_steps", result->steps);
    int cursor = 0;
    for (int p = 0; p < points; p++) {
        float target = final_steps * (float)p / (float)(points - 1);
        while (cursor + 1 < history->size &&
                log_value(&history->items[cursor], "agent_steps", 0) < target) {
            cursor++;
        }
        Dict* log = &history->items[cursor];
        result->scores[p] = log_value(log, target_key, result->score);
        result->costs[p] = log_value(log, "uptime", result->cost);
        result->step_points[p] = log_value(log, "agent_steps", target);
    }
    result->scores[points - 1] = result->score;
    result->costs[points - 1] = result->cost;
    result->step_points[points - 1] = result->steps;
}

static EvalResult run_eval(Config* cfg, TrainContext* ctx, int mode, int verbose) {
    int render = mode == EVAL_RENDER;
    int match = mode == EVAL_MATCH;
    EvalResult result = {0};
    long num_games = puf_config_long(cfg, "base", "num_games");
    if (!num_games) {
        num_games = puf_config_long(cfg, "base", "eval_episodes");
    }
    long burnin_games = puf_config_long(cfg, "base", "burnin_games");
    if (!render && (num_games <= 0 || burnin_games < 0)) {
        fprintf(stderr, "eval requires positive num_games and nonnegative burnin_games\n");
        exit(1);
    }
    if (!render) {
        long eval_agents = puf_config_long(cfg, "base", "eval_agents");
        if (!eval_agents && match) {
            eval_agents = puf_config_long(cfg, "sweep", "league_match_eval_agents");
        }
        if (eval_agents <= 0 && match) {
            eval_agents = 8192;
        } else if (eval_agents <= 0) {
            eval_agents = num_games / 8;
            if (eval_agents < 1024) {
                eval_agents = 1024;
            }
            if (eval_agents > 4096) {
                eval_agents = 4096;
            }
            if (eval_agents > num_games && num_games >= 1024) {
                eval_agents = num_games;
            }
        } else if (eval_agents > num_games) {
            eval_agents = num_games;
        }
        eval_agents += (-eval_agents) % (match ? 4 : 2);

        char buf[64];
        snprintf(buf, sizeof(buf), "%ld", eval_agents);
        puf_config_put(cfg, "vec.num_buffers", "2");
        puf_config_put(cfg, "vec.total_agents", buf);
    }
    if (match) {
        puf_config_put(cfg, "vec.num_frozen_banks", "1");
        puf_config_put(cfg, "vec.frozen_bank_pct", "1");
        puf_config_put(cfg, "selfplay.enabled", "0");
        puf_config_put(cfg, "env.dr", "0");
        puf_config_put(cfg, "env.num_agents", "2");
        puf_config_put(cfg, "env.num_bots", "0");
    }

    puf_config_put(cfg, "base.reset_state", "0");
    puf_config_put(cfg, "train.horizon", "1");

    PuffeRL* pufferl = create_trainer(cfg, ctx);
    if (match) {
        char a_path_buf[4096];
        char b_path_buf[4096];
        const char* a_path = puf_checkpoint_path_key(cfg,
            "load_model_path", a_path_buf, sizeof(a_path_buf));
        const char* b_path = puf_checkpoint_path_key(cfg,
            "load_enemy_model_path", b_path_buf, sizeof(b_path_buf));
        if (!a_path || !b_path) {
            fprintf(stderr, "match requires base.load_model_path and base.load_enemy_model_path\n");
            exit(1);
        }
        puf_load_weights_into(pufferl->master_weights,
            pufferl->param_puf, pufferl->default_stream, a_path);
        pufferl_load_frozen_bank(pufferl, 0, b_path);
    } else {
        puf_load_primary_if_configured(pufferl, cfg);
    }

    Dict baseline = {0};
    long baseline_n = 0;
    for (;;) {
        if (render) {
            puf_render(&pufferl->vec->envs[0]);
        }
        rollouts(pufferl);
        Dict log = {0};
        trainer_eval_log(pufferl, &log);
        if (render) {
            puf_dashboard_print(cfg, pufferl, &log, 0);
            continue;
        }

        long n = (long)puf_log_get_or(&log, "env/n", 0);
        if (match) {
            double score = puf_log_get_or(&log, "env/slot_0_score", 0);
            double draw = puf_log_get_or(&log, "env/draw_rate", 0);
            if (verbose) {
                double b = puf_log_get_or(&log, "env/slot_1_score", 0);
                printf("\rgames=%ld/%ld  A=%.3f  B=%.3f  draw=%.3f",
                    n, num_games, score, b, draw);
            }
            if (n >= num_games) {
                result.score = (float)score;
                result.draw = (float)draw;
                result.games = (int)n;
                break;
            }
            continue;
        }

        if (burnin_games > 0 && baseline_n == 0 && n >= burnin_games) {
            baseline = log;
            baseline_n = n;
            if (verbose) {
                printf("\rbot_eval_burnin=%ld/%ld", n, burnin_games);
            }
            continue;
        }

        double scored_n = n - baseline_n;
        double score = puf_log_get_or(&log, "env/score", 0);
        double perf = puf_log_get_or(&log, "env/perf", 0);
        if (baseline_n > 0 && scored_n > 0) {
            double base_n = (double)baseline_n;
            double cur_n = (double)n;
            score = (score * cur_n - puf_log_get_or(&baseline, "env/score", 0) * base_n) / scored_n;
            perf = (perf * cur_n - puf_log_get_or(&baseline, "env/perf", 0) * base_n) / scored_n;
        }
        if (verbose) {
            printf("\rbot_eval=%.0f/%ld  perf=%.4f  score=%.3f",
                scored_n, num_games, perf, score);
        }
        if ((n - baseline_n) >= num_games && (!burnin_games || baseline_n > 0)) {
            result.score = (float)score;
            result.games = (int)scored_n;
            break;
        }
    }
    if (!render && verbose) {
        printf("\n");
    }
    close_trainer(pufferl);
    return result;
}

static void train_checkpoint_path(PuffeRL* p, const char* dir,
        char* out, size_t out_size) {
    snprintf(out, out_size, "%s/%016ld.bin", dir, p->global_step);
}

TrainResult run_train(Config* cfg, TrainContext* ctx) {
    int use_selfplay = puf_config_int(cfg, "selfplay", "enabled");
    if (!use_selfplay) {
        puf_config_put(cfg, "vec.num_frozen_banks", "0");
        puf_config_put(cfg, "vec.frozen_bank_pct", "0");
    }

    char run_id[64];
    const char* configured_run_id = puf_config_str(cfg, "base", "run_id");
    if (!configured_run_id[0] || strcmp(configured_run_id, "None") == 0) {
        snprintf(run_id, sizeof(run_id), "%ld", (long)(1000.0 * wall_clock()));
        puf_config_put(cfg, "base.run_id", run_id);
    } else {
        snprintf(run_id, sizeof(run_id), "%s", configured_run_id);
    }

    char checkpoint_dir[2048];
    char log_dir[2048];
    snprintf(checkpoint_dir, sizeof(checkpoint_dir), "%s/%s/%s",
        puf_config_str(cfg, "base", "checkpoint_dir"),
        puf_config_str(cfg, "base", "env_name"), run_id);
    snprintf(log_dir, sizeof(log_dir), "%s/%s",
        puf_config_str(cfg, "base", "log_dir"),
        puf_config_str(cfg, "base", "env_name"));
    if (ctx->artifact_owner) {
        mkdir_p(checkpoint_dir);
        mkdir_p(log_dir);
    }

    PuffeRL* pufferl = create_trainer(cfg, ctx);
    char initial_checkpoint[4096] = {0};
    if (use_selfplay) {
        train_checkpoint_path(pufferl, checkpoint_dir,
            initial_checkpoint, sizeof(initial_checkpoint));
        if (ctx->artifact_owner) {
            puf_save_weights(pufferl, initial_checkpoint);
        }
    }

    Selfplay* selfplay = NULL;
    if (use_selfplay) {
        selfplay = (Selfplay*)calloc(1, sizeof(Selfplay));
        selfplay_init(selfplay, cfg, pufferl, initial_checkpoint);
    }

    long total_timesteps = puf_config_long(cfg, "train", "total_timesteps");
    long batch_size = (long)puf_config_int(cfg, "vec", "total_agents") *
        (long)puf_config_int(cfg, "train", "horizon");
    long local_timesteps = total_timesteps / ctx->world_size;
    long train_epochs = local_timesteps / batch_size;
    long eval_epochs = train_epochs / 2;
    long checkpoint_interval = puf_config_long(cfg, "base", "checkpoint_interval");
    long eval_episodes = puf_config_long(cfg, "base", "eval_episodes");
    const char* target_key = "env/score";
    Dict last_log = {0};
    PufLogHistory log_history = {0};
    TrainResult result = {0};

    for (long epoch = 0; epoch < train_epochs + eval_epochs; epoch++) {
        rollouts(pufferl);
        if (epoch < train_epochs) {
            train_impl(*pufferl);
        }

        bool is_final = epoch == train_epochs - 1;
        bool interval_save = checkpoint_interval > 0 &&
            (epoch + 1) % checkpoint_interval == 0;
        bool should_save = epoch < train_epochs && (interval_save || is_final);
        char saved_checkpoint[4096] = {0};
        if (should_save) {
            train_checkpoint_path(pufferl, checkpoint_dir,
                saved_checkpoint, sizeof(saved_checkpoint));
            if (ctx->artifact_owner) {
                puf_save_weights(pufferl, saved_checkpoint);
                snprintf(result.checkpoint_path, sizeof(result.checkpoint_path),
                    "%s", saved_checkpoint);
            }
        }
        if (selfplay && saved_checkpoint[0]) {
            selfplay_add_checkpoint(selfplay, saved_checkpoint);
        }

        if (wall_clock() < pufferl->last_log_time + 0.6 && epoch < train_epochs - 1) {
            continue;
        }

        Dict new_log = {0};
        if (epoch >= train_epochs) {
            trainer_eval_log(pufferl, &new_log);
        } else {
            trainer_log(pufferl, &new_log);
        }
        puf_log_update(&last_log, &new_log);
        if (selfplay && epoch < train_epochs) {
            selfplay_step(selfplay, pufferl, &last_log);
        }
        if (ctx->artifact_owner) {
            puf_dashboard_print(cfg, pufferl, &last_log, (int)epoch);
        }

        if (puf_log_get_or(&last_log, target_key, -1) < 0) {
            continue;
        }
        if (epoch < train_epochs) {
            puf_log_history_add(&log_history, &last_log);
        }
        if (epoch >= train_epochs && puf_log_get_or(&last_log, "env/n", 0) > eval_episodes) {
            break;
        }
    }

    train_result_fill(&result, &log_history, &last_log, cfg, target_key);
    if (ctx->artifact_owner) {
        puf_log_history_add(&log_history, &last_log);
        char log_path[4096];
        snprintf(log_path, sizeof(log_path), "%s/%s.ini", log_dir, run_id);
        puf_log_write(log_path, cfg, &log_history);
    }
    puf_log_history_free(&log_history);
    free(selfplay);
    close_trainer(pufferl);
    return result;
}

static int gpu_for_rank(int rank, int world_size) {
    if (rank == 0) {
        return world_size - 1;
    }
    return rank - 1;
}

static void wait_children(pid_t* pids, int num_pids) {
    for (int i = 0; i < num_pids; i++) {
        int status = 0;
        if (waitpid(pids[i], &status, 0) < 0) {
            fprintf(stderr, "waitpid failed for child %d: %s\n", (int)pids[i], strerror(errno));
            exit(1);
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "worker pid %d failed\n", (int)pids[i]);
            exit(1);
        }
    }
}

TrainResult launch_train(Config* cfg) {
    int world_size = puf_config_int(cfg, "train", "gpus");
    if (world_size < 1) {
        fprintf(stderr, "config error: [train] gpus must be >= 1\n");
        exit(1);
    }
    int gpu_offset = puf_config_int(cfg, "base", "gpu_offset");

    ncclUniqueId nccl_id;
    ncclUniqueId* nccl_ptr = NULL;
    if (world_size > 1) {
        ncclGetUniqueId(&nccl_id);
        nccl_ptr = &nccl_id;
    }

    pid_t* pids = (pid_t*)calloc(world_size > 1 ? world_size - 1 : 1, sizeof(pid_t));
    for (int rank = world_size - 1; rank >= 1; rank--) {
        pid_t pid = fork();
        if (pid < 0) {
            fprintf(stderr, "fork failed: %s\n", strerror(errno));
            exit(1);
        }

        if (pid == 0) {
            if (!freopen("/dev/null", "w", stdout)) {
                fprintf(stderr, "failed to redirect child stdout: %s\n", strerror(errno));
                exit(1);
            }
            TrainContext child = {
                .rank = rank,
                .world_size = world_size,
                .gpu_id = gpu_offset + gpu_for_rank(rank, world_size),
                .artifact_owner = 0,
                .nccl_id = nccl_ptr,
            };
            run_train(cfg, &child);
            puf_config_free(cfg);
            exit(0);
        }

        pids[rank - 1] = pid;
    }

    TrainContext host = {
        .rank = 0,
        .world_size = world_size,
        .gpu_id = gpu_offset + gpu_for_rank(0, world_size),
        .artifact_owner = 1,
        .nccl_id = nccl_ptr,
    };
    TrainResult result = run_train(cfg, &host);
    wait_children(pids, world_size - 1);
    free(pids);
    return result;
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    if (argc < 3) {
        fprintf(stderr, "usage: %s train|eval|eval_bot|match|sweep ENV [section.key=value ...]\n", argv[0]);
        exit(1);
    }

    const char* mode = argv[1];
    const char* env_name = argv[2];
    Config* cfg = (Config*)calloc(1, sizeof(Config));
    puf_config_load_env(cfg, env_name, argc - 3, argv + 3);
    puf_config_validate(cfg);
    TrainContext ctx = {
        .rank = 0,
        .world_size = 1,
        .gpu_id = 0,
        .artifact_owner = 1,
        .nccl_id = NULL,
    };

    if (strcmp(mode, "train") == 0) {
        TrainResult result = launch_train(cfg);
        int result_fd = puf_config_int(cfg, "base", "result_fd");
        if (result_fd) {
            int fd = result_fd;
            if (write(fd, &result, sizeof(result)) != sizeof(result)) {
                fprintf(stderr, "failed to write train result\n");
                exit(1);
            }
            close(fd);
        }
    } else if (strcmp(mode, "sweep") == 0) {
        run_sweep(cfg, argv[0]);
    } else if (strcmp(mode, "eval") == 0 || strcmp(mode, "eval_bot") == 0) {
        int bot = strcmp(mode, "eval_bot") == 0;
        if (bot) {
            puf_config_put(cfg, "vec.num_frozen_banks", "0");
            puf_config_put(cfg, "vec.frozen_bank_pct", "0");
            puf_config_put(cfg, "selfplay.enabled", "0");
            puf_config_put(cfg, "env.dr", "0");
            puf_config_put(cfg, "env.num_agents", "1");
            puf_config_put(cfg, "env.num_bots", "1");
        }
        run_eval(cfg, &ctx, bot ? EVAL_SCORE : EVAL_RENDER, 1);
    } else if (strcmp(mode, "match") == 0) {
        run_eval(cfg, &ctx, EVAL_MATCH, 1);
    } else if (strcmp(mode, "league_match_worker") == 0) {
        int gpu_offset = puf_config_int(cfg, "base", "gpu_offset");
        if (gpu_offset) {
            ctx.gpu_id = gpu_offset;
        }
        run_league_match_worker(cfg, &ctx);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        exit(1);
    }

    puf_config_free(cfg);
    free(cfg);
    return 0;
}

#endif

#endif // PUFFERLIB_KERNELS_ONLY
