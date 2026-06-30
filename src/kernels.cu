#ifndef PUFFERLIB_KERNELS_CU
#define PUFFERLIB_KERNELS_CU

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdint>
#include <cublas_v2.h>
#include <curand.h>
#include <cassert>
#include <stdlib.h>

#include <cuda_bf16.h>

#ifdef PRECISION_FLOAT
typedef float precision_t;
constexpr bool USE_BF16 = false;
constexpr int PRECISION_SIZE = 4;
static constexpr cudaDataType_t CUBLAS_PRECISION = CUDA_R_32F;
static constexpr cublasComputeType_t CUBLAS_COMPUTE_PRECISION = CUBLAS_COMPUTE_32F;
#define NCCL_PRECISION ncclFloat
#define to_float(x) (x)
#define from_float(x) (x)
#else
typedef __nv_bfloat16 precision_t;
constexpr bool USE_BF16 = true;
constexpr int PRECISION_SIZE = 2;
static constexpr cudaDataType_t CUBLAS_PRECISION = CUDA_R_16BF;
static constexpr cublasComputeType_t CUBLAS_COMPUTE_PRECISION = CUBLAS_COMPUTE_32F;
#define NCCL_PRECISION ncclBfloat16
#define to_float(x) __bfloat162float(x)
#define from_float(x) __float2bfloat16(x)
#endif

#include "tensor.h"

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

inline const char* _puf_repr_impl(const char* name, const char* dtype,
        const int64_t* shape, int nd, int64_t ne, bool empty) {
    static thread_local char buf[256];
    if (empty) { snprintf(buf, sizeof(buf), "%s(empty)", name); return buf; }
    int pos = snprintf(buf, sizeof(buf), "%s(%s, [", name, dtype);
    for (int i = 0; i < nd && pos < (int)sizeof(buf) - 32; i++)
        pos += snprintf(buf + pos, sizeof(buf) - pos, "%s%lld", i ? ", " : "", (long long)shape[i]);
    snprintf(buf + pos, sizeof(buf) - pos, "], %lld elems)", (long long)ne);
    return buf;
}

inline const char* puf_repr(const PrecisionTensor* t) {
    return _puf_repr_impl("PrecisionTensor", USE_BF16 ? "bf16" : "f32",
        t->shape, ndim(t->shape), numel(t->shape), !t->data);
}

inline const char* puf_repr(const FloatTensor* t) {
    return _puf_repr_impl("FloatTensor", "f32",
        t->shape, ndim(t->shape), numel(t->shape), !t->data);
}

#ifndef CUDART_INF_F
#define CUDART_INF_F __int_as_float(0x7f800000)
#endif

#define PPO_THREADS 256
#define SELECT_COPY_THREADS 256
#define MAX_ATN_HEADS 16


#define BLOCK_SIZE 256
inline int grid_size(int N) {
    return (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

#define SEQ_SIZE 256
inline int seq_size(int N) {
    return (N + SEQ_SIZE - 1) / SEQ_SIZE;
}

#define SOFTPLUS_BETA 1.0f
#define SOFTPLUS_THRESHOLD 20.0f
__device__ __forceinline__ float softplus_fwd(float x) {
    float x_scaled = x * SOFTPLUS_BETA;
    return (x_scaled > SOFTPLUS_THRESHOLD) ? x : log1pf(expf(x_scaled)) / SOFTPLUS_BETA;
}

__device__ __forceinline__ float softplus_bwd(float grad_output, float x) {
    float beta_x = SOFTPLUS_BETA * x;
    if (beta_x > SOFTPLUS_THRESHOLD) {
        return grad_output;
    }
    float exp_beta_x = expf(beta_x);
    return grad_output * (exp_beta_x / (1.0f + exp_beta_x));
}

__device__ __forceinline__ float relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ __forceinline__ float relu_backward(float x, float grad_output) {
    return (x > 0.0f) ? grad_output : 0.0f;
}

__device__ __forceinline__ float sigmoid(float x) {
    float z = expf(-fabsf(x));
    return x >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
}

__device__ __forceinline__ float sigmoid_backward(float x, float grad_output) {
    float sig = sigmoid(x);
    return grad_output * sig * (1.0f - sig);
}

__device__ __inline__ float fast_tanh(float x) {
    float v1 = fminf(fmaxf(x, -9.0f), 9.0f);
    float v2 = v1 * v1;
    float p = v2 * -2.76076847742355e-16f + 2.00018790482477e-13f;
    p = v2 * p + -8.60467152213735e-11f;
    p = v2 * p + 5.12229709037114e-08f;
    p = v2 * p + 1.48572235717979e-05f;
    p = v2 * p + 6.37261928875436e-04f;
    p = v2 * p + 4.89352455891786e-03f;
    p = v1 * p;
    float q = v2 * 1.19825839466702e-06f + 1.18534705686654e-04f;
    q = v2 * q + 2.26843463243900e-03f;
    q = v2 * q + 4.89352518554385e-03f;
    return p / q;
}

__device__ __inline__ float fast_sigmoid(float x) {
    return fminf(1.0f, fmaxf(0.0f, (fast_tanh(x * 0.5f) + 1.0f) * 0.5f));
}

__device__ __forceinline__ float lerp(float a, float b, float w) {
    float diff = b - a;
    return (fabsf(w) < 0.5f) ? a + w * diff : b - diff * (1.0f - w);
}

__device__ __forceinline__ float logaddexp(float a, float b) {
    float m = fmaxf(a, b), diff = fminf(a, b) - m;
    return (diff < -88.0f) ? m : m + log1pf(__expf(diff));
}

//TODO: Speed up. The previous version was misaligned.
__device__ __forceinline__ void copy_bytes(
    const char* __restrict__ src, char* __restrict__ dst,
    int src_row, int dst_row, int row_bytes) {
    const char* s = src + (int64_t)src_row * row_bytes;
    char* d = dst + (int64_t)dst_row * row_bytes;
    for (int i = threadIdx.x; i < row_bytes; i += blockDim.x) {
        d[i] = s[i];
    }
}

/*
__device__ __forceinline__ void copy_bytes(const char* __restrict__ src,
        char* __restrict__ dst, int src_row, int dst_row, int row_bytes) {
    const int* soffset = (const int*)(src + (int64_t)src_row * row_bytes);
    int* doffset = (int*)(dst + (int64_t)dst_row * row_bytes);
    for (int i = threadIdx.x; i < row_bytes / 4; i += blockDim.x) {
        doffset[i] = soffset[i];
    }
}
*/

// Transpose dims 0,1: [A, B, C] -> [B, A, C]. For 2D, pass C=1.
__global__ void transpose_102(precision_t* __restrict__ dst,
        const precision_t* __restrict__ src, int A, int B, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = A * B * C;
    if (idx >= total) {
        return;
    }
    int a = idx / (B * C), rem = idx % (B * C), b = rem / C, c = rem % C;
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

__global__ void add_kernel(float* __restrict__ dst, const precision_t* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += to_float(src[idx]);
    }
}

#ifndef PRECISION_FLOAT
__global__ void add_kernel(precision_t* __restrict__ dst, const precision_t* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(to_float(dst[idx]) + to_float(src[idx]));
    }
}
#endif

// merge shape[dim] into shape[dim+1] ---
inline PrecisionTensor* puf_squeeze(PrecisionTensor* t, int dim) {
    int n = ndim(t->shape);
    t->shape[dim + 1] *= t->shape[dim];
    for (int i = dim; i < n - 1; i++) t->shape[i] = t->shape[i + 1];
    t->shape[n - 1] = 0;
    return t;
}
inline FloatTensor* puf_squeeze(FloatTensor* t, int dim) {
    int n = ndim(t->shape);
    t->shape[dim + 1] *= t->shape[dim];
    for (int i = dim; i < n - 1; i++) t->shape[i] = t->shape[i + 1];
    t->shape[n - 1] = 0;
    return t;
}

// split shape[dim] into {d0, d1} ---
inline PrecisionTensor* puf_unsqueeze(PrecisionTensor* t, int dim, int64_t d0, int64_t d1) {
    assert(d0 * d1 == t->shape[dim] && "puf_unsqueeze: d0 * d1 must equal shape[dim]");
    int n = ndim(t->shape);
    for (int i = n; i > dim; i--) t->shape[i] = t->shape[i - 1];
    t->shape[dim] = d0;
    t->shape[dim + 1] = d1;
    return t;
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
void puf_mm_nn(PrecisionTensor* a, PrecisionTensor* b, PrecisionTensor* out, cudaStream_t stream) {
    int M = batch_size(a->shape) * a->shape[ndim(a->shape)-2];
    int K = a->shape[ndim(a->shape)-1];
    int N = b->shape[ndim(b->shape)-1];
    cublasGemmExDense(CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
        a->data, b->data, out->data, stream);
}

static void puf_addmm_nn(PrecisionTensor* a, PrecisionTensor* b, PrecisionTensor* out,
        float alpha, float beta, cudaStream_t stream) {
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
inline void cast_dispatch(precision_t* dst, const precision_t* src, int n, cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, n * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream);
}
#endif
inline void cast_dispatch(precision_t* dst, const float* src, int n, cudaStream_t stream) {
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(dst, src, n);
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

inline void cast_dispatch(precision_t* dst, const unsigned char* src, int n, cudaStream_t stream) {
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(dst, src, n);
}

void puf_copy(PrecisionTensor* dst, const PrecisionTensor* src, cudaStream_t stream) {
    assert(numel(dst->shape) == numel(src->shape) && "puf_copy: size mismatch");
    cudaMemcpyAsync(dst->data, src->data, numel(dst->shape) * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream);
}

void puf_zero(PrecisionTensor* dst, cudaStream_t stream) {
    cudaMemsetAsync(dst->data, 0, numel(dst->shape) * sizeof(precision_t), stream);
}

void puf_zero(FloatTensor* dst, cudaStream_t stream) {
    cudaMemsetAsync(dst->data, 0, numel(dst->shape) * sizeof(float), stream);
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

// Normal(0, std). Used for embeddings
void puf_normal_init(PrecisionTensor* dst, float std, ulong seed, cudaStream_t stream) {
    long n = numel(dst->shape);
    assert(n > 0);
    long rand_count = (n % 2 == 0) ? n : n + 1;
    float* buf;
    cudaMalloc(&buf, rand_count * sizeof(float));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, buf, rand_count, 0.0f, std);
    curandDestroyGenerator(gen);
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

#endif // PUFFERLIB_KERNELS_CU
