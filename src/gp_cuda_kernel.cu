// gp_cuda_kernel.cu -- Matern32+Linear GP covariance kernel (CUDA), ARD.

#ifndef GP_CUDA_KERNEL_CU
#define GP_CUDA_KERNEL_CU

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// clang-format off
static inline void _cuda_check(cudaError_t e, const char* f, int l) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", f, l, cudaGetErrorString(e)); abort(); }
}
static inline void _cublas_check(cublasStatus_t s, const char* f, int l) {
    if (s != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS error %s:%d: %d\n", f, l, (int)s); abort(); }
}
#define CUDA_CHECK(call) _cuda_check((call), __FILE__, __LINE__)
#define CUBLAS_CHECK(call) _cublas_check((call), __FILE__, __LINE__)
// clang-format on

#define SP_LB 1e-4
__host__ __device__ __forceinline__ float softplus(float x) { return (x > 20.0 ? x : log1p(exp(x))) + SP_LB; }
__host__ __device__ __forceinline__ float inv_softplus(float x) {
    float v = x - SP_LB;
    return v > 20.0 ? v : log(expm1(v));
}
__host__ __device__ __forceinline__ float softplus_grad(float x) { return 1.0 / (1.0 + exp(-x)); }

typedef struct GPKernel GPKernel;
struct GPKernel {
    int n_params;
    float *raw_params;
    char tag[4];
    void (*build_K)(const GPKernel *k, const float *d_X, int n, int d, float sigma_n, float *d_K, cudaStream_t stream);
    void (*build_Ks)(const GPKernel *k, const float *d_Xtr, const float *d_Xte, int n, int m, int d, float *d_Ks,
                     cudaStream_t stream);
    void (*build_kself_batch)(const GPKernel *k, const float *d_X, int m, int d, float *d_out, cudaStream_t stream);
    float (*k_self)(const GPKernel *k, const float *x, int d);
    void (*mll_grad)(const GPKernel *k, const float *d_X, int n, int d, cublasHandle_t cublas, cudaStream_t stream,
                     const float *d_alpha, const float *d_Kinv, float *kernel_grads);
    void (*destroy)(GPKernel *k);
};

#define GP_KERNEL_TAG_MATERN32_LINEAR "M32L"
#define SF_IDX(np) ((np)-2)
#define OFF_IDX(np) ((np)-1)
#define GP_BLOCK 16
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

inline int gp_grid(int n) { return (n + GP_BLOCK - 1) / GP_BLOCK; }

typedef struct {
    float sigma_f, offset;
} KParams;

static inline KParams kernel_params(const GPKernel *k) {
    int np = k->n_params;
    return (KParams){softplus(k->raw_params[SF_IDX(np)]), softplus(k->raw_params[OFF_IDX(np)])};
}

__global__ void matern32lin_k_kernel(const float *__restrict__ X1, const float *__restrict__ X2, float *__restrict__ K,
                                     const float *__restrict__ inv_ells, int n, int m, int d, float sigma_f,
                                     float diag_noise, float offset) {
    int col = blockIdx.x * GP_BLOCK + threadIdx.x;
    int row = blockIdx.y * GP_BLOCK + threadIdx.y;
    if (row >= n || col >= m)
        return;
    float dot = 0.0, r2 = 0.0;
    for (int k = 0; k < d; k++) {
        float xr = X1[row * d + k], xc = X2[col * d + k];
        dot += xr * xc;
        float diff = (xr - xc) * inv_ells[k];
        r2 += diff * diff;
    }
    if (r2 < 0.0)
        r2 = 0.0;
    float u = sqrt(3.0) * sqrt(r2);
    float val = sigma_f * (dot + offset + (1.0 + u) * exp(-u));
    if (diag_noise != 0.0 && row == col)
        val += diag_noise;
    K[col * n + row] = val;
}

__global__ void matern32lin_k_build_D_ard(const float *__restrict__ X, float *__restrict__ D,
                                          const float *__restrict__ inv_ells, int n, int d) {
    int col = blockIdx.x * GP_BLOCK + threadIdx.x;
    int row = blockIdx.y * GP_BLOCK + threadIdx.y;
    if (row >= n || col >= n)
        return;
    float sq = 0.0;
    for (int k = 0; k < d; k++) {
        float diff = (X[row * d + k] - X[col * d + k]) * inv_ells[k];
        sq += diff * diff;
    }
    D[col * n + row] = sq < 0.0 ? 0.0 : sq;
}

__global__ void matern32lin_k_compute_W(const float *__restrict__ alpha, const float *__restrict__ Kinv,
                                        float *__restrict__ W, int n) {
    int col = blockIdx.x * GP_BLOCK + threadIdx.x;
    int row = blockIdx.y * GP_BLOCK + threadIdx.y;
    if (row >= n || col >= n)
        return;
    W[col * n + row] = alpha[row] * alpha[col] - Kinv[col * n + row];
}

__global__ void matern32lin_k_dk_dell_d(const float *__restrict__ X, const float *__restrict__ D_ard,
                                        float *__restrict__ out, int n, int d, int dd, float sigma_f,
                                        float inv_ell_d3) {
    int col = blockIdx.x * GP_BLOCK + threadIdx.x;
    int row = blockIdx.y * GP_BLOCK + threadIdx.y;
    if (row >= n || col >= n)
        return;
    float r2 = D_ard[col * n + row];
    float diff = X[row * d + dd] - X[col * d + dd];
    out[col * n + row] = sigma_f * 3.0 * diff * diff * inv_ell_d3 * exp(-sqrt(3.0) * sqrt(r2));
}

__global__ void matern32lin_k_fill(float *v, int n, float val) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n)
        v[i] = val;
}

__global__ void matern32lin_k_kself_batch(const float *__restrict__ X, float *__restrict__ out, int m, int d,
                                          float sigma_f, float offset) {
    int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= m)
        return;
    float norm2 = 0.0;
    for (int k = 0; k < d; k++) {
        float v = X[row * d + k];
        norm2 += v * v;
    }
    out[row] = sigma_f * (norm2 + offset + 1.0);
}

static float *h2d(const float *h, int n, cudaStream_t stream) {
    float *d;
    CUDA_CHECK(cudaMallocAsync(&d, (size_t)n * sizeof(float), stream));
    CUDA_CHECK(cudaMemcpyAsync(d, h, (size_t)n * sizeof(float), cudaMemcpyHostToDevice, stream));
    return d;
}

static float *device_inv_ells(const GPKernel *k, int d, cudaStream_t stream) {
    float *h = (float *)malloc(d * sizeof(float));
    for (int i = 0; i < d; i++)
        h[i] = 1.0 / softplus(k->raw_params[i]);
    float *dev = h2d(h, d, stream);
    free(h);
    return dev;
}

static void matern32lin_build_K(const GPKernel *k, const float *d_X, int n, int d, float sigma_n, float *d_K,
                                cudaStream_t stream) {
    KParams p = kernel_params(k);
    float *d_inv = device_inv_ells(k, d, stream);
    dim3 block(GP_BLOCK, GP_BLOCK), grid(gp_grid(n), gp_grid(n));
    matern32lin_k_kernel<<<grid, block, 0, stream>>>(d_X, d_X, d_K, d_inv, n, n, d, p.sigma_f, sigma_n, p.offset);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(d_inv, stream));
}

static void matern32lin_build_Ks(const GPKernel *k, const float *d_Xtr, const float *d_Xte, int n, int m, int d,
                                 float *d_Ks, cudaStream_t stream) {
    KParams p = kernel_params(k);
    float *d_inv = device_inv_ells(k, d, stream);
    dim3 block(GP_BLOCK, GP_BLOCK), grid(gp_grid(m), gp_grid(n));
    matern32lin_k_kernel<<<grid, block, 0, stream>>>(d_Xtr, d_Xte, d_Ks, d_inv, n, m, d, p.sigma_f, 0.0, p.offset);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(d_inv, stream));
}

static float matern32lin_k_self(const GPKernel *k, const float *x, int d) {
    KParams p = kernel_params(k);
    float norm2 = 0.0;
    for (int i = 0; i < d; i++)
        norm2 += x[i] * x[i];
    return p.sigma_f * (norm2 + p.offset + 1.0);
}

static void matern32lin_build_kself_batch(const GPKernel *k, const float *d_X, int m, int d, float *d_out,
                                          cudaStream_t stream) {
    KParams p = kernel_params(k);
    matern32lin_k_kself_batch<<<(m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(d_X, d_out, m, d, p.sigma_f,
                                                                                            p.offset);
    CUDA_CHECK(cudaGetLastError());
}

static void matern32lin_mll_grad(const GPKernel *k, const float *d_X, int n, int d, cublasHandle_t cublas,
                                 cudaStream_t stream, const float *d_alpha, const float *d_Kinv, float *kernel_grads) {
    int np = k->n_params;
    KParams p = kernel_params(k);
    const float one = 1.0, zero = 0.0;
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    float *d_inv = device_inv_ells(k, d, stream);
    float *d_D, *d_W, *d_temp;
    CUDA_CHECK(cudaMallocAsync(&d_D, (size_t)n * n * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_W, (size_t)n * n * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_temp, (size_t)n * n * sizeof(float), stream));
    dim3 block(GP_BLOCK, GP_BLOCK), grid(gp_grid(n), gp_grid(n));

    matern32lin_k_build_D_ard<<<grid, block, 0, stream>>>(d_X, d_D, d_inv, n, d);
    CUDA_CHECK(cudaGetLastError());
    matern32lin_k_compute_W<<<grid, block, 0, stream>>>(d_alpha, d_Kinv, d_W, n);
    CUDA_CHECK(cudaGetLastError());

    for (int dd = 0; dd < d; dd++) {
        float ell_d = softplus(k->raw_params[dd]);
        float inv_ell3 = 1.0 / (ell_d * ell_d * ell_d);
        matern32lin_k_dk_dell_d<<<grid, block, 0, stream>>>(d_X, d_D, d_temp, n, d, dd, p.sigma_f, inv_ell3);
        CUDA_CHECK(cudaGetLastError());
        float dot_val;
        CUBLAS_CHECK(cublasSdot(cublas, n * n, d_W, 1, d_temp, 1, &dot_val));
        kernel_grads[dd] = 0.5 * dot_val * softplus_grad(k->raw_params[dd]);
    }

    matern32lin_k_kernel<<<grid, block, 0, stream>>>(d_X, d_X, d_temp, d_inv, n, n, d, p.sigma_f, 0.0, p.offset);
    CUDA_CHECK(cudaGetLastError());
    float dot_val;
    CUBLAS_CHECK(cublasSdot(cublas, n * n, d_W, 1, d_temp, 1, &dot_val));
    kernel_grads[SF_IDX(np)] = 0.5 * dot_val / p.sigma_f * softplus_grad(k->raw_params[SF_IDX(np)]);

    float *d_ones, *d_wrow;
    CUDA_CHECK(cudaMallocAsync(&d_ones, (size_t)n * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_wrow, (size_t)n * sizeof(float), stream));
    matern32lin_k_fill<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(d_ones, n, 1.0);
    CUDA_CHECK(cudaGetLastError());
    float w_sum;
    CUBLAS_CHECK(cublasSgemv(cublas, CUBLAS_OP_N, n, n, &one, d_W, n, d_ones, 1, &zero, d_wrow, 1));
    CUBLAS_CHECK(cublasSdot(cublas, n, d_wrow, 1, d_ones, 1, &w_sum));
    kernel_grads[OFF_IDX(np)] = 0.5 * p.sigma_f * w_sum * softplus_grad(k->raw_params[OFF_IDX(np)]);
    CUDA_CHECK(cudaFreeAsync(d_ones, stream));
    CUDA_CHECK(cudaFreeAsync(d_wrow, stream));

    CUDA_CHECK(cudaFreeAsync(d_inv, stream));
    CUDA_CHECK(cudaFreeAsync(d_D, stream));
    CUDA_CHECK(cudaFreeAsync(d_W, stream));
    CUDA_CHECK(cudaFreeAsync(d_temp, stream));
}

static void matern32lin_destroy(GPKernel *k) { free(k); }

GPKernel *gp_kernel_matern32_linear(int dim, float lengthscale, float outputscale, float offset) {
    int n_params = dim + 2;
    GPKernel *k = (GPKernel *)malloc(sizeof(GPKernel) + (size_t)n_params * sizeof(float));
    k->n_params = n_params;
    k->raw_params = (float *)(k + 1);
    memcpy(k->tag, GP_KERNEL_TAG_MATERN32_LINEAR, 4);
    for (int i = 0; i < dim; i++)
        k->raw_params[i] = inv_softplus(lengthscale);
    k->raw_params[SF_IDX(n_params)] = inv_softplus(outputscale);
    k->raw_params[OFF_IDX(n_params)] = inv_softplus(offset);
    k->build_K = matern32lin_build_K;
    k->build_Ks = matern32lin_build_Ks;
    k->build_kself_batch = matern32lin_build_kself_batch;
    k->k_self = matern32lin_k_self;
    k->mll_grad = matern32lin_mll_grad;
    k->destroy = matern32lin_destroy;
    return k;
}

float gp_kernel_get_lengthscale(const GPKernel *k, int d) { return softplus(k->raw_params[d]); }
float gp_kernel_get_outputscale(const GPKernel *k) { return softplus(k->raw_params[SF_IDX(k->n_params)]); }
float gp_kernel_get_offset(const GPKernel *k) { return softplus(k->raw_params[OFF_IDX(k->n_params)]); }
void gp_kernel_set_lengthscale(GPKernel *k, int d, float v) { k->raw_params[d] = inv_softplus(v); }
void gp_kernel_set_outputscale(GPKernel *k, float v) { k->raw_params[SF_IDX(k->n_params)] = inv_softplus(v); }
void gp_kernel_set_offset(GPKernel *k, float v) { k->raw_params[OFF_IDX(k->n_params)] = inv_softplus(v); }

#endif // GP_CUDA_KERNEL_CU
