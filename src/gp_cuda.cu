// Exact GP regression, CUDA (cuBLAS/cuSOLVER).

#ifndef GP_CUDA_CU
#define GP_CUDA_CU

#define _USE_MATH_DEFINES
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gp_cuda_kernel.cu"

// clang-format off
static inline void _cusolver_check(cusolverStatus_t s, const char* f, int l) {
    if (s != CUSOLVER_STATUS_SUCCESS) { fprintf(stderr, "cuSOLVER error %s:%d: %d\n", f, l, (int)s); abort(); }
}
#define CUSOLVER_CHECK(call) _cusolver_check((call), __FILE__, __LINE__)
// clang-format on

typedef struct {
    int dim, n, cap;
    float *d_X;
    float *d_y;
    float *d_L;
    float *d_alpha;
    float *d_work;
    int lwork;
    int *d_info;
    float raw_noise;
    float dedup_threshold;
    cublasHandle_t cublas;
    cusolverDnHandle_t cusolver;
    GPKernel *kernel;
} GaussianProcess;

__global__ void gp_k_add_diag(float *A, int n, float val) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n)
        A[(size_t)i * (n + 1)] += val;
}

__global__ void gp_k_extract_diag(const float *A, float *d, int n) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n)
        d[i] = A[(size_t)i * (n + 1)];
}

__global__ void gp_k_var_from_chol(float *vars, const float *V, int n, int m) {
    int j = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (j >= m)
        return;
    float sq = 0.0;
    const float *col = V + (size_t)j * n;
    for (int i = 0; i < n; i++)
        sq += col[i] * col[i];
    float v = vars[j] - sq;
    vars[j] = v > 0.0 ? v : 0.0;
}

#define KD_LEAF 16

typedef struct {
    int lo, hi, left, right, split_dim;
    float split_val;
} KDNode;
typedef struct {
    const float *X;
    int n, dim, n_nodes;
    int *idx;
    KDNode *nodes;
} KDTree;

static int g_kd_split_dim, g_kd_d;
static const float *g_kd_X;

static int kd_cmp_fn(const void *a, const void *b) {
    float va = g_kd_X[*(const int *)a * g_kd_d + g_kd_split_dim];
    float vb = g_kd_X[*(const int *)b * g_kd_d + g_kd_split_dim];
    return (va > vb) - (va < vb);
}

static void kd_build(KDTree *t, int node, int lo, int hi) {
    KDNode *nd = &t->nodes[node];
    nd->lo = lo;
    nd->hi = hi;
    nd->left = nd->right = -1;
    nd->split_dim = -1;
    if (hi - lo <= KD_LEAF)
        return;

    int best = 0;
    float spread = -1.0;
    for (int k = 0; k < t->dim; k++) {
        float mn = t->X[t->idx[lo] * t->dim + k], mx = mn;
        for (int i = lo + 1; i < hi; i++) {
            float v = t->X[t->idx[i] * t->dim + k];
            if (v < mn)
                mn = v;
            if (v > mx)
                mx = v;
        }
        if (mx - mn > spread) {
            spread = mx - mn;
            best = k;
        }
    }
    int mid = (lo + hi) / 2;
    g_kd_split_dim = best;
    g_kd_d = t->dim;
    g_kd_X = t->X;
    qsort(t->idx + lo, hi - lo, sizeof(int), kd_cmp_fn);
    nd->split_dim = best;
    nd->split_val = t->X[t->idx[mid] * t->dim + best];
    int ln = t->n_nodes++, rn = t->n_nodes++;
    nd->left = ln;
    nd->right = rn;
    kd_build(t, ln, lo, mid);
    kd_build(t, rn, mid, hi);
}

static KDTree kd_create(const float *X, int n, int dim) {
    KDTree t = {.X = X, .n = n, .dim = dim, .n_nodes = 1};
    t.idx = (int *)malloc(n * sizeof(int));
    t.nodes = (KDNode *)malloc(2 * n * sizeof(KDNode));
    for (int i = 0; i < n; i++)
        t.idx[i] = i;
    kd_build(&t, 0, 0, n);
    return t;
}

static void kd_query_ball(const KDTree *t, int node, const float *q, float r, float r2, int *out, int *cnt) {
    const KDNode *nd = &t->nodes[node];
    if (nd->split_dim < 0) {
        for (int i = nd->lo; i < nd->hi; i++) {
            int p = t->idx[i];
            float sq = 0.0;
            for (int k = 0; k < t->dim; k++) {
                float df = q[k] - t->X[p * t->dim + k];
                sq += df * df;
            }
            if (sq <= r2)
                out[(*cnt)++] = p;
        }
        return;
    }
    float dv = q[nd->split_dim] - nd->split_val;
    if (nd->left >= 0 && dv <= r)
        kd_query_ball(t, nd->left, q, r, r2, out, cnt);
    if (nd->right >= 0 && dv >= -r)
        kd_query_ball(t, nd->right, q, r, r2, out, cnt);
}

static int gp_filter_near_duplicates(const float *X, int n, int dim, float threshold, int *kept_indices) {
    if (n <= 0)
        return 0;
    if (n == 1) {
        kept_indices[0] = 0;
        return 1;
    }

    KDTree tree = kd_create(X, n, dim);
    int *keep = (int *)malloc(n * sizeof(int));
    int *nearby = (int *)malloc(n * sizeof(int));
    float r2 = threshold * threshold;
    for (int i = 0; i < n; i++)
        keep[i] = 1;

    for (int i = n - 1; i >= 0; i--) {
        if (!keep[i])
            continue;
        int cnt = 0;
        kd_query_ball(&tree, 0, &X[(size_t)i * dim], threshold, r2, nearby, &cnt);
        for (int j = 0; j < cnt; j++)
            if (nearby[j] != i)
                keep[nearby[j]] = 0;
    }

    int count = 0;
    for (int i = 0; i < n; i++)
        if (keep[i])
            kept_indices[count++] = i;

    free(nearby);
    free(keep);
    free(tree.idx);
    free(tree.nodes);
    return count;
}

float gp_get_noise(const GaussianProcess *gp) { return softplus(gp->raw_noise); }
void gp_set_noise(GaussianProcess *gp, float v) { gp->raw_noise = inv_softplus(v); }

GaussianProcess *gp_create(int dim, int cap, GPKernel *kernel, float noise) {
    GaussianProcess *gp = (GaussianProcess *)calloc(1, sizeof(GaussianProcess));
    gp->dim = dim;
    gp->cap = cap;
    gp->kernel = kernel;
    gp_set_noise(gp, noise);
    CUDA_CHECK(cudaMalloc(&gp->d_X, (size_t)cap * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gp->d_y, (size_t)cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gp->d_L, (size_t)cap * cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gp->d_alpha, (size_t)cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gp->d_info, sizeof(int)));
    CUBLAS_CHECK(cublasCreate(&gp->cublas));
    CUSOLVER_CHECK(cusolverDnCreate(&gp->cusolver));
    return gp;
}

void gp_destroy(GaussianProcess *gp) {
    if (!gp)
        return;
    cudaFree(gp->d_X);
    cudaFree(gp->d_y);
    cudaFree(gp->d_L);
    cudaFree(gp->d_alpha);
    cudaFree(gp->d_work);
    cudaFree(gp->d_info);
    cublasDestroy(gp->cublas);
    cusolverDnDestroy(gp->cusolver);
    if (gp->kernel && gp->kernel->destroy)
        gp->kernel->destroy(gp->kernel);
    free(gp);
}

static int run_potrf(GaussianProcess *gp, float *d_K, int n, cudaStream_t stream) {
    CUSOLVER_CHECK(cusolverDnSetStream(gp->cusolver, stream));
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(gp->cusolver, CUBLAS_FILL_MODE_LOWER, n, d_K, n, &lwork));
    if (lwork > gp->lwork) {
        cudaFree(gp->d_work);
        CUDA_CHECK(cudaMalloc(&gp->d_work, (size_t)lwork * sizeof(float)));
        gp->lwork = lwork;
    }
    CUSOLVER_CHECK(
        cusolverDnSpotrf(gp->cusolver, CUBLAS_FILL_MODE_LOWER, n, d_K, n, gp->d_work, gp->lwork, gp->d_info));
    int h_info;
    CUDA_CHECK(cudaMemcpyAsync(&h_info, gp->d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return h_info;
}

int gp_recompute(GaussianProcess *gp, cudaStream_t stream) {
    int n = gp->n;
    if (n == 0)
        return 0;

    gp->kernel->build_K(gp->kernel, gp->d_X, n, gp->dim, gp_get_noise(gp), gp->d_L, stream);

    int info = run_potrf(gp, gp->d_L, n, stream);
    if (info != 0) {
        gp->kernel->build_K(gp->kernel, gp->d_X, n, gp->dim, gp_get_noise(gp), gp->d_L, stream);
        gp_k_add_diag<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(gp->d_L, n, 1e-8);
        CUDA_CHECK(cudaGetLastError());
        info = run_potrf(gp, gp->d_L, n, stream);
        if (info != 0)
            return -2;
    }

    CUDA_CHECK(cudaMemcpyAsync(gp->d_alpha, gp->d_y, (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUBLAS_CHECK(cublasSetStream(gp->cublas, stream));
    CUBLAS_CHECK(cublasStrsv(gp->cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, gp->d_L, n,
                             gp->d_alpha, 1));
    CUBLAS_CHECK(cublasStrsv(gp->cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n, gp->d_L, n,
                             gp->d_alpha, 1));
    return 0;
}

int gp_fit(GaussianProcess *gp, const float *X, const float *y, int n, cudaStream_t stream) {
    if (n > gp->cap)
        return -1;

    if (gp->dedup_threshold > 0.0 && n > 1) {
        int *idx = (int *)malloc(n * sizeof(int));
        int n_kept = gp_filter_near_duplicates(X, n, gp->dim, gp->dedup_threshold, idx);
        float *X_c = (float *)malloc((size_t)n_kept * gp->dim * sizeof(float));
        float *y_c = (float *)malloc((size_t)n_kept * sizeof(float));
        for (int i = 0; i < n_kept; i++) {
            memcpy(&X_c[i * gp->dim], &X[idx[i] * gp->dim], gp->dim * sizeof(float));
            y_c[i] = y[idx[i]];
        }
        free(idx);
        CUDA_CHECK(
            cudaMemcpyAsync(gp->d_X, X_c, (size_t)n_kept * gp->dim * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(gp->d_y, y_c, (size_t)n_kept * sizeof(float), cudaMemcpyHostToDevice, stream));
        free(X_c);
        free(y_c);
        gp->n = n_kept;
    } else {
        CUDA_CHECK(cudaMemcpyAsync(gp->d_X, X, (size_t)n * gp->dim * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(gp->d_y, y, (size_t)n * sizeof(float), cudaMemcpyHostToDevice, stream));
        gp->n = n;
    }
    return gp_recompute(gp, stream);
}

static void predict_dev(const GaussianProcess *gp, const float *d_Xte, float *d_means, float *d_vars, int m,
                        cudaStream_t stream) {
    int n = gp->n, d = gp->dim;
    const float one = 1.0, zero = 0.0;
    CUBLAS_CHECK(cublasSetStream(gp->cublas, stream));

    float *d_Ks;
    CUDA_CHECK(cudaMallocAsync(&d_Ks, (size_t)n * m * sizeof(float), stream));
    gp->kernel->build_Ks(gp->kernel, gp->d_X, d_Xte, n, m, d, d_Ks, stream);
    CUBLAS_CHECK(cublasSgemv(gp->cublas, CUBLAS_OP_T, n, m, &one, d_Ks, n, gp->d_alpha, 1, &zero, d_means, 1));

    if (d_vars) {
        gp->kernel->build_kself_batch(gp->kernel, d_Xte, m, d, d_vars, stream);
        CUBLAS_CHECK(cublasStrsm(gp->cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                 CUBLAS_DIAG_NON_UNIT, n, m, &one, gp->d_L, n, d_Ks, n));
        gp_k_var_from_chol<<<(m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(d_vars, d_Ks, n, m);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaFreeAsync(d_Ks, stream));
}

void gp_predict(const GaussianProcess *gp, const float *Xs, float *means, float *vars, int m, cudaStream_t stream) {
    int n = gp->n;
    if (n == 0) {
        for (int j = 0; j < m; j++) {
            means[j] = 0.0;
            if (vars)
                vars[j] = gp->kernel->k_self(gp->kernel, &Xs[j * gp->dim], gp->dim);
        }
        return;
    }

    float *d_Xte;
    CUDA_CHECK(cudaMallocAsync(&d_Xte, (size_t)m * gp->dim * sizeof(float), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_Xte, Xs, (size_t)m * gp->dim * sizeof(float), cudaMemcpyHostToDevice, stream));

    float *d_means, *d_vars = NULL;
    CUDA_CHECK(cudaMallocAsync(&d_means, (size_t)m * sizeof(float), stream));
    if (vars)
        CUDA_CHECK(cudaMallocAsync(&d_vars, (size_t)m * sizeof(float), stream));

    predict_dev(gp, d_Xte, d_means, d_vars, m, stream);
    CUDA_CHECK(cudaFreeAsync(d_Xte, stream));
    CUDA_CHECK(cudaMemcpyAsync(means, d_means, (size_t)m * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaFreeAsync(d_means, stream));
    if (vars) {
        CUDA_CHECK(cudaMemcpyAsync(vars, d_vars, (size_t)m * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaFreeAsync(d_vars, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void gp_predict_d(const GaussianProcess *gp, const float *d_Xs, float *d_means, float *d_vars, int m,
                  cudaStream_t stream) {
    if (gp->n == 0) {
        CUDA_CHECK(cudaMemsetAsync(d_means, 0, (size_t)m * sizeof(float), stream));
        if (d_vars)
            gp->kernel->build_kself_batch(gp->kernel, d_Xs, m, gp->dim, d_vars, stream);
        return;
    }
    predict_dev(gp, d_Xs, d_means, d_vars, m, stream);
}

float gp_marginal_log_likelihood(const GaussianProcess *gp) {
    int n = gp->n;
    if (n == 0)
        return 0.0;

    CUBLAS_CHECK(cublasSetStream(gp->cublas, 0));
    float data_fit;
    CUBLAS_CHECK(cublasSdot(gp->cublas, n, gp->d_y, 1, gp->d_alpha, 1, &data_fit));

    float *d_diag;
    CUDA_CHECK(cudaMalloc(&d_diag, (size_t)n * sizeof(float)));
    gp_k_extract_diag<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(gp->d_L, d_diag, n);
    CUDA_CHECK(cudaGetLastError());

    float *h_diag = (float *)malloc(n * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_diag, d_diag, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_diag));

    float log_det = 0.0;
    for (int i = 0; i < n; i++)
        log_det += log(h_diag[i]);
    free(h_diag);

    return (-0.5 * data_fit - log_det - 0.5 * n * log(2.0 * M_PI)) / n;
}

void gp_mll_grad(const GaussianProcess *gp, float *d_raw_noise, float *kernel_grads, cudaStream_t stream) {
    int n = gp->n;
    if (n == 0) {
        if (d_raw_noise)
            *d_raw_noise = 0.0;
        if (kernel_grads)
            for (int i = 0; i < gp->kernel->n_params; i++)
                kernel_grads[i] = 0.0;
        return;
    }

    CUSOLVER_CHECK(cusolverDnSetStream(gp->cusolver, stream));
    CUBLAS_CHECK(cublasSetStream(gp->cublas, stream));

    float *d_Kinv;
    CUDA_CHECK(cudaMallocAsync(&d_Kinv, (size_t)n * n * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_Kinv, 0, (size_t)n * n * sizeof(float), stream));
    gp_k_add_diag<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(d_Kinv, n, 1.0);
    CUDA_CHECK(cudaGetLastError());
    CUSOLVER_CHECK(cusolverDnSpotrs(gp->cusolver, CUBLAS_FILL_MODE_LOWER, n, n, gp->d_L, n, d_Kinv, n, gp->d_info));

    if (d_raw_noise) {
        float term1, term2;
        CUBLAS_CHECK(cublasSdot(gp->cublas, n, gp->d_alpha, 1, gp->d_alpha, 1, &term1));
        CUBLAS_CHECK(cublasSasum(gp->cublas, n, d_Kinv, n + 1, &term2));
        *d_raw_noise = 0.5 * (term1 - term2) * softplus_grad(gp->raw_noise) / n;
    }

    if (kernel_grads) {
        for (int i = 0; i < gp->kernel->n_params; i++)
            kernel_grads[i] = 0.0;
        gp->kernel->mll_grad(gp->kernel, gp->d_X, n, gp->dim, gp->cublas, stream, gp->d_alpha, d_Kinv, kernel_grads);
        for (int i = 0; i < gp->kernel->n_params; i++)
            kernel_grads[i] /= n;
    }
    CUDA_CHECK(cudaFreeAsync(d_Kinv, stream));
}

typedef struct {
    const char *tag;
    GPKernel *(*make)(float *, int);
} KernelFactory;

static GPKernel *kf_matern32lin(float *rp, int np) {
    int dim = np - 2;
    GPKernel *k = gp_kernel_matern32_linear(dim, 1.0, 1.0, 1.0);
    memcpy(k->raw_params, rp, (size_t)np * sizeof(float));
    return k;
}

static const KernelFactory kernel_registry[] = {{GP_KERNEL_TAG_MATERN32_LINEAR, kf_matern32lin}, {NULL, NULL}};

static GPKernel *kernel_from_tag(const char *tag, float *rp, int np) {
    for (int i = 0; kernel_registry[i].tag; i++)
        if (memcmp(tag, kernel_registry[i].tag, 4) == 0)
            return kernel_registry[i].make(rp, np);
    return NULL;
}

int gp_save(const GaussianProcess *gp, const char *path) {
    if (gp->n == 0)
        return -1;
    CUDA_CHECK(cudaDeviceSynchronize());
    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;

    int n = gp->n, dim = gp->dim, np = gp->kernel->n_params;
    fwrite("GC04", 1, 4, f);
    fwrite(&dim, sizeof(int), 1, f);
    fwrite(&n, sizeof(int), 1, f);
    fwrite(&gp->raw_noise, sizeof(float), 1, f);
    fwrite(gp->kernel->tag, 1, 4, f);
    fwrite(&np, sizeof(int), 1, f);
    fwrite(gp->kernel->raw_params, sizeof(float), np, f);

    const float *d_ptrs[] = {gp->d_X, gp->d_y, gp->d_L, gp->d_alpha};
    const size_t sizes[] = {(size_t)n * dim, (size_t)n, (size_t)n * n, (size_t)n};
    for (int i = 0; i < 4; i++) {
        float *h = (float *)malloc(sizes[i] * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h, d_ptrs[i], sizes[i] * sizeof(float), cudaMemcpyDeviceToHost));
        fwrite(h, sizeof(float), sizes[i], f);
        free(h);
    }

    fclose(f);
    return 0;
}

GaussianProcess *gp_load(const char *path, int extra_cap) {
    GaussianProcess *gp = NULL;
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;

#define RD(ptr, sz, cnt)                                                                                               \
    if (fread(ptr, sz, cnt, f) != (size_t)(cnt))                                                                       \
    break

    do {
        char magic[4], ktag[4];
        int dim, n, np;
        float rn;

        RD(magic, 1, 4);
        if (memcmp(magic, "GC04", 4) != 0)
            break;
        RD(&dim, sizeof(int), 1);
        RD(&n, sizeof(int), 1);
        RD(&rn, sizeof(float), 1);
        RD(ktag, 1, 4);
        RD(&np, sizeof(int), 1);

        float *rp = (float *)malloc((size_t)np * sizeof(float));
        if (!rp)
            break;
        if (fread(rp, sizeof(float), np, f) != (size_t)np) {
            free(rp);
            break;
        }
        GPKernel *k = kernel_from_tag(ktag, rp, np);
        free(rp);
        if (!k)
            break;

        int cap = n + (extra_cap > 0 ? extra_cap : 0);
        gp = gp_create(dim, cap, k, 1.0);
        gp->raw_noise = rn;
        gp->n = n;

        float *d_ptrs[] = {gp->d_X, gp->d_y, gp->d_L, gp->d_alpha};
        const size_t sizes[] = {(size_t)n * dim, (size_t)n, (size_t)n * n, (size_t)n};
        int ok = 1;
        for (int i = 0; i < 4 && ok; i++) {
            float *h = (float *)malloc(sizes[i] * sizeof(float));
            if (fread(h, sizeof(float), sizes[i], f) != sizes[i]) {
                free(h);
                ok = 0;
            } else {
                CUDA_CHECK(cudaMemcpy(d_ptrs[i], h, sizes[i] * sizeof(float), cudaMemcpyHostToDevice));
                free(h);
            }
        }
        if (!ok)
            break;

        fclose(f);
        return gp;
    } while (0);

#undef RD
    if (gp)
        gp_destroy(gp);
    fclose(f);
    return NULL;
}

#endif // GP_CUDA_CU
