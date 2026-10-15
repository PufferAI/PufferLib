// protein.cu -- Protein hyperparameter optimizer

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PROTEIN_EPSILON 1e-6f
#define PROTEIN_NUM_COST_RATIOS 6
#define PROTEIN_ACQ_MAX_CAP 65536
#define PROTEIN_COST_QUANTILE 0.97f
#define PROTEIN_MIN_OBS_NO_FAIL 100
#define PROTEIN_COST_GROWTH 1.01f
#define PROTEIN_CLF_ITERS 100
#define PROTEIN_THRESHOLD_COST_CAP 1.2f
#define PROTEIN_THRESHOLD_FALLBACK 0.9f
#define NOISE_PRIOR_MU (-4.60517f)
#define NOISE_PRIOR_SIGMA 0.5f
#define SP_LB 1e-4
#define SF_IDX(np) ((np)-2)
#define OFF_IDX(np) ((np)-1)
#define GP_BLOCK 16
#define KD_LEAF 16

typedef enum {
    SPACE_LINEAR,
    SPACE_LOG,
    SPACE_POW2,
    SPACE_LOGIT,
} SpaceType;

typedef struct {
    SpaceType type;
    float min, max, scale;
    int is_integer;
} Space;

typedef struct {
    Space *spaces;
    int num;
    int cost_idx;
    int optimize_direction;
} SweepSpace;

// Map raw hyperparam <-> interpolation domain (identity / log10 / log2 / logit).
static inline float space_fwd(SpaceType type, float v) {
    if (type == SPACE_LOG) return log10f(v);
    if (type == SPACE_POW2) return log2f(v);
    if (type == SPACE_LOGIT) return log10f(1.0f - v);
    return v;
}

static inline float space_normalize(const Space *s, float value) {
    if (s->type == SPACE_LOGIT) {
        value = fmaxf(s->min, fminf(value, s->max));
    }
    float t = space_fwd(s->type, value);
    float lo = space_fwd(s->type, s->min);
    float hi = space_fwd(s->type, s->max);
    return 2.0f * ((t - lo) / (hi - lo)) - 1.0f;
}

static inline float space_unnormalize(const Space *s, float norm) {
    float u = (norm + 1.0f) * 0.5f;
    float lo = space_fwd(s->type, s->min);
    float hi = space_fwd(s->type, s->max);
    float t = u * (hi - lo) + lo;
    if (s->type == SPACE_POW2) {
        t = roundf(t);
    }
    float val = t;
    if (s->type == SPACE_LOG) val = powf(10.0f, t);
    if (s->type == SPACE_POW2) val = powf(2.0f, t);
    if (s->type == SPACE_LOGIT) val = 1.0f - powf(10.0f, t);
    if (s->is_integer) {
        val = roundf(val);
    }
    return val;
}

// Softplus + Matern32+linear kernel
__host__ __device__ __forceinline__ float softplus(float x) {
    return (x > 20.0 ? x : log1p(exp(x))) + SP_LB;
}
__host__ __device__ __forceinline__ float inv_softplus(float x) {
    float v = x - SP_LB;
    return v > 20.0 ? v : log(expm1(v));
}
__host__ __device__ __forceinline__ float softplus_grad(float x) {
    return 1.0 / (1.0 + exp(-x));
}

typedef struct {
    int n_params;
    float *raw_params;
} GPKernel;

static inline int gp_grid(int n) {
    return (n + GP_BLOCK - 1) / GP_BLOCK;
}

// Gram / cross-covariance: K = sigma_f * (x·x' + offset + Matern32(r))
__global__ void matern32lin_k_kernel(const float *__restrict__ X1,
        const float *__restrict__ X2, float *__restrict__ K,
        const float *__restrict__ inv_ells, int n, int m, int d, float sigma_f,
        float diag_noise, float offset) {
    int col = blockIdx.x * GP_BLOCK + threadIdx.x;
    int row = blockIdx.y * GP_BLOCK + threadIdx.y;
    if (row >= n || col >= m) {
        return;
    }
    float dot = 0.0, r2 = 0.0;
    for (int k = 0; k < d; k++) {
        float xr = X1[row * d + k], xc = X2[col * d + k];
        dot += xr * xc;
        float diff = (xr - xc) * inv_ells[k];
        r2 += diff * diff;
    }
    float u = sqrt(3.0) * sqrt(r2);
    float val = sigma_f * (dot + offset + (1.0 + u) * exp(-u));
    if (diag_noise != 0.0 && row == col) {
        val += diag_noise;
    }
    K[col * n + row] = val;
}

// Write constant onto diagonal (stride n+1). Used to form I for spotrs → K^{-1}.
__global__ void gp_k_set_diag(float *A, int n, float val) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) {
        A[i * (n + 1)] = val;
    }
}

// Copy matrix diagonal (stride n+1) to a dense vector (L_ii for host logdet).
__global__ void gp_k_extract_diag(const float *A, float *d, int n) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) {
        d[i] = A[i * (n + 1)];
    }
}

// Fused MLL kernel grads: per (i,j) W=ααᵀ−K⁻¹ terms for ∂/∂ℓ, ∂/∂σ_f, ∂/∂offset.
// Block partials via block_reduce_sum into out_partials[bid * (d+2) + ·]; host folds
// in fixed block order. Dynamic smem: (d+2) * GP_BLOCK² floats.
__global__ void matern32lin_mll_grad_fuse(const float *__restrict__ X,
        const float *__restrict__ alpha, const float *__restrict__ Kinv,
        const float *__restrict__ inv_ells, int n, int d, float sigma_f,
        float offset, float *__restrict__ out_partials) {
    extern __shared__ float sh[];
    int col = blockIdx.x * GP_BLOCK + threadIdx.x;
    int row = blockIdx.y * GP_BLOCK + threadIdx.y;
    int tid = threadIdx.y * GP_BLOCK + threadIdx.x;
    int nthreads = GP_BLOCK * GP_BLOCK;
    int nsum = d + 2;

    for (int c = 0; c < nsum; c++) {
        sh[c * nthreads + tid] = 0.0f;
    }
    if (row < n && col < n) {
        float dot = 0.0f, r2 = 0.0f;
        for (int k = 0; k < d; k++) {
            float xr = X[row * d + k], xc = X[col * d + k];
            dot += xr * xc;
            float t = (xr - xc) * inv_ells[k];
            r2 += t * t;
        }
        float u = sqrtf(3.0f) * sqrtf(r2);
        float e = expf(-u);
        float W = alpha[row] * alpha[col] - Kinv[col * n + row];
        for (int dd = 0; dd < d; dd++) {
            float diff = X[row * d + dd] - X[col * d + dd];
            float inv = inv_ells[dd];
            sh[dd * nthreads + tid] =
                W * (sigma_f * 3.0f * diff * diff * inv * inv * inv * e);
        }
        sh[d * nthreads + tid] = W * (dot + offset + (1.0f + u) * e);
        sh[(d + 1) * nthreads + tid] = W;
    }
    int bid = blockIdx.y * gridDim.x + blockIdx.x;
    block_reduce_sum(sh, out_partials + bid * nsum, tid, nthreads, nsum);
}

typedef struct {
    int dim, n, cap;
    float *d_X, *d_y, *d_L, *d_alpha, *d_work;
    int lwork;
    int *d_info;
    float raw_noise;
    float mean;
    cublasHandle_t cublas;
    cusolverDnHandle_t cusolver;
    GPKernel *kernel;
    float *d_inv_ells, *d_diag, *d_Kinv, *d_partials, *d_Ks, *d_ones;
    float *h_kg, *h_diag, *h_partials;
} GaussianProcess;

static void matern32lin_build_K(GaussianProcess *gp, const float *d_X1,
        const float *d_X2, int n, int m, float diag_noise, float *d_K,
        cudaStream_t stream) {
    int d = gp->dim, np = gp->kernel->n_params;
    float h_inv[d];
    for (int i = 0; i < d; i++) {
        h_inv[i] = 1.0f / softplus(gp->kernel->raw_params[i]);
    }
    cudaMemcpyAsync(gp->d_inv_ells, h_inv, d * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    float sigma_f = softplus(gp->kernel->raw_params[SF_IDX(np)]);
    float offset = softplus(gp->kernel->raw_params[OFF_IDX(np)]);
    dim3 block(GP_BLOCK, GP_BLOCK), grid(gp_grid(m), gp_grid(n));
    matern32lin_k_kernel<<<grid, block, 0, stream>>>(
        d_X1, d_X2, d_K, gp->d_inv_ells, n, m, d, sigma_f, diag_noise, offset);
}

static void gp_init(GaussianProcess *gp, int dim, int cap, int pred_m, float noise) {
    memset(gp, 0, sizeof(*gp));
    gp->dim = dim;
    gp->cap = cap;
    int n_params = dim + 2;
    int nsum = dim + 2;
    int nblocks = gp_grid(cap) * gp_grid(cap);
    GPKernel *k = (GPKernel *)malloc(sizeof(GPKernel) + n_params * sizeof(float));
    k->n_params = n_params;
    k->raw_params = (float *)(k + 1);
    for (int i = 0; i < dim; i++) {
        k->raw_params[i] = inv_softplus(1.0f);
    }
    k->raw_params[SF_IDX(n_params)] = inv_softplus(1.0f);
    k->raw_params[OFF_IDX(n_params)] = inv_softplus(1.0f);
    gp->kernel = k;
    gp->raw_noise = inv_softplus(noise);
    gp->mean = 0.0f;
    cudaMalloc(&gp->d_X, cap * dim * sizeof(float));
    cudaMalloc(&gp->d_y, cap * sizeof(float));
    cudaMalloc(&gp->d_L, cap * cap * sizeof(float));
    cudaMalloc(&gp->d_alpha, cap * sizeof(float));
    cudaMalloc(&gp->d_info, sizeof(int));
    cudaMalloc(&gp->d_inv_ells, dim * sizeof(float));
    cudaMalloc(&gp->d_diag, cap * sizeof(float));
    cudaMalloc(&gp->d_Kinv, cap * cap * sizeof(float));
    cudaMalloc(&gp->d_partials, nblocks * nsum * sizeof(float));
    // Dominant device buffer: cap × acq_cap floats (×2 GPs)
    assert(cudaMalloc(&gp->d_Ks, cap * pred_m * sizeof(float)) == cudaSuccess);
    cudaMalloc(&gp->d_ones, sizeof(float));
    float one = 1.0f;
    cudaMemcpy(gp->d_ones, &one, sizeof(float), cudaMemcpyHostToDevice);
    gp->h_kg = (float *)malloc(n_params * sizeof(float));
    gp->h_diag = (float *)malloc(cap * sizeof(float));
    gp->h_partials = (float *)malloc(nblocks * nsum * sizeof(float));
    cublasCreate(&gp->cublas);
    cusolverDnCreate(&gp->cusolver);
    cusolverDnSpotrf_bufferSize(
        gp->cusolver, CUBLAS_FILL_MODE_LOWER, cap, gp->d_L, cap, &gp->lwork);
    cudaMalloc(&gp->d_work, gp->lwork * sizeof(float));
}

static int run_potrf(GaussianProcess *gp, float *d_K, int n, cudaStream_t stream) {
    cusolverDnSetStream(gp->cusolver, stream);
    cusolverDnSpotrf(gp->cusolver, CUBLAS_FILL_MODE_LOWER,
        n, d_K, n, gp->d_work, gp->lwork, gp->d_info);
    int h_info;
    cudaMemcpyAsync(&h_info, gp->d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return h_info;
}

static int gp_recompute(GaussianProcess *gp, cudaStream_t stream) {
    int n = gp->n;
    float noise = softplus(gp->raw_noise);
    matern32lin_build_K(gp, gp->d_X, gp->d_X, n, n, noise, gp->d_L, stream);
    int info = run_potrf(gp, gp->d_L, n, stream);
    if (info != 0) {
        matern32lin_build_K(gp, gp->d_X, gp->d_X, n, n, noise + 1e-8f, gp->d_L, stream);
        info = run_potrf(gp, gp->d_L, n, stream);
    }
    if (info != 0) {
        return -2;
    }
    // alpha = K^{-1}(y - mean)
    cudaMemcpyAsync(gp->d_alpha, gp->d_y, n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cublasSetStream(gp->cublas, stream);
    float neg_mean = -gp->mean;
    cublasSaxpy(gp->cublas, n, &neg_mean, gp->d_ones, 0, gp->d_alpha, 1);
    cublasStrsv(gp->cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
        n, gp->d_L, n, gp->d_alpha, 1);
    cublasStrsv(gp->cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
        n, gp->d_L, n, gp->d_alpha, 1);
    return 0;
}

static void gp_predict(GaussianProcess *gp, const float *d_Xte, float *d_means, int m,
        cudaStream_t stream) {
    int n = gp->n;
    const float one = 1.0, zero = 0.0;
    cublasSetStream(gp->cublas, stream);
    matern32lin_build_K(gp, gp->d_X, d_Xte, n, m, 0.0f, gp->d_Ks, stream);
    cublasSgemv(gp->cublas, CUBLAS_OP_T, n, m, &one, gp->d_Ks, n, gp->d_alpha, 1,
        &zero, d_means, 1);
    cublasSaxpy(gp->cublas, m, &gp->mean, gp->d_ones, 0, d_means, 1);
}

// Train kernel hyperparameters with Adam (score GP and cost GP).
static float gp_train(GaussianProcess *gp, float *opt_m, float *opt_v,
        float *opt_vmax, int *opt_t, float lr, const float *X, const float *y,
        int n_data, int training_iter, cudaStream_t stream) {
    cudaMemcpyAsync(gp->d_X, X, n_data * gp->dim * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(gp->d_y, y, n_data * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    gp->n = n_data;
    if (gp_recompute(gp, stream) != 0) {
        return 0.0f;
    }

    int n_kp = gp->kernel->n_params;
    int n_opt = n_kp + 2;
    int d = gp->dim;
    int nsum = d + 2;
    float loss = 0.0f;
    const float beta1 = 0.9f, beta2 = 0.999f;

    for (int iter = 0; iter < training_iter; iter++) {
        if (gp_recompute(gp, stream) != 0) {
            break;
        }
        int n = gp->n;
        const GPKernel *k = gp->kernel;
        int np = k->n_params;
        cublasSetStream(gp->cublas, stream);
        float data_fit, sum_alpha;
        cublasSdot(gp->cublas, n, gp->d_y, 1, gp->d_alpha, 1, &data_fit);
        cublasSdot(gp->cublas, n, gp->d_alpha, 1, gp->d_ones, 0, &sum_alpha);
        data_fit -= gp->mean * sum_alpha;
        gp_k_extract_diag<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE, 0, stream>>>(gp->d_L, gp->d_diag, n);
        cudaMemcpyAsync(gp->h_diag, gp->d_diag, n * sizeof(float),
            cudaMemcpyDeviceToHost, stream);
        cudaMemsetAsync(gp->d_Kinv, 0, n * n * sizeof(float), stream);
        gp_k_set_diag<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            gp->d_Kinv, n, 1.0f);
        cusolverDnSetStream(gp->cusolver, stream);
        cusolverDnSpotrs(gp->cusolver, CUBLAS_FILL_MODE_LOWER, n, n, gp->d_L, n,
            gp->d_Kinv, n, gp->d_info);
        float term1, term2;
        cublasSdot(gp->cublas, n, gp->d_alpha, 1, gp->d_alpha, 1, &term1);
        cublasSasum(gp->cublas, n, gp->d_Kinv, n + 1, &term2);
        float d_noise = 0.5f * (term1 - term2) * softplus_grad(gp->raw_noise) / n;

        float sigma_f = softplus(k->raw_params[SF_IDX(np)]);
        float offset = softplus(k->raw_params[OFF_IDX(np)]);
        float h_inv[d];
        for (int i = 0; i < d; i++) {
            h_inv[i] = 1.0f / softplus(k->raw_params[i]);
        }
        cudaMemcpyAsync(gp->d_inv_ells, h_inv, d * sizeof(float),
            cudaMemcpyHostToDevice, stream);
        dim3 b(GP_BLOCK, GP_BLOCK), g(gp_grid(n), gp_grid(n));
        int nblocks = g.x * g.y;
        int nthreads = GP_BLOCK * GP_BLOCK;
        matern32lin_mll_grad_fuse<<<g, b, nsum * nthreads * sizeof(float), stream>>>(
            gp->d_X, gp->d_alpha, gp->d_Kinv, gp->d_inv_ells, n, d, sigma_f,
            offset, gp->d_partials);
        cudaMemcpyAsync(gp->h_partials, gp->d_partials,
            nblocks * nsum * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        float log_det = 0.0f;
        for (int i = 0; i < n; i++) {
            log_det += logf(gp->h_diag[i]);
        }
        float h_sums[nsum];
        for (int c = 0; c < nsum; c++) {
            h_sums[c] = 0.0f;
        }
        for (int bi = 0; bi < nblocks; bi++) {
            for (int c = 0; c < nsum; c++) {
                h_sums[c] += gp->h_partials[bi * nsum + c];
            }
        }
        float inv_n = 0.5f / n;
        for (int dd = 0; dd < d; dd++) {
            gp->h_kg[dd] = inv_n * h_sums[dd] * softplus_grad(k->raw_params[dd]);
        }
        gp->h_kg[SF_IDX(np)] =
            inv_n * h_sums[d] * softplus_grad(k->raw_params[SF_IDX(np)]);
        gp->h_kg[OFF_IDX(np)] =
            inv_n * sigma_f * h_sums[d + 1] * softplus_grad(k->raw_params[OFF_IDX(np)]);

        float mll = (-0.5f * data_fit - log_det - 0.5f * n * logf(2.0f * M_PI)) / n;
        float noise_val = softplus(gp->raw_noise);
        float ln_noise = logf(noise_val);
        float sig = softplus_grad(gp->raw_noise);
        float z = (ln_noise - NOISE_PRIOR_MU) / NOISE_PRIOR_SIGMA;
        float np_grad = ((-z / NOISE_PRIOR_SIGMA - 1.0f) * sig / noise_val) / n;
        loss = -mll - (-0.5f * z * z - ln_noise - logf(NOISE_PRIOR_SIGMA)) / n;

        (*opt_t)++;
        float bc1 = 1.0f - powf(beta1, (*opt_t));
        float bc2 = 1.0f - powf(beta2, (*opt_t));
        float d_mean = sum_alpha / n;
        for (int i = 0; i < n_opt; i++) {
            float g;
            if (i < n_kp) {
                g = -gp->h_kg[i];
            } else if (i == n_kp) {
                g = -d_noise - np_grad;
            } else {
                g = -d_mean;
            }
            opt_m[i] = beta1 * opt_m[i] + (1.0f - beta1) * g;
            opt_v[i] = beta2 * opt_v[i] + (1.0f - beta2) * g * g;
            float m_hat = opt_m[i] / bc1;
            float v_hat = opt_v[i] / bc2;
            opt_vmax[i] = fmaxf(opt_vmax[i], v_hat);
            float step = lr * m_hat / (sqrtf(opt_vmax[i]) + 1e-8f);
            if (i < n_kp) {
                gp->kernel->raw_params[i] -= step;
            } else if (i == n_kp) {
                gp->raw_noise -= step;
            } else {
                gp->mean -= step;
            }
        }
    }
    gp_recompute(gp, stream);
    return loss;
}

// Candidate sampling (score / argmax run on host after GP predict)
__global__ void protein_k_init_rng(curandStatePhilox4_32_10_t *states, int n,
        unsigned long long seed) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) {
        curand_init(seed, i, 0, &states[i]);
    }
}

__global__ void protein_k_sample(float *__restrict__ candidates,
        const float *__restrict__ centers, const float *__restrict__ scales,
        curandStatePhilox4_32_10_t *__restrict__ rng, int n_candidates,
        int n_centers, int dim, float global_scale, int cost_dim,
        float fixed_cost) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= n_candidates) {
        return;
    }
    curandStatePhilox4_32_10_t state = rng[i];
    int center = ((int)(curand_uniform(&state) * n_centers)) % n_centers;
    for (int d = 0; d < dim; d++) {
        float s = scales[d] * global_scale;
        float val = s * (2.0f * curand_uniform(&state) - 1.0f)
            + centers[center * dim + d];
        val = fmaxf(-1.0f, fminf(1.0f, val));
        candidates[i * dim + d] = val;
    }
    if (cost_dim >= 0 && !isnan(fixed_cost)) {
        candidates[i * dim + cost_dim] = fixed_cost;
    }
    rng[i] = state;
}

typedef struct {
    float score_loss, cost_loss, predicted_score, predicted_cost, rating;
    int n_pareto, n_gp_obs, n_candidates, is_random;
} ProteinSweepInfo;

typedef struct {
    int lo, hi, left, right, split_dim;
    float split_val;
} KDNode;

typedef struct {
    // Caller populates config variables
    SweepSpace *space;
    int num_random_samples;
    int suggestions_per_pareto;
    int gp_training_iter;
    float gp_learning_rate;
    int optimizer_reset_frequency;
    int gp_max_obs;
    int infer_batch_size;
    int use_success_prob;
    int prune_pareto;
    int use_logit;
    float global_search_scale;
    float max_suggestion_cost;
    float expansion_rate;
    float cost_random_suggestion;
    float early_stop_quantile;
    int success_cap;
    int failure_cap;
    int top_k;
    unsigned long long rng_seed;
    // Internal
    float *succ_params, *succ_scores, *succ_costs;
    float *fail_params, *fail_scores, *fail_costs;
    int succ_n, fail_n, n_top, dim, cost_dim;
    int *top_idx;
    float min_score, max_score, log_c_min, log_c_max;
    int acq_cap;
    float *d_scales, *d_candidates, *d_pred_y, *d_pred_c;
    curandStatePhilox4_32_10_t *d_rng;
    float cm_A, cm_B, cm_max_score, cm_upper;
    int cm_min_samples, cm_fitted;
    float *clf_w, clf_bias;
    int clf_fitted;
    GaussianProcess gp_score, gp_cost;
    float *opt_m; // 6 * n_opt: score m,v,vmax, cost m,v,vmax
    int opt_t_score, opt_t_cost, n_opt;
    curandGenerator_t sobol;
    cudaStream_t stream;
    int suggestion_idx;
    float upper_cost_threshold;
    float ratio_pool[PROTEIN_NUM_COST_RATIOS];
    int pool_remaining;
    float *gp_train_params, *gp_train_y, *gp_train_c;
    int *pareto_buf, *pruned_buf;
    float *centers_buf;
    // suggest scratch (sized to caps; ext_buf/kept_buf also pinball + classifier)
    float *log_c_buf, *ext_buf, *h_cands, *h_pred;
    int *kept_buf, *kd_idx, *kd_keep;
    KDNode *kd_nodes;
    float *d_centers;
} ProteinSweep;

ProteinSweep *protein_sweep_create(ProteinSweep init) {
    ProteinSweep *sw = (ProteinSweep *)malloc(sizeof(ProteinSweep));
    *sw = init;
    int dim = sw->space->num;
    sw->dim = dim;
    sw->cost_dim = sw->space->cost_idx;
    sw->succ_n = sw->fail_n = sw->n_top = sw->suggestion_idx = 0;
    sw->cm_fitted = sw->clf_fitted = sw->pool_remaining = 0;
    sw->opt_t_score = sw->opt_t_cost = 0;
    sw->upper_cost_threshold = -FLT_MAX;
    sw->cm_min_samples = 30;
    sw->min_score = FLT_MAX;
    sw->max_score = -FLT_MAX;
    sw->log_c_min = FLT_MAX;
    sw->log_c_max = -FLT_MAX;
    sw->clf_bias = 0.0f;
    sw->cm_A = sw->cm_B = sw->cm_max_score = sw->cm_upper = 0.0f;

    static const float default_ratios[PROTEIN_NUM_COST_RATIOS] = {
        0.16f, 0.32f, 0.48f, 0.64f, 0.8f, 1.0f};
    memcpy(sw->ratio_pool, default_ratios, sizeof(default_ratios));

    sw->succ_params = (float *)calloc(sw->success_cap * (dim + 2), sizeof(float));
    sw->succ_scores = sw->succ_params + sw->success_cap * dim;
    sw->succ_costs = sw->succ_scores + sw->success_cap;
    sw->fail_params = (float *)calloc(sw->failure_cap * (dim + 2), sizeof(float));
    sw->fail_scores = sw->fail_params + sw->failure_cap * dim;
    sw->fail_costs = sw->fail_scores + sw->failure_cap;
    sw->top_idx = (int *)malloc(sw->top_k * sizeof(int));

    int max_centers = sw->success_cap + 3 * sw->top_k;
    int acq_cap = max_centers * sw->suggestions_per_pareto;
    if (acq_cap < sw->infer_batch_size) acq_cap = sw->infer_batch_size;
    if (acq_cap > PROTEIN_ACQ_MAX_CAP) acq_cap = PROTEIN_ACQ_MAX_CAP;
    sw->acq_cap = acq_cap;
    int max_ext = sw->success_cap + sw->failure_cap;
    int kd_cap = max_ext > acq_cap ? max_ext : acq_cap;
    float scales[dim];
    for (int i = 0; i < dim; i++) {
        scales[i] = sw->space->spaces[i].scale;
    }
    cudaMalloc(&sw->d_scales, dim * sizeof(float));
    cudaMemcpy(sw->d_scales, scales, dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&sw->d_candidates, acq_cap * dim * sizeof(float));
    cudaMalloc(&sw->d_pred_y, 2 * acq_cap * sizeof(float));
    sw->d_pred_c = sw->d_pred_y + acq_cap;
    cudaMalloc(&sw->d_rng, acq_cap * sizeof(curandStatePhilox4_32_10_t));
    cudaMalloc(&sw->d_centers, max_centers * dim * sizeof(float));
    protein_k_init_rng<<<(acq_cap + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        sw->d_rng, acq_cap, sw->rng_seed);

    sw->clf_w = (float *)calloc(dim, sizeof(float));
    gp_init(&sw->gp_score, dim, sw->gp_max_obs, acq_cap, 0.01f);
    gp_init(&sw->gp_cost, dim, sw->gp_max_obs, acq_cap, 0.01f);
    sw->n_opt = sw->gp_score.kernel->n_params + 2;
    sw->opt_m = (float *)calloc(6 * sw->n_opt, sizeof(float));

    curandCreateGeneratorHost(&sw->sobol, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
    curandSetQuasiRandomGeneratorDimensions(sw->sobol, dim);
    curandSetGeneratorOffset(sw->sobol, sw->rng_seed);

    sw->gp_train_params = (float *)malloc(sw->gp_max_obs * (dim + 2) * sizeof(float));
    sw->gp_train_y = sw->gp_train_params + sw->gp_max_obs * dim;
    sw->gp_train_c = sw->gp_train_y + sw->gp_max_obs;
    sw->pareto_buf = (int *)malloc(2 * sw->success_cap * sizeof(int));
    sw->pruned_buf = sw->pareto_buf + sw->success_cap;
    sw->centers_buf = (float *)malloc(max_centers * dim * sizeof(float));
    sw->log_c_buf = (float *)malloc(sw->success_cap * sizeof(float));
    sw->ext_buf = (float *)malloc(max_ext * (dim + 2) * sizeof(float));
    sw->kept_buf = (int *)malloc(kd_cap * sizeof(int));
    sw->h_cands = (float *)malloc(acq_cap * dim * sizeof(float));
    sw->h_pred = (float *)malloc(2 * acq_cap * sizeof(float));
    sw->kd_idx = (int *)malloc(kd_cap * sizeof(int));
    sw->kd_nodes = (KDNode *)malloc(2 * kd_cap * sizeof(KDNode));
    sw->kd_keep = (int *)malloc(kd_cap * sizeof(int));
    cudaStreamCreate(&sw->stream);
    return sw;
}

static float logit_transform(float value) {
    value = fmaxf(1e-9f, fminf(1.0f - 1e-9f, value));
    return fmaxf(-5.0f, logf(value / (1.0f - value)));
}

void protein_sweep_observe(ProteinSweep *sw, const float *norm_params,
        float score, float cost, int is_failure) {
    if (sw->use_logit) {
        score = logit_transform(score);
    }
    score *= sw->space->optimize_direction;
    int dim = sw->dim;
    if (is_failure || !isfinite(score)) {
        if (sw->fail_n >= sw->failure_cap) {
            return;
        }
        int i = sw->fail_n++;
        memcpy(&sw->fail_params[i * dim], norm_params, dim * sizeof(float));
        sw->fail_scores[i] = score;
        sw->fail_costs[i] = cost;
        return;
    }
    for (int i = 0; i < sw->succ_n; i++) {
        float dist2 = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = norm_params[d] - sw->succ_params[i * dim + d];
            dist2 += diff * diff;
        }
        if (dist2 < PROTEIN_EPSILON * PROTEIN_EPSILON) {
            memcpy(&sw->succ_params[i * dim], norm_params, dim * sizeof(float));
            sw->succ_scores[i] = score;
            sw->succ_costs[i] = cost;
            return;
        }
    }
    if (sw->cost_dim >= 0 && norm_params[sw->cost_dim] <= -1.0f) {
        return;
    }
    if (sw->succ_n >= sw->success_cap) {
        return;
    }
    int idx = sw->succ_n++;
    memcpy(&sw->succ_params[idx * dim], norm_params, dim * sizeof(float));
    sw->succ_scores[idx] = score;
    sw->succ_costs[idx] = cost;
    if (sw->n_top >= sw->top_k
            && score <= sw->succ_scores[sw->top_idx[sw->n_top - 1]]) {
        return;
    }
    if (sw->n_top < sw->top_k) {
        sw->top_idx[sw->n_top++] = idx;
    } else {
        sw->top_idx[sw->n_top - 1] = idx;
    }
    for (int j = sw->n_top - 1; j > 0; j--) {
        if (sw->succ_scores[sw->top_idx[j]] <= sw->succ_scores[sw->top_idx[j - 1]]) {
            break;
        }
        int tmp = sw->top_idx[j];
        sw->top_idx[j] = sw->top_idx[j - 1];
        sw->top_idx[j - 1] = tmp;
    }
}

int protein_sweep_should_stop(const ProteinSweep *sw, float score, float cost) {
    if (sw->use_logit) {
        score = logit_transform(score);
    }
    score *= sw->space->optimize_direction;
    if (!sw->cm_fitted) {
        return 0;
    }
    float min_allowed = sw->cm_upper * 0.3f + 10.0f;
    if (cost < min_allowed) {
        return 0;
    }
    float threshold;
    if (cost > PROTEIN_THRESHOLD_COST_CAP * sw->cm_upper) {
        float fraction = sw->cm_max_score < 0
            ? 2.0f - PROTEIN_THRESHOLD_FALLBACK : PROTEIN_THRESHOLD_FALLBACK;
        threshold = fraction * sw->cm_max_score;
    } else {
        threshold = sw->cm_A + sw->cm_B * logf(cost);
    }
    return score < threshold;
}

static float noise01(void) {
    float u1 = (rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = (rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    return 0.1f * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static void sobol_fallback(ProteinSweep *sw, float *out, int is_fixed_cost,
        float fixed_cost_norm) {
    int dim = sw->dim, cost_dim = sw->cost_dim;
    curandGenerateUniform(sw->sobol, out, dim);
    for (int d = 0; d < dim; d++) {
        out[d] = 2.0f * out[d] - 1.0f;
    }
    if (cost_dim < 0) {
        return;
    }
    if (is_fixed_cost) {
        out[cost_dim] = fixed_cost_norm;
    } else {
        out[cost_dim] = fmaxf(-1.0f,
            fminf(1.0f, sw->cost_random_suggestion + noise01()));
    }
}

// KD Tree near-duplicate filter
typedef struct {
    const float *X;
    int dim, n_nodes;
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
    if (hi - lo <= KD_LEAF) {
        return;
    }
    int best = 0;
    float spread = -1.0;
    for (int k = 0; k < t->dim; k++) {
        float mn = t->X[t->idx[lo] * t->dim + k], mx = mn;
        for (int i = lo + 1; i < hi; i++) {
            float v = t->X[t->idx[i] * t->dim + k];
            mn = fminf(mn, v);
            mx = fmaxf(mx, v);
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

static void kd_remove_near(const KDTree *t, int node, const float *q, float r,
        float r2, int self, int *keep) {
    const KDNode *nd = &t->nodes[node];
    if (nd->split_dim < 0) {
        for (int i = nd->lo; i < nd->hi; i++) {
            int p = t->idx[i];
            float sq = 0.0;
            for (int k = 0; k < t->dim; k++) {
                float df = q[k] - t->X[p * t->dim + k];
                sq += df * df;
            }
            if (p != self && sq <= r2) {
                keep[p] = 0;
            }
        }
        return;
    }
    float dv = q[nd->split_dim] - nd->split_val;
    if (dv <= r) {
        kd_remove_near(t, nd->left, q, r, r2, self, keep);
    }
    if (dv >= -r) {
        kd_remove_near(t, nd->right, q, r, r2, self, keep);
    }
}

static int filter_near_duplicates(ProteinSweep *sw, const float *X,
        int n, int dim, float threshold, int *kept_indices) {
    KDTree tree = {
        .X = X, .dim = dim, .n_nodes = 1, .idx = sw->kd_idx, .nodes = sw->kd_nodes};
    for (int i = 0; i < n; i++) {
        sw->kd_idx[i] = i;
    }
    kd_build(&tree, 0, 0, n);
    float r2 = threshold * threshold;
    for (int i = 0; i < n; i++) {
        sw->kd_keep[i] = 1;
    }
    for (int i = n - 1; i >= 0; i--) {
        if (sw->kd_keep[i]) {
            kd_remove_near(&tree, 0, &X[i * dim], threshold, r2, i, sw->kd_keep);
        }
    }
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (sw->kd_keep[i]) {
            kept_indices[count++] = i;
        }
    }
    return count;
}

static int float_cmp(const void *a, const void *b) {
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static float pinball(float a, float b, const float *x,
        const float *y, int n, float q) {
    float loss = 0.0f;
    for (int i = 0; i < n; i++) {
        float r = y[i] - (a + b * x[i]);
        loss += (r > 0.0f) ? q * r : (q - 1.0f) * r;
    }
    return loss;
}

static const float *g_pareto_costs;
static int pareto_cmp(const void *a, const void *b) {
    float ca = g_pareto_costs[*(const int *)a];
    float cb = g_pareto_costs[*(const int *)b];
    return (ca > cb) - (ca < cb);
}

ProteinSweepInfo protein_sweep_suggest(ProteinSweep *sw,
        float *out, float fixed_cost_norm) {
    ProteinSweepInfo info = {0};
    sw->suggestion_idx++;
    int dim = sw->dim, cost_dim = sw->cost_dim;
    int is_fixed_cost = !isnan(fixed_cost_norm);

    if (sw->suggestion_idx <= sw->num_random_samples || sw->succ_n == 0) {
        sobol_fallback(sw, out, is_fixed_cost, fixed_cost_norm);
        info.is_random = 1;
        return info;
    }

    // sample observations for GP
    sw->min_score = sw->max_score = sw->succ_scores[0];
    sw->log_c_min = logf(fmaxf(sw->succ_costs[0], PROTEIN_EPSILON));
    sw->log_c_buf[0] = sw->log_c_min;
    for (int i = 1; i < sw->succ_n; i++) {
        float s = sw->succ_scores[i];
        sw->min_score = fminf(sw->min_score, s);
        sw->max_score = fmaxf(sw->max_score, s);
        sw->log_c_buf[i] = logf(fmaxf(sw->succ_costs[i], PROTEIN_EPSILON));
        sw->log_c_min = fminf(sw->log_c_min, sw->log_c_buf[i]);
    }
    qsort(sw->log_c_buf, sw->succ_n, sizeof(float), float_cmp);
    float qidx = PROTEIN_COST_QUANTILE * (sw->succ_n - 1);
    int qlo = (int)qidx, qhi = qlo + 1 < sw->succ_n ? qlo + 1 : sw->succ_n - 1;
    float qfrac = qidx - qlo;
    sw->log_c_max = sw->log_c_buf[qlo] * (1.0f - qfrac) + sw->log_c_buf[qhi] * qfrac;

    int use_failures = (sw->succ_n < PROTEIN_MIN_OBS_NO_FAIL && sw->fail_n > 0);
    int combined_n = use_failures ? sw->fail_n + sw->succ_n : sw->succ_n;
    int ext_dim = dim + 2;
    int ci = 0;
    if (use_failures) {
        for (int i = 0; i < sw->fail_n; i++) {
            memcpy(&sw->ext_buf[ci * ext_dim], &sw->fail_params[i * dim],
                dim * sizeof(float));
            sw->ext_buf[ci * ext_dim + dim] = sw->min_score;
            sw->ext_buf[ci * ext_dim + dim + 1] = sw->fail_costs[i];
            ci++;
        }
    }
    for (int i = 0; i < sw->succ_n; i++) {
        memcpy(&sw->ext_buf[ci * ext_dim], &sw->succ_params[i * dim],
            dim * sizeof(float));
        sw->ext_buf[ci * ext_dim + dim] = sw->succ_scores[i];
        sw->ext_buf[ci * ext_dim + dim + 1] = sw->succ_costs[i];
        ci++;
    }
    int n_kept = filter_near_duplicates(
        sw, sw->ext_buf, combined_n, ext_dim, PROTEIN_EPSILON, sw->kept_buf);
    if (n_kept > sw->gp_max_obs) {
        int recent_size = (int)(0.5f * sw->gp_max_obs);
        int older_size = n_kept - recent_size;
        int num_sample = sw->gp_max_obs - recent_size;
        for (int i = 0; i < num_sample; i++) {
            int j = i + rand() % (older_size - i);
            int tmp = sw->kept_buf[i];
            sw->kept_buf[i] = sw->kept_buf[j];
            sw->kept_buf[j] = tmp;
        }
        memmove(sw->kept_buf + num_sample, sw->kept_buf + older_size,
            recent_size * sizeof(int));
        n_kept = sw->gp_max_obs;
    }
    for (int i = 0; i < n_kept; i++) {
        int idx = sw->kept_buf[i];
        memcpy(&sw->gp_train_params[i * dim], &sw->ext_buf[idx * ext_dim],
            dim * sizeof(float));
        sw->gp_train_y[i] = sw->ext_buf[idx * ext_dim + dim];
        sw->gp_train_c[i] = sw->ext_buf[idx * ext_dim + dim + 1];
    }
    int n_gp = n_kept;

    float min_score = sw->min_score, max_score = sw->max_score;
    float log_c_min = sw->log_c_min, log_c_max = sw->log_c_max;
    float score_range = fabsf(max_score - min_score) + PROTEIN_EPSILON;
    float cost_range = log_c_max - log_c_min + PROTEIN_EPSILON;
    for (int i = 0; i < n_gp; i++) {
        sw->gp_train_y[i] = (sw->gp_train_y[i] - min_score) / score_range;
        float lc = logf(fmaxf(sw->gp_train_c[i], PROTEIN_EPSILON));
        sw->gp_train_c[i] = (lc - log_c_min) / cost_range;
    }

    // Train GPs
    float *sm = sw->opt_m, *sv = sm + sw->n_opt, *svx = sv + sw->n_opt;
    float *cm = svx + sw->n_opt, *cv = cm + sw->n_opt, *cvx = cv + sw->n_opt;
    float score_loss = gp_train(
        &sw->gp_score, sm, sv, svx, &sw->opt_t_score, sw->gp_learning_rate,
        sw->gp_train_params, sw->gp_train_y, n_gp, sw->gp_training_iter, sw->stream);
    float cost_loss = gp_train(
        &sw->gp_cost, cm, cv, cvx, &sw->opt_t_cost, sw->gp_learning_rate,
        sw->gp_train_params, sw->gp_train_c, n_gp, sw->gp_training_iter, sw->stream);
    if (sw->optimizer_reset_frequency > 0
            && sw->suggestion_idx % sw->optimizer_reset_frequency == 0) {
        memset(sw->opt_m, 0, 6 * sw->n_opt * sizeof(float));
        sw->opt_t_score = sw->opt_t_cost = 0;
    }

    // Pareto front + prune
    for (int i = 0; i < sw->succ_n; i++) {
        sw->pruned_buf[i] = i;
    }
    g_pareto_costs = sw->succ_costs;
    qsort(sw->pruned_buf, sw->succ_n, sizeof(int), pareto_cmp);
    int n_pareto = 0;
    float max_ps = -FLT_MAX;
    for (int i = 0; i < sw->succ_n; i++) {
        int idx = sw->pruned_buf[i];
        if (sw->succ_scores[idx] > max_ps + PROTEIN_EPSILON) {
            sw->pareto_buf[n_pareto++] = idx;
            max_ps = sw->succ_scores[idx];
        }
    }
    memcpy(sw->pruned_buf, sw->pareto_buf, n_pareto * sizeof(int));
    int n_pruned = n_pareto;
    if (n_pruned >= 2) {
        float s_min = sw->succ_scores[sw->pruned_buf[0]], s_max = s_min;
        float c_min = sw->succ_costs[sw->pruned_buf[0]], c_max = c_min;
        for (int i = 1; i < n_pruned; i++) {
            float ss = sw->succ_scores[sw->pruned_buf[i]];
            float cc = sw->succ_costs[sw->pruned_buf[i]];
            s_min = fminf(s_min, ss);
            s_max = fmaxf(s_max, ss);
            c_min = fminf(c_min, cc);
            c_max = fmaxf(c_max, cc);
        }
        float s_range = fmaxf(s_max - s_min, PROTEIN_EPSILON);
        float c_range2 = fmaxf(c_max - c_min, PROTEIN_EPSILON);
        float max_pareto_s = sw->succ_scores[sw->pruned_buf[n_pruned - 1]];
        float near_best = (max_pareto_s < 0 ? 1.02f : 0.98f) * max_pareto_s;
        for (int i = n_pruned - 1; i > 1; i--) {
            if (sw->succ_scores[sw->pruned_buf[i - 1]] < near_best) {
                break;
            }
            float ng = (sw->succ_scores[sw->pruned_buf[i]]
                - sw->succ_scores[sw->pruned_buf[i - 1]]) / s_range;
            float nc = (sw->succ_costs[sw->pruned_buf[i]]
                - sw->succ_costs[sw->pruned_buf[i - 1]]) / c_range2;
            if (ng / (nc + PROTEIN_EPSILON) >= 0.5f) {
                break;
            }
            n_pruned--;
        }
    }

    if (n_pruned > 0) {
        float pruned_max_cost = sw->succ_costs[sw->pruned_buf[n_pruned - 1]];
        if (sw->upper_cost_threshold < 0) {
            sw->upper_cost_threshold = pruned_max_cost;
        } else if (pruned_max_cost > sw->upper_cost_threshold) {
            sw->upper_cost_threshold *= PROTEIN_COST_GROWTH;
        }
    }

    // Cost model fit (pinball NM)
    sw->cm_fitted = 0;
    sw->cm_max_score = max_score;
    sw->cm_upper = sw->upper_cost_threshold;
    int n_valid = 0;
    for (int i = 0; i < sw->succ_n; i++) {
        if (sw->succ_costs[i] > PROTEIN_EPSILON && isfinite(sw->succ_scores[i])) {
            n_valid++;
        }
    }
    if (n_valid >= sw->cm_min_samples) {
        float *x = sw->ext_buf;
        float *y = x + n_valid;
        int j = 0;
        for (int i = 0; i < sw->succ_n; i++) {
            if (sw->succ_costs[i] > PROTEIN_EPSILON && isfinite(sw->succ_scores[i])) {
                x[j] = logf(sw->succ_costs[i]);
                y[j] = sw->succ_scores[i];
                j++;
            }
        }
        float sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = 0; i < n_valid; i++) {
            sx += x[i];
            sy += y[i];
            sxx += x[i] * x[i];
            sxy += x[i] * y[i];
        }
        float det = n_valid * sxx - sx * sx;
        float a_init = sy / fmaxf(n_valid, 1.0f), b_init = 0.0f;
        if (fabsf(det) >= 1e-10f) {
            b_init = (n_valid * sxy - sx * sy) / det;
            a_init = (sy - b_init * sx) / n_valid;
        }
        float q = sw->early_stop_quantile;
        float nm_x[3] = {a_init, a_init + 0.05f, a_init - 0.05f};
        float nm_y[3] = {b_init, b_init - 0.05f, b_init + 0.05f};
        float nm_v[3];
        for (int i = 0; i < 3; i++) {
            nm_v[i] = pinball(nm_x[i], nm_y[i], x, y, n_valid, q);
        }
        for (int iter = 0; iter < 500; iter++) {
            for (int i = 0; i < 2; i++) {
                for (int jj = i + 1; jj < 3; jj++) {
                    if (nm_v[jj] < nm_v[i]) {
                        float t;
                        t = nm_x[i]; nm_x[i] = nm_x[jj]; nm_x[jj] = t;
                        t = nm_y[i]; nm_y[i] = nm_y[jj]; nm_y[jj] = t;
                        t = nm_v[i]; nm_v[i] = nm_v[jj]; nm_v[jj] = t;
                    }
                }
            }
            if (fabsf(nm_v[2] - nm_v[0]) < 1e-7f) {
                break;
            }
            float cx = (nm_x[0] + nm_x[1]) * 0.5f;
            float cy = (nm_y[0] + nm_y[1]) * 0.5f;
            float rx = cx + (cx - nm_x[2]), ry = cy + (cy - nm_y[2]);
            float rv = pinball(rx, ry, x, y, n_valid, q);
            if (rv < nm_v[1] && rv >= nm_v[0]) {
                nm_x[2] = rx;
                nm_y[2] = ry;
                nm_v[2] = rv;
                continue;
            }
            if (rv < nm_v[0]) {
                float ex = cx + 2.0f * (rx - cx);
                float ey = cy + 2.0f * (ry - cy);
                float ev = pinball(ex, ey, x, y, n_valid, q);
                if (ev >= rv) {
                    ex = rx;
                    ey = ry;
                    ev = rv;
                }
                nm_x[2] = ex;
                nm_y[2] = ey;
                nm_v[2] = ev;
                continue;
            }
            float ccx = (rv < nm_v[2])
                ? cx + 0.5f * (rx - cx) : cx + 0.5f * (nm_x[2] - cx);
            float ccy = (rv < nm_v[2])
                ? cy + 0.5f * (ry - cy) : cy + 0.5f * (nm_y[2] - cy);
            float ccv = pinball(ccx, ccy, x, y, n_valid, q);
            if (ccv < nm_v[2]) {
                nm_x[2] = ccx;
                nm_y[2] = ccy;
                nm_v[2] = ccv;
                continue;
            }
            for (int i = 1; i < 3; i++) {
                nm_x[i] = nm_x[0] + 0.5f * (nm_x[i] - nm_x[0]);
                nm_y[i] = nm_y[0] + 0.5f * (nm_y[i] - nm_y[0]);
                nm_v[i] = pinball(nm_x[i], nm_y[i], x, y, n_valid, q);
            }
        }
        sw->cm_A = nm_x[0];
        sw->cm_B = nm_y[0];
        sw->cm_fitted = 1;
    }

    int *use_pareto = sw->prune_pareto ? sw->pruned_buf : sw->pareto_buf;
    int n_use = sw->prune_pareto ? n_pruned : n_pareto;

    // Search centers
    int n_centers = 0;
    for (int i = 0; i < n_use; i++) {
        memcpy(&sw->centers_buf[n_centers++ * dim],
            &sw->succ_params[use_pareto[i] * dim], dim * sizeof(float));
    }
    float divisors[] = {2.0f, 3.0f};
    int n_shift = (cost_dim >= 0) ? 2 : 1;
    for (int r = 0; r < n_shift; r++) {
        for (int i = 0; i < sw->n_top; i++) {
            const float *src = &sw->succ_params[sw->top_idx[i] * dim];
            float *dst = &sw->centers_buf[n_centers * dim];
            memcpy(dst, src, dim * sizeof(float));
            if (cost_dim >= 0) {
                float orig = src[cost_dim];
                dst[cost_dim] = orig - (orig + 1.0f) / divisors[r];
            }
            n_centers++;
        }
    }

    // Target cost
    float target_cost = 0.0f;
    if (!is_fixed_cost) {
        if (sw->pool_remaining <= 0) {
            for (int i = PROTEIN_NUM_COST_RATIOS - 1; i > 0; i--) {
                int j = rand() % (i + 1);
                float tmp = sw->ratio_pool[i];
                sw->ratio_pool[i] = sw->ratio_pool[j];
                sw->ratio_pool[j] = tmp;
            }
            sw->pool_remaining = PROTEIN_NUM_COST_RATIOS;
        }
        float ratio = sw->ratio_pool[--sw->pool_remaining];
        ratio = fmaxf(0.0f, fminf(1.0f, ratio + noise01()));
        target_cost = (1.0f + sw->expansion_rate) * ratio;
    }

    // Sample candidates
    int n_total = n_centers * sw->suggestions_per_pareto;
    if (n_total > sw->acq_cap) {
        n_total = sw->acq_cap;
    }
    cudaMemcpyAsync(sw->d_centers, sw->centers_buf, n_centers * dim * sizeof(float),
        cudaMemcpyHostToDevice, sw->stream);
    protein_k_sample<<<(n_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, sw->stream>>>(
        sw->d_candidates, sw->d_centers, sw->d_scales, sw->d_rng, n_total,
        n_centers, dim, sw->global_search_scale,
        cost_dim, is_fixed_cost ? fixed_cost_norm : NAN);

    cudaMemcpyAsync(sw->h_cands, sw->d_candidates, n_total * dim * sizeof(float),
        cudaMemcpyDeviceToHost, sw->stream);
    cudaStreamSynchronize(sw->stream);
    int n_cands = filter_near_duplicates(
        sw, sw->h_cands, n_total, dim, PROTEIN_EPSILON, sw->kept_buf);
    if (n_cands < n_total) {
        for (int i = 0; i < n_cands; i++) {
            memmove(&sw->h_cands[i * dim], &sw->h_cands[sw->kept_buf[i] * dim],
                dim * sizeof(float));
        }
        cudaMemcpyAsync(sw->d_candidates, sw->h_cands, n_cands * dim * sizeof(float),
            cudaMemcpyHostToDevice, sw->stream);
    }

    if (n_cands == 0) {
        sobol_fallback(sw, out, is_fixed_cost, fixed_cost_norm);
        info.is_random = 1;
        return info;
    }

    // Success classifier
    sw->clf_fitted = 0;
    if (sw->use_success_prob && sw->succ_n > 9 && sw->fail_n > 9) {
        int n_s = sw->succ_n, n_f = sw->fail_n, n_clf = n_s + n_f;
        float *X = sw->ext_buf;
        int *yy = sw->kept_buf;
        memcpy(X, sw->succ_params, n_s * dim * sizeof(float));
        memcpy(X + n_s * dim, sw->fail_params, n_f * dim * sizeof(float));
        for (int i = 0; i < n_s; i++) {
            yy[i] = 1;
        }
        for (int i = 0; i < n_f; i++) {
            yy[n_s + i] = 0;
        }
        int n_pos = 0;
        for (int i = 0; i < n_clf; i++) {
            n_pos += yy[i];
        }
        int n_neg = n_clf - n_pos;
        if (n_pos > 0 && n_neg > 0) {
            float w_pos = n_clf / (2.0f * n_pos);
            float w_neg = n_clf / (2.0f * n_neg);
            memset(sw->clf_w, 0, dim * sizeof(float));
            sw->clf_bias = 0.0f;
            float grad[dim];
            for (int iter = 0; iter < PROTEIN_CLF_ITERS; iter++) {
                float lr = 0.1f / (1.0f + 0.01f * iter);
                for (int d = 0; d < dim; d++) {
                    grad[d] = 0.0f;
                }
                float grad_b = 0.0f;
                for (int i = 0; i < n_clf; i++) {
                    float z = sw->clf_bias;
                    for (int d = 0; d < dim; d++) {
                        z += X[i * dim + d] * sw->clf_w[d];
                    }
                    float sig = 1.0f / (1.0f + expf(-z));
                    float err = (yy[i] ? w_pos : w_neg) * (sig - yy[i]) / n_clf;
                    for (int d = 0; d < dim; d++) {
                        grad[d] += err * X[i * dim + d];
                    }
                    grad_b += err;
                }
                for (int d = 0; d < dim; d++) {
                    sw->clf_w[d] -= lr * (grad[d] + sw->clf_w[d]);
                }
                sw->clf_bias -= lr * grad_b;
            }
            sw->clf_fitted = 1;
        }
    }

    // GP predict + score/argmax
    gp_predict(&sw->gp_score, sw->d_candidates, sw->d_pred_y, n_cands, sw->stream);
    gp_predict(&sw->gp_cost, sw->d_candidates, sw->d_pred_c, n_cands, sw->stream);
    cudaMemcpyAsync(sw->h_pred, sw->d_pred_y, n_cands * sizeof(float),
        cudaMemcpyDeviceToHost, sw->stream);
    cudaMemcpyAsync(sw->h_pred + n_cands, sw->d_pred_c, n_cands * sizeof(float),
        cudaMemcpyDeviceToHost, sw->stream);
    cudaStreamSynchronize(sw->stream);

    int best = 0;
    float best_s = -FLT_MAX;
    for (int i = 0; i < n_cands; i++) {
        float s = sw->h_pred[i];
        if (!is_fixed_cost) {
            float cn = sw->h_pred[n_cands + i];
            float c = expf(cn * (log_c_max - log_c_min) + log_c_min);
            s *= (c < sw->max_suggestion_cost ? 1.0f : 0.0f)
                * (1.0f - fabsf(target_cost - cn));
        }
        if (sw->clf_fitted) {
            float z = sw->clf_bias;
            for (int d = 0; d < dim; d++) {
                z += sw->h_cands[i * dim + d] * sw->clf_w[d];
            }
            s *= 1.0f / (1.0f + expf(-z));
        }
        if (s > best_s) {
            best_s = s;
            best = i;
        }
    }
    memcpy(out, &sw->h_cands[best * dim], dim * sizeof(float));
    float y_norm = sw->h_pred[best], c_norm = sw->h_pred[n_cands + best];

    info.score_loss = score_loss;
    info.cost_loss = cost_loss;
    info.predicted_score = sw->space->optimize_direction
        * (y_norm * (max_score - min_score) + min_score);
    info.predicted_cost = expf(c_norm * (log_c_max - log_c_min) + log_c_min);
    info.rating = best_s;
    info.n_pareto = n_use;
    info.n_gp_obs = n_gp;
    info.n_candidates = n_cands;
    return info;
}
