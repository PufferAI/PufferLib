// protein.cu -- Protein hyperparameter optimizer (CUDA).

#ifndef PROTEIN_CU
#define PROTEIN_CU

#include <curand.h>
#include <curand_kernel.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PROTEIN_EPSILON 1e-6f
#define PROTEIN_NUM_COST_RATIOS 6
#define PROTEIN_ACQ_MAX_CAP 65536
#define PROTEIN_COST_QUANTILE 0.97f
#define PROTEIN_MIN_OBS_NO_FAIL 100
#define PROTEIN_COST_GROWTH 1.01f
#define PROTEIN_CLF_ITERS 100
#define PROTEIN_THRESHOLD_COST_CAP 1.2f
#define PROTEIN_THRESHOLD_FALLBACK 0.9f
#define PROTEIN_RUNNING_BUF_CAP 30

// clang-format off
#include "protein_util.h"
#include "gp_cuda.cu"
// clang-format on

// clang-format off
static inline void _curand_check(curandStatus_t s, const char* f, int l) {
    if (s != CURAND_STATUS_SUCCESS) { fprintf(stderr, "cuRAND error %s:%d: %d\n", f, l, (int)s); abort(); }
}
#define CURAND_CHECK(call) _curand_check((call), __FILE__, __LINE__)
// clang-format on

typedef struct {
    int dim;
    int capacity;
    float *d_bounds_min;
    float *d_bounds_max;
    float *d_scales;
    float *d_candidates;
    float *d_pred_y;
    float *d_pred_c;
    float *d_scores;
    float *d_success_prob;
    int *d_best_idx;
    curandStatePhilox4_32_10_t *d_rng;
} ProteinAcq;

typedef struct {
    int best_idx;
    float predicted_score;
    float predicted_cost;
    float rating;
} ProteinAcqResult;

typedef struct {
    float *weights;
    float *d_weights;
    float bias;
    int dim;
    int is_fitted;
} ProteinClassifier;

typedef struct {
    float *params;
    float *scores;
    float *costs;
    int n, cap;
} ProteinObsList;

typedef struct {
    ProteinObsList success;
    ProteinObsList failure;
    int *top_idx;
    int n_top, top_k;
    int dim, cost_dim;
    float min_score, max_score;
    float log_c_min, log_c_max;
} ProteinObs;

typedef struct {
    curandGenerator_t gen;
    int dim;
} Sobol;

__global__ void protein_k_init_rng(curandStatePhilox4_32_10_t *states, int n, unsigned long long seed) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n)
        curand_init(seed, i, 0, &states[i]);
}

__global__ void protein_k_sample(float *__restrict__ candidates, const float *__restrict__ centers,
                                 const float *__restrict__ bounds_min, const float *__restrict__ bounds_max,
                                 const float *__restrict__ scales, curandStatePhilox4_32_10_t *__restrict__ rng,
                                 int n_candidates, int n_centers, int dim, float global_scale, int cost_dim,
                                 float fixed_cost) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= n_candidates)
        return;

    curandStatePhilox4_32_10_t state = rng[i];
    int center = ((int)(curand_uniform(&state) * n_centers)) % n_centers;

    for (int d = 0; d < dim; d++) {
        float s = scales[d] * global_scale;
        float val = s * (2.0f * curand_uniform(&state) - 1.0f) + centers[center * dim + d];
        val = fmaxf(bounds_min[d], fminf(bounds_max[d], val));
        candidates[i * dim + d] = val;
    }

    if (cost_dim >= 0 && !isnan(fixed_cost))
        candidates[i * dim + cost_dim] = fixed_cost;

    rng[i] = state;
}

__global__ void protein_k_score(const float *__restrict__ pred_y_norm, const float *__restrict__ pred_c_norm,
                                const float *__restrict__ success_prob, float *__restrict__ scores, int m,
                                float min_score, float max_score, float log_c_min, float log_c_max, float max_cost,
                                float target_cost, int optimize_dir, int fixed_cost) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= m)
        return;

    float y_norm = pred_y_norm[i];
    float c_norm = pred_c_norm[i];
    float s = (float)optimize_dir * y_norm;

    if (!fixed_cost) {
        float log_c = c_norm * (log_c_max - log_c_min) + log_c_min;
        float c = expf(log_c);
        float mask = (c < max_cost) ? 1.0f : 0.0f;
        float w = 1.0f - fabsf(target_cost - c_norm);
        s *= mask * w;
    }

    if (success_prob)
        s *= success_prob[i];

    scores[i] = s;
}

__global__ void protein_k_argmax(const float *__restrict__ scores, int n, int *out_idx) {
    __shared__ float s_val[BLOCK_SIZE];
    __shared__ int s_idx[BLOCK_SIZE];

    int tid = threadIdx.x;
    float best = -FLT_MAX;
    int best_i = 0;

    for (int i = tid; i < n; i += BLOCK_SIZE) {
        if (scores[i] > best) {
            best = scores[i];
            best_i = i;
        }
    }
    s_val[tid] = best;
    s_idx[tid] = best_i;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s && s_val[tid + s] > s_val[tid]) {
            s_val[tid] = s_val[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        *out_idx = s_idx[0];
}

__global__ void protein_k_classifier_predict(const float *__restrict__ X, const float *__restrict__ weights, float bias,
                                             float *__restrict__ probs, int m, int dim) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= m)
        return;
    float z = bias;
    for (int d = 0; d < dim; d++)
        z += X[i * dim + d] * weights[d];
    probs[i] = 1.0f / (1.0f + expf(-z));
}

ProteinClassifier *protein_classifier_create(int dim) {
    ProteinClassifier *clf = (ProteinClassifier *)calloc(1, sizeof(ProteinClassifier));
    clf->dim = dim;
    clf->weights = (float *)calloc((size_t)dim, sizeof(float));
    CUDA_CHECK(cudaMalloc(&clf->d_weights, (size_t)dim * sizeof(float)));
    return clf;
}

void protein_classifier_destroy(ProteinClassifier *clf) {
    if (!clf)
        return;
    free(clf->weights);
    cudaFree(clf->d_weights);
    free(clf);
}

void protein_classifier_fit(ProteinClassifier *clf, const float *X, const int *y, int n, float C_reg, int max_iter) {
    int dim = clf->dim;
    clf->is_fitted = 0;

    int n_pos = 0;
    for (int i = 0; i < n; i++)
        n_pos += y[i];
    int n_neg = n - n_pos;
    if (n_pos == 0 || n_neg == 0)
        return;

    float w_pos = (float)n / (2.0f * (float)n_pos);
    float w_neg = (float)n / (2.0f * (float)n_neg);
    float lambda = 1.0f / C_reg;

    memset(clf->weights, 0, (size_t)dim * sizeof(float));
    clf->bias = 0.0f;
    float *grad = (float *)calloc((size_t)dim, sizeof(float));

    for (int iter = 0; iter < max_iter; iter++) {
        float lr = 0.1f / (1.0f + 0.01f * (float)iter);
        memset(grad, 0, (size_t)dim * sizeof(float));
        float grad_b = 0.0f;

        for (int i = 0; i < n; i++) {
            float z = clf->bias;
            for (int d = 0; d < dim; d++)
                z += X[(size_t)i * dim + d] * clf->weights[d];
            float sig = 1.0f / (1.0f + expf(-z));
            float sw = y[i] ? w_pos : w_neg;
            float err = sw * (sig - (float)y[i]) / (float)n;
            for (int d = 0; d < dim; d++)
                grad[d] += err * X[(size_t)i * dim + d];
            grad_b += err;
        }

        for (int d = 0; d < dim; d++)
            grad[d] += lambda * clf->weights[d];

        for (int d = 0; d < dim; d++)
            clf->weights[d] -= lr * grad[d];
        clf->bias -= lr * grad_b;
    }

    free(grad);
    clf->is_fitted = 1;

    CUDA_CHECK(cudaMemcpy(clf->d_weights, clf->weights, (size_t)dim * sizeof(float), cudaMemcpyHostToDevice));
}

void protein_classifier_predict_d(const ProteinClassifier *clf, const float *d_X, float *d_probs, int m,
                                  cudaStream_t stream) {
    protein_k_classifier_predict<<<(m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        d_X, clf->d_weights, clf->bias, d_probs, m, clf->dim);
    CUDA_CHECK(cudaGetLastError());
}

ProteinAcq *protein_acq_create(int dim, int capacity, const float *bounds_min, const float *bounds_max,
                               const float *scales, unsigned long long rng_seed) {
    ProteinAcq *acq = (ProteinAcq *)calloc(1, sizeof(ProteinAcq));
    acq->dim = dim;
    acq->capacity = capacity;

    CUDA_CHECK(cudaMalloc(&acq->d_bounds_min, (size_t)dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&acq->d_bounds_max, (size_t)dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&acq->d_scales, (size_t)dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(acq->d_bounds_min, bounds_min, (size_t)dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(acq->d_bounds_max, bounds_max, (size_t)dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(acq->d_scales, scales, (size_t)dim * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&acq->d_candidates, (size_t)capacity * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&acq->d_pred_y, (size_t)capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&acq->d_pred_c, (size_t)capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&acq->d_scores, (size_t)capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&acq->d_success_prob, (size_t)capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&acq->d_best_idx, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&acq->d_rng, (size_t)capacity * sizeof(curandStatePhilox4_32_10_t)));

    protein_k_init_rng<<<(capacity + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(acq->d_rng, capacity, rng_seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return acq;
}

void protein_acq_destroy(ProteinAcq *acq) {
    if (!acq)
        return;
    void *ptrs[] = {acq->d_bounds_min, acq->d_bounds_max, acq->d_scales,       acq->d_candidates, acq->d_pred_y,
                    acq->d_pred_c,     acq->d_scores,     acq->d_success_prob, acq->d_best_idx,   acq->d_rng};
    for (int i = 0; i < 10; i++)
        cudaFree(ptrs[i]);
    free(acq);
}

int protein_acq_sample(ProteinAcq *acq, const float *centers, int n_centers, int n_total, float global_scale,
                       int cost_dim, float fixed_cost_norm, float dedup_threshold, cudaStream_t stream) {
    int dim = acq->dim;
    if (n_total > acq->capacity)
        n_total = acq->capacity;

    float *d_centers;
    CUDA_CHECK(cudaMallocAsync(&d_centers, (size_t)n_centers * dim * sizeof(float), stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_centers, centers, (size_t)n_centers * dim * sizeof(float), cudaMemcpyHostToDevice, stream));

    protein_k_sample<<<(n_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        acq->d_candidates, d_centers, acq->d_bounds_min, acq->d_bounds_max, acq->d_scales, acq->d_rng, n_total,
        n_centers, dim, global_scale, cost_dim, fixed_cost_norm);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(d_centers, stream));

    if (dedup_threshold <= 0.0f) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return n_total;
    }

    float *h_cands = (float *)malloc((size_t)n_total * dim * sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(h_cands, acq->d_candidates, (size_t)n_total * dim * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int *kept = (int *)malloc((size_t)n_total * sizeof(int));
    int n_kept = gp_filter_near_duplicates(h_cands, n_total, dim, dedup_threshold, kept);

    if (n_kept > 0 && n_kept < n_total) {
        float *h_compact = (float *)malloc((size_t)n_kept * dim * sizeof(float));
        for (int i = 0; i < n_kept; i++)
            memcpy(&h_compact[(size_t)i * dim], &h_cands[(size_t)kept[i] * dim], (size_t)dim * sizeof(float));
        CUDA_CHECK(cudaMemcpyAsync(acq->d_candidates, h_compact, (size_t)n_kept * dim * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        free(h_compact);
    }

    free(h_cands);
    free(kept);
    return n_kept;
}

ProteinAcqResult protein_acq_suggest(ProteinAcq *acq, int m, const GaussianProcess *gp_score,
                                     const GaussianProcess *gp_cost, float min_score, float max_score, float log_c_min,
                                     float log_c_max, float max_suggestion_cost, float target_cost_ratio,
                                     int optimize_direction, int fixed_cost, const float *d_success_prob,
                                     cudaStream_t stream) {
    gp_predict_d(gp_score, acq->d_candidates, acq->d_pred_y, NULL, m, stream);
    gp_predict_d(gp_cost, acq->d_candidates, acq->d_pred_c, NULL, m, stream);

    protein_k_score<<<(m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        acq->d_pred_y, acq->d_pred_c, d_success_prob, acq->d_scores, m, min_score, max_score, log_c_min, log_c_max,
        max_suggestion_cost, target_cost_ratio, optimize_direction, fixed_cost);
    CUDA_CHECK(cudaGetLastError());

    protein_k_argmax<<<1, BLOCK_SIZE, 0, stream>>>(acq->d_scores, m, acq->d_best_idx);
    CUDA_CHECK(cudaGetLastError());

    int best;
    CUDA_CHECK(cudaMemcpyAsync(&best, acq->d_best_idx, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float y_norm, c_norm, rating;
    CUDA_CHECK(cudaMemcpy(&y_norm, &acq->d_pred_y[best], sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&c_norm, &acq->d_pred_c[best], sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&rating, &acq->d_scores[best], sizeof(float), cudaMemcpyDeviceToHost));

    return (ProteinAcqResult){
        .best_idx = best,
        .predicted_score = y_norm * (max_score - min_score) + min_score,
        .predicted_cost = expf(c_norm * (log_c_max - log_c_min) + log_c_min),
        .rating = rating,
    };
}

#define NOISE_PRIOR_MU (-4.60517f)
#define NOISE_PRIOR_SIGMA 0.5f

typedef struct {
    float *m, *v, *v_max;
    float lr, beta1, beta2, eps;
    int t, n;
} Adam;

Adam *adam_create(int n_kernel_params, float lr) {
    int n = n_kernel_params + 1;
    Adam *opt = (Adam *)calloc(1, sizeof(Adam));
    opt->n = n;
    opt->lr = lr;
    opt->beta1 = 0.9f;
    opt->beta2 = 0.999f;
    opt->eps = 1e-8f;
    opt->m = (float *)calloc((size_t)n, sizeof(float));
    opt->v = (float *)calloc((size_t)n, sizeof(float));
    opt->v_max = (float *)calloc((size_t)n, sizeof(float));
    return opt;
}

void adam_destroy(Adam *opt) {
    if (!opt)
        return;
    free(opt->m);
    free(opt->v);
    free(opt->v_max);
    free(opt);
}

void adam_reset(Adam *opt) {
    memset(opt->m, 0, (size_t)opt->n * sizeof(float));
    memset(opt->v, 0, (size_t)opt->n * sizeof(float));
    memset(opt->v_max, 0, (size_t)opt->n * sizeof(float));
    opt->t = 0;
}

static float noise_log_prior_grad(float raw_noise) {
    float sig = softplus_grad(raw_noise);
    float noise = softplus(raw_noise);
    float log_noise = logf(noise);
    float d_log_p = (-(log_noise - NOISE_PRIOR_MU) / (NOISE_PRIOR_SIGMA * NOISE_PRIOR_SIGMA) - 1.0f) * sig / noise;
    float d_log_jac = 1.0f - sig;
    return d_log_p + d_log_jac;
}

float protein_train_gp(GaussianProcess *gp, Adam *opt, const float *X, const float *y, int n_data, int training_iter,
                       cudaStream_t stream) {
    int rc = gp_fit(gp, X, y, n_data, stream);
    if (rc != 0)
        return 0.0f;

    int n_kp = gp->kernel->n_params;
    float *kg = (float *)malloc((size_t)n_kp * sizeof(float));
    float loss = 0.0f;

    for (int iter = 0; iter < training_iter; iter++) {
        rc = gp_recompute(gp, stream);
        if (rc != 0)
            break;

        float mll = gp_marginal_log_likelihood(gp);
        float d_noise;
        gp_mll_grad(gp, &d_noise, kg, stream);
        cudaStreamSynchronize(stream);

        float np_grad = noise_log_prior_grad(gp->raw_noise);

        float noise_val = softplus(gp->raw_noise);
        float ln_noise = logf(noise_val);
        float lp =
            -0.5f * powf((ln_noise - NOISE_PRIOR_MU) / NOISE_PRIOR_SIGMA, 2.0f) - ln_noise - logf(NOISE_PRIOR_SIGMA);
        float lj = -log1pf(expf(-gp->raw_noise));
        loss = -mll - (lp + lj);

        opt->t++;
        float bc1 = 1.0f - powf(opt->beta1, (float)opt->t);
        float bc2 = 1.0f - powf(opt->beta2, (float)opt->t);

        for (int i = 0; i < opt->n; i++) {
            float g = (i < n_kp) ? -kg[i] : -d_noise - np_grad;
            opt->m[i] = opt->beta1 * opt->m[i] + (1.0f - opt->beta1) * g;
            opt->v[i] = opt->beta2 * opt->v[i] + (1.0f - opt->beta2) * g * g;
            float m_hat = opt->m[i] / bc1;
            float v_hat = opt->v[i] / bc2;
            if (v_hat > opt->v_max[i])
                opt->v_max[i] = v_hat;
            float step = opt->lr * m_hat / (sqrtf(opt->v_max[i]) + opt->eps);
            if (i < n_kp)
                gp->kernel->raw_params[i] -= step;
            else
                gp->raw_noise -= step;
        }
    }

    gp_recompute(gp, stream);
    free(kg);
    return loss;
}

Sobol *sobol_create(int dim, unsigned long long seed) {
    Sobol *sob = (Sobol *)calloc(1, sizeof(Sobol));
    sob->dim = dim;
    CURAND_CHECK(curandCreateGeneratorHost(&sob->gen, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32));
    CURAND_CHECK(curandSetQuasiRandomGeneratorDimensions(sob->gen, dim));
    CURAND_CHECK(curandSetGeneratorOffset(sob->gen, seed));
    return sob;
}

void sobol_destroy(Sobol *sob) {
    if (!sob)
        return;
    curandDestroyGenerator(sob->gen);
    free(sob);
}

ProteinObs *protein_obs_create(int dim, int success_cap, int failure_cap, int top_k, int cost_dim) {
    ProteinObs *obs = (ProteinObs *)calloc(1, sizeof(ProteinObs));
    obs->dim = dim;
    obs->cost_dim = cost_dim;
    obs->top_k = top_k;
    obs->top_idx = (int *)malloc((size_t)top_k * sizeof(int));
    obs->min_score = FLT_MAX;
    obs->max_score = -FLT_MAX;
    obs->log_c_min = FLT_MAX;
    obs->log_c_max = -FLT_MAX;
    obs->success.params = (float *)calloc((size_t)success_cap * dim, sizeof(float));
    obs->success.scores = (float *)calloc((size_t)success_cap, sizeof(float));
    obs->success.costs = (float *)calloc((size_t)success_cap, sizeof(float));
    obs->success.cap = success_cap;
    obs->failure.params = (float *)calloc((size_t)failure_cap * dim, sizeof(float));
    obs->failure.scores = (float *)calloc((size_t)failure_cap, sizeof(float));
    obs->failure.costs = (float *)calloc((size_t)failure_cap, sizeof(float));
    obs->failure.cap = failure_cap;
    return obs;
}

void protein_obs_destroy(ProteinObs *obs) {
    if (!obs)
        return;
    free(obs->success.params);
    free(obs->success.scores);
    free(obs->success.costs);
    free(obs->failure.params);
    free(obs->failure.scores);
    free(obs->failure.costs);
    free(obs->top_idx);
    free(obs);
}

void protein_obs_add(ProteinObs *obs, const float *params, float score, float cost, int is_failure) {
    int dim = obs->dim;

    if (is_failure || !isfinite(score) || isnan(score)) {
        ProteinObsList *f = &obs->failure;
        if (f->n < f->cap) {
            int i = f->n++;
            memcpy(&f->params[(size_t)i * dim], params, (size_t)dim * sizeof(float));
            f->scores[i] = score;
            f->costs[i] = cost;
        }
        return;
    }

    ProteinObsList *s = &obs->success;

    for (int i = 0; i < s->n; i++) {
        float dist2 = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = params[d] - s->params[(size_t)i * dim + d];
            dist2 += diff * diff;
        }
        if (sqrtf(dist2) < PROTEIN_EPSILON) {
            memcpy(&s->params[(size_t)i * dim], params, (size_t)dim * sizeof(float));
            s->scores[i] = score;
            s->costs[i] = cost;
            return;
        }
    }

    if (obs->cost_dim >= 0 && params[obs->cost_dim] <= -1.0f)
        return;

    if (s->n >= s->cap)
        return;

    int idx = s->n++;
    memcpy(&s->params[(size_t)idx * dim], params, (size_t)dim * sizeof(float));
    s->scores[idx] = score;
    s->costs[idx] = cost;

    if (obs->n_top < obs->top_k)
        obs->top_idx[obs->n_top++] = idx;
    else if (score > s->scores[obs->top_idx[obs->n_top - 1]])
        obs->top_idx[obs->n_top - 1] = idx;
    else
        return;

    for (int j = obs->n_top - 1; j > 0; j--) {
        if (s->scores[obs->top_idx[j]] > s->scores[obs->top_idx[j - 1]]) {
            int tmp = obs->top_idx[j];
            obs->top_idx[j] = obs->top_idx[j - 1];
            obs->top_idx[j - 1] = tmp;
        } else
            break;
    }
}

static void protein_obs_extract(const float *ext, int ext_dim, int dim, int idx, float *out_params, float *out_score,
                                float *out_cost) {
    memcpy(out_params, &ext[(size_t)idx * ext_dim], (size_t)dim * sizeof(float));
    *out_score = ext[(size_t)idx * ext_dim + dim];
    *out_cost = ext[(size_t)idx * ext_dim + dim + 1];
}

int protein_obs_sample_for_gp(ProteinObs *obs, int max_size, float recent_ratio, float *out_params, float *out_scores,
                              float *out_costs) {
    ProteinObsList *s = &obs->success;
    int dim = obs->dim;
    if (s->n == 0)
        return 0;

    obs->min_score = s->scores[0];
    obs->max_score = s->scores[0];
    for (int i = 1; i < s->n; i++) {
        if (s->scores[i] < obs->min_score)
            obs->min_score = s->scores[i];
        if (s->scores[i] > obs->max_score)
            obs->max_score = s->scores[i];
    }

    float *log_c_buf = (float *)malloc((size_t)s->n * sizeof(float));
    for (int i = 0; i < s->n; i++)
        log_c_buf[i] = logf(fmaxf(s->costs[i], PROTEIN_EPSILON));
    obs->log_c_min = log_c_buf[0];
    for (int i = 1; i < s->n; i++)
        if (log_c_buf[i] < obs->log_c_min)
            obs->log_c_min = log_c_buf[i];
    obs->log_c_max = protein_quantile(log_c_buf, s->n, PROTEIN_COST_QUANTILE);
    free(log_c_buf);

    ProteinObsList *f = &obs->failure;
    int use_failures = (s->n < PROTEIN_MIN_OBS_NO_FAIL && f->n > 0);
    int combined_n = use_failures ? f->n + s->n : s->n;
    int ext_dim = dim + 2;
    float *ext = (float *)malloc((size_t)combined_n * ext_dim * sizeof(float));

    int ci = 0;
    if (use_failures) {
        for (int i = 0; i < f->n; i++) {
            memcpy(&ext[(size_t)ci * ext_dim], &f->params[(size_t)i * dim], (size_t)dim * sizeof(float));
            ext[(size_t)ci * ext_dim + dim] = obs->min_score;
            ext[(size_t)ci * ext_dim + dim + 1] = f->costs[i];
            ci++;
        }
    }
    for (int i = 0; i < s->n; i++) {
        memcpy(&ext[(size_t)ci * ext_dim], &s->params[(size_t)i * dim], (size_t)dim * sizeof(float));
        ext[(size_t)ci * ext_dim + dim] = s->scores[i];
        ext[(size_t)ci * ext_dim + dim + 1] = s->costs[i];
        ci++;
    }

    int *kept = (int *)malloc((size_t)combined_n * sizeof(int));
    int n_kept = gp_filter_near_duplicates(ext, combined_n, ext_dim, PROTEIN_EPSILON, kept);

    if (n_kept <= max_size) {
        for (int i = 0; i < n_kept; i++)
            protein_obs_extract(ext, ext_dim, dim, kept[i], &out_params[(size_t)i * dim], &out_scores[i],
                                &out_costs[i]);
        free(kept);
        free(ext);
        return n_kept;
    }

    int recent_size = (int)(recent_ratio * (float)max_size);
    int older_size = n_kept - recent_size;
    int num_sample = max_size - recent_size;

    int *sample_idx = (int *)malloc((size_t)older_size * sizeof(int));
    for (int i = 0; i < older_size; i++)
        sample_idx[i] = i;
    for (int i = 0; i < num_sample && i < older_size; i++) {
        int j = i + rand() % (older_size - i);
        int tmp = sample_idx[i];
        sample_idx[i] = sample_idx[j];
        sample_idx[j] = tmp;
    }

    int out_n = 0;
    for (int i = 0; i < num_sample && i < older_size; i++) {
        protein_obs_extract(ext, ext_dim, dim, kept[sample_idx[i]], &out_params[(size_t)out_n * dim],
                            &out_scores[out_n], &out_costs[out_n]);
        out_n++;
    }
    for (int i = n_kept - recent_size; i < n_kept; i++) {
        protein_obs_extract(ext, ext_dim, dim, kept[i], &out_params[(size_t)out_n * dim], &out_scores[out_n],
                            &out_costs[out_n]);
        out_n++;
    }

    free(sample_idx);
    free(kept);
    free(ext);
    return out_n;
}

typedef struct {
    float score_loss;
    float cost_loss;
    float predicted_score;
    float predicted_cost;
    float rating;
    int n_pareto;
    int n_gp_obs;
    int n_candidates;
    int is_random;
} ProteinSweepInfo;

typedef struct {
    Hyperparameters *hypers;
    ProteinObs *obs;
    ProteinAcq *acq;
    ProteinCostModel cost_model;
    ProteinClassifier *clf;
    Sobol *sobol;
    GaussianProcess *gp_score;
    GaussianProcess *gp_cost;
    Adam *opt_score;
    Adam *opt_cost;
    cudaStream_t stream;
    int suggestion_idx;
    int num_random_samples;
    int suggestions_per_pareto;
    int gp_training_iter;
    int optimizer_reset_frequency;
    int gp_max_obs;
    int use_success_prob;
    int prune_pareto;
    int use_logit;
    float global_search_scale;
    float max_suggestion_cost;
    float expansion_rate;
    float cost_random_suggestion;
    float upper_cost_threshold;
    float ratio_pool[PROTEIN_NUM_COST_RATIOS];
    int pool_remaining;
    float running_buf[PROTEIN_RUNNING_BUF_CAP];
    int running_pos;
    int running_len;
    float *gp_train_params;
    float *gp_train_y;
    float *gp_train_c;
    int *pareto_buf;
    int *pruned_buf;
    float *centers_buf;
    int obs_cap, centers_cap;
} ProteinSweep;

ProteinSweep *protein_sweep_create(Hyperparameters *hypers, int num_random_samples, int suggestions_per_pareto,
                                   int gp_training_iter, float gp_learning_rate, int optimizer_reset_frequency,
                                   int gp_max_obs, int infer_batch_size, int use_success_prob, int prune_pareto,
                                   int use_logit, float global_search_scale, float max_suggestion_cost,
                                   float expansion_rate, float cost_random_suggestion, float early_stop_quantile,
                                   int success_cap, int failure_cap, int top_k, unsigned long long rng_seed) {
    int dim = hypers->num;
    ProteinSweep *sw = (ProteinSweep *)calloc(1, sizeof(ProteinSweep));
    sw->hypers = hypers;
    sw->num_random_samples = num_random_samples;
    sw->suggestions_per_pareto = suggestions_per_pareto;
    sw->gp_training_iter = gp_training_iter;
    sw->optimizer_reset_frequency = optimizer_reset_frequency;
    sw->gp_max_obs = gp_max_obs;
    sw->use_success_prob = use_success_prob;
    sw->prune_pareto = prune_pareto;
    sw->use_logit = use_logit;
    sw->global_search_scale = global_search_scale;
    sw->max_suggestion_cost = max_suggestion_cost;
    sw->expansion_rate = expansion_rate;
    sw->cost_random_suggestion = cost_random_suggestion;
    sw->upper_cost_threshold = -FLT_MAX;

    static const float default_ratios[PROTEIN_NUM_COST_RATIOS] = {0.16f, 0.32f, 0.48f, 0.64f, 0.8f, 1.0f};
    memcpy(sw->ratio_pool, default_ratios, sizeof(default_ratios));

    sw->obs = protein_obs_create(dim, success_cap, failure_cap, top_k, hypers->cost_idx);

    int max_centers = success_cap + 3 * top_k;
    int acq_cap = max_centers * suggestions_per_pareto;
    if (acq_cap < infer_batch_size)
        acq_cap = infer_batch_size;
    if (acq_cap > PROTEIN_ACQ_MAX_CAP)
        acq_cap = PROTEIN_ACQ_MAX_CAP;
    sw->acq = protein_acq_create(dim, acq_cap, hypers->bounds_min, hypers->bounds_max, hypers->scales, rng_seed);
    sw->cost_model.quantile = early_stop_quantile;
    sw->cost_model.min_samples = 30;
    sw->clf = protein_classifier_create(dim);
    sw->sobol = sobol_create(dim, rng_seed);

    sw->gp_score = gp_create(dim, gp_max_obs, gp_kernel_matern32_linear(dim, 1.0f, 1.0f, 1.0f), 0.01f);
    sw->gp_cost = gp_create(dim, gp_max_obs, gp_kernel_matern32_linear(dim, 1.0f, 1.0f, 1.0f), 0.01f);
    sw->opt_score = adam_create(sw->gp_score->kernel->n_params, gp_learning_rate);
    sw->opt_cost = adam_create(sw->gp_cost->kernel->n_params, gp_learning_rate);

    sw->obs_cap = success_cap;
    sw->centers_cap = max_centers;
    sw->gp_train_params = (float *)malloc((size_t)gp_max_obs * dim * sizeof(float));
    sw->gp_train_y = (float *)malloc((size_t)gp_max_obs * sizeof(float));
    sw->gp_train_c = (float *)malloc((size_t)gp_max_obs * sizeof(float));
    sw->pareto_buf = (int *)malloc((size_t)success_cap * sizeof(int));
    sw->pruned_buf = (int *)malloc((size_t)success_cap * sizeof(int));
    sw->centers_buf = (float *)malloc((size_t)max_centers * dim * sizeof(float));

    CUDA_CHECK(cudaStreamCreate(&sw->stream));
    return sw;
}

void protein_sweep_destroy(ProteinSweep *sw) {
    if (!sw)
        return;
    hyperparameters_destroy(sw->hypers);
    protein_obs_destroy(sw->obs);
    protein_acq_destroy(sw->acq);
    protein_classifier_destroy(sw->clf);
    sobol_destroy(sw->sobol);
    gp_destroy(sw->gp_score);
    gp_destroy(sw->gp_cost);
    adam_destroy(sw->opt_score);
    adam_destroy(sw->opt_cost);
    if (sw->stream)
        cudaStreamDestroy(sw->stream);
    free(sw->gp_train_params);
    free(sw->gp_train_y);
    free(sw->gp_train_c);
    free(sw->pareto_buf);
    free(sw->pruned_buf);
    free(sw->centers_buf);
    free(sw);
}

void protein_sweep_observe(ProteinSweep *sw, const float *norm_params, float score, float cost, int is_failure) {
    if (sw->use_logit)
        score = protein_logit_transform(score);
    protein_obs_add(sw->obs, norm_params, score, cost, is_failure);
}

void protein_sweep_add_running(ProteinSweep *sw, float val) {
    sw->running_buf[sw->running_pos] = val;
    sw->running_pos = (sw->running_pos + 1) % PROTEIN_RUNNING_BUF_CAP;
    if (sw->running_len < PROTEIN_RUNNING_BUF_CAP)
        sw->running_len++;
}

float protein_sweep_running_mean(const ProteinSweep *sw) {
    if (sw->running_len == 0)
        return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < sw->running_len; i++)
        sum += sw->running_buf[i];
    return sum / (float)sw->running_len;
}

float protein_sweep_get_threshold(const ProteinSweep *sw, float cost) {
    return protein_cost_model_threshold(&sw->cost_model, cost, 0.3f, 10.0f);
}

int protein_sweep_should_stop(const ProteinSweep *sw, float score, float cost) {
    float threshold = protein_sweep_get_threshold(sw, cost);
    if (sw->use_logit)
        score = protein_logit_transform(score);
    return score < threshold;
}

static void protein_sobol_fallback(ProteinSweep *sw, float *out, int is_fixed_cost, float fixed_cost_norm) {
    int dim = sw->hypers->num;
    int cost_dim = sw->hypers->cost_idx;
    CURAND_CHECK(curandGenerateUniform(sw->sobol->gen, out, sw->sobol->dim));
    for (int d = 0; d < dim; d++)
        out[d] = 2.0f * out[d] - 1.0f;
    if (is_fixed_cost && cost_dim >= 0) {
        out[cost_dim] = fixed_cost_norm;
    } else if (cost_dim >= 0) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        float noise = 0.1f * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
        out[cost_dim] = fmaxf(-1.0f, fminf(1.0f, sw->cost_random_suggestion + noise));
    }
}

ProteinSweepInfo protein_sweep_suggest(ProteinSweep *sw, float *out, float fixed_cost_norm) {
    ProteinSweepInfo info = {0};
    sw->suggestion_idx++;

    int dim = sw->hypers->num;
    int cost_dim = sw->hypers->cost_idx;
    int is_fixed_cost = !isnan(fixed_cost_norm);

    ProteinObsList *s = &sw->obs->success;
    if (sw->suggestion_idx <= sw->num_random_samples || s->n == 0) {
        protein_sobol_fallback(sw, out, is_fixed_cost, fixed_cost_norm);
        info.is_random = 1;
        return info;
    }

    int n_gp =
        protein_obs_sample_for_gp(sw->obs, sw->gp_max_obs, 0.5f, sw->gp_train_params, sw->gp_train_y, sw->gp_train_c);

    float min_score = sw->obs->min_score;
    float max_score = sw->obs->max_score;
    float log_c_min = sw->obs->log_c_min;
    float log_c_max = sw->obs->log_c_max;
    float score_range = fabsf(max_score - min_score) + PROTEIN_EPSILON;
    float cost_range = log_c_max - log_c_min + PROTEIN_EPSILON;

    float *y_norm = sw->gp_train_y;
    float *c_norm = sw->gp_train_c;
    for (int i = 0; i < n_gp; i++) {
        y_norm[i] = (y_norm[i] - min_score) / score_range;
        float lc = logf(fmaxf(c_norm[i], PROTEIN_EPSILON));
        c_norm[i] = (lc - log_c_min) / cost_range;
    }

    float score_loss = protein_train_gp(sw->gp_score, sw->opt_score, sw->gp_train_params, y_norm, n_gp,
                                        sw->gp_training_iter, sw->stream);
    float cost_loss = protein_train_gp(sw->gp_cost, sw->opt_cost, sw->gp_train_params, c_norm, n_gp,
                                       sw->gp_training_iter, sw->stream);

    if (sw->optimizer_reset_frequency > 0 && sw->suggestion_idx % sw->optimizer_reset_frequency == 0) {
        adam_reset(sw->opt_score);
        adam_reset(sw->opt_cost);
    }

    int n_pareto = protein_pareto_front(s->scores, s->costs, s->n, sw->pareto_buf);

    memcpy(sw->pruned_buf, sw->pareto_buf, (size_t)n_pareto * sizeof(int));
    int n_pruned = protein_prune_pareto(s->scores, s->costs, sw->pruned_buf, n_pareto, 0.5f, 0.98f);

    if (n_pruned > 0) {
        float pruned_max_cost = s->costs[sw->pruned_buf[n_pruned - 1]];
        if (sw->upper_cost_threshold < 0)
            sw->upper_cost_threshold = pruned_max_cost;
        else if (sw->upper_cost_threshold < pruned_max_cost)
            sw->upper_cost_threshold *= PROTEIN_COST_GROWTH;
    }
    protein_cost_model_fit(&sw->cost_model, s->scores, s->costs, s->n, sw->upper_cost_threshold);

    int *use_pareto = sw->prune_pareto ? sw->pruned_buf : sw->pareto_buf;
    int n_use = sw->prune_pareto ? n_pruned : n_pareto;

    int n_top = sw->obs->n_top;
    int n_centers = protein_build_search_centers(s->params, dim, use_pareto, n_use, sw->obs->top_idx, n_top, cost_dim,
                                                 sw->centers_buf);

    float target_cost = 0.0f;
    if (!is_fixed_cost) {
        target_cost = protein_sample_target_cost(sw->ratio_pool, &sw->pool_remaining, PROTEIN_NUM_COST_RATIOS,
                                                 sw->expansion_rate);
    }

    int n_total = n_centers * sw->suggestions_per_pareto;
    int n_cands = protein_acq_sample(sw->acq, sw->centers_buf, n_centers, n_total, sw->global_search_scale, cost_dim,
                                     is_fixed_cost ? fixed_cost_norm : NAN, PROTEIN_EPSILON, sw->stream);

    if (n_cands == 0) {
        protein_sobol_fallback(sw, out, is_fixed_cost, fixed_cost_norm);
        info.is_random = 1;
        return info;
    }

    float *d_success_prob = NULL;
    if (sw->use_success_prob && s->n > 9 && sw->obs->failure.n > 9) {
        int n_s = s->n, n_f = sw->obs->failure.n;
        int n_clf = n_s + n_f;
        float *X = (float *)malloc((size_t)n_clf * dim * sizeof(float));
        int *y = (int *)malloc((size_t)n_clf * sizeof(int));
        memcpy(X, s->params, (size_t)n_s * dim * sizeof(float));
        memcpy(X + (size_t)n_s * dim, sw->obs->failure.params, (size_t)n_f * dim * sizeof(float));
        for (int i = 0; i < n_s; i++)
            y[i] = 1;
        for (int i = 0; i < n_f; i++)
            y[n_s + i] = 0;
        protein_classifier_fit(sw->clf, X, y, n_clf, 1.0f, PROTEIN_CLF_ITERS);
        free(X);
        free(y);

        if (sw->clf->is_fitted) {
            protein_classifier_predict_d(sw->clf, sw->acq->d_candidates, sw->acq->d_success_prob, n_cands, sw->stream);
            d_success_prob = sw->acq->d_success_prob;
        }
    }

    ProteinAcqResult acq_result =
        protein_acq_suggest(sw->acq, n_cands, sw->gp_score, sw->gp_cost, min_score, max_score, log_c_min, log_c_max,
                            sw->max_suggestion_cost, target_cost, sw->hypers->optimize_direction, is_fixed_cost,
                            d_success_prob, sw->stream);

    CUDA_CHECK(cudaMemcpyAsync(out, &sw->acq->d_candidates[(size_t)acq_result.best_idx * sw->acq->dim],
                               (size_t)sw->acq->dim * sizeof(float), cudaMemcpyDeviceToHost, sw->stream));
    CUDA_CHECK(cudaStreamSynchronize(sw->stream));

    info.score_loss = score_loss;
    info.cost_loss = cost_loss;
    info.predicted_score = acq_result.predicted_score;
    info.predicted_cost = acq_result.predicted_cost;
    info.rating = acq_result.rating;
    info.n_pareto = n_use;
    info.n_gp_obs = n_gp;
    info.n_candidates = n_cands;
    return info;
}

#endif // PROTEIN_CU
