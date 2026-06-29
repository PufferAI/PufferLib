// protein_util.h -- Pure-C utilities for the Protein optimizer:
//                   search space, Pareto front, cost model, numeric helpers.
//
// Included by protein.cu.  No CUDA dependency.

#ifndef PROTEIN_UTIL_H
#define PROTEIN_UTIL_H

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// -- search space (Space types) -----------------------------------------------

enum SpaceType {
    SPACE_LINEAR,
    SPACE_LOG,
    SPACE_POW2,
    SPACE_LOGIT,
};

typedef struct {
    SpaceType type;
    float min, max, scale;
    float norm_min, norm_max;
    int is_integer;
} Space;

static float space_normalize(const Space* s, float value)
{
    float zero_one;
    switch (s->type) {
    case SPACE_LINEAR:
        zero_one = (value - s->min) / (s->max - s->min);
        break;
    case SPACE_LOG:
        zero_one = (log10f(value) - log10f(s->min))
            / (log10f(s->max) - log10f(s->min));
        break;
    case SPACE_POW2:
        zero_one = (log2f(value) - log2f(s->min))
            / (log2f(s->max) - log2f(s->min));
        break;
    case SPACE_LOGIT: {
        float clamped = fmaxf(s->min, fminf(value, s->max));
        zero_one = (log10f(1.0f - clamped) - log10f(1.0f - s->min))
            / (log10f(1.0f - s->max) - log10f(1.0f - s->min));
        break;
    }
    }
    return 2.0f * zero_one - 1.0f;
}

static float space_unnormalize(const Space* s, float norm)
{
    float zero_one = (norm + 1.0f) * 0.5f;
    float val;
    switch (s->type) {
    case SPACE_LINEAR:
        val = zero_one * (s->max - s->min) + s->min;
        if (s->is_integer)
            val = roundf(val);
        break;
    case SPACE_LOG: {
        float log_val = zero_one * (log10f(s->max) - log10f(s->min))
            + log10f(s->min);
        val = powf(10.0f, log_val);
        if (s->is_integer)
            val = roundf(val);
        break;
    }
    case SPACE_POW2: {
        float log_val = zero_one * (log2f(s->max) - log2f(s->min))
            + log2f(s->min);
        val = powf(2.0f, roundf(log_val));
        break;
    }
    case SPACE_LOGIT: {
        float log_val = zero_one * (log10f(1.0f - s->max) - log10f(1.0f - s->min))
            + log10f(1.0f - s->min);
        val = 1.0f - powf(10.0f, log_val);
        break;
    }
    }
    return val;
}

static void space_init(Space* s, SpaceType type, float min, float max,
    float scale, int is_integer)
{
    s->type = type;
    s->min = min;
    s->max = max;
    s->scale = scale;
    s->is_integer = is_integer;
    s->norm_min = space_normalize(s, min);
    s->norm_max = space_normalize(s, max);
}

// -- Hyperparameters ----------------------------------------------------------

typedef struct {
    Space* spaces; // [num]
    float* bounds_min; // [num] normalised
    float* bounds_max; // [num] normalised
    float* scales; // [num]
    int num;
    int cost_idx; // index of cost parameter (-1 if none)
    int optimize_direction; // 1 = maximise, -1 = minimise
} Hyperparameters;

Hyperparameters* hyperparameters_create(const Space* spaces, int num,
    int cost_idx, int optimize_direction)
{
    Hyperparameters* h = (Hyperparameters*)calloc(1, sizeof(Hyperparameters));
    h->num = num;
    h->cost_idx = cost_idx;
    h->optimize_direction = optimize_direction;
    h->spaces = (Space*)malloc((size_t)num * sizeof(Space));
    h->bounds_min = (float*)malloc((size_t)num * sizeof(float));
    h->bounds_max = (float*)malloc((size_t)num * sizeof(float));
    h->scales = (float*)malloc((size_t)num * sizeof(float));
    memcpy(h->spaces, spaces, (size_t)num * sizeof(Space));
    for (int i = 0; i < num; i++) {
        h->bounds_min[i] = h->spaces[i].norm_min;
        h->bounds_max[i] = h->spaces[i].norm_max;
        h->scales[i] = h->spaces[i].scale;
    }
    return h;
}

void hyperparameters_destroy(Hyperparameters* h)
{
    if (!h)
        return;
    free(h->spaces);
    free(h->bounds_min);
    free(h->bounds_max);
    free(h->scales);
    free(h);
}

// -- numeric helpers ----------------------------------------------------------

static int protein_float_cmp(const void* a, const void* b)
{
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

static float protein_quantile(const float* data, int n, float q)
{
    float* sorted = (float*)malloc((size_t)n * sizeof(float));
    memcpy(sorted, data, (size_t)n * sizeof(float));
    qsort(sorted, n, sizeof(float), protein_float_cmp);
    float idx = q * (float)(n - 1);
    int lo = (int)idx, hi = lo + 1;
    if (hi >= n)
        hi = n - 1;
    float frac = idx - (float)lo;
    float val = sorted[lo] * (1.0f - frac) + sorted[hi] * frac;
    free(sorted);
    return val;
}

typedef struct {
    const float* x;
    const float* y;
    int n;
    float q;
} PinballData;

static float pinball_loss(float a, float b, const void* data)
{
    const PinballData* pd = (const PinballData*)data;
    float loss = 0.0f;
    for (int i = 0; i < pd->n; i++) {
        float r = pd->y[i] - (a + b * pd->x[i]);
        loss += (r > 0.0f) ? pd->q * r : (pd->q - 1.0f) * r;
    }
    return loss;
}

static void nelder_mead_2d(float (*f)(float, float, const void*),
    const void* data,
    float* out_a, float* out_b,
    float a0, float b0, int max_iter, float tol)
{
    float sx[3] = { a0, a0 + 0.05f, a0 - 0.05f };
    float sy[3] = { b0, b0 - 0.05f, b0 + 0.05f };
    float sv[3];
    for (int i = 0; i < 3; i++)
        sv[i] = f(sx[i], sy[i], data);

    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < 2; i++)
            for (int j = i + 1; j < 3; j++)
                if (sv[j] < sv[i]) {
                    float t;
                    t = sx[i];
                    sx[i] = sx[j];
                    sx[j] = t;
                    t = sy[i];
                    sy[i] = sy[j];
                    sy[j] = t;
                    t = sv[i];
                    sv[i] = sv[j];
                    sv[j] = t;
                }

        if (fabsf(sv[2] - sv[0]) < tol)
            break;

        float cx = (sx[0] + sx[1]) * 0.5f;
        float cy = (sy[0] + sy[1]) * 0.5f;

        float rx = cx + (cx - sx[2]), ry = cy + (cy - sy[2]);
        float rv = f(rx, ry, data);
        if (rv < sv[1] && rv >= sv[0]) {
            sx[2] = rx;
            sy[2] = ry;
            sv[2] = rv;
            continue;
        }
        if (rv < sv[0]) {
            float ex = cx + 2.0f * (rx - cx), ey = cy + 2.0f * (ry - cy);
            float ev = f(ex, ey, data);
            if (ev < rv) {
                sx[2] = ex;
                sy[2] = ey;
                sv[2] = ev;
            } else {
                sx[2] = rx;
                sy[2] = ry;
                sv[2] = rv;
            }
            continue;
        }
        float ccx, ccy;
        if (rv < sv[2]) {
            ccx = cx + 0.5f * (rx - cx);
            ccy = cy + 0.5f * (ry - cy);
        } else {
            ccx = cx + 0.5f * (sx[2] - cx);
            ccy = cy + 0.5f * (sy[2] - cy);
        }
        float ccv = f(ccx, ccy, data);
        if (ccv < sv[2]) {
            sx[2] = ccx;
            sy[2] = ccy;
            sv[2] = ccv;
            continue;
        }

        for (int i = 1; i < 3; i++) {
            sx[i] = sx[0] + 0.5f * (sx[i] - sx[0]);
            sy[i] = sy[0] + 0.5f * (sy[i] - sy[0]);
            sv[i] = f(sx[i], sy[i], data);
        }
    }
    *out_a = sx[0];
    *out_b = sy[0];
}

static void polyfit_1d(const float* x, const float* y, int n,
    float* intercept, float* slope)
{
    float sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < n; i++) {
        sx += x[i];
        sy += y[i];
        sxx += x[i] * x[i];
        sxy += x[i] * y[i];
    }
    float det = (float)n * sxx - sx * sx;
    if (fabsf(det) < 1e-10f) {
        *intercept = sy / fmaxf((float)n, 1.0f);
        *slope = 0.0f;
        return;
    }
    *slope = ((float)n * sxy - sx * sy) / det;
    *intercept = (sy - *slope * sx) / (float)n;
}

// -- cost model ---------------------------------------------------------------

typedef struct {
    float A, B;
    float max_score;
    float upper_cost_threshold;
    float quantile;
    int min_samples;
    int is_fitted;
} ProteinCostModel;

void protein_cost_model_init(ProteinCostModel* m, float quantile, int min_samples)
{
    memset(m, 0, sizeof(*m));
    m->quantile = quantile;
    m->min_samples = min_samples;
}

void protein_cost_model_fit(ProteinCostModel* m,
    const float* scores, const float* costs, int n,
    float upper_cost_threshold)
{
    m->is_fitted = 0;
    if (n == 0)
        return;

    float s_max = scores[0];
    for (int i = 1; i < n; i++)
        if (scores[i] > s_max)
            s_max = scores[i];
    m->max_score = s_max;
    m->upper_cost_threshold = upper_cost_threshold;

    int n_valid = 0;
    for (int i = 0; i < n; i++)
        if (costs[i] > PROTEIN_EPSILON && isfinite(scores[i]))
            n_valid++;

    if (n_valid < m->min_samples)
        return;

    float* x = (float*)malloc((size_t)n_valid * sizeof(float));
    float* y = (float*)malloc((size_t)n_valid * sizeof(float));
    int j = 0;
    for (int i = 0; i < n; i++) {
        if (costs[i] > PROTEIN_EPSILON && isfinite(scores[i])) {
            x[j] = logf(costs[i]);
            y[j] = scores[i];
            j++;
        }
    }

    float a_init, b_init;
    polyfit_1d(x, y, n_valid, &a_init, &b_init);

    PinballData pd = { x, y, n_valid, m->quantile };
    nelder_mead_2d(pinball_loss, &pd, &m->A, &m->B,
        a_init, b_init, 500, 1e-7f);

    free(x);
    free(y);
    m->is_fitted = 1;
}

float protein_cost_model_threshold(const ProteinCostModel* m, float cost,
    float min_cost_frac, float abs_min_cost)
{
    if (!m->is_fitted)
        return -FLT_MAX;

    float min_allowed = m->upper_cost_threshold * min_cost_frac + abs_min_cost;
    if (cost < min_allowed)
        return -FLT_MAX;
    if (cost > PROTEIN_THRESHOLD_COST_CAP * m->upper_cost_threshold)
        return PROTEIN_THRESHOLD_FALLBACK * m->max_score;
    return m->A + m->B * logf(cost);
}

// -- Pareto utilities ---------------------------------------------------------

static const float* g_protein_pareto_costs;

static int protein_pareto_cmp(const void* a, const void* b)
{
    float ca = g_protein_pareto_costs[*(const int*)a];
    float cb = g_protein_pareto_costs[*(const int*)b];
    return (ca > cb) - (ca < cb);
}

int protein_pareto_front(const float* scores, const float* costs, int n,
    int* out_indices)
{
    if (n == 0)
        return 0;

    int* sorted = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++)
        sorted[i] = i;
    g_protein_pareto_costs = costs;
    qsort(sorted, n, sizeof(int), protein_pareto_cmp);

    int count = 0;
    float max_score = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        int idx = sorted[i];
        if (scores[idx] > max_score + PROTEIN_EPSILON) {
            out_indices[count++] = idx;
            max_score = scores[idx];
        }
    }

    free(sorted);
    return count;
}

int protein_prune_pareto(const float* scores, const float* costs,
    int* indices, int n,
    float eff_threshold, float stop_fraction)
{
    if (n < 2)
        return n;

    float s_min = scores[indices[0]], s_max = s_min;
    float c_min = costs[indices[0]], c_max = c_min;
    for (int i = 1; i < n; i++) {
        float s = scores[indices[i]], c = costs[indices[i]];
        if (s < s_min)
            s_min = s;
        if (s > s_max)
            s_max = s;
        if (c < c_min)
            c_min = c;
        if (c > c_max)
            c_max = c;
    }
    float s_range = fmaxf(s_max - s_min, PROTEIN_EPSILON);
    float c_range = fmaxf(c_max - c_min, PROTEIN_EPSILON);
    float max_pareto_s = scores[indices[n - 1]];

    int pruned = n;
    for (int i = pruned - 1; i > 1; i--) {
        if (scores[indices[i - 1]] < stop_fraction * max_pareto_s)
            break;
        float ng = (scores[indices[i]] - scores[indices[i - 1]]) / s_range;
        float nc = (costs[indices[i]] - costs[indices[i - 1]]) / c_range;
        float eff = ng / (nc + PROTEIN_EPSILON);
        if (eff < eff_threshold) {
            for (int j = i; j < pruned - 1; j++)
                indices[j] = indices[j + 1];
            pruned--;
        } else {
            break;
        }
    }
    return pruned;
}

int protein_build_search_centers(const float* obs_params, int dim,
    const int* pareto_indices, int n_pareto,
    const int* top_indices, int n_top,
    int cost_dim, float* out_centers)
{
    int count = 0;
    for (int i = 0; i < n_pareto; i++) {
        memcpy(&out_centers[(size_t)count * dim],
            &obs_params[(size_t)pareto_indices[i] * dim],
            (size_t)dim * sizeof(float));
        count++;
    }
    if (cost_dim >= 0) {
        for (int i = 0; i < n_top; i++) {
            const float* src = &obs_params[(size_t)top_indices[i] * dim];
            float orig = src[cost_dim];
            memcpy(&out_centers[(size_t)count * dim], src, (size_t)dim * sizeof(float));
            out_centers[(size_t)count * dim + cost_dim] = orig - (orig + 1.0f) / 2.0f;
            count++;
        }
        for (int i = 0; i < n_top; i++) {
            const float* src = &obs_params[(size_t)top_indices[i] * dim];
            float orig = src[cost_dim];
            memcpy(&out_centers[(size_t)count * dim], src, (size_t)dim * sizeof(float));
            out_centers[(size_t)count * dim + cost_dim] = orig - (orig + 1.0f) / 3.0f;
            count++;
        }
    } else {
        for (int i = 0; i < n_top; i++) {
            memcpy(&out_centers[(size_t)count * dim],
                &obs_params[(size_t)top_indices[i] * dim],
                (size_t)dim * sizeof(float));
            count++;
        }
    }
    return count;
}

float protein_sample_target_cost(float* ratio_pool, int* pool_remaining,
    int pool_total, float expansion_rate)
{
    if (*pool_remaining <= 0) {
        for (int i = pool_total - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            float tmp = ratio_pool[i];
            ratio_pool[i] = ratio_pool[j];
            ratio_pool[j] = tmp;
        }
        *pool_remaining = pool_total;
    }
    float ratio = ratio_pool[--(*pool_remaining)];

    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float noise = 0.1f * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);

    ratio = fmaxf(0.0f, fminf(1.0f, ratio + noise));
    return (1.0f + expansion_rate) * ratio;
}

static float protein_logit_transform(float value)
{
    float epsilon = 1e-9f;
    value = fmaxf(epsilon, fminf(1.0f - epsilon, value));
    float logit = logf(value / (1.0f - value));
    return fmaxf(-5.0f, fminf(100.0f, logit));
}

#endif // PROTEIN_UTIL_H
