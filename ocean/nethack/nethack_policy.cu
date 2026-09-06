// Included by src/ocean.cu when building nethack (-DPUFFER_NETHACK).
// Verb-conditional heads + verb-eps. Not a separate translation unit.

__device__ const signed char* d_hc_dev = NULL;
__device__ int d_hc_stride = 0;
__device__ float* d_veps_dev = NULL;
static float* g_veps_dev = NULL;
static float g_veps_base = 0;
static float g_veps_start = 0.4f;
static float g_veps_end = 1.0f;

void nethack_policy_init(Ini* ini) {
    int nv = 0, na = 0;
    const signed char* host = env_head_consume_map(&nv, &na);
    if (host && nv > 0 && na > 0) {
        signed char* dev = NULL;
        assert(cudaMalloc(&dev, (size_t)nv * na) == cudaSuccess);
        assert(cudaMemcpy(dev, host, (size_t)nv * na,
            cudaMemcpyHostToDevice) == cudaSuccess);
        cudaMemcpyToSymbol(d_hc_dev, &dev, sizeof(dev));
        cudaMemcpyToSymbol(d_hc_stride, &na, sizeof(na));
    }
    g_veps_base = puf_ini_get(ini, "train", "verb_eps");
    g_veps_start = puf_ini_get(ini, "train", "verb_eps_anneal_start");
    g_veps_end = puf_ini_get(ini, "train", "verb_eps_anneal_end");
    if (g_veps_base > 0.0f) {
        cudaMalloc(&g_veps_dev, sizeof(float));
        cudaMemcpy(g_veps_dev, &g_veps_base, sizeof(float),
            cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(d_veps_dev, &g_veps_dev, sizeof(g_veps_dev));
    }
}

void nethack_policy_on_rollout(long step, long total) {
    if (g_veps_base <= 0.0f || !g_veps_dev) {
        return;
    }
    float a = g_veps_start;
    float ae = g_veps_end;
    if (a > 0.99f) {
        a = 0.99f;
    }
    if (a < 0.0f) {
        a = 0.0f;
    }
    if (ae > 1.0f || ae <= 0.0f) {
        ae = 1.0f;
    }
    if (ae < a + 0.01f) {
        ae = a + 0.01f;
    }
    double frac = total > 0 ? (double)step / (double)total : 0.0;
    float eps = frac < a ? g_veps_base
        : frac >= ae ? 0.0f
        : g_veps_base * (float)((ae - frac) / (ae - a));
    cudaMemcpy(g_veps_dev, &eps, sizeof(float), cudaMemcpyHostToDevice);
}

__device__ int nethack_head_used(int verb, int h) {
    if (h == 0 || !d_hc_dev) {
        return 1;
    }
    return (int)d_hc_dev[verb * d_hc_stride + h];
}

__device__ float nethack_verb_eps_load(const precision_t* mask_row,
        int A, float* inv_K) {
    *inv_K = 0.0f;
    if (!d_veps_dev) {
        return 0.0f;
    }
    float eps = *d_veps_dev;
    if (eps <= 0.0f) {
        return 0.0f;
    }
    int K = 0;
    for (int a = 0; a < A; a++) {
        if (to_float(mask_row[a]) != 0.0f) {
            K++;
        }
    }
    if (K == 0) {
        K = A;
    }
    *inv_K = 1.0f / (float)K;
    return eps;
}

__device__ float nethack_verb_eps_mix(float prob, precision_t mask,
        float eps, float inv_K) {
    float legal = to_float(mask) != 0.0f ? 1.0f : 0.0f;
    return (1.0f - eps) * prob + eps * legal * inv_K;
}

__device__ float nethack_verb_train_logp(const precision_t* mask_row,
        int A, float lp, float* scale) {
    if (!d_veps_dev) {
        return lp;
    }
    float eps = *d_veps_dev;
    if (eps <= 0.0f) {
        return lp;
    }
    int K = 0;
    for (int j = 0; j < A; ++j) {
        if (to_float(mask_row[j]) != 0.0f) {
            K++;
        }
    }
    if (K == 0) {
        K = A;
    }
    float p_act = __expf(lp);
    float p_mix = (1.0f - eps) * p_act + eps / (float)K;
    *scale = (1.0f - eps) * p_act / p_mix;
    return __logf(p_mix);
}
