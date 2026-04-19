// CPU rollout inference path for Metal.

#ifndef PUFFERLIB_CPU_INFERENCE_H
#define PUFFERLIB_CPU_INFERENCE_H

#include <Accelerate/Accelerate.h>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cassert>

// ============================================================================
// CPU Philox 4x32-10 RNG — matches MSL philox4x32_10 exactly
// ============================================================================

// Single Philox round: mulhi-based bijection.
// MSL uses mulhi(M, x) = (uint64_t(M) * x) >> 32.
static inline void cpu_philox_round(uint32_t ctr[4], const uint32_t key[2]) {
    constexpr uint32_t M0 = 0xD2511F53u;
    constexpr uint32_t M1 = 0xCD9E8D57u;
    uint32_t hi0 = (uint32_t)(((uint64_t)M0 * ctr[0]) >> 32);
    uint32_t lo0 = M0 * ctr[0];
    uint32_t hi1 = (uint32_t)(((uint64_t)M1 * ctr[2]) >> 32);
    uint32_t lo1 = M1 * ctr[2];
    // MSL: uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0)
    uint32_t new0 = hi1 ^ ctr[1] ^ key[0];
    uint32_t new1 = lo1;
    uint32_t new2 = hi0 ^ ctr[3] ^ key[1];
    uint32_t new3 = lo0;
    ctr[0] = new0; ctr[1] = new1; ctr[2] = new2; ctr[3] = new3;
}

// 10-round Philox: matches MSL philox4x32_10.
// counter = {idx, offset, 0, 0}, key = {seed_lo, seed_hi}.
static inline void cpu_philox4x32_10(uint32_t counter[4], uint32_t key[2]) {
    constexpr uint32_t W0 = 0x9E3779B9u;
    constexpr uint32_t W1 = 0xBB67AE85u;
    for (int i = 0; i < 10; i++) {
        cpu_philox_round(counter, key);
        key[0] += W0;
        key[1] += W1;
    }
}

// Per-agent RNG state for CPU sampling.
struct CpuPhiloxState {
    uint32_t counter[4];
    uint32_t key[2];
    uint32_t output[4];
    int consumed;  // how many of output[0..3] have been used
};

static inline void cpu_philox_init(CpuPhiloxState &s, uint32_t idx,
                                    uint32_t offset, uint64_t seed) {
    s.counter[0] = idx;
    s.counter[1] = offset;
    s.counter[2] = 0;
    s.counter[3] = 0;
    s.key[0] = (uint32_t)(seed & 0xFFFFFFFF);
    s.key[1] = (uint32_t)(seed >> 32);
    // Generate first 4 random values
    uint32_t ctr[4] = {s.counter[0], s.counter[1], s.counter[2], s.counter[3]};
    uint32_t key[2] = {s.key[0], s.key[1]};
    cpu_philox4x32_10(ctr, key);
    memcpy(s.output, ctr, 16);
    s.consumed = 0;
}

// MSL philox_uniform: reads output[state_idx & 3], increments state_idx.
// When all 4 consumed, bumps counter.z and regenerates.
static inline float cpu_philox_uniform(CpuPhiloxState &s) {
    if (s.consumed >= 4) {
        s.counter[2]++;
        uint32_t ctr[4] = {s.counter[0], s.counter[1], s.counter[2], s.counter[3]};
        uint32_t key[2] = {s.key[0], s.key[1]};
        cpu_philox4x32_10(ctr, key);
        memcpy(s.output, ctr, 16);
        s.consumed = 0;
    }
    uint32_t val = s.output[s.consumed++];
    return ((float)(val >> 8) + 0.5f) / 16777216.0f;
}

// ============================================================================
// CPU activation functions — matching MSL implementations exactly
// ============================================================================

// Stable sigmoid: matches MSL sigmoid_f. Used for gate and proj.
static inline float cpu_sigmoid(float x) {
    float z = expf(-fabsf(x));
    return x >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
}

// Horner polynomial tanh: matches MSL fast_tanh_f.
static inline float cpu_fast_tanh(float x) {
    float v1 = x < -9.0f ? -9.0f : (x > 9.0f ? 9.0f : x);
    float v2 = v1 * v1;
    float p = v2 * (-2.76076847742355e-16f) + 2.00018790482477e-13f;
    p = v2 * p + (-8.60467152213735e-11f);
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

// Polynomial sigmoid: matches MSL fast_sigmoid_f. Used inside tilde_relu only.
static inline float cpu_fast_sigmoid(float x) {
    float y = cpu_fast_tanh(x * 0.5f);
    float result = (y + 1.0f) * 0.5f;
    return result < 0.0f ? 0.0f : (result > 1.0f ? 1.0f : result);
}

// tilde_relu: x >= 0 ? x + 0.5 : fast_sigmoid(x). Matches MSL tilde_relu_fwd.
static inline float cpu_tilde_relu(float x) {
    return x >= 0.0f ? x + 0.5f : cpu_fast_sigmoid(x);
}

// Careful lerp avoiding catastrophic cancellation: matches MSL lerp_f.
static inline float cpu_lerp(float a, float b, float w) {
    float diff = b - a;
    return fabsf(w) < 0.5f ? a + w * diff : b - diff * (1.0f - w);
}

// ============================================================================
// CPU GEMM — cblas_sgemm via Accelerate (AMX/SME on Apple Silicon)
// ============================================================================

// out(M,N) = a(M,K) @ b(N,K)^T — matches puf_mm convention.
// Weight b is stored as (N,K) row-major (N rows of K elements).
static inline void cpu_mm_nt(const float *a, const float *b, float *out,
                              int M, int K, int N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, a, K, b, K, 0.0f, out, N);
}

// ============================================================================
// CPU MinGRU gate — element-wise matching MSL mingru_gate_inference
// ============================================================================

// combined: (B, 3*H) = [hidden | gate | proj] per row
// state_in, x_in, out, next_state: all (B, H)
static void cpu_mingru_gate(float *out, float *next_state,
                             const float *combined, const float *state_in,
                             const float *x_in, int H, int B) {
    int BH = B * H;
    for (int idx = 0; idx < BH; idx++) {
        int b = idx / H;
        int h = idx % H;
        int base = b * 3 * H;

        float hidden = combined[base + h];
        float gate   = combined[base + H + h];
        float proj   = combined[base + 2 * H + h];
        float state  = state_in[idx];
        float x      = x_in[idx];

        float gate_sig = cpu_sigmoid(gate);
        float hidden_tilde = cpu_tilde_relu(hidden);
        float mingru_out = cpu_lerp(state, hidden_tilde, gate_sig);
        float proj_sig = cpu_sigmoid(proj);

        next_state[idx] = fmaxf(mingru_out, 1e-30f);
        out[idx] = proj_sig * mingru_out + (1.0f - proj_sig) * x;
    }
}

// ============================================================================
// CPU discrete action sampling — matches MSL sample_logits_kernel
// ============================================================================

// Apply action mask: invalid actions get -1e9.
static inline float cpu_mask_logit(float logit, float mask) {
    if (mask < 0.5f) logit = -1e9f;
    return logit;
}

// Sample all action heads for B agents.
// dec_out: (B, fused_cols) — logits + value in last column
// act_sizes: (num_atns,) — number of discrete actions per head
// action_out_f32: (B, num_atns) — sampled action indices
// logprobs: (B,) — scalar joint log-probability (sum across heads, matches CUDA)
// value_out: (B,) — value head output
static void cpu_sample_logits(
        const float *dec_out, int fused_cols,
        const int *act_sizes, int num_atns,
        float *action_out_f32, float *logprobs, float *value_out,
        const float *action_mask, int mask_stride,
        uint64_t seed, uint32_t *offset_ptr, int B) {

    uint32_t offset_snapshot = *offset_ptr;
    *offset_ptr = offset_snapshot + 1u;

    for (int idx = 0; idx < B; idx++) {
        CpuPhiloxState rng;
        cpu_philox_init(rng, (uint32_t)idx, offset_snapshot, seed);

        const float *logits = dec_out + idx * fused_cols;
        // mask_stride=0 means all agents read the same mask (all-ones fallback)
        const float *mask = (mask_stride == 0)
            ? action_mask
            : action_mask + idx * mask_stride;

        int logits_offset = 0;
        // CUDA joint-ratio: accumulate scalar total_log_prob across heads
        float total_log_prob = 0.0f;

        for (int h = 0; h < num_atns; h++) {
            int A = act_sizes[h];

            // max for numerical stability
            float max_val = -INFINITY;
            for (int a = 0; a < A; a++) {
                float l = cpu_mask_logit(logits[logits_offset + a],
                                          mask[logits_offset + a]);
                if (l > max_val) max_val = l;
            }

            // logsumexp
            float sum_exp = 0.0f;
            for (int a = 0; a < A; a++) {
                float l = cpu_mask_logit(logits[logits_offset + a],
                                          mask[logits_offset + a]);
                sum_exp += expf(l - max_val);
            }
            float logsumexp_val = max_val + logf(sum_exp);

            // Philox uniform sample
            float rand_val = cpu_philox_uniform(rng);

            // Inverse CDF sampling
            float cumsum = 0.0f;
            int sampled = A - 1;
            for (int a = 0; a < A; a++) {
                float l = cpu_mask_logit(logits[logits_offset + a],
                                          mask[logits_offset + a]);
                cumsum += expf(l - logsumexp_val);
                if (rand_val < cumsum) {
                    sampled = a;
                    break;
                }
            }

            float sl = cpu_mask_logit(logits[logits_offset + sampled],
                                       mask[logits_offset + sampled]);
            total_log_prob += sl - logsumexp_val;

            action_out_f32[idx * num_atns + h] = (float)sampled;
            logits_offset += A;
        }
        // Scalar joint logprob matching CUDA: logprobs[idx] = sum of per-head log probs
        logprobs[idx] = total_log_prob;
        value_out[idx] = dec_out[idx * fused_cols + (fused_cols - 1)];
    }
}

// ============================================================================
// CPU forward pass — complete rollout step for one buffer
// ============================================================================

// Full CPU forward: encoder → MinGRU layers → decoder → sampling.
// Reads weights (unified memory, read-only). Writes to activation buffers
// (unified memory, per-buffer isolation). No GPU dispatch, no sync.
static void cpu_forward_and_sample(
        PufTensor &obs,             // (B, obs_dim)
        PufTensor &state,           // (num_layers, B, H)
        PolicyWeights &weights,
        int hidden_dim,
        PolicyActivations &acts,
        IntTensor &act_sizes_puf,
        FloatTensor &act_f32_buf,   // (B, num_atns) scratch
        float *logprobs_out,        // (B,)
        float *values_out,          // (B,)
        const float *action_mask, int mask_stride,
        uint64_t rng_seed, uint32_t *rng_offset_ptr) {

    int B = (int)obs.shape[0];
    int obs_dim = (int)obs.shape[1];
    int H = hidden_dim;

    EncoderWeights *ew = (EncoderWeights *)weights.encoder;
    MinGRUWeights *mw = (MinGRUWeights *)weights.network;
    MinGRUActivations *ma = (MinGRUActivations *)acts.network;
    DecoderWeights *dw = (DecoderWeights *)weights.decoder;
    DecoderActivations *da = (DecoderActivations *)acts.decoder;

    // --- Encoder ---
    EncoderActivations *ea = (EncoderActivations *)acts.encoder;
    cpu_mm_nt((const float *)obs.bytes, ew->weight.data,
              ea->out.data, B, obs_dim, ew->out_dim);
    float *layer_input = ea->out.data;

    // --- MinGRU layers ---
    for (int i = 0; i < mw->num_layers; i++) {
        float *state_i = (float *)state.bytes + i * B * H;
        int input_K = H;
        cpu_mm_nt(layer_input, mw->weights[i].data,
                  ma->combined[i].data, B, input_K, 3 * H);

        cpu_mingru_gate(ma->out.data, ma->next_state.data,
                        ma->combined[i].data,
                        state_i, layer_input, H, B);

        // Update RNN state
        memcpy(state_i, ma->next_state.data, B * H * sizeof(float));

        layer_input = ma->out.data;
    }

    // --- Decoder ---
    int fused_cols = dw->output_dim + 1;
    cpu_mm_nt(layer_input, dw->weight.data,
              da->out.data, B, H, fused_cols);

    // --- Sampling ---
    cpu_sample_logits(
        da->out.data, fused_cols,
        act_sizes_puf.data, (int)puf_numel(act_sizes_puf.shape),
        act_f32_buf.data, logprobs_out, values_out,
        action_mask, mask_stride,
        rng_seed, rng_offset_ptr, B);
}

#endif // PUFFERLIB_CPU_INFERENCE_H
