#ifndef PUFFERLIB_METAL_SHADER_SRC_H
#define PUFFERLIB_METAL_SHADER_SRC_H

static const char *get_all_metal_shader_source() {
  return R"METAL(
#include <metal_stdlib>
#include <metal_math>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_atomic>
using namespace metal;

inline float sigmoid_f(float x) {
    float z = exp(-abs(x));
    return x >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
}

inline float sigmoid_backward_f(float x, float grad_output) {
    float sig = sigmoid_f(x);
    return grad_output * sig * (1.0f - sig);
}

inline float fast_tanh_f(float x) {
    float v1 = clamp(x, -9.0f, 9.0f);
    float v2 = v1 * v1;
    // Horner polynomial (matches PyTorch implementation)
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

inline float fast_sigmoid_f(float x) {
    float y = fast_tanh_f(x * 0.5f);
    return clamp((y + 1.0f) * 0.5f, 0.0f, 1.0f);
}

inline float tilde_relu_fwd(float x) {
    return x >= 0.0f ? x + 0.5f : fast_sigmoid_f(x);
}

inline float tilde_relu_bwd(float x, float grad) {
    if (x >= 0.0f) return grad;
    float sig = fast_sigmoid_f(x);
    return grad * sig * (1.0f - sig);
}

inline float lerp_f(float a, float b, float w) {
    float diff = b - a;
    return abs(w) < 0.5f ? a + w * diff : b - diff * (1.0f - w);
}

// MSL has no built-in log1p. Goldberg's trick: compensates for rounding in 1+x.
inline float log1p_f(float x) {
    float u = 1.0f + x;
    return (u == 1.0f) ? x : log(u) * x / (u - 1.0f);
}

constant float SOFTPLUS_BETA = 1.0f;
constant float SOFTPLUS_THRESHOLD = 20.0f;

inline float softplus_fwd(float x) {
    float xs = x * SOFTPLUS_BETA;
    return xs > SOFTPLUS_THRESHOLD ? x : log1p_f(exp(xs)) / SOFTPLUS_BETA;
}

inline float softplus_bwd(float grad_output, float x) {
    float beta_x = SOFTPLUS_BETA * x;
    if (beta_x > SOFTPLUS_THRESHOLD) return grad_output;
    float exp_beta_x = exp(beta_x);
    return grad_output * (exp_beta_x / (1.0f + exp_beta_x));
}

inline void log_coeffs_and_values_fwd(float gate, float hidden,
                                       thread float& log_coeff, thread float& log_value) {
    float abs_gate = abs(gate);
    float sp_neg = log1p_f(exp(-abs_gate));
    float softplus_gate, softplus_neg_gate;
    if (gate >= 0.0f) {
        softplus_gate = gate + sp_neg;
        softplus_neg_gate = sp_neg;
    } else {
        softplus_gate = sp_neg;
        softplus_neg_gate = -gate + sp_neg;
    }
    log_coeff = -softplus_gate;
    float log_z = -softplus_neg_gate;
    float log_tilde_h = hidden >= 0.0f ? log(hidden + 0.5f) : -softplus_fwd(-hidden);
    log_value = log_z + log_tilde_h;
}

inline void log_coeffs_and_values_bwd(float grad_lc, float grad_lv,
                                       float gate, float hidden,
                                       thread float& grad_gate, thread float& grad_hidden) {
    float sig_gate = sigmoid_f(gate);
    grad_gate = -grad_lc * sig_gate + grad_lv * (1.0f - sig_gate);
    if (hidden >= 0.0f) {
        grad_hidden = grad_lv / (hidden + 0.5f);
    } else {
        grad_hidden = grad_lv * sigmoid_f(-hidden);
    }
}

inline float relu_f(float x) { return max(0.0f, x); }
inline float relu_backward_f(float x, float grad_output) { return (x > 0.0f) ? grad_output : 0.0f; }

struct Philox4x32 {
    uint4 counter;
    uint2 key;
};

inline uint4 philox4x32_round(uint4 ctr, uint2 key) {
    constexpr uint PHILOX_M0 = 0xD2511F53u;
    constexpr uint PHILOX_M1 = 0xCD9E8D57u;
    uint hi0 = mulhi(PHILOX_M0, ctr.x);
    uint lo0 = PHILOX_M0 * ctr.x;
    uint hi1 = mulhi(PHILOX_M1, ctr.z);
    uint lo1 = PHILOX_M1 * ctr.z;
    return uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
}

inline uint4 philox4x32_10(uint4 counter, uint2 key) {
    constexpr uint PHILOX_W0 = 0x9E3779B9u;
    constexpr uint PHILOX_W1 = 0xBB67AE85u;
    for (int i = 0; i < 10; i++) {
        counter = philox4x32_round(counter, key);
        key.x += PHILOX_W0;
        key.y += PHILOX_W1;
    }
    return counter;
}

inline float philox_uniform(thread uint& state_idx, uint4 rng_out) {
    uint val;
    switch (state_idx & 3) {
        case 0: val = rng_out.x; break;
        case 1: val = rng_out.y; break;
        case 2: val = rng_out.z; break;
        default: val = rng_out.w; break;
    }
    state_idx++;
    // Convert to (0, 1] uniform
    return (float(val >> 8) + 0.5f) / 16777216.0f;
}

// Box-Muller transform for Normal(0,1)
inline float philox_normal(float u1, float u2) {
    return sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI_F * u2);
}

// mingru_gate_inference: fused chunk + tilde_relu + lerp + highway output gate
// combined is (B, 3*H) = [hidden, gate, proj], state is (B, H)
// x_in is (B, H) = input before projection
// out = sigmoid(proj) * mingru_out + (1 - sigmoid(proj)) * x_in
// next_state = mingru_out
struct MingruGateParams {
    int H;
    int B;
};

kernel void mingru_gate_inference(
    device float* out               [[buffer(0)]],
    device float* next_state        [[buffer(1)]],
    const device float* combined    [[buffer(2)]],
    const device float* state_in    [[buffer(3)]],
    const device float* x_in        [[buffer(4)]],
    constant MingruGateParams& p    [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    int N = p.B * p.H;
    if ((int)idx >= N) return;

    int b = (int)idx / p.H;
    int h = (int)idx % p.H;
    int base = b * 3 * p.H;

    float hidden = combined[base + h];
    float gate = combined[base + p.H + h];
    float proj = combined[base + 2 * p.H + h];
    float state = state_in[idx];
    float x = x_in[idx];

    float gate_sig = sigmoid_f(gate);
    float hidden_tilde = tilde_relu_fwd(hidden);
    float mingru_out = lerp_f(state, hidden_tilde, gate_sig);
    float proj_sig = sigmoid_f(proj);

    next_state[idx] = max(mingru_out, 1e-30f);
    out[idx] = proj_sig * mingru_out + (1.0f - proj_sig) * x;
}

constant int CHECKPOINT_INTERVAL = 4;

struct ScanParams {
    int T_seq;
    int H;
    int B;
};

kernel void mingru_scan_forward_checkpointed(
    device float* out               [[buffer(0)]],
    device float* next_state        [[buffer(1)]],
    device float* a_star_buf        [[buffer(2)]],
    device float* s_buf             [[buffer(3)]],
    device float* log_values_buf    [[buffer(4)]],
    const device float* combined    [[buffer(5)]],
    const device float* state       [[buffer(6)]],
    const device float* input       [[buffer(7)]],
    constant ScanParams& p          [[buffer(8)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx >= p.B * p.H) return;

    int b = (int)idx / p.H;
    int h = (int)idx % p.H;
    int bH = b * p.H;
    int H3 = 3 * p.H;
    int bHT = bH * p.T_seq;
    int out_base = bHT + h;
    int cbase = 3 * bHT;

    float a_star = 0.0f;
    float log_value = 0.0f;
    // fast:: matches CUDA __expf/__logf (kernels.cu:270-305)
    float s = fast::log(state[bH + h]);
    log_value = s;

    int T_out = p.T_seq + 1;
    int buf_base = b * T_out * p.H + h;
    int buf_curr = buf_base;
    a_star_buf[buf_curr] = a_star;
    s_buf[buf_curr] = s;
    log_values_buf[buf_curr] = log_value;

    int out_curr = out_base;
    int t_offset = 0;

    for (int t = 1; t < p.T_seq + 1; t++) {
        float hidden_val = combined[cbase + h + t_offset];
        float gate_val = combined[cbase + p.H + h + t_offset];
        float proj_val = combined[cbase + 2 * p.H + h + t_offset];
        float x_val = input[out_base + (t - 1) * p.H];

        float log_coeff_val;
        log_coeffs_and_values_fwd(gate_val, hidden_val, log_coeff_val, log_value);

        a_star += log_coeff_val;

        float z = log_value - a_star;
        float max_val = fmax(s, z);
        s = max_val + log1p_f(fast::exp(-abs(s - z)));

        float scan_result = fast::exp(a_star + s);
        float proj_sigmoid = sigmoid_f(proj_val);

        out[out_curr] = proj_sigmoid * scan_result + (1.0f - proj_sigmoid) * x_val;

        buf_curr += p.H;
        out_curr += p.H;
        t_offset += H3;

        if (t % CHECKPOINT_INTERVAL == 0) {
            a_star_buf[buf_curr] = a_star;
            s_buf[buf_curr] = s;
            log_values_buf[buf_curr] = log_value;
        }
    }

    // Floor at 1e-30 to prevent log(0)=-inf on the next forward pass.
    // exp(a_star+s) underflows to exactly 0.0f in fp32 when a_star+s < -87.3.
    // A zero state causes log(0)=-inf → permanent -inf propagation through
    // all subsequent scan steps. 1e-30 is well above fp32 denormal range
    // and below any meaningful state value.
    next_state[bH + h] = max(fast::exp(a_star + s), 1e-30f);
}

kernel void mingru_scan_backward_checkpointed(
    device float* grad_combined          [[buffer(0)]],
    device float* grad_state             [[buffer(1)]],
    device float* grad_input             [[buffer(2)]],
    const device float* grad_out         [[buffer(3)]],
    const device float* grad_next_state  [[buffer(4)]],
    const device float* combined         [[buffer(5)]],
    const device float* state            [[buffer(6)]],
    const device float* input            [[buffer(7)]],
    const device float* a_star_buf       [[buffer(8)]],
    const device float* s_buf            [[buffer(9)]],
    const device float* log_values_buf   [[buffer(10)]],
    constant ScanParams& p               [[buffer(11)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx >= p.B * p.H) return;

    int b = (int)idx / p.H;
    int h = (int)idx % p.H;
    int bHT = b * p.H * p.T_seq;
    int cbase = 3 * bHT;
    int H3 = 3 * p.H;
    int state_idx = b * p.H + h;
    int out_base = bHT + h;

    int T_out = p.T_seq + 1;
    int buf_base = b * T_out * p.H + h;

    float acc = 0.0f;
    float s_val_next = 0.0f;
    float carry_grad_a = 0.0f;

    for (int chunk_end = p.T_seq; chunk_end > 0; chunk_end -= CHECKPOINT_INTERVAL) {
        int chunk_start = (chunk_end > CHECKPOINT_INTERVAL) ? (chunk_end - CHECKPOINT_INTERVAL) : 0;
        int chunk_len = chunk_end - chunk_start;

        // Chunk storage in thread-local arrays
        float chunk_a_star[CHECKPOINT_INTERVAL];
        float chunk_s[CHECKPOINT_INTERVAL];
        float chunk_log_values[CHECKPOINT_INTERVAL];
        float chunk_hidden[CHECKPOINT_INTERVAL];
        float chunk_gate[CHECKPOINT_INTERVAL];

        // Load checkpoint
        int ckpt_buf_idx = buf_base + chunk_start * p.H;
        float recomp_a_star = a_star_buf[ckpt_buf_idx];
        float recomp_s = s_buf[ckpt_buf_idx];
        float recomp_log_value = log_values_buf[ckpt_buf_idx];

        // Forward recompute within chunk
        for (int i = 0; i < chunk_len; i++) {
            int t = chunk_start + 1 + i;
            int t_offset = (t - 1) * H3;
            float hv = combined[cbase + h + t_offset];
            float gv = combined[cbase + p.H + h + t_offset];

            float lc;
            log_coeffs_and_values_fwd(gv, hv, lc, recomp_log_value);
            recomp_a_star += lc;

            float z = recomp_log_value - recomp_a_star;
            float mv = fmax(recomp_s, z);
            recomp_s = mv + log1p_f(fast::exp(-abs(recomp_s - z)));

            chunk_a_star[i] = recomp_a_star;
            chunk_s[i] = recomp_s;
            chunk_log_values[i] = recomp_log_value;
            chunk_hidden[i] = hv;
            chunk_gate[i] = gv;
        }

        // Backward through chunk
        for (int i = chunk_len - 1; i >= 0; i--) {
            int t = chunk_start + 1 + i;
            int t_offset = (t - 1) * H3;

            float a_star_t = chunk_a_star[i];
            float s_t = chunk_s[i];
            float log_value_t = chunk_log_values[i];
            float hidden_val = chunk_hidden[i];
            float gate_val = chunk_gate[i];
            float proj_val = combined[cbase + 2 * p.H + h + t_offset];
            int input_idx = out_base + (t - 1) * p.H;
            float x_val = input[input_idx];

            float scan_result = fast::exp(a_star_t + s_t);
            float z = log_value_t - a_star_t;

            float grad_out_val = grad_out[input_idx];
            float grad_scan_from_next = (t == p.T_seq) ? grad_next_state[state_idx] : 0.0f;

            float proj_sigmoid = sigmoid_f(proj_val);
            float grad_scan_result = grad_scan_from_next + grad_out_val * proj_sigmoid;
            float grad_proj = grad_out_val * (scan_result - x_val) * proj_sigmoid * (1.0f - proj_sigmoid);
            grad_input[input_idx] = grad_out_val * (1.0f - proj_sigmoid);

            float grad_log_h = grad_scan_result * scan_result;
            float grad_s = grad_log_h;

            if (t == p.T_seq) {
                acc = grad_s;
            } else {
                acc = grad_s + acc * fast::exp(s_t - s_val_next);
            }
            float grad_z = acc * fast::exp(z - s_t);
            s_val_next = s_t;

            float grad_a = grad_log_h + carry_grad_a - grad_z;
            carry_grad_a = grad_a;

            float grad_g, grad_h;
            log_coeffs_and_values_bwd(grad_a, grad_z, gate_val, hidden_val, grad_g, grad_h);

            grad_combined[cbase + h + t_offset] = grad_h;
            grad_combined[cbase + p.H + h + t_offset] = grad_g;
            grad_combined[cbase + 2 * p.H + h + t_offset] = grad_proj;
        }
    }

    // Gradient for initial state (t=0)
    int ckpt_0_idx = buf_base;
    float a_star_0 = a_star_buf[ckpt_0_idx];
    float s_0 = s_buf[ckpt_0_idx];
    float log_value_0 = log_values_buf[ckpt_0_idx];

    acc = acc * fast::exp(s_0 - s_val_next);
    float grad_z_0 = acc * fast::exp((log_value_0 - a_star_0) - s_0);

    grad_state[state_idx] = (state[state_idx] > 0.0f) ? (grad_z_0 / state[state_idx]) : 0.0f;
}

struct SampleParams {
    uint64_t seed;
    uint offset;
    int num_atns;
    int num_atns_total;  // sum of act_sizes
    int B;
    int logits_stride;
    int logstd_stride;
    int value_stride;
    int is_continuous;  // 1 for continuous, 0 for discrete
    int mask_stride;    // stride between rows in mask buffer (may differ from num_atns_total)
};

// Apply action mask to a logit: invalid actions get -1e9.
inline float masked_logit(float l, float m) {
    if (m < 0.5f) l = -1e9f;
    return l;
}

kernel void sample_logits_kernel(
    device float* actions               [[buffer(0)]],
    device float* logprobs              [[buffer(1)]],
    device float* value_out             [[buffer(2)]],
    const device float* logits          [[buffer(3)]],
    const device float* logstd          [[buffer(4)]],
    const device float* value           [[buffer(5)]],
    const device int* act_sizes         [[buffer(6)]],
    constant SampleParams& sp           [[buffer(7)]],
    const device float* action_mask     [[buffer(8)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx >= sp.B) return;

    uint offset = sp.offset;

    // Generate Philox RNG state
    uint4 counter = uint4((uint)idx, offset, 0u, 0u);
    uint2 key = uint2((uint)(sp.seed & 0xFFFFFFFF), (uint)(sp.seed >> 32));
    uint4 rng_out = philox4x32_10(counter, key);
    uint rng_idx = 0;

    int logits_base = (int)idx * sp.logits_stride;
    float total_log_prob = 0.0f;

    if (sp.is_continuous) {
        constexpr float LOG_2PI = 1.8378770664093453f;
        int logstd_base = (int)idx * sp.logstd_stride;

        for (int h = 0; h < sp.num_atns; h++) {
            float mean = logits[logits_base + h];
            float log_std = logstd[logstd_base + h];
            float std = exp(log_std);

            // Need 2 uniforms for Box-Muller
            float u1 = philox_uniform(rng_idx, rng_out);
            float u2 = philox_uniform(rng_idx, rng_out);
            if (rng_idx >= 4) {
                counter.z++;
                rng_out = philox4x32_10(counter, key);
                rng_idx = 0;
            }
            float noise = philox_normal(u1, u2);
            float action = mean + std * noise;

            float normalized = (action - mean) / std;
            float log_prob = -0.5f * normalized * normalized - 0.5f * LOG_2PI - log_std;

            actions[(int)idx * sp.num_atns + h] = action;
            total_log_prob += log_prob;
        }
    } else {
        // Discrete action sampling (multinomial)
        // CUDA joint-ratio: accumulate scalar total_log_prob across heads
        int logits_offset = 0;

        for (int h = 0; h < sp.num_atns; h++) {
            int A = act_sizes[h];

            // Mask base index for this env (mask_stride allows non-contiguous layout)
            int mask_base = (int)idx * sp.mask_stride;

            // Max + logsumexp (with mask)
            float max_val = -INFINITY;
            for (int a = 0; a < A; a++) {
                float l = masked_logit(logits[logits_base + logits_offset + a],
                                       action_mask[mask_base + logits_offset + a]);
                max_val = fmax(max_val, l);
            }
            float sum_exp = 0.0f;
            for (int a = 0; a < A; a++) {
                float l = masked_logit(logits[logits_base + logits_offset + a],
                                       action_mask[mask_base + logits_offset + a]);
                sum_exp += exp(l - max_val);
            }
            float logsumexp_val = max_val + log(sum_exp);

            // Random sample
            float rand_val = philox_uniform(rng_idx, rng_out);
            if (rng_idx >= 4) {
                counter.z++;
                rng_out = philox4x32_10(counter, key);
                rng_idx = 0;
            }

            // Inverse CDF sampling (with mask)
            float cumsum = 0.0f;
            int sampled_action = A - 1;
            for (int a = 0; a < A; a++) {
                float l = masked_logit(logits[logits_base + logits_offset + a],
                                       action_mask[mask_base + logits_offset + a]);
                float prob = exp(l - logsumexp_val);
                cumsum += prob;
                if (rand_val < cumsum) {
                    sampled_action = a;
                    break;
                }
            }

            float log_prob = masked_logit(
                logits[logits_base + logits_offset + sampled_action],
                action_mask[mask_base + logits_offset + sampled_action]) - logsumexp_val;

            actions[(int)idx * sp.num_atns + h] = float(sampled_action);
            total_log_prob += log_prob;

            logits_offset += A;
        }
    }
    // Scalar joint logprob (matches CUDA kernels.cu:995)
    logprobs[(int)idx] = total_log_prob;
    value_out[idx] = value[(int)idx * sp.value_stride];
}

//
// Used when CPU inference produces actions but logprobs need GPU-precision
// exp/log to match PPO training kernels. Without this, the importance ratio
// exp(new_logp - old_logp) sees a systematic IEEE vs fast::exp bias that
// causes NaN with overlap (stale weights amplify the mismatch).

struct RecomputeLogprobsParams {
    int B;
    int num_atns;
    int logits_stride;   // fused_cols per row
    int mask_stride;     // 0 = broadcast (all-ones)
};

kernel void recompute_logprobs_kernel(
    device float* logprobs              [[buffer(0)]],
    const device float* logits          [[buffer(1)]],
    const device float* actions_f32     [[buffer(2)]],
    const device int* act_sizes         [[buffer(3)]],
    const device float* action_mask     [[buffer(4)]],
    constant RecomputeLogprobsParams& rp [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx >= rp.B) return;

    int logits_base = (int)idx * rp.logits_stride;
    int mask_base = (rp.mask_stride == 0) ? 0 : (int)idx * rp.mask_stride;

    float total_log_prob = 0.0f;
    int logits_offset = 0;

    for (int h = 0; h < rp.num_atns; h++) {
        int A = act_sizes[h];
        int act = int(actions_f32[(int)idx * rp.num_atns + h]);

        // Max + logsumexp (with mask)
        float max_val = -INFINITY;
        for (int a = 0; a < A; a++) {
            max_val = fmax(max_val, masked_logit(
                logits[logits_base + logits_offset + a],
                action_mask[mask_base + logits_offset + a]));
        }
        float sum_exp = 0.0f;
        for (int a = 0; a < A; a++) {
            sum_exp += exp(masked_logit(
                logits[logits_base + logits_offset + a],
                action_mask[mask_base + logits_offset + a]) - max_val);
        }
        float logsumexp_val = max_val + log(sum_exp);

        float head_lp = masked_logit(
            logits[logits_base + logits_offset + act],
            action_mask[mask_base + logits_offset + act]) - logsumexp_val;
        total_log_prob += head_lp;

        logits_offset += A;
    }
    // Scalar joint logprob (matches CUDA kernels.cu:995)
    logprobs[(int)idx] = total_log_prob;
}

constant int PPO_THREADS = 256;
constant int LOSS_PG = 0;
constant int LOSS_VF = 1;
constant int LOSS_ENT = 2;
constant int LOSS_TOTAL = 3;
constant int LOSS_OLD_APPROX_KL = 4;
constant int LOSS_APPROX_KL = 5;
constant int LOSS_CLIPFRAC = 6;
constant int LOSS_N = 7;
constant int MAX_ATN_HEADS = 16;

// Float atomic add via CAS loop (MSL has no native atomic<float>)
inline void atomic_add_float(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        float current = as_type<float>(expected);
        float desired = current + val;
        uint desired_bits = as_type<uint>(desired);
        if (atomic_compare_exchange_weak_explicit(addr, &expected, desired_bits,
                memory_order_relaxed, memory_order_relaxed)) {
            return;
        }
    }
}

// PPO helper: compute logsumexp, entropy, log_prob for a single discrete head with masks.
// mask pointer + mask_offset index into the action mask for this head.
// Invalid actions (mask < 0.5) get logit = -1e9, matching rollout sampling.
inline void ppo_discrete_head(
    const device float* logits,
    int logits_base, int logits_stride_a, int logits_offset,
    int A, int act,
    const device float* mask, int mask_offset,
    thread float& out_logsumexp, thread float& out_entropy, thread float& out_logp
) {
    float max_logit = -INFINITY;
    float sum = 0.0f;
    float act_logit = 0.0f;

    for (int a = 0; a < A; a++) {
        float l = logits[logits_base + (logits_offset + a) * logits_stride_a];
        if (mask[mask_offset + a] < 0.5f) l = -1e9f;
        if (a == act) act_logit = l;
        if (l > max_logit) {
            sum *= exp(max_logit - l);
            max_logit = l;
        }
        sum += exp(l - max_logit);
    }
    // Degenerate input (all masked or non-finite model output): propagate NaN
    // so the corruption surfaces immediately in the PPO loss rather than
    // silently producing logp=0 (ratio=1) which poisons gradients.
    if (!isfinite(max_logit) || !isfinite(sum) || sum <= 0.0f) {
        out_logsumexp = NAN;
        out_entropy = NAN;
        out_logp = NAN;
        return;
    }
    float lse = max_logit + log(sum);

    float ent = 0.0f;
    for (int a = 0; a < A; a++) {
        float l = logits[logits_base + (logits_offset + a) * logits_stride_a];
        if (mask[mask_offset + a] < 0.5f) l = -1e9f;
        float logp = l - lse;
        float p = exp(clamp(logp, -80.0f, 80.0f));
        ent -= p * logp;
    }

    out_logsumexp = lse;
    out_entropy = ent;
    out_logp = act_logit - lse;
}

// PPO helper: compute log_prob and entropy for a single continuous head
inline void ppo_continuous_head(
    float mean, float log_std, float action,
    thread float& out_logp, thread float& out_entropy
) {
    constexpr float HALF_LOG_2PI = 0.9189385332046727f;
    constexpr float HALF_1_PLUS_LOG_2PI = 1.4189385332046727f;
    float std = exp(log_std);
    float normalized = (action - mean) / std;
    out_logp = -0.5f * normalized * normalized - HALF_LOG_2PI - log_std;
    out_entropy = HALF_1_PLUS_LOG_2PI + log_std;
}

struct PPOFusedParams {
    int num_atns;
    float clip_coef;
    float vf_clip_coef;
    float vf_coef;
    float ent_coef;
    int T_seq;
    int A_total;
    int N;
    int logits_stride_n;
    int logits_stride_t;
    int logits_stride_a;
    int values_stride_n;
    int values_stride_t;
    int is_continuous;
    int num_atns_total;  // sum of act_sizes, for mask buffer indexing
    int mask_stride;     // stride in floats between consecutive mask rows in obs
};

// Fused PPO forward + backward: computes loss partials AND gradients in one pass.
// Per-block partial sums written to ppo_partials, reduced by ppo_loss_reduce_kernel.
kernel void ppo_loss_fwd_bwd_kernel(
    device float* ppo_partials              [[buffer(0)]],
    device float* grad_logits               [[buffer(1)]],
    device float* grad_logstd               [[buffer(2)]],
    device float* grad_values_pred          [[buffer(3)]],
    const device float* logits              [[buffer(4)]],
    const device float* logstd              [[buffer(5)]],
    const device float* values_pred         [[buffer(6)]],
    const device float* actions             [[buffer(7)]],
    const device float* old_logprobs        [[buffer(8)]],
    const device float* advantages          [[buffer(9)]],
    const device float* prio                [[buffer(10)]],
    const device float* values              [[buffer(11)]],
    const device float* returns_buf         [[buffer(12)]],
    const device float* adv_mean            [[buffer(13)]],
    const device float* adv_var             [[buffer(14)]],
    const device int* act_sizes             [[buffer(15)]],
    constant PPOFusedParams& pp             [[buffer(16)]],
    const device float* action_mask         [[buffer(17)]],
    device float* out_ratio                 [[buffer(18)]],
    device float* out_newvalue              [[buffer(19)]],
    uint idx [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint block_id [[threadgroup_position_in_grid]]
) {
    int total_elements = pp.N * pp.T_seq;
    float inv_NT = 1.0f / float(total_elements);

    threadgroup float block_losses[7][PPO_THREADS];
    for (int c = 0; c < LOSS_N; c++) {
        block_losses[c][tid] = 0.0f;
    }

    // MSL has no goto — use if/else instead of the CUDA goto pattern
    if ((int)idx < total_elements) {
        int n = (int)idx / pp.T_seq;
        int t = (int)idx % pp.T_seq;
        int nt = n * pp.T_seq + t;

        int logits_base = n * pp.logits_stride_n + t * pp.logits_stride_t;
        int values_idx = n * pp.values_stride_n + t * pp.values_stride_t;
        int grad_logits_base = nt * pp.A_total;

        // Shared computation (used by both forward and backward)
        float adv = advantages[nt];
        float w = prio[n];
        float val = values[nt];
        float ret = returns_buf[nt];
        float val_pred = values_pred[values_idx];
        out_newvalue[nt] = val_pred;

        float adv_std = sqrt(adv_var[0]);
        float adv_normalized = (adv - adv_mean[0]) / (adv_std + 1e-8f);

        // grad_loss is always 1.0
        float dL = inv_NT;
        float d_pg_loss = dL;
        float d_entropy_term = dL * (-pp.ent_coef);

        // Value loss (forward) + value gradient (backward)
        float v_error = val_pred - val;
        float v_clipped = val + clamp(v_error, -pp.vf_clip_coef, pp.vf_clip_coef);
        float v_loss_unclipped = (val_pred - ret) * (val_pred - ret);
        float v_loss_clipped = (v_clipped - ret) * (v_clipped - ret);
        float v_loss = 0.5f * fmax(v_loss_unclipped, v_loss_clipped);

        float d_val_pred = 0.0f;
        if (v_loss_clipped > v_loss_unclipped) {
            if (v_error >= -pp.vf_clip_coef && v_error <= pp.vf_clip_coef) {
                d_val_pred = v_clipped - ret;
            }
        } else {
            d_val_pred = val_pred - ret;
        }
        grad_values_pred[nt] = dL * w * pp.vf_coef * d_val_pred;

        // Policy loss + gradients. Both branches produce pg_loss, total_entropy,
        // logratio, ratio — the block-loss accumulation is shared after the if/else.
        float pg_loss, total_entropy, logratio, ratio;

        if (pp.is_continuous) {
            // Continuous: old_logprobs is scalar (matches CUDA)
            float old_logp = old_logprobs[nt];
            float total_log_prob = 0.0f;
            total_entropy = 0.0f;
            for (int h = 0; h < pp.num_atns; h++) {
                float mean = logits[logits_base + h * pp.logits_stride_a];
                float log_std = logstd[logits_base + h * pp.logits_stride_a];
                float action = actions[nt * pp.num_atns + h];
                float lp, ent;
                ppo_continuous_head(mean, log_std, action, lp, ent);
                total_log_prob += lp;
                total_entropy += ent;
            }

            logratio = total_log_prob - old_logp;
            ratio = exp(logratio);
            out_ratio[nt] = ratio;
            float ratio_clipped = clamp(ratio, 1.0f - pp.clip_coef, 1.0f + pp.clip_coef);
            float wa = -w * adv_normalized;
            pg_loss = fmax(wa * ratio, wa * ratio_clipped);

            // Backward: policy gradient
            float d_ratio = wa * d_pg_loss;
            if (wa * ratio_clipped > wa * ratio) {
                if (ratio <= (1.0f - pp.clip_coef) || ratio >= (1.0f + pp.clip_coef))
                    d_ratio = 0.0f;
            }
            float d_new_logp = d_ratio * ratio;

            for (int h = 0; h < pp.num_atns; h++) {
                float mean = logits[logits_base + h * pp.logits_stride_a];
                float log_std = logstd[logits_base + h * pp.logits_stride_a];
                float std = exp(log_std);
                float var = std * std;
                float action = actions[nt * pp.num_atns + h];
                float diff = action - mean;

                grad_logits[grad_logits_base + h] = d_new_logp * diff / var;
                grad_logstd[grad_logits_base + h] = d_new_logp * (diff * diff / var - 1.0f) + d_entropy_term;
            }
        } else {
            // Discrete joint-ratio clipping (matches CUDA kernels.cu:738-807).
            // Sum per-head logprobs into scalar, compute single joint ratio, clip once.
            float head_logsumexp[MAX_ATN_HEADS];
            float head_entropy[MAX_ATN_HEADS];
            int head_act[MAX_ATN_HEADS];
            int mask_base = (int)idx * pp.mask_stride;

            int logits_offset = 0;
            float total_log_prob = 0.0f;
            total_entropy = 0.0f;

            for (int h = 0; h < pp.num_atns; h++) {
                int A = act_sizes[h];
                int act = int(actions[nt * pp.num_atns + h]);
                head_act[h] = act;
                float lse, ent, lp;
                ppo_discrete_head(logits, logits_base, pp.logits_stride_a, logits_offset,
                    A, act, action_mask, mask_base + logits_offset, lse, ent, lp);
                head_logsumexp[h] = lse;
                head_entropy[h] = ent;
                total_log_prob += lp;
                total_entropy += ent;
                logits_offset += A;
            }

            float old_logp = old_logprobs[nt];
            logratio = total_log_prob - old_logp;
            ratio = exp(logratio);
            out_ratio[nt] = ratio;
            float ratio_clipped = clamp(ratio, 1.0f - pp.clip_coef, 1.0f + pp.clip_coef);
            float wa = -w * adv_normalized;
            pg_loss = fmax(wa * ratio, wa * ratio_clipped);

            // Backward: single joint d_new_logp shared across all heads
            float d_ratio = wa * d_pg_loss;
            if (wa * ratio_clipped > wa * ratio) {
                if (ratio <= (1.0f - pp.clip_coef) || ratio >= (1.0f + pp.clip_coef))
                    d_ratio = 0.0f;
            }
            float d_new_logp = d_ratio * ratio;

            // Gradient pass over logits (reuses head_logsumexp, head_entropy)
            logits_offset = 0;
            for (int h = 0; h < pp.num_atns; h++) {
                int A = act_sizes[h];
                int act = head_act[h];
                float lse = head_logsumexp[h];
                float ent = head_entropy[h];

                for (int a = 0; a < A; a++) {
                    float raw_l = logits[logits_base + (logits_offset + a) * pp.logits_stride_a];
                    float m = action_mask[mask_base + logits_offset + a];
                    float l = (m < 0.5f) ? -1e9f : raw_l;
                    float logp = l - lse;
                    float p = exp(logp);
                    float d_logit = (a == act) ? d_new_logp : 0.0f;
                    d_logit -= p * d_new_logp;
                    d_logit += d_entropy_term * p * (-ent - logp);
                    if (m < 0.5f) d_logit = 0.0f;
                    grad_logits[grad_logits_base + logits_offset + a] = d_logit;
                }
                logits_offset += A;
            }
        }

        // Shared loss accumulation (both branches produce pg_loss, total_entropy, logratio, ratio)
        block_losses[LOSS_PG][tid] = pg_loss * inv_NT;
        block_losses[LOSS_VF][tid] = v_loss * inv_NT;
        block_losses[LOSS_ENT][tid] = total_entropy * inv_NT;
        block_losses[LOSS_TOTAL][tid] = (pg_loss + pp.vf_coef * v_loss - pp.ent_coef * total_entropy) * inv_NT;
        block_losses[LOSS_OLD_APPROX_KL][tid] = (-logratio) * inv_NT;
        block_losses[LOSS_APPROX_KL][tid] = ((ratio - 1.0f) - logratio) * inv_NT;
        block_losses[LOSS_CLIPFRAC][tid] = (abs(ratio - 1.0f) > pp.clip_coef ? 1.0f : 0.0f) * inv_NT;
    } // end if (idx < total_elements)

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Block reduction (tree reduction)
    for (int stride = PPO_THREADS / 2; stride > 0; stride >>= 1) {
        if ((int)tid < stride) {
            for (int c = 0; c < LOSS_N; c++) {
                block_losses[c][tid] += block_losses[c][tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        int base_out = (int)block_id * (LOSS_N + 1);
        ppo_partials[base_out] = block_losses[LOSS_TOTAL][0];
        for (int c = 0; c < LOSS_N; c++) {
            ppo_partials[base_out + 1 + c] = block_losses[c][0];
        }
    }
}

// Deterministic reduction of per-block PPO loss partials + count increment
struct PPOReduceParams {
    int num_blocks;
};

kernel void ppo_loss_reduce_kernel(
    device float* loss                      [[buffer(0)]],
    device float* losses_acc                [[buffer(1)]],
    const device float* partials            [[buffer(2)]],
    constant PPOReduceParams& pp            [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if ((int)tid > LOSS_N) return;

    float sum = 0.0f;
    for (int b = 0; b < pp.num_blocks; b++) {
        sum += partials[b * (LOSS_N + 1) + (int)tid];
    }

    if (tid == 0) {
        *loss += sum;
    } else {
        losses_acc[(int)tid - 1] += sum;
    }

    // Fold add_scalar: increment epoch count
    if (tid == 0) {
        losses_acc[LOSS_N] += 1.0f;
    }
}

struct AdvantageParams {
    float gamma;
    float lambda;
    float rho_clip;
    float c_clip;
    int num_steps;
    int horizon;
};

kernel void puff_advantage_kernel(
    const device float* values      [[buffer(0)]],
    const device float* rewards     [[buffer(1)]],
    const device float* dones       [[buffer(2)]],
    const device float* importance  [[buffer(3)]],
    device float* advantages        [[buffer(4)]],
    constant AdvantageParams& p     [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if ((int)row >= p.num_steps) return;

    int offset = (int)row * p.horizon;
    const device float* v = values + offset;
    const device float* r = rewards + offset;
    const device float* d = dones + offset;
    const device float* imp = importance + offset;
    device float* adv = advantages + offset;

    float lastpufferlam = 0.0f;
    for (int t = p.horizon - 2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0f - d[t_next];
        float rho_t = min(imp[t], p.rho_clip);
        float c_t = min(imp[t], p.c_clip);
        float delta = rho_t * (r[t_next] + p.gamma * v[t_next] * nextnonterminal - v[t]);
        lastpufferlam = delta + p.gamma * p.lambda * c_t * lastpufferlam * nextnonterminal;
        adv[t] = lastpufferlam;
    }
}

struct PrioParams {
    float prio_alpha;
    int stride;
};

kernel void prio_adv_reduction_kernel(
    const device float* advantages  [[buffer(0)]],
    device float* prio_weights      [[buffer(1)]],
    constant PrioParams& pp         [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tx [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    int offset = (int)row * pp.stride;

    float local_sum = 0.0f;
    for (int t = (int)tx; t < pp.stride; t += 32) {
        local_sum += abs(advantages[offset + t]);
    }

    // Simdgroup reduction
    local_sum = simd_sum(local_sum);

    if (simd_lane == 0 && tx < 32) {
        // epsilon floor prevents pow(0, alpha>0) = 0 which would permanently
        // exclude zero-advantage segments from sampling in sparse-reward envs
        float pw = pow(local_sum + 1e-6f, pp.prio_alpha);
        if (isnan(pw) || isinf(pw)) pw = 1e-6f;
        prio_weights[row] = pw;
    }
}

struct PrioNormParams {
    int length;
};

kernel void prio_normalize_kernel(
    device float* prio_weights          [[buffer(0)]],
    constant PrioNormParams& pp         [[buffer(1)]],
    uint tx [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    constexpr float eps = 1e-6f;
    constexpr int NUM_WARPS = 8;  // 256 / 32

    threadgroup float shmem[NUM_WARPS];
    threadgroup float block_sum;

    float local_sum = 0.0f;
    for (int t = (int)tx; t < pp.length; t += 256) {
        local_sum += prio_weights[t];
    }

    // Simdgroup reduction
    local_sum = simd_sum(local_sum);

    if (simd_lane == 0) shmem[simd_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float val = (simd_lane < NUM_WARPS) ? shmem[simd_lane] : 0.0f;
        val = simd_sum(val);
        // add eps * length so numerator and denominator are consistent:
        // sum((w_i + eps)) = sum(w_i) + eps*length
        if (tx == 0) block_sum = val + eps * float(pp.length);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = (int)tx; t < pp.length; t += 256) {
        prio_weights[t] = (prio_weights[t] + eps) / block_sum;
    }
}

struct PrioImpParams {
    int total_agents;
    float anneal_beta;
    int minibatch_segments;
};

struct PrioSampleParams {
    uint64_t seed;
    uint base_offset;
    int total_segments;
    int minibatch_segments;
};

kernel void prio_sample_kernel(
    device int64_t* indices            [[buffer(0)]],
    const device float* prio_probs     [[buffer(1)]],
    constant PrioSampleParams& pp      [[buffer(2)]],
    uint tx [[thread_position_in_grid]]
) {
    if ((int)tx >= pp.minibatch_segments) return;

    uint4 counter = uint4(pp.base_offset + tx, 0u, 0u, 0u);
    uint2 key = uint2((uint)(pp.seed & 0xFFFFFFFF), (uint)(pp.seed >> 32));
    uint4 rng_out = philox4x32_10(counter, key);
    uint rng_idx = 0;
    float u = philox_uniform(rng_idx, rng_out);

    float cumsum = 0.0f;
    int sampled = pp.total_segments - 1;
    for (int i = 0; i < pp.total_segments; i++) {
        cumsum += prio_probs[i];
        if (u <= cumsum) {
            sampled = i;
            break;
        }
    }
    indices[tx] = sampled;
}

kernel void prio_imp_weights_kernel(
    const device int64_t* indices       [[buffer(0)]],
    const device float* prio_probs      [[buffer(1)]],
    device float* mb_prio               [[buffer(2)]],
    constant PrioImpParams& pp          [[buffer(3)]],
    uint tx [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    // Compute raw importance weights
    float w = 1.0f;
    if ((int)tx < pp.minibatch_segments) {
        float value = prio_probs[indices[tx]] * float(pp.total_agents);
        w = pow(value, -pp.anneal_beta);
    }

    // Max-normalize: w_i /= max(w_j) so all weights are in [0, 1].
    // Prevents gradient amplification from undersampled segments.
    constexpr int NUM_WARPS = 8;
    threadgroup float shmem[NUM_WARPS];
    float local_max = ((int)tx < pp.minibatch_segments) ? w : 0.0f;
    local_max = simd_max(local_max);
    if (simd_lane == 0) shmem[simd_id] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        float val = (simd_lane < NUM_WARPS) ? shmem[simd_lane] : 0.0f;
        val = simd_max(val);
        if (simd_lane == 0) shmem[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float w_max = max(shmem[0], 1e-8f);

    if ((int)tx < pp.minibatch_segments) {
        mb_prio[tx] = w / w_max;
    }
}

struct FillParams {
    float val;
    int n;
};

kernel void fill_f32(
    device float* dst           [[buffer(0)]],
    constant FillParams& p      [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx < p.n) dst[idx] = p.val;
}

kernel void copy_f32(
    device float* dst           [[buffer(0)]],
    device const float* src     [[buffer(1)]],
    constant int& n             [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx < n) dst[idx] = src[idx];
}

struct ClampParams {
    float lo;
    float hi;
    int n;
};

kernel void clamp_f32(
    device float* dst           [[buffer(0)]],
    constant ClampParams& p     [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx < p.n) dst[idx] = clamp(dst[idx], p.lo, p.hi);
}

struct ScaleParams {
    float alpha;
    int n;
};

kernel void scale_f32(
    device float* dst           [[buffer(0)]],
    constant ScaleParams& p     [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx < p.n) dst[idx] *= p.alpha;
}

struct AxpyParams {
    float alpha;
    int n;
};

kernel void axpy_f32(
    device float* dst               [[buffer(0)]],
    const device float* src         [[buffer(1)]],
    constant AxpyParams& p          [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx < p.n) dst[idx] += p.alpha * src[idx];
}

struct AddParams {
    int n;
};

kernel void add_f32(
    device float* dst               [[buffer(0)]],
    const device float* src         [[buffer(1)]],
    constant AddParams& p           [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx < p.n) dst[idx] += src[idx];
}

kernel void add_f16(
    device half* dst                [[buffer(0)]],
    const device half* src          [[buffer(1)]],
    constant AddParams& p           [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx < p.n) dst[idx] = half(float(dst[idx]) + float(src[idx]));
}

struct NesterovParams {
    float mu;
    int n;
};

// Fused Nesterov momentum: mb = mu*mb + gc; gc = gc + mu*mb (note: gc uses updated mb)
kernel void nesterov_f32(
    device float* mb                [[buffer(0)]],
    device float* gc                [[buffer(1)]],
    constant NesterovParams& p      [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx < p.n) {
        float m = p.mu * mb[idx] + gc[idx];
        mb[idx] = m;
        gc[idx] += p.mu * m;
    }
}

struct ScaleDevParams {
    int n;
};

// Scale by device-side scalar: dst[i] *= *alpha_ptr
kernel void scale_f32_dev(
    device float* dst                   [[buffer(0)]],
    const device float* alpha_ptr       [[buffer(1)]],
    constant ScaleDevParams& p          [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    float alpha = *alpha_ptr;
    if ((int)idx < p.n) dst[idx] *= alpha;
}

struct AxpyDevParams {
    int n;
};

// dst += (*alpha) * src
kernel void axpy_f32_dev(
    device float* dst                   [[buffer(0)]],
    const device float* src             [[buffer(1)]],
    const device float* alpha_ptr       [[buffer(2)]],
    constant AxpyDevParams& p           [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    float alpha = *alpha_ptr;
    if ((int)idx < p.n) dst[idx] += alpha * src[idx];
}

struct AddScalarParams {
    float val;
};

// *ptr += val (single element)
kernel void add_scalar(
    device float* ptr               [[buffer(0)]],
    constant AddScalarParams& p     [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx == 0) *ptr += p.val;
}

// Reads LR from device, computes neg_lr = -lr
kernel void compute_lr_scalars_kernel(
    const device float* lr          [[buffer(0)]],
    device float* neg_lr            [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx == 0) {
        *neg_lr = -(*lr);
    }
}

struct MuonParams {
    int n;
    float scale;
};

// Weight update: wb = wb * (1 - lr * wd) - lr * scale * up  (matches CUDA muon.cu)
kernel void muon_weight_update_kernel(
    device float* wb                    [[buffer(0)]],
    const device float* up              [[buffer(1)]],
    const device float* lr_ptr          [[buffer(2)]],
    constant MuonParams& p              [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx >= p.n) return;
    float lr = *lr_ptr;
    wb[idx] = wb[idx] - lr * p.scale * up[idx];
}

struct TransposeParams {
    int R;
    int C;
};

// Transpose R x C matrix: dst[c*R + r] = src[r*C + c]
kernel void transpose_f32(
    device float* dst               [[buffer(0)]],
    const device float* src         [[buffer(1)]],
    constant TransposeParams& p     [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx >= p.R * p.C) return;
    dst[(idx % p.C) * p.R + idx / p.C] = src[idx];
}

struct Transpose01Params {
    int A;
    int B;
    int C;
};

// Transpose dims 0 and 1 of (A, B, C) tensor: dst[b*A*C + a*C + c] = src[a*B*C + b*C + c]
kernel void transpose_01(
    device float* dst               [[buffer(0)]],
    const device float* src         [[buffer(1)]],
    constant Transpose01Params& p   [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    int total = p.A * p.B * p.C;
    if ((int)idx >= total) return;
    int a = (int)idx / (p.B * p.C);
    int rem = (int)idx % (p.B * p.C);
    int b = rem / p.C;
    int c = rem % p.C;
    dst[b * p.A * p.C + a * p.C + c] = src[idx];
}

// Transpose dims 0 and 1 for 8-byte elements (f64/u64), using uint2 pairs.
// Metal has no native double support, so we treat each 8-byte element as uint2.
// Same index math as transpose_01 — just operates on uint2 instead of float.
kernel void transpose_01_u64(
    device uint2* dst                   [[buffer(0)]],
    const device uint2* src             [[buffer(1)]],
    constant Transpose01Params& p       [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    int total = p.A * p.B * p.C;
    if ((int)idx >= total) return;
    int a = (int)idx / (p.B * p.C);
    int rem = (int)idx % (p.B * p.C);
    int b = rem / p.C;
    int c = rem % p.C;
    dst[b * p.A * p.C + a * p.C + c] = src[idx];
}

struct NormParams {
    int n;
};

// Per-block sum of squares (partial reduction)
kernel void norm_f32_kernel(
    device float* partials              [[buffer(0)]],
    const device float* src             [[buffer(1)]],
    constant NormParams& p              [[buffer(2)]],
    uint idx [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint block_id [[threadgroup_position_in_grid]],
    uint grid_size [[threads_per_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    constexpr int NUM_WARPS = 8;  // 256 / 32
    threadgroup float sdata[NUM_WARPS];
    float sum = 0.0f;
    for (int i = (int)idx; i < p.n; i += (int)grid_size) {
        sum += src[i] * src[i];
    }
    sum = simd_sum(sum);
    if (simd_lane == 0) sdata[simd_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        sum = (simd_lane < NUM_WARPS) ? sdata[simd_lane] : 0.0f;
        sum = simd_sum(sum);
        if (simd_lane == 0) partials[block_id] = sum;
    }
}

struct NormReduceParams {
    int num_blocks;
};

// Reduce per-block partials to a single sum-of-squares value
kernel void norm_reduce_kernel(
    device float* out                   [[buffer(0)]],
    const device float* partials        [[buffer(1)]],
    constant NormReduceParams& p        [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    constexpr int NUM_WARPS = 8;
    threadgroup float sdata[NUM_WARPS];
    float val = ((int)tid < p.num_blocks) ? partials[tid] : 0.0f;
    val = simd_sum(val);
    if (simd_lane == 0) sdata[simd_id] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        val = (simd_lane < NUM_WARPS) ? sdata[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) *out = val;
    }
}

struct ClipByNormParams {
    float max_norm;
    float eps;
    int n;
};

// Clip gradient by global norm: dst[i] *= min(max_norm / (sqrt(sum_sq) + eps), 1.0)
// When clip_coef is 0 (norm overflow), zero directly to avoid inf*0=NaN.
kernel void clip_by_norm_f32(
    device float* dst                       [[buffer(0)]],
    const device float* sum_sq_ptr          [[buffer(1)]],
    constant ClipByNormParams& p            [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    float clip_coef = min(p.max_norm / (sqrt(*sum_sq_ptr) + p.eps), 1.0f);
    if ((int)idx < p.n) {
        dst[idx] = (clip_coef > 0.0f) ? dst[idx] * clip_coef : 0.0f;
    }
}

struct NormalizeParams {
    float eps;
    int n;
};

// dst[i] /= max(sqrt(*norm), eps) — matches CUDA (no cap)
kernel void normalize_f32(
    device float* dst                       [[buffer(0)]],
    const device float* norm_ptr            [[buffer(1)]],
    constant NormalizeParams& p             [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    float inv_norm = 1.0f / max(sqrt(*norm_ptr), p.eps);
    if ((int)idx < p.n) dst[idx] = dst[idx] * inv_norm;
}

struct VarMeanParams {
    int n;
};

// Compute variance and mean of a float array (single threadgroup)
kernel void var_mean_kernel(
    const device float* src         [[buffer(0)]],
    device float* var_out           [[buffer(1)]],
    device float* mean_out          [[buffer(2)]],
    constant VarMeanParams& p       [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    constexpr int NUM_WARPS = 8;
    threadgroup float sdata[NUM_WARPS];

    // Pass 1: compute mean
    float sum = 0.0f;
    for (int i = (int)tid; i < p.n; i += 256) sum += src[i];
    sum = simd_sum(sum);
    if (simd_lane == 0) sdata[simd_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        sum = (simd_lane < NUM_WARPS) ? sdata[simd_lane] : 0.0f;
        sum = simd_sum(sum);
        if (simd_lane == 0) sdata[0] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = sdata[0] / float(p.n);
    if (tid == 0) *mean_out = mean;

    // Pass 2: compute variance
    float ss = 0.0f;
    for (int i = (int)tid; i < p.n; i += 256) {
        float d = src[i] - mean;
        ss += d * d;
    }
    ss = simd_sum(ss);
    if (simd_lane == 0) sdata[simd_id] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        ss = (simd_lane < NUM_WARPS) ? sdata[simd_lane] : 0.0f;
        ss = simd_sum(ss);
        if (simd_lane == 0) sdata[0] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) *var_out = sdata[0] / float(p.n - 1);
}

struct SumRowsParams {
    int R;
    int C;
};

// dst[c] = sum over rows of src[:, c]
kernel void sum_rows_to_f32_kernel(
    device float* dst               [[buffer(0)]],
    const device float* src         [[buffer(1)]],
    constant SumRowsParams& p       [[buffer(2)]],
    uint col [[thread_position_in_grid]]
) {
    if ((int)col >= p.C) return;
    float sum = 0.0f;
    for (int r = 0; r < p.R; r++) sum += src[r * p.C + (int)col];
    dst[col] = sum;
}

// --- Sum rows fp16 (for bias/LN param grads) ---

kernel void sum_rows_f16_kernel(
    device half* dst                [[buffer(0)]],
    const device half* src          [[buffer(1)]],
    constant SumRowsParams& p       [[buffer(2)]],
    uint col [[thread_position_in_grid]]
) {
    if ((int)col >= p.C) return;
    float sum = 0.0f;
    for (int r = 0; r < p.R; r++) sum += float(src[r * p.C + (int)col]);
    dst[col] = half(sum);
}

struct AssembleDecoderGradParams {
    int B_TT;
    int od;
    int od_plus_1;
};

// Assemble gradient: dst = [grad_logits | grad_value] in fused layout
kernel void assemble_decoder_grad_f32(
    device float* dst                       [[buffer(0)]],
    const device float* grad_logits         [[buffer(1)]],
    const device float* grad_value          [[buffer(2)]],
    constant AssembleDecoderGradParams& p   [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx >= p.B_TT * p.od_plus_1) return;
    int row = (int)idx / p.od_plus_1;
    int col = (int)idx % p.od_plus_1;
    dst[idx] = (col < p.od) ? grad_logits[row * p.od + col] : grad_value[row];
}

struct SelectCopyParams {
    int obs_row_bytes;
    int act_row_bytes;
    int lp_row_bytes;
    int horizon;
};

// Minibatch assembly: copy observations, actions, logprobs, values+advantages+returns, prio
// Channel 0 fuses obs gather + f32→f16 cast: reads f32 src, writes f16 directly to fp16_obs_out.
// Dispatched as (minibatch_size, 5) threadgroups, each handles one channel for one row.
kernel void select_copy_kernel(
    device char* mb_obs                 [[buffer(0)]],
    device char* mb_actions             [[buffer(1)]],
    device char* mb_logprobs            [[buffer(2)]],
    device float* mb_values             [[buffer(3)]],
    device float* mb_advantages         [[buffer(4)]],
    device float* mb_returns            [[buffer(5)]],
    device float* mb_prio_out           [[buffer(6)]],
    const device char* src_obs          [[buffer(7)]],
    const device char* src_actions      [[buffer(8)]],
    const device char* src_logprobs     [[buffer(9)]],
    const device float* src_values      [[buffer(10)]],
    const device float* advantages      [[buffer(11)]],
    const device int64_t* idx           [[buffer(12)]],
    const device float* mb_prio_in      [[buffer(13)]],
    constant SelectCopyParams& p        [[buffer(14)]],
    device half* fp16_obs_out           [[buffer(15)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    int mb = (int)group_id.x;
    int ch = (int)group_id.y;
    int src_row = (int)idx[mb];

    if (ch == 0) {
        // Fused obs gather + f32→f16 cast: copy f32 to mb_obs AND write f16 directly.
        // mb_obs f32 copy is needed because PPO reads embedded action masks from it.
        const device float* sptr = (const device float*)(src_obs + (int64_t)src_row * p.obs_row_bytes);
        int count = p.obs_row_bytes / 4;  // number of floats
        device float* f32ptr = (device float*)(mb_obs + (int64_t)mb * p.obs_row_bytes);
        device half* f16ptr = fp16_obs_out + (int64_t)mb * count;
        for (int i = (int)tid; i < count; i += 256) {
            float val = sptr[i];
            f32ptr[i] = val;
            f16ptr[i] = half(val);
        }
    } else if (ch == 1) {
        // Copy actions
        const device int* sptr = (const device int*)(src_actions + (int64_t)src_row * p.act_row_bytes);
        device int* dptr = (device int*)(mb_actions + (int64_t)mb * p.act_row_bytes);
        for (int i = (int)tid; i < p.act_row_bytes / 4; i += 256) dptr[i] = sptr[i];
    } else if (ch == 2) {
        // Copy logprobs
        const device int* sptr = (const device int*)(src_logprobs + (int64_t)src_row * p.lp_row_bytes);
        device int* dptr = (device int*)(mb_logprobs + (int64_t)mb * p.lp_row_bytes);
        for (int i = (int)tid; i < p.lp_row_bytes / 4; i += 256) dptr[i] = sptr[i];
    } else if (ch == 3) {
        // Copy values + advantages, compute returns = values + advantages
        int srh = src_row * p.horizon;
        int drh = mb * p.horizon;
        for (int i = (int)tid; i < p.horizon; i += 256) {
            float val = src_values[srh + i];
            float adv = advantages[srh + i];
            mb_values[drh + i] = val;
            mb_advantages[drh + i] = adv;
            mb_returns[drh + i] = val + adv;
        }
    } else if (ch == 4) {
        // Copy prio weight
        if (tid == 0) {
            mb_prio_out[mb] = mb_prio_in[mb];
        }
    }
}

struct IndexCopyParams {
    int num_idx;
    int row_bytes;
};

// Indexed copy: for each i, copy src row i to dst row idx[i]
kernel void index_copy_kernel(
    device char* dst                    [[buffer(0)]],
    const device int64_t* idx           [[buffer(1)]],
    const device char* src              [[buffer(2)]],
    constant IndexCopyParams& p         [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if ((int)i >= p.num_idx) return;
    int64_t dst_row = idx[i];
    const device char* s = src + (int64_t)i * p.row_bytes;
    device char* d = dst + dst_row * p.row_bytes;
    // Copy as 4-byte words, then handle remainder bytes
    int words = p.row_bytes / 4;
    const device uint* s4 = (const device uint*)s;
    device uint* d4 = (device uint*)d;
    for (int b = 0; b < words; b++) d4[b] = s4[b];
    for (int b = words * 4; b < p.row_bytes; b++) d[b] = s[b];
}

// index_gather_kernel: dst[i] = src[idx[i]] (gather, inverse of index_copy)
kernel void index_gather_kernel(
    device char* dst                    [[buffer(0)]],
    const device int64_t* idx           [[buffer(1)]],
    const device char* src              [[buffer(2)]],
    constant IndexCopyParams& p         [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if ((int)i >= p.num_idx) return;
    int64_t src_row = idx[i];
    const device char* s = src + src_row * p.row_bytes;
    device char* d = dst + (int64_t)i * p.row_bytes;
    int words = p.row_bytes / 4;
    const device uint* s4 = (const device uint*)s;
    device uint* d4 = (device uint*)d;
    for (int b = 0; b < words; b++) d4[b] = s4[b];
    for (int b = words * 4; b < p.row_bytes; b++) d[b] = s[b];
}

struct CastU8Params {
    int n;
};

kernel void cast_u8_to_f32(
    device float* dst                       [[buffer(0)]],
    const device uchar* src                 [[buffer(1)]],
    constant CastU8Params& p                [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx < p.n) dst[idx] = float(src[idx]);
}

// IEEE 754 f64→f32 bit cast. Metal has no native double, so we read each
// 8-byte double as uint2 and extract
// sign, exponent, mantissa via bit manipulation. Subnormals flush to zero.
kernel void cast_f64_to_f32(
    device const uint2* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant int& count     [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= count) return;
    uint2 bits = src[gid];
    uint hi = bits.y, lo = bits.x;
    uint sign = hi >> 31;
    int biased_exp = int((hi >> 20) & 0x7FFu);
    int exp_f32 = biased_exp - 1023 + 127;
    // Top 23 bits of the 52-bit mantissa: 20 from hi + 3 from lo
    uint mantissa = ((hi & 0xFFFFFu) << 3) | (lo >> 29);
    uint result;
    if (biased_exp == 0)          result = sign << 31;          // zero / subnormal → ±0
    else if (biased_exp == 0x7FF) result = (sign << 31) | 0x7F800000u; // inf/nan → ±inf
    else if (exp_f32 <= 0)        result = sign << 31;          // underflow → ±0
    else if (exp_f32 >= 255)      result = (sign << 31) | 0x7F800000u; // overflow → ±inf
    else                          result = (sign << 31) | (uint(exp_f32) << 23) | mantissa;
    dst[gid] = as_type<float>(result);
}

// ============================================================================
// Section 20: Tiled GEMM — C = alpha * op(A) @ op(B) + beta * C
//
// 64x64 tiled simdgroup_matrix GEMM for f32. Supports NT, NN, TN layouts
// via trans_a/trans_b parameters. Runs on the compute encoder.
// ============================================================================

struct GemmParams {
    int M;       // result rows
    int N;       // result columns
    int K;       // inner dimension
    int lda;     // leading dimension of A (physical columns)
    int ldb;     // leading dimension of B (physical columns)
    int ldc;     // leading dimension of C (= N)
    float alpha;
    float beta;
    int trans_a; // 0 = no transpose, 1 = transpose
    int trans_b;
};

constant int BM = 32;   // tile rows (shared by ksplit and fp16 register GEMMs)
constant int BN = 32;   // tile cols
constant int BK = 16;   // tile K dimension
constant int TM = 4;    // per-thread tile rows
constant int TN = 4;    // per-thread tile cols

// ============================================================================
// Section 22: K-split GEMM — for tall-K backward weight-gradient GEMMs
//
// When M and N are small but K is large (e.g. M=128, N=128, K=32768),
// regular sgemm_reg has too few threadgroups (M/32 * N/32 = 16). K-split
// partitions K across multiple TGs in the Z axis, writing partial sums to
// a scratch buffer. A second kernel reduces the partials into C.
// ============================================================================

kernel void sgemm_ksplit(
    device const float* A          [[buffer(0)]],
    device const float* B          [[buffer(1)]],
    device float* partials         [[buffer(2)]],
    constant GemmParams& p         [[buffer(3)]],
    constant int& k_per_split      [[buffer(4)]],
    uint3 group_id    [[threadgroup_position_in_grid]],
    uint3 local_id    [[thread_position_in_threadgroup]]
) {
    threadgroup float sA[BM][BK];
    threadgroup float sB[BK][BN];

    int trow = (int)local_id.y;
    int tcol = (int)local_id.x;
    int tid = trow * (BN / TN) + tcol;

    float acc[TM][TN];
    for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++)
            acc[m][n] = 0.0f;

    int k_start = (int)group_id.z * k_per_split;
    int k_end = min(k_start + k_per_split, p.K);
    int num_k_tiles = (k_end - k_start + BK - 1) / BK;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k_base = k_start + kt * BK;

        for (int i = 0; i < (BM * BK) / 64; i++) {
            int idx = tid + i * 64;
            int r = idx / BK;
            int c = idx % BK;
            int gr = (int)group_id.y * BM + r;
            int gc = k_base + c;
            sA[r][c] = (gr < p.M && gc < k_end)
                ? (p.trans_a ? A[gc * p.lda + gr] : A[gr * p.lda + gc])
                : 0.0f;
        }

        for (int i = 0; i < (BK * BN) / 64; i++) {
            int idx = tid + i * 64;
            int r = idx / BN;
            int c = idx % BN;
            int gr = k_base + r;
            int gc = (int)group_id.x * BN + c;
            sB[r][c] = (gr < k_end && gc < p.N)
                ? (p.trans_b ? B[gc * p.ldb + gr] : B[gr * p.ldb + gc])
                : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int k = 0; k < BK; k++) {
            float a_reg[TM];
            float b_reg[TN];
            for (int m = 0; m < TM; m++) a_reg[m] = sA[trow * TM + m][k];
            for (int n = 0; n < TN; n++) b_reg[n] = sB[k][tcol * TN + n];
            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n++)
                    acc[m][n] += a_reg[m] * b_reg[n];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partial results: partials[split_idx * M * N + row * N + col]
    int split_offset = (int)group_id.z * p.M * p.N;
    int row_base = (int)group_id.y * BM + trow * TM;
    int col_base = (int)group_id.x * BN + tcol * TN;
    for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n++) {
            int r = row_base + m;
            int c = col_base + n;
            if (r < p.M && c < p.N) {
                partials[split_offset + r * p.N + c] = acc[m][n];
            }
        }
    }
}

struct ReduceKsplitParams {
    int MN;
    int num_splits;
    float alpha;
    float beta;
};

kernel void reduce_ksplit(
    device const float* partials   [[buffer(0)]],
    device float* C                [[buffer(1)]],
    constant ReduceKsplitParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= p.MN) return;
    float sum = 0.0f;
    for (int s = 0; s < p.num_splits; s++) {
        sum += partials[s * p.MN + (int)gid];
    }
    C[gid] = p.alpha * sum + p.beta * C[gid];
}

// Reads fp32 grad_logits and grad_value (from PPO kernel), writes fp16 grad_out.
// grad_out[row * od1 + col] = (col < od) ? grad_logits[row * od + col] : grad_value[row]
kernel void assemble_decoder_grad_f32_to_f16(
    device half* grad_out                   [[buffer(0)]],
    const device float* grad_logits         [[buffer(1)]],
    const device float* grad_value          [[buffer(2)]],
    constant AssembleDecoderGradParams& p   [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= p.B_TT * p.od_plus_1) return;
    int row = (int)gid / p.od_plus_1;
    int col = (int)gid % p.od_plus_1;
    float val = (col < p.od) ? grad_logits[row * p.od + col] : grad_value[row];
    // Clamp to fp16 range to prevent inf (Metal fp16 max ~65504, unlike CUDA bf16)
    grad_out[gid] = half(clamp(val, -65000.0f, 65000.0f));
}

kernel void cast_f32_to_f16(
    device half* dst            [[buffer(0)]],
    const device float* src     [[buffer(1)]],
    constant int& count         [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= count) return;
    dst[gid] = half(src[gid]);
}

kernel void cast_f16_to_f32(
    device float* dst           [[buffer(0)]],
    const device half* src      [[buffer(1)]],
    constant int& count         [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= count) return;
    dst[gid] = float(src[gid]);
}

kernel void mingru_scan_forward_checkpointed_fp16(
    device half* out                [[buffer(0)]],
    device half* next_state         [[buffer(1)]],
    device float* a_star_buf        [[buffer(2)]],
    device float* s_buf             [[buffer(3)]],
    device float* log_values_buf    [[buffer(4)]],
    const device half* combined     [[buffer(5)]],
    const device half* state        [[buffer(6)]],
    const device half* input        [[buffer(7)]],
    constant ScanParams& p          [[buffer(8)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx >= p.B * p.H) return;

    int b = (int)idx / p.H;
    int h = (int)idx % p.H;
    int bH = b * p.H;
    int H3 = 3 * p.H;
    int bHT = bH * p.T_seq;
    int out_base = bHT + h;
    int cbase = 3 * bHT;

    float a_star = 0.0f;
    float log_value = 0.0f;
    float s = log(float(state[bH + h]));
    log_value = s;

    int T_out = p.T_seq + 1;
    int buf_base = b * T_out * p.H + h;
    int buf_curr = buf_base;
    a_star_buf[buf_curr] = a_star;
    s_buf[buf_curr] = s;
    log_values_buf[buf_curr] = log_value;

    int out_curr = out_base;
    int t_offset = 0;

    for (int t = 1; t < p.T_seq + 1; t++) {
        float hidden_val = float(combined[cbase + h + t_offset]);
        float gate_val = float(combined[cbase + p.H + h + t_offset]);
        float proj_val = float(combined[cbase + 2 * p.H + h + t_offset]);
        float x_val = float(input[out_base + (t - 1) * p.H]);

        float log_coeff_val;
        log_coeffs_and_values_fwd(gate_val, hidden_val, log_coeff_val, log_value);

        a_star += log_coeff_val;

        float z = log_value - a_star;
        float max_val = fmax(s, z);
        s = max_val + log1p_f(exp(-abs(s - z)));

        float scan_result = exp(a_star + s);
        float proj_sigmoid = sigmoid_f(proj_val);
        float out_val = proj_sigmoid * scan_result + (1.0f - proj_sigmoid) * x_val;

        out[out_curr] = half(clamp(out_val, -65000.0f, 65000.0f));

        buf_curr += p.H;
        out_curr += p.H;
        t_offset += H3;

        if (t % CHECKPOINT_INTERVAL == 0) {
            a_star_buf[buf_curr] = a_star;
            s_buf[buf_curr] = s;
            log_values_buf[buf_curr] = log_value;
        }
    }

    float next_state_val = max(exp(a_star + s), 1e-30f);
    next_state[bH + h] = half(min(next_state_val, 65000.0f));
}

kernel void mingru_scan_backward_checkpointed_fp16(
    device half* grad_combined            [[buffer(0)]],
    device half* grad_state               [[buffer(1)]],
    device half* grad_input               [[buffer(2)]],
    const device half* grad_out           [[buffer(3)]],
    const device half* grad_next_state    [[buffer(4)]],
    const device half* combined           [[buffer(5)]],
    const device half* state              [[buffer(6)]],
    const device half* input              [[buffer(7)]],
    const device float* a_star_buf        [[buffer(8)]],
    const device float* s_buf             [[buffer(9)]],
    const device float* log_values_buf    [[buffer(10)]],
    constant ScanParams& p                [[buffer(11)]],
    uint idx [[thread_position_in_grid]]
) {
    if ((int)idx >= p.B * p.H) return;

    int b = (int)idx / p.H;
    int h = (int)idx % p.H;
    int bHT = b * p.H * p.T_seq;
    int cbase = 3 * bHT;
    int H3 = 3 * p.H;
    int state_idx = b * p.H + h;
    int out_base = bHT + h;

    int T_out = p.T_seq + 1;
    int buf_base = b * T_out * p.H + h;

    float acc = 0.0f;
    float s_val_next = 0.0f;
    float carry_grad_a = 0.0f;

    for (int chunk_end = p.T_seq; chunk_end > 0; chunk_end -= CHECKPOINT_INTERVAL) {
        int chunk_start = (chunk_end > CHECKPOINT_INTERVAL) ? (chunk_end - CHECKPOINT_INTERVAL) : 0;
        int chunk_len = chunk_end - chunk_start;

        float chunk_a_star[CHECKPOINT_INTERVAL];
        float chunk_s[CHECKPOINT_INTERVAL];
        float chunk_log_values[CHECKPOINT_INTERVAL];
        float chunk_hidden[CHECKPOINT_INTERVAL];
        float chunk_gate[CHECKPOINT_INTERVAL];

        int ckpt_buf_idx = buf_base + chunk_start * p.H;
        float recomp_a_star = a_star_buf[ckpt_buf_idx];
        float recomp_s = s_buf[ckpt_buf_idx];
        float recomp_log_value = log_values_buf[ckpt_buf_idx];

        for (int i = 0; i < chunk_len; i++) {
            int t = chunk_start + 1 + i;
            int t_offset = (t - 1) * H3;
            float hv = float(combined[cbase + h + t_offset]);
            float gv = float(combined[cbase + p.H + h + t_offset]);

            float lc;
            log_coeffs_and_values_fwd(gv, hv, lc, recomp_log_value);
            recomp_a_star += lc;

            float z = recomp_log_value - recomp_a_star;
            float mv = fmax(recomp_s, z);
            recomp_s = mv + log1p_f(exp(-abs(recomp_s - z)));

            chunk_a_star[i] = recomp_a_star;
            chunk_s[i] = recomp_s;
            chunk_log_values[i] = recomp_log_value;
            chunk_hidden[i] = hv;
            chunk_gate[i] = gv;
        }

        for (int i = chunk_len - 1; i >= 0; i--) {
            int t = chunk_start + 1 + i;
            int t_offset = (t - 1) * H3;

            float a_star_t = chunk_a_star[i];
            float s_t = chunk_s[i];
            float log_value_t = chunk_log_values[i];
            float hidden_val = chunk_hidden[i];
            float gate_val = chunk_gate[i];
            float proj_val = float(combined[cbase + 2 * p.H + h + t_offset]);
            int input_idx = out_base + (t - 1) * p.H;
            float x_val = float(input[input_idx]);

            float scan_result = exp(a_star_t + s_t);
            float z = log_value_t - a_star_t;

            float grad_out_val = float(grad_out[input_idx]);
            float grad_scan_from_next = (t == p.T_seq) ? float(grad_next_state[state_idx]) : 0.0f;

            float proj_sigmoid = sigmoid_f(proj_val);
            float grad_scan_result = grad_scan_from_next + grad_out_val * proj_sigmoid;
            float grad_proj = grad_out_val * (scan_result - x_val) * proj_sigmoid * (1.0f - proj_sigmoid);
            float grad_input_val = grad_out_val * (1.0f - proj_sigmoid);
            grad_input[input_idx] = half(clamp(grad_input_val, -65000.0f, 65000.0f));

            float grad_log_h = grad_scan_result * scan_result;
            float grad_s = grad_log_h;

            if (t == p.T_seq) {
                acc = grad_s;
            } else {
                acc = grad_s + acc * exp(s_t - s_val_next);
            }
            float grad_z = acc * exp(z - s_t);
            s_val_next = s_t;

            float grad_a = grad_log_h + carry_grad_a - grad_z;
            carry_grad_a = grad_a;

            float grad_g, grad_h;
            log_coeffs_and_values_bwd(grad_a, grad_z, gate_val, hidden_val, grad_g, grad_h);

            // Clamp to fp16 range to prevent inf (Metal fp16 max ~65504)
            grad_combined[cbase + h + t_offset] = half(clamp(grad_h, -65000.0f, 65000.0f));
            grad_combined[cbase + p.H + h + t_offset] = half(clamp(grad_g, -65000.0f, 65000.0f));
            grad_combined[cbase + 2 * p.H + h + t_offset] = half(clamp(grad_proj, -65000.0f, 65000.0f));
        }
    }

    int ckpt_0_idx = buf_base;
    float a_star_0 = a_star_buf[ckpt_0_idx];
    float s_0 = s_buf[ckpt_0_idx];
    float log_value_0 = log_values_buf[ckpt_0_idx];

    acc = acc * exp(s_0 - s_val_next);
    float grad_z_0 = acc * exp((log_value_0 - a_star_0) - s_0);

    float state_val = float(state[state_idx]);
    float grad_state_val = (state_val > 0.0f) ? (grad_z_0 / state_val) : 0.0f;
    grad_state[state_idx] = half(clamp(grad_state_val, -65000.0f, 65000.0f));
}

// ============================================================================
// Section 26: Steel GEMM — C = alpha * op(A) @ op(B) + beta * C
//
// MLX-inspired 64x64 tiled GEMM using simdgroup_matrix (Apple Silicon M3+).
// 4 simdgroups in 2x2 layout (128 threads), each computing 32x32 output
// via a 4x4 grid of 8x8 simdgroup_matrix multiply-accumulate operations.
//
// Hot loop uses direct device memory loads (ThunderMittens-validated:
// Apple Silicon's L2 cache provides effective data reuse without explicit
// threadgroup staging, eliminating barrier overhead). K-remainder and edge
// tile stores use threadgroup fallback.
//
// ============================================================================

kernel void steel_gemm(
    device const float* A      [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float* C            [[buffer(2)]],
    constant GemmParams& p     [[buffer(3)]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    uint sgid                  [[simdgroup_index_in_threadgroup]],
    uint lane                  [[thread_index_in_simdgroup]]
) {
    // 64x64 output tile, 4 simdgroups (2x2), each 32x32 = 4x4 grid of 8x8
    constexpr int BM = 64, BN = 64;
    constexpr int WM = 2, WN = 2;
    constexpr int TM = BM / (8 * WM); // 4
    constexpr int TN = BN / (8 * WN); // 4

    const int bm = (int)tgid.y * BM;
    const int bn = (int)tgid.x * BN;
    const int wm = (int)(sgid / WN);
    const int wn = (int)(sgid % WN);
    const int tid = (int)(sgid * 32 + lane);

    // Per-simdgroup 32x32 output starts at (sm, sn)
    const int sm = bm + wm * 32;
    const int sn = bn + wn * 32;

    // 4x4 accumulator grid of 8x8 simdgroup matrices (16 per simd group)
    simdgroup_float8x8 acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            acc[i][j] = simdgroup_float8x8(0);

    const int K_aligned = (p.K / 8) * 8;

    // ---- Main loop: direct device loads, no threadgroup, no barriers ----
    // Each simd group loads its own A and B fragments independently.
    // L2 cache provides cross-simdgroup data reuse (validated by
    // ThunderMittens: 9% faster than MLX's threadgroup approach on M2 Pro).

    if (!p.trans_a && p.trans_b) {
        // NT: C = A(M,K) @ B(N,K)^T — forward pass, Muon
        for (int k = 0; k < K_aligned; k += 8) {
            simdgroup_float8x8 a_frag[TM];
            for (int i = 0; i < TM; i++)
                simdgroup_load(a_frag[i], A + (long)(sm + i*8) * p.lda + k, p.lda);
            for (int j = 0; j < TN; j++) {
                simdgroup_float8x8 b_frag;
                simdgroup_load(b_frag, B + (long)(sn + j*8) * p.ldb + k, p.ldb,
                               ulong2(0,0), true);
                for (int i = 0; i < TM; i++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag, acc[i][j]);
            }
        }
    } else if (!p.trans_a && !p.trans_b) {
        // NN: C = A(M,K) @ B(K,N) — backward input grad, Muon addmm
        for (int k = 0; k < K_aligned; k += 8) {
            simdgroup_float8x8 a_frag[TM];
            for (int i = 0; i < TM; i++)
                simdgroup_load(a_frag[i], A + (long)(sm + i*8) * p.lda + k, p.lda);
            for (int j = 0; j < TN; j++) {
                simdgroup_float8x8 b_frag;
                simdgroup_load(b_frag, B + (long)k * p.ldb + sn + j*8, p.ldb);
                for (int i = 0; i < TM; i++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag, acc[i][j]);
            }
        }
    } else if (p.trans_a && !p.trans_b) {
        // TN: C = A(K,M)^T @ B(K,N) — backward weight grad
        for (int k = 0; k < K_aligned; k += 8) {
            simdgroup_float8x8 a_frag[TM];
            for (int i = 0; i < TM; i++)
                simdgroup_load(a_frag[i], A + (long)k * p.lda + sm + i*8, p.lda,
                               ulong2(0,0), true);
            for (int j = 0; j < TN; j++) {
                simdgroup_float8x8 b_frag;
                simdgroup_load(b_frag, B + (long)k * p.ldb + sn + j*8, p.ldb);
                for (int i = 0; i < TM; i++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag, acc[i][j]);
            }
        }
    }

    // Single threadgroup allocation shared between K-remainder and store phases.
    // These phases are sequential (barriers between), so memory is safely reused.
    // The compiler doubles explicit threadgroup memory (simdgroup register spill),
    // so separate arrays for sA+sB+sC exceed 32KB. Aliasing them into one buffer
    // keeps total at BM*BN*4*2 = 32768 = limit.
    constexpr int SMEM_STRIDE = BN;
    threadgroup float _smem[BM * SMEM_STRIDE];

    // ---- K-remainder: threadgroup fallback for last partial chunk ----
    // Only triggered when K % 8 != 0 (e.g., K=373). Loads remaining
    // elements into zero-padded 8-wide threadgroup tiles.
    // Reinterprets _smem as sA (BM×9 float) and sB (8×(BN+1) float).
    if (K_aligned < p.K) {
        threadgroup float* sA = _smem;                // BM*9 = 576 floats
        threadgroup float* sB = _smem + BM * 9;       // 8*(BN+1) = 520 floats
        constexpr int sB_stride = BN + 1;

        int k = K_aligned;
        int rem = p.K - k;

        // Cooperative load A: BM × 8 (zero-padded beyond rem)
        int total_a = BM * 8;
        for (int idx = tid; idx < total_a; idx += 128) {
            int r = idx / 8;
            int c = idx % 8;
            int gr = bm + r;
            int gc = k + c;
            sA[r * 9 + c] = (gr < p.M && c < rem)
                ? (p.trans_a ? A[(long)gc * p.lda + gr] : A[(long)gr * p.lda + gc])
                : 0.0f;
        }

        // Cooperative load B: 8 × BN (zero-padded beyond rem)
        int total_b = 8 * BN;
        for (int idx = tid; idx < total_b; idx += 128) {
            int r = idx / BN;
            int c = idx % BN;
            int gr = k + r;
            int gc = bn + c;
            sB[r * sB_stride + c] = (r < rem && gc < p.N)
                ? (p.trans_b ? B[(long)gc * p.ldb + gr] : B[(long)gr * p.ldb + gc])
                : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 a_frag[TM];
        for (int i = 0; i < TM; i++)
            simdgroup_load(a_frag[i], sA + (wm * 32 + i * 8) * 9, 9);
        for (int j = 0; j < TN; j++) {
            simdgroup_float8x8 b_frag;
            simdgroup_load(b_frag, sB + (wn * 32 + j * 8), sB_stride);
            for (int i = 0; i < TM; i++)
                simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag, acc[i][j]);
        }
    }

    // ---- Store results ----
    // Fast path: direct simdgroup_store for interior tiles with alpha=1, beta=0.
    // Slow path: threadgroup staging via _smem for edge tiles and non-trivial alpha/beta.
    // CRITICAL: condition must be uniform across the threadgroup — all simdgroups
    // must take the same branch, otherwise the threadgroup_barrier in the slow
    // path causes undefined behavior (some threads skip it).
    bool fast_store = (p.alpha == 1.0f && p.beta == 0.0f
                       && bm + BM <= p.M && bn + BN <= p.N);
    if (fast_store) {
        for (int i = 0; i < TM; i++)
            for (int j = 0; j < TN; j++)
                simdgroup_store(acc[i][j],
                    C + (long)(sm + i*8) * p.ldc + sn + j*8, p.ldc);
    } else {
        for (int i = 0; i < TM; i++)
            for (int j = 0; j < TN; j++)
                simdgroup_store(acc[i][j],
                    _smem + (wm*32 + i*8) * SMEM_STRIDE + (wn*32 + j*8),
                    SMEM_STRIDE);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cooperative store to device: each thread handles BM*BN/128 = 32 elements
        for (int idx = tid; idx < BM * BN; idx += 128) {
            int r = idx / BN;
            int c = idx % BN;
            int gr = bm + r;
            int gc = bn + c;
            if (gr < p.M && gc < p.N) {
                long out_idx = (long)gr * p.ldc + gc;
                float val = _smem[r * SMEM_STRIDE + c];
                if (p.beta == 0.0f)
                    C[out_idx] = p.alpha * val;
                else
                    C[out_idx] = p.alpha * val + p.beta * C[out_idx];
            }
        }
    }
}

// Used when N doesn't meet tensor_ops alignment (N%32!=0).
// One threadgroup per output row, threads partition N columns.
// C(M,N) = A(M,K) @ B(N,K)^T, all row-major.

struct SmallGemmParams {
    uint M;
    uint N;
    uint K;
};

kernel void small_gemm_nt_f32(
    const device float* A          [[buffer(0)]],
    const device float* B          [[buffer(1)]],
    device float* C                [[buffer(2)]],
    constant SmallGemmParams& p    [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    // tgid = output row, tid = output column
    uint m = tgid;
    if (tid >= p.N) return;

    const device float* a_row = A + m * p.K;
    const device float* b_row = B + tid * p.K;

    float sum = 0.0f;
    // float4 vectorized accumulation (K must be multiple of 4)
    uint K4 = p.K & ~3u;
    for (uint k = 0; k < K4; k += 4) {
        float4 a4 = *reinterpret_cast<const device float4*>(a_row + k);
        float4 b4 = *reinterpret_cast<const device float4*>(b_row + k);
        sum += dot(a4, b4);
    }
    // handle remainder (K not multiple of 4)
    for (uint k = K4; k < p.K; k++)
        sum += a_row[k] * b_row[k];

    C[m * p.N + tid] = sum;
}

// ============================================================================
// Section 27: Steel GEMM fp16 — half I/O, float accumulation
//
// Same 64x64 tiled GEMM as steel_gemm but with half-precision inputs/outputs.
// Uses simdgroup_half8x8 for loads, simdgroup_float8x8 for accumulation
// (mixed-precision multiply_accumulate). Stores back as half.
// ============================================================================

kernel void steel_gemm_f16(
    device const half* A       [[buffer(0)]],
    device const half* B       [[buffer(1)]],
    device half* C             [[buffer(2)]],
    constant GemmParams& p     [[buffer(3)]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    uint sgid                  [[simdgroup_index_in_threadgroup]],
    uint lane                  [[thread_index_in_simdgroup]]
) {
    constexpr int BM = 64, BN = 64;
    constexpr int WM = 2, WN = 2;
    constexpr int TM = BM / (8 * WM); // 4
    constexpr int TN = BN / (8 * WN); // 4

    const int bm = (int)tgid.y * BM;
    const int bn = (int)tgid.x * BN;
    const int wm = (int)(sgid / WN);
    const int wn = (int)(sgid % WN);
    const int tid = (int)(sgid * 32 + lane);

    const int sm = bm + wm * 32;
    const int sn = bn + wn * 32;

    // f32 accumulators for numerical stability
    simdgroup_float8x8 acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            acc[i][j] = simdgroup_float8x8(0);

    const int K_aligned = (p.K / 8) * 8;

    if (!p.trans_a && p.trans_b) {
        // NT: C = A(M,K) @ B(N,K)^T
        for (int k = 0; k < K_aligned; k += 8) {
            simdgroup_half8x8 a_frag[TM];
            for (int i = 0; i < TM; i++)
                simdgroup_load(a_frag[i], A + (long)(sm + i*8) * p.lda + k, p.lda);
            for (int j = 0; j < TN; j++) {
                simdgroup_half8x8 b_frag;
                simdgroup_load(b_frag, B + (long)(sn + j*8) * p.ldb + k, p.ldb,
                               ulong2(0,0), true);
                for (int i = 0; i < TM; i++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag, acc[i][j]);
            }
        }
    } else if (!p.trans_a && !p.trans_b) {
        // NN: C = A(M,K) @ B(K,N)
        for (int k = 0; k < K_aligned; k += 8) {
            simdgroup_half8x8 a_frag[TM];
            for (int i = 0; i < TM; i++)
                simdgroup_load(a_frag[i], A + (long)(sm + i*8) * p.lda + k, p.lda);
            for (int j = 0; j < TN; j++) {
                simdgroup_half8x8 b_frag;
                simdgroup_load(b_frag, B + (long)k * p.ldb + sn + j*8, p.ldb);
                for (int i = 0; i < TM; i++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag, acc[i][j]);
            }
        }
    } else if (p.trans_a && !p.trans_b) {
        // TN: C = A(K,M)^T @ B(K,N)
        for (int k = 0; k < K_aligned; k += 8) {
            simdgroup_half8x8 a_frag[TM];
            for (int i = 0; i < TM; i++)
                simdgroup_load(a_frag[i], A + (long)k * p.lda + sm + i*8, p.lda,
                               ulong2(0,0), true);
            for (int j = 0; j < TN; j++) {
                simdgroup_half8x8 b_frag;
                simdgroup_load(b_frag, B + (long)k * p.ldb + sn + j*8, p.ldb);
                for (int i = 0; i < TM; i++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag, acc[i][j]);
            }
        }
    }

    // Single threadgroup allocation shared between K-remainder and store phases.
    // These phases are sequential (barriers between), so memory is safely reused.
    // Store needs BM*BN floats = 16384 bytes (largest consumer).
    // K-remainder needs BM*9 + 8*(BN+1) halves = 2192 bytes (fits inside).
    // Without aliasing, the compiler allocates all three arrays statically and
    // exceeds the 32KB threadgroup memory limit. The compiler doubles explicit
    // threadgroup memory (simdgroup register spill), so BM*BN*4*2 = 32768 = limit.
    // No +1 stride padding: minor bank conflict on simdgroup_store vs. correctness.
    constexpr int SMEM_STRIDE = BN;
    threadgroup float _smem[BM * SMEM_STRIDE];

    // K-remainder: reinterpret _smem as half arrays for partial-K accumulation
    if (K_aligned < p.K) {
        threadgroup half* sA = (threadgroup half*)_smem;
        threadgroup half* sB = sA + BM * 9;
        constexpr int sB_stride = BN + 1;

        int k = K_aligned;
        int rem = p.K - k;

        int total_a = BM * 8;
        for (int idx = tid; idx < total_a; idx += 128) {
            int r = idx / 8;
            int c = idx % 8;
            int gr = bm + r;
            int gc = k + c;
            sA[r * 9 + c] = (gr < p.M && c < rem)
                ? (p.trans_a ? A[(long)gc * p.lda + gr] : A[(long)gr * p.lda + gc])
                : half(0);
        }

        int total_b = 8 * BN;
        for (int idx = tid; idx < total_b; idx += 128) {
            int r = idx / BN;
            int c = idx % BN;
            int gr = k + r;
            int gc = bn + c;
            sB[r * sB_stride + c] = (r < rem && gc < p.N)
                ? (p.trans_b ? B[(long)gc * p.ldb + gr] : B[(long)gr * p.ldb + gc])
                : half(0);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_half8x8 a_frag[TM];
        for (int i = 0; i < TM; i++)
            simdgroup_load(a_frag[i], sA + (wm * 32 + i * 8) * 9, 9);
        for (int j = 0; j < TN; j++) {
            simdgroup_half8x8 b_frag;
            simdgroup_load(b_frag, sB + (wn * 32 + j * 8), sB_stride);
            for (int i = 0; i < TM; i++)
                simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag, acc[i][j]);
        }
    }

    // Barrier: K-remainder reads _smem as half, store writes it as float.
    // Without this, a fast simdgroup's float writes corrupt a slow one's half reads.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Store: f32 acc → half output via _smem staging (reuses same memory)
    // simdgroup_store of float8x8 requires float* destination, so we stage
    // through _smem then convert element-by-element to half for output.
    {
        for (int i = 0; i < TM; i++)
            for (int j = 0; j < TN; j++)
                simdgroup_store(acc[i][j],
                    _smem + (wm*32 + i*8) * SMEM_STRIDE + (wn*32 + j*8),
                    SMEM_STRIDE);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int idx = tid; idx < BM * BN; idx += 128) {
            int r = idx / BN;
            int c = idx % BN;
            int gr = bm + r;
            int gc = bn + c;
            if (gr < p.M && gc < p.N) {
                long out_idx = (long)gr * p.ldc + gc;
                float val = _smem[r * SMEM_STRIDE + c];
                if (p.beta == 0.0f)
                    C[out_idx] = half(p.alpha * val);
                else
                    C[out_idx] = half(p.alpha * val + p.beta * float(C[out_idx]));
            }
        }
    }
}

)METAL";
}

#endif // PUFFERLIB_METAL_SHADER_SRC_H
