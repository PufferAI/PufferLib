#ifndef PUFFERLIB_KERNELS_CU
#define PUFFERLIB_KERNELS_CU

/* Kernels must launch on the current torch stream to be traced by cudagraphs.
 * This file is included by modules.cu which calls kernels directly with <<<>>>.
 * Callers should use at::cuda::getCurrentCUDAStream() when using with torch.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "ops.cuh"
#include <curand_kernel.h>

#include <cstdio>
#include <cstdint>

// Compile-time precision: default bf16, pass -DPRECISION_FLOAT for float32
#ifdef PRECISION_FLOAT
typedef float precision_t;
#else
typedef __nv_bfloat16 precision_t;
#endif

#define PPO_THREADS 256

#define SEQ_SIZE 256
#define BLOCK_SIZE 256
#define CHECKPOINT_INTERVAL 4  // Sparse checkpoint interval for optimized kernels

// Maximum number of action heads supported for MultiDiscrete
// Using register arrays to avoid dynamic allocation
#define MAX_ATN_HEADS 16


inline int grid_size(int N) {
    return (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
}
inline int seq_size(int N) {
    return (N + SEQ_SIZE - 1) / SEQ_SIZE;
}

// Compile-time precision conversion macros
#ifdef PRECISION_FLOAT
#define to_float(x) (x)
#define from_float(x) (x)
#else
#define to_float(x) __bfloat162float(x)
#define from_float(x) __float2bfloat16(x)
#endif


// Fused kernel: chunk + mingru_gate + sigmoid(proj) * out
// combined is (B, 1, 3*H) containing [hidden, gate, proj] concatenated on last dim
// state is (B, 1, H)
// out is (B, H) = sigmoid(proj) * mingru_out (final output)
// next_state is (B, H) = mingru_out (recurrent state, without proj)
__global__ void mingru_gate_inference_kernel(
    precision_t* out,
    precision_t* next_state,
    const precision_t* combined,    // (B, 3*H) = [hidden, gate, proj]
    const precision_t* state_in,    // (B, H)
    int H,
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * H;
    if (idx >= N) return;

    int b = idx / H;
    int h = idx % H;

    // Read from combined: layout is [hidden(H), gate(H), proj(H)] for each batch
    int combined_base = b * 3 * H;
    float hidden = to_float(combined[combined_base + h]);
    float gate = to_float(combined[combined_base + H + h]);
    float proj = to_float(combined[combined_base + 2 * H + h]);
    float state = to_float(state_in[idx]);

    // mingru_gate computation
    float gate_sigmoid = sigmoid(gate);
    float hidden_tilde = tilde_relu_fwd(hidden);
    float mingru_out = lerp(state, hidden_tilde, gate_sigmoid);

    // next_state is mingru_out (for recurrence)
    next_state[idx] = from_float(mingru_out);

    // out is sigmoid(proj) * mingru_out (final output)
    float proj_sigmoid = sigmoid(proj);
    out[idx] = from_float(proj_sigmoid * mingru_out);
}


__device__ __forceinline__ double logcumsumexp_forward(double x, double acc) {
    if (acc == -INFINITY) {
        return x;
    } else {
        double min_val = fmin(acc, x);
        double max_val = fmax(acc, x);
        return max_val + log1pf(expf(min_val - max_val));
    }
}

__device__ __forceinline__ double logcumsumexp_backward(double x, double* acc, double grad, double s, double* s_nxt) {
    *acc = grad + *acc * exp(s - *s_nxt);
    *s_nxt = s;
    return *acc * exp(x - s);
}

// Optimized forward kernel with checkpointing
// Writes checkpoints only every CHECKPOINT_INTERVAL timesteps (vs every time)
// Uses fast math intrinsics for better performance
__global__ void fused_scan_forward_kernel_checkpointed(
    precision_t* __restrict__ out,                 // (B, T, H)
    precision_t* __restrict__ next_state,          // (B, 1, H)
    float* __restrict__ a_star_buf,      // (B, T+1, H)
    float* __restrict__ s_buf,           // (B, T+1, H)
    float* __restrict__ log_values_buf,  // (B, T+1, H)
    const precision_t* __restrict__ combined,      // (B, T, 3*H)
    const precision_t* __restrict__ state,         // (B, 1, H)
    int T_seq,
    int H,
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;

    int b = idx / H;
    int h = idx % H;

    int bH = b * H;
    int H3 = 3 * H;
    int H2 = 2 * H;
    int bHT = bH * T_seq;
    int out_base = bHT + h;
    int cbase = 3 * bHT;

    float a_star = 0.0f;
    float log_value = 0.0f;

    // Handle t=0 outside the loop: use log(state), coeff = 0
    float s = __logf(to_float(state[bH + h]));
    log_value = s;

    int T_out = T_seq + 1;
    int buf_base = b * T_out * H + h;
    int buf_curr = buf_base;
    a_star_buf[buf_curr] = a_star;
    s_buf[buf_curr] = s;
    log_values_buf[buf_curr] = log_value;

    const precision_t* combined_h_base = &combined[cbase + h];
    const precision_t* combined_g_base = &combined[cbase + H + h];
    const precision_t* combined_p_base = &combined[cbase + H2 + h];

    // Loop t=1..T_seq with sparse checkpointing
    float scan_result = 0.0f;
    int out_curr = out_base;
    int t_offset = 0;

    for (int t = 1; t < T_seq + 1; t++) {
        float hidden_val = to_float(combined_h_base[t_offset]);
        float gate_val = to_float(combined_g_base[t_offset]);
        float proj_val = to_float(combined_p_base[t_offset]);

        float log_coeff_val;
        log_coeffs_and_values_fwd(gate_val, hidden_val, &log_coeff_val, &log_value);

        // a_star[t] = sum_{i=0}^t log_coeffs[i]
        a_star += log_coeff_val;

        float z = log_value - a_star;
        float max_val = fmaxf(s, z);
        s = max_val + log1pf(__expf(-fabsf(s - z)));

        scan_result = __expf(a_star + s);
        float proj_sigmoid = sigmoid(proj_val);

        out[out_curr] = from_float(proj_sigmoid * scan_result);

        buf_curr += H;
        out_curr += H;
        t_offset += H3;

        if (t % CHECKPOINT_INTERVAL == 0) {
            a_star_buf[buf_curr] = a_star;
            s_buf[buf_curr] = s;
            log_values_buf[buf_curr] = log_value;
        }
    }

    // Write timestep T to next_state (raw scan_result, no proj, for recurrence)
    next_state[bH + h] = from_float(scan_result);
}

// Optimized backward kernel with sparse checkpoint loading
// Reads sparse checkpoints from forward pass, recomputes intermediate values in chunks
// Uses fast math intrinsics for better performance
__global__ void fused_scan_backward_kernel_checkpointed(
    precision_t* __restrict__ grad_combined,         // (B, T, 3*H)
    precision_t* __restrict__ grad_state,            // (B, 1, H)
    const precision_t* __restrict__ grad_out,        // (B, T, H)
    const precision_t* __restrict__ grad_next_state, // (B, 1, H)
    const precision_t* __restrict__ combined,        // (B, T, 3*H)
    const precision_t* __restrict__ state,           // (B, 1, H)
    const float* __restrict__ a_star_buf,  // (B, T+1, H)
    const float* __restrict__ s_buf,       // (B, T+1, H)
    const float* __restrict__ log_values_buf, // (B, T+1, H)
    int T_seq,                             // (T)
    int H,
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;

    int b = idx / H;
    int h = idx % H;

    int bHT = b * H * T_seq;
    int cbase = 3 * bHT;
    int H3 = 3 * H;
    int H2 = 2 * H;
    const int state_idx = b * H + h;
    const int out_base = bHT + h;

    const precision_t* combined_h_base = &combined[cbase + h];
    const precision_t* combined_g_base = &combined[cbase + H + h];
    const precision_t* combined_p_base = &combined[cbase + H2 + h];

    precision_t* grad_combined_h_base = &grad_combined[cbase + h];
    precision_t* grad_combined_g_base = &grad_combined[cbase + H + h];
    precision_t* grad_combined_p_base = &grad_combined[cbase + H2 + h];

    int T_out = T_seq + 1;
    int buf_base = b * T_out * H + h;

    float acc = 0.0;
    float s_val_next = 0.0;
    float carry_grad_a = 0.0;

    for (int chunk_end = T_seq; chunk_end > 0; chunk_end -= CHECKPOINT_INTERVAL) {
        int chunk_start = (chunk_end > CHECKPOINT_INTERVAL) ? (chunk_end - CHECKPOINT_INTERVAL) : 0;
        int chunk_len = chunk_end - chunk_start;

        // Chunk storage in registers
        float chunk_a_star[CHECKPOINT_INTERVAL];
        float chunk_s[CHECKPOINT_INTERVAL];
        float chunk_log_values[CHECKPOINT_INTERVAL];
        float chunk_hidden[CHECKPOINT_INTERVAL];
        float chunk_gate[CHECKPOINT_INTERVAL];

        // Load checkpoint from global memory
        int ckpt_buf_idx = buf_base + chunk_start * H;
        float recomp_a_star = a_star_buf[ckpt_buf_idx];
        float recomp_s = s_buf[ckpt_buf_idx];
        float recomp_log_value = log_values_buf[ckpt_buf_idx];

        // Recompute and store from chunk_start to chunk_end
        for (int i = 0; i < chunk_len; ++i) {
            int t = chunk_start + 1 + i;
            int t_offset = (t - 1) * H3;
            float hv = to_float(combined_h_base[t_offset]);
            float gv = to_float(combined_g_base[t_offset]);

            float lc;
            log_coeffs_and_values_fwd(gv, hv, &lc, &recomp_log_value);
            recomp_a_star += lc;

            float z = recomp_log_value - recomp_a_star;
            float mv = fmaxf(recomp_s, z);
            recomp_s = mv + log1pf(__expf(-fabsf(recomp_s - z)));

            chunk_a_star[i] = recomp_a_star;
            chunk_s[i] = recomp_s;
            chunk_log_values[i] = recomp_log_value;
            chunk_hidden[i] = hv;
            chunk_gate[i] = gv;
        }

        for (int i = chunk_len - 1; i >= 0; --i) {
            int t = chunk_start + 1 + i;
            int t_offset = (t - 1) * H3;

            float a_star_t = chunk_a_star[i];
            float s_t = chunk_s[i];
            float log_value_t = chunk_log_values[i];
            float hidden_val = chunk_hidden[i];
            float gate_val = chunk_gate[i];

            float proj_val = to_float(combined_p_base[t_offset]);

            float scan_result = __expf(a_star_t + s_t);
            float z = log_value_t - a_star_t;

            float grad_out_val = to_float(grad_out[out_base + (t - 1) * H]);

            float grad_scan_from_next = (t == T_seq) ? to_float(grad_next_state[state_idx]) : 0.0f;

            float proj_sigmoid = sigmoid(proj_val);
            float grad_scan_result = grad_scan_from_next + grad_out_val * proj_sigmoid;
            float grad_proj = grad_out_val * scan_result * proj_sigmoid * (1.0f - proj_sigmoid);

            float grad_log_h = grad_scan_result * scan_result;
            float grad_s = grad_log_h;

            if (t == T_seq) {
                acc = grad_s;
            } else {
                acc = grad_s + acc * __expf(s_t - s_val_next);
            }
            float grad_z = acc * __expf(z - s_t);
            s_val_next = s_t;

            float grad_a = grad_log_h + carry_grad_a - grad_z;
            carry_grad_a = grad_a;

            float grad_g, grad_h;
            log_coeffs_and_values_bwd(grad_a, grad_z, gate_val, hidden_val, &grad_g, &grad_h);

            grad_combined_h_base[t_offset] = from_float(grad_h);
            grad_combined_g_base[t_offset] = from_float(grad_g);
            grad_combined_p_base[t_offset] = from_float(grad_proj);
        }
    }

    int ckpt_0_idx = buf_base;
    float a_star_0 = a_star_buf[ckpt_0_idx];
    float s_0 = s_buf[ckpt_0_idx];
    float log_value_0 = log_values_buf[ckpt_0_idx];

    float scan_result_0 = __expf(a_star_0 + s_0);
    float z_0 = log_value_0 - a_star_0;

    float grad_scan_result_0 = 0.0f;
    float grad_log_h_0 = grad_scan_result_0 * scan_result_0;
    float grad_s_0 = grad_log_h_0;

    acc = grad_s_0 + acc * __expf(s_0 - s_val_next);
    float grad_z_0 = acc * __expf(z_0 - s_0);

    grad_state[state_idx] = from_float(grad_z_0 / to_float(state[state_idx]));
}



// This exactly matches pytorch in double, but not in float
__global__ void logcumsumexp_forward_kernel(
    precision_t* __restrict__ out,           // exp(s[t])
    double* __restrict__ s_buf,     // s[t] = logcumsumexp(x[0..t])
    const precision_t* __restrict__ x,       // input: log_values
    int T_total,
    int H,
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;

    int b = idx / H;
    int h = idx % H;

    int base = b * T_total * H + h;

    double s = -INFINITY;

    for (int t = 0; t < T_total; t++) {
        int curr = base + t * H;
        double x_val = (double)to_float(x[curr]);
        s = logcumsumexp_forward(x_val, s);
        out[curr] = from_float((float)s);
        s_buf[curr] = s;
    }
}
__global__ void logcumsumexp_backward_kernel(
    precision_t* __restrict__ grad_x,
    const precision_t* __restrict__ grad_out,
    const precision_t* __restrict__ x,
    const double* __restrict__ s_buf,
    int T_total,
    int H,
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;

    int b = idx / H;
    int h = idx % H;

    int base = b * T_total * H + h;

    double acc = 0.0;
    double s_val_next = 0.0;

    for (int t = T_total - 1; t >= 0; --t) {
        int curr = base + t * H;

        double x_val = (double)to_float(x[curr]);
        double s_val = double(s_buf[curr]);
        double g_val = (double)to_float(grad_out[curr]);
        grad_x[curr] = from_float((float)logcumsumexp_backward(x_val, &acc, g_val, s_val, &s_val_next));
    }
}



__global__ void ppo_loss_forward_kernel_optimized(
    float* __restrict__ loss,
    double* __restrict__ saved_for_backward,
    precision_t* __restrict__ ratio_out,
    precision_t* __restrict__ newvalue_out,
    const precision_t* __restrict__ logits,       // For continuous: mean
    const precision_t* __restrict__ logstd,       // For continuous: log standard deviation (nullptr for discrete)
    const precision_t* __restrict__ values_pred,
    const double* __restrict__ actions, // float64 for both continuous and discrete
    const precision_t* __restrict__ old_logprobs,
    const float* __restrict__ advantages,
    const precision_t* __restrict__ prio,
    const precision_t* __restrict__ values,
    const precision_t* __restrict__ returns,
    const float* __restrict__ adv_mean,
    const float* __restrict__ adv_var,
    const int* __restrict__ act_sizes,  // NEW: array of action head sizes
    int num_atns,                        // NEW: number of action heads
    float clip_coef,
    float vf_clip_coef,
    float vf_coef,
    float ent_coef,
    int T_seq,
    int A_total,                         // renamed: sum of all action sizes
    int N,
    int logits_stride_n,
    int logits_stride_t,
    int logits_stride_a,
    int values_stride_n,
    int values_stride_t,
    bool is_continuous
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * T_seq;
    if (idx >= total_elements) return;

    __shared__ float block_loss[PPO_THREADS];

    int n = idx / T_seq;
    int t = idx % T_seq;
    int nt = n * T_seq + t;

    int logits_base = n * logits_stride_n + t * logits_stride_t;
    int values_idx = n * values_stride_n + t * values_stride_t;

    float total_log_prob = 0.0f;
    float total_entropy = 0.0f;

    if (is_continuous) {
        // Continuous actions: Normal distribution
        // log_prob = -0.5 * ((action - mean) / std)^2 - 0.5 * log(2*pi) - log(std)
        // Differential entropy = 0.5 * (1 + log(2*pi)) + log_std
        constexpr float LOG_2PI = 1.8378770664093453f;
        constexpr float HALF_LOG_2PI = 0.9189385332046727f;
        constexpr float HALF_1_PLUS_LOG_2PI = 1.4189385332046727f;  // 0.5 * (1 + log(2*pi))

        for (int h = 0; h < num_atns; ++h) {
            float mean = to_float(logits[logits_base + h * logits_stride_a]);
            float log_std = to_float(logstd[logits_base + h * logits_stride_a]);
            float std = __expf(log_std);
            float action = float(actions[nt * num_atns + h]);

            float normalized = (action - mean) / std;
            float log_prob = -0.5f * normalized * normalized - HALF_LOG_2PI - log_std;
            // Differential entropy for Normal: 0.5 + 0.5*log(2*pi) + log_std
            float entropy = HALF_1_PLUS_LOG_2PI + log_std;

            total_log_prob += log_prob;
            total_entropy += entropy;
        }
    } else {
        // Discrete actions: Categorical distribution
        // Loop over action heads for MultiDiscrete support
        int logits_offset = 0;

        for (int h = 0; h < num_atns; ++h) {
            int A = act_sizes[h];  // size of this action head
            int act = static_cast<int>(actions[nt * num_atns + h]);  // action for this head

            // Find max for numerical stability and cache action's logit
            float max_logit = -INFINITY;
            float sum = 0.0f;
            float act_logit = 0.0f;

            for (int a = 0; a < A; ++a) {
                float l = to_float(logits[logits_base + (logits_offset + a) * logits_stride_a]);

                if (a == act) {
                    act_logit = l;
                }

                if (l > max_logit) {
                    sum *= __expf(max_logit - l);
                    max_logit = l;
                }
                sum += __expf(l - max_logit);
            }

            float logsumexp = max_logit + __logf(sum);

            // Compute entropy for this head
            float head_entropy = 0.0f;
            for (int a = 0; a < A; ++a) {
                float l = to_float(logits[logits_base + (logits_offset + a) * logits_stride_a]);
                float logp = l - logsumexp;
                float p = __expf(logp);
                head_entropy -= p * logp;
            }

            // Log prob for this head
            float head_logp = act_logit - logsumexp;

            // Accumulate across heads
            total_log_prob += head_logp;
            total_entropy += head_entropy;

            // Advance to next action head
            logits_offset += A;
        }
    }

    // Use accumulated values for the rest of computation
    float new_logp = total_log_prob;
    float entropy = total_entropy;

    float old_logp = to_float(old_logprobs[nt]);
    float adv = float(advantages[nt]);
    float w = to_float(prio[n]);
    float adv_std = sqrtf(float(adv_var[0]));
    float adv_normalized = (adv - float(adv_mean[0])) / (adv_std + 1e-8f);

    float logratio = new_logp - old_logp;
    float ratio = __expf(logratio);

    float ratio_clipped = fmaxf(1.0f - clip_coef, fminf(1.0f + clip_coef, ratio));
    float wa = -w * adv_normalized;
    float pg_loss1 = wa * ratio;
    float pg_loss2 = wa * ratio_clipped;
    float pg_loss = fmaxf(pg_loss1, pg_loss2);

    float val = to_float(values[nt]);
    float ret = to_float(returns[nt]);
    float val_pred = to_float(values_pred[values_idx]);

    float v_error = val_pred - val;
    float v_clipped = val + fmaxf(-vf_clip_coef, fminf(vf_clip_coef, v_error));
    float v_loss_unclipped = (val_pred - ret) * (val_pred - ret);
    float v_loss_clipped = (v_clipped - ret) * (v_clipped - ret);
    float v_loss = 0.5f * fmaxf(v_loss_unclipped, v_loss_clipped);

    float thread_loss = (pg_loss + vf_coef * v_loss - ent_coef * entropy) / float(total_elements);

    // Write ratio and newvalue outputs
    //ratio_out[nt] = T(ratio);
    //newvalue_out[nt] = T(val_pred);

    int tid = threadIdx.x;
    block_loss[tid] = thread_loss;
    __syncthreads();

    for (int stride = PPO_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            block_loss[tid] += block_loss[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(loss, block_loss[0]);
    }
}


__global__ void ppo_loss_backward_kernel_optimized(
    float* __restrict__ grad_logits,    // For continuous: grad_mean
    float* __restrict__ grad_logstd,    // For continuous: grad_logstd (nullptr for discrete)
    float* __restrict__ grad_values_pred,
    const float* __restrict__ grad_loss,
    const precision_t* __restrict__ logits,       // For continuous: mean
    const precision_t* __restrict__ logstd,       // For continuous: log standard deviation (nullptr for discrete)
    const precision_t* __restrict__ values_pred,
    const double* __restrict__ actions, // float64 for both continuous and discrete
    const precision_t* __restrict__ old_logprobs,
    const float* __restrict__ advantages,
    const precision_t* __restrict__ prio,
    const precision_t* __restrict__ values,
    const precision_t* __restrict__ returns,
    const float* __restrict__ adv_mean,
    const float* __restrict__ adv_var,
    const int* __restrict__ act_sizes,  // NEW: array of action head sizes
    int num_atns,                        // NEW: number of action heads
    float clip_coef,
    float vf_clip_coef,
    float vf_coef,
    float ent_coef,
    int T_seq,
    int A_total,                         // renamed: sum of all action sizes
    int N,
    int logits_stride_n,
    int logits_stride_t,
    int logits_stride_a,
    int values_stride_n,
    int values_stride_t,
    bool is_continuous
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * T_seq;
    if (idx >= total_elements) return;

    float inv_NT = 1.0f / float(total_elements);
    int n = idx / T_seq;
    int t = idx % T_seq;
    int nt = n * T_seq + t;

    int logits_base = n * logits_stride_n + t * logits_stride_t;
    int values_idx = n * values_stride_n + t * values_stride_t;

    float old_logp = to_float(old_logprobs[nt]);
    float adv = float(advantages[nt]);
    float w = to_float(prio[n]);
    float val = to_float(values[nt]);
    float ret = to_float(returns[nt]);
    float val_pred = to_float(values_pred[values_idx]);

    // Normalize advantage
    float adv_std_val = sqrtf(float(adv_var[0]));
    float adv_normalized = (adv - float(adv_mean[0])) / (adv_std_val + 1e-8f);

    // Loss gradient scaling
    float dL = grad_loss[0] * inv_NT;
    float d_pg_loss = dL;
    float d_entropy_term = dL * (-ent_coef);

    // Gradient wrt value function prediction
    float v_error = val_pred - val;
    float v_clipped = val + fmaxf(-vf_clip_coef, fminf(vf_clip_coef, v_error));
    float v_loss_unclipped = (val_pred - ret) * (val_pred - ret);
    float v_loss_clipped = (v_clipped - ret) * (v_clipped - ret);
    bool use_clipped_vf = (v_loss_clipped > v_loss_unclipped);

    float d_val_pred = 0.0f;
    if (use_clipped_vf) {
        if (v_error >= -vf_clip_coef && v_error <= vf_clip_coef) {
            d_val_pred = v_clipped - ret;
        }
    } else {
        d_val_pred = val_pred - ret;
    }
    grad_values_pred[values_idx] = dL * vf_coef * d_val_pred;

    if (is_continuous) {
        // Continuous: compute total log prob first for ratio
        constexpr float HALF_LOG_2PI = 0.9189385332046727f;
        float total_log_prob = 0.0f;

        for (int h = 0; h < num_atns; ++h) {
            float mean = to_float(logits[logits_base + h * logits_stride_a]);
            float log_std = to_float(logstd[logits_base + h * logits_stride_a]);
            float std = __expf(log_std);
            float action = float(actions[nt * num_atns + h]);

            float normalized = (action - mean) / std;
            float log_prob = -0.5f * normalized * normalized - HALF_LOG_2PI - log_std;
            total_log_prob += log_prob;
        }

        float new_logp = total_log_prob;
        float ratio = __expf(new_logp - old_logp);

        // Policy loss gradient
        float ratio_clipped = fmaxf(1.0f - clip_coef, fminf(1.0f + clip_coef, ratio));
        float pg_loss1 = -w * adv_normalized * ratio;
        float pg_loss2 = -w * adv_normalized * ratio_clipped;

        float d_ratio = -w * adv_normalized * d_pg_loss;
        if (pg_loss2 > pg_loss1) {
            if (ratio <= (1.0f - clip_coef) || ratio >= (1.0f + clip_coef)) {
                d_ratio = 0.0f;
            }
        }
        float d_new_logp = d_ratio * ratio;

        // Compute gradients for continuous actions
        // log_prob = -0.5 * ((action - mean) / std)^2 - 0.5 * log(2*pi) - log_std
        // d_log_prob/d_mean = (action - mean) / var
        // d_log_prob/d_log_std = (action - mean)^2 / var - 1
        // entropy = 0.5 + 0.5*log(2*pi) + log_std
        // d_entropy/d_log_std = 1

        for (int h = 0; h < num_atns; ++h) {
            float mean = to_float(logits[logits_base + h * logits_stride_a]);
            float log_std = to_float(logstd[logits_base + h * logits_stride_a]);
            float std = __expf(log_std);
            float var = std * std;
            float action = float(actions[nt * num_atns + h]);

            float diff = action - mean;

            // Gradient wrt mean: d_log_prob/d_mean = (action - mean) / var
            float d_mean = d_new_logp * diff / var;
            grad_logits[logits_base + h * logits_stride_a] = d_mean;

            // Gradient wrt log_std:
            // d_log_prob/d_log_std = (action - mean)^2 / var - 1
            // d_entropy/d_log_std = 1
            // Total: d_new_logp * ((diff^2/var) - 1) + d_entropy_term * 1
            float d_log_std = d_new_logp * (diff * diff / var - 1.0f) + d_entropy_term;
            grad_logstd[logits_base + h * logits_stride_a] = d_log_std;
        }
    } else {
        // Discrete: original implementation
        // First pass: compute per-head logsumexp and entropy, accumulate total log prob
        // Store per-head values for gradient computation (use register arrays)
        float head_logsumexp[MAX_ATN_HEADS];
        float head_entropy[MAX_ATN_HEADS];
        int head_act[MAX_ATN_HEADS];

        int logits_offset = 0;
        float total_log_prob = 0.0f;

        for (int h = 0; h < num_atns; ++h) {
            int A = act_sizes[h];
            int act = static_cast<int>(actions[nt * num_atns + h]);
            head_act[h] = act;

            // Compute logsumexp for this head
            float max_logit = -INFINITY;
            float sum = 0.0f;
            float act_logit = 0.0f;

            for (int a = 0; a < A; ++a) {
                float l = to_float(logits[logits_base + (logits_offset + a) * logits_stride_a]);
                if (a == act) act_logit = l;

                if (l > max_logit) {
                    sum *= __expf(max_logit - l);
                    max_logit = l;
                }
                sum += __expf(l - max_logit);
            }
            float logsumexp = max_logit + __logf(sum);
            head_logsumexp[h] = logsumexp;

            // Compute entropy for this head
            float ent = 0.0f;
            for (int a = 0; a < A; ++a) {
                float l = to_float(logits[logits_base + (logits_offset + a) * logits_stride_a]);
                float logp = l - logsumexp;
                float p = __expf(logp);
                ent -= p * logp;
            }
            head_entropy[h] = ent;

            // Accumulate total log prob
            total_log_prob += act_logit - logsumexp;

            logits_offset += A;
        }

        // Compute ratio and policy gradient
        float new_logp = total_log_prob;
        float ratio = __expf(new_logp - old_logp);

        // Policy loss gradient
        float ratio_clipped = fmaxf(1.0f - clip_coef, fminf(1.0f + clip_coef, ratio));
        float pg_loss1 = -w * adv_normalized * ratio;
        float pg_loss2 = -w * adv_normalized * ratio_clipped;

        float d_ratio = -w * adv_normalized * d_pg_loss;
        if (pg_loss2 > pg_loss1) {
            if (ratio <= (1.0f - clip_coef) || ratio >= (1.0f + clip_coef)) {
                d_ratio = 0.0f;
            }
        }
        // d_new_logp flows to each head's log prob equally since total = sum of head log probs
        float d_new_logp = d_ratio * ratio;

        // Second pass: compute gradients per head
        logits_offset = 0;
        for (int h = 0; h < num_atns; ++h) {
            int A = act_sizes[h];
            int act = head_act[h];
            float logsumexp = head_logsumexp[h];
            float ent = head_entropy[h];

            for (int a = 0; a < A; ++a) {
                float l = to_float(logits[logits_base + (logits_offset + a) * logits_stride_a]);
                float logp = l - logsumexp;
                float p = __expf(logp);

                // Policy gradient: d/dlogits[a] of head_logp = delta(a,act) - p
                float d_logit = (a == act) ? d_new_logp : 0.0f;
                d_logit -= p * d_new_logp;

                // Entropy gradient: d/dlogits[a] of head_entropy = p * (-entropy - logp)
                // Each head's entropy contributes independently to total entropy
                d_logit += d_entropy_term * p * (-ent - logp);

                grad_logits[logits_base + (logits_offset + a) * logits_stride_a] = d_logit;
            }

            logits_offset += A;
        }
    }
}



// ============================================================================
// Fused sample_logits kernel: nan_to_num + log_softmax + multinomial + gather + value copy
// Inference-only (no gradients needed)
// Uses inline cuRAND to avoid separate torch::rand() kernel launch
//
// NOTE: This kernel supports strided (non-contiguous) logits/value input.
// This is needed for fused logit+value decoder output where logits is a view
// of a larger (B, V+A) tensor. The stride parameters handle this case
// to avoid .contiguous() kernel launches.
// ============================================================================

// Single kernel that handles both discrete and continuous action sampling
// Discrete: nan_to_num, log_softmax, multinomial sampling, logprob gather, value copy
// Continuous: sample from Normal(mean, exp(logstd)), compute log_prob, value copy
// Input: logits (B, num_atns) for continuous or (B, sum(act_sizes)) for discrete
//        logstd (B, num_atns) for continuous, nullptr for discrete
// Output: actions (B, num_atns) as float64, logprobs (B,), value_out (B,)
// NOTE: offset is read from a pointer (not passed by value) so it works correctly with CUDA graphs.
__global__ void sample_logits_kernel(
    double* __restrict__ actions,         // (B, num_atns) output - sampled actions as float64
    precision_t* __restrict__ logprobs,             // (B,) output - sum of log probs across action dims
    precision_t* __restrict__ value_out,            // (B,) output - copied value (flattened)
    const precision_t* __restrict__ logits,         // (B, num_atns or sum(act_sizes)) input - mean for continuous, logits for discrete
    const precision_t* __restrict__ logstd,         // (B, num_atns) input - log std for continuous, nullptr for discrete
    const precision_t* __restrict__ value,          // (B, 1) input - value from fused output (may be non-contiguous)
    const int* __restrict__ act_sizes,    // (num_atns,) input - size of each action head
    uint64_t seed,                        // RNG seed
    const int64_t* __restrict__ offset_ptr, // RNG offset pointer (read at execution time for CUDA graph support)
    int num_atns,                         // number of action heads/dimensions
    int B,                                // batch size
    int logits_stride,                    // stride between rows (for non-contiguous logits from fused output)
    int logstd_stride,                    // stride between rows for logstd (may be 0 for broadcast)
    int value_stride,                     // stride between rows (for non-contiguous value from fused output)
    bool is_continuous                    // true for continuous actions, false for discrete
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) return;

    // Read offset at execution time (important for CUDA graph replay)
    uint64_t offset = static_cast<uint64_t>(*offset_ptr);

    // Initialize RNG state once per thread
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, offset, &state);

    int logits_base = idx * logits_stride;
    float total_log_prob = 0.0f;

    if (is_continuous) {
        // Continuous action sampling from Normal(mean, exp(logstd))
        constexpr float LOG_2PI = 1.8378770664093453f;  // log(2*pi)
        int logstd_base = idx * logstd_stride;  // separate stride for logstd (may be 0 for broadcast)

        for (int h = 0; h < num_atns; ++h) {
            float mean = to_float(logits[logits_base + h]);
            float log_std = to_float(logstd[logstd_base + h]);
            float std = expf(log_std);

            // Sample from N(0,1) and transform: action = mean + std * noise
            float noise = curand_normal(&state);
            float action = mean + std * noise;

            // Log probability: -0.5 * ((action - mean) / std)^2 - 0.5 * log(2*pi) - log(std)
            float normalized = (action - mean) / std;
            float log_prob = -0.5f * normalized * normalized - 0.5f * LOG_2PI - log_std;

            actions[idx * num_atns + h] = double(action);
            total_log_prob += log_prob;
        }
    } else {
        // Discrete action sampling (original multinomial logic)
        int logits_offset = 0;  // offset within row for current action head

        for (int h = 0; h < num_atns; ++h) {
            int A = act_sizes[h];  // size of this action head

            // Step 1: Find max for numerical stability (with nan_to_num)
            float max_val = -INFINITY;
            for (int a = 0; a < A; ++a) {
                float l = to_float(logits[logits_base + logits_offset + a]);
                if (isnan(l)) l = 0.0f;
                if (isinf(l)) l = (l > 0) ? 3.4028e+38f : -3.4028e+38f;
                max_val = fmaxf(max_val, l);
            }

            // Step 2: Compute logsumexp for log_softmax denominator
            float sum_exp = 0.0f;
            for (int a = 0; a < A; ++a) {
                float l = to_float(logits[logits_base + logits_offset + a]);
                if (isnan(l)) l = 0.0f;
                if (isinf(l)) l = (l > 0) ? 3.4028e+38f : -3.4028e+38f;
                sum_exp += expf(l - max_val);
            }
            float logsumexp = max_val + logf(sum_exp);

            // Step 3: Generate random value for this action head
            float rand_val = curand_uniform(&state);

            // Step 4: Multinomial sampling using inverse CDF
            float cumsum = 0.0f;
            int sampled_action = A - 1;  // default to last action

            for (int a = 0; a < A; ++a) {
                float l = to_float(logits[logits_base + logits_offset + a]);
                if (isnan(l)) l = 0.0f;
                if (isinf(l)) l = (l > 0) ? 3.4028e+38f : -3.4028e+38f;
                float prob = expf(l - logsumexp);
                cumsum += prob;
                if (rand_val < cumsum) {
                    sampled_action = a;
                    break;
                }
            }

            // Step 5: Gather log probability of sampled action
            float sampled_logit = to_float(logits[logits_base + logits_offset + sampled_action]);
            if (isnan(sampled_logit)) sampled_logit = 0.0f;
            if (isinf(sampled_logit)) sampled_logit = (sampled_logit > 0) ? 3.4028e+38f : -3.4028e+38f;
            float log_prob = sampled_logit - logsumexp;

            // Write action for this head
            actions[idx * num_atns + h] = double(sampled_action);
            total_log_prob += log_prob;

            // Advance to next action head
            logits_offset += A;
        }
    }

    // Write summed log probability (log of joint probability)
    logprobs[idx] = from_float(total_log_prob);

    // Copy value (fused to avoid separate elementwise kernel for strided->contiguous copy)
    value_out[idx] = value[idx * value_stride];

    // Increment RNG offset for next call (thread 0 only, fused to avoid separate kernel)
    if (idx == 0) {
        atomicAdd((unsigned long long*)offset_ptr, 1ULL);
    }
}



// =============================================================================
// FCMax: Fused FC -> Max kernel
// Input: x (B, N, D_in), W (D_out, D_in), b (D_out)
// Output: out (B, D_out) = max_over_N(x @ W.T + b)
// Each thread computes one (b, d_out) output element
// N-fold memory bandwidth reduction vs separate FC + Max kernels
// W and b are always float32 (mixed precision for bf16 activations)
// =============================================================================

__global__ void fc_max_forward_kernel(
    precision_t* __restrict__ out,                // (B, D_out)
    int* __restrict__ argmax_indices,   // (B, D_out) - which N produced the max
    const precision_t* __restrict__ x,            // (B, N, D_in)
    const float* __restrict__ W,        // (D_out, D_in) - always float32
    const float* __restrict__ b,        // (D_out) - always float32
    int B, int N, int D_in, int D_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * D_out) return;

    int batch = idx / D_out;
    int d_out = idx % D_out;

    float bias = b[d_out];
    float max_val = -INFINITY;
    int argmax_n = 0;

    // Iterate over all N points, compute FC output, track max
    for (int n = 0; n < N; n++) {
        float val = bias;
        for (int di = 0; di < D_in; di++) {
            val += to_float(x[batch * N * D_in + n * D_in + di]) * W[d_out * D_in + di];
        }
        if (val > max_val) {
            max_val = val;
            argmax_n = n;
        }
    }

    out[idx] = from_float(max_val);
    argmax_indices[idx] = argmax_n;
}

// Backward: grad_x, grad_W, grad_b are always float32 (atomicAdd requires fp32)
// grad_out and x may be bf16
__global__ void fc_max_backward_kernel(
    float* __restrict__ grad_x,             // (B, N, D_in) - always float32 for atomicAdd
    float* __restrict__ grad_W,             // (D_out, D_in) - always float32
    float* __restrict__ grad_b,             // (D_out) - always float32
    const precision_t* __restrict__ grad_out,         // (B, D_out)
    const precision_t* __restrict__ x,                // (B, N, D_in)
    const float* __restrict__ W,            // (D_out, D_in) - always float32
    const int* __restrict__ argmax_indices, // (B, D_out)
    int B, int N, int D_in, int D_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * D_out) return;

    int batch = idx / D_out;
    int d_out = idx % D_out;

    float g_out = to_float(grad_out[idx]);
    int argmax_n = argmax_indices[idx];

    // grad_b[d_out] += g_out
    atomicAdd(&grad_b[d_out], g_out);

    // Backprop through FC at argmax position only
    for (int di = 0; di < D_in; di++) {
        int x_idx = batch * N * D_in + argmax_n * D_in + di;
        int w_idx = d_out * D_in + di;

        // grad_W[d_out, di] += g_out * x[batch, argmax_n, di]
        atomicAdd(&grad_W[w_idx], g_out * to_float(x[x_idx]));

        // grad_x[batch, argmax_n, di] += g_out * W[d_out, di]
        atomicAdd(&grad_x[x_idx], g_out * W[w_idx]);
    }
}



#endif // PUFFERLIB_KERNELS_CU
