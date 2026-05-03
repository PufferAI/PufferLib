// Removed vector dependency for MinGRU activations - now uses raw pointers

#ifndef PUFFERLIB_MODELS_CU
#define PUFFERLIB_MODELS_CU

#include <cstdint>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "kernels.cu"

// Signatures used by encoder and decoder. Writing custom nets in 4.0 requires a fair bit of code,
// because you are responsible for defining your own activation and gradient buffers.
// In practice, this is fairly simple. See our Encoder and Decoder for examples.
// You probably only ever need a custom Encoder
typedef void (*init_weights_fn)(void* weights, ulong* seed, cudaStream_t stream);
typedef void (*reg_params_fn)(void* weights, Allocator* alloc);
typedef void (*reg_train_fn)(void* weights, void* buf, Allocator* acts, Allocator* grads, int B_TT);
typedef void (*reg_rollout_fn)(void* weights, void* buf, Allocator* alloc, int B);
typedef void* (*create_weights_fn)(void* self);
typedef void  (*free_weights_fn)(void* weights);
typedef void  (*free_activations_fn)(void* activations);
typedef PrecisionTensor (*forward_fn)(void* weights, void* activations, PrecisionTensor input, cudaStream_t stream);
typedef void (*encoder_backward_fn)(void* weights, void* activations,
    PrecisionTensor grad, cudaStream_t stream);
typedef PrecisionTensor (*decoder_backward_fn)(void* weights, void* activations,
    FloatTensor grad_logits, FloatTensor grad_logstd, FloatTensor grad_value, cudaStream_t stream);
typedef PrecisionTensor (*network_forward_fn)(void* weights, PrecisionTensor x,
    PrecisionTensor state, void* activations, cudaStream_t stream);
typedef PrecisionTensor (*network_forward_train_fn)(void* weights, PrecisionTensor x,
    PrecisionTensor state, void* activations, cudaStream_t stream);
typedef PrecisionTensor (*network_backward_fn)(void* weights,
    PrecisionTensor grad, void* activations, cudaStream_t stream);

struct Encoder {
    forward_fn forward;
    encoder_backward_fn backward;
    init_weights_fn init_weights;
    reg_params_fn reg_params;
    reg_train_fn reg_train;
    reg_rollout_fn reg_rollout;
    create_weights_fn create_weights;
    free_weights_fn free_weights;
    free_activations_fn free_activations;
    int in_dim, out_dim;
    size_t activation_size;  // sizeof(EncoderActivations) or custom override
};

struct Decoder {
    forward_fn forward;
    decoder_backward_fn backward;
    init_weights_fn init_weights;
    reg_params_fn reg_params;
    reg_train_fn reg_train;
    reg_rollout_fn reg_rollout;
    create_weights_fn create_weights;
    free_weights_fn free_weights;
    free_activations_fn free_activations;
    int hidden_dim, output_dim;
    bool continuous;
};

struct Network {
    network_forward_fn forward;
    network_forward_train_fn forward_train;
    network_backward_fn backward;
    init_weights_fn init_weights;
    reg_params_fn reg_params;
    reg_train_fn reg_train;
    reg_rollout_fn reg_rollout;
    create_weights_fn create_weights;
    free_weights_fn free_weights;
    free_activations_fn free_activations;
    int hidden, num_layers, horizon;
};

struct EncoderWeights {
    PrecisionTensor weight;
    int in_dim, out_dim;
};

struct EncoderActivations {
    PrecisionTensor out, saved_input, wgrad_scratch;
};

// The core of 4.0 is the MinGRU fused scan operation. This allows us to parallelize
// training across the sequence dimension and scale to longer sequences
__device__ __forceinline__ void log_coeffs_and_values_fwd(float gate, float hidden,
        float* log_coeff_out, float* log_value_out) {
    float abs_gate = fabsf(gate);
    float sp_neg = log1pf(expf(-abs_gate));
    float softplus_gate = (gate >= 0.0f) ? gate + sp_neg : sp_neg;
    float softplus_neg_gate = (gate >= 0.0f) ? sp_neg : -gate + sp_neg;
    *log_coeff_out = -softplus_gate;
    float log_tilde_h = (hidden >= 0.0f) ? logf(hidden + 0.5f) : -softplus_fwd(-hidden);
    *log_value_out = -softplus_neg_gate + log_tilde_h;
}

__device__ __forceinline__ void log_coeffs_and_values_bwd(float grad_log_coeffs, float grad_log_values,
        float gate, float hidden, float* grad_gate_out, float* grad_hidden_out) {
    float sig_gate = sigmoid(gate);
    *grad_gate_out = -grad_log_coeffs * sig_gate + grad_log_values * (1.0f - sig_gate);
    *grad_hidden_out = (hidden >= 0.0f) ? grad_log_values / (hidden + 0.5f) : grad_log_values * sigmoid(-hidden);
}

__global__ void mingru_gate(precision_t* out, precision_t* next_state,
        const precision_t* combined, const precision_t* state_in,
        const precision_t* x_in, int H, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * H;
    if (idx >= N) {
        return;
    }

    int b = idx / H;
    int h = idx % H;

    // combined = linear(x_in) = (B, H) -> (B, 3*H)
    int combined_base = b * 3 * H;
    float hidden = to_float(combined[combined_base + h]);
    float gate = to_float(combined[combined_base + H + h]);
    float proj = to_float(combined[combined_base + 2*H + h]);
    float state = to_float(state_in[idx]);
    float x = to_float(x_in[idx]);

    // mingru_gate computation
    float gate_sigmoid = sigmoid(gate);
    float hidden_tilde = (hidden >= 0.0f) ? hidden + 0.5f : fast_sigmoid(hidden);
    float mingru_out = lerp(state, hidden_tilde, gate_sigmoid);

    // next_state is mingru_out (for recurrence)
    next_state[idx] = from_float(mingru_out);

    // Highway connection: sigmoid(proj) * mingru_out + (1 - sigmoid(proj)) * x (highway gate)
    float proj_sigmoid = sigmoid(proj);
    out[idx] = from_float(proj_sigmoid * mingru_out + (1.0f - proj_sigmoid) * x);
}

// Prefix scan buffers
struct PrefixScan {
    precision_t* combined_ptr = nullptr;
    precision_t* state_ptr = nullptr;
    precision_t* input_ptr = nullptr;  // (B, T, H) original input before projection (for highway gate)
    int B = 0, T = 0, H = 0;
    FloatTensor a_star, s_vals, log_values_buf;
    PrecisionTensor out, next_state;
    PrecisionTensor grad_combined, grad_state;
    PrecisionTensor grad_input;        // (B, T, H) highway gate gradient w.r.t. input
};

// Checkpointing trades off partial recomputation for memory bandwidth.
#define CHECKPOINT_INTERVAL 4
constexpr int MINGRU_SCAN_VEC128_WIDTH = 16 / sizeof(precision_t);
constexpr int MINGRU_SCAN_VEC64_WIDTH = 8 / sizeof(precision_t);

template<int CKPT_INTERVAL>
__device__ __forceinline__ void mingru_scan_forward_ckpt_tuned_body(PrefixScan scan) {
    int T_seq = scan.T, H = scan.H, B = scan.B;
    precision_t* __restrict__ out = scan.out.data;
    precision_t* __restrict__ next_state = scan.next_state.data;
    float* __restrict__ a_star_buf = scan.a_star.data;
    float* __restrict__ s_buf = scan.s_vals.data;
    float* __restrict__ log_values_buf = scan.log_values_buf.data;
    const precision_t* __restrict__ combined = scan.combined_ptr;
    const precision_t* __restrict__ state = scan.state_ptr;
    const precision_t* __restrict__ input = scan.input_ptr;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) {
        return;
    }

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

    float scan_result = 0.0f;
    int out_curr = out_base;
    int t_offset = 0;

    for (int t = 1; t <= T_seq; t++) {
        float hidden_val = to_float(combined_h_base[t_offset]);
        float gate_val = to_float(combined_g_base[t_offset]);
        float proj_val = to_float(combined_p_base[t_offset]);
        int input_idx = out_base + (t - 1) * H;
        float x_val = to_float(input[input_idx]);

        float log_coeff_val;
        log_coeffs_and_values_fwd(gate_val, hidden_val, &log_coeff_val, &log_value);

        a_star += log_coeff_val;

        float z = log_value - a_star;
        s = logaddexp(s, z);

        scan_result = __expf(a_star + s);
        float proj_sigmoid = sigmoid(proj_val);

        out[out_curr] = from_float(proj_sigmoid * scan_result + (1.0f - proj_sigmoid) * x_val);

        buf_curr += H;
        out_curr += H;
        t_offset += H3;

        if (t % CKPT_INTERVAL == 0) {
            a_star_buf[buf_curr] = a_star;
            s_buf[buf_curr] = s;
            log_values_buf[buf_curr] = log_value;
        }
    }

    next_state[bH + h] = from_float(scan_result);
}

__global__ void mingru_scan_forward(PrefixScan scan) {
    mingru_scan_forward_ckpt_tuned_body<CHECKPOINT_INTERVAL>(scan);
}

// Reads sparse checkpoints from forward pass, recomputes intermediate values in chunks
template<int CKPT_INTERVAL>
__device__ __forceinline__ void mingru_scan_backward_ckpt_tuned_body(PrefixScan scan,
        const precision_t* __restrict__ grad_out,
        const precision_t* __restrict__ grad_next_state) {
    int T_seq = scan.T, H = scan.H, B = scan.B;
    precision_t* __restrict__ grad_combined = scan.grad_combined.data;
    precision_t* __restrict__ grad_state = scan.grad_state.data;
    precision_t* __restrict__ grad_input = scan.grad_input.data;
    const precision_t* __restrict__ combined = scan.combined_ptr;
    const precision_t* __restrict__ state = scan.state_ptr;
    const precision_t* __restrict__ input = scan.input_ptr;
    const float* __restrict__ a_star_buf = scan.a_star.data;
    const float* __restrict__ s_buf = scan.s_vals.data;
    const float* __restrict__ log_values_buf = scan.log_values_buf.data;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) {
        return;
    }

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

    // Backward recompute must start each chunk at a stored checkpoint index.
    // Floor-to-multiple handles ragged tails (e.g. with CKPT_INTERVAL=4: T=10 -> starts 8,4,0).
    for (int chunk_end = T_seq; chunk_end > 0;) {
        int chunk_start = ((chunk_end - 1) / CKPT_INTERVAL) * CKPT_INTERVAL;
        int chunk_len = chunk_end - chunk_start;

        // Chunk storage in registers
        float chunk_a_star[CKPT_INTERVAL];
        float chunk_s[CKPT_INTERVAL];
        float chunk_log_values[CKPT_INTERVAL];
        float chunk_hidden[CKPT_INTERVAL];
        float chunk_gate[CKPT_INTERVAL];

        // Load checkpoint from global memory
        int ckpt_buf_idx = buf_base + chunk_start * H;
        float recomp_a_star = a_star_buf[ckpt_buf_idx];
        float recomp_s = s_buf[ckpt_buf_idx];
        float recomp_log_value = log_values_buf[ckpt_buf_idx];

        // Phase 1: recompute per-timestep values from checkpoint start to chunk end.
        for (int chunk_i = 0; chunk_i < chunk_len; ++chunk_i) {
            int t = chunk_start + 1 + chunk_i;
            int t_offset = (t - 1) * H3;
            float hidden_val = to_float(combined_h_base[t_offset]);
            float gate_val = to_float(combined_g_base[t_offset]);

            float log_coeff_val;
            log_coeffs_and_values_fwd(gate_val, hidden_val, &log_coeff_val, &recomp_log_value);
            recomp_a_star += log_coeff_val;

            float z = recomp_log_value - recomp_a_star;
            recomp_s = logaddexp(recomp_s, z);

            chunk_a_star[chunk_i] = recomp_a_star;
            chunk_s[chunk_i] = recomp_s;
            chunk_log_values[chunk_i] = recomp_log_value;
            chunk_hidden[chunk_i] = hidden_val;
            chunk_gate[chunk_i] = gate_val;
        }

        // Phase 2: backprop through the chunk in reverse time order.
        for (int chunk_i = chunk_len - 1; chunk_i >= 0; --chunk_i) {
            int t = chunk_start + 1 + chunk_i;
            int t_offset = (t - 1) * H3;
            const bool is_last_t = (t == T_seq);

            float a_star_t = chunk_a_star[chunk_i];
            float s_t = chunk_s[chunk_i];
            float log_value_t = chunk_log_values[chunk_i];
            float hidden_val = chunk_hidden[chunk_i];
            float gate_val = chunk_gate[chunk_i];

            float proj_val = to_float(combined_p_base[t_offset]);
            int input_idx = out_base + (t - 1) * H;
            float x_val = to_float(input[input_idx]);

            float scan_result = __expf(a_star_t + s_t);
            float z = log_value_t - a_star_t;

            float grad_out_val = to_float(grad_out[input_idx]);
            float grad_scan_from_next = is_last_t ? to_float(grad_next_state[state_idx]) : 0.0f;
            float proj_sigmoid = sigmoid(proj_val);

            // Highway gate gradients: out = sigmoid(proj) * scan_result + (1 - sigmoid(proj)) * x
            float grad_scan_result = grad_scan_from_next + grad_out_val * proj_sigmoid;
            float grad_proj = grad_out_val * (scan_result - x_val) * proj_sigmoid * (1.0f - proj_sigmoid);
            grad_input[input_idx] = from_float(grad_out_val * (1.0f - proj_sigmoid));

            float grad_log_h = grad_scan_result * scan_result;
            float grad_s = grad_log_h;

            if (is_last_t) {
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

        chunk_end = chunk_start;
    }

    int ckpt_0_idx = buf_base;
    float a_star_0 = a_star_buf[ckpt_0_idx];
    float s_0 = s_buf[ckpt_0_idx];
    float log_value_0 = log_values_buf[ckpt_0_idx];
    float z_0 = log_value_0 - a_star_0;
    acc = acc * __expf(s_0 - s_val_next);
    float grad_z_0 = acc * __expf(z_0 - s_0);

    grad_state[state_idx] = from_float(grad_z_0 / to_float(state[state_idx]));
}

__global__ void mingru_scan_backward(PrefixScan scan,
        const precision_t* __restrict__ grad_out,
        const precision_t* __restrict__ grad_next_state) {
    mingru_scan_backward_ckpt_tuned_body<CHECKPOINT_INTERVAL>(scan, grad_out, grad_next_state);
}

// Shared packed/vector helper layer for log vec32/vec64/vec128 kernels.

__device__ __forceinline__ float2 scan_load_pair(const float* ptr) {
    return make_float2(ptr[0], ptr[1]);
}

__device__ __forceinline__ void scan_store_pair(float* ptr, float2 v) {
    ptr[0] = v.x;
    ptr[1] = v.y;
}

__device__ __forceinline__ float2 scan_logaddexp_pair(float2 a, float2 b) {
    return make_float2(logaddexp(a.x, b.x), logaddexp(a.y, b.y));
}

__device__ __forceinline__ void scan_log_coeffs_and_values_fwd_pair(
        float2 gate, float2 hidden, float2* log_coeff_out, float2* log_value_io) {
    log_coeffs_and_values_fwd(gate.x, hidden.x, &log_coeff_out->x, &log_value_io->x);
    log_coeffs_and_values_fwd(gate.y, hidden.y, &log_coeff_out->y, &log_value_io->y);
}

__device__ __forceinline__ void scan_log_coeffs_and_values_bwd_pair(
        float2 grad_log_coeffs, float2 grad_log_values, float2 gate, float2 hidden,
        float2* grad_gate_out, float2* grad_hidden_out) {
    log_coeffs_and_values_bwd(
        grad_log_coeffs.x, grad_log_values.x, gate.x, hidden.x,
        &grad_gate_out->x, &grad_hidden_out->x);
    log_coeffs_and_values_bwd(
        grad_log_coeffs.y, grad_log_values.y, gate.y, hidden.y,
        &grad_gate_out->y, &grad_hidden_out->y);
}

template<int VEC_WIDTH>
__device__ __forceinline__ void scan_load_precision_vec(const precision_t* ptr, float* out) {
#ifdef PRECISION_FLOAT
    static_assert(VEC_WIDTH == 2 || VEC_WIDTH == 4,
        "float build supports vec64/vec128 widths");
    if constexpr (VEC_WIDTH == 2) {
        float2 v = *reinterpret_cast<const float2*>(ptr);
        out[0] = v.x;
        out[1] = v.y;
    } else {
        float4 v = *reinterpret_cast<const float4*>(ptr);
        out[0] = v.x;
        out[1] = v.y;
        out[2] = v.z;
        out[3] = v.w;
    }
#else
    static_assert(VEC_WIDTH == 2 || VEC_WIDTH == 4 || VEC_WIDTH == 8,
        "bf16 build supports vec32/vec64/vec128 widths");
    if constexpr (VEC_WIDTH == 2) {
        uint32_t raw = *reinterpret_cast<const uint32_t*>(ptr);
        const precision_t* bf = reinterpret_cast<const precision_t*>(&raw);
        out[0] = to_float(bf[0]);
        out[1] = to_float(bf[1]);
    } else if constexpr (VEC_WIDTH == 4) {
        uint2 raw = *reinterpret_cast<const uint2*>(ptr);
        const precision_t* bf = reinterpret_cast<const precision_t*>(&raw);
        #pragma unroll
        for (int lane = 0; lane < 4; lane++) {
            out[lane] = to_float(bf[lane]);
        }
    } else {
        uint4 raw = *reinterpret_cast<const uint4*>(ptr);
        const precision_t* bf = reinterpret_cast<const precision_t*>(&raw);
        #pragma unroll
        for (int lane = 0; lane < 8; lane++) {
            out[lane] = to_float(bf[lane]);
        }
    }
#endif
}

template<int VEC_WIDTH>
__device__ __forceinline__ void scan_store_precision_vec(precision_t* ptr, const float* in) {
#ifdef PRECISION_FLOAT
    static_assert(VEC_WIDTH == 2 || VEC_WIDTH == 4,
        "float build supports vec64/vec128 widths");
    if constexpr (VEC_WIDTH == 2) {
        *reinterpret_cast<float2*>(ptr) = make_float2(in[0], in[1]);
    } else {
        *reinterpret_cast<float4*>(ptr) = make_float4(in[0], in[1], in[2], in[3]);
    }
#else
    static_assert(VEC_WIDTH == 2 || VEC_WIDTH == 4 || VEC_WIDTH == 8,
        "bf16 build supports vec32/vec64/vec128 widths");
    if constexpr (VEC_WIDTH == 2) {
        alignas(4) precision_t tmp[2];
        tmp[0] = from_float(in[0]);
        tmp[1] = from_float(in[1]);
        *reinterpret_cast<uint32_t*>(ptr) = *reinterpret_cast<const uint32_t*>(tmp);
    } else if constexpr (VEC_WIDTH == 4) {
        alignas(8) precision_t tmp[4];
        #pragma unroll
        for (int lane = 0; lane < 4; lane++) {
            tmp[lane] = from_float(in[lane]);
        }
        *reinterpret_cast<uint2*>(ptr) = *reinterpret_cast<const uint2*>(tmp);
    } else {
        alignas(16) precision_t tmp[8];
        #pragma unroll
        for (int lane = 0; lane < 8; lane++) {
            tmp[lane] = from_float(in[lane]);
        }
        *reinterpret_cast<uint4*>(ptr) = *reinterpret_cast<const uint4*>(tmp);
    }
#endif
}

template<int VEC_WIDTH>
__device__ __forceinline__ void scan_load_float_vec(const float* ptr, float* out) {
#ifdef PRECISION_FLOAT
    static_assert(VEC_WIDTH == 2 || VEC_WIDTH == 4,
        "float build supports vec64/vec128 widths");
#else
    static_assert(VEC_WIDTH == 2 || VEC_WIDTH == 4 || VEC_WIDTH == 8,
        "bf16 build supports vec32/vec64/vec128 widths");
#endif
    if constexpr (VEC_WIDTH == 2) {
        float2 v = *reinterpret_cast<const float2*>(ptr);
        out[0] = v.x;
        out[1] = v.y;
    } else if constexpr (VEC_WIDTH == 4) {
        float4 v = *reinterpret_cast<const float4*>(ptr);
        out[0] = v.x;
        out[1] = v.y;
        out[2] = v.z;
        out[3] = v.w;
    } else {
        float4 lo = *reinterpret_cast<const float4*>(ptr);
        float4 hi = *reinterpret_cast<const float4*>(ptr + 4);
        out[0] = lo.x;
        out[1] = lo.y;
        out[2] = lo.z;
        out[3] = lo.w;
        out[4] = hi.x;
        out[5] = hi.y;
        out[6] = hi.z;
        out[7] = hi.w;
    }
}

template<int VEC_WIDTH>
__device__ __forceinline__ void scan_store_float_vec(float* ptr, const float* in) {
#ifdef PRECISION_FLOAT
    static_assert(VEC_WIDTH == 2 || VEC_WIDTH == 4,
        "float build supports vec64/vec128 widths");
#else
    static_assert(VEC_WIDTH == 2 || VEC_WIDTH == 4 || VEC_WIDTH == 8,
        "bf16 build supports vec32/vec64/vec128 widths");
#endif
    if constexpr (VEC_WIDTH == 2) {
        *reinterpret_cast<float2*>(ptr) = make_float2(in[0], in[1]);
    } else if constexpr (VEC_WIDTH == 4) {
        *reinterpret_cast<float4*>(ptr) = make_float4(in[0], in[1], in[2], in[3]);
    } else {
        *reinterpret_cast<float4*>(ptr) = make_float4(in[0], in[1], in[2], in[3]);
        *reinterpret_cast<float4*>(ptr + 4) = make_float4(in[4], in[5], in[6], in[7]);
    }
}

template<int CKPT_INTERVAL>
__global__ void mingru_scan_forward_ckpt_tuned(PrefixScan scan) {
    mingru_scan_forward_ckpt_tuned_body<CKPT_INTERVAL>(scan);
}

template<int CKPT_INTERVAL>
__global__ void mingru_scan_backward_ckpt_tuned(PrefixScan scan,
        const precision_t* __restrict__ grad_out,
        const precision_t* __restrict__ grad_next_state) {
    mingru_scan_backward_ckpt_tuned_body<CKPT_INTERVAL>(scan, grad_out, grad_next_state);
}

template<int CKPT_INTERVAL, int VEC_WIDTH>
__device__ __forceinline__ void mingru_scan_forward_ckpt_tuned_vec_body(PrefixScan scan) {
    static_assert((VEC_WIDTH % 2) == 0, "vectorized kernels require even width");
#ifdef PRECISION_FLOAT
    static_assert(VEC_WIDTH == MINGRU_SCAN_VEC64_WIDTH || VEC_WIDTH == MINGRU_SCAN_VEC128_WIDTH,
        "float build supports vec64/vec128 widths");
#else
    static_assert(VEC_WIDTH == 2 || VEC_WIDTH == MINGRU_SCAN_VEC64_WIDTH || VEC_WIDTH == MINGRU_SCAN_VEC128_WIDTH,
        "bf16 build supports vec32/vec64/vec128 widths");
#endif

    int T_seq = scan.T, H = scan.H, B = scan.B;
    int HW = H / VEC_WIDTH;
    if (HW == 0) {
        return;
    }

    precision_t* __restrict__ out = scan.out.data;
    precision_t* __restrict__ next_state = scan.next_state.data;
    float* __restrict__ a_star_buf = scan.a_star.data;
    float* __restrict__ s_buf = scan.s_vals.data;
    float* __restrict__ log_values_buf = scan.log_values_buf.data;
    const precision_t* __restrict__ combined = scan.combined_ptr;
    const precision_t* __restrict__ state = scan.state_ptr;
    const precision_t* __restrict__ input = scan.input_ptr;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * HW) {
        return;
    }

    int b = idx / HW;
    int hw = idx % HW;
    int h = hw * VEC_WIDTH;
    int bH = b * H;
    int bTH = b * T_seq * H;
    int cbase = 3 * bTH;
    int H3 = 3 * H;
    int H2 = 2 * H;
    int out_base = bTH + h;

    const precision_t* combined_h_base = &combined[cbase + h];
    const precision_t* combined_g_base = &combined[cbase + H + h];
    const precision_t* combined_p_base = &combined[cbase + H2 + h];

    float h_state[VEC_WIDTH];
    float a_star[VEC_WIDTH];
    float s[VEC_WIDTH];
    float log_value[VEC_WIDTH];
    float scan_result[VEC_WIDTH];
    float hidden[VEC_WIDTH];
    float gate[VEC_WIDTH];
    float proj[VEC_WIDTH];
    float x[VEC_WIDTH];
    float out_chunk[VEC_WIDTH];

    scan_load_precision_vec<VEC_WIDTH>(&state[bH + h], h_state);
    #pragma unroll
    for (int lane = 0; lane < VEC_WIDTH; lane += 2) {
        float2 state_v = scan_load_pair(&h_state[lane]);
        float2 s_v = make_float2(__logf(state_v.x), __logf(state_v.y));
        scan_store_pair(&a_star[lane], make_float2(0.0f, 0.0f));
        scan_store_pair(&s[lane], s_v);
        scan_store_pair(&log_value[lane], s_v);
        scan_store_pair(&scan_result[lane], state_v);
    }

    int T_out = T_seq + 1;
    int buf_base = b * T_out * H + h;
    scan_store_float_vec<VEC_WIDTH>(&a_star_buf[buf_base], a_star);
    scan_store_float_vec<VEC_WIDTH>(&s_buf[buf_base], s);
    scan_store_float_vec<VEC_WIDTH>(&log_values_buf[buf_base], log_value);

    int out_curr = out_base;
    int t_offset = 0;
    for (int t = 1; t <= T_seq; t++) {
        scan_load_precision_vec<VEC_WIDTH>(&combined_h_base[t_offset], hidden);
        scan_load_precision_vec<VEC_WIDTH>(&combined_g_base[t_offset], gate);
        scan_load_precision_vec<VEC_WIDTH>(&combined_p_base[t_offset], proj);
        scan_load_precision_vec<VEC_WIDTH>(&input[out_curr], x);

        #pragma unroll
        for (int lane = 0; lane < VEC_WIDTH; lane += 2) {
            float2 gate_v = scan_load_pair(&gate[lane]);
            float2 hidden_v = scan_load_pair(&hidden[lane]);
            float2 proj_v = scan_load_pair(&proj[lane]);
            float2 x_v = scan_load_pair(&x[lane]);
            float2 a_star_v = scan_load_pair(&a_star[lane]);
            float2 s_v = scan_load_pair(&s[lane]);
            float2 log_value_v = scan_load_pair(&log_value[lane]);
            float2 log_coeff_v;
            scan_log_coeffs_and_values_fwd_pair(gate_v, hidden_v, &log_coeff_v, &log_value_v);
            a_star_v.x += log_coeff_v.x;
            a_star_v.y += log_coeff_v.y;
            float2 z_v = make_float2(log_value_v.x - a_star_v.x, log_value_v.y - a_star_v.y);
            s_v = scan_logaddexp_pair(s_v, z_v);
            float2 scan_result_v = make_float2(__expf(a_star_v.x + s_v.x), __expf(a_star_v.y + s_v.y));
            float2 proj_sigmoid_v = make_float2(sigmoid(proj_v.x), sigmoid(proj_v.y));
            float2 out_v = make_float2(
                proj_sigmoid_v.x * scan_result_v.x + (1.0f - proj_sigmoid_v.x) * x_v.x,
                proj_sigmoid_v.y * scan_result_v.y + (1.0f - proj_sigmoid_v.y) * x_v.y);
            scan_store_pair(&a_star[lane], a_star_v);
            scan_store_pair(&s[lane], s_v);
            scan_store_pair(&log_value[lane], log_value_v);
            scan_store_pair(&scan_result[lane], scan_result_v);
            scan_store_pair(&out_chunk[lane], out_v);
        }
        scan_store_precision_vec<VEC_WIDTH>(&out[out_curr], out_chunk);

        if ((t % CKPT_INTERVAL) == 0) {
            int buf_idx = buf_base + t * H;
            scan_store_float_vec<VEC_WIDTH>(&a_star_buf[buf_idx], a_star);
            scan_store_float_vec<VEC_WIDTH>(&s_buf[buf_idx], s);
            scan_store_float_vec<VEC_WIDTH>(&log_values_buf[buf_idx], log_value);
        }

        out_curr += H;
        t_offset += H3;
    }

    scan_store_precision_vec<VEC_WIDTH>(&next_state[bH + h], scan_result);
}

template<int CKPT_INTERVAL, int VEC_WIDTH>
__device__ __forceinline__ void mingru_scan_backward_ckpt_tuned_vec_body(
        PrefixScan scan, const precision_t* __restrict__ grad_out,
        const precision_t* __restrict__ grad_next_state) {
    static_assert((VEC_WIDTH % 2) == 0, "vectorized kernels require even width");
#ifdef PRECISION_FLOAT
    static_assert(VEC_WIDTH == MINGRU_SCAN_VEC64_WIDTH || VEC_WIDTH == MINGRU_SCAN_VEC128_WIDTH,
        "float build supports vec64/vec128 widths");
#else
    static_assert(VEC_WIDTH == 2 || VEC_WIDTH == MINGRU_SCAN_VEC64_WIDTH || VEC_WIDTH == MINGRU_SCAN_VEC128_WIDTH,
        "bf16 build supports vec32/vec64/vec128 widths");
#endif

    int T_seq = scan.T, H = scan.H, B = scan.B;
    int HW = H / VEC_WIDTH;
    if (HW == 0) {
        return;
    }

    precision_t* __restrict__ grad_combined = scan.grad_combined.data;
    precision_t* __restrict__ grad_state = scan.grad_state.data;
    precision_t* __restrict__ grad_input = scan.grad_input.data;
    const precision_t* __restrict__ combined = scan.combined_ptr;
    const precision_t* __restrict__ state = scan.state_ptr;
    const precision_t* __restrict__ input = scan.input_ptr;
    const float* __restrict__ a_star_buf = scan.a_star.data;
    const float* __restrict__ s_buf = scan.s_vals.data;
    const float* __restrict__ log_values_buf = scan.log_values_buf.data;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * HW) {
        return;
    }

    int b = idx / HW;
    int hw = idx % HW;
    int h = hw * VEC_WIDTH;

    int bH = b * H;
    int bTH = b * T_seq * H;
    int cbase = 3 * bTH;
    int H3 = 3 * H;
    int H2 = 2 * H;
    int out_base = bTH + h;

    const precision_t* combined_h_base = &combined[cbase + h];
    const precision_t* combined_g_base = &combined[cbase + H + h];
    const precision_t* combined_p_base = &combined[cbase + H2 + h];

    precision_t* grad_combined_h_base = &grad_combined[cbase + h];
    precision_t* grad_combined_g_base = &grad_combined[cbase + H + h];
    precision_t* grad_combined_p_base = &grad_combined[cbase + H2 + h];

    int T_out = T_seq + 1;
    int buf_base = b * T_out * H + h;

    float acc[VEC_WIDTH];
    float s_val_next[VEC_WIDTH];
    float carry_grad_a[VEC_WIDTH];
    float grad_next[VEC_WIDTH];
    #pragma unroll
    for (int lane = 0; lane < VEC_WIDTH; lane++) {
        acc[lane] = 0.0f;
        s_val_next[lane] = 0.0f;
        carry_grad_a[lane] = 0.0f;
    }
    scan_load_precision_vec<VEC_WIDTH>(&grad_next_state[bH + h], grad_next);

    for (int chunk_end = T_seq; chunk_end > 0;) {
        int chunk_start = ((chunk_end - 1) / CKPT_INTERVAL) * CKPT_INTERVAL;
        int chunk_len = chunk_end - chunk_start;

        float chunk_a_star[CKPT_INTERVAL][VEC_WIDTH];
        float chunk_s[CKPT_INTERVAL][VEC_WIDTH];
        float chunk_log_values[CKPT_INTERVAL][VEC_WIDTH];

        float recomp_a_star[VEC_WIDTH];
        float recomp_s[VEC_WIDTH];
        float recomp_log_value[VEC_WIDTH];
        int ckpt_buf_idx = buf_base + chunk_start * H;
        scan_load_float_vec<VEC_WIDTH>(&a_star_buf[ckpt_buf_idx], recomp_a_star);
        scan_load_float_vec<VEC_WIDTH>(&s_buf[ckpt_buf_idx], recomp_s);
        scan_load_float_vec<VEC_WIDTH>(&log_values_buf[ckpt_buf_idx], recomp_log_value);

        // Phase 1: recompute per-timestep values from checkpoint start to chunk end.
        for (int chunk_i = 0; chunk_i < chunk_len; ++chunk_i) {
            int t = chunk_start + 1 + chunk_i;
            int t_offset = (t - 1) * H3;
            float hidden[VEC_WIDTH];
            float gate[VEC_WIDTH];
            scan_load_precision_vec<VEC_WIDTH>(&combined_h_base[t_offset], hidden);
            scan_load_precision_vec<VEC_WIDTH>(&combined_g_base[t_offset], gate);
            #pragma unroll
            for (int lane = 0; lane < VEC_WIDTH; lane += 2) {
                float2 gate_v = scan_load_pair(&gate[lane]);
                float2 hidden_v = scan_load_pair(&hidden[lane]);
                float2 recomp_a_star_v = scan_load_pair(&recomp_a_star[lane]);
                float2 recomp_s_v = scan_load_pair(&recomp_s[lane]);
                float2 recomp_log_value_v = scan_load_pair(&recomp_log_value[lane]);
                float2 log_coeff_v;
                scan_log_coeffs_and_values_fwd_pair(gate_v, hidden_v, &log_coeff_v, &recomp_log_value_v);
                recomp_a_star_v.x += log_coeff_v.x;
                recomp_a_star_v.y += log_coeff_v.y;
                float2 z_v = make_float2(
                    recomp_log_value_v.x - recomp_a_star_v.x,
                    recomp_log_value_v.y - recomp_a_star_v.y);
                recomp_s_v = scan_logaddexp_pair(recomp_s_v, z_v);
                scan_store_pair(&recomp_a_star[lane], recomp_a_star_v);
                scan_store_pair(&recomp_s[lane], recomp_s_v);
                scan_store_pair(&recomp_log_value[lane], recomp_log_value_v);
                scan_store_pair(&chunk_a_star[chunk_i][lane], recomp_a_star_v);
                scan_store_pair(&chunk_s[chunk_i][lane], recomp_s_v);
                scan_store_pair(&chunk_log_values[chunk_i][lane], recomp_log_value_v);
            }
        }

        // Phase 2: backprop through the chunk in reverse time order.
        for (int chunk_i = chunk_len - 1; chunk_i >= 0; --chunk_i) {
            int t = chunk_start + 1 + chunk_i;
            int t_offset = (t - 1) * H3;
            int input_idx = out_base + (t - 1) * H;
            const bool is_last_t = (t == T_seq);

            float proj[VEC_WIDTH];
            float x[VEC_WIDTH];
            float grad_out_chunk[VEC_WIDTH];
            float grad_hidden[VEC_WIDTH];
            float grad_gate[VEC_WIDTH];
            float grad_proj[VEC_WIDTH];
            float grad_in[VEC_WIDTH];
            float hidden[VEC_WIDTH];
            float gate[VEC_WIDTH];
            scan_load_precision_vec<VEC_WIDTH>(&combined_p_base[t_offset], proj);
            scan_load_precision_vec<VEC_WIDTH>(&combined_h_base[t_offset], hidden);
            scan_load_precision_vec<VEC_WIDTH>(&combined_g_base[t_offset], gate);
            scan_load_precision_vec<VEC_WIDTH>(&input[input_idx], x);
            scan_load_precision_vec<VEC_WIDTH>(&grad_out[input_idx], grad_out_chunk);

            #pragma unroll
            for (int lane = 0; lane < VEC_WIDTH; lane += 2) {
                float2 a_star_t_v = scan_load_pair(&chunk_a_star[chunk_i][lane]);
                float2 s_t_v = scan_load_pair(&chunk_s[chunk_i][lane]);
                float2 log_value_t_v = scan_load_pair(&chunk_log_values[chunk_i][lane]);
                float2 scan_result_v = make_float2(
                    __expf(a_star_t_v.x + s_t_v.x),
                    __expf(a_star_t_v.y + s_t_v.y));
                float2 z_v = make_float2(
                    log_value_t_v.x - a_star_t_v.x,
                    log_value_t_v.y - a_star_t_v.y);

                float2 proj_v = scan_load_pair(&proj[lane]);
                float2 x_v = scan_load_pair(&x[lane]);
                float2 grad_out_v = scan_load_pair(&grad_out_chunk[lane]);
                float2 proj_sigmoid_v = make_float2(sigmoid(proj_v.x), sigmoid(proj_v.y));
                float2 grad_in_v = make_float2(
                    grad_out_v.x * (1.0f - proj_sigmoid_v.x),
                    grad_out_v.y * (1.0f - proj_sigmoid_v.y));
                float2 grad_proj_v = make_float2(
                    grad_out_v.x * (scan_result_v.x - x_v.x) * proj_sigmoid_v.x * (1.0f - proj_sigmoid_v.x),
                    grad_out_v.y * (scan_result_v.y - x_v.y) * proj_sigmoid_v.y * (1.0f - proj_sigmoid_v.y));
                scan_store_pair(&grad_in[lane], grad_in_v);
                scan_store_pair(&grad_proj[lane], grad_proj_v);

                float2 grad_scan_from_next_v = is_last_t
                    ? scan_load_pair(&grad_next[lane])
                    : make_float2(0.0f, 0.0f);
                float2 grad_scan_result_v = make_float2(
                    grad_scan_from_next_v.x + grad_out_v.x * proj_sigmoid_v.x,
                    grad_scan_from_next_v.y + grad_out_v.y * proj_sigmoid_v.y);
                float2 grad_log_h_v = make_float2(
                    grad_scan_result_v.x * scan_result_v.x,
                    grad_scan_result_v.y * scan_result_v.y);

                float2 acc_v;
                if (is_last_t) {
                    acc_v = grad_log_h_v;
                } else {
                    float2 acc_prev_v = scan_load_pair(&acc[lane]);
                    float2 s_val_next_v = scan_load_pair(&s_val_next[lane]);
                    acc_v = make_float2(
                        grad_log_h_v.x + acc_prev_v.x * __expf(s_t_v.x - s_val_next_v.x),
                        grad_log_h_v.y + acc_prev_v.y * __expf(s_t_v.y - s_val_next_v.y));
                }
                scan_store_pair(&acc[lane], acc_v);

                float2 grad_z_v = make_float2(
                    acc_v.x * __expf(z_v.x - s_t_v.x),
                    acc_v.y * __expf(z_v.y - s_t_v.y));
                scan_store_pair(&s_val_next[lane], s_t_v);

                float2 carry_grad_a_v = scan_load_pair(&carry_grad_a[lane]);
                float2 grad_a_v = make_float2(
                    grad_log_h_v.x + carry_grad_a_v.x - grad_z_v.x,
                    grad_log_h_v.y + carry_grad_a_v.y - grad_z_v.y);
                scan_store_pair(&carry_grad_a[lane], grad_a_v);

                float2 gate_v = scan_load_pair(&gate[lane]);
                float2 hidden_v = scan_load_pair(&hidden[lane]);
                float2 grad_gate_v;
                float2 grad_hidden_v;
                scan_log_coeffs_and_values_bwd_pair(
                    grad_a_v, grad_z_v, gate_v, hidden_v, &grad_gate_v, &grad_hidden_v);
                scan_store_pair(&grad_gate[lane], grad_gate_v);
                scan_store_pair(&grad_hidden[lane], grad_hidden_v);
            }

            scan_store_precision_vec<VEC_WIDTH>(&grad_combined_h_base[t_offset], grad_hidden);
            scan_store_precision_vec<VEC_WIDTH>(&grad_combined_g_base[t_offset], grad_gate);
            scan_store_precision_vec<VEC_WIDTH>(&grad_combined_p_base[t_offset], grad_proj);
            scan_store_precision_vec<VEC_WIDTH>(&grad_input[input_idx], grad_in);
        }

        chunk_end = chunk_start;
    }

    float a_star_0[VEC_WIDTH];
    float s_0[VEC_WIDTH];
    float log_value_0[VEC_WIDTH];
    float grad_state_chunk[VEC_WIDTH];
    float state_chunk[VEC_WIDTH];
    int ckpt_0_idx = buf_base;
    scan_load_float_vec<VEC_WIDTH>(&a_star_buf[ckpt_0_idx], a_star_0);
    scan_load_float_vec<VEC_WIDTH>(&s_buf[ckpt_0_idx], s_0);
    scan_load_float_vec<VEC_WIDTH>(&log_values_buf[ckpt_0_idx], log_value_0);
    scan_load_precision_vec<VEC_WIDTH>(&state[bH + h], state_chunk);
    #pragma unroll
    for (int lane = 0; lane < VEC_WIDTH; lane += 2) {
        float2 a_star_0_v = scan_load_pair(&a_star_0[lane]);
        float2 s_0_v = scan_load_pair(&s_0[lane]);
        float2 log_value_0_v = scan_load_pair(&log_value_0[lane]);
        float2 acc_v = scan_load_pair(&acc[lane]);
        float2 s_val_next_v = scan_load_pair(&s_val_next[lane]);
        float2 state_chunk_v = scan_load_pair(&state_chunk[lane]);

        float2 z_0_v = make_float2(
            log_value_0_v.x - a_star_0_v.x,
            log_value_0_v.y - a_star_0_v.y);
        acc_v.x = acc_v.x * __expf(s_0_v.x - s_val_next_v.x);
        acc_v.y = acc_v.y * __expf(s_0_v.y - s_val_next_v.y);
        float2 grad_z_0_v = make_float2(
            acc_v.x * __expf(z_0_v.x - s_0_v.x),
            acc_v.y * __expf(z_0_v.y - s_0_v.y));
        float2 grad_state_v = make_float2(
            grad_z_0_v.x / state_chunk_v.x,
            grad_z_0_v.y / state_chunk_v.y);
        scan_store_pair(&grad_state_chunk[lane], grad_state_v);
    }
    scan_store_precision_vec<VEC_WIDTH>(&grad_state[bH + h], grad_state_chunk);
}

#ifndef PRECISION_FLOAT
template<int CKPT_INTERVAL>
__global__ void mingru_scan_forward_ckpt_tuned_vec32(PrefixScan scan) {
    mingru_scan_forward_ckpt_tuned_vec_body<CKPT_INTERVAL, 2>(scan);
}

template<int CKPT_INTERVAL>
__global__ void mingru_scan_backward_ckpt_tuned_vec32(PrefixScan scan,
        const precision_t* __restrict__ grad_out,
        const precision_t* __restrict__ grad_next_state) {
    mingru_scan_backward_ckpt_tuned_vec_body<CKPT_INTERVAL, 2>(
        scan, grad_out, grad_next_state);
}
#endif

template<int CKPT_INTERVAL>
__global__ void mingru_scan_forward_ckpt_tuned_vec64(PrefixScan scan) {
    mingru_scan_forward_ckpt_tuned_vec_body<CKPT_INTERVAL, MINGRU_SCAN_VEC64_WIDTH>(scan);
}

template<int CKPT_INTERVAL>
__global__ void mingru_scan_backward_ckpt_tuned_vec64(PrefixScan scan,
        const precision_t* __restrict__ grad_out,
        const precision_t* __restrict__ grad_next_state) {
    mingru_scan_backward_ckpt_tuned_vec_body<CKPT_INTERVAL, MINGRU_SCAN_VEC64_WIDTH>(
        scan, grad_out, grad_next_state);
}

template<int CKPT_INTERVAL>
__global__ void mingru_scan_forward_ckpt_tuned_vec128(PrefixScan scan) {
    mingru_scan_forward_ckpt_tuned_vec_body<CKPT_INTERVAL, MINGRU_SCAN_VEC128_WIDTH>(scan);
}

template<int CKPT_INTERVAL>
__global__ void mingru_scan_backward_ckpt_tuned_vec128(PrefixScan scan,
        const precision_t* __restrict__ grad_out,
        const precision_t* __restrict__ grad_next_state) {
    mingru_scan_backward_ckpt_tuned_vec_body<CKPT_INTERVAL, MINGRU_SCAN_VEC128_WIDTH>(
        scan, grad_out, grad_next_state);
}

enum class MingruScanVariant : int {
    kScalar = 0,
    kVec32 = 1,
    kVec64 = 2,
    kVec128 = 3,
};

struct MingruScanKernelSelection {
    int ckpt_interval;
    MingruScanVariant fwd_variant;
    MingruScanVariant bwd_variant;
};

static inline const char* mingru_scan_variant_name(MingruScanVariant variant) {
    switch (variant) {
        case MingruScanVariant::kScalar: return "log_scalar";
        case MingruScanVariant::kVec32: return "log_vec32";
        case MingruScanVariant::kVec64: return "log_vec64";
        case MingruScanVariant::kVec128: return "log_vec128";
        default: return "log_scalar";
    }
}

static inline bool mingru_scan_variant_supported_for_hidden(MingruScanVariant variant, int H) {
    switch (variant) {
        case MingruScanVariant::kScalar:
            return true;
        case MingruScanVariant::kVec32:
#ifdef PRECISION_FLOAT
            return false;
#else
            return (H % 2) == 0;
#endif
        case MingruScanVariant::kVec64:
            return (H % MINGRU_SCAN_VEC64_WIDTH) == 0;
        case MingruScanVariant::kVec128:
            return (H % MINGRU_SCAN_VEC128_WIDTH) == 0;
        default:
            return false;
    }
}

static inline MingruScanVariant mingru_scan_resolve_variant(MingruScanVariant variant, int H) {
    if (mingru_scan_variant_supported_for_hidden(variant, H)) {
        return variant;
    }
    switch (variant) {
        case MingruScanVariant::kVec128:
            if (mingru_scan_variant_supported_for_hidden(MingruScanVariant::kVec64, H)) {
                return MingruScanVariant::kVec64;
            }
#ifndef PRECISION_FLOAT
            if (mingru_scan_variant_supported_for_hidden(MingruScanVariant::kVec32, H)) {
                return MingruScanVariant::kVec32;
            }
#endif
            return MingruScanVariant::kScalar;
        case MingruScanVariant::kVec64:
#ifndef PRECISION_FLOAT
            if (mingru_scan_variant_supported_for_hidden(MingruScanVariant::kVec32, H)) {
                return MingruScanVariant::kVec32;
            }
#endif
            return MingruScanVariant::kScalar;
        case MingruScanVariant::kVec32:
        case MingruScanVariant::kScalar:
        default:
            return MingruScanVariant::kScalar;
    }
}

static inline MingruScanKernelSelection mingru_scan_select_baseline_policy(int /*B*/, int /*T*/, int H) {
    MingruScanKernelSelection selection = {
        CHECKPOINT_INTERVAL,
        MingruScanVariant::kScalar,
        MingruScanVariant::kScalar,
    };
    selection.fwd_variant = mingru_scan_resolve_variant(selection.fwd_variant, H);
    selection.bwd_variant = mingru_scan_resolve_variant(selection.bwd_variant, H);
    return selection;
}

// Depth-2 trees fitted on fused_scan sweep results (B, T, H grid).
static inline MingruScanKernelSelection mingru_scan_select_depth2_policy(int B, int T, int H) {
    int64_t HB = (int64_t)B * (int64_t)H;
    int64_t HT = (int64_t)H * (int64_t)T;
    int64_t HBT = HB * (int64_t)T;

    int ckpt_interval = 4;
    if (HBT < 4194304LL) {
        ckpt_interval = (HBT < 1048576LL) ? 16 : 1;
    } else {
        ckpt_interval = (HB < 262144LL) ? 8 : 4;
    }

    MingruScanVariant fwd_variant;
    if (HB < 262144LL) {
        fwd_variant = (HBT < 67108864LL) ? MingruScanVariant::kScalar : MingruScanVariant::kVec32;
    } else {
        fwd_variant = (HT < 262144LL) ? MingruScanVariant::kVec64 : MingruScanVariant::kVec128;
    }

    MingruScanVariant bwd_variant;
    if (HB < 262144LL) {
        bwd_variant = (HB < 131072LL) ? MingruScanVariant::kScalar : MingruScanVariant::kVec32;
    } else {
        bwd_variant = (HB < 524288LL) ? MingruScanVariant::kVec64 : MingruScanVariant::kVec32;
    }

    MingruScanKernelSelection selection = {ckpt_interval, fwd_variant, bwd_variant};
    selection.fwd_variant = mingru_scan_resolve_variant(selection.fwd_variant, H);
    selection.bwd_variant = mingru_scan_resolve_variant(selection.bwd_variant, H);
    return selection;
}

template<int CKPT_INTERVAL>
static inline void mingru_scan_launch_forward_ckpt(
        PrefixScan scan, MingruScanVariant variant, cudaStream_t stream) {
    switch (variant) {
        case MingruScanVariant::kVec128:
            mingru_scan_forward_ckpt_tuned_vec128<CKPT_INTERVAL><<<
                grid_size(scan.B * (scan.H / MINGRU_SCAN_VEC128_WIDTH)),
                BLOCK_SIZE, 0, stream>>>(scan);
            return;
        case MingruScanVariant::kVec64:
            mingru_scan_forward_ckpt_tuned_vec64<CKPT_INTERVAL><<<
                grid_size(scan.B * (scan.H / MINGRU_SCAN_VEC64_WIDTH)),
                BLOCK_SIZE, 0, stream>>>(scan);
            return;
        case MingruScanVariant::kVec32:
#ifndef PRECISION_FLOAT
            mingru_scan_forward_ckpt_tuned_vec32<CKPT_INTERVAL><<<
                grid_size(scan.B * (scan.H / 2)),
                BLOCK_SIZE, 0, stream>>>(scan);
            return;
#else
            break;
#endif
        case MingruScanVariant::kScalar:
        default:
            mingru_scan_forward_ckpt_tuned<CKPT_INTERVAL><<<
                grid_size(scan.B * scan.H),
                BLOCK_SIZE, 0, stream>>>(scan);
            return;
    }
}

template<int CKPT_INTERVAL>
static inline void mingru_scan_launch_backward_ckpt(
        PrefixScan scan, const precision_t* grad_out, const precision_t* grad_next_state,
        MingruScanVariant variant, cudaStream_t stream) {
    switch (variant) {
        case MingruScanVariant::kVec128:
            mingru_scan_backward_ckpt_tuned_vec128<CKPT_INTERVAL><<<
                grid_size(scan.B * (scan.H / MINGRU_SCAN_VEC128_WIDTH)),
                BLOCK_SIZE, 0, stream>>>(
                scan, grad_out, grad_next_state);
            return;
        case MingruScanVariant::kVec64:
            mingru_scan_backward_ckpt_tuned_vec64<CKPT_INTERVAL><<<
                grid_size(scan.B * (scan.H / MINGRU_SCAN_VEC64_WIDTH)),
                BLOCK_SIZE, 0, stream>>>(
                scan, grad_out, grad_next_state);
            return;
        case MingruScanVariant::kVec32:
#ifndef PRECISION_FLOAT
            mingru_scan_backward_ckpt_tuned_vec32<CKPT_INTERVAL><<<
                grid_size(scan.B * (scan.H / 2)),
                BLOCK_SIZE, 0, stream>>>(
                scan, grad_out, grad_next_state);
            return;
#else
            break;
#endif
        case MingruScanVariant::kScalar:
        default:
            mingru_scan_backward_ckpt_tuned<CKPT_INTERVAL><<<
                grid_size(scan.B * scan.H),
                BLOCK_SIZE, 0, stream>>>(
                scan, grad_out, grad_next_state);
            return;
    }
}

static inline void mingru_scan_launch_forward_selected(
        PrefixScan scan, int ckpt_interval, MingruScanVariant variant, cudaStream_t stream) {
    variant = mingru_scan_resolve_variant(variant, scan.H);
    switch (ckpt_interval) {
        case 1: mingru_scan_launch_forward_ckpt<1>(scan, variant, stream); return;
        case 2: mingru_scan_launch_forward_ckpt<2>(scan, variant, stream); return;
        case 4: mingru_scan_launch_forward_ckpt<4>(scan, variant, stream); return;
        case 8: mingru_scan_launch_forward_ckpt<8>(scan, variant, stream); return;
        case 16: mingru_scan_launch_forward_ckpt<16>(scan, variant, stream); return;
        case 32: mingru_scan_launch_forward_ckpt<32>(scan, variant, stream); return;
        default:
            mingru_scan_launch_forward_ckpt<CHECKPOINT_INTERVAL>(scan, variant, stream);
            return;
    }
}

static inline void mingru_scan_launch_backward_selected(
        PrefixScan scan, const precision_t* grad_out, const precision_t* grad_next_state,
        int ckpt_interval, MingruScanVariant variant, cudaStream_t stream) {
    variant = mingru_scan_resolve_variant(variant, scan.H);
    switch (ckpt_interval) {
        case 1:
            mingru_scan_launch_backward_ckpt<1>(
                scan, grad_out, grad_next_state, variant, stream);
            return;
        case 2:
            mingru_scan_launch_backward_ckpt<2>(
                scan, grad_out, grad_next_state, variant, stream);
            return;
        case 4:
            mingru_scan_launch_backward_ckpt<4>(
                scan, grad_out, grad_next_state, variant, stream);
            return;
        case 8:
            mingru_scan_launch_backward_ckpt<8>(
                scan, grad_out, grad_next_state, variant, stream);
            return;
        case 16:
            mingru_scan_launch_backward_ckpt<16>(
                scan, grad_out, grad_next_state, variant, stream);
            return;
        case 32:
            mingru_scan_launch_backward_ckpt<32>(
                scan, grad_out, grad_next_state, variant, stream);
            return;
        default:
            mingru_scan_launch_backward_ckpt<CHECKPOINT_INTERVAL>(
                scan, grad_out, grad_next_state, variant, stream);
            return;
    }
}


__global__ void sum_rows_to_precision_kernel(precision_t* __restrict__ dst,
        const float* __restrict__ src, int R, int C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= C) {
        return;
    }
    float sum = 0.0f;
    for (int r = 0; r < R; r++) {
        sum += src[r * C + col];
    }
    dst[col] = from_float(sum);
}

__global__ void assemble_decoder_grad(
        precision_t* __restrict__ dst, const float* __restrict__ grad_logits,
        const float* __restrict__ grad_value, int B_TT, int od, int od_plus_1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B_TT * od_plus_1) {
        return;
    }
    int row = idx / od_plus_1, col = idx % od_plus_1;
    dst[idx] = from_float((col < od) ? grad_logits[row * od + col] : grad_value[row]);
}

static PrecisionTensor encoder_forward(void* w, void* activations, PrecisionTensor input, cudaStream_t stream) {
    EncoderWeights* ew = (EncoderWeights*)w;
    EncoderActivations* a = (EncoderActivations*)activations;
    if (a->saved_input.data) puf_copy(&a->saved_input, &input, stream);
    puf_mm(&input, &ew->weight, &a->out, stream);
    return a->out;
}

static void encoder_backward(void* w, void* activations, PrecisionTensor grad, cudaStream_t stream) {
    EncoderActivations* a = (EncoderActivations*)activations;
    puf_mm_tn(&grad, &a->saved_input, &a->wgrad_scratch, stream);
}

static void encoder_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    EncoderWeights* ew = (EncoderWeights*)w;
    PrecisionTensor wt = {
        .data = ew->weight.data,
        .shape = {ew->out_dim, ew->in_dim},
    };
    puf_kaiming_init(&wt, std::sqrt(2.0f), (*seed)++, stream);
}

static void encoder_reg_params(void* w, Allocator* alloc) {
    EncoderWeights* ew = (EncoderWeights*)w;
    ew->weight = {.shape = {ew->out_dim, ew->in_dim}};
    alloc_register(alloc,&ew->weight);
}

static void encoder_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    EncoderWeights* ew = (EncoderWeights*)w;
    EncoderActivations* a = (EncoderActivations*)activations;
    *a = (EncoderActivations){
        .out =              {.shape = {B_TT, ew->out_dim}},
        .saved_input =      {.shape = {B_TT, ew->in_dim}},
        .wgrad_scratch =    {.shape = {ew->out_dim, ew->in_dim}},
    };
    alloc_register(acts,&a->out);
    alloc_register(acts,&a->saved_input);
    alloc_register(grads,&a->wgrad_scratch);
}

static void encoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    EncoderWeights* ew = (EncoderWeights*)w;
    EncoderActivations* a = (EncoderActivations*)activations;
    a->out = {.shape = {B, ew->out_dim}};
    alloc_register(alloc,&a->out);
}

static void* encoder_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    EncoderWeights* ew = (EncoderWeights*)calloc(1, sizeof(EncoderWeights));
    ew->in_dim = e->in_dim; ew->out_dim = e->out_dim;
    return ew;
}

static void encoder_free_weights(void* weights) {
    free(weights);
}

static void encoder_free_activations(void* activations) {
    free(activations);
}

struct DecoderWeights {
    PrecisionTensor weight, logstd;
    int hidden_dim, output_dim;
    bool continuous;
};

struct DecoderActivations {
    PrecisionTensor out, grad_out, saved_input, grad_input, wgrad_scratch, logstd_scratch;
};

static PrecisionTensor decoder_forward(void* w, void* activations, PrecisionTensor input, cudaStream_t stream) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    if (a->saved_input.data) {
        puf_copy(&a->saved_input, &input, stream);
    }
    puf_mm(&input, &dw->weight, &a->out, stream);
    return a->out;
}

static void decoder_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    DecoderWeights* dw = (DecoderWeights*)w;
    PrecisionTensor wt = {
        .data = dw->weight.data,
        .shape = {dw->output_dim + 1, dw->hidden_dim},
    };
    puf_kaiming_init(&wt, 1.0f, (*seed)++, stream);
}

static void decoder_reg_params(void* w, Allocator* alloc) {
    DecoderWeights* dw = (DecoderWeights*)w;
    dw->weight = {.shape = {dw->output_dim + 1, dw->hidden_dim}};
    alloc_register(alloc,&dw->weight);
    if (dw->continuous) {
        dw->logstd = {.shape = {1, dw->output_dim}};
        alloc_register(alloc,&dw->logstd);
    }
}

static void decoder_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    int od1 = dw->output_dim + 1;
    *a = (DecoderActivations){
        .out =              {.shape = {B_TT, od1}},
        .grad_out =         {.shape = {B_TT, od1}},
        .saved_input =      {.shape = {B_TT, dw->hidden_dim}},
        .grad_input =       {.shape = {B_TT, dw->hidden_dim}},
        .wgrad_scratch =    {.shape = {od1, dw->hidden_dim}},
        .logstd_scratch =   {.shape = {1, dw->output_dim}},
    };
    alloc_register(acts,&a->out);
    alloc_register(acts,&a->saved_input);
    alloc_register(acts,&a->grad_out);
    alloc_register(acts,&a->grad_input);
    alloc_register(grads,&a->wgrad_scratch);
    if (dw->continuous) alloc_register(grads,&a->logstd_scratch);
}

static void decoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    a->out = {.shape = {B, dw->output_dim + 1}};
    alloc_register(alloc,&a->out);
}

static void* decoder_create_weights(void* self) {
    Decoder* d = (Decoder*)self;
    DecoderWeights* dw = (DecoderWeights*)calloc(1, sizeof(DecoderWeights));
    dw->hidden_dim = d->hidden_dim; dw->output_dim = d->output_dim; dw->continuous = d->continuous;
    return dw;
}

static void decoder_free_weights(void* weights) {
    free(weights);
}

static void decoder_free_activations(void* activations) {
    free(activations);
}

static PrecisionTensor decoder_backward(void* w, void* activations,
    FloatTensor grad_logits, FloatTensor grad_logstd, FloatTensor grad_value, cudaStream_t stream) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    int B_TT = a->saved_input.shape[0];
    int od = dw->output_dim, od1 = od + 1;
    assemble_decoder_grad<<<grid_size(B_TT * od1), BLOCK_SIZE, 0, stream>>>(
        a->grad_out.data, grad_logits.data, grad_value.data, B_TT, od, od1);
    puf_mm_tn(&a->grad_out, &a->saved_input, &a->wgrad_scratch, stream);
    if (dw->continuous && grad_logstd.data != nullptr) {
        sum_rows_to_precision_kernel<<<grid_size(dw->output_dim), BLOCK_SIZE, 0, stream>>>(
            a->logstd_scratch.data, grad_logstd.data, B_TT, dw->output_dim);
    }
    puf_mm_nn(&a->grad_out, &dw->weight, &a->grad_input, stream);
    return a->grad_input;
}

struct MinGRUActivations {
    int num_layers;
    // Rollout
    PrecisionTensor* combined;       // (B rollout, 3*T)[num_layers]
    PrecisionTensor out;             // (B rollout, T)
    PrecisionTensor next_state;      // (B rollout, T)
    // Training
    PrecisionTensor* saved_inputs;   // (B, TT, T)[num_layers]
    PrefixScan* scan_bufs;           // [num_layers]
    PrecisionTensor* combined_bufs;  // (B*TT, 3*T)[num_layers]
    PrecisionTensor* wgrad_scratch;  // (3*T, T)[num_layers]
    PrecisionTensor grad_input_buf;  // (B*TT, T)
    PrecisionTensor grad_next_state; // (B, 1, T)
};

void mingru_activations_free(MinGRUActivations* a) {
    free(a->combined);
    free(a->saved_inputs);
    free(a->scan_bufs);
    free(a->combined_bufs);
    free(a->wgrad_scratch);
}

struct MinGRUWeights {
    int hidden, num_layers, horizon;
    PrecisionTensor* weights;  // [num_layers]
};

static PrecisionTensor mingru_state_layer(MinGRUWeights* m, PrecisionTensor& state, int layer_i) {
    long B = state.shape[1], H = state.shape[2];
    return {.data = state.data + layer_i * B * H, .shape = {B, H}};
}

static void mingru_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    for (int layer_i = 0; layer_i < m->num_layers; layer_i++) {
        PrecisionTensor w2d = {
            .data = m->weights[layer_i].data,
            .shape = {3 * m->hidden, m->hidden},
        };
        puf_kaiming_init(&w2d, 1.0f, (*seed)++, stream);
    }
}

static void mingru_reg_params(void* w, Allocator* alloc) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    for (int layer_i = 0; layer_i < m->num_layers; layer_i++) {
        m->weights[layer_i] = {.shape = {3 * m->hidden, m->hidden}};
        alloc_register(alloc,&m->weights[layer_i]);
    }
}

static void mingru_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int H = m->hidden, TT = m->horizon, B = B_TT / TT;
    a->num_layers = m->num_layers;
    a->saved_inputs = (PrecisionTensor*)calloc(m->num_layers, sizeof(PrecisionTensor));
    a->scan_bufs = (PrefixScan*)calloc(m->num_layers, sizeof(PrefixScan));
    a->combined_bufs = (PrecisionTensor*)calloc(m->num_layers, sizeof(PrecisionTensor));
    a->wgrad_scratch = (PrecisionTensor*)calloc(m->num_layers, sizeof(PrecisionTensor));
    a->grad_input_buf = {.shape = {B_TT, H}};
    a->grad_next_state = {.shape = {B, 1, H}};
    alloc_register(acts,&a->grad_input_buf);
    alloc_register(acts,&a->grad_next_state);
    for (int layer_i = 0; layer_i < m->num_layers; layer_i++) {
        a->scan_bufs[layer_i] = {
            .B = B, .T = TT, .H = H,
            .a_star =           {.shape = {B, TT + 1, H}},
            .s_vals =           {.shape = {B, TT + 1, H}},
            .log_values_buf =   {.shape = {B, TT + 1, H}},
            .out =              {.shape = {B, TT, H}},
            .next_state =       {.shape = {B, 1, H}},
            .grad_combined =    {.shape = {B, TT, 3 * H}},
            .grad_state =       {.shape = {B, 1, H}},
            .grad_input =       {.shape = {B, TT, H}},
        };
        a->saved_inputs[layer_i]  = {.shape = {B, TT, H}};
        a->combined_bufs[layer_i] = {.shape = {B_TT, 3 * H}};
        a->wgrad_scratch[layer_i] = {.shape = {3 * H, H}};
        alloc_register(acts,&a->saved_inputs[layer_i]);
        alloc_register(acts,&a->combined_bufs[layer_i]);
        alloc_register(acts,&a->scan_bufs[layer_i].out);
        alloc_register(acts,&a->scan_bufs[layer_i].next_state);
        alloc_register(acts,&a->scan_bufs[layer_i].a_star);
        alloc_register(acts,&a->scan_bufs[layer_i].s_vals);
        alloc_register(acts,&a->scan_bufs[layer_i].log_values_buf);
        alloc_register(acts,&a->scan_bufs[layer_i].grad_combined);
        alloc_register(acts,&a->scan_bufs[layer_i].grad_state);
        alloc_register(acts,&a->scan_bufs[layer_i].grad_input);
        alloc_register(grads,&a->wgrad_scratch[layer_i]);
    }
}

static void mingru_reg_rollout(void* weights, void* activations, Allocator* alloc, int B_inf) {
    MinGRUWeights* w = (MinGRUWeights*)weights;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int H = w->hidden;
    a->num_layers = w->num_layers;
    a->combined = (PrecisionTensor*)calloc(w->num_layers, sizeof(PrecisionTensor));
    for (int layer_i = 0; layer_i < w->num_layers; layer_i++) {
        a->combined[layer_i] = {.shape = {B_inf, 3 * H}};
        alloc_register(alloc,&a->combined[layer_i]);
    }
    a->out = {.shape = {B_inf, H}};
    a->next_state = {.shape = {B_inf, H}};
    alloc_register(alloc,&a->out);
    alloc_register(alloc,&a->next_state);
}

static void* mingru_create_weights(void* self) {
    Network* n = (Network*)self;
    MinGRUWeights* mw = (MinGRUWeights*)calloc(1, sizeof(MinGRUWeights));
    mw->hidden = n->hidden;
    mw->num_layers = n->num_layers;
    mw->horizon = n->horizon;
    mw->weights = (PrecisionTensor*)calloc(n->num_layers, sizeof(PrecisionTensor));
    return mw;
}

static void mingru_free_weights(void* weights) {
    MinGRUWeights* mw = (MinGRUWeights*)weights;
    free(mw->weights);
    free(mw);
}

static void mingru_free_activations(void* activations) {
    MinGRUActivations* a = (MinGRUActivations*)activations;
    mingru_activations_free(a);
    free(a);
}

static PrecisionTensor mingru_forward(void* w, PrecisionTensor x, PrecisionTensor state,
        void* activations, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int B = state.shape[1];
    int H = state.shape[2];
    for (int layer_i = 0; layer_i < m->num_layers; layer_i++) {
        PrecisionTensor state_i = mingru_state_layer(m, state, layer_i);
        puf_mm(&x, &m->weights[layer_i], &a->combined[layer_i], stream);
        mingru_gate<<<grid_size(B*H), BLOCK_SIZE, 0, stream>>>(
            a->out.data, a->next_state.data,
            a->combined[layer_i].data, state_i.data, x.data, H, B);
        puf_copy(&state_i, &a->next_state, stream);
        x = a->out;
    }
    return x;
}

static PrecisionTensor mingru_forward_train(void* w, PrecisionTensor x, PrecisionTensor state,
        void* activations, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    MingruScanKernelSelection scan_selection = mingru_scan_select_depth2_policy(
        state.shape[1], m->horizon, m->hidden);
    for (int layer_i = 0; layer_i < m->num_layers; layer_i++) {
        puf_copy(&a->saved_inputs[layer_i], &x, stream);
        PrecisionTensor state_i = mingru_state_layer(m, state, layer_i);
        puf_mm(&x, &m->weights[layer_i], &a->combined_bufs[layer_i], stream);
        a->scan_bufs[layer_i].combined_ptr = a->combined_bufs[layer_i].data;
        a->scan_bufs[layer_i].state_ptr = state_i.data;
        a->scan_bufs[layer_i].input_ptr = a->saved_inputs[layer_i].data;
        mingru_scan_launch_forward_selected(
            a->scan_bufs[layer_i], scan_selection.ckpt_interval, scan_selection.fwd_variant, stream);
        x = a->scan_bufs[layer_i].out;
    }
    return x;
}

static PrecisionTensor mingru_backward(void* w, PrecisionTensor grad, void* activations, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    MingruScanKernelSelection scan_selection = mingru_scan_select_depth2_policy(
        a->scan_bufs[0].B, a->scan_bufs[0].T, a->scan_bufs[0].H);
    for (int layer_i = m->num_layers - 1; layer_i >= 0; layer_i--) {
        PrefixScan& scan = a->scan_bufs[layer_i];
        mingru_scan_launch_backward_selected(
            scan, grad.data, a->grad_next_state.data,
            scan_selection.ckpt_interval, scan_selection.bwd_variant, stream);
        puf_mm_tn(&scan.grad_combined, &a->saved_inputs[layer_i], &a->wgrad_scratch[layer_i], stream);
        puf_mm_nn(&scan.grad_combined, &m->weights[layer_i], &a->grad_input_buf, stream);
        int n = numel(scan.grad_input.shape);
        add_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            a->grad_input_buf.data, scan.grad_input.data, n);
        grad = a->grad_input_buf;
    }
    return grad;
}

struct Policy {
    Encoder encoder;
    Decoder decoder;
    Network network;
    int input_dim, hidden_dim, output_dim;
    int num_atns;
};

struct PolicyActivations {
    void* encoder;
    void* decoder;
    void* network;
};

struct PolicyWeights {
    void* encoder;
    void* decoder;
    void* network;
};

static void policy_activations_free(Policy* p, PolicyActivations& a) {
    p->encoder.free_activations(a.encoder);
    p->decoder.free_activations(a.decoder);
    p->network.free_activations(a.network);
}

PrecisionTensor policy_forward(Policy* p, PolicyWeights& w, PolicyActivations& activations,
        PrecisionTensor obs, PrecisionTensor state, cudaStream_t stream) {
    PrecisionTensor enc_out = p->encoder.forward(w.encoder, activations.encoder, obs, stream);
    PrecisionTensor h = p->network.forward(w.network, enc_out, state, activations.network, stream);
    return p->decoder.forward(w.decoder, activations.decoder, h, stream);
}

PrecisionTensor policy_forward_train(Policy* p, PolicyWeights& w, PolicyActivations& activations,
        PrecisionTensor x, PrecisionTensor state, cudaStream_t stream) {
    int B = x.shape[0], TT = x.shape[1];
    PrecisionTensor h = p->encoder.forward(w.encoder, activations.encoder, *puf_squeeze(&x, 0), stream);
    h = p->network.forward_train(w.network, *puf_unsqueeze(&h, 0, B, TT), state, activations.network, stream);
    PrecisionTensor dec_out = p->decoder.forward(w.decoder, activations.decoder, *puf_squeeze(&h, 0), stream);
    return *puf_unsqueeze(&dec_out, 0, B, TT);
}

void policy_backward(Policy* p, PolicyWeights& w, PolicyActivations& activations,
        FloatTensor grad_logits, FloatTensor grad_logstd, FloatTensor grad_value, cudaStream_t stream) {
    int B = grad_logits.shape[0], TT = grad_logits.shape[1];
    PrecisionTensor grad_h = p->decoder.backward(w.decoder, activations.decoder,
        *puf_squeeze(&grad_logits, 0), grad_logstd, *puf_squeeze(&grad_value, 0), stream);
    grad_h = p->network.backward(w.network, *puf_unsqueeze(&grad_h, 0, B, TT), activations.network, stream);
    p->encoder.backward(w.encoder, activations.encoder, grad_h, stream);
}

PolicyActivations policy_reg_train(Policy* p, PolicyWeights& w,
        Allocator* acts, Allocator* grads, int B_TT) {
    PolicyActivations a;
    a.encoder = calloc(1, p->encoder.activation_size);
    a.decoder = calloc(1, sizeof(DecoderActivations));
    a.network = calloc(1, sizeof(MinGRUActivations));
    p->encoder.reg_train(w.encoder, a.encoder, acts, grads, B_TT);
    p->decoder.reg_train(w.decoder, a.decoder, acts, grads, B_TT);
    p->network.reg_train(w.network, a.network, acts, grads, B_TT);
    return a;
}

PolicyActivations policy_reg_rollout(Policy* p, PolicyWeights& w, Allocator* acts, int B_inf) {
    PolicyActivations a;
    a.encoder = calloc(1, p->encoder.activation_size);
    a.decoder = calloc(1, sizeof(DecoderActivations));
    a.network = calloc(1, sizeof(MinGRUActivations));
    p->encoder.reg_rollout(w.encoder, a.encoder, acts, B_inf);
    p->decoder.reg_rollout(w.decoder, a.decoder, acts, B_inf);
    p->network.reg_rollout(w.network, a.network, acts, B_inf);
    return a;
}

void policy_init_weights(Policy* p, PolicyWeights& w, uint64_t* seed, cudaStream_t stream) {
    p->encoder.init_weights(w.encoder, seed, stream);
    p->decoder.init_weights(w.decoder, seed, stream);
    p->network.init_weights(w.network, seed, stream);
}

PolicyWeights policy_weights_create(Policy* p, Allocator* params) {
    PolicyWeights w;
    w.encoder = p->encoder.create_weights(&p->encoder);
    w.decoder = p->decoder.create_weights(&p->decoder);
    w.network = p->network.create_weights(&p->network);
    p->encoder.reg_params(w.encoder, params);
    p->decoder.reg_params(w.decoder, params);
    p->network.reg_params(w.network, params);
    return w;
}

void policy_weights_free(Policy* p, PolicyWeights* w) {
    p->encoder.free_weights(w->encoder);
    p->decoder.free_weights(w->decoder);
    p->network.free_weights(w->network);
}

#endif // PUFFERLIB_MODELS_CU
