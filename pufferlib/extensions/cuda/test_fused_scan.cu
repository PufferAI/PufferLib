/*
 * Standalone test file for fused_scan forward/backward kernel optimization.
 * 
 * OPTIMIZATION TARGET: Production RL training workloads
 * - Batch sizes (B): 512-2048 (typical vectorized environments)
 * - Sequence length (T): 64-128 (standard rollout horizons)
 * - Hidden dimension (H): 256-512 (common model sizes)
 * 
 * Hybrid Checkpointing Strategy (simplified):
 * - Forward: Writes sparse checkpoints every CHECKPOINT_INTERVAL=16 timesteps
 * - Backward: Loads checkpoint, recomputes that chunk, processes backward
 * - Unified: chunk size = checkpoint interval (simpler, less register pressure)
 * 
 * Features:
 * - Original forward kernel (writes dense buffers every timestep)
 * - Optimized forward kernel (writes sparse checkpoints every 16 timesteps)
 * - Original backward kernel (reads dense buffers from forward pass)
 * - Checkpointed backward kernel (reads sparse checkpoints, recomputes chunks)
 * 
 * Configuration:
 * - CHECKPOINT_INTERVAL = 16 (unified checkpoint + chunk size)
 * - THREADS_PER_BLOCK = 128 (standard)
 * - Register arrays: 5 × CHECKPOINT_INTERVAL = 80 registers for chunk storage
 *   (a_star, s, log_values, hidden, gate - no proj to save registers)
 * 
 * Build with:
 *   cd pufferlib/extensions/cuda
 *   nvcc -O3 -arch=sm_86 -shared -Xcompiler -fPIC test_fused_scan.cu -o test_fused_scan.so
 * 
 * Then run:
 *   python test_fused_scan.py                     # Test forward only
 *   python test_fused_scan.py --test-backward     # Test backward (checkpointed vs original)
 *   python test_fused_scan.py --benchmark         # Benchmark forward
 *   python test_fused_scan.py --test-backward --benchmark  # Full test + benchmark
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>

// ============================================================================
// Inlined ops from ops.cuh
// ============================================================================

#ifndef CUDART_INF_F
#define CUDART_INF_F __int_as_float(0x7f800000)
#endif

#define SOFTPLUS_BETA 1.0f
#define SOFTPLUS_THRESHOLD 20.0f
#define COARSE 4

__device__ __forceinline__ float softplus_fwd(float x) {
    float x_scaled = x * SOFTPLUS_BETA;
    if (x_scaled > SOFTPLUS_THRESHOLD) {
        return x;
    } else {
        return log1pf(expf(x_scaled)) / SOFTPLUS_BETA;
    }
}

__device__ __forceinline__ float relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ __inline__ float fast_tanh(float x) {
    const float plus_9 = 9.0f;
    const float minus_9 = -9.0f;
    float v1 = fminf(x, plus_9);
    v1 = fmaxf(v1, minus_9);

    const float alpha_1 = 4.89352455891786e-03f;
    const float alpha_3 = 6.37261928875436e-04f;
    const float alpha_5 = 1.48572235717979e-05f;
    const float alpha_7 = 5.12229709037114e-08f;
    const float alpha_9 = -8.60467152213735e-11f;
    const float alpha_11 = 2.00018790482477e-13f;
    const float alpha_13 = -2.76076847742355e-16f;
    const float beta_0 = 4.89352518554385e-03f;
    const float beta_2 = 2.26843463243900e-03f;
    const float beta_4 = 1.18534705686654e-04f;
    const float beta_6 = 1.19825839466702e-06f;

    float v2 = v1 * v1;
    float p = v2 * alpha_13 + alpha_11;
    p = v2 * p + alpha_9;
    p = v2 * p + alpha_7;
    p = v2 * p + alpha_5;
    p = v2 * p + alpha_3;
    p = v2 * p + alpha_1;
    p = v1 * p;

    float q = v2 * beta_6 + beta_4;
    q = v2 * q + beta_2;
    q = v2 * q + beta_0;

    return p / q;
}

__device__ __inline__ float fast_sigmoid(float x) {
    const float one_v = 1.0f;
    const float half_v = 0.5f;
    const float zero_v = 0.0f;
    float x2 = x * half_v;
    float y = fast_tanh(x2);
    float z = (y + one_v) * half_v;
    return fminf(one_v, fmaxf(zero_v, z));
}

__device__ __forceinline__ float sigmoid(float x) {
    float z = expf(-fabsf(x));
    return x >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
}

__device__ __forceinline__ void log_coeffs_and_values_fwd(
    float gate,
    float hidden,
    float* log_coeff_out,
    float* log_value_out
) {
    *log_coeff_out = -softplus_fwd(gate);

    float log_z = -softplus_fwd(-gate);
    float log_tilde_h;
    if (hidden >= 0.0f) {
        float relu_h = relu(hidden);
        log_tilde_h = logf(relu_h + 0.5f);
    } else {
        log_tilde_h = -softplus_fwd(-hidden);
    }
    *log_value_out = log_z + log_tilde_h;
}

__device__ __forceinline__ void log_coeffs_and_values_bwd(
    float grad_log_coeffs,
    float grad_log_values,
    float gate,
    float hidden,
    float* grad_gate_out,
    float* grad_hidden_out
) {
    // Optimization: sigmoid(-x) = 1 - sigmoid(x), so compute sigmoid(gate) once
    // softplus'(x) = sigmoid(x), so:
    //   d(-softplus(g))/dg = -sigmoid(g)
    //   d(-softplus(-g))/dg = sigmoid(-g) = 1 - sigmoid(g)
    float sig_gate = sigmoid(gate);
    *grad_gate_out = -grad_log_coeffs*sig_gate + grad_log_values*(1.0f - sig_gate);

    // grad_hidden from log_tilde_h
    if (hidden >= 0.0f) {
        // log_tilde_h = log(hidden + 0.5)
        // d(log_tilde_h)/d(hidden) = 1/(hidden + 0.5)
        *grad_hidden_out = grad_log_values / (hidden + 0.5f);
    } else {
        // log_tilde_h = -softplus(-hidden)
        // d(-softplus(-h))/dh = sigmoid(-h)
        *grad_hidden_out = grad_log_values * sigmoid(-hidden);
    }
}

// ============================================================================
// Launch configuration
// ============================================================================

// Original kernels: use 32 threads per block (unchanged)
#define SEQ_SIZE 32

inline int seq_size(int N) {
    return (N + SEQ_SIZE - 1) / SEQ_SIZE;
}

// Optimized kernels: use 128 threads per block (same as original)
#define OPT_BLOCK_SIZE 256

inline int opt_grid_size(int N) {
    return (N + OPT_BLOCK_SIZE - 1) / OPT_BLOCK_SIZE;
}

// ============================================================================
// Original fused_scan_forward_kernel (baseline)
// ============================================================================

template<typename T>
__global__ void fused_scan_forward_kernel(
    T* __restrict__ out,                 // (B, T, H) - sigmoid(proj) * scan_result
    T* __restrict__ next_state,          // (B, 1, H) - raw scan_result at T (for recurrence)
    float* __restrict__ a_star_buf,      // (B, T+1, H) - for backward
    float* __restrict__ s_buf,           // (B, T+1, H) - for backward
    float* __restrict__ log_values_buf,  // (B, T+1, H) - cached log_values for backward
    const T* __restrict__ combined,      // (B, T, 3*H) = [hidden(H), gate(H), proj(H)]
    const T* __restrict__ state,         // (B, 1, H)
    int T_seq,                           // sequence length (T)
    int H,
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;

    int b = idx / H;
    int h = idx % H;

    int T_out = T_seq + 1;
    int buf_base = b * T_out * H + h;    // base for a_star/s/log_values buffers (T+1 timesteps)
    int out_base = b * T_seq * H + h;    // base for output (T timesteps)
    int state_idx = b * H + h;           // state is (B, 1, H) -> flatten to (B, H)

    float a_star = 0.0f;
    float s = -INFINITY;  // logcumsumexp accumulator

    // Handle t=0 outside the loop: use log(state), coeff = 0
    float log_value_0 = logf(float(state[state_idx]));
    log_values_buf[buf_base] = log_value_0;
    s = log_value_0;  // z = log_value - a_star = log_value - 0 = log_value
    a_star_buf[buf_base] = a_star;
    s_buf[buf_base] = s;

    // Loop t=1..T_seq (no branches needed)
    float scan_result = 0.0f;
    for (int t = 1; t < T_out; t++) {
        int buf_curr = buf_base + t * H;
        int combined_base = b * T_seq * 3 * H + (t - 1) * 3 * H;

        float hidden_val = float(combined[combined_base + h]);
        float gate_val = float(combined[combined_base + H + h]);
        float proj_val = float(combined[combined_base + 2 * H + h]);

        float log_coeff_val, log_value_val;
        log_coeffs_and_values_fwd(gate_val, hidden_val, &log_coeff_val, &log_value_val);

        // Cache log_value for backward (avoid recomputation)
        log_values_buf[buf_curr] = log_value_val;

        // a_star[t] = sum_{i=0}^t log_coeffs[i]
        a_star += log_coeff_val;

        float z = log_value_val - a_star;

        if (s == -INFINITY) {
            s = z;
        } else {
            float min_val = fminf(s, z);
            float max_val = fmaxf(s, z);
            s = max_val + log1pf(expf(min_val - max_val));
        }

        scan_result = expf(a_star + s);

        // sigmoid(proj) * out
        int out_curr = out_base + (t - 1) * H;
        float proj_sigmoid = sigmoid(proj_val);
        out[out_curr] = T(proj_sigmoid * scan_result);

        a_star_buf[buf_curr] = a_star;
        s_buf[buf_curr] = s;
    }
    // Write timestep T to next_state (raw scan_result, no proj, for recurrence)
    next_state[state_idx] = T(scan_result);
}

template<typename T>
void launch_fused_scan_forward(
    T* out,
    T* next_state,
    float* a_star,
    float* s_vals,
    float* log_values_buf,
    const T* combined,
    const T* state,
    int T_seq,
    int H,
    int B,
    cudaStream_t stream
) {
    int total = B * H;
    int grid = seq_size(total);

    fused_scan_forward_kernel<T><<<grid, SEQ_SIZE, 0, stream>>>(
        out,
        next_state,
        a_star,
        s_vals,
        log_values_buf,
        combined,
        state,
        T_seq,
        H,
        B
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error in forward: %s\n", cudaGetErrorString(err));
    }
}

// Gradient checkpointing configuration
// Unified: checkpoint interval = backward chunk size (simpler, less register pressure)
#define CHECKPOINT_INTERVAL 4  // Save checkpoints every 16 timesteps, process backward in same chunks

// ============================================================================
// NEW optimized fused_scan_forward_kernel with sparse checkpoint writing
// Writes sparse checkpoints every CHECKPOINT_INTERVAL timesteps (reduces writes by 16x)
// ============================================================================

#define TILE_SIZE 8

template<typename T>
__global__ void fused_scan_forward_kernel_new(
    T* __restrict__ out,
    T* __restrict__ next_state,
    float* __restrict__ a_star_buf,
    float* __restrict__ s_buf,
    float* __restrict__ log_values_buf,
    const T* __restrict__ combined,
    const T* __restrict__ state,
    int T_seq,
    int H,
    int B
) {
    // Standard indexing with 128 threads per block
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
    float s = __logf(float(state[bH + h]));
    log_value = s;

    // Save checkpoint at t=0
    int T_out = T_seq + 1;
    int buf_base = b * T_out * H + h;
    a_star_buf[buf_base] = a_star;
    s_buf[buf_base] = s;
    log_values_buf[buf_base] = log_value;

    // precompute base pointers
    const T* combined_h_base = &combined[cbase + h];
    const T* combined_g_base = &combined[cbase + H + h];
    const T* combined_p_base = &combined[cbase + H2 + h];

    // Loop t=1..T_seq with sparse checkpointing
    float scan_result = 0.0f;

    for (int t = 1; t < T_seq + 1; t++) {
        int t_offset = (t - 1) * H3;

        float hidden_val = float(combined_h_base[t_offset]);
        float gate_val = float(combined_g_base[t_offset]);
        float proj_val = float(combined_p_base[t_offset]);

        float log_coeff_val;
        log_coeffs_and_values_fwd(gate_val, hidden_val, &log_coeff_val, &log_value);

        // a_star[t] = sum_{i=0}^t log_coeffs[i]
        a_star += log_coeff_val;

        float z = log_value - a_star;
        float max_val = fmaxf(s, z);
        s = max_val + log1pf(__expf(-fabsf(s - z)));

        scan_result = __expf(a_star + s);
        float proj_sigmoid = sigmoid(proj_val);

        out[out_base + (t - 1) * H] = T(proj_sigmoid * scan_result);

        // Write sparse checkpoints every CHECKPOINT_INTERVAL timesteps
        if (t % CHECKPOINT_INTERVAL == 0) {
            int buf_curr = buf_base + t * H;
            a_star_buf[buf_curr] = a_star;
            s_buf[buf_curr] = s;
            log_values_buf[buf_curr] = log_value;
        }
    }

    // Write timestep T to next_state (raw scan_result, no proj, for recurrence)
    next_state[bH + h] = T(scan_result);
}

template<typename T>
void launch_fused_scan_forward_new(
    T* out,
    T* next_state,
    float* a_star,
    float* s_vals,
    float* log_values_buf,
    const T* combined,
    const T* state,
    int T_seq,
    int H,
    int B,
    cudaStream_t stream
) {
    // Standard launch with 128 threads per block
    int total = B * H;
    int grid = opt_grid_size(total);
    fused_scan_forward_kernel_new<T><<<grid, OPT_BLOCK_SIZE, 0, stream>>>(
        out,
        next_state,
        a_star,
        s_vals,
        log_values_buf,
        combined,
        state,
        T_seq,
        H,
        B
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error in forward_new: %s\n", cudaGetErrorString(err));
    }
}

// ============================================================================
// Backward kernels (original and checkpointed)
// ============================================================================


// Helper function to recompute forward scan from checkpoint to target timestep

// Original fused backward (uses saved buffers)
template<typename T>
__global__ void fused_scan_backward_kernel(
    T* __restrict__ grad_combined,         // (B, T, 3*H) = [grad_hidden, grad_gate, grad_proj]
    T* __restrict__ grad_state,            // (B, 1, H)
    const T* __restrict__ grad_out,        // (B, T, H) - gradient of sigmoid(proj)*scan_result
    const T* __restrict__ grad_next_state, // (B, 1, H) - gradient of raw scan_result at T
    const T* __restrict__ combined,        // (B, T, 3*H) = [hidden, gate, proj]
    const T* __restrict__ state,           // (B, 1, H)
    const float* __restrict__ a_star_buf,  // (B, T+1, H)
    const float* __restrict__ s_buf,       // (B, T+1, H)
    const float* __restrict__ log_values_buf, // (B, T+1, H) - cached from forward
    int T_seq,                             // sequence length (T)
    int H,
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;

    int b = idx / H;
    int h = idx % H;

    int T_out = T_seq + 1;
    int buf_base = b * T_out * H + h;    // base for a_star/s/log_values buffers (T+1 timesteps)
    int out_base = b * T_seq * H + h;    // base for grad_out (T timesteps)
    int state_idx = b * H + h;           // state is (B, 1, H) -> flatten to (B, H)

    float acc = 0.0;
    float s_val_next = 0.0;
    float carry_grad_a = 0.0;

    for (int t = T_out - 1; t >= 0; --t) {
        int base_adr = b*T_seq*3*H + (t-1)*3*H;
        int hidden_adr = base_adr + h;
        int gate_adr = base_adr + H + h;
        int proj_adr = base_adr + 2*H + h;

        int buf_curr = buf_base + t * H;

        float a_star = a_star_buf[buf_curr];
        float s = s_buf[buf_curr];
        float scan_result = expf(a_star + s);

        float log_value_val = log_values_buf[buf_curr];

        float gate_val = 0.0f, hidden_val = 0.0f, proj_val = 0.0f;

        if (t >= 1) {
            hidden_val = float(combined[hidden_adr]);
            gate_val = float(combined[gate_adr]);
            proj_val = float(combined[proj_adr]);
        }

        float z = log_value_val - a_star;

        float grad_gated_out = 0.0f;
        float grad_scan_from_next = 0.0f;

        if (t >= 1) {
            int grad_out_idx = out_base + (t - 1) * H;
            grad_gated_out = float(grad_out[grad_out_idx]);
        }
        if (t == T_seq) {
            grad_scan_from_next = float(grad_next_state[state_idx]);
        }

        float grad_scan_result = grad_scan_from_next;
        float grad_proj = 0.0f;

        if (t >= 1) {
            float proj_sigmoid = sigmoid(proj_val);
            grad_scan_result += grad_gated_out * proj_sigmoid;
            grad_proj = grad_gated_out * scan_result * proj_sigmoid * (1.0f - proj_sigmoid);
        }

        float grad_log_h = grad_scan_result * scan_result;
        float grad_s = grad_log_h;

        if (t == T_out - 1) {
            acc = grad_s;
        } else {
            acc = grad_s + acc * expf(s - s_val_next);
        }
        float grad_z = acc * expf(z - s);
        s_val_next = s;

        float grad_a = grad_log_h + carry_grad_a - grad_z;
        carry_grad_a = grad_a;

        if (t == 0) {
            grad_state[state_idx] = T(grad_z / float(state[state_idx]));
        } else {
            float grad_g, grad_h;
            log_coeffs_and_values_bwd(grad_a, grad_z, gate_val, hidden_val, &grad_g, &grad_h);

            grad_combined[gate_adr] = T(grad_g);
            grad_combined[hidden_adr] = T(grad_h);
            grad_combined[proj_adr] = T(grad_proj);
        }
    }
}

// Hybrid Checkpointed backward (reads sparse checkpoints from forward, minimal recomputation)
// Simplified: CHECKPOINT_INTERVAL = chunk size (no separate CHUNK_SIZE)
// Key idea: Forward writes checkpoints every CHECKPOINT_INTERVAL, backward processes in same-sized chunks
template<typename T>
__global__ void fused_scan_backward_kernel_checkpointed(
    T* __restrict__ grad_combined,         // (B, T, 3*H) = [grad_hidden, grad_gate, grad_proj]
    T* __restrict__ grad_state,            // (B, 1, H)
    const T* __restrict__ grad_out,        // (B, T, H) - gradient of sigmoid(proj)*scan_result
    const T* __restrict__ grad_next_state, // (B, 1, H) - gradient of raw scan_result at T
    const T* __restrict__ combined,        // (B, T, 3*H) = [hidden, gate, proj]
    const T* __restrict__ state,           // (B, 1, H)
    const float* __restrict__ a_star_buf,  // (B, T+1, H) sparse checkpoints from forward
    const float* __restrict__ s_buf,       // (B, T+1, H) sparse checkpoints from forward
    const float* __restrict__ log_values_buf, // (B, T+1, H) sparse checkpoints from forward
    int T_seq,                             // sequence length (T)
    int H,
    int B
) {
    // Standard indexing with 128 threads per block
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;

    int b = idx / H;
    int h = idx % H;

    // Precompute all base addresses once
    int bHT = b * H * T_seq;
    int cbase = 3 * bHT;
    int H3 = 3 * H;
    int H2 = 2 * H;
    const int state_idx = b * H + h;
    const int out_base = bHT + h;
    
    // Base pointers for reads
    const T* combined_h_base = &combined[cbase + h];
    const T* combined_g_base = &combined[cbase + H + h];
    const T* combined_p_base = &combined[cbase + H2 + h];
    
    // Base pointers for writes
    T* grad_combined_h_base = &grad_combined[cbase + h];
    T* grad_combined_g_base = &grad_combined[cbase + H + h];
    T* grad_combined_p_base = &grad_combined[cbase + H2 + h];
    
    // Base pointer for checkpoint buffers
    int T_out = T_seq + 1;
    int buf_base = b * T_out * H + h;

    // Backward pass state
    float acc = 0.0;
    float s_val_next = 0.0;
    float carry_grad_a = 0.0;
    
    // Process chunks from end to beginning (chunk size = CHECKPOINT_INTERVAL)
    for (int chunk_end = T_seq; chunk_end > 0; chunk_end -= CHECKPOINT_INTERVAL) {
        int chunk_start = (chunk_end > CHECKPOINT_INTERVAL) ? (chunk_end - CHECKPOINT_INTERVAL) : 0;
        int chunk_len = chunk_end - chunk_start;
        
        // Chunk storage in registers (CHECKPOINT_INTERVAL timesteps max)
        // Store intermediate values AND hidden/gate (not proj - read it separately)
        float chunk_a_star[CHECKPOINT_INTERVAL];
        float chunk_s[CHECKPOINT_INTERVAL];
        float chunk_log_values[CHECKPOINT_INTERVAL];
        float chunk_hidden[CHECKPOINT_INTERVAL];
        float chunk_gate[CHECKPOINT_INTERVAL];
        
        // Load checkpoint at chunk_start from global memory
        int ckpt_buf_idx = buf_base + chunk_start * H;
        float recomp_a_star = a_star_buf[ckpt_buf_idx];
        float recomp_s = s_buf[ckpt_buf_idx];
        float recomp_log_value = log_values_buf[ckpt_buf_idx];
        
        // Recompute from chunk_start to chunk_end, storing values for backward
        for (int i = 0; i < chunk_len; ++i) {
            int t = chunk_start + 1 + i;
            int t_offset = (t - 1) * H3;
            float hv = float(combined_h_base[t_offset]);
            float gv = float(combined_g_base[t_offset]);
            
            float lc;
            log_coeffs_and_values_fwd(gv, hv, &lc, &recomp_log_value);
            recomp_a_star += lc;
            
            float z = recomp_log_value - recomp_a_star;
            float mv = fmaxf(recomp_s, z);
            recomp_s = mv + log1pf(__expf(-fabsf(recomp_s - z)));
            
            // Store in chunk arrays
            chunk_a_star[i] = recomp_a_star;
            chunk_s[i] = recomp_s;
            chunk_log_values[i] = recomp_log_value;
            chunk_hidden[i] = hv;
            chunk_gate[i] = gv;
        }
        
        // Process backward through this chunk using stored values
        for (int i = chunk_len - 1; i >= 0; --i) {
            int t = chunk_start + 1 + i;
            int t_offset = (t - 1) * H3;
            
            float a_star_t = chunk_a_star[i];
            float s_t = chunk_s[i];
            float log_value_t = chunk_log_values[i];
            float hidden_val = chunk_hidden[i];
            float gate_val = chunk_gate[i];
            
            // Read proj from global memory (not stored to save registers)
            float proj_val = float(combined_p_base[t_offset]);
            
            float scan_result = __expf(a_star_t + s_t);
            float z = log_value_t - a_star_t;
            
            // Read grad_out from global memory
            float grad_out_val = float(grad_out[out_base + (t - 1) * H]);
            
            // Handle grad_next_state only for t == T_seq
            float grad_scan_from_next = (t == T_seq) ? float(grad_next_state[state_idx]) : 0.0f;
            
            float proj_sigmoid = sigmoid(proj_val);
            float grad_scan_result = grad_scan_from_next + grad_out_val * proj_sigmoid;
            float grad_proj = grad_out_val * scan_result * proj_sigmoid * (1.0f - proj_sigmoid);
            
            float grad_log_h = grad_scan_result * scan_result;
            float grad_s = grad_log_h;
            
            // Accumulator logic
            if (t == T_seq) {
                acc = grad_s;
            } else {
                acc = grad_s + acc * __expf(s_t - s_val_next);
            }
            float grad_z = acc * __expf(z - s_t);
            s_val_next = s_t;
            
            float grad_a = grad_log_h + carry_grad_a - grad_z;
            carry_grad_a = grad_a;
            
            // Compute and write gradients
            float grad_g, grad_h;
            log_coeffs_and_values_bwd(grad_a, grad_z, gate_val, hidden_val, &grad_g, &grad_h);
            
            grad_combined_h_base[t_offset] = T(grad_h);
            grad_combined_g_base[t_offset] = T(grad_g);
            grad_combined_p_base[t_offset] = T(grad_proj);
        }
    }
    
    // Handle t = 0 (initial state gradient)
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
    
    grad_state[state_idx] = T(grad_z_0 / float(state[state_idx]));
}

template<typename T>
void launch_fused_scan_backward(
    T* grad_combined,
    T* grad_state,
    const T* grad_out,
    const T* grad_next_state,
    const T* combined,
    const T* state,
    const float* a_star_buf,
    const float* s_buf,
    const float* log_values_buf,
    int T_seq,
    int H,
    int B,
    cudaStream_t stream
) {
    int total = B * H;
    int grid = seq_size(total);

    fused_scan_backward_kernel<T><<<grid, SEQ_SIZE, 0, stream>>>(
        grad_combined,
        grad_state,
        grad_out,
        grad_next_state,
        combined,
        state,
        a_star_buf,
        s_buf,
        log_values_buf,
        T_seq,
        H,
        B
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error in backward: %s\n", cudaGetErrorString(err));
    }
}

template<typename T>
void launch_fused_scan_backward_checkpointed(
    T* grad_combined,
    T* grad_state,
    const T* grad_out,
    const T* grad_next_state,
    const T* combined,
    const T* state,
    const float* a_star_buf,     // Sparse checkpoints from forward pass
    const float* s_buf,           // Sparse checkpoints from forward pass
    const float* log_values_buf,  // Sparse checkpoints from forward pass
    int T_seq,
    int H,
    int B,
    cudaStream_t stream
) {
    // Standard launch with 128 threads per block
    int total = B * H;
    int grid = opt_grid_size(total);

    fused_scan_backward_kernel_checkpointed<T><<<grid, OPT_BLOCK_SIZE, 0, stream>>>(
        grad_combined,
        grad_state,
        grad_out,
        grad_next_state,
        combined,
        state,
        a_star_buf,
        s_buf,
        log_values_buf,
        T_seq,
        H,
        B
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error in checkpointed backward: %s\n", cudaGetErrorString(err));
    }
}

// ============================================================================
// C interface for ctypes
// ============================================================================

extern "C" {

void launch_fused_scan_forward_original(
    float* out,
    float* next_state,
    float* a_star,
    float* s_vals,
    float* log_values_buf,
    const float* combined,
    const float* state,
    int T_seq,
    int H,
    int B
) {
    launch_fused_scan_forward<float>(
        out, next_state, a_star, s_vals, log_values_buf,
        combined, state, T_seq, H, B, nullptr
    );
}

void launch_fused_scan_forward_optimized(
    float* out,
    float* next_state,
    float* a_star,
    float* s_vals,
    float* log_values_buf,
    const float* combined,
    const float* state,
    int T_seq,
    int H,
    int B
) {
    launch_fused_scan_forward_new<float>(
        out, next_state, a_star, s_vals, log_values_buf,
        combined, state, T_seq, H, B, nullptr
    );
}

void launch_fused_scan_backward_original(
    float* grad_combined,
    float* grad_state,
    const float* grad_out,
    const float* grad_next_state,
    const float* combined,
    const float* state,
    const float* a_star_buf,
    const float* s_buf,
    const float* log_values_buf,
    int T_seq,
    int H,
    int B
) {
    launch_fused_scan_backward<float>(
        grad_combined, grad_state, grad_out, grad_next_state,
        combined, state, a_star_buf, s_buf, log_values_buf,
        T_seq, H, B, nullptr
    );
}

void launch_fused_scan_backward_checkpointed_wrapper(
    float* grad_combined,
    float* grad_state,
    const float* grad_out,
    const float* grad_next_state,
    const float* combined,
    const float* state,
    const float* a_star_buf,
    const float* s_buf,
    const float* log_values_buf,
    int T_seq,
    int H,
    int B
) {
    launch_fused_scan_backward_checkpointed<float>(
        grad_combined, grad_state, grad_out, grad_next_state,
        combined, state, a_star_buf, s_buf, log_values_buf,
        T_seq, H, B, nullptr
    );
}

void sync_device() {
    cudaDeviceSynchronize();
}

const char* get_last_error() {
    cudaError_t err = cudaGetLastError();
    return cudaGetErrorString(err);
}

} // extern "C"
