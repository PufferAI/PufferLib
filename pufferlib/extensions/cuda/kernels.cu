#include <torch/extension.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "ops.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <iostream>

#define SEQ_SIZE 32
#define BLOCK_SIZE 256
inline int grid_size(int N) {
    return (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
}
inline int seq_size(int N) {
    return (N + SEQ_SIZE - 1) / SEQ_SIZE;
}

// If you can get this to work, go ahead. I tried.
// NVCC won't parse templated types in kernel launches
/*
template <template <class> class KernelFn, typename... Args>
void dispatch_and_launch(const at::Tensor& example_tensor, Args... args) {
    const int64_t N = example_tensor.numel();
    const int64_t block = LAUNCH_BLOCK_SIZE;
    const int64_t grid = (N + block - 1) / block;
    auto stream = at::cuda::getCurrentCUDAStream();
    at::cuda::CUDAGuard device_guard(example_tensor.device());

    at::ScalarType dtype = example_tensor.scalar_type();
    if (dtype == at::ScalarType::Float) {
        KernelFn<float><<<grid, block, 0, stream>>>(args..., N);
    } else if (dtype == at::ScalarType::Half) {
        KernelFn<__half><<<grid, block, 0, stream>>>(args..., N);
    } else if (dtype == at::ScalarType::BFloat16) {
        KernelFn<__nv_bfloat16><<<grid, block, 0, stream>>>(args..., N);
    } else {
        AT_ERROR("Unsupported dtype: ", dtype);
    }
}
*/

template<typename T>
__global__ void mingru_gate_inference_kernel(
    T* out,
    const T* gate_in,
    const T* hidden_in,
    const T* state_in,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float gate = float(gate_in[idx]);
    float hidden = float(hidden_in[idx]);
    float state = float(state_in[idx]);
    float gate_sigmoid = fast_sigmoid(gate);
    float hidden_tilde = tilde_relu_fwd(hidden);
    float out_val = lerp(state, hidden_tilde, gate_sigmoid);
    out[idx] = T(out_val);
}

template<typename T>
void launch_mingru_gate_inference(
    T* out,
    const T* gate_in,
    const T* hidden_in,
    const T* state_in,
    int N
) {
    int grid = grid_size(N);
    mingru_gate_inference_kernel<T><<<grid, BLOCK_SIZE>>>(
        out,
        gate_in,
        hidden_in,
        state_in,
        N
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}


template<typename T>
__global__ void log_coeffs_and_values_kernel(
    T* log_coeffs,
    T* log_values,
    const T* gate,
    const T* hidden,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float g = float(gate[idx]);
    float h = float(hidden[idx]);

    log_coeffs[idx] = -softplus_fwd(g);
    float log_z = -softplus_fwd(-g);
    float log_tilde_h;
    if (h >= 0.0f) {
        float relu_h = relu(h);
        log_tilde_h = logf(relu_h + 0.5f);
    } else {
        log_tilde_h = -softplus_fwd(-h);
    }
    log_values[idx] = log_z + log_tilde_h;
}

template<typename T>
__global__ void log_coeffs_and_values_backward_kernel(
    T* grad_gate,
    T* grad_hidden,
    const T* grad_log_coeffs,
    const T* grad_log_values,
    const T* gate,
    const T* hidden,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float g = float(gate[idx]);
    float h = float(hidden[idx]);

    float grad_lc = float(grad_log_coeffs[idx]);
    float grad_lv = float(grad_log_values[idx]);
    float grad_g_from_lc = -softplus_bwd(grad_lc, g);
    float grad_g_from_lz = -softplus_bwd(-grad_lv, -g);
    float grad_g_total = grad_g_from_lc + grad_g_from_lz;
    grad_gate[idx] = T(grad_g_total);
    float log_tilde_h;
    float grad_h_from_lt;
    if (h >= 0.0f) {
        float relu_h = relu(h);
        log_tilde_h = logf(relu_h + 0.5f);
        float inner_grad = 1.0f / (relu_h + 0.5f);
        grad_h_from_lt = relu_backward(h, inner_grad * grad_lv);
    } else {
        log_tilde_h = -softplus_fwd(-h);
        grad_h_from_lt = -softplus_bwd(-grad_lv, -h);
    }
    grad_hidden[idx] = T(grad_h_from_lt);
}

template<typename T>
void launch_log_coeffs_and_values(
    T* log_coeffs,
    T* log_values,
    const T* gate,
    const T* hidden,
    int N
) {
    int grid = grid_size(N);
    log_coeffs_and_values_kernel<T><<<grid, BLOCK_SIZE>>>(
        log_coeffs,
        log_values,
        gate,
        hidden,
        N
    );

    // Optional: Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

template<typename T>
void launch_log_coeffs_and_values_backward(
    T* grad_gate,
    T* grad_hidden,
    const T* grad_log_coeffs,
    const T* grad_log_values,
    const T* gate,
    const T* hidden,
    int N
) {
    int grid = grid_size(N);
    log_coeffs_and_values_backward_kernel<T><<<grid, BLOCK_SIZE>>>(
        grad_gate,
        grad_hidden,
        grad_log_coeffs,
        grad_log_values,
        gate,
        hidden,
        N
    );

    // Optional: Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

template<typename T>
__global__ void fused_scan_forward_kernel(
    T* __restrict__ out,
    double* __restrict__ a_star_buf,
    double* __restrict__ s_buf,
    const T* __restrict__ log_coeffs,
    const T* __restrict__ log_values,
    int T_total,
    int H,
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;

    int b = idx / H;
    int h = idx % H;

    int base = b * T_total * H + h;

    double a_star = 0.0f;
    double s = -INFINITY;  // this will be logcumsumexp(z[0..t])

    for (int t = 0; t < T_total; t++) {
        int curr = base + t * H;

        // Step 1: Update a_star[t] = sum_{i=0}^t log_coeffs[i]
        a_star += double(log_coeffs[curr]);

        // Step 2: Compute z[t] = log_values[t] - a_star[t]
        double z_val = double(log_values[curr]) - a_star;

        // Step 3: logcumsumexp on z — EXACTLY as in logcumsumexp_forward_kernel
        if (s == -INFINITY) {
            s = z_val;
        } else {
            double max_val = fmax(s, z_val);
            double diff = fabs(s - z_val);
            s = max_val + log1p(exp(-diff));
        }

        // Step 4: log_h[t] = a_star[t] + s[t], then out[t] = exp(log_h[t])
        double log_h = a_star + s;
        out[curr] = T(exp(log_h));

        // Step 5: Save intermediates (same as before)
        a_star_buf[curr] = a_star;
        s_buf[curr] = s;
    }
}

template<typename T>
__global__ void fused_scan_backward_kernel(
    T* __restrict__ grad_log_coeffs,
    T* __restrict__ grad_log_values,
    const T* __restrict__ d_out,
    const T* __restrict__ out,
    const double* __restrict__ a_star_buf,
    const double* __restrict__ s_buf,
    const T* __restrict__ log_values,
    int T_total,
    int H,
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;

    int b = idx / H;
    int h = idx % H;

    int base = b * T_total * H + h;

    double carry_d_a = 0.0;
    double carry_d_s = 0.0;

    for (int t = T_total - 1; t >= 0; --t) {
        int curr = base + t * H;

        double A = a_star_buf[curr];
        double S = s_buf[curr];
        double Z = double(log_values[curr]) - A;
        double out_val = double(out[curr]);
        double d_log_h = double(d_out[curr]) * out_val;

        double d_S = d_log_h + carry_d_s;

        double S_prev = (t == 0) ? -INFINITY : s_buf[base + (t - 1) * H];

        double max_val = fmax(S_prev, Z);
        double exp_prev = (S_prev == -INFINITY) ? 0.0 : exp(S_prev - max_val);
        double exp_z = (Z == -INFINITY) ? 0.0 : exp(Z - max_val);
        double denom = exp_prev + exp_z;
        double frac_prev = (denom == 0.0) ? 0.0 : exp_prev / denom;
        double frac_z = (denom == 0.0) ? 0.0 : exp_z / denom;

        double d_Z = frac_z * d_S;
        double d_A = d_log_h + carry_d_a - d_Z;

        grad_log_values[curr] = T(d_Z);
        grad_log_coeffs[curr] = T(d_A);

        carry_d_a = d_A;
        carry_d_s = frac_prev * d_S;
    }
}

/*
template<typename T>
__global__ void fused_scan_backward_kernel(
    T* __restrict__ grad_log_coeffs,
    T* __restrict__ grad_log_values,
    const T* __restrict__ grad_out,
    const T* __restrict__ log_coeffs,
    const T* __restrict__ log_values,
    const T* __restrict__ out,
    const double* __restrict__ a_star_buf,
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

    double grad_a_star[1025] = {0};  // Assuming T_total <= 1024
    double W = 0.0;  // Accumulates sum_{i=t}^{T-1} [grad_log_h[i] * exp(-s[i])]

    for (int t = T_total - 1; t >= 0; t--) {
        int curr = base + t * H;

        double a_star = a_star_buf[curr];
        double s_val = s_buf[curr];
        double z_val = double(log_values[curr]) - a_star;

        // Compute dL/d(log_h[t]) = dL/d(out[t]) * d(out[t])/d(log_h[t])
        double grad_log_h = double(grad_out[curr]) * double(out[curr]);

        // Update W: W[t] = grad_log_h[t] * exp(-s_val) + W[t+1]
        W = grad_log_h * exp(-s_val) + W;

        // Compute dL/d(z[t]) = exp(z_val) * W[t]
        double grad_z = exp(z_val) * W;

        // dL/d(log_values[t]) = dL/d(z[t]) * dz[t]/d(log_values[t]) = grad_z
        grad_log_values[curr] = T(grad_z);

        // dL/da_star[t] = dL/d(log_h[t]) - dL/d(z[t]) (due to chain rule)
        grad_a_star[t] = grad_log_h - grad_z;
    }

    // Compute dL/d(log_coeffs) via cumulative sum of dL/da_star
    double accum = 0.0;
    for (int t = T_total - 1; t >= 0; t--) {
        accum += grad_a_star[t];
        grad_log_coeffs[base + t * H] = T(accum);
    }
}
*/


/*
template<typename T>
__global__ void fused_scan_backward_kernel(
    T* __restrict__ grad_log_coeffs,
    T* __restrict__ grad_log_values,
    const T* __restrict__ grad_out,
    const T* __restrict__ log_coeffs,
    const T* __restrict__ log_values,
    const T* __restrict__ out,
    const float* __restrict__ a_star_buf,
    const float* __restrict__ s_buf,
    int T_total,
    int H,
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;

    int b = idx / H;
    int h = idx % H;

    int base = b * T_total * H + h;

    float grad_a_star[1025] = {0};
    float G = 0.0f;  // G[t] = sum_{i=t}^{T-1} grad_s[i]
    for (int t = T_total - 1; t >= 0; t--) {
        int curr = base + t * H;

        float a_star = a_star_buf[curr];
        float s_val = s_buf[curr];
        float z = float(log_values[curr]) - a_star;

        // grad_log_h[t] = grad_out[t] * out[t]
        float grad_log_h = float(grad_out[curr]) * float(out[curr]);

        // G = sum of grad_s from t to end (grad_s[t] = grad_log_h[t])
        G += grad_log_h;

        // grad_z[t] = exp(z - s_val) * G
        float prob = expf(z - s_val);
        float grad_z = prob * G;

        // grad_log_values[t] = grad_z
        grad_log_values[curr] = T(grad_z);

        // grad_a_star[t] gets:
        // - +grad_log_h (from log_h = a_star + s)
        // - -grad_z    (from z = log_values - a_star)
        grad_a_star[t] = grad_log_h - grad_z;
    }

    // grad_log_coeffs[t] = sum_{i=t}^{T-1} grad_a_star[i]
    float accum = 0.0f;
    for (int t = T_total - 1; t >= 0; t--) {
        accum += grad_a_star[t];
        grad_log_coeffs[base + t * H] = T(accum);
    }
}

 template<typename T>
__global__ void fused_scan_backward_kernel(
    T* __restrict__ grad_log_coeffs,
    T* __restrict__ grad_log_values,
    const T* __restrict__ grad_out,
    const T* __restrict__ log_coeffs,
    const T* __restrict__ log_values,
    const T* __restrict__ out,
    const float* __restrict__ a_star_buf,
    const float* __restrict__ s_buf,
    int T_total,
    int H,
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;

    int b = idx / H;
    int h = idx % H;

    int base = b * T_total * H + h;

    // Recompute z[t] = log_values[t] - a_star[t]
    float z[1025];
    for (int t = 0; t < T_total; t++) {
        int curr = base + t * H;
        z[t] = float(log_values[curr]) - a_star_buf[curr];
    }

    // g_log_h[t] = grad_out[t] * out[t]
    float g_log_h[1025];
    for (int t = 0; t < T_total; t++) {
        int curr = base + t * H;
        g_log_h[t] = float(grad_out[curr]) * float(out[curr]);
    }

    // Step: Online logcumsumexp backward for g_z
    float g_z[1025] = {0};
    g_z[T_total - 1] = g_log_h[T_total - 1];

    for (int t = T_total - 2; t >= 0; t--) {
        float exp_term = expf(z[t] - s_buf[base + (t + 1) * H]);
        g_z[t] = g_log_h[t] + g_z[t + 1] * exp_term;
    }

    // grad_log_values[t] = g_z[t]
    for (int t = 0; t < T_total; t++) {
        int curr = base + t * H;
        grad_log_values[curr] = T(g_z[t]);
    }

    // g_a_star[t] = g_log_h[t] - g_z[t]
    float g_a_star[1025] = {0};
    for (int t = 0; t < T_total; t++) {
        g_a_star[t] = g_log_h[t] - g_z[t];
    }

    // grad_log_coeffs[t] = reverse cumsum of g_a_star
    float accum = 0.0f;
    for (int t = T_total - 1; t >= 0; t--) {
        accum += g_a_star[t];
        grad_log_coeffs[base + t * H] = T(accum);
    }
}
*/
// This one tests correct but asserts
template<typename T>
void launch_fused_scan_forward(
    T* out,
    double* a_star,
    double* s_vals,
    const T* log_coeffs,
    const T* log_values,
    int T_seq,
    int H,
    int B
) {
    int total = B * H;
    int grid = seq_size(total);

    fused_scan_forward_kernel<T><<<grid, SEQ_SIZE>>>(
        out,
        a_star,
        s_vals,
        log_coeffs,
        log_values,
        T_seq,
        H,
        B
    );
 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error in forward: %s\n", cudaGetErrorString(err));
    }
}

template<typename T>
void launch_fused_scan_backward(
    T* grad_log_coeffs,
    T* grad_log_values,
    const T* grad_out,
    const T* log_coeffs,
    const T* log_values,
    const T* out,
    const double* a_star_buf,
    const double* s_buf,
    int T_seq,
    int H,
    int B
) {
    int total = B * H;
    int grid = seq_size(total);

    fused_scan_backward_kernel<T><<<grid, SEQ_SIZE>>>(
        grad_log_coeffs,
        grad_log_values,
        grad_out,
        out,
        a_star_buf,
        s_buf,
        log_values,
        T_seq,
        H,
        B
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error in backward: %s\n", cudaGetErrorString(err));
    }
}

/*
__device__ __forceinline__ float log_add_exp(const float a, const float b) {
  if (::isnan(a) || ::isnan(b)) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  float min_val = fminf(a, b);
  float max_val = fmaxf(a, b);
  if (min_val != max_val || ::isfinite(min_val)) {
    return max_val + log1pf(expf(min_val - max_val));
  } else {
      return a;
  }
}

__device__ __forceinline__ float log_add_exp_backward(float x_val, float s_val) {
  if (::isnan(x_val) || ::isnan(s_val)) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return expf(x_val - s_val);
}
*/

__device__ __forceinline__ double log_add_exp(const double a, const double b) {
  double min_val = fmin(a, b);
  double max_val = fmax(a, b);
  return max_val + log1p(exp(min_val - max_val));
}

__device__ __forceinline__ double log_add_exp_backward(double x, double s) {
    return exp(x - s);
}
  
// This exactly matches pytorch in double, but not in float
template<typename T>
__global__ void logcumsumexp_forward_kernel(
    T* __restrict__ out,           // exp(s[t])
    double* __restrict__ s_buf,     // s[t] = logcumsumexp(x[0..t])
    const T* __restrict__ x,       // input: log_values
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
        double x_val = double(x[curr]);

        if (s == -INFINITY) {
            s = x_val;
        } else {
            s = log_add_exp(s, x_val);
        }

        out[curr] = T(s);
        s_buf[curr] = s;
    }
}
template<typename T>
__global__ void logcumsumexp_backward_kernel(
    T* __restrict__ grad_x,
    const T* __restrict__ grad_s,
    const T* __restrict__ x,
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

    for (int t = T_total - 1; t >= 0; --t) {
        int curr = base + t * H;

        double x_val = double(x[curr]);
        double s_val = double(s_buf[curr]);
        double g_val = double(grad_s[curr]);

        if (t == T_total - 1) {
            acc = g_val;
        } else {
            double s_next_val = double(s_buf[curr + H]);
            acc = g_val + acc*log_add_exp_backward(s_val, s_next_val);
        }

        grad_x[curr] = T(acc*log_add_exp_backward(x_val, s_val));
    }
}
/*
template<typename T>
__global__ void logcumsumexp_backward_kernel(
    T* __restrict__ grad_x,
    const T* __restrict__ grad_out,
    const T* __restrict__ x,
    const float* __restrict__ s_buf,
    int T_total,
    int H,
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;

    int b = idx / H;
    int h = idx % H;

    int base = b * T_total * H + h;

    // grad_x[i] = sum_{t≥i} grad_out[t] * exp(x[i] - s[t])
    for (int i = 0; i < T_total; i++) {
        int curr_i = base + i * H;
        float x_i = float(x[curr_i]);
        float g = 0.0f;

        for (int t = i; t < T_total; t++) {
            int curr_t = base + t * H;
            float s_t = s_buf[curr_t];
            float prob = expf(x_i - s_t);
            //float prob = log_add_exp_backward(x_i, s_t);
            g += float(grad_out[curr_t]) * prob;
        }

        grad_x[curr_i] = T(g);
    }
}
*/

template<typename T>
void launch_logcumsumexp_forward(
    T* out,
    double* s_buf,
    const T* x,
    int T_total,
    int H,
    int B
) {
    int total = B * H;
    int grid = grid_size(total);

    logcumsumexp_forward_kernel<T><<<grid, BLOCK_SIZE>>>(
        out, s_buf, x, T_total, H, B
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "Forward kernel error: %s\n", cudaGetErrorString(err));
}

template<typename T>
void launch_logcumsumexp_backward(
    T* grad_x,
    const T* grad_out,
    const T* x,
    const double* s_buf,
    int T_total,
    int H,
    int B
) {
    int total = B * H;
    int grid = grid_size(total);

    logcumsumexp_backward_kernel<T><<<grid, BLOCK_SIZE>>>(
        grad_x, grad_out, x, s_buf, T_total, H, B
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "Backward kernel error: %s\n", cudaGetErrorString(err));
}

template<typename T>
__global__ void ppo_loss_forward_kernel(
    float* __restrict__ loss,
    double* __restrict__ saved_for_backward,
    const T* __restrict__ logits,
    const T* __restrict__ values_pred,
    const int64_t* __restrict__ actions,
    const T* __restrict__ old_logprobs,
    const T* __restrict__ advantages,
    const T* __restrict__ prio,
    const T* __restrict__ values,
    const T* __restrict__ returns,
    double adv_mean,
    double adv_std,
    double clip_coef,
    double vf_clip_coef,
    double vf_coef,
    double ent_coef,
    int T_seq,
    int A,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * T_seq;
    if (idx >= total_elements) return;
    __shared__ float block_loss[BLOCK_SIZE];

    int n = idx / T_seq;  // batch index
    int t = idx % T_seq;  // timestep

    // === Direct indexing: no lambdas ===
    int nt = n * T_seq + t;                    // index into (N, T_seq) tensors
    int logits_offset = n * T_seq * A + t * A; // base index into logits

    // === Step 1: Read action and compute logsumexp ===
    int act = actions[nt];  // action taken at (n,t)

    // Compute logsumexp: log(sum_a exp(logits[a]))
    double max_logit = -INFINITY;
    for (int a = 0; a < A; a++) {
        double l = double(logits[logits_offset + a]);
        max_logit = fmax(max_logit, l);
    }

    double logsumexp = 0.0;
    double sum = 0.0;
    for (int a = 0; a < A; a++) {
        double l = double(logits[logits_offset + a]);
        sum += exp(l - max_logit);
    }
    logsumexp = max_logit + log(sum);

    // === Step 2: new_logprob[action] = logits[action] - logsumexp ===
    // log_softmax = (logits - max_logit) - max_logit - logsumexp
    double new_logp = double(logits[logits_offset + act]) - logsumexp;

    // === Step 3: entropy = -sum_a p_a * log p_a ===
    double entropy = 0.0;
    for (int a = 0; a < A; a++) {
        double l = double(logits[logits_offset + a]);
        double p = exp(l - logsumexp);
        double logp = l - logsumexp;
        entropy -= p * logp;
    }

    // === Step 4: policy gradient loss ===
    double old_logp = double(old_logprobs[nt]);
    double adv = double(advantages[nt]);
    double w = double(prio[n]);  // importance weight, per-sequence
    double adv_normalized = (adv - adv_mean) / (adv_std + 1e-8);

    double logratio = new_logp - old_logp;
    double ratio = exp(logratio);

    double ratio_clipped = fmax(1.0 - clip_coef, fmin(1.0 + clip_coef, ratio));
    double pg_loss1 = -w * adv_normalized * ratio;
    double pg_loss2 = -w * adv_normalized * ratio_clipped;
    double pg_loss = fmax(pg_loss1, pg_loss2);  // PPO clipped surrogate loss

    // === Step 5: value function loss ===
    double val = double(values[nt]);
    double ret = double(returns[nt]);
    double val_pred = double(values_pred[nt]);

    double v_error = val_pred - val;
    double v_clipped = val + fmax(-vf_clip_coef, fmin(vf_clip_coef, v_error));
    double v_loss_unclipped = (val_pred - ret) * (val_pred - ret);
    double v_loss_clipped = (v_clipped - ret) * (v_clipped - ret);
    double v_loss = 0.5f * fmax(v_loss_unclipped, v_loss_clipped);

    // === Step 6: total sample loss ===
    double thread_loss = pg_loss + vf_coef * v_loss - ent_coef * entropy;

    // === Save for backward ===
    double* saved_row = saved_for_backward + idx * 5;
    saved_row[0] = new_logp;
    saved_row[1] = ratio;
    saved_row[2] = val_pred;
    saved_row[3] = v_clipped;
    saved_row[4] = entropy;

    // === Block-local reduction using shared memory ===
    int tid = threadIdx.x;
    block_loss[tid] = thread_loss;
    __syncthreads();

    // Reduce within block using tree reduction
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            block_loss[tid] += block_loss[tid + stride];
        }
        __syncthreads();
    }

    // === Accumulate into loss_output (scalar via atomic add) ===
    if (tid == 0) {
        atomicAdd(loss, block_loss[0]);
    }
}

template<typename T>
__global__ void ppo_loss_backward_kernel(
    T* __restrict__ grad_logits,
    T* __restrict__ grad_values_pred,
    const float* __restrict__ grad_loss,  // scalar, [1], dL/dloss
    const T* __restrict__ logits,
    const int64_t* __restrict__ actions,
    const T* __restrict__ old_logprobs,
    const T* __restrict__ advantages,
    const T* __restrict__ prio,
    const T* __restrict__ values,
    const T* __restrict__ returns,
    const double* __restrict__ saved_for_backward,
    double adv_mean,
    double adv_std,
    double clip_coef,
    double vf_clip_coef,
    double vf_coef,
    double ent_coef,
    int T_seq,
    int A,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * T_seq;
    if (idx >= total_elements) return;

    double inv_NT = 1.0f / (N * T_seq);
    int n = idx / T_seq;
    int t = idx % T_seq;

    // === Direct indexing ===
    int nt = n * T_seq + t;
    int logits_offset = n * T_seq * A + t * A;

    // === Retrieve saved values from forward pass ===
    const double* saved = saved_for_backward + idx * 5;
    double new_logp = saved[0];   // new log prob of selected action
    double ratio = saved[1];      // exp(new_logp - old_logp)
    double val_pred = saved[2];   // value prediction
    double v_clipped = saved[3];  // clipped value target
    double entropy = saved[4];    // entropy at (n,t)

    // === Read inputs ===
    double old_logp = double(old_logprobs[nt]);
    double adv = double(advantages[nt]);
    double w = double(prio[n]);  // importance weight
    double val = double(values[nt]);
    double ret = double(returns[nt]);

    // === Normalize advantage (same as forward) ===
    double adv_normalized = (adv - adv_mean) / (adv_std + 1e-8f);

    // Total loss gradient (scalar from autograd)
    double dL = grad_loss[0] * inv_NT;  // dL/dloss

    // Gradients w.r.t. components
    double d_pg_loss = dL;                    // policy loss contributes dL
    double d_v_loss = dL * vf_coef;           // value loss scaled by vf_coef
    double d_entropy_term = dL * (-ent_coef); // entropy bonus gradient

    // ===================================================
    // 1. Gradient w.r.t. value function prediction
    // ===================================================
    double v_loss_unclipped = (val_pred - ret) * (val_pred - ret);
    double v_loss_clipped = (v_clipped - ret) * (v_clipped - ret);

    // Which branch was taken in forward? (same logic as PyTorch: use unclipped if tie)
    bool use_clipped_vf = (v_loss_clipped > v_loss_unclipped);
    double d_val_pred = 0.0;

    if (use_clipped_vf) {
        double v_error = val_pred - val;
        if (v_error >= -vf_clip_coef && v_error <= vf_clip_coef) {
            d_val_pred = v_clipped - ret;  // = val_pred - ret
        }
    } else {
        d_val_pred = val_pred - ret;
    }

    d_val_pred = dL * vf_coef * d_val_pred;
    grad_values_pred[nt] = T(d_val_pred);

    // ===================================================
    // 2. Gradient w.r.t. policy and entropy (logits)
    // ===================================================
    // Recompute logsumexp for gradient
    double max_logit = -INFINITY;
    for (int a = 0; a < A; a++) {
        double l = double(logits[logits_offset + a]);
        max_logit = fmax(max_logit, l);
    }

    double logsumexp = 0.0;
    double sum = 0.0;
    for (int a = 0; a < A; a++) {
        double l = double(logits[logits_offset + a]);
        sum += exp(l - max_logit);
    }
    logsumexp = max_logit + log(sum);
 
    // Zero grad_logits for this (n,t)
    for (int a = 0; a < A; a++) {
        grad_logits[logits_offset + a] = T(0.0f);
    }

    // --- Policy Loss Gradient ---
    double logratio = new_logp - old_logp;
    double ratio_clipped = fmax(1.0f - clip_coef, fmin(1.0f + clip_coef, ratio));
    double pg_loss1 = -w * adv_normalized * ratio;
    double pg_loss2 = -w * adv_normalized * ratio_clipped;

    double d_ratio = -w * adv_normalized * d_pg_loss;
    if (pg_loss2 > pg_loss1) {
        if (ratio <= (1.0 - clip_coef) || ratio >= (1.0 + clip_coef)) {
            d_ratio = 0.0;
        }
    }

    // d(ratio)/d(new_logp) = ratio
    double d_new_logp = d_ratio * ratio;

    // --- Entropy Gradient ---
    // dH/dlogits[a] = p_a * (entropy - log p_a)
    for (int a = 0; a < A; a++) {
        double l = double(logits[logits_offset + a]);
        double p = exp(l - logsumexp);
        double logp = l - logsumexp;

        // Gradient from policy loss: d/dlogits[a] new_logp = δ_{a,act} - p_a
        double d_logit = 0.0f;
        if (a == actions[nt]) {
            d_logit += d_new_logp;
        }
        d_logit -= p * d_new_logp;

        // Gradient from entropy
        // TODO: Grad is a bit more off than I would like (1e-6)
        // Probably need to check logsumexp (not cumulative) vs
        // torch / actually look at the puffer 3 entropy impl
        double d_entropy_dlogit = p * (entropy - logp);
        d_logit += d_entropy_term * d_entropy_dlogit;

        grad_logits[logits_offset + a] = T(d_logit);
    }
}

template<typename T>
inline void launch_ppo_loss_forward(
    float* loss_output,
    double* saved_for_backward,
    const T* logits,
    const T* values_pred,
    const int64_t* actions,
    const T* old_logprobs,
    const T* advantages,
    const T* prio,
    const T* values,
    const T* returns,
    double adv_mean,
    double adv_std,
    double clip_coef,
    double vf_clip_coef,
    double vf_coef,
    double ent_coef,
    int T_seq,
    int A,
    int N
) {
    int total_elements = N * T_seq;
    int grid = grid_size(total_elements);

    ppo_loss_forward_kernel<T><<<grid, BLOCK_SIZE>>>(
        loss_output,
        saved_for_backward,
        logits,
        values_pred,
        actions,
        old_logprobs,
        advantages,
        prio,
        values,
        returns,
        adv_mean,
        adv_std,
        clip_coef,
        vf_clip_coef,
        vf_coef,
        ent_coef,
        T_seq,
        A,
        N
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "PPO forward kernel error: %s\n", cudaGetErrorString(err));
    }
}

template<typename T>
void launch_ppo_loss_backward(
    T* grad_logits,
    T* grad_values_pred,
    const float* grad_loss,
    const T* logits,
    const int64_t* actions,
    const T* old_logprobs,
    const T* advantages,
    const T* prio,
    const T* values,
    const T* returns,
    const double* saved_for_backward,
    double adv_mean,
    double adv_std,
    double clip_coef,
    double vf_clip_coef,
    double vf_coef,
    double ent_coef,
    int T_seq,
    int A,
    int N
) {
    int total_elements = N * T_seq;
    int grid = grid_size(total_elements);

    ppo_loss_backward_kernel<T><<<grid, BLOCK_SIZE>>>(
        grad_logits,
        grad_values_pred,
        grad_loss,
        logits,
        actions,
        old_logprobs,
        advantages,
        prio,
        values,
        returns,
        saved_for_backward,
        adv_mean,
        adv_std,
        clip_coef,
        vf_clip_coef,
        vf_coef,
        ent_coef,
        T_seq,
        A,
        N
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "PPO backward kernel error: %s\n", cudaGetErrorString(err));
    }
}
