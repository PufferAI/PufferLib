#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace pufferlib {

__host__ __device__ void puff_advantage_row_cuda(float* values, float* rewards, float* dones,
        float* importance, float* advantages, float value_bs, float reward_bs, float done_bs,
        float trunc_bs, float gamma, float lambda, float rho_clip, float c_clip, int horizon) {
    float lastpufferlam = 0.0;
    float next_value = 0.0;
    float nextnonterminal = 1.0;
    float next_reward = 0.0;
    for (int t = horizon-1; t >= 0; t--) {
        int t_next = t + 1;
        if ((t+1) == horizon) {
            nextnonterminal = 1.0 - done_bs;
            next_value = value_bs;
            next_reward = reward_bs;
        } else {
            nextnonterminal = 1.0 - dones[t_next];
            next_value = values[t_next];
            next_reward = rewards[t_next];
        }
        float rho_t = fminf(importance[t], rho_clip);
        float c_t = fminf(importance[t], c_clip);
        float delta = rho_t*(next_reward + gamma*next_value*nextnonterminal - values[t]);
        lastpufferlam = delta + gamma*lambda*c_t*lastpufferlam*nextnonterminal;
        advantages[t] = lastpufferlam;
    }
}

void vtrace_check_cuda(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        torch::Tensor values_bs, torch::Tensor rewards_bs, torch::Tensor dones_bs,
        torch::Tensor truncs_bs, int num_steps, int horizon) {

    // Validate input tensors
    torch::Device device = values.device();
    for (const torch::Tensor& t : {values, rewards, dones, importance, advantages}) {
        TORCH_CHECK(t.is_cuda(), "All tensors must be on GPU");
        TORCH_CHECK(t.dim() == 2, "Tensor must be 2D");
        TORCH_CHECK(t.device() == device, "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
        TORCH_CHECK(t.size(1) == horizon, "Second dimension must match horizon");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "All tensors must be float32");
        if (!t.is_contiguous()) {
            t.contiguous();
        }
    }
    for (const torch::Tensor& t : {values_bs, rewards_bs, dones_bs, truncs_bs}) {
        TORCH_CHECK(t.is_cuda(), "All tensors must be on GPU");
        TORCH_CHECK(t.dim() == 1, "Bootstrap Tensors must be 1D");
        TORCH_CHECK(t.device() == device, "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "All tensors must be float32");
        if (!t.is_contiguous()) {
            t.contiguous();
        }
    }
}

// [num_steps, horizon]
__global__ void puff_advantage_kernel(float* values, float* rewards,
        float* dones, float* importance, float* advantages, float* values_bs,
        float* rewards_bs, float* dones_bs, float* truncs_bs, float gamma,
        float lambda, float rho_clip, float c_clip, int num_steps, int horizon) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= num_steps) {
        return;
    }
    int offset = row*horizon;
    float value_bs = values_bs[row];
    float reward_bs = rewards_bs[row];
    float done_bs = dones_bs[row];
    float trunc_bs = truncs_bs[row]; 
    puff_advantage_row_cuda(values + offset, rewards + offset, dones + offset,
        importance + offset, advantages + offset, value_bs, reward_bs, done_bs, trunc_bs,
        gamma, lambda, rho_clip, c_clip, horizon);
}

void compute_puff_advantage_cuda(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        torch::Tensor values_bs, torch::Tensor rewards_bs, torch::Tensor dones_bs,
        torch::Tensor truncs_bs, double gamma, double lambda, double rho_clip, double c_clip) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    vtrace_check_cuda(values, rewards, dones, importance, advantages, values_bs, rewards_bs,
                      dones_bs, truncs_bs, num_steps, horizon);

    int threads_per_block = 256;
    int blocks = (num_steps + threads_per_block - 1) / threads_per_block;

    puff_advantage_kernel<<<blocks, threads_per_block>>>(
        values.data_ptr<float>(),
        rewards.data_ptr<float>(),
        dones.data_ptr<float>(),
        importance.data_ptr<float>(),
        advantages.data_ptr<float>(),
        values_bs.data_ptr<float>(),
        rewards_bs.data_ptr<float>(),
        dones_bs.data_ptr<float>(),
        truncs_bs.data_ptr<float>(),
        gamma,
        lambda,
        rho_clip,
        c_clip,
        num_steps,
        horizon
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

TORCH_LIBRARY_IMPL(pufferlib, CUDA, m) {
  m.impl("compute_puff_advantage", &compute_puff_advantage_cuda);
}

}
