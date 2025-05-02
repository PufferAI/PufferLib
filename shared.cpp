#include <torch/extension.h>

// TODO: Find a better way to do conditional compilation
#ifndef __CUDA_ARCH__
#define __host__
#define __device__
#endif

// [horizon]
__host__ __device__ void gae_row(float* values, float* rewards, float* dones, float* advantages,
        float gamma, float gae_lambda, int horizon) {
    float lastgaelam = 0;
    for (int t = horizon-2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0 - dones[t_next];
        // Should this rewards[t_next] be rewards[t]?
        float delta = rewards[t_next] + gamma*values[t_next]*nextnonterminal - values[t];
        lastgaelam = delta + gamma*gae_lambda*nextnonterminal * lastgaelam;
        advantages[t] = lastgaelam;
    }
}

torch::Tensor gae_check(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, int num_steps, int horizon) {

    // Validate input tensors
    torch::Device device = values.device();
    for (const torch::Tensor& t : {values, rewards, dones}) {
        TORCH_CHECK(t.dim() == 2, "Tensor must be 2D");
        TORCH_CHECK(t.device() == device, "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
        TORCH_CHECK(t.size(1) == horizon, "Second dimension must match horizon");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "All tensors must be float32");
        if (!t.is_contiguous()) {
            t.contiguous();
        }
    }

    torch::Tensor advantages = torch::zeros(
        {num_steps, horizon}, 
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(device)
    );
    return advantages;
}

// [horizon]
const int max_horizon = 256;
__host__ __device__ void vtrace_row(float* values, float* rewards, float* dones,
        float* importance, float* vs, float* advantages, float gamma, float rho_clip, float c_clip, int horizon) {
    float accum = 0.0;//values[horizon-1]; // Is this correct?
    vs[horizon-1] = values[horizon-1];
    for (int t = horizon-2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0 - dones[t_next];
        float rho_t = fminf(importance[t], rho_clip);
        float c_t = fminf(importance[t], c_clip);
        float delta = rho_t*(rewards[t] + gamma*values[t_next]*nextnonterminal - values[t]);
        accum =  delta + gamma*c_t*accum*nextnonterminal;
        advantages[t] = rho_t*(rewards[t] + gamma*vs[t_next]*nextnonterminal - values[t]);
        vs[t] = accum + values[t];
    }
}

__host__ __device__ void puff_advantage_row(float* values, float* rewards, float* dones,
        float* importance, float* vs, float* advantages, float gamma, float lambda,
        float rho_clip, float c_clip, int horizon) {
    vs[horizon-1] = values[horizon-1];
    float lastpufferlam = 0;
    for (int t = horizon-2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0 - dones[t_next];
        float rho_t = fminf(importance[t], rho_clip);
        float c_t = fminf(importance[t], c_clip);
        // TODO: t_next works and t doesn't. Check original formula
        float delta = rho_t*(rewards[t_next] + gamma*values[t_next]*nextnonterminal - values[t]);
        lastpufferlam = delta + gamma*lambda*c_t*lastpufferlam*nextnonterminal;
        
        //float delta = rewards[t_next] + gamma*values[t_next]*nextnonterminal - values[t];
        //lastpufferlam = delta + gamma*lambda*lastpufferlam*nextnonterminal;


        advantages[t] = lastpufferlam;
        vs[t] = advantages[t] + values[t];
        //advantages[t] = rho_t*(rewards[t] + gamma*vs[t_next]*nextnonterminal - values[t]);
        //vs[t] = lastpufferlam + values[t];
    }
}

void vtrace_check(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor vs, torch::Tensor advantages,
        int num_steps, int horizon) {

    // Validate input tensors
    torch::Device device = values.device();
    for (const torch::Tensor& t : {values, rewards, dones, importance, vs, advantages}) {
        TORCH_CHECK(t.dim() == 2, "Tensor must be 2D");
        TORCH_CHECK(t.device() == device, "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
        TORCH_CHECK(t.size(1) == horizon, "Second dimension must match horizon");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "All tensors must be float32");
        assert(horizon <= max_horizon);
        if (!t.is_contiguous()) {
            t.contiguous();
        }
    }
}




