/*
Build:
nvcc -O3 -arch=sm_86 -shared -Xcompiler -fPIC puff_advantage_standalone.cu -o libpuff_advantage.so
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

// =============================================================================
// ORIGINAL IMPLEMENTATION
// =============================================================================

__device__ void puff_advantage_row_original(float* values, float* rewards, float* dones,
        float* importance, float* advantages, float gamma, float lambda,
        float rho_clip, float c_clip, int horizon) {
    float lastpufferlam = 0;
    for (int t = horizon-2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0f - dones[t_next];
        float rho_t = fminf(importance[t], rho_clip);
        float c_t = fminf(importance[t], c_clip);
        float delta = rho_t*(rewards[t_next] + gamma*values[t_next]*nextnonterminal - values[t]);
        lastpufferlam = delta + gamma*lambda*c_t*lastpufferlam*nextnonterminal;
        advantages[t] = lastpufferlam;
    }
}

__global__ void puff_advantage_kernel_original(float* values, float* rewards,
        float* dones, float* importance, float* advantages, float gamma,
        float lambda, float rho_clip, float c_clip, int num_steps, int horizon) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_steps) {
        return;
    }
    int offset = row * horizon;
    puff_advantage_row_original(values + offset, rewards + offset, dones + offset,
        importance + offset, advantages + offset, gamma, lambda, rho_clip, c_clip, horizon);
}

// =============================================================================
// NEW IMPLEMENTATION
// =============================================================================

__device__ __forceinline__ void puff_advantage_row_new(float* values, float* rewards, float* dones,
        float* importance, float* advantages, float gamma, float lambda,
        float rho_clip, float c_clip, int horizon) {
    
    // Fall back to original if horizon not divisible by 4
    if (horizon % 4 != 0) {
        puff_advantage_row_original(values, rewards, dones,
            importance, advantages, gamma, lambda, rho_clip, c_clip, horizon);
        return;
    }
    
    float lastpufferlam = 0.0f;
    int num_chunks = horizon / 4;
    
    // need to track values across chunks
    float next_value = values[horizon - 1];
    float next_done = dones[horizon - 1];
    float next_reward = rewards[horizon - 1];
    
    // Process chunks from end to beginning
    for (int chunk = num_chunks - 1; chunk >= 0; chunk--) {
        int base = chunk * 4;
        
        // Load 4 elements at once
        float4 v4 = *reinterpret_cast<float4*>(values + base);
        float4 r4 = *reinterpret_cast<float4*>(rewards + base);
        float4 d4 = *reinterpret_cast<float4*>(dones + base);
        float4 i4 = *reinterpret_cast<float4*>(importance + base);
        
        float v[4] = {v4.x, v4.y, v4.z, v4.w};
        float r[4] = {r4.x, r4.y, r4.z, r4.w};
        float d[4] = {d4.x, d4.y, d4.z, d4.w};
        float imp[4] = {i4.x, i4.y, i4.z, i4.w};
        float adv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        
        int start_idx = (chunk == num_chunks - 1) ? 2 : 3;
        
        #pragma unroll
        for (int i = start_idx; i >= 0; i--) {
            float nextnonterminal = 1.0f - next_done;
            float rho_t = fminf(imp[i], rho_clip);
            float c_t = fminf(imp[i], c_clip);
            float delta = rho_t * (next_reward + gamma * next_value * nextnonterminal - v[i]);
            lastpufferlam = delta + gamma * lambda * c_t * lastpufferlam * nextnonterminal;
            adv[i] = lastpufferlam;
            
            next_value = v[i];
            next_done = d[i];
            next_reward = r[i];
        }
        
        float4 adv4 = make_float4(adv[0], adv[1], adv[2], adv[3]);
        *reinterpret_cast<float4*>(advantages + base) = adv4;
    }
}

__global__ void puff_advantage_kernel_new(float* values, float* rewards,
        float* dones, float* importance, float* advantages, float gamma,
        float lambda, float rho_clip, float c_clip, int num_steps, int horizon) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_steps) {
        return;
    }
    int offset = row * horizon;
    puff_advantage_row_new(values + offset, rewards + offset, dones + offset,
        importance + offset, advantages + offset, gamma, lambda, rho_clip, c_clip, horizon);
}

// =============================================================================
// C API - Exported functions callable from Python via ctypes
// =============================================================================

extern "C" {

void launch_original(float* values, float* rewards, float* dones,
        float* importance, float* advantages, float gamma, float lambda,
        float rho_clip, float c_clip, int num_steps, int horizon) {
    int threads_per_block = 256;
    int blocks = (num_steps + threads_per_block - 1) / threads_per_block;
    puff_advantage_kernel_original<<<blocks, threads_per_block>>>(
        values, rewards, dones, importance, advantages,
        gamma, lambda, rho_clip, c_clip, num_steps, horizon);
}

void launch_new(float* values, float* rewards, float* dones,
        float* importance, float* advantages, float gamma, float lambda,
        float rho_clip, float c_clip, int num_steps, int horizon) {
    int threads_per_block = 32;
    int blocks = (num_steps + threads_per_block - 1) / threads_per_block;
    puff_advantage_kernel_new<<<blocks, threads_per_block>>>(
        values, rewards, dones, importance, advantages,
        gamma, lambda, rho_clip, c_clip, num_steps, horizon);
}

void sync_device() {
    cudaDeviceSynchronize();
}

const char* get_last_error() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return cudaGetErrorString(err);
    }
    return NULL;
}

}  // extern "C"

