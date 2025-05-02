#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "c_advantage.cu"

// Kernel declaration
__global__ void advantage_kernel(
    float* reward_block, float* reward_mask, float* values_mean,
    float* values_std, float* buf, float* dones, float* rewards,
    float* advantages, int* bounds, int num_steps, float r_std, int horizon
);

#define NUM_STEPS 6
#define HORIZON 4

float test_values_mean[NUM_STEPS * HORIZON] = {
    1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f,
};

float g = sqrt(0.5f);

float test_values_std[NUM_STEPS * HORIZON] = {
    g, g, g, g,
    g, g, g, g,
    g, g, g, g,
    g, g, g, g,
    g, g, g, g,
};

float test_dones[NUM_STEPS] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
float test_rewards[NUM_STEPS] = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};

int main() {
    // Test parameters
    //const int num_steps = 4;
    //const int horizon = 3;
    const float r_std = 2.0f;
    
    // Calculate sizes
    const int block_size = NUM_STEPS * HORIZON * sizeof(float);
    const int steps_size = NUM_STEPS * sizeof(float);
    const int bounds_size = NUM_STEPS * sizeof(int);

    // Host buffers
    float* h_reward_block = (float*)malloc(block_size);
    float* h_reward_mask = (float*)malloc(block_size);
    float* h_values_mean = (float*)malloc(block_size);
    float* h_values_std = (float*)malloc(block_size);
    float* h_buf = (float*)malloc(block_size);
    float* h_dones = (float*)malloc(steps_size);
    float* h_rewards = (float*)malloc(steps_size);
    float* h_advantages = (float*)malloc(steps_size);
    int* h_bounds = (int*)malloc(bounds_size);

    // Device buffers
    float *d_reward_block, *d_reward_mask, *d_values_mean, *d_values_std;
    float *d_buf, *d_dones, *d_rewards, *d_advantages;
    int* d_bounds;

    // Allocate device memory
    cudaMalloc((void**)&d_reward_block, block_size);
    cudaMalloc((void**)&d_reward_mask, block_size);
    cudaMalloc((void**)&d_values_mean, block_size);
    cudaMalloc((void**)&d_values_std, block_size);
    cudaMalloc((void**)&d_buf, block_size);
    cudaMalloc((void**)&d_dones, steps_size);
    cudaMalloc((void**)&d_rewards, steps_size);
    cudaMalloc((void**)&d_advantages, steps_size);
    cudaMalloc((void**)&d_bounds, bounds_size);

    // Initialize test data
    // Copy input data to device
    cudaMemcpy(d_values_mean, test_values_mean, block_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_std, test_values_std, block_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dones, test_dones, steps_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rewards, test_rewards, steps_size, cudaMemcpyHostToDevice);

    // Launch configuration
    int threadsPerBlock = 256;
    int blocks = (NUM_STEPS + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    advantage_kernel<<<blocks, threadsPerBlock>>>(
        d_reward_block, d_reward_mask, d_values_mean, d_values_std,
        d_buf, d_dones, d_rewards, d_advantages, d_bounds,
        NUM_STEPS, r_std, HORIZON
    );

    cudaGetLastError();
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_reward_block, d_reward_block, block_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_reward_mask, d_reward_mask, block_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_advantages, d_advantages, steps_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bounds, d_bounds, bounds_size, cudaMemcpyDeviceToHost);

    // Print results
    printf("Advantages:\n");
    for (int i = 0; i < NUM_STEPS; i++) {
        printf("%.2f ", h_advantages[i]);
    }
    printf("\nBounds:\n");
    for (int i = 0; i < NUM_STEPS; i++) {
        printf("%d ", h_bounds[i]);
    }
    printf("\nReward Block:\n");
    for (int i = 0; i < NUM_STEPS; i++) {
        for (int j = 0; j < HORIZON; j++) {
            printf("%.2f ", h_reward_block[i * HORIZON + j]);
        }
        printf("\n");
    }

    // Cleanup
    cudaFree(d_reward_block); cudaFree(d_reward_mask);
    cudaFree(d_values_mean); cudaFree(d_values_std);
    cudaFree(d_buf); cudaFree(d_dones);
    cudaFree(d_rewards); cudaFree(d_advantages);
    cudaFree(d_bounds);

    free(h_reward_block); free(h_reward_mask);
    free(h_values_mean); free(h_values_std);
    free(h_buf); free(h_dones);
    free(h_rewards); free(h_advantages);
    free(h_bounds);

    return 0;
}
