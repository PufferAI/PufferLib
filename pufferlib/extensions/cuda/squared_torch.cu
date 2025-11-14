#include <torch/extension.h>
#include <torch/torch.h>
#include <cuda_runtime.h>

#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

namespace pufferlib {

static constexpr unsigned char NOOP   = 0;
static constexpr unsigned char DOWN   = 1;
static constexpr unsigned char UP     = 2;
static constexpr unsigned char LEFT   = 3;
static constexpr unsigned char RIGHT  = 4;

static constexpr unsigned char EMPTY  = 0;
static constexpr unsigned char AGENT  = 1;
static constexpr unsigned char TARGET = 2;

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

struct __align__(16) Squared {
    curandState rng;
    unsigned char* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int size;
    int tick;
    int r, c;
    Log log;
    int padding[3];
};

__device__ void add_log(Squared* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}


// Device: Reset environment
__device__ void cuda_reset(Squared* env, curandState* rng) {
    int tiles = env->size * env->size;
    for (int i = 0; i < tiles; i++) {
        env->observations[i] = EMPTY;
    }
    env->observations[tiles/2] = AGENT;
    env->r = env->size/2;
    env->c = env->size/2;
    env->tick = 0;
    int target_idx = 0; // Deterministic for testing
    do {
        target_idx = curand(rng) % tiles;
    } while (target_idx == tiles/2);
    env->observations[target_idx] = TARGET;
}

// Device: Step environment
__device__ void cuda_step(Squared* env) {
    env->tick += 1;
    int action = env->actions[0];
    env->terminals[0] = 0;
    env->rewards[0] = 0.0f;

    int pos = env->r * env->size + env->c;
    env->observations[pos] = EMPTY;

    if (action == DOWN) {
        env->r += 1;
    } else if (action == UP) {
        env->r -= 1;
    } else if (action == RIGHT) {
        env->c += 1;
    } else if (action == LEFT) {
        env->c -= 1;
    }


    if (env->tick > 3 * env->size
            || env->r < 0
            || env->c < 0
            || env->r >= env->size
            || env->c >= env->size) {
        env->terminals[0] = 1;
        env->rewards[0] = -1.0f;
        add_log(env);
        cuda_reset(env, &env->rng);
        return;
    }

    pos = env->r*env->size + env->c;
    if (env->observations[pos] == TARGET) {
        env->terminals[0] = 1;
        env->rewards[0] = 1.0f;
        add_log(env);
        cuda_reset(env, &env->rng);
        return;
    }

    env->observations[pos] = AGENT;
}

// Kernel: Step all environments
__global__ void step_environments(Squared* envs, int* indices, int num_envs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;
    cuda_step(&envs[idx]);
}

// Kernel: Reset specific environments
__global__ void reset_environments(Squared* envs, int* indices, int num_reset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_reset) return;
    int env_idx = indices[idx];
    cuda_reset(&envs[env_idx], &envs[env_idx].rng);
}

// Kernel: Reset specific environment logs
__global__ void reset_logs(Squared* envs, int* indices, int num_reset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_reset) return;
    int env_idx = indices[idx];
    envs[env_idx].log = {0};
}

// Kernel: Initialize all environments
__global__ void init_environments(Squared* envs,
                                 unsigned char* obs_mem,
                                 int* actions_mem,
                                 float* rewards_mem,
                                 unsigned char* terminals_mem,
                                 int num_envs,
                                 int grid_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;

    Squared* env = &envs[idx];

    // Initialize log
    env->log.perf = 0;
    env->log.score = 0;
    env->log.episode_return = 0;
    env->log.episode_length = 0;
    env->log.n = 0;

    // Set pointers into memory pools
    env->observations = obs_mem + idx * grid_size * grid_size;
    env->actions = actions_mem + idx;
    env->rewards = rewards_mem + idx;
    env->terminals = terminals_mem + idx;
    env->size = grid_size;

    // Initialize RNG
    curand_init(clock64(), idx, 0, &env->rng);

    // Initial reset
    cuda_reset(env, &env->rng);
}


inline dim3 make_grid(int n) {
    return dim3((n + 255) / 256);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
create_squared_environments(int64_t num_envs, int64_t grid_size, torch::Tensor dummy) {
    auto device = dummy.device();
    TORCH_CHECK(device.type() == at::kCUDA, "Dummy tensor must be on CUDA device");

    auto envs_tensor = torch::empty({static_cast<int64_t>(num_envs * sizeof(Squared))}, torch::kUInt8).to(device);
    auto obs = torch::zeros({num_envs, grid_size, grid_size}, torch::kUInt8).to(device);
    auto actions = torch::zeros({num_envs}, torch::kInt32).to(device);
    auto rewards = torch::zeros({num_envs}, torch::kFloat32).to(device);
    auto terminals = torch::zeros({num_envs}, torch::kUInt8).to(device);

    Squared* envs = reinterpret_cast<Squared*>(envs_tensor.data_ptr<unsigned char>());

    init_environments<<<make_grid(num_envs), 256>>>(
        envs,
        obs.data_ptr<unsigned char>(),
        actions.data_ptr<int>(),
        rewards.data_ptr<float>(),
        terminals.data_ptr<unsigned char>(),
        num_envs,
        grid_size
    );
    cudaDeviceSynchronize();

    return std::make_tuple(envs_tensor, obs, actions, rewards, terminals);
}

void step_environments_cuda(torch::Tensor envs_tensor, torch::Tensor indices_tensor) {
    Squared* envs = reinterpret_cast<Squared*>(envs_tensor.data_ptr<unsigned char>());
    auto indices = indices_tensor.data_ptr<int>();
    int num_envs = indices_tensor.size(0);
    step_environments<<<make_grid(num_envs), 256>>>(envs, indices, num_envs);
    cudaDeviceSynchronize();
}

void reset_environments_cuda(torch::Tensor envs_tensor, torch::Tensor indices_tensor) {
    Squared* envs = reinterpret_cast<Squared*>(envs_tensor.data_ptr<unsigned char>());
    auto indices = indices_tensor.data_ptr<int>();
    int num_reset = indices_tensor.size(0);

    reset_environments<<<make_grid(num_reset), 256>>>(envs, indices, num_reset);
    cudaDeviceSynchronize();
}

Log log_environments_cuda(torch::Tensor envs_tensor, torch::Tensor indices_tensor) {
    Squared* envs = reinterpret_cast<Squared*>(envs_tensor.cpu().data_ptr<unsigned char>());
    torch::Tensor cpu_indices = indices_tensor.cpu();
    auto indices = cpu_indices.data_ptr<int>();
    int num_log = indices_tensor.size(0);
    Log log = {0};
    for (int i=0; i<num_log; i++) {
        log.perf += envs[indices[i]].log.perf;
        log.score += envs[indices[i]].log.score;
        log.episode_return += envs[indices[i]].log.episode_return;
        log.episode_length += envs[indices[i]].log.episode_length;
        log.n += envs[indices[i]].log.n;
    }
    log.perf /= log.n;
    log.score /= log.n;
    log.episode_return /= log.n;
    log.episode_length /= log.n;

    // Reset must be done in cuda
    Squared* envs_gpu = reinterpret_cast<Squared*>(envs_tensor.data_ptr<unsigned char>());
    auto indices_gpu = indices_tensor.data_ptr<int>();
    reset_logs<<<make_grid(num_log), 256>>>(envs_gpu, indices_gpu, num_log);
    cudaDeviceSynchronize();

    return log;
}

}
