#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "pufferl.cu"

#ifndef PUFFER_BREAKOUT_CUDA
#error "test_breakout_cuda.cu must be built with the Breakout environment"
#endif

// fbr: Exercise CPU and CUDA Breakout from identical states and actions so the
// device adapter can be reviewed independently from throughput measurements.
static void configure_breakout(Breakout* env, int seed) {
    *env = {};
    env->num_agents = 1;
    env->frameskip = 4;
    env->width = 576;
    env->height = 330;
    env->initial_paddle_width = 62;
    env->paddle_height = 8;
    env->ball_width = 32;
    env->ball_height = 32;
    env->brick_width = 32;
    env->brick_height = 12;
    env->brick_rows = 6;
    env->brick_cols = 18;
    env->initial_ball_speed = 256;
    env->max_ball_speed = 448;
    env->paddle_speed = 620;
    env->continuous = 0;
    env->rng = seed;
    allocate(env);
    puf_reset(env);
}

static bool close_enough(float actual, float expected) {
    float atol = USE_BF16 ? 4.0e-3f : 2.0e-4f;
    float rtol = USE_BF16 ? 4.0e-3f : 2.0e-5f;
    return fabsf(actual - expected) <= atol + rtol * fabsf(expected);
}

int main() {
    constexpr int B = 65;
    constexpr int STEPS = 4096;
    std::vector<Breakout> cpu(B);
    std::vector<BreakoutCudaState> initial(B);
    for (int i = 0; i < B; ++i) {
        configure_breakout(&cpu[i], i + 1);
        initial[i] = bc_from_host(cpu[i]);
    }

    BreakoutCudaState* states;
    precision_t *actions, *observations, *rewards, *terminals;
    cudaMalloc(&states, B * sizeof(*states));
    cudaMalloc(&actions, B * sizeof(*actions));
    cudaMalloc(&observations, (size_t)B * OBS_SIZE * sizeof(*observations));
    cudaMalloc(&rewards, B * sizeof(*rewards));
    cudaMalloc(&terminals, B * sizeof(*terminals));
    cudaMemcpy(states, initial.data(), B * sizeof(*states), cudaMemcpyHostToDevice);

    size_t shared_bytes = (size_t)bc_block_size * sizeof(BreakoutCudaState);
    cudaFuncSetAttribute(bc_step<precision_t>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shared_bytes);
    std::vector<precision_t> host_actions(B);
    std::vector<precision_t> gpu_obs((size_t)B * OBS_SIZE);
    std::vector<precision_t> gpu_rewards(B), gpu_terminals(B);
    int episodes = 0;

    for (int step = 0; step < STEPS; ++step) {
        for (int i = 0; i < B; ++i) {
            float action = (float)((step / 7 + i) % 3);
            host_actions[i] = from_float(action);
            cpu[i].agents[0].actions[0] = action;
            puf_step(&cpu[i]);
        }
        cudaMemcpy(actions, host_actions.data(), B * sizeof(*actions), cudaMemcpyHostToDevice);
        bc_step<precision_t><<<(B + bc_block_size - 1) / bc_block_size,
            bc_block_size, shared_bytes>>>(
            states, B, actions, observations, rewards, terminals);
        bc_observe_warp<precision_t><<<(B + BC_OBSERVE_WARPS_PER_BLOCK - 1)
                / BC_OBSERVE_WARPS_PER_BLOCK,
            BC_OBSERVE_BLOCK_SIZE>>>(states, B, observations);
        cudaError_t error = cudaGetLastError();
        if (error == cudaSuccess) error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            fprintf(stderr, "CUDA Breakout step failed: %s\n", cudaGetErrorString(error));
            return 1;
        }
        cudaMemcpy(gpu_obs.data(), observations,
            gpu_obs.size() * sizeof(precision_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(gpu_rewards.data(), rewards,
            B * sizeof(precision_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(gpu_terminals.data(), terminals,
            B * sizeof(precision_t), cudaMemcpyDeviceToHost);

        for (int i = 0; i < B; ++i) {
            float expected_reward = cpu[i].agents[0].rewards[0];
            float expected_terminal = cpu[i].agents[0].terminals[0];
            float actual_reward = to_float(gpu_rewards[i]);
            float actual_terminal = to_float(gpu_terminals[i]);
            if (actual_reward != expected_reward || actual_terminal != expected_terminal) {
                fprintf(stderr,
                    "step %d env %d reward/terminal mismatch: gpu=(%g,%g) cpu=(%g,%g)\n",
                    step, i, actual_reward, actual_terminal,
                    expected_reward, expected_terminal);
                return 1;
            }
            episodes += expected_terminal != 0.0f;
            for (int j = 0; j < OBS_SIZE; ++j) {
                float expected = ((obs_t*)cpu[i].agents[0].observations)[j];
                float actual = to_float(gpu_obs[(size_t)i * OBS_SIZE + j]);
                if (!close_enough(actual, expected)) {
                    fprintf(stderr,
                        "step %d env %d obs %d mismatch: gpu=%g cpu=%g error=%g\n",
                        step, i, j, actual, expected, fabsf(actual - expected));
                    return 1;
                }
            }
        }
    }

    std::vector<BreakoutCudaState> final_states(B);
    cudaMemcpy(final_states.data(), states, B * sizeof(*states), cudaMemcpyDeviceToHost);
    for (int i = 0; i < B; ++i) {
        const Log& gpu = final_states[i].log;
        const Log& host = cpu[i].log;
        if (gpu.n != host.n || gpu.score != host.score
                || gpu.episode_return != host.episode_return
                || gpu.episode_length != host.episode_length
                || !close_enough(gpu.perf, host.perf)) {
            fprintf(stderr, "env %d log mismatch after %d steps\n", i, STEPS);
            return 1;
        }
    }

    for (Breakout& env : cpu) free_allocated(&env);
    cudaFree(states); cudaFree(actions); cudaFree(observations);
    cudaFree(rewards); cudaFree(terminals);
    printf("Breakout CUDA %s direct-write parity PASS: %d envs x %d steps, "
        "%d completed episodes, %zu-byte state\n",
        USE_BF16 ? "BF16" : "FP32", B, STEPS, episodes,
        sizeof(BreakoutCudaState));
    return 0;
}
