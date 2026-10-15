// CPU vs GPU parity of an ocean/mujoco env: the same envs (same seeds and
// actions) are stepped by mjc_step on the host and by the --cu backend kernels,
// and qpos/qvel/obs/reward are compared after every step. Build with
// -DENV_HEADER='"../ocean/mujoco/mjc_ENV.h"'. Usage: test_mujoco_gpu MODEL_BIN [N] [STEPS]
#include <stdio.h>
#include <time.h>
#define PUF_BACKEND PUF_GPU
typedef float obs_t;
#include ENV_HEADER

int main(int argc, char** argv) {
    int n = argc > 2 ? atoi(argv[2]) : 256;
    int steps = argc > 3 ? atoi(argv[3]) : 200;
    Dict kwargs = {0};
    dict_set_str(&kwargs, "model", argv[1]);
    dict_set(&kwargs, "max_steps", 50);
    dict_set(&kwargs, "reset_noise_scale", 0.1);
    dict_set(&kwargs, "forward_reward_weight", 1.0);
    dict_set(&kwargs, "ctrl_cost_weight", 0.1);
    dict_set(&kwargs, "contact_cost_weight", 5e-4);
    dict_set(&kwargs, "healthy_reward", 1.0);
    obs_t* d_obs;
    float* d_act;
    float* d_rew;
    float* d_term;
    cudaMalloc((void**)&d_obs, n*OBS_SIZE*sizeof(obs_t));
    cudaMalloc((void**)&d_act, n*NUM_ATNS*sizeof(float));
    cudaMalloc((void**)&d_rew, n*sizeof(float));
    cudaMalloc((void**)&d_term, n*sizeof(float));
    Env* d_envs = puf_vec_create(n, &kwargs, d_obs, d_act, d_rew, d_term);
    // host twins with the same seeds
    Env* envs = (Env*)calloc(n, sizeof(Env));
    float* obs = (float*)calloc(n*OBS_SIZE, sizeof(float));
    float* act = (float*)calloc(n*NUM_ATNS, sizeof(float));
    float* rew = (float*)calloc(n, sizeof(float));
    float* term = (float*)calloc(n, sizeof(float));
    for (int i = 0; i < n; i++) {
        mjc_init(&envs[i], &kwargs);
        mj_makeData(&envs[i].d, (float*)calloc(MJ_SCRATCH, sizeof(float)), 1);
        envs[i].rng = i + 1;
        envs[i].agents[0].observations = obs + i*OBS_SIZE;
        envs[i].agents[0].actions = act + i*NUM_ATNS;
        envs[i].agents[0].rewards = rew + i;
        envs[i].agents[0].terminals = term + i;
        mjc_reset(&envs[i]);
    }
    puf_reset(d_envs);
    cudaDeviceSynchronize();
    size_t free0, total;
    cudaMemGetInfo(&free0, &total);
    float* g_obs = (float*)calloc(n*OBS_SIZE, sizeof(float));
    float* g_rew = (float*)calloc(n, sizeof(float));
    float* g_term = (float*)calloc(n, sizeof(float));
    Env* g_env = (Env*)calloc(1, sizeof(Env));
    double max_o = 0, max_r = 0, max_q = 0, sum_r = 0;
    int term_mismatch = 0;
    double gpu_ms = 0;
    for (int t = 0; t < steps; t++) {
        for (int i = 0; i < n*NUM_ATNS; i++) {
            act[i] = 2.0f*((float)rand() / RAND_MAX) - 1.0f;
        }
        cudaMemcpy(d_act, act, n*NUM_ATNS*sizeof(float), cudaMemcpyHostToDevice);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventRecord(e0);
        puf_step(d_envs);
        cudaEventRecord(e1);
        cudaDeviceSynchronize();
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        gpu_ms += ms;
        for (int i = 0; i < n; i++) {
            mjc_step(&envs[i]);
        }
        cudaMemcpy(g_obs, d_obs, n*OBS_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(g_rew, d_rew, n*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(g_term, d_term, n*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(g_env, d_envs, sizeof(Env), cudaMemcpyDeviceToHost);
        for (int i = 0; i < n*OBS_SIZE; i++) {
            max_o = fmax(max_o, fabs(g_obs[i] - obs[i]));
        }
        for (int i = 0; i < n; i++) {
            max_r = fmax(max_r, fabs(g_rew[i] - rew[i]));
            sum_r += fabs(g_rew[i] - rew[i]);
            term_mismatch += (g_term[i] != 0) != (term[i] != 0);
        }
        double eq = 0, eo = 0;
        for (int i = 0; i < mj_model.nq; i++) {
            eq = fmax(eq, fabs(g_env->d.qpos[i] - envs[0].d.qpos[i]));
        }
        for (int i = 0; i < n*OBS_SIZE; i++) {
            eo = fmax(eo, fabs(g_obs[i] - obs[i]));
        }
        max_q = fmax(max_q, eq);
        int k = t + 1;
        if (k == 1 || k == 2 || k == 5 || k == 10 || k == 20 || k == 50 || k == steps) {
            printf("step %4d: max|dq|(env0) %.3e max|dobs|(all) %.3e\n", k, eq, eo);
        }
    }
    printf("cpu-vs-gpu over %d envs x %d steps: max|dobs| %.3e max|dr| %.3e mean|dr| %.3e "
        "max|dq|(env0) %.3e terminal mismatch %d\n", n, steps, max_o, max_r, sum_r / (n*steps),
        max_q, term_mismatch);
    printf("gpu step kernel: %.2f ms per batch of %d (%.0f env steps/s)\n", gpu_ms / steps, n,
        n*steps / (gpu_ms / 1000.0));
    size_t free1;
    cudaMemGetInfo(&free1, &total);
    printf("device memory taken by the step kernel launch (local memory): %.0f MB; envs %.0f MB\n",
        (free0 - free1) / 1048576.0, n*sizeof(Env) / 1048576.0);
    return 0;
}
