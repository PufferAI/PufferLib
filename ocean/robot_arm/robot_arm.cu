#ifndef PUFFER_ROBOT_ARM_GPU_CU
#define PUFFER_ROBOT_ARM_GPU_CU

#define PUF_BACKEND PUF_GPU

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef __nv_bfloat16 obs_t;

#include "robot_arm_cuda.cuh"

static int g_ra_no_timeout;
static int g_ra_stack;
static int g_ra_basketball;
static const char* g_ra_model_glb = "resources/robot_arm/franka_panda.glb";
static RaRenderHost g_ra_render_host;

static struct {
    Env* envs;
    int n;
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    cudaStream_t stream;
} g_gpu;

static int ra_flag(Dict* kwargs, const char* key) {
    DictItem* item = kwargs != NULL ? dict_find(kwargs, key) : NULL;
    return item != NULL && item->value != 0.0;
}

static int ra_kgrid(int n) {
    return (n + RA_CUDA_BLOCK_SIZE - 1) / RA_CUDA_BLOCK_SIZE;
}

static void ra_fill(Env* env, unsigned int rng) {
    memset(env, 0, sizeof(*env));
    env->num_agents = 1;
    env->rng = rng ? rng : 1u;
    env->world.state.rng = env->rng;
    env->world.state.no_timeout = g_ra_no_timeout;
    env->world.state.stack_mode = g_ra_stack;
    env->world.state.basketball_mode = g_ra_basketball;
    ra_reset(&env->world.state);
    ra_rbrst(&env->world.rigid, ra_topo(&env->world.state));
}

Env* puf_vec_create(int n, Dict* env_kwargs,
        obs_t* observations, float* actions, float* rewards, float* terminals) {
    g_ra_no_timeout = ra_flag(env_kwargs, "no_timeout");
    g_ra_stack = ra_flag(env_kwargs, "stack");
    g_ra_basketball = ra_flag(env_kwargs, "basketball");
    assert(!(g_ra_stack && g_ra_basketball));
    DictItem* model = env_kwargs != NULL
        ? dict_find(env_kwargs, "model_glb") : NULL;
    if (model != NULL && model->str != NULL && model->str[0] != '\0'
            && strcmp(model->str, "None") != 0) {
        g_ra_model_glb = model->str;
    }
    g_ra_render_host.model_glb = g_ra_model_glb;
    g_ra_render_host.camera_distance = g_ra_basketball ? 2.35f : 1.55f;
    g_ra_render_host.camera_yaw = 0.78f;
    g_ra_render_host.camera_pitch = 0.48f;

    Env* host_envs = (Env*)calloc((size_t)n, sizeof(Env));
    for (int i = 0; i < n; i++) {
        ra_fill(&host_envs[i], (unsigned int)(i + 1));
    }
    Env* envs = NULL;
    assert(cudaMalloc((void**)&envs, (size_t)n * sizeof(Env)) == cudaSuccess);
    assert(cudaMemcpy(envs, host_envs, (size_t)n * sizeof(Env),
        cudaMemcpyHostToDevice) == cudaSuccess);
    free(host_envs);
    g_gpu.envs = envs;
    g_gpu.n = n;
    g_gpu.observations = observations;
    g_gpu.actions = actions;
    g_gpu.rewards = rewards;
    g_gpu.terminals = terminals;
    g_gpu.stream = 0;
    return envs;
}

void puf_bind_stream(cudaStream_t stream) {
    g_gpu.stream = stream;
}

void puf_init(Env* env, Dict* kwargs) {
    (void)env;
    (void)kwargs;
}

void puf_reset(Env* env) {
    (void)env;
    ra_kinit<<<ra_kgrid(g_gpu.n), RA_CUDA_BLOCK_SIZE>>>(
        g_gpu.envs, g_gpu.observations, g_gpu.rewards, g_gpu.terminals,
        g_gpu.n);
    assert(cudaGetLastError() == cudaSuccess);
}

void puf_step(Env* env) {
    (void)env;
    dim3 grid(ra_kgrid(g_gpu.n));
    dim3 block(RA_CUDA_BLOCK_SIZE);
    ra_kbegin<<<grid, block, 0, g_gpu.stream>>>(
        g_gpu.envs, 0, g_gpu.n, g_gpu.actions);
    assert(cudaGetLastError() == cudaSuccess);
    ra_kphys<<<grid, block, 0, g_gpu.stream>>>(
        g_gpu.envs, 0, g_gpu.n);
    assert(cudaGetLastError() == cudaSuccess);
    ra_kfin<<<grid, block, 0, g_gpu.stream>>>(
        g_gpu.envs, 0, g_gpu.n, g_gpu.observations, g_gpu.rewards,
        g_gpu.terminals);
    assert(cudaGetLastError() == cudaSuccess);
}

void puf_close(Env* env) {
    (void)env;
    ra_rclose(&g_ra_render_host);
    cudaFree(g_gpu.envs);
    g_gpu.envs = NULL;
}

void puf_render(Env* env) {
    (void)env;
    if (!g_gpu.envs || g_gpu.n < 1) {
        return;
    }
    if (g_gpu.stream) {
        cudaStreamSynchronize(g_gpu.stream);
    }
    RaState state;
    assert(cudaMemcpy(&state, &g_gpu.envs->world.state, sizeof(RaState),
        cudaMemcpyDeviceToHost) == cudaSuccess);
    RaPose links[RA_LINKS];
    ra_fk(state.q, state.gripper_width, links, NULL, NULL, &state.end_effector);
    ra_draw(&g_ra_render_host, &state, links);
    if (!g_ra_render_host.reset_requested) {
        return;
    }
    g_ra_render_host.reset_requested = 0;
    Env host_env;
    ra_fill(&host_env, state.rng ? state.rng : 1u);
    assert(cudaMemcpy(g_gpu.envs, &host_env, sizeof(Env),
        cudaMemcpyHostToDevice) == cudaSuccess);
}

#endif
