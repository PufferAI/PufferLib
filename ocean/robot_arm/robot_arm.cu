#pragma once
#ifndef PUFFER_GPU_ENV
#error "robot_arm.cu requires -DPUFFER_GPU_ENV (build with --gpu)"
#endif

#include "robot_arm_cuda.cuh"

static int g_ra_no_timeout;
static int g_ra_stack;
static int g_ra_basketball;
static const char* g_ra_model_glb = "resources/robot_arm/franka_panda.glb";
static RaRenderHost g_ra_render_host;

static int ra_flag(Dict* kwargs, const char* key) {
    DictItem* item = dict_find(kwargs, key);
    return item != NULL && item->value != 0.0;
}

static void ra_fill(Env* env, unsigned int rng) {
    memset(env, 0, sizeof(*env));
    env->world.state.rng = rng ? rng : 1u;
    env->world.state.no_timeout = g_ra_no_timeout;
    env->world.state.stack_mode = g_ra_stack;
    env->world.state.basketball_mode = g_ra_basketball;
    ra_reset(&env->world.state);
    ra_rbrst(&env->world.rigid,
        ra_topo(&env->world.state));
}

static Env* puf_envs_create(int total_agents, Dict* env_kwargs) {
    g_ra_no_timeout = ra_flag(env_kwargs, "no_timeout");
    g_ra_stack = ra_flag(env_kwargs, "stack");
    g_ra_basketball = ra_flag(env_kwargs, "basketball");
    assert(!(g_ra_stack && g_ra_basketball));
    DictItem* model = dict_find(env_kwargs, "model_glb");
    if (model != NULL && model->str != NULL && model->str[0] != '\0'
            && strcmp(model->str, "None") != 0) {
        g_ra_model_glb = model->str;
    }
    g_ra_render_host.model_glb = g_ra_model_glb;
    g_ra_render_host.camera_distance = g_ra_basketball ? 2.35f : 1.55f;
    g_ra_render_host.camera_yaw = 0.78f;
    g_ra_render_host.camera_pitch = 0.48f;

    Env* host_envs = (Env*)calloc((size_t)total_agents, sizeof(Env));
    for (int i = 0; i < total_agents; i++) {
        ra_fill(&host_envs[i], (unsigned int)(i + 1));
    }
    Env* envs = (Env*)xcuda((size_t)total_agents * sizeof(Env));
    assert(cudaMemcpy(envs, host_envs, (size_t)total_agents * sizeof(Env),
        cudaMemcpyHostToDevice) == cudaSuccess);
    free(host_envs);
    return envs;
}

static void puf_envs_reset(Env* envs, obs_t* observations, float* rewards,
        float* terminals, int total_agents) {
    ra_kinit<<<
        (total_agents + RA_CUDA_BLOCK_SIZE - 1) / RA_CUDA_BLOCK_SIZE,
        RA_CUDA_BLOCK_SIZE>>>(
        envs, observations, rewards, terminals, total_agents);
    assert(cudaGetLastError() == cudaSuccess);
}

static void puf_envs_step(Env* envs, const float* actions,
        obs_t* observations, float* rewards, float* terminals, int start,
        int count, cudaStream_t stream) {
    dim3 grid((count + RA_CUDA_BLOCK_SIZE - 1) / RA_CUDA_BLOCK_SIZE);
    dim3 block(RA_CUDA_BLOCK_SIZE);
    ra_kbegin<<<grid, block, 0, stream>>>(
        envs, start, count, actions);
    assert(cudaGetLastError() == cudaSuccess);
    ra_kphys<<<grid, block, 0, stream>>>(
        envs, start, count);
    assert(cudaGetLastError() == cudaSuccess);
    ra_kfin<<<grid, block, 0, stream>>>(
        envs, start, count, observations, rewards, terminals);
    assert(cudaGetLastError() == cudaSuccess);
}

static void puf_envs_close(Env* envs) {
    ra_rclose(&g_ra_render_host);
    cudaFree(envs);
}

void puf_render(Env* env) {
    RaState state;
    assert(cudaMemcpy(&state, &env->world.state, sizeof(RaState),
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
    assert(cudaMemcpy(env, &host_env, sizeof(Env),
        cudaMemcpyHostToDevice) == cudaSuccess);
}
