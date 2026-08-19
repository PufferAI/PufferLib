#ifndef PUFFER_ROBOT_ARM_GPU_CU
#define PUFFER_ROBOT_ARM_GPU_CU

#define PUF_BACKEND PUF_GPU

#ifndef RA_OBS_T_DEFINED
#define RA_OBS_T_DEFINED
#if defined(from_float) && !defined(PRECISION_FLOAT)
typedef precision_t obs_t;
#else
typedef float obs_t;
#endif
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    DictItem* item = dict_find(kwargs, key);
    return item != NULL && item->value != 0.0;
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "perf", log->score);
    if (log->basketball_mode > 0.5f) {
        dict_set(out, "baskets", log->baskets);
        dict_set(out, "grasp_rate", log->grasp_rate);
        dict_set(out, "lift_rate", log->lift_rate);
        dict_set(out, "release_rate", log->release_rate);
        dict_set(out, "slip_rate", log->slip_rate);
        float release_count = log->release_center_miss_count;
        dict_set(out, "avg_release_miss_cm", release_count > 0.0f
            ? log->release_center_miss_cm_sum / release_count : 0.0f);
        dict_set(out, "episode_length", log->episode_length);
        return;
    }
    dict_set(out, "success_rate", log->success_rate);
    dict_set(out, "grasp_rate", log->grasp_rate);
    dict_set(out, "lift_rate", log->lift_rate);
    dict_set(out, "transport_rate", log->transport_rate);
    dict_set(out, "release_rate", log->release_rate);
    dict_set(out, "episode_return", log->return_value);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "reach_distance", log->reach_distance);
    dict_set(out, "place_distance", log->place_distance);
    dict_set(out, "energy", log->energy);
    dict_set(out, "pinch_force", log->pinch_force);
    dict_set(out, "slip_rate", log->slip_rate);
    dict_set(out, "stack_rate", log->stack_rate);
    dict_set(out, "stable_stack_rate", log->stable_stack_rate);
    dict_set(out, "stack_alignment_rate", log->stack_alignment_rate);
    dict_set(out, "valid_stack_contact_rate",
        log->valid_stack_contact_rate);
    dict_set(out, "clearance_rate", log->clearance_rate);
    dict_set(out, "settle_rate", log->settle_rate);
    dict_set(out, "stack_alignment", log->stack_alignment);
    dict_set(out, "base_slide_distance", log->base_slide_distance);
    dict_set(out, "cube_angular_speed", log->cube_angular_speed);
    dict_set(out, "base_angular_speed", log->base_angular_speed);
    dict_set(out, "orientation_error", log->orientation_error);
    dict_set(out, "n", log->n);
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
    ra_rbrst(&env->world.rigid, env->world.state.basketball_mode ? 3u
        : (env->world.state.stack_mode ? 2u : 1u));
}

Env* puf_vec_create(int n, Dict* env_kwargs,
        obs_t* observations, float* actions, float* rewards, float* terminals) {
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
    g_ra_render_host.renderer = (RaRenderer*)calloc(1, sizeof(RaRenderer));
    assert(g_ra_render_host.renderer != NULL);

    Env* host_envs = (Env*)calloc(n, sizeof(Env));
    for (int i = 0; i < n; i++) {
        ra_fill(&host_envs[i], i + 1);
    }
    Env* envs = NULL;
    assert(cudaMalloc((void**)&envs, n * sizeof(Env)) == cudaSuccess);
    assert(cudaMemcpy(envs, host_envs, n * sizeof(Env),
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

void puf_init(Env*, Dict*) {
}

void puf_reset(Env*) {
    ra_kinit<<<(g_gpu.n + RA_CUDA_BLOCK_SIZE - 1) / RA_CUDA_BLOCK_SIZE,
        RA_CUDA_BLOCK_SIZE>>>(
        g_gpu.envs, g_gpu.observations, g_gpu.rewards, g_gpu.terminals,
        g_gpu.n);
    assert(cudaGetLastError() == cudaSuccess);
}

void puf_step(Env*) {
    dim3 grid((g_gpu.n + RA_CUDA_BLOCK_SIZE - 1) / RA_CUDA_BLOCK_SIZE);
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

void puf_close(Env*) {
    ra_rclose(&g_ra_render_host);
    cudaFree(g_gpu.envs);
    g_gpu.envs = NULL;
}

void puf_render(Env*) {
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
