// Vector backends for the ocean/mujoco envs, included at the end of each
// mjc_<env>.h after Env, mjc_reset, mjc_step, mjc_render and mjc_init. All
// memory is bound at create time: the solver scratch (MJ_SCRATCH floats per
// env, see mj_makeData) and, on the GPU, the whole batch. CPU: the puf_* per-env
// API calls straight through. GPU (mjc_<env>.cu defines PUF_BACKEND PUF_GPU):
// one thread per env steps a thread-local Env (local memory is lane interleaved,
// so every access coalesces) and copies only the persistent prefix, everything
// before MjData.xquat, in and out of the device batch; the scratch is laid out
// lane interleaved too (element stride 32). The trainer's device obs/action/
// reward/terminal buffers are bound into Env.agents at create time.

#if PUF_BACKEND == PUF_GPU
#define MJC_BLOCK 128
#define MJC_STATE (offsetof(Env, d) + offsetof(MjData, xquat))

struct {
    Env* envs;
    int n;
    cudaStream_t stream;
} mjc_gpu;

__global__ void mjc_reset_kernel(Env* envs, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        Env env;
        memcpy(&env, &envs[i], MJC_STATE);
        mjc_reset(&env);
        memcpy(&envs[i], &env, MJC_STATE);
    }
}

__global__ void mjc_step_kernel(Env* envs, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        Env env;
        memcpy(&env, &envs[i], MJC_STATE);
        mjc_step(&env);
        memcpy(&envs[i], &env, MJC_STATE);
    }
}

Env* puf_vec_create(int n, Dict* kwargs, obs_t* observations, float* actions, float* rewards,
    float* terminals) {
    Env* host = (Env*)calloc(n, sizeof(Env));
    MjModel* m;
    assert(cudaMalloc((void**)&m, sizeof(MjModel)) == cudaSuccess);
    float* scratch;
    size_t groups = (n + 31) / 32;
    assert(cudaMalloc((void**)&scratch, groups*32*MJ_SCRATCH*sizeof(float)) == cudaSuccess
        && "GPU env solver scratch does not fit in device memory");
    for (int i = 0; i < n; i++) {
        Env* env = &host[i];
        mjc_init(env, kwargs);
        mj_makeData(&env->d, scratch + (size_t)(i / 32)*32*MJ_SCRATCH + i % 32, 32);
        env->m = m;
        env->rng = i + 1;
        env->agents[0].observations = observations + (long)i*OBS_SIZE;
        env->agents[0].actions = actions + (long)i*NUM_ATNS;
        env->agents[0].rewards = rewards + i;
        env->agents[0].terminals = terminals + i;
    }
    cudaMemcpy(m, &mj_model, sizeof(MjModel), cudaMemcpyHostToDevice);
    assert(cudaMalloc((void**)&mjc_gpu.envs, (size_t)n*sizeof(Env)) == cudaSuccess
        && "GPU env batch does not fit in device memory");
    cudaMemcpy(mjc_gpu.envs, host, (size_t)n*sizeof(Env), cudaMemcpyHostToDevice);
    free(host);
    mjc_gpu.n = n;
    return mjc_gpu.envs;
}

void puf_bind_stream(cudaStream_t stream) {
    mjc_gpu.stream = stream;
}

void puf_init(Env* env, Dict* kwargs) {
}

void puf_reset(Env* envs) {
    mjc_reset_kernel<<<(mjc_gpu.n + MJC_BLOCK - 1) / MJC_BLOCK, MJC_BLOCK>>>(mjc_gpu.envs,
        mjc_gpu.n);
    assert(cudaGetLastError() == cudaSuccess);
}

void puf_step(Env* envs) {
    mjc_step_kernel<<<(mjc_gpu.n + MJC_BLOCK - 1) / MJC_BLOCK, MJC_BLOCK, 0, mjc_gpu.stream>>>(
        mjc_gpu.envs, mjc_gpu.n);
    assert(cudaGetLastError() == cudaSuccess);
}

// Copy env 0's state back and draw it with the host model (mj_render recomputes
// kinematics and contacts from qpos)
void puf_render(Env* envs) {
    Env env;
    cudaStreamSynchronize(mjc_gpu.stream);
    cudaMemcpy(&env, mjc_gpu.envs, sizeof(Env), cudaMemcpyDeviceToHost);
    env.m = &mj_model;
    mjc_render(&env);
}

void puf_close(Env* envs) {
    cudaFree(mjc_gpu.envs);
    if (IsWindowReady()) {
        CloseWindow();
    }
}
#else
void puf_init(Env* env, Dict* kwargs) {
    mjc_init(env, kwargs);
    mj_makeData(&env->d, (float*)calloc(MJ_SCRATCH, sizeof(float)), 1);
}

void puf_reset(Env* env) {
    mjc_reset(env);
}

void puf_step(Env* env) {
    mjc_step(env);
}

void puf_render(Env* env) {
    mjc_render(env);
}

void puf_close(Env* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
#endif
