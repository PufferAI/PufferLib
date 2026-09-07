// Vector backend, included at the end of each mjc_<env>.h.

#if PUF_BACKEND == PUF_GPU
#define MJC_BLOCK 128
#define MJC_STATE (offsetof(Env, d) + offsetof(MjData, xquat))

struct {
    Env* envs;
    int n;
    cudaStream_t stream;
} mjc_gpu;

__global__ void mjc_kernel(Env* envs, int n, int reset) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        Env env;
        memcpy(&env, &envs[i], MJC_STATE);
        if (reset) {
            mjc_reset(&env);
        } else {
            mjc_step(&env);
        }
        memcpy(&envs[i], &env, MJC_STATE);
    }
}

// The trainer resets before binding its stream, so resets go to the null stream
void mjc_launch(int reset) {
    mjc_kernel<<<(mjc_gpu.n + MJC_BLOCK - 1) / MJC_BLOCK, MJC_BLOCK, 0, mjc_gpu.stream>>>(
        mjc_gpu.envs, mjc_gpu.n, reset);
    assert(cudaGetLastError() == cudaSuccess);
}

Env* puf_vec_create(int n, Dict* kwargs, obs_t* observations, float* actions, float* rewards,
    float* terminals) {
    Env* host = (Env*)calloc(n, sizeof(Env));
    MjModel* m;
    assert(cudaMalloc((void**)&m, sizeof(MjModel)) == cudaSuccess);
    float* scratch;
    assert(cudaMalloc((void**)&scratch, sizeof(float)*((n + 31) / 32*32)*MJ_SCRATCH) == cudaSuccess
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
    mjc_launch(1);
}

void puf_step(Env* envs) {
    mjc_launch(0);
}

// Draw env 0 with the host model (mj_render recomputes kinematics from qpos)
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

#ifdef PUFFERCPU_EVAL_MAIN
// Whole control steps (MJC_FRAME_SKIP substeps at once) strobe on screen at
// 20 Hz. Run the real mjc_step for exact rewards, logs and resets, then
// rewind and re-integrate the same deterministic substeps one per rendered
// frame; PUF_STEPS_PER_SEC is therefore the substep rate 1/opt_timestep.
#define PUF_EVAL_SHOULD_FORWARD
#define MJC_STATE (offsetof(MjData, xquat))
char mjc_true[MJC_STATE];

void puf_reset(Env* env) {
    mjc_reset(env);
    memcpy(mjc_true, &env->d, MJC_STATE);
}

void puf_step(Env* env) {
    MjData* d = &env->d;
    if (env->tick_frames_left > 0) {
        env->tick_frames_left--;
        mj_step(env->m, d);
        return;
    }
    memcpy(d, mjc_true, MJC_STATE);
    mjc_step(env);
    float ctrl[MJ_MAX_NU];
    memcpy(ctrl, d->ctrl, sizeof(ctrl));
    char post[MJC_STATE];
    memcpy(post, d, MJC_STATE);
    memcpy(d, mjc_true, MJC_STATE);
    memcpy(mjc_true, post, MJC_STATE);
    memcpy(d->ctrl, ctrl, sizeof(ctrl));
    mj_step(env->m, d);
    env->tick_frames_left = MJC_FRAME_SKIP - 1;
}
#else
void puf_reset(Env* env) {
    mjc_reset(env);
}

void puf_step(Env* env) {
    mjc_step(env);
}
#endif

void puf_render(Env* env) {
    mjc_render(env);
}

void puf_close(Env* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
#endif
