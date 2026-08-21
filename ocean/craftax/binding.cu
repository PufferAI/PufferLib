// binding.cu - device-resident Craftax env for the puffer static vec.
//
// Wraps the single-file CUDA port (craftax.cu, an env-only snapshot of
// github.com/Infatoshi/craftax.cu craftax_full.cu: the standalone repo's
// trainer/bench harness is stripped by its trim_for_embed.py; game logic
// and the step/reset/encode kernels are verbatim)
// behind the MY_GPU_ENV hooks in src/vecenv.h. The whole env lives on the GPU:
// per rollout step the only traffic is device-to-device copies between the
// core's SoA buffers and the vec's gpu_* tensors — no pinned-host staging, no
// per-step synchronization, no OpenMP. All kernels launch on the caller's
// (per-buffer, PyTorch-managed) stream so the policy forward and the env step
// chain back-to-back asynchronously.
//
// Seeding matches the CPU binding exactly: env i gets rng=i, seed=seed_offset+i
// (k_env_init in the core), so a device rollout is comparable to the CPU env
// stepped with the same actions. Build with the same -DCRAFTAX_COMPACT_OBS
// setting as binding.c or the obs element types will disagree.

#define CRAFTAX_CU_LIB
#include "craftax.cu"

// tensor.h expects the trainer's precision_t under __CUDACC__; PrecisionTensor
// is unused here, so only the element size must match the build mode.
#ifdef PRECISION_FLOAT
typedef float precision_t;
#else
typedef unsigned short precision_t;  // bf16 storage
#endif
#include "vecenv.h"  // types only: OBS_SIZE is not defined in this TU

static CuVec g_cf_gpu_vec;
static int g_cf_gpu_active = 0;
static CuVec g_cf_gpu_pool_vec;
static int g_cf_gpu_pool_active = 0;
static int g_cf_gpu_pool_size = 0;

typedef struct CfgpuPoolSoaState {
#define CFGPU_POOL_SOA_FIELD(f, t, k) t f[k];
    CF_SOA_FIELDS(CFGPU_POOL_SOA_FIELD)
#undef CFGPU_POOL_SOA_FIELD
} CfgpuPoolSoaState;

static CfgpuPoolSoaState* d_cf_gpu_pool_soa = NULL;
static __constant__ CfgpuPoolSoaState* g_cfgpu_pool_soa = NULL;

#ifdef CRAFTAX_COMPACT_OBS
static __global__ void k_cfgpu_canonicalize_rewards(float* rewards, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float reward_grid = 1048576.0f;  // 2^20, exact in float32
    rewards[i] = nearbyintf(rewards[i] * reward_grid) / reward_grid;
}
#endif

// CraftaxState's cold maps remain AoS, but the core stores hot fields in
// per-field SoA allocations.  A usable reset pool therefore needs both the
// dormant core AoS pointer and matching SoA pointers.  The pool is generated
// as a temporary CuVec so it uses exactly the core's normal c_init/worldgen
// path; these symbols retain its SoA allocations after the live vec becomes
// the process-wide core arena.
#define CFGPU_POOL_SOA_DECL(f, t, k) \
    static __constant__ t* g_cfgpu_pool_##f = NULL;
CF_SOA_FIELDS(CFGPU_POOL_SOA_DECL)
#undef CFGPU_POOL_SOA_DECL

static __device__ inline void cfgpu_copy_pool_aos(
    CraftaxState* dst, int pool_idx, int pool_size,
    unsigned lane, unsigned total_lanes
) {
    (void)pool_size;
    static_assert(sizeof(CraftaxState) % sizeof(uint4) == 0,
                  "CraftaxState pool entries must remain uint4-aligned");
    static_assert(offsetof(CraftaxState, fractal_noise_angles) == 24,
                  "re-audit the unused pool field omission");
    static_assert(offsetof(CraftaxState, map) == 40,
                  "re-audit the aligned pool copy ranges");
    static_assert(offsetof(CraftaxState, lazy_floor_keys) == 62392,
                  "re-audit the eager-pool key omission");
    uint4* dst_vec = (uint4*)(void*)dst;
    const uint4* src_vec =
        (const uint4*)(const void*)&g_craftax_reset_pool[pool_idx];
    uint64_t* dst_u64 = (uint64_t*)(void*)dst;
    const uint64_t* src_u64 =
        (const uint64_t*)(const void*)&g_craftax_reset_pool[pool_idx];
    // potion_mapping is live; fractal_noise_angles is never referenced.
    if (lane == 0) {
        dst_vec[0] = src_vec[0];
        dst_u64[2] = src_u64[2];
        // The live map span starts/ends 8B off a uint4 boundary.
        dst_u64[5] = src_u64[5];
        dst_u64[7798] = src_u64[7798];
    }
    // Maps and ladders vary by pool entry. lazy_floor_keys is ignored for
    // these eager worlds because lazy_floors_pending is always zero.
    for (size_t i = 3 + lane; i < 3899; i += total_lanes) {
        dst_vec[i] = src_vec[i];
    }
}

static __device__ inline void cfgpu_copy_pool_soa_flat(
    CraftaxState* dst, int pool_idx, int pool_size,
    unsigned lane, unsigned total_lanes
) {
#define CFGPU_POOL_SOA_COPY_FLAT(f, t, k) \
    for (int _j = (int)lane; _j < (k); _j += (int)total_lanes) { \
        CF2(f, _j, dst) = g_cfgpu_pool_soa[pool_idx].f[_j]; \
    }
    CF_SOA_FIELDS_FLAT(CFGPU_POOL_SOA_COPY_FLAT)
#undef CFGPU_POOL_SOA_COPY_FLAT
}

static __device__ inline void cfgpu_copy_pool_soa_level(
    CraftaxState* dst, int pool_idx, int pool_size,
    unsigned lane, unsigned total_lanes
) {
#define CFGPU_POOL_SOA_COPY_LEVEL(f, t, k) \
    for (int _j = (int)lane; _j < (k); _j += (int)total_lanes) { \
        CF2(f, _j, dst) = g_cfgpu_pool_soa[pool_idx].f[_j]; \
    }
    CF_SOA_FIELDS_LEVEL(CFGPU_POOL_SOA_COPY_LEVEL)
#undef CFGPU_POOL_SOA_COPY_LEVEL
}

static __global__ void k_cfgpu_pack_pool_soa(int pool_size) {
    int pool_idx = (int)blockIdx.x;
    if (pool_idx >= pool_size) return;
#define CFGPU_POOL_SOA_PACK(f, t, k) \
    for (int _j = (int)threadIdx.x; _j < (k); _j += blockDim.x) { \
        g_cfgpu_pool_soa[pool_idx].f[_j] = \
            g_cfgpu_pool_##f[(size_t)_j * (size_t)pool_size \
                             + (size_t)pool_idx]; \
    }
    CF_SOA_FIELDS(CFGPU_POOL_SOA_PACK)
#undef CFGPU_POOL_SOA_PACK
}

// Host c_init chooses pool[env->seed % pool_size].  Run this once after the
// live CuVec has initialized its per-env RNG keys, then refresh observations.
static __global__ void k_cfgpu_init_from_pool(
    Craftax* envs, int num_envs, int pool_size
) {
    int env_idx = (int)blockIdx.x;
    if (env_idx >= num_envs) return;
    Craftax* env = &envs[env_idx];
    int pool_idx = (int)(env->seed % (uint64_t)pool_size);
    cfgpu_copy_pool_aos(
        env->state, pool_idx, pool_size, threadIdx.x, blockDim.x);
    cfgpu_copy_pool_soa_flat(
        env->state, pool_idx, pool_size, threadIdx.x, blockDim.x);
    cfgpu_copy_pool_soa_level(
        env->state, pool_idx, pool_size, threadIdx.x, blockDim.x);
}

// Host episode resets choose pool[reset_key.word[0] % pool_size].  The core's
// scalar and lazy reset-list kernels both bypass its dormant seed-reset pool
// branch, so the wrapper routes only pool-enabled list resets through here.
static __global__ void __launch_bounds__(256) k_cfgpu_reset_list_pool(
    Craftax* envs, const CraftaxResetRec* resets, int pool_size
) {
    const int roles_per_reset = 13;
    const int reset_group = (int)blockIdx.x / roles_per_reset;
    const int role = (int)blockIdx.x % roles_per_reset;
    const int reset_groups = (int)gridDim.x / roles_per_reset;
    for (int idx = reset_group; idx < g_reset_count; idx += reset_groups) {
        Craftax* env = &envs[resets[idx].env];
        int pool_idx = (int)(resets[idx].key0 % (uint32_t)pool_size);
        if (role < 8) {
            unsigned lane =
                (unsigned)(role * (int)blockDim.x + threadIdx.x);
            cfgpu_copy_pool_aos(
                env->state, pool_idx, pool_size, lane, 8 * blockDim.x);
        } else if (role < 12) {
            unsigned lane = (unsigned)(
                (role - 8) * (int)blockDim.x + threadIdx.x);
            cfgpu_copy_pool_soa_level(
                env->state, pool_idx, pool_size, lane, 4 * blockDim.x);
        } else if (threadIdx.x < 128) {
            cfgpu_copy_pool_soa_flat(
                env->state, pool_idx, pool_size, threadIdx.x, 128);
        }
    }
}

static void cfgpu_clear_core_pool(void) {
    int zero = 0;
    CraftaxState* null_pool = NULL;
    CU_CHECK(cudaMemcpyToSymbol(g_craftax_reset_pool_size, &zero, sizeof(zero)));
    CU_CHECK(cudaMemcpyToSymbol(
        g_craftax_reset_pool, &null_pool, sizeof(null_pool)));
    CU_CHECK(cudaMemcpyToSymbol(g_craftax_reset_pool_ready, &zero, sizeof(zero)));
}

static void cfgpu_bind_pool(CuVec* pool, int pool_size) {
    int soa_idx = 0;
#define CFGPU_POOL_SOA_BIND(f, t, k) { \
        t* ptr = (t*)pool->d_soa[soa_idx++]; \
        CU_CHECK(cudaMemcpyToSymbol(g_cfgpu_pool_##f, &ptr, sizeof(ptr))); \
    }
    CF_SOA_FIELDS(CFGPU_POOL_SOA_BIND)
#undef CFGPU_POOL_SOA_BIND
    if (soa_idx != pool->num_soa) {
        fprintf(stderr, "craftax gpu env: reset-pool SoA layout mismatch\n");
        exit(1);
    }
    CU_CHECK(cudaMemcpyToSymbol(
        g_craftax_reset_pool, &pool->d_states, sizeof(pool->d_states)));
    CU_CHECK(cudaMemcpyToSymbol(
        g_craftax_reset_pool_size, &pool_size, sizeof(pool_size)));
    int ready = 1;
    CU_CHECK(cudaMemcpyToSymbol(
        g_craftax_reset_pool_ready, &ready, sizeof(ready)));
    CU_CHECK(cudaMalloc(
        &d_cf_gpu_pool_soa,
        (size_t)pool_size * sizeof(CfgpuPoolSoaState)));
    CU_CHECK(cudaMemcpyToSymbol(
        g_cfgpu_pool_soa, &d_cf_gpu_pool_soa, sizeof(d_cf_gpu_pool_soa)));
    k_cfgpu_pack_pool_soa<<<pool_size, 128>>>(pool_size);
}

extern "C" void my_gpu_init(StaticVec* vec, Dict* vec_kwargs, Dict* env_kwargs) {
    (void)vec_kwargs;
    if (vec->buffers != 1) {
        fprintf(stderr,
                "craftax gpu env: num_buffers must be 1 (got %d) — the device "
                "env has nothing to overlap with CPU stepping\n", vec->buffers);
        exit(1);
    }
    uint64_t seed_offset = 0;
    DictItem* item = dict_get_unsafe(env_kwargs, "seed_offset");
    if (item != NULL) seed_offset = (uint64_t)item->value;

    int reset_pool_size = 0;
    DictItem* pool_item = dict_get_unsafe(env_kwargs, "reset_pool_size");
    if (pool_item != NULL) reset_pool_size = (int)pool_item->value;

    if (reset_pool_size > 0) {
        // The host creates pool entry i from seed i before it enables lazy
        // floors, so pooled worlds are fully generated even when the live vec
        // uses lazy mode.  Force eager generation only while constructing the
        // temporary pool CuVec, then restore the caller's process setting.
        cfgpu_clear_core_pool();
        const char* old_lazy = getenv("CRAFTAX_CU_LAZY");
        char* saved_lazy = old_lazy == NULL ? NULL : strdup(old_lazy);
        CU_CHECK(cudaDeviceSynchronize());
        setenv("CRAFTAX_CU_LAZY", "0", 1);
        cu_vec_init(&g_cf_gpu_pool_vec, reset_pool_size, 0);
        if (saved_lazy != NULL) {
            setenv("CRAFTAX_CU_LAZY", saved_lazy, 1);
            free(saved_lazy);
        } else {
            unsetenv("CRAFTAX_CU_LAZY");
        }
        g_cf_gpu_pool_active = 1;
        g_cf_gpu_pool_size = reset_pool_size;
    }

    cu_vec_init(&g_cf_gpu_vec, vec->total_agents, seed_offset);
    if (reset_pool_size > 0) {
        cfgpu_bind_pool(&g_cf_gpu_pool_vec, reset_pool_size);
        k_cfgpu_init_from_pool<<<vec->total_agents, 128>>>(
            g_cf_gpu_vec.d_envs, vec->total_agents, reset_pool_size);
        dim3 enc_block(32, CRAFTAX_ENC_WARPS_PER_BLOCK);
        int enc_grid =
            (vec->total_agents + CRAFTAX_ENC_WARPS_PER_BLOCK - 1)
            / CRAFTAX_ENC_WARPS_PER_BLOCK;
        k_encode<<<enc_grid, enc_block>>>(
            g_cf_gpu_vec.d_envs, vec->total_agents);
        k_encode_tail<<<(vec->total_agents + 255) / 256, 256>>>(
            g_cf_gpu_vec.d_envs, vec->total_agents);
    }
    CU_CHECK(cudaDeviceSynchronize());
    g_cf_gpu_active = 1;
}

// Publish the core's freshly-initialized obs (k_env_init already ran c_init +
// encode for every env) into the vec's device buffers.
extern "C" void my_gpu_reset(StaticVec* vec) {
    CuVec* v = &g_cf_gpu_vec;
    size_t n = (size_t)v->num_envs;
    CU_CHECK(cudaMemcpy(vec->gpu_observations, v->d_obs,
                        n * CRAFTAX_OBS_SIZE * sizeof(CraftaxObs),
                        cudaMemcpyDeviceToDevice));
    CU_CHECK(cudaMemset(vec->gpu_rewards, 0, n * sizeof(float)));
    CU_CHECK(cudaMemset(vec->gpu_terminals, 0, n * sizeof(float)));
    CU_CHECK(cudaDeviceSynchronize());
}

// One env step, fully async on `stream`: actions in (D2D), gameplay + spawn
// compaction + done-list resets + canonical obs encode (the hash-sealed bench
// sequence from cu_step_launch, with actions read from env->actions instead of
// the bench's RNG stream), then obs/rewards/terminals out (D2D).
extern "C" void my_gpu_step(StaticVec* vec, cudaStream_t st) {
    CuVec* v = &g_cf_gpu_vec;
    int n = v->num_envs;
    CU_CHECK(cudaMemcpyAsync(v->d_actions, vec->gpu_actions,
                             (size_t)n * sizeof(float),
                             cudaMemcpyDeviceToDevice, st));
    CU_CHECK(cudaMemsetAsync(v->d_reset_count, 0, sizeof(int), st));
    CU_CHECK(cudaMemsetAsync(v->d_spawn_count, 0, sizeof(int), st));
    cu_launch_step_run(v->d_envs, n, v->d_resets, NULL, st);
    {
        int tail_blocks = (n * 32 + 255) / 256;
        if (tail_blocks > 512) tail_blocks = 512;
        k_spawn_tail<<<tail_blocks, 256, 0, st>>>();
    }
    if (g_cf_gpu_pool_size > 0) {
        const int roles_per_reset = 13;
        int reset_groups = n > 32 ? 32 : n;
        k_cfgpu_reset_list_pool
            <<<(reset_groups * roles_per_reset), 256, 0, st>>>(
            v->d_envs, v->d_resets, g_cf_gpu_pool_size);
    } else if (v->lazy) {
        cu_launch_reset_list_warp(v->d_envs, v->d_resets, n, st);
    } else {
        k_reset_list<<<(n + 63) / 64, 64, 0, st>>>(v->d_envs, v->d_resets);
    }
    {
        dim3 enc_block(32, CRAFTAX_ENC_WARPS_PER_BLOCK);
        int enc_grid = (n + CRAFTAX_ENC_WARPS_PER_BLOCK - 1)
            / CRAFTAX_ENC_WARPS_PER_BLOCK;
        k_encode<<<enc_grid, enc_block, 0, st>>>(v->d_envs, n);
        // k_encode covers only the packed map block; the scalar tail
        // (inventory/intrinsics/light) is written by k_step in the core's
        // bench path, and k_step_run writes no obs bytes at all — so
        // materialize it explicitly (the core's runverify does the same).
        k_encode_tail<<<(n + 255) / 256, 256, 0, st>>>(v->d_envs, n);
    }
#ifdef CRAFTAX_COMPACT_OBS
    k_cfgpu_canonicalize_rewards<<<(n + 255) / 256, 256, 0, st>>>(
        v->d_rewards, n);
#endif
    CU_CHECK(cudaMemcpyAsync(vec->gpu_observations, v->d_obs,
                             (size_t)n * CRAFTAX_OBS_SIZE * sizeof(CraftaxObs),
                             cudaMemcpyDeviceToDevice, st));
    CU_CHECK(cudaMemcpyAsync(vec->gpu_rewards, v->d_rewards,
                             (size_t)n * sizeof(float),
                             cudaMemcpyDeviceToDevice, st));
    CU_CHECK(cudaMemcpyAsync(vec->gpu_terminals, v->d_terminals,
                             (size_t)n * sizeof(float),
                             cudaMemcpyDeviceToDevice, st));
    CU_CHECK(cudaGetLastError());
}

static __global__ void k_cfgpu_clear_logs(Craftax* envs, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float* f = (float*)&envs[i].log;
    for (int j = 0; j < (int)(sizeof(Log) / sizeof(float)); j++) f[j] = 0.0f;
}

// Drain per-env device logs into `logs_out` (an array of `n` Log structs —
// identical field layout to the CPU env's Log) and zero them device-side.
// binding.c's my_gpu_sync_logs folds these into the host envs' logs so the
// normal static_vec_log aggregation path works unchanged.
extern "C" void craftax_gpu_collect_logs(void* logs_out, int n) {
    CuVec* v = &g_cf_gpu_vec;
    if (!g_cf_gpu_active || n > v->num_envs) return;
    Craftax* h_envs = (Craftax*)malloc((size_t)n * sizeof(Craftax));
    CU_CHECK(cudaMemcpy(h_envs, v->d_envs, (size_t)n * sizeof(Craftax),
                        cudaMemcpyDeviceToHost));
    Log* out = (Log*)logs_out;
    for (int i = 0; i < n; i++) out[i] = h_envs[i].log;
    free(h_envs);
    k_cfgpu_clear_logs<<<(n + 255) / 256, 256>>>(v->d_envs, n);
    CU_CHECK(cudaDeviceSynchronize());
}

extern "C" void my_gpu_close(StaticVec* vec) {
    (void)vec;
    if (!g_cf_gpu_active) return;
    CU_CHECK(cudaDeviceSynchronize());
    if (g_cf_gpu_pool_active) cfgpu_clear_core_pool();
    cu_vec_free(&g_cf_gpu_vec);
    if (g_cf_gpu_pool_active) {
        cudaFree(d_cf_gpu_pool_soa);
        d_cf_gpu_pool_soa = NULL;
        cu_vec_free(&g_cf_gpu_pool_vec);
        g_cf_gpu_pool_active = 0;
        g_cf_gpu_pool_size = 0;
    }
    g_cf_gpu_active = 0;
}
