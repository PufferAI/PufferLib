// craftax_full.cu -- CUDA port of the full Craftax game (milestone 1:
// correctness only; one thread per env, naive AoS CraftaxState in global
// memory).
//
// GENERATED mechanically from craftax_full.c by scratchpad gen_cu.py:
//   - game logic kept token-identical except `static ` -> `static __device__ `
//     at line starts, `#pragma omp` lines dropped, and the targeted edits
//     marked "[cuda port]" below (host-alloc conveniences, light-level table,
//     pool-free reset-on-done).
//   - the pthread world pool and the OpenMP CLI harness are replaced by a
//     CUDA host harness at the bottom of this file (hash / cmp / stats /
//     bench modes; hash mode computes the same FNV-1a trajectory hash over
//     (obs, rewards, dones) as ./craftax_full hash).
//
// Build:  make craftax_full_cuda   (nvcc -O3 -arch=native --expt-relaxed-constexpr)
// Usage:  ./craftax_full_cuda hash  --envs N --steps M [--seed S]
//         ./craftax_full_cuda cmp   --dump FILE
//         ./craftax_full_cuda stats --envs N --steps M [--seed S]
//         ./craftax_full_cuda bench --envs N --iters M [--seed S]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <curand_kernel.h>  // policy weight init + action sampling (run mode)

// [cuda port] compatibility shims for the transformed C body below.
// gcc atomics: only touched by the single-threaded global-init kernel and
// (afterwards) read-only, so plain accesses are sufficient.
#define __atomic_load_n(ptr, order) (*(ptr))
#define __atomic_store_n(ptr, val, order) ((void)(*(ptr) = (val)))
// device ctz for the spawn bitboard scans (callers guarantee x != 0)
static __device__ inline int32_t craftax_cu_ctzll(unsigned long long x) {
    return __ffsll((long long)x) - 1;
}
#define __builtin_ctzll(x) craftax_cu_ctzll(x)

// ============================================================
// ===== device game logic, transformed from craftax_full.c =====
// ============================================================
// craftax_full.c -- standalone single-file C port of the full Craftax game.
//
// Amalgamated mechanically (gen_amalgam.py) from PufferLib's parity-verified
// C implementation (ocean/craftax/*.h), which matches the JAX original with
// zero trajectory divergences. Game logic is byte-identical to the source
// headers; only the raylib renderer is #if 0-gated and a standalone
// bench/hash harness is appended at the bottom.
//
// Build:  make full   (gcc -O3 -march=native -ffast-math -fopenmp ...)
// Usage:  ./craftax_full bench --envs N --iters M [--threads T] [--seed S]
//         ./craftax_full hash  --envs N --steps M [--seed S]

#define CRAFTAX_ENABLE_ENV_IMPL

// ============================================================
// ===== threefry.h =====
// ============================================================
// Fast RNG helpers for Craftax.
// Replaces JAX Threefry with SplitMix64-based hashing for ~20-50x speedup.
// NOT cryptographically secure and NOT JAX-compatible.


#include <stdint.h>
#include <stddef.h>
#include <string.h>

typedef struct CraftaxThreefryKey {
    uint32_t word[2];
} CraftaxThreefryKey;

static __device__ inline uint64_t craftax_key_to_u64(CraftaxThreefryKey key) {
    return ((uint64_t)key.word[1] << 32) | key.word[0];
}

static __device__ inline CraftaxThreefryKey craftax_u64_to_key(uint64_t x) {
    CraftaxThreefryKey key = {{(uint32_t)x, (uint32_t)(x >> 32)}};
    return key;
}

static __device__ inline uint32_t craftax_rotl32(uint32_t x, uint32_t k) {
    return (uint32_t)((x << k) | (x >> (32u - k)));
}

static __device__ inline CraftaxThreefryKey craftax_prng_key(uint32_t seed) {
    CraftaxThreefryKey key = {{seed, seed ^ 0x9E3779B9u}};
    return key;
}

// MurmurHash3 64-bit finalizer — fast and good mixing
static __device__ inline uint64_t craftax_mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

// Core hash: mixes key state with counter, returns 64 bits of pseudo-randomness
static __device__ inline uint64_t craftax_fast_hash64(CraftaxThreefryKey key, uint64_t counter) {
    uint64_t x = craftax_key_to_u64(key);
    x ^= counter;
    return craftax_mix64(x);
}

static __device__ inline void craftax_threefry2x32(
    CraftaxThreefryKey key,
    uint32_t count0,
    uint32_t count1,
    uint32_t out[2]
) {
    uint64_t h = craftax_fast_hash64(key, ((uint64_t)count1 << 32) | count0);
    out[0] = (uint32_t)h;
    out[1] = (uint32_t)(h >> 32);
}

static __device__ inline CraftaxThreefryKey craftax_threefry_counter_key(
    CraftaxThreefryKey key,
    uint32_t count0,
    uint32_t count1
) {
    return craftax_u64_to_key(craftax_fast_hash64(key, ((uint64_t)count1 << 32) | count0));
}

// Fast split: sequential PCG-style advancement
static __device__ inline void craftax_threefry_split(
    CraftaxThreefryKey key,
    CraftaxThreefryKey* left,
    CraftaxThreefryKey* right
) {
    uint64_t state = craftax_key_to_u64(key);
    uint64_t s1 = state * 6364136223846793005ULL + 1;
    uint64_t s2 = s1 * 6364136223846793005ULL + 1;
    *left = craftax_u64_to_key(s1);
    *right = craftax_u64_to_key(s2);
}

static __device__ inline void craftax_threefry_split_n(
    CraftaxThreefryKey key,
    CraftaxThreefryKey* out,
    size_t count
) {
    uint64_t state = craftax_key_to_u64(key);
    for (size_t i = 0; i < count; i++) {
        state = state * 6364136223846793005ULL + 1;
        out[i] = craftax_u64_to_key(state);
    }
}

static __device__ inline CraftaxThreefryKey craftax_threefry_fold_in(
    CraftaxThreefryKey key,
    uint32_t data
) {
    return craftax_threefry_counter_key(key, 0u, data);
}

static __device__ inline uint32_t craftax_threefry_uniform_u32_at(
    CraftaxThreefryKey key,
    uint64_t index
) {
    uint64_t h = craftax_fast_hash64(key, index);
    return (uint32_t)h ^ (uint32_t)(h >> 32);
}

static __device__ inline uint32_t craftax_threefry_uniform_u32(CraftaxThreefryKey key) {
    return craftax_threefry_uniform_u32_at(key, 0u);
}

static __device__ inline float craftax_threefry_uniform_f32_at(
    CraftaxThreefryKey key,
    uint64_t index
) {
    uint32_t bits = craftax_threefry_uniform_u32_at(key, index);
    uint32_t float_bits = (bits >> 9u) | 0x3F800000u;
    float value;
    memcpy(&value, &float_bits, sizeof(value));
    return value - 1.0f;
}

static __device__ inline float craftax_threefry_uniform_f32(CraftaxThreefryKey key) {
    return craftax_threefry_uniform_f32_at(key, 0u);
}

// ============================================================
// ===== noise.h =====
// ============================================================
// Native C port of craftax/craftax/util/noise.py.


#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


#ifndef CRAFTAX_NOISE_PI2
#define CRAFTAX_NOISE_PI2 6.28318530717958647692f
#endif

#ifndef CRAFTAX_NOISE_SQRT2
#define CRAFTAX_NOISE_SQRT2 1.41421356237309504880f
#endif

static __device__ inline float craftax_noise_interpolant(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

static __device__ inline float craftax_noise_gradient_angle(
    CraftaxThreefryKey angle_key,
    int res_cols,
    int row,
    int col,
    const float* override_angles
) {
    int width = res_cols + 1;
    uint64_t index = (uint64_t)row * (uint64_t)width + (uint64_t)col;
    float unit = override_angles == NULL
        ? craftax_threefry_uniform_f32_at(angle_key, index)
        : override_angles[index];
    return CRAFTAX_NOISE_PI2 * unit;
}

static __device__ inline void craftax_noise_gradient(
    CraftaxThreefryKey angle_key,
    int res_cols,
    int row,
    int col,
    const float* override_angles,
    float* gx,
    float* gy
) {
    float angle = craftax_noise_gradient_angle(
        angle_key,
        res_cols,
        row,
        col,
        override_angles
    );
    *gx = cosf(angle);
    *gy = sinf(angle);
}

#ifndef CRAFTAX_NOISE_MAX_GRAD
// Largest gradient grid used by worldgen: res (6,24) at 48x48 -> 7x25 = 175.
// Sized tightly ([cuda port]: was 1024 = 8KB of per-thread stack); a grid
// larger than this falls back to per-cell lookups, which never triggers.
#define CRAFTAX_NOISE_MAX_GRAD 176
#endif

static __device__ inline void craftax_generate_perlin_noise_2d_scalar(
    CraftaxThreefryKey rng,
    int rows,
    int cols,
    int res_rows,
    int res_cols,
    const float* override_angles,
    float* out
) {
    CraftaxThreefryKey unused;
    CraftaxThreefryKey angle_key;
    craftax_threefry_split(rng, &unused, &angle_key);

    int cell_rows = rows / res_rows;
    int cell_cols = cols / res_cols;

    // The gradient grid is tiny compared to the output map (e.g. 4x4 vs
    // 48x48), but the naive loop recomputes 4 sincos per output cell.
    // Precompute cos/sin for every grid corner once; values and their
    // consumers are unchanged, so output is bit-identical.
    int grad_w = res_cols + 1;
    int grad_h = res_rows + 1;
    float grad_x[CRAFTAX_NOISE_MAX_GRAD];
    float grad_y[CRAFTAX_NOISE_MAX_GRAD];
    bool table_ok = grad_w * grad_h <= CRAFTAX_NOISE_MAX_GRAD;
    if (table_ok) {
        for (int r = 0; r < grad_h; r++) {
            for (int c = 0; c < grad_w; c++) {
                float angle = craftax_noise_gradient_angle(
                    angle_key, res_cols, r, c, override_angles);
                grad_x[r * grad_w + c] = cosf(angle);
                grad_y[r * grad_w + c] = sinf(angle);
            }
        }
    }

    for (int row = 0; row < rows; row++) {
        int grad_row = row / cell_rows;
        float local_row = (float)(row - grad_row * cell_rows) / (float)cell_rows;
        float interp_row = craftax_noise_interpolant(local_row);

        for (int col = 0; col < cols; col++) {
            int grad_col = col / cell_cols;
            float local_col = (float)(col - grad_col * cell_cols) / (float)cell_cols;
            float interp_col = craftax_noise_interpolant(local_col);

            float g00x;
            float g00y;
            float g10x;
            float g10y;
            float g01x;
            float g01y;
            float g11x;
            float g11y;
            if (table_ok) {
                int i00 = grad_row * grad_w + grad_col;
                int i10 = i00 + grad_w;
                g00x = grad_x[i00];     g00y = grad_y[i00];
                g10x = grad_x[i10];     g10y = grad_y[i10];
                g01x = grad_x[i00 + 1]; g01y = grad_y[i00 + 1];
                g11x = grad_x[i10 + 1]; g11y = grad_y[i10 + 1];
            } else {
                craftax_noise_gradient(
                    angle_key, res_cols, grad_row, grad_col,
                    override_angles, &g00x, &g00y);
                craftax_noise_gradient(
                    angle_key, res_cols, grad_row + 1, grad_col,
                    override_angles, &g10x, &g10y);
                craftax_noise_gradient(
                    angle_key, res_cols, grad_row, grad_col + 1,
                    override_angles, &g01x, &g01y);
                craftax_noise_gradient(
                    angle_key, res_cols, grad_row + 1, grad_col + 1,
                    override_angles, &g11x, &g11y);
            }

            float n00 = local_row * g00x;
            n00 += local_col * g00y;
            float n10 = (local_row - 1.0f) * g10x;
            n10 += local_col * g10y;
            float n01 = local_row * g01x;
            n01 += (local_col - 1.0f) * g01y;
            float n11 = (local_row - 1.0f) * g11x;
            n11 += (local_col - 1.0f) * g11y;

            float n0 = n00 * (1.0f - interp_row) + interp_row * n10;
            float n1 = n01 * (1.0f - interp_row) + interp_row * n11;
            out[(size_t)row * (size_t)cols + (size_t)col] =
                CRAFTAX_NOISE_SQRT2 * ((1.0f - interp_col) * n0 + interp_col * n1);
        }
    }
}

static __device__ inline void craftax_generate_perlin_noise_2d(
    CraftaxThreefryKey rng,
    int rows,
    int cols,
    int res_rows,
    int res_cols,
    const float* override_angles,
    float* out
) {
    craftax_generate_perlin_noise_2d_scalar(
        rng, rows, cols, res_rows, res_cols, override_angles, out);
}

static __device__ inline void craftax_generate_fractal_noise_2d(
    CraftaxThreefryKey rng,
    int rows,
    int cols,
    int res_rows,
    int res_cols,
    int octaves,
    float persistence,
    int lacunarity,
    const float* override_angles,
    float* out
) {
    size_t size = (size_t)rows * (size_t)cols;
    // [cuda port] every worldgen call site uses octaves == 1, so the perlin
    // field is generated straight into `out` instead of staging it in a 9KB
    // per-thread buffer. The accumulation ops (out = 0.0f, out += 1.0f * v)
    // are kept verbatim so the result stays bit-identical (incl. -0 -> +0).
    if (octaves != 1) { __trap(); }
    (void)persistence;
    (void)lacunarity;

    CraftaxThreefryKey next_rng;
    CraftaxThreefryKey noise_key;
    craftax_threefry_split(rng, &next_rng, &noise_key);
    rng = next_rng;

    craftax_generate_perlin_noise_2d(
        noise_key,
        rows,
        cols,
        res_rows,
        res_cols,
        override_angles,
        out
    );

    float amplitude = 1.0f;
    for (size_t i = 0; i < size; i++) {
        float perlin_value = out[i];
        out[i] = 0.0f;
        out[i] += amplitude * perlin_value;
    }

    float min_value = out[0];
    float max_value = out[0];
    for (size_t i = 1; i < size; i++) {
        if (out[i] < min_value) {
            min_value = out[i];
        }
        if (out[i] > max_value) {
            max_value = out[i];
        }
    }

    float scale = max_value - min_value;
    for (size_t i = 0; i < size; i++) {
        out[i] = (out[i] - min_value) / scale;
    }
}

// ============================================================
// ===== worldgen.h =====
// ============================================================
// Native Craftax reset world generation.
//
// This mirrors craftax/craftax/world_gen/world_gen.py for the default
// EnvParams and StaticEnvParams used by Craftax-Symbolic-v1 reset.


#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>


#define CRAFTAX_WG_MAP_SIZE 48
#define CRAFTAX_WG_MAP_CELLS (CRAFTAX_WG_MAP_SIZE * CRAFTAX_WG_MAP_SIZE)
#define CRAFTAX_WG_NUM_LEVELS 9
#define CRAFTAX_WG_OBS_ROWS 9
#define CRAFTAX_WG_OBS_COLS 11
#define CRAFTAX_WG_NUM_BLOCK_TYPES 37
#define CRAFTAX_WG_NUM_ITEM_TYPES 5
#define CRAFTAX_WG_NUM_MOB_CLASSES 5
#define CRAFTAX_WG_NUM_MOB_TYPES 8
#define CRAFTAX_WG_INVENTORY_OBS_SIZE 51

// Compact binary observation encoding.
// Each cell uses binary channels instead of one-hot:
//   6 bits: block type (0-63, covers 37 block types)
//   3 bits: item type+1 (0=no item, 1-5=item types)
//   4 bits per mob class: mob type+1 (0=no mob, 1-8=types) x 5 classes
//   1 bit : visibility
// Total: 30 binary channels per cell.
#define CRAFTAX_WG_BINARY_BLOCK_BITS 6
#define CRAFTAX_WG_BINARY_ITEM_BITS 3
#define CRAFTAX_WG_BINARY_MOB_BITS 4
#define CRAFTAX_WG_BINARY_VISIBILITY_BITS 1

#define CRAFTAX_WG_BINARY_CHANNELS_PER_CELL ( \
    CRAFTAX_WG_BINARY_BLOCK_BITS + \
    CRAFTAX_WG_BINARY_ITEM_BITS + \
    CRAFTAX_WG_NUM_MOB_CLASSES * CRAFTAX_WG_BINARY_MOB_BITS + \
    CRAFTAX_WG_BINARY_VISIBILITY_BITS \
)

#define CRAFTAX_WG_BINARY_MAP_OBS_SIZE ( \
    CRAFTAX_WG_OBS_ROWS * CRAFTAX_WG_OBS_COLS * CRAFTAX_WG_BINARY_CHANNELS_PER_CELL \
)
#define CRAFTAX_WG_OBS_WINDOW_CELLS (CRAFTAX_WG_OBS_ROWS * CRAFTAX_WG_OBS_COLS)
#define CRAFTAX_WG_CELL_TEMPLATE_BYTES ( \
    CRAFTAX_WG_BINARY_CHANNELS_PER_CELL * sizeof(float) \
)
#define CRAFTAX_WG_FULL_OBS_SIZE ( \
    CRAFTAX_WG_BINARY_MAP_OBS_SIZE + CRAFTAX_WG_INVENTORY_OBS_SIZE \
)

// Moonshot symbolic observation. Each visible cell stores compact float IDs:
// block, item+1, visible, and one mob type+1 slot for each mob class.
// The 51 scalar channels remain exact floats for oracle-expandability.
#define CRAFTAX_WG_PACKED_CHANNELS_PER_CELL (3 + CRAFTAX_WG_NUM_MOB_CLASSES)
#define CRAFTAX_WG_PACKED_MAP_OBS_SIZE ( \
    CRAFTAX_WG_OBS_ROWS * CRAFTAX_WG_OBS_COLS * CRAFTAX_WG_PACKED_CHANNELS_PER_CELL \
)
#define CRAFTAX_WG_PACKED_OBS_SIZE ( \
    CRAFTAX_WG_PACKED_MAP_OBS_SIZE + CRAFTAX_WG_INVENTORY_OBS_SIZE \
)

// Compact byte observation: every packed map channel is a small integer ID
// (block < 37, item+1 <= 5, visible bit, mob type+1 <= 8), so the 792 map
// channels fit in uint8 losslessly. The 51-float scalar tail is appended as
// raw float32 bytes; the GPU expands map bytes with .float() and reinterprets
// the tail, reproducing the float observation bit-for-bit at 3.4x less DMA.
#define CRAFTAX_WG_COMPACT_OBS_SIZE ( \
    CRAFTAX_WG_PACKED_MAP_OBS_SIZE \
    + CRAFTAX_WG_INVENTORY_OBS_SIZE * (int)sizeof(float) \
)

// Lookup tables for fast binary bit writing (eliminates loops/branches)
static __device__ const float CRAFTAX_WG_BLOCK_LUT[64][6] = {
    {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},{1.0f,0.0f,0.0f,0.0f,0.0f,0.0f},{0.0f,1.0f,0.0f,0.0f,0.0f,0.0f},{1.0f,1.0f,0.0f,0.0f,0.0f,0.0f},
    {0.0f,0.0f,1.0f,0.0f,0.0f,0.0f},{1.0f,0.0f,1.0f,0.0f,0.0f,0.0f},{0.0f,1.0f,1.0f,0.0f,0.0f,0.0f},{1.0f,1.0f,1.0f,0.0f,0.0f,0.0f},
    {0.0f,0.0f,0.0f,1.0f,0.0f,0.0f},{1.0f,0.0f,0.0f,1.0f,0.0f,0.0f},{0.0f,1.0f,0.0f,1.0f,0.0f,0.0f},{1.0f,1.0f,0.0f,1.0f,0.0f,0.0f},
    {0.0f,0.0f,1.0f,1.0f,0.0f,0.0f},{1.0f,0.0f,1.0f,1.0f,0.0f,0.0f},{0.0f,1.0f,1.0f,1.0f,0.0f,0.0f},{1.0f,1.0f,1.0f,1.0f,0.0f,0.0f},
    {0.0f,0.0f,0.0f,0.0f,1.0f,0.0f},{1.0f,0.0f,0.0f,0.0f,1.0f,0.0f},{0.0f,1.0f,0.0f,0.0f,1.0f,0.0f},{1.0f,1.0f,0.0f,0.0f,1.0f,0.0f},
    {0.0f,0.0f,1.0f,0.0f,1.0f,0.0f},{1.0f,0.0f,1.0f,0.0f,1.0f,0.0f},{0.0f,1.0f,1.0f,0.0f,1.0f,0.0f},{1.0f,1.0f,1.0f,0.0f,1.0f,0.0f},
    {0.0f,0.0f,0.0f,1.0f,1.0f,0.0f},{1.0f,0.0f,0.0f,1.0f,1.0f,0.0f},{0.0f,1.0f,0.0f,1.0f,1.0f,0.0f},{1.0f,1.0f,0.0f,1.0f,1.0f,0.0f},
    {0.0f,0.0f,1.0f,1.0f,1.0f,0.0f},{1.0f,0.0f,1.0f,1.0f,1.0f,0.0f},{0.0f,1.0f,1.0f,1.0f,1.0f,0.0f},{1.0f,1.0f,1.0f,1.0f,1.0f,0.0f},
    {0.0f,0.0f,0.0f,0.0f,0.0f,1.0f},{1.0f,0.0f,0.0f,0.0f,0.0f,1.0f},{0.0f,1.0f,0.0f,0.0f,0.0f,1.0f},{1.0f,1.0f,0.0f,0.0f,0.0f,1.0f},
    {0.0f,0.0f,1.0f,0.0f,0.0f,1.0f},{1.0f,0.0f,1.0f,0.0f,0.0f,1.0f},{0.0f,1.0f,1.0f,0.0f,0.0f,1.0f},{1.0f,1.0f,1.0f,0.0f,0.0f,1.0f},
    {0.0f,0.0f,0.0f,1.0f,0.0f,1.0f},{1.0f,0.0f,0.0f,1.0f,0.0f,1.0f},{0.0f,1.0f,0.0f,1.0f,0.0f,1.0f},{1.0f,1.0f,0.0f,1.0f,0.0f,1.0f},
    {0.0f,0.0f,1.0f,1.0f,0.0f,1.0f},{1.0f,0.0f,1.0f,1.0f,0.0f,1.0f},{0.0f,1.0f,1.0f,1.0f,0.0f,1.0f},{1.0f,1.0f,1.0f,1.0f,0.0f,1.0f},
    {0.0f,0.0f,0.0f,0.0f,1.0f,1.0f},{1.0f,0.0f,0.0f,0.0f,1.0f,1.0f},{0.0f,1.0f,0.0f,0.0f,1.0f,1.0f},{1.0f,1.0f,0.0f,0.0f,1.0f,1.0f},
    {0.0f,0.0f,1.0f,0.0f,1.0f,1.0f},{1.0f,0.0f,1.0f,0.0f,1.0f,1.0f},{0.0f,1.0f,1.0f,0.0f,1.0f,1.0f},{1.0f,1.0f,1.0f,0.0f,1.0f,1.0f},
    {0.0f,0.0f,0.0f,1.0f,1.0f,1.0f},{1.0f,0.0f,0.0f,1.0f,1.0f,1.0f},{0.0f,1.0f,0.0f,1.0f,1.0f,1.0f},{1.0f,1.0f,0.0f,1.0f,1.0f,1.0f},
    {0.0f,0.0f,1.0f,1.0f,1.0f,1.0f},{1.0f,0.0f,1.0f,1.0f,1.0f,1.0f},{0.0f,1.0f,1.0f,1.0f,1.0f,1.0f},{1.0f,1.0f,1.0f,1.0f,1.0f,1.0f},
};
static __device__ const float CRAFTAX_WG_ITEM_LUT[8][3] = {
    {0.0f,0.0f,0.0f},{1.0f,0.0f,0.0f},{0.0f,1.0f,0.0f},{1.0f,1.0f,0.0f},
    {0.0f,0.0f,1.0f},{1.0f,0.0f,1.0f},{0.0f,1.0f,1.0f},{1.0f,1.0f,1.0f},
};
static __device__ const float CRAFTAX_WG_MOB_LUT[16][4] = {
    {0.0f,0.0f,0.0f,0.0f},{1.0f,0.0f,0.0f,0.0f},{0.0f,1.0f,0.0f,0.0f},{1.0f,1.0f,0.0f,0.0f},
    {0.0f,0.0f,1.0f,0.0f},{1.0f,0.0f,1.0f,0.0f},{0.0f,1.0f,1.0f,0.0f},{1.0f,1.0f,1.0f,0.0f},
    {0.0f,0.0f,0.0f,1.0f},{1.0f,0.0f,0.0f,1.0f},{0.0f,1.0f,0.0f,1.0f},{1.0f,1.0f,0.0f,1.0f},
    {0.0f,0.0f,1.0f,1.0f},{1.0f,0.0f,1.0f,1.0f},{0.0f,1.0f,1.0f,1.0f},{1.0f,1.0f,1.0f,1.0f},
};
static __device__ float CRAFTAX_WG_VISIBLE_CELL_TEMPLATE_LUT[64][8][CRAFTAX_WG_BINARY_CHANNELS_PER_CELL];
static __device__ float CRAFTAX_WG_EMPTY_CELL_TEMPLATE[CRAFTAX_WG_BINARY_CHANNELS_PER_CELL];
static __device__ bool CRAFTAX_WG_CELL_TEMPLATE_READY = false;

static __device__ inline void craftax_wg_init_cell_templates(void) {
    if (CRAFTAX_WG_CELL_TEMPLATE_READY) {
        return;
    }

    for (int block = 0; block < 64; block++) {
        for (int item = 0; item < 8; item++) {
            float* cell = CRAFTAX_WG_VISIBLE_CELL_TEMPLATE_LUT[block][item];
            memcpy(cell, CRAFTAX_WG_BLOCK_LUT[block], 6 * sizeof(float));
            memcpy(cell + CRAFTAX_WG_BINARY_BLOCK_BITS, CRAFTAX_WG_ITEM_LUT[item], 3 * sizeof(float));
            cell[CRAFTAX_WG_BINARY_CHANNELS_PER_CELL - 1] = 1.0f;
        }
    }

    CRAFTAX_WG_CELL_TEMPLATE_READY = true;
}

#define CRAFTAX_WG_OBS_SIZE CRAFTAX_WG_PACKED_OBS_SIZE
#define CRAFTAX_WG_NUM_ACHIEVEMENTS 67
#define CRAFTAX_WG_MAX_MELEE_MOBS 3
#define CRAFTAX_WG_MAX_PASSIVE_MOBS 3
#define CRAFTAX_WG_MAX_RANGED_MOBS 2
#define CRAFTAX_WG_MAX_MOB_PROJECTILES 3
#define CRAFTAX_WG_MAX_PLAYER_PROJECTILES 3
#define CRAFTAX_WG_MAX_GROWING_PLANTS 10
#define CRAFTAX_WG_MONSTERS_KILLED_TO_CLEAR_LEVEL 8

// Backwards-compatible names used by the phase-1 floor-0 test.
#define CRAFTAX_OVERWORLD_SIZE CRAFTAX_WG_MAP_SIZE
#define CRAFTAX_OVERWORLD_CELLS CRAFTAX_WG_MAP_CELLS

#define CRAFTAX_WG_BLOCK_INVALID 0
#define CRAFTAX_WG_BLOCK_OUT_OF_BOUNDS 1
#define CRAFTAX_WG_BLOCK_GRASS 2
#define CRAFTAX_WG_BLOCK_WATER 3
#define CRAFTAX_WG_BLOCK_STONE 4
#define CRAFTAX_WG_BLOCK_TREE 5
#define CRAFTAX_WG_BLOCK_WOOD 6
#define CRAFTAX_WG_BLOCK_PATH 7
#define CRAFTAX_WG_BLOCK_COAL 8
#define CRAFTAX_WG_BLOCK_IRON 9
#define CRAFTAX_WG_BLOCK_DIAMOND 10
#define CRAFTAX_WG_BLOCK_CRAFTING_TABLE 11
#define CRAFTAX_WG_BLOCK_FURNACE 12
#define CRAFTAX_WG_BLOCK_SAND 13
#define CRAFTAX_WG_BLOCK_LAVA 14
#define CRAFTAX_WG_BLOCK_PLANT 15
#define CRAFTAX_WG_BLOCK_RIPE_PLANT 16
#define CRAFTAX_WG_BLOCK_WALL 17
#define CRAFTAX_WG_BLOCK_DARKNESS 18
#define CRAFTAX_WG_BLOCK_WALL_MOSS 19
#define CRAFTAX_WG_BLOCK_STALAGMITE 20
#define CRAFTAX_WG_BLOCK_SAPPHIRE 21
#define CRAFTAX_WG_BLOCK_RUBY 22
#define CRAFTAX_WG_BLOCK_CHEST 23
#define CRAFTAX_WG_BLOCK_FOUNTAIN 24
#define CRAFTAX_WG_BLOCK_FIRE_GRASS 25
#define CRAFTAX_WG_BLOCK_ICE_GRASS 26
#define CRAFTAX_WG_BLOCK_GRAVEL 27
#define CRAFTAX_WG_BLOCK_FIRE_TREE 28
#define CRAFTAX_WG_BLOCK_ICE_SHRUB 29
#define CRAFTAX_WG_BLOCK_ENCHANTMENT_TABLE_FIRE 30
#define CRAFTAX_WG_BLOCK_ENCHANTMENT_TABLE_ICE 31
#define CRAFTAX_WG_BLOCK_NECROMANCER 32
#define CRAFTAX_WG_BLOCK_GRAVE 33
#define CRAFTAX_WG_BLOCK_GRAVE2 34
#define CRAFTAX_WG_BLOCK_GRAVE3 35
#define CRAFTAX_WG_BLOCK_NECROMANCER_VULNERABLE 36

#define CRAFTAX_WG_ITEM_NONE 0
#define CRAFTAX_WG_ITEM_TORCH 1
#define CRAFTAX_WG_ITEM_LADDER_DOWN 2
#define CRAFTAX_WG_ITEM_LADDER_UP 3
#define CRAFTAX_WG_ITEM_LADDER_DOWN_BLOCKED 4

#define CRAFTAX_WG_ACTION_UP 3
#define CRAFTAX_WG_BOSS_FIGHT_SPAWN_TURNS 7
#define CRAFTAX_WG_PI 3.14159265358979323846f
// [cuda port] table indexed by timestep 0..100000 inclusive
#define CRAFTAX_DEFAULT_MAX_TIMESTEPS_TABLE 100001


// ============================================================
// [cuda port M3] SoA split of hot scalar state. Each field listed here is
// removed from CraftaxState/CraftaxWorldState and stored in a per-field
// global array laid out [sub_index][env] (stride g_cf_n), so a warp of
// adjacent envs coalesces its accesses. Pure layout change: values, types
// and every arithmetic op are identical, so trajectories are bit-exact.
// X(name, ctype, per_env_count)
// ============================================================
// Flat (non-level-indexed) fields: always cleared on reset.
#define CF_SOA_FIELDS_FLAT(X) \
    X(player_position, int32_t, 2) \
    X(player_level, int32_t, 1) \
    X(player_direction, int32_t, 1) \
    X(player_health, float, 1) \
    X(player_food, int32_t, 1) \
    X(player_drink, int32_t, 1) \
    X(player_energy, int32_t, 1) \
    X(player_mana, int32_t, 1) \
    X(is_sleeping, bool, 1) \
    X(is_resting, bool, 1) \
    X(player_recover, float, 1) \
    X(player_hunger, float, 1) \
    X(player_thirst, float, 1) \
    X(player_fatigue, float, 1) \
    X(player_recover_mana, float, 1) \
    X(player_xp, int32_t, 1) \
    X(player_dexterity, int32_t, 1) \
    X(player_strength, int32_t, 1) \
    X(player_intelligence, int32_t, 1) \
    X(inv_wood, int32_t, 1) \
    X(inv_stone, int32_t, 1) \
    X(inv_coal, int32_t, 1) \
    X(inv_iron, int32_t, 1) \
    X(inv_diamond, int32_t, 1) \
    X(inv_sapling, int32_t, 1) \
    X(inv_pickaxe, int32_t, 1) \
    X(inv_sword, int32_t, 1) \
    X(inv_bow, int32_t, 1) \
    X(inv_arrows, int32_t, 1) \
    X(inv_torches, int32_t, 1) \
    X(inv_ruby, int32_t, 1) \
    X(inv_sapphire, int32_t, 1) \
    X(inv_books, int32_t, 1) \
    X(inv_armour, int32_t, 4) \
    X(inv_potions, int32_t, 6) \
    X(learned_spells, bool, 2) \
    X(sword_enchantment, int32_t, 1) \
    X(bow_enchantment, int32_t, 1) \
    X(armour_enchantments, int32_t, 4) \
    X(boss_progress, int32_t, 1) \
    X(boss_timesteps_to_spawn_this_round, int32_t, 1) \
    X(light_level, float, 1) \
    X(achievements, bool, 67) \
    X(state_rng, uint32_t, 2) \
    X(timestep, int32_t, 1) \
    X(growing_plants_positions, int32_t, 20) \
    X(growing_plants_age, int32_t, 10) \
    X(growing_plants_mask, bool, 10) \
    X(lazy_floors_pending, uint32_t, 1)

// Level-indexed fields: on the lazy warp reset only levels that were
// actually generated during the dying episode need re-clearing; the
// per-entry level is recovered from the flat index _j below.
#define CF_SOA_FIELDS_LEVEL(X) \
    X(monsters_killed, int32_t, 9) \
    X(mob_position, int32_t, 270) \
    X(mob_health, float, 135) \
    X(mob_mask, bool, 135) \
    X(mob_attack_cooldown, int32_t, 135) \
    X(mob_type_id, int32_t, 135) \
    X(mob_bits, uint64_t, 432) \
    X(spawn_all_bits, uint64_t, 432) \
    X(spawn_grave_bits, uint64_t, 432) \
    X(spawn_water_bits, uint64_t, 432) \
    X(mob_projectile_directions, int32_t, 54) \
    X(player_projectile_directions, int32_t, 54) \
    X(chests_opened, bool, 9)

#define CF_SOA_FIELDS(X) CF_SOA_FIELDS_FLAT(X) CF_SOA_FIELDS_LEVEL(X)

// The SoA field pointers, the state base pointer and the env count are
// written exactly once from the host (cudaMemcpyToSymbol at arena init) and
// never from device code. Declaring them __constant__ instead of __device__
// (a) serves every CF() pointer fetch from the constant bank instead of a
// global-memory load (the single largest long-scoreboard stall site in
// k_step_run), and (b) lets the compiler CSE cf_slot()'s pointer-difference
// division, which was ~34% of the kernel's executed instructions. Pure
// address-path change: the loaded values and all arithmetic on the field
// data are untouched, so trajectories stay bit-exact.
#define CF_SOA_DECL(f, t, k) __constant__ t* g_cf_##f = NULL;
CF_SOA_FIELDS(CF_SOA_DECL)
#undef CF_SOA_DECL
__constant__ char* g_cf_state_base = NULL;
__constant__ int g_cf_n = 0;

// cf_slot is defined after CraftaxWorldState (needs its sizeof); any pointer
// into an env's state block (including interior pointers) maps to that env.
#define CF(f, s) (g_cf_##f[(size_t)cf_slot(s)])
#define CF2(f, i, s) \
    (g_cf_##f[(size_t)(i) * (size_t)g_cf_n + (size_t)cf_slot(s)])

// Mob SoA: class c (0 melee, 1 passive, 2 ranged, 3 mob proj, 4 player
// proj), level l (0..8), slot i (0..2; ranged pads slot 2, never accessed).
#define CFM(c, l, i) ((((c) * 9) + (l)) * 3 + (i))
#define MOB_POS(c, l, i, a, s) CF2(mob_position, CFM(c, l, i) * 2 + (a), s)
#define MOB_HP(c, l, i, s) CF2(mob_health, CFM(c, l, i), s)
#define MOB_MASK(c, l, i, s) CF2(mob_mask, CFM(c, l, i), s)
#define MOB_CD(c, l, i, s) CF2(mob_attack_cooldown, CFM(c, l, i), s)
#define MOB_TYPE(c, l, i, s) CF2(mob_type_id, CFM(c, l, i), s)
// Row bitmaps: 9 levels x 48 rows of 48-bit column masks.
#define CF_BITS(f, l, r, s) CF2(f, (l) * CRAFTAX_MAP_SIZE + (r), s)

typedef struct CraftaxOverworldFloor {
    uint8_t map[CRAFTAX_OVERWORLD_SIZE][CRAFTAX_OVERWORLD_SIZE];
    uint8_t item_map[CRAFTAX_OVERWORLD_SIZE][CRAFTAX_OVERWORLD_SIZE];
    uint8_t light_map[CRAFTAX_OVERWORLD_SIZE][CRAFTAX_OVERWORLD_SIZE];
    int32_t ladder_down[2];
    int32_t ladder_up[2];
} CraftaxOverworldFloor;

typedef struct CraftaxWGInventory {
    int32_t wood;
    int32_t stone;
    int32_t coal;
    int32_t iron;
    int32_t diamond;
    int32_t sapling;
    int32_t pickaxe;
    int32_t sword;
    int32_t bow;
    int32_t arrows;
    int32_t armour[4];
    int32_t torches;
    int32_t ruby;
    int32_t sapphire;
    int32_t potions[6];
    int32_t books;
} CraftaxWGInventory;

typedef struct CraftaxWGMobs3 {
    int32_t position[CRAFTAX_WG_NUM_LEVELS][3][2];
    float health[CRAFTAX_WG_NUM_LEVELS][3];
    bool mask[CRAFTAX_WG_NUM_LEVELS][3];
    int32_t attack_cooldown[CRAFTAX_WG_NUM_LEVELS][3];
    int32_t type_id[CRAFTAX_WG_NUM_LEVELS][3];
} CraftaxWGMobs3;

typedef struct CraftaxWGMobs2 {
    int32_t position[CRAFTAX_WG_NUM_LEVELS][2][2];
    float health[CRAFTAX_WG_NUM_LEVELS][2];
    bool mask[CRAFTAX_WG_NUM_LEVELS][2];
    int32_t attack_cooldown[CRAFTAX_WG_NUM_LEVELS][2];
    int32_t type_id[CRAFTAX_WG_NUM_LEVELS][2];
} CraftaxWGMobs2;

typedef struct CraftaxWorldState {
    // === Hot data (accessed every step) ===








    int32_t potion_mapping[6];



    int32_t fractal_noise_angles[4];

    // === Medium-hot bitmaps ===

    // === Cold data (large maps) ===
    uint8_t map[CRAFTAX_WG_NUM_LEVELS][CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE];
    uint8_t item_map[CRAFTAX_WG_NUM_LEVELS][CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE];
    uint8_t light_map[CRAFTAX_WG_NUM_LEVELS][CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE];

    int32_t down_ladders[CRAFTAX_WG_NUM_LEVELS][2];
    int32_t up_ladders[CRAFTAX_WG_NUM_LEVELS][2];

    // Lazy floor generation. Bit L of lazy_floors_pending set means floor L
    // has not been generated yet; its worldgen key is in lazy_floor_keys[L].
    // Zero-initialized states (test fixtures, JAX mirrors) read as
    // "all floors generated", so laziness is opt-in per state.
    uint32_t lazy_floor_keys[CRAFTAX_WG_NUM_LEVELS][2];
} CraftaxWorldState;

static __device__ inline int cf_slot(const void* p) {
    return (int)(((const char*)p - g_cf_state_base)
                 / (ptrdiff_t)sizeof(CraftaxWorldState));
}

// Reset-path equivalent of the old memset(state, 0, ...) for the SoA fields.
static __device__ inline void cf_soa_zero_env(const void* s) {
#define CF_SOA_ZERO(f, t, k) \
    for (int _i = 0; _i < (k); _i++) CF2(f, _i, s) = (t)0;
    CF_SOA_FIELDS(CF_SOA_ZERO)
#undef CF_SOA_ZERO
}

// Warp-distributed variant for the warp-cooperative reset: entry j of every
// field is always written by lane j % 32, so later same-stride writes from
// the same warp (mob health init, projectile directions) need no sync.
static __device__ inline void cf_soa_zero_env_warp(const void* s, unsigned lane) {
#define CF_SOA_ZERO_W(f, t, k) \
    for (int _i = (int)lane; _i < (k); _i += 32) CF2(f, _i, s) = (t)0;
    CF_SOA_FIELDS(CF_SOA_ZERO_W)
#undef CF_SOA_ZERO_W
}

// Per-entry level for each level-indexed SoA field (flat index j).
#define CF_LVL_mob_position(j) (((j) / 6) % 9)
#define CF_LVL_mob_health(j) (((j) / 3) % 9)
#define CF_LVL_mob_mask(j) (((j) / 3) % 9)
#define CF_LVL_mob_attack_cooldown(j) (((j) / 3) % 9)
#define CF_LVL_mob_type_id(j) (((j) / 3) % 9)
#define CF_LVL_mob_bits(j) ((j) / 48)
#define CF_LVL_spawn_all_bits(j) ((j) / 48)
#define CF_LVL_spawn_grave_bits(j) ((j) / 48)
#define CF_LVL_spawn_water_bits(j) ((j) / 48)
#define CF_LVL_mob_projectile_directions(j) ((j) / 6)
#define CF_LVL_player_projectile_directions(j) ((j) / 6)
#define CF_LVL_chests_opened(j) (j)
#define CF_LVL_monsters_killed(j) (j)

// Lazy variant: clears flat fields fully but level-indexed fields only for
// levels in gen_mask (bit L = level L was generated and must be re-cleared).
// Never-generated levels provably still hold their post-reset values.
template <int NT>
static __device__ inline void cf_soa_zero_env_warp_lazy(
    const void* s, unsigned lane, uint32_t gen_mask
) {
#define CF_SOA_ZERO_WF(f, t, k) \
    for (int _j = (int)lane; _j < (k); _j += NT) CF2(f, _j, s) = (t)0;
#define CF_SOA_ZERO_WL(f, t, k) \
    for (int _j = (int)lane; _j < (k); _j += NT) { \
        if (((gen_mask >> CF_LVL_##f(_j)) & 1u) == 0u) continue; \
        CF2(f, _j, s) = (t)0; \
    }
    CF_SOA_FIELDS_FLAT(CF_SOA_ZERO_WF)
    CF_SOA_FIELDS_LEVEL(CF_SOA_ZERO_WL)
#undef CF_SOA_ZERO_WF
#undef CF_SOA_ZERO_WL
}

// Warp-distributed mirror of the five craftax_init_empty_mobs* calls:
// health = 1.0f for every (class, level, slot) except the padded ranged
// slot 2, which the scalar path never writes (stays 0 from the zero pass).
// Same gen_mask predicate: skipped levels already hold health = 1.0f.
template <int NT>
static __device__ inline void cf_init_empty_mobs_warp(
    void* s, unsigned lane, uint32_t gen_mask
) {
    for (int j = (int)lane; j < 135; j += NT) {
        int c = j / 27;
        int r = j % 27;
        if (c == 2 && r % 3 == 2) continue;
        if (((gen_mask >> (r / 3)) & 1u) == 0u) continue;
        CF2(mob_health, j, s) = 1.0f;
    }
}


typedef struct CraftaxSmoothGenConfig {
    int32_t default_block;
    int32_t sea_block;
    int32_t coast_block;
    int32_t mountain_block;
    int32_t path_block;
    int32_t inner_mountain_block;
    int32_t ore_requirement_blocks[5];
    int32_t ores[5];
    float ore_chances[5];
    int32_t tree_requirement_block;
    int32_t tree;
    int32_t lava;
    int32_t player_spawn;
    int32_t valid_ladder;
    bool ladder_up;
    bool ladder_down;
    float player_proximity_map_water_strength;
    float player_proximity_map_water_max;
    float player_proximity_map_mountain_strength;
    float player_proximity_map_mountain_max;
    float default_light;
    float water_threshold;
    float sand_threshold;
    float tree_threshold_uniform;
    float tree_threshold_perlin;
} CraftaxSmoothGenConfig;

typedef struct CraftaxDungeonConfig {
    int32_t special_block;
    int32_t fountain_block;
    int32_t rare_path_replacement_block;
} CraftaxDungeonConfig;

static __device__ const CraftaxSmoothGenConfig CRAFTAX_SMOOTHGEN_CONFIGS[6] = {
    {
        CRAFTAX_WG_BLOCK_GRASS,
        CRAFTAX_WG_BLOCK_WATER,
        CRAFTAX_WG_BLOCK_SAND,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_PATH,
        {CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE},
        {CRAFTAX_WG_BLOCK_COAL, CRAFTAX_WG_BLOCK_IRON, CRAFTAX_WG_BLOCK_DIAMOND, CRAFTAX_WG_BLOCK_OUT_OF_BOUNDS, CRAFTAX_WG_BLOCK_OUT_OF_BOUNDS},
        {0.03f, 0.02f, 0.001f, 0.0f, 0.0f},
        CRAFTAX_WG_BLOCK_GRASS,
        CRAFTAX_WG_BLOCK_TREE,
        CRAFTAX_WG_BLOCK_LAVA,
        CRAFTAX_WG_BLOCK_GRASS,
        CRAFTAX_WG_BLOCK_PATH,
        false,
        true,
        5.0f,
        1.0f,
        5.0f,
        1.0f,
        1.0f,
        0.7f,
        0.6f,
        0.8f,
        0.5f,
    },
    {
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_WATER,
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        {CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE},
        {CRAFTAX_WG_BLOCK_COAL, CRAFTAX_WG_BLOCK_IRON, CRAFTAX_WG_BLOCK_DIAMOND, CRAFTAX_WG_BLOCK_SAPPHIRE, CRAFTAX_WG_BLOCK_RUBY},
        {0.04f, 0.02f, 0.005f, 0.0025f, 0.0025f},
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_STALAGMITE,
        CRAFTAX_WG_BLOCK_LAVA,
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_PATH,
        true,
        true,
        5.0f,
        1.0f,
        17.0f,
        1.5f,
        0.0f,
        0.7f,
        0.6f,
        0.8f,
        0.5f,
    },
    {
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_WATER,
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        {CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE},
        {CRAFTAX_WG_BLOCK_COAL, CRAFTAX_WG_BLOCK_IRON, CRAFTAX_WG_BLOCK_DIAMOND, CRAFTAX_WG_BLOCK_SAPPHIRE, CRAFTAX_WG_BLOCK_RUBY},
        {0.04f, 0.03f, 0.01f, 0.01f, 0.01f},
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_STALAGMITE,
        CRAFTAX_WG_BLOCK_LAVA,
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_PATH,
        true,
        true,
        5.0f,
        1.0f,
        17.0f,
        1.5f,
        0.0f,
        0.7f,
        0.6f,
        0.8f,
        0.5f,
    },
    {
        CRAFTAX_WG_BLOCK_FIRE_GRASS,
        CRAFTAX_WG_BLOCK_LAVA,
        CRAFTAX_WG_BLOCK_SAND,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        {CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE},
        {CRAFTAX_WG_BLOCK_COAL, CRAFTAX_WG_BLOCK_IRON, CRAFTAX_WG_BLOCK_DIAMOND, CRAFTAX_WG_BLOCK_SAPPHIRE, CRAFTAX_WG_BLOCK_RUBY},
        {0.05f, 0.0f, 0.0f, 0.0f, 0.025f},
        CRAFTAX_WG_BLOCK_FIRE_GRASS,
        CRAFTAX_WG_BLOCK_FIRE_TREE,
        CRAFTAX_WG_BLOCK_LAVA,
        CRAFTAX_WG_BLOCK_FIRE_GRASS,
        CRAFTAX_WG_BLOCK_FIRE_GRASS,
        true,
        true,
        5.0f,
        1.0f,
        5.0f,
        1.0f,
        1.0f,
        0.5f,
        0.6f,
        0.8f,
        0.5f,
    },
    {
        CRAFTAX_WG_BLOCK_ICE_GRASS,
        CRAFTAX_WG_BLOCK_WATER,
        CRAFTAX_WG_BLOCK_ICE_GRASS,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        CRAFTAX_WG_BLOCK_STONE,
        {CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE, CRAFTAX_WG_BLOCK_STONE},
        {CRAFTAX_WG_BLOCK_COAL, CRAFTAX_WG_BLOCK_IRON, CRAFTAX_WG_BLOCK_DIAMOND, CRAFTAX_WG_BLOCK_SAPPHIRE, CRAFTAX_WG_BLOCK_RUBY},
        {0.0f, 0.0f, 0.005f, 0.02f, 0.0f},
        CRAFTAX_WG_BLOCK_ICE_GRASS,
        CRAFTAX_WG_BLOCK_ICE_SHRUB,
        CRAFTAX_WG_BLOCK_WATER,
        CRAFTAX_WG_BLOCK_ICE_GRASS,
        CRAFTAX_WG_BLOCK_ICE_GRASS,
        true,
        true,
        5.0f,
        1.0f,
        17.0f,
        1.5f,
        0.0f,
        0.5f,
        0.6f,
        0.4f,
        0.5f,
    },
    {
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_WALL,
        CRAFTAX_WG_BLOCK_WALL,
        CRAFTAX_WG_BLOCK_WALL,
        {CRAFTAX_WG_BLOCK_WALL, CRAFTAX_WG_BLOCK_GRAVE, CRAFTAX_WG_BLOCK_GRAVE, CRAFTAX_WG_BLOCK_WALL, CRAFTAX_WG_BLOCK_WALL},
        {CRAFTAX_WG_BLOCK_WALL_MOSS, CRAFTAX_WG_BLOCK_GRAVE2, CRAFTAX_WG_BLOCK_GRAVE3, CRAFTAX_WG_BLOCK_SAPPHIRE, CRAFTAX_WG_BLOCK_RUBY},
        {0.1f, 0.333f, 0.5f, 0.0f, 0.0f},
        CRAFTAX_WG_BLOCK_PATH,
        CRAFTAX_WG_BLOCK_GRAVE,
        CRAFTAX_WG_BLOCK_WALL,
        CRAFTAX_WG_BLOCK_NECROMANCER,
        CRAFTAX_WG_BLOCK_PATH,
        false,
        false,
        5.0f,
        1.0f,
        10.0f,
        10.0f,
        0.0f,
        0.7f,
        0.6f,
        0.95f,
        -1.0f,
    },
};

static __device__ const CraftaxDungeonConfig CRAFTAX_DUNGEON_CONFIGS[3] = {
    {CRAFTAX_WG_BLOCK_PATH, CRAFTAX_WG_BLOCK_FOUNTAIN, CRAFTAX_WG_BLOCK_PATH},
    {CRAFTAX_WG_BLOCK_ENCHANTMENT_TABLE_ICE, CRAFTAX_WG_BLOCK_WATER, CRAFTAX_WG_BLOCK_WATER},
    {CRAFTAX_WG_BLOCK_ENCHANTMENT_TABLE_FIRE, CRAFTAX_WG_BLOCK_FOUNTAIN, CRAFTAX_WG_BLOCK_PATH},
};

static __device__ inline float craftax_wg_clampf(float value, float low, float high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static __device__ inline int craftax_wg_clampi(int value, int low, int high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static __device__ inline size_t craftax_wg_index(int row, int col) {
    return (size_t)row * (size_t)CRAFTAX_WG_MAP_SIZE + (size_t)col;
}

static __device__ inline void craftax_threefry_split3(
    CraftaxThreefryKey key,
    CraftaxThreefryKey* first,
    CraftaxThreefryKey* second,
    CraftaxThreefryKey* third
) {
    CraftaxThreefryKey keys[3];
    craftax_threefry_split_n(key, keys, 3);
    *first = keys[0];
    *second = keys[1];
    *third = keys[2];
}

static __device__ inline CraftaxThreefryKey craftax_worldgen_key_from_seed(uint32_t seed) {
    CraftaxThreefryKey key = craftax_prng_key(seed);
    CraftaxThreefryKey carry;
    CraftaxThreefryKey reset_key;
    craftax_threefry_split(key, &carry, &reset_key);

    CraftaxThreefryKey reset_carry;
    CraftaxThreefryKey world_key;
    craftax_threefry_split(reset_key, &reset_carry, &world_key);
    return world_key;
}

static __device__ inline uint32_t craftax_randint_u32_at(
    CraftaxThreefryKey key,
    uint64_t index,
    uint32_t minval,
    uint32_t maxval
) {
    uint32_t span = maxval > minval ? maxval - minval : 1u;
    // Fast path for power-of-2 spans: just mask
    if ((span & (span - 1)) == 0) {
        uint32_t bits = craftax_threefry_uniform_u32_at(key, index);
        return minval + (bits & (span - 1));
    }
    // General path: use top-32 of hash, scale to span
    uint64_t h = craftax_fast_hash64(key, index);
    return minval + (uint32_t)(((h >> 32) * (uint64_t)span) >> 32);
}

static __device__ inline int32_t craftax_randint_i32_at(
    CraftaxThreefryKey key,
    uint64_t index,
    int32_t minval,
    int32_t maxval
) {
    return (int32_t)craftax_randint_u32_at(
        key,
        index,
        (uint32_t)minval,
        (uint32_t)maxval
    );
}

static __device__ inline int craftax_choice_bool_flat(
    CraftaxThreefryKey key,
    const bool* valid,
    int count
) {
    int valid_count = 0;
    int last_valid = 0;
    for (int i = 0; i < count; i++) {
        if (valid[i]) {
            valid_count++;
            last_valid = i;
        }
    }
    if (valid_count == 0) {
        return 0;
    }

    float draw = (float)valid_count * (1.0f - craftax_threefry_uniform_f32(key));
    float cumulative = 0.0f;
    for (int i = 0; i < count; i++) {
        if (valid[i]) {
            cumulative += 1.0f;
        }
        if (cumulative >= draw) {
            return i;
        }
    }
    return last_valid;
}

static __device__ inline float craftax_torch_light_value(int row, int col, float default_light) {
    float dr = (float)(row - 4);
    float dc = (float)(col - 4);
    float distance = sqrtf(dr * dr + dc * dc);
    float torch = craftax_wg_clampf(1.0f - distance * (1.0f / 5.0f), 0.0f, 1.0f);
    return torch * (1.0f - default_light) + default_light;
}

static __device__ inline void craftax_apply_ladder_light(
    uint8_t light_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    const int32_t ladder_up[2],
    float default_light
) {
    int start_row = ladder_up[0] - 4;
    int start_col = ladder_up[1] - 4;
    if (start_row < 0) {
        start_row += CRAFTAX_WG_MAP_SIZE;
    }
    if (start_col < 0) {
        start_col += CRAFTAX_WG_MAP_SIZE;
    }
    start_row = craftax_wg_clampi(start_row, 0, CRAFTAX_WG_MAP_SIZE - 9);
    start_col = craftax_wg_clampi(start_col, 0, CRAFTAX_WG_MAP_SIZE - 9);
    for (int row = 0; row < 9; row++) {
        for (int col = 0; col < 9; col++) {
            light_map[start_row + row][start_col + col] =
                (uint8_t)(craftax_torch_light_value(row, col, default_light) * 255.0f);
        }
    }
}

static __device__ inline void craftax_add_lava_light(
    uint8_t light_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    const bool lava_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    bool lava_emits_light
) {
    if (!lava_emits_light) {
        return;
    }

    static const float kernel[3][3] = {
        {0.2f, 0.7f, 0.2f},
        {0.7f, 1.0f, 0.7f},
        {0.2f, 0.7f, 0.2f},
    };

    for (int row = 0; row < CRAFTAX_WG_MAP_SIZE; row++) {
        for (int col = 0; col < CRAFTAX_WG_MAP_SIZE; col++) {
            float add = 0.0f;
            for (int kr = 0; kr < 3; kr++) {
                int src_row = row + kr - 1;
                if (src_row < 0 || src_row >= CRAFTAX_WG_MAP_SIZE) {
                    continue;
                }
                for (int kc = 0; kc < 3; kc++) {
                    int src_col = col + kc - 1;
                    if (src_col < 0 || src_col >= CRAFTAX_WG_MAP_SIZE) {
                        continue;
                    }
                    add += lava_map[src_row][src_col] ? kernel[kr][kc] : 0.0f;
                }
            }
            float new_light = craftax_wg_clampf(light_map[row][col] * (1.0f / 255.0f) + add, 0.0f, 1.0f);
            light_map[row][col] = (uint8_t)(new_light * 255.0f);
        }
    }
}

static __device__ inline int craftax_smooth_config_index_for_floor(int floor_idx) {
    switch (floor_idx) {
        case 0:
            return 0;
        case 2:
            return 1;
        case 5:
            return 2;
        case 6:
            return 3;
        case 7:
            return 4;
        case 8:
            return 5;
        default:
            return -1;
    }
}

static __device__ inline int craftax_dungeon_config_index_for_floor(int floor_idx) {
    switch (floor_idx) {
        case 1:
            return 0;
        case 3:
            return 1;
        case 4:
            return 2;
        default:
            return -1;
    }
}

// ============================================================
// [cuda port] Per-thread worldgen scratch arena. The scalar generators used
// to keep their noise fields / padded maps in ~60KB of thread-local stack,
// which forces a huge cudaLimitStackSize and throttles k_step residency.
// Instead each generator grabs a global-memory slot indexed by the calling
// thread's flat id (every kernel that reaches these generators launches at
// most one thread per env, so num_envs slots suffice). The warp-cooperative
// reset path has its own shared-memory scratch and never touches this.
// ============================================================
typedef struct CraftaxWorldgenScratch {
    union {
        struct {
            float water[CRAFTAX_WG_MAP_CELLS];
            float mountain[CRAFTAX_WG_MAP_CELLS];
            float path_x[CRAFTAX_WG_MAP_CELLS];
            float tree_noise[CRAFTAX_WG_MAP_CELLS];
            bool lava_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE];
            bool valid[CRAFTAX_WG_MAP_CELLS];
        } smooth;
        struct {
            uint8_t padded_map[68][68];
            uint8_t padded_item_map[68][68];
            bool adjacent_path[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE];
            bool valid[CRAFTAX_WG_MAP_CELLS];
        } dungeon;
    } u;
} CraftaxWorldgenScratch;

__device__ CraftaxWorldgenScratch* g_craftax_wg_scratch = NULL;

// Scratch slot keyed by the env that owns `p` (any pointer into the state
// arena, e.g. the floor map being generated): one slot per env regardless
// of which thread/kernel runs the generation. The previous tid-based slot
// assumed every worldgen-capable grid has exactly num_envs threads at
// tid==env; env-keyed slots keep the arena in bounds for any launch shape
// and can never alias two envs onto one slot. Scratch is fully rewritten
// before use, so slot choice is invisible in results.
static __device__ inline CraftaxWorldgenScratch* craftax_wg_scratch_slot(
    const void* p
) {
    return &g_craftax_wg_scratch[(size_t)cf_slot(p)];
}

static __device__ inline void craftax_smoothworld_classify_scalar(
    const CraftaxSmoothGenConfig* config,
    CraftaxThreefryKey tree_uniform_key,
    float* water,
    float* mountain,
    const float* path_x,
    const float* tree_noise,
    uint8_t map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t item_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t light_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE]
) {
    const int size = CRAFTAX_WG_MAP_SIZE;
    const int player_row = CRAFTAX_WG_MAP_SIZE / 2;
    const int player_col = CRAFTAX_WG_MAP_SIZE / 2;
    for (int row = 0; row < size; row++) {
        int dr = row > player_row ? row - player_row : player_row - row;
        for (int col = 0; col < size; col++) {
            int dc = col > player_col ? col - player_col : player_col - col;
            float distance = sqrtf((float)(dr * dr + dc * dc));
            float proximity_water = craftax_wg_clampf(
                distance / config->player_proximity_map_water_strength,
                0.0f,
                config->player_proximity_map_water_max
            );
            float proximity_mountain = craftax_wg_clampf(
                distance / config->player_proximity_map_mountain_strength,
                0.0f,
                config->player_proximity_map_mountain_max
            );
            size_t idx = craftax_wg_index(row, col);

            water[idx] = water[idx] + proximity_water - 1.0f;
            int32_t block = water[idx] > config->water_threshold
                ? config->sea_block
                : config->default_block;
            bool sand = water[idx] > config->sand_threshold && block != config->sea_block;
            if (sand) {
                block = config->coast_block;
            }

            mountain[idx] = mountain[idx] + 0.05f + proximity_mountain - 1.0f;
            if (mountain[idx] > 0.7f) {
                block = config->mountain_block;
            }

            bool path = mountain[idx] > 0.7f && path_x[idx] > 0.8f;
            if (path) {
                block = config->path_block;
            }

            float path_y = path_x[craftax_wg_index(col, row)];
            path = mountain[idx] > 0.7f && path_y > 0.8f;
            if (path) {
                block = config->path_block;
            }

            bool cave = mountain[idx] > 0.85f && water[idx] > 0.4f;
            if (cave) {
                block = config->inner_mountain_block;
            }

            float tree_draw = craftax_threefry_uniform_f32_at(tree_uniform_key, idx);
            bool tree = tree_noise[idx] > config->tree_threshold_perlin
                && tree_draw > config->tree_threshold_uniform;
            if (tree && block == config->tree_requirement_block) {
                block = config->tree;
            }

            map[row][col] = (uint8_t)block;
            item_map[row][col] = CRAFTAX_WG_ITEM_NONE;
            light_map[row][col] = (uint8_t)(config->default_light * 255.0f);
        }
    }
}

static __device__ inline void craftax_smoothworld_ore_scalar(
    const CraftaxSmoothGenConfig* config,
    int ore_index,
    CraftaxThreefryKey ore_key,
    uint8_t map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE]
) {
    const int size = CRAFTAX_WG_MAP_SIZE;
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            size_t idx = craftax_wg_index(row, col);
            bool is_ore = map[row][col] == config->ore_requirement_blocks[ore_index]
                && craftax_threefry_uniform_f32_at(ore_key, idx) < config->ore_chances[ore_index];
            if (is_ore) {
                map[row][col] = (uint8_t)config->ores[ore_index];
            }
        }
    }
}

static __device__ inline void craftax_smoothworld_lava_scalar(
    const CraftaxSmoothGenConfig* config,
    const float* mountain,
    const float* tree_noise,
    uint8_t map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    bool lava_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE]
) {
    const int size = CRAFTAX_WG_MAP_SIZE;
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            size_t idx = craftax_wg_index(row, col);
            lava_map[row][col] = mountain[idx] > 0.85f && tree_noise[idx] > 0.7f;
            if (lava_map[row][col]) {
                map[row][col] = (uint8_t)config->lava;
            }
        }
    }
}

static __device__ inline void craftax_generate_smoothworld_config(
    CraftaxThreefryKey rng,
    int config_idx,
    uint8_t map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t item_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t light_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    int32_t ladder_down[2],
    int32_t ladder_up[2]
) {
    const CraftaxSmoothGenConfig* config = &CRAFTAX_SMOOTHGEN_CONFIGS[config_idx];
    const int size = CRAFTAX_WG_MAP_SIZE;
    const int player_row = CRAFTAX_WG_MAP_SIZE / 2;
    const int player_col = CRAFTAX_WG_MAP_SIZE / 2;
    const size_t cells = CRAFTAX_WG_MAP_CELLS;

    CraftaxThreefryKey subkey;
    CraftaxWorldgenScratch* scratch = craftax_wg_scratch_slot(map);
    float* water = scratch->u.smooth.water;
    float* mountain = scratch->u.smooth.mountain;
    float* path_x = scratch->u.smooth.path_x;
    float* tree_noise = scratch->u.smooth.tree_noise;
    bool (*lava_map)[CRAFTAX_WG_MAP_SIZE] = scratch->u.smooth.lava_map;

    craftax_threefry_split(rng, &rng, &subkey);
    craftax_generate_fractal_noise_2d(subkey, size, size, 3, 3, 1, 0.5f, 2, NULL, water);

    craftax_threefry_split(rng, &rng, &subkey);
    (void)subkey;

    craftax_threefry_split(rng, &rng, &subkey);
    craftax_generate_fractal_noise_2d(subkey, size, size, 3, 3, 1, 0.5f, 2, NULL, mountain);

    craftax_threefry_split(rng, &rng, &subkey);
    craftax_generate_fractal_noise_2d(subkey, size, size, 6, 24, 1, 0.5f, 2, NULL, path_x);

    craftax_threefry_split(rng, &rng, &subkey);
    (void)subkey;

    craftax_threefry_split(rng, &rng, &subkey);
    CraftaxThreefryKey tree_uniform_key = rng;
    craftax_generate_fractal_noise_2d(subkey, size, size, 12, 12, 1, 0.5f, 2, NULL, tree_noise);

    craftax_smoothworld_classify_scalar(
        config, tree_uniform_key, water, mountain, path_x, tree_noise,
        map, item_map, light_map);
    (void)player_col;

    CraftaxThreefryKey ore_rng;
    craftax_threefry_split(rng, &rng, &ore_rng);
    for (int ore_index = 0; ore_index < 5; ore_index++) {
        CraftaxThreefryKey ore_key;
        craftax_threefry_split(ore_rng, &ore_rng, &ore_key);
        craftax_smoothworld_ore_scalar(config, ore_index, ore_key, map);
    }

    craftax_smoothworld_lava_scalar(config, mountain, tree_noise, map, lava_map);

    craftax_threefry_split(rng, &rng, &subkey);
    bool* valid_diamond = scratch->u.smooth.valid;
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            valid_diamond[craftax_wg_index(row, col)] = map[row][col] == CRAFTAX_WG_BLOCK_STONE;
        }
    }
    int diamond_index = craftax_choice_bool_flat(subkey, valid_diamond, (int)cells);
    map[diamond_index / size][diamond_index % size] = (uint8_t)CRAFTAX_WG_BLOCK_STONE;

    map[player_row][player_col] = (uint8_t)config->player_spawn;

    bool* valid_ladder = scratch->u.smooth.valid;
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            valid_ladder[craftax_wg_index(row, col)] = map[row][col] == config->valid_ladder;
        }
    }

    craftax_threefry_split(rng, &rng, &subkey);
    int ladder_down_index = craftax_choice_bool_flat(subkey, valid_ladder, (int)cells);
    ladder_down[0] = ladder_down_index / size;
    ladder_down[1] = ladder_down_index % size;
    if (config->ladder_down) {
        item_map[ladder_down[0]][ladder_down[1]] = CRAFTAX_WG_ITEM_LADDER_DOWN;
    }

    craftax_threefry_split(rng, &rng, &subkey);
    int ladder_up_index = craftax_choice_bool_flat(subkey, valid_ladder, (int)cells);
    ladder_up[0] = ladder_up_index / size;
    ladder_up[1] = ladder_up_index % size;

    craftax_apply_ladder_light(light_map, ladder_up, config->default_light);
    craftax_add_lava_light(light_map, lava_map, config->lava == CRAFTAX_WG_BLOCK_LAVA);

    if (config->ladder_up) {
        item_map[ladder_up[0]][ladder_up[1]] = CRAFTAX_WG_ITEM_LADDER_UP;
    }
}

static __device__ inline void craftax_generate_dungeon_config(
    CraftaxThreefryKey rng,
    int config_idx,
    uint8_t map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t item_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t light_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    int32_t ladder_down[2],
    int32_t ladder_up[2]
) {
    const CraftaxDungeonConfig* config = &CRAFTAX_DUNGEON_CONFIGS[config_idx];
    const int chunk_size = 16;
    const int world_chunk_height = CRAFTAX_WG_MAP_SIZE / chunk_size;
    const int num_rooms = 8;
    const int min_room_size = 5;
    const int max_room_size = 10;
    const int padded_size = CRAFTAX_WG_MAP_SIZE + 2 * max_room_size;

    CraftaxWorldgenScratch* scratch = craftax_wg_scratch_slot(map);
    uint8_t (*padded_map)[68] = scratch->u.dungeon.padded_map;
    uint8_t (*padded_item_map)[68] = scratch->u.dungeon.padded_item_map;
    bool room_occupancy_chunks[9];
    int32_t room_sizes[8][2];
    int32_t room_positions[8][2];

    for (int row = 0; row < padded_size; row++) {
        for (int col = 0; col < padded_size; col++) {
            bool inner = row >= max_room_size
                && row < max_room_size + CRAFTAX_WG_MAP_SIZE
                && col >= max_room_size
                && col < max_room_size + CRAFTAX_WG_MAP_SIZE;
            padded_map[row][col] = inner ? CRAFTAX_WG_BLOCK_WALL : 0;
            padded_item_map[row][col] = CRAFTAX_WG_ITEM_NONE;
        }
    }
    for (int i = 0; i < 9; i++) {
        room_occupancy_chunks[i] = true;
    }

    CraftaxThreefryKey room_scan_ignored_key;
    CraftaxThreefryKey room_size_key;
    craftax_threefry_split3(rng, &rng, &room_scan_ignored_key, &room_size_key);
    (void)room_scan_ignored_key;
    for (int room = 0; room < num_rooms; room++) {
        room_sizes[room][0] = craftax_randint_i32_at(room_size_key, (uint64_t)room * 2u, min_room_size, max_room_size);
        room_sizes[room][1] = craftax_randint_i32_at(room_size_key, (uint64_t)room * 2u + 1u, min_room_size, max_room_size);
    }

    CraftaxThreefryKey room_rng;
    craftax_threefry_split(rng, &rng, &room_rng);

    for (int room_index = 0; room_index < num_rooms; room_index++) {
        CraftaxThreefryKey choice_key;
        craftax_threefry_split(room_rng, &room_rng, &choice_key);
        int room_chunk = craftax_choice_bool_flat(choice_key, room_occupancy_chunks, 9);
        room_occupancy_chunks[room_chunk] = false;

        int room_row = (room_chunk % world_chunk_height) * chunk_size + max_room_size;
        int room_col = (room_chunk / world_chunk_height) * chunk_size + max_room_size;
        CraftaxThreefryKey position_key;
        craftax_threefry_split(room_rng, &room_rng, &position_key);
        room_row += craftax_randint_i32_at(position_key, 0, 0, chunk_size - min_room_size);
        room_col += craftax_randint_i32_at(position_key, 1, 0, chunk_size - min_room_size);
        room_positions[room_index][0] = room_row;
        room_positions[room_index][1] = room_col;

        for (int row = 0; row < max_room_size; row++) {
            for (int col = 0; col < max_room_size; col++) {
                if (row < room_sizes[room_index][0] && col < room_sizes[room_index][1]) {
                    padded_map[room_row + row][room_col + col] = CRAFTAX_WG_BLOCK_PATH;
                }
            }
        }

        padded_item_map[room_row][room_col] = CRAFTAX_WG_ITEM_TORCH;
        padded_item_map[room_row + room_sizes[room_index][0] - 1][room_col] = CRAFTAX_WG_ITEM_TORCH;
        padded_item_map[room_row][room_col + room_sizes[room_index][1] - 1] = CRAFTAX_WG_ITEM_TORCH;
        padded_item_map[room_row + room_sizes[room_index][0] - 1][room_col + room_sizes[room_index][1] - 1] = CRAFTAX_WG_ITEM_TORCH;

        CraftaxThreefryKey chest_key;
        craftax_threefry_split(room_rng, &room_rng, &chest_key);
        int chest_row = craftax_randint_i32_at(chest_key, 0, 1, room_sizes[room_index][0] - 1);
        int chest_col = craftax_randint_i32_at(chest_key, 1, 1, room_sizes[room_index][1] - 1);
        padded_map[room_row + chest_row][room_col + chest_col] = CRAFTAX_WG_BLOCK_CHEST;

        CraftaxThreefryKey fountain_key;
        CraftaxThreefryKey fountain_uniform_key;
        craftax_threefry_split3(room_rng, &room_rng, &fountain_key, &fountain_uniform_key);
        int fountain_row = craftax_randint_i32_at(fountain_key, 0, 1, room_sizes[room_index][0] - 1);
        int fountain_col = craftax_randint_i32_at(fountain_key, 1, 1, room_sizes[room_index][1] - 1);
        bool room_has_fountain = craftax_threefry_uniform_f32(fountain_uniform_key) > 0.5f;
        if (room_has_fountain) {
            padded_map[room_row + fountain_row][room_col + fountain_col] = config->fountain_block;
        }
    }

    CraftaxThreefryKey path_rng;
    craftax_threefry_split(rng, &rng, &path_rng);
    bool included_rooms_mask[8] = {false, false, false, false, false, false, false, true};

    for (int path_index = 0; path_index < num_rooms; path_index++) {
        int source_row = room_positions[path_index][0];
        int source_col = room_positions[path_index][1];

        CraftaxThreefryKey sink_key;
        craftax_threefry_split(path_rng, &path_rng, &sink_key);
        int sink_index = craftax_choice_bool_flat(sink_key, included_rooms_mask, num_rooms);
        int sink_row = room_positions[sink_index][0];
        int sink_col = room_positions[sink_index][1];

        int horizontal_distance = sink_col - source_col;
        int horizontal_sign = (horizontal_distance > 0) - (horizontal_distance < 0);
        if (horizontal_sign != 0) {
            int abs_distance = horizontal_distance > 0 ? horizontal_distance : -horizontal_distance;
            for (int col = 0; col < padded_size; col++) {
                int path_index_col = (col - source_col) * horizontal_sign;
                bool horizontal_mask = path_index_col >= 0
                    && path_index_col <= abs_distance
                    && padded_map[source_row][col] == CRAFTAX_WG_BLOCK_WALL;
                if (horizontal_mask) {
                    padded_map[source_row][col] = CRAFTAX_WG_BLOCK_PATH;
                }
            }
        }

        int vertical_distance = sink_row - source_row;
        int vertical_sign = (vertical_distance > 0) - (vertical_distance < 0);
        if (vertical_sign != 0) {
            int abs_distance = vertical_distance > 0 ? vertical_distance : -vertical_distance;
            for (int row = 0; row < padded_size; row++) {
                int path_index_row = (row - source_row) * vertical_sign;
                bool vertical_mask = path_index_row >= 0
                    && path_index_row <= abs_distance
                    && padded_map[row][sink_col] == CRAFTAX_WG_BLOCK_WALL;
                if (vertical_mask) {
                    padded_map[row][sink_col] = CRAFTAX_WG_BLOCK_PATH;
                }
            }
        }

        CraftaxThreefryKey unused_left;
        CraftaxThreefryKey next_path_rng;
        craftax_threefry_split(path_rng, &unused_left, &next_path_rng);
        path_rng = next_path_rng;
        included_rooms_mask[path_index] = true;
    }

    int special_row = room_positions[0][0] + 2;
    int special_col = room_positions[0][1] + 2;
    padded_map[special_row][special_col] = config->special_block;

    for (int row = 0; row < CRAFTAX_WG_MAP_SIZE; row++) {
        for (int col = 0; col < CRAFTAX_WG_MAP_SIZE; col++) {
            map[row][col] = padded_map[row + max_room_size][col + max_room_size];
            item_map[row][col] = padded_item_map[row + max_room_size][col + max_room_size];
        }
    }

    bool (*adjacent_path)[CRAFTAX_WG_MAP_SIZE] = scratch->u.dungeon.adjacent_path;
    for (int row = 0; row < CRAFTAX_WG_MAP_SIZE; row++) {
        for (int col = 0; col < CRAFTAX_WG_MAP_SIZE; col++) {
            bool adjacent = map[row][col] != CRAFTAX_WG_BLOCK_WALL;
            adjacent = adjacent || (row > 0 && map[row - 1][col] != CRAFTAX_WG_BLOCK_WALL);
            adjacent = adjacent || (row + 1 < CRAFTAX_WG_MAP_SIZE && map[row + 1][col] != CRAFTAX_WG_BLOCK_WALL);
            adjacent = adjacent || (col > 0 && map[row][col - 1] != CRAFTAX_WG_BLOCK_WALL);
            adjacent = adjacent || (col + 1 < CRAFTAX_WG_MAP_SIZE && map[row][col + 1] != CRAFTAX_WG_BLOCK_WALL);
            adjacent_path[row][col] = adjacent;
        }
    }

    CraftaxThreefryKey rare_key;
    craftax_threefry_split(rng, &rng, &rare_key);
    for (int row = 0; row < CRAFTAX_WG_MAP_SIZE; row++) {
        for (int col = 0; col < CRAFTAX_WG_MAP_SIZE; col++) {
            size_t idx = craftax_wg_index(row, col);
            bool rare = (1.0f - craftax_threefry_uniform_f32_at(rare_key, idx)) > 0.9f;
            int32_t wall_map = rare ? CRAFTAX_WG_BLOCK_WALL_MOSS : CRAFTAX_WG_BLOCK_WALL;
            bool rare_path = rare && map[row][col] == CRAFTAX_WG_BLOCK_PATH && item_map[row][col] == CRAFTAX_WG_ITEM_NONE;
            int32_t path_map = rare_path ? config->rare_path_replacement_block : map[row][col];
            bool is_wall_map = map[row][col] == CRAFTAX_WG_BLOCK_WALL && adjacent_path[row][col];
            bool is_darkness_map = !adjacent_path[row][col];

            if (is_darkness_map) {
                map[row][col] = CRAFTAX_WG_BLOCK_DARKNESS;
            } else if (is_wall_map) {
                map[row][col] = wall_map;
            } else {
                map[row][col] = path_map;
            }
            light_map[row][col] = 255;
        }
    }

    bool* valid_ladder = scratch->u.dungeon.valid;
    for (int row = 0; row < CRAFTAX_WG_MAP_SIZE; row++) {
        for (int col = 0; col < CRAFTAX_WG_MAP_SIZE; col++) {
            valid_ladder[craftax_wg_index(row, col)] = map[row][col] == CRAFTAX_WG_BLOCK_PATH;
        }
    }

    CraftaxThreefryKey ladder_down_key;
    craftax_threefry_split(rng, &rng, &ladder_down_key);
    int ladder_down_index = craftax_choice_bool_flat(ladder_down_key, valid_ladder, CRAFTAX_WG_MAP_CELLS);
    ladder_down[0] = ladder_down_index / CRAFTAX_WG_MAP_SIZE;
    ladder_down[1] = ladder_down_index % CRAFTAX_WG_MAP_SIZE;
    item_map[ladder_down[0]][ladder_down[1]] = CRAFTAX_WG_ITEM_LADDER_DOWN;

    CraftaxThreefryKey ladder_up_key;
    craftax_threefry_split(rng, &rng, &ladder_up_key);
    int ladder_up_index = craftax_choice_bool_flat(ladder_up_key, valid_ladder, CRAFTAX_WG_MAP_CELLS);
    ladder_up[0] = ladder_up_index / CRAFTAX_WG_MAP_SIZE;
    ladder_up[1] = ladder_up_index % CRAFTAX_WG_MAP_SIZE;
    item_map[ladder_up[0]][ladder_up[1]] = CRAFTAX_WG_ITEM_LADDER_UP;
}

static __device__ inline void craftax_generate_dungeon_floor(
    CraftaxThreefryKey seed_key,
    int floor_idx,
    uint8_t map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t item_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t light_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    int32_t ladder_down[2],
    int32_t ladder_up[2]
) {
    int config_idx = craftax_dungeon_config_index_for_floor(floor_idx);
    if (config_idx < 0) {
        memset(map, 0, CRAFTAX_WG_MAP_CELLS * sizeof(uint8_t));
        memset(item_map, 0, CRAFTAX_WG_MAP_CELLS * sizeof(uint8_t));
        memset(light_map, 0, CRAFTAX_WG_MAP_CELLS * sizeof(uint8_t));
        ladder_down[0] = 0;
        ladder_down[1] = 0;
        ladder_up[0] = 0;
        ladder_up[1] = 0;
        return;
    }
    craftax_generate_dungeon_config(
        seed_key,
        config_idx,
        map,
        item_map,
        light_map,
        ladder_down,
        ladder_up
    );
}

static __device__ inline void craftax_permutation_6(CraftaxThreefryKey key, int32_t out[6]) {
    CraftaxThreefryKey carry;
    CraftaxThreefryKey sort_key;
    craftax_threefry_split(key, &carry, &sort_key);
    (void)carry;

    uint32_t keys[6];
    for (int i = 0; i < 6; i++) {
        keys[i] = craftax_threefry_uniform_u32_at(sort_key, (uint64_t)i);
        out[i] = i;
    }

    for (int i = 1; i < 6; i++) {
        uint32_t key_value = keys[i];
        int32_t value = out[i];
        int j = i - 1;
        while (j >= 0 && keys[j] > key_value) {
            keys[j + 1] = keys[j];
            out[j + 1] = out[j];
            j--;
        }
        keys[j + 1] = key_value;
        out[j + 1] = value;
    }
}

// [cuda port] Light level is a pure function of timestep computed with
// host libm (cosf) + gcc -ffast-math powf(x,3)->x*x*x expansion in the C
// reference. Device libm rounds differently, so the host harness computes
// the whole table with host code and uploads it (see harness below).
__device__ float g_craftax_light_table[CRAFTAX_DEFAULT_MAX_TIMESTEPS_TABLE];

static __device__ inline float craftax_calculate_initial_light_level(void) {
    // reference: 1.0f - powf(fabsf(cosf(pi * 0.3f)), 3.0f) == table[0]
    return g_craftax_light_table[0];
}

static __device__ inline void craftax_init_empty_mobs3(void* mobs, int mc) {
    for (int level = 0; level < CRAFTAX_WG_NUM_LEVELS; level++) {
        for (int mob = 0; mob < 3; mob++) {
            MOB_HP(mc, level, mob, mobs) = 1.0f;
        }
    }
}

static __device__ inline void craftax_init_empty_mobs2(void* mobs, int mc) {
    for (int level = 0; level < CRAFTAX_WG_NUM_LEVELS; level++) {
        for (int mob = 0; mob < 2; mob++) {
            MOB_HP(mc, level, mob, mobs) = 1.0f;
        }
    }
}

// Per-level generator dispatch: floors {0,2,5,6,7,8} are smoothworld configs
// 0..5, floors {1,3,4} are dungeon configs 0..2 (must match the floor orders
// in craftax_generate_world_from_key_lazy below).
static __device__ inline void craftax_generate_floor_from_key(
    CraftaxThreefryKey key,
    int level,
    CraftaxWorldState* out
) {
    static const int8_t is_dungeon[CRAFTAX_WG_NUM_LEVELS] =
        {0, 1, 0, 1, 1, 0, 0, 0, 0};
    static const int8_t cfg_index[CRAFTAX_WG_NUM_LEVELS] =
        {0, 0, 1, 1, 2, 2, 3, 4, 5};
    if (is_dungeon[level]) {
        craftax_generate_dungeon_config(
            key, cfg_index[level],
            out->map[level], out->item_map[level], out->light_map[level],
            out->down_ladders[level], out->up_ladders[level]);
    } else {
        craftax_generate_smoothworld_config(
            key, cfg_index[level],
            out->map[level], out->item_map[level], out->light_map[level],
            out->down_ladders[level], out->up_ladders[level]);
    }
}

static __device__ inline void craftax_generate_world_from_key_lazy(
    CraftaxThreefryKey rng,
    CraftaxWorldState* out,
    bool lazy
) {
    memset(out, 0, sizeof(*out));
    cf_soa_zero_env(out);

    CraftaxThreefryKey smooth_split[7];
    craftax_threefry_split_n(rng, smooth_split, 7);
    rng = smooth_split[0];

    static const int smooth_floor_order[6] = {0, 2, 5, 6, 7, 8};
    for (int i = 0; i < 6; i++) {
        int level = smooth_floor_order[i];
        out->lazy_floor_keys[level][0] = smooth_split[i + 1].word[0];
        out->lazy_floor_keys[level][1] = smooth_split[i + 1].word[1];
        if (lazy && level != 0) continue;
        craftax_generate_smoothworld_config(
            smooth_split[i + 1],
            i,
            out->map[level],
            out->item_map[level],
            out->light_map[level],
            out->down_ladders[level],
            out->up_ladders[level]
        );
    }

    CraftaxThreefryKey dungeon_split[4];
    craftax_threefry_split_n(rng, dungeon_split, 4);
    rng = dungeon_split[0];

    static const int dungeon_floor_order[3] = {1, 3, 4};
    for (int i = 0; i < 3; i++) {
        int level = dungeon_floor_order[i];
        out->lazy_floor_keys[level][0] = dungeon_split[i + 1].word[0];
        out->lazy_floor_keys[level][1] = dungeon_split[i + 1].word[1];
        if (lazy) continue;
        craftax_generate_dungeon_config(
            dungeon_split[i + 1],
            i,
            out->map[level],
            out->item_map[level],
            out->light_map[level],
            out->down_ladders[level],
            out->up_ladders[level]
        );
    }

    CF(lazy_floors_pending, out) = lazy ? 0x1FEu : 0u;  // floors 1..8 deferred

    craftax_init_empty_mobs3(out, 0);
    craftax_init_empty_mobs3(out, 1);
    craftax_init_empty_mobs2(out, 2);
    craftax_init_empty_mobs3(out, 3);
    craftax_init_empty_mobs3(out, 4);
    for (int level = 0; level < CRAFTAX_WG_NUM_LEVELS; level++) {
        for (int projectile = 0; projectile < CRAFTAX_WG_MAX_MOB_PROJECTILES; projectile++) {
            CF2(mob_projectile_directions, (level) * 6 + (projectile) * 2 + (0), out) = 1;
            CF2(mob_projectile_directions, (level) * 6 + (projectile) * 2 + (1), out) = 1;
        }
        for (int projectile = 0; projectile < CRAFTAX_WG_MAX_PLAYER_PROJECTILES; projectile++) {
            CF2(player_projectile_directions, (level) * 6 + (projectile) * 2 + (0), out) = 1;
            CF2(player_projectile_directions, (level) * 6 + (projectile) * 2 + (1), out) = 1;
        }
    }

    CraftaxThreefryKey potion_key;
    craftax_threefry_split(rng, &rng, &potion_key);
    craftax_permutation_6(potion_key, out->potion_mapping);

    CraftaxThreefryKey state_key;
    craftax_threefry_split(rng, &rng, &state_key);
    (void)rng;
    CF2(state_rng, 0, out) = state_key.word[0];
    CF2(state_rng, 1, out) = state_key.word[1];

    CF2(monsters_killed, 0, out) = 10;
    CF2(player_position, 0, out) = CRAFTAX_WG_MAP_SIZE / 2;
    CF2(player_position, 1, out) = CRAFTAX_WG_MAP_SIZE / 2;
    CF(player_level, out) = 0;
    CF(player_direction, out) = CRAFTAX_WG_ACTION_UP;
    CF(player_health, out) = 9.0f;
    CF(player_food, out) = 9;
    CF(player_drink, out) = 9;
    CF(player_energy, out) = 9;
    CF(player_mana, out) = 9;
    CF(player_dexterity, out) = 1;
    CF(player_strength, out) = 1;
    CF(player_intelligence, out) = 1;
    CF(boss_timesteps_to_spawn_this_round, out) = CRAFTAX_WG_BOSS_FIGHT_SPAWN_TURNS;
    CF(light_level, out) = craftax_calculate_initial_light_level();
}

static __device__ inline int craftax_wg_jax_index(int32_t index, int32_t size) {
    if (index < 0) {
        index += size;
    }
    if (index < 0) {
        return 0;
    }
    if (index >= size) {
        return size - 1;
    }
    return index;
}

static __device__ inline bool craftax_wg_scatter_index(
    int32_t index,
    int32_t size,
    int* mapped_index
) {
    if (index < -size || index >= size) {
        return false;
    }
    *mapped_index = index < 0 ? index + size : index;
    return true;
}

static __device__ inline bool craftax_wg_is_boss_vulnerable(
    const CraftaxWorldState* state
) {
    int level = craftax_wg_jax_index(CF(player_level, state), CRAFTAX_WG_NUM_LEVELS);
    bool has_melee = false;
    bool has_ranged = false;
    for (int i = 0; i < CRAFTAX_WG_MAX_MELEE_MOBS; i++) {
        has_melee = has_melee || MOB_MASK(0, level, i, state);
    }
    for (int i = 0; i < CRAFTAX_WG_MAX_RANGED_MOBS; i++) {
        has_ranged = has_ranged || MOB_MASK(2, level, i, state);
    }
    return !has_melee
        && !has_ranged
        && CF(boss_timesteps_to_spawn_this_round, state) <= 0;
}

static __device__ inline void craftax_encode_mobs3_observation(
    const CraftaxWorldState* state,
    const void* mobs, int mc,
    int mob_class_index,
    int channels,
    int mob_channels_offset,
    float* obs
) {
    int level = craftax_wg_jax_index(CF(player_level, state), CRAFTAX_WG_NUM_LEVELS);
    for (int i = 0; i < 3; i++) {
        int local_row = MOB_POS(mc, level, i, 0, mobs)
            - CF2(player_position, 0, state)
            + CRAFTAX_WG_OBS_ROWS / 2;
        int local_col = MOB_POS(mc, level, i, 1, mobs)
            - CF2(player_position, 1, state)
            + CRAFTAX_WG_OBS_COLS / 2;
        int type_id = MOB_TYPE(mc, level, i, mobs);
        int scatter_row;
        int scatter_col;
        if (!craftax_wg_scatter_index(
                local_row,
                CRAFTAX_WG_OBS_ROWS,
                &scatter_row
            )
            || !craftax_wg_scatter_index(
                local_col,
                CRAFTAX_WG_OBS_COLS,
                &scatter_col
            )
            || type_id < 0
            || type_id >= CRAFTAX_WG_NUM_MOB_TYPES) {
            continue;
        }

        bool on_screen = local_row >= 0
            && local_row < CRAFTAX_WG_OBS_ROWS
            && local_col >= 0
            && local_col < CRAFTAX_WG_OBS_COLS;
        int world_row = MOB_POS(mc, level, i, 0, mobs);
        int world_col = MOB_POS(mc, level, i, 1, mobs);
        bool in_bounds = world_row >= 0
            && world_row < CRAFTAX_WG_MAP_SIZE
            && world_col >= 0
            && world_col < CRAFTAX_WG_MAP_SIZE;
        bool visible = in_bounds && state->light_map[level][world_row][world_col] > 12;
        int obs_base = (scatter_row * CRAFTAX_WG_OBS_COLS + scatter_col) * channels;
        int channel = mob_channels_offset
            + mob_class_index * CRAFTAX_WG_NUM_MOB_TYPES
            + type_id;
        obs[obs_base + channel] =
            MOB_MASK(mc, level, i, mobs) && on_screen && visible ? 1.0f : 0.0f;
    }
}

static __device__ inline void craftax_encode_mobs2_observation(
    const CraftaxWorldState* state,
    const void* mobs, int mc,
    int mob_class_index,
    int channels,
    int mob_channels_offset,
    float* obs
) {
    int level = craftax_wg_jax_index(CF(player_level, state), CRAFTAX_WG_NUM_LEVELS);
    for (int i = 0; i < 2; i++) {
        int local_row = MOB_POS(mc, level, i, 0, mobs)
            - CF2(player_position, 0, state)
            + CRAFTAX_WG_OBS_ROWS / 2;
        int local_col = MOB_POS(mc, level, i, 1, mobs)
            - CF2(player_position, 1, state)
            + CRAFTAX_WG_OBS_COLS / 2;
        int type_id = MOB_TYPE(mc, level, i, mobs);
        int scatter_row;
        int scatter_col;
        if (!craftax_wg_scatter_index(
                local_row,
                CRAFTAX_WG_OBS_ROWS,
                &scatter_row
            )
            || !craftax_wg_scatter_index(
                local_col,
                CRAFTAX_WG_OBS_COLS,
                &scatter_col
            )
            || type_id < 0
            || type_id >= CRAFTAX_WG_NUM_MOB_TYPES) {
            continue;
        }

        bool on_screen = local_row >= 0
            && local_row < CRAFTAX_WG_OBS_ROWS
            && local_col >= 0
            && local_col < CRAFTAX_WG_OBS_COLS;
        int world_row = MOB_POS(mc, level, i, 0, mobs);
        int world_col = MOB_POS(mc, level, i, 1, mobs);
        bool in_bounds = world_row >= 0
            && world_row < CRAFTAX_WG_MAP_SIZE
            && world_col >= 0
            && world_col < CRAFTAX_WG_MAP_SIZE;
        bool visible = in_bounds && state->light_map[level][world_row][world_col] > 12;
        int obs_base = (scatter_row * CRAFTAX_WG_OBS_COLS + scatter_col) * channels;
        int channel = mob_channels_offset
            + mob_class_index * CRAFTAX_WG_NUM_MOB_TYPES
            + type_id;
        obs[obs_base + channel] =
            MOB_MASK(mc, level, i, mobs) && on_screen && visible ? 1.0f : 0.0f;
    }
}

static __device__ inline void craftax_write_binary_bits(
    float* obs,
    int base,
    int value,
    int num_bits
) {
    if (num_bits == 6) {
        memcpy(obs + base, CRAFTAX_WG_BLOCK_LUT[value], 6 * sizeof(float));
    } else if (num_bits == 3) {
        memcpy(obs + base, CRAFTAX_WG_ITEM_LUT[value], 3 * sizeof(float));
    } else if (num_bits == 4) {
        memcpy(obs + base, CRAFTAX_WG_MOB_LUT[value], 4 * sizeof(float));
    } else {
        for (int i = 0; i < num_bits; i++) {
            obs[base + i] = (value & (1 << i)) ? 1.0f : 0.0f;
        }
    }
}

static __device__ inline void craftax_encode_mobs3_binary(
    const CraftaxWorldState* state,
    const void* mobs, int mc,
    int mob_class_index,
    int channels_per_cell,
    int mob_bits_offset,
    float* obs
) {
    int level = craftax_wg_jax_index(CF(player_level, state), CRAFTAX_WG_NUM_LEVELS);
    for (int i = 0; i < 3; i++) {
        int type_id = MOB_TYPE(mc, level, i, mobs);
        if (type_id < 0 || type_id >= CRAFTAX_WG_NUM_MOB_TYPES
            || !MOB_MASK(mc, level, i, mobs)) {
            continue;
        }

        int local_row = MOB_POS(mc, level, i, 0, mobs)
            - CF2(player_position, 0, state)
            + CRAFTAX_WG_OBS_ROWS / 2;
        int local_col = MOB_POS(mc, level, i, 1, mobs)
            - CF2(player_position, 1, state)
            + CRAFTAX_WG_OBS_COLS / 2;
        if (local_row < 0 || local_row >= CRAFTAX_WG_OBS_ROWS
            || local_col < 0 || local_col >= CRAFTAX_WG_OBS_COLS) {
            continue;
        }

        int world_row = MOB_POS(mc, level, i, 0, mobs);
        int world_col = MOB_POS(mc, level, i, 1, mobs);
        if (world_row < 0 || world_row >= CRAFTAX_WG_MAP_SIZE
            || world_col < 0 || world_col >= CRAFTAX_WG_MAP_SIZE
            || state->light_map[level][world_row][world_col] <= 12) {
            continue;
        }

        int obs_base = (local_row * CRAFTAX_WG_OBS_COLS + local_col)
            * channels_per_cell;
        int class_offset = mob_bits_offset
            + mob_class_index * CRAFTAX_WG_BINARY_MOB_BITS;
        memcpy(obs + obs_base + class_offset,
               CRAFTAX_WG_MOB_LUT[type_id + 1],
               CRAFTAX_WG_BINARY_MOB_BITS * sizeof(float));
    }
}

static __device__ inline void craftax_encode_mobs2_binary(
    const CraftaxWorldState* state,
    const void* mobs, int mc,
    int mob_class_index,
    int channels_per_cell,
    int mob_bits_offset,
    float* obs
) {
    int level = craftax_wg_jax_index(CF(player_level, state), CRAFTAX_WG_NUM_LEVELS);
    for (int i = 0; i < 2; i++) {
        int type_id = MOB_TYPE(mc, level, i, mobs);
        if (type_id < 0 || type_id >= CRAFTAX_WG_NUM_MOB_TYPES
            || !MOB_MASK(mc, level, i, mobs)) {
            continue;
        }

        int local_row = MOB_POS(mc, level, i, 0, mobs)
            - CF2(player_position, 0, state)
            + CRAFTAX_WG_OBS_ROWS / 2;
        int local_col = MOB_POS(mc, level, i, 1, mobs)
            - CF2(player_position, 1, state)
            + CRAFTAX_WG_OBS_COLS / 2;
        if (local_row < 0 || local_row >= CRAFTAX_WG_OBS_ROWS
            || local_col < 0 || local_col >= CRAFTAX_WG_OBS_COLS) {
            continue;
        }

        int world_row = MOB_POS(mc, level, i, 0, mobs);
        int world_col = MOB_POS(mc, level, i, 1, mobs);
        if (world_row < 0 || world_row >= CRAFTAX_WG_MAP_SIZE
            || world_col < 0 || world_col >= CRAFTAX_WG_MAP_SIZE
            || state->light_map[level][world_row][world_col] <= 12) {
            continue;
        }

        int obs_base = (local_row * CRAFTAX_WG_OBS_COLS + local_col)
            * channels_per_cell;
        int class_offset = mob_bits_offset
            + mob_class_index * CRAFTAX_WG_BINARY_MOB_BITS;
        memcpy(obs + obs_base + class_offset,
               CRAFTAX_WG_MOB_LUT[type_id + 1],
               CRAFTAX_WG_BINARY_MOB_BITS * sizeof(float));
    }
}

static __device__ inline void craftax_encode_map_base_observation(
    const CraftaxWorldState* state,
    float* obs
) {
    const int channels = CRAFTAX_WG_BINARY_CHANNELS_PER_CELL;
    const int top = CF2(player_position, 0, state) - CRAFTAX_WG_OBS_ROWS / 2;
    const int left = CF2(player_position, 1, state) - CRAFTAX_WG_OBS_COLS / 2;
    const int level = CF(player_level, state);
    const float* empty_cell = CRAFTAX_WG_EMPTY_CELL_TEMPLATE;

    for (int row = 0; row < CRAFTAX_WG_OBS_ROWS; row++) {
        int world_row = top + row;
        bool row_in_bounds = world_row >= 0 && world_row < CRAFTAX_WG_MAP_SIZE;
        for (int col = 0; col < CRAFTAX_WG_OBS_COLS; col++) {
            int world_col = left + col;
            int obs_base = (row * CRAFTAX_WG_OBS_COLS + col) * channels;
            const float* cell = empty_cell;

            if (row_in_bounds && world_col >= 0 && world_col < CRAFTAX_WG_MAP_SIZE
                && state->light_map[level][world_row][world_col] > 12) {
                uint8_t block = state->map[level][world_row][world_col];
                uint8_t item = state->item_map[level][world_row][world_col];
                cell = CRAFTAX_WG_VISIBLE_CELL_TEMPLATE_LUT[block][item + 1];
            }

            memcpy(obs + obs_base, cell, CRAFTAX_WG_CELL_TEMPLATE_BYTES);
        }
    }
}

static __device__ inline void craftax_encode_packed_map_base_observation(
    const CraftaxWorldState* state,
    float* obs
) {
    const int channels = CRAFTAX_WG_PACKED_CHANNELS_PER_CELL;
    const int top = CF2(player_position, 0, state) - CRAFTAX_WG_OBS_ROWS / 2;
    const int left = CF2(player_position, 1, state) - CRAFTAX_WG_OBS_COLS / 2;
    const int level = CF(player_level, state);

    memset(obs, 0, CRAFTAX_WG_PACKED_MAP_OBS_SIZE * sizeof(float));
    for (int row = 0; row < CRAFTAX_WG_OBS_ROWS; row++) {
        int world_row = top + row;
        bool row_in_bounds = world_row >= 0 && world_row < CRAFTAX_WG_MAP_SIZE;
        for (int col = 0; col < CRAFTAX_WG_OBS_COLS; col++) {
            int world_col = left + col;
            int obs_base = (row * CRAFTAX_WG_OBS_COLS + col) * channels;
            if (row_in_bounds && world_col >= 0 && world_col < CRAFTAX_WG_MAP_SIZE
                && state->light_map[level][world_row][world_col] > 12) {
                obs[obs_base + 0] = (float)state->map[level][world_row][world_col];
                obs[obs_base + 1] = (float)state->item_map[level][world_row][world_col] + 1.0f;
                obs[obs_base + 2] = 1.0f;
            }
        }
    }
}

static __device__ inline void craftax_clear_mob_channels_observation(float* obs) {
    const int channels = CRAFTAX_WG_BINARY_CHANNELS_PER_CELL;
    const int mob_bits_offset = CRAFTAX_WG_BINARY_BLOCK_BITS + CRAFTAX_WG_BINARY_ITEM_BITS;
    const size_t mob_channel_bytes =
        CRAFTAX_WG_NUM_MOB_CLASSES * CRAFTAX_WG_BINARY_MOB_BITS * sizeof(float);

    for (int cell = 0; cell < CRAFTAX_WG_OBS_WINDOW_CELLS; cell++) {
        memset(obs + cell * channels + mob_bits_offset, 0, mob_channel_bytes);
    }
}

static __device__ inline void craftax_encode_mobs3_packed(
    const CraftaxWorldState* state,
    const void* mobs, int mc,
    int mob_class_index,
    float* obs
) {
    const int level = craftax_wg_jax_index(CF(player_level, state), CRAFTAX_WG_NUM_LEVELS);
    const int mob_slot_offset = 3 + mob_class_index;
    for (int i = 0; i < 3; i++) {
        int type_id = MOB_TYPE(mc, level, i, mobs);
        if (type_id < 0 || type_id >= CRAFTAX_WG_NUM_MOB_TYPES
            || !MOB_MASK(mc, level, i, mobs)) {
            continue;
        }

        int local_row = MOB_POS(mc, level, i, 0, mobs)
            - CF2(player_position, 0, state)
            + CRAFTAX_WG_OBS_ROWS / 2;
        int local_col = MOB_POS(mc, level, i, 1, mobs)
            - CF2(player_position, 1, state)
            + CRAFTAX_WG_OBS_COLS / 2;
        if (local_row < 0 || local_row >= CRAFTAX_WG_OBS_ROWS
            || local_col < 0 || local_col >= CRAFTAX_WG_OBS_COLS) {
            continue;
        }

        int world_row = MOB_POS(mc, level, i, 0, mobs);
        int world_col = MOB_POS(mc, level, i, 1, mobs);
        if (world_row < 0 || world_row >= CRAFTAX_WG_MAP_SIZE
            || world_col < 0 || world_col >= CRAFTAX_WG_MAP_SIZE
            || state->light_map[level][world_row][world_col] <= 12) {
            continue;
        }

        int obs_base = (local_row * CRAFTAX_WG_OBS_COLS + local_col)
            * CRAFTAX_WG_PACKED_CHANNELS_PER_CELL;
        obs[obs_base + mob_slot_offset] = (float)(type_id + 1);
    }
}

static __device__ inline void craftax_encode_mobs2_packed(
    const CraftaxWorldState* state,
    const void* mobs, int mc,
    int mob_class_index,
    float* obs
) {
    const int level = craftax_wg_jax_index(CF(player_level, state), CRAFTAX_WG_NUM_LEVELS);
    const int mob_slot_offset = 3 + mob_class_index;
    for (int i = 0; i < 2; i++) {
        int type_id = MOB_TYPE(mc, level, i, mobs);
        if (type_id < 0 || type_id >= CRAFTAX_WG_NUM_MOB_TYPES
            || !MOB_MASK(mc, level, i, mobs)) {
            continue;
        }

        int local_row = MOB_POS(mc, level, i, 0, mobs)
            - CF2(player_position, 0, state)
            + CRAFTAX_WG_OBS_ROWS / 2;
        int local_col = MOB_POS(mc, level, i, 1, mobs)
            - CF2(player_position, 1, state)
            + CRAFTAX_WG_OBS_COLS / 2;
        if (local_row < 0 || local_row >= CRAFTAX_WG_OBS_ROWS
            || local_col < 0 || local_col >= CRAFTAX_WG_OBS_COLS) {
            continue;
        }

        int world_row = MOB_POS(mc, level, i, 0, mobs);
        int world_col = MOB_POS(mc, level, i, 1, mobs);
        if (world_row < 0 || world_row >= CRAFTAX_WG_MAP_SIZE
            || world_col < 0 || world_col >= CRAFTAX_WG_MAP_SIZE
            || state->light_map[level][world_row][world_col] <= 12) {
            continue;
        }

        int obs_base = (local_row * CRAFTAX_WG_OBS_COLS + local_col)
            * CRAFTAX_WG_PACKED_CHANNELS_PER_CELL;
        obs[obs_base + mob_slot_offset] = (float)(type_id + 1);
    }
}

static __device__ inline void craftax_encode_packed_mobs_observation(
    const CraftaxWorldState* state,
    float* obs
) {
    craftax_encode_mobs3_packed(state, state, 0, 0, obs);
    craftax_encode_mobs3_packed(state, state, 1, 1, obs);
    craftax_encode_mobs2_packed(state, state, 2, 2, obs);
    craftax_encode_mobs3_packed(state, state, 3, 3, obs);
    craftax_encode_mobs3_packed(state, state, 4, 4, obs);
}

static __device__ inline void craftax_encode_mobs_observation(
    const CraftaxWorldState* state,
    float* obs
) {
    const int channels = CRAFTAX_WG_BINARY_CHANNELS_PER_CELL;
    const int mob_bits_offset = CRAFTAX_WG_BINARY_BLOCK_BITS + CRAFTAX_WG_BINARY_ITEM_BITS;

    craftax_encode_mobs3_binary(
        state,
        state, 0,
        0,
        channels,
        mob_bits_offset,
        obs
    );
    craftax_encode_mobs3_binary(
        state,
        state, 1,
        1,
        channels,
        mob_bits_offset,
        obs
    );
    craftax_encode_mobs2_binary(
        state,
        state, 2,
        2,
        channels,
        mob_bits_offset,
        obs
    );
    craftax_encode_mobs3_binary(
        state,
        state, 3,
        3,
        channels,
        mob_bits_offset,
        obs
    );
    craftax_encode_mobs3_binary(
        state,
        state, 4,
        4,
        channels,
        mob_bits_offset,
        obs
    );
}

static __device__ inline void craftax_encode_scalar_observation_tail_at(
    const CraftaxWorldState* state,
    float* obs,
    int index
) {
    const int level = CF(player_level, state);
    obs[index++] = sqrtf((float)CF(inv_wood, state)) * (1.0f / 10.0f);
    obs[index++] = sqrtf((float)CF(inv_stone, state)) * (1.0f / 10.0f);
    obs[index++] = sqrtf((float)CF(inv_coal, state)) * (1.0f / 10.0f);
    obs[index++] = sqrtf((float)CF(inv_iron, state)) * (1.0f / 10.0f);
    obs[index++] = sqrtf((float)CF(inv_diamond, state)) * (1.0f / 10.0f);
    obs[index++] = sqrtf((float)CF(inv_sapphire, state)) * (1.0f / 10.0f);
    obs[index++] = sqrtf((float)CF(inv_ruby, state)) * (1.0f / 10.0f);
    obs[index++] = sqrtf((float)CF(inv_sapling, state)) * (1.0f / 10.0f);
    obs[index++] = sqrtf((float)CF(inv_torches, state)) * (1.0f / 10.0f);
    obs[index++] = sqrtf((float)CF(inv_arrows, state)) * (1.0f / 10.0f);
    obs[index++] = (float)CF(inv_books, state) * (1.0f / 2.0f);
    obs[index++] = (float)CF(inv_pickaxe, state) * (1.0f / 4.0f);
    obs[index++] = (float)CF(inv_sword, state) * (1.0f / 4.0f);
    obs[index++] = (float)CF(sword_enchantment, state);
    obs[index++] = (float)CF(bow_enchantment, state);
    obs[index++] = (float)CF(inv_bow, state);

    for (int i = 0; i < 6; i++) {
        obs[index++] = sqrtf((float)CF2(inv_potions, i, state)) * (1.0f / 10.0f);
    }

    obs[index++] = CF(player_health, state) * (1.0f / 10.0f);
    obs[index++] = (float)CF(player_food, state) * (1.0f / 10.0f);
    obs[index++] = (float)CF(player_drink, state) * (1.0f / 10.0f);
    obs[index++] = (float)CF(player_energy, state) * (1.0f / 10.0f);
    obs[index++] = (float)CF(player_mana, state) * (1.0f / 10.0f);
    obs[index++] = (float)CF(player_xp, state) * (1.0f / 10.0f);
    obs[index++] = (float)CF(player_dexterity, state) * (1.0f / 10.0f);
    obs[index++] = (float)CF(player_strength, state) * (1.0f / 10.0f);
    obs[index++] = (float)CF(player_intelligence, state) * (1.0f / 10.0f);

    int direction_index = CF(player_direction, state) - 1;
    for (int i = 0; i < 4; i++) {
        obs[index++] = i == direction_index ? 1.0f : 0.0f;
    }

    for (int i = 0; i < 4; i++) {
        obs[index++] = (float)CF2(inv_armour, i, state) * (1.0f / 2.0f);
    }
    for (int i = 0; i < 4; i++) {
        obs[index++] = (float)CF2(armour_enchantments, i, state);
    }

    obs[index++] = CF(light_level, state);
    obs[index++] = CF(is_sleeping, state) ? 1.0f : 0.0f;
    obs[index++] = CF(is_resting, state) ? 1.0f : 0.0f;
    obs[index++] = CF2(learned_spells, 0, state) ? 1.0f : 0.0f;
    obs[index++] = CF2(learned_spells, 1, state) ? 1.0f : 0.0f;
    obs[index++] = (float)CF(player_level, state) * (1.0f / 10.0f);
    obs[index++] = CF2(monsters_killed, level, state) >= CRAFTAX_WG_MONSTERS_KILLED_TO_CLEAR_LEVEL ? 1.0f : 0.0f;
    obs[index++] = craftax_wg_is_boss_vulnerable(state) ? 1.0f : 0.0f;
}

static __device__ inline void craftax_encode_scalar_observation_tail(
    const CraftaxWorldState* state,
    float* obs
) {
    craftax_encode_scalar_observation_tail_at(state, obs, CRAFTAX_WG_BINARY_MAP_OBS_SIZE);
}

static __device__ inline void craftax_encode_reset_observation(
    const CraftaxWorldState* state,
    float* obs
) {
    craftax_encode_packed_map_base_observation(state, obs);
    craftax_encode_packed_mobs_observation(state, obs);
    craftax_encode_scalar_observation_tail_at(state, obs, CRAFTAX_WG_PACKED_MAP_OBS_SIZE);
}

static __device__ inline void craftax_encode_compact_map_base_observation(
    const CraftaxWorldState* state,
    uint8_t* obs
) {
    const int channels = CRAFTAX_WG_PACKED_CHANNELS_PER_CELL;
    const int top = CF2(player_position, 0, state) - CRAFTAX_WG_OBS_ROWS / 2;
    const int left = CF2(player_position, 1, state) - CRAFTAX_WG_OBS_COLS / 2;
    const int level = CF(player_level, state);

    memset(obs, 0, CRAFTAX_WG_PACKED_MAP_OBS_SIZE);
    for (int row = 0; row < CRAFTAX_WG_OBS_ROWS; row++) {
        int world_row = top + row;
        bool row_in_bounds = world_row >= 0 && world_row < CRAFTAX_WG_MAP_SIZE;
        for (int col = 0; col < CRAFTAX_WG_OBS_COLS; col++) {
            int world_col = left + col;
            int obs_base = (row * CRAFTAX_WG_OBS_COLS + col) * channels;
            if (row_in_bounds && world_col >= 0 && world_col < CRAFTAX_WG_MAP_SIZE
                && state->light_map[level][world_row][world_col] > 12) {
                obs[obs_base + 0] = (uint8_t)state->map[level][world_row][world_col];
                obs[obs_base + 1] = (uint8_t)(state->item_map[level][world_row][world_col] + 1);
                obs[obs_base + 2] = 1;
            }
        }
    }
}

static __device__ inline void craftax_encode_mobs3_compact(
    const CraftaxWorldState* state,
    const void* mobs, int mc,
    int mob_class_index,
    uint8_t* obs
) {
    const int level = craftax_wg_jax_index(CF(player_level, state), CRAFTAX_WG_NUM_LEVELS);
    const int mob_slot_offset = 3 + mob_class_index;
    for (int i = 0; i < 3; i++) {
        int type_id = MOB_TYPE(mc, level, i, mobs);
        if (type_id < 0 || type_id >= CRAFTAX_WG_NUM_MOB_TYPES
            || !MOB_MASK(mc, level, i, mobs)) {
            continue;
        }

        int local_row = MOB_POS(mc, level, i, 0, mobs)
            - CF2(player_position, 0, state)
            + CRAFTAX_WG_OBS_ROWS / 2;
        int local_col = MOB_POS(mc, level, i, 1, mobs)
            - CF2(player_position, 1, state)
            + CRAFTAX_WG_OBS_COLS / 2;
        if (local_row < 0 || local_row >= CRAFTAX_WG_OBS_ROWS
            || local_col < 0 || local_col >= CRAFTAX_WG_OBS_COLS) {
            continue;
        }

        int world_row = MOB_POS(mc, level, i, 0, mobs);
        int world_col = MOB_POS(mc, level, i, 1, mobs);
        if (world_row < 0 || world_row >= CRAFTAX_WG_MAP_SIZE
            || world_col < 0 || world_col >= CRAFTAX_WG_MAP_SIZE
            || state->light_map[level][world_row][world_col] <= 12) {
            continue;
        }

        int obs_base = (local_row * CRAFTAX_WG_OBS_COLS + local_col)
            * CRAFTAX_WG_PACKED_CHANNELS_PER_CELL;
        obs[obs_base + mob_slot_offset] = (uint8_t)(type_id + 1);
    }
}

static __device__ inline void craftax_encode_mobs2_compact(
    const CraftaxWorldState* state,
    const void* mobs, int mc,
    int mob_class_index,
    uint8_t* obs
) {
    const int level = craftax_wg_jax_index(CF(player_level, state), CRAFTAX_WG_NUM_LEVELS);
    const int mob_slot_offset = 3 + mob_class_index;
    for (int i = 0; i < 2; i++) {
        int type_id = MOB_TYPE(mc, level, i, mobs);
        if (type_id < 0 || type_id >= CRAFTAX_WG_NUM_MOB_TYPES
            || !MOB_MASK(mc, level, i, mobs)) {
            continue;
        }

        int local_row = MOB_POS(mc, level, i, 0, mobs)
            - CF2(player_position, 0, state)
            + CRAFTAX_WG_OBS_ROWS / 2;
        int local_col = MOB_POS(mc, level, i, 1, mobs)
            - CF2(player_position, 1, state)
            + CRAFTAX_WG_OBS_COLS / 2;
        if (local_row < 0 || local_row >= CRAFTAX_WG_OBS_ROWS
            || local_col < 0 || local_col >= CRAFTAX_WG_OBS_COLS) {
            continue;
        }

        int world_row = MOB_POS(mc, level, i, 0, mobs);
        int world_col = MOB_POS(mc, level, i, 1, mobs);
        if (world_row < 0 || world_row >= CRAFTAX_WG_MAP_SIZE
            || world_col < 0 || world_col >= CRAFTAX_WG_MAP_SIZE
            || state->light_map[level][world_row][world_col] <= 12) {
            continue;
        }

        int obs_base = (local_row * CRAFTAX_WG_OBS_COLS + local_col)
            * CRAFTAX_WG_PACKED_CHANNELS_PER_CELL;
        obs[obs_base + mob_slot_offset] = (uint8_t)(type_id + 1);
    }
}

// See CRAFTAX_WG_COMPACT_OBS_SIZE for the layout contract. The scalar tail
// goes through the float encoder and is memcpy'd, so the reinterpreted floats
// on the learner side are bit-identical to the float observation path.
static __device__ inline void craftax_encode_compact_observation(
    const CraftaxWorldState* state,
    uint8_t* obs
) {
    craftax_encode_compact_map_base_observation(state, obs);
    craftax_encode_mobs3_compact(state, state, 0, 0, obs);
    craftax_encode_mobs3_compact(state, state, 1, 1, obs);
    craftax_encode_mobs2_compact(state, state, 2, 2, obs);
    craftax_encode_mobs3_compact(state, state, 3, 3, obs);
    craftax_encode_mobs3_compact(state, state, 4, 4, obs);

    float tail[CRAFTAX_WG_INVENTORY_OBS_SIZE];
    craftax_encode_scalar_observation_tail_at(state, tail, 0);
    memcpy(obs + CRAFTAX_WG_PACKED_MAP_OBS_SIZE, tail, sizeof(tail));
}

// ============================================================
// ===== craftax.h =====
// ============================================================
// Full native Craftax environment for PufferLib Ocean.


#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// ============================================================
// Optional step profiling (compile with -DCRAFTAX_PROFILE)
// ============================================================
#ifdef CRAFTAX_PROFILE

#define CRAFTAX_NUM_PROFILE_ZONES 18

typedef struct {
    const char* name;
    uint64_t total_ns;
    uint64_t count;
} CraftaxProfileZone;

static __device__ CraftaxProfileZone craftax_profile_zones[CRAFTAX_NUM_PROFILE_ZONES] = {
    {"change_floor", 0, 0},
    {"crafting", 0, 0},
    {"do_action", 0, 0},
    {"place+shoot+spell+potion", 0, 0},
    {"read_book", 0, 0},
    {"enchant", 0, 0},
    {"boss+attr+move", 0, 0},
    {"update_mobs", 0, 0},
    {"spawn_mobs", 0, 0},
    {"plants+intrinsics+achieve", 0, 0},
    {"reward+bookkeeping", 0, 0},
    {"encode_obs", 0, 0},
    {"rng_split", 0, 0},
    {"is_game_over", 0, 0},
    {"reset_on_done", 0, 0},
    {"copy_achievements", 0, 0},
    {"reward_bookkeeping", 0, 0},
    {"unprofiled", 0, 0},
};

static __device__ inline uint64_t craftax_profile_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static __device__ inline void craftax_profile_record(int zone, uint64_t start) {
    craftax_profile_zones[zone].total_ns += craftax_profile_now() - start;
    craftax_profile_zones[zone].count++;
}

static __device__ inline void craftax_profile_report(void) {
    fprintf(stderr, "\n=== Craftax Step Profile ===\n");
    uint64_t total = 0;
    for (int i = 0; i < CRAFTAX_NUM_PROFILE_ZONES; i++) {
        total += craftax_profile_zones[i].total_ns;
    }
    for (int i = 0; i < CRAFTAX_NUM_PROFILE_ZONES; i++) {
        CraftaxProfileZone* z = &craftax_profile_zones[i];
        if (z->count == 0) continue;
        double pct = total > 0 ? (100.0 * (double)z->total_ns / (double)total) : 0.0;
        double avg_us = (double)z->total_ns / (double)z->count / 1000.0;
        fprintf(stderr, "%-28s %8.3f%%  %10.2f us/step  (%lu calls)\n",
                z->name, pct, avg_us, (unsigned long)z->count);
    }
    fprintf(stderr, "%-28s %8.3f%%  %10.2f us/step\n",
            "TOTAL", 100.0, (double)total / (double)craftax_profile_zones[0].count / 1000.0);
}

#define CRAFTAX_PROFILE_START() uint64_t _prof_start = craftax_profile_now(); uint64_t _prof_zone_start;
#define CRAFTAX_PROFILE_ZONE(n) do { _prof_zone_start = craftax_profile_now(); } while(0)
#define CRAFTAX_PROFILE_END(n) craftax_profile_record((n), _prof_zone_start)
#define CRAFTAX_PROFILE_FINAL(n) craftax_profile_record((n), _prof_start)

#elif defined(CRAFTAX_CU_PROFILE)

// Device-side zone profiler: clock64 deltas accumulated with atomics. Only
// for perf investigation builds (-DCRAFTAX_CU_PROFILE); off by default.
#define CRAFTAX_NUM_PROFILE_ZONES 18
__device__ unsigned long long g_cu_prof_cycles[CRAFTAX_NUM_PROFILE_ZONES];
__device__ unsigned long long g_cu_prof_count[CRAFTAX_NUM_PROFILE_ZONES];

#define CRAFTAX_PROFILE_START() \
    unsigned long long _prof_zone_start = 0; (void)_prof_zone_start
#define CRAFTAX_PROFILE_ZONE(n) do { _prof_zone_start = clock64(); } while (0)
#define CRAFTAX_PROFILE_END(n) do { \
    atomicAdd(&g_cu_prof_cycles[n], \
              (unsigned long long)(clock64() - _prof_zone_start)); \
    atomicAdd(&g_cu_prof_count[n], 1ULL); \
} while (0)
#define CRAFTAX_PROFILE_FINAL(n) ((void)0)
#define craftax_profile_report() ((void)0)

#else

#define CRAFTAX_PROFILE_START() ((void)0)
#define CRAFTAX_PROFILE_ZONE(n) ((void)0)
#define CRAFTAX_PROFILE_END(n) ((void)0)
#define CRAFTAX_PROFILE_FINAL(n) ((void)0)
#define craftax_profile_report() ((void)0)

#endif // CRAFTAX_PROFILE

// ============================================================
// Constants
// ============================================================
#define CRAFTAX_OBS_ROWS 9
#define CRAFTAX_OBS_COLS 11
#define CRAFTAX_MAP_SIZE 48
#define CRAFTAX_NUM_LEVELS 9

#define CRAFTAX_NUM_BLOCK_TYPES 37
#define CRAFTAX_NUM_ITEM_TYPES 5
#define CRAFTAX_NUM_MOB_CLASSES 5
#define CRAFTAX_NUM_MOB_TYPES 8
#define CRAFTAX_INVENTORY_OBS_SIZE 51
// Compile with -DCRAFTAX_COMPACT_OBS to emit uint8 observations (996 bytes:
// 792 map ID bytes + 51 float32 scalars as raw bytes) instead of 843 floats.
// Pair with OBS_TENSOR_T ByteTensor and the CraftaxCompactEncoder policy
// encoder, which expands back to the identical 843-float observation on GPU.
#ifdef CRAFTAX_COMPACT_OBS
#define CRAFTAX_OBS_SIZE CRAFTAX_WG_COMPACT_OBS_SIZE
typedef uint8_t CraftaxObs;
#else
#define CRAFTAX_OBS_SIZE CRAFTAX_WG_OBS_SIZE
typedef float CraftaxObs;
#endif

#define CRAFTAX_NUM_ACTIONS 43
#define CRAFTAX_NUM_ACHIEVEMENTS 67

#define CRAFTAX_MAX_MELEE_MOBS 3
#define CRAFTAX_MAX_PASSIVE_MOBS 3
#define CRAFTAX_MAX_RANGED_MOBS 2
#define CRAFTAX_MAX_MOB_PROJECTILES 3
#define CRAFTAX_MAX_PLAYER_PROJECTILES 3
#define CRAFTAX_MAX_GROWING_PLANTS 10

#define CRAFTAX_DEFAULT_MAX_TIMESTEPS 100000
#define CRAFTAX_DAY_LENGTH 300
#define CRAFTAX_MAX_ATTRIBUTE 5
#define CRAFTAX_MOB_DESPAWN_DISTANCE 14
#define CRAFTAX_MONSTERS_KILLED_TO_CLEAR_LEVEL 8

// ============================================================
// Enums copied from craftax/craftax/constants.py
// ============================================================
typedef enum CraftaxBlockType {
    CRAFTAX_BLOCK_INVALID = 0,
    CRAFTAX_BLOCK_OUT_OF_BOUNDS = 1,
    CRAFTAX_BLOCK_GRASS = 2,
    CRAFTAX_BLOCK_WATER = 3,
    CRAFTAX_BLOCK_STONE = 4,
    CRAFTAX_BLOCK_TREE = 5,
    CRAFTAX_BLOCK_WOOD = 6,
    CRAFTAX_BLOCK_PATH = 7,
    CRAFTAX_BLOCK_COAL = 8,
    CRAFTAX_BLOCK_IRON = 9,
    CRAFTAX_BLOCK_DIAMOND = 10,
    CRAFTAX_BLOCK_CRAFTING_TABLE = 11,
    CRAFTAX_BLOCK_FURNACE = 12,
    CRAFTAX_BLOCK_SAND = 13,
    CRAFTAX_BLOCK_LAVA = 14,
    CRAFTAX_BLOCK_PLANT = 15,
    CRAFTAX_BLOCK_RIPE_PLANT = 16,
    CRAFTAX_BLOCK_WALL = 17,
    CRAFTAX_BLOCK_DARKNESS = 18,
    CRAFTAX_BLOCK_WALL_MOSS = 19,
    CRAFTAX_BLOCK_STALAGMITE = 20,
    CRAFTAX_BLOCK_SAPPHIRE = 21,
    CRAFTAX_BLOCK_RUBY = 22,
    CRAFTAX_BLOCK_CHEST = 23,
    CRAFTAX_BLOCK_FOUNTAIN = 24,
    CRAFTAX_BLOCK_FIRE_GRASS = 25,
    CRAFTAX_BLOCK_ICE_GRASS = 26,
    CRAFTAX_BLOCK_GRAVEL = 27,
    CRAFTAX_BLOCK_FIRE_TREE = 28,
    CRAFTAX_BLOCK_ICE_SHRUB = 29,
    CRAFTAX_BLOCK_ENCHANTMENT_TABLE_FIRE = 30,
    CRAFTAX_BLOCK_ENCHANTMENT_TABLE_ICE = 31,
    CRAFTAX_BLOCK_NECROMANCER = 32,
    CRAFTAX_BLOCK_GRAVE = 33,
    CRAFTAX_BLOCK_GRAVE2 = 34,
    CRAFTAX_BLOCK_GRAVE3 = 35,
    CRAFTAX_BLOCK_NECROMANCER_VULNERABLE = 36,
} CraftaxBlockType;

typedef enum CraftaxItemType {
    CRAFTAX_ITEM_NONE = 0,
    CRAFTAX_ITEM_TORCH = 1,
    CRAFTAX_ITEM_LADDER_DOWN = 2,
    CRAFTAX_ITEM_LADDER_UP = 3,
    CRAFTAX_ITEM_LADDER_DOWN_BLOCKED = 4,
} CraftaxItemType;

typedef enum CraftaxAction {
    CRAFTAX_ACTION_NOOP = 0,
    CRAFTAX_ACTION_LEFT = 1,
    CRAFTAX_ACTION_RIGHT = 2,
    CRAFTAX_ACTION_UP = 3,
    CRAFTAX_ACTION_DOWN = 4,
    CRAFTAX_ACTION_DO = 5,
    CRAFTAX_ACTION_SLEEP = 6,
    CRAFTAX_ACTION_PLACE_STONE = 7,
    CRAFTAX_ACTION_PLACE_TABLE = 8,
    CRAFTAX_ACTION_PLACE_FURNACE = 9,
    CRAFTAX_ACTION_PLACE_PLANT = 10,
    CRAFTAX_ACTION_MAKE_WOOD_PICKAXE = 11,
    CRAFTAX_ACTION_MAKE_STONE_PICKAXE = 12,
    CRAFTAX_ACTION_MAKE_IRON_PICKAXE = 13,
    CRAFTAX_ACTION_MAKE_WOOD_SWORD = 14,
    CRAFTAX_ACTION_MAKE_STONE_SWORD = 15,
    CRAFTAX_ACTION_MAKE_IRON_SWORD = 16,
    CRAFTAX_ACTION_REST = 17,
    CRAFTAX_ACTION_DESCEND = 18,
    CRAFTAX_ACTION_ASCEND = 19,
    CRAFTAX_ACTION_MAKE_DIAMOND_PICKAXE = 20,
    CRAFTAX_ACTION_MAKE_DIAMOND_SWORD = 21,
    CRAFTAX_ACTION_MAKE_IRON_ARMOUR = 22,
    CRAFTAX_ACTION_MAKE_DIAMOND_ARMOUR = 23,
    CRAFTAX_ACTION_SHOOT_ARROW = 24,
    CRAFTAX_ACTION_MAKE_ARROW = 25,
    CRAFTAX_ACTION_CAST_FIREBALL = 26,
    CRAFTAX_ACTION_CAST_ICEBALL = 27,
    CRAFTAX_ACTION_PLACE_TORCH = 28,
    CRAFTAX_ACTION_DRINK_POTION_RED = 29,
    CRAFTAX_ACTION_DRINK_POTION_GREEN = 30,
    CRAFTAX_ACTION_DRINK_POTION_BLUE = 31,
    CRAFTAX_ACTION_DRINK_POTION_PINK = 32,
    CRAFTAX_ACTION_DRINK_POTION_CYAN = 33,
    CRAFTAX_ACTION_DRINK_POTION_YELLOW = 34,
    CRAFTAX_ACTION_READ_BOOK = 35,
    CRAFTAX_ACTION_ENCHANT_SWORD = 36,
    CRAFTAX_ACTION_ENCHANT_ARMOUR = 37,
    CRAFTAX_ACTION_MAKE_TORCH = 38,
    CRAFTAX_ACTION_LEVEL_UP_DEXTERITY = 39,
    CRAFTAX_ACTION_LEVEL_UP_STRENGTH = 40,
    CRAFTAX_ACTION_LEVEL_UP_INTELLIGENCE = 41,
    CRAFTAX_ACTION_ENCHANT_BOW = 42,
} CraftaxAction;

typedef enum CraftaxMobType {
    CRAFTAX_MOB_PASSIVE = 0,
    CRAFTAX_MOB_MELEE = 1,
    CRAFTAX_MOB_RANGED = 2,
    CRAFTAX_MOB_PROJECTILE = 3,
} CraftaxMobType;

typedef enum CraftaxProjectileType {
    CRAFTAX_PROJECTILE_ARROW = 0,
    CRAFTAX_PROJECTILE_DAGGER = 1,
    CRAFTAX_PROJECTILE_FIREBALL = 2,
    CRAFTAX_PROJECTILE_ICEBALL = 3,
    CRAFTAX_PROJECTILE_ARROW2 = 4,
    CRAFTAX_PROJECTILE_SLIMEBALL = 5,
    CRAFTAX_PROJECTILE_FIREBALL2 = 6,
    CRAFTAX_PROJECTILE_ICEBALL2 = 7,
} CraftaxProjectileType;

typedef enum CraftaxAchievement {
    CRAFTAX_ACH_COLLECT_WOOD = 0,
    CRAFTAX_ACH_PLACE_TABLE = 1,
    CRAFTAX_ACH_EAT_COW = 2,
    CRAFTAX_ACH_COLLECT_SAPLING = 3,
    CRAFTAX_ACH_COLLECT_DRINK = 4,
    CRAFTAX_ACH_MAKE_WOOD_PICKAXE = 5,
    CRAFTAX_ACH_MAKE_WOOD_SWORD = 6,
    CRAFTAX_ACH_PLACE_PLANT = 7,
    CRAFTAX_ACH_DEFEAT_ZOMBIE = 8,
    CRAFTAX_ACH_COLLECT_STONE = 9,
    CRAFTAX_ACH_PLACE_STONE = 10,
    CRAFTAX_ACH_EAT_PLANT = 11,
    CRAFTAX_ACH_DEFEAT_SKELETON = 12,
    CRAFTAX_ACH_MAKE_STONE_PICKAXE = 13,
    CRAFTAX_ACH_MAKE_STONE_SWORD = 14,
    CRAFTAX_ACH_WAKE_UP = 15,
    CRAFTAX_ACH_PLACE_FURNACE = 16,
    CRAFTAX_ACH_COLLECT_COAL = 17,
    CRAFTAX_ACH_COLLECT_IRON = 18,
    CRAFTAX_ACH_COLLECT_DIAMOND = 19,
    CRAFTAX_ACH_MAKE_IRON_PICKAXE = 20,
    CRAFTAX_ACH_MAKE_IRON_SWORD = 21,
    CRAFTAX_ACH_MAKE_ARROW = 22,
    CRAFTAX_ACH_MAKE_TORCH = 23,
    CRAFTAX_ACH_PLACE_TORCH = 24,
    CRAFTAX_ACH_MAKE_DIAMOND_SWORD = 25,
    CRAFTAX_ACH_MAKE_IRON_ARMOUR = 26,
    CRAFTAX_ACH_MAKE_DIAMOND_ARMOUR = 27,
    CRAFTAX_ACH_ENTER_GNOMISH_MINES = 28,
    CRAFTAX_ACH_ENTER_DUNGEON = 29,
    CRAFTAX_ACH_ENTER_SEWERS = 30,
    CRAFTAX_ACH_ENTER_VAULT = 31,
    CRAFTAX_ACH_ENTER_TROLL_MINES = 32,
    CRAFTAX_ACH_ENTER_FIRE_REALM = 33,
    CRAFTAX_ACH_ENTER_ICE_REALM = 34,
    CRAFTAX_ACH_ENTER_GRAVEYARD = 35,
    CRAFTAX_ACH_DEFEAT_GNOME_WARRIOR = 36,
    CRAFTAX_ACH_DEFEAT_GNOME_ARCHER = 37,
    CRAFTAX_ACH_DEFEAT_ORC_SOLIDER = 38,
    CRAFTAX_ACH_DEFEAT_ORC_MAGE = 39,
    CRAFTAX_ACH_DEFEAT_LIZARD = 40,
    CRAFTAX_ACH_DEFEAT_KOBOLD = 41,
    CRAFTAX_ACH_DEFEAT_TROLL = 42,
    CRAFTAX_ACH_DEFEAT_DEEP_THING = 43,
    CRAFTAX_ACH_DEFEAT_PIGMAN = 44,
    CRAFTAX_ACH_DEFEAT_FIRE_ELEMENTAL = 45,
    CRAFTAX_ACH_DEFEAT_FROST_TROLL = 46,
    CRAFTAX_ACH_DEFEAT_ICE_ELEMENTAL = 47,
    CRAFTAX_ACH_DAMAGE_NECROMANCER = 48,
    CRAFTAX_ACH_DEFEAT_NECROMANCER = 49,
    CRAFTAX_ACH_EAT_BAT = 50,
    CRAFTAX_ACH_EAT_SNAIL = 51,
    CRAFTAX_ACH_FIND_BOW = 52,
    CRAFTAX_ACH_FIRE_BOW = 53,
    CRAFTAX_ACH_COLLECT_SAPPHIRE = 54,
    CRAFTAX_ACH_LEARN_FIREBALL = 55,
    CRAFTAX_ACH_CAST_FIREBALL = 56,
    CRAFTAX_ACH_LEARN_ICEBALL = 57,
    CRAFTAX_ACH_CAST_ICEBALL = 58,
    CRAFTAX_ACH_COLLECT_RUBY = 59,
    CRAFTAX_ACH_MAKE_DIAMOND_PICKAXE = 60,
    CRAFTAX_ACH_OPEN_CHEST = 61,
    CRAFTAX_ACH_DRINK_POTION = 62,
    CRAFTAX_ACH_ENCHANT_SWORD = 63,
    CRAFTAX_ACH_ENCHANT_ARMOUR = 64,
    CRAFTAX_ACH_DEFEAT_KNIGHT = 65,
    CRAFTAX_ACH_DEFEAT_ARCHER = 66,
} CraftaxAchievement;

// ============================================================
// State layout declarations matching craftax_state.py field order
// ============================================================
typedef struct CraftaxInventory {
    int32_t wood;
    int32_t stone;
    int32_t coal;
    int32_t iron;
    int32_t diamond;
    int32_t sapling;
    int32_t pickaxe;
    int32_t sword;
    int32_t bow;
    int32_t arrows;
    int32_t armour[4];
    int32_t torches;
    int32_t ruby;
    int32_t sapphire;
    int32_t potions[6];
    int32_t books;
} CraftaxInventory;

typedef struct CraftaxMobs3 {
    int32_t position[CRAFTAX_NUM_LEVELS][3][2];
    float health[CRAFTAX_NUM_LEVELS][3];
    bool mask[CRAFTAX_NUM_LEVELS][3];
    int32_t attack_cooldown[CRAFTAX_NUM_LEVELS][3];
    int32_t type_id[CRAFTAX_NUM_LEVELS][3];
} CraftaxMobs3;

typedef struct CraftaxMobs2 {
    int32_t position[CRAFTAX_NUM_LEVELS][2][2];
    float health[CRAFTAX_NUM_LEVELS][2];
    bool mask[CRAFTAX_NUM_LEVELS][2];
    int32_t attack_cooldown[CRAFTAX_NUM_LEVELS][2];
    int32_t type_id[CRAFTAX_NUM_LEVELS][2];
} CraftaxMobs2;

typedef struct CraftaxState {
    // === Hot data (accessed every step) ===








    int32_t potion_mapping[6];



    int32_t fractal_noise_angles[4];

    // === Medium-hot bitmaps, read during mob updates, spawn scans, encode_obs ===

    // === Cold data (large maps, scattered access) ===
    uint8_t map[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE][CRAFTAX_MAP_SIZE];
    uint8_t item_map[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE][CRAFTAX_MAP_SIZE];
    uint8_t light_map[CRAFTAX_NUM_LEVELS][CRAFTAX_MAP_SIZE][CRAFTAX_MAP_SIZE];

    int32_t down_ladders[CRAFTAX_NUM_LEVELS][2];
    int32_t up_ladders[CRAFTAX_NUM_LEVELS][2];

    // Mirrors CraftaxWorldState: lazy floor generation bookkeeping.
    uint32_t lazy_floor_keys[CRAFTAX_NUM_LEVELS][2];
} CraftaxState;

typedef char CraftaxStateMatchesWorldState[
    (sizeof(CraftaxState) == sizeof(CraftaxWorldState)) ? 1 : -1
];

static __device__ inline uint64_t craftax_spawn_all_bit(uint8_t block) {
    return (uint64_t)(
        block == CRAFTAX_BLOCK_GRASS
        || block == CRAFTAX_BLOCK_PATH
        || block == CRAFTAX_BLOCK_FIRE_GRASS
        || block == CRAFTAX_BLOCK_ICE_GRASS
    );
}

static __device__ inline uint64_t craftax_spawn_grave_bit(uint8_t block) {
    return (uint64_t)(
        block == CRAFTAX_BLOCK_GRAVE
        || block == CRAFTAX_BLOCK_GRAVE2
        || block == CRAFTAX_BLOCK_GRAVE3
    );
}

static __device__ inline uint64_t craftax_spawn_water_bit(uint8_t block) {
    return (uint64_t)(block == CRAFTAX_BLOCK_WATER);
}

static __device__ inline void craftax_refresh_spawn_bits_cell(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col
) {
    uint64_t bit = 1ULL << col;
    uint8_t block = state->map[level][row][col];

    CF_BITS(spawn_all_bits, level, row, state) =
        (CF_BITS(spawn_all_bits, level, row, state) & ~bit)
        | ((0ULL - craftax_spawn_all_bit(block)) & bit);
    CF_BITS(spawn_grave_bits, level, row, state) =
        (CF_BITS(spawn_grave_bits, level, row, state) & ~bit)
        | ((0ULL - craftax_spawn_grave_bit(block)) & bit);
    CF_BITS(spawn_water_bits, level, row, state) =
        (CF_BITS(spawn_water_bits, level, row, state) & ~bit)
        | ((0ULL - craftax_spawn_water_bit(block)) & bit);
}

static __device__ inline void craftax_set_map_block(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col,
    int32_t block
) {
    state->map[level][row][col] = (uint8_t)block;
    craftax_refresh_spawn_bits_cell(state, level, row, col);
}

static __device__ inline void craftax_refresh_spawn_bits_level(
    CraftaxState* state,
    int32_t level
) {
    for (int32_t row = 0; row < CRAFTAX_MAP_SIZE; row++) {
        uint64_t all_bits = 0;
        uint64_t grave_bits = 0;
        uint64_t water_bits = 0;
        for (int32_t col = 0; col < CRAFTAX_MAP_SIZE; col++) {
            uint8_t block = state->map[level][row][col];
            uint64_t bit = 1ULL << col;
            all_bits |= (0ULL - craftax_spawn_all_bit(block)) & bit;
            grave_bits |= (0ULL - craftax_spawn_grave_bit(block)) & bit;
            water_bits |= (0ULL - craftax_spawn_water_bit(block)) & bit;
        }
        CF_BITS(spawn_all_bits, level, row, state) = all_bits;
        CF_BITS(spawn_grave_bits, level, row, state) = grave_bits;
        CF_BITS(spawn_water_bits, level, row, state) = water_bits;
    }
}

// Generate a deferred floor on first visit. No-op when the floor is already
// present (bit clear), so zero-initialized fixture states are unaffected.
static __device__ inline void craftax_ensure_floor_generated(
    CraftaxState* state,
    int32_t level
) {
    if (level < 0 || level >= CRAFTAX_NUM_LEVELS) return;
    uint32_t bit = 1u << (uint32_t)level;
    if (!(CF(lazy_floors_pending, state) & bit)) return;
    CraftaxThreefryKey key = {{
        state->lazy_floor_keys[level][0],
        state->lazy_floor_keys[level][1],
    }};
    craftax_generate_floor_from_key(key, level, (CraftaxWorldState*)(void*)state);
    CF(lazy_floors_pending, state) &= ~bit;
    craftax_refresh_spawn_bits_level(state, level);
}

static __device__ inline void craftax_refresh_spawn_bits_all(CraftaxState* state) {
    for (int32_t level = 0; level < CRAFTAX_NUM_LEVELS; level++) {
        craftax_refresh_spawn_bits_level(state, level);
    }
}

#define CRAFTAX_ARENA_PACKET_SIZE 64

typedef struct CraftaxArena {
    CraftaxState* states;
    int num_envs;
    int packet_size;
    int num_packets;
} CraftaxArena;

#ifdef CRAFTAX_ENABLE_ENV_IMPL
static __device__ inline void craftax_change_floor_native(CraftaxState* state, int32_t action);
static __device__ inline void craftax_do_crafting_native(CraftaxState* state, int32_t action);
static __device__ inline void craftax_do_action_native(
    CraftaxState* state,
    int32_t action,
    CraftaxThreefryKey rng
);
static __device__ inline void craftax_place_block_native(CraftaxState* state, int32_t action);
static __device__ inline void craftax_shoot_projectile_native(
    CraftaxState* state,
    int32_t action
);
static __device__ inline void craftax_cast_spell_native(CraftaxState* state, int32_t action);
static __device__ inline void craftax_drink_potion_native(CraftaxState* state, int32_t action);
static __device__ inline void craftax_read_book_native(
    CraftaxState* state,
    const uint32_t rng_words[2],
    int32_t action
);
static __device__ inline void craftax_enchant_native(
    CraftaxState* state,
    int32_t action,
    CraftaxThreefryKey rng
);
static __device__ inline void craftax_boss_logic_native(CraftaxState* state);
static __device__ inline void craftax_level_up_attributes_native(
    CraftaxState* state,
    int32_t action,
    int32_t max_attribute
);
static __device__ inline void craftax_move_player_native(
    CraftaxState* state,
    int32_t action,
    bool god_mode
);
static __device__ inline void craftax_update_mobs_native(
    CraftaxState* state,
    CraftaxThreefryKey rng
);
static __device__ inline void craftax_spawn_mobs_native(
    CraftaxState* state,
    CraftaxThreefryKey rng
);
static __device__ inline void craftax_update_plants_native(CraftaxState* state);
static __device__ inline void craftax_update_player_intrinsics_native(
    CraftaxState* state,
    int32_t action
);
static __device__ inline void craftax_clip_inventory_and_intrinsics_native(
    CraftaxState* state,
    bool god_mode
);
static __device__ inline void craftax_calculate_inventory_achievements_native(
    CraftaxState* state
);
#endif

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float achievements[CRAFTAX_NUM_ACHIEVEMENTS];
    float n;
} Log;

typedef struct Client {
    int unused;
} Client;

typedef struct Craftax {
    Client* client;
    Log log;
    Log curr_log;  // curriculum-start episodes only (canonical stats stay in log)

    CraftaxObs* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;

    unsigned int rng;
    uint64_t seed;
    CraftaxThreefryKey rng_key;
    CraftaxArena* arena;
    CraftaxState* state;
    int32_t packet_id;
    int32_t lane_id;
    bool owns_state_storage;

    float achievements[CRAFTAX_NUM_ACHIEVEMENTS];
    float episode_return_accum;
    int32_t episode_length_accum;
    int32_t curriculum_episode;  // 1: started pre-equipped on floor>0; excluded from add_log
} Craftax;

#ifdef CRAFTAX_ENABLE_ENV_IMPL

// ============================================================
// Native reset, observation, reward, and step glue
// ============================================================
static __device__ const float CRAFTAX_ACHIEVEMENT_REWARD_MAP[CRAFTAX_NUM_ACHIEVEMENTS] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 5.0f, 5.0f,
    5.0f, 8.0f, 8.0f, 8.0f, 3.0f, 3.0f, 3.0f, 3.0f,
    5.0f, 5.0f, 5.0f, 5.0f, 8.0f, 8.0f, 8.0f, 8.0f,
    8.0f, 8.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 5.0f,
    5.0f, 5.0f, 5.0f, 3.0f, 3.0f, 3.0f, 3.0f, 5.0f,
    5.0f, 5.0f, 5.0f,
};

static __device__ inline CraftaxThreefryKey craftax_step_native_next_key(
    CraftaxThreefryKey* rng
) {
    CraftaxThreefryKey subkey;
    craftax_threefry_split(*rng, rng, &subkey);
    return subkey;
}

// Lazy floor generation toggle (process-wide, like the reset pool). When on,
// resets generate only floor 0 and defer floors 1..8 to first visit. Maps are
// bit-identical to eager generation; only the time of generation changes.
static __device__ int g_craftax_lazy_floors = 0;

// Descend-shaped reward: multiplier on the eight enter-floor achievement
// rewards (CRAFTAX_ACH_ENTER_GNOMISH_MINES..CRAFTAX_ACH_ENTER_GRAVEYARD).
// 1.0 = canonical rewards (hash anchors unchanged).
static __device__ float g_craftax_descend_bonus = 1.0f;

// Curriculum resets (CRAFTAX_CU_CURR_P / CRAFTAX_CU_CURR_MAXF): with
// probability p an episode starts on a uniform floor 1..maxf, shallower
// floors pre-cleared and randomized gear granted, so deep-floor skills get
// dense experience. Applied at the first step of an episode (timestep==0),
// which covers every reset path; the one obs the policy saw at reset is the
// pre-teleport floor-0 obs (1-step glitch, accepted). Curriculum episodes
// train normally but are excluded from add_log so reported metrics stay
// canonical (floor-0 starts only). p=0 (default) is behavior-identical.
static __device__ float g_craftax_curr_p = 0.0f;
static __device__ int g_craftax_curr_maxf = 2;
static __device__ void craftax_apply_curriculum(Craftax* env);

static __device__ inline void craftax_set_lazy_floors(int enabled) {
    g_craftax_lazy_floors = enabled;
}

static __device__ inline void craftax_generate_state_from_world_key(
    CraftaxThreefryKey world_key,
    CraftaxState* out
) {
    // CraftaxState and CraftaxWorldState are layout-identical (see the
    // sizeof static assert), so generate in place instead of filling an
    // 80KB temporary and copying it over.
    craftax_generate_world_from_key_lazy(
        world_key,
        (CraftaxWorldState*)(void*)out,
        g_craftax_lazy_floors != 0);
    // Deferred floors keep zeroed spawn bits; craftax_ensure_floor_generated
    // refreshes them when the floor materializes.
    for (int32_t level = 0; level < CRAFTAX_NUM_LEVELS; level++) {
        if (CF(lazy_floors_pending, out) & (1u << (uint32_t)level)) continue;
        craftax_refresh_spawn_bits_level(out, level);
    }
}

static __device__ inline void craftax_reset_state_from_reset_key(
    CraftaxState* out,
    CraftaxThreefryKey reset_key
) {
    CraftaxThreefryKey unused;
    CraftaxThreefryKey world_key;
    craftax_threefry_split(reset_key, &unused, &world_key);
    craftax_generate_state_from_world_key(world_key, out);
}

// ============================================================
// Reset pool: pre-generate N worlds once, then memcpy on reset.
// Trades world diversity (<= pool_size unique maps per process) for
// ~500x faster reset. Set pool_size=0 to disable (exact per-seed
// world; required for the parity harness).
// ============================================================
static __device__ int g_craftax_reset_pool_size = 0;
static __device__ CraftaxState* g_craftax_reset_pool = NULL;
static __device__ int g_craftax_reset_pool_ready = 0;

// [cuda port] craftax_set_reset_pool_size removed: host-side calloc pool
// machinery is never used by this harness (pools stay disabled, size 0).

static __device__ inline void craftax_ensure_state_storage(Craftax* env) {
    if (env->state != NULL) {
        return;
    }
    // [cuda port] state storage is always preassigned by the host harness;
    // the C file's calloc fallback is unreachable on device.
    __trap();
}

static __device__ inline void craftax_reset_state_from_seed(Craftax* env) {
    craftax_ensure_state_storage(env);
    CraftaxThreefryKey initial_key = craftax_prng_key((uint32_t)env->seed);
    if (g_craftax_reset_pool_size > 0) {
        CraftaxThreefryKey discard;
        craftax_threefry_split(initial_key, &env->rng_key, &discard);
        int idx = (int)(env->seed % (uint64_t)g_craftax_reset_pool_size);
        memcpy(env->state, &g_craftax_reset_pool[idx], sizeof(CraftaxState));
        return;
    }
    CraftaxThreefryKey reset_key;
    craftax_threefry_split(initial_key, &env->rng_key, &reset_key);
    craftax_reset_state_from_reset_key(env->state, reset_key);
}

// ============================================================
// [cuda port] The pipelined pthread world pool and static reset pool from
// craftax_full.c (lines 3616-3744) are host-side conveniences and are
// removed. With pools disabled the hot-path reset is exactly the direct
// per-key generation, matching the C hash-mode configuration.
// ============================================================
static __device__ inline void craftax_reset_state_on_done(
    CraftaxState* out,
    CraftaxThreefryKey reset_key
) {
    craftax_reset_state_from_reset_key(out, reset_key);
}

// ============================================================
// Warp-cooperative world regeneration (lazy mode).
//
// All worldgen randomness is either (a) key-chain derivation — a serial but
// tiny sequence of 64-bit LCG advances, replicated identically on every lane
// — or (b) counter-based draws craftax_threefry_uniform_*_at(key, cell_index)
// that are pure functions of (key, cell). Every per-cell arithmetic
// expression below is copied verbatim from the scalar generators, and
// -fmad=false forbids contraction differences, so a warp generating cells in
// parallel produces bit-identical maps to the single-thread path. min/max
// reductions commute bitwise (no NaNs); the weighted-choice scan is chunked
// so each lane reproduces the scalar cumulative sums exactly.
//
// Scratch (the noise fields the scalar code kept in 46KB of thread-local
// stack) lives in shared memory: one warp per block, ~41KB static shared.
// ============================================================
typedef struct CraftaxWarpScratch {
    float water[CRAFTAX_WG_MAP_CELLS];
    float mountain[CRAFTAX_WG_MAP_CELLS];
    float path_x[CRAFTAX_WG_MAP_CELLS];
    float tree_noise[CRAFTAX_WG_MAP_CELLS];
    float grad_x[176];  // largest gradient grid: res (6,24) -> 7x25 = 175
    float grad_y[176];
    bool lava_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE];
} CraftaxWarpScratch;

#define CRAFTAX_WARP_ALL 0xffffffffu

// Block-width generalizations of the warp primitives used by the worldgen.
// Every reduction here is exact (int sums, float min/max on NaN-free data),
// so reduction order is irrelevant and any NT produces bit-identical
// results. NT=32 keeps the original warp-shuffle forms -- k_mega runs the
// worldgen per-warp inside larger blocks, where __syncthreads is illegal.
// NT>32 runs one env per block (k_reset_list_block) with smem trees: at
// small env counts the reset is a serial-latency tail on the rollout
// critical path (~30 resets across 188 SMs = duration is ONE env's
// worldgen), so more lanes per env cut wall latency directly.
template <int NT>
static __device__ inline void cf_wg_sync(void) {
    if constexpr (NT == 32) __syncwarp();
    else __syncthreads();
}

template <int NT>
static __device__ inline void cf_wg_minmax_f(float* mn, float* mx,
                                             unsigned lane) {
    if constexpr (NT == 32) {
        for (int off = 16; off > 0; off >>= 1) {
            float omn = __shfl_xor_sync(CRAFTAX_WARP_ALL, *mn, off);
            float omx = __shfl_xor_sync(CRAFTAX_WARP_ALL, *mx, off);
            if (omn < *mn) *mn = omn;
            if (omx > *mx) *mx = omx;
        }
    } else {
        __shared__ float smn[NT], smx[NT];
        smn[lane] = *mn;
        smx[lane] = *mx;
        __syncthreads();
        for (int off = NT / 2; off > 0; off >>= 1) {
            if (lane < (unsigned)off) {
                if (smn[lane + off] < smn[lane]) smn[lane] = smn[lane + off];
                if (smx[lane + off] > smx[lane]) smx[lane] = smx[lane + off];
            }
            __syncthreads();
        }
        *mn = smn[0];
        *mx = smx[0];
        __syncthreads();  // smem reused by the next call
    }
}

// Inclusive prefix sum of per-lane int counts plus the block total.
template <int NT>
static __device__ inline void cf_wg_scan_incl(int count, int* incl,
                                              int* total, unsigned lane) {
    if constexpr (NT == 32) {
        int v = count;
        for (int off = 1; off < 32; off <<= 1) {
            int n = __shfl_up_sync(CRAFTAX_WARP_ALL, v, off);
            if (lane >= (unsigned)off) v += n;
        }
        *incl = v;
        *total = __shfl_sync(CRAFTAX_WARP_ALL, v, 31);
    } else {
        __shared__ int sc[NT];
        sc[lane] = count;
        __syncthreads();
        int acc = 0;
        for (int k = 0; k <= (int)lane; k++) acc += sc[k];
        int tot = acc;
        for (int k = (int)lane + 1; k < NT; k++) tot += sc[k];
        *incl = acc;
        *total = tot;
        __syncthreads();
    }
}

template <int NT>
static __device__ inline int cf_wg_min_i(int v, unsigned lane) {
    if constexpr (NT == 32) {
        for (int off = 16; off > 0; off >>= 1) {
            int other = __shfl_xor_sync(CRAFTAX_WARP_ALL, v, off);
            if (other < v) v = other;
        }
        return v;
    } else {
        __shared__ int sv[NT];
        sv[lane] = v;
        __syncthreads();
        for (int off = NT / 2; off > 0; off >>= 1) {
            if (lane < (unsigned)off) {
                if (sv[lane + off] < sv[lane]) sv[lane] = sv[lane + off];
            }
            __syncthreads();
        }
        int r = sv[0];
        __syncthreads();
        return r;
    }
}

template <int NT>
static __device__ inline int cf_wg_max_i(int v, unsigned lane) {
    if constexpr (NT == 32) {
        for (int off = 16; off > 0; off >>= 1) {
            int other = __shfl_xor_sync(CRAFTAX_WARP_ALL, v, off);
            if (other > v) v = other;
        }
        return v;
    } else {
        __shared__ int sv[NT];
        sv[lane] = v;
        __syncthreads();
        for (int off = NT / 2; off > 0; off >>= 1) {
            if (lane < (unsigned)off) {
                if (sv[lane + off] > sv[lane]) sv[lane] = sv[lane + off];
            }
            __syncthreads();
        }
        int r = sv[0];
        __syncthreads();
        return r;
    }
}

// Fractal noise with octaves=1 (every smoothworld call), warp-parallel over
// cells. Reproduces craftax_generate_fractal_noise_2d + perlin_2d_scalar
// bitwise: same key chain, same per-cell expressions, order-free min/max.
template <int NT>
static __device__ inline void craftax_fractal_noise_warp(
    CraftaxThreefryKey rng,
    int res_rows,
    int res_cols,
    float* grad_x,
    float* grad_y,
    float* out,
    unsigned lane
) {
    const int rows = CRAFTAX_WG_MAP_SIZE;
    const int cols = CRAFTAX_WG_MAP_SIZE;
    // fractal: single octave -> one perlin field, then min/max normalize.
    CraftaxThreefryKey next_rng;
    CraftaxThreefryKey noise_key;
    craftax_threefry_split(rng, &next_rng, &noise_key);
    // perlin: derive the angle key exactly as the scalar path.
    CraftaxThreefryKey unused;
    CraftaxThreefryKey angle_key;
    craftax_threefry_split(noise_key, &unused, &angle_key);

    int cell_rows = rows / res_rows;
    int cell_cols = cols / res_cols;
    int grad_w = res_cols + 1;
    int grad_h = res_rows + 1;
    for (int g = (int)lane; g < grad_h * grad_w; g += NT) {
        // craftax_noise_gradient_angle with width == grad_w: index == g.
        float angle = CRAFTAX_NOISE_PI2
            * craftax_threefry_uniform_f32_at(angle_key, (uint64_t)g);
        grad_x[g] = cosf(angle);
        grad_y[g] = sinf(angle);
    }
    cf_wg_sync<NT>();

    float local_min = INFINITY;
    float local_max = -INFINITY;
    for (int i = (int)lane; i < rows * cols; i += NT) {
        int row = i / cols;
        int col = i % cols;
        int grad_row = row / cell_rows;
        float local_row = (float)(row - grad_row * cell_rows) / (float)cell_rows;
        float interp_row = craftax_noise_interpolant(local_row);
        int grad_col = col / cell_cols;
        float local_col = (float)(col - grad_col * cell_cols) / (float)cell_cols;
        float interp_col = craftax_noise_interpolant(local_col);

        int i00 = grad_row * grad_w + grad_col;
        int i10 = i00 + grad_w;
        float g00x = grad_x[i00];     float g00y = grad_y[i00];
        float g10x = grad_x[i10];     float g10y = grad_y[i10];
        float g01x = grad_x[i00 + 1]; float g01y = grad_y[i00 + 1];
        float g11x = grad_x[i10 + 1]; float g11y = grad_y[i10 + 1];

        float n00 = local_row * g00x;
        n00 += local_col * g00y;
        float n10 = (local_row - 1.0f) * g10x;
        n10 += local_col * g10y;
        float n01 = local_row * g01x;
        n01 += (local_col - 1.0f) * g01y;
        float n11 = (local_row - 1.0f) * g11x;
        n11 += (local_col - 1.0f) * g11y;

        float n0 = n00 * (1.0f - interp_row) + interp_row * n10;
        float n1 = n01 * (1.0f - interp_row) + interp_row * n11;
        float perlin =
            CRAFTAX_NOISE_SQRT2 * ((1.0f - interp_col) * n0 + interp_col * n1);
        // fractal accumulate, octave 0: out = 0 + 1.0f * perlin (kept as the
        // same ops so -0.0f flushes to +0.0f exactly like the scalar code).
        float acc = 0.0f;
        acc += 1.0f * perlin;
        out[i] = acc;
        if (acc < local_min) local_min = acc;
        if (acc > local_max) local_max = acc;
    }
    cf_wg_minmax_f<NT>(&local_min, &local_max, lane);
    float scale = local_max - local_min;
    cf_wg_sync<NT>();
    for (int i = (int)lane; i < rows * cols; i += NT) {
        out[i] = (out[i] - local_min) / scale;
    }
    cf_wg_sync<NT>();
}

// Warp version of craftax_choice_bool_flat over the predicate
// map[cell] == target. Chunked so lane L owns cells [CL, CL+C), C =
// CELLS/NT: the lane whose chunk straddles the draw replays the scalar
// cumulative loop with the exact float sums the serial code would have at
// its chunk boundary (the cumsum only ever adds 1.0f to an exact integer,
// so any chunk size reproduces the scalar values bitwise).
template <int NT>
static __device__ inline int craftax_warp_choice_map_eq(
    CraftaxThreefryKey key,
    const uint8_t* map_flat,
    uint8_t target,
    unsigned lane
) {
    const int chunk = CRAFTAX_WG_MAP_CELLS / NT;
    int base = (int)lane * chunk;
    int count = 0;
    int last_valid_local = -1;
    for (int k = 0; k < chunk; k++) {
        if (map_flat[base + k] == target) {
            count++;
            last_valid_local = base + k;
        }
    }
    int incl, total;
    cf_wg_scan_incl<NT>(count, &incl, &total, lane);
    if (total == 0) return 0;

    float draw = (float)total * (1.0f - craftax_threefry_uniform_f32(key));
    int excl = incl - count;
    int result = INT_MAX;
    if ((float)excl < draw && draw <= (float)incl) {
        float cumulative = (float)excl;
        for (int k = 0; k < chunk; k++) {
            if (map_flat[base + k] == target) cumulative += 1.0f;
            if (cumulative >= draw) { result = base + k; break; }
        }
    }
    result = cf_wg_min_i<NT>(result, lane);
    if (result != INT_MAX) return result;
    // Scalar fallback: cumulative never reached draw -> last valid cell.
    int last_valid = cf_wg_max_i<NT>(last_valid_local, lane);
    return last_valid < 0 ? 0 : last_valid;
}

// Warp version of craftax_generate_smoothworld_config: identical key chain
// (replicated on all lanes), per-cell passes strided across the warp.
template <int NT>
static __device__ inline void craftax_smoothworld_config_warp(
    CraftaxThreefryKey rng,
    int config_idx,
    uint8_t map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t item_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    uint8_t light_map[CRAFTAX_WG_MAP_SIZE][CRAFTAX_WG_MAP_SIZE],
    int32_t ladder_down[2],
    int32_t ladder_up[2],
    CraftaxWarpScratch* s,
    unsigned lane
) {
    const CraftaxSmoothGenConfig* config = &CRAFTAX_SMOOTHGEN_CONFIGS[config_idx];
    const int size = CRAFTAX_WG_MAP_SIZE;
    const int player_row = CRAFTAX_WG_MAP_SIZE / 2;
    const int player_col = CRAFTAX_WG_MAP_SIZE / 2;

    CraftaxThreefryKey subkey;
    craftax_threefry_split(rng, &rng, &subkey);
    craftax_fractal_noise_warp<NT>(subkey, 3, 3, s->grad_x, s->grad_y, s->water, lane);

    craftax_threefry_split(rng, &rng, &subkey);
    (void)subkey;

    craftax_threefry_split(rng, &rng, &subkey);
    craftax_fractal_noise_warp<NT>(subkey, 3, 3, s->grad_x, s->grad_y, s->mountain, lane);

    craftax_threefry_split(rng, &rng, &subkey);
    craftax_fractal_noise_warp<NT>(subkey, 6, 24, s->grad_x, s->grad_y, s->path_x, lane);

    craftax_threefry_split(rng, &rng, &subkey);
    (void)subkey;

    craftax_threefry_split(rng, &rng, &subkey);
    CraftaxThreefryKey tree_uniform_key = rng;
    craftax_fractal_noise_warp<NT>(subkey, 12, 12, s->grad_x, s->grad_y, s->tree_noise, lane);

    // classify (verbatim per-cell body from craftax_smoothworld_classify_scalar)
    for (int i = (int)lane; i < size * size; i += NT) {
        int row = i / size;
        int col = i % size;
        int dr = row > player_row ? row - player_row : player_row - row;
        int dc = col > player_col ? col - player_col : player_col - col;
        float distance = sqrtf((float)(dr * dr + dc * dc));
        float proximity_water = craftax_wg_clampf(
            distance / config->player_proximity_map_water_strength,
            0.0f,
            config->player_proximity_map_water_max
        );
        float proximity_mountain = craftax_wg_clampf(
            distance / config->player_proximity_map_mountain_strength,
            0.0f,
            config->player_proximity_map_mountain_max
        );
        size_t idx = craftax_wg_index(row, col);

        s->water[idx] = s->water[idx] + proximity_water - 1.0f;
        int32_t block = s->water[idx] > config->water_threshold
            ? config->sea_block
            : config->default_block;
        bool sand = s->water[idx] > config->sand_threshold && block != config->sea_block;
        if (sand) {
            block = config->coast_block;
        }

        s->mountain[idx] = s->mountain[idx] + 0.05f + proximity_mountain - 1.0f;
        if (s->mountain[idx] > 0.7f) {
            block = config->mountain_block;
        }

        bool path = s->mountain[idx] > 0.7f && s->path_x[idx] > 0.8f;
        if (path) {
            block = config->path_block;
        }

        float path_y = s->path_x[craftax_wg_index(col, row)];
        path = s->mountain[idx] > 0.7f && path_y > 0.8f;
        if (path) {
            block = config->path_block;
        }

        bool cave = s->mountain[idx] > 0.85f && s->water[idx] > 0.4f;
        if (cave) {
            block = config->inner_mountain_block;
        }

        float tree_draw = craftax_threefry_uniform_f32_at(tree_uniform_key, idx);
        bool tree = s->tree_noise[idx] > config->tree_threshold_perlin
            && tree_draw > config->tree_threshold_uniform;
        if (tree && block == config->tree_requirement_block) {
            block = config->tree;
        }

        map[row][col] = (uint8_t)block;
        item_map[row][col] = CRAFTAX_WG_ITEM_NONE;
        light_map[row][col] = (uint8_t)(config->default_light * 255.0f);
    }
    cf_wg_sync<NT>();

    // ores
    CraftaxThreefryKey ore_rng;
    craftax_threefry_split(rng, &rng, &ore_rng);
    for (int ore_index = 0; ore_index < 5; ore_index++) {
        CraftaxThreefryKey ore_key;
        craftax_threefry_split(ore_rng, &ore_rng, &ore_key);
        for (int i = (int)lane; i < size * size; i += NT) {
            int row = i / size;
            int col = i % size;
            size_t idx = craftax_wg_index(row, col);
            bool is_ore = map[row][col] == config->ore_requirement_blocks[ore_index]
                && craftax_threefry_uniform_f32_at(ore_key, idx) < config->ore_chances[ore_index];
            if (is_ore) {
                map[row][col] = (uint8_t)config->ores[ore_index];
            }
        }
        cf_wg_sync<NT>();
    }

    // lava
    for (int i = (int)lane; i < size * size; i += NT) {
        int row = i / size;
        int col = i % size;
        size_t idx = craftax_wg_index(row, col);
        s->lava_map[row][col] = s->mountain[idx] > 0.85f && s->tree_noise[idx] > 0.7f;
        if (s->lava_map[row][col]) {
            map[row][col] = (uint8_t)config->lava;
        }
    }
    cf_wg_sync<NT>();

    // diamond placement (choice over stone cells; the write is a no-op value
    // change but kept for parity with the scalar code)
    craftax_threefry_split(rng, &rng, &subkey);
    int diamond_index = craftax_warp_choice_map_eq<NT>(
        subkey, &map[0][0], (uint8_t)CRAFTAX_WG_BLOCK_STONE, lane);
    if (lane == 0) {
        map[diamond_index / size][diamond_index % size] =
            (uint8_t)CRAFTAX_WG_BLOCK_STONE;
        map[player_row][player_col] = (uint8_t)config->player_spawn;
    }
    cf_wg_sync<NT>();

    // ladders (both draws read the same map state, like the scalar code)
    craftax_threefry_split(rng, &rng, &subkey);
    int ladder_down_index = craftax_warp_choice_map_eq<NT>(
        subkey, &map[0][0], (uint8_t)config->valid_ladder, lane);
    if (lane == 0) {
        ladder_down[0] = ladder_down_index / size;
        ladder_down[1] = ladder_down_index % size;
        if (config->ladder_down) {
            item_map[ladder_down_index / size][ladder_down_index % size] =
                CRAFTAX_WG_ITEM_LADDER_DOWN;
        }
    }

    craftax_threefry_split(rng, &rng, &subkey);
    int ladder_up_index = craftax_warp_choice_map_eq<NT>(
        subkey, &map[0][0], (uint8_t)config->valid_ladder, lane);
    int lu_row = ladder_up_index / size;
    int lu_col = ladder_up_index % size;
    if (lane == 0) {
        ladder_up[0] = lu_row;
        ladder_up[1] = lu_col;
    }
    cf_wg_sync<NT>();

    // craftax_apply_ladder_light, 9x9 patch strided across the warp
    {
        int start_row = lu_row - 4;
        int start_col = lu_col - 4;
        if (start_row < 0) start_row += CRAFTAX_WG_MAP_SIZE;
        if (start_col < 0) start_col += CRAFTAX_WG_MAP_SIZE;
        start_row = craftax_wg_clampi(start_row, 0, CRAFTAX_WG_MAP_SIZE - 9);
        start_col = craftax_wg_clampi(start_col, 0, CRAFTAX_WG_MAP_SIZE - 9);
        for (int p = (int)lane; p < 81; p += NT) {
            int row = p / 9;
            int col = p % 9;
            light_map[start_row + row][start_col + col] = (uint8_t)(
                craftax_torch_light_value(row, col, config->default_light) * 255.0f);
        }
    }
    cf_wg_sync<NT>();

    // craftax_add_lava_light, per-cell over the shared lava map
    if (config->lava == CRAFTAX_WG_BLOCK_LAVA) {
        static const float kernel[3][3] = {
            {0.2f, 0.7f, 0.2f},
            {0.7f, 1.0f, 0.7f},
            {0.2f, 0.7f, 0.2f},
        };
        for (int i = (int)lane; i < size * size; i += NT) {
            int row = i / size;
            int col = i % size;
            float add = 0.0f;
            for (int kr = 0; kr < 3; kr++) {
                int src_row = row + kr - 1;
                if (src_row < 0 || src_row >= CRAFTAX_WG_MAP_SIZE) continue;
                for (int kc = 0; kc < 3; kc++) {
                    int src_col = col + kc - 1;
                    if (src_col < 0 || src_col >= CRAFTAX_WG_MAP_SIZE) continue;
                    add += s->lava_map[src_row][src_col] ? kernel[kr][kc] : 0.0f;
                }
            }
            float new_light = craftax_wg_clampf(
                light_map[row][col] * (1.0f / 255.0f) + add, 0.0f, 1.0f);
            light_map[row][col] = (uint8_t)(new_light * 255.0f);
        }
    }
    cf_wg_sync<NT>();

    if (config->ladder_up && lane == 0) {
        item_map[lu_row][lu_col] = CRAFTAX_WG_ITEM_LADDER_UP;
    }
    cf_wg_sync<NT>();
}

// Warp version of craftax_generate_world_from_key_lazy(lazy=true) plus the
// spawn-bit refresh from craftax_generate_state_from_world_key.
template <int NT>
static __device__ inline void craftax_generate_state_from_world_key_warp(
    CraftaxThreefryKey rng,
    CraftaxState* state,
    CraftaxWarpScratch* s,
    unsigned lane
) {
    CraftaxWorldState* out = (CraftaxWorldState*)(void*)state;

    // cooperative memset(out, 0, sizeof(*out))
    {
        uint32_t* words = (uint32_t*)(void*)out;
        const size_t nwords = sizeof(*out) / sizeof(uint32_t);
        for (size_t i = lane; i < nwords; i += NT) words[i] = 0u;
        if (lane == 0) {
            uint8_t* bytes = (uint8_t*)(void*)out;
            for (size_t b = nwords * sizeof(uint32_t); b < sizeof(*out); b++) {
                bytes[b] = 0;
            }
        }
    }
    // Levels never generated during the dying episode still hold their
    // post-reset values (mask/pos/type/cd/bits 0, health 1.0f, dirs 1), so
    // only generated levels (prev pending bit clear) need re-clearing.
    const uint32_t cf_gen_mask =
        ~CF(lazy_floors_pending, out) & 0x1FFu;
    cf_wg_sync<NT>();
    cf_soa_zero_env_warp_lazy<NT>(out, lane, cf_gen_mask);
    cf_wg_sync<NT>();

    CraftaxThreefryKey smooth_split[7];
    craftax_threefry_split_n(rng, smooth_split, 7);
    rng = smooth_split[0];

    static const int smooth_floor_order[6] = {0, 2, 5, 6, 7, 8};
    if (lane == 0) {
        for (int i = 0; i < 6; i++) {
            int level = smooth_floor_order[i];
            out->lazy_floor_keys[level][0] = smooth_split[i + 1].word[0];
            out->lazy_floor_keys[level][1] = smooth_split[i + 1].word[1];
        }
    }
    // floor 0 (smooth config 0), warp-cooperative; floors 1..8 stay deferred
    craftax_smoothworld_config_warp<NT>(
        smooth_split[1], 0,
        out->map[0], out->item_map[0], out->light_map[0],
        out->down_ladders[0], out->up_ladders[0], s, lane);

    CraftaxThreefryKey dungeon_split[4];
    craftax_threefry_split_n(rng, dungeon_split, 4);
    rng = dungeon_split[0];

    static const int dungeon_floor_order[3] = {1, 3, 4};
    if (lane == 0) {
        for (int i = 0; i < 3; i++) {
            int level = dungeon_floor_order[i];
            out->lazy_floor_keys[level][0] = dungeon_split[i + 1].word[0];
            out->lazy_floor_keys[level][1] = dungeon_split[i + 1].word[1];
        }
        CF(lazy_floors_pending, out) = 0x1FEu;  // floors 1..8 deferred
    }
    // warp-distributed: same lane-per-entry stride as the zero pass above
    cf_init_empty_mobs_warp<NT>(out, lane, cf_gen_mask);
    for (int j = (int)lane; j < 54; j += NT) {
        if (((cf_gen_mask >> (j / 6)) & 1u) != 0u) {
            CF2(mob_projectile_directions, j, out) = 1;
            CF2(player_projectile_directions, j, out) = 1;
        }
    }

    CraftaxThreefryKey potion_key;
    craftax_threefry_split(rng, &rng, &potion_key);
    CraftaxThreefryKey state_key;
    craftax_threefry_split(rng, &rng, &state_key);
    (void)rng;
    if (lane == 0) {
        craftax_permutation_6(potion_key, out->potion_mapping);
        CF2(state_rng, 0, out) = state_key.word[0];
        CF2(state_rng, 1, out) = state_key.word[1];

        CF2(monsters_killed, 0, out) = 10;
        CF2(player_position, 0, out) = CRAFTAX_WG_MAP_SIZE / 2;
        CF2(player_position, 1, out) = CRAFTAX_WG_MAP_SIZE / 2;
        CF(player_level, out) = 0;
        CF(player_direction, out) = CRAFTAX_WG_ACTION_UP;
        CF(player_health, out) = 9.0f;
        CF(player_food, out) = 9;
        CF(player_drink, out) = 9;
        CF(player_energy, out) = 9;
        CF(player_mana, out) = 9;
        CF(player_dexterity, out) = 1;
        CF(player_strength, out) = 1;
        CF(player_intelligence, out) = 1;
        CF(boss_timesteps_to_spawn_this_round, out) = CRAFTAX_WG_BOSS_FIGHT_SPAWN_TURNS;
        CF(light_level, out) = craftax_calculate_initial_light_level();
    }
    cf_wg_sync<NT>();

    // spawn bits for the one generated floor, one row per lane
    for (int row = (int)lane; row < CRAFTAX_MAP_SIZE; row += NT) {
        uint64_t all_bits = 0;
        uint64_t grave_bits = 0;
        uint64_t water_bits = 0;
        for (int32_t col = 0; col < CRAFTAX_MAP_SIZE; col++) {
            uint8_t block = state->map[0][row][col];
            uint64_t bit = 1ULL << col;
            all_bits |= (0ULL - craftax_spawn_all_bit(block)) & bit;
            grave_bits |= (0ULL - craftax_spawn_grave_bit(block)) & bit;
            water_bits |= (0ULL - craftax_spawn_water_bit(block)) & bit;
        }
        CF_BITS(spawn_all_bits, 0, row, state) = all_bits;
        CF_BITS(spawn_grave_bits, 0, row, state) = grave_bits;
        CF_BITS(spawn_water_bits, 0, row, state) = water_bits;
    }
    cf_wg_sync<NT>();
}

// Full on-done reset, warp-cooperative: mirrors craftax_reset_state_on_done
// -> craftax_reset_state_from_reset_key (lazy path).
template <int NT>
static __device__ inline void craftax_reset_state_on_done_warp(
    CraftaxState* state,
    CraftaxThreefryKey reset_key,
    CraftaxWarpScratch* s,
    unsigned lane
) {
    CraftaxThreefryKey unused;
    CraftaxThreefryKey world_key;
    craftax_threefry_split(reset_key, &unused, &world_key);
    craftax_generate_state_from_world_key_warp<NT>(world_key, state, s, lane);
}

static __device__ inline void craftax_encode_native_observation(
    const CraftaxState* state,
    CraftaxObs* obs
) {
    if (obs == NULL) {
        return;
    }
#ifdef CRAFTAX_COMPACT_OBS
    craftax_encode_compact_observation((const CraftaxWorldState*)(const void*)state, obs);
#else
    craftax_encode_reset_observation((const CraftaxWorldState*)(const void*)state, obs);
#endif
}

static __device__ inline float craftax_calculate_light_level_native(int32_t timestep) {
    // [cuda port] host-computed table; reference formula:
    //   progress = fmodf(timestep * (1.0f / 300.0f), 1.0f) + 0.3f
    //   1.0f - powf(fabsf(cosf(pi * progress)), 3.0f)
    return g_craftax_light_table[timestep];
}

static __device__ inline bool craftax_is_game_over_native(const CraftaxState* state) {
    return CF(timestep, state) >= CRAFTAX_DEFAULT_MAX_TIMESTEPS
        || CF(player_health, state) <= 0.0f;
}

static __device__ inline void craftax_copy_achievements_to_env(
    Craftax* env,
    const CraftaxState* state
) {
    for (int i = 0; i < CRAFTAX_NUM_ACHIEVEMENTS; i++) {
        env->achievements[i] = CF2(achievements, i, state) ? 1.0f : 0.0f;
    }
}

static __device__ void add_log(Craftax* env) {
    Log* log = env->curriculum_episode ? &env->curr_log : &env->log;
    int unlocked = 0;
    for (int i = 0; i < CRAFTAX_NUM_ACHIEVEMENTS; i++) {
        if (env->achievements[i] > 0.5f) {
            unlocked++;
            log->achievements[i] += 1.0f;
        }
    }
    log->perf += (float)unlocked / (float)CRAFTAX_NUM_ACHIEVEMENTS;
    log->score += env->episode_return_accum;
    log->episode_return += env->episode_return_accum;
    log->episode_length += (float)env->episode_length_accum;
    log->n += 1.0f;
}

static __device__ float craftax_gameplay_step_native(
    CraftaxState* state,
    int32_t action,
    CraftaxThreefryKey rng
) {
    CRAFTAX_PROFILE_START();
    bool init_achievements[CRAFTAX_NUM_ACHIEVEMENTS];
    for (int i = 0; i < CRAFTAX_NUM_ACHIEVEMENTS; i++) {
        init_achievements[i] = CF2(achievements, i, state);
    }
    float init_health = CF(player_health, state);

    action = CF(is_sleeping, state) ? CRAFTAX_ACTION_NOOP : action;
    action = CF(is_resting, state) ? CRAFTAX_ACTION_NOOP : action;

    CRAFTAX_PROFILE_ZONE(0);
    craftax_change_floor_native(state, action);
    craftax_do_crafting_native(state, action);
    CRAFTAX_PROFILE_END(0);

    CraftaxThreefryKey subkey = craftax_step_native_next_key(&rng);
    CRAFTAX_PROFILE_ZONE(2);
    craftax_do_action_native(state, action, subkey);
    CRAFTAX_PROFILE_END(2);

    CRAFTAX_PROFILE_ZONE(3);
    craftax_place_block_native(state, action);
    craftax_shoot_projectile_native(state, action);
    craftax_cast_spell_native(state, action);
    craftax_drink_potion_native(state, action);
    CRAFTAX_PROFILE_END(3);

    subkey = craftax_step_native_next_key(&rng);
    CRAFTAX_PROFILE_ZONE(4);
    craftax_read_book_native(state, subkey.word, action);
    CRAFTAX_PROFILE_END(4);

    subkey = craftax_step_native_next_key(&rng);
    CRAFTAX_PROFILE_ZONE(5);
    craftax_enchant_native(state, action, subkey);
    CRAFTAX_PROFILE_END(5);

    CRAFTAX_PROFILE_ZONE(6);
    craftax_boss_logic_native(state);
    craftax_level_up_attributes_native(state, action, CRAFTAX_MAX_ATTRIBUTE);
    craftax_move_player_native(state, action, false);
    CRAFTAX_PROFILE_END(6);

    subkey = craftax_step_native_next_key(&rng);
    CRAFTAX_PROFILE_ZONE(7);
    craftax_update_mobs_native(state, subkey);
    CRAFTAX_PROFILE_END(7);

    subkey = craftax_step_native_next_key(&rng);
    CRAFTAX_PROFILE_ZONE(8);
    craftax_spawn_mobs_native(state, subkey);
    CRAFTAX_PROFILE_END(8);

    CRAFTAX_PROFILE_ZONE(9);
    craftax_update_plants_native(state);
    craftax_update_player_intrinsics_native(state, action);
    craftax_clip_inventory_and_intrinsics_native(state, false);
    craftax_calculate_inventory_achievements_native(state);
    CRAFTAX_PROFILE_END(9);

    CRAFTAX_PROFILE_ZONE(10);
    float reward = 0.0f;
    for (int i = 0; i < CRAFTAX_NUM_ACHIEVEMENTS; i++) {
        int32_t delta = (int32_t)CF2(achievements, i, state)
            - (int32_t)init_achievements[i];
        float w = CRAFTAX_ACHIEVEMENT_REWARD_MAP[i];
        if (i >= CRAFTAX_ACH_ENTER_GNOMISH_MINES
            && i <= CRAFTAX_ACH_ENTER_GRAVEYARD) {
            w *= g_craftax_descend_bonus;
        }
        reward += (float)delta * w;
    }
    reward += (CF(player_health, state) - init_health) * 0.1f;

    subkey = craftax_step_native_next_key(&rng);
    CF(timestep, state) += 1;
    CF(light_level, state) = craftax_calculate_light_level_native(CF(timestep, state));
    CF2(state_rng, 0, state) = subkey.word[0];
    CF2(state_rng, 1, state) = subkey.word[1];
    CRAFTAX_PROFILE_END(10);

    return reward;
}

// ============================================================
// Public API expected by vecenv.h
// ============================================================
static __device__ void c_init(Craftax* env) {
    env->client = NULL;
    env->num_agents = 1;
    craftax_ensure_state_storage(env);
    env->episode_return_accum = 0.0f;
    env->episode_length_accum = 0;
    memset(env->achievements, 0, sizeof(env->achievements));
    memset(&env->log, 0, sizeof(env->log));
    craftax_wg_init_cell_templates();
    craftax_reset_state_from_seed(env);
}

static __device__ void c_reset(Craftax* env) {
    if (env->rewards != NULL) {
        env->rewards[0] = 0.0f;
    }
    if (env->terminals != NULL) {
        env->terminals[0] = 0.0f;
    }
    env->episode_return_accum = 0.0f;
    env->episode_length_accum = 0;
    memset(env->achievements, 0, sizeof(env->achievements));

    craftax_reset_state_from_seed(env);
    craftax_encode_native_observation(env->state, env->observations);
}

#ifdef CRAFTAX_PROFILE
static __device__ void c_step_native(Craftax* env) {
    CRAFTAX_PROFILE_START();
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    int action = (int)env->actions[0];
    if (action < 0) {
        action = CRAFTAX_ACTION_NOOP;
    }
    if (action >= CRAFTAX_NUM_ACTIONS) {
        action = CRAFTAX_NUM_ACTIONS - 1;
    }

    CRAFTAX_PROFILE_ZONE(12);
    CraftaxThreefryKey step_key;
    craftax_threefry_split(env->rng_key, &env->rng_key, &step_key);

    CraftaxThreefryKey step_rng;
    CraftaxThreefryKey reset_key;
    craftax_threefry_split(step_key, &step_rng, &reset_key);
    CRAFTAX_PROFILE_END(12);

    if (g_craftax_curr_p > 0.0f && CF(timestep, env->state) == 0) {
        craftax_apply_curriculum(env);
    }
    float reward = craftax_gameplay_step_native(env->state, action, step_rng);

    CRAFTAX_PROFILE_ZONE(13);
    bool done = craftax_is_game_over_native(env->state);
    CRAFTAX_PROFILE_END(13);

    CRAFTAX_PROFILE_ZONE(15);
    craftax_copy_achievements_to_env(env, env->state);
    CRAFTAX_PROFILE_END(15);

    CRAFTAX_PROFILE_ZONE(16);
    env->rewards[0] = reward;
    env->terminals[0] = done ? 1.0f : 0.0f;
    env->episode_return_accum += reward;
    env->episode_length_accum += 1;
    CRAFTAX_PROFILE_END(16);

    if (done) {
        add_log(env);
        env->episode_return_accum = 0.0f;
        env->episode_length_accum = 0;
        memset(env->achievements, 0, sizeof(env->achievements));
        CRAFTAX_PROFILE_ZONE(14);
        craftax_reset_state_on_done(env->state, reset_key);
        CRAFTAX_PROFILE_END(14);
    }

    CRAFTAX_PROFILE_ZONE(11);
    craftax_encode_native_observation(env->state, env->observations);
    CRAFTAX_PROFILE_END(11);

    // Record unprofiled time
    CRAFTAX_PROFILE_ZONE(17);
    CRAFTAX_PROFILE_END(17);

#ifdef CRAFTAX_PROFILE
    static int profile_step_count = 0;
    profile_step_count++;
    if (profile_step_count >= 100000) {
        craftax_profile_report();
        profile_step_count = 0;
    }

#endif
}

#endif

// Gameplay step with the on-done world regeneration factored out: performs
// everything c_step_gameplay does (including the done-side log/accumulator
// bookkeeping) EXCEPT craftax_reset_state_on_done. Returns whether the env
// finished and the reset key it must be regenerated with — identical key
// derivation, so running the regeneration later (in a dedicated kernel) is
// bit-exact vs the inline version.
static __device__ bool c_step_gameplay_core(
    Craftax* env, CraftaxThreefryKey* reset_key_out
) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    int action = (int)env->actions[0];
    if (action < 0) action = CRAFTAX_ACTION_NOOP;
    if (action >= CRAFTAX_NUM_ACTIONS) action = CRAFTAX_NUM_ACTIONS - 1;

    CraftaxThreefryKey step_key;
    craftax_threefry_split(env->rng_key, &env->rng_key, &step_key);
    CraftaxThreefryKey step_rng;
    CraftaxThreefryKey reset_key;
    craftax_threefry_split(step_key, &step_rng, &reset_key);

    if (g_craftax_curr_p > 0.0f && CF(timestep, env->state) == 0) {
        craftax_apply_curriculum(env);
    }
    float reward = craftax_gameplay_step_native(env->state, action, step_rng);
    bool done = craftax_is_game_over_native(env->state);
    craftax_copy_achievements_to_env(env, env->state);

    env->rewards[0] = reward;
    env->terminals[0] = done ? 1.0f : 0.0f;
    env->episode_return_accum += reward;
    env->episode_length_accum += 1;

    if (done) {
        add_log(env);
        env->episode_return_accum = 0.0f;
        env->episode_length_accum = 0;
        memset(env->achievements, 0, sizeof(env->achievements));
    }
    *reset_key_out = reset_key;
    return done;
}

static __device__ void c_step_gameplay(Craftax* env) {
    CraftaxThreefryKey reset_key;
    if (c_step_gameplay_core(env, &reset_key)) {
        craftax_reset_state_on_done(env->state, reset_key);
    }
}

static __device__ void c_step_encode(Craftax* env) {
    craftax_encode_native_observation(env->state, env->observations);
}

static __device__ void c_step(Craftax* env) {
    c_step_gameplay(env);
    c_step_encode(env);
}

static __device__ void c_close(Craftax* env) {
    if (!env->owns_state_storage || env->arena == NULL) {
        return;
    }
    free(env->arena->states);
    free(env->arena);
    env->arena = NULL;
    env->state = NULL;
    env->owns_state_storage = false;
}

#endif

// ============================================================
// ===== step_simple.h =====
// ============================================================
// Standalone native ports of simple Craftax step subsystems.
//
// These helpers intentionally are not integrated into c_step yet. They mutate a
// full CraftaxState in place so tests can compare each subsystem directly
// against the installed JAX implementation.



static __device__ inline int32_t craftax_step_jax_index(int32_t index, int32_t size) {
    if (index < 0) {
        index += size;
    }
    if (index < 0) {
        return 0;
    }
    if (index >= size) {
        return size - 1;
    }
    return index;
}

static __device__ inline int32_t craftax_step_mini32(int32_t a, int32_t b) {
    return a < b ? a : b;
}

static __device__ inline int32_t craftax_step_maxi32(int32_t a, int32_t b) {
    return a > b ? a : b;
}

static __device__ inline float craftax_step_minf32(float a, float b) {
    if (isnan(a) || isnan(b)) {
        return NAN;
    }
    return a < b ? a : b;
}

static __device__ inline float craftax_step_maxf32(float a, float b) {
    if (isnan(a) || isnan(b)) {
        return NAN;
    }
    return a > b ? a : b;
}

static __device__ inline int32_t craftax_step_get_max_health(const CraftaxState* state) {
    return 8 + CF(player_strength, state);
}

static __device__ inline int32_t craftax_step_get_max_food(const CraftaxState* state) {
    return 7 + 2 * CF(player_dexterity, state);
}

static __device__ inline int32_t craftax_step_get_max_drink(const CraftaxState* state) {
    return 7 + 2 * CF(player_dexterity, state);
}

static __device__ inline int32_t craftax_step_get_max_energy(const CraftaxState* state) {
    return 7 + 2 * CF(player_dexterity, state);
}

static __device__ inline int32_t craftax_step_get_max_mana(const CraftaxState* state) {
    return 6 + 3 * CF(player_intelligence, state);
}

static __device__ inline bool craftax_step_is_fighting_boss(const CraftaxState* state) {
    return CF(player_level, state) == CRAFTAX_NUM_LEVELS - 1;
}

static __device__ inline bool craftax_step_has_beaten_boss(const CraftaxState* state) {
    return CF(boss_progress, state) >= CRAFTAX_NUM_LEVELS - 1;
}

static __device__ inline void craftax_step_direction(int32_t action, int32_t direction[2]) {
    direction[0] = 0;
    direction[1] = 0;
    int32_t direction_index = craftax_step_jax_index(action, 16);
    if (direction_index == CRAFTAX_ACTION_LEFT) {
        direction[1] = -1;
    } else if (direction_index == CRAFTAX_ACTION_RIGHT) {
        direction[1] = 1;
    } else if (direction_index == CRAFTAX_ACTION_UP) {
        direction[0] = -1;
    } else if (direction_index == CRAFTAX_ACTION_DOWN) {
        direction[0] = 1;
    }
}

static __device__ inline bool craftax_step_is_solid_block(int32_t block) {
    switch (block) {
    case CRAFTAX_BLOCK_STONE:
    case CRAFTAX_BLOCK_TREE:
    case CRAFTAX_BLOCK_COAL:
    case CRAFTAX_BLOCK_IRON:
    case CRAFTAX_BLOCK_DIAMOND:
    case CRAFTAX_BLOCK_CRAFTING_TABLE:
    case CRAFTAX_BLOCK_FURNACE:
    case CRAFTAX_BLOCK_PLANT:
    case CRAFTAX_BLOCK_RIPE_PLANT:
    case CRAFTAX_BLOCK_WALL:
    case CRAFTAX_BLOCK_WALL_MOSS:
    case CRAFTAX_BLOCK_STALAGMITE:
    case CRAFTAX_BLOCK_RUBY:
    case CRAFTAX_BLOCK_SAPPHIRE:
    case CRAFTAX_BLOCK_CHEST:
    case CRAFTAX_BLOCK_FOUNTAIN:
    case CRAFTAX_BLOCK_FIRE_TREE:
    case CRAFTAX_BLOCK_ENCHANTMENT_TABLE_FIRE:
    case CRAFTAX_BLOCK_ENCHANTMENT_TABLE_ICE:
    case CRAFTAX_BLOCK_GRAVE:
    case CRAFTAX_BLOCK_GRAVE2:
    case CRAFTAX_BLOCK_GRAVE3:
    case CRAFTAX_BLOCK_NECROMANCER:
        return true;
    default:
        return false;
    }
}

static __device__ inline bool craftax_step_is_in_mob(
    const CraftaxState* state,
    int32_t row,
    int32_t col
) {
    int32_t level = craftax_step_jax_index(CF(player_level, state), CRAFTAX_NUM_LEVELS);
    int32_t map_row = craftax_step_jax_index(row, CRAFTAX_MAP_SIZE);
    int32_t map_col = craftax_step_jax_index(col, CRAFTAX_MAP_SIZE);
    bool player_here = CF2(player_position, 0, state) == row
        && CF2(player_position, 1, state) == col;
    return ((CF_BITS(mob_bits, level, map_row, state) >> map_col) & 1ULL) || player_here;
}

static __device__ inline bool craftax_step_valid_land_position(
    const CraftaxState* state,
    int32_t row,
    int32_t col
) {
    bool pos_in_bounds = row >= 0
        && row < CRAFTAX_MAP_SIZE
        && col >= 0
        && col < CRAFTAX_MAP_SIZE;
    int32_t level = craftax_step_jax_index(CF(player_level, state), CRAFTAX_NUM_LEVELS);
    int32_t map_row = craftax_step_jax_index(row, CRAFTAX_MAP_SIZE);
    int32_t map_col = craftax_step_jax_index(col, CRAFTAX_MAP_SIZE);
    int32_t block = state->map[level][map_row][map_col];
    bool in_solid_block = craftax_step_is_solid_block(block);
    bool in_mob = craftax_step_is_in_mob(state, row, col);
    bool in_lava = block == CRAFTAX_BLOCK_LAVA;
    bool in_water = block == CRAFTAX_BLOCK_WATER;

    bool valid_move = pos_in_bounds && !in_mob && !in_solid_block;
    valid_move = valid_move && !in_water;
    valid_move = valid_move && !in_lava;
    return valid_move;
}

static __device__ inline void craftax_move_player_native(
    CraftaxState* state,
    int32_t action,
    bool god_mode
) {
    int32_t direction[2];
    craftax_step_direction(action, direction);

    int32_t proposed_row = CF2(player_position, 0, state) + direction[0];
    int32_t proposed_col = CF2(player_position, 1, state) + direction[1];
    bool valid_move = craftax_step_valid_land_position(
        state,
        proposed_row,
        proposed_col
    );
    valid_move = valid_move || god_mode;

    CF2(player_position, 0, state) += (int32_t)valid_move * direction[0];
    CF2(player_position, 1, state) += (int32_t)valid_move * direction[1];

    bool is_new_direction = direction[0] != 0 || direction[1] != 0;
    CF(player_direction, state) = CF(player_direction, state) * (1 - (int32_t)is_new_direction)
        + action * (int32_t)is_new_direction;
}

static __device__ inline void craftax_update_plants_native(CraftaxState* state) {
    bool finished_growing_plants[CRAFTAX_MAX_GROWING_PLANTS];

    for (int plant = 0; plant < CRAFTAX_MAX_GROWING_PLANTS; plant++) {
        CF2(growing_plants_age, plant, state) =
            (CF2(growing_plants_age, plant, state) + 1)
            * (int32_t)CF2(growing_plants_mask, plant, state);
        finished_growing_plants[plant] = CF2(growing_plants_age, plant, state) >= 600;
    }

    for (int plant = 0; plant < CRAFTAX_MAX_GROWING_PLANTS; plant++) {
        int32_t row = craftax_step_jax_index(
            CF2(growing_plants_positions, (plant) * 2 + (0), state),
            CRAFTAX_MAP_SIZE
        );
        int32_t col = craftax_step_jax_index(
            CF2(growing_plants_positions, (plant) * 2 + (1), state),
            CRAFTAX_MAP_SIZE
        );
        int32_t new_block = finished_growing_plants[plant]
            ? CRAFTAX_BLOCK_RIPE_PLANT
            : state->map[0][row][col];
        craftax_set_map_block(state, 0, row, col, new_block);
    }
}

static __device__ inline void craftax_boss_logic_native(CraftaxState* state) {
    CF2(achievements, CRAFTAX_ACH_DEFEAT_NECROMANCER, state) =
        CF2(achievements, CRAFTAX_ACH_DEFEAT_NECROMANCER, state)
        || craftax_step_has_beaten_boss(state);
    CF(boss_timesteps_to_spawn_this_round, state) -=
        (int32_t)craftax_step_is_fighting_boss(state);
}

static __device__ inline void craftax_level_up_attributes_native(
    CraftaxState* state,
    int32_t action,
    int32_t max_attribute
) {
    bool can_level_up = CF(player_xp, state) >= 1;
    bool is_levelling_up_dex = can_level_up
        && action == CRAFTAX_ACTION_LEVEL_UP_DEXTERITY
        && CF(player_dexterity, state) < max_attribute;
    bool is_levelling_up_str = can_level_up
        && action == CRAFTAX_ACTION_LEVEL_UP_STRENGTH
        && CF(player_strength, state) < max_attribute;
    bool is_levelling_up_int = can_level_up
        && action == CRAFTAX_ACTION_LEVEL_UP_INTELLIGENCE
        && CF(player_intelligence, state) < max_attribute;
    bool is_levelling_up = is_levelling_up_dex
        || is_levelling_up_str
        || is_levelling_up_int;

    CF(player_dexterity, state) += (int32_t)is_levelling_up_dex;
    CF(player_strength, state) += (int32_t)is_levelling_up_str;
    CF(player_intelligence, state) += (int32_t)is_levelling_up_int;
    CF(player_xp, state) -= (int32_t)is_levelling_up;
}

static __device__ inline void craftax_clip_inventory_and_intrinsics_native(
    CraftaxState* state,
    bool god_mode
) {
    CF(inv_wood, state) = craftax_step_mini32(CF(inv_wood, state), 99);
    CF(inv_stone, state) = craftax_step_mini32(CF(inv_stone, state), 99);
    CF(inv_coal, state) = craftax_step_mini32(CF(inv_coal, state), 99);
    CF(inv_iron, state) = craftax_step_mini32(CF(inv_iron, state), 99);
    CF(inv_diamond, state) = craftax_step_mini32(CF(inv_diamond, state), 99);
    CF(inv_sapling, state) = craftax_step_mini32(CF(inv_sapling, state), 99);
    CF(inv_pickaxe, state) = craftax_step_mini32(CF(inv_pickaxe, state), 99);
    CF(inv_sword, state) = craftax_step_mini32(CF(inv_sword, state), 99);
    CF(inv_bow, state) = craftax_step_mini32(CF(inv_bow, state), 99);
    CF(inv_arrows, state) = craftax_step_mini32(CF(inv_arrows, state), 99);
    for (int i = 0; i < 4; i++) {
        CF2(inv_armour, i, state) = craftax_step_mini32(
            CF2(inv_armour, i, state),
            99
        );
    }
    CF(inv_torches, state) = craftax_step_mini32(CF(inv_torches, state), 99);
    CF(inv_ruby, state) = craftax_step_mini32(CF(inv_ruby, state), 99);
    CF(inv_sapphire, state) = craftax_step_mini32(CF(inv_sapphire, state), 99);
    for (int i = 0; i < 6; i++) {
        CF2(inv_potions, i, state) = craftax_step_mini32(
            CF2(inv_potions, i, state),
            99
        );
    }
    CF(inv_books, state) = craftax_step_mini32(CF(inv_books, state), 99);

    float min_health = god_mode ? 9.0f : 0.0f;
    CF(player_health, state) = craftax_step_minf32(
        craftax_step_maxf32(CF(player_health, state), min_health),
        (float)craftax_step_get_max_health(state)
    );
    CF(player_food, state) = craftax_step_mini32(
        craftax_step_maxi32(CF(player_food, state), 0),
        craftax_step_get_max_food(state)
    );
    CF(player_drink, state) = craftax_step_mini32(
        craftax_step_maxi32(CF(player_drink, state), 0),
        craftax_step_get_max_drink(state)
    );
    CF(player_energy, state) = craftax_step_mini32(
        craftax_step_maxi32(CF(player_energy, state), 0),
        craftax_step_get_max_energy(state)
    );
    CF(player_mana, state) = craftax_step_mini32(
        craftax_step_maxi32(CF(player_mana, state), 0),
        craftax_step_get_max_mana(state)
    );
}

static __device__ inline void craftax_calculate_inventory_achievements_native(
    CraftaxState* state
) {
    CF2(achievements, CRAFTAX_ACH_COLLECT_WOOD, state) =
        CF2(achievements, CRAFTAX_ACH_COLLECT_WOOD, state) || CF(inv_wood, state) > 0;
    CF2(achievements, CRAFTAX_ACH_COLLECT_STONE, state) =
        CF2(achievements, CRAFTAX_ACH_COLLECT_STONE, state) || CF(inv_stone, state) > 0;
    CF2(achievements, CRAFTAX_ACH_COLLECT_COAL, state) =
        CF2(achievements, CRAFTAX_ACH_COLLECT_COAL, state) || CF(inv_coal, state) > 0;
    CF2(achievements, CRAFTAX_ACH_COLLECT_IRON, state) =
        CF2(achievements, CRAFTAX_ACH_COLLECT_IRON, state) || CF(inv_iron, state) > 0;
    CF2(achievements, CRAFTAX_ACH_COLLECT_DIAMOND, state) =
        CF2(achievements, CRAFTAX_ACH_COLLECT_DIAMOND, state) || CF(inv_diamond, state) > 0;
    CF2(achievements, CRAFTAX_ACH_COLLECT_RUBY, state) =
        CF2(achievements, CRAFTAX_ACH_COLLECT_RUBY, state) || CF(inv_ruby, state) > 0;
    CF2(achievements, CRAFTAX_ACH_COLLECT_SAPPHIRE, state) =
        CF2(achievements, CRAFTAX_ACH_COLLECT_SAPPHIRE, state)
        || CF(inv_sapphire, state) > 0;
    CF2(achievements, CRAFTAX_ACH_COLLECT_SAPLING, state) =
        CF2(achievements, CRAFTAX_ACH_COLLECT_SAPLING, state)
        || CF(inv_sapling, state) > 0;
    CF2(achievements, CRAFTAX_ACH_FIND_BOW, state) =
        CF2(achievements, CRAFTAX_ACH_FIND_BOW, state) || CF(inv_bow, state) > 0;
    CF2(achievements, CRAFTAX_ACH_MAKE_ARROW, state) =
        CF2(achievements, CRAFTAX_ACH_MAKE_ARROW, state) || CF(inv_arrows, state) > 0;
    CF2(achievements, CRAFTAX_ACH_MAKE_TORCH, state) =
        CF2(achievements, CRAFTAX_ACH_MAKE_TORCH, state) || CF(inv_torches, state) > 0;

    CF2(achievements, CRAFTAX_ACH_MAKE_WOOD_PICKAXE, state) =
        CF2(achievements, CRAFTAX_ACH_MAKE_WOOD_PICKAXE, state)
        || CF(inv_pickaxe, state) >= 1;
    CF2(achievements, CRAFTAX_ACH_MAKE_STONE_PICKAXE, state) =
        CF2(achievements, CRAFTAX_ACH_MAKE_STONE_PICKAXE, state)
        || CF(inv_pickaxe, state) >= 2;
    CF2(achievements, CRAFTAX_ACH_MAKE_IRON_PICKAXE, state) =
        CF2(achievements, CRAFTAX_ACH_MAKE_IRON_PICKAXE, state)
        || CF(inv_pickaxe, state) >= 3;
    CF2(achievements, CRAFTAX_ACH_MAKE_DIAMOND_PICKAXE, state) =
        CF2(achievements, CRAFTAX_ACH_MAKE_DIAMOND_PICKAXE, state)
        || CF(inv_pickaxe, state) >= 4;

    CF2(achievements, CRAFTAX_ACH_MAKE_WOOD_SWORD, state) =
        CF2(achievements, CRAFTAX_ACH_MAKE_WOOD_SWORD, state)
        || CF(inv_sword, state) >= 1;
    CF2(achievements, CRAFTAX_ACH_MAKE_STONE_SWORD, state) =
        CF2(achievements, CRAFTAX_ACH_MAKE_STONE_SWORD, state)
        || CF(inv_sword, state) >= 2;
    CF2(achievements, CRAFTAX_ACH_MAKE_IRON_SWORD, state) =
        CF2(achievements, CRAFTAX_ACH_MAKE_IRON_SWORD, state)
        || CF(inv_sword, state) >= 3;
    CF2(achievements, CRAFTAX_ACH_MAKE_DIAMOND_SWORD, state) =
        CF2(achievements, CRAFTAX_ACH_MAKE_DIAMOND_SWORD, state)
        || CF(inv_sword, state) >= 4;
}

static __device__ inline void craftax_update_player_intrinsics_native(
    CraftaxState* state,
    int32_t action
) {
    bool is_starting_sleep = action == CRAFTAX_ACTION_SLEEP
        && CF(player_energy, state) < craftax_step_get_max_energy(state);
    CF(is_sleeping, state) = CF(is_sleeping, state) || is_starting_sleep;

    bool is_waking_up = CF(player_energy, state) >= craftax_step_get_max_energy(state)
        && CF(is_sleeping, state);
    CF(is_sleeping, state) = CF(is_sleeping, state) && !is_waking_up;
    CF2(achievements, CRAFTAX_ACH_WAKE_UP, state) =
        CF2(achievements, CRAFTAX_ACH_WAKE_UP, state) || is_waking_up;

    bool is_starting_rest = action == CRAFTAX_ACTION_REST
        && CF(player_health, state) < (float)craftax_step_get_max_health(state);
    CF(is_resting, state) = CF(is_resting, state) || is_starting_rest;

    is_waking_up = CF(is_resting, state)
        && (
            CF(player_health, state) >= (float)craftax_step_get_max_health(state)
            || CF(player_food, state) <= 0
            || CF(player_drink, state) <= 0
        );
    CF(is_resting, state) = CF(is_resting, state) && !is_waking_up;

    bool not_boss = !craftax_step_is_fighting_boss(state);
    float intrinsic_decay_coeff =
        1.0f - (0.125f * (float)(CF(player_dexterity, state) - 1));

    float hunger_add = (CF(is_sleeping, state) ? 0.5f : 1.0f) * intrinsic_decay_coeff;
    float new_hunger = CF(player_hunger, state) + hunger_add;
    int32_t hungered_food = craftax_step_maxi32(
        CF(player_food, state) - (int32_t)not_boss,
        0
    );
    int32_t new_food = new_hunger > 25.0f ? hungered_food : CF(player_food, state);
    new_hunger = new_hunger > 25.0f ? 0.0f : new_hunger;
    CF(player_hunger, state) = new_hunger;
    CF(player_food, state) = new_food;

    float thirst_add = (CF(is_sleeping, state) ? 0.5f : 1.0f) * intrinsic_decay_coeff;
    float new_thirst = CF(player_thirst, state) + thirst_add;
    int32_t thirsted_drink = craftax_step_maxi32(
        CF(player_drink, state) - (int32_t)not_boss,
        0
    );
    int32_t new_drink = new_thirst > 20.0f ? thirsted_drink : CF(player_drink, state);
    new_thirst = new_thirst > 20.0f ? 0.0f : new_thirst;
    CF(player_thirst, state) = new_thirst;
    CF(player_drink, state) = new_drink;

    float new_fatigue = CF(is_sleeping, state)
        ? craftax_step_minf32(CF(player_fatigue, state) - 1.0f, 0.0f)
        : CF(player_fatigue, state) + intrinsic_decay_coeff;
    int32_t new_energy = new_fatigue > 30.0f
        ? craftax_step_maxi32(CF(player_energy, state) - (int32_t)not_boss, 0)
        : CF(player_energy, state);
    new_fatigue = new_fatigue > 30.0f ? 0.0f : new_fatigue;
    new_energy = new_fatigue < -10.0f
        ? craftax_step_mini32(
            CF(player_energy, state) + 1,
            craftax_step_get_max_energy(state)
        )
        : new_energy;
    new_fatigue = new_fatigue < -10.0f ? 0.0f : new_fatigue;
    CF(player_fatigue, state) = new_fatigue;
    CF(player_energy, state) = new_energy;

    bool all_necessities = CF(player_food, state) > 0
        && CF(player_drink, state) > 0
        && (CF(player_energy, state) > 0 || CF(is_sleeping, state));
    float recover_all = CF(is_sleeping, state) ? 2.0f : 1.0f;
    float recover_not_all = (CF(is_sleeping, state) ? -0.5f : -1.0f)
        * (float)(int32_t)not_boss;
    float recover_add = all_necessities ? recover_all : recover_not_all;
    float new_recover = CF(player_recover, state) + recover_add;

    float recovered_health = craftax_step_minf32(
        CF(player_health, state) + 1.0f,
        (float)craftax_step_get_max_health(state)
    );
    float derecovered_health = CF(player_health, state) - 1.0f;
    float new_health = new_recover > 25.0f
        ? recovered_health
        : CF(player_health, state);
    new_recover = new_recover > 25.0f ? 0.0f : new_recover;
    new_health = new_recover < -15.0f ? derecovered_health : new_health;
    new_recover = new_recover < -15.0f ? 0.0f : new_recover;
    CF(player_recover, state) = new_recover;
    CF(player_health, state) = new_health;

    float mana_recover_coeff =
        1.0f + 0.25f * (float)(CF(player_intelligence, state) - 1);
    float new_recover_mana = (
        CF(is_sleeping, state)
            ? CF(player_recover_mana, state) + 2.0f
            : CF(player_recover_mana, state) + 1.0f
    ) * mana_recover_coeff;
    int32_t new_mana = new_recover_mana > 30.0f
        ? CF(player_mana, state) + 1
        : CF(player_mana, state);
    new_recover_mana = new_recover_mana > 30.0f ? 0.0f : new_recover_mana;
    CF(player_recover_mana, state) = new_recover_mana;
    CF(player_mana, state) = new_mana;
}

static __device__ inline void craftax_drink_potion_native(
    CraftaxState* state,
    int32_t action
) {
    int32_t drinking_potion_index = -1;
    bool is_drinking_potion = false;

    bool is_drinking_red_potion = action == CRAFTAX_ACTION_DRINK_POTION_RED
        && CF2(inv_potions, 0, state) > 0;
    drinking_potion_index = (int32_t)is_drinking_red_potion * 0
        + (1 - (int32_t)is_drinking_red_potion) * drinking_potion_index;
    is_drinking_potion = is_drinking_potion || is_drinking_red_potion;

    bool is_drinking_green_potion = action == CRAFTAX_ACTION_DRINK_POTION_GREEN
        && CF2(inv_potions, 1, state) > 0;
    drinking_potion_index = (int32_t)is_drinking_green_potion * 1
        + (1 - (int32_t)is_drinking_green_potion) * drinking_potion_index;
    is_drinking_potion = is_drinking_potion || is_drinking_green_potion;

    bool is_drinking_blue_potion = action == CRAFTAX_ACTION_DRINK_POTION_BLUE
        && CF2(inv_potions, 2, state) > 0;
    drinking_potion_index = (int32_t)is_drinking_blue_potion * 2
        + (1 - (int32_t)is_drinking_blue_potion) * drinking_potion_index;
    is_drinking_potion = is_drinking_potion || is_drinking_blue_potion;

    bool is_drinking_pink_potion = action == CRAFTAX_ACTION_DRINK_POTION_PINK
        && CF2(inv_potions, 3, state) > 0;
    drinking_potion_index = (int32_t)is_drinking_pink_potion * 3
        + (1 - (int32_t)is_drinking_pink_potion) * drinking_potion_index;
    is_drinking_potion = is_drinking_potion || is_drinking_pink_potion;

    bool is_drinking_cyan_potion = action == CRAFTAX_ACTION_DRINK_POTION_CYAN
        && CF2(inv_potions, 4, state) > 0;
    drinking_potion_index = (int32_t)is_drinking_cyan_potion * 4
        + (1 - (int32_t)is_drinking_cyan_potion) * drinking_potion_index;
    is_drinking_potion = is_drinking_potion || is_drinking_cyan_potion;

    bool is_drinking_yellow_potion = action == CRAFTAX_ACTION_DRINK_POTION_YELLOW
        && CF2(inv_potions, 5, state) > 0;
    drinking_potion_index = (int32_t)is_drinking_yellow_potion * 5
        + (1 - (int32_t)is_drinking_yellow_potion) * drinking_potion_index;
    is_drinking_potion = is_drinking_potion || is_drinking_yellow_potion;

    int32_t potion_index = craftax_step_jax_index(drinking_potion_index, 6);
    int32_t potion_effect_index = state->potion_mapping[potion_index];

    int32_t delta_health = 0;
    delta_health += (int32_t)is_drinking_potion * (int32_t)(potion_effect_index == 0) * 8;
    delta_health += (int32_t)is_drinking_potion * (int32_t)(potion_effect_index == 1) * -3;

    int32_t delta_mana = 0;
    delta_mana += (int32_t)is_drinking_potion * (int32_t)(potion_effect_index == 2) * 8;
    delta_mana += (int32_t)is_drinking_potion * (int32_t)(potion_effect_index == 3) * -3;

    int32_t delta_energy = 0;
    delta_energy += (int32_t)is_drinking_potion * (int32_t)(potion_effect_index == 4) * 8;
    delta_energy += (int32_t)is_drinking_potion * (int32_t)(potion_effect_index == 5) * -3;

    CF2(achievements, CRAFTAX_ACH_DRINK_POTION, state) =
        CF2(achievements, CRAFTAX_ACH_DRINK_POTION, state) || is_drinking_potion;
    CF2(inv_potions, potion_index, state) =
        CF2(inv_potions, potion_index, state) - (int32_t)is_drinking_potion;
    CF(player_health, state) += (float)delta_health;
    CF(player_mana, state) += delta_mana;
    CF(player_energy, state) += delta_energy;
}

static __device__ inline void craftax_read_book_native(
    CraftaxState* state,
    const uint32_t rng_words[2],
    int32_t action
) {
    bool is_reading_book = action == CRAFTAX_ACTION_READ_BOOK
        && CF(inv_books, state) > 0;

    CraftaxThreefryKey rng = {{rng_words[0], rng_words[1]}};
    CraftaxThreefryKey unused;
    CraftaxThreefryKey choice_key;
    craftax_threefry_split(rng, &unused, &choice_key);

    float p0 = CF2(learned_spells, 0, state) ? 0.0f : 1.0f;
    float p1 = CF2(learned_spells, 1, state) ? 0.0f : 1.0f;
    float p_sum = p0 + p1;
    int32_t spell_to_learn_index = 0;
    if (p_sum != 0.0f) {
        p0 /= p_sum;
        float r = 1.0f - craftax_threefry_uniform_f32(choice_key);
        spell_to_learn_index = r <= p0 ? 0 : 1;
    }

    int32_t learn_spell_achievement = spell_to_learn_index
        ? CRAFTAX_ACH_LEARN_ICEBALL
        : CRAFTAX_ACH_LEARN_FIREBALL;

    CF2(achievements, learn_spell_achievement, state) =
        CF2(achievements, learn_spell_achievement, state) || is_reading_book;
    CF(inv_books, state) -= (int32_t)is_reading_book;
    CF2(learned_spells, spell_to_learn_index, state) =
        CF2(learned_spells, spell_to_learn_index, state) || is_reading_book;
}

// ============================================================
// ===== step_crafting.h =====
// ============================================================
// Standalone native ports of Craftax crafting and placement subsystems.
//
// These helpers intentionally are not integrated into c_step yet. They mutate a
// full CraftaxState in place so tests can compare each subsystem directly
// against the installed JAX implementation.



static __device__ inline bool craftax_crafting_is_near_block(
    const CraftaxState* state,
    int32_t block_type
) {
    static const int32_t close_blocks[8][2] = {
        {0, -1},
        {0, 1},
        {-1, 0},
        {1, 0},
        {-1, -1},
        {-1, 1},
        {1, -1},
        {1, 1},
    };

    int32_t level = craftax_step_jax_index(
        CF(player_level, state),
        CRAFTAX_NUM_LEVELS
    );
    for (int32_t i = 0; i < 8; i++) {
        int32_t row = CF2(player_position, 0, state) + close_blocks[i][0];
        int32_t col = CF2(player_position, 1, state) + close_blocks[i][1];
        bool in_bounds = row >= 0
            && row < CRAFTAX_MAP_SIZE
            && col >= 0
            && col < CRAFTAX_MAP_SIZE;
        if (in_bounds && state->map[level][row][col] == block_type) {
            return true;
        }
    }
    return false;
}

static __device__ inline int32_t craftax_crafting_first_armour_below(
    const void* inventory,
    int32_t threshold,
    int32_t* count
) {
    int32_t first = 0;
    *count = 0;
    for (int32_t i = 0; i < 4; i++) {
        bool below = CF2(inv_armour, i, inventory) < threshold;
        first = (*count == 0 && below) ? i : first;
        *count += (int32_t)below;
    }
    return first;
}

static __device__ inline void craftax_do_crafting_native(
    CraftaxState* state,
    int32_t action
) {
    bool is_at_crafting_table = craftax_crafting_is_near_block(
        state,
        CRAFTAX_BLOCK_CRAFTING_TABLE
    );
    bool is_at_furnace = craftax_crafting_is_near_block(
        state,
        CRAFTAX_BLOCK_FURNACE
    );

    const void* const inventory = (const void*)state;

    bool can_craft_wood_pickaxe = CF(inv_wood, inventory) >= 1;
    bool is_crafting_wood_pickaxe =
        action == CRAFTAX_ACTION_MAKE_WOOD_PICKAXE
        && can_craft_wood_pickaxe
        && is_at_crafting_table
        && CF(inv_pickaxe, inventory) < 1;
    CF(inv_wood, inventory) -= 1 * (int32_t)is_crafting_wood_pickaxe;
    CF(inv_pickaxe, inventory) =
        CF(inv_pickaxe, inventory) * (1 - (int32_t)is_crafting_wood_pickaxe)
        + 1 * (int32_t)is_crafting_wood_pickaxe;

    bool can_craft_stone_pickaxe =
        CF(inv_wood, inventory) >= 1 && CF(inv_stone, inventory) >= 1;
    bool is_crafting_stone_pickaxe =
        action == CRAFTAX_ACTION_MAKE_STONE_PICKAXE
        && can_craft_stone_pickaxe
        && is_at_crafting_table
        && CF(inv_pickaxe, inventory) < 2;
    CF(inv_stone, inventory) -= 1 * (int32_t)is_crafting_stone_pickaxe;
    CF(inv_wood, inventory) -= 1 * (int32_t)is_crafting_stone_pickaxe;
    CF(inv_pickaxe, inventory) =
        CF(inv_pickaxe, inventory) * (1 - (int32_t)is_crafting_stone_pickaxe)
        + 2 * (int32_t)is_crafting_stone_pickaxe;

    bool can_craft_iron_pickaxe =
        CF(inv_wood, inventory) >= 1
        && CF(inv_stone, inventory) >= 1
        && CF(inv_iron, inventory) >= 1
        && CF(inv_coal, inventory) >= 1;
    bool is_crafting_iron_pickaxe =
        action == CRAFTAX_ACTION_MAKE_IRON_PICKAXE
        && can_craft_iron_pickaxe
        && is_at_furnace
        && is_at_crafting_table
        && CF(inv_pickaxe, inventory) < 3;
    CF(inv_iron, inventory) -= 1 * (int32_t)is_crafting_iron_pickaxe;
    CF(inv_wood, inventory) -= 1 * (int32_t)is_crafting_iron_pickaxe;
    CF(inv_stone, inventory) -= 1 * (int32_t)is_crafting_iron_pickaxe;
    CF(inv_coal, inventory) -= 1 * (int32_t)is_crafting_iron_pickaxe;
    CF(inv_pickaxe, inventory) =
        CF(inv_pickaxe, inventory) * (1 - (int32_t)is_crafting_iron_pickaxe)
        + 3 * (int32_t)is_crafting_iron_pickaxe;

    bool can_craft_diamond_pickaxe =
        CF(inv_wood, inventory) >= 1 && CF(inv_diamond, inventory) >= 3;
    bool is_crafting_diamond_pickaxe =
        action == CRAFTAX_ACTION_MAKE_DIAMOND_PICKAXE
        && can_craft_diamond_pickaxe
        && is_at_crafting_table
        && CF(inv_pickaxe, inventory) < 4;
    CF(inv_diamond, inventory) -= 3 * (int32_t)is_crafting_diamond_pickaxe;
    CF(inv_wood, inventory) -= 1 * (int32_t)is_crafting_diamond_pickaxe;
    CF(inv_pickaxe, inventory) =
        CF(inv_pickaxe, inventory) * (1 - (int32_t)is_crafting_diamond_pickaxe)
        + 4 * (int32_t)is_crafting_diamond_pickaxe;

    bool can_craft_wood_sword = CF(inv_wood, inventory) >= 1;
    bool is_crafting_wood_sword =
        action == CRAFTAX_ACTION_MAKE_WOOD_SWORD
        && can_craft_wood_sword
        && is_at_crafting_table
        && CF(inv_sword, inventory) < 1;
    CF(inv_wood, inventory) -= 1 * (int32_t)is_crafting_wood_sword;
    CF(inv_sword, inventory) =
        CF(inv_sword, inventory) * (1 - (int32_t)is_crafting_wood_sword)
        + 1 * (int32_t)is_crafting_wood_sword;

    bool can_craft_stone_sword =
        CF(inv_stone, inventory) >= 1 && CF(inv_wood, inventory) >= 1;
    bool is_crafting_stone_sword =
        action == CRAFTAX_ACTION_MAKE_STONE_SWORD
        && can_craft_stone_sword
        && is_at_crafting_table
        && CF(inv_sword, inventory) < 2;
    CF(inv_wood, inventory) -= 1 * (int32_t)is_crafting_stone_sword;
    CF(inv_stone, inventory) -= 1 * (int32_t)is_crafting_stone_sword;
    CF(inv_sword, inventory) =
        CF(inv_sword, inventory) * (1 - (int32_t)is_crafting_stone_sword)
        + 2 * (int32_t)is_crafting_stone_sword;

    bool can_craft_iron_sword =
        CF(inv_iron, inventory) >= 1
        && CF(inv_wood, inventory) >= 1
        && CF(inv_stone, inventory) >= 1
        && CF(inv_coal, inventory) >= 1;
    bool is_crafting_iron_sword =
        action == CRAFTAX_ACTION_MAKE_IRON_SWORD
        && can_craft_iron_sword
        && is_at_furnace
        && is_at_crafting_table
        && CF(inv_sword, inventory) < 3;
    CF(inv_wood, inventory) -= 1 * (int32_t)is_crafting_iron_sword;
    CF(inv_iron, inventory) -= 1 * (int32_t)is_crafting_iron_sword;
    CF(inv_stone, inventory) -= 1 * (int32_t)is_crafting_iron_sword;
    CF(inv_coal, inventory) -= 1 * (int32_t)is_crafting_iron_sword;
    CF(inv_sword, inventory) =
        CF(inv_sword, inventory) * (1 - (int32_t)is_crafting_iron_sword)
        + 3 * (int32_t)is_crafting_iron_sword;

    bool can_craft_diamond_sword =
        CF(inv_diamond, inventory) >= 2 && CF(inv_wood, inventory) >= 1;
    bool is_crafting_diamond_sword =
        action == CRAFTAX_ACTION_MAKE_DIAMOND_SWORD
        && can_craft_diamond_sword
        && is_at_crafting_table
        && CF(inv_sword, inventory) < 4;
    CF(inv_wood, inventory) -= 1 * (int32_t)is_crafting_diamond_sword;
    CF(inv_diamond, inventory) -= 2 * (int32_t)is_crafting_diamond_sword;
    CF(inv_sword, inventory) =
        CF(inv_sword, inventory) * (1 - (int32_t)is_crafting_diamond_sword)
        + 4 * (int32_t)is_crafting_diamond_sword;

    int32_t armour_count = 0;
    int32_t iron_armour_index_to_craft =
        craftax_crafting_first_armour_below(inventory, 1, &armour_count);
    bool can_craft_iron_armour =
        armour_count > 0 && CF(inv_iron, inventory) >= 3 && CF(inv_coal, inventory) >= 3;
    bool is_crafting_iron_armour =
        action == CRAFTAX_ACTION_MAKE_IRON_ARMOUR
        && can_craft_iron_armour
        && is_at_crafting_table
        && is_at_furnace;
    CF(inv_iron, inventory) -= 3 * (int32_t)is_crafting_iron_armour;
    CF(inv_coal, inventory) -= 3 * (int32_t)is_crafting_iron_armour;
    CF2(inv_armour, iron_armour_index_to_craft, inventory) =
        (int32_t)is_crafting_iron_armour * 1
        + (1 - (int32_t)is_crafting_iron_armour)
        * CF2(inv_armour, iron_armour_index_to_craft, inventory);
    CF2(achievements, CRAFTAX_ACH_MAKE_IRON_ARMOUR, state) =
        CF2(achievements, CRAFTAX_ACH_MAKE_IRON_ARMOUR, state)
        || is_crafting_iron_armour;

    int32_t diamond_armour_count = 0;
    int32_t diamond_armour_index_to_craft =
        craftax_crafting_first_armour_below(inventory, 2, &diamond_armour_count);
    bool can_craft_diamond_armour =
        diamond_armour_count > 0 && CF(inv_diamond, inventory) >= 3;
    bool is_crafting_diamond_armour =
        action == CRAFTAX_ACTION_MAKE_DIAMOND_ARMOUR
        && can_craft_diamond_armour
        && is_at_crafting_table;
    CF(inv_diamond, inventory) -= 3 * (int32_t)is_crafting_diamond_armour;
    CF2(inv_armour, diamond_armour_index_to_craft, inventory) =
        (int32_t)is_crafting_diamond_armour * 2
        + (1 - (int32_t)is_crafting_diamond_armour)
        * CF2(inv_armour, diamond_armour_index_to_craft, inventory);
    CF2(achievements, CRAFTAX_ACH_MAKE_DIAMOND_ARMOUR, state) =
        CF2(achievements, CRAFTAX_ACH_MAKE_DIAMOND_ARMOUR, state)
        || is_crafting_diamond_armour;

    bool can_craft_arrow = CF(inv_stone, inventory) >= 1 && CF(inv_wood, inventory) >= 1;
    bool is_crafting_arrow =
        action == CRAFTAX_ACTION_MAKE_ARROW
        && can_craft_arrow
        && is_at_crafting_table
        && CF(inv_arrows, inventory) < 99;
    CF(inv_wood, inventory) -= 1 * (int32_t)is_crafting_arrow;
    CF(inv_stone, inventory) -= 1 * (int32_t)is_crafting_arrow;
    CF(inv_arrows, inventory) += 2 * (int32_t)is_crafting_arrow;

    bool can_craft_torch = CF(inv_coal, inventory) >= 1 && CF(inv_wood, inventory) >= 1;
    bool is_crafting_torch =
        action == CRAFTAX_ACTION_MAKE_TORCH
        && can_craft_torch
        && is_at_crafting_table
        && CF(inv_torches, inventory) < 99;
    CF(inv_wood, inventory) -= 1 * (int32_t)is_crafting_torch;
    CF(inv_coal, inventory) -= 1 * (int32_t)is_crafting_torch;
    CF(inv_torches, inventory) += 4 * (int32_t)is_crafting_torch;
}

static __device__ inline bool craftax_crafting_can_place_item(int32_t block) {
    switch (block) {
    case CRAFTAX_BLOCK_GRASS:
    case CRAFTAX_BLOCK_SAND:
    case CRAFTAX_BLOCK_PATH:
    case CRAFTAX_BLOCK_FIRE_GRASS:
    case CRAFTAX_BLOCK_ICE_GRASS:
        return true;
    default:
        return false;
    }
}

static __device__ inline float craftax_crafting_torch_light(int32_t row, int32_t col) {
    static const float torch_light_map[9][9] = {
        {0.0f, 0.0f, 0.10557288f, 0.17537886f, 0.19999999f, 0.17537886f, 0.10557288f, 0.0f, 0.0f},
        {0.0f, 0.15147191f, 0.27888972f, 0.36754447f, 0.39999998f, 0.36754447f, 0.27888972f, 0.15147191f, 0.0f},
        {0.10557288f, 0.27888972f, 0.43431455f, 0.55278647f, 0.6f, 0.55278647f, 0.43431455f, 0.27888972f, 0.10557288f},
        {0.17537886f, 0.36754447f, 0.55278647f, 0.71715724f, 0.8f, 0.71715724f, 0.55278647f, 0.36754447f, 0.17537886f},
        {0.19999999f, 0.39999998f, 0.6f, 0.8f, 1.0f, 0.8f, 0.6f, 0.39999998f, 0.19999999f},
        {0.17537886f, 0.36754447f, 0.55278647f, 0.71715724f, 0.8f, 0.71715724f, 0.55278647f, 0.36754447f, 0.17537886f},
        {0.10557288f, 0.27888972f, 0.43431455f, 0.55278647f, 0.6f, 0.55278647f, 0.43431455f, 0.27888972f, 0.10557288f},
        {0.0f, 0.15147191f, 0.27888972f, 0.36754447f, 0.39999998f, 0.36754447f, 0.27888972f, 0.15147191f, 0.0f},
        {0.0f, 0.0f, 0.10557288f, 0.17537886f, 0.19999999f, 0.17537886f, 0.10557288f, 0.0f, 0.0f},
    };
    return torch_light_map[row][col];
}

static __device__ inline void craftax_crafting_add_torch_light(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col
) {
    for (int32_t dr = -4; dr <= 4; dr++) {
        int32_t map_row = row + dr;
        if (map_row < 0 || map_row >= CRAFTAX_MAP_SIZE) {
            continue;
        }
        for (int32_t dc = -4; dc <= 4; dc++) {
            int32_t map_col = col + dc;
            if (map_col < 0 || map_col >= CRAFTAX_MAP_SIZE) {
                continue;
            }
            float light = state->light_map[level][map_row][map_col] * (1.0f / 255.0f)
                + craftax_crafting_torch_light(dr + 4, dc + 4);
            state->light_map[level][map_row][map_col] =
                (uint8_t)(craftax_step_minf32(craftax_step_maxf32(light, 0.0f), 1.0f) * 255.0f);
        }
    }
}

static __device__ inline void craftax_add_new_growing_plant_native(
    CraftaxState* state,
    const int32_t position[2],
    bool is_placing_sapling
) {
    int32_t plant_index = 0;
    int32_t empty_count = 0;
    for (int32_t i = 0; i < CRAFTAX_MAX_GROWING_PLANTS; i++) {
        bool is_empty = !CF2(growing_plants_mask, i, state);
        plant_index = (empty_count == 0 && is_empty) ? i : plant_index;
        empty_count += (int32_t)is_empty;
    }

    bool is_adding_plant = empty_count > 0 && is_placing_sapling;
    if (!is_adding_plant) {
        return;
    }

    CF2(growing_plants_positions, (plant_index) * 2 + (0), state) = position[0];
    CF2(growing_plants_positions, (plant_index) * 2 + (1), state) = position[1];
    CF2(growing_plants_age, plant_index, state) = 0;
    CF2(growing_plants_mask, plant_index, state) = true;
}

static __device__ inline void craftax_place_block_native(
    CraftaxState* state,
    int32_t action
) {
    int32_t direction[2];
    craftax_step_direction(CF(player_direction, state), direction);

    int32_t row = CF2(player_position, 0, state) + direction[0];
    int32_t col = CF2(player_position, 1, state) + direction[1];
    bool in_bounds = row >= 0
        && row < CRAFTAX_MAP_SIZE
        && col >= 0
        && col < CRAFTAX_MAP_SIZE;
    bool in_mob = in_bounds && craftax_step_is_in_mob(state, row, col);
    if (!in_bounds || in_mob) {
        return;
    }

    int32_t level = craftax_step_jax_index(
        CF(player_level, state),
        CRAFTAX_NUM_LEVELS
    );
    int32_t original_block = state->map[level][row][col];
    int32_t original_item = state->item_map[level][row][col];
    bool is_placement_on_solid_block_or_item =
        craftax_step_is_solid_block(original_block)
        || original_item != CRAFTAX_ITEM_NONE;

    const void* const inventory = (const void*)state;

    bool is_placing_crafting_table =
        action == CRAFTAX_ACTION_PLACE_TABLE
        && !is_placement_on_solid_block_or_item
        && CF(inv_wood, inventory) >= 2;
    if (is_placing_crafting_table) {
        craftax_set_map_block(state, level, row, col, CRAFTAX_BLOCK_CRAFTING_TABLE);
    }
    CF(inv_wood, inventory) -= 2 * (int32_t)is_placing_crafting_table;
    CF2(achievements, CRAFTAX_ACH_PLACE_TABLE, state) =
        CF2(achievements, CRAFTAX_ACH_PLACE_TABLE, state)
        || is_placing_crafting_table;

    bool is_placing_furnace =
        action == CRAFTAX_ACTION_PLACE_FURNACE
        && !is_placement_on_solid_block_or_item
        && CF(inv_stone, inventory) > 0;
    if (is_placing_furnace) {
        craftax_set_map_block(state, level, row, col, CRAFTAX_BLOCK_FURNACE);
    }
    CF(inv_stone, inventory) -= 1 * (int32_t)is_placing_furnace;
    CF2(achievements, CRAFTAX_ACH_PLACE_FURNACE, state) =
        CF2(achievements, CRAFTAX_ACH_PLACE_FURNACE, state)
        || is_placing_furnace;

    bool is_placing_on_valid_stone_block =
        original_block == CRAFTAX_BLOCK_WATER
        || !is_placement_on_solid_block_or_item;
    bool is_placing_stone =
        action == CRAFTAX_ACTION_PLACE_STONE
        && is_placing_on_valid_stone_block
        && CF(inv_stone, inventory) > 0;
    if (is_placing_stone) {
        craftax_set_map_block(state, level, row, col, CRAFTAX_BLOCK_STONE);
    }
    CF(inv_stone, inventory) -= 1 * (int32_t)is_placing_stone;
    CF2(achievements, CRAFTAX_ACH_PLACE_STONE, state) =
        CF2(achievements, CRAFTAX_ACH_PLACE_STONE, state)
        || is_placing_stone;

    bool is_placing_on_valid_torch_block =
        craftax_crafting_can_place_item(original_block)
        && state->item_map[level][row][col] == CRAFTAX_ITEM_NONE;
    bool is_placing_torch =
        action == CRAFTAX_ACTION_PLACE_TORCH
        && is_placing_on_valid_torch_block
        && CF(inv_torches, inventory) > 0;
    if (is_placing_torch) {
        state->item_map[level][row][col] = CRAFTAX_ITEM_TORCH;
        craftax_crafting_add_torch_light(state, level, row, col);
    }
    CF(inv_torches, inventory) -= 1 * (int32_t)is_placing_torch;
    CF2(achievements, CRAFTAX_ACH_PLACE_TORCH, state) =
        CF2(achievements, CRAFTAX_ACH_PLACE_TORCH, state)
        || is_placing_torch;

    bool is_placing_sapling =
        action == CRAFTAX_ACTION_PLACE_PLANT
        && state->map[level][row][col] == CRAFTAX_BLOCK_GRASS
        && CF(inv_sapling, inventory) > 0
        && state->item_map[level][row][col] == CRAFTAX_ITEM_NONE;
    if (is_placing_sapling) {
        int32_t position[2] = {row, col};
        craftax_set_map_block(state, level, row, col, CRAFTAX_BLOCK_PLANT);
        craftax_add_new_growing_plant_native(
            state,
            position,
            is_placing_sapling
        );
    }
    CF(inv_sapling, inventory) -= 1 * (int32_t)is_placing_sapling;
    CF2(achievements, CRAFTAX_ACH_PLACE_PLANT, state) =
        CF2(achievements, CRAFTAX_ACH_PLACE_PLANT, state)
        || is_placing_sapling;
}

// ============================================================
// ===== step_medium.h =====
// ============================================================
// Standalone native ports of medium Craftax step subsystems.
//
// These helpers intentionally are not integrated into c_step yet. They mutate a
// full CraftaxState, or an Inventory plus read-only state context, so tests can
// compare each subsystem directly against the installed JAX implementation.



static __device__ inline CraftaxThreefryKey craftax_medium_next_random_key(
    CraftaxThreefryKey* rng
) {
    CraftaxThreefryKey draw;
    craftax_threefry_split(*rng, rng, &draw);
    return draw;
}

static __device__ inline int32_t craftax_medium_randint(
    CraftaxThreefryKey key,
    int32_t minval,
    int32_t maxval
) {
    return craftax_randint_i32_at(key, 0u, minval, maxval);
}

static __device__ inline int32_t craftax_medium_choice_weighted(
    CraftaxThreefryKey key,
    const float* weights,
    int32_t count
) {
    float total = 0.0f;
    for (int32_t i = 0; i < count; i++) {
        total += weights[i];
    }

    float draw = total * (1.0f - craftax_threefry_uniform_f32(key));
    float cumulative = 0.0f;
    for (int32_t i = 0; i < count; i++) {
        cumulative += weights[i];
        if (cumulative >= draw) {
            return i;
        }
    }
    return count - 1;
}

static __device__ inline int32_t craftax_medium_projectile_count(const CraftaxState* state) {
    int32_t level = craftax_step_jax_index(
        CF(player_level, state),
        CRAFTAX_NUM_LEVELS
    );
    int32_t count = 0;
    for (int32_t i = 0; i < CRAFTAX_MAX_PLAYER_PROJECTILES; i++) {
        count += (int32_t)MOB_MASK(4, level, i, state);
    }
    return count;
}

static __device__ inline int32_t craftax_medium_first_projectile_slot(
    const CraftaxState* state
) {
    int32_t level = craftax_step_jax_index(
        CF(player_level, state),
        CRAFTAX_NUM_LEVELS
    );
    for (int32_t i = 0; i < CRAFTAX_MAX_PLAYER_PROJECTILES; i++) {
        if (!MOB_MASK(4, level, i, state)) {
            return i;
        }
    }
    return 0;
}

static __device__ inline void craftax_medium_spawn_player_projectile(
    CraftaxState* state,
    bool is_spawning_projectile,
    const int32_t new_projectile_position[2],
    const int32_t direction[2],
    int32_t projectile_type
) {
    if (!is_spawning_projectile) {
        return;
    }

    int32_t level = craftax_step_jax_index(
        CF(player_level, state),
        CRAFTAX_NUM_LEVELS
    );
    int32_t index = craftax_medium_first_projectile_slot(state);
    MOB_POS(4, level, index, 0, state) = new_projectile_position[0];
    MOB_POS(4, level, index, 1, state) = new_projectile_position[1];
    MOB_MASK(4, level, index, state) = true;
    MOB_TYPE(4, level, index, state) = projectile_type;
    CF2(player_projectile_directions, (level) * 6 + (index) * 2 + (0), state) = direction[0];
    CF2(player_projectile_directions, (level) * 6 + (index) * 2 + (1), state) = direction[1];
}

static __device__ inline int32_t craftax_medium_level_achievement(int32_t level) {
    switch (craftax_step_jax_index(level, CRAFTAX_NUM_LEVELS)) {
    case 1:
        return CRAFTAX_ACH_ENTER_DUNGEON;
    case 2:
        return CRAFTAX_ACH_ENTER_GNOMISH_MINES;
    case 3:
        return CRAFTAX_ACH_ENTER_SEWERS;
    case 4:
        return CRAFTAX_ACH_ENTER_VAULT;
    case 5:
        return CRAFTAX_ACH_ENTER_TROLL_MINES;
    case 6:
        return CRAFTAX_ACH_ENTER_FIRE_REALM;
    case 7:
        return CRAFTAX_ACH_ENTER_ICE_REALM;
    case 8:
        return CRAFTAX_ACH_ENTER_GRAVEYARD;
    default:
        return CRAFTAX_ACH_COLLECT_WOOD;
    }
}

// See the declaration comment next to g_craftax_curr_p. Runs in the scalar
// step-kernel context at the first step of an episode; draws come from the
// state_rng chain (written back), so canonical envs (p=0) never touch it.
static __device__ void craftax_apply_curriculum(Craftax* env) {
    CraftaxState* state = env->state;
    CraftaxThreefryKey chain;
    chain.word[0] = CF2(state_rng, 0, state);
    chain.word[1] = CF2(state_rng, 1, state);

    CraftaxThreefryKey draw = craftax_medium_next_random_key(&chain);
    int32_t is_curr = craftax_threefry_uniform_f32(draw) < g_craftax_curr_p;
    env->curriculum_episode = is_curr;
    if (is_curr) {
        draw = craftax_medium_next_random_key(&chain);
        int32_t f = craftax_medium_randint(draw, 1, g_craftax_curr_maxf + 1);
        for (int32_t l = 1; l <= f; l++) {
            craftax_ensure_floor_generated(state, l);
            CF2(achievements, craftax_medium_level_achievement(l), state) = true;
            CF2(monsters_killed, l - 1, state) =
                CRAFTAX_MONSTERS_KILLED_TO_CLEAR_LEVEL;
        }
        CF(player_level, state) = f;
        CF2(player_position, 0, state) = state->up_ladders[f][0];
        CF2(player_position, 1, state) = state->up_ladders[f][1];

        draw = craftax_medium_next_random_key(&chain);
        CF(inv_pickaxe, state) = craftax_medium_randint(draw, 0, 5);
        draw = craftax_medium_next_random_key(&chain);
        CF(inv_sword, state) = craftax_medium_randint(draw, 0, 5);
        draw = craftax_medium_next_random_key(&chain);
        CF(inv_torches, state) = craftax_medium_randint(draw, 0, 21);
        draw = craftax_medium_next_random_key(&chain);
        if (craftax_threefry_uniform_f32(draw) < 0.5f) {
            CF(inv_bow, state) = 1;
            draw = craftax_medium_next_random_key(&chain);
            CF(inv_arrows, state) = craftax_medium_randint(draw, 1, 21);
        }
        for (int32_t a = 0; a < 4; a++) {
            draw = craftax_medium_next_random_key(&chain);
            CF2(inv_armour, a, state) = craftax_medium_randint(draw, 0, 3);
        }
        // Pre-mark the achievements the granted gear implies, so the first
        // step's achievement-delta reward doesn't hand out a free burst.
        craftax_calculate_inventory_achievements_native(state);
    }
    CF2(state_rng, 0, state) = chain.word[0];
    CF2(state_rng, 1, state) = chain.word[1];
}

static __device__ inline void craftax_shoot_projectile_native(
    CraftaxState* state,
    int32_t action
) {
    bool is_shooting_arrow = action == CRAFTAX_ACTION_SHOOT_ARROW
        && CF(inv_bow, state) >= 1
        && CF(inv_arrows, state) >= 1
        && craftax_medium_projectile_count(state) < CRAFTAX_MAX_PLAYER_PROJECTILES;

    int32_t direction[2];
    craftax_step_direction(CF(player_direction, state), direction);
    int32_t cf_pp_tmp[2] = {
        CF2(player_position, 0, state), CF2(player_position, 1, state)};
    craftax_medium_spawn_player_projectile(
        state,
        is_shooting_arrow,
        cf_pp_tmp,
        direction,
        CRAFTAX_PROJECTILE_ARROW2
    );

    CF2(achievements, CRAFTAX_ACH_FIRE_BOW, state) =
        CF2(achievements, CRAFTAX_ACH_FIRE_BOW, state) || is_shooting_arrow;
    CF(inv_arrows, state) -= (int32_t)is_shooting_arrow;
}

static __device__ inline void craftax_cast_spell_native(
    CraftaxState* state,
    int32_t action
) {
    bool has_projectile_slot =
        craftax_medium_projectile_count(state) < CRAFTAX_MAX_PLAYER_PROJECTILES;
    bool has_mana = CF(player_mana, state) >= 2;
    bool is_casting_fireball = action == CRAFTAX_ACTION_CAST_FIREBALL
        && has_mana
        && has_projectile_slot
        && CF2(learned_spells, 0, state);
    bool is_casting_iceball = action == CRAFTAX_ACTION_CAST_ICEBALL
        && has_mana
        && has_projectile_slot
        && CF2(learned_spells, 1, state);
    bool is_casting_spell = is_casting_fireball || is_casting_iceball;

    int32_t projectile_type =
        (int32_t)is_casting_fireball * CRAFTAX_PROJECTILE_FIREBALL
        + (int32_t)is_casting_iceball * CRAFTAX_PROJECTILE_ICEBALL;

    int32_t direction[2];
    craftax_step_direction(CF(player_direction, state), direction);
    int32_t cf_pp_tmp[2] = {
        CF2(player_position, 0, state), CF2(player_position, 1, state)};
    craftax_medium_spawn_player_projectile(
        state,
        is_casting_spell,
        cf_pp_tmp,
        direction,
        projectile_type
    );

    if (is_casting_fireball) {
        CF2(achievements, CRAFTAX_ACH_CAST_FIREBALL, state) = true;
    }
    if (is_casting_iceball) {
        CF2(achievements, CRAFTAX_ACH_CAST_ICEBALL, state) = true;
    }
    CF(player_mana, state) -= (int32_t)is_casting_spell * 2;
}

static __device__ inline void craftax_enchant_native(
    CraftaxState* state,
    int32_t action,
    CraftaxThreefryKey rng
) {
    int32_t direction[2];
    craftax_step_direction(CF(player_direction, state), direction);

    int32_t level = craftax_step_jax_index(
        CF(player_level, state),
        CRAFTAX_NUM_LEVELS
    );
    int32_t target_row = craftax_step_jax_index(
        CF2(player_position, 0, state) + direction[0],
        CRAFTAX_MAP_SIZE
    );
    int32_t target_col = craftax_step_jax_index(
        CF2(player_position, 1, state) + direction[1],
        CRAFTAX_MAP_SIZE
    );
    int32_t target_block = state->map[level][target_row][target_col];

    bool is_fire_table = target_block == CRAFTAX_BLOCK_ENCHANTMENT_TABLE_FIRE;
    bool is_ice_table = target_block == CRAFTAX_BLOCK_ENCHANTMENT_TABLE_ICE;
    bool target_block_is_enchantment_table = is_fire_table || is_ice_table;
    int32_t enchantment_type = is_fire_table ? 1 : 2;
    int32_t num_gems = is_fire_table
        ? CF(inv_ruby, state)
        : CF(inv_sapphire, state);

    bool could_enchant = CF(player_mana, state) >= 9
        && target_block_is_enchantment_table
        && num_gems >= 1;
    bool is_enchanting_bow = could_enchant
        && action == CRAFTAX_ACTION_ENCHANT_BOW
        && CF(inv_bow, state) > 0;
    bool is_enchanting_sword = could_enchant
        && action == CRAFTAX_ACTION_ENCHANT_SWORD
        && CF(inv_sword, state) > 0;

    int32_t armour_count = 0;
    for (int32_t i = 0; i < 4; i++) {
        armour_count += CF2(inv_armour, i, state);
    }
    bool is_enchanting_armour = could_enchant
        && action == CRAFTAX_ACTION_ENCHANT_ARMOUR
        && armour_count > 0;

    CraftaxThreefryKey armour_key = craftax_medium_next_random_key(&rng);
    int32_t unenchanted_count = 0;
    for (int32_t i = 0; i < 4; i++) {
        unenchanted_count += (int32_t)(CF2(armour_enchantments, i, state) == 0);
    }

    float armour_targets[4];
    for (int32_t i = 0; i < 4; i++) {
        bool unenchanted = CF2(armour_enchantments, i, state) == 0;
        bool opposite_enchanted = CF2(armour_enchantments, i, state) != 0
            && CF2(armour_enchantments, i, state) != enchantment_type;
        armour_targets[i] = (unenchanted || (
            unenchanted_count == 0 && opposite_enchanted
        )) ? 1.0f : 0.0f;
    }
    int32_t armour_target = craftax_medium_choice_weighted(
        armour_key,
        armour_targets,
        4
    );

    bool is_enchanting = is_enchanting_sword
        || is_enchanting_bow
        || is_enchanting_armour;
    if (is_enchanting_sword) {
        CF(sword_enchantment, state) = enchantment_type;
        CF2(achievements, CRAFTAX_ACH_ENCHANT_SWORD, state) = true;
    }
    if (is_enchanting_bow) {
        CF(bow_enchantment, state) = enchantment_type;
    }
    if (is_enchanting_armour) {
        CF2(armour_enchantments, armour_target, state) = enchantment_type;
        CF2(achievements, CRAFTAX_ACH_ENCHANT_ARMOUR, state) = true;
    }

    CF(inv_sapphire, state) -=
        (int32_t)is_enchanting * (int32_t)(enchantment_type == 2);
    CF(inv_ruby, state) -=
        (int32_t)is_enchanting * (int32_t)(enchantment_type == 1);
    CF(player_mana, state) -= (int32_t)is_enchanting * 9;
}

static __device__ inline void craftax_change_floor_native(
    CraftaxState* state,
    int32_t action
) {
    int32_t level = craftax_step_jax_index(
        CF(player_level, state),
        CRAFTAX_NUM_LEVELS
    );
    int32_t player_row = craftax_step_jax_index(
        CF2(player_position, 0, state),
        CRAFTAX_MAP_SIZE
    );
    int32_t player_col = craftax_step_jax_index(
        CF2(player_position, 1, state),
        CRAFTAX_MAP_SIZE
    );

    bool on_down_ladder =
        state->item_map[level][player_row][player_col] == CRAFTAX_ITEM_LADDER_DOWN;
    bool is_moving_down = action == CRAFTAX_ACTION_DESCEND
        && on_down_ladder
        && CF2(monsters_killed, level, state) >= CRAFTAX_MONSTERS_KILLED_TO_CLEAR_LEVEL
        && CF(player_level, state) < CRAFTAX_NUM_LEVELS - 1;

    bool on_up_ladder =
        state->item_map[level][player_row][player_col] == CRAFTAX_ITEM_LADDER_UP;
    bool is_moving_up = action == CRAFTAX_ACTION_ASCEND
        && on_up_ladder
        && CF(player_level, state) > 0;

    int32_t delta_floor = (int32_t)is_moving_down - (int32_t)is_moving_up;
    int32_t new_level = CF(player_level, state) + delta_floor;
    int32_t achievement = craftax_medium_level_achievement(new_level);
    bool new_floor = new_level != 0 && !CF2(achievements, achievement, state);

    if (is_moving_down) {
        int32_t ladder_level = craftax_step_jax_index(
            CF(player_level, state) + 1,
            CRAFTAX_NUM_LEVELS
        );
        craftax_ensure_floor_generated(state, ladder_level);
        CF2(player_position, 0, state) = state->up_ladders[ladder_level][0];
        CF2(player_position, 1, state) = state->up_ladders[ladder_level][1];
    } else if (is_moving_up) {
        int32_t ladder_level = craftax_step_jax_index(
            CF(player_level, state) - 1,
            CRAFTAX_NUM_LEVELS
        );
        craftax_ensure_floor_generated(state, ladder_level);
        CF2(player_position, 0, state) = state->down_ladders[ladder_level][0];
        CF2(player_position, 1, state) = state->down_ladders[ladder_level][1];
    }

    CF(player_level, state) = new_level;
    CF2(achievements, achievement, state) =
        CF2(achievements, achievement, state) || new_level != 0;
    CF(player_xp, state) += (int32_t)new_floor;
}

static __device__ inline void craftax_add_items_from_chest_native(
    const CraftaxState* state,
    const void* inventory,
    bool is_opening_chest,
    CraftaxThreefryKey rng
) {
    CraftaxThreefryKey draw_key;

    draw_key = craftax_medium_next_random_key(&rng);
    bool is_looting_wood = craftax_threefry_uniform_f32(draw_key) < 0.6f;
    draw_key = craftax_medium_next_random_key(&rng);
    int32_t wood_loot_amount =
        craftax_medium_randint(draw_key, 1, 6) * (int32_t)is_looting_wood;
    (void)wood_loot_amount;

    draw_key = craftax_medium_next_random_key(&rng);
    bool is_looting_torch = craftax_threefry_uniform_f32(draw_key) < 0.6f;
    draw_key = craftax_medium_next_random_key(&rng);
    int32_t torch_loot_amount =
        craftax_medium_randint(draw_key, 4, 8) * (int32_t)is_looting_torch;

    draw_key = craftax_medium_next_random_key(&rng);
    bool is_looting_ore = craftax_threefry_uniform_f32(draw_key) < 0.6f;
    draw_key = craftax_medium_next_random_key(&rng);
    float ore_weights[5] = {0.3f, 0.3f, 0.15f, 0.125f, 0.125f};
    int32_t ore_loot_id = craftax_medium_choice_weighted(
        draw_key,
        ore_weights,
        5
    );
    draw_key = craftax_medium_next_random_key(&rng);

    int32_t coal_loot_amount =
        craftax_medium_randint(draw_key, 1, 4)
        * (int32_t)(ore_loot_id == 0)
        * (int32_t)is_looting_ore;
    int32_t iron_loot_amount =
        craftax_medium_randint(draw_key, 1, 3)
        * (int32_t)(ore_loot_id == 1)
        * (int32_t)is_looting_ore;
    int32_t diamond_loot_amount =
        craftax_medium_randint(draw_key, 1, 2)
        * (int32_t)(ore_loot_id == 2)
        * (int32_t)is_looting_ore;
    int32_t sapphire_loot_amount =
        craftax_medium_randint(draw_key, 1, 2)
        * (int32_t)(ore_loot_id == 3)
        * (int32_t)is_looting_ore;
    int32_t ruby_loot_amount =
        craftax_medium_randint(draw_key, 1, 2)
        * (int32_t)(ore_loot_id == 4)
        * (int32_t)is_looting_ore;

    draw_key = craftax_medium_next_random_key(&rng);
    bool is_looting_potion = craftax_threefry_uniform_f32(draw_key) < 0.5f;
    draw_key = craftax_medium_next_random_key(&rng);
    int32_t potion_loot_index = craftax_medium_randint(draw_key, 0, 6);
    draw_key = craftax_medium_next_random_key(&rng);
    int32_t potion_loot_amount = craftax_medium_randint(draw_key, 1, 3);

    draw_key = craftax_medium_next_random_key(&rng);
    bool is_looting_arrows = craftax_threefry_uniform_f32(draw_key) < 0.25f;
    draw_key = craftax_medium_next_random_key(&rng);
    int32_t arrows_loot_amount =
        craftax_medium_randint(draw_key, 1, 5) * (int32_t)is_looting_arrows;

    draw_key = craftax_medium_next_random_key(&rng);
    bool is_looting_tool = craftax_threefry_uniform_f32(draw_key) < 0.2f;
    draw_key = craftax_medium_next_random_key(&rng);
    int32_t tool_id = craftax_medium_randint(draw_key, 0, 2);

    bool is_looting_pickaxe = is_looting_tool
        && tool_id == 0
        && is_opening_chest;
    draw_key = craftax_medium_next_random_key(&rng);
    float tool_weights[4] = {0.4f, 0.3f, 0.2f, 0.1f};
    int32_t pickaxe_loot_level = (
        craftax_medium_choice_weighted(draw_key, tool_weights, 4) + 1
    ) * (int32_t)is_looting_pickaxe;
    pickaxe_loot_level = craftax_step_maxi32(
        pickaxe_loot_level,
        CF(inv_pickaxe, inventory)
    );
    int32_t new_pickaxe_level = is_looting_pickaxe
        ? pickaxe_loot_level
        : CF(inv_pickaxe, inventory);

    bool is_looting_sword = is_looting_tool
        && tool_id == 1
        && is_opening_chest;
    draw_key = craftax_medium_next_random_key(&rng);
    int32_t sword_loot_level = (
        craftax_medium_choice_weighted(draw_key, tool_weights, 4) + 1
    ) * (int32_t)is_looting_sword;
    sword_loot_level = craftax_step_maxi32(sword_loot_level, CF(inv_sword, inventory));
    int32_t new_sword_level = is_looting_sword
        ? sword_loot_level
        : CF(inv_sword, inventory);

    int32_t level = craftax_step_jax_index(
        CF(player_level, state),
        CRAFTAX_NUM_LEVELS
    );
    bool is_looting_bow = is_opening_chest
        && CF(player_level, state) == 1
        && !CF2(chests_opened, level, state);
    int32_t new_bow_level = is_looting_bow ? 1 : CF(inv_bow, inventory);

    bool is_looting_book = !CF2(chests_opened, level, state)
        && (CF(player_level, state) == 3 || CF(player_level, state) == 4);

    int32_t opening = (int32_t)is_opening_chest;
    CF(inv_torches, inventory) += torch_loot_amount * opening;
    CF(inv_coal, inventory) += coal_loot_amount * opening;
    CF(inv_iron, inventory) += iron_loot_amount * opening;
    CF(inv_diamond, inventory) += diamond_loot_amount * opening;
    CF(inv_sapphire, inventory) += sapphire_loot_amount * opening;
    CF(inv_ruby, inventory) += ruby_loot_amount * opening;
    CF(inv_arrows, inventory) += arrows_loot_amount * opening;
    CF(inv_pickaxe, inventory) = new_pickaxe_level;
    CF(inv_sword, inventory) = new_sword_level;
    CF2(inv_potions, potion_loot_index, inventory) +=
        potion_loot_amount * (int32_t)is_looting_potion * opening;
    CF(inv_bow, inventory) = new_bow_level;
    CF(inv_books, inventory) += (int32_t)is_looting_book * opening;
}

// ============================================================
// ===== step_do_action.h =====
// ============================================================
// Standalone native port of Craftax do_action.



#define CRAFTAX_DO_ACTION_BOSS_FIGHT_SPAWN_TURNS 7

static __device__ inline float craftax_do_action_mob_defense(
    int32_t type_id,
    int32_t mob_class_index,
    int32_t damage_index
) {
    static const float defenses[8][4][3] = {
        {
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {0.5f, 0.0f, 0.0f},
            {0.5f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {0.2f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {0.9f, 1.0f, 0.0f},
            {0.9f, 1.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {0.9f, 0.0f, 1.0f},
            {0.9f, 0.0f, 1.0f},
            {0.0f, 0.0f, 0.0f},
        },
    };

    int32_t type_index = craftax_step_jax_index(type_id, 8);
    int32_t class_index = craftax_step_jax_index(mob_class_index, 4);
    int32_t component = craftax_step_jax_index(damage_index, 3);
    return defenses[type_index][class_index][component];
}

static __device__ inline int32_t craftax_do_action_mob_achievement(
    int32_t mob_class_index,
    int32_t type_id
) {
    static const int32_t achievements[3][8] = {
        {
            CRAFTAX_ACH_EAT_COW,
            CRAFTAX_ACH_EAT_BAT,
            CRAFTAX_ACH_EAT_SNAIL,
            0,
            0,
            0,
            0,
            0,
        },
        {
            CRAFTAX_ACH_DEFEAT_ZOMBIE,
            CRAFTAX_ACH_DEFEAT_GNOME_WARRIOR,
            CRAFTAX_ACH_DEFEAT_ORC_SOLIDER,
            CRAFTAX_ACH_DEFEAT_LIZARD,
            CRAFTAX_ACH_DEFEAT_KNIGHT,
            CRAFTAX_ACH_DEFEAT_TROLL,
            CRAFTAX_ACH_DEFEAT_PIGMAN,
            CRAFTAX_ACH_DEFEAT_FROST_TROLL,
        },
        {
            CRAFTAX_ACH_DEFEAT_SKELETON,
            CRAFTAX_ACH_DEFEAT_GNOME_ARCHER,
            CRAFTAX_ACH_DEFEAT_ORC_MAGE,
            CRAFTAX_ACH_DEFEAT_KOBOLD,
            CRAFTAX_ACH_DEFEAT_ARCHER,
            CRAFTAX_ACH_DEFEAT_DEEP_THING,
            CRAFTAX_ACH_DEFEAT_FIRE_ELEMENTAL,
            CRAFTAX_ACH_DEFEAT_ICE_ELEMENTAL,
        },
    };

    int32_t class_index = craftax_step_jax_index(mob_class_index, 3);
    int32_t type_index = craftax_step_jax_index(type_id, 8);
    return achievements[class_index][type_index];
}

static __device__ inline void craftax_do_action_player_damage_vector(
    const CraftaxState* state,
    float damage_vector[3]
) {
    static const float physical_damages[5] = {1.0f, 2.0f, 3.0f, 5.0f, 8.0f};

    int32_t sword_index = craftax_step_jax_index(CF(inv_sword, state), 5);
    float physical_damage = physical_damages[sword_index];
    float fire_damage =
        physical_damage * (float)(CF(sword_enchantment, state) == 1) * 0.5f;
    float ice_damage =
        physical_damage * (float)(CF(sword_enchantment, state) == 2) * 0.5f;

    physical_damage *= 1.0f + 0.25f * (float)(CF(player_strength, state) - 1);
    fire_damage *= 1.0f + 0.05f * (float)(CF(player_intelligence, state) - 1);
    ice_damage *= 1.0f + 0.05f * (float)(CF(player_intelligence, state) - 1);

    damage_vector[0] = physical_damage;
    damage_vector[1] = fire_damage;
    damage_vector[2] = ice_damage;
}

static __device__ inline float craftax_do_action_damage_done(
    const float damage_vector[3],
    int32_t type_id,
    int32_t mob_class_index
) {
    float damage = 0.0f;
    for (int32_t i = 0; i < 3; i++) {
        float defense = craftax_do_action_mob_defense(
            type_id,
            mob_class_index,
            i
        );
        damage += (1.0f - defense) * damage_vector[i];
    }
    return damage;
}

static __device__ inline void craftax_do_action_refresh_mobs3_masks(void* mobs, int mc) {
    for (int32_t level = 0; level < CRAFTAX_NUM_LEVELS; level++) {
        for (int32_t i = 0; i < 3; i++) {
            MOB_MASK(mc, level, i, mobs) =
                MOB_MASK(mc, level, i, mobs) && MOB_HP(mc, level, i, mobs) > 0.0f;
        }
    }
}

static __device__ inline void craftax_do_action_refresh_mobs2_masks(void* mobs, int mc) {
    for (int32_t level = 0; level < CRAFTAX_NUM_LEVELS; level++) {
        for (int32_t i = 0; i < 2; i++) {
            MOB_MASK(mc, level, i, mobs) =
                MOB_MASK(mc, level, i, mobs) && MOB_HP(mc, level, i, mobs) > 0.0f;
        }
    }
}

static __device__ inline void craftax_do_action_attack_mobs3(
    CraftaxState* state,
    void* mobs, int mc,
    int32_t row,
    int32_t col,
    const float damage_vector[3],
    bool can_get_achievement,
    int32_t mob_class_index,
    bool* did_kill_mob,
    bool* is_attacking_mob
) {
    int32_t level = craftax_step_jax_index(
        CF(player_level, state),
        CRAFTAX_NUM_LEVELS
    );
    bool is_attacking_array[3];
    *is_attacking_mob = false;
    int32_t target_mob_index = 0;

    for (int32_t i = 0; i < 3; i++) {
        bool in_mob = MOB_POS(mc, level, i, 0, mobs) == row
            && MOB_POS(mc, level, i, 1, mobs) == col;
        is_attacking_array[i] = in_mob && MOB_MASK(mc, level, i, mobs);
        if (is_attacking_array[i] && !*is_attacking_mob) {
            target_mob_index = i;
        }
        *is_attacking_mob = *is_attacking_mob || is_attacking_array[i];
    }

    int32_t target_type_id = MOB_TYPE(mc, level, target_mob_index, mobs);
    float damage = craftax_do_action_damage_done(
        damage_vector,
        target_type_id,
        mob_class_index
    );
    MOB_HP(mc, level, target_mob_index, mobs) -=
        damage * (float)(int32_t)(*is_attacking_mob);

    bool old_mask = MOB_MASK(mc, level, target_mob_index, mobs);
    craftax_do_action_refresh_mobs3_masks(mobs, mc);
    *did_kill_mob = old_mask && !MOB_MASK(mc, level, target_mob_index, mobs);

    int32_t achievement_for_kill = craftax_do_action_mob_achievement(
        mob_class_index,
        target_type_id
    );
    bool unlock = *did_kill_mob && can_get_achievement;
    CF2(achievements, achievement_for_kill, state) =
        CF2(achievements, achievement_for_kill, state) || unlock;
}

static __device__ inline void craftax_do_action_attack_mobs2(
    CraftaxState* state,
    void* mobs, int mc,
    int32_t row,
    int32_t col,
    const float damage_vector[3],
    bool can_get_achievement,
    int32_t mob_class_index,
    bool* did_kill_mob,
    bool* is_attacking_mob
) {
    int32_t level = craftax_step_jax_index(
        CF(player_level, state),
        CRAFTAX_NUM_LEVELS
    );
    bool is_attacking_array[2];
    *is_attacking_mob = false;
    int32_t target_mob_index = 0;

    for (int32_t i = 0; i < 2; i++) {
        bool in_mob = MOB_POS(mc, level, i, 0, mobs) == row
            && MOB_POS(mc, level, i, 1, mobs) == col;
        is_attacking_array[i] = in_mob && MOB_MASK(mc, level, i, mobs);
        if (is_attacking_array[i] && !*is_attacking_mob) {
            target_mob_index = i;
        }
        *is_attacking_mob = *is_attacking_mob || is_attacking_array[i];
    }

    int32_t target_type_id = MOB_TYPE(mc, level, target_mob_index, mobs);
    float damage = craftax_do_action_damage_done(
        damage_vector,
        target_type_id,
        mob_class_index
    );
    MOB_HP(mc, level, target_mob_index, mobs) -=
        damage * (float)(int32_t)(*is_attacking_mob);

    bool old_mask = MOB_MASK(mc, level, target_mob_index, mobs);
    craftax_do_action_refresh_mobs2_masks(mobs, mc);
    *did_kill_mob = old_mask && !MOB_MASK(mc, level, target_mob_index, mobs);

    int32_t achievement_for_kill = craftax_do_action_mob_achievement(
        mob_class_index,
        target_type_id
    );
    bool unlock = *did_kill_mob && can_get_achievement;
    CF2(achievements, achievement_for_kill, state) =
        CF2(achievements, achievement_for_kill, state) || unlock;
}

static __device__ inline bool craftax_do_action_update_index(
    int32_t index,
    int32_t size,
    int32_t* mapped_index
) {
    if (index < -size || index >= size) {
        return false;
    }
    *mapped_index = index < 0 ? index + size : index;
    return true;
}

static __device__ inline void craftax_do_action_update_mob_map(
    CraftaxState* state,
    int32_t row,
    int32_t col,
    bool did_kill_mob
) {
    int32_t update_row;
    int32_t update_col;
    if (!craftax_do_action_update_index(row, CRAFTAX_MAP_SIZE, &update_row)
        || !craftax_do_action_update_index(col, CRAFTAX_MAP_SIZE, &update_col)) {
        return;
    }

    int32_t level = craftax_step_jax_index(
        CF(player_level, state),
        CRAFTAX_NUM_LEVELS
    );
    int32_t read_row = craftax_step_jax_index(row, CRAFTAX_MAP_SIZE);
    int32_t read_col = craftax_step_jax_index(col, CRAFTAX_MAP_SIZE);
    bool old_value = (CF_BITS(mob_bits, level, read_row, state) >> read_col) & 1ULL;
    bool new_value = old_value && !did_kill_mob;
    if (new_value) {
        CF_BITS(mob_bits, level, update_row, state) |= (1ULL << update_col);
    } else {
        CF_BITS(mob_bits, level, update_row, state) &= ~(1ULL << update_col);
    }
}

static __device__ inline void craftax_do_action_attack_mob(
    CraftaxState* state,
    int32_t row,
    int32_t col,
    bool can_eat,
    bool* did_attack_mob,
    bool* did_kill_mob
) {
    float damage_vector[3];
    craftax_do_action_player_damage_vector(state, damage_vector);

    bool did_kill_melee_mob = false;
    bool is_attacking_melee_mob = false;
    craftax_do_action_attack_mobs3(
        state,
        state, 0,
        row,
        col,
        damage_vector,
        true,
        1,
        &did_kill_melee_mob,
        &is_attacking_melee_mob
    );

    bool did_kill_passive_mob = false;
    bool is_attacking_passive_mob = false;
    craftax_do_action_attack_mobs3(
        state,
        state, 1,
        row,
        col,
        damage_vector,
        can_eat,
        0,
        &did_kill_passive_mob,
        &is_attacking_passive_mob
    );

    if (did_kill_passive_mob && can_eat) {
        CF(player_food, state) = craftax_step_mini32(
            craftax_step_get_max_food(state),
            CF(player_food, state) + 6
        );
        CF(player_hunger, state) = 0.0f;
    }

    bool did_kill_ranged_mob = false;
    bool is_attacking_ranged_mob = false;
    craftax_do_action_attack_mobs2(
        state,
        state, 2,
        row,
        col,
        damage_vector,
        true,
        2,
        &did_kill_ranged_mob,
        &is_attacking_ranged_mob
    );

    *did_attack_mob = is_attacking_melee_mob
        || is_attacking_passive_mob
        || is_attacking_ranged_mob;
    bool did_kill_monster = did_kill_melee_mob || did_kill_ranged_mob;
    *did_kill_mob = did_kill_monster || did_kill_passive_mob;

    craftax_do_action_update_mob_map(state, row, col, *did_kill_mob);

    int32_t level = craftax_step_jax_index(
        CF(player_level, state),
        CRAFTAX_NUM_LEVELS
    );
    CF2(monsters_killed, level, state) += (int32_t)did_kill_monster;
}

static __device__ inline bool craftax_do_action_in_bounds(int32_t row, int32_t col) {
    return row >= 0
        && row < CRAFTAX_MAP_SIZE
        && col >= 0
        && col < CRAFTAX_MAP_SIZE;
}

static __device__ inline bool craftax_do_action_boss_vulnerable(
    const CraftaxState* state
) {
    int32_t level = craftax_step_jax_index(
        CF(player_level, state),
        CRAFTAX_NUM_LEVELS
    );
    int32_t melee_count = 0;
    int32_t ranged_count = 0;
    for (int32_t i = 0; i < CRAFTAX_MAX_MELEE_MOBS; i++) {
        melee_count += (int32_t)MOB_MASK(0, level, i, state);
    }
    for (int32_t i = 0; i < CRAFTAX_MAX_RANGED_MOBS; i++) {
        ranged_count += (int32_t)MOB_MASK(2, level, i, state);
    }
    return melee_count == 0
        && ranged_count == 0
        && CF(boss_timesteps_to_spawn_this_round, state) <= 0;
}

static __device__ inline void craftax_do_action_update_plants_with_eat(
    CraftaxState* state,
    int32_t row,
    int32_t col
) {
    int32_t plant_index = 0;
    bool found = false;
    for (int32_t i = 0; i < CRAFTAX_MAX_GROWING_PLANTS; i++) {
        bool is_plant = CF2(growing_plants_positions, (i) * 2 + (0), state) == row
            && CF2(growing_plants_positions, (i) * 2 + (1), state) == col;
        if (is_plant && !found) {
            plant_index = i;
            found = true;
        }
    }
    CF2(growing_plants_age, plant_index, state) = 0;
}

static __device__ inline void craftax_do_action_native(
    CraftaxState* state,
    int32_t action,
    CraftaxThreefryKey rng
) {
    if (action != CRAFTAX_ACTION_DO) {
        return;
    }

    int32_t direction[2];
    craftax_step_direction(CF(player_direction, state), direction);
    int32_t target_row = CF2(player_position, 0, state) + direction[0];
    int32_t target_col = CF2(player_position, 1, state) + direction[1];

    bool did_attack_mob = false;
    bool did_kill_mob = false;
    craftax_do_action_attack_mob(
        state,
        target_row,
        target_col,
        true,
        &did_attack_mob,
        &did_kill_mob
    );
    (void)did_kill_mob;

    int32_t level = craftax_step_jax_index(
        CF(player_level, state),
        CRAFTAX_NUM_LEVELS
    );
    int32_t read_row = craftax_step_jax_index(target_row, CRAFTAX_MAP_SIZE);
    int32_t read_col = craftax_step_jax_index(target_col, CRAFTAX_MAP_SIZE);
    int32_t target_block = state->map[level][read_row][read_col];

    CraftaxThreefryKey sapling_key = craftax_medium_next_random_key(&rng);
    CraftaxThreefryKey chest_key = craftax_medium_next_random_key(&rng);

    bool is_opening_chest = target_block == CRAFTAX_BLOCK_CHEST;
    bool is_damaging_boss = target_block == CRAFTAX_BLOCK_NECROMANCER
        && craftax_do_action_boss_vulnerable(state)
        && craftax_step_is_fighting_boss(state);

    bool action_block_in_bounds =
        craftax_do_action_in_bounds(target_row, target_col) && !did_attack_mob;

    if (action_block_in_bounds) {
        bool is_block_tree = target_block == CRAFTAX_BLOCK_TREE;
        bool is_block_fire_tree = target_block == CRAFTAX_BLOCK_FIRE_TREE;
        bool is_block_ice_shrub = target_block == CRAFTAX_BLOCK_ICE_SHRUB;
        bool is_mining_tree =
            is_block_tree || is_block_fire_tree || is_block_ice_shrub;
        if (is_mining_tree) {
            int32_t replacement = is_block_tree
                ? CRAFTAX_BLOCK_GRASS
                : (is_block_fire_tree
                    ? CRAFTAX_BLOCK_FIRE_GRASS
                    : CRAFTAX_BLOCK_ICE_GRASS);
            craftax_set_map_block(state, level, target_row, target_col, replacement);
            CF(inv_wood, state) += 1;
        }

        bool is_mining_stone = target_block == CRAFTAX_BLOCK_STONE
            && CF(inv_pickaxe, state) >= 1;
        if (is_mining_stone) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            CF(inv_stone, state) += 1;
        }

        if (target_block == CRAFTAX_BLOCK_FURNACE) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
        }

        if (target_block == CRAFTAX_BLOCK_CRAFTING_TABLE) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
        }

        bool is_mining_coal = target_block == CRAFTAX_BLOCK_COAL
            && CF(inv_pickaxe, state) >= 1;
        if (is_mining_coal) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            CF(inv_coal, state) += 1;
        }

        bool is_mining_iron = target_block == CRAFTAX_BLOCK_IRON
            && CF(inv_pickaxe, state) >= 2;
        if (is_mining_iron) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            CF(inv_iron, state) += 1;
        }

        bool is_mining_diamond = target_block == CRAFTAX_BLOCK_DIAMOND
            && CF(inv_pickaxe, state) >= 3;
        if (is_mining_diamond) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            CF(inv_diamond, state) += 1;
        }

        bool is_mining_sapphire = target_block == CRAFTAX_BLOCK_SAPPHIRE
            && CF(inv_pickaxe, state) >= 4;
        if (is_mining_sapphire) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            CF(inv_sapphire, state) += 1;
        }

        bool is_mining_ruby = target_block == CRAFTAX_BLOCK_RUBY
            && CF(inv_pickaxe, state) >= 4;
        if (is_mining_ruby) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            CF(inv_ruby, state) += 1;
        }

        bool is_mining_sapling = target_block == CRAFTAX_BLOCK_GRASS
            && craftax_threefry_uniform_f32(sapling_key) < 0.1f;
        CF(inv_sapling, state) += (int32_t)is_mining_sapling;

        bool is_drinking_water = target_block == CRAFTAX_BLOCK_WATER
            || target_block == CRAFTAX_BLOCK_FOUNTAIN;
        if (is_drinking_water) {
            CF(player_drink, state) = craftax_step_mini32(
                craftax_step_get_max_drink(state),
                CF(player_drink, state) + 1
            );
            CF(player_thirst, state) = 0.0f;
            CF2(achievements, CRAFTAX_ACH_COLLECT_DRINK, state) = true;
        }

        bool is_eating_plant = target_block == CRAFTAX_BLOCK_RIPE_PLANT;
        if (is_eating_plant) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PLANT);
            CF(player_food, state) = craftax_step_mini32(
                craftax_step_get_max_food(state),
                CF(player_food, state) + 4
            );
            CF(player_hunger, state) = 0.0f;
            CF2(achievements, CRAFTAX_ACH_EAT_PLANT, state) = true;
            craftax_do_action_update_plants_with_eat(
                state,
                target_row,
                target_col
            );
        }

        bool is_mining_stalagmite = target_block == CRAFTAX_BLOCK_STALAGMITE
            && CF(inv_pickaxe, state) >= 1;
        if (is_mining_stalagmite) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            CF(inv_stone, state) += 1;
        }

        if (is_opening_chest) {
            craftax_set_map_block(state, level, target_row, target_col, CRAFTAX_BLOCK_PATH);
            craftax_add_items_from_chest_native(
                state,
                state,
                true,
                chest_key
            );
            CF2(achievements, CRAFTAX_ACH_OPEN_CHEST, state) = true;
        }

        if (is_damaging_boss) {
            CF2(achievements, CRAFTAX_ACH_DAMAGE_NECROMANCER, state) = true;
        }
    }

    CF2(chests_opened, level, state) =
        CF2(chests_opened, level, state) || is_opening_chest;

    CF(boss_progress, state) += (int32_t)is_damaging_boss;
    if (is_damaging_boss) {
        CF(boss_timesteps_to_spawn_this_round, state) =
            CRAFTAX_DO_ACTION_BOSS_FIGHT_SPAWN_TURNS;
    }
}

// ============================================================
// ===== step_update_mobs.h =====
// ============================================================
// Standalone native port of Craftax update_mobs.



#define CRAFTAX_UPDATE_BOSS_FIGHT_EXTRA_DAMAGE 0.5f

static __device__ inline CraftaxThreefryKey craftax_update_mobs_next_random_key(
    CraftaxThreefryKey* rng
) {
    CraftaxThreefryKey draw;
    craftax_threefry_split(*rng, rng, &draw);
    return draw;
}

static __device__ inline bool craftax_update_mobs_scatter_index(
    int32_t index,
    int32_t size,
    int32_t* mapped_index
) {
    if (index < -size || index >= size) {
        return false;
    }
    *mapped_index = index < 0 ? index + size : index;
    return true;
}

static __device__ inline bool craftax_update_mobs_in_bounds(
    int32_t row,
    int32_t col
) {
    return row >= 0
        && row < CRAFTAX_MAP_SIZE
        && col >= 0
        && col < CRAFTAX_MAP_SIZE;
}

static __device__ inline int32_t craftax_update_mobs_read_block(
    const CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col
) {
    int32_t map_level = craftax_step_jax_index(level, CRAFTAX_NUM_LEVELS);
    int32_t map_row = craftax_step_jax_index(row, CRAFTAX_MAP_SIZE);
    int32_t map_col = craftax_step_jax_index(col, CRAFTAX_MAP_SIZE);
    return state->map[map_level][map_row][map_col];
}

static __device__ inline void craftax_update_mobs_set_block(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col,
    int32_t block
) {
    int32_t map_level;
    int32_t map_row;
    int32_t map_col;
    if (!craftax_update_mobs_scatter_index(
            level,
            CRAFTAX_NUM_LEVELS,
            &map_level
        )
        || !craftax_update_mobs_scatter_index(
            row,
            CRAFTAX_MAP_SIZE,
            &map_row
        )
        || !craftax_update_mobs_scatter_index(
            col,
            CRAFTAX_MAP_SIZE,
            &map_col
        )) {
        return;
    }
    craftax_set_map_block(state, map_level, map_row, map_col, block);
}

static __device__ inline bool craftax_update_mobs_read_mob_map(
    const CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col
) {
    int32_t map_level = craftax_step_jax_index(level, CRAFTAX_NUM_LEVELS);
    int32_t map_row = craftax_step_jax_index(row, CRAFTAX_MAP_SIZE);
    int32_t map_col = craftax_step_jax_index(col, CRAFTAX_MAP_SIZE);
    return (CF_BITS(mob_bits, map_level, map_row, state) >> map_col) & 1ULL;
}

static __device__ inline void craftax_update_mobs_set_mob_map(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col,
    bool value
) {
    int32_t map_level;
    int32_t map_row;
    int32_t map_col;
    if (!craftax_update_mobs_scatter_index(
            level,
            CRAFTAX_NUM_LEVELS,
            &map_level
        )
        || !craftax_update_mobs_scatter_index(
            row,
            CRAFTAX_MAP_SIZE,
            &map_row
        )
        || !craftax_update_mobs_scatter_index(
            col,
            CRAFTAX_MAP_SIZE,
            &map_col
        )) {
        return;
    }
    if (value) {
        CF_BITS(mob_bits, map_level, map_row, state) |= (1ULL << map_col);
    } else {
        CF_BITS(mob_bits, map_level, map_row, state) &= ~(1ULL << map_col);
    }
}

static __device__ inline void craftax_update_mobs_clear_old_map_entry(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col,
    bool old_mask
) {
    bool old_value = craftax_update_mobs_read_mob_map(state, level, row, col);
    craftax_update_mobs_set_mob_map(
        state,
        level,
        row,
        col,
        old_value && !old_mask
    );
}

static __device__ inline void craftax_update_mobs_enter_new_map_entry(
    CraftaxState* state,
    int32_t level,
    int32_t row,
    int32_t col,
    bool new_mask
) {
    bool old_value = craftax_update_mobs_read_mob_map(state, level, row, col);
    craftax_update_mobs_set_mob_map(
        state,
        level,
        row,
        col,
        old_value || new_mask
    );
}

static __device__ inline void craftax_update_mobs_damage_vector(
    int32_t type_id,
    int32_t mob_class_index,
    float damage[3]
) {
    static const float damages[CRAFTAX_NUM_MOB_TYPES][4][3] = {
        {
            {0.0f, 0.0f, 0.0f},
            {2.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {2.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {4.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {4.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {3.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 3.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {5.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 3.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {6.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {5.0f, 0.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {6.0f, 1.0f, 1.0f},
            {0.0f, 0.0f, 0.0f},
            {4.0f, 3.0f, 3.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {3.0f, 5.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {3.0f, 5.0f, 0.0f},
        },
        {
            {0.0f, 0.0f, 0.0f},
            {4.0f, 0.0f, 5.0f},
            {0.0f, 0.0f, 0.0f},
            {4.0f, 0.0f, 5.0f},
        },
    };

    int32_t type_index = craftax_step_jax_index(
        type_id,
        CRAFTAX_NUM_MOB_TYPES
    );
    int32_t class_index = craftax_step_jax_index(mob_class_index, 4);
    for (int32_t i = 0; i < 3; i++) {
        damage[i] = damages[type_index][class_index][i];
    }
}

static __device__ inline void craftax_update_mobs_collision_map(
    int32_t type_id,
    int32_t mob_class_index,
    bool collision[3]
) {
    static const bool collisions[CRAFTAX_NUM_MOB_TYPES][4][3] = {
        {
            {false, true, true},
            {false, true, true},
            {false, true, true},
            {false, false, false},
        },
        {
            {false, false, false},
            {false, true, true},
            {false, true, true},
            {false, false, false},
        },
        {
            {false, true, true},
            {false, true, true},
            {false, true, true},
            {false, false, false},
        },
        {
            {false, true, true},
            {false, false, true},
            {false, true, true},
            {false, false, false},
        },
        {
            {false, true, true},
            {false, true, true},
            {false, true, true},
            {false, false, false},
        },
        {
            {false, true, true},
            {false, true, true},
            {true, false, true},
            {false, false, false},
        },
        {
            {false, true, true},
            {false, true, true},
            {false, false, false},
            {false, false, false},
        },
        {
            {false, true, true},
            {false, true, true},
            {false, false, false},
            {false, false, false},
        },
    };

    int32_t type_index = craftax_step_jax_index(
        type_id,
        CRAFTAX_NUM_MOB_TYPES
    );
    int32_t class_index = craftax_step_jax_index(mob_class_index, 4);
    for (int32_t i = 0; i < 3; i++) {
        collision[i] = collisions[type_index][class_index][i];
    }
}

static __device__ inline int32_t craftax_update_mobs_projectile_type_for_ranged(
    int32_t ranged_type
) {
    static const int32_t mapping[CRAFTAX_NUM_MOB_TYPES] = {
        CRAFTAX_PROJECTILE_ARROW,
        CRAFTAX_PROJECTILE_ARROW,
        CRAFTAX_PROJECTILE_FIREBALL,
        CRAFTAX_PROJECTILE_DAGGER,
        CRAFTAX_PROJECTILE_ARROW2,
        CRAFTAX_PROJECTILE_SLIMEBALL,
        CRAFTAX_PROJECTILE_FIREBALL2,
        CRAFTAX_PROJECTILE_ICEBALL2,
    };
    int32_t type_index = craftax_step_jax_index(
        ranged_type,
        CRAFTAX_NUM_MOB_TYPES
    );
    return mapping[type_index];
}

static __device__ inline void craftax_update_mobs_direction_choice(
    CraftaxThreefryKey key,
    int32_t count,
    int32_t direction[2]
) {
    int32_t choice = craftax_medium_randint(key, 0, count);
    direction[0] = 0;
    direction[1] = 0;
    if (choice == 0) {
        direction[1] = -1;
    } else if (choice == 1) {
        direction[1] = 1;
    } else if (choice == 2) {
        direction[0] = -1;
    } else if (choice == 3) {
        direction[0] = 1;
    }
}

static __device__ inline int32_t craftax_update_mobs_abs_i32(int32_t value) {
    return value < 0 ? -value : value;
}

static __device__ inline int32_t craftax_update_mobs_sign_i32(int32_t value) {
    if (value < 0) {
        return -1;
    }
    return value > 0 ? 1 : 0;
}

static __device__ inline int32_t craftax_update_mobs_player_axis_choice(
    CraftaxThreefryKey key,
    int32_t distance_row,
    int32_t distance_col
) {
    int32_t max_distance = distance_row > distance_col
        ? distance_row
        : distance_col;
    int32_t total_distance = distance_row + distance_col;
    if (total_distance == 0) {
        return 1;
    }

    float weights[2] = {
        (distance_row == max_distance) ? 1.0f / (float)total_distance : 0.0f,
        (distance_col == max_distance) ? 1.0f / (float)total_distance : 0.0f,
    };
    return craftax_medium_choice_weighted(key, weights, 2);
}

static __device__ inline bool craftax_update_mobs_valid_position(
    const CraftaxState* state,
    int32_t row,
    int32_t col,
    const bool collision[3]
) {
    int32_t level = craftax_step_jax_index(
        CF(player_level, state),
        CRAFTAX_NUM_LEVELS
    );
    bool pos_in_bounds = craftax_update_mobs_in_bounds(row, col);
    int32_t block = craftax_update_mobs_read_block(state, level, row, col);
    bool in_solid_block = craftax_step_is_solid_block(block);
    bool in_mob = craftax_step_is_in_mob(state, row, col);
    bool in_lava = block == CRAFTAX_BLOCK_LAVA;
    bool in_water = block == CRAFTAX_BLOCK_WATER;
    bool on_ground_block = !in_solid_block && !in_water && !in_lava;

    bool valid_move = pos_in_bounds && !in_mob && !in_solid_block;
    valid_move = valid_move && (!collision[0] || !on_ground_block);
    valid_move = valid_move && (!collision[1] || !in_water);
    valid_move = valid_move && (!collision[2] || !in_lava);
    return valid_move;
}

static __device__ inline int32_t craftax_update_mobs_manhattan_to_player(
    const CraftaxState* state,
    int32_t row,
    int32_t col
) {
    return craftax_update_mobs_abs_i32(row - CF2(player_position, 0, state))
        + craftax_update_mobs_abs_i32(col - CF2(player_position, 1, state));
}

static __device__ inline float craftax_update_mobs_damage_done_to_player(
    const CraftaxState* state,
    const float damage_vector[3]
) {
    float defense_vector[3] = {0.0f, 0.0f, 0.0f};
    for (int32_t i = 0; i < 4; i++) {
        defense_vector[0] += (float)CF2(inv_armour, i, state) * 0.1f;
        defense_vector[1] +=
            (float)(int32_t)(CF2(armour_enchantments, i, state) == 1) * 0.2f;
        defense_vector[2] +=
            (float)(int32_t)(CF2(armour_enchantments, i, state) == 2) * 0.2f;
    }

    float boss_coeff = craftax_step_is_fighting_boss(state)
        ? 1.0f + CRAFTAX_UPDATE_BOSS_FIGHT_EXTRA_DAMAGE
        : 1.0f;
    float damage = 0.0f;
    for (int32_t i = 0; i < 3; i++) {
        damage += (1.0f - defense_vector[i]) * damage_vector[i] * boss_coeff;
    }
    return damage;
}

static __device__ inline int32_t craftax_update_mobs_count_mob_projectiles(
    const CraftaxState* state,
    int32_t level
) {
    return (int32_t)MOB_MASK(3, level, 0, state)
        + (int32_t)MOB_MASK(3, level, 1, state)
        + (int32_t)MOB_MASK(3, level, 2, state);
}

static __device__ inline int32_t craftax_update_mobs_first_empty_mob_projectile(
    const CraftaxState* state,
    int32_t level
) {
    if (!MOB_MASK(3, level, 0, state)) return 0;
    if (!MOB_MASK(3, level, 1, state)) return 1;
    if (!MOB_MASK(3, level, 2, state)) return 2;
    return 0;
}

static __device__ inline void craftax_update_mobs_spawn_mob_projectile(
    CraftaxState* state,
    int32_t level,
    bool is_spawning_projectile,
    const int32_t position[2],
    const int32_t direction[2],
    int32_t projectile_type
) {
    if (!is_spawning_projectile) {
        return;
    }

    int32_t index = craftax_update_mobs_first_empty_mob_projectile(
        state,
        level
    );
    MOB_POS(3, level, index, 0, state) = position[0];
    MOB_POS(3, level, index, 1, state) = position[1];
    MOB_MASK(3, level, index, state) = true;
    MOB_TYPE(3, level, index, state) = projectile_type;
    CF2(mob_projectile_directions, (level) * 6 + (index) * 2 + (0), state) = direction[0];
    CF2(mob_projectile_directions, (level) * 6 + (index) * 2 + (1), state) = direction[1];
}

static __device__ inline void craftax_update_mobs_attack_mob_with_damage(
    CraftaxState* state,
    int32_t row,
    int32_t col,
    const float damage_vector[3],
    bool can_eat,
    bool* did_attack_mob,
    bool* did_kill_mob
) {
    bool did_kill_melee_mob = false;
    bool is_attacking_melee_mob = false;
    craftax_do_action_attack_mobs3(
        state,
        state, 0,
        row,
        col,
        damage_vector,
        true,
        CRAFTAX_MOB_MELEE,
        &did_kill_melee_mob,
        &is_attacking_melee_mob
    );

    bool did_kill_passive_mob = false;
    bool is_attacking_passive_mob = false;
    craftax_do_action_attack_mobs3(
        state,
        state, 1,
        row,
        col,
        damage_vector,
        can_eat,
        CRAFTAX_MOB_PASSIVE,
        &did_kill_passive_mob,
        &is_attacking_passive_mob
    );

    if (did_kill_passive_mob && can_eat) {
        CF(player_food, state) = craftax_step_mini32(
            craftax_step_get_max_food(state),
            CF(player_food, state) + 6
        );
        CF(player_hunger, state) = 0.0f;
    }

    bool did_kill_ranged_mob = false;
    bool is_attacking_ranged_mob = false;
    craftax_do_action_attack_mobs2(
        state,
        state, 2,
        row,
        col,
        damage_vector,
        true,
        CRAFTAX_MOB_RANGED,
        &did_kill_ranged_mob,
        &is_attacking_ranged_mob
    );

    *did_attack_mob = is_attacking_melee_mob
        || is_attacking_passive_mob
        || is_attacking_ranged_mob;
    bool did_kill_monster = did_kill_melee_mob || did_kill_ranged_mob;
    *did_kill_mob = did_kill_monster || did_kill_passive_mob;

    craftax_do_action_update_mob_map(state, row, col, *did_kill_mob);

    int32_t level = craftax_step_jax_index(
        CF(player_level, state),
        CRAFTAX_NUM_LEVELS
    );
    CF2(monsters_killed, level, state) += (int32_t)did_kill_monster;
}

static __device__ inline void craftax_update_mobs_player_projectile_damage_vector(
    const CraftaxState* state,
    int32_t level,
    int32_t projectile_index,
    float damage_vector[3]
) {
    int32_t projectile_type =
        MOB_TYPE(4, level, projectile_index, state);
    craftax_update_mobs_damage_vector(
        projectile_type,
        CRAFTAX_MOB_PROJECTILE,
        damage_vector
    );

    float mask = (float)(int32_t)
        MOB_MASK(4, level, projectile_index, state);
    for (int32_t i = 0; i < 3; i++) {
        damage_vector[i] *= mask;
    }

    bool is_arrow = projectile_type == CRAFTAX_PROJECTILE_ARROW
        || projectile_type == CRAFTAX_PROJECTILE_ARROW2;
    if (is_arrow) {
        float arrow_damage_add[3] = {0.0f, 0.0f, 0.0f};
        int32_t enchantment_index;
        if (craftax_update_mobs_scatter_index(
                CF(bow_enchantment, state),
                3,
                &enchantment_index
            )) {
            arrow_damage_add[enchantment_index] = damage_vector[0] * (1.0f / 2.0f);
        }
        arrow_damage_add[0] = 0.0f;
        for (int32_t i = 0; i < 3; i++) {
            damage_vector[i] += arrow_damage_add[i];
        }
    }

    if (is_arrow) {
        float arrow_damage_coeff =
            1.0f + 0.2f * (float)(CF(player_dexterity, state) - 1);
        for (int32_t i = 0; i < 3; i++) {
            damage_vector[i] *= arrow_damage_coeff;
        }
    }

    bool is_magic_projectile = projectile_type == CRAFTAX_PROJECTILE_FIREBALL
        || projectile_type == CRAFTAX_PROJECTILE_ICEBALL;
    if (is_magic_projectile) {
        float magic_damage_coeff =
            1.0f + 0.5f * (float)(CF(player_intelligence, state) - 1);
        for (int32_t i = 0; i < 3; i++) {
            damage_vector[i] *= magic_damage_coeff;
        }
    }
}

static __device__ inline void craftax_update_mobs_move_melee(
    CraftaxState* state,
    CraftaxThreefryKey* rng,
    int32_t index
) {
    int32_t level = CF(player_level, state);
    bool old_mask = MOB_MASK(0, level, index, state);
    // Dead slot early-out: no observable effect on obs/reward/terminal.
    // Skip body and RNG draws for speed. Breaks per-seed replay against
    // JAX; define CRAFTAX_JAX_PARITY at build time to restore the
    // branchless slow path (same pattern in every move_* below).
#ifndef CRAFTAX_JAX_PARITY
    if (!old_mask) return;
#endif
    int32_t old_row = MOB_POS(0, level, index, 0, state);
    int32_t old_col = MOB_POS(0, level, index, 1, state);
    int32_t old_cooldown = MOB_CD(0, level, index, state);
    int32_t mob_type = MOB_TYPE(0, level, index, state);

    CraftaxThreefryKey draw_key =
        craftax_update_mobs_next_random_key(rng);
    int32_t random_direction[2];
    craftax_update_mobs_direction_choice(draw_key, 4, random_direction);
    int32_t random_row = old_row + random_direction[0];
    int32_t random_col = old_col + random_direction[1];

    int32_t distance_row =
        craftax_update_mobs_abs_i32(CF2(player_position, 0, state) - old_row);
    int32_t distance_col =
        craftax_update_mobs_abs_i32(CF2(player_position, 1, state) - old_col);
    draw_key = craftax_update_mobs_next_random_key(rng);
    int32_t player_move_axis = craftax_update_mobs_player_axis_choice(
        draw_key,
        distance_row,
        distance_col
    );
    int32_t player_direction[2] = {0, 0};
    if (player_move_axis == 0) {
        player_direction[0] =
            craftax_update_mobs_sign_i32(CF2(player_position, 0, state) - old_row);
    } else {
        player_direction[1] =
            craftax_update_mobs_sign_i32(CF2(player_position, 1, state) - old_col);
    }
    int32_t player_row = old_row + player_direction[0];
    int32_t player_col = old_col + player_direction[1];

    int32_t distance_to_player = distance_row + distance_col;
    bool close_to_player = distance_to_player < 10
        || craftax_step_is_fighting_boss(state);
    draw_key = craftax_update_mobs_next_random_key(rng);
    close_to_player = close_to_player
        && craftax_threefry_uniform_f32(draw_key) < 0.75f;

    int32_t proposed_row = close_to_player ? player_row : random_row;
    int32_t proposed_col = close_to_player ? player_col : random_col;

    bool is_attacking_player = distance_to_player == 1
        && old_cooldown <= 0
        && old_mask;
    if (is_attacking_player) {
        proposed_row = old_row;
        proposed_col = old_col;
    }

    float base_damage[3];
    craftax_update_mobs_damage_vector(
        mob_type,
        CRAFTAX_MOB_MELEE,
        base_damage
    );
    float sleeping_coeff = 1.0f + 2.5f * (float)(int32_t)CF(is_sleeping, state);
    for (int32_t i = 0; i < 3; i++) {
        base_damage[i] *= sleeping_coeff;
    }
    float damage = craftax_update_mobs_damage_done_to_player(
        state,
        base_damage
    );

    int32_t new_cooldown = is_attacking_player ? 5 : old_cooldown - 1;
    bool is_waking_player = CF(is_sleeping, state) && is_attacking_player;
    CF(player_health, state) -= damage * (float)(int32_t)is_attacking_player;
    CF(is_sleeping, state) = CF(is_sleeping, state) && !is_attacking_player;
    CF(is_resting, state) = CF(is_resting, state) && !is_attacking_player;
    CF2(achievements, CRAFTAX_ACH_WAKE_UP, state) =
        CF2(achievements, CRAFTAX_ACH_WAKE_UP, state) || is_waking_player;

    bool collision[3];
    craftax_update_mobs_collision_map(
        mob_type,
        CRAFTAX_MOB_MELEE,
        collision
    );
    bool valid_move = craftax_update_mobs_valid_position(
        state,
        proposed_row,
        proposed_col,
        collision
    );
    int32_t new_row = valid_move ? proposed_row : old_row;
    int32_t new_col = valid_move ? proposed_col : old_col;

    bool should_not_despawn = distance_to_player < CRAFTAX_MOB_DESPAWN_DISTANCE
        || craftax_step_is_fighting_boss(state);

    CraftaxThreefryKey unused_left;
    CraftaxThreefryKey returned_key;
    craftax_threefry_split(*rng, &unused_left, &returned_key);
    *rng = returned_key;

    craftax_update_mobs_clear_old_map_entry(
        state,
        level,
        old_row,
        old_col,
        old_mask
    );
    bool new_mask = old_mask && should_not_despawn;
    craftax_update_mobs_enter_new_map_entry(
        state,
        level,
        new_row,
        new_col,
        new_mask
    );

    MOB_POS(0, level, index, 0, state) = new_row;
    MOB_POS(0, level, index, 1, state) = new_col;
    MOB_CD(0, level, index, state) = new_cooldown;
    MOB_MASK(0, level, index, state) = new_mask;
}

static __device__ inline void craftax_update_mobs_move_passive(
    CraftaxState* state,
    CraftaxThreefryKey* rng,
    int32_t index
) {
    int32_t level = CF(player_level, state);
    bool old_mask = MOB_MASK(1, level, index, state);
#ifndef CRAFTAX_JAX_PARITY
    if (!old_mask) return;
#endif
    int32_t old_row = MOB_POS(1, level, index, 0, state);
    int32_t old_col = MOB_POS(1, level, index, 1, state);
    int32_t mob_type = MOB_TYPE(1, level, index, state);

    CraftaxThreefryKey draw_key =
        craftax_update_mobs_next_random_key(rng);
    int32_t direction[2];
    craftax_update_mobs_direction_choice(draw_key, 8, direction);
    int32_t proposed_row = old_row + direction[0];
    int32_t proposed_col = old_col + direction[1];

    bool collision[3];
    craftax_update_mobs_collision_map(
        mob_type,
        CRAFTAX_MOB_PASSIVE,
        collision
    );
    bool valid_move = craftax_update_mobs_valid_position(
        state,
        proposed_row,
        proposed_col,
        collision
    );
    int32_t new_row = valid_move ? proposed_row : old_row;
    int32_t new_col = valid_move ? proposed_col : old_col;

    int32_t distance_to_player = craftax_update_mobs_manhattan_to_player(
        state,
        old_row,
        old_col
    );
    bool should_not_despawn =
        distance_to_player < CRAFTAX_MOB_DESPAWN_DISTANCE;

    craftax_update_mobs_clear_old_map_entry(
        state,
        level,
        old_row,
        old_col,
        old_mask
    );
    bool new_mask = old_mask && should_not_despawn;
    craftax_update_mobs_enter_new_map_entry(
        state,
        level,
        new_row,
        new_col,
        new_mask
    );

    MOB_POS(1, level, index, 0, state) = new_row;
    MOB_POS(1, level, index, 1, state) = new_col;
    MOB_MASK(1, level, index, state) = new_mask;
}

static __device__ inline void craftax_update_mobs_move_ranged(
    CraftaxState* state,
    CraftaxThreefryKey* rng,
    int32_t index
) {
    int32_t level = CF(player_level, state);
    bool old_mask = MOB_MASK(2, level, index, state);
#ifndef CRAFTAX_JAX_PARITY
    if (!old_mask) return;
#endif
    int32_t old_row = MOB_POS(2, level, index, 0, state);
    int32_t old_col = MOB_POS(2, level, index, 1, state);
    int32_t old_cooldown = MOB_CD(2, level, index, state);
    int32_t mob_type = MOB_TYPE(2, level, index, state);

    CraftaxThreefryKey draw_key =
        craftax_update_mobs_next_random_key(rng);
    int32_t random_direction[2];
    craftax_update_mobs_direction_choice(draw_key, 4, random_direction);
    int32_t random_row = old_row + random_direction[0];
    int32_t random_col = old_col + random_direction[1];

    int32_t distance_row =
        craftax_update_mobs_abs_i32(CF2(player_position, 0, state) - old_row);
    int32_t distance_col =
        craftax_update_mobs_abs_i32(CF2(player_position, 1, state) - old_col);
    draw_key = craftax_update_mobs_next_random_key(rng);
    int32_t player_move_axis = craftax_update_mobs_player_axis_choice(
        draw_key,
        distance_row,
        distance_col
    );
    int32_t player_direction[2] = {0, 0};
    if (player_move_axis == 0) {
        player_direction[0] =
            craftax_update_mobs_sign_i32(CF2(player_position, 0, state) - old_row);
    } else {
        player_direction[1] =
            craftax_update_mobs_sign_i32(CF2(player_position, 1, state) - old_col);
    }
    int32_t towards_row = old_row + player_direction[0];
    int32_t towards_col = old_col + player_direction[1];
    int32_t away_row = old_row - player_direction[0];
    int32_t away_col = old_col - player_direction[1];

    int32_t distance_to_player = distance_row + distance_col;
    bool far_from_player = distance_to_player >= 6;
    bool too_close_to_player = distance_to_player <= 3;
    int32_t proposed_row = far_from_player ? towards_row : random_row;
    int32_t proposed_col = far_from_player ? towards_col : random_col;
    if (too_close_to_player) {
        proposed_row = away_row;
        proposed_col = away_col;
    }

    draw_key = craftax_update_mobs_next_random_key(rng);
    if (!(craftax_threefry_uniform_f32(draw_key) > 0.85f)) {
        proposed_row = random_row;
        proposed_col = random_col;
    }

    bool collision[3];
    craftax_update_mobs_collision_map(
        mob_type,
        CRAFTAX_MOB_RANGED,
        collision
    );

    bool is_attacking_player =
        distance_to_player >= 4 && distance_to_player <= 5;
    bool proposed_valid = craftax_update_mobs_valid_position(
        state,
        proposed_row,
        proposed_col,
        collision
    );
    is_attacking_player = is_attacking_player
        || (too_close_to_player && !proposed_valid);
    is_attacking_player = is_attacking_player
        && old_cooldown <= 0
        && old_mask;

    bool can_spawn_projectile =
        craftax_update_mobs_count_mob_projectiles(state, level)
            < CRAFTAX_MAX_MOB_PROJECTILES;
    bool is_spawning_projectile =
        is_attacking_player && can_spawn_projectile;
    int32_t projectile_position[2] = {old_row, old_col};
    int32_t projectile_type =
        craftax_update_mobs_projectile_type_for_ranged(mob_type);
    craftax_update_mobs_spawn_mob_projectile(
        state,
        level,
        is_spawning_projectile,
        projectile_position,
        player_direction,
        projectile_type
    );

    if (is_attacking_player) {
        proposed_row = old_row;
        proposed_col = old_col;
    }
    int32_t new_cooldown = is_attacking_player ? 4 : old_cooldown - 1;

    bool valid_move = craftax_update_mobs_valid_position(
        state,
        proposed_row,
        proposed_col,
        collision
    );
    int32_t new_row = valid_move ? proposed_row : old_row;
    int32_t new_col = valid_move ? proposed_col : old_col;

    bool should_not_despawn = distance_to_player < CRAFTAX_MOB_DESPAWN_DISTANCE
        || craftax_step_is_fighting_boss(state);

    craftax_update_mobs_clear_old_map_entry(
        state,
        level,
        old_row,
        old_col,
        old_mask
    );
    bool new_mask = old_mask && should_not_despawn;
    craftax_update_mobs_enter_new_map_entry(
        state,
        level,
        new_row,
        new_col,
        new_mask
    );

    MOB_POS(2, level, index, 0, state) = new_row;
    MOB_POS(2, level, index, 1, state) = new_col;
    MOB_CD(2, level, index, state) = new_cooldown;
    MOB_MASK(2, level, index, state) = new_mask;
}

static __device__ inline void craftax_update_mobs_move_mob_projectile(
    CraftaxState* state,
    int32_t index
) {
    int32_t level = CF(player_level, state);
    bool old_mask = MOB_MASK(3, level, index, state);
#ifndef CRAFTAX_JAX_PARITY
    if (!old_mask) return;
#endif
    int32_t old_row = MOB_POS(3, level, index, 0, state);
    int32_t old_col = MOB_POS(3, level, index, 1, state);
    int32_t proposed_row =
        old_row + CF2(mob_projectile_directions, (level) * 6 + (index) * 2 + (0), state);
    int32_t proposed_col =
        old_col + CF2(mob_projectile_directions, (level) * 6 + (index) * 2 + (1), state);

    bool proposed_in_player =
        proposed_row == CF2(player_position, 0, state)
        && proposed_col == CF2(player_position, 1, state);
    bool proposed_in_bounds = craftax_update_mobs_in_bounds(
        proposed_row,
        proposed_col
    );
    int32_t proposed_block = craftax_update_mobs_read_block(
        state,
        level,
        proposed_row,
        proposed_col
    );
    bool in_wall = craftax_step_is_solid_block(proposed_block)
        && proposed_block != CRAFTAX_BLOCK_WATER;
    bool in_mob = craftax_step_is_in_mob(state, proposed_row, proposed_col);
    bool continue_move = proposed_in_bounds && !in_wall && !in_mob;

    bool hit_player0 =
        old_row == CF2(player_position, 0, state)
        && old_col == CF2(player_position, 1, state)
        && old_mask;
    bool hit_player1 = proposed_in_player && old_mask;
    bool hit_player = hit_player0 || hit_player1;
    continue_move = continue_move && !hit_player;

    bool new_mask = continue_move && old_mask;

    bool hit_bench_or_furnace = proposed_block == CRAFTAX_BLOCK_FURNACE
        || proposed_block == CRAFTAX_BLOCK_CRAFTING_TABLE;
    bool removing_block = hit_bench_or_furnace && old_mask;
    int32_t new_block = removing_block ? CRAFTAX_BLOCK_PATH : proposed_block;

    int32_t projectile_type =
        MOB_TYPE(3, level, index, state);
    float damage_vector[3];
    craftax_update_mobs_damage_vector(
        projectile_type,
        CRAFTAX_MOB_PROJECTILE,
        damage_vector
    );
    float damage = craftax_update_mobs_damage_done_to_player(
        state,
        damage_vector
    );

    MOB_POS(3, level, index, 0, state) = proposed_row;
    MOB_POS(3, level, index, 1, state) = proposed_col;
    MOB_MASK(3, level, index, state) = new_mask;
    CF(player_health, state) -= damage * (float)(int32_t)hit_player;
    CF(is_sleeping, state) = CF(is_sleeping, state) && !hit_player;
    CF(is_resting, state) = CF(is_resting, state) && !hit_player;
    craftax_update_mobs_set_block(
        state,
        level,
        proposed_row,
        proposed_col,
        new_block
    );
}

static __device__ inline void craftax_update_mobs_move_player_projectile(
    CraftaxState* state,
    int32_t index
) {
    int32_t level = CF(player_level, state);
    bool old_mask = MOB_MASK(4, level, index, state);
#ifndef CRAFTAX_JAX_PARITY
    if (!old_mask) return;
#endif
    int32_t old_row = MOB_POS(4, level, index, 0, state);
    int32_t old_col = MOB_POS(4, level, index, 1, state);
    int32_t proposed_row =
        old_row + CF2(player_projectile_directions, (level) * 6 + (index) * 2 + (0), state);
    int32_t proposed_col =
        old_col + CF2(player_projectile_directions, (level) * 6 + (index) * 2 + (1), state);

    float damage_vector[3];
    craftax_update_mobs_player_projectile_damage_vector(
        state,
        level,
        index,
        damage_vector
    );

    bool proposed_in_bounds = craftax_update_mobs_in_bounds(
        proposed_row,
        proposed_col
    );
    int32_t proposed_block = craftax_update_mobs_read_block(
        state,
        level,
        proposed_row,
        proposed_col
    );
    bool in_wall = craftax_step_is_solid_block(proposed_block)
        && proposed_block != CRAFTAX_BLOCK_WATER;

    bool did_attack_mob0 = false;
    bool did_kill_mob0 = false;
    craftax_update_mobs_attack_mob_with_damage(
        state,
        old_row,
        old_col,
        damage_vector,
        false,
        &did_attack_mob0,
        &did_kill_mob0
    );
    (void)did_kill_mob0;

    float second_damage_vector[3];
    for (int32_t i = 0; i < 3; i++) {
        second_damage_vector[i] =
            damage_vector[i] * (float)(int32_t)(!did_attack_mob0);
    }

    bool did_attack_mob1 = false;
    bool did_kill_mob1 = false;
    craftax_update_mobs_attack_mob_with_damage(
        state,
        proposed_row,
        proposed_col,
        second_damage_vector,
        false,
        &did_attack_mob1,
        &did_kill_mob1
    );
    (void)did_kill_mob1;

    bool did_attack_mob = did_attack_mob0 || did_attack_mob1;
    bool continue_move = proposed_in_bounds && !in_wall && !did_attack_mob;
    bool new_mask = continue_move && old_mask;

    MOB_POS(4, level, index, 0, state) = proposed_row;
    MOB_POS(4, level, index, 1, state) = proposed_col;
    MOB_MASK(4, level, index, state) = new_mask;
}

static __device__ inline void craftax_update_mobs_native(
    CraftaxState* state,
    CraftaxThreefryKey rng
) {
    CraftaxThreefryKey unused;

    craftax_threefry_split(rng, &rng, &unused);
    craftax_update_mobs_move_melee(state, &rng, 0);
    craftax_update_mobs_move_melee(state, &rng, 1);
    craftax_update_mobs_move_melee(state, &rng, 2);

    craftax_threefry_split(rng, &rng, &unused);
    craftax_update_mobs_move_passive(state, &rng, 0);
    craftax_update_mobs_move_passive(state, &rng, 1);
    craftax_update_mobs_move_passive(state, &rng, 2);

    craftax_threefry_split(rng, &rng, &unused);
    craftax_update_mobs_move_ranged(state, &rng, 0);
    craftax_update_mobs_move_ranged(state, &rng, 1);

    craftax_threefry_split(rng, &rng, &unused);
    craftax_update_mobs_move_mob_projectile(state, 0);
    craftax_update_mobs_move_mob_projectile(state, 1);
    craftax_update_mobs_move_mob_projectile(state, 2);

    craftax_threefry_split(rng, &rng, &unused);
    craftax_update_mobs_move_player_projectile(state, 0);
    craftax_update_mobs_move_player_projectile(state, 1);
    craftax_update_mobs_move_player_projectile(state, 2);
}

// ============================================================
// ===== step_spawn_mobs.h =====
// ============================================================
// Craftax spawn_mobs, optimized for CPU.
//
// Bitwise-equivalent to the prior JAX-transliterated baseline (verified by
// ocean/craftax_exp/parity_vs_baseline.c over 1.28M paired steps), ~6-9x
// faster per step by stripping JAX-isms:
//   - full-grid validity masks -> compact coord list collected in one pass
//   - bounding-box scan (only cells within MOB_DESPAWN_DISTANCE)
//   - early return on mob-cap / probability-roll failure (no dead writes)
//   - merged count + first_empty loops
//
// The prior reference implementation is archived at
// ocean/craftax_exp/step_spawn_mobs_baseline.h.



#define CRAFTAX_SPAWN_MAP_CELLS (CRAFTAX_MAP_SIZE * CRAFTAX_MAP_SIZE)
#define CRAFTAX_SPAWN_BBOX_MAX_CELLS 729  // (2*DESPAWN-1)^2 at 14 = 27*27
#define CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK ( \
    (1ULL << CRAFTAX_BLOCK_GRASS) \
    | (1ULL << CRAFTAX_BLOCK_PATH) \
    | (1ULL << CRAFTAX_BLOCK_FIRE_GRASS) \
    | (1ULL << CRAFTAX_BLOCK_ICE_GRASS))
#define CRAFTAX_SPAWN_GRAVE_BLOCK_MASK ( \
    (1ULL << CRAFTAX_BLOCK_GRAVE) \
    | (1ULL << CRAFTAX_BLOCK_GRAVE2) \
    | (1ULL << CRAFTAX_BLOCK_GRAVE3))
#define CRAFTAX_SPAWN_WATER_BLOCK_MASK (1ULL << CRAFTAX_BLOCK_WATER)

typedef struct { int8_t dr, dc0, dc1; } CraftaxSpawnOffsetSpan;

static __device__ CraftaxSpawnOffsetSpan craftax_spawn_passive_spans[CRAFTAX_SPAWN_BBOX_MAX_CELLS];
static __device__ CraftaxSpawnOffsetSpan craftax_spawn_hostile_spans[CRAFTAX_SPAWN_BBOX_MAX_CELLS];
static __device__ CraftaxSpawnOffsetSpan craftax_spawn_boss_spans[CRAFTAX_SPAWN_BBOX_MAX_CELLS];
static __device__ int32_t craftax_spawn_passive_span_count = 0;
static __device__ int32_t craftax_spawn_hostile_span_count = 0;
static __device__ int32_t craftax_spawn_boss_span_count = 0;
static __device__ int32_t craftax_spawn_offsets_initialized = 0;

static __device__ inline void craftax_spawn_append_span(
    CraftaxSpawnOffsetSpan* spans,
    int32_t* count,
    int32_t dr,
    int32_t dc0,
    int32_t dc1
) {
    spans[*count] = (CraftaxSpawnOffsetSpan){
        (int8_t)dr, (int8_t)dc0, (int8_t)dc1
    };
    *count += 1;
}

static __device__ inline void craftax_spawn_build_spans_for_row(
    CraftaxSpawnOffsetSpan* spans,
    int32_t* count,
    int32_t dr,
    int32_t limit,
    int32_t min_exclusive,
    int32_t max_exclusive
) {
    bool active = false;
    int32_t start = 0;
    for (int32_t dc = -limit; dc <= limit; dc++) {
        int32_t distance2 = dr * dr + dc * dc;
        bool valid = distance2 > min_exclusive && distance2 < max_exclusive;
        if (valid && !active) {
            active = true;
            start = dc;
        } else if (!valid && active) {
            craftax_spawn_append_span(spans, count, dr, start, dc - 1);
            active = false;
        }
    }
    if (active) {
        craftax_spawn_append_span(spans, count, dr, start, limit);
    }
}

static __device__ inline void craftax_spawn_init_offsets_once(void) {
    if (__atomic_load_n(
            &craftax_spawn_offsets_initialized, __ATOMIC_ACQUIRE
    )) return;

    // [cuda port] #pragma omp removed
    {
        if (!__atomic_load_n(
                &craftax_spawn_offsets_initialized, __ATOMIC_RELAXED
        )) {
            int32_t passive_count = 0;
            int32_t hostile_count = 0;
            int32_t boss_count = 0;
            int32_t limit = CRAFTAX_MOB_DESPAWN_DISTANCE - 1;
            int32_t limit2 = CRAFTAX_MOB_DESPAWN_DISTANCE
                           * CRAFTAX_MOB_DESPAWN_DISTANCE;
            for (int32_t dr = -limit; dr <= limit; dr++) {
                craftax_spawn_build_spans_for_row(
                    craftax_spawn_passive_spans,
                    &passive_count,
                    dr,
                    limit,
                    9,
                    limit2
                );
                craftax_spawn_build_spans_for_row(
                    craftax_spawn_hostile_spans,
                    &hostile_count,
                    dr,
                    limit,
                    81,
                    limit2
                );
                craftax_spawn_build_spans_for_row(
                    craftax_spawn_boss_spans,
                    &boss_count,
                    dr,
                    limit,
                    -1,
                    37
                );
            }
            craftax_spawn_passive_span_count = passive_count;
            craftax_spawn_hostile_span_count = hostile_count;
            craftax_spawn_boss_span_count = boss_count;
            __atomic_store_n(
                &craftax_spawn_offsets_initialized, 1, __ATOMIC_RELEASE
            );
        }
    }
}

static __device__ inline bool craftax_spawn_block_matches(uint8_t block, uint64_t mask) {
    return ((mask >> block) & 1ULL) != 0;
}

static __device__ inline uint64_t craftax_spawn_row_bits_for_mask(
    const CraftaxState* state,
    int32_t level,
    int32_t row,
    uint64_t terrain_mask
) {
    if (terrain_mask == CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK) {
        return CF_BITS(spawn_all_bits, level, row, state);
    }
    if (terrain_mask == CRAFTAX_SPAWN_GRAVE_BLOCK_MASK) {
        return CF_BITS(spawn_grave_bits, level, row, state);
    }
    return CF_BITS(spawn_water_bits, level, row, state);
}

static __device__ inline uint64_t craftax_spawn_col_mask(int32_t col0, int32_t col1) {
    uint64_t hi = (1ULL << (col1 + 1)) - 1ULL;
    uint64_t lo = col0 <= 0 ? 0ULL : ((1ULL << col0) - 1ULL);
    return hi & ~lo;
}

static __device__ inline CraftaxThreefryKey craftax_spawn_next_random_key(
    CraftaxThreefryKey* rng
) {
    CraftaxThreefryKey draw;
    craftax_threefry_split(*rng, rng, &draw);
    return draw;
}

static __device__ inline int32_t craftax_spawn_floor_mob_type(
    int32_t floor, int32_t mob_class
) {
    static const int32_t mapping[CRAFTAX_NUM_LEVELS][3] = {
        {0, 0, 0}, {2, 2, 2}, {1, 1, 1}, {2, 3, 3}, {2, 4, 4},
        {1, 5, 5}, {1, 6, 6}, {1, 7, 7}, {0, 0, 0},
    };
    int32_t level = craftax_step_jax_index(floor, CRAFTAX_NUM_LEVELS);
    int32_t class_index = craftax_step_jax_index(mob_class, 3);
    return mapping[level][class_index];
}

static __device__ inline float craftax_spawn_floor_spawn_chance(
    int32_t floor, int32_t chance_index
) {
    static const float chances[CRAFTAX_NUM_LEVELS][4] = {
        {0.1f, 0.02f, 0.05f, 0.1f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.0f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
    };
    int32_t level = craftax_step_jax_index(floor, CRAFTAX_NUM_LEVELS);
    int32_t index = craftax_step_jax_index(chance_index, 4);
    return chances[level][index];
}

static __device__ inline float craftax_spawn_mob_type_health(
    int32_t mob_type, int32_t mob_class
) {
    static const float health[CRAFTAX_NUM_MOB_TYPES][4] = {
        {3.0f, 5.0f, 3.0f, 0.0f}, {4.0f, 7.0f, 5.0f, 0.0f},
        {6.0f, 9.0f, 6.0f, 0.0f}, {8.0f, 11.0f, 8.0f, 0.0f},
        {0.0f, 12.0f, 12.0f, 0.0f}, {0.0f, 20.0f, 4.0f, 0.0f},
        {0.0f, 20.0f, 14.0f, 0.0f}, {0.0f, 24.0f, 16.0f, 0.0f},
    };
    int32_t type_index = craftax_step_jax_index(mob_type, CRAFTAX_NUM_MOB_TYPES);
    int32_t class_index = craftax_step_jax_index(mob_class, 4);
    return health[type_index][class_index];
}

static __device__ inline bool craftax_spawn_is_all_valid_block(int32_t block) {
    static const uint8_t flags[CRAFTAX_NUM_BLOCK_TYPES] = {
        [CRAFTAX_BLOCK_GRASS] = 1,
        [CRAFTAX_BLOCK_PATH] = 1,
        [CRAFTAX_BLOCK_FIRE_GRASS] = 1,
        [CRAFTAX_BLOCK_ICE_GRASS] = 1,
    };
    int32_t idx = craftax_step_jax_index(block, CRAFTAX_NUM_BLOCK_TYPES);
    return flags[idx] != 0;
}

static __device__ inline bool craftax_spawn_is_grave_block(int32_t block) {
    static const uint8_t flags[CRAFTAX_NUM_BLOCK_TYPES] = {
        [CRAFTAX_BLOCK_GRAVE] = 1,
        [CRAFTAX_BLOCK_GRAVE2] = 1,
        [CRAFTAX_BLOCK_GRAVE3] = 1,
    };
    int32_t idx = craftax_step_jax_index(block, CRAFTAX_NUM_BLOCK_TYPES);
    return flags[idx] != 0;
}

static __device__ inline bool craftax_spawn_is_water_block(int32_t block) {
    static const uint8_t flags[CRAFTAX_NUM_BLOCK_TYPES] = {
        [CRAFTAX_BLOCK_WATER] = 1,
    };
    int32_t idx = craftax_step_jax_index(block, CRAFTAX_NUM_BLOCK_TYPES);
    return flags[idx] != 0;
}

static __device__ inline int32_t craftax_spawn_player_distance_squared(
    const CraftaxState* state, int32_t row, int32_t col
) {
    int32_t dr = row - CF2(player_position, 0, state);
    int32_t dc = col - CF2(player_position, 1, state);
    if (dr < 0) dr = -dr;
    if (dc < 0) dc = -dc;
    return dr * dr + dc * dc;
}

static __device__ inline int32_t craftax_spawn_count_mobs3(
    const void* mobs, int mc, int32_t level
) {
    int32_t count = 0;
    for (int32_t i = 0; i < 3; i++) count += (int32_t)MOB_MASK(mc, level, i, mobs);
    return count;
}

static __device__ inline int32_t craftax_spawn_count_mobs2(
    const void* mobs, int mc, int32_t level
) {
    int32_t count = 0;
    for (int32_t i = 0; i < 2; i++) count += (int32_t)MOB_MASK(mc, level, i, mobs);
    return count;
}

static __device__ inline int32_t craftax_spawn_first_empty_mobs3(
    const void* mobs, int mc, int32_t level
) {
    for (int32_t i = 0; i < 3; i++) if (!MOB_MASK(mc, level, i, mobs)) return i;
    return 0;
}

static __device__ inline int32_t craftax_spawn_first_empty_mobs2(
    const void* mobs, int mc, int32_t level
) {
    for (int32_t i = 0; i < 2; i++) if (!MOB_MASK(mc, level, i, mobs)) return i;
    return 0;
}

static __device__ inline void craftax_spawn_mobs3_count_and_empty(
    const void* mobs, int mc, int32_t level,
    int32_t* count_out, int32_t* first_empty_out
) {
    int32_t count = 0, first_empty = 0;
    bool found = false;
    for (int32_t i = 0; i < 3; i++) {
        bool m = MOB_MASK(mc, level, i, mobs);
        count += (int32_t)m;
        if (!m && !found) { first_empty = i; found = true; }
    }
    *count_out = count;
    *first_empty_out = first_empty;
}

static __device__ inline void craftax_spawn_mobs2_count_and_empty(
    const void* mobs, int mc, int32_t level,
    int32_t* count_out, int32_t* first_empty_out
) {
    int32_t count = 0, first_empty = 0;
    bool found = false;
    for (int32_t i = 0; i < 2; i++) {
        bool m = MOB_MASK(mc, level, i, mobs);
        count += (int32_t)m;
        if (!m && !found) { first_empty = i; found = true; }
    }
    *count_out = count;
    *first_empty_out = first_empty;
}

// Baseline algorithm on a bool mask:
//   draw = valid_count * (1.0 - uniform_f32(key));
//   cum = 0;
//   for i: if valid[i] { cum += 1.0; if (cum >= draw) return i; }
// Over a compact list of length valid_count this collapses to a short loop
// using the same FP arithmetic, preserving bitwise-identical choice.
static __device__ inline int32_t craftax_spawn_pick_kth(
    int32_t valid_count, CraftaxThreefryKey key
) {
    float draw = (float)valid_count * (1.0f - craftax_threefry_uniform_f32(key));
    float cum = 0.0f;
    for (int32_t k = 0; k < valid_count; k++) {
        cum += 1.0f;
        if (cum >= draw) return k;
    }
    return valid_count - 1;
}

typedef struct { int16_t row, col; } CraftaxSpawnCoord;

typedef struct {
    CraftaxSpawnCoord passive[CRAFTAX_SPAWN_BBOX_MAX_CELLS];
    CraftaxSpawnCoord melee[CRAFTAX_SPAWN_BBOX_MAX_CELLS];
    CraftaxSpawnCoord ranged[CRAFTAX_SPAWN_BBOX_MAX_CELLS];
    int32_t passive_count;
    int32_t melee_count;
    int32_t ranged_count;
} CraftaxSpawnLists;

static __device__ inline int32_t craftax_spawn_collect_spans(
    const CraftaxState* state,
    int32_t level,
    const CraftaxSpawnOffsetSpan* spans,
    int32_t span_count,
    uint64_t terrain_mask,
    CraftaxSpawnCoord* coords
) {
    int32_t pr = CF2(player_position, 0, state);
    int32_t pc = CF2(player_position, 1, state);
    int32_t n = 0;
    for (int32_t i = 0; i < span_count; i++) {
        int32_t row = pr + spans[i].dr;
        if ((uint32_t)row >= CRAFTAX_MAP_SIZE) continue;
        int32_t col0 = pc + spans[i].dc0;
        int32_t col1 = pc + spans[i].dc1;
        if (col0 < 0) col0 = 0;
        if (col1 >= CRAFTAX_MAP_SIZE) col1 = CRAFTAX_MAP_SIZE - 1;
        if (col0 > col1) continue;
        uint64_t candidates =
            craftax_spawn_row_bits_for_mask(state, level, row, terrain_mask)
            & ~CF_BITS(mob_bits, level, row, state)
            & craftax_spawn_col_mask(col0, col1);
        while (candidates != 0) {
            int32_t col = __builtin_ctzll(candidates);
            coords[n].row = (int16_t)row;
            coords[n].col = (int16_t)col;
            n++;
            candidates &= candidates - 1;
        }
    }
    return n;
}

static __device__ inline bool craftax_spawn_scan_spans(
    const CraftaxState* state,
    int32_t level,
    const CraftaxSpawnOffsetSpan* spans,
    int32_t span_count,
    uint64_t terrain_mask,
    CraftaxThreefryKey pos_key,
    int32_t* out_row,
    int32_t* out_col
) {
    // [cuda port] Two-pass count+select instead of materialising the
    // candidate list (which cost hundreds of local-memory writes per scan).
    // Pass 1 counts candidates with popcounts; the k-th pick reproduces
    // craftax_spawn_pick_kth bitwise (cum goes 1.0f, 2.0f, ... exactly, so
    // the first k with (float)(k+1) >= draw is ceilf(draw) - 1); pass 2
    // walks the same span order to the k-th set bit.
    int32_t pr = CF2(player_position, 0, state);
    int32_t pc = CF2(player_position, 1, state);
    int32_t n = 0;
    for (int32_t i = 0; i < span_count; i++) {
        int32_t row = pr + spans[i].dr;
        if ((uint32_t)row >= CRAFTAX_MAP_SIZE) continue;
        int32_t col0 = pc + spans[i].dc0;
        int32_t col1 = pc + spans[i].dc1;
        if (col0 < 0) col0 = 0;
        if (col1 >= CRAFTAX_MAP_SIZE) col1 = CRAFTAX_MAP_SIZE - 1;
        if (col0 > col1) continue;
        uint64_t candidates =
            craftax_spawn_row_bits_for_mask(state, level, row, terrain_mask)
            & ~CF_BITS(mob_bits, level, row, state)
            & craftax_spawn_col_mask(col0, col1);
        n += __popcll(candidates);
    }
    if (n == 0) return false;

    float draw = (float)n * (1.0f - craftax_threefry_uniform_f32(pos_key));
    int32_t k = (int32_t)ceilf(draw) - 1;
    if (k < 0) k = 0;
    if (k > n - 1) k = n - 1;

    int32_t seen = 0;
    for (int32_t i = 0; i < span_count; i++) {
        int32_t row = pr + spans[i].dr;
        if ((uint32_t)row >= CRAFTAX_MAP_SIZE) continue;
        int32_t col0 = pc + spans[i].dc0;
        int32_t col1 = pc + spans[i].dc1;
        if (col0 < 0) col0 = 0;
        if (col1 >= CRAFTAX_MAP_SIZE) col1 = CRAFTAX_MAP_SIZE - 1;
        if (col0 > col1) continue;
        uint64_t candidates =
            craftax_spawn_row_bits_for_mask(state, level, row, terrain_mask)
            & ~CF_BITS(mob_bits, level, row, state)
            & craftax_spawn_col_mask(col0, col1);
        int32_t c = __popcll(candidates);
        if (seen + c <= k) {
            seen += c;
            continue;
        }
        int32_t need = k - seen;
        while (need-- > 0) candidates &= candidates - 1;
        *out_row = row;
        *out_col = __builtin_ctzll(candidates);
        return true;
    }
    return false;
}

static __device__ inline bool craftax_spawn_coord_matches(
    CraftaxSpawnCoord coord, bool exclude, int32_t row, int32_t col
) {
    return exclude && coord.row == row && coord.col == col;
}

static __device__ inline bool craftax_spawn_pick_excluding(
    const CraftaxSpawnCoord* coords, int32_t count, CraftaxThreefryKey key,
    bool exclude_a, int32_t row_a, int32_t col_a,
    bool exclude_b, int32_t row_b, int32_t col_b,
    int32_t* out_row, int32_t* out_col
) {
    int32_t valid_count = 0;
    for (int32_t i = 0; i < count; i++) {
        bool excluded = craftax_spawn_coord_matches(
            coords[i], exclude_a, row_a, col_a
        ) || craftax_spawn_coord_matches(coords[i], exclude_b, row_b, col_b);
        valid_count += excluded ? 0 : 1;
    }
    if (valid_count == 0) return false;

    int32_t k = craftax_spawn_pick_kth(valid_count, key);
    for (int32_t i = 0; i < count; i++) {
        bool excluded = craftax_spawn_coord_matches(
            coords[i], exclude_a, row_a, col_a
        ) || craftax_spawn_coord_matches(coords[i], exclude_b, row_b, col_b);
        if (excluded) continue;
        if (k == 0) {
            *out_row = coords[i].row;
            *out_col = coords[i].col;
            return true;
        }
        k--;
    }
    return false;
}

static __device__ inline void craftax_spawn_scan_all(
    const CraftaxState* state,
    int32_t level,
    int32_t ranged_type,
    bool fighting_boss,
    bool need_passive,
    bool need_melee,
    bool need_ranged,
    CraftaxSpawnLists* out
) {
    out->passive_count = 0;
    out->melee_count = 0;
    out->ranged_count = 0;

    craftax_spawn_init_offsets_once();

    if (need_passive) {
        out->passive_count = craftax_spawn_collect_spans(
            state,
            level,
            craftax_spawn_passive_spans,
            craftax_spawn_passive_span_count,
            CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK,
            out->passive
        );
    }

    if (!need_melee && !need_ranged) return;

    int32_t pr = CF2(player_position, 0, state);
    int32_t pc = CF2(player_position, 1, state);
    const CraftaxSpawnOffsetSpan* spans = fighting_boss
        ? craftax_spawn_boss_spans
        : craftax_spawn_hostile_spans;
    int32_t span_count = fighting_boss
        ? craftax_spawn_boss_span_count
        : craftax_spawn_hostile_span_count;
    bool ranged_water_type = (ranged_type == 5);

    uint64_t melee_terrain_mask = fighting_boss
        ? CRAFTAX_SPAWN_GRAVE_BLOCK_MASK
        : CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK;
    uint64_t ranged_terrain_mask;
    if (fighting_boss) {
        ranged_terrain_mask = CRAFTAX_SPAWN_GRAVE_BLOCK_MASK;
    } else if (ranged_water_type) {
        ranged_terrain_mask = CRAFTAX_SPAWN_WATER_BLOCK_MASK;
    } else {
        ranged_terrain_mask = CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK;
    }

    for (int32_t i = 0; i < span_count; i++) {
        int32_t row = pr + spans[i].dr;
        if ((uint32_t)row >= CRAFTAX_MAP_SIZE) continue;
        int32_t col0 = pc + spans[i].dc0;
        int32_t col1 = pc + spans[i].dc1;
        if (col0 < 0) col0 = 0;
        if (col1 >= CRAFTAX_MAP_SIZE) col1 = CRAFTAX_MAP_SIZE - 1;
        if (col0 > col1) continue;
        uint64_t open_bits =
            ~CF_BITS(mob_bits, level, row, state) & craftax_spawn_col_mask(col0, col1);

        if (need_melee) {
            uint64_t melee_candidates =
                craftax_spawn_row_bits_for_mask(
                    state, level, row, melee_terrain_mask
                ) & open_bits;
            while (melee_candidates != 0) {
                int32_t col = __builtin_ctzll(melee_candidates);
                int32_t n = out->melee_count++;
                out->melee[n].row = (int16_t)row;
                out->melee[n].col = (int16_t)col;
                melee_candidates &= melee_candidates - 1;
            }
        }

        if (need_ranged) {
            uint64_t ranged_candidates =
                craftax_spawn_row_bits_for_mask(
                    state, level, row, ranged_terrain_mask
                ) & open_bits;
            while (ranged_candidates != 0) {
                int32_t col = __builtin_ctzll(ranged_candidates);
                int32_t n = out->ranged_count++;
                out->ranged[n].row = (int16_t)row;
                out->ranged[n].col = (int16_t)col;
                ranged_candidates &= ranged_candidates - 1;
            }
        }
    }
}

static __device__ inline bool craftax_spawn_scan_passive(
    const CraftaxState* state, int32_t level, CraftaxThreefryKey pos_key,
    int32_t* out_row, int32_t* out_col
) {
    craftax_spawn_init_offsets_once();
    return craftax_spawn_scan_spans(
        state,
        level,
        craftax_spawn_passive_spans,
        craftax_spawn_passive_span_count,
        CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK,
        pos_key,
        out_row,
        out_col
    );
}

static __device__ inline bool craftax_spawn_scan_melee(
    const CraftaxState* state, int32_t level, bool fighting_boss,
    CraftaxThreefryKey pos_key, int32_t* out_row, int32_t* out_col
) {
    craftax_spawn_init_offsets_once();
    const CraftaxSpawnOffsetSpan* spans = fighting_boss
        ? craftax_spawn_boss_spans
        : craftax_spawn_hostile_spans;
    int32_t span_count = fighting_boss
        ? craftax_spawn_boss_span_count
        : craftax_spawn_hostile_span_count;
    uint64_t terrain_mask = fighting_boss
        ? CRAFTAX_SPAWN_GRAVE_BLOCK_MASK
        : CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK;
    return craftax_spawn_scan_spans(
        state, level, spans, span_count, terrain_mask, pos_key,
        out_row, out_col
    );
}

static __device__ inline bool craftax_spawn_scan_ranged(
    const CraftaxState* state, int32_t level, int32_t new_type,
    bool fighting_boss, CraftaxThreefryKey pos_key,
    int32_t* out_row, int32_t* out_col
) {
    craftax_spawn_init_offsets_once();
    const CraftaxSpawnOffsetSpan* spans = fighting_boss
        ? craftax_spawn_boss_spans
        : craftax_spawn_hostile_spans;
    int32_t span_count = fighting_boss
        ? craftax_spawn_boss_span_count
        : craftax_spawn_hostile_span_count;
    uint64_t terrain_mask;
    if (fighting_boss) {
        terrain_mask = CRAFTAX_SPAWN_GRAVE_BLOCK_MASK;
    } else if (new_type == 5) {
        terrain_mask = CRAFTAX_SPAWN_WATER_BLOCK_MASK;
    } else {
        terrain_mask = CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK;
    }
    return craftax_spawn_scan_spans(
        state, level, spans, span_count, terrain_mask, pos_key,
        out_row, out_col
    );
}

// Both RNG keys are always consumed (preserves baseline RNG sequence).
// Baseline quirk: type_id[level][slot] is written unconditionally, even
// when no mob spawns. We match that for bitwise parity.

static __device__ inline void craftax_spawn_passive_mob(
    CraftaxState* state, CraftaxThreefryKey* rng,
    int32_t level, bool fighting_boss
) {
    int32_t count, slot;
    craftax_spawn_mobs3_count_and_empty(state, 1, level, &count, &slot);

    CraftaxThreefryKey prob_key = craftax_spawn_next_random_key(rng);
    CraftaxThreefryKey pos_key  = craftax_spawn_next_random_key(rng);

    int32_t type = craftax_spawn_floor_mob_type(level, CRAFTAX_MOB_PASSIVE);
    MOB_TYPE(1, level, slot, state) = type;

    if (fighting_boss) return;
    if (count >= CRAFTAX_MAX_PASSIVE_MOBS) return;
    if (craftax_threefry_uniform_f32(prob_key)
        >= craftax_spawn_floor_spawn_chance(level, 0)) return;

    int32_t row, col;
    if (!craftax_spawn_scan_passive(state, level, pos_key, &row, &col)) return;

    MOB_POS(1, level, slot, 0, state) = row;
    MOB_POS(1, level, slot, 1, state) = col;
    MOB_HP(1, level, slot, state)      =
        craftax_spawn_mob_type_health(type, CRAFTAX_MOB_PASSIVE);
    MOB_MASK(1, level, slot, state)        = true;
    CF_BITS(mob_bits, level, row, state) |= (1ULL << col);
}

static __device__ inline void craftax_spawn_melee_mob(
    CraftaxState* state, CraftaxThreefryKey* rng,
    int32_t level, bool fighting_boss, int32_t monster_spawn_coeff
) {
    int32_t count, slot;
    craftax_spawn_mobs3_count_and_empty(state, 0, level, &count, &slot);

    int32_t type = fighting_boss
        ? craftax_spawn_floor_mob_type(CF(boss_progress, state), CRAFTAX_MOB_MELEE)
        : craftax_spawn_floor_mob_type(level, CRAFTAX_MOB_MELEE);

    CraftaxThreefryKey prob_key = craftax_spawn_next_random_key(rng);
    float night_coeff = 1.0f - CF(light_level, state);
    float spawn_chance = craftax_spawn_floor_spawn_chance(level, 1)
        + craftax_spawn_floor_spawn_chance(level, 3) * night_coeff * night_coeff;
    CraftaxThreefryKey pos_key = craftax_spawn_next_random_key(rng);

    MOB_TYPE(0, level, slot, state) = type;

    if (count >= CRAFTAX_MAX_MELEE_MOBS) return;
    if (craftax_threefry_uniform_f32(prob_key)
        >= spawn_chance * (float)monster_spawn_coeff) return;

    int32_t row, col;
    if (!craftax_spawn_scan_melee(state, level, fighting_boss, pos_key, &row, &col))
        return;

    MOB_POS(0, level, slot, 0, state) = row;
    MOB_POS(0, level, slot, 1, state) = col;
    MOB_HP(0, level, slot, state)      =
        craftax_spawn_mob_type_health(type, CRAFTAX_MOB_MELEE);
    MOB_MASK(0, level, slot, state)        = true;
    CF_BITS(mob_bits, level, row, state) |= (1ULL << col);
}

static __device__ inline void craftax_spawn_ranged_mob(
    CraftaxState* state, CraftaxThreefryKey* rng,
    int32_t level, bool fighting_boss, int32_t monster_spawn_coeff
) {
    int32_t count, slot;
    craftax_spawn_mobs2_count_and_empty(state, 2, level, &count, &slot);

    int32_t type = fighting_boss
        ? craftax_spawn_floor_mob_type(CF(boss_progress, state), CRAFTAX_MOB_RANGED)
        : craftax_spawn_floor_mob_type(level, CRAFTAX_MOB_RANGED);

    CraftaxThreefryKey prob_key = craftax_spawn_next_random_key(rng);
    CraftaxThreefryKey pos_key  = craftax_spawn_next_random_key(rng);

    MOB_TYPE(2, level, slot, state) = type;

    if (count >= CRAFTAX_MAX_RANGED_MOBS) return;
    if (craftax_threefry_uniform_f32(prob_key)
        >= craftax_spawn_floor_spawn_chance(level, 2) * (float)monster_spawn_coeff)
        return;

    int32_t row, col;
    if (!craftax_spawn_scan_ranged(state, level, type, fighting_boss, pos_key,
                                    &row, &col)) return;

    MOB_POS(2, level, slot, 0, state) = row;
    MOB_POS(2, level, slot, 1, state) = col;
    MOB_HP(2, level, slot, state)      =
        craftax_spawn_mob_type_health(type, CRAFTAX_MOB_RANGED);
    MOB_MASK(2, level, slot, state)        = true;
    CF_BITS(mob_bits, level, row, state) |= (1ULL << col);
}

// [cuda port] Spawn request compaction: the prologue below (RNG draws, type
// writes, try_* probability rolls) runs for every env every step, but the
// span scans only run for the rare envs whose rolls succeed (~1 in 40 under
// random actions). Inline, those scans make every warp pay the full
// divergent cost. When g_cf_spawn_queue is set (split-kernel path), envs
// that would scan are appended to a compacted worklist instead and
// k_spawn_tail processes them densely between k_step and the reset kernel.
// Bit-exactness: the scans read only spawn_{all,grave,water}_bits, mob_bits
// and player_position; no phase after spawn_mobs in the step mutates any of
// those (update_plants only toggles PLANT/RIPE_PLANT, which are in no spawn
// mask, and its set_map_block rewrites recompute identical bit values), so
// running the tail after k_step reads exactly the same inputs. Per-env
// writes are independent, so worklist order does not matter. Envs that
// finish this step still get their spawn writes before the reset kernel
// wipes them -- same net state as the inline order.
typedef struct CraftaxSpawnRec {
    int32_t env;
    uint32_t pkey0, pkey1;   // passive_pos_key
    uint32_t mkey0, mkey1;   // melee_pos_key
    uint32_t rkey0, rkey1;   // ranged_pos_key
    uint8_t level;
    uint8_t flags;           // bit0 try_passive, bit1 try_melee,
                             // bit2 try_ranged, bit3 fighting_boss
    uint8_t pslot, mslot, rslot;
    uint8_t ptype, mtype, rtype;
} CraftaxSpawnRec;

__device__ CraftaxSpawnRec* g_cf_spawn_queue = NULL;  // NULL => inline tail
__device__ int g_cf_spawn_count = 0;

static __device__ inline void craftax_spawn_mobs_tail(
    CraftaxState* state, int32_t level, bool fighting_boss,
    bool try_passive, bool try_melee, bool try_ranged,
    int32_t passive_slot, int32_t melee_slot, int32_t ranged_slot,
    int32_t passive_type, int32_t melee_type, int32_t ranged_type,
    CraftaxThreefryKey passive_pos_key, CraftaxThreefryKey melee_pos_key,
    CraftaxThreefryKey ranged_pos_key
) {
    // [cuda port] The list-building multi-spawn path (scan_all into 8KB of
    // per-thread CraftaxSpawnLists + pick_excluding) is replaced by three
    // sequential span scans. Each successful spawn sets its mob_bits bit
    // before the next scan runs, which removes exactly the coordinates that
    // pick_excluding excluded; iteration order and the k-th pick arithmetic
    // are unchanged, so the chosen cells are bit-identical.
    craftax_spawn_init_offsets_once();

    bool passive_spawned = false;
    int32_t passive_row = 0;
    int32_t passive_col = 0;
    if (try_passive && craftax_spawn_scan_passive(
            state, level, passive_pos_key, &passive_row, &passive_col
        )) {
        MOB_POS(1, level, passive_slot, 0, state) = passive_row;
        MOB_POS(1, level, passive_slot, 1, state) = passive_col;
        MOB_HP(1, level, passive_slot, state) =
            craftax_spawn_mob_type_health(passive_type, CRAFTAX_MOB_PASSIVE);
        MOB_MASK(1, level, passive_slot, state) = true;
        CF_BITS(mob_bits, level, passive_row, state) |= (1ULL << passive_col);
        passive_spawned = true;
    }

    bool melee_spawned = false;
    int32_t melee_row = 0;
    int32_t melee_col = 0;
    (void)passive_spawned;
    if (try_melee && craftax_spawn_scan_melee(
            state, level, fighting_boss, melee_pos_key,
            &melee_row, &melee_col
        )) {
        MOB_POS(0, level, melee_slot, 0, state) = melee_row;
        MOB_POS(0, level, melee_slot, 1, state) = melee_col;
        MOB_HP(0, level, melee_slot, state) =
            craftax_spawn_mob_type_health(melee_type, CRAFTAX_MOB_MELEE);
        MOB_MASK(0, level, melee_slot, state) = true;
        CF_BITS(mob_bits, level, melee_row, state) |= (1ULL << melee_col);
        melee_spawned = true;
    }

    int32_t ranged_row = 0;
    int32_t ranged_col = 0;
    (void)melee_spawned;
    if (try_ranged && craftax_spawn_scan_ranged(
            state, level, ranged_type, fighting_boss, ranged_pos_key,
            &ranged_row, &ranged_col
        )) {
        MOB_POS(2, level, ranged_slot, 0, state) = ranged_row;
        MOB_POS(2, level, ranged_slot, 1, state) = ranged_col;
        MOB_HP(2, level, ranged_slot, state) =
            craftax_spawn_mob_type_health(ranged_type, CRAFTAX_MOB_RANGED);
        MOB_MASK(2, level, ranged_slot, state) = true;
        CF_BITS(mob_bits, level, ranged_row, state) |= (1ULL << ranged_col);
    }
}

// Warp-cooperative version of craftax_spawn_scan_spans: lanes stride the
// span list. All selection arithmetic is integer (popcounts, prefix sums)
// except the single uniform draw, which is computed exactly as the scalar
// version from the same n and key, so the chosen (row, col) is bit-identical.
static __device__ inline bool craftax_spawn_scan_spans_warp(
    const CraftaxState* state, int32_t level,
    const CraftaxSpawnOffsetSpan* spans, int32_t span_count,
    uint64_t terrain_mask, CraftaxThreefryKey pos_key,
    int32_t* out_row, int32_t* out_col, unsigned lane
) {
    const unsigned FULL = 0xffffffffu;
    int32_t pr = CF2(player_position, 0, state);
    int32_t pc = CF2(player_position, 1, state);

    int32_t n_lane = 0;
    for (int32_t i = (int32_t)lane; i < span_count; i += 32) {
        int32_t row = pr + spans[i].dr;
        if ((uint32_t)row >= CRAFTAX_MAP_SIZE) continue;
        int32_t col0 = pc + spans[i].dc0;
        int32_t col1 = pc + spans[i].dc1;
        if (col0 < 0) col0 = 0;
        if (col1 >= CRAFTAX_MAP_SIZE) col1 = CRAFTAX_MAP_SIZE - 1;
        if (col0 > col1) continue;
        uint64_t candidates =
            craftax_spawn_row_bits_for_mask(state, level, row, terrain_mask)
            & ~CF_BITS(mob_bits, level, row, state)
            & craftax_spawn_col_mask(col0, col1);
        n_lane += __popcll(candidates);
    }
    int32_t n = n_lane;
    for (int off = 16; off > 0; off >>= 1)
        n += __shfl_xor_sync(FULL, n, off);
    if (n == 0) return false;

    float draw = (float)n * (1.0f - craftax_threefry_uniform_f32(pos_key));
    int32_t k = (int32_t)ceilf(draw) - 1;
    if (k < 0) k = 0;
    if (k > n - 1) k = n - 1;

    int32_t seen = 0;
    for (int32_t base = 0; base < span_count; base += 32) {
        int32_t i = base + (int32_t)lane;
        uint64_t candidates = 0;
        int32_t row = 0;
        if (i < span_count) {
            row = pr + spans[i].dr;
            int32_t col0 = pc + spans[i].dc0;
            int32_t col1 = pc + spans[i].dc1;
            if (col0 < 0) col0 = 0;
            if (col1 >= CRAFTAX_MAP_SIZE) col1 = CRAFTAX_MAP_SIZE - 1;
            if ((uint32_t)row < CRAFTAX_MAP_SIZE && col0 <= col1) {
                candidates =
                    craftax_spawn_row_bits_for_mask(state, level, row,
                                                    terrain_mask)
                    & ~CF_BITS(mob_bits, level, row, state)
                    & craftax_spawn_col_mask(col0, col1);
            }
        }
        int32_t c = __popcll(candidates);
        int32_t incl = c;
        for (int off = 1; off < 32; off <<= 1) {
            int32_t v = __shfl_up_sync(FULL, incl, off);
            if ((int)lane >= off) incl += v;
        }
        int32_t chunk_total = __shfl_sync(FULL, incl, 31);
        if (seen + chunk_total > k) {
            int32_t excl = incl - c;
            bool mine = (seen + excl <= k) && (k < seen + incl);
            unsigned m = __ballot_sync(FULL, mine);
            int src = __ffs((int)m) - 1;
            int32_t r_row = 0, r_col = 0;
            if (mine) {
                int32_t need = k - seen - excl;
                while (need-- > 0) candidates &= candidates - 1;
                r_row = row;
                r_col = (int32_t)__builtin_ctzll(candidates);
            }
            *out_row = __shfl_sync(FULL, r_row, src);
            *out_col = __shfl_sync(FULL, r_col, src);
            return true;
        }
        seen += chunk_total;
    }
    return false;
}

// Warp-cooperative spawn tail: the three class scans stay sequential (each
// successful spawn's mob_bits update must be visible to the next scan), but
// each scan is lane-parallel. Writes happen on lane 0; __syncwarp() orders
// them before the next scan reads.
static __device__ inline void craftax_spawn_mobs_tail_warp(
    CraftaxState* state, int32_t level, bool fighting_boss,
    bool try_passive, bool try_melee, bool try_ranged,
    int32_t passive_slot, int32_t melee_slot, int32_t ranged_slot,
    int32_t passive_type, int32_t melee_type, int32_t ranged_type,
    CraftaxThreefryKey passive_pos_key, CraftaxThreefryKey melee_pos_key,
    CraftaxThreefryKey ranged_pos_key, unsigned lane
) {
    int32_t row = 0, col = 0;
    if (try_passive && craftax_spawn_scan_spans_warp(
            state, level, craftax_spawn_passive_spans,
            craftax_spawn_passive_span_count,
            CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK, passive_pos_key,
            &row, &col, lane)) {
        if (lane == 0) {
            MOB_POS(1, level, passive_slot, 0, state) = row;
            MOB_POS(1, level, passive_slot, 1, state) = col;
            MOB_HP(1, level, passive_slot, state) =
                craftax_spawn_mob_type_health(passive_type,
                                              CRAFTAX_MOB_PASSIVE);
            MOB_MASK(1, level, passive_slot, state) = true;
            CF_BITS(mob_bits, level, row, state) |= (1ULL << col);
        }
    }
    __syncwarp();

    const CraftaxSpawnOffsetSpan* hostile_spans = fighting_boss
        ? craftax_spawn_boss_spans
        : craftax_spawn_hostile_spans;
    int32_t hostile_span_count = fighting_boss
        ? craftax_spawn_boss_span_count
        : craftax_spawn_hostile_span_count;

    uint64_t melee_terrain_mask = fighting_boss
        ? CRAFTAX_SPAWN_GRAVE_BLOCK_MASK
        : CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK;
    if (try_melee && craftax_spawn_scan_spans_warp(
            state, level, hostile_spans, hostile_span_count,
            melee_terrain_mask, melee_pos_key, &row, &col, lane)) {
        if (lane == 0) {
            MOB_POS(0, level, melee_slot, 0, state) = row;
            MOB_POS(0, level, melee_slot, 1, state) = col;
            MOB_HP(0, level, melee_slot, state) =
                craftax_spawn_mob_type_health(melee_type, CRAFTAX_MOB_MELEE);
            MOB_MASK(0, level, melee_slot, state) = true;
            CF_BITS(mob_bits, level, row, state) |= (1ULL << col);
        }
    }
    __syncwarp();

    uint64_t ranged_terrain_mask;
    if (fighting_boss) {
        ranged_terrain_mask = CRAFTAX_SPAWN_GRAVE_BLOCK_MASK;
    } else if (ranged_type == 5) {
        ranged_terrain_mask = CRAFTAX_SPAWN_WATER_BLOCK_MASK;
    } else {
        ranged_terrain_mask = CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK;
    }
    if (try_ranged && craftax_spawn_scan_spans_warp(
            state, level, hostile_spans, hostile_span_count,
            ranged_terrain_mask, ranged_pos_key, &row, &col, lane)) {
        if (lane == 0) {
            MOB_POS(2, level, ranged_slot, 0, state) = row;
            MOB_POS(2, level, ranged_slot, 1, state) = col;
            MOB_HP(2, level, ranged_slot, state) =
                craftax_spawn_mob_type_health(ranged_type, CRAFTAX_MOB_RANGED);
            MOB_MASK(2, level, ranged_slot, state) = true;
            CF_BITS(mob_bits, level, row, state) |= (1ULL << col);
        }
    }
    __syncwarp();
}

static __device__ inline void craftax_spawn_mobs_native(
    CraftaxState* state, CraftaxThreefryKey rng
) {
    int32_t level = craftax_step_jax_index(
        CF(player_level, state), CRAFTAX_NUM_LEVELS
    );
    bool fighting_boss = craftax_step_is_fighting_boss(state);
    int32_t monster_spawn_coeff =
        1
        + (int32_t)(CF2(monsters_killed, level, state)
                    < CRAFTAX_MONSTERS_KILLED_TO_CLEAR_LEVEL) * 2;

    bool boss_spawn_wave =
        fighting_boss && CF(boss_timesteps_to_spawn_this_round, state) >= 1;
    if (fighting_boss) {
        monster_spawn_coeff *= (int32_t)boss_spawn_wave * 1000;
    }

    int32_t passive_count, passive_slot;
    craftax_spawn_mobs3_count_and_empty(
        state, 1, level, &passive_count, &passive_slot
    );
    CraftaxThreefryKey passive_prob_key = craftax_spawn_next_random_key(&rng);
    CraftaxThreefryKey passive_pos_key = craftax_spawn_next_random_key(&rng);
    int32_t passive_type = craftax_spawn_floor_mob_type(
        level, CRAFTAX_MOB_PASSIVE
    );
    MOB_TYPE(1, level, passive_slot, state) = passive_type;

    int32_t melee_count, melee_slot;
    craftax_spawn_mobs3_count_and_empty(
        state, 0, level, &melee_count, &melee_slot
    );
    int32_t melee_type = fighting_boss
        ? craftax_spawn_floor_mob_type(CF(boss_progress, state), CRAFTAX_MOB_MELEE)
        : craftax_spawn_floor_mob_type(level, CRAFTAX_MOB_MELEE);
    CraftaxThreefryKey melee_prob_key = craftax_spawn_next_random_key(&rng);
    float night_coeff = 1.0f - CF(light_level, state);
    float melee_spawn_chance = craftax_spawn_floor_spawn_chance(level, 1)
        + craftax_spawn_floor_spawn_chance(level, 3) * night_coeff * night_coeff;
    CraftaxThreefryKey melee_pos_key = craftax_spawn_next_random_key(&rng);
    MOB_TYPE(0, level, melee_slot, state) = melee_type;

    int32_t ranged_count, ranged_slot;
    craftax_spawn_mobs2_count_and_empty(
        state, 2, level, &ranged_count, &ranged_slot
    );
    int32_t ranged_type = fighting_boss
        ? craftax_spawn_floor_mob_type(CF(boss_progress, state), CRAFTAX_MOB_RANGED)
        : craftax_spawn_floor_mob_type(level, CRAFTAX_MOB_RANGED);
    CraftaxThreefryKey ranged_prob_key = craftax_spawn_next_random_key(&rng);
    CraftaxThreefryKey ranged_pos_key = craftax_spawn_next_random_key(&rng);
    MOB_TYPE(2, level, ranged_slot, state) = ranged_type;

    bool try_passive = !fighting_boss
        && passive_count < CRAFTAX_MAX_PASSIVE_MOBS
        && craftax_threefry_uniform_f32(passive_prob_key)
            < craftax_spawn_floor_spawn_chance(level, 0);
    bool try_melee = melee_count < CRAFTAX_MAX_MELEE_MOBS
        && craftax_threefry_uniform_f32(melee_prob_key)
            < melee_spawn_chance * (float)monster_spawn_coeff;
    bool try_ranged = ranged_count < CRAFTAX_MAX_RANGED_MOBS
        && craftax_threefry_uniform_f32(ranged_prob_key)
            < craftax_spawn_floor_spawn_chance(level, 2)
                * (float)monster_spawn_coeff;

    if (!try_passive && !try_melee && !try_ranged) return;

    if (g_cf_spawn_queue != NULL) {
        int slot = atomicAdd(&g_cf_spawn_count, 1);
        CraftaxSpawnRec* r = &g_cf_spawn_queue[slot];
        r->env = cf_slot(state);
        r->pkey0 = passive_pos_key.word[0];
        r->pkey1 = passive_pos_key.word[1];
        r->mkey0 = melee_pos_key.word[0];
        r->mkey1 = melee_pos_key.word[1];
        r->rkey0 = ranged_pos_key.word[0];
        r->rkey1 = ranged_pos_key.word[1];
        r->level = (uint8_t)level;
        r->flags = (uint8_t)((try_passive ? 1 : 0) | (try_melee ? 2 : 0)
                             | (try_ranged ? 4 : 0) | (fighting_boss ? 8 : 0));
        r->pslot = (uint8_t)passive_slot;
        r->mslot = (uint8_t)melee_slot;
        r->rslot = (uint8_t)ranged_slot;
        r->ptype = (uint8_t)passive_type;
        r->mtype = (uint8_t)melee_type;
        r->rtype = (uint8_t)ranged_type;
        return;
    }

    craftax_spawn_mobs_tail(
        state, level, fighting_boss, try_passive, try_melee, try_ranged,
        passive_slot, melee_slot, ranged_slot,
        passive_type, melee_type, ranged_type,
        passive_pos_key, melee_pos_key, ranged_pos_key);
}

// ============================================================
// ===== CUDA host harness (replaces the pthread/OpenMP harness) =====
#define CU_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, \
                __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

// token-identical copy of the C harness action RNG, as device code
static __device__ inline uint32_t cf_xorshift32(uint32_t* s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x ? x : 0xdeadbeefu;
    return *s;
}

__global__ void k_global_init(void) {
    craftax_wg_init_cell_templates();
    craftax_spawn_init_offsets_once();
}

// Mirrors cf_vec_init in craftax_full.c: shared arena, env->rng = index,
// env->seed = seed + index, pools disabled. [env_lo, env_hi) per launch so
// worldgen-heavy init kernels stay short (display GPU watchdog).
__global__ void k_env_init(
    Craftax* envs, CraftaxState* states, CraftaxObs* observations,
    float* actions, float* rewards, float* terminals, uint32_t* action_rng,
    int env_lo, int env_hi, uint64_t seed
) {
    int i = env_lo + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= env_hi) return;
    Craftax* env = &envs[i];
    env->num_agents = 1;
    env->rng = (unsigned int)i;
    env->seed = seed + (uint64_t)i;
    env->arena = NULL;
    env->state = &states[i];
    env->packet_id = i / CRAFTAX_ARENA_PACKET_SIZE;
    env->lane_id = i % CRAFTAX_ARENA_PACKET_SIZE;
    env->owns_state_storage = false;
    env->observations = &observations[(size_t)i * CRAFTAX_OBS_SIZE];
    env->actions = &actions[i];
    env->rewards = &rewards[i];
    env->terminals = &terminals[i];
    c_init(env);
    craftax_encode_native_observation(env->state, env->observations);
    // Deterministic per-env action stream, identical to the C harness.
    action_rng[i] = (uint32_t)(seed ^ ((uint64_t)i * 2654435761ULL)) | 1u;
}

// Done-list compaction: k_step runs gameplay only and appends finished envs
// (with their reset keys) to a compacted list; k_reset_list then regenerates
// those worlds and writes their post-reset observations. The combination is
// bit-exact vs the inline-reset c_step: same keys, same values, same obs.
typedef struct CraftaxResetRec {
    int32_t env;
    uint32_t key0;
    uint32_t key1;
} CraftaxResetRec;

__device__ int g_reset_count = 0;

// Scalar-tail encode (inventory/intrinsics block after the packed map).
// Runs thread-per-env in k_step (state is register/L1-hot there) and in the
// reset kernels for just-reset envs, replacing the serial lane-0 tail in
// k_encode. Values and destination bytes are identical.
static __device__ inline void cf_encode_tail(
    const CraftaxState* state, CraftaxObs* obs
) {
    const CraftaxWorldState* ws = (const CraftaxWorldState*)(const void*)state;
#ifdef CRAFTAX_COMPACT_OBS
    float tail[CRAFTAX_WG_INVENTORY_OBS_SIZE];
    craftax_encode_scalar_observation_tail_at(ws, tail, 0);
    memcpy(obs + CRAFTAX_WG_PACKED_MAP_OBS_SIZE, tail, sizeof(tail));
#else
    craftax_encode_scalar_observation_tail_at(
        ws, obs, CRAFTAX_WG_PACKED_MAP_OBS_SIZE);
#endif
}

// Optional occupancy forcing for the 64-thread step kernels (k_step,
// k_step_run, k_step_policy, k_step_policy_train). N min blocks/SM caps
// registers at floor(65536/(64*N)) and ptxas spills the excess to local;
// 0 (default) keeps the unconstrained allocation. Pure scheduling: results
// are bit-identical either way, only regs/occupancy move.
//
// MEASURED on an idle 3090 @65536 envs (fused run mode SPS / bench SPS),
// 2026-07 register-pressure pass -- occupancy is NOT the lever here:
//   N=0: 254 regs, 0 spill, 4 blocks/SM (16.7% occ)  29.0M run / 39.3M bench
//   N=5: 168 regs, 0 spill, 6 blocks/SM (25%)        23.3M     / 39.5M
//   N=8: 128 regs, 0 spill, 8 blocks/SM (33%)        19.3M     / 38.0M
//   N=10: 96 regs, ~150B spill, 10 blocks (41.7%)    16.7M     / --
//   N=12: 80 regs, ~250B spill, 12 blocks (50%)      14.8M     / --
// More resident envs shrink the per-env L1 share and the hit rate collapses
// (78.8% -> 67.3% at N=5); the kernel is L1TEX-latency-bound, and 256
// threads/SM is the measured cache-vs-latency optimum. Occupancy DOWN loses
// too (192 thr = 22.2M, 128 thr = 19.2M via smem padding), as do smem-staged
// weights (73KB smem carveout eats L1: 20.7M) and __constant__ weights
// (17.7KB cycling thrashes IMC: 26.7M). Keep N=0.
#ifndef CRAFTAX_STEP_MIN_BLOCKS
#define CRAFTAX_STEP_MIN_BLOCKS 0
#endif
#if CRAFTAX_STEP_MIN_BLOCKS > 0
#define CRAFTAX_STEP_LB __launch_bounds__(64, CRAFTAX_STEP_MIN_BLOCKS)
#else
#define CRAFTAX_STEP_LB
#endif

__global__ void CRAFTAX_STEP_LB k_step(
    Craftax* envs, uint32_t* action_rng, int num_envs, CraftaxResetRec* resets
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_envs) return;
    Craftax* env = &envs[i];
    env->actions[0] =
        (float)(cf_xorshift32(&action_rng[i]) % CRAFTAX_NUM_ACTIONS);
    CraftaxThreefryKey reset_key;
    CRAFTAX_PROFILE_START();
    bool done = c_step_gameplay_core(env, &reset_key);
    if (done) {
        int slot = atomicAdd(&g_reset_count, 1);
        resets[slot].env = i;
        resets[slot].key0 = reset_key.word[0];
        resets[slot].key1 = reset_key.word[1];
    } else {
        cf_encode_tail(env->state, env->observations);
    }
}

// Warp-transposed observation encode: one warp per env, lane L writing obs
// channels L, L+32, ... so stores coalesce across the warp (the per-thread
// encode issued ~1150 scalar stores per env into non-contiguous sectors and
// dominated k_step). Values are computed with the exact expressions of
// craftax_encode_reset_observation / craftax_encode_compact_observation, so
// the buffer is bit-identical; it runs after the reset kernel and therefore
// also covers post-reset observations.
#define CRAFTAX_ENC_WARPS_PER_BLOCK 4

// Warp-cooperative packed-map encode for one env (lanes stride the packed
// map channels; the scalar tail is written by k_step / the reset kernels).
static __device__ inline void cf_encode_map_warp(
    const Craftax* env, unsigned lane
) {
    const CraftaxWorldState* state =
        (const CraftaxWorldState*)(const void*)env->state;
    CraftaxObs* obs = env->observations;

    const int level = CF(player_level, state);
    const int mob_level =
        craftax_wg_jax_index(CF(player_level, state), CRAFTAX_WG_NUM_LEVELS);
    const int top = CF2(player_position, 0, state) - CRAFTAX_WG_OBS_ROWS / 2;
    const int left = CF2(player_position, 1, state) - CRAFTAX_WG_OBS_COLS / 2;

    // One lane per cell: map/item/light loaded once per cell (the previous
    // per-(cell,channel) loop reloaded light_map for each of the 8 channels)
    // and the 8 channel values are written as one contiguous 8-element run,
    // so a full warp stores a coalesced 32 * 8 * sizeof(CraftaxObs) block.
    for (int cell = (int)lane;
         cell < CRAFTAX_WG_OBS_ROWS * CRAFTAX_WG_OBS_COLS; cell += 32) {
        int row = cell / CRAFTAX_WG_OBS_COLS;
        int col = cell % CRAFTAX_WG_OBS_COLS;
        int world_row = top + row;
        int world_col = left + col;
        bool in_bounds = world_row >= 0 && world_row < CRAFTAX_WG_MAP_SIZE
            && world_col >= 0 && world_col < CRAFTAX_WG_MAP_SIZE;
        bool visible =
            in_bounds && state->light_map[level][world_row][world_col] > 12;

        CraftaxObs* cell_obs =
            obs + (size_t)cell * CRAFTAX_WG_PACKED_CHANNELS_PER_CELL;
        CraftaxObs v0 = 0;
        CraftaxObs v1 = 0;
        if (visible) {
            v0 = (CraftaxObs)state->map[level][world_row][world_col];
#ifdef CRAFTAX_COMPACT_OBS
            v1 = (uint8_t)(state->item_map[level][world_row][world_col] + 1);
#else
            v1 = (float)state->item_map[level][world_row][world_col] + 1.0f;
#endif
        }
        cell_obs[0] = v0;
        cell_obs[1] = v1;
        cell_obs[2] = visible ? (CraftaxObs)1 : (CraftaxObs)0;
        // Mob channels (ch >= 3): zero here, scattered below. The per-cell
        // scan visited 5 classes x 3 slots for each of 130 cells; scattering
        // the <= 14 mobs directly writes the same values in the same
        // last-slot-wins order per class (classes write disjoint channels).
#pragma unroll
        for (int c = 0; c < CRAFTAX_WG_NUM_MOB_CLASSES; c++) {
            cell_obs[3 + c] = 0;
        }
    }
    __syncwarp();
    if (lane < 5) {
        const int c = (int)lane;
        const int nslots = c == 2 ? 2 : 3;
        for (int i = 0; i < nslots; i++) {
            int type_id = MOB_TYPE(c, mob_level, i, state);
            if (type_id < 0 || type_id >= CRAFTAX_WG_NUM_MOB_TYPES
                || !MOB_MASK(c, mob_level, i, state)) {
                continue;
            }
            int mob_row = MOB_POS(c, mob_level, i, 0, state);
            int mob_col = MOB_POS(c, mob_level, i, 1, state);
            int local_row = mob_row - top;
            int local_col = mob_col - left;
            if (local_row < 0 || local_row >= CRAFTAX_WG_OBS_ROWS
                || local_col < 0 || local_col >= CRAFTAX_WG_OBS_COLS) {
                continue;
            }
            bool in_bounds = mob_row >= 0 && mob_row < CRAFTAX_WG_MAP_SIZE
                && mob_col >= 0 && mob_col < CRAFTAX_WG_MAP_SIZE;
            if (!in_bounds
                || state->light_map[mob_level][mob_row][mob_col] <= 12) {
                continue;
            }
            int cell = local_row * CRAFTAX_WG_OBS_COLS + local_col;
            obs[cell * CRAFTAX_WG_PACKED_CHANNELS_PER_CELL + 3 + c] =
                (CraftaxObs)(type_id + 1);
        }
    }
}

__global__ void k_encode(Craftax* envs, int num_envs) {
    int env_idx = blockIdx.x * CRAFTAX_ENC_WARPS_PER_BLOCK + threadIdx.y;
    if (env_idx >= num_envs) return;
    cf_encode_map_warp(&envs[env_idx], threadIdx.x);
}

// Compacted spawn-scan tail (see CraftaxSpawnRec): one thread per env that
// rolled a spawn attempt this step. Runs after k_step and before the reset
// kernels, so done envs still receive their (about-to-be-wiped) spawn writes
// in the same order as the inline path.
__global__ void k_spawn_tail(void) {
    const unsigned lane = (unsigned)(threadIdx.x & 31);
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int total_warps = (int)((gridDim.x * blockDim.x) >> 5);
    const int count = g_cf_spawn_count;
    for (int idx = warp; idx < count; idx += total_warps) {
        const CraftaxSpawnRec r = g_cf_spawn_queue[idx];
        CraftaxState* state = &((CraftaxState*)g_cf_state_base)[r.env];
        CraftaxThreefryKey pkey = {{r.pkey0, r.pkey1}};
        CraftaxThreefryKey mkey = {{r.mkey0, r.mkey1}};
        CraftaxThreefryKey rkey = {{r.rkey0, r.rkey1}};
        craftax_spawn_mobs_tail_warp(
            state, (int32_t)r.level, (r.flags & 8) != 0,
            (r.flags & 1) != 0, (r.flags & 2) != 0, (r.flags & 4) != 0,
            (int32_t)r.pslot, (int32_t)r.mslot, (int32_t)r.rslot,
            (int32_t)r.ptype, (int32_t)r.mtype, (int32_t)r.rtype,
            pkey, mkey, rkey, lane);
    }
}

__global__ void k_reset_list(Craftax* envs, const CraftaxResetRec* resets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= g_reset_count) return;
    Craftax* env = &envs[resets[idx].env];
    CraftaxThreefryKey reset_key;
    reset_key.word[0] = resets[idx].key0;
    reset_key.word[1] = resets[idx].key1;
    craftax_reset_state_on_done(env->state, reset_key);
    cf_encode_tail(env->state, env->observations);
}

// Warp-cooperative reset: one warp (= one block) processes finished envs
// from the compacted reset list, noise scratch in shared memory instead of
// 46KB of per-thread stack. Only valid in lazy-floors mode (floor 0
// regenerated here, floors 1-8 on descent).
//
// Grid-strided over the worklist so the host can launch a fixed small grid
// (graph-friendly, ~num_SMs worth of blocks) instead of `num_envs` mostly
// empty blocks. Each env's worldgen is a pure function of its reset key, so
// processing order is irrelevant and bit-identical to 1-block-per-reset.
// Typical load is ~n/eplen resets/step (~30 @8192); the previous n-block
// launch paid scheduler overhead on thousands of early-exit warps every step.
__global__ void __launch_bounds__(32) k_reset_list_warp(
    Craftax* envs, const CraftaxResetRec* resets
) {
    __shared__ CraftaxWarpScratch s;
    unsigned lane = threadIdx.x;
    // Stride the compacted list; idle blocks exit on the first check.
    for (int idx = (int)blockIdx.x; idx < g_reset_count; idx += (int)gridDim.x) {
        Craftax* env = &envs[resets[idx].env];
        CraftaxThreefryKey reset_key;
        reset_key.word[0] = resets[idx].key0;
        reset_key.word[1] = resets[idx].key1;
        craftax_reset_state_on_done_warp<32>(env->state, reset_key, &s, lane);
        if (lane == 0) {
            cf_encode_tail(env->state, env->observations);
        }
        __syncwarp();
    }
}

// Block-cooperative reset: one NT-thread block per finished env instead of
// one warp. Bit-identical to k_reset_list_warp (see the cf_wg_* helpers:
// every per-cell draw is a pure function of (key, cell) and every reduction
// is exact), but ~NT/32 lanes attack the single-env worldgen latency. At
// small env counts (~4 resets/step @1024) reset duration IS one env's
// worldgen wall time on the rollout critical path, so this converts thread
// parallelism the GPU has to spare directly into latency.
template <int NT>
__global__ void __launch_bounds__(NT) k_reset_list_block(
    Craftax* envs, const CraftaxResetRec* resets
) {
    __shared__ CraftaxWarpScratch s;
    unsigned tid = threadIdx.x;
    for (int idx = (int)blockIdx.x; idx < g_reset_count; idx += (int)gridDim.x) {
        Craftax* env = &envs[resets[idx].env];
        CraftaxThreefryKey reset_key;
        reset_key.word[0] = resets[idx].key0;
        reset_key.word[1] = resets[idx].key1;
        craftax_reset_state_on_done_warp<NT>(env->state, reset_key, &s, tid);
        if (tid == 0) {
            cf_encode_tail(env->state, env->observations);
        }
        __syncthreads();
    }
}

// Reset-list launch: CRAFTAX_CU_RESET_BLOCK picks the per-env thread count
// (128 default; 32 = legacy warp kernel for A/B).
static int cu_reset_block_threads(void) {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("CRAFTAX_CU_RESET_BLOCK");
        v = e ? atoi(e) : 128;
        if (v != 32 && v != 128 && v != 256) v = 128;
    }
    return v;
}

static void cu_launch_reset_list_warp(Craftax* envs,
                                      const CraftaxResetRec* resets,
                                      int num_envs, cudaStream_t st) {
    // Cap the grid: worldgen is ~one block of work per finished env and the
    // list is almost always tiny; 512 blocks still covers a worst-case
    // all-die step via the grid-stride loop.
    int rgrid = num_envs > 512 ? 512 : num_envs;
    int nt = cu_reset_block_threads();
    if (nt == 256)
        k_reset_list_block<256><<<rgrid, 256, 0, st>>>(envs, resets);
    else if (nt == 128)
        k_reset_list_block<128><<<rgrid, 128, 0, st>>>(envs, resets);
    else
        k_reset_list_warp<<<rgrid, 32, 0, st>>>(envs, resets);
}

// ============================================================
// Megakernel: fused step + reset + encode, num_steps per launch.
// One thread per env for gameplay; the owning warp then handles resets
// (warp-cooperative worldgen, scratch in a per-warp global arena instead of
// shared memory so occupancy is not smem-capped) and the packed-map encode
// for its 32 envs. Envs never interact, so multi-step launches need no
// grid-wide synchronization and are bitwise identical to running the
// split kernels num_steps times.
// ============================================================
__global__ void k_mega(
    Craftax* envs, uint32_t* action_rng, int num_envs, int num_steps,
    CraftaxWarpScratch* warp_scratch
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = (unsigned)(threadIdx.x & 31);
    const int warp_id = i >> 5;
    const int warp_base = warp_id << 5;
    const bool active = i < num_envs;
    CraftaxWarpScratch* ws = &warp_scratch[warp_id];
    uint32_t arng = active ? action_rng[i] : 0u;

    for (int step = 0; step < num_steps; step++) {
        bool done = false;
        CraftaxThreefryKey reset_key;
        reset_key.word[0] = 0u;
        reset_key.word[1] = 0u;
        if (active) {
            Craftax* env = &envs[i];
            env->actions[0] =
                (float)(cf_xorshift32(&arng) % CRAFTAX_NUM_ACTIONS);
            done = c_step_gameplay_core(env, &reset_key);
            if (!done) cf_encode_tail(env->state, env->observations);
        }
        unsigned done_mask = __ballot_sync(0xffffffffu, done);
        while (done_mask != 0u) {
            int src = __ffs((int)done_mask) - 1;
            done_mask &= done_mask - 1u;
            CraftaxThreefryKey rk;
            rk.word[0] = __shfl_sync(0xffffffffu, reset_key.word[0], src);
            rk.word[1] = __shfl_sync(0xffffffffu, reset_key.word[1], src);
            Craftax* denv = &envs[warp_base + src];
            craftax_reset_state_on_done_warp<32>(denv->state, rk, ws, lane);
            if (lane == 0) cf_encode_tail(denv->state, denv->observations);
        }
        __syncwarp();
        for (int e = warp_base; e < warp_base + 32 && e < num_envs; e++) {
            cf_encode_map_warp(&envs[e], lane);
        }
        __syncwarp();
    }
    if (active) action_rng[i] = arng;
}


// Run-mode step kernels: batched gameplay step + reset-flag
// record, actions provided by the caller (the vec / the policy).

template <int EPW>
__global__ void CRAFTAX_STEP_LB k_step_run(Craftax* envs, int num_envs, CraftaxResetRec* resets,
                                           uint8_t* __restrict__ reset_flags) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = t & 31;
    int i = (t >> 5) * EPW + lane;
    if (lane >= EPW || i >= num_envs) return;
    Craftax* env = &envs[i];
    CraftaxThreefryKey reset_key;
    bool done = c_step_gameplay_core(env, &reset_key);
    if (reset_flags) reset_flags[i] = done ? 1 : 0;
    if (done) {
        int slot = atomicAdd(&g_reset_count, 1);
        resets[slot].env = i;
        resets[slot].key0 = reset_key.word[0];
        resets[slot].key1 = reset_key.word[1];
    }
}

// EPW schedule: keep total step warps around the count that fills the SMs
// without spilling into extra register-file waves (the kernel runs at ~250
// regs/thread, so ~8 warps/SM is the ceiling). 3090-measured rollout SPS
// winners (graph=1, 3000 iters, CF_GEMM_BF16=0): 1024->EPW2 (+8.9% vs the
// dense mapping), 2048->EPW4 (+5.3%), 4096->EPW8/16 (+1.2%), 8192->EPW16
// (+0.7%); EPW1 regresses everywhere (2x the warp waves). Above the mid
// band the dense mapping is already even with the best retile, so keep it.
// CF_STEP_EPW=<1|2|4|8|16|32> forces a value (0/unset = this schedule);
// larger-SM parts (sm_120: 188 SMs) may prefer one notch lower per n.
static int cf_step_epw_for(int n) {
    static int forced = -1;
    if (forced < 0) {
        const char* e = getenv("CF_STEP_EPW");
        forced = e ? atoi(e) : 0;
    }
    if (forced > 0) return forced;
    if (n <= 1536) return 2;
    if (n <= 3072) return 4;
    if (n <= 6144) return 8;
    if (n <= 12288) return 16;
    return 32;
}

// Launch tiling for k_step_run: ceil(n/EPW) warps carry the envs. At EPW=32
// this reproduces the historical dense grids exactly
// (ceil(ceil(n/32)*32/64) == ceil(n/64)); the 32-wide mid-band retile is
// the measured-and-kept configuration from the register-pressure pass.
static void cu_launch_step_run(Craftax* envs, int n, CraftaxResetRec* resets,
                               uint8_t* rf, cudaStream_t st) {
    int epw = cf_step_epw_for(n);
    long threads = (long)((n + epw - 1) / epw) * 32;
    dim3 grid, block;
    if (n >= 4096 && n <= 12288) {
        grid = dim3((unsigned)((threads + 31) / 32));
        block = dim3(32);
    } else {
        grid = dim3((unsigned)((threads + 63) / 64));
        block = dim3(64);
    }
    switch (epw) {
        case 1:  k_step_run<1><<<grid, block, 0, st>>>(envs, n, resets, rf); break;
        case 2:  k_step_run<2><<<grid, block, 0, st>>>(envs, n, resets, rf); break;
        case 4:  k_step_run<4><<<grid, block, 0, st>>>(envs, n, resets, rf); break;
        case 8:  k_step_run<8><<<grid, block, 0, st>>>(envs, n, resets, rf); break;
        case 16: k_step_run<16><<<grid, block, 0, st>>>(envs, n, resets, rf); break;
        default: k_step_run<32><<<grid, block, 0, st>>>(envs, n, resets, rf); break;
    }
}

// Scalar-tail encode as a standalone kernel (runverify materializes the
// full obs; the batched rollout path never writes observation bytes).
__global__ void k_encode_tail(Craftax* envs, int num_envs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_envs) return;
    cf_encode_tail(envs[i].state, envs[i].observations);
}

typedef struct {
    int num_envs;
    Craftax* d_envs;
    CraftaxState* d_states;
    CraftaxObs* d_obs;
    float* d_actions;
    float* d_rewards;
    float* d_terminals;
    uint32_t* d_action_rng;
    CraftaxResetRec* d_resets;
    uint8_t* d_reset_flags;  // per-env "reset pending" (k_step_run, overlap)
    int* d_reset_count;  // symbol address of g_reset_count
    CraftaxSpawnRec* d_spawn_queue;
    int* d_spawn_count;  // symbol address of g_cf_spawn_count
    CraftaxWorldgenScratch* d_wg_scratch;
    void* d_soa[64];
    int num_soa;
    int lazy;
    int mega;
    float curr_p;
    CraftaxWarpScratch* d_warp_scratch;
    CraftaxObs* h_obs;
    float* h_rewards;
    float* h_terminals;
} CuVec;

static void cu_fill_light_table(void) {
    // Same formula as craftax_calculate_light_level_native, computed with
    // HOST libm cosf (glibc, as in the reference binary) and the gcc
    // -ffast-math expansion powf(x, 3.0f) -> x*x*x.
    static float table[CRAFTAX_DEFAULT_MAX_TIMESTEPS_TABLE];
    for (int t = 0; t < CRAFTAX_DEFAULT_MAX_TIMESTEPS_TABLE; t++) {
        // gcc -ffast-math turns t / 300.0f into t * (1.0f / 300.0f)
        float progress =
            fmodf((float)t * (1.0f / (float)CRAFTAX_DAY_LENGTH), 1.0f) + 0.3f;
        float c = cosf(CRAFTAX_WG_PI * progress);
        float a = fabsf(c);
        // gcc contracts 1.0f - a*a*a into vfnmadd (verified bit-exact
        // against the reference binary for all dumped timesteps)
        table[t] = fmaf(-(a * a), a, 1.0f);
    }
    CU_CHECK(cudaMemcpyToSymbol(g_craftax_light_table, table, sizeof(table)));
}

static void cu_vec_init(CuVec* v, int num_envs, uint64_t seed) {
    v->num_envs = num_envs;
    // Worldgen scratch lives in a per-thread global arena (g_craftax_wg_scratch),
    // so the residual stack is small (~13KB gameplay/encode locals per ptxas -v).
    size_t stack_bytes = 16 << 10;
    const char* stack_env = getenv("CRAFTAX_CU_STACK");
    if (stack_env != NULL) stack_bytes = (size_t)atol(stack_env);
    CU_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, stack_bytes));

    CU_CHECK(cudaMalloc(&v->d_envs, (size_t)num_envs * sizeof(Craftax)));
    CU_CHECK(cudaMalloc(&v->d_states, (size_t)num_envs * sizeof(CraftaxState)));
    CU_CHECK(cudaMalloc(&v->d_obs,
                        (size_t)num_envs * CRAFTAX_OBS_SIZE * sizeof(CraftaxObs)));
    CU_CHECK(cudaMalloc(&v->d_actions, (size_t)num_envs * sizeof(float)));
    CU_CHECK(cudaMalloc(&v->d_rewards, (size_t)num_envs * sizeof(float)));
    CU_CHECK(cudaMalloc(&v->d_terminals, (size_t)num_envs * sizeof(float)));
    CU_CHECK(cudaMalloc(&v->d_action_rng, (size_t)num_envs * sizeof(uint32_t)));
    CU_CHECK(cudaMalloc(&v->d_resets,
                        (size_t)num_envs * sizeof(CraftaxResetRec)));
    CU_CHECK(cudaMalloc(&v->d_reset_flags, (size_t)num_envs));
    CU_CHECK(cudaMemset(v->d_reset_flags, 0, (size_t)num_envs));
    CU_CHECK(cudaGetSymbolAddress((void**)&v->d_reset_count, g_reset_count));
    CU_CHECK(cudaMemset(v->d_reset_count, 0, sizeof(int)));
    CU_CHECK(cudaMalloc(&v->d_spawn_queue,
                        (size_t)num_envs * sizeof(CraftaxSpawnRec)));
    CU_CHECK(cudaGetSymbolAddress((void**)&v->d_spawn_count, g_cf_spawn_count));
    CU_CHECK(cudaMalloc(&v->d_wg_scratch,
                        (size_t)num_envs * sizeof(CraftaxWorldgenScratch)));
    CU_CHECK(cudaMemcpyToSymbol(g_craftax_wg_scratch, &v->d_wg_scratch,
                                sizeof(v->d_wg_scratch)));
    {
        char* base = (char*)v->d_states;
        CU_CHECK(cudaMemcpyToSymbol(g_cf_state_base, &base, sizeof(base)));
        CU_CHECK(cudaMemcpyToSymbol(g_cf_n, &num_envs, sizeof(num_envs)));
        v->num_soa = 0;
#define CF_SOA_ALLOC(f, t, k) { \
        t* p = NULL; \
        size_t bytes = sizeof(t) * (size_t)(k) * (size_t)num_envs; \
        CU_CHECK(cudaMalloc(&p, bytes)); \
        CU_CHECK(cudaMemset(p, 0, bytes)); \
        CU_CHECK(cudaMemcpyToSymbol(g_cf_##f, &p, sizeof(p))); \
        v->d_soa[v->num_soa++] = p; }
        CF_SOA_FIELDS(CF_SOA_ALLOC)
#undef CF_SOA_ALLOC
    }
    // calloc semantics of the C harness
    CU_CHECK(cudaMemset(v->d_envs, 0, (size_t)num_envs * sizeof(Craftax)));
    CU_CHECK(cudaMemset(v->d_states, 0, (size_t)num_envs * sizeof(CraftaxState)));
    CU_CHECK(cudaMemset(v->d_obs, 0,
                        (size_t)num_envs * CRAFTAX_OBS_SIZE * sizeof(CraftaxObs)));
    CU_CHECK(cudaMemset(v->d_actions, 0, (size_t)num_envs * sizeof(float)));
    CU_CHECK(cudaMemset(v->d_rewards, 0, (size_t)num_envs * sizeof(float)));
    CU_CHECK(cudaMemset(v->d_terminals, 0, (size_t)num_envs * sizeof(float)));

    v->h_obs = (CraftaxObs*)malloc(
        (size_t)num_envs * CRAFTAX_OBS_SIZE * sizeof(CraftaxObs));
    v->h_rewards = (float*)malloc((size_t)num_envs * sizeof(float));
    v->h_terminals = (float*)malloc((size_t)num_envs * sizeof(float));

    cu_fill_light_table();
    // Lazy floor generation (floor 0 on reset, floors 1-8 on first descent)
    // produces bit-identical maps with ~9x less reset work; verified against
    // both hash anchors. CRAFTAX_CU_LAZY=0 restores eager worldgen.
    const char* lazy_env = getenv("CRAFTAX_CU_LAZY");
    int lazy = lazy_env == NULL ? 1 : atoi(lazy_env);
    CU_CHECK(cudaMemcpyToSymbol(g_craftax_lazy_floors, &lazy, sizeof(int)));
    v->lazy = lazy;
    // Descend-shaped reward probe: CRAFTAX_CU_DESCEND_BONUS=<mult> scales the
    // enter-floor achievement rewards. Default 1.0 keeps canonical rewards.
    const char* db_env = getenv("CRAFTAX_CU_DESCEND_BONUS");
    float descend_bonus = db_env == NULL ? 1.0f : (float)atof(db_env);
    CU_CHECK(cudaMemcpyToSymbol(g_craftax_descend_bonus, &descend_bonus,
                                sizeof(float)));
    if (descend_bonus != 1.0f) {
        printf("descend_bonus=%.2f (enter-floor rewards scaled)\n", descend_bonus);
    }
    // Curriculum resets: CRAFTAX_CU_CURR_P=<frac> [CRAFTAX_CU_CURR_MAXF=<1..8>].
    // Default 0 = off (canonical behavior, hash anchors unchanged).
    const char* cp_env = getenv("CRAFTAX_CU_CURR_P");
    float curr_p = cp_env == NULL ? 0.0f : (float)atof(cp_env);
    const char* cmf_env = getenv("CRAFTAX_CU_CURR_MAXF");
    int curr_maxf = cmf_env == NULL ? 2 : atoi(cmf_env);
    if (curr_maxf < 1) curr_maxf = 1;
    if (curr_maxf > 8) curr_maxf = 8;
    CU_CHECK(cudaMemcpyToSymbol(g_craftax_curr_p, &curr_p, sizeof(float)));
    CU_CHECK(cudaMemcpyToSymbol(g_craftax_curr_maxf, &curr_maxf, sizeof(int)));
    v->curr_p = curr_p;
    if (curr_p > 0.0f) {
        printf("curriculum p=%.2f maxf=%d (curriculum episodes excluded from stats)\n",
               curr_p, curr_maxf);
    }
    // Megakernel path (fused step+reset+encode, multi-step launches in
    // bench mode), CRAFTAX_CU_MEGA=1. Verified bitwise identical to the
    // split kernels (statehash mode), but measured SLOWER on sm_120 (64.0M
    // vs 69.9M SPS @65k envs): one fused kernel forces a single
    // register/occupancy configuration on phases with opposite needs
    // (gameplay wants registers, encode wants resident warps), so the split
    // path stays the default. Requires lazy floors (warp reset).
    const char* mega_env = getenv("CRAFTAX_CU_MEGA");
    v->mega = (mega_env == NULL ? 0 : atoi(mega_env)) && lazy;
    // Spawn request compaction is a split-path optimization; the megakernel
    // keeps the inline tail (queue symbol stays NULL there).
    if (!v->mega) {
        CU_CHECK(cudaMemcpyToSymbol(g_cf_spawn_queue, &v->d_spawn_queue,
                                    sizeof(v->d_spawn_queue)));
    }
    v->d_warp_scratch = NULL;
    if (v->mega) {
        size_t n_warps = ((size_t)num_envs + 63) / 64 * 2;
        CU_CHECK(cudaMalloc(&v->d_warp_scratch,
                            n_warps * sizeof(CraftaxWarpScratch)));
    }
    k_global_init<<<1, 1>>>();
    CU_CHECK(cudaDeviceSynchronize());

    int batch = 8192;  // bounded init launches (full 9-level worldgen each)
    for (int lo = 0; lo < num_envs; lo += batch) {
        int hi = lo + batch < num_envs ? lo + batch : num_envs;
        int n = hi - lo;
        k_env_init<<<(n + 63) / 64, 64>>>(
            v->d_envs, v->d_states, v->d_obs, v->d_actions, v->d_rewards,
            v->d_terminals, v->d_action_rng, lo, hi, seed);
        CU_CHECK(cudaDeviceSynchronize());
    }
}

static void cu_vec_free(CuVec* v) {
    cudaFree(v->d_envs); cudaFree(v->d_states); cudaFree(v->d_obs);
    cudaFree(v->d_actions); cudaFree(v->d_rewards); cudaFree(v->d_terminals);
    cudaFree(v->d_action_rng); cudaFree(v->d_resets);
    cudaFree(v->d_reset_flags);
    cudaFree(v->d_spawn_queue);
    cudaFree(v->d_wg_scratch);
    if (v->d_warp_scratch != NULL) cudaFree(v->d_warp_scratch);
    for (int i = 0; i < v->num_soa; i++) cudaFree(v->d_soa[i]);
    free(v->h_obs); free(v->h_rewards); free(v->h_terminals);
}

static void cu_copy_back(CuVec* v, bool with_obs) {
    if (with_obs) {
        CU_CHECK(cudaMemcpy(
            v->h_obs, v->d_obs,
            (size_t)v->num_envs * CRAFTAX_OBS_SIZE * sizeof(CraftaxObs),
            cudaMemcpyDeviceToHost));
    }
    CU_CHECK(cudaMemcpy(v->h_rewards, v->d_rewards,
                        (size_t)v->num_envs * sizeof(float),
                        cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(v->h_terminals, v->d_terminals,
                        (size_t)v->num_envs * sizeof(float),
                        cudaMemcpyDeviceToHost));
}

static void cu_print_logs(CuVec* v, bool histogram) {
    Craftax* h_envs = (Craftax*)malloc((size_t)v->num_envs * sizeof(Craftax));
    CU_CHECK(cudaMemcpy(h_envs, v->d_envs,
                        (size_t)v->num_envs * sizeof(Craftax),
                        cudaMemcpyDeviceToHost));
    double episodes = 0.0, ep_len = 0.0, ep_ret = 0.0, ach = 0.0;
    double hist[CRAFTAX_NUM_ACHIEVEMENTS] = {0};
    for (int i = 0; i < v->num_envs; i++) {
        Log* log = &h_envs[i].log;
        episodes += (double)log->n;
        ep_len += (double)log->episode_length;
        ep_ret += (double)log->score;
        for (int a = 0; a < CRAFTAX_NUM_ACHIEVEMENTS; a++) {
            ach += (double)log->achievements[a];
            hist[a] += (double)log->achievements[a];
        }
    }
    printf("episodes_completed=%.0f mean_episode_length=%.2f\n", episodes,
           episodes > 0.0 ? ep_len / episodes : 0.0);
    printf("completed_episode_reward=%.3f total_achievements=%.0f\n",
           ep_ret, ach);
    if (histogram) {
        printf("achievement_histogram");
        for (int a = 0; a < CRAFTAX_NUM_ACHIEVEMENTS; a++) {
            printf(" %.0f", hist[a]);
        }
        printf("\n");
    }
    free(h_envs);
}

static void cu_mega_launch(CuVec* v, int num_steps) {
    k_mega<<<(v->num_envs + 63) / 64, 64>>>(
        v->d_envs, v->d_action_rng, v->num_envs, num_steps,
        v->d_warp_scratch);
}

static void cu_step_launch(CuVec* v) {
    if (v->mega) {
        cu_mega_launch(v, 1);
        return;
    }
    CU_CHECK(cudaMemsetAsync(v->d_reset_count, 0, sizeof(int)));
    CU_CHECK(cudaMemsetAsync(v->d_spawn_count, 0, sizeof(int)));
    k_step<<<(v->num_envs + 63) / 64, 64>>>(
        v->d_envs, v->d_action_rng, v->num_envs, v->d_resets);
    // Compacted spawn scans; must precede the reset kernels (done envs get
    // their spawn writes into the pre-reset world, as inline order did).
    // One warp per request, grid-stride over the worklist.
    {
        int tail_blocks = (v->num_envs * 32 + 255) / 256;
        if (tail_blocks > 512) tail_blocks = 512;
        k_spawn_tail<<<tail_blocks, 256>>>();
    }
    // Cap the reset grid: worldgen is ~one warp of work per finished env and
    // typical load is tens of resets, not num_envs. Stride covers the rare
    // all-die step. (See k_reset_list_warp.)
    if (v->lazy) {
        cu_launch_reset_list_warp(v->d_envs, v->d_resets, v->num_envs, 0);
    } else {
        k_reset_list<<<(v->num_envs + 63) / 64, 64>>>(v->d_envs, v->d_resets);
    }
    dim3 enc_block(32, CRAFTAX_ENC_WARPS_PER_BLOCK);
    int enc_grid = (v->num_envs + CRAFTAX_ENC_WARPS_PER_BLOCK - 1)
        / CRAFTAX_ENC_WARPS_PER_BLOCK;
    k_encode<<<enc_grid, enc_block>>>(v->d_envs, v->num_envs);
}

// ------------------------------------------------------------
// hash mode: identical FNV-1a stream to ./craftax_full hash
// ------------------------------------------------------------
#include <mma.h>

