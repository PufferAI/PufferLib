// Fast RNG helpers for Craftax.
// Replaces JAX Threefry with SplitMix64-based hashing for ~20-50x speedup.
// NOT cryptographically secure and NOT JAX-compatible.

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <string.h>

typedef struct CraftaxThreefryKey {
    uint32_t word[2];
} CraftaxThreefryKey;

static inline uint64_t craftax_key_to_u64(CraftaxThreefryKey key) {
    return ((uint64_t)key.word[1] << 32) | key.word[0];
}

static inline CraftaxThreefryKey craftax_u64_to_key(uint64_t x) {
    CraftaxThreefryKey key = {{(uint32_t)x, (uint32_t)(x >> 32)}};
    return key;
}

static inline uint32_t craftax_rotl32(uint32_t x, uint32_t k) {
    return (uint32_t)((x << k) | (x >> (32u - k)));
}

static inline CraftaxThreefryKey craftax_prng_key(uint32_t seed) {
    CraftaxThreefryKey key = {{seed, seed ^ 0x9E3779B9u}};
    return key;
}

// MurmurHash3 64-bit finalizer — fast and good mixing
static inline uint64_t craftax_mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

// Core hash: mixes key state with counter, returns 64 bits of pseudo-randomness
static inline uint64_t craftax_fast_hash64(CraftaxThreefryKey key, uint64_t counter) {
    uint64_t x = craftax_key_to_u64(key);
    x ^= counter;
    return craftax_mix64(x);
}

static inline void craftax_threefry2x32(
    CraftaxThreefryKey key,
    uint32_t count0,
    uint32_t count1,
    uint32_t out[2]
) {
    uint64_t h = craftax_fast_hash64(key, ((uint64_t)count1 << 32) | count0);
    out[0] = (uint32_t)h;
    out[1] = (uint32_t)(h >> 32);
}

static inline CraftaxThreefryKey craftax_threefry_counter_key(
    CraftaxThreefryKey key,
    uint32_t count0,
    uint32_t count1
) {
    return craftax_u64_to_key(craftax_fast_hash64(key, ((uint64_t)count1 << 32) | count0));
}

// Fast split: sequential PCG-style advancement
static inline void craftax_threefry_split(
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

static inline void craftax_threefry_split_n(
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

static inline CraftaxThreefryKey craftax_threefry_fold_in(
    CraftaxThreefryKey key,
    uint32_t data
) {
    return craftax_threefry_counter_key(key, 0u, data);
}

static inline uint32_t craftax_threefry_uniform_u32_at(
    CraftaxThreefryKey key,
    uint64_t index
) {
    uint64_t h = craftax_fast_hash64(key, index);
    return (uint32_t)h ^ (uint32_t)(h >> 32);
}

static inline uint32_t craftax_threefry_uniform_u32(CraftaxThreefryKey key) {
    return craftax_threefry_uniform_u32_at(key, 0u);
}

static inline float craftax_threefry_uniform_f32_at(
    CraftaxThreefryKey key,
    uint64_t index
) {
    uint32_t bits = craftax_threefry_uniform_u32_at(key, index);
    uint32_t float_bits = (bits >> 9u) | 0x3F800000u;
    float value;
    memcpy(&value, &float_bits, sizeof(value));
    return value - 1.0f;
}

static inline float craftax_threefry_uniform_f32(CraftaxThreefryKey key) {
    return craftax_threefry_uniform_f32_at(key, 0u);
}
