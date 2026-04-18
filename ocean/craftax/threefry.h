// JAX-compatible threefry2x32 helpers for Craftax.
//
// The local JAX version uses the partitionable threefry split path by default:
// split(key, n)[i] is threefry2x32(key, counter=(0, i)). 32-bit random bits
// are bits0 ^ bits1 from the same counter schedule.

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <string.h>

typedef struct CraftaxThreefryKey {
    uint32_t word[2];
} CraftaxThreefryKey;

static inline uint32_t craftax_rotl32(uint32_t x, uint32_t k) {
    return (uint32_t)((x << k) | (x >> (32u - k)));
}

static inline CraftaxThreefryKey craftax_prng_key(uint32_t seed) {
    CraftaxThreefryKey key = {{0u, seed}};
    return key;
}

static inline void craftax_threefry2x32(
    CraftaxThreefryKey key,
    uint32_t count0,
    uint32_t count1,
    uint32_t out[2]
) {
    static const uint32_t rotations[2][4] = {
        {13u, 15u, 26u, 6u},
        {17u, 29u, 16u, 24u},
    };

    uint32_t ks[3] = {
        key.word[0],
        key.word[1],
        key.word[0] ^ key.word[1] ^ 0x1BD11BDAu,
    };
    uint32_t x0 = count0 + ks[0];
    uint32_t x1 = count1 + ks[1];

    for (uint32_t block = 0; block < 5u; block++) {
        const uint32_t* rs = rotations[block & 1u];
        for (int i = 0; i < 4; i++) {
            x0 += x1;
            x1 = craftax_rotl32(x1, rs[i]);
            x1 ^= x0;
        }
        x0 += ks[(block + 1u) % 3u];
        x1 += ks[(block + 2u) % 3u] + block + 1u;
    }

    out[0] = x0;
    out[1] = x1;
}

static inline CraftaxThreefryKey craftax_threefry_counter_key(
    CraftaxThreefryKey key,
    uint32_t count0,
    uint32_t count1
) {
    uint32_t out[2];
    craftax_threefry2x32(key, count0, count1, out);
    CraftaxThreefryKey result = {{out[0], out[1]}};
    return result;
}

static inline void craftax_threefry_split(
    CraftaxThreefryKey key,
    CraftaxThreefryKey* left,
    CraftaxThreefryKey* right
) {
    *left = craftax_threefry_counter_key(key, 0u, 0u);
    *right = craftax_threefry_counter_key(key, 0u, 1u);
}

static inline void craftax_threefry_split_n(
    CraftaxThreefryKey key,
    CraftaxThreefryKey* out,
    size_t count
) {
    for (size_t i = 0; i < count; i++) {
        uint64_t counter = (uint64_t)i;
        out[i] = craftax_threefry_counter_key(
            key,
            (uint32_t)(counter >> 32),
            (uint32_t)counter
        );
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
    uint32_t out[2];
    craftax_threefry2x32(
        key,
        (uint32_t)(index >> 32),
        (uint32_t)index,
        out
    );
    return out[0] ^ out[1];
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
