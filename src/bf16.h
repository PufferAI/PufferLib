// Usage:
//     #include "bf16.h"
//
//     Set env_name/binding.c obs to PrecisionTensor
//
//     bf16* observations;
//     observations[0] = f32_to_bf16(some_float);                // scalar
//
//     // SIMD fast-path for inner loops with 8 floats already in an __m256:
//     __m256 v = _mm256_mul_ps(x, scale);
//     store_f32x8_as_bf16(&observations[i], v);                 // 1 store, 8 vals
//
//     // Reverse if you ever need to read back as float:
//     float f = bf16_to_f32(observations[0]);
//
// x86_64 only — uses AVX intrinsics. To port to ARM/Apple Silicon, replace
// store_f32x8_as_bf16 with a NEON equivalent (or remove and use scalar
// f32_to_bf16 in a loop — the compiler auto-vectorizes well).

#include <stdint.h>
#include <string.h>
#include <immintrin.h>

typedef uint16_t bf16;

static inline bf16 f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}

static inline float bf16_to_f32(bf16 b) {
    uint32_t bits = (uint32_t)b << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static inline void store_f32x8_as_bf16(bf16* dst, __m256 v) {
    __m256i vi = _mm256_srli_epi32(_mm256_castps_si256(v), 16);
    __m128i lo = _mm256_castsi256_si128(vi);
    __m128i hi = _mm256_extracti128_si256(vi, 1);
    _mm_storeu_si128((__m128i*)dst, _mm_packus_epi32(lo, hi));
}
