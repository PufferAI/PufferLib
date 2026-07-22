// Native C port of craftax/craftax/util/noise.py.

#pragma once

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "threefry.h"

#ifndef CRAFTAX_NOISE_PI2
#define CRAFTAX_NOISE_PI2 6.28318530717958647692f
#endif

#ifndef CRAFTAX_NOISE_SQRT2
#define CRAFTAX_NOISE_SQRT2 1.41421356237309504880f
#endif

static inline float craftax_noise_interpolant(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

static inline float craftax_noise_gradient_angle(
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

static inline void craftax_noise_gradient(
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
// Sized with headroom; a grid larger than this falls back to per-cell lookups.
#define CRAFTAX_NOISE_MAX_GRAD 1024
#endif

static inline void craftax_generate_perlin_noise_2d_scalar(
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

#if defined(__AVX512F__) && !defined(CRAFTAX_NO_SIMD_NOISE)
#include <immintrin.h>

static inline __m512 craftax_noise_interpolant_v(__m512 t) {
    // t*t*t*(t*(t*6 - 15) + 10), plain mul/add to track the scalar formula.
    __m512 a = _mm512_add_ps(
        _mm512_mul_ps(t, _mm512_set1_ps(6.0f)), _mm512_set1_ps(-15.0f));
    a = _mm512_add_ps(_mm512_mul_ps(t, a), _mm512_set1_ps(10.0f));
    __m512 t3 = _mm512_mul_ps(_mm512_mul_ps(t, t), t);
    return _mm512_mul_ps(t3, a);
}

// AVX-512 inner loop, 16 output cells at a time. Requires cols % 16 == 0,
// power-of-two cell_cols, and a precomputed gradient table (all true for
// every worldgen call: 48-wide maps, cell_cols in {2,4,16}). May differ from
// the scalar path by ~1 ULP where the compiler contracted scalar mul+add
// into FMA; worlds are distributionally identical, not bit-identical.
static inline bool craftax_generate_perlin_noise_2d_avx512(
    CraftaxThreefryKey rng,
    int rows,
    int cols,
    int res_rows,
    int res_cols,
    const float* override_angles,
    float* out
) {
    int cell_rows = rows / res_rows;
    int cell_cols = cols / res_cols;
    int grad_w = res_cols + 1;
    int grad_h = res_rows + 1;
    if (cols % 16 != 0) return false;
    if (cell_cols <= 0 || (cell_cols & (cell_cols - 1)) != 0) return false;
    if (grad_w * grad_h > CRAFTAX_NOISE_MAX_GRAD) return false;

    CraftaxThreefryKey unused;
    CraftaxThreefryKey angle_key;
    craftax_threefry_split(rng, &unused, &angle_key);

    float grad_x[CRAFTAX_NOISE_MAX_GRAD];
    float grad_y[CRAFTAX_NOISE_MAX_GRAD];
    for (int r = 0; r < grad_h; r++) {
        for (int c = 0; c < grad_w; c++) {
            float angle = craftax_noise_gradient_angle(
                angle_key, res_cols, r, c, override_angles);
            grad_x[r * grad_w + c] = cosf(angle);
            grad_y[r * grad_w + c] = sinf(angle);
        }
    }

    uint32_t col_shift = (uint32_t)__builtin_ctz((unsigned)cell_cols);
    __m512i shift_v = _mm512_set1_epi32((int)col_shift);
    __m512i col_mask = _mm512_set1_epi32(cell_cols - 1);
    __m512 inv_cell_cols = _mm512_set1_ps(1.0f / (float)cell_cols);
    __m512i grad_w_v = _mm512_set1_epi32(grad_w);
    __m512i lane = _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 sqrt2 = _mm512_set1_ps(CRAFTAX_NOISE_SQRT2);

    for (int row = 0; row < rows; row++) {
        int grad_row = row / cell_rows;
        float local_row_s =
            (float)(row - grad_row * cell_rows) / (float)cell_rows;
        float interp_row_s = craftax_noise_interpolant(local_row_s);
        __m512 local_row = _mm512_set1_ps(local_row_s);
        __m512 local_row_m1 = _mm512_set1_ps(local_row_s - 1.0f);
        __m512 interp_row = _mm512_set1_ps(interp_row_s);
        __m512 one_m_interp_row = _mm512_set1_ps(1.0f - interp_row_s);
        __m512i row_base = _mm512_set1_epi32(grad_row * grad_w);

        for (int col = 0; col < cols; col += 16) {
            __m512i col_v = _mm512_add_epi32(_mm512_set1_epi32(col), lane);
            __m512i grad_col = _mm512_srlv_epi32(col_v, shift_v);
            __m512 local_col = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_and_si512(col_v, col_mask)),
                inv_cell_cols);
            __m512 interp_col = craftax_noise_interpolant_v(local_col);

            __m512i i00 = _mm512_add_epi32(row_base, grad_col);
            __m512i i10 = _mm512_add_epi32(i00, grad_w_v);
            __m512i i01 = _mm512_add_epi32(i00, _mm512_set1_epi32(1));
            __m512i i11 = _mm512_add_epi32(i10, _mm512_set1_epi32(1));

            __m512 g00x = _mm512_i32gather_ps(i00, grad_x, 4);
            __m512 g00y = _mm512_i32gather_ps(i00, grad_y, 4);
            __m512 g10x = _mm512_i32gather_ps(i10, grad_x, 4);
            __m512 g10y = _mm512_i32gather_ps(i10, grad_y, 4);
            __m512 g01x = _mm512_i32gather_ps(i01, grad_x, 4);
            __m512 g01y = _mm512_i32gather_ps(i01, grad_y, 4);
            __m512 g11x = _mm512_i32gather_ps(i11, grad_x, 4);
            __m512 g11y = _mm512_i32gather_ps(i11, grad_y, 4);

            __m512 local_col_m1 = _mm512_sub_ps(local_col, one);
            __m512 n00 = _mm512_add_ps(
                _mm512_mul_ps(local_row, g00x),
                _mm512_mul_ps(local_col, g00y));
            __m512 n10 = _mm512_add_ps(
                _mm512_mul_ps(local_row_m1, g10x),
                _mm512_mul_ps(local_col, g10y));
            __m512 n01 = _mm512_add_ps(
                _mm512_mul_ps(local_row, g01x),
                _mm512_mul_ps(local_col_m1, g01y));
            __m512 n11 = _mm512_add_ps(
                _mm512_mul_ps(local_row_m1, g11x),
                _mm512_mul_ps(local_col_m1, g11y));

            __m512 n0 = _mm512_add_ps(
                _mm512_mul_ps(n00, one_m_interp_row),
                _mm512_mul_ps(interp_row, n10));
            __m512 n1 = _mm512_add_ps(
                _mm512_mul_ps(n01, one_m_interp_row),
                _mm512_mul_ps(interp_row, n11));
            __m512 result = _mm512_mul_ps(sqrt2, _mm512_add_ps(
                _mm512_mul_ps(_mm512_sub_ps(one, interp_col), n0),
                _mm512_mul_ps(interp_col, n1)));
            _mm512_storeu_ps(&out[(size_t)row * (size_t)cols + (size_t)col],
                             result);
        }
    }
    return true;
}
#endif  // __AVX512F__ && !CRAFTAX_NO_SIMD_NOISE

static inline void craftax_generate_perlin_noise_2d(
    CraftaxThreefryKey rng,
    int rows,
    int cols,
    int res_rows,
    int res_cols,
    const float* override_angles,
    float* out
) {
#if defined(__AVX512F__) && !defined(CRAFTAX_NO_SIMD_NOISE)
    if (craftax_generate_perlin_noise_2d_avx512(
            rng, rows, cols, res_rows, res_cols, override_angles, out)) {
        return;
    }
#endif
    craftax_generate_perlin_noise_2d_scalar(
        rng, rows, cols, res_rows, res_cols, override_angles, out);
}

static inline void craftax_generate_fractal_noise_2d(
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
    for (size_t i = 0; i < size; i++) {
        out[i] = 0.0f;
    }

    int frequency = 1;
    float amplitude = 1.0f;
    float perlin[size];

    for (int octave = 0; octave < octaves; octave++) {
        CraftaxThreefryKey next_rng;
        CraftaxThreefryKey noise_key;
        craftax_threefry_split(rng, &next_rng, &noise_key);
        rng = next_rng;

        craftax_generate_perlin_noise_2d(
            noise_key,
            rows,
            cols,
            frequency * res_rows,
            frequency * res_cols,
            override_angles,
            perlin
        );

        for (size_t i = 0; i < size; i++) {
            out[i] += amplitude * perlin[i];
        }

        frequency *= lacunarity;
        amplitude *= persistence;
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
