// Native C port of craftax/craftax/util/noise.py.

#pragma once

#include <math.h>
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

static inline void craftax_generate_perlin_noise_2d(
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
            craftax_noise_gradient(
                angle_key,
                res_cols,
                grad_row,
                grad_col,
                override_angles,
                &g00x,
                &g00y
            );
            craftax_noise_gradient(
                angle_key,
                res_cols,
                grad_row + 1,
                grad_col,
                override_angles,
                &g10x,
                &g10y
            );
            craftax_noise_gradient(
                angle_key,
                res_cols,
                grad_row,
                grad_col + 1,
                override_angles,
                &g01x,
                &g01y
            );
            craftax_noise_gradient(
                angle_key,
                res_cols,
                grad_row + 1,
                grad_col + 1,
                override_angles,
                &g11x,
                &g11y
            );

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
